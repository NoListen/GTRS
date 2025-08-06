# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Union
from typing import Dict
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.dp.dp_config import DPConfig
from navsim.agents.dp.dp_model import DPModel
from navsim.agents.gtrs_dense.hydra_features import HydraFeatureBuilder, HydraTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)



def dp_rl_loss_bev(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: DPConfig,
    traj_head,
    pdm_scores                # 可能是 np.ndarray 或 tensor
):
    # 1. 将 pdm_scores 转为 torch.Tensor
    if isinstance(pdm_scores, np.ndarray):
        device = predictions['log_step_probs'].device
        pdm_scores = torch.from_numpy(pdm_scores).to(device)
    else:
        pdm_scores = pdm_scores.to(predictions['log_step_probs'].device)

    # 2. 调用封装好的 get_dp_loss 接口（默认只 RL + BC）
    dp_loss, dp_metrics = traj_head.get_dp_loss(
        log_step_probs=predictions['log_step_probs'],       # tensor B×G×T
        pdm_scores=pdm_scores,                              # tensor B×G
        bc_log_step_probs=predictions['bc_log_step_probs']  # tensor B×G×T
    )
    dp_loss = dp_loss * config.dp_loss_weight

    # 3. 超参数中保留 BEV loss（如果你还需要）
    bev_semantic_loss = F.cross_entropy(
        predictions["bev_semantic_map"],
        targets["bev_semantic_map"].long()
    ) * config.bev_loss_weight

    total_loss = dp_loss + bev_semantic_loss

    # 4. 汇总 metrics 信息
    metrics = {
        'dp_rl_loss': dp_loss.item(),
        'bev_semantic_loss': bev_semantic_loss.item(),
        **dp_metrics
    }

    return total_loss, metrics

class DPRLAgent(AbstractAgent):
    def __init__(
            self,
            config: DPConfig,
            lr: float,
            checkpoint_path: str = None
    ):
        super().__init__(
            trajectory_sampling=config.trajectory_sampling
        )
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self.model = DPModel(config)
        self.backbone_wd = config.backbone_wd
        self.scheduler = config.scheduler

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
            "state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[0, 1, 2, 3],
            cam_l0=[0, 1, 2, 3],
            cam_l1=[0, 1, 2, 3],
            cam_l2=[0, 1, 2, 3],
            cam_r0=[0, 1, 2, 3],
            cam_r1=[0, 1, 2, 3],
            cam_r2=[0, 1, 2, 3],
            cam_b0=[0, 1, 2, 3],
            lidar_pc=[],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [HydraTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [HydraFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(features)

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
            pdm_scores: pd.DataFr
    ):
        return dp_loss_bev(targets, predictions, self._config, self.model._trajectory_head)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        backbone_params_name = '_backbone.image_encoder'
        reference_transformer_name = '_trajectory_head.reference_transformer'
        ori_transformer_name = '_trajectory_head.ori_transformer'
        img_backbone_params = list(
            filter(lambda kv: backbone_params_name in kv[0], self.model.named_parameters()))
        default_params = list(filter(lambda kv:
                                     backbone_params_name not in kv[0] and
                                     reference_transformer_name not in kv[0] and
                                     ori_transformer_name not in kv[0], self.model.named_parameters()))
        params_lr_dict = [
            {'params': [tmp[1] for tmp in default_params]},
            {
                'params': [tmp[1] for tmp in img_backbone_params],
                'lr': self._lr * self._config.lr_mult_backbone,
                'weight_decay': self.backbone_wd
            }
        ]

        if self.scheduler == 'default':
            return torch.optim.Adam(params_lr_dict, lr=self._lr, weight_decay=self._config.weight_decay)
        elif self.scheduler == 'cycle':
            optim = torch.optim.Adam(params_lr_dict, lr=self._lr)
            return {
                "optimizer": optim,
                "lr_scheduler": OneCycleLR(
                    optim,
                    max_lr=0.01,
                    total_steps=100 * 202
                )
            }
        else:
            raise ValueError('Unsupported lr scheduler')

    def get_training_callbacks(self) -> List[pl.Callback]:
        ckpt_callback = ModelCheckpoint(
            save_top_k=100,
            monitor="val/loss_epoch",
            mode="min",
            dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/",
            filename="{epoch:02d}-{step:04d}",
        )
        return [
            ckpt_callback
        ]
