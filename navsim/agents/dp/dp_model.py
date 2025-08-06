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

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

from navsim.agents.dp.dp_config import DPConfig
from navsim.agents.gtrs_dense.hydra_backbone_bev import HydraBackboneBEV

x_diff_min = -1.2698211669921875
x_diff_max = 7.475563049316406
x_diff_mean = 2.950225591659546

# Y difference statistics
y_diff_min = -5.012081146240234
y_diff_max = 4.8563690185546875
y_diff_mean = 0.0607292577624321

# Calculate scaling factors for differences
x_diff_scale = max(abs(x_diff_max - x_diff_mean), abs(x_diff_min - x_diff_mean))
y_diff_scale = max(abs(y_diff_max - y_diff_mean), abs(y_diff_min - y_diff_mean))

HORIZON = 8
ACTION_DIM = 4
ACTION_DIM_ORI = 3


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SimpleDiffusionTransformer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, dp_nlayers, input_dim, obs_len):
        super().__init__()
        self.dp_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), dp_nlayers
        )
        self.input_emb = nn.Linear(input_dim, d_model)
        self.time_emb = SinusoidalPosEmb(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_emb = nn.Linear(d_model, input_dim)
        token_len = obs_len + 1
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, token_len, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, SimpleDiffusionTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(self,
                sample,
                timestep,
                cond):
        B, HORIZON, DIM = sample.shape
        sample = sample.view(B, -1).float()
        input_emb = self.input_emb(sample)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,To,n_emb)
        cond_embeddings = torch.cat([time_emb, cond], dim=1)
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[
                              :, :tc, :
                              ]  # each position maps to a (learnable) vector
        x = cond_embeddings + position_embeddings
        memory = x
        # (B,T_cond,n_emb)

        # decoder
        token_embeddings = input_emb.unsqueeze(1)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
                              :, :t, :
                              ]  # each position maps to a (learnable) vector
        x = token_embeddings + position_embeddings
        # (B,T,n_emb)
        x = self.dp_transformer(
            tgt=x,
            memory=memory,
        )
        # (B,T,n_emb)
        x = self.ln_f(x)
        x = self.output_emb(x)
        return x.squeeze(1).view(B, HORIZON, DIM)


def diff_traj(traj):
    B, L, _ = traj.shape
    sin = traj[..., -1:].sin()
    cos = traj[..., -1:].cos()
    zero_pad = torch.zeros((B, 1, 1), dtype=traj.dtype, device=traj.device)
    x_diff = traj[..., 0:1].diff(n=1, dim=1, prepend=zero_pad)
    x_diff = x_diff - x_diff_mean
    x_diff_range = max(abs(x_diff_max - x_diff_mean), abs(x_diff_min - x_diff_mean))
    x_diff_norm = x_diff / x_diff_range

    zero_pad = torch.zeros((B, 1, 1), dtype=traj.dtype, device=traj.device)
    y_diff = traj[..., 1:2].diff(n=1, dim=1, prepend=zero_pad)
    y_diff = y_diff - y_diff_mean
    y_diff_range = max(abs(y_diff_max - y_diff_mean), abs(y_diff_min - y_diff_mean))
    y_diff_norm = y_diff / y_diff_range

    return torch.cat([x_diff_norm, y_diff_norm, sin, cos], -1)


def cumsum_traj(norm_trajs):
    B, L, _ = norm_trajs.shape
    sin_values = norm_trajs[..., 2:3]
    cos_values = norm_trajs[..., 3:4]
    heading = torch.atan2(sin_values, cos_values)

    # Denormalize x differences
    x_diff_range = max(abs(x_diff_max - x_diff_mean), abs(x_diff_min - x_diff_mean))
    x_diff = norm_trajs[..., 0:1] * x_diff_range + x_diff_mean

    # Denormalize y differences
    y_diff_range = max(abs(y_diff_max - y_diff_mean), abs(y_diff_min - y_diff_mean))
    y_diff = norm_trajs[..., 1:2] * y_diff_range + y_diff_mean

    # Cumulative sum to get absolute positions
    x = x_diff.cumsum(dim=1)
    y = y_diff.cumsum(dim=1)

    return torch.cat([x, y, heading], -1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=self.mlp[0].weight.device)
        if t.dim() == 0:
            t = t[None]
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        emb = self.mlp(emb)
        return emb


class DiTBlock1D(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        q = k = v = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(q, k, v)
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class DiTBlockWithCrossAttention1D(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim):
        super().__init__()
        self.dit_block = DiTBlock1D(hidden_size, num_heads, mlp_dim)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size),
        )

    def forward(self, x, c, context):
        x = self.dit_block(x, c)
        # layer norm1
        q = self.norm1(x)
        k = v = context
        attn_out, _ = self.attn(q, k, v)
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x

class FinalLayer1D(nn.Module):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTDiffusionTransformer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, depth, input_dim, obs_len):
        super().__init__()
        self.input_emb = nn.Linear(input_dim, d_model)
        self.time_emb = TimestepEmbedder(d_model)
        self.cond_proj = nn.Linear(d_model, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, HORIZON, d_model))  # HORIZON needs to be defined

        self.blocks = nn.ModuleList([
            DiTBlockWithCrossAttention1D(d_model, nhead, d_ffn) for _ in range(depth)
        ])
        self.final_layer = FinalLayer1D(d_model, input_dim)  # Assumed already defined

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, sample, timestep, cond, context):
        """
        sample: (B, L, input_dim) - Noised input
        timestep: (B,) or scalar - Diffusion timestep
        cond: (B, T_cond, d_model) - VLM tokens or similar condition
        context: (B, T_ctx, d_model) - Cross-attention tokens (same as cond if no separate context)
        """
        B, L, D = sample.shape
        x = self.input_emb(sample) + self.pos_emb[:, :L]  # Add positional encoding

        # Process timestep and cond
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        if timestep.dim() == 0:
            timestep = timestep[None]
        timestep = timestep.expand(B)
        t_emb = self.time_emb(timestep)  # (B, d_model)

        # Combine time + average VLM condition
        pooled_cond = cond.mean(dim=1)  # (B, d_model)
        c = t_emb + self.cond_proj(pooled_cond)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x, c, context)

        x = self.final_layer(x, c)
        return x

class DPHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: DPConfig = None
                 ):
        super().__init__()
        self.config = config
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.denoising_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_small',
            clip_sample=True,
            clip_sample_range=1.0,
            prediction_type='epsilon'
        )
        img_num = 2 if config.use_back_view else 1

        self.transformer_dp = DiTDiffusionTransformer(
            d_model, nhead, d_ffn, config.dp_layers,
            input_dim=ACTION_DIM * HORIZON,
            obs_len=config.img_vert_anchors * config.img_horz_anchors * img_num + 1,
        )
        self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps

    def forward(self, cond, context) -> Dict[str, torch.Tensor]:
        B = context.shape[0]
        NUM = self.config.num_proposals
        T = self.noise_scheduler.num_inference_steps
        device = context.device

        context = context.repeat_interleave(NUM, dim=0)
        cond = cond.repeat_interleave(NUM, dim=0)
        noise = torch.randn((B*NUM, HORIZON, ACTION_DIM), device=device, dtype=cond.dtype)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        log_step = torch.zeros((B*NUM, T), device=device)
        bc_log_step = torch.zeros((B*NUM, T), device=device)

        for idx, t in enumerate(self.noise_scheduler.timesteps):
            out = self.transformer_dp(noise, t, cond=cond, context=context)
            prev = self.noise_scheduler.step(out, t, noise).prev_sample
            log_step[:, idx] = self.compute_log_prob_step(out, t, noise, prev)

            with torch.no_grad():
                ref_out = self.ref_model.transformer_dp(noise, t, cond=cond, context=context)
            bc_log_step[:, idx] = self.compute_log_prob_step(ref_out, t, noise, prev)

            noise = prev

        traj = cumsum_traj(noise)
        return {
            'trajectories': traj.view(B, NUM, HORIZON, ACTION_DIM_ORI),
            'log_step_probs': log_step.view(B, NUM, T),
            'bc_log_step_probs': bc_log_step.view(B, NUM, T)
        }
    
    def get_dp_loss(
        self,
        log_step_probs: torch.Tensor,      # 主模型 per-step log π: (B, G, T)
        pdm_scores: torch.Tensor,          # PDMS reward: (B, G)
        bc_log_step_probs: torch.Tensor     # π_ref per-step log π: (B, G, T)
    ):
        B, G, T = log_step_probs.shape
        device = log_step_probs.device

        # 1. Group-standardized advantage Â_i
        rewards = pdm_scores.to(device)
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True)
        adv = (rewards - mean_r) / (std_r + self.config.std_eps)  # B×G

        # 2. 为每步加 γ^(t−1)，计算 chain 平均 log-likelihood
        gamma = torch.tensor([self.config.gamma ** t for t in range(T)], device=device)
        discounted_ll = (log_step_probs * gamma.view(1,1,T)).mean(dim=2)  # B×G

        # 3. RL 目标项
        rl_loss = -(discounted_ll * adv).mean()

        # 4. Reference 模型的 BC 项
        bc_ll = (bc_log_step_probs * gamma.view(1,1,T)).mean(dim=2)  # B×G
        bc_loss = -self.config.bc_weight * bc_ll.mean()

        total_loss = self.config.rl_weight * rl_loss + bc_loss
        metrics = {
            'rl_loss': rl_loss.item(),
            'bc_loss': bc_ll.mean().item(),
            'mean_adv': adv.mean().item(),
            'std_adv': rewards.std(dim=1).mean().item()
        }
        return total_loss, metrics
    
    def compute_log_prob_step(self, model_output, t_idx, x_t, x_prev):
        """
        计算 pθ(x_{t-1} | x_t) 的 log 概率：
        log π = -½ * ||x_prev - μθ||² / σ_t²  + const
        使用 DDPM 中的 beta_t, alpha_t 和累积 alpha 来推导 μθ 和 σ_t.
        """
        # 从 scheduler 获取 beta_t 和 alpha cumulative prod
        beta_t = self.noise_scheduler.betas[t_idx]                      # βₜ
        alpha_t = 1.0 - beta_t                                            # αₜ
        alpha_cum = self.noise_scheduler.alphas_cumprod[t_idx]          # \bar{α}_t
        
        sqrt_alpha_cum = torch.sqrt(alpha_cum)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_cum)
        
        # 假设 model_output 是 εθ(x_t, t)
        mu_theta = (x_t - sqrt_one_minus * model_output) / sqrt_alpha_cum

        # 反向扩散的 σ_t
        # 对于 DDPM：σ_t² = β_t（常规简化假设）
        sigma_t = torch.sqrt(beta_t)

        delta = x_prev - mu_theta  # B×H×D
        # Gaussian log-prob 梯度方向常数项忽略
        log_prob = -0.5 * torch.sum(delta ** 2, dim=[1,2]) / (sigma_t ** 2 + 1e-8)

        return log_prob  # shape (B,)

    # def get_dp_loss(self, cond, context, gt_trajectory):
    #     B = context.shape[0]
    #     device = context.device
    #     gt_trajectory = gt_trajectory.float()
    #     gt_trajectory = diff_traj(gt_trajectory)

    #     noise = torch.randn(gt_trajectory.shape, device=device, dtype=torch.float)
    #     # Sample a random timestep for each image
    #     timesteps = torch.randint(
    #         0, self.noise_scheduler.config.num_train_timesteps,
    #         (B,), device=device
    #     ).long()
    #     # Add noise to the clean images according to the noise magnitude at each timestep
    #     # (this is the forward diffusion process)
    #     noisy_dp_input = self.noise_scheduler.add_noise(
    #         gt_trajectory, noise, timesteps
    #     )

    #     # Predict the noise residual
    #     pred = self.transformer_dp(
    #         noisy_dp_input,
    #         timesteps,
    #         cond=cond,
    #         context=context
    #     )
    #     return F.mse_loss(pred, noise)


class DPModel(nn.Module):
    def __init__(self, config: DPConfig):
        super().__init__()
        self._config = config
        self._backbone = HydraBackboneBEV(config)

        kv_len = self._backbone.bev_w * self._backbone.bev_h
        emb_len = kv_len + 1
        if self._config.use_hist_ego_status:
            emb_len += 1
        self._keyval_embedding = nn.Embedding(
            emb_len, config.tf_d_model
        )  # 8x8 feature grid + trajectory

        # usually, the BEV features are variable in size.
        self.downscale_layer = nn.Linear(self._backbone.img_feat_c, config.tf_d_model)
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    config.lidar_resolution_height // 2,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)
        if self._config.use_hist_ego_status:
            self._hist_status_encoding = nn.Linear((2 + 2 + 3) * 3, config.tf_d_model)

        self._trajectory_head = DPHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )
        if self._config.use_temporal_bev_kv:
            self.temporal_bev_fusion = nn.Conv2d(
                config.tf_d_model * 2,
                config.tf_d_model,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            )

    def forward(self, features: Dict[str, torch.Tensor],
                interpolated_traj=None) -> Dict[str, torch.Tensor]:
        camera_feature: torch.Tensor = features["camera_feature"]
        camera_feature_back: torch.Tensor = features["camera_feature_back"]
        status_feature: torch.Tensor = features["status_feature"][0]

        batch_size = status_feature.shape[0]
        assert (camera_feature[-1].shape[0] == batch_size)

        camera_feature_curr = camera_feature[-1]
        if isinstance(camera_feature_back, list):
            camera_feature_back_curr = camera_feature_back[-1]
        else:
            camera_feature_back_curr = camera_feature_back
        img_tokens, bev_tokens, up_bev = self._backbone(camera_feature_curr, camera_feature_back_curr)
        keyval = self.downscale_layer(bev_tokens)
        assert not self._config.use_temporal_bev_kv
        if self._config.use_temporal_bev_kv:
            with torch.no_grad():
                camera_feature_prev = camera_feature[-2]
                camera_feature_back_prev = camera_feature_back[-2]
                img_tokens, bev_tokens, up_bev = self._backbone(camera_feature_prev, camera_feature_back_prev)
                keyval_prev = self.downscale_layer(bev_tokens)
            # grad for fusion layer
            C = keyval.shape[-1]
            keyval = self.temporal_bev_fusion(
                torch.cat([
                    keyval.permute(0, 2, 1).view(batch_size, C, self._backbone.bev_h, self._backbone.bev_w),
                    keyval_prev.permute(0, 2, 1).view(batch_size, C, self._backbone.bev_h, self._backbone.bev_w)
                ], 1)
            ).view(batch_size, C, -1).permute(0, 2, 1).contiguous()

        bev_semantic_map = self._bev_semantic_head(up_bev)
        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([keyval, status_encoding[:, None]], dim=1)
        if self._config.use_hist_ego_status:
            hist_status_encoding = self._hist_status_encoding(features['hist_status_feature'])
            keyval = torch.concatenate([keyval, hist_status_encoding[:, None]], dim=1)

        keyval += self._keyval_embedding.weight[None, ...]

        output: Dict[str, torch.Tensor] = {}
        trajectory = self._trajectory_head(cond=status_feature, cotext=keyval)

        output.update(trajectory)

        output['env_kv'] = keyval
        output['bev_semantic_map'] = bev_semantic_map

        return output
