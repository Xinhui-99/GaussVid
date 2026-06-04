"""
camera_token_encoder.py

将 Plücker 坐标编码为 token 序列，用于 EasyControl 风格的 DiT 注入。

输入格式：已做首帧重复4次打包的 Plücker 坐标
  [B, 6, T_packed, H_pixel, W_pixel]
  其中 T_packed = 4 + (num_pixel_frames - 1)

时间压缩对齐：
  视频的 VAE 用 3D Conv(temporal_kernel=4, temporal_stride=4) 将
  T_packed 帧压缩为 F_latent 帧。本模块的 Conv3d 使用相同的
  temporal_k=4 做对齐的时间压缩，使得：
    N_cam = F_latent × H_patch × W_patch = N_video
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange


class PluckerTokenEncoder(nn.Module):
    """
    将打包后的 Plücker [B, 6, T_packed, H, W] 编码为 token 序列 [B, N_cam, dim]。
    
    架构：
      1. 6 维 Plücker 坐标整体输入 3D patch embedding
      2. MLP fusion 提取几何特征
      3. 投射到 DiT dim，输出 token 序列
    
    时间维 kernel/stride = vae_temporal_factor × patch_size[0] = 4 × 1 = 4
    空间维 kernel/stride = downscale_factor × patch_size[1] = 8 × 2 = 16
    
    N_cam = (T_packed / 4) × (H / 16) × (W / 16)
    """
    def __init__(
        self,
        dim: int = 5120,              # DiT hidden dim
        patch_size: Tuple[int, int, int] = (1, 2, 2),  # DiT patch size
        downscale_factor: int = 8,     # Spatial downscale (to match VAE spatial)
        vae_temporal_factor: int = 4,  # VAE temporal compression ratio
        hidden_dim: int = 256,
        bottleneck_dim: int = 512,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.downscale_factor = downscale_factor
        self.vae_temporal_factor = vae_temporal_factor
        
        # 空间 kernel: downscale × patch_spatial = 8 × 2 = 16
        spatial_k = downscale_factor * patch_size[1]
        # 时间 kernel: vae_temporal × patch_temporal = 4 × 1 = 4
        # 与 VAE 的时间压缩率完全对齐
        temporal_k = vae_temporal_factor * patch_size[0]
        
        self._temporal_k = temporal_k
        self._spatial_k = spatial_k
        
        # ========== 6 维 Plücker 整体 3D Patch Embedding ==========
        self.patch_embed = nn.Conv3d(
            6, hidden_dim,
            kernel_size=(temporal_k, spatial_k, spatial_k),
            stride=(temporal_k, spatial_k, spatial_k),
        )
        
        # ========== MLP Fusion ==========
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # ========== 投射到 DiT dim ==========
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)
        
        # 可学习缩放，初始化小值
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Zero-init 最后一层
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)
    
    def forward(self, plucker: torch.Tensor) -> torch.Tensor:
        """
        Args:
            plucker: [B, 6, T_packed, H_pixel, W_pixel]
                已做首帧重复4次打包：T_packed = 4 + (num_pixel_frames - 1)
                channels 0-2: dxo (moment, 已归一化)
                channels 3-5: d   (direction, 单位向量)
        
        Returns:
            cam_tokens: [B, N_cam, dim]
            N_cam = (T_packed / temporal_k) × (H / spatial_k) × (W / spatial_k)
        """
        # 1. 6 维 Plücker 整体做 3D patch embedding（temporal_k=4 自动做时间压缩）
        feat = self.patch_embed(plucker)  # [B, hidden_dim, f, h, w]
        
        # 2. Flatten 到 token 序列 [B, N, hidden_dim]
        feat = rearrange(feat, 'b c f h w -> b (f h w) c')
        
        # 3. MLP fusion
        feat = feat + self.fusion(feat)  # 残差连接
        
        # 4. 投射到 DiT dim
        cam_tokens = self.out_norm(self.out_proj(feat))  # [B, N, dim]
        
        return cam_tokens * self.scale


def pack_plucker_like_video(plucker_pixel: torch.Tensor) -> torch.Tensor:
    """
    将像素帧级的 Plücker 坐标按 Wan VAE 的打包方式处理。
    
    Wan VAE 的时间打包规则：
      输入 T 帧 → 首帧重复4次 + 后续(T-1)帧 → T_packed = T + 3
      然后 3D Conv(kernel=4, stride=4) 压缩 → F_latent = T_packed / 4

    Args:
        plucker_pixel: [1, 6, T, H, W]  原始像素帧级 Plücker
    
    Returns:
        plucker_packed: [1, 6, T_packed, H, W]  首帧重复4次打包后
    """
    plucker_packed = torch.cat([
        plucker_pixel[:, :, 0:1].repeat(1, 1, 4, 1, 1),  # [1, 6, 4, H, W]
        plucker_pixel[:, :, 1:],                            # [1, 6, T-1, H, W]
    ], dim=2)
    return plucker_packed


def prepare_camera_position_ids(
    num_cam_tokens: int,
    f: int, h: int, w: int,
    device: torch.device,
    dtype: torch.dtype,
    temporal_offset: int = 0,
    spatial_offset: int = 64,
) -> torch.Tensor:
    """为相机 tokens 生成位置 ID，h 维加 offset=64。"""
    cam_ids = []
    for t_idx in range(f):
        for h_idx in range(h):
            for w_idx in range(w):
                cam_ids.append([
                    t_idx + temporal_offset,
                    h_idx + spatial_offset,
                    w_idx,
                ])
    return torch.tensor(cam_ids, device=device, dtype=dtype)


def build_camera_attention_mask(
    n_video: int,
    n_text: int,
    n_cam: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    EasyControl 风格 attention mask。
    video 能看 camera，camera 只看自己。
    """
    total = n_text + n_video + n_cam
    mask = torch.zeros((total, total), device=device, dtype=dtype)
    cam_start = n_text + n_video
    mask[cam_start:, :cam_start] = -1e20
    return mask