# diffsynth/models/wan_video_gsfixer_feature.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Projector: Linear + Normalization
# 论文里两个 projector 都用 linear + norm 实现
# ---------------------------------------------------------------------------
class FeatureProjector(nn.Module):
    """把外部特征 (VGGT / DINOv2) 投到 DiT 的 hidden dim。

    结构: Linear -> LayerNorm  (可选再加一层 Linear 增强表达)
    """
    def __init__(self, in_dim: int, out_dim: int, use_two_layer: bool = False):
        super().__init__()
        self.use_two_layer = use_two_layer
        if use_two_layer:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
        else:
            self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B, N, in_dim]
        return self.norm(self.proj(x))


# ---------------------------------------------------------------------------
# VGGT 3D 几何特征提取器封装
# ---------------------------------------------------------------------------
class VGGTFeatureExtractor(nn.Module):
    """封装 VGGT，输入首尾帧图像，输出 3D 几何 token。

    输出 token 维度依据 VGGT 的 aggregator 输出 (通常是 2*embed_dim 拼接 或 embed_dim)。
    这里按 VGGT 默认 embed_dim=1024 处理，实际以模型为准。
    """
    def __init__(self, vggt_model=None, freeze: bool = True):
        super().__init__()
        self.vggt = vggt_model
        self.freeze = freeze
        if self.vggt is not None and freeze:
            for p in self.vggt.parameters():
                p.requires_grad = False
            self.vggt.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """
        images: [B, V, 3, H, W]，V=2 (首帧、尾帧)，已归一化到 VGGT 期望的范围
        return: [B, V*N_patch, C_vggt]  3D 几何 token 序列
        """
        if self.vggt is None:
            raise RuntimeError("VGGT model not loaded.")

        # VGGT aggregator 接口: 返回 aggregated_tokens_list, patch_start_idx
        # ★ 对齐 dtype：输入图像转成 VGGT 权重的 dtype
        model_dtype = next(self.vggt.parameters()).dtype
        model_device = next(self.vggt.parameters()).device
        images = images.to(device=model_device, dtype=model_dtype)
        aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images)

        # 这里取最后一层 aggregated token 作为几何特征
        #aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images)
        tokens = aggregated_tokens_list[-1]  # [B, V, N_total, C]

        # 去掉 camera/register token，只保留 patch token
        patch_tokens = tokens[:, :, patch_start_idx:, :]  # [B, V, N_patch, C]

        B, V, N, C = patch_tokens.shape
        patch_tokens = patch_tokens.reshape(B, V * N, C)
        return patch_tokens


# ---------------------------------------------------------------------------
# DINOv2 2D 语义特征提取器封装
# ---------------------------------------------------------------------------
class DINOv2FeatureExtractor(nn.Module):
    """封装 DINOv2，输入首尾帧图像，输出 2D 语义 token。"""
    def __init__(self, dinov2_model=None, freeze: bool = True):
        super().__init__()
        self.dinov2 = dinov2_model
        self.freeze = freeze
        if self.dinov2 is not None and freeze:
            for p in self.dinov2.parameters():
                p.requires_grad = False
            self.dinov2.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """
        images: [B, V, 3, H, W]，V=2
        return: [B, V*N_patch, C_dino]
        """
        if self.dinov2 is None:
            raise RuntimeError("DINOv2 model not loaded.")

        #B, V, C, H, W = images.shape
        #images = images.reshape(B * V, C, H, W)
        model_device = next(self.dinov2.parameters()).device
        model_dtype = next(self.dinov2.parameters()).dtype
        images = images.to(device=model_device, dtype=model_dtype)
        #model_dtype = next(self.dinov2.parameters()).dtype
        #images = images.to(dtype=model_dtype)
        B, V, C, H, W = images.shape
        images = images.reshape(B * V, C, H, W)
        feats = self.dinov2.forward_features(images)

        # DINOv2 forward_features 返回 dict，取 patch token
        feats = self.dinov2.forward_features(images)
        patch_tokens = feats["x_norm_patchtokens"]  # [B*V, N_patch, C_dino]

        N, Cd = patch_tokens.shape[1], patch_tokens.shape[2]
        patch_tokens = patch_tokens.reshape(B, V * N, Cd)
        return patch_tokens


# ---------------------------------------------------------------------------
# Fusion 模块：3D token 与 2D token 各自 project 后相加
# ---------------------------------------------------------------------------
class GSFixerFusion(nn.Module):
    """
    VGGT (3D 几何) + DINOv2 (2D 语义) -> 各自 Projector -> 相加 = fusion token
    """
    def __init__(
        self,
        dim: int,                 # DiT hidden dim
        vggt_dim: int = 1024,     # VGGT 输出维度，按实际模型调整
        dino_dim: int = 1024,     # DINOv2 输出维度 (vitl14=1024)
        use_two_layer_proj: bool = False,
    ):
        super().__init__()
        self.proj_3d = FeatureProjector(vggt_dim, dim, use_two_layer_proj)
        self.proj_2d = FeatureProjector(dino_dim, dim, use_two_layer_proj)

    def forward(self, vggt_tokens: torch.Tensor, dino_tokens: torch.Tensor):
        """
        vggt_tokens: [B, N3d, vggt_dim]
        dino_tokens: [B, N2d, dino_dim]
        两者 token 数需对齐 (同样的 V*N_patch)，相加得到 fusion token
        return: [B, N, dim]
        """
        f3d = self.proj_3d(vggt_tokens)   # [B, N, dim]
        f2d = self.proj_2d(dino_tokens)   # [B, N, dim]

        # token 数对齐：若 patch 数不同则插值对齐到 3D token 的长度
        if f3d.shape[1] != f2d.shape[1]:
            f2d = self._align_tokens(f2d, f3d.shape[1])

        fusion = f3d + f2d
        return fusion

    @staticmethod
    def _align_tokens(x, target_len):
        # x: [B, N, C] -> [B, target_len, C] 用 1d 插值对齐 token 数
        B, N, C = x.shape
        x = x.transpose(1, 2)  # [B, C, N]
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x.transpose(1, 2)