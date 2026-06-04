"""
prope_utils.py — Sub-Block Multi-Camera PRoPE 工具模块
放置路径: diffsynth/models/prope_utils.py

核心功能:
  1. 将每个 latent 帧映射到 4 个像素帧相机
  2. 构建 block-diagonal 变换 (4 个投影子块 + 2D RoPE)
  3. 提供 PropeData 对象，可被 SelfAttention 直接使用
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Callable
from functools import partial
from einops import rearrange


# ============================================================
# PropeData: 传递给每个 DiTBlock 的预计算数据
# ============================================================
class PropeData:
    """
    存储预计算的 PRoPE 变换函数。
    
    使用方式:
      prope_data.apply_q(q)   # q: (B, S, num_heads*head_dim)
      prope_data.apply_kv(kv)
      prope_data.apply_o(o)
    
    所有变换都不改变 tensor shape，flash_attention 完全不受影响。
    """
    def __init__(
        self,
        apply_fn_q: Callable,
        apply_fn_kv: Callable,
        apply_fn_o: Callable,
        num_heads: int,
    ):
        self.apply_fn_q = apply_fn_q
        self.apply_fn_kv = apply_fn_kv
        self.apply_fn_o = apply_fn_o
        self.num_heads = num_heads

    def apply_q(self, q_flat: torch.Tensor) -> torch.Tensor:
        """q_flat: (B, S, num_heads * head_dim) → same shape"""
        B, S, D = q_flat.shape
        n, d = self.num_heads, D // self.num_heads
        q = rearrange(q_flat, 'b s (n d) -> b n s d', n=n)
        q = self.apply_fn_q(q)
        return rearrange(q, 'b n s d -> b s (n d)')

    def apply_kv(self, kv_flat: torch.Tensor) -> torch.Tensor:
        B, S, D = kv_flat.shape
        n = self.num_heads
        kv = rearrange(kv_flat, 'b s (n d) -> b n s d', n=n)
        kv = self.apply_fn_kv(kv)
        return rearrange(kv, 'b n s d -> b s (n d)')

    def apply_o(self, o_flat: torch.Tensor) -> torch.Tensor:
        B, S, D = o_flat.shape
        n = self.num_heads
        o = rearrange(o_flat, 'b s (n d) -> b n s d', n=n)
        o = self.apply_fn_o(o)
        return rearrange(o, 'b n s d -> b s (n d)')


# ============================================================
# 主入口: compute_multi_camera_prope_data
# ============================================================
def compute_multi_camera_prope_data(
    pixel_c2ws: torch.Tensor,         # (T_pixel, 4, 4)
    pixel_Ks: torch.Tensor,           # (T_pixel, 3, 3)
    num_latent_frames: int,
    patches_h: int,                    # latent H after patchify
    patches_w: int,                    # latent W after patchify
    image_height: int,
    image_width: int,
    head_dim: int,
    num_heads: int,
    vae_temporal_factor: int = 4,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    freq_base: float = 100.0,
    has_reference_latents: bool = False,  # 是否有 reference image 拼接
) -> PropeData:
    """
    方案 B: Sub-Block Multi-Camera PRoPE
    
    head_dim=128 的分配:
      [0:16)   投影 P_0 (sub-camera 0)
      [16:32)  投影 P_1 (sub-camera 1)
      [32:48)  投影 P_2 (sub-camera 2)
      [48:64)  投影 P_3 (sub-camera 3)
      [64:96)  2D RoPE_x (patch w 坐标)
      [96:128) 2D RoPE_y (patch h 坐标)
    
    Args:
        pixel_c2ws: 所有像素帧的 camera-to-world 矩阵
        pixel_Ks: 所有像素帧的内参
        num_latent_frames: VAE 压缩后的帧数 = (T_pixel-1)//4 + 1
        patches_h/w: patchify 后的空间分辨率
        has_reference_latents: 如果有 reference image，会多一个 "帧"
                               该帧使用 identity 投影矩阵
    """
    assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
    
    T_pixel = pixel_c2ws.shape[0]
    pixel_c2ws = pixel_c2ws.to(device=device, dtype=dtype)
    pixel_Ks = pixel_Ks.to(device=device, dtype=dtype)
    
    n_sub_cameras = vae_temporal_factor  # 4
    proj_dim_total = head_dim // 2       # 64
    proj_dim_per_cam = proj_dim_total // n_sub_cameras  # 16
    
    assert proj_dim_per_cam >= 4, \
        f"proj_dim_per_cam={proj_dim_per_cam} too small (need >=4 for 4x4 matrix)"
    assert proj_dim_per_cam % 4 == 0, \
        f"proj_dim_per_cam={proj_dim_per_cam} must be divisible by 4"
    
    # ─── 1. 相对位姿 ───
    first_w2c = torch.linalg.inv(pixel_c2ws[0])
    rel_c2ws = first_w2c.unsqueeze(0) @ pixel_c2ws  # (T_pixel, 4, 4)
    rel_c2ws[0] = torch.eye(4, device=device, dtype=dtype)
    
    # ─── 2. 为每个 sub-camera 收集逐 latent 帧的投影矩阵 ───
    # cameras = num_latent_frames (可能 +1 如果有 reference)
    cameras = num_latent_frames + (1 if has_reference_latents else 0)
    
    sub_P_list = []     # 每个 sub-camera 的 P    (1, cameras, 4, 4)
    sub_PT_list = []    # P^T
    sub_Pinv_list = []  # P^{-1}
    
    for sub_idx in range(n_sub_cameras):
        latent_w2cs = []
        latent_Ks_list = []
        
        # 如果有 reference_latents，第一个 "camera" 是 identity
        if has_reference_latents:
            latent_w2cs.append(torch.eye(4, device=device, dtype=dtype))
            latent_Ks_list.append(torch.eye(3, device=device, dtype=dtype))
        
        for k in range(num_latent_frames):
            if k == 0:
                # latent 帧 0 = 像素帧 0
                w2c = _invert_SE3_single(rel_c2ws[0])
                latent_w2cs.append(w2c)
                latent_Ks_list.append(pixel_Ks[0])
            else:
                # latent 帧 k → 像素帧 [4(k-1)+1 + sub_idx]
                start = (k - 1) * vae_temporal_factor + 1
                pixel_idx = min(start + sub_idx, T_pixel - 1)
                w2c = _invert_SE3_single(rel_c2ws[pixel_idx])
                latent_w2cs.append(w2c)
                latent_Ks_list.append(pixel_Ks[pixel_idx])
        
        viewmats = torch.stack(latent_w2cs).unsqueeze(0)  # (1, cameras, 4, 4)
        Ks = torch.stack(latent_Ks_list).unsqueeze(0)     # (1, cameras, 3, 3)
        
        P, P_T, P_inv = _compute_projection_matrices(
            viewmats, Ks, image_width, image_height
        )
        sub_P_list.append(P)
        sub_PT_list.append(P_T)
        sub_Pinv_list.append(P_inv)
    
    # ─── 3. 2D RoPE 系数 ───
    # patches_w 变化快 (x 方向), patches_h 变化慢 (y 方向)
    coeffs_x = _rope_precompute_coeffs(
        torch.tile(torch.arange(patches_w, device=device), (patches_h,)),
        freq_base=freq_base, feat_dim=head_dim // 4,
    )
    coeffs_y = _rope_precompute_coeffs(
        torch.repeat_interleave(torch.arange(patches_h, device=device), patches_w),
        freq_base=freq_base, feat_dim=head_dim // 4,
    )
    
    # ─── 4. 构建 block-diagonal 变换 ───
    transforms_q, transforms_kv, transforms_o = [], [], []
    
    for sub_idx in range(n_sub_cameras):
        transforms_q.append((
            partial(_apply_tiled_projmat, matrix=sub_PT_list[sub_idx]),
            proj_dim_per_cam
        ))
        transforms_kv.append((
            partial(_apply_tiled_projmat, matrix=sub_Pinv_list[sub_idx]),
            proj_dim_per_cam
        ))
        transforms_o.append((
            partial(_apply_tiled_projmat, matrix=sub_P_list[sub_idx]),
            proj_dim_per_cam
        ))
    
    # 添加 2D RoPE (Q/K/V 用正向, O 用逆向)
    transforms_q.extend([
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ])
    transforms_kv.extend([
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ])
    transforms_o.extend([
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
    ])
    
    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    
    return PropeData(apply_fn_q, apply_fn_kv, apply_fn_o, num_heads)


# ============================================================
# 底层工具函数
# ============================================================

def _invert_SE3_single(T: torch.Tensor) -> torch.Tensor:
    """单个 4×4 矩阵的 SE(3) 逆。"""
    Rinv = T[:3, :3].T
    out = torch.zeros_like(T)
    out[:3, :3] = Rinv
    out[:3, 3] = -Rinv @ T[:3, 3]
    out[3, 3] = 1.0
    return out


def _invert_SE3(T: torch.Tensor) -> torch.Tensor:
    """批量 SE(3) 逆。T: (..., 4, 4)"""
    Rinv = T[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, T[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """3×3 → 4×4 齐次。"""
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """内参逆（无 skew）。"""
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0].clamp(min=1e-8)
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1].clamp(min=1e-8)
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0].clamp(min=1e-8)
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1].clamp(min=1e-8)
    out[..., 2, 2] = 1.0
    return out


def _compute_projection_matrices(
    viewmats: torch.Tensor,  # (B, cameras, 4, 4) — w2c
    Ks: torch.Tensor,        # (B, cameras, 3, 3)
    image_width: int,
    image_height: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 P, P^T, P^{-1}。"""
    # 归一化内参
    Ks_norm = torch.zeros_like(Ks)
    Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
    Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
    Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
    Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
    Ks_norm[..., 2, 2] = 1.0
    
    P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
    P_T = P.transpose(-1, -2)
    P_inv = torch.einsum(
        "...ij,...jk->...ik",
        _invert_SE3(viewmats),
        _lift_K(_invert_K(Ks_norm)),
    )
    return P, P_T, P_inv


def _apply_tiled_projmat(
    feats: torch.Tensor,   # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, 4, 4)
) -> torch.Tensor:
    """
    对特征施加分块投影矩阵变换。
    feat_dim 被分成 feat_dim//4 组，每组 4 维与 4×4 矩阵相乘。
    """
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    D = matrix.shape[-1]  # 4
    assert seqlen % cameras == 0, \
        f"seqlen ({seqlen}) must be divisible by cameras ({cameras})"
    assert feat_dim % D == 0
    
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix.to(feats.dtype),
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _rope_precompute_coeffs(
    positions: torch.Tensor,  # (seqlen_per_image,)
    freq_base: float = 100.0,
    freq_scale: float = 1.0,
    feat_dim: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE cos/sin。"""
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    angles = positions[None, None, :, None] * freqs
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """施加 RoPE（split 顺序）。自动 repeat 到 cameras 数量。"""
    cos, sin = coeffs
    cos = cos.to(feats.dtype)
    sin = sin.to(feats.dtype)
    # cos/sin: (1, 1, patches_per_image, feat_dim//2)
    # feats:   (batch, heads, seqlen, feat_dim)
    # seqlen = cameras * patches_per_image
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    
    x_in = feats[..., :feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2:]
    
    if not inverse:
        return torch.cat([cos * x_in + sin * y_in, -sin * x_in + cos * y_in], dim=-1)
    else:
        return torch.cat([cos * x_in - sin * y_in, sin * x_in + cos * y_in], dim=-1)


def _apply_block_diagonal(
    feats: torch.Tensor,
    func_size_pairs: List[Tuple[Callable, int]],
) -> torch.Tensor:
    """block-diagonal 变换：将 feat_dim 拆分成多个块，分别施加不同变换。"""
    funcs, block_sizes = zip(*func_size_pairs)
    x_blocks = torch.split(feats, list(block_sizes), dim=-1)
    return torch.cat([f(xb) for f, xb in zip(funcs, x_blocks)], dim=-1)