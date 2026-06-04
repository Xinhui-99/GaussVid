import torch
import math
from typing import Optional, Tuple
 
 
# ═══════════════════════════════════════════════════════════════════════
# 新建文件: models/camera_rope_utils.py
# ═══════════════════════════════════════════════════════════════════════
 
def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    3x3 旋转矩阵 → axis-angle 表示 [ax*θ, ay*θ, az*θ]
    
    为什么不用 Euler 角？
    ─────────────────────
    Euler 角 (yaw, pitch, roll) 存在万向节死锁 (Gimbal Lock)：
    当 pitch = ±90° 时，yaw 和 roll 的旋转轴重合，
    三个角度只能描述两个自由度。在此奇异点附近，
    微小的旋转变化导致角度值剧烈跳变，不适合作为连续的 RoPE 位置。
    
    为什么不用四元数？
    ─────────────────
    四元数有反对称性：q 和 -q 表示同一旋转。
    作为 RoPE 位置时，这会导致"相同位姿"产生不同的频率编码。
    
    为什么选择 axis-angle？
    ─────────────────────
    axis-angle 在 θ∈(0,π) 时是 SO(3)→R³ 的连续单射映射。
    唯一退化点 θ=0 处，axis·θ→0，作为位置编码值恰好正确。
    视频帧间的相机运动通常是小角度旋转，远离 θ=π 的退化区。
    
    Args:
        R: [..., 3, 3] 旋转矩阵
    Returns:
        [..., 3] axis-angle 向量
    """
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    
    # Rodrigues: trace(R) = 1 + 2cos(θ)
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    cos_angle = ((trace - 1) / 2).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)  # [N]
    sin_angle = torch.sin(angle).clamp(min=1e-7)
    
    # 旋转轴 = (R - R^T) 的反对称部分 / (2 sin θ)
    ax = (R_flat[:, 2, 1] - R_flat[:, 1, 2]) / (2 * sin_angle)
    ay = (R_flat[:, 0, 2] - R_flat[:, 2, 0]) / (2 * sin_angle)
    az = (R_flat[:, 1, 0] - R_flat[:, 0, 1]) / (2 * sin_angle)
    
    axis_angle = torch.stack([ax * angle, ay * angle, az * angle], dim=-1)
    axis_angle[angle < 1e-6] = 0.0  # 单位旋转 → 零向量
    
    return axis_angle.reshape(*batch_shape, 3)
 
 
def compute_camera_temporal_freqs(
    c2ws: torch.Tensor,
    Ks: torch.Tensor,
    num_latent_frames: int,
    temporal_rope_dim: int,
    vae_temporal_factor: int = 4,
    theta: float = 10000.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.float64,
    aggregation: str = "center_weighted",
) -> torch.Tensor:
    """
    计算相机感知的时间维 RoPE 频率，正确处理 VAE 4:1 压缩。
    
    ┌─────────────────────────────────────────────────────┐
    │ 像素帧:  [0] [1 2 3 4] [5 6 7 8] ... [T-4..T-1]   │
    │            ↓     ↓         ↓              ↓         │
    │ Latent帧: [0]   [1]       [2]    ...    [f-1]       │
    │                                                      │
    │ 每个 latent 帧聚合 4 个像素帧的相机状态              │
    │ weights = [0.1, 0.4, 0.4, 0.1] (中心加权)           │
    └─────────────────────────────────────────────────────┘
    
    聚合策略的物理意义：
    ──────────────────
    VAE 的 3D 卷积核 (temporal kernel_size=4) 对输入帧进行加权求和。
    卷积核的权重分布通常近似高斯，中心帧贡献最大。
    
    因此 latent 帧的 "等效相机位姿" 应是 4 帧的加权平均：
      cam_latent[k] = Σ_i w_i · cam_pixel[4(k-1)+1+i]
    
    其中 w = [0.1, 0.4, 0.4, 0.1] 近似高斯核。
    
    这确保了 RoPE 编码的相机位置与 VAE latent 实际承载的
    视觉内容在几何上一致。
    
    Args:
        c2ws:               [T_pixel, 4, 4] camera-to-world 矩阵
        Ks:                 [T_pixel, 3, 3] 内参矩阵
        num_latent_frames:  latent 帧数 f = (T_pixel-1)//4 + 1
        temporal_rope_dim:  时间维 RoPE 维度
        vae_temporal_factor: VAE 时间压缩率 (默认 4)
        theta:              RoPE 基础频率
        device:             目标设备
        dtype:              计算精度
        aggregation:        聚合方式
    
    Returns:
        [num_latent_frames, temporal_rope_dim//2] 复数频率张量
        可直接替换 dit.freqs[0][:f] 使用
    """
    T_pixel = c2ws.shape[0]
    c2ws = c2ws.to(dtype=dtype, device=device)
    
    # ─── Step 1: 计算相对位姿 ───
    # 理论：相对编码消除全局坐标系偏好，
    # 使模型学到的几何关系在任意参考系下成立。
    first_w2c = torch.linalg.inv(c2ws[0])
    rel_c2ws = first_w2c.unsqueeze(0) @ c2ws  # [T, 4, 4]
    rel_c2ws[0] = torch.eye(4, dtype=dtype, device=device)
    
    # ─── Step 2: 提取逐帧 6D 相机状态 ───
    # 6D = [axis_angle_x, axis_angle_y, axis_angle_z, tx, ty, tz]
    all_positions = []
    for t in range(T_pixel):
        rot_aa = rotation_matrix_to_axis_angle(rel_c2ws[t, :3, :3])  # [3]
        trans = rel_c2ws[t, :3, 3]  # [3]
        all_positions.append(torch.cat([rot_aa, trans]))  # [6]
    all_positions = torch.stack(all_positions)  # [T_pixel, 6]
    
    # ─── Step 3: VAE 压缩映射 + 加权聚合 ───
    weight_map = {
        "center_weighted": [0.1, 0.4, 0.4, 0.1],
        "mean":            [0.25, 0.25, 0.25, 0.25],
        "center_frame":    [0.0, 0.0, 1.0, 0.0],
    }
    weights = torch.tensor(weight_map[aggregation], dtype=dtype, device=device)
    
    latent_positions = []
    for k in range(num_latent_frames):
        if k == 0:
            # latent 帧 0 = 参考帧，位置 = 零向量
            latent_positions.append(torch.zeros(6, dtype=dtype, device=device))
        else:
            # latent 帧 k ← 像素帧 [4(k-1)+1, ..., 4k]
            start = (k - 1) * vae_temporal_factor + 1
            indices = [min(start + i, T_pixel - 1) for i in range(vae_temporal_factor)]
            group = all_positions[indices]  # [4, 6]
            aggregated = (group * weights.unsqueeze(1)).sum(dim=0)  # [6]
            latent_positions.append(aggregated)
    
    latent_positions = torch.stack(latent_positions)  # [num_latent_frames, 6]
    
    # ─── Step 4: 6D 位置 → RoPE 复数频率 ───
    # 将 temporal_rope_dim//2 个频率通道分配给 6 个参数
    # 每个参数 p 的通道数 ≈ half_dim / 6
    # 频率计算: freq[k, j] = exp(i · position[k, p] · θ_j)
    half_dim = temporal_rope_dim // 2
    dims_per_param = half_dim // 6
    remainder = half_dim - dims_per_param * 6
    
    all_freqs = []
    for p in range(6):
        n = dims_per_param + (1 if p < remainder else 0)
        if n == 0:
            continue
        # 每个参数有独立的频率基
        base = 1.0 / (theta ** (torch.arange(0, n, dtype=dtype, device=device) / n))
        positions = latent_positions[:, p]  # [num_latent_frames]
        angles = torch.outer(positions, base)  # [num_latent_frames, n]
        all_freqs.append(torch.polar(torch.ones_like(angles), angles))
    
    freqs_cis = torch.cat(all_freqs, dim=-1)  # [num_latent_frames, half_dim]
    return freqs_cis
 