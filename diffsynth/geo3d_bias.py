"""
Geo3DAnchorAttentionBias: 3D-Anchored Hierarchical Attention Bias Module

基于 PEF 分层思想 + 静态场景 3D 刚体先验的 Attention Bias 模块。
用于 WAN2.1 视频去噪训练，将清晰参考帧的纹理通过 3D 距离引导注入噪声帧。

可训练参数: 仅 80 个标量 (num_heads × 2)
  - log_sigma: [H] 控制每个 head 的空间感受野
  - bias_scale: [H] 控制每个 head 的 bias 强度

使用方式:
  module = Geo3DAnchorAttentionBias(num_heads=40)

  # Phase 1: 最小验证 (cross-bias only, ~4GB)
  bias = module.forward_cross_only(noisy_coords, ref_coords)

  # Phase 2: 高效训练 (sparse self + full cross, ~5-6GB)
  bias = module.forward_efficient(noisy_coords, ref_coords, frame_ids)
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class Geo3DAnchorAttentionBias(nn.Module):
    """
    分层 3D 距离 Attention Bias。

    理论:
      B(p,q) = -||P_world(p) - P_world(q)||² / (2σ²) × scale

      不同 head 使用不同 σ，实现频率分解:
        Group A (25%): σ ∈ [0.01, 0.10] → 纹理复制 (高频)
        Group B (50%): σ ∈ [0.20, 1.50] → 几何一致 (中频)
        Group C (25%): σ ∈ [5.00, 10.0] → 全局上下文 (低频, bias≈0)

    Args:
        num_heads: int, attention head 数量 (WAN2.1-14B: 40)
        num_groups: int, 频率分组数量 (默认 3)
    """

    def __init__(self, num_heads: int = 40, num_groups: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups

        # ======== 分层 σ 初始化 ========
        sigmas = torch.ones(num_heads)
        h_per_group = num_heads // num_groups

        # Group A: Texture Copying Heads (delta-like, 点对点复制)
        n_a = h_per_group
        sigmas[:n_a] = torch.linspace(0.01, 0.10, n_a)

        # Group B: Geometric Consistency Heads (局部平滑)
        n_b = h_per_group
        sigmas[n_a:n_a + n_b] = torch.linspace(0.20, 1.50, n_b)

        # Group C: Global Context Heads (近零 bias, 保持原始 attention)
        n_c = num_heads - n_a - n_b
        sigmas[n_a + n_b:] = torch.linspace(5.0, 10.0, n_c)

        # log 空间参数化 (保证正数, 梯度更稳定)
        self.log_sigma = nn.Parameter(torch.log(sigmas))

        # 每个 head 的 bias 强度缩放
        self.bias_scale = nn.Parameter(torch.ones(num_heads))

        # 记录分组信息 (用于 logging)
        self._group_boundaries = (n_a, n_a + n_b)

    @property
    def sigma(self):
        """当前 σ 值 (用于 logging)"""
        return self.log_sigma.exp()

    def _gaussian_bias(self, coords_q: torch.Tensor, coords_k: torch.Tensor
                       ) -> torch.Tensor:
        """
        计算高斯距离 bias。

        Args:
            coords_q: [B, N_q, 3]
            coords_k: [B, N_k, 3]

        Returns:
            bias: [B, H, N_q, N_k]
        """
        # 距离平方: [B, N_q, N_k]
        # 使用 float32 计算避免 bfloat16 精度问题
        cq = coords_q.float()
        ck = coords_k.float()
        diff = cq.unsqueeze(2) - ck.unsqueeze(1)  # [B, N_q, N_k, 3]
        dist_sq = (diff ** 2).sum(-1)               # [B, N_q, N_k]

        # σ² 和 scale: [H] → [1, H, 1, 1]
        sigma = self.log_sigma.exp().float()
        sigma_sq = (sigma ** 2).view(1, -1, 1, 1)
        scale = self.bias_scale.float().view(1, -1, 1, 1)

        # Gaussian kernel: -d²/(2σ²) × scale
        bias = -dist_sq.unsqueeze(1) / (2 * sigma_sq) * scale

        return bias.to(coords_q.dtype)

    # ================================================================
    #  Phase 1: Cross-Bias Only (最小验证, ~4GB)
    # ================================================================
    def forward_cross_only(self, noisy_coords: torch.Tensor,
                           ref_coords: torch.Tensor) -> torch.Tensor:
        """
        只计算 noisy → ref 的 cross-bias，self-bias 为零。

        适用: Phase 1 最小验证，验证 bias 能否引导模型利用参考帧。

        Args:
            noisy_coords: [B, N_noisy, 3]  (e.g., [1, 11200, 3])
            ref_coords:   [B, N_ref, 3]    (e.g., [1, 4480, 3])

        Returns:
            bias: [B, H, N_noisy, N_ref + N_noisy]
                  = [cross_bias(→ref) | zeros(→self)]
                  列顺序对应 K = [K_ref, K_noisy]

        显存: N_noisy × N_ref × H × 2B ≈ 11200 × 4480 × 40 × 2B ≈ 4.0 GB
        """
        B, N_noisy, _ = noisy_coords.shape
        N_ref = ref_coords.shape[1]

        # Cross-bias: noisy(Q) → ref(K)
        cross_bias = self._gaussian_bias(noisy_coords, ref_coords)
        # [B, H, N_noisy, N_ref]

        # Self-bias: 全零
        self_bias = torch.zeros(
            B, self.num_heads, N_noisy, N_noisy,
            device=noisy_coords.device, dtype=noisy_coords.dtype
        )

        # 拼接顺序: [ref | noisy] 对应 K 的拼接顺序
        return torch.cat([cross_bias, self_bias], dim=-1)

    # ================================================================
    #  Phase 2: Efficient Sparse (self+cross, ~5-6GB)
    # ================================================================
    def forward_efficient(self, noisy_coords: torch.Tensor,
                          ref_coords: torch.Tensor,
                          noisy_frame_ids: torch.Tensor,
                          max_frame_gap: int = 1) -> torch.Tensor:
        """
        稀疏 self-bias (帧内 + 相邻帧) + 完整 cross-bias。

        远距离帧之间 3D 距离大 → Gaussian bias ≈ 0，可安全省略。

        Args:
            noisy_coords:    [B, N_noisy, 3]
            ref_coords:      [B, N_ref, 3]
            noisy_frame_ids: [N_noisy] 每个 token 的 frame index
            max_frame_gap:   只计算帧间距 ≤ 此值的 self-bias

        Returns:
            bias: [B, H, N_noisy, N_ref + N_noisy]

        显存: ~5-6 GB (大部分来自 cross_bias)
        """
        B, N_noisy, _ = noisy_coords.shape
        device = noisy_coords.device
        dtype = noisy_coords.dtype

        # --- 完整 Cross-bias ---
        cross_bias = self._gaussian_bias(noisy_coords, ref_coords)

        # --- 稀疏 Self-bias ---
        self_bias = torch.zeros(
            B, self.num_heads, N_noisy, N_noisy,
            device=device, dtype=dtype
        )

        unique_frames = noisy_frame_ids.unique().tolist()

        for i, fi in enumerate(unique_frames):
            idx_i = (noisy_frame_ids == fi).nonzero(as_tuple=True)[0]
            coords_i = noisy_coords[:, idx_i]

            for j, fj in enumerate(unique_frames):
                if abs(i - j) > max_frame_gap:
                    continue

                idx_j = (noisy_frame_ids == fj).nonzero(as_tuple=True)[0]
                coords_j = noisy_coords[:, idx_j]

                block_bias = self._gaussian_bias(coords_i, coords_j)

                # 写入稀疏矩阵的对应块
                i_start, i_end = idx_i[0].item(), idx_i[-1].item() + 1
                j_start, j_end = idx_j[0].item(), idx_j[-1].item() + 1
                self_bias[:, :, i_start:i_end, j_start:j_end] = block_bias

        return torch.cat([cross_bias, self_bias], dim=-1)

    # ================================================================
    #  Logging / Debug
    # ================================================================
    def log_sigma_stats(self) -> dict:
        """返回当前 σ 统计信息（用于 TensorBoard）"""
        s = self.sigma.detach()
        n_a, n_ab = self._group_boundaries
        return {
            "sigma/group_a_mean": s[:n_a].mean().item(),
            "sigma/group_a_min": s[:n_a].min().item(),
            "sigma/group_b_mean": s[n_a:n_ab].mean().item(),
            "sigma/group_c_mean": s[n_ab:].mean().item(),
            "scale/mean": self.bias_scale.detach().mean().item(),
            "scale/std": self.bias_scale.detach().std().item(),
        }


# ================================================================
#  辅助函数
# ================================================================

def build_frame_ids(num_latent_frames: int, tokens_per_frame: int,
                    device: str = 'cuda') -> torch.Tensor:
    """
    构建 noisy token 的 frame id 向量。

    Args:
        num_latent_frames: latent 帧数 (e.g., 5)
        tokens_per_frame: 每帧 token 数 (e.g., 35 × 64 = 2240)

    Returns:
        frame_ids: [num_latent_frames × tokens_per_frame] long tensor
    """
    return torch.arange(num_latent_frames, device=device).repeat_interleave(
        tokens_per_frame
    )


def pad_bias_for_ref_queries(bias: torch.Tensor, n_ref_tokens: int
                              ) -> torch.Tensor:
    """
    在 bias 矩阵上方补零行，使其匹配完整序列 [ref, noisy] 的 Q 维度。

    WAN2.1 将 ref latents prepend 到序列前面:
      x = [ref_tokens, noisy_tokens]

    但 bias 只对 noisy_tokens 的 Q 计算了值。
    需要补零行让 ref_tokens 作为 Q 时 bias=0（不影响其原始 attention）。

    Args:
        bias: [B, H, N_noisy, N_total] 原始 bias (只有 noisy 的 Q 维度)
        n_ref_tokens: int, ref token 数量

    Returns:
        padded_bias: [B, H, N_noisy + N_ref, N_total]
    """
    if n_ref_tokens == 0:
        return bias

    B, H, N_q, N_k = bias.shape
    zero_rows = torch.zeros(
        B, H, n_ref_tokens, N_k,
        device=bias.device, dtype=bias.dtype
    )
    return torch.cat([zero_rows, bias], dim=2)