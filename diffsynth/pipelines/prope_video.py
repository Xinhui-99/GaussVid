"""
prope_video.py — 将 PRoPE (Cameras as Relative Positional Encoding) 集成到视频 DiT 中

核心思路：
  - head_dim 分为 [temporal | projection | x_rope | y_rope]
  - temporal 部分保持原有 3D RoPE（complex 乘法）
  - spatial 部分用 PRoPE：projection matrix + 2D RoPE (cos/sin 实数)
  
  对于 VAE 时间压缩（4 pixel frames → 1 latent frame）：
  - latent frame 0 → pixel frame 0
  - latent frame k (k>0) → pixel frame (k-1)*4 + 2 （取中间帧）
"""

import torch
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, List, Callable
from einops import rearrange


# ============================================================
# 1. 为每个 latent frame 选择对应的 camera
# ============================================================

def select_cameras_for_latent_frames(
    Ks: torch.Tensor,       # (T_pixel, 3, 3)
    c2ws: torch.Tensor,     # (T_pixel, 4, 4)
    num_latent_frames: int,
    vae_temporal_factor: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """为每个 latent frame 选择对应 pixel frame 的 camera 参数。
    
    Wan VAE 时间压缩: 第一帧单独，之后每4帧压成1帧
    - latent frame 0 → pixel frame 0
    - latent frame k (k>0) → 取 pixel frame (k-1)*4+1 到 (k-1)*4+4 的中间帧
    
    Returns:
        selected_Ks:  (num_latent_frames, 3, 3)
        selected_c2ws: (num_latent_frames, 4, 4)
    """
    T_pixel = c2ws.shape[0]
    selected_indices = []
    for k in range(num_latent_frames):
        if k == 0:
            selected_indices.append(0)
        else:
            start = (k - 1) * vae_temporal_factor + 1
            mid = start + vae_temporal_factor // 2  # 中间帧
            mid = min(mid, T_pixel - 1)
            selected_indices.append(mid)
    
    selected_Ks = Ks[selected_indices]
    selected_c2ws = c2ws[selected_indices]
    return selected_Ks, selected_c2ws


# ============================================================
# 2. PRoPE 核心计算（从 prope.py 改造为视频版本）
# ============================================================

def _rope_precompute_coeffs(
    positions: torch.Tensor,  # (seqlen,)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE cos/sin 系数。"""
    assert len(positions.shape) == 1
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
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """用 split 方式对 features 施加 RoPE。"""
    cos, sin = coeffs
    cos = cos.to(dtype=feats.dtype, device=feats.device)  # ← 加这行
    sin = sin.to(dtype=feats.dtype, device=feats.device)  # ← 加这行
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    if not inverse:
        return torch.cat([cos * x_in + sin * y_in, -sin * x_in + cos * y_in], dim=-1)
    else:
        return torch.cat([cos * x_in - sin * y_in, sin * x_in + cos * y_in], dim=-1)


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """对 features 施加逐帧 projection matrix。"""
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    D = matrix.shape[-1]
    matrix = matrix.to(dtype=feats.dtype, device=feats.device)  # ← 加这行
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """求 4x4 SE(3) 矩阵的逆。"""
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """3x3 → 齐次 4x4"""
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """3x3 内参矩阵的逆（假设无 skew）。"""
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def _apply_block_diagonal(
    feats: torch.Tensor,
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """对 head_dim 的不同块施加不同的变换函数。"""
    funcs, block_sizes = zip(*func_size_pairs)
    x_blocks = torch.split(feats, list(block_sizes), dim=-1)
    return torch.cat([f(x_block) for f, x_block in zip(funcs, x_blocks)], dim=-1)


# ============================================================
# 3. PropeVideoHelper — 视频级 PRoPE 管理器
# ============================================================

class PropeVideoHelper:
    """管理视频 DiT 中的 PRoPE 变换。
    
    将 head_dim 拆分为:
      [temporal_dim | proj_dim | rope_x_dim | rope_y_dim]
    
    - temporal_dim: 保留原有的 complex RoPE（只作用于 q, k）
    - proj_dim + rope_x_dim + rope_y_dim: 使用 PRoPE（作用于 q, k, v, o）
    
    用法:
        helper = PropeVideoHelper.from_cameras(
            Ks, c2ws, head_dim, num_latent_frames, patches_h, patches_w,
            image_height, image_width
        )
        # 在 self-attention 中:
        q, k, v = helper.apply_to_qkv(q, k, v, temporal_freqs, num_heads)
        o = F.scaled_dot_product_attention(q, k, v)
        o = helper.apply_to_output(o, temporal_freqs, num_heads)
    """
    
    def __init__(
        self,
        apply_fn_q: Callable,
        apply_fn_kv: Callable,
        apply_fn_o: Callable,
        temporal_dim: int,
        spatial_dim: int,
    ):
        self.apply_fn_q = apply_fn_q
        self.apply_fn_kv = apply_fn_kv
        self.apply_fn_o = apply_fn_o
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
    
    @staticmethod
    def from_cameras(
        Ks: torch.Tensor,          # (num_latent_frames, 3, 3)
        c2ws: torch.Tensor,        # (num_latent_frames, 4, 4)
        head_dim: int,
        num_latent_frames: int,
        patches_h: int,
        patches_w: int,
        image_height: int,
        image_width: int,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ) -> "PropeVideoHelper":
        """从 camera 参数构建 PRoPE 变换。
        
        Args:
            Ks: 每个 latent frame 的内参 (f, 3, 3)
            c2ws: 每个 latent frame 的 camera-to-world (f, 4, 4)
            head_dim: attention head 维度
            num_latent_frames: latent 帧数 f
            patches_h: 空间 height 方向的 patch 数
            patches_w: 空间 width 方向的 patch 数
            image_height: 原始图像高度（用于归一化内参）
            image_width: 原始图像宽度（用于归一化内参）
        """
        device = c2ws.device
        dtype = c2ws.dtype
        
        # 维度拆分
        temporal_dim = head_dim - 2 * (head_dim // 3)
        spatial_dim = head_dim - temporal_dim  # = 2 * (head_dim // 3)
        
        # PRoPE 在 spatial_dim 上的分配: proj + x_rope + y_rope
        # proj_dim 必须是 4 的倍数（用于 4x4 projection matrix）

        # from_cameras 方法中，替换原来的三行
        proj_dim = (spatial_dim // 2 // 4) * 4
        remaining = spatial_dim - proj_dim
        rope_x_dim = (remaining // 2 // 2) * 2
        rope_y_dim = remaining - rope_x_dim

        #proj_dim = spatial_dim // 2
        #rope_x_dim = spatial_dim // 4
        #rope_y_dim = spatial_dim - proj_dim - rope_x_dim
        
        assert proj_dim % 4 == 0, f"proj_dim={proj_dim} must be multiple of 4"
        assert rope_x_dim % 2 == 0, f"rope_x_dim={rope_x_dim} must be even"
        assert rope_y_dim % 2 == 0, f"rope_y_dim={rope_y_dim} must be even"
        
        # --- 计算 viewmats (world-to-camera) ---
        # c2ws 是 camera-to-world，viewmats = c2ws^{-1} 是 world-to-camera
        viewmats = torch.linalg.inv(c2ws)  # (f, 4, 4)
        viewmats = viewmats.unsqueeze(0).to(dtype=torch.float32)  # (1, f, 4, 4)
        Ks_batch = Ks.unsqueeze(0).to(dtype=torch.float32)         # (1, f, 3, 3)
        
        f = num_latent_frames
        
        # --- 归一化内参 ---
        Ks_norm = torch.zeros_like(Ks_batch)
        Ks_norm[..., 0, 0] = Ks_batch[..., 0, 0] / image_width
        Ks_norm[..., 1, 1] = Ks_batch[..., 1, 1] / image_height
        Ks_norm[..., 0, 2] = Ks_batch[..., 0, 2] / image_width - 0.5
        Ks_norm[..., 1, 2] = Ks_batch[..., 1, 2] / image_height - 0.5
        Ks_norm[..., 2, 2] = 1.0
        
        # --- 计算 projection matrices ---
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        )
        # P, P_T, P_inv shape: (1, f, 4, 4)
        
        # --- 预计算空间 RoPE (x 方向和 y 方向) ---
        # 每帧内 patches_h * patches_w 个 token
        # x 方向: 0, 1, ..., patches_w-1 重复 patches_h 次
        # y 方向: 0, 0, ..., 0, 1, 1, ..., 1, ... 每个值重复 patches_w 次
        # 跨帧重复 f 次
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_w, device=device), (patches_h * f,)),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=rope_x_dim,
        )
        coeffs_y = _rope_precompute_coeffs(
            torch.tile(
                torch.repeat_interleave(
                    torch.arange(patches_h, device=device), patches_w
                ),
                (f,),
            ),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=rope_y_dim,
        )
        
        # --- 构建 block-diagonal transforms ---
        transforms_q = [
            (partial(_apply_tiled_projmat, matrix=P_T), proj_dim),
            (partial(_rope_apply_coeffs, coeffs=coeffs_x), rope_x_dim),
            (partial(_rope_apply_coeffs, coeffs=coeffs_y), rope_y_dim),
        ]
        transforms_kv = [
            (partial(_apply_tiled_projmat, matrix=P_inv), proj_dim),
            (partial(_rope_apply_coeffs, coeffs=coeffs_x), rope_x_dim),
            (partial(_rope_apply_coeffs, coeffs=coeffs_y), rope_y_dim),
        ]
        transforms_o = [
            (partial(_apply_tiled_projmat, matrix=P), proj_dim),
            (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), rope_x_dim),
            (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), rope_y_dim),
        ]
        
        apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
        apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
        apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
        
        return PropeVideoHelper(
            apply_fn_q=apply_fn_q,
            apply_fn_kv=apply_fn_kv,
            apply_fn_o=apply_fn_o,
            temporal_dim=temporal_dim,
            spatial_dim=spatial_dim,
        )
    
    def apply_to_qkv(
        self,
        q: torch.Tensor,       # (B, S, N*D) 或 (B, N, S, D)
        k: torch.Tensor,
        v: torch.Tensor,
        temporal_freqs,         # complex RoPE freqs for temporal dim
        num_heads: int,
        input_format: str = "bsnd",  # "bsnd" = (B, S, N*D), "bnsd" = (B, N, S, D)
    ):
        """对 q, k, v 施加 temporal RoPE + PRoPE spatial 变换。
        
        返回格式: (B, N, S, D) — 适合直接传入 scaled_dot_product_attention
        """
        if input_format == "bsnd":
            q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        
        d_t = self.temporal_dim
        
        # 拆分 temporal 和 spatial
        q_t, q_s = q[..., :d_t], q[..., d_t:]
        k_t, k_s = k[..., :d_t], k[..., d_t:]
        v_t, v_s = v[..., :d_t], v[..., d_t:]
        
        # temporal RoPE (complex) — 只作用于 q, k
        q_t = self._apply_temporal_rope(q_t, temporal_freqs)
        k_t = self._apply_temporal_rope(k_t, temporal_freqs)
        # v_t 不变
        
        # PRoPE spatial — 作用于 q, k, v
        q_s = self.apply_fn_q(q_s)
        k_s = self.apply_fn_kv(k_s)
        v_s = self.apply_fn_kv(v_s)
        
        # 拼接
        q = torch.cat([q_t, q_s], dim=-1)
        k = torch.cat([k_t, k_s], dim=-1)
        v = torch.cat([v_t, v_s], dim=-1)
        
        return q, k, v
    
    def apply_to_output(
        self,
        o: torch.Tensor,       # (B, N, S, D)
        temporal_freqs,         # 不实际使用，temporal 部分不需要逆变换
        num_heads: int,
        output_format: str = "bsnd",  # 输出格式
    ):
        """对 attention output 施加 PRoPE 逆变换（仅 spatial 部分）。"""
        d_t = self.temporal_dim
        
        o_t, o_s = o[..., :d_t], o[..., d_t:]
        
        # temporal 部分不做逆变换
        # spatial 部分做 PRoPE 逆变换
        o_s = self.apply_fn_o(o_s)
        
        o = torch.cat([o_t, o_s], dim=-1)
        
        if output_format == "bsnd":
            o = rearrange(o, "b n s d -> b s (n d)")
        
        return o
    
    @staticmethod
    def _apply_temporal_rope(x, temporal_freqs):
        B, N, S, d = x.shape
        orig_dtype = x.dtype
        # 用 float32 替代 float64，减少内存和计算开销
        x_complex = torch.view_as_complex(
            x.to(torch.float32).reshape(B, N, S, d // 2, 2)  # float64 → float32
        )
        freqs = temporal_freqs.permute(1, 0, 2).unsqueeze(0)
        if "npu" in str(freqs.device):
            freqs = freqs.to(torch.complex64)
        else:
            freqs = freqs.to(torch.complex64)  # 匹配 float32 的 complex 类型
        x_out = torch.view_as_real(x_complex * freqs).reshape(B, N, S, d)
        return x_out.to(orig_dtype)

# ============================================================
# 4. 构建 temporal-only freqs（只包含时间维度的 RoPE）
# ============================================================

def build_temporal_only_freqs(
    dit_freqs,      # dit.freqs — 预计算的 (f_freqs, h_freqs, w_freqs) complex tensors
    f: int, h: int, w: int,
    temporal_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """只提取 temporal 维度的 complex RoPE freqs。
    
    原始 3D freqs 拼接为 [temporal | height | width]。
    这里只取 temporal 部分。
    
    Returns:
        (f*h*w, 1, temporal_dim//2) complex tensor
    """
    # dit_freqs[0] 是 temporal freqs, shape (end, dim//2) complex
    # 对于 temporal_dim，需要 temporal_dim//2 个 complex 频率
    temporal_half_dim = temporal_dim // 2
    
    # dit.freqs[0][:f] shape: (f, temporal_half_dim) — 但原始 freqs 可能维度不完全匹配
    # 原始计算: precompute_freqs_cis(dim - 2*(dim//3), end, theta)
    # dim - 2*(dim//3) = temporal_rope_full_dim (每个 head 的 temporal 维度)
    # 取前 temporal_half_dim 个频率
    
    f_cis = dit_freqs[0][:f, :temporal_half_dim]  # (f, temporal_half_dim) complex
    
    # 扩展到 (f, h, w, temporal_half_dim)
    temporal_freqs = f_cis.view(f, 1, 1, -1).expand(f, h, w, -1)
    
    # reshape 到 (f*h*w, 1, temporal_half_dim)
    temporal_freqs = temporal_freqs.reshape(f * h * w, 1, -1).to(device)
    
    return temporal_freqs


# ============================================================
# 5. 修改后的 Self-Attention forward（PRoPE 版本）
# ============================================================

def prope_self_attention_forward(
    self_attn,                  # SelfAttention module
    x: torch.Tensor,           # (B, S, dim)
    freqs,                      # 原始 3D freqs（用于提取 temporal 部分）
    prope_helper: PropeVideoHelper,
    attn_bias=None,
):
    """用 PRoPE 替代原始空间 RoPE 的 self-attention forward。
    
    保持 temporal RoPE 不变，空间部分用 PRoPE。
    """
    q = self_attn.norm_q(self_attn.q(x))
    k = self_attn.norm_k(self_attn.k(x))
    v = self_attn.v(x)
    
    num_heads = self_attn.num_heads
    
    # 用 PRoPE helper 处理 q, k, v
    # 这里 freqs 只用于 temporal 部分
    q, k, v = prope_helper.apply_to_qkv(
        q, k, v, freqs, num_heads, input_format="bsnd"
    )
    # 返回 (B, N, S, D) 格式
    
    if attn_bias is not None:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    else:
        o = F.scaled_dot_product_attention(q, k, v)
    
    # 对 output 施加 PRoPE 逆变换
    o = prope_helper.apply_to_output(o, freqs, num_heads, output_format="bsnd")
    
    return self_attn.o(o)


# ============================================================
# 6. PRoPE 版本的 DiTBlock forward
# ============================================================

def prope_dit_block_forward(
    block,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    temporal_freqs,
    prope_helper: PropeVideoHelper,
    attn_bias=None,
):
    """使用 PRoPE 的 DiTBlock forward。
    
    与原始 DiTBlock.forward 相同的结构，只是 self_attn 部分换成 PRoPE 版本。
    """
    from diffsynth.models.wan_video_dit_easy import modulate
    
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
    ).chunk(6, dim=chunk_dim)
    
    if has_seq:
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        shift_mlp = shift_mlp.squeeze(2)
        scale_mlp = scale_mlp.squeeze(2)
        gate_mlp = gate_mlp.squeeze(2)
    
    # Self-Attention with PRoPE
    input_x = modulate(block.norm1(x), shift_msa, scale_msa)
    attn_out = prope_self_attention_forward(
        block.self_attn, input_x, temporal_freqs, prope_helper, attn_bias
    )
    x = block.gate(x, gate_msa, attn_out)
    
    # Cross-Attention (unchanged)
    x = x + block.cross_attn(block.norm3(x), context)
    
    # FFN (unchanged)
    input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
    x = block.gate(x, gate_mlp, block.ffn(input_x))
    
    return x


# ============================================================
# 7. 从 pipeline 参数构建 PropeVideoHelper 的便捷函数
# ============================================================

def build_prope_helper_from_vace_camera(
    vace_camera: dict,          # {"Ks": (T,3,3), "c2ws": (T,4,4)}
    head_dim: int,
    num_latent_frames: int,
    patches_h: int,
    patches_w: int,
    image_height: int,
    image_width: int,
    vae_temporal_factor: int = 4,
    device: torch.device = None,
    use_relative_pose: bool = True,
) -> PropeVideoHelper:
    """从 vace_camera 参数构建 PropeVideoHelper。
    
    包括：
    1. 选择每个 latent frame 对应的 camera
    2. 可选：转换为相对 pose
    3. 构建 PRoPE 变换
    """
    Ks = vace_camera["Ks"].to(dtype=torch.float32)
    c2ws = vace_camera["c2ws"].to(dtype=torch.float32)
    
    if device is not None:
        Ks = Ks.to(device=device)
        c2ws = c2ws.to(device=device)
    
    # 相对 pose：以第一帧为参考
    if use_relative_pose:
        first_w2c = torch.linalg.inv(c2ws[0])
        c2ws_rel = first_w2c.unsqueeze(0) @ c2ws
        c2ws_rel[0] = torch.eye(4, device=c2ws.device, dtype=c2ws.dtype)
        c2ws = c2ws_rel
    
    # 选择每个 latent frame 的 camera
    sel_Ks, sel_c2ws = select_cameras_for_latent_frames(
        Ks, c2ws, num_latent_frames, vae_temporal_factor
    )
    
    # 构建 PRoPE helper
    helper = PropeVideoHelper.from_cameras(
        Ks=sel_Ks,
        c2ws=sel_c2ws,
        head_dim=head_dim,
        num_latent_frames=num_latent_frames,
        patches_h=patches_h,
        patches_w=patches_w,
        image_height=image_height,
        image_width=image_width,
    )
    
    return helper