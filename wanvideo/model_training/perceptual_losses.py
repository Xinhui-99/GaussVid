"""
perceptual_losses.py
感知损失模块：LPIPS (官方库) + MS-SSIM + Focal Frequency Loss + L1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. MS-SSIM Loss (不变)
# ============================================================
def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return (g / g.sum()).unsqueeze(0).unsqueeze(0)


def _ssim_per_channel(x, y, kernel, data_range=1.0, k1=0.01, k2=0.03):
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    ch = x.shape[1]
    kernel = kernel.expand(ch, -1, -1, -1).to(x.device, x.dtype)
    mu_x = F.conv2d(x, kernel, groups=ch, padding="same")
    mu_y = F.conv2d(y, kernel, groups=ch, padding="same")
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_xx = F.conv2d(x * x, kernel, groups=ch, padding="same") - mu_xx
    sigma_yy = F.conv2d(y * y, kernel, groups=ch, padding="same") - mu_yy
    sigma_xy = F.conv2d(x * y, kernel, groups=ch, padding="same") - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
               ((mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2))
    cs_map = (2 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    return ssim_map.mean(dim=[2, 3]), cs_map.mean(dim=[2, 3])


class MS_SSIM_Loss(nn.Module):
    def __init__(self, data_range=1.0, levels=5, kernel_size=11, sigma=1.5):
        super().__init__()
        self.data_range = data_range
        self.levels = levels
        kernel_1d = _fspecial_gauss_1d(kernel_size, sigma)
        self.register_buffer("kernel", kernel_1d[:, :, :, None] * kernel_1d[:, :, None, :])
        self.register_buffer("level_weights",
            torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:levels]))

    def forward(self, pred, target):
        mssim_vals, mcs_vals = [], []
        for i in range(self.levels):
            ssim_val, cs_val = _ssim_per_channel(pred, target, self.kernel, self.data_range)
            mssim_vals.append(ssim_val)
            mcs_vals.append(cs_val)
            if i < self.levels - 1:
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)
        mssim_vals = torch.stack(mssim_vals, dim=-1)
        mcs_vals = torch.stack(mcs_vals, dim=-1)
        w = self.level_weights.to(pred.device)
        ms_ssim = torch.prod(mcs_vals[:, :, :-1] ** w[:-1], dim=-1) * \
                  mssim_vals[:, :, -1] ** w[-1]
        return 1.0 - ms_ssim.mean()


# ============================================================
# 2. Focal Frequency Loss (不变)
# ============================================================
class FocalFrequencyLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        freq_diff = (pred_fft - target_fft).abs()
        weight = freq_diff.detach() ** self.alpha
        weight = weight / (weight.mean() + 1e-8)
        return (weight * freq_diff).mean()


# ============================================================
# 3. 组合感知损失 (LPIPS 替换为官方库)
# ============================================================
class PerceptualLossBundle(nn.Module):
    def __init__(
        self,
        use_lpips=True,
        use_ms_ssim=True,
        use_freq=True,
        use_l1=True,
        lpips_weight=1.0,
        ms_ssim_weight=0.5,
        freq_weight=0.05,
        l1_weight=1.0,
        device="cuda",
    ):
        super().__init__()
        self.lpips_weight = lpips_weight
        self.ms_ssim_weight = ms_ssim_weight
        self.freq_weight = freq_weight
        self.l1_weight = l1_weight
        self.use_l1 = use_l1
        self.use_lpips = use_lpips
        self.use_ms_ssim = use_ms_ssim
        self.use_freq = use_freq

        # ★ 使用官方 lpips 库
        self.lpips_fn = None
        if use_lpips:
            try:
                import lpips as lpips_lib
                self.lpips_fn = lpips_lib.LPIPS(net='vgg').to(device).eval()
                for p in self.lpips_fn.parameters():
                    p.requires_grad = False
                print(f"[PerceptualLoss] Official LPIPS (VGG) on {device}")
            except ImportError:
                print("[PerceptualLoss] lpips not installed, skipping LPIPS")
                self.use_lpips = False

        self.ms_ssim_fn = MS_SSIM_Loss().to(device) if use_ms_ssim else None
        self.freq_fn = FocalFrequencyLoss(alpha=1.0).to(device) if use_freq else None

    def _ensure_lpips_device(self, device):
        """确保 LPIPS 网络在正确设备上"""
        if self.lpips_fn is not None:
            if next(self.lpips_fn.parameters()).device != device:
                self.lpips_fn = self.lpips_fn.to(device)

    def forward(self, pred_pixels, gt_pixels):
        """
        pred_pixels: [B, 3, H, W] in [0, 1], 有梯度
        gt_pixels:   [B, 3, H, W] in [0, 1], 无梯度
        returns: (total_loss, info_dict)

        注意：官方 lpips 库期望输入范围 [-1, 1]，
        所以内部做 [0,1] → [-1,1] 的转换。
        """
        pred_pixels = pred_pixels.clamp(0, 1)
        gt_pixels = gt_pixels.clamp(0, 1).detach()

        losses = {}
        total = torch.tensor(0.0, device=pred_pixels.device, dtype=pred_pixels.dtype)

        # ★ LPIPS: 官方库，输入 [-1, 1]
        if self.use_lpips and self.lpips_fn is not None:
            self._ensure_lpips_device(pred_pixels.device)
            # [0, 1] → [-1, 1]
            pred_lpips = pred_pixels * 2.0 - 1.0   # 有梯度
            gt_lpips = gt_pixels * 2.0 - 1.0       # 无梯度
            l = self.lpips_fn(pred_lpips.float(), gt_lpips.float()).mean()
            losses["lpips"] = l.item()
            total = total + self.lpips_weight * l

        # MS-SSIM
        if self.use_ms_ssim and self.ms_ssim_fn is not None:
            l = self.ms_ssim_fn(pred_pixels, gt_pixels)
            losses["ms_ssim"] = l.item()
            total = total + self.ms_ssim_weight * l 

        # Focal Frequency
        '''if self.use_freq and self.freq_fn is not None:
            l = self.freq_fn(pred_pixels, gt_pixels)
            losses["freq"] = l.item()
            total = total + self.freq_weight * l'''

        # L1
        if self.use_l1:
            l = F.l1_loss(pred_pixels, gt_pixels)
            losses["l1"] = l.item()
            total = total + self.l1_weight * l

        losses["perceptual_total"] = total.item()
        return total, losses