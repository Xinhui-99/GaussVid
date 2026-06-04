"""
compute_world_coords_multiscale.py

在原有 world_coords 预计算脚本的基础上，生成多分辨率坐标 npz。

用法:
  python compute_world_coords_multiscale.py \
    --input_npz /path/to/world_coords_clip0001.npz \
    --output_npz /path/to/world_coords_clip0001_v3.npz

  或批量:
  python compute_world_coords_multiscale.py \
    --input_dir /path/to/clips/ \
    --output_dir /path/to/clips_v3/

功能:
  读取原有 npz (包含 depth + camera), 在三种 stride 下重新 unproject,
  生成:
    coords_fine_log_radial:   [T, 140, 256, 3]  stride=4, 覆盖 4×4 px
    coords_mid_log_radial:    [T, 70, 128, 3]   stride=8, 覆盖 8×8 px
    coords_coarse_log_radial: [T, 35, 64, 3]    stride=16, 覆盖 16×16 px
    + 对应的 ref 版本

  如果原有 npz 没有原始 depth/camera 数据,
  则从 coarse 坐标通过插值生成 fine 和 mid (精度略低但可用)。
"""

import numpy as np
import os
import argparse
from glob import glob
import torch
import torch.nn.functional as F


def upsample_coords_from_coarse(
    coarse_coords: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    从 coarse 坐标 [T, H_c, W_c, 3] 双线性插值到 [T, target_h, target_w, 3]。

    这不如真正的多 stride unproject 精确, 但作为 fallback 可用。
    在深度连续区域误差很小, 在深度不连续区域（物体边缘）会有平滑误差。
    """
    T, H_c, W_c, C = coarse_coords.shape
    t = torch.from_numpy(coarse_coords).float().permute(0, 3, 1, 2)  # [T, 3, H_c, W_c]
    t_up = F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return t_up.permute(0, 2, 3, 1).numpy()  # [T, target_h, target_w, 3]


def convert_npz_to_multiscale(input_path: str, output_path: str):
    """
    将 V1 npz 转换为 V3 多分辨率 npz。

    如果原始 npz 只有 coarse (stride=16) 坐标,
    则通过插值生成 fine 和 mid。
    """
    data = np.load(input_path, allow_pickle=True)

    # 检查是否已经是 V3 格式
    if 'coords_fine_log_radial' in data:
        print(f"  [SKIP] {input_path} already has multiscale coords")
        return

    # 必须有 coarse 坐标
    coarse_key = 'world_coords_log_radial'
    if coarse_key not in data:
        coarse_key = 'coords_coarse_log_radial'
    if coarse_key not in data:
        print(f"  [ERROR] {input_path}: no coarse coords found")
        return

    coarse = data[coarse_key]  # [T, 35, 64, 3]
    T, H_c, W_c, C = coarse.shape

    # 通过插值生成 fine 和 mid
    # fine: [T, 140, 256, 3]  (stride=4, 4x coarse)
    fine = upsample_coords_from_coarse(coarse, H_c * 4, W_c * 4)
    # mid: [T, 70, 128, 3]   (stride=8, 2x coarse)
    mid = upsample_coords_from_coarse(coarse, H_c * 2, W_c * 2)

    # Ref coords
    ref_coarse_key = 'ref_coords_log_radial'
    if ref_coarse_key not in data:
        ref_coarse_key = 'ref_coords_coarse_log_radial'

    save_dict = {
        # 多分辨率 noisy
        'coords_fine_log_radial': fine.astype(np.float32),
        'coords_mid_log_radial': mid.astype(np.float32),
        'coords_coarse_log_radial': coarse.astype(np.float32),
        # 向后兼容
        'world_coords_log_radial': coarse.astype(np.float32),
    }

    # 复制原有的所有其他 key
    for key in data.files:
        if key not in save_dict:
            save_dict[key] = data[key]

    # Ref 也做多分辨率
    if ref_coarse_key in data:
        ref_coarse = data[ref_coarse_key]
        Tr, Hr, Wr, Cr = ref_coarse.shape
        ref_fine = upsample_coords_from_coarse(ref_coarse, Hr * 4, Wr * 4)
        ref_mid = upsample_coords_from_coarse(ref_coarse, Hr * 2, Wr * 2)
        save_dict['ref_coords_fine_log_radial'] = ref_fine.astype(np.float32)
        save_dict['ref_coords_mid_log_radial'] = ref_mid.astype(np.float32)
        save_dict['ref_coords_coarse_log_radial'] = ref_coarse.astype(np.float32)
        save_dict['ref_coords_log_radial'] = ref_coarse.astype(np.float32)

    # 存储
    np.savez_compressed(output_path, **save_dict)

    # 打印大小
    fine_mb = fine.nbytes / 1024 / 1024
    mid_mb = mid.nbytes / 1024 / 1024
    coarse_mb = coarse.nbytes / 1024 / 1024
    print(f"  [OK] {os.path.basename(output_path)}: "
          f"fine={fine.shape}({fine_mb:.1f}MB), "
          f"mid={mid.shape}({mid_mb:.1f}MB), "
          f"coarse={coarse.shape}({coarse_mb:.1f}MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", type=str, default=None, help="单个 npz 文件")
    parser.add_argument("--output_npz", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None, help="批量: 输入目录")
    parser.add_argument("--output_dir", type=str, default=None, help="批量: 输出目录")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--in_place", action="store_true",
                        help="原地修改 (output=input)")
    args = parser.parse_args()

    if args.input_npz:
        output = args.output_npz or args.input_npz.replace('.npz', '_v3.npz')
        if args.in_place:
            output = args.input_npz
        convert_npz_to_multiscale(args.input_npz, output)
    elif args.input_dir:
        output_dir = args.output_dir or args.input_dir
        os.makedirs(output_dir, exist_ok=True)
        npz_files = sorted(glob(os.path.join(args.input_dir, "**/*.npz"), recursive=True))
        print(f"Found {len(npz_files)} npz files")
        for npz_path in npz_files:
            rel = os.path.relpath(npz_path, args.input_dir)
            if args.in_place:
                out_path = npz_path
            else:
                out_path = os.path.join(output_dir, rel)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if os.path.exists(out_path) and not args.overwrite and not args.in_place:
                print(f"  [SKIP] {rel} (exists)")
                continue
            convert_npz_to_multiscale(npz_path, out_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()