#!/usr/bin/env python3
"""
补丁脚本：为已有的 world_coords_*.npz 补上 ref_coords_log_radial
=================================================================

问题：旧版预处理脚本生成的 npz 中只有 world_coords_log_radial (noisy frames)，
      缺少 ref_coords_log_radial (reference frames)，导致 Geo3D bias 无法激活。

原理：
  - 从 metadata CSV 中读取每个 clip 的参考帧信息
  - 用参考帧的深度 + 相机参数反投影计算 world coords
  - 做 log-radial 归一化后追加到 npz 文件

用法：
  python patch_add_ref_coords.py \
    --metadata_csv /mnt/dongxu-fs1/data-hdd/xinhuiliu/WAN2.1_5K_17_world/metadata_train.csv \
    --dry_run          # 先 dry run 检查
  
  python patch_add_ref_coords.py \
    --metadata_csv /mnt/dongxu-fs1/data-hdd/xinhuiliu/WAN2.1_5K_17_world/metadata_train.csv
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
from tqdm import tqdm


def normalize_world_coords_log_radial(pts_world, scene_center, near_r, far_r):
    """Log-radial 归一化世界坐标 (与预处理脚本中的函数一致)"""
    offset = pts_world - scene_center
    dists = np.linalg.norm(offset, axis=-1, keepdims=True)
    valid = (dists.squeeze(-1) > 0)

    if not np.any(valid):
        return np.zeros_like(pts_world)

    log_near = np.log(max(near_r, 1e-6))
    log_far = np.log(max(far_r, near_r + 1e-6))

    if log_far - log_near < 1e-6:
        return np.zeros_like(pts_world)

    direction = np.zeros_like(pts_world)
    direction[valid] = offset[valid] / dists[valid]

    dists_flat = dists.squeeze(-1)
    dists_clipped = np.clip(dists_flat, near_r, far_r)

    log_norm = np.zeros_like(dists_flat)
    log_norm[valid] = (np.log(dists_clipped[valid]) - log_near) / (log_far - log_near)

    return (direction * log_norm[..., np.newaxis]).astype(np.float32)


def compute_ref_world_coords(ref_depth, ref_cam, target_w, target_h,
                              scene_center, near_r, far_r, patch_stride=16):
    """
    为单张参考帧计算 patch 级别的 log-radial world coords。

    与 compute_world_coords_for_clip() 保持一致:
      - 使用 extrinsic_matrix (即 camtoworld)，而非 c2w (optimized_camtoworld)
      - 使用相同的 unproject + log-radial 归一化逻辑

    Args:
        ref_depth:  [H, W] float32 原始深度
        ref_cam:    dict with 'intrinsic_matrix', 'extrinsic_matrix'/'c2w', 'width', 'height'
        target_w, target_h: 目标分辨率 (与训练一致)

    Returns:
        coords_log: [H_patch, W_patch, 3] float32
    """
    import cv2

    h_patch = target_h // patch_stride
    w_patch = target_w // patch_stride

    # Resize depth if needed
    if ref_depth.shape[0] != target_h or ref_depth.shape[1] != target_w:
        ref_depth = cv2.resize(ref_depth, (target_w, target_h),
                               interpolation=cv2.INTER_NEAREST)

    # Patch center coordinates
    us = np.arange(patch_stride // 2, target_w, patch_stride)
    vs = np.arange(patch_stride // 2, target_h, patch_stride)
    us_grid, vs_grid = np.meshgrid(us, vs)

    # Camera intrinsics (scaled to target resolution)
    K = np.array(ref_cam['intrinsic_matrix'], dtype=np.float64)
    # ★ 关键：使用 extrinsic_matrix (camtoworld)，与 compute_world_coords_for_clip 一致
    # 不用 c2w (可能是 optimized_camtoworld，与 noisy frames 用的不同)
    c2w = np.array(ref_cam.get('extrinsic_matrix', ref_cam['c2w']), dtype=np.float64)

    orig_w = ref_cam.get('width', target_w)
    orig_h = ref_cam.get('height', target_h)
    fx = K[0, 0] * target_w / orig_w
    fy = K[1, 1] * target_h / orig_h
    cx = K[0, 2] * target_w / orig_w
    cy = K[1, 2] * target_h / orig_h

    # Sample depth at patch centers
    depth_patch = ref_depth[vs_grid.astype(int), us_grid.astype(int)]

    # Unproject to camera coords
    x_c = (us_grid - cx) * depth_patch / fx
    y_c = (vs_grid - cy) * depth_patch / fy
    z_c = depth_patch

    pts_cam = np.stack([x_c, y_c, z_c, np.ones_like(z_c)], axis=-1)
    pts_cam_flat = pts_cam.reshape(-1, 4)
    pts_world = (c2w @ pts_cam_flat.T).T[:, :3]
    pts_world = pts_world.reshape(h_patch, w_patch, 3)

    invalid_mask = depth_patch <= 0

    # Log-radial normalization
    pts_log = normalize_world_coords_log_radial(
        pts_world, scene_center, near_r, far_r
    )
    pts_log[invalid_mask] = 0.0

    return pts_log.astype(np.float32)


def process_one_row(row, target_w=1024, target_h=560, patch_stride=16, dry_run=False):
    """处理一行 metadata，为对应的 npz 补上 ref_coords_log_radial"""

    wc_path = row.get('world_coords_path', '')
    if not wc_path or not os.path.exists(wc_path):
        return "skip_no_wc"

    # 检查是否已有 ref_coords
    existing = np.load(wc_path, allow_pickle=True)
    existing_keys = list(existing.files)

    if 'ref_coords_log_radial' in existing_keys:
        arr = existing['ref_coords_log_radial']
        if arr.shape[0] > 0 and arr.shape[-1] == 3:
            return "already_has"

    # --- 需要补 ref_coords ---

    # 解析参考帧深度路径
    ref_depth_paths_str = row.get('ref_depth_paths', '[]')
    try:
        ref_depth_paths = json.loads(ref_depth_paths_str)
    except json.JSONDecodeError:
        return "skip_bad_ref_depth_json"

    # 解析参考帧相机参数
    ref_cam_str = row.get('ref_camera_params', '[]')
    try:
        ref_cameras = json.loads(ref_cam_str)
    except json.JSONDecodeError:
        return "skip_bad_ref_cam_json"

    if not ref_depth_paths or not ref_cameras:
        return "skip_no_ref_data"

    # 场景参数
    scene_center_str = row.get('scene_center', '[0,0,0]')
    try:
        scene_center = np.array(json.loads(scene_center_str), dtype=np.float64)
    except:
        scene_center = np.array([0.0, 0.0, 0.0])

    near_r = float(row.get('near_r', 0.01))
    far_r = float(row.get('far_r', 10.0))

    # 计算每个参考帧的 world coords
    ref_coords_list = []
    for i, (depth_path, cam) in enumerate(zip(ref_depth_paths, ref_cameras)):
        if not depth_path or not os.path.exists(depth_path):
            # Fallback: 零填充
            h_p = target_h // patch_stride
            w_p = target_w // patch_stride
            ref_coords_list.append(np.zeros((h_p, w_p, 3), dtype=np.float32))
            continue

        ref_depth = np.load(depth_path).astype(np.float32)
        ref_log = compute_ref_world_coords(
            ref_depth, cam, target_w, target_h,
            scene_center, near_r, far_r, patch_stride
        )
        ref_coords_list.append(ref_log)

    ref_coords_log_radial = np.stack(ref_coords_list, axis=0)  # [num_ref, H_p, W_p, 3]

    if dry_run:
        return f"would_add:{ref_coords_log_radial.shape}"

    # --- 写回 npz (保留所有已有数据 + 新增 ref_coords_log_radial) ---
    save_dict = {k: existing[k] for k in existing_keys}
    save_dict['ref_coords_log_radial'] = ref_coords_log_radial

    # 写到临时文件再 rename (原子操作，防止写入中断导致数据损坏)
    tmp_path = wc_path + '.tmp'
    np.savez_compressed(tmp_path, **save_dict)
    os.replace(tmp_path, wc_path)

    return f"patched:{ref_coords_log_radial.shape}"


def main():
    parser = argparse.ArgumentParser(
        description="Patch existing world_coords npz files to add ref_coords_log_radial")
    parser.add_argument("--metadata_csv", type=str, required=True,
                        help="Path to metadata_train.csv")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=560)
    parser.add_argument("--patch_stride", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true",
                        help="Only check, don't modify files")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Process only first N samples (for testing)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Patch: Add ref_coords_log_radial to world_coords npz")
    print(f"  CSV:   {args.metadata_csv}")
    print(f"  Mode:  {'DRY RUN' if args.dry_run else 'WRITE'}")
    print(f"{'='*60}\n")

    with open(args.metadata_csv) as f:
        rows = list(csv.DictReader(f))

    if args.max_samples:
        rows = rows[:args.max_samples]

    stats = {}
    for row in tqdm(rows, desc="Patching"):
        result = process_one_row(
            row,
            target_w=args.width,
            target_h=args.height,
            patch_stride=args.patch_stride,
            dry_run=args.dry_run,
        )
        key = result.split(":")[0]
        stats[key] = stats.get(key, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Results:")
    for k, v in sorted(stats.items()):
        print(f"    {k:30s}: {v}")
    print(f"  Total: {len(rows)} rows")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n⚠️  This was a DRY RUN. Remove --dry_run to actually patch files.")
    else:
        print("\n✅  Patching complete. Re-run validation to verify:")
        print(f"    python validate_ablation.py --metadata_csv {args.metadata_csv} "
              f"--num_samples 3 --data_check_only")


if __name__ == "__main__":
    main()