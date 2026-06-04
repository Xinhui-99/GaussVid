import os
import shutil
import csv

# === 配置 ===
CSV_PATH = "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/1New3DGS/Octree-GS-VideoDif/DiffSynth-Studio/examples/wanvideo/model_training/metric/metadata_test_git.csv"  # 修改为你的CSV文件路径

DST_CLIPS = "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/1New3DGS/Octree-GS-VideoDif/GaussVid/wanvideo/model_training/test/clips"
DST_CAMERAS = "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/1New3DGS/Octree-GS-VideoDif/GaussVid/wanvideo/model_training/test/clips"
# 注意：你指定的 mp4 和 json 目标路径相同，如果 json 应该放到 cameras/ 子目录，请修改上面 DST_CAMERAS

# === 创建目标目录 ===
os.makedirs(DST_CLIPS, exist_ok=True)
os.makedirs(DST_CAMERAS, exist_ok=True)

# === 读取CSV并复制文件 ===
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"共 {len(rows)} 条记录")

for i, row in enumerate(rows):
    noisy_src = row["noisy_video_path"]
    gt_src = row["gt_video_path"]
    cam_src = row["camera_params_path"]

    for src in [noisy_src, gt_src]:
        dst = os.path.join(DST_CLIPS, os.path.basename(src))
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[{i+1}] 复制 {os.path.basename(src)}")
        else:
            print(f"[{i+1}] ⚠ 源文件不存在: {src}")

    dst = os.path.join(DST_CAMERAS, os.path.basename(cam_src))
    if os.path.exists(cam_src):
        shutil.copy2(cam_src, dst)
        print(f"[{i+1}] 复制 {os.path.basename(cam_src)}")
    else:
        print(f"[{i+1}] ⚠ 源文件不存在: {cam_src}")

# === 生成新的CSV（路径已更新） ===
new_csv_path = os.path.join(os.path.dirname(CSV_PATH), "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/1New3DGS/Octree-GS-VideoDif/GaussVid/wanvideo/model_training/test/metadata_test.csv")
with open(new_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["noisy_video_path", "gt_video_path", "camera_params_path"])
    for row in rows:
        writer.writerow([
            os.path.join(DST_CLIPS, os.path.basename(row["noisy_video_path"])),
            os.path.join(DST_CLIPS, os.path.basename(row["gt_video_path"])),
            os.path.join(DST_CAMERAS, os.path.basename(row["camera_params_path"])),
        ])

print(f"\n新CSV已保存: {new_csv_path}")
print("完成!")