import os
import json

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
from accelerate import Accelerator
from safetensors import safe_open
from safetensors.torch import save_file

from diffsynth.utils.data import VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


class MetricsCalculator:
    """Per-process metric computation (PSNR / SSIM / LPIPS)."""

    def __init__(self, accelerator, compute_lpips=True):
        self.accelerator = accelerator
        self.device = accelerator.device

        self.loss_fn_vgg = None
        if compute_lpips:
            self.loss_fn_vgg = lpips.LPIPS(net="vgg").to(self.device).eval()

        # LPIPS expects inputs in [-1, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def calculate_single_pair(self, img_ref_pil, img_gen_pil):
        img_ref_np = np.array(img_ref_pil)
        img_gen_np = np.array(img_gen_pil)

        psnr_val = psnr_func(img_ref_np, img_gen_np, data_range=255)
        ssim_val = ssim_func(img_ref_np, img_gen_np, channel_axis=2, data_range=255)

        lpips_val = 0.0
        if self.loss_fn_vgg is not None:
            img_ref_tensor = self.transform(img_ref_pil).unsqueeze(0).to(self.device)
            img_gen_tensor = self.transform(img_gen_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                lpips_val = float(self.loss_fn_vgg(img_ref_tensor, img_gen_tensor).item())

        return psnr_val, ssim_val, lpips_val

    def calculate_video_metrics(self, gt_frames, compare_frames):
        psnr_values, ssim_values, lpips_values = [], [], []
        num_frames = min(len(gt_frames), len(compare_frames))

        for i in range(1, num_frames - 1):
            gt_img = gt_frames[i]
            compare_img = compare_frames[i]

            if isinstance(compare_img, torch.Tensor):
                compare_img = Image.fromarray(
                    compare_img.detach().cpu()
                    .mul(255).add_(0.5).clamp_(0, 255)
                    .permute(1, 2, 0).to(torch.uint8).numpy()
                )

            if gt_img.size != compare_img.size:
                compare_img = compare_img.resize(gt_img.size, Image.BILINEAR)

            psnr_val, ssim_val, lpips_val = self.calculate_single_pair(gt_img, compare_img)
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            lpips_values.append(lpips_val)

        return {
            "psnr": float(np.mean(psnr_values)) if psnr_values else 0.0,
            "ssim": float(np.mean(ssim_values)) if ssim_values else 0.0,
            "lpips": float(np.mean(lpips_values)) if lpips_values else 0.0,
            "num_frames": int(num_frames),
        }


def save_frames_to_folder(frames, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    for i, frame in enumerate(frames):
        if isinstance(frame, torch.Tensor):
            frame = Image.fromarray(
                frame.detach().cpu()
                .mul(255).add_(0.5).clamp_(0, 255)
                .permute(1, 2, 0).to(torch.uint8).numpy()
            )
        frame.save(os.path.join(folder_name, f"frame_{i:04d}.png"))


def append_metrics_to_file(file_path, record):
    """Append a single metric record to a .jsonl or .csv file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if file_path.endswith(".jsonl"):
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    elif file_path.endswith(".csv"):
        df = pd.DataFrame([record])
        file_exists = os.path.exists(file_path)
        df.to_csv(file_path, mode="a", header=not file_exists, index=False, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def load_video_frames(video_path, height, width):
    """Read a video and return a list of PIL Images."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (width, height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def get_best_records_for_clip(all_combo_records):
    """Pick the best result per metric (PSNR / SSIM / LPIPS) for one clip."""
    if len(all_combo_records) == 0:
        return None

    best_psnr = max(all_combo_records, key=lambda x: x["gt_gen_psnr"])
    best_ssim = max(all_combo_records, key=lambda x: x["gt_gen_ssim"])
    best_lpips = min(all_combo_records, key=lambda x: x["gt_gen_lpips"])

    return {
        "clip_idx": best_psnr["clip_idx"],
        "vace_video": best_psnr["vace_video"],

        "best_psnr": best_psnr["gt_gen_psnr"],
        "best_psnr_strength": best_psnr["strength"],
        "best_psnr_steps": best_psnr["steps"],
        "best_psnr_ssim": best_psnr["gt_gen_ssim"],
        "best_psnr_lpips": best_psnr["gt_gen_lpips"],

        "best_ssim": best_ssim["gt_gen_ssim"],
        "best_ssim_strength": best_ssim["strength"],
        "best_ssim_steps": best_ssim["steps"],
        "best_ssim_psnr": best_ssim["gt_gen_psnr"],
        "best_ssim_lpips": best_ssim["gt_gen_lpips"],

        "best_lpips": best_lpips["gt_gen_lpips"],
        "best_lpips_strength": best_lpips["strength"],
        "best_lpips_steps": best_lpips["steps"],
        "best_lpips_psnr": best_lpips["gt_gen_psnr"],
        "best_lpips_ssim": best_lpips["gt_gen_ssim"],

        "control_psnr": best_psnr["gt_ctrl_psnr"],
        "control_ssim": best_psnr["gt_ctrl_ssim"],
        "control_lpips": best_psnr["gt_ctrl_lpips"],
    }


def run_inference_with_metrics(
    csv_path,
    output_base,
    pipe,
    metrics_calculator,
    accelerator,
    split_name="test",
    per_clip_metrics_path=None,
    best_metrics_output_path=None,
):
    os.makedirs(output_base, exist_ok=True)

    df = pd.read_csv(csv_path)
    total_clips = len(df)

    # Split clips across processes
    per = total_clips // accelerator.num_processes
    start = accelerator.process_index * per
    end = (accelerator.process_index + 1) * per
    if accelerator.process_index == accelerator.num_processes - 1:
        end = total_clips

    df_subset = df.iloc[start:end].copy()
    df_subset["global_idx"] = df_subset.index
    df_subset = df_subset.reset_index(drop=True)

    if accelerator.is_main_process:
        print(f"Using {accelerator.num_processes} processes")
    print(f"[proc {accelerator.process_index}] Processing clips [{start}, {end}) = {len(df_subset)} clips")

    all_combo_results_local = []
    best_clip_results_local = []

    step_candidates = [10]
    strength_candidates = [0.2]

    for local_i, row in df_subset.iterrows():
        global_idx = int(row["global_idx"])
        print(f"\n[proc {accelerator.process_index}] [{split_name}] clip {global_idx}: {row['noisy_video_path']}")

        h, w = 560, 1024

        # Load input / GT / control
        control_video = VideoData(row["noisy_video_path"], height=h, width=w)
        gt_frames = load_video_frames(row["gt_video_path"], h, w)
        control_frames = load_video_frames(row["noisy_video_path"], h, w)

        # Load camera parameters
        vace_camera = None
        if os.path.exists(row["camera_params_path"]):
            with open(row["camera_params_path"], "r") as f:
                cam_json = json.load(f)
            all_frames = cam_json.get("frames", [])
            ks = [torch.tensor(f["intrinsic_matrix"], dtype=torch.float64) for f in all_frames]
            c2ws = [torch.tensor(f["c2w"], dtype=torch.float64) for f in all_frames]
            vace_camera = {
                "Ks": torch.stack(ks),
                "c2ws": torch.stack(c2ws),
            }

        # Control metrics only need to be computed once
        metrics_gt_control = metrics_calculator.calculate_video_metrics(gt_frames, control_frames)

        clip_combo_results = []

        for steps, strength in zip(step_candidates, strength_candidates):
            clip_output_dir = os.path.join(
                output_base,
                f"clip_{global_idx:05d}_strength_{strength:.1f}_steps_{steps}",
            )

            # Reuse existing frames if already generated
            already_processed = False
            expected_frames = len(control_video)
            frame_files = []
            if os.path.exists(clip_output_dir):
                frame_files = sorted([
                    f for f in os.listdir(clip_output_dir)
                    if f.startswith("frame_") and f.endswith(".png")
                ])
                if len(frame_files) >= expected_frames:
                    already_processed = True

            if already_processed:
                video = [
                    Image.open(os.path.join(clip_output_dir, f)).convert("RGB")
                    for f in frame_files
                ]
                print(f"[proc {accelerator.process_index}] Reuse clip {global_idx}, strength={strength}, steps={steps}")
            else:
                with torch.no_grad():
                    video = pipe(
                        prompt="Remove degradation and restore clean video.",
                        negative_prompt="blur, noise, distortion, low quality",
                        input_video=control_video,
                        vace_video=control_video,
                        height=h,
                        width=w,
                        vace_camera=vace_camera,
                        denoising_strength=strength,
                        num_inference_steps=steps,
                        cfg_scale=1.0,
                        num_frames=len(control_video),
                        seed=1,
                        tiled=True,
                    )
                save_frames_to_folder(video, clip_output_dir)

            metrics_gt_gen = metrics_calculator.calculate_video_metrics(gt_frames, video)

            print(
                f"[proc {accelerator.process_index}] clip={global_idx}, "
                f"strength={strength}, steps={steps} | "
                f"Gen PSNR={metrics_gt_gen['psnr']:.3f}, "
                f"SSIM={metrics_gt_gen['ssim']:.4f}, "
                f"LPIPS={metrics_gt_gen['lpips']:.4f} | "
                f"Control PSNR={metrics_gt_control['psnr']:.3f}, "
                f"SSIM={metrics_gt_control['ssim']:.4f}, "
                f"LPIPS={metrics_gt_control['lpips']:.4f}"
            )

            combo_record = {
                "clip_idx": global_idx,
                "strength": strength,
                "steps": steps,
                "vace_video": row["noisy_video_path"],
                "num_frames": metrics_gt_gen["num_frames"],

                "gt_gen_psnr": metrics_gt_gen["psnr"],
                "gt_gen_ssim": metrics_gt_gen["ssim"],
                "gt_gen_lpips": metrics_gt_gen["lpips"],

                "gt_ctrl_psnr": metrics_gt_control["psnr"],
                "gt_ctrl_ssim": metrics_gt_control["ssim"],
                "gt_ctrl_lpips": metrics_gt_control["lpips"],

                "proc_index": accelerator.process_index,
            }

            clip_combo_results.append(combo_record)
            all_combo_results_local.append(combo_record)

            if per_clip_metrics_path is not None:
                proc_file_path = per_clip_metrics_path
                if accelerator.num_processes > 1:
                    base, ext = os.path.splitext(per_clip_metrics_path)
                    proc_file_path = f"{base}_proc{accelerator.process_index}{ext}"
                append_metrics_to_file(proc_file_path, combo_record)

        # Select best combo for the current clip
        best_record = get_best_records_for_clip(clip_combo_results)
        if best_record is not None:
            best_record["proc_index"] = accelerator.process_index
            best_clip_results_local.append(best_record)

        # Save GT / control frames once
        gt_output_dir = os.path.join(output_base, f"gt_{global_idx:05d}")
        control_output_dir = os.path.join(output_base, f"control_{global_idx:05d}")
        if not os.path.exists(gt_output_dir):
            save_frames_to_folder(gt_frames, gt_output_dir)
        if not os.path.exists(control_output_dir):
            save_frames_to_folder(control_frames, control_output_dir)

    # Gather results across processes via temp files
    temp_dir = os.path.join(output_base, "temp_gather")
    os.makedirs(temp_dir, exist_ok=True)

    temp_file = os.path.join(temp_dir, f"proc_{accelerator.process_index}.json")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump({
            "all_combo_results": all_combo_results_local,
            "best_clip_results": best_clip_results_local,
        }, f, ensure_ascii=False)

    print(f"[proc {accelerator.process_index}] Saved temporary results to {temp_file}")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\n[Main Process] Gathering results from all processes...")

        gathered_all_combo_results = []
        gathered_best_clip_results = []

        for proc_idx in range(accelerator.num_processes):
            proc_file = os.path.join(temp_dir, f"proc_{proc_idx}.json")
            if os.path.exists(proc_file):
                with open(proc_file, "r", encoding="utf-8") as f:
                    proc_data = json.load(f)
                gathered_all_combo_results.extend(proc_data["all_combo_results"])
                gathered_best_clip_results.extend(proc_data["best_clip_results"])
                print(f"  - Loaded process {proc_idx}")
            else:
                print(f"  - Warning: missing {proc_file}")

        # Deduplicate by clip index
        best_by_clip = {item["clip_idx"]: item for item in gathered_best_clip_results}
        dedup_best_clip_results = [best_by_clip[k] for k in sorted(best_by_clip.keys())]

        all_combo_df = pd.DataFrame(gathered_all_combo_results).sort_values(["clip_idx", "steps", "strength"])
        best_df = pd.DataFrame(dedup_best_clip_results).sort_values(["clip_idx"])

        if per_clip_metrics_path is not None:
            os.makedirs(os.path.dirname(per_clip_metrics_path), exist_ok=True)
            all_combo_df.to_csv(per_clip_metrics_path, index=False, encoding="utf-8")
            print(f"[Main Process] All combinations saved to {per_clip_metrics_path}")

        if best_metrics_output_path is not None:
            os.makedirs(os.path.dirname(best_metrics_output_path), exist_ok=True)
            best_df.to_csv(best_metrics_output_path, index=False, encoding="utf-8")
            print(f"[Main Process] Best-per-clip metrics saved to {best_metrics_output_path}")

        final_summary = {
            "split": split_name,
            "num_clips_total": total_clips,
            "num_clips_processed": len(dedup_best_clip_results),

            "avg_best_psnr": float(best_df["best_psnr"].mean()) if len(best_df) > 0 else 0.0,
            "avg_best_ssim": float(best_df["best_ssim"].mean()) if len(best_df) > 0 else 0.0,
            "avg_best_lpips": float(best_df["best_lpips"].mean()) if len(best_df) > 0 else 0.0,

            "avg_control_psnr": float(best_df["control_psnr"].mean()) if len(best_df) > 0 else 0.0,
            "avg_control_ssim": float(best_df["control_ssim"].mean()) if len(best_df) > 0 else 0.0,
            "avg_control_lpips": float(best_df["control_lpips"].mean()) if len(best_df) > 0 else 0.0,

            "best_clip_results": dedup_best_clip_results,
        }

        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp dir: {e}")

        return final_summary

    return None


def main():
    accelerator = Accelerator(
        mixed_precision="bf16",
        device_placement=True,
    )
    print(f"Using {accelerator.num_processes} GPUs (process index: {accelerator.process_index})")

    # ---- Load base pipeline ----
    if accelerator.is_main_process:
        print("Loading model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=accelerator.device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    )

    # ---- Load fine-tuned checkpoint (LoRA + VAE decoder) ----
    checkpoint_path = "/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/1New3DGS/Octree-GS-VideoDif/DiffSynth-Studio/examples/wanvideo/model_training/models/train_stage1_ditvace_flow_ssim_26_again/Wan2.1-VACE-14B_lora/epoch-4.safetensors"

    with safe_open(checkpoint_path, framework="pt") as f:
        all_weights = {k: f.get_tensor(k) for k in f.keys()}

    vace_weights = {}
    dit_weights = {}
    vae_decoder_weights = {}

    for k, v in all_weights.items():
        if k.startswith("pipe.vace."):
            vace_weights[k.replace("pipe.vace.", "")] = v
        elif k.startswith("pipe.dit."):
            dit_weights[k.replace("pipe.dit.", "")] = v
        elif k.startswith("pipe.vae.model.decoder."):
            vae_decoder_weights[k.replace("pipe.vae.model.decoder.", "")] = v
        elif k.startswith("pipe.vae.") and "decoder" in k:
            vae_decoder_weights[k.replace("pipe.vae.model.decoder.", "")] = v

    if len(vace_weights) > 0:
        save_file(vace_weights, "/tmp/vace_lora_only.safetensors")
        pipe.load_lora(pipe.vace, "/tmp/vace_lora_only.safetensors", alpha=1.0)
        print("VACE LoRA loaded")

    if len(dit_weights) > 0:
        save_file(dit_weights, "/tmp/dit_lora_only.safetensors")
        pipe.load_lora(pipe.dit, "/tmp/dit_lora_only.safetensors", alpha=1.0)
        print("DiT LoRA loaded")

    if len(vae_decoder_weights) > 0:
        current_decoder_state = pipe.vae.model.decoder.state_dict()
        updated_count = 0
        for key, value in vae_decoder_weights.items():
            if key in current_decoder_state:
                if current_decoder_state[key].shape == value.shape:
                    current_decoder_state[key] = value.to(current_decoder_state[key].dtype)
                    updated_count += 1
                else:
                    print(f"Shape mismatch for {key}: {current_decoder_state[key].shape} vs {value.shape}")
            else:
                print(f"Key not found in decoder: {key}")
        pipe.vae.model.decoder.load_state_dict(current_decoder_state, strict=True)
        print(f"VAE Decoder weights loaded ({updated_count}/{len(vae_decoder_weights)} keys)")

    pipe = accelerator.prepare(pipe)

    metrics_calculator = MetricsCalculator(accelerator, compute_lpips=True)
    if accelerator.is_main_process:
        print("Metrics calculator initialized")

    # ---- Paths ----
    test_csv_path = "./metadata_test.csv"
    all_combinations_output_path = "experiments/Gaussvid/all_combinations.csv"
    best_metrics_output_path = "experiments/Gaussvid/baseline_gaussvid.csv"
    output_base_test = "experiments/baseline_gaussvid"

    results_test = run_inference_with_metrics(
        csv_path=test_csv_path,
        output_base=output_base_test,
        pipe=pipe,
        metrics_calculator=metrics_calculator,
        accelerator=accelerator,
        split_name="test",
        per_clip_metrics_path=all_combinations_output_path,
        best_metrics_output_path=best_metrics_output_path,
    )

    if accelerator.is_main_process and results_test is not None:
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"\nTEST SET (Total: {results_test['num_clips_total']}, "
              f"Processed: {results_test['num_clips_processed']} clips):")
        print("-" * 60)

        print("Average BEST results over all clips:")
        print(f"  Best PSNR : {results_test['avg_best_psnr']:.3f}")
        print(f"  Best SSIM : {results_test['avg_best_ssim']:.4f}")
        print(f"  Best LPIPS: {results_test['avg_best_lpips']:.4f}")

        print("\nAverage CONTROL results over all clips:")
        print(f"  Control PSNR : {results_test['avg_control_psnr']:.3f}")
        print(f"  Control SSIM : {results_test['avg_control_ssim']:.4f}")
        print(f"  Control LPIPS: {results_test['avg_control_lpips']:.4f}")

        print("\nImprovement:")
        print(f"  PSNR : {results_test['avg_best_psnr'] - results_test['avg_control_psnr']:+.3f}")
        print(f"  SSIM : {results_test['avg_best_ssim'] - results_test['avg_control_ssim']:+.4f}")
        print(f"  LPIPS: {results_test['avg_control_lpips'] - results_test['avg_best_lpips']:+.4f} (lower is better)")

        summary_path = "experiments/metrics_summary_gaussvid.json"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results_test, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to {summary_path}")
        print("\nAll clips processed!")


if __name__ == "__main__":
    main()