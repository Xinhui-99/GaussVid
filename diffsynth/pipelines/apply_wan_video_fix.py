"""
apply_wan_video_fix.py

在 wan_video.py 中找到 InputVideoEmbedder.process 的 ref_prefix_images 编码部分，
将批量编码替换为逐帧编码。

用法:
    python apply_wan_video_fix.py wan_video.py wan_video_fixed.py
"""
import sys

def apply_fix(input_path, output_path):
    with open(input_path, 'r') as f:
        content = f.read()
    
    # 定位要替换的原始代码块
    old_code = '''        if ref_prefix_images is not None:
            if not isinstance(ref_prefix_images, list):
                ref_prefix_images = [ref_prefix_images]
            ref_prefix_images = pipe.preprocess_video(ref_prefix_images)
            ref_prefix_latents = pipe.vae.encode(ref_prefix_images, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([ref_prefix_latents, input_latents], dim=2)'''
    
    new_code = '''        if ref_prefix_images is not None:
            if not isinstance(ref_prefix_images, list):
                ref_prefix_images = [ref_prefix_images]
            # ★ FIX: 逐帧编码参考图，确保每张图 → 1 个 latent 帧
            #
            # BUG 原因:
            #   原代码 preprocess_video(N张图) 将它们打包为 [1,3,N,H,W] 的"视频"，
            #   VAE encode 会做时间压缩: T_latent = (N-1)//4 + 1
            #   当 N=2 时: T_latent = (2-1)//4 + 1 = 1 (只有1个latent帧！)
            #   但 NoiseInitializer 按 length += N 分配了 N 个 latent 帧的 noise
            #   → noise T ≠ input_latents T，维度不匹配
            #
            # 修复: 逐张编码，每张图 → [1,C,1,H',W']，dim=2 拼接 → N 个 latent 帧
            ref_latent_list = []
            for ref_img in ref_prefix_images:
                ref_tensor = pipe.preprocess_video([ref_img])          # [1, 3, 1, H, W]
                ref_lat = pipe.vae.encode(ref_tensor, device=pipe.device)  # [1, C, 1, H', W']
                ref_lat = ref_lat.to(dtype=pipe.torch_dtype, device=pipe.device)
                ref_latent_list.append(ref_lat)
            ref_prefix_latents = torch.concat(ref_latent_list, dim=2)  # [1, C, N, H', W']
            input_latents = torch.concat([ref_prefix_latents, input_latents], dim=2)'''
    
    if old_code not in content:
        print("ERROR: Could not find the target code block to replace!")
        print("Please verify wan_video.py has the expected InputVideoEmbedder code.")
        sys.exit(1)
    
    count = content.count(old_code)
    if count != 1:
        print(f"WARNING: Found {count} occurrences, expected 1. Replacing first only.")
    
    content = content.replace(old_code, new_code, 1)
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"Fix applied: {input_path} → {output_path}")
    print(f"  Changed: InputVideoEmbedder.process ref encoding (batch → per-frame)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_wan_video.py> <output_wan_video.py>")
        sys.exit(1)
    apply_fix(sys.argv[1], sys.argv[2])