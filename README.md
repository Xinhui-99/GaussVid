# GaussVid: Sparse-View Gaussian Splatting with 3D-Aware Video Diffusion Priors



## 📦 Environment Setup

### 1. Clone Repository and Setup Environment

```bash
git clone https://github.com/Xinhui-99/GaussVid.git
cd GaussVid
conda create -n gaussvid python=3.10 -y
conda activate gaussvid
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Download Models

GaussVid relies on two sets of weights. Please download them and place them in the `./checkpoints` folder.

* Wan2.1-VACE-14B [[download](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B)]
* GaussVid-LoRA-weights [[download](https://huggingface.co/XinhuiLiu001/GaussVid/tree/main/Wan2.1-VACE-14B_lora)]

---

## 🚀 Inference

### Inference

```bash
# Prepare camera pose and video clips
cp ./wanvideo/model_training/test/cameras
cp ./wanvideo/model_training/test/clips
cp ./wanvideo/model_training/experiments/metadata.csv 

# Inference / repair and evaluate the video clips
python ./wanvideo/model_training/metric_mm/metric.py 
```

---
---
## 📊 Results

We provide part of our experimental results in the [`./wanvideo/model_training/experiments`](./wanvideo/model_training/experiments) folder, including rendered, restored clips,  and evaluation metrics.


## 🙏 Acknowledgments

Thanks to these great repositories: [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [EasyControl](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_EasyControl_Adding_Efficient_and_Flexible_Control_for_Diffusion_Transformer_ICCV_2025_paper.html),[Wan2.1](https://github.com/Wan-Video/Wan2.1) and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

---

## 🔗 Citation

If you find our work helpful, please cite it:

```bibtex
@article{gaussvid,
    title={GaussVid: Sparse-View Gaussian Splatting with 3D-Aware Video Diffusion Priors},
    author={<Author List>},
    journal={arXiv preprint arXiv:XXXX.XXXXX},
    year={2025}
}
```
