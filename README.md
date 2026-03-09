# EasyVideoR1

[中文](README_zh.md)

Built on [EasyR1](https://github.com/hiyouga/EasyR1), with support for **Qwen3-VL** series models for video understanding reinforcement learning training.

This project builds upon the excellent work of EasyR1 and [veRL](https://github.com/volcengine/verl). We thank all authors for providing such a high-performance RL training framework.

## Features

- Support for **Qwen3-VL** series vision-language models
- **Mixed image-video training** with optimized gradient flow
- Independent resolution control for images and videos
- Video metadata support for precise frame processing
- Comprehensive reward functions for video understanding tasks
  - Multiple choice, numerical, OCR, open-ended QA
  - Temporal grounding, spatial grounding, spatial-temporal grounding
  - Object tracking, image/video segmentation

## Installation

### Step 1: Create Conda Environment

```bash
conda create -n easyvideorl python=3.11
conda activate easyvideorl
```

### Step 2: Clone and Install

```bash
git clone https://github.com/cyuQ1n/EasyVideoR1.git
cd EasyVideoR1
pip install -e .
```

### Step 3: Install Flash Attention

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

## Quick Start

### 1. Prepare Dataset

The dataset should be in JSON format with the following structure:

```json
[
  {
    "problem": "Your question",
    "answer": "Ground truth answer",
    "videos": ["path/to/video.mp4"],
    "data_type": "video",
    "problem_type": "multiple_choice"
  }
]
```

Supported `problem_type` values:
- `multiple_choice`, `numerical`, `regression`, `ocr`, `open_ended`, `math`
- `temporal_grounding`, `spatial_grounding`, `spatial_temporal_grounding`
- `tracking`, `image_segmentation`, `video_segmentation`

### 2. Configure Training

Edit `examples/videorl/video_rl.yaml`:

```yaml
data:
  train_files: /path/to/your/train.json
  val_files: /path/to/your/val.json
  image_dir: /path/to/your/data/root
  video_fps: 2.0
  video_max_frames: 64
  max_prompt_length: 16384
  max_response_length: 4096
  # Resolution settings
  image_max_pixels: 1048576
  video_max_pixels: 262144

worker:
  actor:
    model:
      model_path: Qwen/Qwen3-VL-8B-Instruct
    # ... other settings

trainer:
  project_name: your_project
  experiment_name: your_experiment
  save_checkpoint_path: /path/to/checkpoints
```

### 3. Start Training

```bash
bash examples/videorl/run_video_rl.sh
```

### 4. Merge Checkpoints

```bash
python3 scripts/model_merger.py --local_dir checkpoints/your_exp/global_step_xxx/actor
```

## Configuration Parameters

### Data Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `video_fps` | Video sampling frame rate | 2.0 |
| `video_max_frames` | Max frames per video | 128 |
| `image_max_pixels` | Max pixels for images | 1048576 |
| `video_max_pixels` | Max pixels for videos | 262144 |

### Worker Configuration

| Parameter | Description |
|-----------|-------------|
| `worker.rollout.tensor_parallel_size` | Tensor parallelism for vLLM inference |
| `worker.rollout.gpu_memory_utilization` | GPU memory utilization ratio for vLLM |
| `worker.actor.fsdp.enable_full_shard` | Enable FSDP full sharding |

## FAQ

**Q: Image features and image tokens do not match**

A: Increase `data.max_prompt_length` or decrease `data.max_pixels`.

**Q: CUDA out of memory**

A: Decrease `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`.

**Q: Multi-node training hangs**

A: Use `ray status` to check Ray cluster status and ensure all nodes are connected.

## Citation

If you use this project, please cite EasyR1 and veRL:

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}

@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and others},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

## License

This project follows the same license as [EasyR1](https://github.com/hiyouga/EasyR1).
