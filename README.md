# EasyVideoR1

[中文](README_zh.md)

Built on [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl), with support for **Qwen3-VL** series models for video understanding reinforcement learning training.

We thank all authors for providing such a high-performance RL training framework.

## 📍 Features

- Support for **Qwen3-VL** series vision-language models
- **Mixed image-video training** with optimized gradient flow
- Independent resolution control for images and videos
- Video metadata support for precise frame processing
- Default example prompt/reward stack for video understanding tasks
  - Fully implemented in the default reward: multiple choice, numerical, temporal grounding, spatial-temporal grounding, open-ended QA
  - Prompt formatting exists for additional task types such as spatial grounding, tracking, OCR, boolean QA, math, and code generation

## 🏆 Performance

Training with EasyVideoR1 yields consistent improvements over the Qwen3-VL-8B base models across 10 video understanding benchmarks, with an average accuracy gain of **+2.3%**.

<div align="center">
  <img src="assets/benchmark.png" alt="Benchmark results" width="90%">
</div>

Our video preprocessing cache reduces per-step training time by **1.47x** compared to on-the-fly decoding, without sacrificing accuracy.

<div align="center">
  <img src="assets/efficiency.png" alt="Training efficiency" width="90%">
</div>

## 📐 Installation

### Step 1: Create Conda Environment

```bash
conda create -n easyvideor1 python=3.11
conda activate easyvideor1
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

## 🚀 Quick Start

Below is a minimal 3-step workflow to get training running.

### Step 1: Prepare Your Data

Create a JSON/JSONL file. Each entry should look like:

```json
{
  "problem": "What happens in this video?",
  "answer": "A cat jumps onto the table.",
  "videos": ["path/to/video.mp4"],
  "data_type": "video",
  "problem_type": "open-ended"
}
```

For multiple-choice tasks, add an `options` field:

```json
{
  "problem": "What color is the car?",
  "answer": "B",
  "videos": ["path/to/video.mp4"],
  "data_type": "video",
  "problem_type": "multiple choice",
  "options": ["A. Red", "B. Blue", "C. Green", "D. White"]
}
```

> See [docs/config_parameters.md](docs/config_parameters.md) for the full list of supported `problem_type` values and data fields.

### Step 2: Edit the Config

Copy and edit the example config:

```bash
cp examples/video_rl/video_rl.yaml my_config.yaml
```

Update at minimum these fields:

```yaml
data:
  train_files: /path/to/your/train.jsonl
  val_files: /path/to/your/val.json

worker:
  actor:
    model:
      model_path: Qwen/Qwen3-VL-8B-Instruct

trainer:
  experiment_name: my_first_run
  save_checkpoint_path: ./checkpoints/my_first_run
```

### Step 3: Launch Training

```bash
# Single-node (8 GPUs)
bash examples/video_rl/run_video_rl.sh

# Multi-node: set WORLD_SIZE, RANK, MASTER_ADDR on each node
WORLD_SIZE=2 RANK=0 MASTER_ADDR=<head_ip> bash examples/video_rl/run_video_rl.sh  # head
WORLD_SIZE=2 RANK=1 MASTER_ADDR=<head_ip> bash examples/video_rl/run_video_rl.sh  # worker
```

After training, merge FSDP checkpoints to Hugging Face format:

```bash
python3 scripts/model_merger.py --local_dir checkpoints/my_first_run/global_step_100/actor
```

## 📂 Project Structure

```
EasyVideoR1/
├── verl/                       # Core RL training framework
│   ├── trainer/                # Training loop & Ray orchestration
│   ├── workers/                # Actor, rollout, reward, critic workers
│   ├── models/                 # Qwen3-VL / Qwen2-VL model support
│   └── utils/                  # Dataset, tokenization, FSDP utilities
├── examples/
│   ├── video_rl/               # Video-only RL pipeline (single-file reward)
│   └── unified_rl/             # Mixed image-video pipeline (modular reward)
├── eval/                       # Evaluation toolkit (25+ benchmarks)
├── scripts/                    # Checkpoint merger, video preprocessing
└── docs/                       # Detailed documentation
```

## 🔧 Example Pipelines

### Video RL (`examples/video_rl/`)

A self-contained pipeline for video-only RL training. The reward function (`video_reward.py`) handles all task types in a single file with a simple `accuracy * 0.9 + format * 0.1` scoring formula.

```bash
bash examples/video_rl/run_video_rl.sh
```

### Unified RL (`examples/unified_rl/`)

A modular pipeline for mixed image-video training. The reward function routes each sample to a task-specific module (multiple choice, grounding, math, etc.) with independent scoring logic.

```bash
bash examples/unified_rl/run_unified_rl.sh
```

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Configuration Parameters](docs/config_parameters.md) | Complete reference for all YAML config options |
| [RL Training Deep Dive](docs/rl_training_deep_dive.md) | GRPO algorithm, system architecture, training flow |
| [Qwen3-VL Multimodal Processing](docs/qwen3_vl_multimodal_processing.md) | Vision-language model internals |
| [Token Calculation](docs/token_calculation.md) | Token counting and memory estimation |

## ❓ FAQ

**Q: `Image features and image tokens do not match`**

A: Increase `data.max_prompt_length` or decrease `data.max_pixels`.

**Q: CUDA out of memory**

A: Decrease `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`.

**Q: Multi-node training hangs**

A: Run `ray status` to check the cluster. Ensure all nodes are connected and NCCL ports are open.

## 🙏 Acknowledgements

This project is built upon the excellent work of:
- [EasyR1](https://github.com/hiyouga/EasyR1) — Efficient, scalable RL training framework
- [veRL](https://github.com/volcengine/verl) — High-performance RL with HybridEngine
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — Vision-language model family

## 📄 Citation

If you use this project, please cite EasyR1 and veRL:

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong, Richong Zhang},
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

## 📜 License

This project follows the same license as [EasyR1](https://github.com/hiyouga/EasyR1).

## ☎️ We're Hiring!

We're hiring multimodal research scientists and interns at JD Explore Academy! If you have top-tier publications and are passionate about video understanding and VLMs, please send your resume to: siqingyi.phoebus@jd.com. We'd love to hear from you!
