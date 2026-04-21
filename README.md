# EasyVideoR1: Easier RL for Video Understanding

[中文](README_zh.md)

In our pursuit of advancing video understanding through post-training of multimodal LLMs, we found that existing RL frameworks were not particularly well-suited for video understanding scenarios. Therefore, we built **EasyVideoR1** to implement relevant optimizations, which we have outlined in this [report](https://github.com/cyuQ1n/EasyVideoR1/blob/main/EasyVideoR1-report.pdf). To the best of our knowledge, this should be the most suitable code repository for research on RL post-training for video understanding at the time of this report's release. It supports a wide range of video understanding tasks, incorporates research-friendly interfaces (mixed off-policy and on-policy training, joint image-video training), enhances training efficiency for video RL through systematic design, and provides an efficient, comprehensive, and accuracy-aligned evaluation framework. We hope this repository can inspire enthusiasm within the multimodal community for video understanding research. We also call upon community researchers to join us in maintaining this codebase, working together to create the most comprehensive and research-friendly repository for video understanding. We welcome and will consider merging any valuable pull requests.

## 📍 Features

### Video Friendly Optimization
-   1. Offline Preprocessing and Cache-Based Training: **accelerates rollout generation by 1.5× and log-probability computation by 2.9×, achieving a 1.47× overall speedup in both wall-clock time per step and token throughput.**
-   2. Task-Aware Prompt and Reward Assignment System: **supports 10+ task types and their accuracy scoring/reward methods.** Specifically, EasyVideoR1 fully implements the following reward types by default: multiple choice, numerical, temporal grounding, spatial-temporal grounding, and open-ended QA. Prompt formatting is also available for additional task types including spatial grounding, tracking, OCR, boolean QA, math, and code generation.
-   3. More flexible video-hyperparameter settings: **Video metadata support for precise frame processing**
-   4. Advanced VLMs: supports **Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL** series vision-language models.
-   5. Rich RL Algorithms: inherited from [EasyR1](https://github.com/hiyouga/EasyR1), supports **GRPO, DAPO, GSPO, CISPO, Reinforce++, ReMax, RLOO** and more.
### Research-Friendly Interfaces for Algorithm Development
-   1. Mixed-Modality Pipeline Adaptation: **supports joint Text-Image-Video training with optimized gradient flow.**
-   2. A Lightweight Mix-policy Interface: **supports hybrid online-offline training.**
### Fast & Comprehensive Evaluation Framework
-   1. Asynchronous Inference: **Precomputed Frame Caching** and **Asynchronous Pipeline with AsyncLLMEngine**  ensure that the GPU remains productive at every scheduling step: cached I/O feeds data continuously, asynchronous queuing removes batch-boundary stalls, and chunked prefill prevents any single long sequence from monopolizing compute.
-   2. Comprehensive and reproducible evaluation: **supports 22+ Video Understanding Benchmarks.**
-   3. Accuracy-aligned: for Qwen3-VL series, **evaluation results align with official scores (within 1% deviation).**
       
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
│   ├── models/                 # Qwen2-VL / Qwen2.5-VL / Qwen3-VL model support
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
- [OneThinker](https://github.com/tulerfeng/OneThinker) - All-in-one Reasoning Model for Image and Video


## 📄 Citation

If you use this project, please cite:

```bibtex
@misc{qin2026easyvideor1,
  title        = {EasyVideoR1: Easier RL for Video Understanding},
  author       = {Chuanyu Qin, Chenxu Yang, Qingyi Si, Naibin Gu, Dingyu Yao, Zheng Lin, Peng Fu, Nan Duan, Jiaqi Wang},
  howpublished = {\url{https://github.com/cyuQ1n/EasyVideoR1}},
  year         = {2026}
}

```

## 📜 License

This project follows the same license as [EasyR1](https://github.com/hiyouga/EasyR1).

## ☎️ We're Hiring!

We're hiring multimodal research scientists and interns at JD Explore Academy! If you have top-tier publications and are passionate about video understanding and VLMs, please send your resume to: siqingyi.phoebus@jd.com. We'd love to hear from you!
