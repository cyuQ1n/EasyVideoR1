<div align="center">
       
# EasyVideoR1: Easier RL for Video Understanding


<img src="https://img.shields.io/github/stars/cyuQ1n/EasyVideoR1.svg?style=social">
<a href="https://github.com/cyuQ1n/EasyVideoR1/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-red"></a>
<a href="https://github.com/cyuQ1n/EasyVideoR1/commits/main"><img src="https://img.shields.io/github/last-commit/cyuQ1n/EasyVideoR1?color=blue"></a>

[![arXiv](https://img.shields.io/badge/arXiv-2604.16893-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2604.16893)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/papers/2604.16893)
</div>

[中文](README_zh.md)

In our pursuit of advancing video understanding through post-training of multimodal LLMs, we found that existing RL frameworks were not particularly well-suited for video understanding scenarios. Therefore, we built **EasyVideoR1** to implement relevant optimizations, which we have outlined in this [report](https://github.com/cyuQ1n/EasyVideoR1/blob/main/EasyVideoR1-report.pdf). To the best of our knowledge, this should be the most suitable code repository for research on RL post-training for video understanding at the time of this report's release. It supports a wide range of video understanding tasks, incorporates research-friendly interfaces (mixed off-policy and on-policy training, joint image-video training), enhances training efficiency for video RL through systematic design, and provides an efficient, comprehensive, and accuracy-aligned evaluation framework. We hope this repository can inspire enthusiasm within the multimodal community for video understanding research. We also call upon community researchers to join us in maintaining this codebase, working together to create the most comprehensive and research-friendly repository for video understanding. We welcome and will consider merging any valuable pull requests.

> Branch note
>
> This branch is maintained as the **Qwen3.5 RL training branch**. The public training entry is [`examples/video_rl_qwen3.5/video_rl_v1_dapo.sh`](examples/video_rl_qwen3.5/video_rl_v1_dapo.sh), and the public config is [`examples/video_rl_qwen3.5/video_rl_v1_qwen3_5.yaml`](examples/video_rl_qwen3.5/video_rl_v1_qwen3_5.yaml). The tested runtime environment export is stored under [`docs/environment/`](docs/environment/README.md).

## 📍 Features

### Video Friendly Optimization
-   1. Offline Preprocessing and Cache-Based Training: **accelerates rollout generation by 1.5× and log-probability computation by 2.9×, achieving a 1.47× overall speedup in both wall-clock time per step and token throughput.**
-   2. Task-Aware Prompt and Reward Assignment System: **supports 10+ task types and their accuracy scoring/reward methods.** Specifically, EasyVideoR1 fully implements the following reward types by default: multiple choice, numerical, temporal grounding, spatial-temporal grounding, and open-ended QA. Prompt formatting is also available for additional task types including spatial grounding, tracking, OCR, boolean QA, math, and code generation.
-   3. More flexible video-hyperparameter settings: **Video metadata support for precise frame processing**
-   4. Advanced VLMs: supports **Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL** series vision-language models.
-   5. Rich RL Algorithms: inherited from [EasyR1](https://github.com/hiyouga/EasyR1), supports **GRPO, DAPO, GSPO, CISPO, Reinforce++, ReMax, RLOO, GDPO** and more.
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

This branch has been validated with the `easyvideor1-for-qwen3.5` environment. The authoritative environment exports are:

- [`docs/environment/easyvideor1-for-qwen3.5.conda.yaml`](docs/environment/easyvideor1-for-qwen3.5.conda.yaml)
- [`docs/environment/easyvideor1-for-qwen3.5.conda-explicit.txt`](docs/environment/easyvideor1-for-qwen3.5.conda-explicit.txt)
- [`docs/environment/easyvideor1-for-qwen3.5.pip-freeze.txt`](docs/environment/easyvideor1-for-qwen3.5.pip-freeze.txt)
- [`docs/environment/easyvideor1-for-qwen3.5.summary.txt`](docs/environment/easyvideor1-for-qwen3.5.summary.txt)

Recommended setup:

```bash
conda env create -f docs/environment/easyvideor1-for-qwen3.5.conda.yaml
conda activate easyvideor1-for-qwen3.5

git clone https://github.com/cyuQ1n/EasyVideoR1.git
cd EasyVideoR1
pip install -e .
```

If you prefer a lightweight install path, [`requirements.txt`](requirements.txt) contains the tested Python-level package set for this branch. The exported environment files remain the source of truth for reproducing the exact runtime.

Tested key versions:

- Python `3.11.14`
- PyTorch `2.10.0+cu129`
- Transformers `5.5.4`
- vLLM `0.19.1`
- Ray `2.54.0`
- qwen-vl-utils `0.0.14`
- flash-attn `2.8.3`
- flash-linear-attention `0.4.2`
- torchcodec `0.11.1`

## 🚀 Quick Start

For this branch, the recommended path is the Qwen3.5 video RL pipeline under `examples/video_rl_qwen3.5/`.

### Step 1: Check the Committed Example Paths

The committed Qwen3.5 example expects a preprocessed video RL layout:

- model: `/path/to/Qwen3.5-2B`
- train data: `/path/to/train.jsonl`
- val data: `/path/to/val.jsonl`
- preprocessed video cache: `/path/to/preprocessed_pt_dir`

These defaults are encoded in [`examples/video_rl_qwen3.5/video_rl_v1_qwen3_5.yaml`](examples/video_rl_qwen3.5/video_rl_v1_qwen3_5.yaml). If you are running outside our internal environment, replace them before launching.

You can override them either by editing the YAML directly or by exporting:

```bash
export EASYVIDEORL_MODEL_PATH=/abs/path/to/Qwen3.5-2B
export EASYVIDEORL_TRAIN_FILES=/abs/path/to/train.jsonl
export EASYVIDEORL_VAL_FILES=/abs/path/to/val.jsonl
export EASYVIDEORL_PREPROCESSED_VIDEO_DIR=/abs/path/to/preprocessed_pt_dir
export EASYVIDEORL_VAL_PREPROCESSED_VIDEO_DIR=/abs/path/to/preprocessed_pt_dir
```

### Step 2: Review the Qwen3.5 Training Files

- launch script: [`examples/video_rl_qwen3.5/video_rl_v1_dapo.sh`](examples/video_rl_qwen3.5/video_rl_v1_dapo.sh)
- training config: [`examples/video_rl_qwen3.5/video_rl_v1_qwen3_5.yaml`](examples/video_rl_qwen3.5/video_rl_v1_qwen3_5.yaml)
- adaptation notes: [`docs/qwen3_5_adaptation_notes.md`](docs/qwen3_5_adaptation_notes.md)

This branch currently packages the Qwen3.5 example around a generic preprocessed JSONL + PT-cache training layout and keeps the prompt/reward logic self-contained under `examples/video_rl_qwen3.5/`.

### Step 3: Launch Training

```bash
# Single-node
bash examples/video_rl_qwen3.5/video_rl_v1_dapo.sh

# Multi-node
WORLD_SIZE=2 RANK=0 MASTER_ADDR=<head_ip> bash examples/video_rl_qwen3.5/video_rl_v1_dapo.sh
WORLD_SIZE=2 RANK=1 MASTER_ADDR=<head_ip> bash examples/video_rl_qwen3.5/video_rl_v1_dapo.sh
```

The script defaults to the tested Conda environment name:

```bash
easyvideor1-for-qwen3.5
```

You can still override it with `CONDA_ENV_PREFIX` or `CONDA_ENV_NAME` if needed.

After training, merge FSDP checkpoints to Hugging Face format:

```bash
python3 scripts/model_merger.py --local_dir checkpoints/qwen3_5_onethinker100k/<experiment>/global_step_*/actor
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
│   ├── video_rl/               # Original video-only RL pipeline
│   ├── video_rl_qwen3.5/       # Public Qwen3.5-oriented video RL pipeline for this branch
│   └── unified_rl/             # Mixed image-video pipeline (modular reward)
├── eval/                       # Evaluation toolkit (25+ benchmarks)
├── scripts/                    # Checkpoint merger, video preprocessing
└── docs/                       # Detailed documentation and environment exports
```

## 🔧 Example Pipelines

### Video RL (`examples/video_rl/`)

A self-contained pipeline for video-only RL training. The reward function (`video_reward.py`) handles all task types in a single file with a simple `accuracy * 0.9 + format * 0.1` scoring formula.

```bash
bash examples/video_rl/run_video_rl.sh
```

### Qwen3.5 Video RL (`examples/video_rl_qwen3.5/`)

This is the main training path for the current branch. It includes:

- Qwen3.5 model support in `verl/models/transformers/qwen3_5.py`
- Qwen3.5-aware position id routing
- the multi-node launch entry `video_rl_v1_dapo.sh`
- the Qwen3.5 config `video_rl_v1_qwen3_5.yaml`

```bash
bash examples/video_rl_qwen3.5/video_rl_v1_dapo.sh
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
| [Qwen3.5 Adaptation Notes](docs/qwen3_5_adaptation_notes.md) | Qwen3.5-specific model and training changes in this branch |
| [Environment Exports](docs/environment/README.md) | Tested Conda/Pip environment artifacts for the Qwen3.5 branch |
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
- [ms-swift](https://github.com/modelscope/ms-swift) — Our Qwen3.5 environment setup also referenced the official [Qwen3.5 Best Practices](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html) and the `ms-swift` codebase.


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
