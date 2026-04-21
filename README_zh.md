# EasyVideoR1: Easier RL for Video Understanding

[English](README.md)

在推进多模态大语言模型后训练以提升视频理解能力的过程中，我们发现现有的强化学习框架在视频理解场景中适配的不太好。因此，我们构建了EasyVideoR1来实现相关优化，具体内容已在[report](https://github.com/cyuQ1n/EasyVideoR1/blob/main/EasyVideoR1-report.pdf)中概述。据我们所知，这应该是迄今为止最适合视频理解RL研究的代码仓库。它支持广泛的视频理解任务，融入了适合社区进行后训练研究和探索的接口（off-policy与on-policy混合训练、图像-视频联合训练），通过系统化设计提升了视频强化学习的训练效率，并提供了高效、全面且与已对齐准确率的视频理解评测框架。我们希望这个仓库能够激发多模态社区对视频理解研究的热情。我们也呼吁社区研究人员加入我们，共同维护这个代码库，携手打造最全面、最适合学术探索的视频理解RL仓库。我们欢迎并会认真考虑合并任何有价值的pull request。

## 📍 特性

### 视频友好的RL管线优化
-  1. 离线预处理与基于缓存的训练：rollout生成加速1.5倍，log-probability计算加速2.9倍，**整体单步时间和token吞吐量均实现1.47倍加速**。
-  2. 任务感知提示与奖励分配系统：**支持10+种任务类型及其准确率评分/奖励方法**。具体而言，EasyVideoR1默认完整实现了以下奖励类型：选择题、数值题、时序定位、时空定位、开放式问答。此外，提示词格式也已为以下额外任务类型准备就绪：空间定位、Tracking、OCR、布尔问答、数学和代码生成。
-  3. 更灵活的视频超参数设置：支持视频元数据以实现精确的帧处理。
-  4. 先进视觉语言模型：支持 **Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL** 系列视觉语言模型。
-  5. 丰富的强化学习算法：继承自 [EasyR1](https://github.com/hiyouga/EasyR1)，支持 **GRPO、DAPO、GSPO、CISPO、Reinforce++、ReMax、RLOO、GDPO** 等多种算法。
### 算法开发研究友好接口
-  1. 混合模态流程适配：通过优化梯度流，**支持联合文本-图像-视频训练**。
-  2. 轻量级混合策略接口：**支持在线-离线混合训练（mix policy）**。
### 快速全面的评估框架
-  1. 异步推理：预计算帧缓存与异步流水线结合AsyncLLMEngine，确保GPU在每个调度步骤都保持高效：缓存I/O持续供给数据，异步队列消除批次边界停滞，分块预填充防止任何单一长序列独占计算资源。
-  2. 全面且可复现的评估：支持22+个视频理解基准测试。
-  3. 精度对齐：对于Qwen3-VL系列模型，**评测结果与官方精度对齐（偏差在1%以内）**。
      
## 🏆 性能

使用 EasyVideoR1 训练后，在 10 个视频理解基准测试上相比 Qwen3-VL-8B 基座模型取得了一致的提升，平均准确率提升 **+2.3%**。

<div align="center">
  <img src="assets/benchmark.png" alt="基准测试结果" width="90%">
</div>

视频预处理缓存机制相比实时解码，将单步训练速度提升 **1.47 倍**，且不影响精度。

<div align="center">
  <img src="assets/efficiency.png" alt="训练效率" width="90%">
</div>

## 📐 安装

### 第一步：创建 Conda 环境

```bash
conda create -n easyvideor1 python=3.11
conda activate easyvideor1
```

### 第二步：克隆并安装

```bash
git clone https://github.com/cyuQ1n/EasyVideoR1.git
cd EasyVideoR1
pip install -e .
```

### 第三步：安装 Flash Attention

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

## 🚀 快速开始

以下是最简 3 步启动训练的流程。

### 第一步：准备数据

创建 JSON/JSONL 文件，每条数据格式如下：

```json
{
  "problem": "这个视频中发生了什么？",
  "answer": "一只猫跳上了桌子。",
  "videos": ["path/to/video.mp4"],
  "data_type": "video",
  "problem_type": "open-ended"
}
```

多选题需要额外添加 `options` 字段：

```json
{
  "problem": "视频中汽车是什么颜色的？",
  "answer": "B",
  "videos": ["path/to/video.mp4"],
  "data_type": "video",
  "problem_type": "multiple choice",
  "options": ["A. 红色", "B. 蓝色", "C. 绿色", "D. 白色"]
}
```

> 完整的 `problem_type` 列表和数据字段说明请参考 [docs/config_parameters.md](docs/config_parameters.md)。

### 第二步：编辑配置

复制并编辑示例配置：

```bash
cp examples/video_rl/video_rl.yaml my_config.yaml
```

至少修改以下字段：

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

### 第三步：启动训练

```bash
# 单机训练（8 GPU）
bash examples/video_rl/run_video_rl.sh

# 多机训练：在每个节点上设置 WORLD_SIZE、RANK、MASTER_ADDR
WORLD_SIZE=2 RANK=0 MASTER_ADDR=<主节点IP> bash examples/video_rl/run_video_rl.sh  # 主节点
WORLD_SIZE=2 RANK=1 MASTER_ADDR=<主节点IP> bash examples/video_rl/run_video_rl.sh  # 工作节点
```

训练完成后，将 FSDP 检查点合并为 Hugging Face 格式：

```bash
python3 scripts/model_merger.py --local_dir checkpoints/my_first_run/global_step_100/actor
```

## 📂 项目结构

```
EasyVideoR1/
├── verl/                       # 核心 RL 训练框架
│   ├── trainer/                # 训练循环 & Ray 编排
│   ├── workers/                # Actor、rollout、reward、critic workers
│   ├── models/                 # Qwen2-VL / Qwen2.5-VL / Qwen3-VL 模型支持
│   └── utils/                  # 数据集、分词、FSDP 工具
├── examples/
│   ├── video_rl/               # 纯视频 RL 管线（单文件 reward）
│   └── unified_rl/             # 图文视频混合管线（模块化 reward）
├── eval/                       # 评测工具集（25+ 基准测试）
├── scripts/                    # 检查点合并、视频预处理
└── docs/                       # 详细文档
```

## 🔧 示例管线

### Video RL (`examples/video_rl/`)

单文件自包含的纯视频 RL 训练管线。reward 函数 (`video_reward.py`) 在一个文件中处理所有任务类型，使用 `accuracy * 0.9 + format * 0.1` 的简单加权公式。

```bash
bash examples/video_rl/run_video_rl.sh
```

### Unified RL (`examples/unified_rl/`)

模块化的图文视频混合训练管线。reward 函数根据每条样本的 `problem_type` 自动路由到对应的任务模块（多选题、grounding、数学等），各模块独立评分。

```bash
bash examples/unified_rl/run_unified_rl.sh
```

## 📖 详细文档

| 文档 | 说明 |
|------|------|
| [配置参数参考](docs/config_parameters.md) | 所有 YAML 配置选项的完整说明 |
| [RL 训练深度解析](docs/rl_training_deep_dive.md) | GRPO 算法、系统架构、训练流程 |
| [Qwen3-VL 多模态处理](docs/qwen3_vl_multimodal_processing.md) | 视觉语言模型内部机制 |
| [Token 计算](docs/token_calculation.md) | Token 计数与显存估算 |

## ❓ 常见问题

**问：`Image features and image tokens do not match`**

答：增大 `data.max_prompt_length` 或减小 `data.max_pixels`。

**问：CUDA out of memory**

答：减小 `worker.rollout.gpu_memory_utilization` 并启用 `worker.actor.offload.offload_params`。

**问：多节点训练卡住**

答：使用 `ray status` 检查集群状态，确保所有节点已连接且 NCCL 端口已开放。

## 🙏 致谢

本项目基于以下优秀工作构建：
- [EasyR1](https://github.com/hiyouga/EasyR1) — 高效可扩展的 RL 训练框架
- [veRL](https://github.com/volcengine/verl) — 高性能 RL 与 HybridEngine
- [OneThinker](https://github.com/tulerfeng/OneThinker) - 图和视频的联合RL框架

## 📄 引用

如果使用本项目，请引用 EasyR1 和 veRL：

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyVideoR1: Easier RL for Video Understanding},
  author       = {},
  howpublished = {\url{https://github.com/cyuQ1n/EasyVideoR1}},
  year         = {2026}
}
```

## 📜 许可证

本项目遵循与 [EasyR1](https://github.com/hiyouga/EasyR1) 相同的许可证。

## ☎️ 我们正在招聘！

我们（京东探索研究院多模态理解研究部）正在招聘多模态大模型研究员和实习生！如果你有顶级会议/期刊论文发表，并对视频理解和视觉语言模型（VLMs）充满热情，请将简历发送至：siqingyi.phoebus@jd.com。期待你的加入！
