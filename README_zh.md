# EasyVideoR1

[English](README.md)

基于 [EasyR1](https://github.com/hiyouga/EasyR1) 修改，支持 **Qwen3-VL** 系列模型进行视频理解强化学习训练。

本项目基于 EasyR1 和 [veRL](https://github.com/volcengine/verl) 的优秀工作。感谢所有作者提供如此高性能的强化学习训练框架。

## 特性

- 支持 **Qwen3-VL** 系列视觉语言模型
- **图像-视频混合训练**，优化梯度流
- 图像和视频分辨率独立控制
- 视频元数据支持，精确帧处理
- 默认示例 prompt/reward 栈面向视频理解任务
  - 默认 reward 已完整实现：多选题、数值计算、时间定位、时空定位、开放式问答
  - 额外还提供了空间定位、目标追踪、OCR、布尔问答、数学、代码生成等 prompt 模板分支，但默认 reward 脚本尚未完整实现这些任务

## 安装

### 第一步：创建 Conda 环境

```bash
conda create -n easyvideorl python=3.11
conda activate easyvideorl
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

## 快速开始

### 1. 准备数据集

数据集可以是 JSON 或 JSONL，字段结构如下：

```json
{
  "problem": "你的问题",
  "answer": "标准答案",
  "videos": ["path/to/video.mp4"],
  "data_type": "video",
  "problem_type": "multiple choice"
}
```

如果使用默认示例链路
(`examples/videorl/format_prompt/unified.jinja` + `examples/videorl/reward_function/video_reward.py`)，
请严格使用以下 `problem_type` 字符串：
- `multiple choice`
- `numerical`
- `temporal grounding`
- `spatial-temporal grounding`
- `open-ended`

其中，prompt 模板还包含 `regression`、`spatial grounding`、`tracking`、`ocr`、`boolean`、`math`、`code`、`svg-code`、`html-code`、`llava` 等分支，但默认 reward 脚本暂未完整支持这些任务类型。

### 2. 配置训练

编辑 `examples/videorl/video_rl.yaml`：

```yaml
data:
  train_files: /path/to/your/train.json
  val_files: /path/to/your/val.json
  image_dir: /path/to/your/data/root
  video_fps: 2.0
  video_max_frames: 64
  max_prompt_length: 16384
  max_response_length: 4096
  # 分辨率设置
  image_max_pixels: 1048576
  video_max_pixels: 262144

worker:
  actor:
    model:
      model_path: Qwen/Qwen3-VL-8B-Instruct
    # ... 其他设置

trainer:
  project_name: your_project
  experiment_name: your_experiment
  save_checkpoint_path: /path/to/checkpoints
```

### 3. 开始训练

```bash
bash examples/videorl/run_video_rl.sh
```

### 4. 合并检查点

```bash
python3 scripts/model_merger.py --local_dir checkpoints/your_exp/global_step_xxx/actor
```

## 配置参数

### 数据配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `video_fps` | 视频采样帧率 | 2.0 |
| `video_max_frames` | 每个视频最大帧数 | 128 |
| `image_max_pixels` | 图像最大像素数 | 1048576 |
| `video_max_pixels` | 视频最大像素数 | 262144 |

### Worker 配置

| 参数 | 说明 |
|------|------|
| `worker.rollout.tensor_parallel_size` | vLLM 推理的张量并行度 |
| `worker.rollout.gpu_memory_utilization` | vLLM 的 GPU 显存占用比例 |
| `worker.actor.fsdp.enable_full_shard` | 启用 FSDP 完全分片 |

## 常见问题

**问：Image features and image tokens do not match**

答：增大 `data.max_prompt_length` 或减小 `data.max_pixels`。

**问：CUDA out of memory**

答：减小 `worker.rollout.gpu_memory_utilization` 并启用 `worker.actor.offload.offload_params`。

**问：多节点训练卡住**

答：使用 `ray status` 检查 Ray 集群状态，确保所有节点已连接。

## 引用

如果使用本项目，请引用原版 EasyR1 和 veRL：

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

## 许可证

本项目遵循与 [EasyR1](https://github.com/hiyouga/EasyR1) 相同的许可证。

## ☎️ 我们正在招聘！
我们（京东探索研究院多模态理解研究部）正在招聘多模态大模型研究员和实习生！如果你有顶级会议/期刊论文发表，并对视频理解和视觉语言模型（VLMs）充满热情，请将简历发送至：siqingyi.phoebus@jd.com。期待你的加入！
