# Qwen3-VL 视觉 Token 计算与显存配置指南

本文档记录 Qwen3-VL 视觉 token 的计算方式、参数关系，以及 H200 显存优化配置建议。

---

## 1. 核心架构参数

Qwen3-VL 视觉编码器的固定参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `patch_size` | 16 | 将图像划分为 16×16 的 patch |
| `spatial_merge_size` | 2 | 将 2×2 个 patch 合并为一个 token |
| `temporal_patch_size` | 2 | 将连续 2 帧合并（时间维度压缩） |

**注意**: 这些参数是模型预训练时固定的架构参数，**不能随意修改**。修改会导致：
- 维度不匹配，直接报错
- 丧失预训练的视觉理解能力

---

## 2. 视觉 Token 计算公式

### 2.1 压缩比

| 压缩类型 | 压缩比 | 计算方式 |
|---------|--------|----------|
| 空间压缩 | 32× | patch_size(16) × spatial_merge_size(2) = 32 |
| 时间压缩 | 2× | temporal_patch_size = 2 |

### 2.2 Token 数量计算

**每帧视觉 token：**
```
每帧 tokens = (H / 32) × (W / 32)
```

**视频总 token：**
```
视频 tokens = (帧数 / 2) × (H / 32) × (W / 32)
                  ↑
         temporal_patch_size
```

### 2.3 计算示例

配置：`video_max_pixels: 262144` (512×512), `video_max_frames: 128`

```
每帧 tokens = (512 / 32) × (512 / 32) = 16 × 16 = 256 tokens/帧

视频 tokens = (128 / 2) × 256 = 64 × 256 = 16384 tokens
```

---

## 3. 像素预算与 Token 预算的换算

### 3.1 为什么需要 ×2

参考 [Qwen3-VL PR #1792](https://github.com/QwenLM/Qwen3-VL/pull/1792/files)：

```python
# 原来的配置（只考虑空间压缩，错误）
max_pixels = token_budget * 32 * 32

# 正确的配置（同时考虑时间压缩）
max_pixels = token_budget * 32 * 32 * 2
```

**原因**：视频处理需要同时考虑空间压缩（32×32）和时间压缩（2×），才能准确换算像素预算和 token 预算。

### 3.2 换算公式

```python
# 从 token 预算计算像素预算
max_pixels = max_visual_tokens * 32 * 32 * 2

# 从像素预算计算 token 数量
visual_tokens = (frames / 2) * (pixels / 32 / 32)
```

---

## 4. 配置参数对 Prompt 空间的影响

### 4.1 参数关系

| 配置项 | 示例值 | 说明 |
|--------|--------|------|
| video_max_frames | 128 | 最大帧数 |
| video_max_pixels | 262144 | 每帧最大像素 (512×512) |
| max_prompt_length | 16384 | prompt 最大 token 数 |
| max_response_length | 4096 | response 最大 token 数 |

### 4.2 空间分配示例

**满载配置（128帧，512×512）：**
```
视觉 tokens: 16384
max_prompt_length: 16384
剩余文本空间: ≈ 0 tokens  ← 问题！
```

补充：在之前的模型训练中使用64帧进行训练；max_pixels设置为262144。训练过程中的prompt length最大为8k左右，也基本符合上述计算。
![alt text](image.png)

**典型配置（60帧，480×320）：**
```
每帧 tokens = (480/32) × (320/32) = 15 × 10 = 150 tokens/帧
视频 tokens = (60/2) × 150 = 4500 tokens
剩余文本空间 = 16384 - 4500 ≈ 11884 tokens
```

### 4.3 调整建议

| 调整方案 | 修改后视觉 token | 释放文本空间 |
|---------|-----------------|-------------|
| video_max_pixels → 131072 | ~8192 | ~8000+ |
| video_max_frames → 64 | ~8192 | ~8000+ |
| 两者都降低 | ~4096 | ~12000+ |
| max_prompt_length → 32768 | - | 直接增加空间 |

---

## 5. H200 (140GB) 显存配置建议

### 5.1 显存占用分解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         显存占用分解 (8B 模型 + FSDP)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【Rollout 阶段 - vLLM】                【Training 阶段 - FSDP】              │
│  ┌─────────────────────┐               ┌─────────────────────┐              │
│  │ 模型权重 (TP=2分片)  │               │ 模型权重 (FSDP分片)  │              │
│  │ ≈ 8GB per GPU       │               │ ≈ 2GB per GPU       │              │
│  ├─────────────────────┤               ├─────────────────────┤              │
│  │ KV Cache            │               │ 优化器状态          │              │
│  │ ∝ seq_len × batch   │               │ ≈ 4GB per GPU       │              │
│  ├─────────────────────┤               ├─────────────────────┤              │
│  │ 视觉编码 pixel_values│               │ 梯度                │              │
│  │ ∝ frames × pixels   │               │ ≈ 2GB per GPU       │              │
│  └─────────────────────┘               ├─────────────────────┤              │
│                                        │ 激活值 (主要开销!)   │              │
│                                        │ ∝ seq_len × batch   │              │
│                                        └─────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 关键公式

```python
# 视觉 token 数量
visual_tokens = (video_max_frames / 2) * (H / 32) * (W / 32)

# 总序列长度
total_seq_len = max_prompt_length + max_response_length

# 激活显存 (粗略估计，有 gradient checkpointing)
activation_memory ≈ seq_len × micro_batch × hidden_size × num_layers × 2 bytes × 0.3~0.5

# KV Cache (vLLM rollout)
kv_cache_per_token ≈ 2 × num_layers × 2 × head_dim × num_kv_heads × 2 bytes / tp_size
```

### 5.3 推荐配置方案

#### 方案 A: 更大 batch + 更长序列（推荐）

```yaml
data:
  video_max_frames: 128
  video_max_pixels: 262144       # 512×512
  max_prompt_length: 24576       # 提升到 24K
  max_response_length: 8192      # 提升到 8K

worker:
  actor:
    micro_batch_size_per_device_for_update: 2
    micro_batch_size_per_device_for_experience: 2

  rollout:
    gpu_memory_utilization: 0.6
    max_num_batched_tokens: 33000  # > 24576 + 8192
```

#### 方案 B: 更多帧数

```yaml
data:
  video_max_frames: 256          # 翻倍帧数
  video_max_pixels: 262144
  max_prompt_length: 32768       # 对应提升
  max_response_length: 4096

worker:
  actor:
    micro_batch_size_per_device_for_update: 1
    micro_batch_size_per_device_for_experience: 1

  rollout:
    gpu_memory_utilization: 0.6
    max_num_batched_tokens: 37000
```

### 5.4 渐进式调优步骤

```yaml
# 第一步：测试基准（保守）
micro_batch_size_per_device_for_experience: 1
micro_batch_size_per_device_for_update: 1
max_prompt_length: 16384
max_response_length: 4096
# 预计显存: ~60-80GB per GPU

# 第二步：提升 batch size
micro_batch_size_per_device_for_experience: 2
micro_batch_size_per_device_for_update: 2
# 预计显存: ~90-110GB per GPU

# 第三步：提升序列长度
max_prompt_length: 24576
max_response_length: 8192
micro_batch_size_per_device_for_experience: 2
micro_batch_size_per_device_for_update: 1  # 可能需要降回来
# 预计显存: ~100-120GB per GPU
```

### 5.5 参数调优优先级

| 优先级 | 参数 | 建议 | 理由 |
|--------|------|------|------|
| 1 | micro_batch_size | 先尝试 2 | 直接提升训练吞吐 |
| 2 | max_prompt_length | 提升到 24K-32K | 给文本问题留空间 |
| 3 | gpu_memory_utilization | 0.6-0.7 | 充分利用 vLLM 显存 |
| 4 | video_max_frames | 根据数据集调整 | 看实际视频长度分布 |
| 5 | max_response_length | 8K | 更长的推理链 |

---

## 6. 显存监控命令

```bash
# 实时监控
watch -n 1 nvidia-smi

# Python 代码中监控
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

---

## 7. 相关参考

- [Qwen3-VL PR #1792](https://github.com/QwenLM/Qwen3-VL/pull/1792/files) - 像素范围修复（×2 的来源）
- [EasyVideoR1 多模态处理文档](./qwen3_vl_multimodal_processing.md) - 详细的数据处理流程
