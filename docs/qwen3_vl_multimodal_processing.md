# Qwen3-VL 多模态数据处理指南

本文档详细说明 Qwen3-VL 在 EasyVideoR1 训练框架中视频和图像数据的完整处理流程，包括各阶段涉及的代码、关键参数的作用，以及如何确保各阶段处理的一致性。

## 目录

1. [概述](#1-概述)
2. [关键参数说明](#2-关键参数说明)
3. [数据处理流程](#3-数据处理流程)
4. [各阶段代码详解](#4-各阶段代码详解)
5. [Token 与 Feature 的对应关系](#5-token-与-feature-的对应关系)
6. [常见问题与解决方案](#6-常见问题与解决方案)
7. [参数配置检查清单](#7-参数配置检查清单)

---

## 1. 概述

EasyVideoR1 训练 Qwen3-VL 视频模型时，多模态数据会经过以下三个主要阶段：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           训练流程概览                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Dataset 阶段        2. Rollout 阶段         3. Training 阶段            │
│  ┌─────────────────┐   ┌─────────────────┐    ┌─────────────────┐          │
│  │ 读取视频/图像    │   │ vLLM 推理生成    │    │ FSDP 训练更新    │          │
│  │ 预处理 + 分词    │──▶│ 重新处理视频     │───▶│ 重新处理视频     │          │
│  │ 生成 input_ids  │   │ 生成 response   │    │ 计算 loss       │          │
│  └─────────────────┘   └─────────────────┘    └─────────────────┘          │
│         │                     │                      │                     │
│         ▼                     ▼                      ▼                     │
│  必须使用相同的          必须使用相同的           必须使用相同的              │
│  min_pixels/max_pixels  min_pixels/max_pixels   min_pixels/max_pixels      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**核心原则：三个阶段必须使用完全相同的视频/图像处理参数，否则会导致 token 数量与 feature 数量不匹配。**

---

## 2. 关键参数说明

### 2.1 配置文件中的参数

在 `config_video_r1.yaml` 中：

```yaml
data:
  min_pixels: 3136        # 最小像素数 (56×56)
  max_pixels: 131072      # 最大像素数 (~362×362)
  video_fps: 2.0          # 视频采样帧率
```

### 2.2 参数作用详解

| 参数 | 说明 | 影响 |
|------|------|------|
| `min_pixels` | 视频帧/图像的最小总像素数 | 太小的图像会被放大到此尺寸 |
| `max_pixels` | 视频帧/图像的最大总像素数 | 太大的图像会被缩小到此尺寸 |
| `video_fps` | 从视频中采样的帧率 | 影响提取的帧数，进而影响 token 数量 |

### 2.3 像素数与 Token 数的关系

Qwen3-VL 的视觉编码器使用以下参数：
- `patch_size = 16`：将图像划分为 16×16 的 patch
- `spatial_merge_size = 2`：将 2×2 个 patch 合并为一个 token
- `temporal_patch_size = 2`：将连续 2 帧合并

**Token 计算公式：**
```
每帧 tokens = (H / 32) × (W / 32)
视频 tokens = (帧数 / 2) × (H / 32) × (W / 32)
```

**示例：**
```
max_pixels = 131072 (~362×362)
假设视频帧为 256×480 = 122880 像素，36 帧

每帧 tokens = (256/32) × (480/32) = 8 × 15 = 120
视频 tokens = (36/2) × 120 = 18 × 120 = 2160
```

---

## 3. 数据处理流程

### 3.1 完整数据流

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              数据处理流水线                                    │
└──────────────────────────────────────────────────────────────────────────────┘

原始视频文件
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 1: fetch_video (qwen_vl_utils)                                         │
│ ─────────────────────────────────────                                       │
│ 输入: video_path, fps, min_pixels, max_pixels                               │
│ 输出: (frames, metadata), sample_fps                                        │
│                                                                             │
│ 处理内容:                                                                    │
│   1. 读取视频文件 (decord/torchvision)                                       │
│   2. 按 fps 采样帧                                                           │
│   3. 调整分辨率到 min_pixels ~ max_pixels 范围                                │
│   4. 返回 frames (T, C, H, W) 和 metadata (fps, frames_indices, ...)        │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 2: video_processor (transformers)                                      │
│ ─────────────────────────────────────                                       │
│ 输入: videos, video_metadata, do_resize, do_sample_frames                   │
│ 输出: pixel_values_videos, video_grid_thw                                   │
│                                                                             │
│ 处理内容:                                                                    │
│   1. 如果 do_resize=True, 再次调整尺寸                                       │
│   2. 如果 do_sample_frames=True, 再次采样帧                                  │
│   3. 将视频转换为 patch 序列                                                 │
│   4. 返回 pixel_values (num_patches, hidden_dim) 和 grid_thw (T, H, W)      │
│                                                                             │
│ 重要: 如果视频已预处理, 应设置 do_resize=False, do_sample_frames=False       │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 3: processor (Qwen3VLProcessor)                                        │
│ ─────────────────────────────────────                                       │
│ 输入: text, videos, video_metadata                                          │
│ 输出: input_ids, attention_mask, pixel_values_videos, video_grid_thw        │
│                                                                             │
│ 处理内容:                                                                    │
│   1. 将 <video> 占位符替换为正确数量的 <|video_pad|> tokens                   │
│   2. 调用 video_processor 处理视频                                           │
│   3. 组合文本 tokens 和视频 tokens                                           │
│                                                                             │
│ 关键: input_ids 中的 video_pad 数量 = video_grid_thw 的乘积 / 4 (spatial)    │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 4: visual encoder (model.visual)                                       │
│ ─────────────────────────────────────                                       │
│ 输入: pixel_values_videos, grid_thw                                         │
│ 输出: video_embeds                                                          │
│                                                                             │
│ 处理内容:                                                                    │
│   1. Patch embedding                                                        │
│   2. Spatial merge (2×2 patches → 1 embedding)                              │
│   3. 返回 video_embeds (num_tokens, hidden_dim)                              │
│                                                                             │
│ 验证: video_embeds.shape[0] 必须 == input_ids 中 video_pad 的数量            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 各阶段代码详解

### 4.1 Dataset 阶段

**文件**: `verl/utils/dataset.py`

**关键函数**: `RLHFDataset.__getitem__`

```python
# 位置: dataset.py, 约 332-342 行
for video in videos:
    # 必须传入 min_pixels, max_pixels, video_fps
    processed_video, video_fps = process_video(
        video,
        min_pixels=self.min_pixels if self.min_pixels else 4 * 32 * 32,
        max_pixels=self.max_pixels if self.max_pixels else 64 * 32 * 32,
        video_fps=self.video_fps,
        return_fps=True
    )
    processed_videos.append(processed_video)
```

**处理逻辑**:
1. 调用 `process_video` 预处理视频
2. 使用 `self.processor` 进行 tokenization
3. 生成 `input_ids` 和 `multi_modal_data`

**输出**:
- `input_ids`: 包含 `<|video_pad|>` tokens
- `multi_modal_data`: `{"videos": [video_path, ...]}` (存储路径，非帧数据)

**注意**: Dataset 只存储视频路径，实际帧数据在后续阶段重新处理。这是为了避免内存占用过大。

### 4.2 Rollout 阶段 (vLLM)

**文件**: `verl/workers/rollout/vllm_rollout_spmd.py`

**关键函数**: `_process_multi_modal_data`

```python
# 位置: vllm_rollout_spmd.py, 约 74-98 行
if "videos" in multi_modal_data:
    for video in multi_modal_data["videos"]:
        processed, _ = process_video(
            video,
            min_pixels=min_pixels,      # 从 meta_info 获取
            max_pixels=max_pixels,      # 从 meta_info 获取
            video_fps=video_fps,        # 从 meta_info 获取
            return_fps=True
        )
        if isinstance(processed, tuple) and len(processed) == 2:
            frames, metadata = processed
            videos.append((frames, metadata))  # vLLM 需要 (frames, metadata) 元组

    if len(videos) > 0:
        mm_kwargs = {"do_sample_frames": False, "do_resize": False}
```

**处理逻辑**:
1. 从 `meta_info` 获取配置参数
2. 重新调用 `process_video` 处理视频
3. 将 `(frames, metadata)` 元组传递给 vLLM
4. 设置 `do_resize=False, do_sample_frames=False` 防止重复处理

**vLLM 内部处理** (`vllm/model_executor/models/qwen3_vl.py`):
```python
# 约 826-856 行
for item in mm_data["videos"]:
    video_array, metadata = item  # 期望元组格式

    # 创建 VideoMetadata
    metadata = VideoMetadata(**{
        k: metadata[k]
        for k in metadata if k != "do_sample_frames"
    })

    # 调用 HuggingFace processor
    video_outputs = super()._call_hf_processor(
        prompt="<|vision_start|><|video_pad|><|vision_end|>",
        mm_data={"videos": [[video_array]], "video_metadata": [[metadata]]},
        mm_kwargs=video_mm_kwargs,  # 包含 do_resize=False
        ...
    )
```

### 4.3 Training 阶段 (FSDP)

**文件**: `verl/workers/fsdp_workers.py`

**关键函数**: `ActorRolloutRefWorker._process_multi_modal_inputs`

```python
# 位置: fsdp_workers.py, 约 484-532 行
if "videos" in multi_modal_data:
    video_metadatas = []
    for video in multi_modal_data["videos"]:
        result = process_video(
            video,
            min_pixels=min_pixels,      # 从 data.meta_info 获取
            max_pixels=max_pixels,      # 从 data.meta_info 获取
            video_fps=video_fps,        # 从 data.meta_info 获取
            return_fps=True
        )
        if isinstance(result, tuple) and len(result) == 2:
            video_data, _ = result
            if isinstance(video_data, tuple) and len(video_data) == 2:
                frames, metadata = video_data
                videos.append(frames)
                video_metadatas.append(metadata)

    # 调用 video_processor
    processor_kwargs = {
        "videos": videos,
        "return_tensors": "pt",
        "do_resize": False,           # 已在 process_video 中 resize
        "do_sample_frames": False,    # 已在 process_video 中采样
    }
    if video_metadatas is not None:
        processor_kwargs["video_metadata"] = video_metadatas

    multi_modal_inputs = dict(self.processor.video_processor(**processor_kwargs))
```

**验证阶段** (`verl/models/transformers/qwen3_vl.py`):
```python
# 约 167-172 行
n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
n_video_features = video_embeds.shape[0]
if n_video_tokens != n_video_features:
    raise ValueError(
        f"Video features and video tokens do not match: "
        f"tokens: {n_video_tokens}, features {n_video_features}"
    )
```

---

## 5. Token 与 Feature 的对应关系

### 5.1 数量对应

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Token 与 Feature 的关系                             │
└─────────────────────────────────────────────────────────────────────────────┘

pixel_values_videos.shape[0] = T × H × W (patches)
                               ↓
                        Visual Encoder
                               ↓
video_embeds.shape[0] = T × (H/2) × (W/2) (spatial merge)
                               ↓
                        必须等于
                               ↓
input_ids 中 video_pad 的数量 = T/2 × H/2 × W/2 (temporal + spatial merge)

示例:
- video_grid_thw = [18, 16, 30]
- pixel_values: 18 × 16 × 30 = 8640 patches
- video_embeds: 18 × 8 × 15 = 2160 embeddings (spatial merge 2×2)
- video_pad tokens: 2160 (必须匹配)
```

### 5.2 不匹配的原因

| 错误场景 | 原因 | 解决方案 |
|---------|------|---------|
| tokens > features | Dataset 使用高分辨率，Training 使用低分辨率 | 统一 max_pixels |
| tokens < features | Dataset 使用低分辨率，Training 使用高分辨率 | 统一 max_pixels |
| Warning: no video metadata | metadata 未正确传递 | 检查 (frames, metadata) 元组格式 |

---

## 6. 常见问题与解决方案

### 6.1 ValueError: Video features and video tokens do not match

**错误信息**:
```
ValueError: Video features and video tokens do not match: tokens: 27846, features 4954
```

**原因分析**:
- 比例 27846/4954 ≈ 5.6，表示分辨率差异约 2.4 倍
- Dataset 使用了默认 max_pixels，Training 使用了配置的 max_pixels

**解决方案**:
检查 `dataset.py` 中 `__getitem__` 的 `process_video` 调用是否传入了正确参数:
```python
# 正确写法
processed_video, video_fps = process_video(
    video,
    min_pixels=self.min_pixels if self.min_pixels else 4 * 32 * 32,
    max_pixels=self.max_pixels if self.max_pixels else 64 * 32 * 32,
    video_fps=self.video_fps,
    return_fps=True
)
```

### 6.2 Warning: Asked to sample fps frames per second but no video metadata was provided

**警告信息**:
```
Asked to sample fps frames per second but no video metadata was provided
```

**原因**:
- `video_metadata` 未正确传递给 processor
- metadata 缺少 `fps` 字段

**解决方案**:
确保 process_video 返回的 metadata 包含必要字段:
```python
metadata = {
    'fps': 30.0,                    # 原始视频 FPS
    'frames_indices': [0, 16, ...], # 采样的帧索引
    'total_num_frames': 561,        # 原始总帧数
    'video_backend': 'decord'       # 可选
}
```

### 6.3 The given max_pixels exceeds limit

**警告信息**:
```
The given max_pixels[1048576] exceeds limit[786432]
```

**原因**:
- max_pixels 设置过大，超过 qwen_vl_utils 的限制

**解决方案**:
```yaml
# 推荐配置
max_pixels: 131072  # ~362×362，适合 16K prompt length
# 或
max_pixels: 262144  # 512×512，需要更大 prompt length
```

---

## 7. 参数配置检查清单

### 7.1 配置文件检查

```yaml
# config_video_r1.yaml
data:
  min_pixels: 3136        # ✓ 必须配置
  max_pixels: 131072      # ✓ 必须配置
  video_fps: 2.0          # ✓ 必须配置
  max_prompt_length: 16384  # ✓ 必须 > video tokens + text tokens
```

### 7.2 代码检查点

| 文件 | 位置 | 检查项 |
|------|------|--------|
| `dataset.py` | `__getitem__` | process_video 是否传入 min_pixels, max_pixels, video_fps |
| `dataset.py` | `_filter_overlong_prompts` | process_video 参数是否一致 |
| `vllm_rollout_spmd.py` | `_process_multi_modal_data` | 是否使用 meta_info 中的参数 |
| `fsdp_workers.py` | `_process_multi_modal_inputs` | 是否使用 meta_info 中的参数 |

### 7.3 验证命令

```python
# 验证各阶段分辨率一致性
from verl.utils.dataset import process_video

video_path = "your_video.mp4"
min_pixels = 3136
max_pixels = 131072
video_fps = 2.0

result = process_video(
    video_path,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    video_fps=video_fps,
    return_fps=True
)
(frames, metadata), _ = result
print(f"Resolution: {frames.shape[2]}x{frames.shape[3]}")
print(f"Frames: {frames.shape[0]}")
print(f"Expected tokens: {frames.shape[0]//2 * (frames.shape[2]//32) * (frames.shape[3]//32)}")
```

---

## 附录: 相关文件索引

| 文件路径 | 功能 |
|---------|------|
| `verl/utils/dataset.py` | Dataset 定义，视频预处理和 tokenization |
| `verl/workers/rollout/vllm_rollout_spmd.py` | vLLM Rollout，推理时的视频处理 |
| `verl/workers/fsdp_workers.py` | FSDP Training，训练时的视频处理 |
| `verl/models/transformers/qwen3_vl.py` | Qwen3-VL 模型集成，包含验证逻辑 |
| `qwen_vl_utils/vision_process.py` | fetch_video 实现，底层视频读取 |
| `transformers/.../video_processing_qwen3_vl.py` | HuggingFace video_processor |
