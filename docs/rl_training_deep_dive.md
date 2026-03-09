# RL Training 深度解析

本文档基于 EasyVideoR1 框架（基于 verl），由浅至深讲解多模态大模型 RL 训练的完整流程。

---

## 目录

1. [基础概念](#1-基础概念)
2. [框架整体架构](#2-框架整体架构)
3. [一个训练 Step 的完整流程](#3-一个训练-step-的完整流程)
4. [数据流转与内存模型](#4-数据流转与内存模型)
5. [分布式训练细节](#5-分布式训练细节)
6. [关键配置参数详解](#6-关键配置参数详解)
7. [常见问题与调优指南](#7-常见问题与调优指南)

---

## 1. 基础概念

### 1.1 为什么要用 RL 训练大模型？

传统的 SFT（监督微调）是让模型模仿标准答案。而 RL 训练的思路不同：

```
SFT:  给模型看标准答案 → 模型学习模仿 → 只能学到一种解题方式
RL:   让模型自己尝试回答 → 给回答打分 → 模型学习得分高的策略
```

RL 的优势在于**探索性**——模型可以发现训练数据中没有的解题路径，只要结果正确即可获得奖励。

### 1.2 GRPO 算法简介

本框架使用 **GRPO（Group Relative Policy Optimization）** 作为核心 RL 算法。

GRPO 的核心思想很简单：

```
1. 对同一个问题，让模型生成 n 个不同的回答（Group）
2. 给每个回答打分（Reward）
3. 在同一组内，做得好的回答获得正的优势值（Advantage），做得差的获得负的优势值
4. 用这些优势值来更新模型：增大好回答的概率，减小差回答的概率
```

与传统 PPO 不同，GRPO **不需要 Critic 模型**（价值网络），而是通过组内相对比较来估计优势值，大大减少了显存占用。

关键公式（简化版）：

```
Advantage_i = (reward_i - mean(rewards_in_group)) / std(rewards_in_group)

Loss = -E[ min(ratio × advantage, clip(ratio, 1-ε, 1+ε) × advantage) ]

其中 ratio = π_new(action|state) / π_old(action|state)
          = exp(new_log_prob - old_log_prob)
```

### 1.3 训练涉及的模型角色

| 角色 | 说明 | 是否可训练 |
|------|------|:---:|
| **Actor（策略模型）** | 当前正在训练的模型，负责生成回答 | 是 |
| **Rollout 引擎** | 用 vLLM 高效生成回答，使用 Actor 的权重 | 否（与 Actor 共享权重） |
| **Reference（参考模型）** | 训练开始时的模型快照，用于计算 KL 散度 | 否（冻结） |
| **Reward（奖励函数）** | 给回答打分，通常是规则函数（非模型） | 否 |

在本框架中，Actor、Rollout、Reference 被**共置（colocated）**在同一组 GPU 上，通过权重同步和内存卸载来复用 GPU 资源。

---

## 2. 框架整体架构

### 2.1 系统架构图

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Head 节点 (Driver)                            │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  RayPPOTrainer.fit()  —— 主训练循环                            │  │
│  │                                                                │  │
│  │  1. _make_batch_data()     ← 调度 Rollout 生成数据             │  │
│  │  2. _balance_batch()       ← 均衡各 GPU 的 token 负载          │  │
│  │  3. compute_reward()       ← 异步调度 Reward 计算              │  │
│  │  4. compute_log_probs()    ← 调度各 GPU 计算 log π_old         │  │
│  │  5. compute_ref_log_probs()← 调度各 GPU 计算 log π_ref         │  │
│  │  6. compute_advantage()    ← 在 Driver 本地计算优势值           │  │
│  │  7. update_actor()         ← 调度各 GPU 执行 PPO 更新          │  │
│  │  8. _validate() / _save()  ← 验证 & 保存                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                          ↕ Ray RPC 调度                              │
├──────────────────────────────────────────────────────────────────────┤
│                     Worker 节点群 (128 GPUs)                         │
│                                                                      │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐    ┌────────────┐    │
│  │  GPU 0     │ │  GPU 1     │ │  GPU 2     │... │  GPU 127   │    │
│  │            │ │            │ │            │    │            │    │
│  │ ┌────────┐ │ │ ┌────────┐ │ │ ┌────────┐ │    │ ┌────────┐ │    │
│  │ │ Actor  │ │ │ │ Actor  │ │ │ │ Actor  │ │    │ │ Actor  │ │    │
│  │ │ (FSDP) │ │ │ │ (FSDP) │ │ │ │ (FSDP) │ │    │ │ (FSDP) │ │    │
│  │ ├────────┤ │ │ ├────────┤ │ │ ├────────┤ │    │ ├────────┤ │    │
│  │ │ vLLM   │ │ │ │ vLLM   │ │ │ │ vLLM   │ │    │ │ vLLM   │ │    │
│  │ │Rollout │ │ │ │Rollout │ │ │ │Rollout │ │    │ │Rollout │ │    │
│  │ ├────────┤ │ │ ├────────┤ │ │ ├────────┤ │    │ ├────────┤ │    │
│  │ │  Ref   │ │ │ │  Ref   │ │ │ │  Ref   │ │    │ │  Ref   │ │    │
│  │ │ Policy │ │ │ │ Policy │ │ │ │ Policy │ │    │ │ Policy │ │    │
│  │ └────────┘ │ │ └────────┘ │ │ └────────┘ │    │ └────────┘ │    │
│  └────────────┘ └────────────┘ └────────────┘    └────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Reward Workers (CPU)  —— 独立进程，不占用 GPU               │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件说明

#### Driver 进程（Head 节点上的 RayPPOTrainer）

- **不占用 GPU**，仅使用 CPU
- 负责**调度**所有训练阶段，决定什么时候做 rollout、什么时候做训练
- 持有完整的 **batch 数据**（存在 CPU 内存 / Ray Object Store 中）
- 通过 Ray RPC 将数据分发给各 GPU Worker

#### GPU Worker（FSDPWorker）

- 每张 GPU 一个 Worker 进程
- 每个 Worker **共置** Actor + Rollout(vLLM) + Reference 三个角色
- 同一时刻只有一个角色占用 GPU 显存（通过卸载机制切换）
- 使用 FSDP（Fully Sharded Data Parallel）将模型参数分片到所有 GPU

#### Reward Worker

- 纯 CPU 进程，不需要 GPU
- 根据预定义的规则函数给模型回答打分
- 与 Driver 异步交互

### 2.3 FSDP 与 vLLM 的 "轮换" 机制

由于 Actor(FSDP) 和 Rollout(vLLM) 共享同一组 GPU，它们需要**轮换**使用显存：

```
时间线：
─────────────────────────────────────────────────────────────────
│ vLLM 生成阶段                    │ FSDP 训练阶段              │
│                                  │                            │
│ 1. Actor FSDP 参数卸载到 CPU     │ 5. vLLM 权重卸载           │
│ 2. Actor 权重同步到 vLLM         │ 6. Actor FSDP 参数加载回 GPU│
│ 3. vLLM 初始化 KV Cache          │ 7. 前向/反向传播            │
│ 4. 生成 n 个回答                 │ 8. 优化器更新               │
│    显存：~80-100 GB              │    显存：~5-10 GB           │
─────────────────────────────────────────────────────────────────
```

关键函数对应关系：
- `prepare_rollout_engine()` → 步骤 1-3（`fsdp_workers.py:611`）
- `release_rollout_engine()` → 步骤 5（`fsdp_workers.py:615`）
- Actor 参数加载 → 在 `compute_log_probs()` / `update_actor()` 开始时自动处理

---

## 3. 一个训练 Step 的完整流程

以下以实际配置为例（16 节点 × 8 GPU = 128 GPU，rollout_batch_size=512，n=10）。

代码入口：`verl/trainer/ray_trainer.py:621` 的 `fit()` 方法。

### 3.1 Phase 1：生成回答（Rollout）

```python
# ray_trainer.py:627-630
self.actor_rollout_ref_wg.prepare_rollout_engine()   # vLLM 加载
batch = self._make_batch_data(metrics=metrics)        # 生成数据
self.actor_rollout_ref_wg.release_rollout_engine()    # vLLM 卸载
```

**详细过程：**

1. **加载 vLLM**：将 Actor FSDP 权重同步到 vLLM 引擎，初始化 KV Cache
2. **从 DataLoader 取数据**：每次取一批 prompts（由 `mini_rollout_batch_size` 控制，默认 = `rollout_batch_size`）
3. **vLLM 生成**：使用 Tensor Parallel（TP=8）在 8 卡上并行推理
   - 每个 prompt 生成 n=10 个不同回答（通过 `temperature=1.0` 采样）
   - 最大生成长度 = `max_response_length`（14768 tokens）
4. **扩展 batch**：原始 512 条 prompt 被 `repeat(n=10, interleave=True)` → 5120 条序列
5. **卸载 vLLM**：释放 KV Cache 和模型权重，回收显存

```
输入:  512 条 prompt（含问题文本 + 图片/视频的多模态数据）
输出:  5120 条完整序列（prompt + response），存在 CPU 内存中
```

### 3.2 Phase 2：均衡分配（Balance Batch）

```python
# ray_trainer.py:635
self._balance_batch(batch, metrics=metrics)
```

不同序列的有效 token 数差异很大（几百到几万），如果随机分配给各 GPU，会导致部分 GPU 等待其他 GPU——**木桶效应**。

均衡算法使用 **Karmarkar-Karp 分区**（`seqlen_balancing.py:153`），将 5120 条序列分配给 128 个 GPU，使每个 GPU 的**总 token 数尽可能相等**。

```
示例（简化，4 条序列分配到 2 个 GPU）：
  序列长度: [8000, 2000, 6000, 4000]

  随机分配:  GPU 0: [8000, 6000] = 14000 tokens
             GPU 1: [2000, 4000] = 6000 tokens  ← GPU 1 空闲等待 GPU 0

  均衡分配:  GPU 0: [8000, 2000] = 10000 tokens
             GPU 1: [6000, 4000] = 10000 tokens  ← 完美均衡
```

### 3.3 Phase 3：计算奖励（Reward，异步）

```python
# ray_trainer.py:641-643
if "token_level_scores" not in batch.batch:
    reward_ref = self.reward_fn.compute_reward.remote(batch)  # 异步！
```

奖励计算是 **CPU 密集型**任务（规则匹配、正则提取答案等），通过 Ray 的 `remote()` **异步**执行，不阻塞 GPU 训练流程。

Reward 函数根据 `problem_type` 分派到不同的评分逻辑：
- 数学题：提取答案与 ground truth 比较
- 选择题：匹配选项
- 代码题：沙箱执行验证
- 开放题：关键词/格式检查

### 3.4 Phase 4：计算 Old Log Probs（Actor 推理）

```python
# ray_trainer.py:646-648
old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
batch = batch.union(old_log_probs)
```

**为什么需要重新计算 old_log_probs？**

vLLM 生成时使用采样（随机性），不会精确记录每个 token 的 log 概率。PPO 需要精确的 log π_old(a|s) 来计算 importance sampling ratio，因此必须用 FSDP 模型重新做一次前向推理。

**执行流程：**

1. **数据分发**：5120 条序列 `chunk(128)` → 每 GPU 分到 40 条（`decorator.py:106`）
2. **动态分批**：40 条序列按 `max_token_len_per_gpu=32768` 分成若干 micro-batch（`dp_actor.py:197`）
3. **前向推理**：每个 micro-batch 做一次前向传播，计算每个 response token 的 log 概率
4. **结果汇总**：128 个 GPU 的结果 concat 回 5120 条

```
每 GPU 处理量:
  40 条序列，平均有效 token ~2000/条
  总 token = 80,000
  micro-batch 数 = ceil(80,000 / 32,768) = 3 个
  每个 micro-batch ≈ 13-14 条序列，≈ 26,000 tokens
```

### 3.5 Phase 5：计算 Reference Log Probs（参考模型推理）

```python
# ray_trainer.py:651-654
if self.use_reference_policy:
    ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
    batch = batch.union(ref_log_probs)
```

与 Phase 4 完全相同的流程，但使用**冻结的初始模型**。输出 log π_ref(a|s) 用于后续 KL 散度计算。

KL 散度的作用是**防止模型偏离太远**——如果新策略与初始策略差异过大，通过 KL 惩罚拉回来。

### 3.6 Phase 6：计算优势值（Advantage，在 Driver 上）

```python
# ray_trainer.py:662-684
# 1. 收集异步的 reward 结果
reward_tensor, reward_metrics = ray.get(reward_ref)
batch.batch["token_level_scores"] = reward_tensor

# 2. 应用 KL 惩罚（如果启用）
batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

# 3. 计算 GRPO 优势值
batch = compute_advantage(batch, adv_estimator="grpo", gamma=1.0, lam=1.0)
```

GRPO 优势值计算的核心逻辑：

```
对于同一个 prompt 的 n=10 个回答:
  rewards = [0.8, 0.0, 1.0, 0.0, 0.6, 0.0, 1.0, 0.0, 0.4, 0.0]
  mean = 0.38,  std = 0.42

  advantages = (rewards - mean) / std
             = [1.0, -0.9, 1.5, -0.9, 0.5, -0.9, 1.5, -0.9, 0.05, -0.9]

→ 正确回答的 advantage > 0（增大概率）
→ 错误回答的 advantage < 0（减小概率）
```

### 3.7 Phase 7：更新 Actor 模型（PPO 训练）

```python
# ray_trainer.py:695-700
actor_output = self.actor_rollout_ref_wg.update_actor(batch)
```

这是**唯一涉及梯度计算和参数更新**的阶段。

**执行流程（每个 GPU 上，`dp_actor.py:222-301`）：**

```
每 GPU 收到: 40 条序列 + old_log_probs + ref_log_probs + advantages

Step 1: 分 mini-batch
  global_batch_size_per_device = 256 × 10 / 128 = 20 条/mini-batch
  40 条 → 2 个 mini-batch

Step 2: 对每个 mini-batch
  ├── 动态分 micro-batch (max_token_len_per_gpu=32768)
  │     20 条 × ~2000 tokens = 40K tokens → 2 个 micro-batch
  │
  ├── 对每个 micro-batch:
  │     ├── 前向传播: 计算 new_log_probs = log π_new(a|s)
  │     ├── 计算 ratio = exp(new_log_probs - old_log_probs)
  │     ├── 计算 PPO clip loss = -min(ratio×adv, clip(ratio,0.2,1.28)×adv)
  │     ├── (可选) 计算 KL loss
  │     └── 反向传播: loss.backward()  ← 梯度累积，不更新参数
  │
  └── 优化器更新: optimizer.step()  ← 每个 mini-batch 更新一次
      └── 梯度裁剪: max_grad_norm=1.0

总计: 每 GPU 每 step 做 2 次 optimizer.step()
     全局做 2 × (128 GPUs) = 256 次同步梯度更新
```

### 3.8 Phase 8：验证和保存

```python
# ray_trainer.py:702-715
if self.global_step % self.config.trainer.val_freq == 0:
    val_metrics = self._validate()

if self.global_step % self.config.trainer.save_freq == 0:
    self._save_checkpoint()
```

- **验证**：与训练流程类似，但 n=1（只生成一个回答），使用不同的采样参数（temperature=0.7 等）
- **保存**：保存 Actor 权重、优化器状态、DataLoader 状态（用于断点续训）

---

## 4. 数据流转与内存模型

### 4.1 DataProto：数据的容器

所有数据通过 `DataProto` 容器在各组件间传递（`protocol.py:165`）：

```python
class DataProto:
    batch: TensorDict          # Tensor 数据 (input_ids, attention_mask, ...)
    non_tensor_batch: dict     # 非 Tensor 数据 (multi_modal_data, uid, ...)
    meta_info: dict            # 元信息 (temperature, image_max_pixels, ...)
```

关键操作：

| 操作 | 说明 | 使用场景 |
|------|------|----------|
| `repeat(n, interleave=True)` | [a,b] → [a,a,b,b] | rollout 后对齐 n 个回答 |
| `chunk(k)` | 均分为 k 份 | 分发给 k 个 GPU |
| `concat([a,b])` | 合并多个 DataProto | 汇总 GPU 结果 |
| `union(other)` | 合并键值对 | 添加 log_probs 等新字段 |
| `reorder(indices)` | 按索引重排 | 均衡 batch 分配 |
| `select(keys)` | 选取部分键 | 只传必要数据给 Worker |

### 4.2 一条序列在内存中占多少空间

以一条 `max_prompt_length=18000, max_response_length=14768` 的序列为例：

```
Tensor 数据（固定，每条序列）:
  input_ids:      32768 × 8 bytes (int64) = 256 KB
  attention_mask:  32768 × 8 bytes        = 256 KB
  position_ids:   4 × 32768 × 8 bytes     = 1 MB    (Qwen3-VL 有 4 维位置编码)
  responses:      14768 × 8 bytes         = 115 KB
  response_mask:  14768 × 8 bytes         = 115 KB
  ──────────────────────────────────────
  合计: ≈ 1.74 MB / 条

5120 条 Tensor 总计: ≈ 8.7 GB

非 Tensor 数据（变长，视数据类型而定）:
  图片样本:  multi_modal_data = {"images": [PIL.Image]}        ≈ 1-10 MB / 条
  视频样本:  multi_modal_data = {"video": [Tensor(128,3,H,W)]} ≈ 200-400 MB / 条
  纯文本:    multi_modal_data = {}                              ≈ 0 MB / 条
```

### 4.3 内存在各位置的分布

```
┌─────────────────────────────────────────────────────────────────────┐
│                         内存分布示意图                               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  CPU 内存 / Ray Object Store (Head 节点)                     │   │
│  │                                                              │   │
│  │  完整 batch (5120 条):                                       │   │
│  │    ├── Tensor 数据: ~8.7 GB                                  │   │
│  │    └── multi_modal_data: ~40-80 GB (视频帧占大头)            │   │
│  │                                                              │   │
│  │  ★ 当 batch 被 dispatch 给 128 个 GPU 时,                   │   │
│  │    Ray 需要序列化每个 chunk → 额外 ~80-150 GB 在 Object Store │   │
│  │  ★ 如果 reward 异步计算与 dispatch 重叠 → 峰值 ~200+ GB      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  GPU 显存 (每张卡)                                           │   │
│  │                                                              │   │
│  │  Phase: Rollout (vLLM)                                       │   │
│  │    ├── 模型权重 (TP=8 分片): ~2 GB                           │   │
│  │    ├── KV Cache (gpu_memory_utilization=0.6): ~80 GB         │   │
│  │    └── 总计: ~80-100 GB                                      │   │
│  │                                                              │   │
│  │  Phase: compute_log_probs / compute_ref_log_probs (推理)     │   │
│  │    ├── FSDP 参数分片 (8B/128卡): ~125 MB                    │   │
│  │    ├── All-gather 临时权重 (当前层): ~571 MB                 │   │
│  │    ├── Batch 数据 (40条): ~640 MB                            │   │
│  │    ├── Activation (~26K tokens): ~2-3 GB                     │   │
│  │    └── 总计: ~3-5 GB                                         │   │
│  │                                                              │   │
│  │  Phase: update_actor (训练)                                   │   │
│  │    ├── FSDP 参数分片: ~125 MB                                │   │
│  │    ├── All-gather 临时权重: ~571 MB                          │   │
│  │    ├── 前向 Activation (gradient checkpointing): ~1-2 GB     │   │
│  │    ├── 反向梯度: ~2 GB                                       │   │
│  │    ├── 优化器状态 (offload=true → CPU): ~0 GB                │   │
│  │    └── 总计: ~4-6 GB                                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.4 关键结论

| 问题 | 答案 |
|------|------|
| GPU 显存与 `rollout_batch_size` 有关吗？ | **无关**。每 GPU 的显存由 `max_token_len_per_gpu` 和 `micro_batch_size` 决定 |
| GPU 显存瓶颈在哪？ | **vLLM 生成阶段**是显存最高的（~80-100 GB），训练阶段仅 ~5 GB |
| CPU 内存与 `rollout_batch_size` 有关吗？ | **有关**。batch_size × n 决定了 Ray Object Store 中的数据量 |
| CPU 内存瓶颈在哪？ | **视频帧数据**（~200-400 MB/条）通过 Ray 序列化 dispatch 时产生大量拷贝 |

---

## 5. 分布式训练细节

### 5.1 FSDP（Fully Sharded Data Parallel）

FSDP 是 PyTorch 的分布式训练策略，相当于 DeepSpeed ZeRO-3。核心思想：

```
传统 DDP:  每张卡存一份完整模型 → 8B 模型 × 128 卡 = 需要 128 × 16 GB = 2 TB 显存

FSDP:     模型参数分片到所有卡 → 8B 模型 / 128 卡 = 每卡仅 125 MB
          需要某层参数时 → all-gather 到所有卡 → 计算完 → 释放
```

FSDP 的**时间线**（单层前向传播）：

```
1. All-gather:  128 个 GPU 各发送自己的参数分片 → 每个 GPU 获得完整层参数
2. Forward:     用完整层参数计算前向输出
3. Release:     丢弃 all-gather 来的参数（只保留自己的分片）
4. 重复 1-3 到下一层
```

### 5.2 动态 Micro-Batching

传统的 micro-batching 按**样本数**切分：每个 micro-batch 固定 k 条序列。但不同序列长度差异巨大，会导致：

```
固定切分 (micro_batch_size=4):
  Micro-batch 1: [100, 100, 100, 100]   总 400 tokens   → GPU 空闲
  Micro-batch 2: [20000, 20000, 5000, 5000] 总 50000 tokens → GPU 爆显存
```

本框架的动态 micro-batching（`seqlen_balancing.py:240`）按**总 token 数**切分：

```
动态切分 (max_token_len_per_gpu=32768):
  Micro-batch 1: [20000, 5000, 5000, 100, 100, ...] 总 ≤ 32768 tokens
  Micro-batch 2: [20000, 100, 100, 100, ...]          总 ≤ 32768 tokens

  → 每个 micro-batch 的 GPU 负载基本相同
```

当 `padding_free=true` 时，配合 Flash Attention 的 `unpad_input()` 函数，实际计算只处理有效 token，无需为 padding 浪费算力。

### 5.3 Tensor Parallel (TP) — 仅用于 vLLM 生成

生成阶段使用 vLLM 的 Tensor Parallel（`tensor_parallel_size=8`），将模型按**矩阵列**切分到 8 张 GPU 上。

```
TP=8 的推理:
  Node 0 的 8 张卡各持有模型的 1/8
  → 每卡只需 16 GB / 8 = 2 GB 模型权重
  → KV Cache 充分利用剩余显存
```

训练阶段**不使用 TP**，而是使用 FSDP 把模型分片到所有 128 张 GPU。

### 5.4 数据并行 (DP) — 用于训练各阶段

训练阶段（compute_log_probs、update_actor 等）采用 Data Parallel：

```
5120 条序列 → chunk(128) → 每 GPU 40 条 → 各自独立处理 → concat 结果
```

各 GPU 之间通过 NCCL 进行 AllReduce 同步梯度（在 FSDP 框架内自动完成）。

---

## 6. 关键配置参数详解

### 6.1 数据相关

```yaml
data:
  rollout_batch_size: 512       # 每 step 多少条 prompt 做 rollout
  # ↑ 决定了每 step 的总数据量 = rollout_batch_size × n
  # ↑ 直接影响 CPU 内存（Ray Object Store）
  # ↑ 不影响 GPU 显存（因为 micro-batch 是分批处理的）

  max_prompt_length: 18000      # prompt 侧的 padding 长度
  max_response_length: 14768    # response 侧的 padding 长度
  # ↑ 两者之和 = 每条序列的最大总长度（32768）
  # ↑ 这是 vLLM 的 max_model_len，也是 tensor 的第二维大小
  # ↑ 增大会增加 CPU 内存和 vLLM KV Cache 消耗

  filter_overlong_prompts: true # 预处理时过滤超过 max_prompt_length 的样本
```

### 6.2 Rollout（生成）相关

```yaml
worker:
  rollout:
    n: 10                       # 每个 prompt 生成多少个不同回答
    # ↑ GRPO 的核心参数。n 越大，优势估计越准，但内存 ×n
    # ↑ 建议范围: 5-16

    temperature: 1.0            # 采样温度。1.0 = 标准采样，>1 更随机
    top_p: 1.0                  # nucleus 采样。1.0 = 不裁剪

    tensor_parallel_size: 8     # vLLM 推理的 TP 并行度
    # ↑ 通常等于每节点的 GPU 数

    gpu_memory_utilization: 0.6 # vLLM 占用 GPU 显存的比例
    # ↑ 剩余的 40% 留给 FSDP 模型权重和其他开销
    # ↑ 值太大会导致 FSDP 加载时 OOM

    max_num_batched_tokens: 32768  # vLLM 单次 prefill 的最大 token 数
    enable_chunked_prefill: true   # 分块 prefill，对长序列更友好
```

### 6.3 Actor（训练）相关

```yaml
worker:
  actor:
    global_batch_size: 256              # 全局 batch 大小（prompt 数）
    # ↑ 实际 mini-batch = global_batch_size × n / world_size
    #   = 256 × 10 / 128 = 20 条 / GPU

    micro_batch_size_per_device_for_update: 1    # 训练时每卡 micro-batch 大小
    micro_batch_size_per_device_for_experience: 1 # 推理时每卡 micro-batch 大小
    # ↑ 当 dynamic_batching=true 时，这两个参数作为 fallback
    #   实际由 max_token_len_per_gpu 控制

    padding_free: true                  # 消除 padding 的无效计算
    dynamic_batching: true              # 按 token 数动态分批
    max_token_len_per_gpu: 32768        # 每个 micro-batch 的最大 token 总数
    # ↑ 这是控制 GPU 显存的核心参数
    # ↑ 减小此值 → 每个 micro-batch 更小 → 显存更低（但 micro-batch 数更多）

    max_grad_norm: 1.0                  # 梯度裁剪阈值

    model:
      enable_gradient_checkpointing: true  # 梯度检查点，节省显存
      freeze_vision_tower: false           # 是否冻结视觉编码器

    fsdp:
      enable_full_shard: true     # ZeRO-3 全分片
      enable_cpu_offload: false   # FSDP 的 CPU 卸载（慢但省显存）

    offload:
      offload_params: true        # Actor 不用时参数卸载到 CPU
      offload_optimizer: true     # Actor 不用时优化器卸载到 CPU
      # ↑ 这两个是 verl 自定义的卸载机制（与 FSDP 的 cpu_offload 不同）
      # ↑ 用于在 vLLM 生成阶段释放 GPU 显存给 vLLM
```

### 6.4 算法相关

```yaml
algorithm:
  adv_estimator: grpo            # 优势估计方式: grpo | gae | reinforce++
  # ↑ grpo: 不需要 Critic，组内相对比较
  # ↑ gae: 需要 Critic 模型，传统 PPO

  disable_kl: true               # 不计算 KL 散度（节省 Reference 模型推理）
  use_kl_loss: true              # 在 loss 中加入 KL 惩罚项
  kl_coef: 0.01                  # KL 惩罚系数
  # ↑ disable_kl=true 只是跳过 KL 在 reward 中的惩罚
  # ↑ use_kl_loss=true 仍然在训练 loss 中加入 KL 正则项
```

### 6.5 Trainer 相关

```yaml
trainer:
  total_epochs: 10              # 训练轮数
  # ↑ training_steps 的计算方式：
  #   如果 max_steps 非 null → training_steps = max_steps
  #   如果 mini_rollout_batch_size 非 null →
  #     training_steps = total_epochs × len(dataloader) × (mini / rollout_batch_size)
  #   否则 → training_steps = total_epochs × len(dataloader)

  val_freq: 10                  # 每 N 步验证一次
  val_before_train: true        # 训练前先做一次验证（baseline）
  save_freq: 20                 # 每 N 步保存 checkpoint
  save_limit: 10                # 最多保留 N 个 checkpoint
  find_last_checkpoint: true    # 自动寻找最新 checkpoint 恢复训练
```

---

## 7. 常见问题与调优指南

### 7.1 GPU 显存不足 (OOM)

**症状：** CUDA Out of Memory，通常在 vLLM 生成阶段

**原因与解决：**

| 参数 | 调整方向 | 说明 |
|------|----------|------|
| `gpu_memory_utilization` | 降低（0.6→0.5） | 减少 vLLM KV Cache，但可能降低吞吐 |
| `max_num_batched_tokens` | 降低（32768→16384） | 减少 vLLM 单次处理的 token 数 |
| `max_token_len_per_gpu` | 降低（32768→16384） | 减少训练阶段每 micro-batch 的显存 |
| `max_response_length` | 降低 | 减少生成长度，全面降低显存 |
| `offload_params` | 开启 | Actor 不用时卸载参数 |

### 7.2 CPU 内存不足 / Ray Object Store 溢出

**症状：** 日志中出现 `Spilled XXX MiB` 或 `Spilled XXX GiB`，之后 NCCL Timeout

**原因：** 视频帧数据（每条 ~200-400 MB）在 Ray Object Store 中被多次拷贝

**解决：**

| 参数 | 调整方向 | 说明 |
|------|----------|------|
| `rollout_batch_size` | 降低（512→256） | 减少每 step 的总数据量 |
| `n` | 降低（10→5） | 减少重复数据量 |
| `video_max_frames` | 降低（128→64） | 每条视频的帧数据减半 |
| `video_max_pixels` | 降低（262144→131072） | 每帧分辨率降低 |

**深层原因：**

当 `compute_log_probs(batch)` 被调用时，Driver 将 batch `chunk(128)` 成 128 份并通过 Ray 发送给各 GPU。Ray 序列化每一份时会**独立拷贝** `multi_modal_data` 中的视频帧 tensor，导致内存中存在多份拷贝。同时如果 `reward` 异步计算尚未完成，Ray Object Store 中同时存在 reward 版本和 dispatch 版本的数据，内存压力翻倍。

### 7.3 NCCL Timeout

**症状：** `Watchdog caught collective operation timeout: WorkNCCL ... ran for 600000 milliseconds`

**原因：** 某个 GPU Worker 因内存不足（CPU 或 GPU）崩溃或卡死，无法参与 AllReduce 通信

**解决：**
1. 优先排查上述 GPU OOM 或 CPU 内存问题
2. 增大超时：`export NCCL_TIMEOUT=3600000`（30 min → 60 min）
3. 检查网络：`export NCCL_DEBUG=INFO` 查看通信日志

### 7.4 训练 loss 不下降

**可能原因与检查项：**

1. **Reward 函数有 bug** → 先用 `val_only: true` 模式检查 reward 输出
2. **学习率过大/过小** → GRPO 通常使用 1e-6 到 5e-6
3. **KL 惩罚过强** → 减小 `kl_coef` 或设置 `use_kl_loss: false`
4. **n 太小** → 优势估计噪声大，增大 n（如 10→16）
5. **clip_ratio 不合适** → 标准值 `clip_ratio_low=0.2, clip_ratio_high=0.28`
6. **视觉编码器冻结** → `freeze_vision_tower: true` 会限制多模态能力

### 7.5 断点续训

训练中断后，只要满足以下条件即可自动恢复：

1. `find_last_checkpoint: true`
2. `save_checkpoint_path` 指向之前的保存目录

恢复时加载：
- Actor 模型权重 + 优化器状态
- DataLoader 状态（从中断的样本继续）
- global_step（从 checkpoint 目录名中解析）

**可以安全修改的参数：** 学习率、KL 系数、clip_ratio、验证频率、保存频率

**不建议修改的参数：** rollout_batch_size（会改变 training_steps 计算）、global_batch_size（影响优化器步骤数）

### 7.6 训练速度优化 Checklist

- [ ] `padding_free: true` + `dynamic_batching: true` — 消除 padding 浪费
- [ ] `enable_gradient_checkpointing: true` — 用计算换显存
- [ ] `enable_chunked_prefill: true` — vLLM 分块 prefill
- [ ] `offload_params: true` + `offload_optimizer: true` — Actor 不用时卸载
- [ ] `enforce_eager: false` — 允许 CUDA Graph 加速
- [ ] `use_torch_compile: true` — Torch 编译加速

---

## 附录：关键代码文件索引

| 文件 | 职责 |
|------|------|
| `verl/trainer/main.py` | 入口：加载配置，启动 Ray，创建 Runner |
| `verl/trainer/config.py` | 所有配置的 dataclass 定义 |
| `verl/trainer/ray_trainer.py` | 核心：RayPPOTrainer 主训练循环 |
| `verl/workers/fsdp_workers.py` | GPU Worker：模型初始化、权重同步、参数卸载 |
| `verl/workers/actor/dp_actor.py` | Actor：log_probs 计算、PPO 更新、micro-batching |
| `verl/workers/rollout/vllm_rollout_spmd.py` | Rollout：vLLM 生成、多模态处理、n>1 采样 |
| `verl/utils/dataset.py` | 数据集：加载、预处理、multi_modal_data 构建 |
| `verl/utils/seqlen_balancing.py` | 动态分批：Karmarkar-Karp 分区算法 |
| `verl/protocol.py` | DataProto：数据容器，repeat/chunk/concat 等操作 |
| `verl/single_controller/base/decorator.py` | 分发逻辑：DP_COMPUTE_PROTO dispatch |
| `verl/single_controller/ray/base.py` | Ray Worker Group：execute_all 分发执行 |
