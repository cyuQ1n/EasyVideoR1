# EasyVideoR1 Qwen3.5 适配说明

> 日期：2026-03-13
> 目标：在 EasyVideoR1 框架中支持 Qwen3.5 系列（model_type: `qwen3_5`）的视频 RL 训练
> 验证模型：Qwen3.5-2B（`/pfs/qcy/models/Qwen3.5-2B`）
> 运行环境：`/pfs/siqingyi/miniconda3/envs/rl-for-qwen3.5`（Python 3.12, vLLM 0.17.0, transformers 5.3.0, torch 2.10.0+cu128）

---

## 一、新增文件

### 1. `verl/models/transformers/qwen3_5.py`（核心适配文件）

参照 `qwen3_vl.py` 结构，为 Qwen3.5 实现自定义 forward 和位置编码：

| 函数 | 用途 |
|------|------|
| `get_rope_index()` | 基于 `mm_token_type_ids`（0=text, 1=image, 2=video）计算 interleaved MRoPE 位置编码，返回 `(3, seq_length)` 的 position_ids |
| `_get_input_embeds()` | 通过 `model.visual` 计算图像/视频 embedding，用 `masked_scatter` 注入文本 embedding；包含 dummy gradient flow 保持 vision encoder 梯度 |
| `qwen3_5_base_forward()` | 覆盖 `Qwen3_5Model.forward`，调用 `_get_input_embeds` 后传入 `language_model` |
| `qwen3_5_model_forward()` | 覆盖 `Qwen3_5ForConditionalGeneration.forward`，调用 `self.model` + `self.lm_head` 返回 logits |

### 2. `examples/video_rl_v2/video_rl_v1_qwen3_5.yaml`（训练配置）

基于 `video_rl_v1_vanilla.yaml` 调整：
- `model_path: /pfs/qcy/models/Qwen3.5-2B`
- `micro_batch_size_per_device_for_update: 1`（减小显存占用）
- `micro_batch_size_per_device_for_experience: 1`
- `max_token_len_per_gpu: 32000`
- `gpu_memory_utilization: 0.2`（为 FSDP 训练腾出显存）
- `project_name: video_rl_qwen3_5`

---

## 二、修改的已有文件

### 3. `verl/models/monkey_patch.py` — 注册 Qwen3.5 模型支持

- `SUPPORTED_MODEL_TYPE` 元组添加 `"qwen3_5"`
- 新增 `QWEN3_5_MODELS = ("qwen3_5",)`
- `apply_ulysses_patch()` 添加 `elif model_type in QWEN3_5_MODELS:` 分支，import 并 patch `Qwen3_5Model.forward` 和 `Qwen3_5ForConditionalGeneration.forward`

### 4. `verl/utils/dataset.py` — 按 model_type 路由 position_ids 计算

- `RLHFDataset.__init__` 新增 `model_type: Optional[str] = None` 参数
- position_ids 计算处（原 ~L526）添加 Qwen3.5 分支：
  ```python
  if self.model_type == "qwen3_5":
      from ..models.transformers.qwen3_5 import get_rope_index
  elif "Qwen3VLProcessor" in ...:
      from ..models.transformers.qwen3_vl import get_rope_index
  ```
- **原因**：Qwen3.5 和 Qwen3-VL 共享 `Qwen3VLProcessor`，无法靠 processor 类名区分

### 5. `verl/trainer/data_loader.py` — 自动检测 model_type

- `create_dataloader()` 新增 `model_path` 参数
- 通过 `AutoConfig.from_pretrained(model_path)` 自动获取 `model_type`
- 将 `model_type` 传递给 `RLHFDataset` 构造函数

### 6. `verl/trainer/main.py` — 传递 model_path

- `create_dataloader` 调用处增加 `model_path=config.worker.actor.model.model_path`

### 7. `verl/utils/flops_counter.py` — 添加 FLOPS 估算

- `_ESTIMATE_FUNC` 字典添加 `"qwen3_5": self._estimate_llama_flops`

---

## 三、vLLM 0.17.0 + transformers 5.3.0 兼容性修复

这些修改不仅对 Qwen3.5 必要，也影响 Qwen3-VL 在新环境下的运行。

### 8. `verl/workers/fsdp_workers.py` — `no_init_weights` 导入路径

transformers 5.x 将 `no_init_weights` 从 `modeling_utils` 移到 `initialization`：

```python
# 修复前
from transformers.modeling_utils import no_init_weights

# 修复后
try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    from transformers.initialization import no_init_weights
```

### 9. `verl/workers/rollout/vllm_rollout_spmd.py` — 两处 vLLM API 变更

**9a. 移除 `disable_mm_preprocessor_cache`**

vLLM 0.17.0 移除了此参数（默认行为等价），直接删除该行。

**9b. `SamplingParams` 变为 `msgspec.Struct`（只读属性）**

vLLM 0.17.0 的 `SamplingParams` 不再支持 `setattr`。`update_sampling_params` 方法重写为：
- 检测 `__struct_fields__`（msgspec.Struct 标志）
- 如果是 Struct：读取当前字段值，合并 overrides，重建 `SamplingParams` 对象
- 如果不是：走旧的 `setattr` 路径（兼容旧版 vLLM）
- 回滚时直接恢复整个对象引用

### 10. `verl/workers/sharding_manager/fsdp_vllm.py` — TP group API 重命名

vLLM 0.17.0 将 `get_tensor_model_parallel_group()` 改为 `get_tp_group()`：

```python
try:
    self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group
except AttributeError:
    self.tp_group = vllm_ps.get_tp_group().device_group
```

### 11. `verl/models/transformers/flash_attention_utils.py` — 支持 3D position_ids

Qwen3.5 的 position_ids 是 `(3, batch_size, seq_length)` 而非 `(batch_size, seq_length)`：
- `prepare_fa2_from_position_ids()` 添加 `if position_ids.ndim == 3: position_ids = position_ids[0]`
- `_custom_flash_attention_forward()` 中的 `torch.diff` 检查也同样处理 3D 情况

---

## 四、数据文件修复

### 12. 训练/评测数据 JSON 字段补全

部分样本缺少 `has_offline_trajectory` 等字段，导致 `datasets` 库推断 schema 时报 `KeyError`：

| 文件 | 修复行数 | 补全字段 |
|------|---------|----------|
| `video_train_data_v1_preprocessed_sampled_1000.json` | 1000/1000 | `has_offline_trajectory=false`, `offline_output=""`, `preprocessed_video=null` |
| `eval_1000_samples_preprocessed_sampled_100.json` | 100/100 | 同上 |

---

## 五、启动脚本修改

### 13. `examples/video_rl_v2/video_rl_v1_dapo.sh`

- `PROJECT_DIR` 改为随脚本位置自动推导，不再绑定固定机器目录
- `CONFIG_PATH` 默认指向仓库内的 `video_rl_v1_qwen3_5.yaml`，同时允许环境变量覆盖
- `MODEL_PATH`、`LOG_DIR`、`SAVE_CHECKPOINT_PATH`、`REWARD_FUNCTION` 改为可配置的相对/环境变量路径
- `WANDB_API_KEY` 改为从环境变量读取，不再硬编码

---

## 六、修改文件汇总

| 文件 | 类型 | 改动说明 |
|------|------|----------|
| `verl/models/transformers/qwen3_5.py` | **新增** | Qwen3.5 模型适配（forward + position_ids） |
| `examples/video_rl_v2/video_rl_v1_qwen3_5.yaml` | **新增** | Qwen3.5-2B 训练配置 |
| `verl/models/monkey_patch.py` | 修改 | 注册 qwen3_5 模型类型 |
| `verl/utils/dataset.py` | 修改 | model_type 参数 + position_ids 路由 |
| `verl/trainer/data_loader.py` | 修改 | 自动检测 model_type |
| `verl/trainer/main.py` | 修改 | 传递 model_path |
| `verl/utils/flops_counter.py` | 修改 | 添加 qwen3_5 FLOPS 估算 |
| `verl/workers/fsdp_workers.py` | 修改 | transformers 5.x 兼容 |
| `verl/workers/rollout/vllm_rollout_spmd.py` | 修改 | vLLM 0.17.0 兼容（2 处） |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 修改 | vLLM 0.17.0 TP group API |
| `verl/models/transformers/flash_attention_utils.py` | 修改 | 支持 3D position_ids |
| `examples/video_rl_v2/video_rl_v1_dapo.sh` | 修改 | 切换到 Qwen3.5 配置和模型 |
