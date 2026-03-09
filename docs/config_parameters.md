# EasyVideoR1 配置参数说明（基于源码）

本文档基于 EasyVideoR1 源码中的 dataclass 定义，总结各类配置参数，以及这些参数在代码中的使用与串联方式。

源码位置：
- `verl/trainer/config.py`
- `verl/workers/config.py`
- `verl/workers/actor/config.py`
- `verl/workers/critic/config.py`
- `verl/workers/rollout/config.py`
- `verl/workers/reward/config.py`
- `verl/trainer/main.py`

## 配置加载与覆盖逻辑

入口脚本：`python3 -m verl.trainer.main`

- 默认配置：`OmegaConf.structured(PPOConfig())`
- 如通过命令行传入 `config=path`，则会将该 YAML 文件与默认配置合并
- 命令行参数具有最高优先级，会在 YAML 之后再次覆盖
- `PPOConfig.deep_post_init()` 会递归调用所有嵌套 dataclass 的 `post_init()`，完成一些依赖关系和路径等的二次处理

## 顶层配置：PPOConfig

定义位置：`verl/trainer/config.py`。

顶层字段：
- `data: DataConfig`
- `worker: WorkerConfig`
- `algorithm: AlgorithmConfig`
- `trainer: TrainerConfig`

`post_init` 中的自动赋值（跨子配置的“连线”）：
- `worker.rollout.prompt_length = data.max_prompt_length`
- `worker.rollout.response_length = data.max_response_length`
- `worker.rollout.trust_remote_code = worker.actor.model.trust_remote_code`
- `worker.actor.disable_kl = algorithm.disable_kl`
- `worker.actor.use_kl_loss = algorithm.use_kl_loss`
- `worker.actor.kl_penalty = algorithm.kl_penalty`
- `worker.actor.kl_coef = algorithm.kl_coef`

## DataConfig

定义位置：`verl/trainer/config.py`。

字段：
- `train_files: str = ""` 训练数据文件路径（支持通配或多个文件拼接，具体参考数据加载脚本）
- `val_files: str = ""` 验证数据文件路径
- `prompt_key: str = "prompt"` 数据中问题/提示字段名
- `answer_key: str = "answer"` 数据中答案字段名
- `image_key: str = "images"` 图像字段名
- `video_key: str = "videos"` 视频字段名
- `image_dir: Optional[str] = None` 图像根目录（如有单独存放）
- `video_fps: float = 2.0` 视频采样帧率（frames per second）
- `max_prompt_length: int = 512` 文本 prompt 最大长度（token 数）
- `max_response_length: int = 512` 模型生成响应的最大长度（token 数）
- `rollout_batch_size: int = 512` 采样/rollout 阶段的批大小
- `mini_rollout_batch_size: Optional[int] = None` 如设置，则用于 dataloader 的实际 batch size，优先于 `rollout_batch_size`
- `val_batch_size: int = -1` 验证批大小，-1 表示自动推断或不启用
- `format_prompt: Optional[str] = None` Prompt 模板（例如 chat 格式封装）
- `override_chat_template: Optional[str] = None` 用于覆盖 tokenizer 自带的 chat template
- `shuffle: bool = True` 是否打乱数据
- `seed: int = 1` 随机种子
- `min_pixels: Optional[int] = 262144` 图像/视频最小像素数过滤阈值
- `max_pixels: Optional[int] = 4194304` 图像/视频最大像素数过滤阈值
- `filter_overlong_prompts: bool = True` 是否过滤超长的 prompt
- `filter_overlong_prompts_workers: int = 16` 过滤超长 prompt 时使用的并发 worker 数

`post_init` 行为：
- 如设置了 `image_dir`, `format_prompt`, `override_chat_template`，会将其解析为绝对路径

使用要点：
- `data.max_prompt_length` 与 `data.max_response_length` 会传递到 `rollout`，作为采样时的长度限制
- 如设置 `mini_rollout_batch_size`，则 dataloader 实际使用该值而不是 `rollout_batch_size`

## AlgorithmConfig

定义位置：`verl/trainer/config.py`。

字段：
- `gamma: float = 1.0` 折扣因子
- `lam: float = 1.0` GAE 的 lambda 参数
- `adv_estimator: str = "grpo"`
  - 支持：`gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`
- `disable_kl: bool = False` 是否完全关闭 KL 正则
- `use_kl_loss: bool = False` 是否显式将 KL 作为 loss 项加入
- `kl_penalty: str = "kl"`
  - 支持：`kl`, `abs`, `mse`, `low_var_kl`, `full`
- `kl_coef: float = 1e-3` KL 系数（权重）
- `kl_type: str = "fixed"`
  - 支持：`fixed`, `adaptive`（自适应 KL）
- `kl_horizon: float = 10000.0` 自适应 KL 时使用的 horizon 参数
- `kl_target: float = 0.1` 自适应 KL 的目标值
- `online_filtering: bool = False` 是否在线过滤样本（基于 reward 分布）
- `filter_key: str = "overall"` 过滤使用的 reward 字段
- `filter_low: float = 0.01` 分位点下界
- `filter_high: float = 0.99` 分位点上界

运行时的校验逻辑：
- `adv_estimator` 必须属于支持的集合
- 当选择 `grpo` 或 `rloo` 时，要求 `worker.rollout.n > 1`

## TrainerConfig

定义位置：`verl/trainer/config.py`。

字段：
- `total_epochs: int = 15` 训练轮数（如未设置 `max_steps`，以 epoch 为主）
- `max_steps: Optional[int] = None` 如设置，则以总 step 数为主，覆盖 `total_epochs`
- `project_name: str = "easy_r1"` 工程名（用于日志与 checkpoint 路径）
- `experiment_name: str = "demo"` 实验名
- `logger: Tuple[str] = ("console", "wandb")`
  - 支持：`console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`
- `nnodes: int = 1` 节点数（分布式训练）
- `n_gpus_per_node: int = 8` 每个节点的 GPU 数量
- `max_try_make_batch: int = 20` 构造一个有效 batch 的最大尝试次数
- `critic_warmup: int = 0` critic 预热 epoch 数
- `val_freq: int = -1` 验证频率（以 epoch/step 计，实际语义参见 trainer 实现），-1 代表不定期或关闭
- `val_before_train: bool = True` 在正式训练前先跑一次验证
- `val_only: bool = False` 仅运行验证，不进行训练
- `val_generations_to_log: int = 0` 验证阶段要额外 log 的生成样本数量
- `save_freq: int = -1` 保存 checkpoint 的频率
- `save_limit: int = -1` 最多保留的 checkpoint 数
- `save_model_only: bool = False` 仅保存模型权重，不保存优化器等
- `save_checkpoint_path: Optional[str] = None` checkpoint 保存路径
- `load_checkpoint_path: Optional[str] = None` 恢复训练时的加载路径
- `ray_timeline: Optional[str] = None` Ray timeline 文件输出路径
- `find_last_checkpoint: bool = True` 如未显式指定 `load_checkpoint_path`，是否自动查找最新 checkpoint

`post_init` 行为：
- 若未设置 `save_checkpoint_path`，则默认使用 `checkpoints/{project_name}/{experiment_name}`
- 如设置了 `load_checkpoint_path`，会转成绝对路径

## WorkerConfig

定义位置：`verl/workers/config.py`。

字段：
- `hybrid_engine: bool = True` 是否启用混合执行引擎
- `actor: ActorConfig` actor 模型配置
- `critic: CriticConfig` critic 模型配置
- `ref: RefConfig` reference 模型配置
- `reward: RewardConfig` 奖励模型/函数配置
- `rollout: RolloutConfig` 采样/推理引擎配置

`post_init` 行为：
- `ref.*` 会从 `actor` 中继承部分字段：
  - `micro_batch_size_per_device_for_experience`
  - `padding_free`
  - `dynamic_batching`
  - `ulysses_size`
  - `use_torch_compile`

## ActorConfig

定义位置：`verl/workers/actor/config.py`。

字段：
- `strategy: str = "fsdp"` 分布式策略
- `global_batch_size: int = 256` 全局 batch 大小（所有设备之和）
- `micro_batch_size_per_device_for_update: int = 4` 每卡用于参数更新的 micro-batch 大小
- `micro_batch_size_per_device_for_experience: int = 16` 每卡用于经验采样的 micro-batch 大小
- `max_grad_norm: float = 1.0` 梯度裁剪上限
- `clip_ratio_low: float = 0.2` PPO 下界裁剪比率
- `clip_ratio_high: float = 0.3` PPO 上界裁剪比率
- `clip_ratio_dual: float = 3.0` dual-clip 相关参数
- `loss_avg_mode: str = "token"` loss 归一化方式（`"token"` 或 `"seq"`）
- `loss_type: str = "default"` loss 类型（`"default"`, `"gspo"`, `"cispo"` 等）
- `ppo_epochs: int = 1` 每个 batch 上 PPO 更新轮数
- `padding_free: bool = True` 是否使用 padding-free 技术
- `dynamic_batching: bool = True` 是否启用动态 batch
- `ulysses_size: int = 1` Ulysses 并行大小
- `use_torch_compile: bool = True` 是否使用 `torch.compile`
- `model: ModelConfig` 模型相关配置
- `optim: OptimConfig` 优化器相关配置
- `fsdp: FSDPConfig` FSDP 细节配置
- `offload: OffloadConfig` CPU/offload 配置

运行时自动赋值字段：
- `global_batch_size_per_device` 根据 `global_batch_size` 和设备数自动计算
- `disable_kl` 来自 `AlgorithmConfig.disable_kl`
- `use_kl_loss` 来自 `AlgorithmConfig.use_kl_loss`
- `kl_penalty` 来自 `AlgorithmConfig.kl_penalty`
- `kl_coef` 来自 `AlgorithmConfig.kl_coef`

### ModelConfig

- `model_path: Optional[str] = None` 模型权重路径
- `tokenizer_path: Optional[str] = None` tokenizer 路径
- `override_config: dict[str, Any] = {}` 用于覆盖模型 config 的字典
- `enable_gradient_checkpointing: bool = True` 是否启用梯度 checkpoint
- `trust_remote_code: bool = True` 是否信任远程代码（用于自定义模型实现）
- `freeze_vision_tower: bool = False` 是否冻结视觉部分参数

`post_init` 行为：
- 如未设置 `tokenizer_path`，则默认与 `model_path` 相同
- 如 `model_path` 与 `tokenizer_path` 在磁盘上存在，会被转换为绝对路径

### OptimConfig

- `lr: float = 1e-6` 学习率
- `betas: tuple[float, float] = (0.9, 0.999)` AdamW 的 betas
- `weight_decay: float = 1e-2` 权重衰减
- `strategy: str = "adamw"` 优化器类型
- `lr_warmup_ratio: float = 0.0` warmup 比例
- `lr_warmup_steps: Optional[int] = None` warmup 步数（如设置则优先生效）
- `min_lr_ratio: Optional[float] = None` 学习率最小缩放比
- `lr_scheduler_type: str = "constant"` 学习率调度器类型
- `training_steps: int` 训练总步数（由 trainer 在运行时自动写入）

### FSDPConfig

- `enable_full_shard: bool = True` 是否启用 full-shard
- `enable_cpu_offload: bool = False` 是否将参数/梯度 offload 到 CPU
- `enable_rank0_init: bool = True` 是否在 rank0 上初始化
- `use_orig_params: bool = False` 是否使用 `use_orig_params` 模式
- `torch_dtype: Optional[str] = None` 显式设置 dtype
- `fsdp_size: int = -1` FSDP 组大小（-1 表示自动）
- `mp_param_dtype: str = "bf16"` 参数混合精度 dtype
- `mp_reduce_dtype: str = "fp32"` reduce 时的 dtype
- `mp_buffer_dtype: str = "fp32"` buffer 的 dtype

### OffloadConfig

- `offload_params: bool = False` 是否 offload 参数
- `offload_optimizer: bool = False` 是否 offload 优化器状态

## RefConfig

定义位置：`verl/workers/actor/config.py`。

字段：
- `strategy: str = "fsdp"`
- `fsdp: FSDPConfig`
- `offload: OffloadConfig`

自动继承字段（通过 `WorkerConfig.post_init()` 从 actor 继承）：
- `micro_batch_size_per_device_for_experience`
- `padding_free`
- `dynamic_batching`
- `ulysses_size`
- `use_torch_compile`

## CriticConfig

定义位置：`verl/workers/critic/config.py`。

字段：
- `strategy: str = "fsdp"` 分布式策略
- `global_batch_size: int = 256` 全局 batch 大小
- `micro_batch_size_per_device_for_update: int = 4` 每卡更新用 micro-batch 大小
- `micro_batch_size_per_device_for_experience: int = 16` 每卡经验采样用 micro-batch 大小
- `max_grad_norm: float = 1.0` 梯度剪裁上限
- `cliprange_value: float = 0.5` value loss 的裁剪范围
- `loss_avg_mode: str = "token"` loss 归一化方式（`"token"` 或 `"seq"`）
- `ppo_epochs: int = 1` 每个 batch 上 PPO 更新轮数
- `padding_free: bool = False` 是否使用 padding-free
- `dynamic_batching: bool = True` 是否启用动态 batch
- `ulysses_size: int = 1` Ulysses 并行大小
- `model: ModelConfig` 模型配置
- `optim: OptimConfig` 优化器配置
- `fsdp: FSDPConfig` FSDP 配置
- `offload: OffloadConfig` offload 配置

自动字段：
- `global_batch_size_per_device` 由全局 batch 和设备数自动计算

## RolloutConfig

定义位置：`verl/workers/rollout/config.py`。

字段：
- `name: str = "vllm"` rollout/推理后端名称
- `n: int = 1` 每个样本生成的候选数
- `temperature: float = 1.0` 采样温度
- `top_p: float = 1.0` nucleus sampling 参数
- `top_k: int = -1` top-k 截断（-1 表示不使用）
- `seed: int = 1` 随机种子
- `limit_images: int = 0` 每个样本最多使用的图片数量（0 表示不限制）
- `dtype: str = "bf16"` 模型推理 dtype
- `gpu_memory_utilization: float = 0.6` GPU 显存利用率上限
- `ignore_eos: bool = False` 是否忽略 EOS token
- `enforce_eager: bool = False` 是否强制 eager 模式
- `enable_chunked_prefill: bool = False` 是否启用 chunked prefill
- `tensor_parallel_size: int = 2` 张量并行大小
- `max_model_len: Optional[int] = None` 最大模型序列长度
- `max_num_batched_tokens: int = 8192` 单批最大 token 数
- `disable_log_stats: bool = True` 是否关闭统计日志
- `disable_tqdm: bool = False` 是否关闭 tqdm 进度条
- `val_override_config: dict[str, Any] = {}` 验证阶段用于覆盖 rollout 设置的字典

自动字段（在 `PPOConfig.post_init()` 中设置）：
- `prompt_length`（来自 `DataConfig.max_prompt_length`）
- `response_length`（来自 `DataConfig.max_response_length`）
- `trust_remote_code`（来自 `ActorConfig.model.trust_remote_code`）

## RewardConfig

定义位置：`verl/workers/reward/config.py`。

字段：
- `reward_function: Optional[str] = None`
  - 形式：`path/to/file.py:func_name`，如省略 `:func_name`，默认使用 `main`
- `reward_function_kwargs: dict = {}` 传给 reward 函数的 kwargs
- `skip_special_tokens: bool = True` 在计算 reward 前是否移除特殊 token
- `num_cpus: int = 1` 计算 reward 时使用的 CPU 数

`post_init` 行为：
- 将 `reward_function` 中的路径部分解析为绝对路径
- 从 `path:func` 字符串中解析出 `reward_function_name`，并在运行时按此加载对应函数
