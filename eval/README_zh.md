# EasyVideoR1 Eval

`EasyVideoR1/eval` 提供了一个面向视频理解模型的离线评测工具链，核心是基于 `vLLM AsyncLLMEngine` 的异步流水线推理脚本，支持视频特征缓存、批量评测、开放题 LLM-as-a-Judge 后处理，以及多种视频理解评测集。

本目录主要包含：

- `code/AsyncLLMEngine_eval_videobench_qwen3vl_multi_task.py`：主评测脚本
- `code/llm_judge.py`：开放题后处理脚本
- `data/valid_data/`：内置的评测标注文件
- `bashes/`：批量运行示例脚本

## 特性

- 基于 `AsyncLLMEngine` 的**真流水线评测**
  - 请求通过 `add_request()` 持续送入引擎
  - 数据加载、预处理与推理解耦并行
  - 支持长视频场景下的连续调度
- 支持**视频缓存**
  - 可先离线预处理视频，再重复复用缓存进行评测
  - 支持 `auto / preprocess / eval / both` 四种运行模式
- 支持 **Qwen3-VL**、**Qwen3.5-VL** 与 **Qwen2.5-VL**
  - `--model_family qwen3`
  - `--model_family qwen25`
- 支持**视频 + 图片联合输入**
  - `VideoMMMU` 这类样本可通过 `image_path` 附带单张图片
- 支持**开放题二阶段打分**
  - 主评测输出原始结果
  - 使用 `llm_judge.py` 对 `open-ended` 样本做语义一致性判分
- 支持结果汇总
  - 明细输出到 `output_dir`
  - 汇总指标输出到 `result_dir`

## 安装

### 第一步：创建环境

```bash
conda create -n easyvideorl python=3.11
conda activate easyvideorl
```

### 第二步：安装仓库

在仓库根目录执行：

```bash
git clone https://github.com/cyuQ1n/EasyVideoR1.git
cd EasyVideoR1
pip install -e .
pip install -r requirements.txt
```

### 第三步：可选安装 Flash Attention

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

## 快速开始

### 1. 准备评测数据目录

主脚本默认通过 `--data_dir_path` 组织数据目录，格式如下：

```text
YOUR_EVAL_DATA_ROOT/
├── valid_data/
│   ├── lvbench.json
│   ├── mmvu.json
│   └── ...
├── LVBench/
├── MMVU/
├── Video-MME/
└── ...
```

仓库已经自带 `eval/data/valid_data/` 中的标注文件，但**原始视频/图片文件不在仓库内**。  
如果你直接使用本仓库内置标注，推荐显式指定：

```bash
--data_dir_path /mnt/public/users/siqingyi/EasyVideoR1/EasyVideoR1/eval/data
```

然后将对应数据集视频目录按脚本约定放到该目录下，或使用 `--dataset_config` 覆盖路径。

### 2. 运行单个数据集评测

在仓库根目录执行：

```bash
python eval/code/AsyncLLMEngine_eval_videobench_qwen3vl_multi_task.py \
  --mode auto \
  --model_path /path/to/your/model \
  --model_family qwen3 \
  --data_dir_path /mnt/public/users/siqingyi/EasyVideoR1/EasyVideoR1/eval/data \
  --datasets lvbench \
  --cache_dir ./eval/video_cache \
  --output_dir ./eval/output-async \
  --result_dir ./eval/result-async \
  --nframes 256 \
  --fps 2.0 \
  --max_pixels 262144 \
  --total_pixels 33554432 \
  --max_tokens 4096 \
  --max_num_seqs 8 \
  --max_concurrent 16 \
  --queue_size 16 \
  --thinking_mode
```

### 3. 仅预处理视频缓存

```bash
python eval/code/AsyncLLMEngine_eval_videobench_qwen3vl_multi_task.py \
  --mode preprocess \
  --model_path /path/to/your/model \
  --data_dir_path /mnt/public/users/siqingyi/EasyVideoR1/EasyVideoR1/eval/data \
  --datasets lvbench videomme \
  --cache_dir ./eval/video_cache
```

### 4. 只使用已有缓存做评测

```bash
python eval/code/AsyncLLMEngine_eval_videobench_qwen3vl_multi_task.py \
  --mode eval \
  --model_path /path/to/your/model \
  --data_dir_path /mnt/public/users/siqingyi/EasyVideoR1/EasyVideoR1/eval/data \
  --datasets lvbench videomme \
  --cache_dir ./eval/video_cache \
  --output_dir ./eval/output-async \
  --result_dir ./eval/result-async
```

### 5. 对开放题结果做 LLM Judge

`open-ended` 样本的最终分数建议通过 `llm_judge.py` 后处理获得：

```bash
python eval/code/llm_judge.py \
  --input_json ./eval/output-async/videoreasonbench_f256_fps2.0_mp256k_tp32768k/YourModel_output.json \
  --model_path /path/to/judge/model \
  --output_json ./eval/output-async/videoreasonbench_f256_fps2.0_mp256k_tp32768k/YourModel_output_judged.json \
  --result_file ./eval/result-async/YourModel.json \
  --dataset_name videoreasonbench \
  --num_gpus 1 \
  --max_tokens 2048 \
  --temperature 0.1
```

## 运行模式

| 模式 | 说明 |
|------|------|
| `auto` | 默认模式；若缓存覆盖率足够则直接评测，否则先预处理再评测 |
| `preprocess` | 只生成视频缓存 |
| `eval` | 只执行评测，要求已有缓存 |
| `both` | 强制预处理后再评测 |

## 支持的评测功能

当前主脚本已经落地支持以下任务及打分方式：

| 任务类型 | 评分方式 |
|------|------|
| `multiple choice` | 精确匹配选项 |
| `numerical` | 数值解析后比较 |
| `regression` | `mean relative accuracy (MRA)` |
| `temporal grounding` | 时间区间 `IoU` |
| `spatial-temporal grounding` | 时间 `IoU` 与框 `mIoU` 的组合分数 |
| `open-ended` | 建议通过 `llm_judge.py` 做语义一致性打分 |

补充说明：

- Prompt 模板中还兼容了 `OCR`、`free-form` 等问答格式
- 但当前自动评分逻辑主要覆盖上表中的任务类型
- `thinking_mode` 会切换到带推理过程的提示模板

## 支持的评测集

主脚本内置了以下数据集映射：

| 数据集 | 任务类型 | 说明 |
|------|------|------|
| `holmes` | 多选题 | Video-Holmes |
| `lvbench` | 多选题 | LVBench |
| `longvideobench` | 多选题 | LongVideoBench |
| `mmvu` | 多选题 | MMVU |
| `mmvu-all` | 多选题、开放题 | MMVU 全量版本 |
| `mvbench` | 多选题 | MVBench |
| `tempcompass` | 多选题 | TempCompass |
| `vsibench` | 多选题、回归 | VSI-Bench |
| `videommmu` | 多选题、数值题 | 使用 `videommmu_with_image.json`，支持额外图片输入 |
| `mlvu` | 多选题 | MLVU Test |
| `mlvu-test` | 多选题 | MLVU Test |
| `mlvu-dev` | 多选题 | MLVU Dev |
| `videomme` | 多选题 | Video-MME |
| `videomathqa` | 多选题 | VideoMathQA |
| `vcrbench` | 多选题、开放题 | VCR-Bench |
| `videoreasonbench` | 开放题 | 建议配合 `llm_judge.py` |
| `longvideoreason` | 多选题 | LongVideo-Reason |
| `minerva` | 多选题 | Minerva / Minderva |
| `stvg` | 时空定位 | ST-Align-Benchmark |
| `charades_sta` | 时间定位 | Charades-STA |
| `motionbench` | 多选题 | MotionBench |

额外说明：

- `eval/data/valid_data/` 中还包含 `mmvu-mc.json`、`videommlu.json` 等文件；如果需要，可通过 `--dataset_config` 自定义接入

## 自定义数据集

可以通过 `--dataset_config` 覆盖默认数据集映射。配置文件格式如下：

```json
{
  "mybench": {
    "json": "/path/to/mybench.json",
    "video": "/path/to/mybench_videos"
  }
}
```

启动示例：

```bash
python eval/code/AsyncLLMEngine_eval_videobench_qwen3vl_multi_task.py \
  --model_path /path/to/your/model \
  --dataset_config /path/to/dataset_config.json \
  --datasets mybench
```

## 输出说明

评测完成后通常会生成两类结果：

- 明细结果：`output_dir/<dataset>_f<nframes>_fps<fps>_mp<...>_tp<...>/<ModelName>_output.json`
- 汇总结果：`result_dir/<ModelName>.json`

其中明细文件包含：

- `results`：逐样本输出、抽取答案、reward 等信息
- `final_acc`：总体指标
- `eval_time_seconds` / `eval_time_minutes`
- `total_output_tokens` / `avg_output_tokens`

缓存目录名会自动编码数据集与采样参数，例如：

```text
lvbench_f256_fps2.0_mp256k_tp32768k
```

## 示例脚本

`eval/bashes/` 下给出了现成的批量评测脚本，可作为模板直接修改：

- `async_run_eval_bench-all-mc-instruct-qwen3vl.sh`
- `async_run_eval_bench-all-mc-instruct-qwen25-1.sh`
- `async_run_eval_bench-all-mc-instruct-qwen35.sh`
- `async_run_eval_bench-all-mc-instruct-parallel.sh`
- `run_judge_test.sh`

这些脚本主要展示了：

- 多卡 `CUDA_VISIBLE_DEVICES` 分配
- `qwen3 / qwen3.5 / qwen25` 模型切换
- 不同数据集下的 `max_num_seqs` 与显存参数调整
- 多组数据集并行评测

补充说明：

- `Qwen3.5-VL` 在当前评测脚本中同样走 `--model_family qwen3` 分支
- 可直接参考 `eval/bashes/async_run_eval_bench-all-mc-instruct-qwen35.sh`

## 常用参数

### 数据与缓存

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir_path` | 数据根目录 | `/mnt/public/users/siqingyi/video_reasoning/data/test` |
| `--datasets` | 需要评测的数据集列表 | 全部内置数据集 |
| `--dataset_config` | 自定义数据集映射 | 空 |
| `--cache_dir` | 视频缓存根目录 | `./video_cache` |
| `--mode` | 运行模式 | `auto` |

### 视频处理

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--nframes` | 最大采样帧数 | `2048` |
| `--fps` | 视频采样帧率 | `2.0` |
| `--max_pixels` | 单帧像素上限 | `640*32*32` |
| `--total_pixels` | 总像素预算 | `160376*32*32` |

### 推理与并发

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max_tokens` | 最大生成长度 | `2048` |
| `--max_model_len` | 最大上下文长度 | `256000` |
| `--gpu_mem_util` | vLLM 显存占用比例 | `0.7` |
| `--max_num_seqs` | vLLM 并发序列数 | `8` |
| `--max_num_batched_tokens` | vLLM batch token 上限 | `262144` |
| `--max_concurrent` | 引擎最大并发请求数 | `16` |
| `--queue_size` | 预取队列大小 | `8` |
| `--thinking_mode` | 启用推理型提示模板 | 关闭 |

## 常见问题

**问：为什么脚本能找到标注文件，但找不到视频？**

答：仓库里主要提供了 `eval/data/valid_data/` 的标注，原始视频通常需要你自行放到 `--data_dir_path` 对应的子目录下，或者用 `--dataset_config` 重定向。

**问：开放题为什么主评测后分数不理想？**

答：`open-ended` 建议走 `llm_judge.py` 二阶段评测，主脚本更适合先产出模型原始答案与中间结果。

**问：什么时候用 `auto`，什么时候用 `eval`？**

答：第一次跑某个数据集建议用 `auto`；已有完整缓存后，重复实验更适合直接用 `eval`。

**问：`videommmu` 为什么既有视频又有图片？**

答：该脚本会在样本带有 `image_path` 时，将图片附加到消息内容中，并加上一句 “The image for this question is at the end of the video.” 的提示。

## 许可证

本目录属于 EasyVideoR1 项目的一部分，遵循仓库根目录中的许可证说明。
