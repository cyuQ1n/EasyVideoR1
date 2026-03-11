# EasyVideoR1 Eval

`EasyVideoR1/eval` provides an offline evaluation toolkit for video understanding models. At its core is an asynchronous pipeline inference script built on `vLLM AsyncLLMEngine`, supporting video feature caching, batch evaluation, open-ended LLM-as-a-Judge post-processing, and a wide range of video understanding benchmarks.

This directory mainly contains:

- `code/AsyncLLMEngine_eval_videobench_qwen3vl_multi_task.py`: Main evaluation script
- `code/llm_judge.py`: Open-ended post-processing script
- `data/valid_data/`: Built-in evaluation annotation files
- `bashes/`: Example batch scripts
- `preprocess/`: Video preprocessing tools (clip trimming, etc.)

## Features

- **True pipeline evaluation** based on `AsyncLLMEngine`
  - Requests are continuously fed into the engine via `add_request()`
  - Data loading, preprocessing, and inference run in parallel
  - Supports continuous scheduling for long video scenarios
- **Video caching** support
  - Preprocess videos offline and reuse the cache for repeated evaluations
  - Supports four run modes: `auto / preprocess / eval / both`
- **Qwen3-VL**, **Qwen3.5-VL**, and **Qwen2.5-VL** support
  - `--model_family qwen3`
  - `--model_family qwen25`
- **Video + image joint input**
  - Benchmarks like `VideoMMMU` can attach an additional image via `image_path`
- **Two-stage scoring for open-ended questions**
  - The main evaluation script produces raw results
  - `llm_judge.py` performs semantic consistency scoring on `open-ended` samples
- **Result aggregation**
  - Per-sample details are saved to `output_dir`
  - Summary metrics are saved to `result_dir`

## Installation

### Step 1: Create Environment

```bash
conda create -n easyvideorl python=3.11
conda activate easyvideorl
```

### Step 2: Install the Repository

Run the following from the repository root:

```bash
git clone https://github.com/cyuQ1n/EasyVideoR1.git
cd EasyVideoR1
pip install -e .
pip install -r requirements.txt
```

### Step 3: (Optional) Install Flash Attention

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

## Quick Start

### 1. Prepare the Evaluation Data Directory

The main script organizes data via `--data_dir_path` with the following structure:

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

The repository ships with annotation files in `eval/data/valid_data/`, but **raw video/image files are not included**.
If you use the built-in annotations, it is recommended to explicitly specify:

```bash
--data_dir_path /path/to/EasyVideoR1/eval/data
```

Then place the corresponding dataset video directories under that path, or use `--dataset_config` to override paths.

### 2. Run Evaluation on a Single Dataset

Run the following from the repository root:

```bash
python eval/code/AsyncLLMEngine_eval_videobench_qwen3vl_multi_task.py \
  --mode auto \
  --model_path /path/to/your/model \
  --model_family qwen3 \
  --data_dir_path /path/to/EasyVideoR1/eval/data \
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

### 3. Preprocess Video Cache Only

```bash
python eval/code/AsyncLLMEngine_eval_videobench_qwen3vl_multi_task.py \
  --mode preprocess \
  --model_path /path/to/your/model \
  --data_dir_path /path/to/EasyVideoR1/eval/data \
  --datasets lvbench videomme \
  --cache_dir ./eval/video_cache
```

### 4. Evaluate Using Existing Cache Only

```bash
python eval/code/AsyncLLMEngine_eval_videobench_qwen3vl_multi_task.py \
  --mode eval \
  --model_path /path/to/your/model \
  --data_dir_path /path/to/EasyVideoR1/eval/data \
  --datasets lvbench videomme \
  --cache_dir ./eval/video_cache \
  --output_dir ./eval/output-async \
  --result_dir ./eval/result-async
```

### 5. Run LLM Judge on Open-Ended Results

Final scores for `open-ended` samples are best obtained via `llm_judge.py` post-processing:

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

## Run Modes

| Mode | Description |
|------|-------------|
| `auto` | Default mode; evaluates directly if cache coverage is sufficient, otherwise preprocesses first |
| `preprocess` | Only generates video cache |
| `eval` | Only runs evaluation; requires existing cache |
| `both` | Forces preprocessing before evaluation |

## Supported Evaluation Tasks

The main script currently supports the following task types and scoring methods:

| Task Type | Scoring Method |
|-----------|---------------|
| `multiple choice` | Exact option matching |
| `numerical` | Numerical parsing and comparison |
| `regression` | Mean Relative Accuracy (MRA) |
| `temporal grounding` | Temporal interval IoU |
| `spatial-temporal grounding` | Combined temporal IoU and bounding box mIoU |
| `open-ended` | Recommended via `llm_judge.py` for semantic consistency scoring |

Additional notes:

- Prompt templates also support `OCR` and `free-form` QA formats
- The automatic scoring logic primarily covers the task types listed above
- `thinking_mode` switches to reasoning-augmented prompt templates

## Supported Benchmarks

The main script includes built-in dataset mappings for the following benchmarks:

| Dataset | Task Type | Description |
|---------|-----------|-------------|
| `holmes` | Multiple choice | Video-Holmes |
| `lvbench` | Multiple choice | LVBench |
| `longvideobench` | Multiple choice | LongVideoBench |
| `mmvu` | Multiple choice | MMVU |
| `mmvu-all` | Multiple choice, open-ended | MMVU full version |
| `mvbench` | Multiple choice | MVBench |
| `tempcompass` | Multiple choice | TempCompass |
| `vsibench` | Multiple choice, regression | VSI-Bench |
| `videommmu` | Multiple choice, numerical | Uses `videommmu_with_image.json`; supports additional image input |
| `mlvu` | Multiple choice | MLVU Test |
| `mlvu-test` | Multiple choice | MLVU Test |
| `mlvu-dev` | Multiple choice | MLVU Dev |
| `videomme` | Multiple choice | Video-MME |
| `videomathqa` | Multiple choice | VideoMathQA |
| `vcrbench` | Multiple choice, open-ended | VCR-Bench |
| `videoreasonbench` | Open-ended | Recommended with `llm_judge.py` |
| `longvideoreason` | Multiple choice | LongVideo-Reason |
| `minerva` | Multiple choice | Minerva / Minderva |
| `stvg` | Spatial-temporal grounding | ST-Align-Benchmark |
| `charades_sta` | Temporal grounding | Charades-STA |
| `motionbench` | Multiple choice | MotionBench |
| `ovobench` | Multiple choice, counting, yes/no | OVO-Bench |
| `odvbench` | Multiple choice | ODV-Bench |
| `livesports3k_qa` | Multiple choice | LiveSports-3K QA |

Additional notes:

- `eval/data/valid_data/` also contains files like `mmvu-mc.json` and `videommlu.json`; these can be integrated via `--dataset_config`
- `odvbench` and `livesports3k_qa` source videos are long videos that need to be trimmed into clips using tools in `preprocess/` before evaluation (see the "Video Preprocessing" section below)

## Video Preprocessing

ODV-Bench and LiveSports-3K annotations correspond to long video segments that must be trimmed before evaluation. 

```bash
cd EasyVideoR1/eval/preprocess

# ODV-Bench
python cilp.py \
    --input_json ../data/valid_data/odvbench.json \
    --video_root /path/to/ODV-Bench \
    --output_dir ../data/ODV-Bench/clips \
    --format odvbench

# LiveSports-3K
python cilp.py \
    --input_json ../data/valid_data/livesports3k_qa.json \
    --video_root /path/to/LiveSports-3K/videos \
    --output_dir ../data/LiveSports-3K-QA/clips \
    --format livesports
```


Data sources: [ODV-Bench](https://huggingface.co/datasets/MCG-NJU/ODV-Bench), [LiveSports-3K](https://huggingface.co/datasets/stdKonjac/LiveSports-3K)

## Custom Datasets

You can override the default dataset mappings via `--dataset_config`. The configuration file format is:

```json
{
  "mybench": {
    "json": "/path/to/mybench.json",
    "video": "/path/to/mybench_videos"
  }
}
```

Example usage:

```bash
python eval/code/AsyncLLMEngine_eval_videobench_qwen3vl_multi_task.py \
  --model_path /path/to/your/model \
  --dataset_config /path/to/dataset_config.json \
  --datasets mybench
```

## Output Description

After evaluation, two types of results are typically generated:

- Per-sample details: `output_dir/<dataset>_f<nframes>_fps<fps>_mp<...>_tp<...>/<ModelName>_output.json`
- Summary results: `result_dir/<ModelName>.json`

The per-sample detail file contains:

- `results`: Per-sample output, extracted answers, rewards, etc.
- `final_acc`: Overall metrics
- `eval_time_seconds` / `eval_time_minutes`
- `total_output_tokens` / `avg_output_tokens`

Cache directory names automatically encode dataset and sampling parameters, for example:

```text
lvbench_f256_fps2.0_mp256k_tp32768k
```

## Example Scripts

`eval/bashes/` provides ready-to-use batch evaluation scripts that can be used as templates:

- `async_run_eval_bench-all-mc-instruct-qwen3vl.sh`
- `async_run_eval_bench-all-mc-instruct-qwen25-1.sh`
- `async_run_eval_bench-all-mc-instruct-qwen35.sh`
- `async_run_eval_bench-all-mc-instruct-parallel.sh`
- `run_judge_test.sh`

These scripts demonstrate:

- Multi-GPU `CUDA_VISIBLE_DEVICES` allocation
- Switching between `qwen3 / qwen3.5 / qwen25` models
- Adjusting `max_num_seqs` and GPU memory parameters for different datasets
- Parallel evaluation across multiple datasets

Additional notes:

- `Qwen3.5-VL` also uses the `--model_family qwen3` branch in the current evaluation scripts
- Refer to `eval/bashes/async_run_eval_bench-all-mc-instruct-qwen35.sh` for details

## Common Parameters

### Data and Caching

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_dir_path` | Data root directory | `./eval/data` |
| `--datasets` | List of datasets to evaluate | All built-in datasets |
| `--dataset_config` | Custom dataset mapping | Empty |
| `--cache_dir` | Video cache root directory | `./video_cache` |
| `--mode` | Run mode | `auto` |

### Video Processing

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--nframes` | Maximum number of sampled frames | `2048` |
| `--fps` | Video sampling frame rate | `2.0` |
| `--max_pixels` | Per-frame pixel limit | `640*32*32` |
| `--total_pixels` | Total pixel budget | `160376*32*32` |

### Inference and Concurrency

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max_tokens` | Maximum generation length | `2048` |
| `--max_model_len` | Maximum context length | `256000` |
| `--gpu_mem_util` | vLLM GPU memory utilization ratio | `0.7` |
| `--max_num_seqs` | vLLM concurrent sequence count | `8` |
| `--max_num_batched_tokens` | vLLM batch token limit | `262144` |
| `--max_concurrent` | Engine maximum concurrent requests | `16` |
| `--queue_size` | Prefetch queue size | `8` |
| `--thinking_mode` | Enable reasoning-augmented prompt templates | Disabled |

## FAQ

**Q: Why can the script find annotation files but not the videos?**

A: The repository primarily provides annotations in `eval/data/valid_data/`. Raw videos typically need to be placed in the corresponding subdirectories under `--data_dir_path`, or redirected via `--dataset_config`.

**Q: Why are the scores for open-ended questions not ideal after the main evaluation?**

A: `open-ended` questions are best evaluated using the two-stage `llm_judge.py` pipeline. The main script is more suited for producing raw model answers and intermediate results.

**Q: When should I use `auto` vs `eval`?**

A: Use `auto` when running a dataset for the first time. Once you have a complete cache, use `eval` directly for repeated experiments.

**Q: Why does `videommmu` have both video and image inputs?**

A: When a sample has an `image_path` field, the script appends the image to the message content along with the hint: "The image for this question is at the end of the video."

## License

This directory is part of the EasyVideoR1 project and follows the license specified in the repository root.
