#!/bin/bash
# LLM-as-a-Judge 评测脚本
# 用法: bash run_judge_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_RPC_TIMEOUT=200000
export VLLM_USE_V1="1"
export FORCE_QWENVL_VIDEO_READER=decord

# Optional proxy settings
# export http_proxy="http://YourProxyHost:Port"
# export https_proxy="http://YourProxyHost:Port"

CONDA_SH="${CONDA_SH:-YourLocalPath/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then
    . "$CONDA_SH"
elif ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Please set CONDA_SH to your conda.sh path."
    exit 1
fi
conda activate easyvideorl

# ============== 配置区 ==============
INPUT_JSON="${INPUT_JSON:-YourLocalPath/eval_outputs/videoreasonbench/YourModel_output.json}"
MODEL_PATH="${MODEL_PATH:-YourLocalPath/models/your_judge_model}"
OUTPUT_JSON="${OUTPUT_JSON:-YourLocalPath/eval_outputs/videoreasonbench/YourModel_output_judged.json}"
RESULT_FILE="${RESULT_FILE:-$REPO_ROOT/eval/result-async/YourModel.json}"
DATASET_NAME="videoreasonbench"
# ====================================

python "$REPO_ROOT/eval/code/llm_judge.py" \
    --input_json    "$INPUT_JSON" \
    --model_path    "$MODEL_PATH" \
    --output_json   "$OUTPUT_JSON" \
    --result_file   "$RESULT_FILE" \
    --dataset_name  "$DATASET_NAME" \
    --num_gpus      1 \
    --max_tokens    4096 \
    --temperature   0.8
