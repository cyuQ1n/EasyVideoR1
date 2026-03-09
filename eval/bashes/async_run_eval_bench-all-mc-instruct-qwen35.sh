#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Optional jemalloc preload
# export LD_PRELOAD="YourLocalPath/miniconda3/envs/easyvideorl/lib/python3.11/site-packages/ray/core/libjemalloc.so"

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

cd "$REPO_ROOT" || exit 1

DATASETS="mmvu"

# 模型和输出配置
MODEL_PATH="${MODEL_PATH:-YourLocalPath/models/Qwen3.5-VL}"
DATA_DIR_PATH="${DATA_DIR_PATH:-$REPO_ROOT/eval/data}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_ROOT/eval/code/AsyncLLMEngine_eval_videobench_qwen3vl_multi_task.py}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/eval/caches}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval/outputs_async-35/thinking_modes-1024frame}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/eval/logs}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/eval/results-35}"

mkdir -p "$LOG_DIR"

MODEL_NAME=$(basename "$MODEL_PATH")

# 视频处理参数
NFRAMES_LIST="1024"
FPS=2.0
MAX_PIXELS=262144
TOTAL_PIXELS=33554432

# 推理参数
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20
PRESENCE_PENALTY=1.5
REPETITION_PENALTY=1.0
MAX_TOKENS=4096

# ============== AsyncLLMEngine 真流水线参数 ==============
NUM_WORKERS=16
LOAD_WORKERS=16
MAX_CONCURRENT=16
QUEUE_SIZE=16
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=140000
GPU_MEM_UTIL=0.7

for NFRAMES in $NFRAMES_LIST; do
    echo "############################################"
    echo "# Starting Async Pipeline experiments with NFRAMES=$NFRAMES"
    echo "############################################"

    for DATASET in $DATASETS; do
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        LOG_FILE="${LOG_DIR}/async_${MODEL_NAME}_${DATASET}_f${NFRAMES}_${TIMESTAMP}.log"

        if [[ "$DATASET" == "videommmu" || "$DATASET" == "longvideobench" || "$DATASET" == "videomathqa" ]]; then
            MAX_NUM_SEQS=4
            GPU_MEM_UTIL=0.6
        else
            MAX_NUM_SEQS=8
            GPU_MEM_UTIL=0.7
        fi

        echo "=========================================="
        echo "[Async] Evaluating dataset: $DATASET (NFRAMES=$NFRAMES)"
        echo "Log file: $LOG_FILE"
        echo "=========================================="

        python "$EVAL_SCRIPT" \
            --mode auto \
            --num_gpus 8 \
            --model_path "$MODEL_PATH" \
            --data_dir_path "$DATA_DIR_PATH" \
            --cache_dir "$CACHE_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --datasets "$DATASET" \
            --nframes "$NFRAMES" \
            --fps "$FPS" \
            --max_pixels "$MAX_PIXELS" \
            --total_pixels "$TOTAL_PIXELS" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --top_k "$TOP_K" \
            --presence_penalty "$PRESENCE_PENALTY" \
            --repetition_penalty "$REPETITION_PENALTY" \
            --max_tokens "$MAX_TOKENS" \
            --num_workers "$NUM_WORKERS" \
            --load_workers "$LOAD_WORKERS" \
            --max_concurrent "$MAX_CONCURRENT" \
            --queue_size "$QUEUE_SIZE" \
            --result_dir "$RESULT_DIR" \
            --max_num_seqs "$MAX_NUM_SEQS" \
            --max_num_batched_tokens "$MAX_NUM_BATCHED_TOKENS" \
            --gpu_mem_util "$GPU_MEM_UTIL" \
            --thinking_mode 2>&1 | tee "$LOG_FILE"

        echo "Finished evaluating: $DATASET (NFRAMES=$NFRAMES)"
        echo ""
    done

    echo "Completed all datasets for NFRAMES=$NFRAMES"
    echo ""
done

echo "All Async Pipeline experiments completed!"
