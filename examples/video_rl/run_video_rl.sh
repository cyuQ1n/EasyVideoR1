#!/bin/bash

set -euo pipefail
set -x

# VideoRL Training Script for Qwen3-VL (multi-node)

# =============================================================================
# Environment Configuration
# =============================================================================
export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}
export TOKENIZERS_PARALLELISM=false
export RAY_worker_num_grpc_internal_threads=1
export RAY_ADDRESS=""
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# CUDA Configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600000
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Fix NCCL communication issues
export NCCL_NVLS_ENABLE=0           # Disable NVLS, fix transport/nvls.cc Cuda failure
export NCCL_IB_RETRY_CNT=20         # Increase IB retry count for InfiniBand network jitter
export NCCL_IB_TIMEOUT=23           # Increase IB timeout

# VERL log level
export VERL_LOG_LEVEL=DEBUG

# =============================================================================
# Multi-node Distributed Configuration
# =============================================================================
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-6379}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}

# =============================================================================
# Project & Log Configuration
# =============================================================================
PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
LOG_DIR=${LOG_DIR:-"${PROJECT_DIR}/logs/video_rl_experiment"}
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/verl_rank${RANK:-0}_${TIMESTAMP}.log"

# =============================================================================
# Model & Data Configuration
# =============================================================================
CONFIG_PATH=${CONFIG_PATH:-"${PROJECT_DIR}/examples/video_rl/video_rl.yaml"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-8B-Instruct"}
TRAIN_DATA=${TRAIN_DATA:-"/path/to/your/train_data.jsonl"}
VAL_DATA=${VAL_DATA:-"/path/to/your/val_data.json"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"video_rl_experiment"}
SAVE_CHECKPOINT_PATH=${SAVE_CHECKPOINT_PATH:-"${PROJECT_DIR}/checkpoints/video_rl/${EXPERIMENT_NAME}"}
FIND_LAST_CHECKPOINT=${FIND_LAST_CHECKPOINT:-true}

# Prompt template & Reward function paths
FORMAT_PROMPT=${FORMAT_PROMPT:-"${PROJECT_DIR}/examples/video_rl/format_prompt/unified.jinja"}
REWARD_FUNCTION=${REWARD_FUNCTION:-"${PROJECT_DIR}/examples/video_rl/reward_function/video_reward.py:compute_score"}

# =============================================================================
# Print Cluster Info
# =============================================================================
echo "============================================================"
echo "  Total nodes: ${WORLD_SIZE}, Current node: ${RANK}"
echo "  Head node: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  GPUs per node: ${NPROC_PER_NODE}, Total GPUs: $((WORLD_SIZE * NPROC_PER_NODE))"
echo "  Config: ${CONFIG_PATH}"
echo "  Model: ${MODEL_PATH}"
echo "  Train data: ${TRAIN_DATA}"
echo "  Val data: ${VAL_DATA}"
echo "  Prompt template: ${FORMAT_PROMPT}"
echo "  Reward function: ${REWARD_FUNCTION}"
echo "  Checkpoint path: ${SAVE_CHECKPOINT_PATH}"
echo "============================================================"

# =============================================================================
# Ray Cluster Management
# =============================================================================
cleanup_ray() {
    ray stop --force 2>/dev/null || true
    sleep 3
}

wait_for_head() {
    local max_attempts=60
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if ray status --address="${MASTER_ADDR}:${MASTER_PORT}" &>/dev/null; then
            return 0
        fi
        attempt=$((attempt + 1))
        echo "Waiting for head node... ($attempt/$max_attempts)"
        sleep 5
    done
    echo "Timeout waiting for head node"
    return 1
}

wait_for_workers() {
    local expected_nodes=$WORLD_SIZE
    local max_attempts=60
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        local connected_nodes
        connected_nodes=$(ray status 2>/dev/null | grep -c "node_" || echo "0")
        echo "Connected nodes: $connected_nodes / $expected_nodes (attempt $attempt/$max_attempts)"

        if [ "$connected_nodes" -ge "$expected_nodes" ]; then
            echo "All nodes connected!"
            ray status
            return 0
        fi

        attempt=$((attempt + 1))
        sleep 10
    done

    echo "Error: not all nodes connected"
    ray status
    return 1
}

# =============================================================================
# Switch to project directory (format_prompt uses relative paths)
# =============================================================================
cd "$PROJECT_DIR"

# =============================================================================
# Execute based on node role
# =============================================================================
if [ "$RANK" == "0" ]; then
    # Head node
    cleanup_ray

    ray start --head \
        --port=${MASTER_PORT} \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=${RAY_DASHBOARD_PORT} \
        --num-gpus=${NPROC_PER_NODE} \
        --disable-usage-stats

    if [ "$WORLD_SIZE" -gt 1 ]; then
        echo "Waiting for worker nodes..."
        if ! wait_for_workers; then
            echo "Cluster not ready, exiting"
            exit 1
        fi
    fi

    # Submit training job
    python3 -m verl.trainer.main \
        config=${CONFIG_PATH} \
        data.train_files=${TRAIN_DATA} \
        data.val_files=${VAL_DATA} \
        data.format_prompt=${FORMAT_PROMPT} \
        worker.actor.model.model_path=${MODEL_PATH} \
        worker.actor.clip_ratio_low=0.2 \
        worker.actor.clip_ratio_high=0.28 \
        worker.reward.reward_function=${REWARD_FUNCTION} \
        algorithm.disable_kl=True \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.n_gpus_per_node=${NPROC_PER_NODE} \
        trainer.nnodes=${WORLD_SIZE} \
        trainer.save_checkpoint_path=${SAVE_CHECKPOINT_PATH} \
        trainer.find_last_checkpoint=${FIND_LAST_CHECKPOINT} \
        2>&1 | tee -a "$LOG_FILE"

    echo "Training complete!"
    cleanup_ray

else
    # Worker node
    cleanup_ray
    wait_for_head
    sleep 20

    ray start \
        --address="${MASTER_ADDR}:${MASTER_PORT}" \
        --num-gpus=${NPROC_PER_NODE} \
        --disable-usage-stats \
        2>&1 | tee -a "$LOG_FILE"

    ray status --address="${MASTER_ADDR}:${MASTER_PORT}"

    echo "Worker node standing by..."
    sleep inf
fi
