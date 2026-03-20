#!/bin/bash
# =============================================================================
# Visual RL Training Script
# RL training launcher for mixed image-video tasks
#
# Usage:
#   Single node: bash examples/visual_rl/visual_rl_train.sh
#   Multi-node:
#     Node 0 (Head): WORLD_SIZE=2 RANK=0 MASTER_ADDR=<head_ip> bash examples/visual_rl/visual_rl_train.sh
#     Node 1 (Worker): WORLD_SIZE=2 RANK=1 MASTER_ADDR=<head_ip> bash examples/visual_rl/visual_rl_train.sh
#
# Environment variables:
#   MODEL_PATH: model path (default: Qwen/Qwen3-VL-8B-Instruct)
#   TRAIN_DATA: training data path
#   VAL_DATA: validation data path
#   OUTPUT_PATH: output path
#   EXPERIMENT_NAME: experiment name
#   NPROC_PER_NODE: GPUs per node (default: 8)
# =============================================================================

set -euo pipefail
set -x

# =============================================================================
# Environment variables
# =============================================================================
export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}
export TOKENIZERS_PARALLELISM=false
export RAY_worker_num_grpc_internal_threads=1
export RAY_ADDRESS=""
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# CUDA configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800000  # 30-minute timeout
export NCCL_DEBUG=INFO       # Enable NCCL debug logs
export NCCL_DEBUG_SUBSYS=ALL

# Important: enable verbose VERL logging
export VERL_LOG_LEVEL=DEBUG

# =============================================================================
# Multi-node distributed configuration
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
LOG_DIR=${LOG_DIR:-"${PROJECT_DIR}/logs/visual_rl_experiment"}
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/verl_rank${RANK:-0}_${TIMESTAMP}.log"

# =============================================================================
# Model & Data Configuration
# =============================================================================
CONFIG_PATH=${CONFIG_PATH:-"${PROJECT_DIR}/examples/visual_rl/visual_rl_config.yaml"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-8B-Instruct"}
TRAIN_DATA=${TRAIN_DATA:-"/path/to/your/train_data.jsonl"}
VAL_DATA=${VAL_DATA:-"/path/to/your/val_data.json"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"visual_rl_experiment"}
SAVE_CHECKPOINT_PATH=${SAVE_CHECKPOINT_PATH:-"${PROJECT_DIR}/checkpoints/visual_rl/${EXPERIMENT_NAME}"}
FIND_LAST_CHECKPOINT=${FIND_LAST_CHECKPOINT:-true}

# Prompt template & Reward function paths
FORMAT_PROMPT=${FORMAT_PROMPT:-"${PROJECT_DIR}/examples/visual_rl/format_prompt/unified.jinja"}
REWARD_FUNCTION=${REWARD_FUNCTION:-"${PROJECT_DIR}/examples/visual_rl/reward_function/unified.py:compute_score"}


# =============================================================================
# Print configuration
# =============================================================================
echo "============================================================"
echo "  Visual RL Training - Distributed Mode"
echo "============================================================"
echo "  Total nodes: ${WORLD_SIZE}, current node: ${RANK}"
echo "  Head node: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  GPUs per node: ${NPROC_PER_NODE}, total GPUs: $((WORLD_SIZE * NPROC_PER_NODE))"
echo "------------------------------------------------------------"
echo "  Project directory: ${PROJECT_DIR}"
echo "  Config file: ${CONFIG_PATH}"
echo "  Model path: ${MODEL_PATH}"
echo "  Training data: ${TRAIN_DATA}"
echo "  Validation data: ${VAL_DATA}"
echo "  Experiment name: ${EXPERIMENT_NAME}"
echo "  Log file: ${LOG_FILE}"
echo "------------------------------------------------------------"
echo "  Prompt template: ${FORMAT_PROMPT}"
echo "  Reward function: ${REWARD_FUNCTION}"
echo "============================================================"

# =============================================================================
# Ray cluster management helpers
# =============================================================================
cleanup_ray() {
    echo "[INFO] Cleaning up Ray processes..."
    ray stop --force 2>/dev/null || true
    sleep 3
}

wait_for_head() {
    local max_attempts=60
    local attempt=0
    echo "[INFO] Waiting for Head node at ${MASTER_ADDR}:${MASTER_PORT}..."
    while [ $attempt -lt $max_attempts ]; do
        if ray status --address="${MASTER_ADDR}:${MASTER_PORT}" &>/dev/null; then
            echo "[INFO] Head node is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        echo "  Waiting for Head node... ($attempt/$max_attempts)"
        sleep 5
    done
    echo "[ERROR] Timed out waiting for Head node"
    return 1
}

wait_for_workers() {
    local expected_nodes=$WORLD_SIZE
    local max_attempts=120
    local attempt=0

    echo "[INFO] Waiting for $expected_nodes nodes to connect..."
    while [ $attempt -lt $max_attempts ]; do
        local connected_nodes
        connected_nodes=$(ray status 2>/dev/null | grep -c "node_" || echo "0")
        echo "  Connected nodes: $connected_nodes / $expected_nodes (attempt $attempt/$max_attempts)"

        if [ "$connected_nodes" -ge "$expected_nodes" ]; then
            echo "[INFO] All nodes are connected!"
            ray status
            return 0
        fi

        attempt=$((attempt + 1))
        sleep 10
    done

    echo "[ERROR] Failed to wait for all nodes to connect"
    ray status
    return 1
}

# =============================================================================
# Switch to the project directory
# =============================================================================
cd "$PROJECT_DIR"

# =============================================================================
# Prepare output directories
# =============================================================================
mkdir -p "${SAVE_CHECKPOINT_PATH}"
export TENSORBOARD_DIR="${SAVE_CHECKPOINT_PATH}/tensorboard_log"
mkdir -p "${TENSORBOARD_DIR}"

# =============================================================================
# Execute based on node role
# =============================================================================
if [ "$RANK" == "0" ]; then
    # =========================================================================
    # Head node
    # =========================================================================
    cleanup_ray

    echo "[HEAD] Starting Ray Head node..."
    ray start --head \
        --port=${MASTER_PORT} \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=${RAY_DASHBOARD_PORT} \
        --num-gpus=${NPROC_PER_NODE} \
        --disable-usage-stats

    # Wait for workers in multi-node mode
    if [ "$WORLD_SIZE" -gt 1 ]; then
        echo "[HEAD] Waiting for Worker nodes to connect..."
        if ! wait_for_workers; then
            echo "[ERROR] Cluster is not ready, exiting"
            cleanup_ray
            exit 1
        fi
    fi

    # Launch training
    echo "[HEAD] Launching Visual RL training..."
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
        trainer.save_checkpoint_path=${SAVE_CHECKPOINT_PATH} \
        trainer.n_gpus_per_node=${NPROC_PER_NODE} \
        trainer.nnodes=${WORLD_SIZE} \
        trainer.save_freq=25 \
        trainer.save_limit=5 \
        2>&1 | tee -a "$LOG_FILE"

    echo "[HEAD] Training completed!"
    cleanup_ray

else
    # =========================================================================
    # Worker node
    # =========================================================================
    cleanup_ray

    echo "[WORKER ${RANK}] Waiting for Head node..."
    if ! wait_for_head; then
        echo "[ERROR] Failed to connect to Head node, exiting"
        exit 1
    fi

    sleep 20

    echo "[WORKER ${RANK}] Connecting to Ray cluster..."
    ray start \
        --address="${MASTER_ADDR}:${MASTER_PORT}" \
        --num-gpus=${NPROC_PER_NODE} \
        --disable-usage-stats \
        2>&1 | tee -a "$LOG_FILE"

    ray status --address="${MASTER_ADDR}:${MASTER_PORT}"

    echo "[WORKER ${RANK}] Connected and waiting for tasks..."
    sleep inf
fi
