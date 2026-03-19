#!/bin/bash

set -euo pipefail
set -x

# Training Script for Video RL v2 (multi-node)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}

# =============================================================================
# 环境与路径配置
# =============================================================================
if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "CONDA_ENV_NAME is set but conda is not available on PATH." >&2
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME}"
fi

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export TOKENIZERS_PARALLELISM=false
export RAY_worker_num_grpc_internal_threads=1
export RAY_ADDRESS=""
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# CUDA 配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800000
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 关键: 添加详细日志
export VERL_LOG_LEVEL=DEBUG

# =============================================================================
# 多节点分布式配置
# =============================================================================
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-6379}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}

# =============================================================================
# 日志配置
# =============================================================================
LOG_DIR=${LOG_DIR:-"${PROJECT_DIR}/logs/video_rl_v2_dapo"}
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/verl_rank${RANK:-0}_${TIMESTAMP}.log"

# =============================================================================
# 模型和配置
# =============================================================================
CONFIG_PATH=${CONFIG_PATH:-"${PROJECT_DIR}/examples/video_rl_v2/video_rl_v1_qwen3_5.yaml"}
MODEL_PATH=${MODEL_PATH:-"./models/Qwen3.5-2B"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"video_rl_v2_dapo"}
SAVE_CHECKPOINT_PATH=${SAVE_CHECKPOINT_PATH:-"${PROJECT_DIR}/checkpoints/video_rl_v2_dapo"}

# Prompt模板和Reward函数路径
FORMAT_PROMPT=${FORMAT_PROMPT:-"examples/video_rl_v2/format_prompt/unified.jinja"}
REWARD_FUNCTION=${REWARD_FUNCTION:-"examples/video_rl_v2/video_v1.py:compute_score"}

# =============================================================================
# 打印集群信息
# =============================================================================
echo "============================================================"
echo "  节点总数: ${WORLD_SIZE}, 当前节点: ${RANK}"
echo "  主节点: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  每节点 GPU: ${NPROC_PER_NODE}, 总 GPU: $((WORLD_SIZE * NPROC_PER_NODE))"
echo "============================================================"

# =============================================================================
# Ray 集群管理函数
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
        echo "等待 Head 节点... ($attempt/$max_attempts)"
        sleep 5
    done
    echo "等待 Head 节点超时"
    return 1
}

wait_for_workers() {
    local expected_nodes=$WORLD_SIZE
    local max_attempts=60
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        local connected_nodes
        connected_nodes=$(ray status 2>/dev/null | grep -c "node_" || echo "0")
        echo "已连接节点: $connected_nodes / $expected_nodes (尝试 $attempt/$max_attempts)"

        if [ "$connected_nodes" -ge "$expected_nodes" ]; then
            echo "所有节点已连接!"
            ray status
            return 0
        fi

        attempt=$((attempt + 1))
        sleep 10
    done

    echo "错误: 未能等到所有节点连接"
    ray status
    return 1
}

# =============================================================================
# 切换到项目目录（format_prompt 使用相对路径，需要在项目根目录下运行）
# =============================================================================
cd "$PROJECT_DIR"

# =============================================================================
# 根据节点角色执行
# =============================================================================
if [ "$RANK" == "0" ]; then
    cleanup_ray

    ray start --head \
        --port=${MASTER_PORT} \
        --dashboard-port=${RAY_DASHBOARD_PORT} \
        --num-gpus=${NPROC_PER_NODE} \
        --disable-usage-stats

    if [ "$WORLD_SIZE" -gt 1 ]; then
        echo "等待 Worker 节点连接..."
        if ! wait_for_workers; then
            echo "集群未就绪，退出"
            exit 1
        fi
    fi

    python3 -m verl.trainer.main \
        config=${CONFIG_PATH} \
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
        2>&1 | tee -a "$LOG_FILE"

    echo "训练完成!"
else
    cleanup_ray
    wait_for_head
    sleep 20

    ray start \
        --address="${MASTER_ADDR}:${MASTER_PORT}" \
        --num-gpus=${NPROC_PER_NODE} \
        --disable-usage-stats \
        2>&1 | tee -a "$LOG_FILE"

    ray status --address="${MASTER_ADDR}:${MASTER_PORT}"

    echo "Worker 节点待命中..."
    sleep inf
fi
