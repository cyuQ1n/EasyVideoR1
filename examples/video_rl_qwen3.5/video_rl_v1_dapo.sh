#!/bin/bash

set -euo pipefail
set -x

# Training Script for Video RL v2 (multi-node)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-easyvideor1-for-qwen3.5}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-}
SKIP_CONDA_ACTIVATE=${SKIP_CONDA_ACTIVATE:-0}
STRICT_ENV_CHECK=${STRICT_ENV_CHECK:-1}

# =============================================================================
# 环境与路径配置
# =============================================================================
activate_conda_env() {
    if [[ "${SKIP_CONDA_ACTIVATE}" == "1" ]]; then
        return 0
    fi

    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is not available on PATH. Set SKIP_CONDA_ACTIVATE=1 only if you already entered the correct env." >&2
        exit 1
    fi

    eval "$(conda shell.bash hook)"
    if [[ -n "${CONDA_ENV_PREFIX}" ]]; then
        conda activate "${CONDA_ENV_PREFIX}"
    else
        conda activate "${CONDA_ENV_NAME}"
    fi
}

require_command() {
    local cmd=$1
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "Required command not found: ${cmd}" >&2
        exit 1
    fi
}

check_qwen35_env() {
    "${PYTHON_BIN}" - <<'PY'
import os
import re
import sys
from importlib import metadata
from importlib.util import find_spec


def parse_version(version: str) -> tuple[int, ...]:
    numbers = [int(part) for part in re.findall(r"\d+", version)]
    return tuple(numbers) if numbers else (0,)


def version_ge(version: str, minimum: str) -> bool:
    lhs = parse_version(version)
    rhs = parse_version(minimum)
    width = max(len(lhs), len(rhs))
    lhs += (0,) * (width - len(lhs))
    rhs += (0,) * (width - len(rhs))
    return lhs >= rhs


strict = os.environ.get("STRICT_ENV_CHECK", "1") == "1"
errors: list[str] = []
warnings: list[str] = []

required = [
    ("qwen-vl-utils", "0.0.14", "Qwen3.5 best practice recommends qwen-vl-utils>=0.0.14."),
    ("vllm", "0.17.0", "Qwen3.5 RL rollout expects a recent vLLM (>=0.17.0)."),
    ("flash-linear-attention", "0.4.2", "Qwen3.5 training benefits from flash-linear-attention>=0.4.2."),
    ("flash-attn", "2.8.3", "The actor/critic stack expects flash-attn to be installed."),
]
optional = [
    ("torchcodec", None, "Install torchcodec if qwen-vl-utils video decoding hangs with decord."),
    ("causal-conv1d", None, "Recommended by the official Qwen3.5 best-practice environment."),
]

print("Python executable:", sys.executable)
print("Conda env:", os.environ.get("CONDA_DEFAULT_ENV", "<none>"))

try:
    transformers_version = metadata.version("transformers")
    print(f"transformers=={transformers_version}")
except metadata.PackageNotFoundError:
    errors.append("Missing required package: transformers. Qwen3.5 requires a transformers build that exposes qwen3_5.")
    transformers_version = None

if find_spec("transformers.models.qwen3_5.modeling_qwen3_5") is None:
    errors.append(
        "Current transformers build does not expose transformers.models.qwen3_5.modeling_qwen3_5. "
        "Install a Qwen3.5-capable transformers version."
    )

for dist_name, min_version, note in required:
    try:
        installed = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        errors.append(f"Missing required package: {dist_name}. {note}")
        continue

    print(f"{dist_name}=={installed}")
    if min_version is not None and not version_ge(installed, min_version):
        errors.append(f"{dist_name}=={installed} is older than the recommended minimum {min_version}. {note}")

for dist_name, min_version, note in optional:
    try:
        installed = metadata.version(dist_name)
        print(f"{dist_name}=={installed}")
    except metadata.PackageNotFoundError:
        warnings.append(f"Optional package missing: {dist_name}. {note}")

if warnings:
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

if errors:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if strict:
        sys.exit(1)
PY
}

activate_conda_env

export PYTHONNOUSERSITE=1
if [[ -n "${CONDA_ENV_PREFIX}" && -x "${CONDA_ENV_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_ENV_PREFIX}/bin/python"
else
    require_command python
    PYTHON_BIN=$(command -v python)
fi

if [[ -n "${CONDA_ENV_PREFIX}" && -x "${CONDA_ENV_PREFIX}/bin/ray" ]]; then
    RAY_BIN="${CONDA_ENV_PREFIX}/bin/ray"
else
    require_command ray
    RAY_BIN=$(command -v ray)
fi

check_qwen35_env

export WANDB_API_KEY=${WANDB_API_KEY:-e72fce795d7e86b88f22eb1218731ec7e748feab}
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
LOG_DIR=${LOG_DIR:-"${PROJECT_DIR}/logs/qwen3_5_onethinker100k"}
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/verl_rank${RANK:-0}_${TIMESTAMP}.log"

# =============================================================================
# 模型和配置
# =============================================================================
CONFIG_PATH=${CONFIG_PATH:-"${PROJECT_DIR}/examples/video_rl_qwen3.5/video_rl_v1_qwen3_5.yaml"}
MODEL_PATH=${MODEL_PATH:-"/path/to/Qwen3.5-2B"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3_5_2b_demo"}
SAVE_CHECKPOINT_PATH=${SAVE_CHECKPOINT_PATH:-"${PROJECT_DIR}/checkpoints/qwen3_5_video_rl/${EXPERIMENT_NAME}"}
FIND_LAST_CHECKPOINT=${FIND_LAST_CHECKPOINT:-false}

# Prompt模板和Reward函数路径
FORMAT_PROMPT=${FORMAT_PROMPT:-"examples/video_rl_qwen3.5/format_prompt/unified.jinja"}
REWARD_FUNCTION=${REWARD_FUNCTION:-"examples/video_rl_qwen3.5/video_v1.py:compute_score"}

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
    "${RAY_BIN}" stop --force 2>/dev/null || true
    sleep 3
}

wait_for_head() {
    local max_attempts=60
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if "${RAY_BIN}" status --address="${MASTER_ADDR}:${MASTER_PORT}" &>/dev/null; then
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
        connected_nodes=$("${RAY_BIN}" status 2>/dev/null | grep -c "node_" || echo "0")
        echo "已连接节点: $connected_nodes / $expected_nodes (尝试 $attempt/$max_attempts)"

        if [ "$connected_nodes" -ge "$expected_nodes" ]; then
            echo "所有节点已连接!"
            "${RAY_BIN}" status
            return 0
        fi

        attempt=$((attempt + 1))
        sleep 10
    done

    echo "错误: 未能等到所有节点连接"
    "${RAY_BIN}" status
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

    "${RAY_BIN}" start --head \
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

    "${PYTHON_BIN}" -m verl.trainer.main \
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
        trainer.find_last_checkpoint=${FIND_LAST_CHECKPOINT} \
        2>&1 | tee -a "$LOG_FILE"

    echo "训练完成!"
else
    cleanup_ray
    wait_for_head
    sleep 20

    "${RAY_BIN}" start \
        --address="${MASTER_ADDR}:${MASTER_PORT}" \
        --num-gpus=${NPROC_PER_NODE} \
        --disable-usage-stats \
        2>&1 | tee -a "$LOG_FILE"

    "${RAY_BIN}" status --address="${MASTER_ADDR}:${MASTER_PORT}"

    echo "Worker 节点待命中..."
    sleep inf
fi
