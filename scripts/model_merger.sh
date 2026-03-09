#!/bin/bash

# =============================================================================
# Model Merger Script
# Merge FSDP sharded checkpoint into a single HuggingFace model.
#
# Usage:
#   bash scripts/model_merger.sh
# =============================================================================

# Path configuration (modify these)
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CKPT_DIR=${CKPT_DIR:-"/path/to/checkpoints/experiment/global_step_xxx/actor"}
DST_MODEL_DIR=${DST_MODEL_DIR:-"/path/to/output/merged_model"}

# =============================================================================
cd "$PROJECT_DIR" || exit

# Merge model
python scripts/model_merger.py --local_dir "$CKPT_DIR"

# Create destination directory
mkdir -p "$DST_MODEL_DIR"

# Copy huggingface model
cp -r "$CKPT_DIR/huggingface" "$DST_MODEL_DIR"

echo "Done. Model copied to: $DST_MODEL_DIR"
