#!/bin/bash
################################################################################
# Video Data Preprocessing Script - EasyVideoR1
# Preprocesses training and evaluation videos for faster training.
#
# Usage:
#   bash scripts/preprocess_video_data.sh
################################################################################

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "Video Data Preprocessing - EasyVideoR1"
echo "================================================================================"

# ==================== Configuration ====================
# Project root directory
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

# Video processing parameters (must match training config!)
VIDEO_FPS=2.0
VIDEO_MAX_FRAMES=128
VIDEO_MIN_PIXELS=3136       # 56*56
VIDEO_MAX_PIXELS=262144     # 512*512
WORKERS=16

# Training data configuration
TRAIN_JSON=${TRAIN_JSON:-"/path/to/your/train_data.json"}
TRAIN_OUTPUT_JSON=${TRAIN_OUTPUT_JSON:-"$PROJECT_DIR/train_data/train_data_preprocessed.json"}

# Evaluation data configuration
EVAL_JSON=${EVAL_JSON:-"/path/to/your/eval_data.json"}
EVAL_OUTPUT_JSON=${EVAL_OUTPUT_JSON:-"$PROJECT_DIR/val_data/eval_data_preprocessed.json"}

# Shared preprocessed video directory
SHARED_PREPROCESSED_DIR=${SHARED_PREPROCESSED_DIR:-"$PROJECT_DIR/preprocessed_videos"}

echo ""
echo -e "${BLUE}Preprocessing parameters:${NC}"
echo "  video_fps: ${VIDEO_FPS}"
echo "  video_max_frames: ${VIDEO_MAX_FRAMES}"
echo "  video_min_pixels: ${VIDEO_MIN_PIXELS}"
echo "  video_max_pixels: ${VIDEO_MAX_PIXELS}"
echo "  workers: ${WORKERS}"
echo ""
echo -e "${BLUE}Data paths:${NC}"
echo "  Training data: ${TRAIN_JSON}"
echo "  Evaluation data: ${EVAL_JSON}"
echo "  Preprocessed output: ${SHARED_PREPROCESSED_DIR}"
echo ""

# Check input files
if [ ! -f "${TRAIN_JSON}" ]; then
    echo -e "${RED}Error: Training data file not found: ${TRAIN_JSON}${NC}"
    exit 1
fi

if [ ! -f "${EVAL_JSON}" ]; then
    echo -e "${RED}Error: Evaluation data file not found: ${EVAL_JSON}${NC}"
    exit 1
fi

# Create preprocessed directory
mkdir -p "${SHARED_PREPROCESSED_DIR}"

# ==================== Step 1: Preprocess training videos ====================
echo "================================================================================"
echo "Step 1/2: Preprocessing training videos"
echo "================================================================================"

if [ -f "${TRAIN_OUTPUT_JSON}" ]; then
    echo -e "${YELLOW}Training data already preprocessed: ${TRAIN_OUTPUT_JSON}${NC}"
    read -p "Re-preprocess? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Skipping training data preprocessing${NC}"
    else
        echo -e "${BLUE}Starting training data preprocessing...${NC}"
        python3 scripts/preprocess_videos.py \
            --input_file "${TRAIN_JSON}" \
            --output_dir "${SHARED_PREPROCESSED_DIR}" \
            --output_file "${TRAIN_OUTPUT_JSON}" \
            --video_fps ${VIDEO_FPS} \
            --video_max_frames ${VIDEO_MAX_FRAMES} \
            --video_min_pixels ${VIDEO_MIN_PIXELS} \
            --video_max_pixels ${VIDEO_MAX_PIXELS} \
            --workers ${WORKERS}
        echo -e "${GREEN}Training data preprocessing complete${NC}"
    fi
else
    echo -e "${BLUE}Starting training data preprocessing...${NC}"
    python3 scripts/preprocess_videos.py \
        --input_file "${TRAIN_JSON}" \
        --output_dir "${SHARED_PREPROCESSED_DIR}" \
        --output_file "${TRAIN_OUTPUT_JSON}" \
        --video_fps ${VIDEO_FPS} \
        --video_max_frames ${VIDEO_MAX_FRAMES} \
        --video_min_pixels ${VIDEO_MIN_PIXELS} \
        --video_max_pixels ${VIDEO_MAX_PIXELS} \
        --workers ${WORKERS}
    echo -e "${GREEN}Training data preprocessing complete${NC}"
fi

# ==================== Step 2: Preprocess evaluation videos ====================
echo ""
echo "================================================================================"
echo "Step 2/2: Preprocessing evaluation videos"
echo "================================================================================"

if [ -f "${EVAL_OUTPUT_JSON}" ]; then
    echo -e "${YELLOW}Evaluation data already preprocessed: ${EVAL_OUTPUT_JSON}${NC}"
    read -p "Re-preprocess? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Skipping evaluation data preprocessing${NC}"
    else
        echo -e "${BLUE}Starting evaluation data preprocessing...${NC}"
        python3 scripts/preprocess_videos.py \
            --input_file "${EVAL_JSON}" \
            --output_dir "${SHARED_PREPROCESSED_DIR}" \
            --output_file "${EVAL_OUTPUT_JSON}" \
            --video_fps ${VIDEO_FPS} \
            --video_max_frames ${VIDEO_MAX_FRAMES} \
            --video_min_pixels ${VIDEO_MIN_PIXELS} \
            --video_max_pixels ${VIDEO_MAX_PIXELS} \
            --workers ${WORKERS}
        echo -e "${GREEN}Evaluation data preprocessing complete${NC}"
    fi
else
    echo -e "${BLUE}Starting evaluation data preprocessing...${NC}"
    python3 scripts/preprocess_videos.py \
        --input_file "${EVAL_JSON}" \
        --output_dir "${SHARED_PREPROCESSED_DIR}" \
        --output_file "${EVAL_OUTPUT_JSON}" \
        --video_fps ${VIDEO_FPS} \
        --video_max_frames ${VIDEO_MAX_FRAMES} \
        --video_min_pixels ${VIDEO_MIN_PIXELS} \
        --video_max_pixels ${VIDEO_MAX_PIXELS} \
        --workers ${WORKERS}
    echo -e "${GREEN}Evaluation data preprocessing complete${NC}"
fi

# ==================== Done ====================
echo ""
echo "================================================================================"
echo -e "${GREEN}Preprocessing complete!${NC}"
echo "================================================================================"

# Count preprocessed files
PREPROCESSED_COUNT=$(find "${SHARED_PREPROCESSED_DIR}" -name "*.pt" 2>/dev/null | wc -l)
PREPROCESSED_SIZE=$(du -sh "${SHARED_PREPROCESSED_DIR}" 2>/dev/null | cut -f1)

echo ""
echo -e "${BLUE}Preprocessing statistics:${NC}"
echo "  Preprocessed files: ${PREPROCESSED_COUNT}"
echo "  Disk usage: ${PREPROCESSED_SIZE}"
echo ""

echo -e "${BLUE}Output files:${NC}"
echo "  Training data: ${TRAIN_OUTPUT_JSON}"
echo "  Evaluation data: ${EVAL_OUTPUT_JSON}"
echo "  Preprocessed dir: ${SHARED_PREPROCESSED_DIR}"
echo ""

echo -e "${BLUE}Next step - Update training config:${NC}"
echo "Edit your config YAML and set the following fields:"
echo ""
echo "  data:"
echo "    train_files: ${TRAIN_OUTPUT_JSON}"
echo "    val_files: ${EVAL_OUTPUT_JSON}"
echo "    use_preprocessed_videos: true"
echo "    preprocessed_video_dir: ${SHARED_PREPROCESSED_DIR}"
echo ""

echo -e "${YELLOW}Notes:${NC}"
echo "  1. Training and evaluation share the preprocessed directory"
echo "  2. Videos use absolute paths, no need to set image_dir"
echo "  3. Preprocessing parameters must match training config"
echo ""
echo "================================================================================"
