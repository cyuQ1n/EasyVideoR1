#!/bin/bash
################################################################################
# Video Data Preprocessing Script - EasyVideoR1
# Preprocesses training and evaluation videos for faster training.
################################################################################

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

VIDEO_FPS=${VIDEO_FPS:-2.0}
VIDEO_MAX_FRAMES=${VIDEO_MAX_FRAMES:-128}
VIDEO_MIN_PIXELS=${VIDEO_MIN_PIXELS:-3136}
VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-262144}
VIDEO_TOTAL_PIXELS=${VIDEO_TOTAL_PIXELS:-}
WORKERS=${WORKERS:-16}
SKIP_ERRORS=${SKIP_ERRORS:-0}
FORCE_REPROCESS=${FORCE_REPROCESS:-0}

TRAIN_JSON=${TRAIN_JSON:-"/path/to/your/train_data.json"}
TRAIN_OUTPUT_JSON=${TRAIN_OUTPUT_JSON:-"$PROJECT_DIR/train_data/train_data_preprocessed.json"}
TRAIN_IMAGE_DIR=${TRAIN_IMAGE_DIR:-}

EVAL_JSON=${EVAL_JSON:-"/path/to/your/eval_data.json"}
EVAL_OUTPUT_JSON=${EVAL_OUTPUT_JSON:-"$PROJECT_DIR/val_data/eval_data_preprocessed.json"}
EVAL_IMAGE_DIR=${EVAL_IMAGE_DIR:-}

SHARED_PREPROCESSED_DIR=${SHARED_PREPROCESSED_DIR:-"$PROJECT_DIR/preprocessed_videos"}

run_preprocess() {
    local input_json="$1"
    local output_json="$2"
    local image_dir="${3:-}"

    local cmd=(
        python3 scripts/preprocess_videos.py
        --input_file "$input_json"
        --output_dir "$SHARED_PREPROCESSED_DIR"
        --output_file "$output_json"
        --video_fps "$VIDEO_FPS"
        --video_max_frames "$VIDEO_MAX_FRAMES"
        --video_min_pixels "$VIDEO_MIN_PIXELS"
        --video_max_pixels "$VIDEO_MAX_PIXELS"
        --workers "$WORKERS"
    )

    if [ -n "$VIDEO_TOTAL_PIXELS" ]; then
        cmd+=(--video_total_pixels "$VIDEO_TOTAL_PIXELS")
    fi
    if [ -n "$image_dir" ]; then
        cmd+=(--image_dir "$image_dir")
    fi
    if [ "$SKIP_ERRORS" = "1" ]; then
        cmd+=(--skip_errors)
    fi

    "${cmd[@]}"
}

maybe_preprocess() {
    local label="$1"
    local input_json="$2"
    local output_json="$3"
    local image_dir="${4:-}"

    if [ ! -f "$input_json" ]; then
        echo -e "${RED}Error: ${label} data file not found: ${input_json}${NC}"
        exit 1
    fi

    if [ -f "$output_json" ] && [ "$FORCE_REPROCESS" != "1" ]; then
        echo -e "${YELLOW}${label} data already preprocessed: ${output_json}${NC}"
        read -p "Re-preprocess ${label}? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Skipping ${label} preprocessing${NC}"
            return
        fi
    fi

    echo -e "${BLUE}Starting ${label} preprocessing...${NC}"
    run_preprocess "$input_json" "$output_json" "$image_dir"
    echo -e "${GREEN}${label} preprocessing complete${NC}"
}

echo "================================================================================"
echo "Video Data Preprocessing - EasyVideoR1"
echo "================================================================================"
echo ""
echo -e "${BLUE}Preprocessing parameters:${NC}"
echo "  video_fps: ${VIDEO_FPS}"
echo "  video_max_frames: ${VIDEO_MAX_FRAMES}"
echo "  video_min_pixels: ${VIDEO_MIN_PIXELS}"
echo "  video_max_pixels: ${VIDEO_MAX_PIXELS}"
echo "  video_total_pixels: ${VIDEO_TOTAL_PIXELS:-<unset>}"
echo "  workers: ${WORKERS}"
echo "  skip_errors: ${SKIP_ERRORS}"
echo ""
echo -e "${BLUE}Data paths:${NC}"
echo "  Training data: ${TRAIN_JSON}"
echo "  Training output: ${TRAIN_OUTPUT_JSON}"
echo "  Training image_dir: ${TRAIN_IMAGE_DIR:-<unset>}"
echo "  Evaluation data: ${EVAL_JSON}"
echo "  Evaluation output: ${EVAL_OUTPUT_JSON}"
echo "  Evaluation image_dir: ${EVAL_IMAGE_DIR:-<unset>}"
echo "  Preprocessed output: ${SHARED_PREPROCESSED_DIR}"
echo ""

mkdir -p "$SHARED_PREPROCESSED_DIR"
mkdir -p "$(dirname "$TRAIN_OUTPUT_JSON")"
mkdir -p "$(dirname "$EVAL_OUTPUT_JSON")"

echo "================================================================================"
echo "Step 1/2: Preprocessing training videos"
echo "================================================================================"
maybe_preprocess "training" "$TRAIN_JSON" "$TRAIN_OUTPUT_JSON" "$TRAIN_IMAGE_DIR"

echo ""
echo "================================================================================"
echo "Step 2/2: Preprocessing evaluation videos"
echo "================================================================================"
maybe_preprocess "evaluation" "$EVAL_JSON" "$EVAL_OUTPUT_JSON" "$EVAL_IMAGE_DIR"

echo ""
echo "================================================================================"
echo -e "${GREEN}Preprocessing complete!${NC}"
echo "================================================================================"

PREPROCESSED_COUNT=$(find "$SHARED_PREPROCESSED_DIR" -name '*.pt' 2>/dev/null | wc -l)
PREPROCESSED_SIZE=$(du -sh "$SHARED_PREPROCESSED_DIR" 2>/dev/null | cut -f1)

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
echo "  data:"
echo "    train_files: ${TRAIN_OUTPUT_JSON}"
echo "    val_files: ${EVAL_OUTPUT_JSON}"
echo "    use_preprocessed_videos: true"
echo "    video_source_mode: prefer_preprocessed"
echo "    preprocessed_video_dir: ${SHARED_PREPROCESSED_DIR}"
if [ -n "$VIDEO_TOTAL_PIXELS" ]; then
    echo "    video_total_pixels: ${VIDEO_TOTAL_PIXELS}"
fi
echo ""
echo -e "${YELLOW}Notes:${NC}"
echo "  1. Training and evaluation share the preprocessed directory"
echo "  2. Set TRAIN_IMAGE_DIR / EVAL_IMAGE_DIR if dataset video paths are relative"
echo "  3. Preprocessing parameters must match the training config, including video_total_pixels when used"
echo "================================================================================"
