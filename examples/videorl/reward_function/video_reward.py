# -*- coding: utf-8 -*-
"""
Reward function for Video RL Training (with format reward & length penalty)
Supported task types:
  - multiple choice
  - temporal grounding
  - spatial-temporal grounding
  - numerical
  - open-ended

Reward composition:
  overall = (1 - format_weight - length_penalty_factor) * accuracy
           + format_weight * format
           + length_penalty_factor * length_penalty
"""
import re
import json
import random
from typing import Any, Dict, List, Optional

from rouge_score import rouge_scorer
from mathruler.grader import grade_answer

# Reward function metadata
REWARD_NAME = "video_reward"
REWARD_TYPE = "batch"


# -------------------------
# Answer extraction pattern
# -------------------------
ANSWER_CAPTURE_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL
)

# -------------------------
# Format check pattern
# -------------------------
FORMAT_PATTERN = re.compile(
    r"<thought>.*</thought>.*<answer>.*</answer>",
    re.DOTALL
)


# -------------------------
# Utilities
# -------------------------
def extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    if not isinstance(text, str):
        return None
    m = ANSWER_CAPTURE_PATTERN.search(text)
    return m.group(1).strip() if m else None


def normalize_number(num_str: str) -> Optional[float]:
    """Convert string to float, handling commas."""
    try:
        return float((num_str or "").replace(",", ""))
    except Exception:
        return None


def compute_rouge_score(reference: str, hypothesis: str) -> float:
    """Compute average ROUGE score (rouge1, rouge2, rougeL)."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference or "", hypothesis or "")
    return (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3.0


# -------------------------
# Format reward
# -------------------------
def format_reward(response: str) -> float:
    """检查格式: <thought>...</thought>...<answer>...</answer>"""
    format_match = re.fullmatch(FORMAT_PATTERN, response)
    return 1.0 if format_match else 0.0


# -------------------------
# Length penalty (bidirectional)
# -------------------------
def soft_length_penalty(response_length: int,
                        max_response_length: int,
                        min_expected_length: int = 128,
                        overlong_buffer_length: int = 3072) -> float:
    """
    双向长度惩罚：鼓励模型输出不长不短
    - 太短 (< min_expected_length): 线性惩罚，从 -1 到 0
    - 正常范围: 0（无惩罚）
    - 接近上限 (expected_len ~ max_response_length): 线性惩罚，从 0 到 -1
    - 超长 (> max_response_length): -1
    """
    # 过短惩罚：length=0 → -1, length=min_expected → 0
    if response_length < min_expected_length:
        if min_expected_length <= 0:
            return 0.0
        return max(-1.0, response_length / min_expected_length - 1.0)

    # 过长惩罚
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 0.0
    elif response_length <= max_response_length:
        return (expected_len - response_length) / overlong_buffer_length
    else:
        return -1.0


# -------------------------
# IoU helpers
# -------------------------
def _is_list_of_numbers(x, n=None) -> bool:
    """Check if x is a list of numbers with optional length constraint."""
    if not isinstance(x, list):
        return False
    if n is not None and len(x) != n:
        return False
    try:
        for v in x:
            float(v)
        return True
    except Exception:
        return False


def iou_1d(pred: List[float], gt: List[float]) -> float:
    """
    Compute 1D IoU for temporal grounding.
    pred, gt: [start, end] time intervals
    Returns IoU ∈ [0, 1]
    """
    if not _is_list_of_numbers(pred, 2) or not _is_list_of_numbers(gt, 2):
        return 0.0
    try:
        s1, e1 = float(pred[0]), float(pred[1])
        s2, e2 = float(gt[0]), float(gt[1])
    except Exception:
        return 0.0
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 1e-12 else 0.0


def iou_2d(box1: List[float], box2: List[float]) -> float:
    """
    Compute 2D IoU for spatial grounding.
    box1, box2: [x1, y1, x2, y2] bounding boxes
    Returns IoU ∈ [0, 1]
    """
    if not _is_list_of_numbers(box1, 4) or not _is_list_of_numbers(box2, 4):
        return 0.0
    try:
        x1, y1, x2, y2 = map(float, box1)
        X1, Y1, X2, Y2 = map(float, box2)
    except Exception:
        return 0.0
    inter_x1, inter_y1 = max(x1, X1), max(y1, Y1)
    inter_x2, inter_y2 = min(x2, X2), min(y2, Y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 1e-12 else 0.0


def mean_iou_over_intersection(pred_boxes: Dict[str, List[float]],
                                gt_boxes: Dict[str, List[float]]) -> float:
    """
    Compute mean IoU over intersection of frame keys.
    For spatial-temporal grounding: IoU averaged only over common frames.
    pred_boxes, gt_boxes: {frame_str: [x1, y1, x2, y2]}
    """
    if not isinstance(pred_boxes, dict) or not isinstance(gt_boxes, dict):
        return 0.0
    common = [k for k in pred_boxes.keys() if k in gt_boxes]
    if not common:
        return 0.0
    vals = [iou_2d(pred_boxes[k], gt_boxes[k]) for k in common]
    return sum(vals) / len(vals) if vals else 0.0


# -------------------------
# JSON parsing helper
# -------------------------
def _load_json(s: str) -> Optional[Any]:
    """Safely load JSON string."""
    try:
        return json.loads(s)
    except Exception:
        return None


# -------------------------
# Accuracy reward function
# -------------------------
def accuracy_reward(response: str,
                    ground_truth: str,
                    data_type: str,
                    problem_type: str) -> float:
    """
    Compute accuracy reward ∈ [0, 1] based on problem type.

    Supported problem types:
      - multiple choice: exact match
      - numerical: numeric comparison (2 decimal places)
      - temporal grounding: 1D IoU
      - spatial-temporal grounding: 0.5 * tIoU + 0.5 * mIoU
      - open-ended: ROUGE score
    """
    try:
        ans = extract_answer(response) or response.strip()
        ptype = (problem_type or "").lower()
        # Training data may store labels as <answer>...</answer>; normalize them
        # before task-specific grading so numerical/JSON tasks remain parseable.
        gt = extract_answer(ground_truth) or (ground_truth or "")

        # ------ Multiple Choice ------
        if ptype == "multiple choice":
            return 1.0 if grade_answer(ans.strip(), gt.strip()) else 0.0

        # ------ Numerical ------
        if ptype == "numerical":
            gt_num = normalize_number(gt)
            pr_num = normalize_number(ans)
            if gt_num is not None and pr_num is not None:
                return 1.0 if round(gt_num, 2) == round(pr_num, 2) else 0.0
            return 0.0

        # ------ Temporal Grounding ------
        # Answer format: {"time": [start, end]}
        if ptype == "temporal grounding":
            pred = _load_json(ans)
            gtj = _load_json(gt)
            if not isinstance(pred, dict) or not isinstance(gtj, dict):
                return 0.0
            return iou_1d(pred.get("time"), gtj.get("time"))

        # ------ Spatial-Temporal Grounding ------
        # Answer format: {"time": [start, end], "boxes": {"frame_id": [x1, y1, x2, y2], ...}}
        if ptype == "spatial-temporal grounding":
            pred = _load_json(ans)
            gtj = _load_json(gt)
            if not isinstance(pred, dict) or not isinstance(gtj, dict):
                return 0.0
            # Temporal IoU
            tiou = iou_1d(pred.get("time"), gtj.get("time"))
            # Spatial IoU (mean over intersection frames)
            pboxes = pred.get("boxes")
            gboxes = gtj.get("boxes")
            if not isinstance(pboxes, dict) or not isinstance(gboxes, dict):
                miou_inter = 0.0
            else:
                miou_inter = mean_iou_over_intersection(pboxes, gboxes)
            # Combined score: 0.5 * temporal + 0.5 * spatial
            return 0.5 * tiou + 0.5 * miou_inter

        # ------ Open-ended ------
        if ptype == "open-ended":
            return max(0.0, min(1.0, compute_rouge_score(gt, ans)))

        # ------ Unknown type ------
        return 0.0

    except Exception:
        return 0.0


# -------------------------
# Public API
# -------------------------
def compute_score(
    reward_inputs: List[Dict[str, Any]],
    max_response_length: int = 4096,
    format_weight: float = 0.1,
    length_penalty_factor: float = 0.0,
    min_expected_length: int = 128,
    overlong_buffer_length: int = 3072,
    **kwargs
) -> List[Dict[str, float]]:
    """
    Batch interface for computing rewards.

    Each input item should contain:
        {
            "response": str,          # Model response
            "response_length": int,   # Token length of response
            "ground_truth": str,      # Ground truth (may contain <answer> tags)
            "data_type": str,         # "video"
            "problem_type": str,      # One of the supported types
        }

    Returns: List of dicts with keys {overall, accuracy, format, length_penalty}
        overall = (1-fw-lp) * accuracy + fw * format + lp * length_penalty

    Args:
        max_response_length: 最大允许响应长度（token数）
        format_weight: 格式奖励权重
        length_penalty_factor: 长度惩罚权重
        min_expected_length: 最短期望响应长度（低于此值有惩罚）
        overlong_buffer_length: 过长缓冲区长度
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    results: List[Dict[str, float]] = []

    for idx, item in enumerate(reward_inputs):
        try:
            # Normalize tag whitespaces
            raw_response = item.get("response", "") or ""
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", raw_response)

            data_type = item.get("data_type", "") or ""
            problem_type = item.get("problem_type", "") or ""
            response_length = item.get("response_length", len(response))

            # Extract ground truth (may be wrapped in <answer> tags)
            raw_gt = item.get("ground_truth", "") or ""
            gt_extracted = extract_answer(raw_gt) or raw_gt

            # Compute sub-scores
            a_score = accuracy_reward(response, gt_extracted, data_type, problem_type)
            f_score = format_reward(response)
            l_penalty = soft_length_penalty(
                response_length, max_response_length,
                min_expected_length=min_expected_length,
                overlong_buffer_length=overlong_buffer_length,
            )

            # Weighted overall
            overall = (
                (1 - format_weight - length_penalty_factor) * a_score
                + format_weight * f_score
                + length_penalty_factor * l_penalty
            )

            results.append({
                "overall": float(overall),
                "accuracy": float(a_score),
                "format": float(f_score),
                "length_penalty": float(l_penalty),
            })

        except Exception:
            results.append({
                "overall": 0.0,
                "accuracy": 0.0,
                "format": 0.0,
                "length_penalty": 0.0,
            })

    # Debug logging (10% sample rate)
    if random.random() < 0.1:
        for idx, item in enumerate(reward_inputs):
            raw_gt = item.get("ground_truth", "") or ""
            gt_display = extract_answer(raw_gt) or raw_gt
            print(f"[video_v1] type: {item.get('problem_type', '')}")
            print(f"[video_v1] gt: {gt_display}")
            print(f"[video_v1] ans: {extract_answer(item.get('response', ''))}")
            print(f"[video_v1] score: {results[idx]}")

    return results
