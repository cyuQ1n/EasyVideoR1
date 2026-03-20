# -*- coding: utf-8 -*-
"""
Grounding Task Reward Functions (Spatial/Temporal/Spatial-Temporal)
"""

from typing import Any, Dict, List

from utils import extract_answer, extract_json_from_text, iou_1d, iou_2d

REWARD_NAME = "grounding"
REWARD_TYPE = "batch"


def mean_iou_over_intersection(pred_boxes: Dict, gt_boxes: Dict) -> float:
    """Compute mean IoU over the shared frame keys."""
    if not isinstance(pred_boxes, dict) or not isinstance(gt_boxes, dict):
        return 0.0

    common_keys = set(pred_boxes.keys()) & set(gt_boxes.keys())
    if not common_keys:
        return 0.0

    ious = [iou_2d(pred_boxes[k], gt_boxes[k]) for k in common_keys]
    return sum(ious) / len(ious)


def spatial_reward(response: str, ground_truth: str) -> float:
    """Spatial grounding reward based on box IoU."""
    try:
        ans = extract_answer(response)
        pred = extract_json_from_text(ans)
        gt_obj = extract_json_from_text(extract_answer(ground_truth))

        if pred is None or gt_obj is None:
            return 0.0

        return iou_2d(pred.get("boxes", []), gt_obj.get("boxes", []))
    except:
        return 0.0


def temporal_reward(response: str, ground_truth: str) -> float:
    """Temporal grounding reward based on time IoU."""
    try:
        ans = extract_answer(response)
        pred = extract_json_from_text(ans)
        gt_obj = extract_json_from_text(extract_answer(ground_truth))

        if pred is None or gt_obj is None:
            return 0.0

        return iou_1d(pred.get("time", []), gt_obj.get("time", []))
    except:
        return 0.0


def spatial_temporal_reward(response: str, ground_truth: str) -> float:
    """Spatio-temporal grounding reward: 0.5 * tIoU + 0.5 * mIoU."""
    try:
        ans = extract_answer(response)
        pred = extract_json_from_text(ans)
        gt_obj = extract_json_from_text(extract_answer(ground_truth))

        if pred is None or gt_obj is None:
            return 0.0

        tiou = iou_1d(pred.get("time", []), gt_obj.get("time", []))
        miou = mean_iou_over_intersection(pred.get("boxes", {}), gt_obj.get("boxes", {}))

        return 0.5 * tiou + 0.5 * miou
    except:
        return 0.0


def spatial_compute_score(reward_inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    scores = []
    for inp in reward_inputs:
        acc = spatial_reward(inp.get("response", ""), inp.get("ground_truth", ""))
        scores.append({"overall": acc, "accuracy": acc, "format": 0.0, "length_penalty": 0.0})
    return scores


def temporal_compute_score(reward_inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    scores = []
    for inp in reward_inputs:
        acc = temporal_reward(inp.get("response", ""), inp.get("ground_truth", ""))
        scores.append({"overall": acc, "accuracy": acc, "format": 0.0, "length_penalty": 0.0})
    return scores


def spatial_temporal_compute_score(reward_inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    scores = []
    for inp in reward_inputs:
        acc = spatial_temporal_reward(inp.get("response", ""), inp.get("ground_truth", ""))
        scores.append({"overall": acc, "accuracy": acc, "format": 0.0, "length_penalty": 0.0})
    return scores
