# -*- coding: utf-8 -*-
"""
Numerical Task Reward Function
"""

from typing import Any, Dict, List, Optional


try:
    from .utils import extract_answer, preprocess_ground_truth
except ImportError:
    from utils import extract_answer, preprocess_ground_truth

REWARD_NAME = "numerical"
REWARD_TYPE = "batch"


def normalize_number(num_str: str) -> Optional[float]:
    try:
        return float((num_str or "").replace(",", ""))
    except Exception:
        return None


def accuracy_reward(response: str, ground_truth: str) -> float:
    ans = extract_answer(response)
    gt_ans = extract_answer(ground_truth)

    pred_num = normalize_number(ans)
    gt_num = normalize_number(gt_ans)

    if pred_num is None or gt_num is None:
        return 0.0

    return 1.0 if round(pred_num, 2) == round(gt_num, 2) else 0.0


def compute_score(reward_inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    scores = []
    for inp in reward_inputs:
        response = inp.get("response", "")
        ground_truth = preprocess_ground_truth(inp.get("ground_truth", ""))
        acc = accuracy_reward(response, ground_truth)
        scores.append(
            {
                "overall": acc,
                "accuracy": acc,
                "format": 0.0,
                "length_penalty": 0.0,
            }
        )
    return scores
