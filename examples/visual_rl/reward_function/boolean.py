# -*- coding: utf-8 -*-
"""
Boolean Task Reward Function
"""

from typing import Any, Dict, List

from utils import extract_answer, preprocess_ground_truth


REWARD_NAME = "boolean"
REWARD_TYPE = "batch"


def accuracy_reward(response: str, ground_truth: str) -> float:
    ans = extract_answer(response).lower().strip()
    gt_ans = extract_answer(ground_truth).lower().strip()

    ans_norm = "yes" if ans in ["yes", "true", "1"] else ("no" if ans in ["no", "false", "0"] else ans)
    gt_norm = "yes" if gt_ans in ["yes", "true", "1"] else ("no" if gt_ans in ["no", "false", "0"] else gt_ans)

    return 1.0 if ans_norm == gt_norm else 0.0


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
