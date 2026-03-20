# -*- coding: utf-8 -*-
"""
Multiple Choice Task Reward Function
"""

from typing import Any, Dict, List

from mathruler.grader import grade_answer
from utils import extract_answer, parse_mcq, preprocess_ground_truth


REWARD_NAME = "multiple_choice"
REWARD_TYPE = "batch"


def accuracy_reward(response: str, ground_truth: str) -> float:
    ans = extract_answer(response)
    gt_ans = extract_answer(ground_truth)

    ans_letter = parse_mcq(ans)
    gt_letter = parse_mcq(gt_ans)

    if ans_letter and gt_letter:
        return 1.0 if ans_letter.upper() == gt_letter.upper() else 0.0

    return 1.0 if grade_answer(ans.strip(), gt_ans.strip()) else 0.0


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
