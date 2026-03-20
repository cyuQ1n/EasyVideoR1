# -*- coding: utf-8 -*-
"""
LLaVA Critic Task Reward Function
"""

from typing import Any, Dict, List

from utils import extract_answer, preprocess_ground_truth

REWARD_NAME = "llava"
REWARD_TYPE = "batch"


def normalize_label(text: str) -> int:
    """Normalize critic labels.

    Supported formats:
    - Response 1 / Response A / 1 / A / first  → 1
    - Response 2 / Response B / 2 / B / second → 2
    - Tie / Two responses are equally good     → 0
    """
    text = text.lower().strip()

    # Format: Response 1, Response A, 1, A, first
    if text in ['response 1', 'response a', '1', 'a', 'first']:
        return 1
    if 'response 1' in text or 'response a' in text:
        return 1

    # Format: Response 2, Response B, 2, B, second
    if text in ['response 2', 'response b', '2', 'b', 'second']:
        return 2
    if 'response 2' in text or 'response b' in text:
        return 2

    # Format: Tie, equally good, both, equal
    if text == 'tie':
        return 0
    if 'equally' in text or 'both' in text or 'equal' in text or 'tie' in text:
        return 0

    # Also handle descriptions that mention first/second.
    if 'first' in text:
        return 1
    if 'second' in text:
        return 2

    return -1  # Invalid label


def accuracy_reward(response: str, ground_truth: str) -> float:
    ans = extract_answer(response).lower().strip()
    gt_ans = extract_answer(ground_truth).lower().strip()

    pred_label = normalize_label(ans)
    gt_label = normalize_label(gt_ans)

    if pred_label >= 0 and pred_label == gt_label:
        return 1.0
    return 0.0


def compute_score(reward_inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    scores = []
    for inp in reward_inputs:
        response = inp.get("response", "")
        ground_truth = preprocess_ground_truth(inp.get("ground_truth", ""))
        acc = accuracy_reward(response, ground_truth)
        scores.append({
            "overall": acc,
            "accuracy": acc,
            "format": 0.0,
            "length_penalty": 0.0,
        })
    return scores
