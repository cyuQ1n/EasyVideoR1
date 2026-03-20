# -*- coding: utf-8 -*-
"""
OCR Task Reward Function
"""

from typing import Any, Dict, List

from utils import extract_answer, preprocess_ground_truth

REWARD_NAME = "ocr"
REWARD_TYPE = "batch"


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate."""
    ref_words = (reference or "").split()
    hyp_words = (hypothesis or "").split()
    m, n = len(ref_words), len(hyp_words)

    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)

    return d[m][n] / max(1, m)


def accuracy_reward(response: str, ground_truth: str) -> float:
    ans = extract_answer(response)
    gt_ans = extract_answer(ground_truth)
    wer_score = wer(gt_ans.lower(), ans.lower())
    return max(0.0, min(1.0, 1.0 - wer_score))


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
