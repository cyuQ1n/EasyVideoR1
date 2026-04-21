# -*- coding: utf-8 -*-
"""
Open-ended Task Reward Function
"""

from typing import Any, Dict, List

from rouge_score import rouge_scorer
from utils import extract_answer, preprocess_ground_truth


REWARD_NAME = "open_ended"
REWARD_TYPE = "batch"


def compute_rouge_score(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference or "", hypothesis or "")
    return (scores["rouge1"].fmeasure + scores["rouge2"].fmeasure + scores["rougeL"].fmeasure) / 3.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    ans = extract_answer(response)
    gt_ans = extract_answer(ground_truth)

    if not ans or not gt_ans:
        return 0.0

    # Use exact matching for short answers.
    if len(gt_ans) <= 10:
        if ans.lower().strip() == gt_ans.lower().strip():
            return 1.0

    # Use ROUGE for longer answers.
    rouge_f1 = compute_rouge_score(gt_ans, ans)
    return max(0.0, min(1.0, rouge_f1))


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
