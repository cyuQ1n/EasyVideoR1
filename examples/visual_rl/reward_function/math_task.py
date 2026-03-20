# -*- coding: utf-8 -*-
"""
Math Task Reward Function
"""

import re
import signal
from typing import Any, Dict, List

from mathruler.grader import grade_answer


try:
    from .utils import extract_answer_math as extract_answer
    from .utils import preprocess_ground_truth, strip_math_string
except ImportError:
    from utils import extract_answer_math as extract_answer
    from utils import preprocess_ground_truth, strip_math_string


class _GradeTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _GradeTimeout()


def grade_answer_safe(pred: str, gt: str, timeout: int = 10) -> bool:
    """grade_answer with a timeout to prevent sympy from hanging or leaking memory."""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        return grade_answer(pred, gt)
    except _GradeTimeout:
        return False
    except Exception:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


REWARD_NAME = "math"
REWARD_TYPE = "batch"


def format_reward(response: str, thinking_tag: str = "thought") -> float:
    """Check whether the response follows the required thought-answer format."""
    pattern = re.compile(rf"<{thinking_tag}>.*</{thinking_tag}>.*<answer>.*</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def soft_overlong_punishment(
    response_length: int, max_response_length: int, overlong_buffer_length: int = 3072
) -> float:
    """
    Length penalty:
    Apply a linear penalty when the generated response exceeds max_response_length.
    This discourages reward hacking through excessively long outputs.
    """
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 0.0
    elif response_length <= max_response_length:
        return (expected_len - response_length) / overlong_buffer_length
    else:
        return -1.0


def math_equivalent(gt: str, pred: str) -> bool:
    """Enhanced mathematical-equivalence check."""
    gt_norm = strip_math_string(gt)
    pred_norm = strip_math_string(pred)

    if grade_answer_safe(pred, gt):
        return True
    if grade_answer_safe(pred_norm, gt_norm):
        return True

    return False


def accuracy_reward(response: str, ground_truth: str) -> float:
    ans = extract_answer(response)
    gt_ans = extract_answer(ground_truth) or (ground_truth or "")
    return 1.0 if math_equivalent(gt_ans, ans) else 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    max_response_length: int = 16384,
    format_weight: float = 0.1,
    overlong_penalty_factor: float = 0.1,
    thinking_tag: str = "thought",
    **kwargs,
) -> List[Dict[str, float]]:
    scores = []
    for inp in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", inp.get("response", ""))
        ground_truth = preprocess_ground_truth(inp.get("ground_truth", ""))
        response_length = inp.get("response_length", len(response))

        format_score = format_reward(response, thinking_tag=thinking_tag)
        accuracy_score = accuracy_reward(response, ground_truth)
        len_penalty = soft_overlong_punishment(response_length, max_response_length)

        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score
                + format_weight * format_score
                + overlong_penalty_factor * len_penalty,
                "format": format_score,
                "accuracy": accuracy_score,
                "length_penalty": len_penalty,
            }
        )

    return scores
