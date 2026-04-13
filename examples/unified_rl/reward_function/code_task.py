# -*- coding: utf-8 -*-
"""
Code Task Reward Function (Python/HTML/SVG)
"""

import re
from typing import Any, Dict, List

from utils import extract_answer


REWARD_NAME = "code"
REWARD_TYPE = "batch"


def extract_code_from_markdown(text: str) -> str:
    """Extract code from markdown code blocks."""
    if not text:
        return ""

    content = text
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()

    code_pattern = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)
    code_blocks = code_pattern.findall(content)

    if code_blocks:
        return "\n".join(code_blocks).strip()

    if match and content:
        return content

    global_blocks = code_pattern.findall(text)
    if global_blocks:
        return "\n".join(global_blocks).strip()

    return ""


def python_code_reward(response: str, ground_truth: str) -> float:
    """Validate Python code by checking syntactic correctness and structure similarity."""
    extracted_code = extract_code_from_markdown(response)
    if not extracted_code:
        return 0.0

    # Check if the code is syntactically valid Python.
    try:
        compile(extracted_code, "<string>", "exec")
    except SyntaxError:
        return 0.0

    gt_code = extract_code_from_markdown(ground_truth) or ground_truth.strip()
    if not gt_code:
        return 0.5  # Syntactically valid but no ground truth to compare

    # Structural similarity: compare defined function/class names and imports.
    def extract_identifiers(code: str) -> set:
        names = set()
        names.update(re.findall(r"^\s*def\s+(\w+)", code, re.MULTILINE))
        names.update(re.findall(r"^\s*class\s+(\w+)", code, re.MULTILINE))
        names.update(re.findall(r"^\s*import\s+(\w+)", code, re.MULTILINE))
        names.update(re.findall(r"^\s*from\s+(\w+)", code, re.MULTILINE))
        return names

    pred_ids = extract_identifiers(extracted_code)
    gt_ids = extract_identifiers(gt_code)

    if not gt_ids:
        return 0.5

    overlap = len(pred_ids & gt_ids) / len(gt_ids) if gt_ids else 0.0
    return max(0.0, min(1.0, overlap))


def svg_code_reward(response: str, ground_truth: str) -> float:
    """Validate SVG code."""
    ans = extract_answer(response)
    gt_ans = extract_answer(ground_truth)

    if not ans:
        return 0.0

    if "<svg" not in ans.lower() or "</svg>" not in ans.lower():
        return 0.0

    ans_clean = "".join(ans.split())
    gt_clean = "".join(gt_ans.split())

    if not gt_clean:
        return 1.0 if not ans_clean else 0.0

    gt_tags = set(re.findall(r"<(\w+)", gt_clean))
    ans_tags = set(re.findall(r"<(\w+)", ans_clean))

    if not gt_tags:
        return 0.0

    tag_match = len(gt_tags & ans_tags) / len(gt_tags)
    length_ratio = min(len(ans_clean), len(gt_clean)) / max(len(ans_clean), len(gt_clean))

    return max(0.0, min(1.0, 0.7 * tag_match + 0.3 * length_ratio))


def html_code_reward(response: str, ground_truth: str) -> float:
    """Validate HTML code."""
    ans = extract_answer(response)
    gt_ans = extract_answer(ground_truth)

    if not ans:
        return 0.0

    if not re.search(r"<(\w+)[^>]*>", ans, re.IGNORECASE):
        return 0.0

    ans_clean = "".join(ans.split())
    gt_clean = "".join(gt_ans.split())

    if not gt_clean:
        return 1.0 if not ans_clean else 0.0

    gt_tags = set(re.findall(r"<(\w+)", gt_clean.lower()))
    ans_tags = set(re.findall(r"<(\w+)", ans_clean.lower()))

    if not gt_tags:
        return 0.0

    tag_match = len(gt_tags & ans_tags) / len(gt_tags)
    length_ratio = min(len(ans_clean), len(gt_clean)) / max(len(ans_clean), len(gt_clean))

    return max(0.0, min(1.0, 0.7 * tag_match + 0.3 * length_ratio))


def compute_score(reward_inputs: List[Dict[str, Any]], code_type: str = "python", **kwargs) -> List[Dict[str, float]]:
    scores = []
    for inp in reward_inputs:
        response = inp.get("response", "")
        ground_truth = inp.get("ground_truth", "")
        problem_type = inp.get("problem_type", "").lower()

        if problem_type == "svg-code" or code_type == "svg":
            acc = svg_code_reward(response, ground_truth)
        elif problem_type == "html-code" or code_type == "html":
            acc = html_code_reward(response, ground_truth)
        else:
            acc = python_code_reward(response, ground_truth)

        scores.append(
            {
                "overall": acc,
                "accuracy": acc,
                "format": 0.0,
                "length_penalty": 0.0,
            }
        )
    return scores
