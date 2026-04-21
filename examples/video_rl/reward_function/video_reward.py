# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""
Accuracy + Format reward function for Video RL (0328 exp_3).

Based on exp_2's video_v1_strict.py but simplified to only use accuracy and
format rewards. Reasoning and length penalty components are removed to avoid
reward hacking.

  overall = accuracy_weight * accuracy + format_weight * format
"""

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from mathruler.grader import grade_answer
from rouge_score import rouge_scorer


# Reward function metadata
REWARD_NAME = "video_v1_acc_fmt"
REWARD_TYPE = "batch"

ANSWER_CAPTURE_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
OPEN_THINK_BLOCK_PATTERN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
WORD_PATTERN = re.compile(r"\b\w+\b")
LEADING_RESPONSE_PREFIX_PATTERN = re.compile(r"^\s*(?:\[[^\]]+\]|output\s*:|response\s*:)\s*", re.IGNORECASE)

FINAL_STATEMENT_PATTERNS = [
    re.compile(
        r"(?im)^\s*(?:therefore|so|thus|hence|finally|in\s+(?:summary|conclusion))[,\s]*"
        r"(?:the\s+)?(?:final\s+|best\s+|correct\s+)?answer\s+"
        r"(?:is|would\s+be|should\s+be)\s*[:：]?\s*(.+?)\s*$"
    ),
    re.compile(
        r"(?im)^\s*(?:the\s+)?(?:final\s+|best\s+|correct\s+)?answer\s+"
        r"(?:is|would\s+be|should\s+be)\s*[:：]?\s*(.+?)\s*$"
    ),
    re.compile(r"(?im)^\s*final\s+answer\s*[:：]\s*(.+?)\s*$"),
]

OPEN_ENDED_TYPES = {"open-ended", "free-form", "video qa", "video description"}
NUMERIC_TYPES = {"numerical"}
REGRESSION_TYPES = {"regression"}
BOOLEAN_TYPES = {"boolean", "binary classification"}
BOXED_TYPES = {"multiple choice", "numerical", "regression", "boolean", "binary classification", "math"}
JSON_TYPES = {"temporal grounding", "spatial grounding", "spatial-temporal grounding", "tracking"}
OPTION_SET = set("ABCDEFGHIJ")

WORD2NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}


def extract_tag_answer(text: str) -> Optional[str]:
    """Extract content inside <answer>...</answer>."""
    if not isinstance(text, str):
        return None
    matches = list(ANSWER_CAPTURE_PATTERN.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_tag_answer_with_index(text: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract content inside <answer>...</answer> and return its start index."""
    if not isinstance(text, str):
        return None, None
    matches = list(ANSWER_CAPTURE_PATTERN.finditer(text))
    if not matches:
        return None, None
    match = matches[-1]
    return match.group(1).strip(), match.start()


def remove_think_block(text: str) -> str:
    """Drop complete or truncated <think> blocks when possible."""
    if not isinstance(text, str):
        return ""
    cleaned = THINK_BLOCK_PATTERN.sub("", text).strip()
    if cleaned:
        return cleaned
    cleaned = OPEN_THINK_BLOCK_PATTERN.sub("", text).strip()
    return cleaned if cleaned else text.strip()


def strip_leading_response_prefix(text: str) -> str:
    """Strip common logging prefixes such as `[output]` from the start of a response."""
    if not isinstance(text, str):
        return ""

    stripped = text
    while True:
        updated = LEADING_RESPONSE_PREFIX_PATTERN.sub("", stripped, count=1)
        if updated == stripped:
            break
        stripped = updated
    return stripped


def get_after_think(text: str) -> str:
    """Return the content after </think> when it exists."""
    if not isinstance(text, str):
        return ""
    match = re.search(r"</think>\s*(.*?)$", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_boxed(text: str) -> str:
    """Extract the content of the last \\boxed{...} span."""
    if not isinstance(text, str):
        return ""

    results: List[str] = []
    i = 0
    while i < len(text):
        if text[i : i + 7] == "\\boxed{":
            i += 7
            brace_level = 1
            start = i
            while i < len(text) and brace_level > 0:
                if text[i] == "{":
                    brace_level += 1
                elif text[i] == "}":
                    brace_level -= 1
                i += 1
            if brace_level == 0:
                results.append(text[start : i - 1])
        else:
            i += 1
    return results[-1].strip() if results else ""


def strip_terminal_punctuation(text: str) -> str:
    """Strip one trailing sentence punctuation mark from the extracted answer."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if text.endswith(("。", "；", ";")):
        return text[:-1].rstrip()
    if text.endswith(".") and not text.endswith("..."):
        return text[:-1].rstrip()
    return text


def extract_final_statement(response: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract the last natural final-answer statement and its start index."""
    if not isinstance(response, str):
        return None, None

    best_match = None
    for pattern in FINAL_STATEMENT_PATTERNS:
        for match in pattern.finditer(response):
            if best_match is None or match.start() >= best_match.start():
                best_match = match

    if best_match is None:
        return None, None

    return strip_terminal_punctuation(best_match.group(1)), best_match.start()


def extract_last_nonempty_line(response: str) -> str:
    """Return the last non-empty line."""
    if not isinstance(response, str):
        return ""
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def word_to_number(text: str) -> Optional[float]:
    """Convert simple English number words into digits."""
    if not isinstance(text, str):
        return None
    normalized = text.lower().strip().replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in WORD2NUM:
        return float(WORD2NUM[normalized])
    parts = normalized.split()
    if len(parts) == 2 and parts[0] in WORD2NUM and parts[1] in WORD2NUM:
        return float(WORD2NUM[parts[0]] + WORD2NUM[parts[1]])
    return None


def normalize_number(num_str: str) -> Optional[float]:
    """Extract numeric values from text, fractions, percentages, or number words."""
    if num_str is None:
        return None

    text = num_str.strip()
    if not text:
        return None

    frac_match = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", text)
    if frac_match:
        try:
            numerator = float(frac_match.group(1))
            denominator = float(frac_match.group(2))
            if denominator != 0:
                return numerator / denominator
        except Exception:
            pass

    pct_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", text)
    if pct_match:
        try:
            return float(pct_match.group(1)) / 100.0
        except Exception:
            pass

    num_match = re.search(r"(-?\d[\d,]*\.?\d*)", text)
    if num_match:
        try:
            return float(num_match.group(1).replace(",", ""))
        except Exception:
            pass

    word_number = word_to_number(text)
    if word_number is not None:
        return word_number

    for token in text.lower().split():
        word_number = word_to_number(token)
        if word_number is not None:
            return word_number

    return None


def _normalize_boolean_token(token: str) -> str:
    if token == "true":
        return "yes"
    if token == "false":
        return "no"
    return token


def parse_boolean_answer(raw: str) -> str:
    """Extract a yes/no style answer from raw output."""
    if not isinstance(raw, str):
        return ""

    candidates = [raw]
    extracted = extract_open_answer(raw)
    if extracted and extracted != raw:
        candidates.insert(0, extracted)

    for candidate in candidates:
        matches = list(re.finditer(r"\b(yes|no|true|false)\b", candidate, re.IGNORECASE))
        if matches:
            return _normalize_boolean_token(matches[-1].group(1).lower())

    return ""


def compute_rouge_score(reference: str, hypothesis: str) -> float:
    """Compute the average of rouge1 / rouge2 / rougeL F1."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference or "", hypothesis or "")
    return (scores["rouge1"].fmeasure + scores["rouge2"].fmeasure + scores["rougeL"].fmeasure) / 3.0


def mean_relative_accuracy(pred_val: float, target_val: float, start: float = 0.5, end: float = 0.95, interval: float = 0.05) -> float:
    """Compute mean relative accuracy without torch dependency."""
    eps = 1e-8
    relative_error = abs(pred_val - target_val) / (abs(target_val) + eps)
    count = 0
    total = 0
    threshold = start
    while threshold <= end + interval / 2:
        total += 1
        if relative_error < (1 - threshold):
            count += 1
        threshold += interval
    return count / total if total > 0 else 0.0


def _is_list_of_numbers(x: Any, n: Optional[int] = None) -> bool:
    """Check whether x is a list of numbers."""
    if not isinstance(x, list):
        return False
    if n is not None and len(x) != n:
        return False
    try:
        for value in x:
            float(value)
        return True
    except Exception:
        return False


def iou_1d(pred: List[float], gt: List[float]) -> float:
    """Compute 1D IoU for temporal intervals."""
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
    """Compute 2D IoU for boxes."""
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


def mean_iou_over_intersection(pred_boxes: Dict[str, List[float]], gt_boxes: Dict[str, List[float]]) -> float:
    """Compute mean IoU over common frame ids."""
    if not isinstance(pred_boxes, dict) or not isinstance(gt_boxes, dict):
        return 0.0
    common = [key for key in pred_boxes.keys() if key in gt_boxes]
    if not common:
        return 0.0
    values = [iou_2d(pred_boxes[key], gt_boxes[key]) for key in common]
    return sum(values) / len(values) if values else 0.0


def _load_json(text: str) -> Optional[Any]:
    """Safely parse JSON."""
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_json_object(text: str) -> str:
    """Extract the last valid JSON object from text."""
    if not isinstance(text, str):
        return ""

    candidates: List[str] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            start = i
            depth = 1
            i += 1
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            if depth == 0:
                candidate = text[start:i]
                if _load_json(candidate) is not None:
                    candidates.append(candidate)
        else:
            i += 1

    return candidates[-1].strip() if candidates else ""


def extract_mc_answer(raw: str) -> str:
    """Robustly extract a multiple-choice option letter from raw model output."""
    if not isinstance(raw, str):
        return ""

    candidates: List[str] = []

    tagged = extract_tag_answer(raw)
    if tagged:
        candidates.append(tagged)

    final_answer, _ = extract_final_statement(raw)
    if final_answer:
        candidates.append(final_answer)

    after_think = get_after_think(raw)
    if after_think:
        candidates.append(after_think)
        after_think_final, _ = extract_final_statement(after_think)
        if after_think_final:
            candidates.append(after_think_final)

    cleaned = remove_think_block(raw)
    if cleaned:
        candidates.append(cleaned)

    candidates.append(raw)

    for candidate in candidates:
        letter = _extract_mc_from_text(candidate)
        if letter:
            return letter

    return ""


def _extract_mc_from_text(text: str) -> str:
    """Try multiple patterns to recover a single option letter."""
    if not isinstance(text, str):
        return ""

    content = text.strip()
    if not content:
        return ""

    boxed = extract_boxed(content)
    if boxed:
        boxed_letter = _extract_mc_from_text(boxed)
        if boxed_letter:
            return boxed_letter

    if len(content) == 1 and content.upper() in OPTION_SET:
        return content.upper()

    match = re.match(r"^([A-Ja-j])\s*[.)\]:：。]", content)
    if match:
        return match.group(1).upper()

    if len(content) < 20:
        match = re.match(r"^([A-Ja-j])\s", content)
        if match:
            return match.group(1).upper()

    match = re.match(r"^([A-Ja-j])\s*$", content, re.MULTILINE)
    if match:
        return match.group(1).upper()

    answer_patterns = [
        (
            r"(?:Therefore|So|Thus|Hence|Finally|In\s+(?:summary|conclusion))[,\s]*"
            r"(?:the\s+)?(?:final\s+|best\s+|correct\s+)?answer\s+"
            r"(?:is|would\s+be|should\s+be)[:\s]*"
            r"(?:option\s+|choice\s+)?\(?([A-Ja-j])\)?\b"
        ),
        (
            r"(?:the\s+)?(?:final\s+|best\s+|correct\s+)?answer\s+"
            r"(?:is|would\s+be|should\s+be)[:\s]*"
            r"(?:option\s+|choice\s+)?\(?([A-Ja-j])\)?\b"
        ),
        r"answer\s*[:：]\s*(?:option\s+|choice\s+)?\(?([A-Ja-j])\)?\b",
        r"答案\s*[是为选：:]\s*\(?([A-Ja-j])\)?\b",
        r"(?:应该)?选(?:择)?\s*\(?([A-Ja-j])\)?\b",
    ]
    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()

    match = re.search(
        r"(?:I\s+)?(?:choose|select|pick|go\s+with)\s+(?:option\s+|choice\s+)?\(?([A-Ja-j])\)?\b",
        content,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    match = re.search(
        r"option\s+\(?([A-Ja-j])\)?\s+is\s+(?:the\s+)?(?:correct|right|best)",
        content,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    match = re.search(
        r"\(?([A-Ja-j])\)?\s+is\s+(?:the\s+)?(?:correct|right|best)\s+(?:answer|option|choice)",
        content,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    match = re.search(r"\*\*\s*([A-Ja-j])\s*\*\*", content)
    if match:
        return match.group(1).upper()

    lines = content.splitlines()
    for line in reversed(lines[-5:]):
        line = line.strip()
        if not line:
            continue
        if len(line) == 1 and line.upper() in OPTION_SET:
            return line.upper()
        match = re.match(r"^([A-Ja-j])\s*[.)\]:：。]", line)
        if match:
            return match.group(1).upper()
        if len(line) < 40:
            match = re.search(r"\b([A-Ja-j])\s*[.):]?\s*$", line)
            if match and match.group(1).upper() in OPTION_SET:
                return match.group(1).upper()

    letters = re.findall(r"\b([A-J])\b", content)
    unique_letters = {letter for letter in letters if letter in OPTION_SET}
    if len(unique_letters) == 1 and len(content) <= 12:
        return unique_letters.pop()

    return ""


def extract_open_answer(raw: str) -> str:
    """Extract an open-form answer from tagged, final-statement, or raw output."""
    if not isinstance(raw, str):
        return ""

    tagged = extract_tag_answer(raw)
    if tagged:
        return tagged.strip()

    final_answer, _ = extract_final_statement(raw)
    if final_answer:
        boxed = extract_boxed(final_answer)
        return strip_terminal_punctuation(boxed or final_answer)

    boxed = extract_boxed(raw)
    if boxed:
        return strip_terminal_punctuation(boxed)

    after_think = get_after_think(raw)
    if after_think:
        after_final, _ = extract_final_statement(after_think)
        if after_final:
            return strip_terminal_punctuation(after_final)
        after_boxed = extract_boxed(after_think)
        if after_boxed:
            return strip_terminal_punctuation(after_boxed)
        return after_think.strip()

    cleaned = remove_think_block(raw)
    if cleaned:
        return cleaned.strip()

    return raw.strip()


def extract_numerical_answer(raw: str) -> str:
    """Extract a numerical answer, returning the normalized numeric string when possible."""
    text = extract_open_answer(raw)
    number = normalize_number(text)
    if number is not None:
        return str(number)
    return text


def _extract_time_range_from_text(text: str) -> Optional[List[float]]:
    """Extract a temporal range from natural language when JSON is missing."""
    if not isinstance(text, str):
        return None

    patterns = [
        r"(?:in|from|between|at)\s+(\d+(?:\.\d+)?)\s*[-–~to]+\s*(\d+(?:\.\d+)?)\s*(?:seconds?|s\b)",
        r"(\d+(?:\.\d+)?)\s*[-–~]\s*(\d+(?:\.\d+)?)\s*(?:seconds?|s\b)",
        r"(\d+(?:\.\d+)?)\s*s?\s+to\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s\b)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return [float(match.group(1)), float(match.group(2))]
    return None


def extract_grounding_answer(raw: str, problem_type: str) -> str:
    """Extract a grounding answer, preferring JSON and falling back to time spans when possible."""
    if not isinstance(raw, str):
        return ""

    tagged = extract_tag_answer(raw)
    if tagged and _load_json(tagged) is not None:
        return tagged

    candidates: List[str] = []

    final_answer, _ = extract_final_statement(raw)
    if final_answer:
        candidates.append(final_answer)

    after_think = get_after_think(raw)
    if after_think:
        candidates.append(after_think)
        after_final, _ = extract_final_statement(after_think)
        if after_final:
            candidates.append(after_final)

    cleaned = remove_think_block(raw)
    if cleaned:
        candidates.append(cleaned)

    candidates.append(raw)

    for candidate in candidates:
        if _load_json(candidate) is not None:
            return candidate.strip()
        json_object = extract_json_object(candidate)
        if json_object:
            return json_object

    ptype = (problem_type or "").lower().strip()
    if ptype in {"temporal grounding", "spatial-temporal grounding"}:
        for candidate in candidates:
            time_range = _extract_time_range_from_text(candidate)
            if time_range is not None:
                if ptype == "temporal grounding":
                    return json.dumps({"time": time_range})
                return json.dumps({"time": time_range, "boxes": {}})

    return ""


def extract_answer(response: str, problem_type: str) -> Optional[str]:
    """Dispatch answer extraction by problem type."""
    ptype = (problem_type or "").lower().strip()
    if ptype == "multiple choice":
        return extract_mc_answer(response)
    if ptype in NUMERIC_TYPES or ptype in REGRESSION_TYPES:
        return extract_numerical_answer(response)
    if ptype in JSON_TYPES:
        return extract_grounding_answer(response, ptype)
    return extract_open_answer(response)


def _validate_json_final_answer(content: str, problem_type: str) -> bool:
    """Validate the JSON shape expected by a grounding task."""
    parsed = _load_json(content)
    ptype = (problem_type or "").lower().strip()
    if not isinstance(parsed, dict):
        return False
    if ptype == "temporal grounding":
        return _is_list_of_numbers(parsed.get("time"), 2)
    if ptype == "spatial grounding":
        return _is_list_of_numbers(parsed.get("boxes"), 4)
    if ptype == "spatial-temporal grounding":
        has_time = _is_list_of_numbers(parsed.get("time"), 2)
        has_boxes = isinstance(parsed.get("boxes"), dict)
        return has_time and has_boxes
    if ptype == "tracking":
        return isinstance(parsed.get("boxes"), dict)
    return False


def format_reward(response: str, problem_type: str) -> float:
    """
    Check whether the response contains a valid <answer>...</answer> tag
    with type-appropriate content.

    Returns:
      1.0 — valid <answer> tag with correct content type
      0.5 — <answer> tag present but content type invalid (e.g. non-letter for MC)
      0.0 — no <answer> tag at all
    """
    ptype = (problem_type or "").lower().strip()

    tag_content = extract_tag_answer(response)
    if not tag_content:
        return 0.0

    # Tag exists — check if the content is type-appropriate
    if ptype == "multiple choice":
        return 1.0 if _extract_mc_from_text(tag_content) else 0.5
    if ptype in NUMERIC_TYPES or ptype in REGRESSION_TYPES:
        return 1.0 if normalize_number(tag_content) is not None else 0.5
    if ptype in BOOLEAN_TYPES:
        return 1.0 if parse_boolean_answer(tag_content) else 0.5
    if ptype in JSON_TYPES:
        return 1.0 if _validate_json_final_answer(tag_content, ptype) else 0.5
    # For open-ended / other types, any non-empty content is valid
    return 1.0 if bool(tag_content.strip()) else 0.5


def accuracy_reward(response: str, ground_truth: str, data_type: str, problem_type: str) -> float:
    """Compute the task accuracy / similarity reward in [0, 1]."""
    try:
        ptype = (problem_type or "").lower().strip()
        gt = extract_tag_answer(ground_truth) or (ground_truth or "")

        if ptype == "multiple choice":
            pred_letter = extract_mc_answer(response)
            gt_letter = extract_mc_answer(gt)
            if pred_letter and gt_letter:
                return 1.0 if pred_letter == gt_letter else 0.0
            pred_answer = extract_answer(response, ptype) or ""
            return 1.0 if grade_answer(pred_answer.strip(), gt.strip()) else 0.0

        if ptype in NUMERIC_TYPES:
            gt_num = normalize_number(gt)
            pred_num = normalize_number(extract_answer(response, ptype) or "")
            if gt_num is None or pred_num is None:
                return 0.0
            if round(gt_num, 4) == round(pred_num, 4):
                return 1.0
            return 1.0 if abs(gt_num - pred_num) < 0.015 else 0.0

        if ptype in REGRESSION_TYPES:
            gt_num = normalize_number(gt)
            pred_num = normalize_number(extract_answer(response, ptype) or "")
            if gt_num is None or pred_num is None:
                return 0.0
            return mean_relative_accuracy(pred_num, gt_num)

        if ptype in BOOLEAN_TYPES:
            pred = parse_boolean_answer(response)
            gold = parse_boolean_answer(gt) or gt.strip().lower()
            return 1.0 if pred and pred == gold else 0.0

        if ptype == "temporal grounding":
            pred = _load_json(extract_answer(response, ptype) or "")
            gt_json = _load_json(gt)
            if not isinstance(pred, dict) or not isinstance(gt_json, dict):
                return 0.0
            return iou_1d(pred.get("time"), gt_json.get("time"))

        if ptype == "spatial-temporal grounding":
            pred = _load_json(extract_answer(response, ptype) or "")
            gt_json = _load_json(gt)
            if not isinstance(pred, dict) or not isinstance(gt_json, dict):
                return 0.0
            tiou = iou_1d(pred.get("time"), gt_json.get("time"))
            pred_boxes = pred.get("boxes")
            gt_boxes = gt_json.get("boxes")
            miou_inter = mean_iou_over_intersection(pred_boxes, gt_boxes) if isinstance(pred_boxes, dict) and isinstance(gt_boxes, dict) else 0.0
            return 0.5 * tiou + 0.5 * miou_inter

        if ptype == "spatial grounding":
            pred = _load_json(extract_answer(response, ptype) or "")
            gt_json = _load_json(gt)
            if not isinstance(pred, dict) or not isinstance(gt_json, dict):
                return 0.0
            return iou_2d(pred.get("boxes"), gt_json.get("boxes"))

        if ptype == "tracking":
            pred = _load_json(extract_answer(response, ptype) or "")
            gt_json = _load_json(gt)
            if not isinstance(pred, dict) or not isinstance(gt_json, dict):
                return 0.0
            pred_boxes = pred.get("boxes")
            gt_boxes = gt_json.get("boxes")
            if not isinstance(pred_boxes, dict) or not isinstance(gt_boxes, dict):
                return 0.0
            return mean_iou_over_intersection(pred_boxes, gt_boxes)

        if ptype in OPEN_ENDED_TYPES:
            pred_text = extract_open_answer(response)
            return max(0.0, min(1.0, compute_rouge_score(gt, pred_text)))

        pred_text = extract_open_answer(response)
        return 1.0 if grade_answer(pred_text.strip(), gt.strip()) else 0.0

    except Exception:
        return 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    accuracy_weight: float = 0.90,
    format_weight: float = 0.10,
    debug_sample_rate: float = 0.1,
    **kwargs: Any,
) -> List[Dict[str, float]]:
    """
    Batch interface for computing rewards (accuracy + format only).

    Simplified from exp_2's reasoning-first reward to avoid hacking:
    - No reasoning_weight (was 0.15): models tend to game word-count-based reasoning rewards
    - No length_penalty_factor (was 0.05): removes incentive to optimize output length

    Reward formula:
      overall = accuracy_weight * accuracy + format_weight * format

    Expected keys in each reward input:
      - response
      - ground_truth
      - data_type
      - problem_type
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    total_weight = accuracy_weight + format_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got {total_weight:.4f} "
            f"(acc={accuracy_weight}, fmt={format_weight})"
        )

    results: List[Dict[str, float]] = []

    for item in reward_inputs:
        try:
            raw_response = item.get("response", "") or ""
            raw_response = strip_leading_response_prefix(raw_response)
            response = re.sub(r"[ \t]*(<|>|/)[ \t]*", r"\1", raw_response)
            data_type = item.get("data_type", "") or ""
            problem_type = item.get("problem_type", "") or ""
            raw_gt = item.get("ground_truth", "") or ""
            gt_extracted = extract_tag_answer(raw_gt) or raw_gt

            format_score = format_reward(response, problem_type)
            accuracy = accuracy_reward(response, gt_extracted, data_type, problem_type)

            overall = accuracy_weight * accuracy + format_weight * format_score

            results.append(
                {
                    "overall": float(overall),
                    "accuracy": float(accuracy),
                    "format": float(format_score),
                }
            )
        except Exception:
            results.append(
                {
                    "overall": 0.0,
                    "accuracy": 0.0,
                    "format": 0.0,
                }
            )

    if debug_sample_rate > 0 and random.random() < debug_sample_rate:
        for index, item in enumerate(reward_inputs):
            print(f"[video_v1_acc_fmt] type: {item.get('problem_type', '')}")
            print(f"[video_v1_acc_fmt] extracted_answer: {extract_answer(item.get('response', ''), item.get('problem_type', '') or '')}")
            print(f"[video_v1_acc_fmt] score: {results[index]}")

    return results
