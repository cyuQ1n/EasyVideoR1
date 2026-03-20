# -*- coding: utf-8 -*-
"""
Shared utility functions.
"""

import json
import re
from typing import Dict, List, Optional


def preprocess_ground_truth(gt: str) -> str:
    """
    Preprocess ground truth text.
    - Remove wrapping $ or $$ markers
    - Preserve \\boxed{}
    """
    if not isinstance(gt, str):
        return ""
    gt = gt.strip()
    if gt.startswith("$$") and gt.endswith("$$"):
        gt = gt[2:-2].strip()
    elif gt.startswith("$") and gt.endswith("$"):
        gt = gt[1:-1].strip()
    return gt


def normalize_response(response: str) -> str:
    """Fix common formatting issues."""
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
    return response


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the answer using multiple fallback strategies.
    """
    if not isinstance(text, str):
        return None

    # Strategy 1: <answer> tag
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Strategy 2: \\boxed{...}
    boxed = extract_boxed(text)
    if boxed:
        return boxed

    # Strategy 3: the full response
    return text.strip()


def extract_answer_math(text: str) -> Optional[str]:
    """
    Extract a math answer using multiple fallback strategies.
    """
    if not isinstance(text, str):
        return None

    # Strategy 1: <answer> tag
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Strategy 2: \\boxed{...}
    boxed = extract_boxed(text)
    if boxed:
        return boxed

    # Strategy 3: return an empty string
    return ""


def extract_boxed(text: str) -> str:
    """Extract the contents of \\boxed{...}, using the last match."""
    results = []
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
    return results[-1] if results else ""


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract a JSON object from text.
    Also supports extraction from markdown code blocks.
    """
    if not text:
        return None

    # Try direct parsing first.
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # Try extracting from markdown code blocks.
    code_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
    matches = code_pattern.findall(text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except Exception:
            continue

    # Try finding a JSON object pattern.
    json_pattern = re.compile(r"\{[^{}]*\}")
    matches = json_pattern.findall(text)
    for match in reversed(matches):  # Search from the end.
        try:
            return json.loads(match)
        except Exception:
            continue

    return None


def strip_math_string(string: str) -> str:
    """Normalize a math answer string."""
    if not string:
        return ""

    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = re.sub(r"\\text\{\s*[^}]*\s*\}", "", string)
    string = re.sub(r"\\mbox\{\s*[^}]*\s*\}", "", string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")

    if string.startswith("."):
        string = "0" + string

    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    if len(string.split("\\approx")) == 2:
        string = string.split("\\approx")[-1]

    string = string.replace(" ", "")
    return string


def parse_mcq(predict_str: str) -> str:
    """Extract a multiple-choice answer using multiple matching patterns."""
    if not predict_str or predict_str.strip() == "":
        return ""

    response = predict_str.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.strip(char)
    response = " " + response + " "

    all_choices = ["A", "B", "C", "D", "E", "F", "G", "H"]
    candidates = []

    # Multiple matching patterns.
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append((choice, response.rfind(f"({choice})"), "parentheses"))
        if f"{choice}." in response:
            candidates.append((choice, response.rfind(f"{choice}."), "period"))
        if f"{choice}:" in response:
            candidates.append((choice, response.rfind(f"{choice}:"), "colon"))
        if f"{choice})" in response:
            candidates.append((choice, response.rfind(f"{choice})"), "right_paren"))
        if f" {choice} " in response:
            candidates.append((choice, response.rfind(f" {choice} "), "space"))

    # Answer-phrase matching.
    answer_phrases = ["the answer is", "answer is", "the correct answer is", "correct answer is"]
    for phrase in answer_phrases:
        if phrase in response.lower():
            phrase_end = response.lower().find(phrase) + len(phrase)
            after_phrase = response[phrase_end:]
            for choice in all_choices:
                for pattern in [f" {choice} ", f" {choice}.", f"({choice})"]:
                    if pattern in after_phrase.upper():
                        choice_pos = phrase_end + after_phrase.upper().find(pattern)
                        candidates.append((choice, choice_pos, "phrase"))
                        break

    # Match at the start or end.
    for choice in all_choices:
        if response.strip().upper().startswith(choice):
            candidates.append((choice, 0, "start"))
        if response.strip().upper().endswith(choice):
            candidates.append((choice, len(response) - 1, "end"))

    if not candidates:
        for choice in all_choices:
            if choice in response.upper():
                candidates.append((choice, response.upper().rfind(choice), "fallback"))

    if candidates:
        format_priority = {
            "start": 10,
            "end": 9,
            "phrase": 7,
            "parentheses": 6,
            "period": 5,
            "colon": 4,
            "right_paren": 3,
            "space": 2,
            "fallback": 0,
        }
        candidates.sort(key=lambda x: (format_priority.get(x[2], 0), x[1]), reverse=True)
        return candidates[0][0]

    return ""


def iou_1d(pred: List[float], gt: List[float]) -> float:
    """1D IoU for temporal intervals."""
    try:
        if not isinstance(pred, list) or len(pred) != 2:
            return 0.0
        if not isinstance(gt, list) or len(gt) != 2:
            return 0.0
        s1, e1 = float(pred[0]), float(pred[1])
        s2, e2 = float(gt[0]), float(gt[1])
        inter = max(0.0, min(e1, e2) - max(s1, s2))
        union = max(e1, e2) - min(s1, s2)
        return inter / union if union > 1e-12 else 0.0
    except Exception:
        return 0.0


def iou_2d(box1: List[float], box2: List[float]) -> float:
    """2D IoU for bounding boxes."""
    try:
        if not isinstance(box1, list) or len(box1) != 4:
            return 0.0
        if not isinstance(box2, list) or len(box2) != 4:
            return 0.0
        x1, y1, x2, y2 = map(float, box1)
        X1, Y1, X2, Y2 = map(float, box2)
        inter_x1, inter_y1 = max(x1, X1), max(y1, Y1)
        inter_x2, inter_y2 = min(x2, X2), min(y2, Y2)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area2 = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
        union = area1 + area2 - inter_area
        return inter_area / union if union > 1e-12 else 0.0
    except Exception:
        return 0.0
