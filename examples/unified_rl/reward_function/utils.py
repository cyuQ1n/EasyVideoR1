# -*- coding: utf-8 -*-
"""
Shared utility functions.
"""

import json
import re
from typing import Dict, List, Optional

OPTION_SET = set("ABCDEFGHIJ")


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
    Supports nested JSON, markdown code blocks, and plain text.
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

    # Use brace-depth counting to extract nested JSON objects.
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
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        candidates.append(candidate)
                except Exception:
                    pass
        else:
            i += 1

    if candidates:
        # Return the last valid JSON object (most likely the final answer).
        return json.loads(candidates[-1])

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
    string = string.replace("°", "")
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


def _extract_between_tag(text: str, tag: str) -> str:
    """Extract content between a specific XML-like tag."""
    if not isinstance(text, str):
        return ""
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _remove_reasoning_blocks(text: str) -> str:
    """Remove <think>/<thought> blocks while preserving the remaining answer text."""
    if not isinstance(text, str):
        return ""

    cleaned = re.sub(r"<(?:think|thought)>.*?</(?:think|thought)>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if cleaned:
        return cleaned

    cleaned = re.sub(r"<(?:think|thought)>.*$", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return cleaned if cleaned else text.strip()


def _get_after_reasoning(text: str) -> str:
    """Return the content after a closing </think> or </thought> tag."""
    if not isinstance(text, str):
        return ""
    match = re.search(r"</(?:think|thought)>\s*(.*?)$", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _normalize_option_text(text: str) -> str:
    """Normalize option text to support text-to-letter matching."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = text.strip()
    text = re.sub(r"^(?:option|choice)\s+[A-Ja-j]\s*[.)\]:：。-]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[A-Ja-j]\s*[.)\]:：。-]\s*", "", text)
    text = re.sub(
        r"^(?:therefore|so|thus|hence|finally|in\s+(?:summary|conclusion))[,\s]*(?:the\s+)?"
        r"(?:final\s+|best\s+|correct\s+)?answer\s+(?:is|would\s+be|should\s+be)[:\s]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:the\s+)?(?:final\s+|best\s+|correct\s+)?answer\s+(?:is|would\s+be|should\s+be)[:\s]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:i\s+)?(?:choose|select|pick|go\s+with)\s+(?:option\s+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[\"'“”‘’`]", "", text)
    return text.strip(" \t\r\n.,;:!?")


def _build_option_map(options: Optional[List[str]]) -> Dict[str, str]:
    """Build a mapping from option letter to normalized option text."""
    option_map: Dict[str, str] = {}
    if not options:
        return option_map

    for idx, option in enumerate(options):
        if not isinstance(option, str):
            continue
        match = re.match(r"^\s*([A-Ja-j])\s*[.)\]:：。-]\s*(.*)$", option, re.DOTALL)
        if match:
            letter = match.group(1).upper()
            option_text = match.group(2)
        else:
            if idx >= len(OPTION_SET):
                continue
            letter = chr(ord("A") + idx)
            option_text = option
        normalized = _normalize_option_text(option_text)
        if normalized:
            option_map[letter] = normalized
    return option_map


def _match_option_text_to_letter(text: str, options: Optional[List[str]]) -> str:
    """Try to recover the option letter from option text."""
    option_map = _build_option_map(options)
    if not option_map:
        return ""

    candidates = []
    normalized = _normalize_option_text(text)
    if normalized:
        candidates.append(normalized)

    for pattern in [
        r"(?:the\s+)?(?:final\s+|best\s+|correct\s+)?answer\s+(?:is|would\s+be|should\s+be)[:\s]+(.+)",
        r"answer\s*[:：]\s*(.+)",
        r"(?:i\s+)?(?:choose|select|pick|go\s+with)\s+(?:option\s+)?(.+)",
    ]:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            candidate = _normalize_option_text(match.group(1))
            if candidate:
                candidates.append(candidate)

    for candidate in candidates:
        for letter, option_text in option_map.items():
            if candidate == option_text:
                return letter
            if candidate.startswith(option_text) or candidate.endswith(option_text):
                return letter

    return ""


def _extract_mc_from_text(text: str) -> str:
    """Robustly extract a multiple-choice letter from free-form text."""
    s = text.strip()
    if not s:
        return ""

    if len(s) == 1 and s.upper() in OPTION_SET:
        return s.upper()

    match = re.match(r"^([A-Ja-j])\s*[.)\]:：。]", s)
    if match:
        return match.group(1).upper()

    if len(s) < 20:
        match = re.match(r"^([A-Ja-j])\s", s)
        if match:
            return match.group(1).upper()

    match = re.match(r"^([A-Ja-j])\s*$", s, re.MULTILINE)
    if match:
        return match.group(1).upper()

    answer_patterns = [
        r"(?:Therefore|So|Thus|Hence|Finally|In\s+(?:summary|conclusion))[,\s]*(?:the\s+)?"
        r"(?:final\s+|best\s+|correct\s+)?answer\s+(?:is|would\s+be|should\s+be)[:\s]*\(?([A-Ja-j])\)?\b",
        r"(?:the\s+)?(?:final\s+|best\s+|correct\s+)?answer\s+(?:is|would\s+be|should\s+be)[:\s]*\(?([A-Ja-j])\)?\b",
        r"answer\s*[:：]\s*\(?([A-Ja-j])\)?\b",
        r"答案\s*[是为选：:]\s*\(?([A-Ja-j])\)?\b",
        r"(?:应该)?选(?:择)?\s*\(?([A-Ja-j])\)?\b",
    ]
    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, s, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()

    match = re.search(
        r"(?:I\s+)?(?:choose|select|pick|go\s+with)\s+(?:option\s+)?\(?([A-Ja-j])\)?\b",
        s,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    match = re.search(r"option\s+\(?([A-Ja-j])\)?\s+is\s+(?:the\s+)?(?:correct|right|best)", s, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(
        r"\(?([A-Ja-j])\)?\s+is\s+(?:the\s+)?(?:correct|right|best)\s+(?:answer|option|choice)",
        s,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    match = re.search(r"\*\*\s*([A-Ja-j])\s*\*\*", s)
    if match:
        return match.group(1).upper()

    lines = s.split("\n")
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
            match = re.search(r"\b([A-J])\s*[.):]?\s*$", line.upper())
            if match:
                return match.group(1).upper()

    found = {match.group(1).upper() for match in re.finditer(r"\b([A-Ja-j])\b", s)}
    if len(found) == 1:
        return found.pop()

    return ""


def parse_mcq(predict_str: str, options: Optional[List[str]] = None) -> str:
    """Extract a multiple-choice answer from free-form output."""
    if not predict_str or predict_str.strip() == "":
        return ""

    raw = predict_str.strip()
    candidates = []

    answer_tag = _extract_between_tag(raw, "answer")
    if answer_tag:
        candidates.append(answer_tag)

    after_reasoning = _get_after_reasoning(raw)
    if after_reasoning:
        candidates.append(after_reasoning)

    cleaned = _remove_reasoning_blocks(raw)
    if cleaned:
        candidates.append(cleaned)

    candidates.append(raw)

    seen = set()
    deduped_candidates = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        letter = _extract_mc_from_text(candidate)
        if letter:
            return letter

    for candidate in deduped_candidates:
        letter = _match_option_text_to_letter(candidate, options)
        if letter:
            return letter

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


# ============================================================================
# Unified format reward
# ============================================================================

# Task type sets for format validation
_NUMERIC_TYPES = {"numerical", "number", "regression"}
_BOOLEAN_TYPES = {"boolean", "binary classification"}
_JSON_TYPES = {"temporal grounding", "spatial grounding", "spatial-temporal grounding", "tracking"}


def _is_list_of_numbers(x, n: Optional[int] = None) -> bool:
    """Check whether x is a list of numbers."""
    if not isinstance(x, list):
        return False
    if n is not None and len(x) != n:
        return False
    try:
        for v in x:
            float(v)
        return True
    except Exception:
        return False


def _validate_json_answer(content: str, problem_type: str) -> bool:
    """Validate the JSON shape expected by a grounding / tracking task."""
    try:
        parsed = json.loads(content)
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False
    ptype = problem_type.lower().strip()
    if ptype == "temporal grounding":
        return _is_list_of_numbers(parsed.get("time"), 2)
    if ptype == "spatial grounding":
        return _is_list_of_numbers(parsed.get("boxes"), 4)
    if ptype == "spatial-temporal grounding":
        return _is_list_of_numbers(parsed.get("time"), 2) and isinstance(parsed.get("boxes"), dict)
    if ptype == "tracking":
        return isinstance(parsed.get("boxes"), dict)
    return False


def format_reward(response: str, problem_type: str) -> float:
    """
    Check whether the response contains a valid <answer>...</answer> tag
    with type-appropriate content.

    Returns:
      1.0 -- valid <answer> tag with correct content type
      0.5 -- <answer> tag present but content type invalid
      0.0 -- no <answer> tag at all
    """
    ptype = (problem_type or "").lower().strip()

    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if not match:
        return 0.0

    tag_content = match.group(1).strip()
    if not tag_content:
        return 0.5

    if ptype == "multiple choice":
        return 1.0 if _extract_mc_from_text(tag_content) else 0.5
    if ptype in _NUMERIC_TYPES:
        try:
            float(tag_content.replace(",", ""))
            return 1.0
        except Exception:
            return 0.5
    if ptype in _BOOLEAN_TYPES:
        token = tag_content.lower()
        return 1.0 if token in ("yes", "no", "true", "false") else 0.5
    if ptype in _JSON_TYPES:
        return 1.0 if _validate_json_answer(tag_content, ptype) else 0.5
    # For open-ended / code / other types, any non-empty content is valid.
    return 1.0
