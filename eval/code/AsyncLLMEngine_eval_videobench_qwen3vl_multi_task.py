"""
Qwen3-VL 视频评测脚本 (AsyncLLMEngine 真流水线版本)

核心优化: 使用 vLLM 的 AsyncLLMEngine 低级 API 实现真正的流水线
    - 请求通过 add_request() 持续流入引擎，不阻塞
    - 数据加载和推理完全并行
    - Prefill 和 Decode 真正交错进行
    - 消除了 batch 间的"冷启动"问题

与 LLM.generate() 的区别:
    - LLM.generate(): 同步阻塞，等所有请求完成才返回
    - AsyncLLMEngine: 异步非阻塞，请求可以持续流入，结果持续流出

运行模式:
    --mode auto       : (默认) 自动检测缓存，有缓存直接评测，无缓存则预处理+评测
    --mode preprocess : 只预处理视频到缓存
    --mode eval       : 只评测 (需要已有缓存)
    --mode both       : 强制预处理+评测

用法示例:
    python AsyncLLMEngine_eval_videobench_qwen3vl_ycx.py \
        --model_path /path/to/model --datasets lvbench
"""

import os
import re
import json
import argparse
import hashlib
import time
import asyncio
import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import multiprocessing as mp
import logging

os.environ['MODEL_SEQ_LEN'] = '224000'
logging.basicConfig(level=logging.INFO)

import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoProcessor, AutoTokenizer

# 可选依赖
try:
    from safetensors.torch import save_file
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("Warning: safetensors not installed, using numpy format for cache")

os.environ["DECORD_EOF_RETRY_MAX"] = "20480"

DEFAULT_QWEN25_UTILS_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "qwen_vl_utils-0.0.8",
)


def str2bool(v):
    """解析布尔值参数"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (true/false, yes/no, 1/0)')


# ============== Prompt 模板 ==============
QUESTION_TEMPLATE_qwen25 = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', "
    "'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your "
    "final answer between the <answer> and </answer> tags."
)

TYPE_TEMPLATE_qwen25 = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "open-ended": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "temporal grounding": (
        " Please provide the time span in seconds as JSON within the <answer>...</answer> tags. "
        "Example: <answer>{\"time\": [12.3, 25.7]}</answer>"
    ),
    "spatial-temporal grounding": (
        " Please provide the time span in seconds and bounding boxes as JSON within the <answer>...</answer> tags. "
        "Example: <answer>{\"time\": [8.125, 13.483], \"boxes\": {\"9\": [317, 422, 582, 997]}}</answer>"
    ),
}

QUESTION_TEMPLATE_qwen3 = """ Select the best answer to the following multiple-choice question based on the video.
Respond with only the letter (A, B, C, or D) of the correct option.
Question: {Question}
""".strip()

QUESTION_TEMPLATE_qwen3_thinking = """ You should watch and learn the video content. Then apply what you learned to select the best answer to the following multiple-choice question based on the video.
Begin by explaining your reasoning process briefly.
Conclude by stating the final answer using the following format: "Therefore, the final answer is: $LETTER" (without quotes), where $LETTER must be one of the options.
Question: {Question}
""".strip()

QUESTION_TEMPLATE_qwen3_open = """ Answer the following question based on the video.
Question: {Question}
""".strip()

# 通用 thinking 模式问题模板 (适用于所有非 MC 任务类型，来自 VideoEval-JD)
QUESTION_TEMPLATE_qwen3_thinking_generic = """ You should watch and learn the video content. Then apply what you learned to answer the following question based on the video.
Begin by explaining your reasoning process briefly.
Question: {Question}
""".strip()

TYPE_TEMPLATE_qwen3 = {
    "multiple choice": "\nThe best answer is: ",
    "numerical": "\nThe best answer is: ",
    "OCR": "\nThe best answer is: ",
    "free-form": "\nThe best answer is: ",
    "open-ended": "\nThe best answer is: ",
    "regression": "\nThe best answer is: ",
    "thinking_multiple_choice": "Please reason step-by-step, identify relevant visual content, analyze key timestamps and clues, and then provide the final answer.",
    "temporal grounding": (
        "\nPlease provide the time span in seconds as JSON within the <answer>...</answer> tags.\n"
        "Example:\n<answer>{\"time\": [12.3, 25.7]}</answer>\n"
    ),
    "spatial-temporal grounding": (
        "\nPlease provide the time span in seconds and bounding boxes as JSON within the <answer>...</answer> tags.\n"
        "You MUST output one bounding box for every integer second within the given time span (inclusive).\n"
        "Example:\n"
        "<answer>{\"time\": [8.125, 13.483], \"boxes\": {\"9\": [317, 422, 582, 997], "
        "\"10\": [332, 175, 442, 369], \"11\": [340, 180, 450, 370]}}</answer>\n"
        "Note: Each key in 'boxes' must be an integer second within the span, and its value must be a 4-number bounding box [x1, y1, x2, y2]."
    ),
}

# nothink 模式: 所有任务类型使用简短的后缀 (来自 VideoEval-JD)
TYPE_TEMPLATE_qwen3_nothink = {
    "multiple choice": "\nThe best answer is: ",
    "numerical": "\nThe best answer is: ",
    "OCR": "\nThe best answer is: ",
    "free-form": "\nThe best answer is: ",
    "open-ended": "\nThe best answer is: ",
    "regression": "\nThe best answer is: ",
    "temporal grounding": (
        "\nPlease provide the time span in seconds as JSON within the <answer>...</answer> tags.\n"
        "Example:\n<answer>{\"time\": [12.3, 25.7]}</answer>\n"
    ),
    "spatial-temporal grounding": (
        "\nPlease provide the time span in seconds and bounding boxes as JSON within the <answer>...</answer> tags.\n"
        "You MUST output one bounding box for every integer second within the given time span (inclusive).\n"
        "Example:\n"
        "<answer>{\"time\": [8.125, 13.483], \"boxes\": {\"9\": [317, 422, 582, 997], "
        "\"10\": [332, 175, 442, 369], \"11\": [340, 180, 450, 370]}}</answer>\n"
        "Note: Each key in 'boxes' must be an integer second within the span, and its value must be a 4-number bounding box [x1, y1, x2, y2]."
    ),
}

# thinking 模式: 每种任务类型使用针对性的结论格式 (来自 VideoEval-JD)
TYPE_TEMPLATE_qwen3_thinking = {
    "multiple choice": '''\nConclude by stating the final answer using the following format: "Therefore, the final answer is: $LETTER" (without quotes), where $LETTER must be one of the options.''',
    "numerical": '''\nConclude by stating the final answer using the following format: "Therefore, the final answer is: $NUMBER" (without quotes), where $NUMBER is your numerical answer.''',
    "OCR": '''\nConclude by stating the final answer using the following format: "Therefore, the final answer is: $TEXT" (without quotes), where $TEXT is the recognized text.''',
    "free-form": '''\nConclude by stating the final answer using the following format: "Therefore, the final answer is: $ANSWER" (without quotes).''',
    "open-ended": '''\nConclude by stating the final answer using the following format: "Therefore, the final answer is: $ANSWER" (without quotes).''',
    "regression": '''\nConclude by stating the final answer using the following format: "Therefore, the final answer is: $NUMBER" (without quotes), where $NUMBER is your numerical answer.''',
    "temporal grounding": (
        "\nConclude by providing the time span in seconds as JSON within the <answer>...</answer> tags.\n"
        "Example:\n<answer>{\"time\": [12.3, 25.7]}</answer>\n"
    ),
    "spatial-temporal grounding": (
        "\nConclude by providing the time span in seconds and bounding boxes as JSON within the <answer>...</answer> tags.\n"
        "You MUST output one bounding box for every integer second within the given time span (inclusive).\n"
        "Example:\n"
        "<answer>{\"time\": [8, 13], \"boxes\": {\"9\": [317, 422, 582, 997], "
        "\"10\": [332, 175, 442, 369], \"11\": [340, 180, 450, 370]}}</answer>\n"
        "Note: Each key in 'boxes' must be an integer second within the span, and its value must be a 4-number bounding box [x1, y1, x2, y2]."
    ),
}


def _resolve_qwen25_vision_process_path(utils_root: str) -> str:
    root = ensure_abs(utils_root)
    candidates = [
        os.path.join(root, "src", "qwen_vl_utils", "vision_process.py"),
        os.path.join(root, "qwen_vl_utils", "vision_process.py"),
        os.path.join(root, "vision_process.py"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find qwen2.5 vision_process.py under {root}. "
        f"Tried: {candidates}"
    )


_QWEN25_VISION_MODULES: Dict[str, Any] = {}


def _load_qwen25_vl_module(utils_root: str):
    resolved_root = ensure_abs(utils_root)
    if resolved_root in _QWEN25_VISION_MODULES:
        return _QWEN25_VISION_MODULES[resolved_root]

    module_path = _resolve_qwen25_vision_process_path(resolved_root)
    spec = importlib.util.spec_from_file_location(
        f"qwen25_vl_utils_{abs(hash(module_path))}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load qwen2.5 vision utils from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _QWEN25_VISION_MODULES[resolved_root] = module
    return module


class VisionBackend:
    def __init__(self, model_family: str, qwen25_utils_root: str):
        self.model_family = model_family
        self.qwen25_utils_root = ensure_abs(qwen25_utils_root)
        if model_family == "qwen25":
            self.module = _load_qwen25_vl_module(self.qwen25_utils_root)
        elif model_family == "qwen3":
            self.module = importlib.import_module("qwen_vl_utils")
        else:
            raise ValueError(f"Unsupported model_family: {model_family}")

    def build_video_content(
        self,
        data_type: str,
        video_path: str,
        nframes: int,
        fps: float,
        max_pixels: int,
        total_pixels: int,
    ) -> dict:
        if self.model_family == "qwen25":
            return {
                "type": data_type,
                "video": video_path,
                "nframes": nframes,
                "max_pixels": max_pixels,
                "total_pixels": total_pixels,
            }
        return {
            "type": data_type,
            "video": video_path,
            "max_frames": nframes,
            "fps": fps,
            "max_pixels": max_pixels,
            "total_pixels": total_pixels,
        }

    def process_vision_info(
        self,
        conversations: list[dict] | list[list[dict]],
        patch_size: int,
        return_video_metadata: bool = False,
    ) -> tuple[Any, Any, dict]:
        if self.model_family == "qwen25":
            image_inputs, video_inputs = self.module.process_vision_info(conversations)
            return image_inputs, video_inputs, {}

        image_inputs, video_inputs, video_kwargs = self.module.process_vision_info(
            conversations,
            image_patch_size=patch_size,
            return_video_kwargs=True,
            return_video_metadata=return_video_metadata,
        )
        return image_inputs, video_inputs, video_kwargs or {}

    def fetch_image(self, ele: dict, patch_size: int):
        if self.model_family == "qwen25":
            return self.module.fetch_image(ele, size_factor=patch_size)
        return self.module.fetch_image(ele, image_patch_size=patch_size)


def build_prompt_text(
    model_family: str,
    question: str,
    problem_type: str,
    thinking_mode: bool,
    image_hint: str = "",
) -> str:
    if model_family == "qwen25":
        suffix = TYPE_TEMPLATE_qwen25.get(problem_type, TYPE_TEMPLATE_qwen25["free-form"])
        return image_hint + QUESTION_TEMPLATE_qwen25.format(Question=question) + suffix

    # qwen3 家族: 根据 thinking_mode 和 problem_type 选择对应的模板
    if thinking_mode:
        # Thinking 模式: 所有任务类型都使用 thinking 专用模板
        if problem_type == "multiple choice":
            # MC thinking: 使用 MC 专用的 thinking 问题模板 (已包含结论格式)
            return image_hint + QUESTION_TEMPLATE_qwen3_thinking.format(Question=question)
        else:
            # 非 MC thinking: 使用通用 thinking 问题模板 + 任务特定的 thinking 后缀
            suffix = TYPE_TEMPLATE_qwen3_thinking.get(
                problem_type, TYPE_TEMPLATE_qwen3_thinking.get("free-form", "")
            )
            return image_hint + QUESTION_TEMPLATE_qwen3_thinking_generic.format(Question=question) + suffix
    else:
        # Non-thinking 模式
        if problem_type == "multiple choice":
            template = QUESTION_TEMPLATE_qwen3
        else:
            template = QUESTION_TEMPLATE_qwen3_open
        suffix = TYPE_TEMPLATE_qwen3_nothink.get(
            problem_type, TYPE_TEMPLATE_qwen3_nothink.get("free-form", "")
        )
        return image_hint + template.format(Question=question) + suffix


# ============== 工具函数 ==============
def load_dataset(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return [obj]
        return obj
    else:
        raise ValueError("Input file must be .json or .jsonl")


def extract_between(text: str, tag: str) -> str:
    m = re.search(fr'<{tag}>\s*(.*?)\s*</{tag}>', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_think(s: str) -> str:
    return extract_between(s, "think")


def extract_answer(s: str, problem_type: str = "multiple choice") -> str:
    """
    统一的答案提取函数，兼容 thinking 和 instruct 模式

    Thinking 模式输出格式: "...推理过程...</think>\n\nTherefore, the final answer is: A"
    Instruct 模式输出格式: "A" 或 "The answer is A"
    """
    # 只匹配大写选项字母，避免匹配到 "is" 中的 "i"
    OPTION_PATTERN_UPPER = r'[A-J]'
    # 匹配大小写选项字母（用于需要的地方）
    OPTION_PATTERN = r'[A-Ja-j]'
    OPTION_SET = 'ABCDEFGHIJ'

    # 0. grounding 类型: 优先提取 <answer> 中的 JSON，避免被选项字母规则误伤
    if problem_type in ("temporal grounding", "spatial-temporal grounding"):
        ans = extract_between(s, "answer")
        if ans:
            return ans
        think_content = extract_between(s, "think")
        if think_content:
            json_match = re.search(r'\{.*\}', think_content, re.DOTALL)
            if json_match:
                return json_match.group(0)
        cleaned = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL | re.IGNORECASE).strip()
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return cleaned if cleaned else s.strip()

    # 1. 优先从 <answer>...</answer> 标签提取
    ans = extract_between(s, "answer")
    if ans and problem_type != "multiple choice":
        return ans
    if ans and len(ans) <= 2:
        # 如果是单个字母或 "A." 格式
        letter = ans.strip().rstrip('.').upper()
        if letter in OPTION_SET:
            return letter

    # 2. Thinking 模式: 匹配 "Therefore, the final answer is: X" 格式
    # 这是最常见的 thinking 模式输出格式，优先匹配
    final_answer_patterns = [
        # "Therefore, the final answer is: A" 或 "Therefore, the final answer is A"
        rf'(?:Therefore|So|Thus|Hence|Finally)[,\s]*(?:the\s+)?(?:final\s+|best\s+|correct\s+)?answer\s+is[:\s]+({OPTION_PATTERN_UPPER})\b',
        # "the answer is A" 格式
        rf'(?:the\s+)?(?:final\s+|best\s+|correct\s+)?answer\s+is[:\s]+({OPTION_PATTERN_UPPER})\b',
        # "答案是 A" 中文格式
        rf'答案[是为：:]\s*({OPTION_PATTERN_UPPER})\b',
    ]

    for pattern in final_answer_patterns:
        match = re.search(pattern, s, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # 2.5 非选择题: 尝试完整提取 "Therefore, the final answer is: ..."
    if problem_type != "multiple choice":
        final_text_match = re.search(
            r'(?:Therefore|So|Thus|Hence|Finally)[,\s]*(?:the\s+)?(?:final\s+|best\s+|correct\s+)?answer\s+is[:\s]+(.+)',
            s,
            re.IGNORECASE | re.DOTALL,
        )
        if final_text_match:
            return final_text_match.group(1).strip()

    # 3. 检查 </think> 后面的内容
    think_end_match = re.search(r'</think>\s*(.*?)$', s, re.DOTALL | re.IGNORECASE)
    if think_end_match:
        after_think = think_end_match.group(1).strip()
        if after_think:
            # 直接是单个选项字母
            if len(after_think) == 1 and after_think.upper() in OPTION_SET:
                return after_think.upper()
            # 以选项字母开头: "A." 或 "A " 或 "A"
            option_match = re.match(rf'^({OPTION_PATTERN_UPPER})(?:\s*[.。\):]|$|\s)', after_think)
            if option_match:
                return option_match.group(1).upper()
            # 再次尝试匹配 "answer is X" 格式（使用大写模式避免误匹配）
            answer_match = re.search(rf'answer\s+is[:\s]+({OPTION_PATTERN_UPPER})\b', after_think, re.IGNORECASE)
            if answer_match:
                return answer_match.group(1).upper()

    # 4. Instruct 模式: 检查简单的选项输出
    # "I choose A" 或 "I select B"
    choose_match = re.search(
        rf'(?:I\s+)?(?:choose|select|pick)\s+(?:option\s+)?({OPTION_PATTERN_UPPER})\b',
        s, re.IGNORECASE
    )
    if choose_match:
        return choose_match.group(1).upper()

    # 5. 检查最后几行是否直接是选项
    lines = s.strip().split('\n')
    for line in reversed(lines[-5:]):
        line = line.strip()
        # 单独一个字母
        if len(line) == 1 and line.upper() in OPTION_SET:
            return line.upper()
        # "A." 或 "A)" 格式
        option_match = re.match(rf'^({OPTION_PATTERN_UPPER})\s*[.):\s。]?\s*$', line)
        if option_match:
            return option_match.group(1).upper()
        # 行尾是选项字母（短行）
        if len(line) < 30:
            end_match = re.search(rf'\b({OPTION_PATTERN_UPPER})\s*[.)]?\s*$', line)
            if end_match:
                return end_match.group(1).upper()

    # 6. 非选择题: 返回清理后的文本
    if problem_type != "multiple choice":
        cleaned = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL | re.IGNORECASE).strip()
        return cleaned if cleaned else s.strip()

    return ""


def normalize_number(num_str: str):
    try:
        cleaned = num_str.strip()
        cleaned = re.sub(r'^[^\d\-+.]*', '', cleaned)
        cleaned = re.sub(r'[^\d.]*$', '', cleaned)
        return float(cleaned.replace(",", ""))
    except Exception:
        return None


def iou_1d(pred: List[float], gt: List[float]) -> float:
    def _is_list_of_numbers(x, n=None):
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

    if not _is_list_of_numbers(pred, 2) or not _is_list_of_numbers(gt, 2):
        return 0.0
    s1, e1 = float(pred[0]), float(pred[1])
    s2, e2 = float(gt[0]), float(gt[1])
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 1e-12 else 0.0


def iou_2d(box1: List[float], box2: List[float]) -> float:
    def _is_list_of_numbers(x, n=None):
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

    if not _is_list_of_numbers(box1, 4) or not _is_list_of_numbers(box2, 4):
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


def mean_iou_over_intersection(pred_boxes: Dict[str, List[float]], gt_boxes: Dict[str, List[float]]) -> float:
    if not isinstance(pred_boxes, dict) or not isinstance(gt_boxes, dict):
        return 0.0
    common = [k for k in pred_boxes.keys() if k in gt_boxes]
    if not common:
        return 0.0
    vals = [iou_2d(pred_boxes[k], gt_boxes[k]) for k in common]
    return sum(vals) / len(vals) if vals else 0.0


def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)
    eps = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + eps)
    thresholds = torch.arange(start, end + interval / 2, interval, dtype=torch.float32)
    mra = (rel_error < (1 - thresholds)).float().mean()
    return mra.item()


def reward_fn(sample: dict, model_output: str, question_type: str) -> float:
    try:
        output_ans = extract_answer(model_output, question_type)
        if output_ans == '':
            output_ans = model_output
        gt_ans = extract_answer(sample.get("solution", ""), question_type)
        if question_type == "multiple choice":
            return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return mean_relative_accuracy(out_number, gt_number)
        elif question_type == "temporal grounding":
            try:
                pred = json.loads(output_ans)
                gtj = json.loads(gt_ans)
                if not isinstance(pred, dict) or not isinstance(gtj, dict):
                    return 0.0
                return iou_1d(pred.get("time"), gtj.get("time"))
            except Exception:
                return 0.0
        elif question_type == "spatial-temporal grounding":
            try:
                pred = json.loads(output_ans)
                gtj = json.loads(gt_ans)
                if not isinstance(pred, dict) or not isinstance(gtj, dict):
                    return 0.0
                tiou = iou_1d(pred.get("time"), gtj.get("time"))
                pboxes, gboxes = pred.get("boxes"), gtj.get("boxes")
                if not isinstance(gboxes, dict) or len(gboxes) == 0:
                    return tiou
                miou = mean_iou_over_intersection(
                    pboxes if isinstance(pboxes, dict) else {}, gboxes
                )
                return 0.5 * tiou + 0.5 * miou
            except Exception:
                return 0.0
        else:
            return 0.0
    except Exception:
        return 0.0


def ensure_abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(p)


def get_dataset_cache_dir(
    base_cache_dir: str,
    dataset_name: str,
    nframes: int,
    fps: float,
    max_pixels: int,
    total_pixels: int,
    model_family: str = "qwen3",
) -> str:
    mp_k = max_pixels // 1024
    tp_k = total_pixels // 1024
    dir_name = f"{dataset_name}_f{nframes}_fps{fps}_mp{mp_k}k_tp{tp_k}k"
    if model_family != "qwen3":
        dir_name = f"{dir_name}_{model_family}"
    return os.path.join(base_cache_dir, dir_name)


def check_cache_exists(
    cache_dir: str,
    data: List[dict],
    video_base: str,
    nframes: int,
    fps: float,
    max_pixels: int,
    patch_size: int,
    model_family: str = "qwen3",
    threshold: float = 0.9,
) -> bool:
    if not os.path.exists(cache_dir):
        return False
    video_paths = set()
    for ex in data:
        video_path = os.path.normpath(os.path.join(video_base, ex["path"]))
        video_paths.add(video_path)
    cached_count = 0
    for video_path in video_paths:
        cache_key = get_cache_key(video_path, nframes, fps, max_pixels, patch_size, model_family)
        cache_path = get_cache_path(cache_dir, cache_key)
        if cache_exists(cache_path):
            cached_count += 1
    total = len(video_paths)
    if total == 0:
        return False
    ratio = cached_count / total
    print(f"[Cache Check] {cached_count}/{total} videos cached ({ratio*100:.1f}%)")
    return ratio >= threshold


def load_dataset_config(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    out = {}
    for k, v in cfg.items():
        out[k] = {
            "json": ensure_abs(v["json"]),
            "video": ensure_abs(v["video"]),
        }
    return out


# ============== 缓存相关函数 ==============
def get_cache_key(
    video_path: str,
    nframes: int,
    fps: float,
    max_pixels: int,
    patch_size: int,
    model_family: str = "qwen3",
) -> str:
    if model_family == "qwen3":
        params = f"{video_path}|{nframes}|{fps}|{max_pixels}|{patch_size}"
    else:
        params = f"{model_family}|{video_path}|{nframes}|{fps}|{max_pixels}|{patch_size}"
    return hashlib.md5(params.encode()).hexdigest()[:16]


def get_cache_path(cache_dir: str, cache_key: str, use_safetensors: bool = True) -> str:
    ext = ".safetensors" if use_safetensors and HAS_SAFETENSORS else ".npy"
    return os.path.join(cache_dir, f"{cache_key}{ext}")


def save_video_cache(cache_path: str, video_inputs: List, video_kwargs: dict) -> bool:
    try:
        if video_inputs is None or len(video_inputs) == 0:
            return False
        raw_video = video_inputs[0]
        video_meta = {}
        if isinstance(raw_video, tuple):
            video_tensor = raw_video[0]
            if len(raw_video) > 1 and isinstance(raw_video[1], dict):
                video_meta = raw_video[1]
        else:
            video_tensor = raw_video
        if not isinstance(video_tensor, torch.Tensor):
            return False
        all_metadata = {
            "video_kwargs": _serialize_video_kwargs(video_kwargs),
            "video_meta": video_meta,
        }
        if cache_path.endswith(".safetensors") and HAS_SAFETENSORS:
            tensors = {"frames": video_tensor.half().contiguous()}
            metadata = {"all_metadata": json.dumps(all_metadata)}
            save_file(tensors, cache_path, metadata=metadata)
        else:
            np.save(cache_path, video_tensor.half().contiguous().numpy())
            meta_path = cache_path.replace(".npy", "_meta.json")
            with open(meta_path, "w") as f:
                json.dump(all_metadata, f)
        return True
    except Exception as e:
        print(f"Error saving cache: {e}")
        return False


def load_video_cache(cache_path: str) -> Tuple[Optional[List], Optional[dict]]:
    try:
        if cache_path.endswith(".safetensors") and HAS_SAFETENSORS:
            with safe_open(cache_path, framework="pt") as f:
                metadata = f.metadata()
                frames = f.get_tensor("frames").float()
            if "all_metadata" in metadata:
                all_metadata = json.loads(metadata["all_metadata"])
                video_kwargs = _deserialize_video_kwargs(all_metadata.get("video_kwargs", {}))
                video_meta = all_metadata.get("video_meta", {})
            else:
                video_kwargs = json.loads(metadata.get("video_kwargs", "{}"))
                video_kwargs = _deserialize_video_kwargs(video_kwargs)
                video_meta = {}
        else:
            frames = torch.from_numpy(np.load(cache_path)).float()
            meta_path = cache_path.replace(".npy", "_meta.json")
            with open(meta_path, "r") as f:
                all_metadata = json.load(f)
            if "video_kwargs" in all_metadata and "video_meta" in all_metadata:
                video_kwargs = _deserialize_video_kwargs(all_metadata.get("video_kwargs", {}))
                video_meta = all_metadata.get("video_meta", {})
            else:
                video_kwargs = _deserialize_video_kwargs(all_metadata)
                video_meta = {}
        video_inputs = [(frames, video_meta)]
        return video_inputs, video_kwargs
    except Exception as e:
        print(f"Error loading cache {cache_path}: {e}")
        return None, None


def _serialize_video_kwargs(video_kwargs: dict) -> dict:
    result = {}
    for k, v in video_kwargs.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.tolist()
        elif isinstance(v, list):
            result[k] = [item.tolist() if isinstance(item, torch.Tensor) else item for item in v]
        else:
            result[k] = v
    return result


def _deserialize_video_kwargs(video_kwargs: dict) -> dict:
    result = {}
    for k, v in video_kwargs.items():
        if k == "video_grid_thw" and isinstance(v, list):
            result[k] = [torch.tensor(item) if isinstance(item, list) else item for item in v]
        else:
            result[k] = v
    return result


def cache_exists(cache_path: str) -> bool:
    if cache_path.endswith(".safetensors"):
        return os.path.exists(cache_path)
    else:
        return os.path.exists(cache_path) and os.path.exists(cache_path.replace(".npy", "_meta.json"))


def skip_marker_path(cache_path: str) -> str:
    """Return the .skip marker file path for a given cache path."""
    base = cache_path.rsplit(".", 1)[0]
    return base + ".skip"


def is_skip_marked(cache_path: str) -> bool:
    return os.path.exists(skip_marker_path(cache_path))


def mark_skip(cache_path: str, reason: str = "") -> None:
    with open(skip_marker_path(cache_path), "w") as f:
        f.write(reason)


# ============== 预处理函数 ==============
def _preprocess_worker(
    video_path: str,
    cache_dir: str,
    nframes: int,
    fps: float,
    max_pixels: int,
    patch_size: int,
    total_pixels: int,
    model_family: str,
    qwen25_utils_root: str,
    force: bool,
    timeout: int,
) -> Tuple[str, bool, str]:
    """在独立进程中预处理视频，内置超时保护防止 decord 卡死"""
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Video processing timed out after {timeout}s")

    # 设置进程级超时信号
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        result = preprocess_single_video(
            video_path,
            cache_dir,
            nframes,
            fps,
            max_pixels,
            patch_size,
            total_pixels,
            model_family,
            qwen25_utils_root,
            force,
        )
        signal.alarm(0)  # 取消超时
        return result
    except TimeoutError as e:
        signal.alarm(0)
        return video_path, False, str(e)
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def _run_in_process(
    video_path,
    cache_dir,
    nframes,
    fps,
    max_pixels,
    patch_size,
    total_pixels,
    model_family,
    qwen25_utils_root,
    force,
    timeout,
    result_dict,
):
    """子进程入口（顶层函数，兼容 spawn 模式），结果写入共享 dict"""
    try:
        vp, ok, msg = _preprocess_worker(
            video_path, cache_dir, nframes, fps, max_pixels, patch_size,
            total_pixels, model_family, qwen25_utils_root, force, timeout=timeout
        )
        result_dict['result'] = (vp, ok, msg)
    except Exception as e:
        result_dict['result'] = (video_path, False, str(e))


def preprocess_single_video(
    video_path: str,
    cache_dir: str,
    nframes: int,
    fps: float,
    max_pixels: int,
    patch_size: int,
    total_pixels: int,
    model_family: str = "qwen3",
    qwen25_utils_root: str = DEFAULT_QWEN25_UTILS_ROOT,
    force: bool = False,
) -> Tuple[str, bool, str]:
    import gc
    vision_backend = VisionBackend(model_family, qwen25_utils_root)
    cache_key = get_cache_key(video_path, nframes, fps, max_pixels, patch_size, model_family)
    cache_path = get_cache_path(cache_dir, cache_key)
    if not force and cache_exists(cache_path):
        return video_path, True, "cached"
    if not os.path.exists(video_path):
        return video_path, False, f"video not found: {video_path}"
    try:
        print(f"[PREPROCESS_START] video_path={video_path}", flush=True)
        msg = [{
            "role": "user",
            "content": [
                vision_backend.build_video_content(
                    "video", video_path, nframes, fps, max_pixels, total_pixels
                ),
                {"type": "text", "text": "placeholder"}
            ]
        }]
        image_inputs, video_inputs, video_kwargs = vision_backend.process_vision_info(
            msg, patch_size=patch_size, return_video_metadata=True
        )
        if model_family == "qwen3":
            video_kwargs["do_resize"] = False
        success = save_video_cache(cache_path, video_inputs, video_kwargs)
        del image_inputs, video_inputs, video_kwargs, msg
        gc.collect()
        if success:
            return video_path, True, "processed"
        else:
            return video_path, False, "save failed"
    except Exception as e:
        gc.collect()
        return video_path, False, str(e)


def preprocess_dataset(
    data: List[dict],
    video_base: str,
    cache_dir: str,
    nframes: int,
    fps: float,
    max_pixels: int,
    patch_size: int,
    total_pixels: int,
    model_family: str = "qwen3",
    qwen25_utils_root: str = DEFAULT_QWEN25_UTILS_ROOT,
    num_workers: int = 4,
    force: bool = False,
    video_timeout: int = 120,
) -> Tuple[int, int]:
    """预处理数据集中的视频。

    使用 mp.Process 代替 ProcessPoolExecutor，确保超时的子进程能被可靠杀掉，
    避免 decord 读取损坏视频时卡死导致整个程序挂住。
    """
    import gc
    os.makedirs(cache_dir, exist_ok=True)
    video_paths = set()
    for ex in data:
        video_path = os.path.normpath(os.path.join(video_base, ex["path"]))
        video_paths.add(video_path)
    video_paths = list(video_paths)
    print(f"Total unique videos to process: {len(video_paths)}")
    success_count = 0
    fail_count = 0
    cached_count = 0

    # 先快速过滤已缓存和已标记跳过的视频
    to_process = []
    for vp in video_paths:
        cache_key = get_cache_key(vp, nframes, fps, max_pixels, patch_size, model_family)
        cache_path = get_cache_path(cache_dir, cache_key)
        if not force and cache_exists(cache_path):
            cached_count += 1
        elif not force and is_skip_marked(cache_path):
            fail_count += 1
        else:
            to_process.append(vp)

    print(f"  Already cached: {cached_count}, to process: {len(to_process)}")

    BATCH_SIZE = num_workers
    for batch_start in range(0, len(to_process), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(to_process))
        batch_paths = to_process[batch_start:batch_end]

        # 每个视频一个独立进程，便于超时后强制 kill
        manager = mp.Manager()
        processes = []
        for vp in batch_paths:
            result_dict = manager.dict()
            p = mp.Process(
                target=_run_in_process,
                args=(vp, cache_dir, nframes, fps, max_pixels, patch_size,
                      total_pixels, model_family, qwen25_utils_root, force, video_timeout, result_dict)
            )
            processes.append((vp, p, result_dict))
            p.start()

        desc = f"Preprocessing [{batch_start+1}-{batch_end}/{len(to_process)}]"
        for vp, p, result_dict in tqdm(processes, desc=desc):
            # 等待进程完成，超时后强制杀掉
            p.join(timeout=video_timeout + 10)  # 给 SIGALRM 额外 10 秒
            if p.is_alive():
                print(f"  Killing hung process: {vp}", flush=True)
                p.kill()
                p.join(timeout=5)
                fail_count += 1
                cache_key = get_cache_key(vp, nframes, fps, max_pixels, patch_size, model_family)
                cache_path = get_cache_path(cache_dir, cache_key)
                mark_skip(cache_path, f"killed after {video_timeout}s timeout")
                print(f"  Timeout ({video_timeout}s): {vp}")
            elif 'result' in result_dict:
                video_path, success, msg = result_dict['result']
                if success:
                    if msg == "cached":
                        cached_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1
                    print(f"  Failed: {video_path} - {msg}")
            else:
                fail_count += 1
                print(f"  Process crashed: {vp} (exit code {p.exitcode})")
                cache_key = get_cache_key(vp, nframes, fps, max_pixels, patch_size, model_family)
                cache_path = get_cache_path(cache_dir, cache_key)
                mark_skip(cache_path, f"process crashed with exit code {p.exitcode}")

        manager.shutdown()
        gc.collect()
    print(f"Preprocessing complete: {success_count} processed, {cached_count} cached, {fail_count} failed")
    return success_count + cached_count, fail_count


# ============== 请求数据结构 ==============
@dataclass
class PendingRequest:
    """待处理的请求"""
    request_id: str
    sample_idx: int
    sample: dict


# ============== AsyncLLMEngine 评测器 ==============
class AsyncEngineEvaluator:
    """
    使用 AsyncLLMEngine 的真正异步评测器

    核心机制:
    1. 数据加载线程: 在后台持续加载数据
    2. 请求提交协程: 将加载好的数据提交给引擎
    3. 结果收集协程: 异步收集推理结果
    4. 三者完全并行，实现真流水线
    """

    def __init__(
        self,
        engine,
        sampling_params,
        messages: List[List[dict]],
        cache_paths: List[str],
        data: List[dict],
        processor,
        patch_size: int,
        vision_backend: VisionBackend,
        load_workers: int,
        output_path: str,
        existing_results: List[dict] = None,
        start_idx: int = 0,
        max_concurrent_requests: int = 32,
    ):
        self.engine = engine
        self.sampling_params = sampling_params
        self.messages = messages
        self.cache_paths = cache_paths
        self.data = data
        self.processor = processor
        self.patch_size = patch_size
        self.vision_backend = vision_backend
        self.load_workers = load_workers
        self.output_path = output_path
        self.results = existing_results or []
        self.start_idx = start_idx
        self.max_concurrent_requests = max_concurrent_requests

        # 状态追踪
        self.pending_requests: Dict[str, PendingRequest] = {}
        self.completed_count = 0
        self.total_count = len(messages) - start_idx

        # 异步队列: 数据加载 -> 请求提交
        self.ready_queue: asyncio.Queue = None

        # 控制信号
        self.loading_done = False
        self.all_submitted = False

    def _load_single_item(self, idx: int) -> Tuple[int, dict, dict]:
        """加载单个样本 (在线程池中运行)"""
        msg = self.messages[idx]
        cp = self.cache_paths[idx]
        sample = self.data[idx]

        prompt_text = self.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )

        video_inputs, video_kwargs = None, None
        image_inputs = None

        if is_skip_marked(cp):
            print(f"  [SKIP] video marked as problematic, skipping: {cp}")
            return idx, sample, None

        if cache_exists(cp):
            video_inputs, video_kwargs = load_video_cache(cp)

        if video_inputs is None:
            image_inputs, video_inputs, video_kwargs = self.vision_backend.process_vision_info(
                msg,
                patch_size=self.patch_size,
                return_video_metadata=True,
            )
            if self.vision_backend.model_family == "qwen3":
                video_kwargs["do_resize"] = False
        elif any(c.get("type") == "image" for c in msg[0]["content"]):
            # 视频从缓存加载，但图片仍需单独处理（图片很小不需要缓存）
            _img_list = []
            for c in msg[0]['content']:
                if c.get('type') == 'image':
                    _img_list.append(self.vision_backend.fetch_image(c, self.patch_size))
            if _img_list:
                image_inputs = _img_list

        mm_data = {}
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        if image_inputs is not None:
            mm_data['image'] = image_inputs

        llm_input = {
            "prompt": prompt_text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs or {},
        }

        return idx, sample, llm_input

    async def _data_loader_task(self):
        """数据加载任务: 在后台线程池中持续加载数据"""
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=self.load_workers) as executor:
            # 并发加载
            pending_futures = {}
            next_load_idx = self.start_idx
            next_yield_idx = self.start_idx
            completed_loads = {}

            # 初始填充
            while next_load_idx < len(self.messages) and len(pending_futures) < self.load_workers * 2:
                future = loop.run_in_executor(executor, self._load_single_item, next_load_idx)
                pending_futures[next_load_idx] = future
                next_load_idx += 1

            # 持续加载并按顺序放入队列
            while next_yield_idx < len(self.messages):
                # 检查已完成的加载任务
                done_indices = []
                for idx, future in list(pending_futures.items()):
                    if future.done():
                        done_indices.append(idx)

                for idx in done_indices:
                    future = pending_futures.pop(idx)
                    try:
                        result = await future
                        completed_loads[idx] = result
                    except Exception as e:
                        print(f"Error loading sample {idx}: {e}")
                        completed_loads[idx] = None

                    # 添加新的加载任务
                    if next_load_idx < len(self.messages):
                        new_future = loop.run_in_executor(executor, self._load_single_item, next_load_idx)
                        pending_futures[next_load_idx] = new_future
                        next_load_idx += 1

                # 按顺序放入 ready_queue
                while next_yield_idx in completed_loads:
                    item = completed_loads.pop(next_yield_idx)
                    if item is not None:
                        await self.ready_queue.put(item)
                    next_yield_idx += 1

                # 避免忙等待
                if not done_indices:
                    await asyncio.sleep(0.01)

        self.loading_done = True
        print("[Loader] All data loaded")

    async def _request_submitter_task(self):
        """请求提交任务: 从 ready_queue 取数据，提交给引擎"""
        submitted_count = 0

        while True:
            # 控制并发数
            while len(self.pending_requests) >= self.max_concurrent_requests:
                await asyncio.sleep(0.01)

            # 检查是否还有数据
            if self.loading_done and self.ready_queue.empty():
                break

            try:
                # 从队列获取数据 (带超时，避免死锁)
                item = await asyncio.wait_for(self.ready_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            idx, sample, llm_input = item
            if llm_input is None:
                # 视频被标记为有问题，跳过该样本
                submitted_count += 1
                continue
            request_id = f"req_{idx}"

            # 记录待处理请求
            self.pending_requests[request_id] = PendingRequest(
                request_id=request_id,
                sample_idx=idx,
                sample=sample,
            )

            # 提交请求给引擎 (非阻塞)
            # vLLM V1 格式: 将所有输入打包成 dict
            inputs = {
                "prompt": llm_input["prompt"],
                "multi_modal_data": llm_input["multi_modal_data"],
                "mm_processor_kwargs": llm_input["mm_processor_kwargs"],
            }
            self.engine.generate(
                inputs,
                self.sampling_params,
                request_id=request_id,
            )

            submitted_count += 1

        self.all_submitted = True
        print(f"[Submitter] All {submitted_count} requests submitted")

    async def _result_collector_task(self, pbar):
        """结果收集任务: 异步收集推理结果"""
        while True:
            # 检查是否完成
            if self.all_submitted and len(self.pending_requests) == 0:
                break

            # 从引擎获取完成的结果
            request_outputs = await self.engine.get_request_outputs()

            for request_output in request_outputs:
                if not request_output.finished:
                    continue

                request_id = request_output.request_id

                if request_id not in self.pending_requests:
                    continue

                pending = self.pending_requests.pop(request_id)

                # 提取结果
                output_text = request_output.outputs[0].text
                num_tokens = len(request_output.outputs[0].token_ids)

                # 处理样本
                sample = pending.sample
                think_chain = extract_think(output_text)
                q_type = sample.get("problem_type", "")
                final_ans = extract_answer(output_text, q_type) or output_text

                sample["output"] = output_text
                sample["prediction"] = final_ans
                sample["reward"] = reward_fn(sample, output_text, q_type)
                sample["correct"] = True if sample["reward"] == 1.0 else False
                sample["num_output_tokens"] = num_tokens
                if think_chain:
                    sample["process"] = f"<think>{think_chain}</think>"

                self.results.append(sample)
                self.completed_count += 1
                pbar.update(1)

                # 定期保存
                if self.completed_count % 10 == 0:
                    self._save_results()

            await asyncio.sleep(0.01)

    def _save_results(self):
        """保存结果到文件"""
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump({"results": self.results}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving results: {e}")

    async def run(self):
        """运行异步评测"""
        # 初始化队列
        self.ready_queue = asyncio.Queue(maxsize=self.max_concurrent_requests * 2)

        # 进度条
        pbar = tqdm(total=self.total_count, desc="[AsyncEngine] Processing")

        # 启动三个并行任务
        loader_task = asyncio.create_task(self._data_loader_task())
        submitter_task = asyncio.create_task(self._request_submitter_task())
        collector_task = asyncio.create_task(self._result_collector_task(pbar))

        # 等待所有任务完成
        await asyncio.gather(loader_task, submitter_task, collector_task)

        pbar.close()

        # 最终保存
        self._save_results()

        return self.results


# ============== 使用标准 generate API 的简化版本 ==============
async def run_async_evaluation_with_generate(
    engine,
    sampling_params,
    messages: List[List[dict]],
    cache_paths: List[str],
    data: List[dict],
    processor,
    patch_size: int,
    vision_backend: VisionBackend,
    load_workers: int,
    output_path: str,
    existing_results: List[dict] = None,
    start_idx: int = 0,
    max_concurrent: int = 16,
    queue_size: int = 8,  # 预加载队列大小，控制内存中缓存的视频数据量
):
    """
    使用 AsyncLLMEngine.generate() 异步生成器的评测

    这是更简洁的实现方式，利用 vLLM 的 generate 异步生成器
    """
    results = existing_results or []
    total_count = len(messages) - start_idx

    pbar = tqdm(total=total_count, desc="[AsyncEngine] Processing")

    # 状态追踪
    pending_requests: Dict[str, Tuple[int, dict]] = {}  # request_id -> (idx, sample)
    completed_count = 0

    # 数据加载函数
    def load_single_item(idx: int):
        msg = messages[idx]
        cp = cache_paths[idx]
        sample = data[idx]

        prompt_text = processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )

        video_inputs, video_kwargs = None, None
        image_inputs = None

        if is_skip_marked(cp):
            print(f"  [SKIP] video marked as problematic, skipping: {cp}")
            return idx, sample, None, None, None

        if cache_exists(cp):
            video_inputs, video_kwargs = load_video_cache(cp)

        if video_inputs is None:
            image_inputs, video_inputs, video_kwargs = vision_backend.process_vision_info(
                msg,
                patch_size=patch_size,
                return_video_metadata=True,
            )
            if vision_backend.model_family == "qwen3":
                video_kwargs["do_resize"] = False
        elif any(c.get("type") == "image" for c in msg[0]["content"]):
            # 视频从缓存加载，但图片仍需单独处理（图片很小不需要缓存）
            _img_list = []
            for c in msg[0]['content']:
                if c.get('type') == 'image':
                    _img_list.append(vision_backend.fetch_image(c, patch_size))
            if _img_list:
                image_inputs = _img_list

        mm_data = {}
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        if image_inputs is not None:
            mm_data['image'] = image_inputs

        return idx, sample, prompt_text, mm_data, video_kwargs or {}

    # 使用正在运行的事件循环
    loop = asyncio.get_running_loop()

    # 已加载但等待提交的数据队列，限制大小避免内存堆积
    # queue_size 控制预加载的视频数量，每个视频约 200-500MB
    loaded_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)

    async def data_loader_task():
        """后台数据加载任务"""
        with ThreadPoolExecutor(max_workers=load_workers) as executor:
            for idx in range(start_idx, len(messages)):
                try:
                    # 在线程池中加载数据
                    result = await loop.run_in_executor(executor, load_single_item, idx)
                    await loaded_queue.put(result)
                except Exception as e:
                    print(f"Error loading sample {idx}: {e}")
            # 发送结束信号
            await loaded_queue.put(None)

    async def process_results(gen, request_id):
        """处理单个请求的结果"""
        try:
            async for request_output in gen:
                if request_output.finished:
                    return request_id, request_output
        except Exception as e:
            print(f"Error in generator {request_id}: {e}")
        return request_id, None

    # 启动数据加载任务
    loader_task = asyncio.create_task(data_loader_task())

    loading_done = False
    pending_tasks: Dict[str, asyncio.Task] = {}

    while completed_count < total_count:
        # 1. 尝试从队列获取已加载的数据并提交
        while not loading_done and len(pending_tasks) < max_concurrent:
            try:
                item = loaded_queue.get_nowait()
                if item is None:
                    loading_done = True
                    break

                idx, sample, prompt_text, mm_data, mm_kwargs = item
                if prompt_text is None:
                    # 视频被标记为有问题，跳过该样本
                    completed_count += 1
                    continue
                request_id = f"req_{idx}"

                # 启动异步生成
                # vLLM V1 格式: 将所有输入打包成 dict
                inputs = {
                    "prompt": prompt_text,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": mm_kwargs,
                }
                gen = engine.generate(
                    inputs,
                    sampling_params,
                    request_id=request_id,
                )

                # 创建处理任务
                task = asyncio.create_task(process_results(gen, request_id))
                pending_tasks[request_id] = task
                pending_requests[request_id] = (idx, sample)

            except asyncio.QueueEmpty:
                break

        # 2. 等待任意一个生成任务完成
        if pending_tasks:
            done, _ = await asyncio.wait(
                pending_tasks.values(),
                timeout=0.1,
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                request_id, request_output = await task

                # 从 pending 中移除
                if request_id in pending_tasks:
                    del pending_tasks[request_id]

                if request_output is not None and request_id in pending_requests:
                    idx, sample = pending_requests.pop(request_id)

                    # 提取结果
                    output_text = request_output.outputs[0].text
                    num_tokens = len(request_output.outputs[0].token_ids)

                    think_chain = extract_think(output_text)
                    q_type = sample.get("problem_type", "")
                    final_ans = extract_answer(output_text, q_type) or output_text

                    sample["output"] = output_text
                    sample["prediction"] = final_ans
                    sample["reward"] = reward_fn(sample, output_text, q_type)
                    sample["correct"] = True if sample["reward"] == 1.0 else False
                    sample["num_output_tokens"] = num_tokens
                    if think_chain:
                        sample["process"] = f"<think>{think_chain}</think>"

                    results.append(sample)
                    completed_count += 1
                    pbar.update(1)

                    # 定期保存
                    if completed_count % 10 == 0:
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
                elif request_id in pending_requests:
                    # 请求失败，也要计数
                    pending_requests.pop(request_id)
                    completed_count += 1
                    pbar.update(1)
        else:
            # 没有活跃任务，等待数据加载
            await asyncio.sleep(0.01)

    # 确保加载任务完成
    await loader_task

    pbar.close()

    # 最终保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    return results


# ============== 主程序 ==============
def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Qwen3-VL AsyncLLMEngine Evaluation (True Pipeline)")

    # 模式选择
    parser.add_argument("--mode", type=str, default="auto", choices=["preprocess", "eval", "both", "auto"])

    # 缓存配置
    parser.add_argument("--cache_dir", type=str, default="./video_cache")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--load_workers", type=int, default=16)
    parser.add_argument("--force_preprocess", action="store_true")

    # 模型配置
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_family", type=str, default="qwen3", choices=["qwen3", "qwen25"])
    parser.add_argument("--qwen25_utils_root", type=str, default=DEFAULT_QWEN25_UTILS_ROOT)
    parser.add_argument("--output_dir", type=str, default="./output-async")
    parser.add_argument("--result_dir", type=str, default="./result-async")

    # 视频处理参数
    parser.add_argument("--nframes", type=int, default=2048)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_pixels", type=int, default=640*32*32)
    parser.add_argument("--total_pixels", type=int, default=160376*32*32)

    # 推理参数
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--presence_penalty", type=float, default=1.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2048)

    # vLLM参数
    parser.add_argument("--max_model_len", type=int, default=256000)
    parser.add_argument("--gpu_mem_util", type=float, default=0.7)
    parser.add_argument("--enforce_eager", type=str2bool, default=True)
    parser.add_argument("--enable_chunked_prefill", type=str2bool, default=True)
    parser.add_argument("--max_num_seqs", type=int, default=8)
    parser.add_argument("--max_num_batched_tokens", type=int, default=262144)
    parser.add_argument("--num_gpus", type=int, default=None)

    # 异步评测参数
    parser.add_argument("--max_concurrent", type=int, default=16,
                        help="Maximum number of concurrent requests in the engine")
    parser.add_argument("--queue_size", type=int, default=8,
                        help="Size of prefetch queue (controls memory for cached video data)")

    # Thinking 模型配置
    parser.add_argument("--thinking_mode", action="store_true")

    # 样本数量控制
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate per dataset (for quick validation). None means all samples.")

    # 数据集配置
    parser.add_argument("--data_dir_path", type=str, default="/mnt/public/users/siqingyi/video_reasoning/data/test")
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--dataset_config", type=str, default="")

    args = parser.parse_args()

    MODEL_PATH = args.model_path
    BASE_CACHE_DIR = ensure_abs(args.cache_dir)
    BASE_OUTPUT_DIR = ensure_abs(args.output_dir)
    RESULT_DIR = ensure_abs(args.result_dir)
    # 模型命名: 当 basename 是通用名(如 checkpoint-xxx)时，拼上上级目录名避免冲突
    _model_base = os.path.basename(MODEL_PATH.rstrip('/'))
    if _model_base.startswith("checkpoint"):
        _parent = os.path.basename(os.path.dirname(MODEL_PATH.rstrip('/')))
        MODEL_NAME = f"{_parent}_{_model_base}"
    else:
        MODEL_NAME = _model_base

    os.makedirs(BASE_CACHE_DIR, exist_ok=True)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    if args.mode in ["eval", "both", "auto"]:
        os.makedirs(RESULT_DIR, exist_ok=True)

    num_gpus = args.num_gpus or torch.cuda.device_count()
    print(f"[Config] AsyncLLMEngine Mode (True Pipeline), using {num_gpus} GPUs")
    print(f"[Config] Max concurrent requests: {args.max_concurrent}")
    print(f"[Config] Model family: {args.model_family}")

    if args.thinking_mode:
        print(f"[Config] Thinking Mode enabled")

    vision_backend = VisionBackend(args.model_family, args.qwen25_utils_root)

    # 数据集映射
    ROOT = ensure_abs(args.data_dir_path)
    default_mapping: Dict[str, Dict[str, str]] = {
        "holmes":        {"json": f"{ROOT}/valid_data/holmes.json",          "video": f"{ROOT}/Video-Holmes"},
        "lvbench":       {"json": f"{ROOT}/valid_data/lvbench.json",         "video": f"{ROOT}/LVBench"},
        "longvideobench":{"json": f"{ROOT}/valid_data/longvideobench.json",  "video": f"{ROOT}/LongVideoBench"},
        "mmvu":          {"json": f"{ROOT}/valid_data/mmvu.json",            "video": f"{ROOT}/MMVU"},
        "mmvu-all":      {"json": f"{ROOT}/valid_data/mmvu-all.json",        "video": f"{ROOT}/MMVU"},
        "mvbench":       {"json": f"{ROOT}/valid_data/mvbench.json",         "video": f"{ROOT}/MVBench"},
        "tempcompass":   {"json": f"{ROOT}/valid_data/tempcompass.json",     "video": f"{ROOT}/TempCompass"},
        "vsibench":      {"json": f"{ROOT}/valid_data/vsibench.json",        "video": f"{ROOT}/VSI-Bench"},
        "videommmu":     {"json": f"{ROOT}/valid_data/videommmu_with_image.json", "video": f"{ROOT}/VideoMMMU"},
        "mlvu":          {"json": f"{ROOT}/valid_data/mlvu.json",            "video": f"{ROOT}/MLVU_Test"},
        "mlvu-test":     {"json": f"{ROOT}/valid_data/mlvu-test.json",       "video": f"{ROOT}/MLVU_Test"},
        "mlvu-dev":      {"json": f"{ROOT}/valid_data/mlvu-dev.json",        "video": f"{ROOT}/MLVU_dev/MLVU_dev"},
        "videomme":      {"json": f"{ROOT}/valid_data/videomme.json",        "video": f"{ROOT}/Video-MME"},
        "videomathqa":   {"json": f"{ROOT}/valid_data/videomathqa.jsonl",    "video": f"{ROOT}/VideoMathQA"},
        "vcrbench":      {"json": f"{ROOT}/valid_data/vcrbench.json",        "video": f"{ROOT}/VCR-Bench/v1/videos"},
        "videoreasonbench": {"json": f"{ROOT}/valid_data/videoreasonbench.json", "video": f"{ROOT}/VideoReasonBench/videos"},
        "longvideoreason": {"json": f"{ROOT}/valid_data/longvideoreason.json", "video": f"/mnt/public/users/siqingyi/video_rl_data/OneThinker-train-data/QA/longvila_videos"},
        "minerva":       {"json": f"{ROOT}/valid_data/minerva.json",         "video": f"{ROOT}/Minderva"},
        "stvg":          {"json": f"{ROOT}/valid_data/stvg.json",            "video": f"{ROOT}/ST-Align-Benchmark"},
        "charades_sta":  {"json": f"{ROOT}/valid_data/charades_sta.json",    "video": f"{ROOT}/Charades_sta/Charades_v1_480"},
        "motionbench":   {"json": f"{ROOT}/valid_data/motionbench.json",     "video": f"{ROOT}/MotionBench-official/MotionBench"},
        "video_count_eval": {"json": f"{ROOT}/valid_data/video_count_eval.json", "video": f"{ROOT}/Molmo2-VideoCountEval/Molmo2-VideoCountEval"},
    }

    for k in list(default_mapping.keys()):
        default_mapping[k]["json"] = ensure_abs(default_mapping[k]["json"])
        default_mapping[k]["video"] = ensure_abs(default_mapping[k]["video"])

    if args.dataset_config:
        override = load_dataset_config(args.dataset_config)
        default_mapping.update(override)

    run_list = args.datasets if args.datasets else list(default_mapping.keys())

    # 获取 patch_size
    print(f"Loading processor from {MODEL_PATH}...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    patch_size = processor.image_processor.patch_size
    print(f"Patch size: {patch_size}")

    # 初始化 AsyncLLMEngine
    engine = None
    sampling_params = None
    tokenizer = None

    if args.mode in ["eval", "both", "auto"]:
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        print(f"[AsyncLLMEngine] Initializing with TP={num_gpus}...")

        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            tensor_parallel_size=num_gpus,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt={"image": 1, "video": 1},
            enforce_eager=args.enforce_eager,
            enable_chunked_prefill=args.enable_chunked_prefill,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            trust_remote_code=True,
        )

        engine = AsyncLLMEngine.from_engine_args(engine_args)

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            stop_token_ids=[],
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        tokenizer.padding_side = "left"
        processor.tokenizer = tokenizer

        print("[AsyncLLMEngine] Initialized successfully")

    # 处理每个数据集
    for name in run_list:
        if name not in default_mapping:
            print(f"[Skip] dataset '{name}' not in mapping")
            continue

        json_path = default_mapping[name]["json"]
        video_base = default_mapping[name]["video"]

        if not os.path.exists(json_path):
            print(f"[Skip] JSON not found: {json_path}")
            continue

        CACHE_DIR = get_dataset_cache_dir(
            BASE_CACHE_DIR, name,
            args.nframes, args.fps, args.max_pixels, args.total_pixels,
            args.model_family,
        )
        os.makedirs(CACHE_DIR, exist_ok=True)

        OUTPUT_DIR = get_dataset_cache_dir(
            BASE_OUTPUT_DIR, name,
            args.nframes, args.fps, args.max_pixels, args.total_pixels,
            args.model_family,
        )
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        print(f"JSON:  {json_path}")
        print(f"VIDEO: {video_base}")
        print(f"CACHE: {CACHE_DIR}")
        print(f"OUTPUT: {OUTPUT_DIR}")

        # 加载数据
        data = load_dataset(json_path)
        original_count = len(data)

        # 截取部分样本（用于快速验证）
        if args.max_samples is not None and args.max_samples > 0:
            data = data[:args.max_samples]
            print(f"Loaded {original_count} samples, using first {len(data)} samples (--max_samples={args.max_samples})")
        else:
            print(f"Loaded {len(data)} samples")

        # 处理 vcrbench 特殊格式
        if name == "vcrbench" and data and isinstance(data, list) and isinstance(data[0], dict) and ("multiple-choice" in data[0] or "video_path" in data[0]):
            converted = []
            for e in data:
                pt = "multiple choice" if e.get("multiple-choice", False) else "open-ended"
                problem = e.get("question", "")
                options = []
                if e.get("multiple-choice", False):
                    ch = e.get("choices", {}) or {}
                    for letter in ["A", "B", "C", "D"]:
                        val = ch.get(letter, "")
                        options.append(f"{letter}: {val}")
                path = e.get("video_path", e.get("path", ""))
                sol = f"<answer>{e.get('answer', '')}</answer>"
                converted.append({
                    "data_type": "video",
                    "data_source": "VCR-Bench",
                    "problem_type": pt,
                    "problem": problem,
                    "options": options,
                    "path": path,
                    "problem_id": e.get("id", None),
                    "solution": sol,
                })
            data = converted

        # 自动检测模式
        current_mode = args.mode
        if args.mode == "auto":
            cache_ready = check_cache_exists(
                CACHE_DIR, data, video_base,
                args.nframes, args.fps, args.max_pixels, patch_size,
                args.model_family,
            )
            if cache_ready:
                current_mode = "eval"
                print(f"[Auto Mode] Cache found and sufficient, using 'eval' mode")
            else:
                current_mode = "both"
                print(f"[Auto Mode] Cache not ready, using 'both' mode (preprocess + eval)")

        # 预处理阶段
        if current_mode in ["preprocess", "both"]:
            print(f"\n--- Preprocessing videos ---")
            success, fail = preprocess_dataset(
                data=data,
                video_base=video_base,
                cache_dir=CACHE_DIR,
                nframes=args.nframes,
                fps=args.fps,
                max_pixels=args.max_pixels,
                total_pixels=args.total_pixels,
                patch_size=patch_size,
                model_family=args.model_family,
                qwen25_utils_root=args.qwen25_utils_root,
                num_workers=args.num_workers,
                force=args.force_preprocess,
            )
            if current_mode == "preprocess":
                continue

        # 评测阶段 (AsyncLLMEngine 真流水线)
        if current_mode in ["eval", "both"]:
            print(f"\n--- Running AsyncLLMEngine Evaluation (True Pipeline) ---")
            eval_start_time = time.time()
            output_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_output.json")
            print(f"OUT: {output_path}")

            # 构造消息和缓存映射
            messages = []
            cache_paths = []

            for ex in data:
                if ex.get("problem_type") == "multiple choice":
                    question = ex['problem'] + " Possible answer choices:\n"
                    for op in ex["options"]:
                        question += op + "\n"
                else:
                    question = ex["problem"]

                video_path = os.path.normpath(os.path.join(video_base, ex["path"]))
                cache_key = get_cache_key(
                    video_path,
                    args.nframes,
                    args.fps,
                    args.max_pixels,
                    patch_size,
                    args.model_family,
                )
                cache_path = get_cache_path(CACHE_DIR, cache_key)
                cache_paths.append(cache_path)

                # 构造 content 列表: video + (可选 image) + text
                content = [
                    vision_backend.build_video_content(
                        ex["data_type"],
                        video_path,
                        args.nframes,
                        args.fps,
                        args.max_pixels,
                        args.total_pixels,
                    ),
                ]

                # 如果有图片 (VideoMMMU Adaptation 子集)，添加 image 内容
                if ex.get("image_path"):
                    img_path = os.path.normpath(os.path.join(video_base, ex["image_path"]))
                    if os.path.exists(img_path):
                        content.append({
                            "type": "image",
                            "image": f"file://{img_path}",
                        })

                # 对有图片的样本 (VideoMMMU Adaptation)，提示图片位置
                image_hint = "The image for this question is at the end of the video.\n" if ex.get("image_path") else ""

                content.append({
                    "type": "text",
                    "text": build_prompt_text(
                        args.model_family,
                        question,
                        ex["problem_type"],
                        args.thinking_mode,
                        image_hint=image_hint,
                    )
                })

                msg = [{"role": "user", "content": content}]
                messages.append(msg)

            # 断点续跑
            existing_results = []
            start_idx = 0
            if os.path.exists(output_path):
                try:
                    with open(output_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                        existing_results = existing.get("results", [])
                        start_idx = len(existing_results)
                        print(f"Resuming from sample index {start_idx}")
                except Exception as e:
                    print(f"Error reading existing output: {e}")

            # 运行异步评测
            print(f"[AsyncEngine] Starting true pipeline evaluation...")
            final_output = asyncio.run(
                run_async_evaluation_with_generate(
                    engine=engine,
                    sampling_params=sampling_params,
                    messages=messages,
                    cache_paths=cache_paths,
                    data=data,
                    processor=processor,
                    patch_size=patch_size,
                    vision_backend=vision_backend,
                    load_workers=args.load_workers,
                    output_path=output_path,
                    existing_results=existing_results,
                    start_idx=start_idx,
                    max_concurrent=args.max_concurrent,
                    queue_size=args.queue_size,
                )
            )

            # 计算运行时间
            eval_end_time = time.time()
            eval_elapsed_time = eval_end_time - eval_start_time
            eval_elapsed_minutes = eval_elapsed_time / 60.0
            print(f"Evaluation time: {eval_elapsed_time:.2f}s ({eval_elapsed_minutes:.2f}min)")

            # 汇总指标
            mean_acc, mean_mra = [], []
            total_output_tokens = 0
            for sample in final_output:
                q_type = sample.get("problem_type", "")
                reward = sample.get("reward", 0.0)
                if q_type != "regression":
                    mean_acc.append(reward)
                else:
                    mean_mra.append(reward)
                total_output_tokens += sample.get("num_output_tokens", 0)

            final_acc = {"mean_acc": 0.0, "mean_mra": 0.0}
            if mean_acc:
                final_acc["mean_acc"] = sum(mean_acc) / len(mean_acc)
            if mean_mra:
                final_acc["mean_mra"] = sum(mean_mra) / len(mean_mra)

            avg_output_tokens = total_output_tokens / len(final_output) if final_output else 0
            print(f"Total output tokens: {total_output_tokens}, Average: {avg_output_tokens:.2f}")

            # 保存详细结果
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "results": final_output,
                    "final_acc": [final_acc],
                    "eval_time_seconds": round(eval_elapsed_time, 2),
                    "eval_time_minutes": round(eval_elapsed_minutes, 2),
                    "total_output_tokens": total_output_tokens,
                    "avg_output_tokens": round(avg_output_tokens, 2),
                }, f, indent=2, ensure_ascii=False)

            # 保存汇总结果
            result_file = os.path.join(RESULT_DIR, f"{MODEL_NAME}.json")
            if os.path.exists(result_file):
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        all_results = json.load(f)
                except Exception:
                    all_results = {}
            else:
                all_results = {}

            all_results[name] = {
                "mean_acc": final_acc["mean_acc"],
                "mean_mra": final_acc["mean_mra"],
                "num_samples": len(final_output),
                "eval_time_seconds": round(eval_elapsed_time, 2),
                "eval_time_minutes": round(eval_elapsed_minutes, 2),
                "total_output_tokens": total_output_tokens,
                "avg_output_tokens": round(avg_output_tokens, 2),
            }

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            print(f"Detailed results saved to {output_path}")
            print(f"Aggregated results saved to {result_file}")
            print(f"Mean accuracy: {final_acc['mean_acc']:.4f}")
            if mean_mra:
                print(f"Mean MRA: {final_acc['mean_mra']:.4f}")


if __name__ == "__main__":
    main()
