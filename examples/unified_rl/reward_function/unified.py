# -*- coding: utf-8 -*-
"""
Unified Reward Function - unified routing
Dispatch to the corresponding reward module automatically by problem_type.
"""

import importlib
import logging
import os
import sys
from typing import Any, Dict, List


# Add the current directory to sys.path to support dynamic loading.
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

_utils = importlib.import_module("utils")
normalize_response = _utils.normalize_response
preprocess_ground_truth = _utils.preprocess_ground_truth
format_reward = _utils.format_reward

bool_reward = importlib.import_module("boolean")
code_reward = importlib.import_module("code_task")
grounding_reward = importlib.import_module("grounding")
llava_reward = importlib.import_module("llava")
math_reward = importlib.import_module("math_task")
mc_reward = importlib.import_module("multiple_choice")
num_reward = importlib.import_module("numerical")
ocr_reward = importlib.import_module("ocr")
oe_reward = importlib.import_module("open_ended")

logger = logging.getLogger(__name__)

REWARD_NAME = "unified"
REWARD_TYPE = "batch"

# Mapping from task type to reward module.
REWARD_MAPPING = {
    # Math tasks
    "math": math_reward,
    "mathematics": math_reward,
    # Multiple-choice tasks
    "multiple choice": mc_reward,
    "multiple_choice": mc_reward,
    # Numerical tasks
    "numerical": num_reward,
    "number": num_reward,
    "regression": num_reward,
    # Open-ended tasks
    "open-ended": oe_reward,
    "open_ended": oe_reward,
    "video qa": oe_reward,
    "video description": oe_reward,
    "free-form": oe_reward,
    # OCR
    "ocr": ocr_reward,
    # Boolean tasks
    "boolean": bool_reward,
    "binary classification": bool_reward,
    # LLaVA Critic
    "llava": llava_reward,
    "critic": llava_reward,
    # Code tasks
    "code": code_reward,
    "coding": code_reward,
    "svg-code": code_reward,
    "html-code": code_reward,
    # Grounding tasks (handled specially)
    "spatial grounding": "spatial",
    "temporal grounding": "temporal",
    "spatial-temporal grounding": "spatial-temporal",
    "tracking": "tracking",
}


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    accuracy_weight: float = 0.90,
    format_weight: float = 0.10,
    **kwargs,
) -> List[Dict[str, float]]:
    """
    Unified reward entry point.
    Route each sample to the matching reward function by problem_type,
    then combine with a unified format reward.

    overall = accuracy_weight * accuracy + format_weight * format
    """
    results = []

    for idx, inp in enumerate(reward_inputs):
        try:
            # Preprocess input fields.
            response = normalize_response(inp.get("response", ""))
            ground_truth = preprocess_ground_truth(inp.get("ground_truth", ""))
            problem_type = (inp.get("problem_type", "") or "").lower().strip()

            # Build a normalized input record.
            inp_processed = {**inp, "response": response, "ground_truth": ground_truth}

            # Resolve the target reward module or handler type.
            reward_target = REWARD_MAPPING.get(problem_type, oe_reward)

            # Special-case grounding tasks.
            if reward_target == "spatial":
                score_list = grounding_reward.spatial_compute_score([inp_processed], **kwargs)
            elif reward_target == "temporal":
                score_list = grounding_reward.temporal_compute_score([inp_processed], **kwargs)
            elif reward_target == "spatial-temporal":
                score_list = grounding_reward.spatial_temporal_compute_score([inp_processed], **kwargs)
            elif reward_target == "tracking":
                score_list = grounding_reward.tracking_compute_score([inp_processed], **kwargs)
            else:
                # Call the target module's compute_score function.
                score_list = reward_target.compute_score([inp_processed], **kwargs)

            score = score_list[0]
            accuracy = score.get("accuracy", 0.0)

            # Compute unified format reward.
            fmt_score = format_reward(response, problem_type)

            # Combine accuracy and format into overall.
            overall = accuracy_weight * accuracy + format_weight * fmt_score

            results.append(
                {
                    "overall": float(overall),
                    "accuracy": float(accuracy),
                    "format": float(fmt_score),
                }
            )

        except Exception as e:
            logger.error(f"Error computing reward for sample {idx}: {e}")
            results.append(
                {
                    "overall": 0.0,
                    "accuracy": 0.0,
                    "format": 0.0,
                }
            )

    return results
