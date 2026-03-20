# Visual RL Reward Functions
# Modular reward function collection.

from .utils import extract_answer, extract_boxed, preprocess_ground_truth
from .math_task import compute_score as math_compute_score
from .multiple_choice import compute_score as multiple_choice_compute_score
from .numerical import compute_score as numerical_compute_score
from .open_ended import compute_score as open_ended_compute_score
from .ocr import compute_score as ocr_compute_score
from .boolean import compute_score as boolean_compute_score
from .code_task import compute_score as code_compute_score
from .llava import compute_score as llava_compute_score
from .grounding import (
    spatial_compute_score,
    temporal_compute_score,
    spatial_temporal_compute_score,
)

__all__ = [
    "extract_answer",
    "extract_boxed",
    "preprocess_ground_truth",
    "math_compute_score",
    "multiple_choice_compute_score",
    "numerical_compute_score",
    "open_ended_compute_score",
    "ocr_compute_score",
    "boolean_compute_score",
    "code_compute_score",
    "llava_compute_score",
    "spatial_compute_score",
    "temporal_compute_score",
    "spatial_temporal_compute_score",
]
