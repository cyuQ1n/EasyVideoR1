from __future__ import annotations

import inspect
import os
from typing import Any, Optional, Sequence, Union

import torch
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video

VIDEO_SOURCE_MODE_PREFER_PREPROCESSED = "prefer_preprocessed"
VIDEO_SOURCE_MODE_PREPROCESSED_ONLY = "preprocessed_only"
VIDEO_SOURCE_MODE_REALTIME_ONLY = "realtime_only"

VIDEO_SOURCE_TYPE_PREPROCESSED = "preprocessed"
VIDEO_SOURCE_TYPE_PATH = "path"

VALID_VIDEO_SOURCE_MODES = {
    VIDEO_SOURCE_MODE_PREFER_PREPROCESSED,
    VIDEO_SOURCE_MODE_PREPROCESSED_ONLY,
    VIDEO_SOURCE_MODE_REALTIME_ONLY,
}

VALID_VIDEO_SOURCE_TYPES = {
    VIDEO_SOURCE_TYPE_PREPROCESSED,
    VIDEO_SOURCE_TYPE_PATH,
}

try:
    _FETCH_VIDEO_PARAM_NAMES = frozenset(inspect.signature(fetch_video).parameters)
except (TypeError, ValueError):
    _FETCH_VIDEO_PARAM_NAMES = frozenset()


def _build_fallback_video_metadata(
    video_data: Union[torch.Tensor, list[ImageObject]],
    sample_fps: float,
) -> dict[str, Any]:
    if isinstance(video_data, torch.Tensor):
        total_num_frames = int(video_data.shape[0])
        height = int(video_data.shape[-2]) if video_data.ndim >= 3 else None
        width = int(video_data.shape[-1]) if video_data.ndim >= 3 else None
    else:
        total_num_frames = len(video_data)
        height = width = None
        if total_num_frames > 0:
            first_frame = video_data[0]
            if isinstance(first_frame, ImageObject):
                width, height = first_frame.size
            elif isinstance(first_frame, torch.Tensor) and first_frame.ndim >= 2:
                height = int(first_frame.shape[-2])
                width = int(first_frame.shape[-1])

    metadata = {
        "fps": float(sample_fps),
        "frames_indices": list(range(total_num_frames)),
        "total_num_frames": total_num_frames,
    }
    if sample_fps > 0:
        metadata["duration"] = total_num_frames / sample_fps
    if width is not None and height is not None:
        metadata["width"] = width
        metadata["height"] = height
    return metadata


def process_video(
    video: str,
    min_pixels: int = 4 * 32 * 32,
    max_pixels: int = 64 * 32 * 32,
    max_frames: int = 128,
    video_fps: float = 2.0,
    total_pixels: Optional[int] = None,
    return_fps: bool = False,
) -> Any:
    vision_info = {
        "video": video,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_frames": max_frames,
        "fps": video_fps,
    }
    if total_pixels is not None:
        vision_info["total_pixels"] = total_pixels

    fetch_kwargs = {}
    if "image_patch_size" in _FETCH_VIDEO_PARAM_NAMES:
        fetch_kwargs["image_patch_size"] = 16
    elif "image_factor" in _FETCH_VIDEO_PARAM_NAMES:
        fetch_kwargs["image_factor"] = 16
    if return_fps and "return_video_sample_fps" in _FETCH_VIDEO_PARAM_NAMES:
        fetch_kwargs["return_video_sample_fps"] = True
    if return_fps and "return_video_metadata" in _FETCH_VIDEO_PARAM_NAMES:
        fetch_kwargs["return_video_metadata"] = True

    result = fetch_video(vision_info, **fetch_kwargs)
    if not return_fps:
        return result

    if isinstance(result, tuple) and len(result) == 2:
        video_data, sample_fps = result
    else:
        video_data, sample_fps = result, video_fps

    sample_fps = float(sample_fps)
    if isinstance(video_data, tuple) and len(video_data) == 2:
        return video_data, sample_fps

    return (video_data, _build_fallback_video_metadata(video_data, sample_fps)), sample_fps


def normalize_video_source_mode(mode: Optional[str], *, use_preprocessed_videos: bool) -> str:
    if mode is None or mode == "":
        return (
            VIDEO_SOURCE_MODE_PREFER_PREPROCESSED
            if use_preprocessed_videos
            else VIDEO_SOURCE_MODE_REALTIME_ONLY
        )

    if mode not in VALID_VIDEO_SOURCE_MODES:
        raise ValueError(
            f"Unsupported data.video_source_mode={mode!r}. "
            f"Expected one of {sorted(VALID_VIDEO_SOURCE_MODES)}."
        )

    return mode


def build_video_multimodal_contract(
    *,
    video_paths: Sequence[str],
    preprocessed_video_path: Optional[str],
    video_source_mode: str,
) -> dict[str, Any]:
    if not video_paths:
        raise ValueError("video_paths must not be empty when building a video contract.")

    if video_source_mode == VIDEO_SOURCE_MODE_PREPROCESSED_ONLY:
        if preprocessed_video_path is None:
            raise FileNotFoundError("video_source_mode=preprocessed_only but no preprocessed video is available.")
        source_type = VIDEO_SOURCE_TYPE_PREPROCESSED
        source_paths = [preprocessed_video_path]
    elif video_source_mode == VIDEO_SOURCE_MODE_PREFER_PREPROCESSED and preprocessed_video_path is not None:
        source_type = VIDEO_SOURCE_TYPE_PREPROCESSED
        source_paths = [preprocessed_video_path]
    elif video_source_mode in {
        VIDEO_SOURCE_MODE_PREFER_PREPROCESSED,
        VIDEO_SOURCE_MODE_REALTIME_ONLY,
    }:
        source_type = VIDEO_SOURCE_TYPE_PATH
        source_paths = list(video_paths)
    else:
        raise ValueError(
            f"Unsupported video_source_mode={video_source_mode!r}. "
            f"Expected one of {sorted(VALID_VIDEO_SOURCE_MODES)}."
        )

    return {
        "video": {
            "source_type": source_type,
            "paths": source_paths,
        }
    }


def has_video_multimodal_data(multi_modal_data: dict[str, Any]) -> bool:
    return any(
        key in multi_modal_data
        for key in ("video", "videos", "preprocessed_video_path")
    )


def validate_multi_modal_data_contract(multi_modal_data: dict[str, Any]) -> None:
    if not isinstance(multi_modal_data, dict):
        raise TypeError(f"multi_modal_data must be a dict, got {type(multi_modal_data).__name__}.")

    if not multi_modal_data:
        return

    has_images = "images" in multi_modal_data
    has_video = has_video_multimodal_data(multi_modal_data)
    if has_images and has_video:
        raise ValueError("A single sample cannot currently contain both images and video in multi_modal_data.")

    if has_images:
        images = multi_modal_data["images"]
        if not hasattr(images, "__len__") or len(images) == 0:
            raise ValueError("multi_modal_data['images'] must be a non-empty sequence.")
        return

    if "video" in multi_modal_data:
        video_value = multi_modal_data["video"]
        if isinstance(video_value, dict) and "source_type" in video_value:
            source_type = video_value.get("source_type")
            paths = video_value.get("paths")
            if source_type not in VALID_VIDEO_SOURCE_TYPES:
                raise ValueError(
                    f"multi_modal_data['video']['source_type']={source_type!r} is invalid. "
                    f"Expected one of {sorted(VALID_VIDEO_SOURCE_TYPES)}."
                )
            if not isinstance(paths, list) or len(paths) == 0 or not all(isinstance(path, str) for path in paths):
                raise ValueError("multi_modal_data['video']['paths'] must be a non-empty list of strings.")
            return
        return

    if "videos" in multi_modal_data:
        videos = multi_modal_data["videos"]
        if not isinstance(videos, list) or len(videos) == 0 or not all(isinstance(path, str) for path in videos):
            raise ValueError("Legacy multi_modal_data['videos'] must be a non-empty list of strings.")
        return

    if "preprocessed_video_path" in multi_modal_data:
        path = multi_modal_data["preprocessed_video_path"]
        if not isinstance(path, str) or path == "":
            raise ValueError("Legacy multi_modal_data['preprocessed_video_path'] must be a non-empty string.")
        return

    raise ValueError(f"Unsupported multi_modal_data keys: {sorted(multi_modal_data.keys())}.")


def _normalize_video_metadata(metadata: Any, frames: Any) -> dict[str, Any]:
    if isinstance(metadata, dict):
        return metadata

    if hasattr(metadata, "keys") and hasattr(metadata, "__getitem__"):
        return {key: metadata[key] for key in metadata.keys()}

    total_num_frames = frames.shape[0] if hasattr(frames, "shape") else len(frames)
    return {
        "fps": 2.0,
        "frames_indices": list(range(total_num_frames)),
        "total_num_frames": total_num_frames,
    }


def _load_preprocessed_video_artifact(path: str) -> tuple[Any, dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed video artifact not found: {path}")

    preprocessed_data = torch.load(path, map_location="cpu", weights_only=False)
    if "frames" not in preprocessed_data or "metadata" not in preprocessed_data:
        raise KeyError(
            f"Preprocessed video artifact {path} must contain 'frames' and 'metadata'. "
            f"Found keys: {sorted(preprocessed_data.keys())}."
        )

    return preprocessed_data["frames"], _normalize_video_metadata(
        preprocessed_data["metadata"],
        preprocessed_data["frames"],
    )


def load_video_tensors_and_metadata(
    multi_modal_data: dict[str, Any],
    *,
    video_min_pixels: int,
    video_max_pixels: int,
    video_max_frames: int,
    video_fps: float,
    video_total_pixels: Optional[int],
) -> tuple[list[Any], Optional[list[dict[str, Any]]]]:
    if not multi_modal_data:
        return [], None

    if "video" in multi_modal_data:
        video_value = multi_modal_data["video"]
        if isinstance(video_value, dict) and "source_type" in video_value:
            source_type = video_value["source_type"]
            source_paths = video_value["paths"]
            videos: list[Any] = []
            video_metadatas: list[dict[str, Any]] = []

            if source_type == VIDEO_SOURCE_TYPE_PREPROCESSED:
                for path in source_paths:
                    frames, metadata = _load_preprocessed_video_artifact(path)
                    videos.append(frames)
                    video_metadatas.append(metadata)
                return videos, video_metadatas

            if source_type == VIDEO_SOURCE_TYPE_PATH:
                for path in source_paths:
                    result = process_video(
                        path,
                        min_pixels=video_min_pixels,
                        max_pixels=video_max_pixels,
                        max_frames=video_max_frames,
                        video_fps=video_fps,
                        total_pixels=video_total_pixels,
                        return_fps=True,
                    )
                    video_data, _ = result
                    if isinstance(video_data, tuple) and len(video_data) == 2:
                        frames, metadata = video_data
                    else:
                        frames = video_data
                        metadata = _normalize_video_metadata(None, frames)
                    videos.append(frames)
                    video_metadatas.append(metadata)
                return videos, video_metadatas

            raise ValueError(
                f"Unsupported multi_modal_data['video']['source_type']={source_type!r}. "
                f"Expected one of {sorted(VALID_VIDEO_SOURCE_TYPES)}."
            )

        videos = multi_modal_data["video"]
        video_metadatas = multi_modal_data.get("video_metadatas", None)
        return videos, video_metadatas

    if "preprocessed_video_path" in multi_modal_data:
        frames, metadata = _load_preprocessed_video_artifact(multi_modal_data["preprocessed_video_path"])
        return [frames], [metadata]

    if "videos" in multi_modal_data:
        videos = []
        video_metadatas = []
        for path in multi_modal_data["videos"]:
            result = process_video(
                path,
                min_pixels=video_min_pixels,
                max_pixels=video_max_pixels,
                max_frames=video_max_frames,
                video_fps=video_fps,
                total_pixels=video_total_pixels,
                return_fps=True,
            )
            video_data, _ = result
            if isinstance(video_data, tuple) and len(video_data) == 2:
                frames, metadata = video_data
            else:
                frames = video_data
                metadata = _normalize_video_metadata(None, frames)
            videos.append(frames)
            video_metadatas.append(metadata)
        return videos, video_metadatas

    return [], None
