#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频预处理脚本 - 离线处理视频数据以加速训练

功能：
1. 读取训练数据集JSON文件
2. 批量处理所有视频文件
3. 将处理后的视频帧张量保存为.pt文件
4. 生成新的JSON文件，包含预处理文件路径

使用方法：
    python scripts/preprocess_videos.py \
        --input_file data/train.json \
        --output_dir data/preprocessed_videos \
        --output_file data/train_preprocessed.json \
        --video_fps 2.0 \
        --video_max_frames 128 \
        --video_min_pixels 5120 \
        --video_max_pixels 131072 \
        --workers 8
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from qwen_vl_utils.vision_process import fetch_video
from tqdm import tqdm


def process_single_video(
    video_path: str,
    min_pixels: int,
    max_pixels: int,
    max_frames: int,
    video_fps: float,
    image_dir: Optional[str] = None,
) -> Tuple[List[Image.Image], dict, float]:
    """
    处理单个视频文件

    Args:
        video_path: 视频文件路径
        min_pixels: 最小像素数
        max_pixels: 最大像素数
        max_frames: 最大帧数
        video_fps: 采样帧率
        image_dir: 视频文件根目录

    Returns:
        (frames, metadata, sample_fps): 视频帧列表、元数据、采样帧率
    """
    # 处理相对路径
    if image_dir is not None and not os.path.isabs(video_path):
        full_video_path = os.path.join(image_dir, video_path)
    else:
        full_video_path = video_path

    if not os.path.exists(full_video_path):
        raise FileNotFoundError(f"Video file not found: {full_video_path}")

    # 使用与Dataset相同的处理逻辑
    vision_info = {
        "video": full_video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_frames": max_frames,
        "fps": video_fps,
    }

    result = fetch_video(
        vision_info,
        image_patch_size=16,
        return_video_sample_fps=True,
        return_video_metadata=True,
    )

    # 解析返回结果
    if isinstance(result, tuple) and len(result) == 2:
        video_data, sample_fps = result
        if isinstance(video_data, tuple) and len(video_data) == 2:
            frames, metadata = video_data
            return frames, metadata, sample_fps
        else:
            # 降级处理：没有metadata
            return video_data, {}, sample_fps
    else:
        # 最简单的情况
        return result, {}, video_fps


def save_preprocessed_video(
    frames: List[Image.Image],
    metadata: dict,
    sample_fps: float,
    output_path: str,
    source_video: str,
    preprocess_config: Dict[str, Any],
) -> None:
    """
    保存预处理的视频数据为.pt文件

    Args:
        frames: 视频帧列表（PIL Image对象）
        metadata: 视频元数据
        sample_fps: 采样帧率
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存数据
    torch.save(
        {
            "frames": frames,  # List[PIL.Image]
            "metadata": metadata,  # dict
            "sample_fps": sample_fps,  # float
        },
        output_path,
    )


def get_video_hash(video_path: str, params: dict) -> str:
    """
    生成视频的唯一hash标识（基于路径和处理参数）

    Args:
        video_path: 视频路径
        params: 处理参数字典

    Returns:
        16字符的hash字符串
    """
    # 将路径和参数序列化
    param_str = f"{video_path}_{params['min_pixels']}_{params['max_pixels']}_{params['max_frames']}_{params['fps']}"
    return hashlib.md5(param_str.encode()).hexdigest()[:16]


def process_video_worker(args: Tuple[int, dict, dict, str, Optional[str]]) -> Tuple[int, dict, Optional[str]]:
    """
    Worker函数用于多进程处理

    Args:
        args: (index, item, params, output_dir, image_dir)

    Returns:
        (index, updated_item, error_message)
    """
    index, item, params, output_dir, image_dir = args

    try:
        # 检查是否有视频字段
        if "videos" not in item or not item["videos"]:
            return index, item, "No videos field"

        video_path = item["videos"][0]  # 假设每个样本只有一个视频

        # 生成唯一的输出文件名
        video_hash = get_video_hash(video_path, params)
        output_filename = f"{video_hash}.pt"
        output_path = os.path.join(output_dir, output_filename)

        # 如果已经处理过，跳过
        if os.path.exists(output_path):
            item["preprocessed_video"] = output_filename
            return index, item, None

        # 处理视频
        frames, metadata, sample_fps = process_single_video(
            video_path=video_path,
            min_pixels=params["min_pixels"],
            max_pixels=params["max_pixels"],
            max_frames=params["max_frames"],
            video_fps=params["fps"],
            image_dir=image_dir,
        )

        # 保存预处理结果
        save_preprocessed_video(
            frames=frames,
            metadata=metadata,
            sample_fps=sample_fps,
            output_path=output_path,
            source_video=video_path,
            preprocess_config={
                "min_pixels": params["min_pixels"],
                "max_pixels": params["max_pixels"],
                "max_frames": params["max_frames"],
                "fps": params["fps"],
                "total_pixels": params.get("total_pixels"),
                "version": PREPROCESS_VERSION,
            },
        )

        # 更新数据项
        item["preprocessed_video"] = output_filename

        return index, item, None

    except Exception as e:
        return index, item, str(e)


def preprocess_dataset(
    input_file: str,
    output_dir: str,
    output_file: str,
    video_fps: float = 2.0,
    video_max_frames: int = 128,
    video_min_pixels: int = 5120,
    video_max_pixels: int = 131072,
    image_dir: Optional[str] = None,
    workers: int = 8,
    skip_errors: bool = False,
) -> None:
    """
    预处理整个数据集

    Args:
        input_file: 输入JSON文件路径
        output_dir: 预处理文件输出目录
        output_file: 输出JSON文件路径
        video_fps: 视频采样帧率
        video_max_frames: 最大帧数
        video_min_pixels: 最小像素数
        video_max_pixels: 最大像素数
        image_dir: 视频文件根目录（如果视频路径是相对路径）
        workers: 并行处理的worker数量
        skip_errors: 是否跳过错误继续处理
    """
    # 加载输入数据（支持JSON数组和JSONL格式）
    print(f"Loading dataset from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # JSON数组格式
            print("Detected JSON array format")
            data = json.load(f)
        else:
            # JSONL格式（每行一个JSON对象）
            print("Detected JSONL format")
            data = [json.loads(line) for line in f]

    print(f"Total samples: {len(data)}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 统计视频数量
    video_count = sum(1 for item in data if "videos" in item and item["videos"])
    print(f"Samples with videos: {video_count}")

    if video_count == 0:
        print("Warning: No videos found in dataset!")
        return

    # 准备处理参数
    params = {
        "min_pixels": video_min_pixels,
        "max_pixels": video_max_pixels,
        "max_frames": video_max_frames,
        "fps": video_fps,
    }

    print("\nProcessing parameters:")
    print(f"  - video_fps: {video_fps}")
    print(f"  - video_max_frames: {video_max_frames}")
    print(f"  - video_min_pixels: {video_min_pixels}")
    print(f"  - video_max_pixels: {video_max_pixels}")
    print(f"  - workers: {workers}")
    print(f"  - output_dir: {output_dir}")
    print()

    # 多进程处理
    tasks = [(i, item.copy(), params, output_dir, image_dir) for i, item in enumerate(data)]

    results = [None] * len(data)
    errors = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_video_worker, task): task[0] for task in tasks}

        with tqdm(total=len(tasks), desc="Processing videos") as pbar:
            for future in as_completed(futures):
                index, updated_item, error = future.result()

                if error:
                    errors.append((index, error))
                    if not skip_errors:
                        print(f"\nError processing item {index}: {error}")
                    results[index] = data[index]  # 保留原始数据
                else:
                    results[index] = updated_item

                pbar.update(1)

    # 报告错误
    if errors:
        print(f"\n{len(errors)} errors encountered:")
        for idx, err in errors[:10]:  # 只显示前10个错误
            print(f"  - Item {idx}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    # 只保存成功预处理的样本
    preprocessed_items = [item for item in results if "preprocessed_video" in item]
    failed_items = [item for item in results if "preprocessed_video" not in item]

    # 保存输出数据
    print(f"\nSaving preprocessed dataset to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in preprocessed_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    failed_output_file = f"{output_file}.failed.jsonl"
    print(f"Saving failed samples to {failed_output_file}...")
    with open(failed_output_file, "w", encoding="utf-8") as f:
        for item in failed_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计信息
    preprocessed_count = len(preprocessed_items)
    print("\nPreprocessing complete!")
    print(f"  - Total samples: {len(results)}")
    print(f"  - Successfully preprocessed: {preprocessed_count}")
    print(f"  - Errors: {len(errors)}")
    print(f"  - Skipped samples (no preprocessed_video): {len(failed_items)}")
    print(f"  - Output file: {output_file}")
    print(f"  - Failed samples file: {failed_output_file}")
    print(f"  - Preprocessed videos directory: {output_dir}")

    # 估算磁盘空间
    total_size = 0
    for file in Path(output_dir).glob("*.pt"):
        total_size += file.stat().st_size
    print(f"  - Total preprocessed data size: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess videos for RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 预处理训练集
  python scripts/preprocess_videos.py \\
      --input_file data/train.json \\
      --output_dir data/preprocessed_videos \\
      --output_file data/train_preprocessed.json \\
      --workers 16

  # 指定视频根目录（如果JSON中是相对路径）
  python scripts/preprocess_videos.py \\
      --input_file data/train.json \\
      --output_dir data/preprocessed_videos \\
      --output_file data/train_preprocessed.json \\
      --image_dir /path/to/video/root \\
      --workers 16

  # 自定义处理参数
  python scripts/preprocess_videos.py \\
      --input_file data/train.json \\
      --output_dir data/preprocessed_videos \\
      --output_file data/train_preprocessed.json \\
      --video_fps 3.0 \\
      --video_max_frames 256 \\
      --workers 16
        """,
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSON file path (one sample per line)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save preprocessed video files (.pt)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file path with preprocessed references",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=2.0,
        help="Video sampling FPS (default: 2.0)",
    )
    parser.add_argument(
        "--video_max_frames",
        type=int,
        default=128,
        help="Maximum number of frames (default: 128)",
    )
    parser.add_argument(
        "--video_min_pixels",
        type=int,
        default=4 * 32 * 32,
        help="Minimum pixels (default: 4096)",
    )
    parser.add_argument(
        "--video_max_pixels",
        type=int,
        default=64 * 32 * 32,
        help="Maximum pixels (default: 65536)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Root directory for video files (if paths in JSON are relative)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Skip errors and continue processing",
    )

    args = parser.parse_args()

    # 执行预处理
    preprocess_dataset(
        input_file=args.input_file,
        output_dir=args.output_dir,
        output_file=args.output_file,
        video_fps=args.video_fps,
        video_max_frames=args.video_max_frames,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        image_dir=args.image_dir,
        workers=args.workers,
        skip_errors=args.skip_errors,
    )


if __name__ == "__main__":
    main()
