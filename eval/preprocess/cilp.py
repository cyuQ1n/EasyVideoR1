#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple


# =====================================================================
#  核心裁剪函数
# =====================================================================

def clip_video(
    src: str, dst: str, start: float, end: float, timeout: int = 120
) -> Tuple[str, bool, str]:
    """
    用 ffmpeg copy 模式裁剪视频片段。

    Args:
        src:     源视频路径
        dst:     输出片段路径
        start:   起始时间 (秒)
        end:     结束时间 (秒)
        timeout: ffmpeg 超时时间 (秒)

    Returns:
        (dst, success, message)
    """
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst, True, "exists"

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if not os.path.exists(src):
        return dst, False, f"source not found: {src}"

    duration = end - start
    if duration <= 0:
        return dst, False, f"invalid duration: start={start}, end={end}"

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", src,
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-loglevel", "error",
        dst,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return dst, True, "clipped"
        return dst, False, "output file empty"
    except subprocess.CalledProcessError as e:
        if os.path.exists(dst):
            os.remove(dst)
        return dst, False, (e.stderr or str(e))[-200:]
    except subprocess.TimeoutExpired:
        if os.path.exists(dst):
            os.remove(dst)
        return dst, False, f"timeout ({timeout}s)"


def parallel_clip(
    tasks: List[Tuple[str, str, float, float]],
    num_workers: int = 16,
    timeout: int = 120,
) -> Tuple[int, int, int]:
    """
    并行裁剪视频。

    Args:
        tasks: [(src, dst, start, end), ...]
        num_workers: 并行线程数
        timeout: 每个 ffmpeg 的超时

    Returns:
        (clipped, cached, failed)
    """
    clipped = cached = failed = 0
    total = len(tasks)

    if total == 0:
        print("No clips to process.")
        return 0, 0, 0

    print(f"Processing {total} clips with {num_workers} workers ...")

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(clip_video, src, dst, s, e, timeout): i
            for i, (src, dst, s, e) in enumerate(tasks)
        }
        for done_count, fut in enumerate(as_completed(futures), 1):
            dst, ok, msg = fut.result()
            if ok:
                if msg == "exists":
                    cached += 1
                else:
                    clipped += 1
            else:
                failed += 1
                if failed <= 20:
                    print(f"  [FAIL] {os.path.basename(dst)}: {msg}")

            if done_count % 500 == 0 or done_count == total:
                print(
                    f"  Progress: {done_count}/{total}  "
                    f"(clipped={clipped}, cached={cached}, failed={failed})"
                )

    print(f"\nDone: {clipped} clipped, {cached} cached, {failed} failed")
    return clipped, cached, failed


# =====================================================================
#  格式解析: 从不同的 JSON 格式提取 (src, dst, start, end)
# =====================================================================

def _build_video_index(video_root: str) -> Dict[str, str]:
    """
    递归扫描 video_root，建立 {视频文件名stem -> 完整路径} 索引。
    """
    index = {}
    for dirpath, _, filenames in os.walk(video_root):
        for fn in filenames:
            if fn.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm")):
                stem = Path(fn).stem
                if stem not in index:
                    index[stem] = os.path.join(dirpath, fn)
    return index


# ODV-Bench path 格式: {index}_{video_stem}_s{start}_e{end}.mp4
_ODVBENCH_PATH_RE = re.compile(r"^(\d+)_(.+)_s([\d.]+)_e([\d.]+)\.mp4$")


def parse_odvbench(
    input_json: str, video_root: str, output_dir: str
) -> List[Tuple[str, str, float, float]]:
    """
    解析 ODV-Bench valid_data JSON。

    path 格式: {index}_{video_stem}_s{start}_e{end}.mp4
    从 path 中提取 video_stem、start、end，在 video_root 中递归查找源视频。
    """
    with open(input_json, encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} samples from {input_json}")

    # 建立视频索引
    video_index = _build_video_index(video_root)
    print(f"Indexed {len(video_index)} videos in {video_root}")

    tasks = []
    for item in raw_data:
        clip_name = item["path"]
        m = _ODVBENCH_PATH_RE.match(clip_name)
        if not m:
            print(f"  [WARN] cannot parse path: {clip_name}")
            continue

        video_stem = m.group(2)
        start = float(m.group(3))
        end = float(m.group(4))
        dst = os.path.join(output_dir, clip_name)

        src = video_index.get(video_stem, os.path.join(video_root, f"{video_stem}.mp4"))
        tasks.append((src, dst, start, end))

    print(f"Unique clips: {len(tasks)}")
    return tasks


def parse_livesports(
    input_json: str, video_root: str, output_dir: str
) -> List[Tuple[str, str, float, float]]:
    """
    解析 LiveSports-3K-QA valid_data JSON。

    path 格式: {video_id}_{begin}_{end}.mp4
    从 path 中提取 video_id、begin、end，在 video_root 中查找 {video_id}.mp4。
    同一片段只裁剪一次 (去重)。
    """
    with open(input_json, encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} samples from {input_json}")

    # path 格式: {video_id}_{start}_{end}.mp4  (video_id 可含下划线和连字符)
    path_re = re.compile(r"^(.+)_([\d.]+)_([\d.]+)\.mp4$")

    unique_clips: Dict[str, Tuple[str, float, float]] = {}
    for e in raw_data:
        clip_name = e["path"]
        if clip_name in unique_clips:
            continue
        m = path_re.match(clip_name)
        if not m:
            print(f"  [WARN] cannot parse path: {clip_name}")
            continue
        video_id = m.group(1)
        begin = float(m.group(2))
        end = float(m.group(3))
        unique_clips[clip_name] = (video_id, begin, end)

    print(f"Unique clips: {len(unique_clips)} (from {len(raw_data)} samples)")

    tasks = []
    for clip_name, (video_id, begin, end) in unique_clips.items():
        src = os.path.join(video_root, f"{video_id}.mp4")
        dst = os.path.join(output_dir, clip_name)
        tasks.append((src, dst, begin, end))

    return tasks


def parse_generic(
    input_json: str, video_root: str, output_dir: str
) -> List[Tuple[str, str, float, float]]:
    """
    解析通用格式 JSON。

    每条记录需要包含:
      - video: 源视频相对路径
      - start: 起始时间 (秒)
      - end:   结束时间 (秒)

    可选字段:
      - clip_name: 输出文件名 (默认自动生成)
    """
    with open(input_json, encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} samples from {input_json}")

    tasks = []
    seen = set()
    for i, item in enumerate(raw_data):
        start = float(item["start"])
        end = float(item["end"])
        video_rel = item["video"]
        src = os.path.join(video_root, video_rel)

        if "clip_name" in item:
            clip_name = item["clip_name"]
        else:
            stem = Path(video_rel).stem
            clip_name = f"{stem}_{start:.3f}_{end:.3f}.mp4"

        if clip_name in seen:
            continue
        seen.add(clip_name)

        dst = os.path.join(output_dir, clip_name)
        tasks.append((src, dst, start, end))

    print(f"Unique clips: {len(tasks)}")
    return tasks


PARSERS = {
    "odvbench": parse_odvbench,
    "livesports": parse_livesports,
    "generic": parse_generic,
}


# =====================================================================
#  主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="视频裁剪工具: 按时间戳从长视频中裁剪片段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_json", required=True, help="原始数据 JSON 路径")
    parser.add_argument("--video_root", required=True, help="源视频根目录")
    parser.add_argument("--output_dir", required=True, help="裁剪片段输出目录")
    parser.add_argument(
        "--format", required=True, choices=list(PARSERS.keys()),
        help="JSON 格式: odvbench / livesports / generic",
    )
    parser.add_argument("--num_workers", type=int, default=16, help="并行 ffmpeg 线程数")
    parser.add_argument("--timeout", type=int, default=120, help="单个 ffmpeg 超时 (秒)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Format     : {args.format}")
    print(f"Input JSON : {args.input_json}")
    print(f"Video root : {args.video_root}")
    print(f"Output dir : {args.output_dir}")
    print(f"Workers    : {args.num_workers}")
    print()

    # 解析 JSON → 裁剪任务列表
    parse_fn = PARSERS[args.format]
    tasks = parse_fn(args.input_json, args.video_root, args.output_dir)

    # 执行裁剪
    parallel_clip(tasks, num_workers=args.num_workers, timeout=args.timeout)


if __name__ == "__main__":
    main()
