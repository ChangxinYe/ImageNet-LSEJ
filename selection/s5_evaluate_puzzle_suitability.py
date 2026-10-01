"""
@file: s5_evaluate_puzzle_suitability.py
@description:
    本脚本评估 S4 生成的 1000 x 1000 PNG 是否适合用于 20 x 20 拼图任务。
    每张图片会按最终任务规则划分为 400 个 50 x 50 piece，并从“局部信息量”而非
    单纯的整图分辨率出发进行判断。

    评估重点包括：低信息 piece 的比例、低信息 piece 在 20 x 20 网格中的最大连通
    区域、近似重复 piece 的比例，以及整图清晰度。这样既能发现大面积天空、纯色墙面
    和虚化背景，也能发现颜色与纹理统计高度重复、可能导致多个 piece 难以区分的图片。

    第一版阈值有意设置得较保守，目标是只拒绝明显不适合的图片，避免过多抛弃样本。
    脚本不会自行使用固定阈值判断图片好坏，而是为每张图计算 0 至 100 的连续分数，
    分数越高表示越适合拼图。所有图片按分数从高到低排序，再由 --top-k 参数决定前
    K 张进入 accepted，其余进入 rejected，因此最终保留数量完全由使用者控制。

    accepted 和 rejected 图片会分别扁平复制到同名目录，不再保留类别子目录；原始类别、
    路径、排名、总分和各分项指标均记录在 CSV 中。ImageNet 图片名通常包含类别 ID，
    脚本仍会在复制前检查重名，发现重名时停止，避免静默覆盖。

    所有阈值均可通过命令行参数调整。建议先查看完整统计和各类别样本，再决定最终筛选
    规则，不应把当前自动评分直接视为不可更改的真实标签。
@author: Changxin Ye
@created: 2026-07-11
@version: 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SIZE = 1000
GRID_SIZE = 20
PIECE_SIZE = 50


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_source_dir() -> Path:
    return project_root() / "select_image" / "s4_center_crop_1000x1000"


def default_output_dir() -> Path:
    return project_root() / "select_image" / "s5_puzzle_suitability"


def largest_connected_component(mask: np.ndarray) -> int:
    """Return the largest 4-connected True region in a small boolean grid."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    largest = 0
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or visited[row, column]:
                continue
            stack = [(row, column)]
            visited[row, column] = True
            size = 0
            while stack:
                current_row, current_column = stack.pop()
                size += 1
                for next_row, next_column in (
                    (current_row - 1, current_column),
                    (current_row + 1, current_column),
                    (current_row, current_column - 1),
                    (current_row, current_column + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and mask[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        visited[next_row, next_column] = True
                        stack.append((next_row, next_column))
            largest = max(largest, size)
    return largest


def repeated_piece_ratio(piece_means: np.ndarray, piece_stds: np.ndarray) -> float:
    """Estimate repeated pieces using conservative quantized color statistics."""
    mean_bins = np.clip(piece_means // 16, 0, 15).astype(np.uint8)
    std_bins = np.clip(piece_stds // 12, 0, 15).astype(np.uint8)
    descriptors = np.concatenate((mean_bins, std_bins), axis=1)
    _, counts = np.unique(descriptors, axis=0, return_counts=True)
    repeated = int(counts[counts >= 4].sum())
    return repeated / len(descriptors)


def evaluate_image(path_string: str, source_root_string: str, args: dict[str, Any]) -> dict[str, Any]:
    path = Path(path_string)
    source_root = Path(source_root_string)
    relative_path = path.relative_to(source_root).as_posix()
    result: dict[str, Any] = {
        "class_id": path.parent.name,
        "relative_path": relative_path,
        "rank": "",
        "score": "",
        "selection": "",
        "low_information_pieces": "",
        "low_information_ratio": "",
        "largest_low_information_component": "",
        "repeated_piece_ratio": "",
        "mean_piece_std": "",
        "mean_gradient": "",
        "reasons": "",
        "error": "",
    }
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        if rgb.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected 1000x1000, got {rgb.shape[1]}x{rgb.shape[0]}")

        pieces = rgb.reshape(GRID_SIZE, PIECE_SIZE, GRID_SIZE, PIECE_SIZE, 3)
        pieces = pieces.transpose(0, 2, 1, 3, 4)
        flat_pieces = pieces.reshape(GRID_SIZE * GRID_SIZE, -1, 3)
        piece_means = flat_pieces.mean(axis=1)
        piece_stds_rgb = flat_pieces.std(axis=1)
        piece_stds = piece_stds_rgb.mean(axis=1)

        gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        gray_pieces = gray.reshape(GRID_SIZE, PIECE_SIZE, GRID_SIZE, PIECE_SIZE)
        gray_pieces = gray_pieces.transpose(0, 2, 1, 3)
        horizontal = np.abs(np.diff(gray_pieces, axis=3)).mean(axis=(2, 3))
        vertical = np.abs(np.diff(gray_pieces, axis=2)).mean(axis=(2, 3))
        gradients = (horizontal + vertical) / 2

        low_mask = (
            (piece_stds.reshape(GRID_SIZE, GRID_SIZE) < args["low_std"])
            & (gradients < args["low_gradient"])
        )
        low_count = int(low_mask.sum())
        low_ratio = low_count / (GRID_SIZE * GRID_SIZE)
        largest_component = largest_connected_component(low_mask)
        repeat_ratio = repeated_piece_ratio(piece_means, piece_stds_rgb)
        mean_gradient = float(gradients.mean())

        # Each term is clipped to [0, 1]. The weights emphasize ambiguity caused
        # by low-information and repeated pieces while retaining a smaller global
        # sharpness/detail contribution. Ranking is more important than the
        # absolute numerical score.
        low_information_score = 1.0 - low_ratio
        component_score = 1.0 - largest_component / (GRID_SIZE * GRID_SIZE)
        uniqueness_score = 1.0 - repeat_ratio
        gradient_score = min(mean_gradient / 10.0, 1.0)
        piece_detail_score = min(float(piece_stds.mean()) / 40.0, 1.0)
        suitability_score = 100.0 * (
            0.30 * low_information_score
            + 0.20 * component_score
            + 0.20 * uniqueness_score
            + 0.15 * gradient_score
            + 0.15 * piece_detail_score
        )

        result.update(
            score=round(suitability_score, 6),
            low_information_pieces=low_count,
            low_information_ratio=round(low_ratio, 6),
            largest_low_information_component=largest_component,
            repeated_piece_ratio=round(repeat_ratio, 6),
            mean_piece_std=round(float(piece_stds.mean()), 4),
            mean_gradient=round(mean_gradient, 4),
            reasons="",
        )
    except Exception as exc:
        result["selection"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 20x20 puzzle suitability.")
    parser.add_argument("--source-dir", type=Path, default=default_source_dir())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--top-k", type=int, default=12000)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-copy", action="store_true", help="Only write ranking reports; do not copy images.")
    parser.add_argument("--low-std", type=float, default=10.0)
    parser.add_argument("--low-gradient", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.is_dir():
        print(f"Source directory does not exist: {source_dir}", file=sys.stderr)
        return 2

    image_paths = sorted(source_dir.glob("*/*.png"))
    if args.limit_images is not None:
        image_paths = image_paths[:args.limit_images]
    if not image_paths:
        print(f"No PNG images found in: {source_dir}", file=sys.stderr)
        return 2

    if not 0 <= args.top_k <= len(image_paths):
        print(f"top-k must be between 0 and {len(image_paths)}.", file=sys.stderr)
        return 2

    names = [path.name for path in image_paths]
    if len(names) != len(set(names)):
        print("Duplicate image names found; flat output would overwrite files.", file=sys.stderr)
        return 2

    threshold_keys = ("low_std", "low_gradient")
    thresholds = {key: getattr(args, key) for key in threshold_keys}
    print(f"Source directory: {source_dir}")
    print(f"Images: {len(image_paths)}")
    print("Puzzle layout: 20x20 pieces, each piece 50x50 pixels")
    print(f"Top-k accepted images: {args.top_k}")
    print(f"Workers: {args.workers}")

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(evaluate_image, str(path), str(source_dir), thresholds)
            for path in image_paths
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if not args.quiet and (index == 1 or index == len(image_paths) or index % 250 == 0):
                print(f"Evaluated images: {index}/{len(image_paths)}")
    failed = [result for result in results if result["selection"] == "failed"]
    scored = [result for result in results if result["selection"] != "failed"]
    scored.sort(key=lambda item: (-float(item["score"]), item["relative_path"]))
    for rank, result in enumerate(scored, start=1):
        result["rank"] = rank
        result["selection"] = "accepted" if rank <= args.top_k else "rejected"
    results = scored + failed

    counts = {
        "accepted": sum(item["selection"] == "accepted" for item in results),
        "rejected": sum(item["selection"] == "rejected" for item in results),
        "failed": len(failed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "s5_scores.csv", results)
    for label in ("accepted", "rejected", "failed"):
        selected = [result for result in results if result["selection"] == label]
        if selected:
            write_csv(output_dir / f"{label}.csv", selected)

    if not args.no_copy:
        accepted_dir = output_dir / "accepted"
        rejected_dir = output_dir / "rejected"
        accepted_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir.mkdir(parents=True, exist_ok=True)
        for index, result in enumerate(scored, start=1):
            source_path = source_dir / str(result["relative_path"])
            target_dir = accepted_dir if result["selection"] == "accepted" else rejected_dir
            other_dir = rejected_dir if result["selection"] == "accepted" else accepted_dir
            target_path = target_dir / source_path.name
            other_path = other_dir / source_path.name
            if other_path.exists():
                other_path.unlink()
            if not target_path.exists() or target_path.stat().st_size != source_path.stat().st_size:
                shutil.copy2(source_path, target_path)
            if not args.quiet and (index == len(scored) or index % 500 == 0):
                print(f"Copied images: {index}/{len(scored)}")

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "image_size": IMAGE_SIZE,
        "grid_size": GRID_SIZE,
        "piece_size": PIECE_SIZE,
        "top_k": args.top_k,
        "image_count": len(results),
        "selection_counts": counts,
        "selection_ratios": {
            key: round(value / len(results), 6) for key, value in counts.items()
        },
        "thresholds": thresholds,
        "score_formula": "30% low-information + 20% component + 20% uniqueness + 15% gradient + 15% piece detail",
        "copied_images": not args.no_copy,
    }
    with (output_dir / "s5_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nTop-k selection summary:")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count} ({count / len(results):.2%})")
    print(f"Scores: {output_dir / 's5_scores.csv'}")
    print(f"Summary: {output_dir / 's5_summary.json'}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
