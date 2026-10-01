"""
@file: s2_imagenet_size_stats.py
@description: 统计 ImageNet 训练集图像短边尺寸，并输出不同裁剪阈值下的类别分布。
@author: Changxin Ye
@created: 2026-07-10
@version: 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def default_image_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "ILSVRC2012_img_train_unzip" / "image",
        project_root / "image",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return candidates[0]


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "s2_imagenet_size_stats"


def parse_extensions(raw_extensions: str) -> tuple[str, ...]:
    extensions = []
    for item in raw_extensions.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        extensions.append(ext)
    return tuple(extensions)


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        header = file.read(24)

    if len(header) < 24 or not header.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Not a PNG file")

    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return width, height


def read_jpeg_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        if file.read(2) != b"\xff\xd8":
            raise ValueError("Not a JPEG file")

        while True:
            byte = file.read(1)
            while byte and byte != b"\xff":
                byte = file.read(1)

            if not byte:
                break

            marker = file.read(1)
            while marker == b"\xff":
                marker = file.read(1)

            if not marker:
                break

            marker_value = marker[0]

            if marker_value == 0x01 or 0xD0 <= marker_value <= 0xD9:
                continue

            length_bytes = file.read(2)
            if len(length_bytes) != 2:
                break

            segment_length = int.from_bytes(length_bytes, "big")
            if segment_length < 2:
                raise ValueError("Invalid JPEG segment length")

            if marker_value in SOF_MARKERS:
                data = file.read(5)
                if len(data) != 5:
                    break
                height = int.from_bytes(data[1:3], "big")
                width = int.from_bytes(data[3:5], "big")
                return width, height

            file.seek(segment_length - 2, os.SEEK_CUR)

    raise ValueError("JPEG size marker not found")


def read_image_size(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        return read_jpeg_size(path)

    if suffix == ".png":
        return read_png_size(path)

    raise ValueError(f"Unsupported image extension: {path.suffix}")


def threshold_key(threshold: int, strict_greater: bool) -> str:
    prefix = "gt" if strict_greater else "ge"
    return f"{prefix}_{threshold}"


def process_class_dir(
    class_dir: str,
    thresholds: tuple[int, ...],
    strict_greater: bool,
    extensions: tuple[str, ...],
) -> dict[str, Any]:
    class_path = Path(class_dir)
    counts = {threshold_key(threshold, strict_greater): 0 for threshold in thresholds}
    failed_samples: list[str] = []

    total_files = 0
    readable_images = 0
    failed_images = 0
    short_side_sum = 0
    min_short_side: int | None = None
    max_short_side: int | None = None

    for image_path in class_path.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in extensions:
            continue

        total_files += 1

        try:
            width, height = read_image_size(image_path)
        except Exception as exc:
            failed_images += 1
            if len(failed_samples) < 5:
                failed_samples.append(f"{image_path.name}: {exc}")
            continue

        readable_images += 1
        short_side = min(width, height)
        short_side_sum += short_side
        min_short_side = (
            short_side if min_short_side is None else min(min_short_side, short_side)
        )
        max_short_side = (
            short_side if max_short_side is None else max(max_short_side, short_side)
        )

        for threshold in thresholds:
            if short_side > threshold if strict_greater else short_side >= threshold:
                counts[threshold_key(threshold, strict_greater)] += 1

    mean_short_side = short_side_sum / readable_images if readable_images else 0

    return {
        "class_id": class_path.name,
        "total_files": total_files,
        "readable_images": readable_images,
        "failed_images": failed_images,
        "min_short_side": min_short_side or 0,
        "max_short_side": max_short_side or 0,
        "mean_short_side": round(mean_short_side, 4),
        "failed_samples": failed_samples,
        **counts,
    }


def collect_results(
    class_dirs: list[Path],
    thresholds: tuple[int, ...],
    strict_greater: bool,
    extensions: tuple[str, ...],
    workers: int,
    quiet: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if workers <= 1:
        for index, class_dir in enumerate(class_dirs, start=1):
            results.append(
                process_class_dir(
                    str(class_dir),
                    thresholds,
                    strict_greater,
                    extensions,
                )
            )
            print_progress(index, len(class_dirs), quiet)
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_class_dir,
                str(class_dir),
                thresholds,
                strict_greater,
                extensions,
            )
            for class_dir in class_dirs
        ]

        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print_progress(index, len(class_dirs), quiet)

    return sorted(results, key=lambda item: item["class_id"])


def print_progress(done: int, total: int, quiet: bool) -> None:
    if quiet:
        return

    if done == 1 or done == total or done % 50 == 0:
        print(f"Processed classes: {done}/{total}")


def build_summary(
    image_dir: Path,
    results: list[dict[str, Any]],
    thresholds: tuple[int, ...],
    strict_greater: bool,
    extensions: tuple[str, ...],
) -> dict[str, Any]:
    total_files = sum(item["total_files"] for item in results)
    readable_images = sum(item["readable_images"] for item in results)
    failed_images = sum(item["failed_images"] for item in results)

    threshold_counts = {}
    threshold_distribution = {}

    for threshold in thresholds:
        key = threshold_key(threshold, strict_greater)
        values = [item[key] for item in results]
        total = sum(values)

        threshold_counts[key] = total
        threshold_distribution[key] = {
            "classes_with_at_least_one_image": sum(1 for value in values if value > 0),
            "min_per_class": min(values) if values else 0,
            "max_per_class": max(values) if values else 0,
            "mean_per_class": round(statistics.mean(values), 4) if values else 0,
            "median_per_class": statistics.median(values) if values else 0,
        }

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_dir": str(image_dir),
        "comparison": "short_side > threshold"
        if strict_greater
        else "short_side >= threshold",
        "thresholds": list(thresholds),
        "extensions": list(extensions),
        "class_count": len(results),
        "total_files": total_files,
        "readable_images": readable_images,
        "failed_images": failed_images,
        "threshold_counts": threshold_counts,
        "threshold_distribution": threshold_distribution,
    }


def write_class_csv(
    output_path: Path,
    results: list[dict[str, Any]],
    thresholds: tuple[int, ...],
    strict_greater: bool,
) -> None:
    threshold_columns = [threshold_key(threshold, strict_greater) for threshold in thresholds]
    rate_columns = [f"rate_{column}" for column in threshold_columns]
    fieldnames = [
        "class_id",
        "total_files",
        "readable_images",
        "failed_images",
        "min_short_side",
        "max_short_side",
        "mean_short_side",
        *threshold_columns,
        *rate_columns,
        "failed_samples",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in results:
            row = dict(item)
            readable_images = row["readable_images"]
            for column in threshold_columns:
                row[f"rate_{column}"] = (
                    round(row[column] / readable_images, 6) if readable_images else 0
                )
            row["failed_samples"] = " | ".join(row.get("failed_samples", []))
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_summary_json(output_path: Path, summary: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count ImageNet training images whose short side satisfies size "
            "thresholds, and save per-class distribution statistics."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=default_image_dir(),
        help="ImageNet class directory root. Defaults to ../ILSVRC2012_img_train_unzip/image.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[500, 750, 1000],
        help="Short-side thresholds to count. Defaults to 500 750 1000.",
    )
    parser.add_argument(
        "--strict-greater",
        action="store_true",
        help="Use short_side > threshold. Default is short_side >= threshold.",
    )
    parser.add_argument(
        "--extensions",
        default=".jpg,.jpeg",
        help="Comma-separated image extensions. Defaults to .jpg,.jpeg.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory for CSV and JSON results.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 2) - 1)),
        help="Number of worker processes. Use 1 for sequential scanning.",
    )
    parser.add_argument(
        "--limit-classes",
        type=int,
        default=None,
        help="Only scan the first N classes. Useful for quick tests.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print progress while scanning.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    thresholds = tuple(sorted(set(args.thresholds)))
    extensions = parse_extensions(args.extensions)

    if not image_dir.is_dir():
        print(f"Image directory does not exist: {image_dir}", file=sys.stderr)
        return 2

    class_dirs = sorted(path for path in image_dir.iterdir() if path.is_dir())
    if args.limit_classes is not None:
        class_dirs = class_dirs[: args.limit_classes]

    if not class_dirs:
        print(f"No class directories found in: {image_dir}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    class_csv_path = output_dir / "class_size_distribution.csv"
    summary_json_path = output_dir / "summary.json"

    print(f"Image directory: {image_dir}")
    print(f"Class directories: {len(class_dirs)}")
    print(
        "Comparison: "
        + ("short_side > threshold" if args.strict_greater else "short_side >= threshold")
    )
    print(f"Thresholds: {', '.join(str(threshold) for threshold in thresholds)}")
    print(f"Workers: {args.workers}")

    results = collect_results(
        class_dirs=class_dirs,
        thresholds=thresholds,
        strict_greater=args.strict_greater,
        extensions=extensions,
        workers=args.workers,
        quiet=args.quiet,
    )

    summary = build_summary(
        image_dir=image_dir,
        results=results,
        thresholds=thresholds,
        strict_greater=args.strict_greater,
        extensions=extensions,
    )

    write_class_csv(
        output_path=class_csv_path,
        results=results,
        thresholds=thresholds,
        strict_greater=args.strict_greater,
    )
    write_summary_json(summary_json_path, summary)

    print("\nSummary:")
    print(f"Total files: {summary['total_files']}")
    print(f"Readable images: {summary['readable_images']}")
    print(f"Failed images: {summary['failed_images']}")

    for threshold in thresholds:
        key = threshold_key(threshold, args.strict_greater)
        distribution = summary["threshold_distribution"][key]
        print(
            f"{key}: {summary['threshold_counts'][key]} images, "
            f"{distribution['classes_with_at_least_one_image']} classes, "
            f"per-class min/median/max="
            f"{distribution['min_per_class']}/"
            f"{distribution['median_per_class']}/"
            f"{distribution['max_per_class']}"
        )

    print(f"\nClass CSV: {class_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
