"""
@file: s4_center_crop_1000px.py
@description:
    本脚本用于处理步骤 S3 筛选出的 ImageNet 大尺寸候选图片：按照图片的视觉方向，
    从几何中心直接裁剪一块 1000 x 1000 像素的正方形，并以 PNG 格式保存。

    默认输入目录为 select_image/greater_than_1000px，默认输出目录为
    select_image/s4_center_crop_1000x1000。输入和输出均保留 WordNet 类别目录结构。
    例如输入 n01440764/n01440764_10026.JPEG，会保存为
    n01440764/n01440764_10026.png。

    本步骤不会缩放、拉伸或补边。程序先应用 JPEG 的 EXIF Orientation，使裁剪依据
    与人眼看到的方向一致；随后使用整数像素坐标从图片中心裁出固定区域。如果宽度或
    高度与 1000 的差为奇数，裁剪区域固定向左或向上偏 1 个像素，以保证结果可复现。

    输出采用 PNG，避免裁剪结果再次经过 JPEG 有损编码。程序默认跳过已经存在的目标
    文件，支持中断后继续运行；也支持 dry-run、覆盖、限制类别数量及多进程处理。
    完成后会生成 manifest CSV 和 summary JSON，记录每个文件的尺寸、裁剪坐标、处理
    状态及失败原因，便于后续检查和拼图适用性筛选。

    推荐先使用 --dry-run 或 --limit-classes 做小规模检查，再执行完整处理。
@author: Changxin Ye
@created: 2026-07-11
@version: 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


CROP_SIZE = 1000


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_source_dir() -> Path:
    return project_root() / "select_image" / "greater_than_1000px"


def default_output_dir() -> Path:
    return project_root() / "select_image" / "s4_center_crop_1000x1000"


def parse_extensions(raw_extensions: str) -> tuple[str, ...]:
    extensions: list[str] = []
    for item in raw_extensions.split(","):
        extension = item.strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        extensions.append(extension)
    return tuple(dict.fromkeys(extensions))


def process_image(
    source_path: str,
    source_root: str,
    output_root: str,
    crop_size: int,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    source = Path(source_path)
    source_root_path = Path(source_root)
    output_root_path = Path(output_root)
    relative_path = source.relative_to(source_root_path)
    output_path = (output_root_path / relative_path).with_suffix(".png")

    result: dict[str, Any] = {
        "class_id": relative_path.parent.as_posix(),
        "source_file": relative_path.as_posix(),
        "output_file": output_path.relative_to(output_root_path).as_posix(),
        "original_width": "",
        "original_height": "",
        "oriented_width": "",
        "oriented_height": "",
        "crop_left": "",
        "crop_top": "",
        "crop_right": "",
        "crop_bottom": "",
        "status": "",
        "error": "",
    }

    try:
        with Image.open(source) as image:
            result["original_width"], result["original_height"] = image.size
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
            result["oriented_width"] = width
            result["oriented_height"] = height

            if width < crop_size or height < crop_size:
                result["status"] = "too_small"
                result["error"] = (
                    f"Oriented image is {width}x{height}, smaller than "
                    f"{crop_size}x{crop_size}"
                )
                return result

            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            result.update(
                crop_left=left,
                crop_top=top,
                crop_right=right,
                crop_bottom=bottom,
            )

            if output_path.exists() and not overwrite:
                result["status"] = "skipped_existing"
                return result

            if dry_run:
                result["status"] = "would_crop"
                return result

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cropped = oriented.crop((left, top, right, bottom))

            # Palette and CMYK JPEGs are normalized for predictable model input.
            if cropped.mode not in {"RGB", "RGBA"}:
                cropped = cropped.convert("RGB")
            cropped.save(output_path, format="PNG", optimize=False)
            result["status"] = "cropped"
            return result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def print_progress(done: int, total: int, quiet: bool) -> None:
    if not quiet and (done == 1 or done == total or done % 250 == 0):
        print(f"Processed images: {done}/{total}")


def collect_results(
    image_paths: list[Path],
    source_dir: Path,
    output_dir: Path,
    crop_size: int,
    dry_run: bool,
    overwrite: bool,
    workers: int,
    quiet: bool,
) -> list[dict[str, Any]]:
    if workers <= 1:
        results = []
        for index, image_path in enumerate(image_paths, start=1):
            results.append(
                process_image(
                    str(image_path), str(source_dir), str(output_dir), crop_size,
                    dry_run, overwrite,
                )
            )
            print_progress(index, len(image_paths), quiet)
        return results

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_image, str(image_path), str(source_dir), str(output_dir),
                crop_size, dry_run, overwrite,
            )
            for image_path in image_paths
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print_progress(index, len(image_paths), quiet)
    return sorted(results, key=lambda item: item["source_file"])


def write_manifest(output_path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = list(results[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def build_summary(
    source_dir: Path,
    output_dir: Path,
    crop_size: int,
    dry_run: bool,
    overwrite: bool,
    workers: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "operation": "EXIF transpose, center crop, save as PNG; no resize",
        "crop_size": crop_size,
        "output_format": "PNG",
        "dry_run": dry_run,
        "overwrite": overwrite,
        "workers": workers,
        "image_count": len(results),
        "status_counts": status_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Center-crop S3 candidate images to 1000x1000 without resizing."
    )
    parser.add_argument("--source-dir", type=Path, default=default_source_dir())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png")
    parser.add_argument(
        "--workers", type=int,
        default=min(4, max(1, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-classes", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    extensions = parse_extensions(args.extensions)

    if not source_dir.is_dir():
        print(f"Source directory does not exist: {source_dir}", file=sys.stderr)
        return 2
    if args.crop_size <= 0:
        print("Crop size must be positive.", file=sys.stderr)
        return 2
    try:
        output_dir.relative_to(source_dir)
        print("Output directory must not be inside source directory.", file=sys.stderr)
        return 2
    except ValueError:
        pass

    class_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if args.limit_classes is not None:
        class_dirs = class_dirs[:args.limit_classes]
    image_paths = sorted(
        path for class_dir in class_dirs for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not image_paths:
        print(f"No supported images found in: {source_dir}", file=sys.stderr)
        return 2

    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Class directories: {len(class_dirs)}")
    print(f"Images: {len(image_paths)}")
    print(f"Operation: center crop {args.crop_size}x{args.crop_size}, no resize")
    print("Output format: PNG")
    print(f"Workers: {args.workers}")
    print(f"Dry run: {args.dry_run}")

    results = collect_results(
        image_paths, source_dir, output_dir, args.crop_size, args.dry_run,
        args.overwrite, args.workers, args.quiet,
    )
    summary = build_summary(
        source_dir, output_dir, args.crop_size, args.dry_run, args.overwrite,
        args.workers, results,
    )

    if not args.dry_run:
        write_manifest(output_dir / "s4_center_crop_manifest.csv", results)
        with (output_dir / "s4_center_crop_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nSummary:")
    for status, count in sorted(summary["status_counts"].items()):
        print(f"{status}: {count}")
    if args.dry_run:
        print("Dry run finished. No image or report files were written.")
    else:
        print(f"Manifest: {output_dir / 's4_center_crop_manifest.csv'}")
        print(f"Summary: {output_dir / 's4_center_crop_summary.json'}")

    failures = summary["status_counts"].get("failed", 0)
    too_small = summary["status_counts"].get("too_small", 0)
    return 0 if failures == 0 and too_small == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
