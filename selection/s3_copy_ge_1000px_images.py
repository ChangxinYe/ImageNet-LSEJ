"""
@file: s3_copy_ge_1000px_images.py
@description:
    本脚本用于从已经解压完成的 ILSVRC2012 ImageNet train 数据集中筛选原图短边尺寸
    大于等于 1000 px 的图像，并把这些图像复制到 select_image/greater_than_1000px
    文件夹中。

    输入目录默认是项目根目录下的 ILSVRC2012_img_train_unzip/image。该目录中每个
    n 开头的 WordNet synset ID 文件夹代表一个 ImageNet 类别，例如 n07615774。

    输出目录默认是项目根目录下的 select_image/greater_than_1000px。本步骤会暂时
    保留原始类别文件夹结构，不会把所有图片拍平成一个目录。例如源文件
    ILSVRC2012_img_train_unzip/image/n07615774/xxx.JPEG 如果满足短边 >= 1000 px，
    会被复制为 select_image/greater_than_1000px/n07615774/xxx.JPEG。

    脚本只复制符合尺寸条件的图片，不移动、不删除、不修改原始 ImageNet 数据。默认
    遇到目标文件已存在时会跳过，因此可以在中断后重复运行。脚本通过读取 JPEG/PNG
    文件头获取宽高，不需要把整张图片解码进内存，也不强制依赖 Pillow。

    运行完成后，会在输出目录写入 manifest CSV 和 summary JSON，用于记录每个类别
    扫描了多少张图、复制了多少张图、跳过了多少张图，以及是否存在尺寸读取失败的
    文件。建议正式复制前先使用 --dry-run 检查筛选数量和输出路径。
@author: Changxin Ye
@created: 2026-07-10
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


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_source_dir() -> Path:
    return project_root() / "ILSVRC2012_img_train_unzip" / "image"


def default_output_dir() -> Path:
    return project_root() / "select_image" / "greater_than_1000px"


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


def is_within_directory(base: Path, target: Path) -> bool:
    base_abs = os.path.abspath(base)
    target_abs = os.path.abspath(target)
    return os.path.commonpath([base_abs, target_abs]) == base_abs


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


def process_class_dir(
    class_dir: str,
    source_root: str,
    output_root: str,
    threshold: int,
    extensions: tuple[str, ...],
    dry_run: bool,
    overwrite: bool,
    create_empty_class_dirs: bool,
) -> dict[str, Any]:
    class_path = Path(class_dir)
    source_root_path = Path(source_root)
    output_root_path = Path(output_root)
    class_id = class_path.name
    output_class_dir = output_root_path / class_id

    if not is_within_directory(source_root_path, class_path):
        raise RuntimeError(f"Class directory is outside source root: {class_path}")

    if create_empty_class_dirs and not dry_run:
        output_class_dir.mkdir(parents=True, exist_ok=True)

    scanned_images = 0
    selected_images = 0
    copied_images = 0
    skipped_small_images = 0
    skipped_existing_images = 0
    failed_images = 0
    failed_samples: list[str] = []

    for image_path in sorted(class_path.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in extensions:
            continue

        scanned_images += 1

        try:
            width, height = read_image_size(image_path)
        except Exception as exc:
            failed_images += 1
            if len(failed_samples) < 5:
                failed_samples.append(f"{image_path.name}: {exc}")
            continue

        short_side = min(width, height)
        if short_side < threshold:
            skipped_small_images += 1
            continue

        selected_images += 1
        output_path = output_class_dir / image_path.name

        if output_path.exists() and not overwrite:
            skipped_existing_images += 1
            continue

        if not dry_run:
            output_class_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, output_path)

        copied_images += 1

    return {
        "class_id": class_id,
        "scanned_images": scanned_images,
        "selected_images": selected_images,
        "copied_images": copied_images,
        "skipped_small_images": skipped_small_images,
        "skipped_existing_images": skipped_existing_images,
        "failed_images": failed_images,
        "failed_samples": failed_samples,
    }


def print_progress(done: int, total: int, quiet: bool) -> None:
    if quiet:
        return

    if done == 1 or done == total or done % 50 == 0:
        print(f"Processed classes: {done}/{total}")


def collect_results(
    class_dirs: list[Path],
    source_dir: Path,
    output_dir: Path,
    threshold: int,
    extensions: tuple[str, ...],
    dry_run: bool,
    overwrite: bool,
    create_empty_class_dirs: bool,
    workers: int,
    quiet: bool,
) -> list[dict[str, Any]]:
    if workers <= 1:
        results = []
        for index, class_dir in enumerate(class_dirs, start=1):
            results.append(
                process_class_dir(
                    str(class_dir),
                    str(source_dir),
                    str(output_dir),
                    threshold,
                    extensions,
                    dry_run,
                    overwrite,
                    create_empty_class_dirs,
                )
            )
            print_progress(index, len(class_dirs), quiet)
        return results

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_class_dir,
                str(class_dir),
                str(source_dir),
                str(output_dir),
                threshold,
                extensions,
                dry_run,
                overwrite,
                create_empty_class_dirs,
            )
            for class_dir in class_dirs
        ]

        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print_progress(index, len(class_dirs), quiet)

    return sorted(results, key=lambda item: item["class_id"])


def build_summary(
    source_dir: Path,
    output_dir: Path,
    threshold: int,
    extensions: tuple[str, ...],
    dry_run: bool,
    overwrite: bool,
    create_empty_class_dirs: bool,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "rule": f"copy images with short_side >= {threshold}px",
        "threshold": threshold,
        "extensions": list(extensions),
        "dry_run": dry_run,
        "overwrite": overwrite,
        "create_empty_class_dirs": create_empty_class_dirs,
        "class_count": len(results),
        "scanned_images": sum(item["scanned_images"] for item in results),
        "selected_images": sum(item["selected_images"] for item in results),
        "copied_images": sum(item["copied_images"] for item in results),
        "skipped_small_images": sum(item["skipped_small_images"] for item in results),
        "skipped_existing_images": sum(
            item["skipped_existing_images"] for item in results
        ),
        "failed_images": sum(item["failed_images"] for item in results),
        "classes_with_selected_images": sum(
            1 for item in results if item["selected_images"] > 0
        ),
        "classes_with_failed_images": sum(1 for item in results if item["failed_images"] > 0),
    }


def write_manifest_csv(output_path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "class_id",
        "scanned_images",
        "selected_images",
        "copied_images",
        "skipped_small_images",
        "skipped_existing_images",
        "failed_images",
        "failed_samples",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = dict(item)
            row["failed_samples"] = " | ".join(item["failed_samples"])
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_summary_json(output_path: Path, summary: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy ImageNet images whose short side is >= 1000 px into "
            "select_image/greater_than_1000px while preserving class folders."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source_dir(),
        help="Source ImageNet class root. Defaults to ../ILSVRC2012_img_train_unzip/image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Output root. Defaults to ../select_image/greater_than_1000px.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1000,
        help="Copy images whose short side is >= this value. Defaults to 1000.",
    )
    parser.add_argument(
        "--extensions",
        default=".jpg,.jpeg",
        help="Comma-separated image extensions. Defaults to .jpg,.jpeg.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 2) - 1)),
        help="Number of worker processes. Defaults to up to 4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report counts without creating folders or copying files.",
    )
    parser.add_argument(
        "--create-empty-class-dirs",
        action="store_true",
        help=(
            "Create output class folders even when a class has no selected image. "
            "By default, folders are created only when at least one image is copied."
        ),
    )
    parser.add_argument(
        "--limit-classes",
        type=int,
        default=None,
        help="Only process the first N classes. Useful for quick tests.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print progress while scanning/copying.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    extensions = parse_extensions(args.extensions)

    if not source_dir.is_dir():
        print(f"Source directory does not exist: {source_dir}", file=sys.stderr)
        return 2

    if is_within_directory(source_dir, output_dir):
        print(
            f"Output directory must not be inside the source directory: {output_dir}",
            file=sys.stderr,
        )
        return 2

    class_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if args.limit_classes is not None:
        class_dirs = class_dirs[: args.limit_classes]

    if not class_dirs:
        print(f"No class directories found in: {source_dir}", file=sys.stderr)
        return 2

    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Class directories: {len(class_dirs)}")
    print(f"Rule: short_side >= {args.threshold}px")
    print(f"Workers: {args.workers}")
    print(f"Dry run: {args.dry_run}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Keep class folders: True")

    results = collect_results(
        class_dirs=class_dirs,
        source_dir=source_dir,
        output_dir=output_dir,
        threshold=args.threshold,
        extensions=extensions,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        create_empty_class_dirs=args.create_empty_class_dirs,
        workers=args.workers,
        quiet=args.quiet,
    )

    summary = build_summary(
        source_dir=source_dir,
        output_dir=output_dir,
        threshold=args.threshold,
        extensions=extensions,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        create_empty_class_dirs=args.create_empty_class_dirs,
        results=results,
    )

    if not args.dry_run:
        write_manifest_csv(output_dir / "s3_copy_manifest.csv", results)
        write_summary_json(output_dir / "s3_copy_summary.json", summary)

    print("\nSummary:")
    print(f"Scanned images: {summary['scanned_images']}")
    print(f"Selected images: {summary['selected_images']}")
    print(f"Copied images: {summary['copied_images']}")
    print(f"Skipped small images: {summary['skipped_small_images']}")
    print(f"Skipped existing images: {summary['skipped_existing_images']}")
    print(f"Failed images: {summary['failed_images']}")
    print(f"Classes with selected images: {summary['classes_with_selected_images']}")

    if args.dry_run:
        print("\nDry run finished. No folders were created and no images were copied.")
    else:
        print(f"\nManifest CSV: {output_dir / 's3_copy_manifest.csv'}")
        print(f"Summary JSON: {output_dir / 's3_copy_summary.json'}")

    return 0 if summary["failed_images"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
