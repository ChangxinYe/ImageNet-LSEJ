"""
@file: s1_create_dataset_splits.py
@description:
    【中文】
    本脚本用于为 ImageNet-LSEJ 的 12,000 张源图片创建可复现的数据集划分，流程如下：

    1. 读取脚本同级 images 文件夹中的 PNG 图片，并从文件名解析 ImageNet 图片 ID
       和类别 ID，例如 n01440764_10281.png 对应类别 n01440764。
    2. 按原始文件名排序，为每张图片生成稳定的 LSEJ ID（如 LSEJ_000001）。编号在
       随机划分前生成，因此修改随机种子不会改变图片身份。
    3. 按 ImageNet 类别整理图片，根据 75% / 8.33% / 16.67% 的目标比例计算每个
       类别在 train、val、test 中的分配配额。
    4. 对整数配额产生的余数进行全局校正，严格保证 train、val、test 的图片数量分别
       为 9,000、1,000 和 2,000；少样本类别不强制同时出现在三个集合中。
    5. 使用固定随机种子在每个类别内部打乱图片，并按照计算后的配额完成分层划分。
    6. 写出前检查图片数量、文件名格式、ID 与路径重名、集合数量以及集合间重叠。
    7. 生成 splits/train.csv、val.csv、test.csv、all.csv 和 summary.json。

    脚本只记录数据划分，不会移动、复制、重命名或修改 images 中的任何源图片。
    本脚本仅用于 ImageNet-LSEJ 数据集的制作与首次划分。数据集发布后，使用者应直接
    读取已经生成的 splits 文件，无需运行或修改本脚本。

    【English】
    This script creates reproducible splits for the 12,000 source images in
    ImageNet-LSEJ. The workflow is as follows:

    1. Read PNG images from the images directory next to this script and parse
       the ImageNet image ID and class ID from each file name. For example,
       n01440764_10281.png belongs to class n01440764.
    2. Sort images by their original file names and assign each image a stable
       LSEJ ID, such as LSEJ_000001. IDs are assigned before random splitting,
       so changing the random seed does not change image identities.
    3. Group images by ImageNet class and calculate per-class quotas using the
       target train/val/test ratios of 75% / 8.33% / 16.67%.
    4. Apply global remainder correction to guarantee exactly 9,000 training,
       1,000 validation, and 2,000 test images. Classes with very few images are
       not required to appear in all three splits.
    5. Shuffle images within each class using a fixed random seed and assign
       them to train, val, and test according to the calculated quotas.
    6. Before writing files, validate image counts, file-name formats, duplicate
       IDs and paths, split sizes, and overlap between splits.
    7. Generate splits/train.csv, val.csv, test.csv, all.csv, and summary.json.

    The script only records dataset splits. It does not move, copy, rename, or
    modify any source image in the images directory. This script is intended
    only for constructing and initially splitting ImageNet-LSEJ. Dataset users
    should directly use the published split files and do not need to run or
    modify this script.
@author: Changxin Ye
@created: 2026-07-11
@version: 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_COUNTS = {"train": 9000, "val": 1000, "test": 2000}
CLASS_PATTERN = re.compile(r"^(n\d{8})_.+$")
FIELDNAMES = [
    "lsej_id",
    "path",
    "split",
    "imagenet_image_id",
    "imagenet_class_id",
]


def default_images_dir() -> Path:
    return Path(__file__).resolve().parent / "images"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "splits"


def load_records(images_dir: Path) -> list[dict[str, str]]:
    image_paths = sorted(images_dir.glob("*.png"), key=lambda path: path.name)
    if len(image_paths) != len({path.name for path in image_paths}):
        raise ValueError("Duplicate PNG file names were found.")

    records: list[dict[str, str]] = []
    id_width = max(6, len(str(len(image_paths))))
    for index, image_path in enumerate(image_paths, start=1):
        image_id = image_path.stem
        match = CLASS_PATTERN.fullmatch(image_id)
        if match is None:
            raise ValueError(
                f"Image name does not match n########_... format: {image_path.name}"
            )
        records.append(
            {
                "lsej_id": f"LSEJ_{index:0{id_width}d}",
                "path": f"images/{image_path.name}",
                "split": "",
                "imagenet_image_id": image_id,
                "imagenet_class_id": match.group(1),
            }
        )
    return records


def allocate_class_quotas(
    class_sizes: dict[str, int], target_counts: dict[str, int]
) -> dict[str, dict[str, int]]:
    total = sum(class_sizes.values())
    splits = tuple(target_counts)
    ratios = {split: target_counts[split] / total for split in splits}
    quotas: dict[str, dict[str, int]] = {}
    remaining_by_class: dict[str, int] = {}

    for class_id, size in class_sizes.items():
        quotas[class_id] = {
            split: int(size * ratios[split]) for split in splits
        }
        remaining_by_class[class_id] = size - sum(quotas[class_id].values())

    deficits = {
        split: target_counts[split]
        - sum(quotas[class_id][split] for class_id in class_sizes)
        for split in splits
    }

    while sum(remaining_by_class.values()) > 0:
        best: tuple[float, str, str] | None = None
        for class_id, remaining in remaining_by_class.items():
            if remaining <= 0:
                continue
            size = class_sizes[class_id]
            for split in splits:
                if deficits[split] <= 0:
                    continue
                ideal = size * ratios[split]
                # Higher value means this class/split pair is further below its
                # ideal proportional quota. Class and split names break ties.
                priority = ideal - quotas[class_id][split]
                candidate = (priority, class_id, split)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            raise RuntimeError("Unable to satisfy exact split targets.")
        _, class_id, split = best
        quotas[class_id][split] += 1
        remaining_by_class[class_id] -= 1
        deficits[split] -= 1

    if any(deficit != 0 for deficit in deficits.values()):
        raise RuntimeError(f"Unresolved split deficits: {deficits}")
    return quotas


def stratified_split(
    records: list[dict[str, str]], target_counts: dict[str, int], seed: int
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["imagenet_class_id"]].append(record)

    quotas = allocate_class_quotas(
        {class_id: len(items) for class_id, items in grouped.items()}, target_counts
    )
    rng = random.Random(seed)
    for class_id in sorted(grouped):
        items = grouped[class_id]
        rng.shuffle(items)
        offset = 0
        for split in target_counts:
            count = quotas[class_id][split]
            for record in items[offset : offset + count]:
                record["split"] = split
            offset += count
        if offset != len(items):
            raise RuntimeError(f"Not all images assigned for class {class_id}.")
    return records


def validate(records: list[dict[str, str]], target_counts: dict[str, int]) -> None:
    ids = [record["lsej_id"] for record in records]
    paths = [record["path"] for record in records]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError("Duplicate IDs or paths found after splitting.")
    actual = {
        split: sum(record["split"] == split for record in records)
        for split in target_counts
    }
    if actual != target_counts:
        raise ValueError(f"Split counts mismatch: expected {target_counts}, got {actual}")


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible stratified ImageNet-LSEJ dataset splits."
    )
    parser.add_argument("--images-dir", type=Path, default=default_images_dir())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--train-count", type=int, default=9000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report without writing files."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()
    target_counts = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }
    if not images_dir.is_dir():
        print(f"Images directory does not exist: {images_dir}", file=sys.stderr)
        return 2
    if any(count < 0 for count in target_counts.values()):
        print("Split counts must be non-negative.", file=sys.stderr)
        return 2

    try:
        records = load_records(images_dir)
        expected_total = sum(target_counts.values())
        if len(records) != expected_total:
            raise ValueError(
                f"Expected {expected_total} PNG images, found {len(records)}."
            )
        stratified_split(records, target_counts, args.seed)
        validate(records, target_counts)
    except (ValueError, RuntimeError) as exc:
        print(f"Split creation failed: {exc}", file=sys.stderr)
        return 2

    class_counts = {
        split: len(
            {
                record["imagenet_class_id"]
                for record in records
                if record["split"] == split
            }
        )
        for split in target_counts
    }
    print(f"Images directory: {images_dir}")
    print(f"Random seed: {args.seed}")
    for split, count in target_counts.items():
        print(f"{split}: {count} images, {class_counts[split]} classes")

    if args.dry_run:
        print("Dry run finished. No split files were written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in target_counts:
        split_records = sorted(
            (record for record in records if record["split"] == split),
            key=lambda record: record["lsej_id"],
        )
        write_csv(output_dir / f"{split}.csv", split_records)
    write_csv(
        output_dir / "all.csv",
        sorted(records, key=lambda record: record["lsej_id"]),
    )

    summary: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": args.seed,
        "method": "exact-size stratified random split with global remainder correction",
        "total_images": len(records),
        "split_counts": target_counts,
        "class_counts": class_counts,
        "csv_fields": FIELDNAMES,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Split files written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
