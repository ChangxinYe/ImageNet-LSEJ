"""
@file: s2_prepare_official_tasks.py
@description:
    【中文】
    本脚本用于生成 ImageNet-LSEJ 的六种官方任务配置和固定拼图排列，保证不同使用者
    能以统一的预处理规则和完全相同的打乱顺序训练、验证和测试。处理流程如下：

    1. 读取 splits/all.csv，并按 lsej_id 排序，确保排列与稳定的数据集 ID 一一对应。
    2. 为每张图片生成一套 10 x 10 排列（100个piece）和一套 20 x 20 排列
       （400个piece）。同一网格下的2px、5px和8px侵蚀任务共享同一套排列，使不同
       侵蚀强度之间的比较只改变侵蚀条件，不改变piece顺序。
    3. 原始piece采用行优先编号，排列定义为：
       shuffled_pieces = original_pieces[permutation]。
       因此 permutation[i] 表示打乱后第i个位置所放置的原始piece编号。
    4. 使用固定随机种子生成排列，并将排列数组本身保存为官方数据。数据集使用者应
       直接读取发布的排列文件，而不是依赖某个随机数库和种子重新生成。
    5. 生成10x10、20x20与2px、5px、8px侵蚀组成的六份JSON配置。侵蚀定义为
       从50x50 piece四周裁去指定像素，再使用Lanczos缩放回50x50。
    6. 输出configs任务配置、permutations排列文件和对应metadata，并在写出前检查
       ID唯一性、排列有效性及配置字段之间的一致性。

    本脚本仅用于制作 ImageNet-LSEJ 官方数据。数据集发布后，普通使用者无需运行或
    修改本脚本，只需读取已提供的官方排列文件。研究者可以自行采用动态排列作为额外
    训练增强，但应在实验报告中明确说明其未使用官方固定训练排列。

    【English】
    This script generates the six official task configurations and fixed
    permutations for ImageNet-LSEJ, ensuring identical preprocessing and puzzle
    orders across users and baselines. The workflow is as follows:

    1. Read splits/all.csv and sort records by lsej_id, creating a deterministic
       one-to-one mapping between dataset IDs and permutation rows.
    2. Generate one 10 x 10 permutation (100 pieces) and one 20 x 20 permutation
       (400 pieces) for every image. The 2px, 5px, and 8px erosion tasks of the
       same grid share the same permutation, isolating erosion as the only
       changing experimental condition.
    3. Original pieces use row-major indexing. The permutation convention is:
       shuffled_pieces = original_pieces[permutation]. Therefore permutation[i]
       is the original piece index placed at shuffled position i.
    4. Generate permutations with a fixed seed, but publish the arrays themselves.
       Dataset users must read the official arrays instead of regenerating them
       from a seed with a potentially different random-number implementation.
    5. Generate six JSON configurations from the 10x10 and 20x20 grids and the
       2px, 5px, and 8px erosion levels. Erosion crops the specified border from
       each 50x50 piece and resizes the remainder back to 50x50 with Lanczos.
    6. Write task configs, permutation files, and metadata after validating IDs,
       permutations, and cross-file configuration consistency.

    This script is only used to construct the official ImageNet-LSEJ dataset.
    Dataset users do not need to run or modify it. Researchers may use dynamic
    permutations as optional training augmentation, but should clearly report
    that they did not use the official fixed training permutations.
@author: Changxin Ye
@created: 2026-07-11
@version: 1.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


GRID_SIZES = (10, 20)
EROSION_PIXELS = (2, 5, 8)
SOURCE_IMAGE_SIZE = 1000
PIECE_SIZE = 50
DEFAULT_SEED = 42


def dataset_root() -> Path:
    return Path(__file__).resolve().parent


def default_split_file() -> Path:
    return dataset_root() / "splits" / "all.csv"


def default_permutations_dir() -> Path:
    return dataset_root() / "permutations"


def default_configs_dir() -> Path:
    return dataset_root() / "configs"


def build_task_config(grid_size: int, erosion_pixels: int) -> dict[str, Any]:
    working_size = grid_size * PIECE_SIZE
    cropped_piece_size = PIECE_SIZE - 2 * erosion_pixels
    if cropped_piece_size <= 0:
        raise ValueError("Erosion removes the entire piece.")
    return {
        "dataset": "ImageNet-LSEJ",
        "version": "1.0",
        "task_id": f"grid{grid_size}_erode{erosion_pixels}",
        "source": {
            "image_size": [SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE],
            "color_mode": "RGB",
        },
        "puzzle": {
            "grid_rows": grid_size,
            "grid_cols": grid_size,
            "piece_count": grid_size * grid_size,
            "piece_size": [PIECE_SIZE, PIECE_SIZE],
            "piece_indexing": "row-major, zero-based",
        },
        "image_preprocessing": {
            "working_image_size": [working_size, working_size],
            "resize_required": working_size != SOURCE_IMAGE_SIZE,
            "resize_method": (
                "lanczos" if working_size != SOURCE_IMAGE_SIZE else None
            ),
        },
        "erosion": {
            "method": "border_crop_and_resize",
            "pixels_per_side": erosion_pixels,
            "cropped_piece_size": [cropped_piece_size, cropped_piece_size],
            "output_piece_size": [PIECE_SIZE, PIECE_SIZE],
            "resize_method": "lanczos",
        },
        "permutation": {
            "file": f"../permutations/grid{grid_size}_permutations.npz",
            "array": "permutations",
            "id_array": "lsej_ids",
            "definition": "shuffled_pieces = original_pieces[permutation]",
            "shared_across_erosion_levels": True,
        },
    }


def validate_task_config(config: dict[str, Any]) -> None:
    puzzle = config["puzzle"]
    preprocessing = config["image_preprocessing"]
    erosion = config["erosion"]
    grid_size = puzzle["grid_rows"]
    if puzzle["grid_cols"] != grid_size:
        raise ValueError("Only square grids are supported.")
    if puzzle["piece_count"] != grid_size * grid_size:
        raise ValueError("piece_count is inconsistent with grid dimensions.")
    if preprocessing["working_image_size"] != [grid_size * PIECE_SIZE] * 2:
        raise ValueError("working_image_size is inconsistent with grid and piece size.")
    expected_crop = PIECE_SIZE - 2 * erosion["pixels_per_side"]
    if erosion["cropped_piece_size"] != [expected_crop, expected_crop]:
        raise ValueError("cropped_piece_size is inconsistent with erosion pixels.")


def load_records(split_file: Path) -> list[dict[str, str]]:
    with split_file.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"lsej_id", "split"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        records = list(reader)

    if not records:
        raise ValueError("Split CSV contains no records.")
    records.sort(key=lambda record: record["lsej_id"])
    ids = [record["lsej_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate lsej_id values found in split CSV.")
    return records


def derive_grid_seed(base_seed: int, grid_size: int) -> int:
    """Derive independent, stable seeds for different puzzle grids."""
    digest = hashlib.sha256(f"ImageNet-LSEJ:{base_seed}:grid{grid_size}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def generate_permutations(
    image_count: int, grid_size: int, base_seed: int
) -> np.ndarray:
    piece_count = grid_size * grid_size
    rng = np.random.default_rng(derive_grid_seed(base_seed, grid_size))
    dtype = np.uint8 if piece_count <= 256 else np.uint16
    permutations = np.empty((image_count, piece_count), dtype=dtype)
    for index in range(image_count):
        permutations[index] = rng.permutation(piece_count)
    return permutations


def validate_permutations(permutations: np.ndarray, grid_size: int) -> None:
    piece_count = grid_size * grid_size
    if permutations.ndim != 2 or permutations.shape[1] != piece_count:
        raise ValueError(
            f"grid{grid_size} has invalid shape: {permutations.shape}"
        )
    expected = np.arange(piece_count, dtype=permutations.dtype)
    # Sorting every row is simple and rigorous; these official arrays are small.
    if not np.all(np.sort(permutations, axis=1) == expected):
        raise ValueError(f"grid{grid_size} contains an invalid permutation.")


def save_npz(
    output_path: Path,
    lsej_ids: np.ndarray,
    splits: np.ndarray,
    permutations: np.ndarray,
) -> None:
    np.savez_compressed(
        output_path,
        lsej_ids=lsej_ids,
        splits=splits,
        permutations=permutations,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate official fixed ImageNet-LSEJ puzzle permutations."
    )
    parser.add_argument("--split-file", type=Path, default=default_split_file())
    parser.add_argument(
        "--configs-dir", type=Path, default=default_configs_dir()
    )
    parser.add_argument(
        "--permutations-dir", type=Path, default=default_permutations_dir()
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate and validate without writing files."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_file = args.split_file.resolve()
    configs_dir = args.configs_dir.resolve()
    permutations_dir = args.permutations_dir.resolve()
    if not split_file.is_file():
        print(f"Split file does not exist: {split_file}", file=sys.stderr)
        return 2

    try:
        records = load_records(split_file)
        lsej_ids = np.asarray([record["lsej_id"] for record in records])
        splits = np.asarray([record["split"] for record in records])
        generated = {
            grid_size: generate_permutations(len(records), grid_size, args.seed)
            for grid_size in GRID_SIZES
        }
        for grid_size, permutations in generated.items():
            validate_permutations(permutations, grid_size)
        task_configs = {
            f"grid{grid_size}_erode{erosion_pixels}": build_task_config(
                grid_size, erosion_pixels
            )
            for grid_size in GRID_SIZES
            for erosion_pixels in EROSION_PIXELS
        }
        for config in task_configs.values():
            validate_task_config(config)
    except (ValueError, OSError) as exc:
        print(f"Permutation generation failed: {exc}", file=sys.stderr)
        return 2

    print(f"Split file: {split_file}")
    print(f"Images: {len(records)}")
    print(f"Base seed: {args.seed}")
    for grid_size, permutations in generated.items():
        print(
            f"grid{grid_size}: shape={permutations.shape}, "
            f"dtype={permutations.dtype}"
        )
    print(f"Task configurations: {len(task_configs)}")

    if args.dry_run:
        print("Dry run finished. No permutation files were written.")
        return 0

    configs_dir.mkdir(parents=True, exist_ok=True)
    permutations_dir.mkdir(parents=True, exist_ok=True)
    for grid_size, permutations in generated.items():
        save_npz(
            permutations_dir / f"grid{grid_size}_permutations.npz",
            lsej_ids,
            splits,
            permutations,
        )

    metadata: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "ImageNet-LSEJ",
        "version": "1.0",
        "image_count": len(records),
        "base_seed": args.seed,
        "piece_indexing": "row-major, zero-based",
        "permutation_definition": (
            "shuffled_pieces = original_pieces[permutation]"
        ),
        "target_definition": (
            "permutation[i] is the original piece index at shuffled position i"
        ),
        "shared_across_erosion_pixels": [2, 5, 8],
        "files": {
            f"grid{grid_size}": {
                "path": f"grid{grid_size}_permutations.npz",
                "piece_count": grid_size * grid_size,
                "arrays": ["lsej_ids", "splits", "permutations"],
                "permutation_shape": [len(records), grid_size * grid_size],
                "permutation_dtype": str(generated[grid_size].dtype),
                "derived_seed": derive_grid_seed(args.seed, grid_size),
            }
            for grid_size in GRID_SIZES
        },
        "usage_note": (
            "Users should load the published permutation arrays instead of "
            "regenerating them from the seed."
        ),
    }
    with (permutations_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    task_index = {
        "dataset": "ImageNet-LSEJ",
        "version": "1.0",
        "default_task": "grid10_erode2",
        "tasks": {
            task_id: f"{task_id}.json" for task_id in task_configs
        },
    }
    for task_id, config in task_configs.items():
        with (configs_dir / f"{task_id}.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
    with (configs_dir / "tasks.json").open("w", encoding="utf-8") as file:
        json.dump(task_index, file, ensure_ascii=False, indent=2)

    print(f"Task configs written to: {configs_dir}")
    print(f"Permutation files written to: {permutations_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
