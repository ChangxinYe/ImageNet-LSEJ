"""
@file: lsej_dataloader.py
@description:
    【中文】
    ImageNet-LSEJ 官方 PyTorch Dataset 实现。用户只需指定数据集根目录、数据划分和
    任务名称，即可获得按照官方配置生成并使用官方固定排列打乱的拼图piece。

    Dataset严格读取发布的splits、configs和permutations文件，不会自行重新划分数据、
    重新生成排列或修改源图片。单个样本的处理流程如下：

    1. 根据splits/{split}.csv读取1000x1000 PNG源图片；
    2. 读取configs/{task}.json。10x10任务使用Lanczos将整图缩放至500x500，
       20x20任务保持1000x1000不变；
    3. 按行优先顺序切成100或400个原始50x50 piece；
    4. 从每个piece四周裁去2px、5px或8px，再使用Lanczos恢复至50x50；
    5. 根据lsej_id读取官方NPZ排列，并执行
       shuffled_pieces = original_pieces[permutation]；
    6. 返回打乱后的pieces、官方排列以及可追溯的样本元数据。

    默认将图像转换为范围[0, 1]的float32张量，形状为[N, 3, 50, 50]。当
    normalize=False时返回uint8张量。官方排列本身可直接作为“打乱后每个位置对应的
    原始位置”标签，即permutation[i]表示打乱后的第i个piece应该放回的原始位置。

    【English】
    Official PyTorch Dataset implementation for ImageNet-LSEJ. Users only need
    to specify the dataset root, split, and task name to obtain puzzle pieces
    produced by the official preprocessing pipeline and shuffled by the official
    fixed permutations.

    The Dataset strictly reads the published split, config, and permutation files.
    It never recreates splits or permutations and never modifies source images.
    For each sample, it:

    1. Loads a 1000x1000 PNG according to splits/{split}.csv;
    2. Reads configs/{task}.json, resizing the full image to 500x500 with Lanczos
       for 10x10 tasks and retaining 1000x1000 for 20x20 tasks;
    3. Splits the image into 100 or 400 original 50x50 pieces in row-major order;
    4. Crops 2px, 5px, or 8px from every side of each piece and resizes the
       remaining center region back to 50x50 with Lanczos;
    5. Loads the official permutation by lsej_id and applies
       shuffled_pieces = original_pieces[permutation];
    6. Returns shuffled pieces, the official permutation, and traceable sample
       metadata.

    By default, pieces are float32 tensors in [0, 1] with shape [N, 3, 50, 50].
    With normalize=False, uint8 tensors are returned. permutation is the target
    target: permutation[i] is the original position to which shuffled piece i
    should be restored.
@author: Changxin Ye
@created: 2026-07-11
@version: 1.0
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SplitName = Literal["train", "val", "test"]
LANCZOS = Image.Resampling.LANCZOS


class ImageNetLSEJDataset(Dataset[dict[str, Any]]):
    """Load an official ImageNet-LSEJ split and task."""

    def __init__(
        self,
        root: str | Path,
        split: SplitName = "train",
        task: str = "grid10_erode2",
        normalize: bool = True,
        verify_images: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.task = task
        self.normalize = normalize

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.config_path = self._resolve_task_config(task)
        self.config = self._load_json(self.config_path)
        self._validate_config()
        self.records = self._load_split_records()
        self.permutations = self._load_split_permutations()

        if verify_images:
            missing = [
                record["path"]
                for record in self.records
                if not (self.root / record["path"]).is_file()
            ]
            if missing:
                preview = ", ".join(missing[:5])
                raise FileNotFoundError(
                    f"Missing {len(missing)} source images; first entries: {preview}"
                )

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Required JSON file does not exist: {path}")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _resolve_task_config(self, task: str) -> Path:
        index_path = self.root / "configs" / "tasks.json"
        index = self._load_json(index_path)
        tasks = index.get("tasks", {})
        if task not in tasks:
            available = ", ".join(sorted(tasks))
            raise ValueError(f"Unknown task '{task}'. Available tasks: {available}")
        config_path = (index_path.parent / tasks[task]).resolve()
        try:
            config_path.relative_to(index_path.parent.resolve())
        except ValueError as exc:
            raise ValueError("Task config path escapes the configs directory.") from exc
        return config_path

    def _validate_config(self) -> None:
        config = self.config
        if config.get("task_id") != self.task:
            raise ValueError("Task ID does not match the requested task.")
        puzzle = config["puzzle"]
        rows = int(puzzle["grid_rows"])
        columns = int(puzzle["grid_cols"])
        piece_height, piece_width = map(int, puzzle["piece_size"])
        working_height, working_width = map(
            int, config["image_preprocessing"]["working_image_size"]
        )
        if rows != columns:
            raise ValueError("Only square puzzle grids are supported.")
        if piece_height != 50 or piece_width != 50:
            raise ValueError("Official ImageNet-LSEJ pieces must be 50x50.")
        if working_height != rows * piece_height or working_width != columns * piece_width:
            raise ValueError("Working image size is inconsistent with grid and piece size.")
        if int(puzzle["piece_count"]) != rows * columns:
            raise ValueError("piece_count is inconsistent with grid dimensions.")

        erosion = config["erosion"]
        pixels = int(erosion["pixels_per_side"])
        expected_crop = piece_height - 2 * pixels
        if erosion["method"] != "border_crop_and_resize":
            raise ValueError("Unsupported erosion method.")
        if list(map(int, erosion["cropped_piece_size"])) != [expected_crop] * 2:
            raise ValueError("Erosion crop size is inconsistent with pixels_per_side.")
        if expected_crop <= 0:
            raise ValueError("Erosion removes the entire piece.")

        permutation = config["permutation"]
        expected_definition = "shuffled_pieces = original_pieces[permutation]"
        if permutation["definition"] != expected_definition:
            raise ValueError("Unsupported permutation definition.")

    def _load_split_records(self) -> list[dict[str, str]]:
        split_path = self.root / "splits" / f"{self.split}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file does not exist: {split_path}")
        with split_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            required = {
                "lsej_id", "path", "split", "imagenet_image_id",
                "imagenet_class_id",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Split CSV is missing columns: {sorted(missing)}")
            records = list(reader)
        if not records:
            raise ValueError(f"Split contains no records: {split_path}")
        if any(record["split"] != self.split for record in records):
            raise ValueError(f"Split CSV contains records outside '{self.split}'.")
        ids = [record["lsej_id"] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate lsej_id values found in {split_path}.")
        return records

    def _load_split_permutations(self) -> np.ndarray:
        permutation_config = self.config["permutation"]
        permutation_path = (
            self.config_path.parent / permutation_config["file"]
        ).resolve()
        if not permutation_path.is_file():
            raise FileNotFoundError(
                f"Permutation file does not exist: {permutation_path}"
            )
        with np.load(permutation_path, allow_pickle=False) as archive:
            id_key = permutation_config["id_array"]
            permutation_key = permutation_config["array"]
            if id_key not in archive or permutation_key not in archive:
                raise ValueError("Permutation NPZ is missing required arrays.")
            official_ids = archive[id_key].astype(str)
            official_permutations = archive[permutation_key]

        if len(official_ids) != len(official_permutations):
            raise ValueError("Permutation IDs and rows have different lengths.")
        id_to_row = {lsej_id: index for index, lsej_id in enumerate(official_ids)}
        if len(id_to_row) != len(official_ids):
            raise ValueError("Permutation file contains duplicate lsej_id values.")
        try:
            row_indices = [id_to_row[record["lsej_id"]] for record in self.records]
        except KeyError as exc:
            raise ValueError(f"No official permutation for lsej_id {exc.args[0]}.") from exc
        selected = official_permutations[row_indices]
        expected_shape = (len(self.records), int(self.config["puzzle"]["piece_count"]))
        if selected.shape != expected_shape:
            raise ValueError(
                f"Unexpected permutation shape {selected.shape}; expected {expected_shape}."
            )
        return selected.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.records)

    def _load_and_prepare_image(self, image_path: Path) -> Image.Image:
        source_size = tuple(map(int, self.config["source"]["image_size"]))
        working_size = tuple(
            map(int, self.config["image_preprocessing"]["working_image_size"])
        )
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != source_size:
                raise ValueError(
                    f"Expected source image {source_size}, got {image.size}: {image_path}"
                )
            if image.size != working_size:
                image = image.resize(working_size, resample=LANCZOS)
            return image.copy()

    def _make_original_pieces(self, image: Image.Image) -> torch.Tensor:
        puzzle = self.config["puzzle"]
        grid_rows = int(puzzle["grid_rows"])
        grid_columns = int(puzzle["grid_cols"])
        piece_height, piece_width = map(int, puzzle["piece_size"])
        erosion_pixels = int(self.config["erosion"]["pixels_per_side"])
        output_height, output_width = map(
            int, self.config["erosion"]["output_piece_size"]
        )

        pieces: list[np.ndarray] = []
        for row in range(grid_rows):
            for column in range(grid_columns):
                left = column * piece_width
                top = row * piece_height
                piece = image.crop(
                    (left, top, left + piece_width, top + piece_height)
                )
                piece = piece.crop(
                    (
                        erosion_pixels,
                        erosion_pixels,
                        piece_width - erosion_pixels,
                        piece_height - erosion_pixels,
                    )
                )
                piece = piece.resize((output_width, output_height), resample=LANCZOS)
                # copy=True makes the array writable before torch conversion.
                pieces.append(np.array(piece, dtype=np.uint8, copy=True))

        array = np.stack(pieces, axis=0)
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
        if self.normalize:
            tensor = tensor.to(torch.float32).div_(255.0)
        return tensor

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = (self.root / record["path"]).resolve()
        try:
            image_path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Image path escapes dataset root: {record['path']}") from exc
        if not image_path.is_file():
            raise FileNotFoundError(f"Source image does not exist: {image_path}")

        image = self._load_and_prepare_image(image_path)
        original_pieces = self._make_original_pieces(image)
        permutation = torch.from_numpy(self.permutations[index].copy()).long()
        shuffled_pieces = original_pieces.index_select(0, permutation)

        return {
            "pieces": shuffled_pieces,
            "permutation": permutation,
            "lsej_id": record["lsej_id"],
            "imagenet_image_id": record["imagenet_image_id"],
            "imagenet_class_id": record["imagenet_class_id"],
            "split": self.split,
            "task": self.task,
            "image_path": record["path"],
        }


def list_official_tasks(root: str | Path) -> list[str]:
    """Return all official task IDs published in configs/tasks.json."""
    index_path = Path(root).expanduser().resolve() / "configs" / "tasks.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Task index does not exist: {index_path}")
    with index_path.open("r", encoding="utf-8") as file:
        index = json.load(file)
    return sorted(index.get("tasks", {}).keys())


__all__ = ["ImageNetLSEJDataset", "list_official_tasks"]
