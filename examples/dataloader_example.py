"""Minimal, directly runnable example for the official ImageNet-LSEJ DataLoader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader


# Make the dataset module importable when this file is run directly from examples/.
DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from lsej_dataloader import ImageNetLSEJDataset, list_official_tasks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageNet-LSEJ DataLoader example")
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--task", default="grid10_erode2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    print("Official tasks:")
    for task_name in list_official_tasks(root):
        print(f"  - {task_name}")

    dataset = ImageNetLSEJDataset(
        root=root,
        split=args.split,
        task=args.task,
    )

    sample = dataset[0]
    print("\nSingle sample:")
    print(f"  dataset length:       {len(dataset)}")
    print(f"  lsej_id:              {sample['lsej_id']}")
    print(f"  ImageNet class:       {sample['imagenet_class_id']}")
    print(f"  pieces shape:         {tuple(sample['pieces'].shape)}")
    print(f"  pieces dtype:         {sample['pieces'].dtype}")
    print(f"  permutation shape:    {tuple(sample['permutation'].shape)}")
    print(f"  first 10 labels:      {sample['permutation'][:10].tolist()}")

    # DataLoader shuffle changes only the order of image samples. It does not
    # change the official piece permutation stored for each image.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    batch = next(iter(loader))

    print("\nOne batch:")
    print(f"  pieces shape:         {tuple(batch['pieces'].shape)}")
    print(f"  permutation shape:    {tuple(batch['permutation'].shape)}")
    print(f"  lsej_ids:             {list(batch['lsej_id'])}")

    print("\nLabel convention:")
    print("  shuffled_pieces = original_pieces[permutation]")
    print("  permutation[i] is the original position of shuffled piece i.")


if __name__ == "__main__":
    main()
