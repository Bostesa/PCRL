"""CelebA-medium dataset: 60k stratified subsample of the 162,770-image train
split, 128x128, ImageNet-normalized.

Stratification key: joint distribution of ``Male`` x ``Young`` (4 cells).

Reads CSVs from ``data/celeba/`` (the user already extracted the archive
locally). On AWS, the same layout is expected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CELEBA_DIR = ROOT / "data" / "celeba"

# CelebA attributes are stored as -1/+1 in the CSV. We map +1 -> 1, -1 -> 0.
ATTR_MALE = "Male"
ATTR_YOUNG = "Young"
ATTR_SMILING = "Smiling"


def _imagenet_train_transform(image_size: int = 128) -> Callable:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])


def _imagenet_eval_transform(image_size: int = 128) -> Callable:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])


def _load_attr_partition(celeba_dir: Path) -> pd.DataFrame:
    attr = pd.read_csv(celeba_dir / "list_attr_celeba.csv")
    part = pd.read_csv(celeba_dir / "list_eval_partition.csv")
    df = attr.merge(part, on="image_id", how="inner")
    # +1 -> 1, -1 -> 0 for binary columns.
    for col in (ATTR_MALE, ATTR_YOUNG, ATTR_SMILING):
        df[col] = (df[col] > 0).astype(np.int64)
    return df


def stratified_subsample(
    df: pd.DataFrame, n: int, seed: int = 0
) -> pd.DataFrame:
    """Stratified subsample on (Male, Young) joint distribution."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_strat"] = df[ATTR_MALE].astype(int) * 2 + df[ATTR_YOUNG].astype(int)
    counts = df["_strat"].value_counts(normalize=True).sort_index()
    out_parts: list[pd.DataFrame] = []
    remaining = n
    cells = sorted(counts.index.tolist())
    for i, cell in enumerate(cells):
        if i == len(cells) - 1:
            take = remaining
        else:
            take = int(round(n * counts.loc[cell]))
            take = min(take, remaining)
        cell_df = df[df["_strat"] == cell]
        if take > len(cell_df):
            take = len(cell_df)
        idx = rng.choice(len(cell_df), size=take, replace=False)
        out_parts.append(cell_df.iloc[idx])
        remaining -= take
    out = pd.concat(out_parts, axis=0).sample(frac=1.0, random_state=seed)
    return out.drop(columns=["_strat"]).reset_index(drop=True)


class CelebAMedium(Dataset):
    """60k stratified subsample of CelebA train split.

    Returns dict per item:
      - ``image``: float tensor (3, H, W), ImageNet-normalized
      - ``smiling``: int (target task)
      - ``male``: int (sensitive)
      - ``young``: int (sensitive)
    """

    def __init__(
        self,
        celeba_dir: str | Path = DEFAULT_CELEBA_DIR,
        n: int = 60_000,
        partition: int = 0,
        image_size: int = 128,
        train: bool = True,
        seed: int = 0,
    ) -> None:
        celeba_dir = Path(celeba_dir)
        df = _load_attr_partition(celeba_dir)
        df = df[df["partition"] == partition].reset_index(drop=True)
        if n is not None and n < len(df):
            df = stratified_subsample(df, n=n, seed=seed)
        self.df = df
        self.celeba_dir = celeba_dir
        # Two-level nesting: data/celeba/img_align_celeba/img_align_celeba/*.jpg
        candidate1 = celeba_dir / "img_align_celeba" / "img_align_celeba"
        candidate2 = celeba_dir / "img_align_celeba"
        self.img_dir = candidate1 if candidate1.is_dir() else candidate2
        self.transform = (
            _imagenet_train_transform(image_size)
            if train
            else _imagenet_eval_transform(image_size)
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = self.img_dir / row["image_id"]
        with Image.open(path) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return {
            "image": x,
            "smiling": int(row[ATTR_SMILING]),
            "male": int(row[ATTR_MALE]),
            "young": int(row[ATTR_YOUNG]),
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    smiling = torch.tensor([b["smiling"] for b in batch], dtype=torch.long)
    male = torch.tensor([b["male"] for b in batch], dtype=torch.long)
    young = torch.tensor([b["young"] for b in batch], dtype=torch.long)
    return {"image": images, "smiling": smiling, "male": male, "young": young}
