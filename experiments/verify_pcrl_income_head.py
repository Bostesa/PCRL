#!/usr/bin/env python3
"""Verify the trained PCRL Adult task head matches the paper's reported
income accuracy.

Loads ``checkpoints/fix_pcrl_adult_0/best.pt`` (the seed-0 PCRL run from
``adult_seeds_fixed.csv``) and runs the income_prediction task head on
the Adult test set under purpose 0. Compares against the 76.3% +/- 0.2%
reported in Table 3 of the paper.

If the head is collapsed (predicts the majority class on every input),
report it explicitly so we know the checkpoint or eval path is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.adult import AdultDataset, get_adult_purposes  # noqa: E402
from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.models.encoder import PurposeConditionedEncoder  # noqa: E402
from pcrl.models.task_head import TaskHead  # noqa: E402

PAPER_VALUE = 0.763  # PCRL Adult income accuracy, Table 3


def load_head_from_flat(state_dict: dict, prefix: str, repr_dim: int, output_dim: int) -> TaskHead:
    head = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
    sub = {}
    p = f"{prefix}."
    for k, v in state_dict.items():
        if k.startswith(p):
            sub[k[len(p):]] = v
    if not sub:
        # try as nested dict instead of flat
        if prefix in state_dict and isinstance(state_dict[prefix], dict):
            head.load_state_dict(state_dict[prefix])
            head.eval()
            return head
        raise KeyError(f"no keys with prefix {p!r} in checkpoint task_heads")
    head.load_state_dict(sub)
    head.eval()
    return head


def evaluate(ckpt_path: Path) -> tuple[float, float]:
    """Returns (head_accuracy, fresh_lr_probe_accuracy)."""
    purposes = get_adult_purposes()
    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=False)
    test_ds = AdultDataset(
        purposes=purposes, root="data", split="test", download=False,
        norm_stats=train_ds.norm_stats,
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False,
                              collate_fn=collate_pcrl_batch, num_workers=0)

    encoder = PurposeConditionedEncoder(
        input_dim=train_ds.info.num_features,
        hidden_dims=[128, 128], repr_dim=64,
        num_purposes=len(purposes), purpose_emb_dim=32,
        conditioning="film", dropout=0.3,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    income_head = load_head_from_flat(ckpt["task_heads"], "income_prediction", 64, 2)

    inc_test = test_ds.task_labels["income"].numpy()

    # Trained-head accuracy on purpose-0 representations.
    @torch.no_grad()
    def reps_purpose(loader, idx):
        out = []
        for b in loader:
            out.append(encoder(b["features"], idx).numpy())
        return np.concatenate(out)

    train_p0 = reps_purpose(train_loader, 0)
    test_p0 = reps_purpose(test_loader, 0)

    with torch.no_grad():
        logits = income_head(torch.from_numpy(test_p0))
        preds = logits.argmax(dim=-1).numpy()
    head_acc = float((preds == inc_test).mean())

    # Fresh-LR probe for sanity check.
    from sklearn.linear_model import LogisticRegression
    inc_train = train_ds.task_labels["income"].numpy()
    probe_acc = float(
        LogisticRegression(max_iter=2000, random_state=42)
        .fit(train_p0, inc_train).score(test_p0, inc_test)
    )

    return head_acc, probe_acc


def main() -> None:
    candidates = [
        ROOT / "checkpoints" / "fix_pcrl_adult_0" / "best.pt",
        ROOT / "checkpoints" / "adult" / "best.pt",
    ]

    print()
    for ckpt_path in candidates:
        print(f"Checking: {ckpt_path}")
        if not ckpt_path.exists():
            print("  (not present locally — skipping)")
            print()
            continue
        head_acc, probe_acc = evaluate(ckpt_path)
        match = "yes" if abs(head_acc - PAPER_VALUE) < 0.01 else "no"
        print(f"Loaded checkpoint:        {ckpt_path}")
        print(f"Task head income accuracy: {head_acc:.1%}")
        print(f"Fresh-LR probe accuracy:   {probe_acc:.1%}  (lower bound on usable signal)")
        print(f"Paper reported value:      {PAPER_VALUE:.1%}")
        print(f"Match (within 1pp):        {match}")
        print()


if __name__ == "__main__":
    main()
