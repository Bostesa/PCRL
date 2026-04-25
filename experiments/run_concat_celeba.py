#!/usr/bin/env python3
"""FiLM vs Concat conditioning on CelebA (v2 HPs).

We need a clean FiLM-vs-Concat comparison, but the existing CelebA v2
baseline (results/celeba/baseline_v2_fixed.csv) uses a per-purpose
projection head, not FiLM. So this script trains a CNN encoder in BOTH
FiLM and Concat conditioning modes at matched v2 HPs and writes per-pair
results to results/celeba/concat_perpair.csv (and also a film_perpair.csv
reference from this same run).

v2 HPs:
    REPR_DIM=128, CONV_CHANNELS=(32,64,128), BATCH=256
    lambda_adv=0.5, lambda_verify=0.3, epochs=50
    auditor_steps=5, patience=None, warmup=5

Seed=0 only (CelebA is expensive).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pcrl.training.trainer as _trainer_mod


class _QuietTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable) if self.iterable is not None else iter([])
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def set_postfix(self, *args, **kwargs):
        pass
    def update(self, *args):
        pass
    def close(self):
        pass

_trainer_mod.tqdm = _QuietTqdm

from pcrl.data.base import collate_pcrl_batch
from pcrl.data.celeba import CelebADataset, get_celeba_purposes
from pcrl.evaluation.certificates import generate_report
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)


# ═════════════════════════════════════════════════════════════════════════
# CNN encoder with switchable per-layer conditioning (FiLM or Concat)
# ═════════════════════════════════════════════════════════════════════════

class CNNEncoderConditioned(nn.Module):
    """CNN encoder with either FiLM or channel-Concat conditioning per conv.

    FiLM: per-channel gamma/beta predicted from purpose embedding.
    Concat: broadcast purpose embedding to an (embD, H, W) map, concat along
            channel dim, then 1x1 conv projects back to C channels.
    """

    def __init__(
        self,
        repr_dim: int,
        num_purposes: int,
        conditioning: str = "film",
        purpose_emb_dim: int = 32,
        in_channels: int = 3,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        assert conditioning in {"film", "concat"}
        self.repr_dim = repr_dim
        self.num_purposes = num_purposes
        self.conditioning = conditioning
        self.purpose_emb_dim = purpose_emb_dim

        self.purpose_embedding = nn.Embedding(num_purposes, purpose_emb_dim)

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.cond_layers = nn.ModuleList()  # per-layer conditioning modules

        channels = [in_channels] + list(conv_channels)
        for i in range(len(conv_channels)):
            self.conv_layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1)
            )
            out_c = channels[i + 1]
            if conditioning == "film":
                # gamma/beta predicted from purpose embedding, applied per-channel.
                self.cond_layers.append(nn.ModuleDict({
                    "gamma": nn.Linear(purpose_emb_dim, out_c),
                    "beta": nn.Linear(purpose_emb_dim, out_c),
                }))
            else:  # concat
                # 1x1 conv projects (C + embD) -> C after spatial broadcast of embedding.
                self.cond_layers.append(
                    nn.Conv2d(out_c + purpose_emb_dim, out_c, kernel_size=1)
                )
            self.bn_layers.append(nn.BatchNorm2d(out_c))

        # Identity init for FiLM
        if conditioning == "film":
            for layer in self.cond_layers:
                nn.init.zeros_(layer["gamma"].weight)
                nn.init.ones_(layer["gamma"].bias)
                nn.init.zeros_(layer["beta"].weight)
                nn.init.zeros_(layer["beta"].bias)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        spatial = 64
        for _ in range(len(conv_channels)):
            spatial = (spatial + 1) // 2
        self._flat_dim = conv_channels[-1] * spatial * spatial

        proj_hidden = max(256, repr_dim)
        self.repr_proj = nn.Sequential(
            nn.Linear(self._flat_dim, proj_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, repr_dim),
        )

    def _apply_cond(self, h: torch.Tensor, purpose_emb: torch.Tensor, idx: int) -> torch.Tensor:
        if self.conditioning == "film":
            gamma = self.cond_layers[idx]["gamma"](purpose_emb).unsqueeze(-1).unsqueeze(-1)
            beta = self.cond_layers[idx]["beta"](purpose_emb).unsqueeze(-1).unsqueeze(-1)
            return gamma * h + beta
        else:
            # Broadcast purpose embedding to spatial map and concat channels.
            b, _, hh, ww = h.shape
            emb_map = purpose_emb.unsqueeze(-1).unsqueeze(-1).expand(b, self.purpose_emb_dim, hh, ww)
            combined = torch.cat([h, emb_map], dim=1)
            return self.cond_layers[idx](combined)

    def _forward_impl(self, x: torch.Tensor, purpose_emb: torch.Tensor) -> torch.Tensor:
        h = x
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            h = conv(h)
            h = self._apply_cond(h, purpose_emb, i)
            h = bn(h)
            h = self.activation(h)
        h = h.flatten(1)
        h = self.dropout(h)
        return self.repr_proj(h)

    def forward(self, x: torch.Tensor, purpose_idx) -> torch.Tensor:
        b = x.shape[0]
        if isinstance(purpose_idx, int):
            idx_t = torch.full((b,), purpose_idx, device=x.device, dtype=torch.long)
        else:
            idx_t = purpose_idx
        emb = self.purpose_embedding(idx_t)
        return self._forward_impl(x, emb)

    def forward_with_embedding(self, x: torch.Tensor, purpose_emb: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x, purpose_emb)

    def encode_all_purposes(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        b = x.shape[0]
        out: dict[int, torch.Tensor] = {}
        for p in range(self.num_purposes):
            idx_t = torch.full((b,), p, device=x.device, dtype=torch.long)
            emb = self.purpose_embedding(idx_t)
            out[p] = self._forward_impl(x, emb)
        return out

    def get_purpose_embedding(self, purpose_idx: int) -> torch.Tensor:
        idx_tensor = torch.tensor([purpose_idx], device=self.purpose_embedding.weight.device)
        return self.purpose_embedding(idx_tensor).squeeze(0)


class MultiTaskHead(nn.Module):
    def __init__(self, heads: dict[str, nn.Module]) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


# ═════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════

REPR_DIM = 128
CONV_CHANNELS = (32, 64, 128)
BATCH_SIZE = 256
LR = 1e-3
LAMBDA_ADV = 0.5
LAMBDA_VERIFY = 0.3
AUDITOR_STEPS = 5
EPOCHS = 50
PATIENCE = None
WARMUP_EPOCHS = 5
DROPOUT = 0.3
PURPOSE_EMB_DIM = 32
SEED = 0


def run_variant(conditioning: str, device: str, train_ds, val_ds, test_ds,
                purposes, registry: PurposeRegistry) -> tuple[list[dict], dict, float, int]:
    print(f"\n{'='*60}\nCelebA ({conditioning}) — seed {SEED}\n{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pcrl_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)

    encoder = CNNEncoderConditioned(
        repr_dim=REPR_DIM, num_purposes=len(purposes),
        conditioning=conditioning, purpose_emb_dim=PURPOSE_EMB_DIM,
        conv_channels=CONV_CHANNELS, dropout=DROPOUT,
    )

    task_heads: dict[str, nn.Module] = {}
    for p in purposes:
        if len(p.allowed_tasks) == 1:
            tn = p.allowed_tasks[0]
            od = p.allowed_task_dims.get(tn, 2)
            task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=od)
        else:
            sub = {}
            for tn in p.allowed_tasks:
                od = p.allowed_task_dims.get(tn, 2)
                sub[tn] = TaskHead(repr_dim=REPR_DIM, output_dim=od)
            task_heads[p.name] = MultiTaskHead(sub)

    auditors: dict[str, nn.Module] = {}
    for p in purposes:
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    config = TrainerConfig(
        batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
        lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY,
        auditor_steps=AUDITOR_STEPS, epochs=EPOCHS, weight_decay=1e-4,
        early_stopping_patience=PATIENCE, log_interval=100,
        confusion_type="entropy", warmup_epochs=WARMUP_EPOCHS,
        gradient_reversal=False, sequential_purposes=False,
        checkpoint_dir=str(project_root / "checkpoints" / f"celeba_{conditioning}_s{SEED}"),
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    print("  Training...")
    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    elapsed = time.time() - t0
    print(f"  Training complete — epoch {state.epoch + 1}, {elapsed:.0f}s")

    eval_metrics = trainer.evaluate(test_loader)
    print("  Task accuracies:")
    for task, acc in sorted(eval_metrics.task_accuracy.items()):
        print(f"    {task}: {acc:.1%}")

    reports = generate_report(
        encoder=encoder, train_loader=train_loader, test_loader=test_loader,
        purpose_registry=registry, device=device,
    )

    rows = []
    pass_count = 0
    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        ok = delta < 0.02 and r.linear_r2 < 0.05
        if ok:
            pass_count += 1
        rows.append({
            "variant": conditioning,
            "seed": SEED,
            "purpose": r.purpose_name,
            "attribute": r.attr_name,
            "linear_r2": round(r.linear_r2, 6),
            "linear_pass": r.linear_certified,
            "empirical_best_acc": round(r.empirical_best_acc, 4),
            "majority_baseline": round(r.majority_proportion, 4),
            "delta": round(delta, 4),
            "adj_pass": ok,
            "nonlinear_bound": round(r.nonlinear_bound, 4) if r.nonlinear_bound else None,
            "train_time_s": round(elapsed, 1),
        })
    total = len(reports)
    print(f"  Adjusted compliance: {pass_count}/{total}")
    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        ok = delta < 0.02 and r.linear_r2 < 0.05
        status = "PASS" if ok else "FAIL"
        print(f"    {r.purpose_name:30s} / {r.attr_name:12s}  Δ={delta:+.1%}  R²={r.linear_r2:.4f}  {status}")

    return rows, eval_metrics.task_accuracy, elapsed, pass_count


def save_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["film", "concat"],
                        choices=["film", "concat"])
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")
    print(f"v2 HPs: repr={REPR_DIM} conv={CONV_CHANNELS} batch={BATCH_SIZE} "
          f"lam_adv={LAMBDA_ADV} lam_ver={LAMBDA_VERIFY} epochs={EPOCHS} "
          f"K={AUDITOR_STEPS} warmup={WARMUP_EPOCHS}")

    purposes = get_celeba_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    print("Loading CelebA (max 10k train / 3k val / 3k test)...")
    train_ds = CelebADataset(purposes, root="data/celeba", split="train", max_samples=10000)
    val_ds = CelebADataset(purposes, root="data/celeba", split="val", max_samples=3000)
    test_ds = CelebADataset(purposes, root="data/celeba", split="test", max_samples=3000)

    summary = {}
    for cond in args.variants:
        rows, accs, elapsed, pc = run_variant(cond, device, train_ds, val_ds, test_ds, purposes, registry)
        out = project_root / "results" / "celeba" / (
            "concat_perpair.csv" if cond == "concat" else "film_v2matched_perpair.csv"
        )
        save_rows(rows, out)
        summary[cond] = {"rows": rows, "task_accs": accs, "elapsed": elapsed, "pass_count": pc}

    print("\n" + "=" * 60)
    print("FiLM vs Concat — CelebA v2 HPs")
    print("=" * 60)
    for cond, s in summary.items():
        print(f"\n{cond.upper()}:  pass={s['pass_count']}/{len(s['rows'])}  time={s['elapsed']:.0f}s")
        for task, acc in sorted(s["task_accs"].items()):
            print(f"  {task:>25s}: {acc:.1%}")


if __name__ == "__main__":
    main()
