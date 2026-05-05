"""LEACE warm-start rank-sufficiency probe.

For each dataset, build the V2 trainer at rank=8, run leace_warm_start,
and report per-pair pre / post_closed linear-R² (auditor metric).

Hypothesis under test: joint LEACE on concatenated [Z_1 | Z_2 | ...] with
sum(c_i - 1) <= rank achieves post_closed R² ~ 0 on every pair. When
required rank exceeds LoRA rank (Diabetes quality_research, req=13 with
race=5 + age_bucket=10), the closed-form eraser cannot zero R² on the
deficient axes — visible as nonzero post_closed R² on age_bucket.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.base import collate_pcrl_batch
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.models.encoder import StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.independence.vclub import VCLUB
from pcrl.training.v2_trainer import V2Trainer, V2TrainerConfig


def build(name: str):
    if name == "adult":
        from pcrl.data.adult import AdultDataset, get_adult_purposes
        purposes = get_adult_purposes()
        train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    elif name == "diabetes":
        from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
        purposes = get_diabetes_purposes()
        train_ds = DiabetesDataset(purposes=purposes, split="train")
    elif name == "hmda":
        from pcrl.data.hmda import HMDADataset, get_hmda_purposes
        purposes = get_hmda_purposes()
        train_ds = HMDADataset(purposes=purposes, root="data", split="train")
    else:
        raise ValueError(name)
    return purposes, train_ds


@torch.no_grad()
def post_truncation_r2(trainer, train_loader) -> dict:
    """After leace_warm_start (which applies SVD truncation to rank=r), compute
    the auditor R² using the encoder's actual output per purpose. This is the
    real post-init R² the trainer sees at epoch 0."""
    from pcrl.training.v2_trainer import _linear_r2_train as _r2

    trainer.encoder.eval()
    feats_per_purpose = {p: [] for p in trainer.purpose_names}
    attrs_collect = {}
    for batch in train_loader:
        batch = trainer._to_device(batch)
        for pn in trainer.purpose_names:
            z = trainer.encoder(batch["features"], trainer._purpose_idx(pn))
            feats_per_purpose[pn].append(z.detach().cpu())
        for k, v in batch["sensitive_attrs"].items():
            attrs_collect.setdefault(k, []).append(v.detach().cpu().long())
    attrs = {k: torch.cat(v, dim=0).numpy() for k, v in attrs_collect.items()}
    out = {}
    for pn, parts in feats_per_purpose.items():
        Zp = torch.cat(parts, dim=0).numpy()
        disallowed = trainer.purpose_configs[pn]["disallowed_attrs"]
        out[pn] = {a: _r2(Zp, attrs[a]) for a in disallowed}
    return out


def run(dataset: str) -> dict:
    torch.manual_seed(0)
    np.random.seed(0)

    purposes, train_ds = build(dataset)
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)
    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0,
    )

    input_dim = train_ds.info.num_features
    repr_dim = 64
    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=repr_dim, dropout=0.3,
    )
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=len(purposes),
        rank=8, alpha=16.0, dropout=0.0,
    )
    task_heads = {}
    vclubs = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        out_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=out_dim)
        for attr in p.disallowed_attrs:
            n_classes = p.disallowed_attr_dims.get(attr, 2)
            vclubs[f"{p.name}__{attr}"] = VCLUB(
                x_dim=repr_dim, z_dim=n_classes, hidden_dim=128,
                z_categorical=True, l2=1e-1,
            )

    config = V2TrainerConfig(repr_threshold=None) if False else V2TrainerConfig(
        r2_threshold=0.05, lora_rank=8, lora_alpha=16.0,
        checkpoint_dir=str(ROOT / "checkpoints" / f"_probe_leace_{dataset}"),
    )
    trainer = V2Trainer(
        encoder=encoder, task_heads=task_heads, vclubs=vclubs,
        purpose_registry=registry, config=config, device="cpu",
    )

    diag = trainer.leace_warm_start(train_loader)
    post_lora = post_truncation_r2(trainer, train_loader)
    for pn, attr_r2 in post_lora.items():
        for a, r2 in attr_r2.items():
            diag[pn][f"r2_post_lora[{a}]"] = float(r2)
    return diag


def main():
    out = {}
    for ds in ("adult", "hmda", "diabetes"):
        print(f"\n=== {ds.upper()} — joint LEACE warm-start (rank=8) ===")
        d = run(ds)
        out[ds] = d
        for purpose, m in d.items():
            print(f"  purpose: {purpose}")
            pre_keys = sorted(k for k in m if k.startswith("r2_pre["))
            for k in pre_keys:
                attr = k[len("r2_pre["):-1]
                pre = m[k]
                post_c = m.get(f"r2_post_closed[{attr}]", float("nan"))
                post_l = m.get(f"r2_post_lora[{attr}]", float("nan"))
                tag = "✓" if post_l < 0.01 else ("✗ RANK-DEFICIT" if post_l > 0.05 else "△")
                print(f"    {attr:20s} pre={pre:.4f}  post_closed={post_c:.4f}  post_lora={post_l:.4f}  {tag}")
    out_path = ROOT / "results" / "leace_rank_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
