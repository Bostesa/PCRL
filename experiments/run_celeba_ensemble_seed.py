#!/usr/bin/env python3
"""CelebA PCRL training for one ensemble seed.

Identical hyperparameters to run_celeba_v2.py.  Reads SEED from argv,
writes checkpoint to checkpoints/celeba_ensemble/seed_{SEED}/best.pt.
No CSV output here — evaluation is done in run_celeba_ensemble_eval.py
after all seeds finish.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

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
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.cnn_encoder import CNNPurposeProjectionEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

REPR_DIM = 128
CONV_CHANNELS = (32, 64, 128)
DROPOUT = 0.3
BATCH_SIZE = 256
LR = 1e-3
LAMBDA_ADV = 0.5
LAMBDA_VERIFY = 0.3
AUDITOR_STEPS = 5
EPOCHS = 50
WARMUP_EPOCHS = 5
AUDITOR_HIDDEN = 256
AUDITOR_LAYERS = 3
MAX_TRAIN = 10000
MAX_VAL = 3000
MAX_TEST = 3000


class MultiTaskHead(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = nn.ModuleDict(heads)
    def forward(self, x):
        return {name: head(x) for name, head in self.heads.items()}


def main():
    if len(sys.argv) != 2:
        print("Usage: run_celeba_ensemble_seed.py <seed>", file=sys.stderr)
        sys.exit(1)
    seed = int(sys.argv[1])

    torch.manual_seed(seed)
    import numpy as np, random
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== CelebA PCRL ensemble — seed {seed}, device={device} ===")

    purposes = get_celeba_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = CelebADataset(purposes, root="data/celeba", split="train", max_samples=MAX_TRAIN)
    val_ds = CelebADataset(purposes, root="data/celeba", split="val", max_samples=MAX_VAL)
    test_ds = CelebADataset(purposes, root="data/celeba", split="test", max_samples=MAX_TEST)
    print(f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pcrl_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)

    encoder = CNNPurposeProjectionEncoder(
        repr_dim=REPR_DIM,
        num_purposes=len(purposes),
        conv_channels=CONV_CHANNELS,
        dropout=DROPOUT,
        backbone_grad_scale=1.0,
    )

    task_heads = {}
    for p in purposes:
        if len(p.allowed_tasks) == 1:
            t = p.allowed_tasks[0]
            task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=p.allowed_task_dims.get(t, 2))
        else:
            sub = {t: TaskHead(repr_dim=REPR_DIM, output_dim=p.allowed_task_dims.get(t, 2))
                   for t in p.allowed_tasks}
            task_heads[p.name] = MultiTaskHead(sub)

    auditors = {p.name: MultiAttributeAuditor(
        repr_dim=REPR_DIM,
        attr_output_dims=p.disallowed_attr_dims,
        hidden_dim=AUDITOR_HIDDEN,
        num_layers=AUDITOR_LAYERS,
    ) for p in purposes}

    ckpt_dir = project_root / "checkpoints" / "celeba_ensemble" / f"seed_{seed}"
    config = TrainerConfig(
        batch_size=BATCH_SIZE,
        lr_encoder=LR,
        lr_auditor=LR,
        lambda_adv=LAMBDA_ADV,
        lambda_verify=LAMBDA_VERIFY,
        auditor_steps=AUDITOR_STEPS,
        epochs=EPOCHS,
        weight_decay=1e-4,
        early_stopping_patience=None,
        log_interval=100,
        confusion_type="entropy",
        warmup_epochs=WARMUP_EPOCHS,
        gradient_reversal=False,
        sequential_purposes=False,
        checkpoint_dir=str(ckpt_dir),
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    print(f"seed={seed} done in {time.time()-t0:.0f}s, best epoch={state.epoch+1}, ckpt={ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
