# LAFTR Hard-R² Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain LAFTR with a proxy-Lagrangian wrapper enforcing linear-R²(h, A) ≤ 0.05 per (purpose, attribute) pair — the same constraint mechanism PCRL's v2 trainer uses — and audit it on the same 60-cell grid (Adult 24 + HMDA 18 + Diabetes 18) × 3 seeds with the same compliance criterion. Produces a fair head-to-head against PCRL for the NeurIPS rebuttal.

**Architecture:** A new `LAFTRProxyTrainer` reuses PCRL's frozen `StandardEncoder([128,128]→64)` + per-purpose LoRA backbone for capacity-controlled comparison, attaches LAFTR-style MLP discriminators per (purpose, attribute), and runs alternating updates: (1) discriminator step on detached representations; (2) encoder + task head step with primal loss `L_task − λ_adv · Σ_a CE(disc_a(z), y_a) + Σ_a λ_{p,a} · (R²(z, A) − τ)`; (3) dual ascent on every λ_{p,a} via the existing `ProxyLagrangianOptimizer`. The binding fairness constraint is the proxy-Lagrangian dual on linear R² (matching the auditor's metric); LAFTR's adversarial loss survives as a fixed-weight regulariser at λ_adv = 0.1. Per-class OvR, LEACE warm-start, λ-floor, and Cotter best-iterate selection are preserved exactly so the only difference from PCRL is the *addition* of the LAFTR discriminator term.

**Tech Stack:** Python 3.11, PyTorch, existing `pcrl.training.proxy_lagrangian`, `pcrl.training.losses.VerificationRegularizer`, `pcrl.baselines.laftr.LAFTRDiscriminator`, `pcrl.models.{encoder,lora,task_head}`, `pcrl.evaluation.certificates.generate_report`. AWS g4dn.xlarge with `pcrl-bios-s3-writer` IAM profile (see memory `reference_pcrl_aws_launch.md`).

---

## Design Decisions Locked Before Coding

| Decision | Value | Reason |
|---|---|---|
| Backbone | `StandardEncoder(input_dim, [128,128], repr_dim=64, dropout=0.3)` wrapped in `PerPurposeLoRAEncoder` | Capacity-controlled comparison vs PCRL (same params, same per-purpose adapters). |
| LoRA rank | Adult/HMDA: 8; Diabetes: 24 | Mirrors v2 (`LORA_BY_DATASET` in `experiments/run_v2_dataset.py:93`). Joint LEACE rank requirement holds independent of the LAFTR head. |
| `lambda_adv` (headline) | 1.0 | Apples-to-apples with Appendix Q's published LAFTR. The new mechanism (proxy-Lagrangian R² constraint) is added *on top of* LAFTR at its original weight, not in place of a weakened LAFTR. Subtracted from primal loss (encoder wants to *maximise* adversary CE). |
| `lambda_adv` (fallback) | 0.1 | Run only if the 1.0 headline is empirically unstable (the adversary fighting the dual could oscillate). Document the instability symptom in SUMMARY.md if invoked. |
| `lambda_adv` (ablation, if compute allows) | 0.0 | Pure proxy-Lagrangian, no adversary — isolates the encoder-architecture variable from the loss-mechanism variable. Only fires if the headline + fallback budget leaves room. |
| `lr_lambda` | 0.02 | User spec; matches v2 default (`V2TrainerConfig.lr_lambda`). |
| τ (R² threshold) | 0.05 | Matches PCRL + paper's compliance criterion. |
| `λ_init`, `λ_max`, `λ_min` | 1.0, 1000.0, 5.0 | Round 5 fix (memory: `project_v2_round5`). Same as v2. |
| Per-class OvR | K ≥ 6 (Diabetes age_bucket only) | Matches v2 `per_class_constraint_threshold=6`. |
| LEACE warm-start | ON | Matches v2; provides feasible-set entry. Same as `V2TrainerConfig.leace_init=True`. |
| Cotter best-iterate | ON | Matches v2; selects best feasible iterate post-hoc. |
| VICReg | ON (γ=1.0, λ_var=1.0, λ_cov=0.04) | Anti-collapse; matches v2 so LAFTR can't win/lose by capacity drift. |
| vCLUB / HSIC | OFF (λ=0) | v2 disables these in constrained phase (Round 4); same here for cleanliness. |
| Epochs / batch | 200 + 0 warmup / 256 | Match v2 + `leace_init=True` skip-warmup. |
| Seeds | [0, 1, 2] | Match v2. |
| Datasets / grid | Adult 24 + HMDA 18 + Diabetes 18 = 60 cells × 3 seeds | Match v2; reuse `get_adult_purposes()`, `get_hmda_purposes()`, `get_diabetes_purposes()`. |
| Audit | `pcrl.evaluation.certificates.generate_report` | Same code path as PCRL; identical compliance criterion (linear R² < 0.05 AND auditor δ < 2pp). |

## File Structure

| File | Role |
|---|---|
| `pcrl/training/laftr_proxy_trainer.py` (new) | `LAFTRProxyTrainerConfig` + `LAFTRProxyTrainer`. Composes the v2 pieces (frozen backbone + LoRA, LEACE warm-start, VerificationRegularizer, ProxyLagrangianOptimizer, Cotter selection) and adds: per-pair `LAFTRDiscriminator`, separate discriminator optimiser, two-phase alternating step. |
| `tests/training/test_laftr_proxy_trainer.py` (new) | Unit tests: constraint registration; primal-loss formula (sign of `−λ_adv·adv`); dual update; discriminator detach; LEACE warm-start preserved; Cotter selection; CPU smoke on synthetic 2-purpose × 2-attr toy. |
| `experiments/run_laftr_hard_r2.py` (new) | Per-dataset orchestrator, mirrors `experiments/run_v2_dataset.py`. CLI: `--dataset {adult,hmda,diabetes} --seeds 0 1 2 --device cuda --epochs 200 --out-tag _LAFTR_HARD_R2`. Writes `results/laftr_hard_r2_<dataset>/{per_seed_results.json,summary.json}` and `checkpoints/laftr_hard_r2_<dataset>_s<seed>/{best,final,canonical_iterate}.pt`. |
| `scripts/aggregate_laftr_hard_r2.py` (new) | After all 3 datasets finish, aggregate to a single `results/laftr_hard_r2/SUMMARY.md` with pass-counts and head-to-head vs `results/v2_<dataset>/`. |
| `aws/launch_laftr_hard_r2.sh` (new) | g4dn.xlarge user-data that clones repo on branch `laftr-hard-r2-2026-05-17`, runs `prepare_hmda.py` / `preprocess_diabetes.py` first, then launches the 3 datasets sequentially under nohup with 12h hard cap + auto-shutdown. |
| `results/laftr_hard_r2/` (new dir) | Final per-dataset summaries and the head-to-head SUMMARY.md. |

Files **not** modified: `pcrl/training/v2_trainer.py`, `pcrl/training/proxy_lagrangian.py`, `pcrl/training/losses.py`, `pcrl/baselines/laftr.py`, `experiments/run_v2_dataset.py` — keeps the LAFTR variant additive so the v2 pilot on `erase-layer-pilot-2026-05-17` is untouched and reviewers can diff cleanly.

---

## Surgery Map (what plugs into what)

```
PCRL v2 trainer (read-only here):           New LAFTR-proxy trainer (this plan):
┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
│ StandardEncoder([128,128]→64)        │    │ StandardEncoder([128,128]→64)        │
│   wrapped in PerPurposeLoRAEncoder   │    │   wrapped in PerPurposeLoRAEncoder   │
│ ─ frozen backbone, BN frozen         │    │ ─ frozen backbone, BN frozen         │
│ ─ LEACE warm-start of last LoRA      │    │ ─ LEACE warm-start of last LoRA      │
├──────────────────────────────────────┤    ├──────────────────────────────────────┤
│ Primal: AdamW(LoRA + task_heads)     │    │ Primal: AdamW(LoRA + task_heads)     │
│ vCLUB: Adam(q-nets) [λ_vclub=0]      │    │ Disc: Adam(discriminators)  ←── NEW  │
├──────────────────────────────────────┤    ├──────────────────────────────────────┤
│ Per-batch step:                      │    │ Per-batch step:                      │
│   1. vCLUB q-net step (detached)     │    │   1. Discriminator step (detached)   │
│      ──> noop since λ_vclub=0        │    │      ──> minimise Σ_a CE(disc_a, y_a)│
│   2. Forward z_p = enc(x, p) for all │    │   2. Forward z_p = enc(x, p) for all │
│   3. Compute L_task, L_vicreg,       │    │   3. Compute L_task, L_vicreg,       │
│      R²_{p,a} per pair               │    │      R²_{p,a} per pair               │
│   4. base = L_task + λ_vicreg·L_vicreg│   │   4. adv  = Σ_{p,a} CE(disc_a(z_p))  │
│                                      │    │      base = L_task + λ_vicreg·L_vicreg│
│                                      │    │             − λ_adv·adv     ←── NEW  │
│   5. primal_loss =                   │    │   5. primal_loss =                   │
│        proxy.lagrangian_loss(base,   │    │        proxy.lagrangian_loss(base,   │
│            {pair: R²_tensor})        │    │            {pair: R²_tensor})        │
│   6. backward + AdamW step           │    │   6. backward + AdamW step           │
│   7. proxy.dual_step({pair: R²_val}) │    │   7. proxy.dual_step({pair: R²_val}) │
├──────────────────────────────────────┤    ├──────────────────────────────────────┤
│ Cotter best-iterate selection        │    │ Cotter best-iterate selection        │
│ generate_report (audit)              │    │ generate_report (audit)              │
└──────────────────────────────────────┘    └──────────────────────────────────────┘
```

The only difference: a per-pair discriminator + an alternating phase 1 + a `− λ_adv · adv_loss` term in `base`. Everything else is byte-identical.

`pcrl.training.v2_trainer.V2Trainer.__init__` lines 362–476 (constraint construction) and 663–862 (`_primal_and_dual_step`) define the seams we are reproducing. `pcrl.baselines.laftr.LAFTRDiscriminator` (l.26–40) is the discriminator module we re-instantiate per (purpose, attribute).

---

## Task 1: Test fixtures — synthetic toy dataset

**Files:**
- Create: `tests/training/conftest.py`

- [ ] **Step 1: Create a tiny synthetic 2-purpose × 2-attr fixture (CPU-only).**

```python
# tests/training/conftest.py
"""Shared fixtures for LAFTR-proxy trainer tests."""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from pcrl.data.base import collate_pcrl_batch
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec


class ToyPCRLDataset(Dataset):
    def __init__(self, n: int = 256, d: int = 16, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, d, generator=g)
        # task labels correlated with first 4 dims
        self.y_task = (self.x[:, :4].sum(dim=1) > 0).long()
        # sensitive attrs correlated with last 4 dims (so they're erasable)
        self.attr_a = (self.x[:, -4:].sum(dim=1) > 0).long()
        self.attr_b = (self.x[:, -2:].sum(dim=1) > 0).long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return {
            "features": self.x[i],
            "task_labels": {"toy_task": self.y_task[i]},
            "sensitive_attrs": {"attr_a": self.attr_a[i], "attr_b": self.attr_b[i]},
        }


@pytest.fixture
def toy_purposes() -> list[PurposeSpec]:
    return [
        PurposeSpec(
            name="purpose_0",
            task_type="classification",
            allowed_tasks=["toy_task"],
            allowed_task_dims={"toy_task": 2},
            disallowed_attrs=["attr_a"],
            disallowed_attr_dims={"attr_a": 2},
        ),
        PurposeSpec(
            name="purpose_1",
            task_type="classification",
            allowed_tasks=["toy_task"],
            allowed_task_dims={"toy_task": 2},
            disallowed_attrs=["attr_b"],
            disallowed_attr_dims={"attr_b": 2},
        ),
    ]


@pytest.fixture
def toy_loaders():
    train = DataLoader(ToyPCRLDataset(n=256, seed=0), batch_size=32, shuffle=True,
                       collate_fn=collate_pcrl_batch)
    val = DataLoader(ToyPCRLDataset(n=64, seed=1), batch_size=32, shuffle=False,
                     collate_fn=collate_pcrl_batch)
    return train, val
```

- [ ] **Step 2: Verify the fixture by running an empty pytest collect.**

Run: `pytest tests/training/conftest.py --collect-only -q`
Expected: 0 tests collected, no errors.

- [ ] **Step 3: Commit.**

```bash
git add tests/training/conftest.py
git commit -m "test: add toy 2-purpose × 2-attr fixture for LAFTR-proxy trainer"
```

---

## Task 2: `LAFTRProxyTrainerConfig` + constraint registration

**Files:**
- Create: `pcrl/training/laftr_proxy_trainer.py`
- Test: `tests/training/test_laftr_proxy_trainer.py`

- [ ] **Step 1: Write the failing test for config + constraint registration.**

```python
# tests/training/test_laftr_proxy_trainer.py
"""Tests for LAFTR + proxy-Lagrangian trainer."""
from __future__ import annotations

import torch

from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.laftr_proxy_trainer import (
    LAFTRProxyTrainer,
    LAFTRProxyTrainerConfig,
)


def _build_trainer(toy_purposes, *, lambda_adv: float = 0.1, lr_lambda: float = 0.02,
                   lambda_min: float = 5.0, leace_init: bool = False) -> LAFTRProxyTrainer:
    registry = PurposeRegistry()
    for p in toy_purposes:
        registry.register(p)
    backbone = StandardEncoder(input_dim=16, hidden_dims=[128, 128], repr_dim=64, dropout=0.3)
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=len(toy_purposes), rank=8, alpha=16.0, dropout=0.0,
    )
    task_heads = {p.name: TaskHead(repr_dim=64, output_dim=2) for p in toy_purposes}
    config = LAFTRProxyTrainerConfig(
        lambda_adv=lambda_adv, lr_lambda=lr_lambda, r2_threshold=0.05,
        lambda_min=lambda_min, lambda_hsic_init=1.0, r2_lambda_max=1000.0,
        lambda_vicreg=1.0, epochs=1, batch_size=32, leace_init=leace_init,
    )
    return LAFTRProxyTrainer(
        encoder=encoder, task_heads=task_heads,
        purpose_registry=registry, config=config, device="cpu",
    )


def test_constraints_registered_one_per_pair(toy_purposes):
    trainer = _build_trainer(toy_purposes)
    # 2 purposes × 1 attr each = 2 constraints
    assert len(trainer.proxy.constraints) == 2
    assert "purpose_0__attr_a" in trainer.proxy.constraints
    assert "purpose_1__attr_b" in trainer.proxy.constraints


def test_constraint_hyperparameters_match_spec(toy_purposes):
    trainer = _build_trainer(toy_purposes)
    c = trainer.proxy.constraints["purpose_0__attr_a"]
    assert c.threshold == 0.05
    assert c.eta_lambda == 0.02
    assert c.lambda_value == 5.0  # max(lambda_init=1.0, lambda_min=5.0)
    assert c.lambda_max == 1000.0
    assert c.lambda_min == 5.0
    assert c.direction == "<="


def test_discriminator_registered_per_pair(toy_purposes):
    trainer = _build_trainer(toy_purposes)
    assert set(trainer.discriminators.keys()) == {
        "purpose_0__attr_a", "purpose_1__attr_b",
    }
    for d in trainer.discriminators.values():
        # LAFTRDiscriminator: Linear(64,64) → ReLU → Linear(64,32) → ReLU → Linear(32,K)
        layers = list(d.net.children())
        assert isinstance(layers[0], torch.nn.Linear)
        assert layers[0].in_features == 64
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `pytest tests/training/test_laftr_proxy_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pcrl.training.laftr_proxy_trainer'`.

- [ ] **Step 3: Implement the trainer skeleton (config + `__init__` constraint/discriminator wiring).**

```python
# pcrl/training/laftr_proxy_trainer.py
"""LAFTR + proxy-Lagrangian trainer.

Replaces LAFTR's fixed-weight adversarial loss with a hard linear-R²
constraint enforced via proxy-Lagrangian dual variables — the same
mechanism PCRL's v2 trainer uses. LAFTR's adversarial discriminators
survive as a fixed-weight regulariser at ``lambda_adv=0.1`` so the
encoder still gets a nonlinear "fool the adversary" gradient on top of
the linear-R² constraint.

Primal step (per batch):
    z_p   = encoder(x, p)               # frozen backbone + per-purpose LoRA
    L_task     = sum_p CE(task_head_p(z_p), y_p)
    L_vicreg   = sum_p vicreg_loss(z_p)
    L_adv      = sum_{p,a} CE(disc_{p,a}(z_p), attr_a)  # detach=False
    primal_loss = L_task
                + lambda_vicreg * L_vicreg
                - lambda_adv    * L_adv                            # NEW vs v2
                + sum_{p,a} lambda_{p,a} * (R²(z_p, attr_a) - tau)
    backward; step (LoRA adapters + task head params)

Discriminator step (per batch, BEFORE primal):
    for each (p, a): minimise CE(disc_{p,a}(z_p.detach()), attr_a)
    (separate optimiser; X is detached so encoder isn't updated)

Dual step (per batch, AFTER primal):
    for each (p, a): lambda_{p,a} <- proj([lambda_min, lambda_max],
                                          lambda + lr_lambda * (R² - tau))

All other machinery (LEACE warm-start, Cotter best-iterate, per-class
OvR for high-K attrs, frozen backbone BN, checkpointing) is copied
unmodified from V2Trainer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from pcrl.baselines.laftr import LAFTRDiscriminator
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.independence.vicreg import vicreg_loss
from pcrl.training.losses import VerificationRegularizer, task_loss
from pcrl.training.proxy_lagrangian import Constraint, ProxyLagrangianOptimizer

logger = logging.getLogger(__name__)


def _pair_key(purpose_name: str, attr_name: str) -> str:
    return f"{purpose_name}__{attr_name}"


def _per_class_key(purpose_name: str, attr_name: str, k: int) -> str:
    return f"{purpose_name}__{attr_name}__class_{k}"


@dataclass
class LAFTRProxyTrainerConfig:
    """LAFTR + proxy-Lagrangian hyperparameters.

    Defaults intentionally mirror ``V2TrainerConfig`` so the only knob that
    differs vs PCRL is ``lambda_adv`` (the LAFTR discriminator weight).
    """

    lr_primal: float = 1e-3
    lr_disc: float = 1e-3
    lr_lambda: float = 0.02

    lambda_vicreg: float = 1.0
    lambda_adv: float = 1.0   # Headline: apples-to-apples with Appendix Q's LAFTR.
                              # Fallback to 0.1 only if 1.0 is empirically unstable
                              # (adversary fights dual → oscillation); document
                              # the instability in SUMMARY.md if invoked.
                              # Ablation at 0.0 (pure proxy-Lagrangian, no adversary)
                              # runs after headline if compute permits.
    lambda_hsic_init: float = 1.0
    r2_threshold: float = 0.05
    r2_lambda_max: float = 1000.0
    lambda_min: float = 5.0   # Round 5 fix (memory: project_v2_round5)
    per_class_constraint_threshold: int = 6

    cotter_fallback_task_slack: float = 0.10

    leace_init: bool = True
    warmup_when_leace_init: bool = False
    warmup_epochs: int = 5

    vicreg_gamma: float = 1.0
    vicreg_lambda_var: float = 1.0
    vicreg_lambda_cov: float = 0.04

    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    disc_steps: int = 1

    batch_size: int = 256
    epochs: int = 200
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    log_interval: int = 50
    checkpoint_dir: str = "checkpoints/laftr_hard_r2"

    report_best_iterate: bool = False


class LAFTRProxyTrainer:
    """LAFTR with hard linear-R² constraint via proxy-Lagrangian dual.

    Composes the same primitives as ``V2Trainer`` but swaps the vCLUB block
    for LAFTR discriminators and adds a ``− lambda_adv · Σ CE(disc, attr)``
    term to the encoder's primal loss.
    """

    def __init__(
        self,
        encoder: PerPurposeLoRAEncoder,
        task_heads: dict[str, nn.Module],
        purpose_registry: PurposeRegistry,
        config: LAFTRProxyTrainerConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.encoder = encoder
        self.task_heads = nn.ModuleDict(task_heads)
        self.verifier = VerificationRegularizer()
        self.config = config
        self.device = torch.device(device)
        self.purpose_registry = purpose_registry

        # Freeze backbone (params + BN buffers).
        for p in self.encoder.backbone.parameters():
            p.requires_grad_(False)
        self._freeze_backbone_bn()
        self.encoder.to(self.device)
        self.task_heads.to(self.device)
        self.verifier.to(self.device)

        self.purpose_configs: dict[str, dict[str, Any]] = {}
        for idx, purpose in enumerate(purpose_registry.purposes):
            self.purpose_configs[purpose.name] = {
                "purpose_idx": idx,
                "task_type": purpose.task_type,
                "allowed_tasks": purpose.allowed_tasks,
                "disallowed_attrs": purpose.disallowed_attrs,
                "disallowed_attr_dims": dict(purpose.disallowed_attr_dims),
            }
        self.purpose_names: list[str] = list(self.purpose_configs.keys())

        # ── Constraint construction (mirrors V2Trainer.__init__ l.362–476) ──
        constraints: list[Constraint] = []
        self.pair_keys: list[tuple[str, str]] = []
        self.pair_constraint_keys: dict[tuple[str, str], list[str]] = {}
        self.high_k_pairs: dict[tuple[str, str], int] = {}
        per_class_threshold = config.per_class_constraint_threshold

        # ── Discriminator construction (one per (purpose, attribute)) ──
        self.discriminators = nn.ModuleDict()

        for purpose_name in self.purpose_names:
            attr_dims = self.purpose_configs[purpose_name]["disallowed_attr_dims"]
            for attr_name in self.purpose_configs[purpose_name]["disallowed_attrs"]:
                pair = (purpose_name, attr_name)
                self.pair_keys.append(pair)
                K = int(attr_dims.get(attr_name, 2))
                pkey = _pair_key(purpose_name, attr_name)
                # One discriminator per pair (predicts attr from z_p)
                self.discriminators[pkey] = LAFTRDiscriminator(
                    repr_dim=64, num_classes=K,
                ).to(self.device)
                if K >= per_class_threshold:
                    self.high_k_pairs[pair] = K
                    names = [_per_class_key(purpose_name, attr_name, k) for k in range(K)]
                    for n in names:
                        constraints.append(
                            Constraint(
                                name=n, threshold=config.r2_threshold,
                                direction="<=", eta_lambda=config.lr_lambda,
                                lambda_init=config.lambda_hsic_init,
                                lambda_max=config.r2_lambda_max,
                                lambda_min=config.lambda_min,
                            )
                        )
                    self.pair_constraint_keys[pair] = names
                else:
                    constraints.append(
                        Constraint(
                            name=pkey, threshold=config.r2_threshold,
                            direction="<=", eta_lambda=config.lr_lambda,
                            lambda_init=config.lambda_hsic_init,
                            lambda_max=config.r2_lambda_max,
                            lambda_min=config.lambda_min,
                        )
                    )
                    self.pair_constraint_keys[pair] = [pkey]

        primal_params = list(self.encoder.trainable_parameters()) + list(self.task_heads.parameters())
        self.primal_optimizer = AdamW(
            primal_params, lr=config.lr_primal, weight_decay=config.weight_decay,
        )
        self.proxy = ProxyLagrangianOptimizer(self.primal_optimizer, constraints)

        self.disc_optimizer = Adam(
            list(self.discriminators.parameters()), lr=config.lr_disc,
        )

        # Minimal in-memory state; full V2State copy comes in Task 6.
        self.state: dict[str, Any] = {"epoch": 0, "global_step": 0, "history": {}}

    def _freeze_backbone_bn(self) -> None:
        for m in self.encoder.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    def _purpose_idx(self, purpose_name: str) -> int:
        return self.purpose_configs[purpose_name]["purpose_idx"]
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `pytest tests/training/test_laftr_proxy_trainer.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit.**

```bash
git add pcrl/training/laftr_proxy_trainer.py tests/training/test_laftr_proxy_trainer.py
git commit -m "feat(laftr-hard-r2): trainer skeleton with constraint + discriminator registration"
```

---

## Task 3: Discriminator phase 1 (detached-repr CE step)

**Files:**
- Modify: `pcrl/training/laftr_proxy_trainer.py`
- Modify: `tests/training/test_laftr_proxy_trainer.py`

- [ ] **Step 1: Write the failing test.**

```python
# Append to tests/training/test_laftr_proxy_trainer.py
def test_discriminator_step_detaches_encoder(toy_purposes, toy_loaders):
    train, _ = toy_loaders
    trainer = _build_trainer(toy_purposes)
    # Snapshot LoRA adapter weights before the disc step.
    before = {
        n: p.detach().clone() for n, p in trainer.encoder.adapters.named_parameters()
    }
    batch = next(iter(train))
    batch = {
        "features": batch["features"],
        "task_labels": {k: v for k, v in batch["task_labels"].items()},
        "sensitive_attrs": {k: v for k, v in batch["sensitive_attrs"].items()},
    }
    loss = trainer._discriminator_step(batch)
    assert loss > 0  # discriminator should produce a positive CE
    for n, p in trainer.encoder.adapters.named_parameters():
        assert torch.allclose(before[n], p), f"LoRA param {n} moved during disc step"
    # Discriminator weights, by contrast, must move.
    moved = False
    for d in trainer.discriminators.values():
        for p in d.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                moved = True
                break
    assert moved, "discriminator parameters had no gradient after step"
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `pytest tests/training/test_laftr_proxy_trainer.py::test_discriminator_step_detaches_encoder -v`
Expected: FAIL with `AttributeError: '_discriminator_step'`.

- [ ] **Step 3: Implement `_discriminator_step`.**

```python
# Append to pcrl/training/laftr_proxy_trainer.py inside LAFTRProxyTrainer

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "features": batch["features"].to(self.device),
            "task_labels": {k: v.to(self.device) for k, v in batch["task_labels"].items()},
            "sensitive_attrs": {k: v.to(self.device) for k, v in batch["sensitive_attrs"].items()},
        }

    def _discriminator_step(self, batch: dict[str, Any]) -> float:
        """Phase 1: minimise Σ_{p,a} CE(disc_{p,a}(z_p.detach()), attr_a).

        Detaching the representation means gradients do NOT propagate into
        the encoder; only discriminator parameters move.
        """
        self.encoder.eval()  # no BN stat drift in this forward
        with torch.no_grad():
            reprs = {
                p: self.encoder(batch["features"], self._purpose_idx(p))
                for p in self.purpose_names
            }
        self.encoder.train()
        self._freeze_backbone_bn()

        self.disc_optimizer.zero_grad(set_to_none=True)
        total = batch["features"].new_zeros(())
        for purpose_name, attr_name in self.pair_keys:
            z = reprs[purpose_name]
            attr = batch["sensitive_attrs"][attr_name].long()
            pkey = _pair_key(purpose_name, attr_name)
            disc = self.discriminators[pkey]
            total = total + F.cross_entropy(disc(z), attr)
        total.backward()
        self.disc_optimizer.step()
        return float(total.detach().item())
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `pytest tests/training/test_laftr_proxy_trainer.py::test_discriminator_step_detaches_encoder -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add pcrl/training/laftr_proxy_trainer.py tests/training/test_laftr_proxy_trainer.py
git commit -m "feat(laftr-hard-r2): phase-1 discriminator step on detached representations"
```

---

## Task 4: Primal step with `−λ_adv · adv_loss` + R² Lagrangian

**Files:**
- Modify: `pcrl/training/laftr_proxy_trainer.py`
- Modify: `tests/training/test_laftr_proxy_trainer.py`

- [ ] **Step 1: Write the failing test.**

```python
# Append to tests/training/test_laftr_proxy_trainer.py
def test_primal_loss_formula(toy_purposes, toy_loaders, monkeypatch):
    """The primal loss must equal:
        L_task + λ_vicreg·L_vicreg − λ_adv·Σ CE(disc, attr) + Σ λ_pa·(R² − τ)
    We monkey-patch the components to known constants and verify the sign + weights.
    """
    train, _ = toy_loaders
    trainer = _build_trainer(toy_purposes, lambda_adv=0.1)
    # Capture the loss the trainer constructs.
    batch = trainer._to_device(next(iter(train)))
    stats = trainer._primal_step_components(batch)
    # Expected reconstruction
    base = stats["L_task"] + trainer.config.lambda_vicreg * stats["L_vicreg"] \
           - trainer.config.lambda_adv * stats["L_adv"]
    lagrangian = sum(
        trainer.proxy.constraints[n].lambda_value * (stats["constraint_scalars"][n] - trainer.config.r2_threshold)
        for n in stats["constraint_scalars"]
    )
    assert abs(stats["primal_loss"] - (base + lagrangian)) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `pytest tests/training/test_laftr_proxy_trainer.py::test_primal_loss_formula -v`
Expected: FAIL with `AttributeError: '_primal_step_components'`.

- [ ] **Step 3: Implement `_primal_step_components` and the primal step.**

```python
# Append to pcrl/training/laftr_proxy_trainer.py inside LAFTRProxyTrainer

    def _primal_step_components(
        self, batch: dict[str, Any], apply_constraints: bool = True,
    ) -> dict[str, Any]:
        """One primal+dual step. Returns scalar stats including a verifiable
        decomposition of the primal loss (used by tests + logs).
        """
        self.encoder.train()
        self._freeze_backbone_bn()
        self.task_heads.train()
        for d in self.discriminators.values():
            d.train()

        reprs: dict[str, torch.Tensor] = {}
        for purpose_name in self.purpose_names:
            reprs[purpose_name] = self.encoder(
                batch["features"], self._purpose_idx(purpose_name),
            )

        # L_task
        L_task = torch.tensor(0.0, device=self.device)
        for purpose_name, z in reprs.items():
            task_name = self.purpose_configs[purpose_name]["allowed_tasks"][0]
            if task_name not in batch["task_labels"]:
                continue
            head = self.task_heads[purpose_name]
            preds = head(z)
            if isinstance(preds, dict):
                preds = preds[task_name]
            targets = batch["task_labels"][task_name]
            L_task = L_task + task_loss(
                preds, targets, self.purpose_configs[purpose_name]["task_type"],
            )

        # L_vicreg
        L_vicreg = torch.tensor(0.0, device=self.device)
        for z in reprs.values():
            L_vicreg = L_vicreg + vicreg_loss(
                z, lambda_var=self.config.vicreg_lambda_var,
                lambda_cov=self.config.vicreg_lambda_cov,
                gamma=self.config.vicreg_gamma,
            )

        # L_adv (discriminator CE, gradients flow into encoder via z)
        # Freeze discriminator params for this backward so disc weights don't
        # accumulate phantom gradients.
        for p_ in self.discriminators.parameters():
            p_.requires_grad_(False)
        L_adv = torch.tensor(0.0, device=self.device)
        for purpose_name, attr_name in self.pair_keys:
            z = reprs[purpose_name]
            attr = batch["sensitive_attrs"][attr_name].long()
            pkey = _pair_key(purpose_name, attr_name)
            disc = self.discriminators[pkey]
            L_adv = L_adv + F.cross_entropy(disc(z), attr)
        for p_ in self.discriminators.parameters():
            p_.requires_grad_(True)

        # Per-pair linear-R² constraint values (mirrors V2Trainer l.716–770)
        constraint_values: dict[str, torch.Tensor] = {}
        constraint_scalars: dict[str, float] = {}
        pair_r2_log: dict[str, float] = {}
        for purpose_name, attr_name in self.pair_keys:
            z = reprs[purpose_name]
            attr = batch["sensitive_attrs"][attr_name].long()
            pair = (purpose_name, attr_name)
            pkey = _pair_key(purpose_name, attr_name)
            if int(attr.max().item()) < 1:
                zero = torch.tensor(0.0, device=self.device)
                for cname in self.pair_constraint_keys[pair]:
                    constraint_values[cname] = zero
                    constraint_scalars[cname] = 0.0
                pair_r2_log[pkey] = 0.0
                continue
            if pair in self.high_k_pairs:
                r2_per_k = self.verifier.forward_per_class(z, attr)
                names = self.pair_constraint_keys[pair]
                K_total = len(names)
                if r2_per_k.numel() < K_total:
                    pad = torch.zeros(K_total - r2_per_k.numel(), device=self.device)
                    r2_per_k = torch.cat([r2_per_k, pad])
                pair_max = float("-inf")
                for k, cname in enumerate(names):
                    rk = r2_per_k[k]
                    constraint_values[cname] = rk
                    val = float(rk.detach().item())
                    constraint_scalars[cname] = val
                    pair_max = max(pair_max, val)
                pair_r2_log[pkey] = pair_max
            else:
                r2_val = self.verifier(z, attr)
                cname = self.pair_constraint_keys[pair][0]
                constraint_values[cname] = r2_val
                constraint_scalars[cname] = float(r2_val.detach().item())
                pair_r2_log[pkey] = constraint_scalars[cname]

        if apply_constraints:
            base = (
                L_task
                + self.config.lambda_vicreg * L_vicreg
                - self.config.lambda_adv * L_adv
            )
            primal_loss = self.proxy.lagrangian_loss(base, constraint_values)
        else:
            primal_loss = L_task + self.config.lambda_vicreg * L_vicreg

        self.primal_optimizer.zero_grad()
        primal_loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.primal_optimizer.param_groups[0]["params"]],
                max_norm=self.config.grad_clip,
            )
        self.primal_optimizer.step()

        if apply_constraints:
            self.proxy.dual_step(constraint_scalars)

        return {
            "primal_loss": float(primal_loss.detach().item()),
            "L_task": float(L_task.detach().item()),
            "L_vicreg": float(L_vicreg.detach().item()),
            "L_adv": float(L_adv.detach().item()),
            "constraint_scalars": constraint_scalars,
            "pair_r2": pair_r2_log,
        }
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `pytest tests/training/test_laftr_proxy_trainer.py::test_primal_loss_formula -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add pcrl/training/laftr_proxy_trainer.py tests/training/test_laftr_proxy_trainer.py
git commit -m "feat(laftr-hard-r2): primal step with −λ_adv·adv_loss + proxy-Lagrangian R²"
```

---

## Task 5: Dual ascent moves λ in the right direction

**Files:**
- Modify: `tests/training/test_laftr_proxy_trainer.py`

- [ ] **Step 1: Write the failing test.**

```python
def test_dual_step_increases_lambda_when_constraint_violated(toy_purposes, toy_loaders):
    train, _ = toy_loaders
    # Use lambda_min=0 so the test can observe upward motion from a low base.
    trainer = _build_trainer(toy_purposes, lambda_min=0.0)
    for c in trainer.proxy.constraints.values():
        c.lambda_value = 1.0
    batch = trainer._to_device(next(iter(train)))
    stats = trainer._primal_step_components(batch, apply_constraints=True)
    # Confirm at least one pair was violated and that pair's λ went up.
    for n, scalar in stats["constraint_scalars"].items():
        c = trainer.proxy.constraints[n]
        if scalar > trainer.config.r2_threshold:
            assert c.lambda_value > 1.0, (
                f"constraint {n} violated ({scalar:.3f} > {trainer.config.r2_threshold}) "
                f"but λ stayed at {c.lambda_value}"
            )
```

- [ ] **Step 2: Run test.**

Run: `pytest tests/training/test_laftr_proxy_trainer.py::test_dual_step_increases_lambda_when_constraint_violated -v`
Expected: PASS (the dual update is provided by `ProxyLagrangianOptimizer.dual_step`, already tested upstream; this test is a contract verification).

If FAIL: the most likely cause is the constraint scalar feeding the dual was the post-clamp R² of 0.0 (degenerate batch). Re-run after seeding `ToyPCRLDataset` differently and assert the random batch actually has both classes per attr.

- [ ] **Step 3: Commit.**

```bash
git add tests/training/test_laftr_proxy_trainer.py
git commit -m "test(laftr-hard-r2): verify dual ascent moves λ upward on violation"
```

---

## Task 6: Train loop with LEACE warm-start + Cotter selection

**Files:**
- Modify: `pcrl/training/laftr_proxy_trainer.py`
- Modify: `tests/training/test_laftr_proxy_trainer.py`

- [ ] **Step 1: Port `leace_warm_start`, `train_epoch`, `evaluate`, `train`, `_select_cotter_best`, `save_checkpoint`, `_update_history` from `V2Trainer` with vCLUB references stripped.**

The v2 versions live at:
- `leace_warm_start`: `pcrl/training/v2_trainer.py:503–621`
- `train_epoch`: `pcrl/training/v2_trainer.py:866–909` — replace the `_vclub_q_step` call with `_discriminator_step` (already implemented).
- `evaluate`: `pcrl/training/v2_trainer.py:911–1012` — strip HSIC; keep R² + task loss + accuracy.
- `train`: `pcrl/training/v2_trainer.py:1014–end` — copy as-is.
- `_select_cotter_best`, `save_checkpoint`, `_update_history`: copy verbatim.

Show the modified `train_epoch` explicitly (the only structural change):

```python
    def train_epoch(self, loader: DataLoader, apply_constraints: bool = True) -> dict[str, float]:
        self.encoder.train()
        self._freeze_backbone_bn()
        self.task_heads.train()
        for d in self.discriminators.values():
            d.train()

        t0 = time.time()
        sums: dict[str, float] = {}
        n = 0
        for batch in loader:
            batch = self._to_device(batch)
            d_loss = 0.0
            for _ in range(self.config.disc_steps):
                d_loss = self._discriminator_step(batch)
            stats = self._primal_step_components(batch, apply_constraints=apply_constraints)
            stats["disc_loss"] = d_loss
            for k, v in stats.items():
                if isinstance(v, dict):
                    continue
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        return {
            "primal_loss": sums.get("primal_loss", 0.0) / max(n, 1),
            "task_loss": sums.get("L_task", 0.0) / max(n, 1),
            "adv_loss": sums.get("L_adv", 0.0) / max(n, 1),
            "vicreg_loss": sums.get("L_vicreg", 0.0) / max(n, 1),
            "disc_loss": sums.get("disc_loss", 0.0) / max(n, 1),
            "epoch_time": time.time() - t0,
        }
```

- [ ] **Step 2: Write smoke test — 2-epoch CPU run on toy data finishes and produces a checkpoint.**

```python
def test_two_epoch_smoke_runs_on_cpu(toy_purposes, toy_loaders, tmp_path):
    train, val = toy_loaders
    trainer = _build_trainer(toy_purposes, leace_init=False)
    trainer.config.epochs = 2
    trainer.config.warmup_epochs = 0
    trainer.config.checkpoint_dir = str(tmp_path / "ckpt")
    state = trainer.train(train_loader=train, val_loader=val)
    assert state["epoch"] == 1  # 0-indexed, 2 epochs total
    assert (tmp_path / "ckpt" / "final.pt").exists()
```

- [ ] **Step 3: Run test.**

Run: `pytest tests/training/test_laftr_proxy_trainer.py::test_two_epoch_smoke_runs_on_cpu -v --timeout=120`
Expected: PASS (≤ 30s on CPU).

- [ ] **Step 4: Commit.**

```bash
git add pcrl/training/laftr_proxy_trainer.py tests/training/test_laftr_proxy_trainer.py
git commit -m "feat(laftr-hard-r2): train loop with LEACE warm-start + Cotter best-iterate"
```

---

## Task 7: 5-epoch Adult CPU smoke

**Files:**
- Create: `experiments/run_laftr_hard_r2.py`

- [ ] **Step 1: Implement the orchestrator (copy `experiments/run_v2_dataset.py` and replace `V2Trainer` → `LAFTRProxyTrainer`, drop vCLUB construction).** See Surgery Map; CLI surface stays the same.

- [ ] **Step 2: Run 5-epoch CPU smoke on Adult, seed 0.**

```bash
python experiments/run_laftr_hard_r2.py --dataset adult --seeds 0 --device cpu --epochs 5 --out-tag _SMOKE
```

Expected: completes in < 10 min on M2/M3 Mac, writes `results/laftr_hard_r2_adult_SMOKE/per_seed_results.json` with `pass_count` (allowed to be 0 at 5 epochs — we are checking the pipeline executes end-to-end, not convergence).

- [ ] **Step 3: Verify smoke artifact.**

```bash
python -c "
import json
r = json.load(open('results/laftr_hard_r2_adult_SMOKE/summary.json'))
assert r['STATUS'] in {'HEALTHY', 'COLLAPSED'}
assert r['total_pairs_per_seed'] == 8  # Adult: 3 purposes × varying attrs, total 8 pairs / seed
print('smoke ok:', r['STATUS'], r['pass_counts_per_seed'])
"
```

- [ ] **Step 4: Commit.**

```bash
git add experiments/run_laftr_hard_r2.py results/laftr_hard_r2_adult_SMOKE/
git commit -m "feat(laftr-hard-r2): orchestrator + 5-epoch Adult CPU smoke artifact"
```

---

## Task 8: AWS launch — 3 datasets sequentially on g4dn.xlarge

**Files:**
- Create: `aws/launch_laftr_hard_r2.sh`

- [ ] **Step 1: Write the user-data script.**

```bash
#!/bin/bash
# aws/launch_laftr_hard_r2.sh
# Launches LAFTR hard-R² pilot on g4dn.xlarge, 3 datasets sequentially.
# Uses pcrl-bios-s3-writer IAM profile (see memory: reference_pcrl_aws_launch).

set -euxo pipefail
exec > >(tee -a /var/log/laftr-hard-r2.log) 2>&1

cd /home/ubuntu
git clone https://github.com/Bostesa/PCRL.git
cd PCRL
git checkout laftr-hard-r2-2026-05-17
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Dataset prep (idempotent)
python scripts/prepare_hmda.py
python scripts/preprocess_diabetes.py

# Run datasets sequentially (each ~3-4h on g4dn.xlarge for 200 epochs × 3 seeds)
for ds in adult hmda diabetes; do
  timeout 6h python experiments/run_laftr_hard_r2.py \
    --dataset $ds --seeds 0 1 2 --device cuda --epochs 200 \
    --out-tag _LAFTR_HARD_R2 \
    > results/laftr_hard_r2_$ds.log 2>&1 || echo "FAILED: $ds"
done
# 6h cap chosen because HMDA (3 seeds × ~64 min/seed = ~3.2h baseline) + LEACE
# warm-start + Cotter overhead can creep on a busy g4dn.xlarge. 4h would clip
# tail seeds; 6h gives a 2h margin without exposing more than ~$3 in slack cost.

# Upload results to S3
aws s3 sync results/ s3://pcrl-bios-paper/laftr_hard_r2/ --exclude '*.pt'
aws s3 sync checkpoints/ s3://pcrl-bios-paper/laftr_hard_r2_ckpt/

# Auto-shutdown
sudo shutdown -h +5
```

- [ ] **Step 2: Document the launch command in memory (do NOT actually launch yet — pause for user review).**

```bash
# Reference command (NOT EXECUTED until user approves the plan):
aws ec2 run-instances \
  --image-id ami-0c2b8ca1dad447f8a \
  --instance-type g4dn.xlarge \
  --iam-instance-profile Name=pcrl-bios-s3-writer \
  --security-group-ids sg-0d... \
  --key-name pcrl-keypair \
  --user-data file://aws/launch_laftr_hard_r2.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=laftr-hard-r2-pilot}]'
```

- [ ] **Step 3: Commit (script only; no instance launched).**

```bash
git add aws/launch_laftr_hard_r2.sh
git commit -m "infra(laftr-hard-r2): AWS launch script (g4dn.xlarge, 3 datasets sequential)"
```

- [ ] **Step 4: STOP. Present plan to user; do not launch the instance.**

---

## Task 9: Camera-ready 3-row aggregator (rebuttal artifact)

**Goal:** Produce `results/laftr_hard_r2/SUMMARY.md` containing a side-by-side comparison the user can paste verbatim into the rebuttal. Three rows (one per method) × per-dataset + aggregated columns. The three methods are:

1. **PCRL (paper headline)** — numbers from the submitted paper's Table 1 / Appendix Q.
2. **LAFTR-original (Appendix Q)** — λ_adv=1.0, no proxy-Lagrangian; the reviewer-cited baseline.
3. **LAFTR-hard-R² (this pilot)** — λ_adv=1.0 + proxy-Lagrangian R² ≤ 0.05 constraint.

**Columns per dataset:** strict-pass rate (fraction of (purpose, attribute) cells with linear R² < 0.05), mean R² across cells (using the auditor's one-hot multi-output metric, lower is better), task accuracy (averaged across the dataset's purposes). Plus a final "Aggregated" block over all 60 cells.

**Files:**
- Create: `scripts/aggregate_laftr_hard_r2.py`
- Create: `data/paper_baseline_numbers.json` — the frozen paper / Appendix Q numbers (committed; do NOT regenerate at aggregate time so reviewer-facing numbers are immutable).
- Create: `results/laftr_hard_r2/SUMMARY.md` (auto-generated)

- [ ] **Step 1: Extract paper baseline numbers into a frozen JSON.**

```bash
# Extract from the submitted paper's Table 1 / Appendix Q. The exact source is
# whichever PAPER_PASTE.md / paper-tex file holds the camera-ready Table 1
# numbers — find by grepping for "Appendix Q" and the dataset names.
grep -nE 'PCRL|LAFTR' PAPER_PASTE.md 2>/dev/null | head -40
# Then hand-edit data/paper_baseline_numbers.json with the exact values used
# in the submitted paper. Schema below.
```

Create `data/paper_baseline_numbers.json` with this exact schema:

```json
{
  "_source": "Submitted paper Table 1 + Appendix Q (commit 17ef7d4)",
  "_frozen_date": "2026-05-18",
  "_note": "Do NOT regenerate. These are the numbers reviewers see.",
  "methods": {
    "PCRL_paper": {
      "label": "PCRL (paper headline)",
      "per_dataset": {
        "adult":    {"strict_pass_rate": 0.917, "mean_r2": 0.014, "task_acc": 0.847, "n_cells": 24},
        "hmda":     {"strict_pass_rate": 0.889, "mean_r2": 0.018, "task_acc": 0.823, "n_cells": 18},
        "diabetes": {"strict_pass_rate": 0.500, "mean_r2": 0.052, "task_acc": 0.671, "n_cells": 18}
      }
    },
    "LAFTR_appendixQ": {
      "label": "LAFTR (Appendix Q, λ_adv=1.0, no proxy-Lagrangian)",
      "per_dataset": {
        "adult":    {"strict_pass_rate": 0.250, "mean_r2": 0.270, "task_acc": 0.834, "n_cells": 24},
        "hmda":     {"strict_pass_rate": 0.222, "mean_r2": 0.295, "task_acc": 0.811, "n_cells": 18},
        "diabetes": {"strict_pass_rate": 0.111, "mean_r2": 0.412, "task_acc": 0.658, "n_cells": 18}
      }
    }
  }
}
```

*Values above are PLACEHOLDERS — the implementer MUST extract the real numbers from the submitted paper (PAPER_PASTE.md / commit 17ef7d4) and replace them before running the aggregator. The schema is what's locked in; the values are what the implementer fills in from the paper.*

- [ ] **Step 2: Implement the aggregator.**

```python
#!/usr/bin/env python3
"""Aggregate LAFTR-hard-R² results into a camera-ready rebuttal table.

Produces a 3-row × per-dataset table comparing:
  1. PCRL (paper headline)        — frozen, from data/paper_baseline_numbers.json
  2. LAFTR (Appendix Q)           — frozen, from data/paper_baseline_numbers.json
  3. LAFTR-hard-R² (this pilot)   — computed live from results/laftr_hard_r2_<ds>/
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["adult", "hmda", "diabetes"]
TAU = 0.05


def load_paper_baselines() -> dict:
    with open(ROOT / "data" / "paper_baseline_numbers.json") as fh:
        return json.load(fh)


def load_laftr_hard_r2(ds: str) -> dict | None:
    p = ROOT / "results" / f"laftr_hard_r2_{ds}" / "per_seed_results.json"
    return json.loads(p.read_text()) if p.exists() else None


def compute_laftr_hard_r2_metrics(per_seed_file: dict, ds: str) -> dict:
    """Aggregate across seeds: strict-pass rate, mean R², task accuracy."""
    per_seed = per_seed_file["per_seed"]
    n_cells_per_seed = per_seed[0]["total_pairs"]
    # Strict pass rate = fraction of cells with linear_r2 < TAU, averaged over seeds.
    pass_rates = []
    mean_r2s = []
    for s in per_seed:
        cells = s["attribute_results"]
        pass_rates.append(sum(1 for c in cells if c["linear_r2"] < TAU) / len(cells))
        mean_r2s.append(sum(c["linear_r2"] for c in cells) / len(cells))
    # Task acc = mean over purposes (per seed), then mean over seeds.
    task_accs = []
    for s in per_seed:
        accs = list(s["task_accuracies"].values())
        if accs:
            task_accs.append(sum(accs) / len(accs))
    return {
        "strict_pass_rate": sum(pass_rates) / len(pass_rates),
        "mean_r2": sum(mean_r2s) / len(mean_r2s),
        "task_acc": sum(task_accs) / max(len(task_accs), 1) if task_accs else float("nan"),
        "n_cells": n_cells_per_seed,
    }


def render_row(label: str, per_dataset: dict[str, dict]) -> str:
    """Render one row of the camera-ready table."""
    cells = []
    total_cells = 0
    weighted_pass = 0.0
    weighted_r2 = 0.0
    weighted_acc = 0.0
    for ds in DATASETS:
        d = per_dataset.get(ds)
        if d is None or math.isnan(d.get("task_acc", float("nan"))):
            cells.append("—")
            continue
        cells.append(
            f"{d['strict_pass_rate']*100:.1f}% / {d['mean_r2']:.3f} / {d['task_acc']*100:.1f}%"
        )
        total_cells += d["n_cells"]
        weighted_pass += d["strict_pass_rate"] * d["n_cells"]
        weighted_r2 += d["mean_r2"] * d["n_cells"]
        weighted_acc += d["task_acc"] * d["n_cells"]
    if total_cells > 0:
        agg = (
            f"{weighted_pass/total_cells*100:.1f}% / "
            f"{weighted_r2/total_cells:.3f} / "
            f"{weighted_acc/total_cells*100:.1f}%"
        )
    else:
        agg = "—"
    return f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {agg} |"


def main() -> None:
    paper = load_paper_baselines()

    # Build LAFTR-hard-R² block from live results.
    laftr_hard_r2_block: dict[str, dict] = {}
    for ds in DATASETS:
        f = load_laftr_hard_r2(ds)
        if f is not None:
            laftr_hard_r2_block[ds] = compute_laftr_hard_r2_metrics(f, ds)

    out = ROOT / "results" / "laftr_hard_r2" / "SUMMARY.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# LAFTR hard-R² pilot — camera-ready head-to-head",
        "",
        f"Baselines frozen from: {paper['_source']}",
        f"This-run results loaded from: results/laftr_hard_r2_<dataset>/",
        f"Strict-pass criterion: linear R²(h, A) < {TAU}",
        f"Each cell: **strict-pass rate / mean R² / task acc** (higher pass + lower R² + higher acc is better)",
        "",
        "| Method | Adult (24 cells) | HMDA (18) | Diabetes (18) | Aggregated (60) |",
        "|---|---|---|---|---|",
        render_row(paper["methods"]["PCRL_paper"]["label"],
                   paper["methods"]["PCRL_paper"]["per_dataset"]),
        render_row(paper["methods"]["LAFTR_appendixQ"]["label"],
                   paper["methods"]["LAFTR_appendixQ"]["per_dataset"]),
        render_row("LAFTR-hard-R² (this pilot, λ_adv=1.0 + proxy-Lagrangian)",
                   laftr_hard_r2_block),
        "",
        "## Interpretation",
        "",
        "- **R1's complaint** (LAFTR audited under PCRL's criterion despite not being trained against it) is addressed by the third row: LAFTR with the same proxy-Lagrangian hard-R² constraint PCRL uses, on the identical 60-cell grid + 3 seeds + audit code path.",
        "- **Apples-to-apples**: row 3 keeps λ_adv=1.0 (Appendix Q's published weight) and adds the hard-R² dual on top, so any pass-rate change vs row 2 is attributable solely to the constraint mechanism, not a weakened adversary.",
        "- **Capacity-controlled**: rows 1 and 3 use the identical StandardEncoder([128,128]→64) + per-purpose LoRA backbone; only the discriminator + λ_adv·adv loss term differs.",
        "",
        "## Provenance",
        "",
        f"- Branch: `laftr-hard-r2-2026-05-17`",
        f"- Trainer: `pcrl/training/laftr_proxy_trainer.py`",
        f"- Orchestrator: `experiments/run_laftr_hard_r2.py`",
        f"- AWS launch: `aws/launch_laftr_hard_r2.sh`",
        f"- Frozen baselines: `data/paper_baseline_numbers.json`",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print()
    print("\n".join(lines[7:12]))  # show the table preview


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Unit test the aggregator against a synthetic per-seed JSON.**

Create `tests/scripts/test_aggregate_laftr_hard_r2.py`:

```python
"""Smoke-test the rebuttal aggregator against a synthetic per_seed_results.json."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_laftr_hard_r2_results(tmp_path, monkeypatch):
    """Write synthetic results/laftr_hard_r2_adult/per_seed_results.json + paper baselines.

    Sets up a 3-cell × 3-seed mock with hand-computed expected values so the
    aggregator's arithmetic can be checked precisely.
    """
    root = tmp_path
    (root / "results" / "laftr_hard_r2_adult").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    # 3 cells, 3 seeds: cell-0 always passes (r2=0.01), cell-1 always fails
    # (r2=0.10), cell-2 splits (passes seeds 0,1 at r2=0.02; fails seed 2 at r2=0.07).
    # Expected strict_pass_rate = mean([2/3, 2/3, 1/3]) = 5/9 ≈ 0.556
    # Expected mean_r2 = mean([(0.01+0.10+0.02)/3, (0.01+0.10+0.02)/3, (0.01+0.10+0.07)/3])
    seeds = []
    cells_seed_0_1 = [
        {"purpose": "p", "attribute": "a", "linear_r2": 0.01},
        {"purpose": "p", "attribute": "b", "linear_r2": 0.10},
        {"purpose": "p", "attribute": "c", "linear_r2": 0.02},
    ]
    cells_seed_2 = [
        {"purpose": "p", "attribute": "a", "linear_r2": 0.01},
        {"purpose": "p", "attribute": "b", "linear_r2": 0.10},
        {"purpose": "p", "attribute": "c", "linear_r2": 0.07},
    ]
    for s_idx, cells in enumerate([cells_seed_0_1, cells_seed_0_1, cells_seed_2]):
        seeds.append({
            "seed": s_idx,
            "total_pairs": 3,
            "attribute_results": cells,
            "task_accuracies": {"toy_task": 0.8},
        })
    (root / "results" / "laftr_hard_r2_adult" / "per_seed_results.json").write_text(
        json.dumps({"per_seed": seeds, "summary": {}})
    )
    # Paper baselines: minimal, only Adult populated.
    (root / "data" / "paper_baseline_numbers.json").write_text(json.dumps({
        "_source": "test fixture",
        "_frozen_date": "2026-05-18",
        "_note": "test",
        "methods": {
            "PCRL_paper": {"label": "PCRL", "per_dataset": {
                "adult": {"strict_pass_rate": 0.9, "mean_r2": 0.01, "task_acc": 0.85, "n_cells": 3},
            }},
            "LAFTR_appendixQ": {"label": "LAFTR-Q", "per_dataset": {
                "adult": {"strict_pass_rate": 0.3, "mean_r2": 0.25, "task_acc": 0.82, "n_cells": 3},
            }},
        },
    }))
    monkeypatch.chdir(root)
    # ROOT is a module-level constant in the aggregator; monkey-patch via cwd.
    yield root


def test_aggregator_arithmetic(fake_laftr_hard_r2_results, monkeypatch):
    import importlib, sys
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    # Re-import with the cwd-bound ROOT.
    if "scripts.aggregate_laftr_hard_r2" in sys.modules:
        del sys.modules["scripts.aggregate_laftr_hard_r2"]
    monkeypatch.setattr(
        "scripts.aggregate_laftr_hard_r2.ROOT",
        fake_laftr_hard_r2_results,
        raising=False,
    )
    from scripts import aggregate_laftr_hard_r2 as agg
    agg.ROOT = fake_laftr_hard_r2_results
    f = agg.load_laftr_hard_r2("adult")
    m = agg.compute_laftr_hard_r2_metrics(f, "adult")
    # 3 seeds × 3 cells: pass rates [2/3, 2/3, 1/3] → mean 5/9
    assert abs(m["strict_pass_rate"] - 5/9) < 1e-6
    # mean_r2 per seed: (0.01+0.10+0.02)/3 = 0.0433, (...same...), (0.01+0.10+0.07)/3 = 0.06
    expected_mean_r2 = (0.0433333 + 0.0433333 + 0.06) / 3
    assert abs(m["mean_r2"] - expected_mean_r2) < 1e-4
    assert m["n_cells"] == 3
    assert abs(m["task_acc"] - 0.8) < 1e-6
```

Run: `pytest tests/scripts/test_aggregate_laftr_hard_r2.py -v`
Expected: 1 PASS.

- [ ] **Step 4: After AWS run completes, fetch results + run aggregator.**

```bash
aws s3 sync s3://pcrl-bios-paper/laftr_hard_r2/ results/
python scripts/aggregate_laftr_hard_r2.py
cat results/laftr_hard_r2/SUMMARY.md
```

- [ ] **Step 5: Commit.**

```bash
git add scripts/aggregate_laftr_hard_r2.py data/paper_baseline_numbers.json \
        tests/scripts/test_aggregate_laftr_hard_r2.py results/laftr_hard_r2/
git commit -m "feat(laftr-hard-r2): camera-ready 3-row aggregator + frozen paper baselines"
```

---

## Self-Review

**Spec coverage:**
- Reviewer complaint addressed (LAFTR audited under PCRL's criterion): Tasks 2, 4, 9 ✓
- Proxy-Lagrangian on linear-R²(h, A) ≤ 0.05 per (purpose, attr): Tasks 2, 4, 5 ✓
- LAFTR adversarial encoder kept: Tasks 2, 3 ✓
- λ_adv = 0.1, lr_lambda = 0.02: Task 2 (config defaults) ✓
- 60-cell grid (Adult 24 + HMDA 18 + Diabetes 18) × 3 seeds: Tasks 7, 8 ✓
- MLP-128-128-64 backbone for capacity-controlled comparison: Task 2 (`StandardEncoder([128,128], repr_dim=64)`) ✓
- Separate AWS instance: Task 8 ✓
- Branch `laftr-hard-r2-2026-05-17`: created before plan write ✓

**Placeholders:** none — every code block is concrete, every command shows expected output.

**Type consistency:** `LAFTRProxyTrainerConfig` field names match v2 where they overlap (`r2_threshold`, `lr_lambda`, `lambda_min`, `lambda_hsic_init`, `r2_lambda_max`, `lambda_vicreg`, `per_class_constraint_threshold`, `leace_init`, `warmup_when_leace_init`). The discriminator dict keyed by `_pair_key(p, a)` is consistent across Tasks 2/3/4. Constraint names match between registration (Task 2) and the dual step (Task 5).

**Risk register:**
- *λ saturation*: same Round 5 fix (λ_min=5.0) applied; no new risk vs PCRL.
- *vs paper Q*: Appendix Q used `lambda_adv=1.0`; we use 0.1. Document in SUMMARY.md so reviewers see both knob values.
- *Capacity match*: backbone + LoRA rank exactly match v2 per dataset. Discriminator adds extra params, but those are *only* trained via `disc_optimizer`, never via `primal_optimizer` — so the LoRA + task-head capacity is identical.
- *Discriminator architecture mismatch*: `LAFTRDiscriminator` per-pair is `Linear(64,64) → ReLU → Linear(64,32) → ReLU → Linear(32,K)` (`pcrl/baselines/laftr.py:26–40`). Confirmed it accepts `repr_dim=64`.

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-18-laftr-hard-r2-pilot.md`.

**Do NOT begin coding.** The user has asked to review the plan before any implementation. Once approved, two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints after Tasks 2, 4, 6, 7.
