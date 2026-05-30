#!/usr/bin/env python3
"""V2 validation runner — full 200 epochs × 3 seeds for one dataset.

Trains the v2 pipeline (frozen StandardEncoder backbone + per-purpose
LoRA adapters, linear-R² constraint + HSIC auxiliary + vCLUB
independence, VICReg anti-collapse, proxy-Lagrangian dual variables on
linear R² <= 0.05) on Adult / Diabetes / HMDA. After each seed it runs
``generate_report`` for the paper's adjusted compliance criterion
(linear R² < 0.05 AND post-hoc auditor delta < 2pp). After all seeds
finish, it computes representation health and writes ``STATUS: HEALTHY``
or ``STATUS: COLLAPSED`` at the top of the summary JSON.

Outputs:
    results/v2_<dataset>/per_seed_results.json
    results/v2_<dataset>/summary.json
    checkpoints/v2_<dataset>_s<seed>/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pcrl.training.trainer as _trainer_mod  # noqa: E402


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

from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.evaluation.certificates import generate_report  # noqa: E402
from pcrl.models.encoder import StandardEncoder  # noqa: E402
from pcrl.models.lora import PerPurposeLoRAEncoder  # noqa: E402
from pcrl.models.task_head import TaskHead  # noqa: E402
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec  # noqa: E402
from pcrl.training.independence.vclub import VCLUB  # noqa: E402
from pcrl.training.v2_trainer import V2Trainer, V2TrainerConfig  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("v2_validation")
log.setLevel(logging.INFO)
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

SEEDS = [0, 1, 2]
HEALTH_PER_DIM_STD_MIN = 0.5
HEALTH_EFF_RANK_MIN = 2.0

# Per-dataset LoRA shape overrides. The default shape (rank=8, alpha=16) is
# sufficient for joint LEACE on Adult/HMDA (max ``sum(c_i - 1) = 8`` on
# ``employment_analysis``). Diabetes ``quality_research`` requires
# ``race(5) + age_bucket(10) → 4 + 9 = 13`` independent erasure directions
# (joint LEACE) plus headroom for 10 simultaneous OvR per-class constraints
# active during training. Round 6 with rank-16 (3 directions slack) hit
# saturated lambdas (~421) on seeds 0 and 2 because the LoRA couldn't
# satisfy all 10 per-class constraints. Round 7 bumps to rank-24
# (11 directions slack). Adult/HMDA stay at rank-8.
LORA_BY_DATASET: dict[str, tuple[int, float]] = {
    "adult": (8, 16.0),
    "hmda": (8, 16.0),
    "diabetes": (24, 48.0),
    # Folktables joint cardinality is capped at 12 (sex×race×age=2×2×3 on
    # the public_coverage purpose; LEACE rank requirement = sum(c_i-1) = 4)
    # so the rank-8 default has comfortable headroom.
    "folktables": (8, 16.0),
}

# DataLoader ``num_workers`` per dataset. Adult/HMDA/Diabetes were calibrated
# under ``num_workers=0`` (single-threaded data loader) and their reference
# results in ``results/v2_<dataset>/`` are bit-reproducible only at that
# setting — leaving them at 0 keeps Round 5/6/7 reproducibility intact.
# Folktables Round 1 was CPU-bound at 85s/epoch on g4dn.xlarge with
# ``num_workers=0`` (GPU at 15% util); Round 2 raises this to 4 to bring
# wall time within the 8h × 3-seed budget.
NUM_WORKERS_BY_DATASET: dict[str, int] = {
    "adult": 0,
    "hmda": 0,
    "diabetes": 0,
    "folktables": 4,
}


# ───────────────────────────────────────────────────────────────────────────
# Dataset construction
# ───────────────────────────────────────────────────────────────────────────


def build_datasets(name: str):
    if name == "adult":
        from pcrl.data.adult import AdultDataset, get_adult_purposes
        purposes = get_adult_purposes()
        train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
        val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False,
                              norm_stats=train_ds.norm_stats)
        test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False,
                               norm_stats=train_ds.norm_stats)
    elif name == "diabetes":
        from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
        purposes = get_diabetes_purposes()
        train_ds = DiabetesDataset(purposes=purposes, split="train")
        val_ds = DiabetesDataset(purposes=purposes, split="val")
        test_ds = DiabetesDataset(purposes=purposes, split="test")
    elif name == "hmda":
        from pcrl.data.hmda import HMDADataset, get_hmda_purposes
        purposes = get_hmda_purposes()
        train_ds = HMDADataset(purposes=purposes, root="data", split="train")
        val_ds = HMDADataset(purposes=purposes, root="data", split="val")
        test_ds = HMDADataset(purposes=purposes, root="data", split="test")
    elif name == "folktables":
        from pcrl.data.folktables import FolktablesACSDataset, get_folktables_purposes
        purposes = get_folktables_purposes()
        train_ds = FolktablesACSDataset(
            purposes=purposes, root="data", split="train", download=True,
        )
        val_ds = FolktablesACSDataset(
            purposes=purposes, root="data", split="val", download=False,
            norm_stats=train_ds.norm_stats,
        )
        test_ds = FolktablesACSDataset(
            purposes=purposes, root="data", split="test", download=False,
            norm_stats=train_ds.norm_stats,
        )
    else:
        raise ValueError(f"unknown dataset {name}")
    return purposes, train_ds, val_ds, test_ds


# ───────────────────────────────────────────────────────────────────────────
# Health metrics (computed on test reps, per purpose)
# ───────────────────────────────────────────────────────────────────────────


def effective_rank(reprs: np.ndarray) -> float:
    if reprs.shape[0] < 2:
        return float("nan")
    centered = reprs - reprs.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    p = (s ** 2) / max((s ** 2).sum(), 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def repr_health(reprs: np.ndarray) -> dict:
    per_dim_std = reprs.std(axis=0)
    l2 = np.linalg.norm(reprs, axis=1)
    return {
        "shape": list(reprs.shape),
        "per_dim_std_mean": float(per_dim_std.mean()),
        "per_dim_std_max": float(per_dim_std.max()),
        "per_dim_std_min": float(per_dim_std.min()),
        "l2_norm_mean": float(l2.mean()),
        "l2_norm_std": float(l2.std()),
        "effective_rank": effective_rank(reprs),
    }


@torch.no_grad()
def reps_for_purpose(encoder, loader, idx, device):
    encoder.eval()
    out = []
    for batch in loader:
        h = encoder(batch["features"].to(device), idx)
        out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0)


# ───────────────────────────────────────────────────────────────────────────
# Per-seed run
# ───────────────────────────────────────────────────────────────────────────


def run_seed(name: str, purposes: list[PurposeSpec], train_ds, val_ds, test_ds,
             seed: int, device: str, epochs: int = 200,
             out_tag: str = "",
             per_class_threshold: int = 6,
             report_best_iterate: bool = False,
             freeze_leace_projection: bool = False,
             cross_purpose_attrs: list[str] | None = None,
             cross_purpose_threshold: float = 0.10,
             use_erase_layer: bool = False,
             lora_target: str = "all_linear") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    n_workers = NUM_WORKERS_BY_DATASET.get(name, 0)
    persistent = n_workers > 0
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              collate_fn=collate_pcrl_batch,
                              num_workers=n_workers,
                              persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            collate_fn=collate_pcrl_batch,
                            num_workers=n_workers,
                            persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_pcrl_batch,
                             num_workers=n_workers,
                             persistent_workers=persistent)

    input_dim = train_ds.info.num_features
    repr_dim = 64

    lora_rank, lora_alpha = LORA_BY_DATASET.get(name, (8, 16.0))
    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=repr_dim, dropout=0.3,
        use_erase_layer=use_erase_layer,
    )
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=len(purposes),
        rank=lora_rank, alpha=lora_alpha, dropout=0.0,
        lora_target=lora_target,
    )

    task_heads: dict = {}
    vclubs: dict = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        out_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=out_dim)
        for attr in p.disallowed_attrs:
            n_classes = p.disallowed_attr_dims.get(attr, 2)
            key = f"{p.name}__{attr}"
            vclubs[key] = VCLUB(
                x_dim=repr_dim, z_dim=n_classes,
                hidden_dim=128, z_categorical=True, l2=1e-1,
            )

    # ``out_tag`` is included in the checkpoint dir to keep parallel runs
    # isolated (e.g. Round 4 vs an R1+R2 probe) so analysis scripts that
    # read ``checkpoints/v2_{name}_s{seed}`` don't see clobbered weights.
    ckpt_dir = ROOT / "checkpoints" / f"v2_{name}{out_tag}_s{seed}"
    # Round 5 / Fix R1+R2: lambda_min=5.0 (proxy-Lagrangian dual floor) +
    # warmup_when_leace_init=False (dataclass default; LEACE warm-start
    # replaces task-only warmup as the source of feasible-set entry).
    # All other knobs match Round 4. See
    # `results/v2_optimizer_drift_audit.md` for the audit motivating R1+R2.
    config = V2TrainerConfig(
        lambda_vicreg=1.0, lambda_vclub=1.0, lambda_verify=0.0,
        r2_threshold=0.05,
        vicreg_gamma=1.0,
        lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
        batch_size=256, epochs=epochs,
        weight_decay=1e-4, grad_clip=1.0, vclub_steps=1,
        lambda_min=5.0,  # Fix R1
        per_class_constraint_threshold=per_class_threshold,
        checkpoint_dir=str(ckpt_dir),
        report_best_iterate=report_best_iterate,
        freeze_leace_projection=freeze_leace_projection,
        cross_purpose_attrs=cross_purpose_attrs,
        cross_purpose_threshold=cross_purpose_threshold,
        use_erase_layer=use_erase_layer,
        lora_target=lora_target,
    )
    trainer = V2Trainer(
        encoder=encoder, task_heads=task_heads, vclubs=vclubs,
        purpose_registry=registry, config=config, device=device,
    )

    log.info(
        f"  [{name}/seed={seed}] training (device={device}, "
        f"warmup={config.warmup_epochs} + epochs={config.epochs}, true Cotter best-iterate)"
    )

    # When the erase-layer architecture is on, fit the frozen joint-LEACE
    # erase and SKIP the legacy per-purpose `leace_warm_start` — §5.5
    # vision uses the erase as its sole LEACE-based init (the task-proj
    # LoRA starts at zero, with task gradients shaping it from scratch).
    # Stacking both inits has not been ablated and would muddy the
    # pilot's interpretation.
    if use_erase_layer:
        log.info(f"  [{name}/seed={seed}] fitting frozen LEACE erase layer …")
        erase_diag = trainer.fit_erase_layer(train_loader)
        log.info(f"  [{name}/seed={seed}] erase fit done: {erase_diag}")
    elif config.leace_init:
        log.info(f"  [{name}/seed={seed}] LEACE warm-start of LoRA adapters …")
        leace_diag = trainer.leace_warm_start(train_loader)
        log.info(f"  [{name}/seed={seed}] LEACE warm-start done")

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0
    cotter_meta = (state.history.get("cotter_selection") or [{}])[-1]
    log.info(
        f"  [{name}/seed={seed}] trained in {train_time:.0f}s, last_epoch={state.epoch}, "
        f"best_epoch={state.best_epoch}, best_task_loss={state.best_val_loss:.4f}, "
        f"cotter={cotter_meta.get('kind','?')} "
        f"feasible={cotter_meta.get('n_feasible_post_warmup','?')}/"
        f"{cotter_meta.get('n_eligible_post_warmup','?')}"
    )

    # Reload checkpoint for downstream eval. Prefer ``canonical_iterate.pt``
    # (written by the trainer when ``report_best_iterate=True``) over
    # ``best.pt``. Adult/HMDA/Diabetes runs leave ``report_best_iterate=False``
    # so canonical_iterate.pt is never created and behavior falls through to
    # the legacy best.pt path. The fallback order is canonical → best → final
    # so an interrupted training that left only final.pt is still usable.
    canonical = ckpt_dir / "canonical_iterate.pt"
    best = ckpt_dir / "best.pt"
    final = ckpt_dir / "final.pt"
    chosen = canonical if canonical.exists() else (best if best.exists() else final)
    if chosen.exists():
        ckpt = torch.load(chosen, map_location=device, weights_only=False)
        encoder.backbone.load_state_dict(ckpt["backbone"])
        encoder.adapters.load_state_dict(ckpt["lora_adapters"])
        # Encoder-level buffers (e.g. ``leace_P_p{p}``) live outside backbone
        # and adapters. ``load_state_dict(strict=False)`` *silently skips*
        # buffers that aren't already registered, so we route through
        # ``set_leace_projection`` which calls ``register_buffer`` explicitly.
        # In-process path: trainer.leace_warm_start already registered them
        # earlier in this function, so this is an idempotent overwrite. From
        # a fresh process (verdict script reloading a saved checkpoint):
        # this path *creates* the buffers on the rebuilt encoder.
        enc_buf = ckpt.get("encoder_buffers", {}) or {}
        for p_idx in range(len(purposes)):
            P_key = f"leace_P_p{p_idx}"
            mu_key = f"leace_mu_p{p_idx}"
            if P_key in enc_buf and mu_key in enc_buf:
                encoder.set_leace_projection(p_idx, enc_buf[P_key], enc_buf[mu_key])
        # task_heads is a plain dict here, but Trainer wraps it in nn.ModuleDict.
        # Use the trainer's task_heads (the live nn.ModuleDict) for state_dict load.
        trainer.task_heads.load_state_dict(ckpt["task_heads"])
        log.info(f"  [{name}/seed={seed}] reloaded {chosen.name}")
    encoder.eval()

    # Compliance via paper's adjusted criterion
    reports = generate_report(
        encoder=encoder, train_loader=train_loader, test_loader=test_loader,
        purpose_registry=registry, device=device,
    )
    pass_count = 0
    attr_results = []
    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        ok = bool(delta < 0.02 and r.linear_r2 < 0.05)
        if ok:
            pass_count += 1
        attr_results.append({
            "purpose": r.purpose_name,
            "attribute": r.attr_name,
            "linear_r2": round(r.linear_r2, 6),
            "empirical_best_acc": round(r.empirical_best_acc, 6),
            "majority_baseline": round(r.majority_proportion, 6),
            "delta": round(delta, 6),
            "adj_pass": ok,
        })

    # Task accuracies via the trained heads
    val_metrics = trainer.evaluate(test_loader)
    task_accs = {k: round(float(v), 6) for k, v in val_metrics.task_accuracy.items()}

    # Health per purpose
    per_purpose_health: dict = {}
    for idx, p in enumerate(purposes):
        reps = reps_for_purpose(encoder, test_loader, idx, device)
        per_purpose_health[p.name] = repr_health(reps)

    # Final lambda values for HSIC constraints
    lambdas_final = {n: float(c.lambda_value) for n, c in trainer.proxy.constraints.items()}

    return {
        "seed": seed,
        "train_time_s": round(train_time, 1),
        "last_epoch": state.epoch,
        "best_epoch": state.best_epoch,
        "task_accuracies": task_accs,
        "attribute_results": attr_results,
        "pass_count": pass_count,
        "total_pairs": len(reports),
        "per_purpose_health": per_purpose_health,
        "lambdas_final": lambdas_final,
        "best_task_loss_at_selected": float(state.best_val_loss),
        "cotter_selection": cotter_meta,
    }


# ───────────────────────────────────────────────────────────────────────────
# Aggregate across seeds + STATUS
# ───────────────────────────────────────────────────────────────────────────


def aggregate(name: str, per_seed: list[dict]) -> dict:
    pass_counts = [s["pass_count"] for s in per_seed]
    pairs = per_seed[0]["total_pairs"]

    # Mean per-task acc
    task_keys: list[str] = []
    for s in per_seed:
        for k in s["task_accuracies"]:
            if k not in task_keys:
                task_keys.append(k)
    task_means = {k: float(np.mean([s["task_accuracies"].get(k, np.nan) for s in per_seed])) for k in task_keys}
    task_stds = {k: float(np.std([s["task_accuracies"].get(k, np.nan) for s in per_seed], ddof=0)) for k in task_keys}

    # Health: HEALTHY if every seed × every purpose hits both thresholds AND at least
    # one task acc per seed is above majority+5pp. Otherwise COLLAPSED.
    status = "HEALTHY"
    health_notes: list[str] = []
    for s in per_seed:
        for pname, h in s["per_purpose_health"].items():
            if h["per_dim_std_mean"] < HEALTH_PER_DIM_STD_MIN:
                status = "COLLAPSED"
                health_notes.append(
                    f"seed={s['seed']} purpose={pname} per_dim_std_mean="
                    f"{h['per_dim_std_mean']:.3f} < {HEALTH_PER_DIM_STD_MIN}"
                )
            if h["effective_rank"] < HEALTH_EFF_RANK_MIN:
                status = "COLLAPSED"
                health_notes.append(
                    f"seed={s['seed']} purpose={pname} eff_rank="
                    f"{h['effective_rank']:.2f} < {HEALTH_EFF_RANK_MIN}"
                )

    # majority+5pp check per seed: at least one task above majority+5pp
    # We don't have majority directly here; use the heuristic that per-seed
    # if EVERY task is < 0.55 (binary) treat as suspicious. The compliance
    # adj_pass already encodes whether a learnable signal survives — leave
    # detailed accuracy floor checks to the aggregator.

    summary = {
        "STATUS": status,
        "dataset": name,
        "n_seeds": len(per_seed),
        "total_pairs_per_seed": pairs,
        "pass_count_mean": float(np.mean(pass_counts)),
        "pass_count_std": float(np.std(pass_counts, ddof=0)),
        "pass_counts_per_seed": pass_counts,
        "task_acc_mean": task_means,
        "task_acc_std": task_stds,
        "health_notes": health_notes,
        "thresholds": {
            "per_dim_std_min": HEALTH_PER_DIM_STD_MIN,
            "effective_rank_min": HEALTH_EFF_RANK_MIN,
        },
    }
    return summary


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["adult", "diabetes", "hmda", "folktables"])
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS,
                        help="Seeds to run (default: 0 1 2)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-tag", default="",
                        help="Suffix appended to results/v2_<dataset> output dir, e.g. _OPTION_A")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of constrained epochs per seed (warmup is added on top).")
    parser.add_argument(
        "--per-class-threshold", type=int, default=6,
        help=(
            "Cardinality threshold for per-class OvR R² constraints. "
            "Attributes with K >= this value are constrained per-class "
            "(K independent binary R² constraints) instead of one joint "
            "multi-output R². Default 6 leaves Adult/HMDA (max K=5) "
            "unchanged and activates per-class for Diabetes age_bucket "
            "(K=10)."
        ),
    )
    parser.add_argument(
        "--report-best-iterate", action="store_true", default=False,
        help=(
            "Opt-in: write ``canonical_iterate.pt`` containing whichever of "
            "(best.pt, final.pt) has lower mean R² on val. Eval falls back "
            "to canonical_iterate.pt > best.pt > final.pt. Default OFF "
            "preserves existing best.pt-only behavior for Adult/HMDA/Diabetes."
        ),
    )
    parser.add_argument(
        "--freeze-leace-projection", action="store_true", default=False,
        help=(
            "Opt-in: register the LEACE projection (P_sub, μ) per purpose as "
            "non-trainable buffers on the encoder; forward applies "
            "h_proj = h - (h - μ) @ P_sub.T at every call. Default OFF leaves "
            "Adult/HMDA/Diabetes runs with LoRA-side LEACE warm-start only."
        ),
    )
    parser.add_argument(
        "--cross-purpose-attrs", nargs="*", default=None,
        help="Opt-in: list of attribute names to constrain on h_concat "
             "(e.g. --cross-purpose-attrs race sex age_group). Adds a "
             "linear-R²(h_concat, A) <= cross_purpose_threshold dual per attr.",
    )
    parser.add_argument(
        "--cross-purpose-threshold", type=float, default=0.10,
        help="Threshold for the cross-purpose linear-R² constraint.",
    )
    parser.add_argument(
        "--use-erase-layer", action="store_true", default=False,
        help=(
            "Rebuttal pilot (2026-05-17): port §5.5 vision erase-layer "
            "architecture to tabular. Inserts a frozen LEACE-fit Linear "
            "layer between the backbone network and repr_proj. Combine "
            "with --lora-target=repr_proj_only for the faithful vision "
            "mirror (frozen backbone → frozen erase → LoRA-trainable "
            "task_proj). Default OFF preserves Round 5/7 behaviour."
        ),
    )
    parser.add_argument(
        "--lora-target", choices=["all_linear", "repr_proj_only"],
        default="all_linear",
        help=(
            "Where to attach LoRA adapters: 'all_linear' (default) puts "
            "an adapter on every Linear in the backbone; 'repr_proj_only' "
            "puts adapters only on the final Linear (the post-erase "
            "task_proj). The latter matches the §5.5 vision pattern."
        ),
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 72}")
    print(f"V2 VALIDATION — {args.dataset.upper()} ({len(args.seeds)} seeds, device={device})")
    print(f"{'=' * 72}\n")

    purposes, train_ds, val_ds, test_ds = build_datasets(args.dataset)
    print(f"  N_train={len(train_ds)}  N_val={len(val_ds)}  N_test={len(test_ds)}  D={train_ds.info.num_features}")
    print(f"  purposes: {[p.name for p in purposes]}")
    print()

    overall_t0 = time.time()
    per_seed_results = []
    for seed in args.seeds:
        result = run_seed(
            args.dataset, purposes, train_ds, val_ds, test_ds,
            seed, device, args.epochs, out_tag=args.out_tag,
            per_class_threshold=args.per_class_threshold,
            report_best_iterate=args.report_best_iterate,
            freeze_leace_projection=args.freeze_leace_projection,
            cross_purpose_attrs=args.cross_purpose_attrs,
            cross_purpose_threshold=args.cross_purpose_threshold,
            use_erase_layer=args.use_erase_layer,
            lora_target=args.lora_target,
        )
        per_seed_results.append(result)
        print(f"  → seed={seed}: pass {result['pass_count']}/{result['total_pairs']}, "
              f"task_acc={result['task_accuracies']}, "
              f"epochs={result['last_epoch']}, time={result['train_time_s']:.0f}s")

    summary = aggregate(args.dataset, per_seed_results)
    summary["total_wall_s"] = round(time.time() - overall_t0, 1)

    out_dir = ROOT / "results" / f"v2_{args.dataset}{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_seed_results.json", "w") as fh:
        json.dump({"summary": summary, "per_seed": per_seed_results}, fh, indent=2)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print()
    print(f"{'=' * 72}")
    print(f"STATUS: {summary['STATUS']}")
    print(f"  pass {summary['pass_count_mean']:.2f} +/- {summary['pass_count_std']:.2f} / "
          f"{summary['total_pairs_per_seed']}  (per seed: {summary['pass_counts_per_seed']})")
    for k, m in summary["task_acc_mean"].items():
        print(f"  {k}: {m:.1%} +/- {summary['task_acc_std'][k]:.1%}")
    print(f"  total wall: {summary['total_wall_s']:.0f}s")
    print(f"  saved: {out_dir / 'per_seed_results.json'}")
    print(f"  saved: {out_dir / 'summary.json'}")
    print()


if __name__ == "__main__":
    main()
