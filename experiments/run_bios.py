#!/usr/bin/env python3
"""BIOS-medium training driver.

Single CLI entry for both phases:
- Phase 1 (default): one marginal gender constraint, single-seed, ~4.5 hr on T4.
- Phase 2: 10 per-occupation conditional gender + 1 marginal = 11 constraints.
  Phase 2 is GATED — it errors out unless ``--phase1-ok`` is passed AND a
  Phase-1 success checkpoint exists in ``--output-dir``.

Diagnostics-only mode (``--diagnostics-only``) runs the four pre-launch
diagnostics and exits without training. Use this on Mac before launching
on AWS.

Reuses (import-only, never modified):
  pcrl.training.proxy_lagrangian.{Constraint, ProxyLagrangianOptimizer}
  pcrl.training.losses.VerificationRegularizer
  pcrl.models.task_head.TaskHead
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcrl.language import (  # noqa: E402
    BertWithLoRA,
    BIOS_TOP10,
    EmaCrossCovPIController,
    FiveSignalMonitor,
    OnlineLeaceRefit,
    bio_length_by_gender,
    build_bios_loaders,
    cls_shape_trace,
    construction_r2,
    describe_modules,
    gender_one_hot,
    leace_warm_start_bert,
    nhsic_linear,
    theil_adjusted_r2,
    tpr_gap_summary,
    tpr_gaps_per_occupation,
)
from pcrl.language.per_class_ovr_constraints import (  # noqa: E402
    build_phase2_constraints,
    evaluate_constraints as evaluate_phase2_constraints,
)
from pcrl.models.task_head import TaskHead  # noqa: E402
from pcrl.training.losses import VerificationRegularizer  # noqa: E402
from pcrl.training.proxy_lagrangian import (  # noqa: E402
    Constraint,
    ProxyLagrangianOptimizer,
)


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pick_device(arg: str) -> torch.device:
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but no CUDA device.")
        return torch.device("cuda")
    if arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS unavailable.")
        return torch.device("mps")
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


# --------------------------------------------------------------------------
# Diagnostics-only mode
# --------------------------------------------------------------------------

def run_diagnostics(args, *, device: torch.device, output_dir: Path) -> dict:
    print("=" * 78)
    print("BIOS-medium pre-launch diagnostics")
    print(f"device={device}  seed={args.seed}  output_dir={output_dir}")
    print("=" * 78)

    print("\n[1/5] Loading dataset, building DataLoaders...")
    train_loader, dev_loader, info, train_ds, dev_ds = build_bios_loaders(
        n_train=args.n_train,
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )
    print(f"  n_train={info['n_train']}  n_dev={info['n_dev']}")
    print(f"  occupations={info['occupations']}")

    print("\n[2/5] Building BertWithLoRA + PEFT injection...")
    model = BertWithLoRA(
        rank=args.rank, alpha=args.alpha, dropout=args.dropout,
    ).to(device)

    print("\n--- Diagnostic #1: named modules + requires_grad after PEFT ---")
    desc = describe_modules(model)
    for k, v in desc.items():
        print(f"  {k}: {v}")
    expected_qv = 24
    expected_od = 1
    if desc["lora_q_v_adapters"] != expected_qv:
        print(f"  WARNING: expected {expected_qv} Q/V LoRA adapters, "
              f"got {desc['lora_q_v_adapters']}")
    if desc["lora_output_dense_adapters"] != expected_od:
        print(f"  WARNING: expected {expected_od} output.dense LoRA adapter, "
              f"got {desc['lora_output_dense_adapters']}")
    ratio = desc["ratio_trainable"]
    if not (0.005 <= ratio <= 0.025):
        print(f"  WARNING: trainable ratio {ratio:.4%} is outside "
              f"the expected 1-2% band (0.5%-2.5% accepted).")

    print("\n--- Diagnostic #2: [CLS] shape trace ---")
    sample_batch = next(iter(dev_loader))
    trace = cls_shape_trace(model, sample_batch, device)
    for k, v in trace.items():
        print(f"  {k}: {v}")
    if not trace["matches"]:
        raise RuntimeError(
            "[CLS] extracted via .last_hidden_state[:, 0, :] does NOT match "
            "model.forward output — the wrapper has a bug."
        )

    print("\n--- Diagnostic #3: bio length distribution by gender ---")
    # Decode the original token strings directly from the train_ds tensors:
    # since we already tokenised, recover lengths from un-truncated re-tokenisation
    # using the dataset's underlying text.
    from datasets import load_dataset
    full_train = load_dataset("LabHC/bias_in_bios", split="train")
    from pcrl.language.bios_dataset import (
        BIOS_TOP10_IDS, _filter_top10, _stratified_subsample,
    )
    train_filtered = _filter_top10(full_train)
    occ_int = np.array(train_filtered["profession"])
    gen_int = np.array(train_filtered["gender"])
    label_to_local = {pid: i for i, pid in enumerate(BIOS_TOP10_IDS)}
    occ_local = np.array([label_to_local[p] for p in occ_int])
    keep_idx = _stratified_subsample(
        occ_local, gen_int, args.n_train, args.seed,
    )
    train_sub = train_filtered.select(keep_idx.tolist())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    length_stats = bio_length_by_gender(
        texts=train_sub["hard_text"],
        genders=train_sub["gender"],
        tokenizer=tok,
        max_length=args.max_length,
    )
    for k, v in length_stats.items():
        print(f"  {k}: {v}")

    print("\n[3/5] LEACE warm-start on layer-11 output.dense...")
    leace_diag = leace_warm_start_bert(model, train_loader, device=device)
    for k, v in leace_diag.items():
        print(f"  {k}: {v}")

    print("\n--- Diagnostic #4: construction-time linear-R²([CLS], gender) ---")
    print(f"  (full dev split, disjoint from LEACE-fit data; using full set "
          f"keeps OLS overfit bias d/N << 0.05)")
    constr = construction_r2(
        model, dev_loader, device,
        max_samples=None, abort_threshold=0.05, decimal_places=4,
    )
    for k, v in constr.items():
        print(f"  {k}: {v}")
    print(f"\n  Construction R² (4 dp): {constr['r2_rounded']:.4f}  "
          f"abort_threshold=0.05  passed={constr['passed']}")

    out = {
        "info": info,
        "diagnostic_1_modules": desc,
        "diagnostic_2_shape_trace": trace,
        "diagnostic_3_bio_length_by_gender": length_stats,
        "leace_warmstart": leace_diag,
        "diagnostic_4_construction_r2": constr,
        "device": str(device),
        "seed": args.seed,
    }
    _save_json(output_dir / "diagnostics.json", out)
    print(f"\nSaved diagnostics to {output_dir / 'diagnostics.json'}")
    return out


# --------------------------------------------------------------------------
# Phase 1 training loop
# --------------------------------------------------------------------------

def _build_phase1_constraints(*, threshold: float, eta: float, lambda_min: float,
                              lambda_init: float, lambda_max: float) -> list[Constraint]:
    return [
        Constraint(
            name="marginal_gender",
            threshold=threshold,
            direction="<=",
            eta_lambda=eta,
            lambda_init=lambda_init,
            lambda_max=lambda_max,
            lambda_min=lambda_min,
        )
    ]


@torch.no_grad()
def _holdout_r2(model, holdout_loader, device, *, ridge: float = 1e-6) -> dict:
    """Closed-form ridge R²([CLS], gender) on a fixed held-out batch +
    Theil-adjusted R² for the dual signal.

    Round 2 (set 2026-05-03): the dual update consumes ``adj_r2`` rather than
    raw ``r2``. Theil's adjusted R² has expectation 0 under independence at
    any (n, d) — the OLS noise floor of d/(n-1) ≈ 0.19 at d=768, n=4096
    that crippled Round 1's dual is corrected away. ``τ = 0.05`` becomes
    a meaningful threshold for the dual ascent.

    Returns a dict with both raw and adjusted R² so callers can choose.
    """
    was_training = model.training
    model.eval()
    z_blocks: list[torch.Tensor] = []
    g_blocks: list[torch.Tensor] = []
    for batch in holdout_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        z = model(ids, mask)
        z_blocks.append(z.detach().cpu())
        g_blocks.append(batch["gender"].long())
    if was_training:
        model.train()
    Z = torch.cat(z_blocks, dim=0).float().numpy().astype(np.float64)
    g = torch.cat(g_blocks, dim=0).numpy().astype(np.int64)
    Z_oh = np.eye(2)[g].astype(np.float64)
    H_c = Z - Z.mean(axis=0, keepdims=True)
    Z_c = Z_oh - Z_oh.mean(axis=0, keepdims=True)
    gram = H_c.T @ H_c + ridge * np.eye(H_c.shape[1])
    W = np.linalg.solve(gram, H_c.T @ Z_c)
    Z_pred = H_c @ W
    ss_res = ((Z_c - Z_pred) ** 2).sum()
    ss_tot = (Z_c ** 2).sum()
    r2 = float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12)))
    n = int(Z.shape[0])
    d = int(Z.shape[1])
    return {
        "r2": r2,
        "adj_r2": theil_adjusted_r2(r2, n, d),
        "n": n,
        "d": d,
    }


@torch.no_grad()
def _evaluate_dev(
    model, task_head, dev_loader, verifier, device,
) -> dict:
    model.eval()
    task_head.eval()
    correct = 0
    total = 0
    z_blocks = []
    g_blocks = []
    occ_blocks = []
    for batch in dev_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        gen = batch["gender"].long().to(device)
        occ = batch["occupation"].long().to(device)
        z = model(ids, mask)
        logits = task_head(z)
        pred = logits.argmax(-1)
        correct += int((pred == occ).sum().item())
        total += int(occ.numel())
        z_blocks.append(z.detach().cpu())
        g_blocks.append(gen.detach().cpu())
        occ_blocks.append(occ.detach().cpu())
    Z = torch.cat(z_blocks, dim=0)
    G = torch.cat(g_blocks, dim=0)
    OCC = torch.cat(occ_blocks, dim=0)

    # Closed-form ridge OLS R² on full dev (same formula as _holdout_r2 / the
    # construction-time check) so values are directly comparable end-to-end.
    Z_np = Z.float().numpy().astype(np.float64)
    g_np = G.numpy().astype(np.int64)
    Z_oh = np.eye(2)[g_np].astype(np.float64)
    H_c = Z_np - Z_np.mean(axis=0, keepdims=True)
    Z_c = Z_oh - Z_oh.mean(axis=0, keepdims=True)
    gram = H_c.T @ H_c + 1e-6 * np.eye(H_c.shape[1])
    W = np.linalg.solve(gram, H_c.T @ Z_c)
    ss_res = ((Z_c - H_c @ W) ** 2).sum()
    ss_tot = (Z_c ** 2).sum()
    r2_marginal = float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12)))
    n_dev_eff = int(Z_np.shape[0])
    d_dev = int(Z_np.shape[1])
    adj_r2 = theil_adjusted_r2(r2_marginal, n_dev_eff, d_dev)

    # Per-occupation TPR gap (BIOS standard fairness metric, De-Arteaga 2019).
    # Round 2 reports this as the **primary** metric; R² is diagnostic.
    preds_np = []
    occ_np = OCC.numpy()
    # Re-run task head over Z (already on CPU) to recover predictions without
    # another model forward pass — matches the predictions counted in
    # ``correct`` above for ``top10_acc`` consistency.
    with torch.no_grad():
        task_head.cpu()
        logits = task_head(Z.float())
        preds_np = logits.argmax(-1).numpy()
        task_head.to(device)
    gaps = tpr_gaps_per_occupation(
        preds_np, occ_np, g_np, n_occupations=10,
    )
    tpr = tpr_gap_summary(gaps)

    return {
        "top10_acc": correct / max(total, 1),
        "marginal_r2": r2_marginal,
        "marginal_adj_r2": adj_r2,
        "tpr_rms_gap": tpr["rms_gap"],
        "tpr_max_abs_gap": tpr["max_abs_gap"],
        "tpr_per_occupation": tpr["per_occupation"],
        "n_dev": int(total),
        "Z": Z, "G": G, "OCC": OCC,
    }


def _check_bail(
    elapsed_s: float,
    top10_acc: float,
    marg_adj_r2: float,
    *,
    phase: int,
) -> str | None:
    """Return a reason string if a bail condition is hit, else None.

    Round 2 thresholds (set 2026-05-03) operate on Theil-adjusted R²
    (``marg_adj_r2``) rather than raw R² — adj_R² has expectation 0 under
    independence at any (n, d), so τ values are meaningful.
    """
    if phase == 1:
        if elapsed_s >= 30 * 60 and top10_acc < 0.65:
            return ("BAIL @ 30 min: top-10 acc {:.3f} < 0.65 — likely LoRA "
                    "wiring is broken.".format(top10_acc))
        if elapsed_s >= 30 * 60 and marg_adj_r2 > 0.30:
            return ("BAIL @ 30 min: marginal adj-R² {:.4f} > 0.30 — "
                    "estimator stack failed early.".format(marg_adj_r2))
        if elapsed_s >= 60 * 60 and marg_adj_r2 > 0.20:
            return ("BAIL @ 1 hr: marginal adj-R² {:.4f} > 0.20 — "
                    "constraint not converging.".format(marg_adj_r2))
        if elapsed_s >= 120 * 60 and marg_adj_r2 > 0.10:
            return ("BAIL @ 2 hr: marginal adj-R² {:.4f} > 0.10 — "
                    "this seed will not converge.".format(marg_adj_r2))
        return None
    return None  # Phase-2 bail logic handled in run_phase2 (multi-constraint)


def run_phase1(args, *, device: torch.device, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)

    print("=" * 78)
    print(f"BIOS-medium PHASE 1 (single marginal gender constraint)")
    print(f"device={device}  seed={args.seed}  output_dir={output_dir}")
    print("=" * 78)

    train_loader, dev_loader, info, train_ds, dev_ds = build_bios_loaders(
        n_train=args.n_train,
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )
    # The full-train loader is what LEACE warm-start fits on (matches the
    # n=50K used during diagnostics). Carving the held-out 4096 OUT of the
    # LEACE-fit data drops eraser quality enough to push the construction-R²
    # check over 0.05 (observed on AWS launch #2: 0.045→0.078 at n=45,904).
    # Held-out exclusion is for the dual signal (no gradient leakage), not
    # for the one-shot eraser fit. Keep the original loader as
    # ``full_train_loader`` and reassign ``train_loader`` to the carved
    # primal subset below.
    full_train_loader = train_loader
    # Option D: carve out a 4096-sample held-out subset (deterministic seed,
    # no overlap with primal mini-batches). Dual update reads R² off this
    # batch every K=holdout_refresh_every primal steps.
    rng_holdout = np.random.default_rng(args.seed + 7919)
    holdout_idx = rng_holdout.choice(
        len(train_ds), size=args.holdout_size, replace=False,
    )
    holdout_set = set(holdout_idx.tolist())
    primal_idx = np.array(
        [i for i in range(len(train_ds)) if i not in holdout_set]
    )
    primal_train_ds = Subset(train_ds, primal_idx.tolist())
    holdout_ds = Subset(train_ds, holdout_idx.tolist())
    train_loader = DataLoader(
        primal_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=args.holdout_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    print(f"n_train={info['n_train']}  n_dev={info['n_dev']}  "
          f"batch_size={args.batch_size}")
    print(f"primal train n={len(primal_train_ds)}  holdout n={len(holdout_ds)}  "
          f"refresh_every={args.holdout_refresh_every}  "
          f"holdout_eval_bs={args.holdout_eval_batch_size}")

    # ── INVARIANT: three disjoint dataset roles wired correctly ──────────
    # full_train_loader   → LEACE one-shot eraser fit (NEEDS full n_train)
    # train_loader        → primal mini-batches      (carved primal subset)
    # holdout_loader      → dual signal R² eval      (4096 held-out)
    # Adding a guard here prevents future refactors from silently dropping
    # the LEACE-fit size below n_train and breaking the construction-R²
    # check (the bug that crashed AWS launch #2).
    n_full_for_leace = len(full_train_loader.dataset)
    print(f"[invariant] LEACE-fit dataset size = {n_full_for_leace}, "
          f"primal-train size = {len(primal_train_ds)}, "
          f"holdout size = {len(holdout_ds)}")
    if n_full_for_leace < args.n_train:
        raise RuntimeError(
            f"[invariant FAILED] full_train_loader.dataset has "
            f"{n_full_for_leace} samples but --n-train={args.n_train}. "
            f"LEACE warm-start MUST fit on the full subsample. Did a "
            f"refactor pass the carved primal subset to "
            f"leace_warm_start_bert by mistake?"
        )
    # primal + holdout MUST exactly partition the full set (no overlap, no
    # missing indices) — critical for Option D's no-gradient-leakage property.
    if len(primal_train_ds) + len(holdout_ds) != n_full_for_leace:
        raise RuntimeError(
            f"[invariant FAILED] primal({len(primal_train_ds)}) + "
            f"holdout({len(holdout_ds)}) != full({n_full_for_leace}). "
            f"Carve-out is broken — there is overlap or missing indices."
        )

    model = BertWithLoRA(
        rank=args.rank, alpha=args.alpha, dropout=args.dropout,
    ).to(device)
    desc = describe_modules(model)
    print(f"PEFT injection: {desc['lora_q_v_adapters']} Q/V + "
          f"{desc['lora_output_dense_adapters']} output.dense LoRAs; "
          f"trainable={desc['trainable']:,}  frozen={desc['frozen']:,}  "
          f"ratio={desc['ratio_trainable']:.4%}")

    print(f"LEACE fitting on N={n_full_for_leace} samples")
    leace_diag = leace_warm_start_bert(model, full_train_loader, device=device)
    post_eraser = leace_diag["post_r2_train_eraser_only"]
    print(f"  LEACE: pre_r2={leace_diag['pre_r2_train']:.4f} → "
          f"post_r2_eraser_only={post_eraser:.4f}")
    print(f"  LEACE post-R²(eraser only): {post_eraser:.4f} "
          f"(expected ≈ 0.008 at N=50000)")
    try:
        constr = construction_r2(model, dev_loader, device, max_samples=None)
    except RuntimeError as e:
        if not args.allow_warmstart_fail:
            raise
        # Smoke-test mode: LEACE warm-start at small n_train cannot generalize
        # to dev (R² scales ~ d/n). Down-grade abort to warning so the rest of
        # the training-loop wiring (Option D, monitor, Plan-B trip detection)
        # can be exercised end-to-end. Production runs MUST not pass this flag.
        print(f"  WARNING --allow-warmstart-fail: {e}")
        constr = construction_r2(
            model, dev_loader, device,
            max_samples=None, abort_threshold=1.0,
        )
    print(f"  Construction-time R² (full dev, n={constr['n_held_out']}): "
          f"{constr['r2_rounded']:.4f}  passed={constr['passed']}")

    task_head = TaskHead(repr_dim=768, output_dim=10, hidden_dim=64).to(device)
    verifier = VerificationRegularizer(regularization=1e-4).to(device)

    constraints = _build_phase1_constraints(
        threshold=args.r2_threshold,
        eta=args.dual_lr,
        lambda_min=args.lambda_min,
        lambda_init=args.lambda_init,
        lambda_max=args.lambda_max,
    )
    primal_params = (
        [p for p in model.parameters() if p.requires_grad]
        + list(task_head.parameters())
    )
    primal_optimizer = torch.optim.AdamW(primal_params, lr=args.lr)
    proxy = ProxyLagrangianOptimizer(primal_optimizer, constraints)

    history: list[dict] = []
    best_state: dict | None = None
    best_acc = -1.0
    best_marg_r2 = float("inf")
    bail_reason: str | None = None
    config = vars(args).copy()
    config["info"] = info
    _save_json(output_dir / "config.json", config)

    # Round 2 estimator stack (set 2026-05-03):
    #   - primal differentiable signal: linear unbiased nHSIC (no d/N pathology)
    #   - dual update signal:           Theil-adjusted held-out R² (E=0 under H₀)
    #   - post-projection (optional):   sliding-buffer online LEACE refit with
    #                                   shrinkage and constrain_cov_trace
    online_leace: OnlineLeaceRefit | None = None
    if args.online_leace:
        online_leace = OnlineLeaceRefit(
            d_x=768, d_z=1,
            buffer_size=args.online_leace_buffer,
            refit_every=args.online_leace_refit_every,
            device="cpu",  # closed-form fit; portable across CUDA/MPS/CPU
        )
        print(f"[round2] online LEACE refit ENABLED  "
              f"(buffer={args.online_leace_buffer}, "
              f"refit_every={args.online_leace_refit_every}, "
              f"shrinkage=True, constrain_cov_trace=True)")
    print(f"[round2] primal=nHSIC(linear, unbiased)  "
          f"dual=Theil-adjusted holdout R²  "
          f"τ={args.r2_threshold}")

    holdout_r2_cached: float | None = None      # raw R² (logging only)
    holdout_adj_r2_cached: float | None = None  # the dual signal
    global_step = 0
    controller_mode = "option_d"  # swaps to "plan_b" if monitor trips in epoch 1
    plan_b: EmaCrossCovPIController | None = None
    monitor = FiveSignalMonitor(threshold=args.r2_threshold)
    swap_log: dict | None = None
    t0 = time.time()
    last_status_milestone = 0  # 0 = none, 1 = 30min hit, 2 = 1hr hit
    print(f"[monitor] FiveSignalMonitor active "
          f"(flip_window=20, lambda_sat_trip=100, flip_rate_trip=0.4, "
          f"mid_window_jump_trip=0.20)")
    for epoch in range(args.epochs):
        model.train()
        task_head.train()
        running_task = running_r2 = 0.0
        running_holdout_r2 = 0.0
        running_holdout_refreshes = 0
        running_steps = 0
        ep_t0 = time.time()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            gen = batch["gender"].long().to(device)
            occ = batch["occupation"].long().to(device)

            z = model(ids, mask)
            logits = task_head(z)
            L_task = F.cross_entropy(logits, occ)

            # Round 2 primal differentiable signal: linear unbiased nHSIC₁
            # (Song et al. 2012). Has E[nHSIC] = 0 under independence at any
            # (n, d) — replaces in-batch OLS R², which saturated at 1.0 due
            # to d/N=24 noise floor at d=768, batch=32 (the failure mode
            # confirmed in Round 1 results/v2_bios_ROUND1/VERDICT.md).
            gen_oh = gender_one_hot(gen)
            primal_signal = nhsic_linear(z, gen_oh)

            # Plan-B mode: overwrite proxy lambda from EMA-cov PI controller
            # before computing the lagrangian loss. The proxy's lambda field
            # is just a scalar — we mutate it directly since dual_step is
            # bypassed in this mode.
            if controller_mode == "plan_b":
                assert plan_b is not None
                plan_b.observe(z, gen)
                lam_new, _ = plan_b.step()
                proxy.constraints["marginal_gender"].lambda_value = lam_new

            primal_loss = proxy.lagrangian_loss(
                L_task, {"marginal_gender": primal_signal}
            )

            primal_optimizer.zero_grad(set_to_none=True)
            primal_loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(primal_params, args.grad_clip)
            primal_optimizer.step()

            # Online LEACE refit (after primal step so the buffer captures the
            # post-step [CLS] distribution). Observe every step; refit on
            # cadence; register on the model.
            if online_leace is not None:
                with torch.no_grad():
                    z_post_step = model(ids, mask)
                online_leace.observe(z_post_step, gen)
                if online_leace.should_refit(global_step):
                    eraser = online_leace.refit(global_step)
                    Q = eraser.P.detach().to(torch.float32)
                    mu = (
                        eraser.bias.detach().to(torch.float32)
                        if eraser.bias is not None
                        else torch.zeros(Q.shape[0], dtype=torch.float32)
                    )
                    model.set_leace_projection(Q, mu)

            # Dual update path
            refreshed_this_step = False
            if controller_mode == "option_d":
                # Option D: Theil-adjusted held-out R² refreshed every K
                # primal steps. The dual ascent operates on adj_R² which
                # has E=0 under independence at any (n, d), so threshold
                # τ=0.05 is meaningful (raw R² has noise floor d/(n-1)≈0.19).
                if (
                    holdout_adj_r2_cached is None
                    or global_step % args.holdout_refresh_every == 0
                ):
                    out = _holdout_r2(model, holdout_loader, device)
                    holdout_r2_cached = out["r2"]
                    holdout_adj_r2_cached = out["adj_r2"]
                    running_holdout_refreshes += 1
                    refreshed_this_step = True
                proxy.dual_step({"marginal_gender": holdout_adj_r2_cached})

            cur_lambda = proxy.constraints["marginal_gender"].lambda_value
            monitor.on_primal_step(
                lambda_now=cur_lambda,
                lambda_max=args.lambda_max,
                inbatch_r2=float(primal_signal.detach().item()),
            )
            if refreshed_this_step:
                snap = monitor.on_dual_refresh(
                    lambda_now=cur_lambda,
                    holdout_r2=holdout_adj_r2_cached,
                )
                snap["holdout_raw_r2"] = holdout_r2_cached
                with (output_dir / "monitor.jsonl").open("a") as f:
                    f.write(json.dumps(
                        {"global_step": global_step, "epoch": epoch, **snap},
                        default=_json_default,
                    ) + "\n")

            # Log every 50 primal steps
            if global_step % 50 == 0:
                _h = (
                    holdout_r2_cached
                    if holdout_r2_cached is not None else float("nan")
                )
                print(f"[step {global_step:5d}] ep={epoch} mode={controller_mode}  "
                      f"λ={cur_lambda:.3f}  |Δλ|_ema={monitor.delta_lambda_ema:.4f}  "
                      f"holdout_R²={_h:.4f}  "
                      f"in-batch_R²_window_mean="
                      f"{monitor.last_inbatch_window_mean:.4f}  "
                      f"flip_rate_20={monitor.flip_rate():.2f}  "
                      f"sat_streak={monitor.lambda_saturation_streak}")

            # Plan-B trip check (epoch 1 only)
            if controller_mode == "option_d" and epoch == 0:
                trip = monitor.trip_check()
                if trip is not None:
                    print(f"!!! {trip}")
                    print(f"!!! SWAP: option_d → plan_b (EMA-cov PI controller). "
                          f"Resetting integral, λ_init={args.lambda_init}.")
                    plan_b = EmaCrossCovPIController(
                        d=768,
                        n_classes=2,
                        threshold=args.r2_threshold,
                        ema_decay=args.plan_b_ema_decay,
                        kp=args.plan_b_kp,
                        ki=args.plan_b_ki,
                        lambda_min=args.lambda_min,
                        lambda_init=args.lambda_init,
                        lambda_max=args.lambda_max,
                    )
                    proxy.constraints["marginal_gender"].lambda_value = max(
                        args.lambda_min, args.lambda_init,
                    )
                    controller_mode = "plan_b"
                    swap_log = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "trip_reason": trip,
                    }
                    with (output_dir / "monitor.jsonl").open("a") as f:
                        f.write(json.dumps(
                            {"event": "PLAN_B_SWAP", **swap_log},
                            default=_json_default,
                        ) + "\n")

            # Status milestone surfacing
            elapsed_now = time.time() - t0
            if last_status_milestone < 1 and elapsed_now >= 30 * 60:
                last_status_milestone = 1
                print(f"[STATUS @ 30min] step={global_step} ep={epoch} "
                      f"mode={controller_mode} λ={cur_lambda:.3f} "
                      f"holdout_R²={(holdout_r2_cached if holdout_r2_cached else 0):.4f} "
                      f"flip_rate={monitor.flip_rate():.2f} "
                      f"sat_streak={monitor.lambda_saturation_streak}")
            elif last_status_milestone < 2 and elapsed_now >= 60 * 60:
                last_status_milestone = 2
                print(f"[STATUS @ 1hr] step={global_step} ep={epoch} "
                      f"mode={controller_mode} λ={cur_lambda:.3f} "
                      f"holdout_R²={(holdout_r2_cached if holdout_r2_cached else 0):.4f} "
                      f"flip_rate={monitor.flip_rate():.2f} "
                      f"sat_streak={monitor.lambda_saturation_streak}")

            # Hard 30-min checkpoint: if no epoch has completed by 30 min,
            # the loop is too slow for the 12-epoch / 8h plan and we abort
            # before burning more compute. Set 2026-05-02 ahead of launch #3.
            if (
                bail_reason is None
                and elapsed_now >= 30 * 60
                and len(history) == 0
            ):
                bail_reason = (
                    f"BAIL @ 30 min hard-cap: epoch 0 still in progress at "
                    f"global_step={global_step} (n_primal={len(primal_train_ds)}, "
                    f"steps_per_epoch≈{len(primal_train_ds)//args.batch_size}). "
                    f"Loop is too slow for 12 epochs in 8h."
                )
                print(f"!!! {bail_reason}")
                break  # break inner step loop

            running_task += float(L_task.item())
            running_r2 += float(primal_signal.detach().item())
            running_holdout_r2 += (
                holdout_adj_r2_cached if holdout_adj_r2_cached is not None else 0.0
            )
            running_steps += 1
            global_step += 1

        # If the inner step loop bailed (e.g. hard 30-min cap with no epoch
        # completed), skip end-of-epoch eval + _check_bail so the inner-set
        # bail_reason is preserved for the summary.
        if bail_reason is not None:
            break

        ep_dt = time.time() - ep_t0
        avg_task = running_task / max(running_steps, 1)
        avg_r2 = running_r2 / max(running_steps, 1)
        avg_holdout_r2 = running_holdout_r2 / max(running_steps, 1)
        ev = _evaluate_dev(model, task_head, dev_loader, verifier, device)
        elapsed = time.time() - t0
        diag = proxy.diagnostics()["marginal_gender"]

        ep_log = {
            "epoch": epoch,
            "elapsed_s": elapsed,
            "epoch_time_s": ep_dt,
            "controller_mode": controller_mode,
            "train_task_loss": avg_task,
            "train_nhsic": avg_r2,
            "train_holdout_adj_r2_avg": avg_holdout_r2,
            "holdout_refreshes_this_epoch": running_holdout_refreshes,
            "dev_top10_acc": ev["top10_acc"],
            "dev_marginal_r2": ev["marginal_r2"],
            "dev_marginal_adj_r2": ev["marginal_adj_r2"],
            "dev_tpr_rms_gap": ev["tpr_rms_gap"],
            "dev_tpr_max_abs_gap": ev["tpr_max_abs_gap"],
            "dev_tpr_per_occupation": ev["tpr_per_occupation"],
            "lambda_marginal_gender": diag["lambda"],
            "delta_lambda_ema_end": monitor.delta_lambda_ema,
            "flip_rate_end": monitor.flip_rate(),
            "saturation_streak_end": monitor.lambda_saturation_streak,
            "online_leace_refit_count": (
                online_leace.refit_count if online_leace is not None else 0
            ),
        }
        history.append(ep_log)
        print(f"[ep {epoch:2d}] elapsed={elapsed/60:.1f}m  "
              f"mode={controller_mode}  "
              f"task_loss={avg_task:.4f}  train_nHSIC={avg_r2:.4f}  "
              f"holdout_adj_R²={avg_holdout_r2:.4f}  "
              f"dev_acc={ev['top10_acc']:.4f}  "
              f"dev_R²={ev['marginal_r2']:.4f}  "
              f"dev_adj_R²={ev['marginal_adj_r2']:.4f}  "
              f"TPR_rms={ev['tpr_rms_gap']:.4f}  "
              f"TPR_max={ev['tpr_max_abs_gap']:.4f}  "
              f"λ={diag['lambda']:.3f}")
        with (output_dir / "history.jsonl").open("a") as f:
            f.write(json.dumps(ep_log, default=_json_default) + "\n")

        # Surface every 5 epochs (status checkpoint visibility).
        if (epoch + 1) % 5 == 0:
            print(f"[STATUS @ ep{epoch}] elapsed={elapsed/60:.1f}m "
                  f"mode={controller_mode} λ={diag['lambda']:.3f} "
                  f"|Δλ|_ema={monitor.delta_lambda_ema:.4f} "
                  f"flip_rate={monitor.flip_rate():.2f} "
                  f"sat_streak={monitor.lambda_saturation_streak} "
                  f"dev_acc={ev['top10_acc']:.4f} "
                  f"dev_adj_R²={ev['marginal_adj_r2']:.4f} "
                  f"TPR_rms={ev['tpr_rms_gap']:.4f}")

        # Track best by (adj_R² ≤ τ AND highest acc).
        is_compliant = ev["marginal_adj_r2"] <= args.r2_threshold
        if is_compliant and ev["top10_acc"] > best_acc:
            best_acc = ev["top10_acc"]
            best_marg_r2 = ev["marginal_adj_r2"]
            best_state = {
                "model": {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()},
                "task_head": {k: v.detach().cpu().clone()
                              for k, v in task_head.state_dict().items()},
                "epoch": epoch,
            }

        bail_reason = _check_bail(
            elapsed, ev["top10_acc"], ev["marginal_adj_r2"], phase=1,
        )
        if bail_reason:
            print(f"!!! {bail_reason}")
            break

    # Save final + best.
    final_path = output_dir / "checkpoint_final.pt"
    torch.save({
        "model": model.state_dict(),
        "task_head": task_head.state_dict(),
        "epoch": history[-1]["epoch"] if history else -1,
    }, final_path)
    if best_state is not None:
        torch.save(best_state, output_dir / "checkpoint_best.pt")

    # CPU pre-flight gate (Round 2, set 2026-05-03). After 1 epoch on a small
    # subsample, decide whether the AWS launch is justified. Gate criteria:
    #   - dev_marginal_adj_r2 < args.gate_adj_r2_max (default 0.10)
    #   - dev_tpr_rms_gap     < args.gate_tpr_rms_max (default 0.15)
    # Both must pass. If either fails, the AWS launch is NOT justified;
    # we accept the §5.5 negative-result framing.
    gate_decision: dict | None = None
    if args.cpu_preflight:
        if not history:
            gate_decision = {
                "decision": "FAIL",
                "reason": "no completed epoch — pre-flight ran out of time or aborted",
            }
        else:
            last = history[-1]
            adj_r2 = float(last["dev_marginal_adj_r2"])
            tpr_rms = float(last["dev_tpr_rms_gap"])
            tpr_max = float(last["dev_tpr_max_abs_gap"])
            acc = float(last["dev_top10_acc"])
            adj_r2_pass = adj_r2 < args.gate_adj_r2_max
            tpr_pass = tpr_rms < args.gate_tpr_rms_max
            gate_decision = {
                "decision": "PASS" if (adj_r2_pass and tpr_pass) else "FAIL",
                "dev_top10_acc": acc,
                "dev_marginal_adj_r2": adj_r2,
                "dev_tpr_rms_gap": tpr_rms,
                "dev_tpr_max_abs_gap": tpr_max,
                "adj_r2_threshold": args.gate_adj_r2_max,
                "tpr_rms_threshold": args.gate_tpr_rms_max,
                "adj_r2_pass": adj_r2_pass,
                "tpr_pass": tpr_pass,
                "epochs_run": len(history),
            }
        print("\n" + "=" * 78)
        print("[CPU PRE-FLIGHT GATE]")
        for k, v in gate_decision.items():
            print(f"  {k}: {v}")
        print("=" * 78)

    summary = {
        "phase": 1,
        "epochs_run": len(history),
        "elapsed_min": (time.time() - t0) / 60.0,
        "bail_reason": bail_reason,
        "final_dev": history[-1] if history else None,
        "best_compliant": (
            None if best_state is None
            else {"epoch": best_state["epoch"],
                  "dev_top10_acc": best_acc,
                  "dev_marginal_adj_r2": best_marg_r2}
        ),
        "leace_warmstart": leace_diag,
        "construction_r2": constr,
        "param_breakdown": desc,
        "info": info,
        "controller_mode_final": controller_mode,
        "plan_b_swap": swap_log,
        "round2_estimator_stack": {
            "primal": "nhsic_linear (Song et al. 2012, unbiased, normalized)",
            "dual": "Theil-adjusted held-out R² (E=0 under H₀)",
            "online_leace": (
                None if online_leace is None
                else {
                    "buffer_size": online_leace.buffer_size,
                    "refit_every": online_leace.refit_every,
                    "shrinkage": online_leace.shrinkage,
                    "constrain_cov_trace": online_leace.constrain_cov_trace,
                    "refit_count": online_leace.refit_count,
                }
            ),
        },
        "cpu_preflight_gate": gate_decision,
    }
    _save_json(output_dir / "summary.json", summary)
    print(f"\nWrote summary to {output_dir / 'summary.json'}")
    return summary


# --------------------------------------------------------------------------
# Phase 2 training loop (gated; not run by default)
# --------------------------------------------------------------------------

def run_phase2(args, *, device: torch.device, output_dir: Path) -> dict:
    if not args.phase1_ok:
        raise RuntimeError(
            "Phase 2 requires --phase1-ok flag (explicit user approval after "
            "reviewing Phase 1 results). Refusing to launch."
        )
    phase1_ckpt = output_dir / "checkpoint_best.pt"
    if not phase1_ckpt.exists():
        phase1_ckpt = output_dir / "checkpoint_final.pt"
    if not phase1_ckpt.exists():
        raise RuntimeError(
            f"Phase 2 requires a Phase-1 checkpoint at "
            f"{output_dir / 'checkpoint_best.pt'} or "
            f"{output_dir / 'checkpoint_final.pt'}. None found."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)

    print("=" * 78)
    print(f"BIOS-medium PHASE 2 (10 per-occupation conditional + 1 marginal)")
    print(f"device={device}  seed={args.seed}  output_dir={output_dir}")
    print(f"loading Phase-1 ckpt: {phase1_ckpt}")
    print("=" * 78)

    train_loader, dev_loader, info, _, _ = build_bios_loaders(
        n_train=args.n_train,
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )
    model = BertWithLoRA(
        rank=args.rank, alpha=args.alpha, dropout=args.dropout,
    ).to(device)
    task_head = TaskHead(repr_dim=768, output_dim=10, hidden_dim=64).to(device)

    ckpt = torch.load(phase1_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    task_head.load_state_dict(ckpt["task_head"])
    verifier = VerificationRegularizer(regularization=1e-4).to(device)

    constraints = build_phase2_constraints(
        threshold=args.r2_threshold,
        eta_lambda=args.dual_lr,
        lambda_init=args.lambda_init,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
    )
    primal_params = (
        [p for p in model.parameters() if p.requires_grad]
        + list(task_head.parameters())
    )
    primal_optimizer = torch.optim.AdamW(primal_params, lr=args.lr)
    proxy = ProxyLagrangianOptimizer(primal_optimizer, constraints)

    history: list[dict] = []
    bail_reason: str | None = None
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        task_head.train()
        ep_t0 = time.time()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            gen = batch["gender"].long().to(device)
            occ = batch["occupation"].long().to(device)
            z = model(ids, mask)
            logits = task_head(z)
            L_task = F.cross_entropy(logits, occ)
            ev2 = evaluate_phase2_constraints(
                z, gen, occ, verifier,
                min_slice_size=args.phase2_min_slice_size,
            )
            primal_loss = proxy.lagrangian_loss(L_task, ev2.differentiable)
            primal_optimizer.zero_grad(set_to_none=True)
            primal_loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(primal_params, args.grad_clip)
            primal_optimizer.step()
            proxy.dual_step(ev2.scalars)

        ep_dt = time.time() - ep_t0
        ev = _evaluate_dev(model, task_head, dev_loader, verifier, device)
        elapsed = time.time() - t0

        # Per-constraint dev eval: per-occupation conditional R² on full dev.
        Z, G, OCC = ev["Z"], ev["G"], ev["OCC"]
        per_occ_r2: dict[str, float] = {}
        for k, occ_name in enumerate(BIOS_TOP10):
            mask_k = (OCC == k)
            if int(mask_k.sum().item()) < 50 or int(G[mask_k].unique().numel()) < 2:
                per_occ_r2[occ_name] = 0.0
                continue
            per_occ_r2[occ_name] = float(
                verifier(Z[mask_k].float(), G[mask_k]).item()
            )

        diags = proxy.diagnostics()
        ep_log = {
            "epoch": epoch,
            "elapsed_s": elapsed,
            "epoch_time_s": ep_dt,
            "dev_top10_acc": ev["top10_acc"],
            "dev_marginal_r2": ev["marginal_r2"],
            "dev_per_occ_r2": per_occ_r2,
            "lambdas": {n: d["lambda"] for n, d in diags.items()},
        }
        history.append(ep_log)
        print(f"[ep {epoch:2d}] elapsed={elapsed/60:.1f}m  "
              f"acc={ev['top10_acc']:.4f}  marg_R²={ev['marginal_r2']:.4f}  "
              f"per-occ R² range=[{min(per_occ_r2.values()):.4f}, "
              f"{max(per_occ_r2.values()):.4f}]")
        with (output_dir / "history_phase2.jsonl").open("a") as f:
            f.write(json.dumps(ep_log, default=_json_default) + "\n")

        marg_r2 = ev["marginal_r2"]
        n_below_010 = sum(1 for v in per_occ_r2.values() if v <= 0.10)
        n_below_008 = sum(1 for v in per_occ_r2.values() if v <= 0.08)
        if elapsed >= 60 * 60 and (marg_r2 > 0.20 or n_below_010 < 6):
            bail_reason = (
                f"BAIL @ 1 hr (Phase 2): marg_r2={marg_r2:.4f} > 0.20 "
                f"OR fewer than 6/10 below 0.10 ({n_below_010}/10)"
            )
            print(f"!!! {bail_reason}")
            break
        if elapsed >= 120 * 60 and n_below_008 < 7:
            bail_reason = (
                f"BAIL @ 2 hr (Phase 2): fewer than 7/10 below 0.08 "
                f"({n_below_008}/10) — abandoning Phase 2; falling back to Phase 1."
            )
            print(f"!!! {bail_reason}")
            break

    torch.save({
        "model": model.state_dict(),
        "task_head": task_head.state_dict(),
        "epoch": history[-1]["epoch"] if history else -1,
    }, output_dir / "checkpoint_phase2_final.pt")

    summary = {
        "phase": 2,
        "epochs_run": len(history),
        "elapsed_min": (time.time() - t0) / 60.0,
        "bail_reason": bail_reason,
        "final_dev": history[-1] if history else None,
        "info": info,
    }
    _save_json(output_dir / "summary_phase2.json", summary)
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BIOS-medium PCRL training driver")
    p.add_argument("--phase", type=int, choices=[1, 2], default=1)
    p.add_argument("--diagnostics-only", action="store_true",
                   help="Run pre-launch diagnostics and exit.")
    p.add_argument("--phase1-ok", action="store_true",
                   help="Required to launch Phase 2 — explicit acknowledgment "
                        "that Phase-1 results were reviewed.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--dual-lr", type=float, default=0.02)
    p.add_argument("--lambda-min", type=float, default=5.0)
    p.add_argument("--lambda-init", type=float, default=1.0)
    p.add_argument("--lambda-max", type=float, default=100.0)
    p.add_argument("--r2-threshold", type=float, default=0.05)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--n-train", type=int, default=50_000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--phase2-min-slice-size", type=int, default=16)
    # Option D: held-out R² for dual update (Phase 1 only). Defaults set
    # 2026-05-02 after the drift diagnostic showed the post-projection alone
    # cannot enforce the constraint under task-loss training.
    p.add_argument("--holdout-size", type=int, default=4096,
                   help="Held-out subset size for dual R² estimator. d/N=0.19 "
                        "at 4096 keeps the in-sample OLS bias below threshold.")
    p.add_argument("--holdout-refresh-every", type=int, default=10,
                   help="Refresh held-out R² every K primal mini-batches.")
    p.add_argument("--holdout-eval-batch-size", type=int, default=128,
                   help="Forward batch size during held-out R² eval.")
    # Plan B (EMA-cov + νPI) hyperparameters. Activated only on a monitor
    # trip in epoch 1 (lambda saturation, sign-flip rate >0.4, mid-window
    # jump >0.20). Pre-committed by the user 2026-05-02.
    p.add_argument("--plan-b-ema-decay", type=float, default=0.99,
                   help="EMA decay for second-moment buffers in Plan B.")
    p.add_argument("--plan-b-kp", type=float, default=50.0,
                   help="Proportional gain on (R²_ema - threshold) in Plan B.")
    p.add_argument("--plan-b-ki", type=float, default=5.0,
                   help="Integral gain on (R²_ema - threshold) in Plan B.")
    p.add_argument("--allow-warmstart-fail", action="store_true",
                   help="Smoke-test only: downgrade construction-R² abort to "
                        "warning. Required when --n-train is too small for "
                        "LEACE to generalize. PRODUCTION RUNS MUST NOT USE THIS.")
    # Round 2 stack (set 2026-05-03)
    p.add_argument("--online-leace", action="store_true",
                   help="Enable sliding-buffer online LEACE refit. Replaces "
                        "the static post-projection with one re-fit every K "
                        "primal steps on the most recent activations.")
    p.add_argument("--online-leace-buffer", type=int, default=512,
                   help="Sliding buffer size for online LEACE refit.")
    p.add_argument("--online-leace-refit-every", type=int, default=10,
                   help="Refit cadence (primal steps).")
    # CPU pre-flight gate (Round 2)
    p.add_argument("--cpu-preflight", action="store_true",
                   help="CPU pre-flight gate: print PASS/FAIL based on "
                        "dev_marginal_adj_r2 and dev_tpr_rms_gap after the "
                        "configured epochs. AWS launch only if PASS.")
    p.add_argument("--gate-adj-r2-max", type=float, default=0.10,
                   help="Pre-flight gate: max acceptable dev adj-R².")
    p.add_argument("--gate-tpr-rms-max", type=float, default=0.15,
                   help="Pre-flight gate: max acceptable RMS TPR-gap.")
    p.add_argument("--output-dir", type=str,
                   default="results/v2_bios_ROUND1")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = _pick_device(args.device)
    output_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() \
        else Path(args.output_dir)

    if args.diagnostics_only:
        run_diagnostics(args, device=device, output_dir=output_dir)
        return

    if args.phase == 1:
        run_phase1(args, device=device, output_dir=output_dir)
    elif args.phase == 2:
        run_phase2(args, device=device, output_dir=output_dir)


if __name__ == "__main__":
    main()
