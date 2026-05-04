"""Stage 3 mech-interp full run on BIOS BERT.

Identifies WHICH attention heads in BERT-base mediate gender re-derivation
when the input pronoun is flipped — the mechanistic complement to the
rank_k_sweep "rank-1 LEACE re-emerges at block-1" finding.

Pipeline:
  1. Cache layer-1 [CLS] reps for clean train (50K stratified BIOS top-10),
     clean dev (~31K), corrupt dev (~31K) — pronoun-swap counterfactual.
  2. Train sklearn LR probe on clean train reps + true gender labels.
  3. Compute baselines: vanilla / corrupt population R² (apples-to-apples
     with rank_k_sweep) + probe mean logits + probe accuracy.
  4. 12 layers × 12 heads patching sweep — per (L, h), patch
     blocks.L.attn.hook_z[..., h, :] from clean cache into corrupt run,
     compute three metrics:
         (a) Δ_logit     = mean(probe(patched)) - mean(probe(corrupt))
         (b) flip_rate   = (# patched-correct & corrupt-wrong) / (# corrupt-wrong)
         (c) normΔ       = Δ_logit / |mean_clean - mean_corrupt|
       plus population R² for apples-to-apples vs rank_k_sweep.
  5. Top-K causal ablation — pick top-K heads (by |Δ_logit|) globally;
     zero-ablate them simultaneously on corrupt run; measure pop R² at
     block-1 [CLS]. K ∈ {2, 3, 5}.
  6. Residual stream decomposition — for each (L, h), compute the OV
     circuit's CLS-position contribution z[L,h] @ W_O[L,h], project onto
       - g_LEACE: the LEACE rank-1 direction in d_model space
       - g_means: (mean(F_clean_layer1_CLS) - mean(M_clean_layer1_CLS))
     (both unit-normalized), record the per-head magnitudes.
  7. Spearman rank-correlation: Stage 2's sklearn-LR ranking (block 1)
     vs Stage 3's population-R² ranking (block 1) — robustness check.

Outputs to ${out_dir}/mech_interp_full_*.{json,md,pdf} + S3 push after each
stage.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def population_linear_r2(X: torch.Tensor, Z: torch.Tensor) -> float:
    X = X.to(torch.float64)
    Z = Z.to(torch.float64).view(-1, 1)
    Xc = X - X.mean(0)
    Zc = Z - Z.mean(0)
    n = X.shape[0]
    Sxx = (Xc.T @ Xc) / n
    Sxz = (Xc.T @ Zc) / n
    var_z = float(Zc.var(unbiased=False))
    Sxx_inv = torch.linalg.pinv(Sxx, rcond=1e-10)
    return float((Sxz.T @ Sxx_inv @ Sxz / max(var_z, 1e-12)).item())


def s3_put(local_path: Path, s3_uri: str) -> None:
    if not s3_uri:
        return
    rc = subprocess.run(["aws", "s3", "cp", str(local_path), s3_uri],
                        capture_output=True)
    print(f"[s3] {s3_uri} rc={rc.returncode}", flush=True)
    if rc.stderr:
        print(f"[s3-err] {rc.stderr.decode(errors='ignore')[:300]}", flush=True)


# ---------------------------------------------------------------------------
# Build pairs and cache layer-1 [CLS]
# ---------------------------------------------------------------------------

def build_counterfactual_pairs(
    texts, tokenizer, *, max_length: int = 128
):
    from pcrl.language.cda_invariance import swap_gender_pronouns
    clean_keep, corrupt_keep, mask = [], [], []
    for t in texts:
        s = swap_gender_pronouns(t)
        if s == t:
            mask.append(False)
            continue
        tk_a = tokenizer(t, max_length=max_length, truncation=True,
                         padding="max_length", return_tensors="pt")["input_ids"]
        tk_b = tokenizer(s, max_length=max_length, truncation=True,
                         padding="max_length", return_tensors="pt")["input_ids"]
        a_len = int((tk_a[0] != tokenizer.pad_token_id).sum())
        b_len = int((tk_b[0] != tokenizer.pad_token_id).sum())
        if a_len != b_len:
            mask.append(False)
            continue
        clean_keep.append(t)
        corrupt_keep.append(s)
        mask.append(True)
    return clean_keep, corrupt_keep, np.array(mask, dtype=bool)


@torch.no_grad()
def cache_layer1_cls(
    model, tokenizer, texts, *, max_length=128, batch_size=64, device="cuda",
):
    """Cache blocks.1.hook_resid_post[:, 0, :] (= layer-1 [CLS])."""
    n = len(texts)
    out = torch.empty((n, model.cfg.d_model), dtype=torch.float32)
    for i in range(0, n, batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, max_length=max_length, truncation=True,
                        padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].to(device)
        _, cache = model.run_with_cache(ids, names_filter="blocks.1.hook_resid_post")
        out[i:i + batch_size] = cache["blocks.1.hook_resid_post"][:, 0, :].float().cpu()
        if i // batch_size % 50 == 0:
            print(f"  [cache] {i}/{n}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Patching: per (L, h), forward corrupt with clean's z[L, h] swapped in,
# return layer-1 [CLS] reps
# ---------------------------------------------------------------------------

@torch.no_grad()
def sweep_patching(
    model, tokenizer, clean_texts, corrupt_texts, *,
    max_length=128, batch_size=64, device="cuda",
):
    """Returns dict: layer_l, head_h -> patched_layer1_cls (n, d_model).

    Strategy: for each layer L, cache clean's blocks.L.attn.hook_z PER BATCH,
    then for each head h, re-run corrupt with patch hook. blocks.1.hook_resid_post
    is captured.
    """
    assert len(clean_texts) == len(corrupt_texts)
    n = len(clean_texts)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model

    patched_cls = torch.empty((n_layers, n_heads, n, d_model), dtype=torch.float32)

    for L in range(n_layers):
        t_L = time.time()
        for i in range(0, n, batch_size):
            clean_chunk = clean_texts[i:i + batch_size]
            corrupt_chunk = corrupt_texts[i:i + batch_size]
            clean_ids = tokenizer(clean_chunk, max_length=max_length, truncation=True,
                                  padding="max_length", return_tensors="pt")["input_ids"].to(device)
            corrupt_ids = tokenizer(corrupt_chunk, max_length=max_length, truncation=True,
                                    padding="max_length", return_tensors="pt")["input_ids"].to(device)
            # Cache clean's hook_z[L]
            _, clean_cache = model.run_with_cache(
                clean_ids, names_filter=f"blocks.{L}.attn.hook_z",
            )
            clean_z = clean_cache[f"blocks.{L}.attn.hook_z"]  # (B, T, H, D_head)

            for h in range(n_heads):
                captured = {}

                def patch_z(z, hook, _ref=clean_z, _h=h):
                    z[..., _h, :] = _ref[..., _h, :]
                    return z

                def cap_resid(act, hook, _store=captured):
                    _store["v"] = act.detach().float().cpu()

                model.run_with_hooks(
                    corrupt_ids,
                    fwd_hooks=[
                        (f"blocks.{L}.attn.hook_z", patch_z),
                        ("blocks.1.hook_resid_post", cap_resid),
                    ],
                )
                patched_cls[L, h, i:i + batch_size] = captured["v"][:, 0, :]
            del clean_z, clean_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[sweep] layer {L} done in {time.time()-t_L:.1f}s", flush=True)
    return patched_cls


# ---------------------------------------------------------------------------
# Top-K causal ablation: zero-ablate top-K heads simultaneously on corrupt
# ---------------------------------------------------------------------------

@torch.no_grad()
def ablate_topk(
    model, tokenizer, corrupt_texts, top_heads, *,
    max_length=128, batch_size=64, device="cuda",
):
    """Run model on corrupt texts with the listed (layer, head) pairs zero-ablated
    in z. Return layer-1 [CLS] reps."""
    n = len(corrupt_texts)
    out = torch.empty((n, model.cfg.d_model), dtype=torch.float32)
    by_layer: dict[int, list[int]] = {}
    for L, h in top_heads:
        by_layer.setdefault(L, []).append(h)

    for i in range(0, n, batch_size):
        chunk = corrupt_texts[i:i + batch_size]
        ids = tokenizer(chunk, max_length=max_length, truncation=True,
                        padding="max_length", return_tensors="pt")["input_ids"].to(device)
        captured = {}

        def cap_resid(act, hook, _s=captured):
            _s["v"] = act.detach().float().cpu()

        hooks = [("blocks.1.hook_resid_post", cap_resid)]
        for L, heads in by_layer.items():
            def make_zero_hook(h_list):
                def fn(z, hook, _h=h_list):
                    for hh in _h:
                        z[..., hh, :] = 0.0
                    return z
                return fn
            hooks.append((f"blocks.{L}.attn.hook_z", make_zero_hook(heads)))
        model.run_with_hooks(ids, fwd_hooks=hooks)
        out[i:i + batch_size] = captured["v"][:, 0, :]
    return out


# ---------------------------------------------------------------------------
# Residual decomposition: per-head OV-circuit contribution at CLS, projected
# onto LEACE direction and class-mean gender direction
# ---------------------------------------------------------------------------

@torch.no_grad()
def per_head_ov_at_cls(
    model, tokenizer, clean_texts, *,
    max_length=128, batch_size=64, device="cuda",
):
    """For each (L, h), return mean_over_examples( z[L,h]_CLS @ W_O[L,h] )
    as a tensor of shape (n_layers, n_heads, d_model)."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    n = len(clean_texts)
    accum = torch.zeros((n_layers, n_heads, d_model), dtype=torch.float64)

    for i in range(0, n, batch_size):
        chunk = clean_texts[i:i + batch_size]
        ids = tokenizer(chunk, max_length=max_length, truncation=True,
                        padding="max_length", return_tensors="pt")["input_ids"].to(device)
        names = [f"blocks.{L}.attn.hook_z" for L in range(n_layers)]
        names_filter = lambda nm, _names=set(names): nm in _names
        _, cache = model.run_with_cache(ids, names_filter=names_filter)
        for L in range(n_layers):
            z = cache[f"blocks.{L}.attn.hook_z"]  # (B, T, H, D_head)
            W_O = model.blocks[L].attn.W_O  # (H, D_head, D_model)
            cls_z = z[:, 0, :, :]  # (B, H, D_head)
            # Per-head OV at CLS: einsum
            ov_cls = torch.einsum("bhd,hde->bhe", cls_z, W_O)  # (B, H, D_model)
            accum[L] += ov_cls.sum(0).double().cpu()
    accum /= float(n)
    return accum  # (n_layers, n_heads, d_model)


def project_onto_unit(v: torch.Tensor, u: torch.Tensor) -> float:
    """Return |<v, u>| / ||u|| — magnitude of v's component along u."""
    u_norm = float(torch.linalg.norm(u))
    if u_norm < 1e-12:
        return float("nan")
    u_hat = u / u_norm
    return float(torch.abs(v @ u_hat))


# ---------------------------------------------------------------------------
# LEACE rank-1 direction in d_model space
# ---------------------------------------------------------------------------

def leace_direction(X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Returns the unit vector u such that LEACE projects out the direction u.
    Computed via the kernel of (I - P) where P = LeaceEraser.fit(X, Z_oh).P."""
    from concept_erasure import LeaceEraser
    Z_oh = torch.nn.functional.one_hot(Z.long(), num_classes=2).float()
    eraser = LeaceEraser.fit(X.float(), Z_oh)
    P = eraser.P.detach()
    I_minus_P = torch.eye(P.shape[0], dtype=P.dtype) - P
    # I - P is rank-1 for binary Z; eigenvector with largest eigenvalue is u.
    # Symmetrize then eigendecompose.
    M = 0.5 * (I_minus_P + I_minus_P.T)
    w, V = torch.linalg.eigh(M)
    u = V[:, -1]  # largest eigenvalue
    return u / torch.linalg.norm(u)


# ---------------------------------------------------------------------------
# Headline writer
# ---------------------------------------------------------------------------

def write_headline(path: Path, payload: dict) -> None:
    s = payload["stage3"]
    pop = s.get("baseline_population_R2", {})
    probe = s.get("baseline_probe", {})
    spread = abs(probe.get("mean_logit_clean", 0) - probe.get("mean_logit_corrupt", 0))

    # Top-3 globally by |delta_logit|
    deltas = []
    for L in range(12):
        for h in range(12):
            cell = s["per_head"].get(f"L{L}_h{h}", {})
            d = cell.get("delta_logit", float("nan"))
            if np.isfinite(d):
                deltas.append(((L, h), d))
    deltas.sort(key=lambda kv: abs(kv[1]), reverse=True)
    top10 = deltas[:10]

    # Ablation
    abl = s.get("ablation", {})

    # Spearman block-1
    sp = s.get("spearman_block1_vs_stage2", {})

    text = (
        "MECH INTERP RESULTS — BIOS BERT layer-1 LEACE re-emergence\n"
        "==========================================================\n"
        f"Wall time: {s.get('wall_time_seconds', 0)/60:.1f} min | "
        f"Cost: ~${s.get('aws_cost_estimate_usd', 0):.2f} | "
        f"Instance: {s.get('aws_instance_id', '?')}\n"
        f"Layer convention: blocks.{1}.hook_resid_post = HF hidden_states[2] = "
        f"block-1 output (matches rank_k_sweep)\n\n"
        f"Probe: {s.get('probe_type', '?')}\n"
        f"Probe train n: {s.get('probe_train_n', '?')} | "
        f"Eval n: {s.get('probe_eval_n', '?')}\n"
        f"Pairs kept: {s.get('n_pairs_kept', '?')}/{s.get('n_pairs_attempted', '?')}\n\n"
        f"Vanilla layer-1 R² (population OLS): {pop.get('vanilla', float('nan')):.3f}  "
        f"(rank_k_sweep baseline 0.821)\n"
        f"Corrupt layer-1 R²: {pop.get('corrupt', float('nan')):.3f}\n"
        f"Probe acc — clean: {probe.get('eval_clean_acc', float('nan')):.3f} | "
        f"corrupt: {probe.get('eval_corrupt_acc', float('nan')):.3f}\n"
        f"Probe logit spread (clean-vs-corrupt): {spread:.4f}\n\n"
        "Top-10 heads by |Δ_logit| (across all 12×12 cells):\n"
        "(L, h)  |  Δ_logit  |  flip_rate  |  normΔ  |  R²_patched\n"
        "--------+-----------+-------------+---------+-------------\n"
    )
    for (L, h), d in top10:
        cell = s["per_head"][f"L{L}_h{h}"]
        text += (
            f"({L:>2},{h:>2}) | {d:+.4f}  |  {cell.get('flip_rate', float('nan')):.3f}     |  "
            f"{cell.get('norm_delta', float('nan')):+.3f}  |  {cell.get('population_R2', float('nan')):.3f}\n"
        )

    text += "\nCausal ablation (zero-ablate top-K heads on corrupt run):\n"
    text += "K   | heads ablated                 | layer-1 R² | Δ vs corrupt\n"
    text += "----+-------------------------------+------------+--------------\n"
    for K in (2, 3, 5):
        a = abl.get(f"top_{K}", {})
        heads_str = str(a.get("heads", []))[:30]
        text += (
            f" {K:>2} | {heads_str:<30}| "
            f"{a.get('population_R2', float('nan')):.3f}     | "
            f"{a.get('delta_vs_corrupt', float('nan')):+.3f}\n"
        )

    if sp:
        text += (
            f"\nSpearman block-1 ranking (Stage 2 LR-Δ vs Stage 3 R²-Δ): "
            f"{sp.get('rho', float('nan')):.3f}  "
            f"(p={sp.get('pvalue', float('nan')):.3g})\n"
        )

    # Branch decision
    abl3 = abl.get("top_3", {})
    abl3_r2 = abl3.get("population_R2", float("nan"))
    if np.isfinite(abl3_r2):
        if abl3_r2 < 0.5:
            branch = "SHARP CIRCUIT (top-3 ablation drops layer-1 R² below 0.5)"
        elif abl3_r2 < 0.7:
            branch = "MIXED (top-3 ablation partially attenuates)"
        else:
            branch = "DIFFUSE MEDIATION (top-3 ablation barely moves R²)"
    else:
        branch = "UNKNOWN"

    text += f"\nBRANCH: {branch}\n"

    sanity = []
    sanity.append(("Vanilla layer-1 R² near 0.821?",
                   "Y" if abs(pop.get("vanilla", 0) - 0.821) < 0.10 else "N"))
    sanity.append(("Existing same-layer LEACE-1 R² ~0.06 (sanity)?",
                   "Y (not re-tested in stage 3 — see qleace_results.json)"))
    sanity.append(("At least 1 head |Δ_logit| > 0.04?",
                   "Y" if (deltas and abs(deltas[0][1]) > 0.04) else "N"))
    text += "\nSanity:\n"
    for q, a in sanity:
        text += f"  - {q} [{a}]\n"

    path.write_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=50_000)
    ap.add_argument("--n-eval", type=int, default=31764)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/v2_bios_FINAL")
    ap.add_argument("--s3-uri", default="")
    ap.add_argument("--instance-id", default="")
    ap.add_argument("--cost-per-hour", type=float, default=0.526)
    ap.add_argument("--stage2-results-uri", default="",
                    help="s3 URI of Stage 2 results JSON, for Spearman comparison")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "mech_interp_full_results.json"
    headline_path = out_dir / "mech_interp_full_HEADLINE.txt"
    done_path = out_dir / "mech_interp_STAGE3_DONE"
    s3_prefix = args.s3_uri.rstrip("/")

    device = args.device if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    payload = {
        "stage3": {
            "implementation_source": "transformer_lens HookedEncoder bert-base-uncased",
            "aws_instance_id": args.instance_id,
            "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "probe_type": "sklearn_LogisticRegression(class_weight=balanced, max_iter=2000)",
            "probe_train_split": f"BIOS_TOP10_50K_stratified_seed={args.seed} (matches rank_k_sweep)",
            "layer_convention": "blocks.L.hook_resid_post = HF hidden_states[L+1]; probe at blocks.1.hook_resid_post",
        }
    }
    t_start = time.time()

    # ------------------------------------------------------------------
    # Stage A: load model + BIOS
    # ------------------------------------------------------------------
    print("[s3] loading HookedEncoder bert-base-uncased...", flush=True)
    from transformer_lens import HookedEncoder
    model = HookedEncoder.from_pretrained("bert-base-uncased")
    model = model.to(device).eval()
    tokenizer = model.tokenizer
    print(f"[s3] model on {device}; n_layers={model.cfg.n_layers} "
          f"n_heads={model.cfg.n_heads} d_model={model.cfg.d_model}", flush=True)

    print("[s3] loading BIOS top-10 train + dev...", flush=True)
    from datasets import load_dataset
    from pcrl.language.bios_dataset import (
        BIOS_TOP10_IDS, _LABEL_TO_LOCAL, _filter_top10, _stratified_subsample,
    )
    full_train = _filter_top10(load_dataset("LabHC/bias_in_bios", split="train"))
    occ_int_tr = np.array(full_train["profession"])
    gen_int_tr = np.array(full_train["gender"])
    occ_local_tr = np.array([_LABEL_TO_LOCAL[p] for p in occ_int_tr])
    keep_idx = _stratified_subsample(occ_local_tr, gen_int_tr, args.n_train, args.seed)
    train_sub = full_train.select(keep_idx.tolist())

    full_dev = _filter_top10(load_dataset("LabHC/bias_in_bios", split="dev"))
    print(f"[s3] train n={len(train_sub)} dev n={len(full_dev)}", flush=True)

    # ------------------------------------------------------------------
    # Stage B: pairs
    # ------------------------------------------------------------------
    print("[s3] building counterfactual pairs (dev)...", flush=True)
    dev_texts = list(full_dev["hard_text"])
    dev_genders = np.array(full_dev["gender"], dtype=np.int64)
    clean_dev, corrupt_dev, mask_dev = build_counterfactual_pairs(
        dev_texts, tokenizer, max_length=args.max_length,
    )
    g_dev = dev_genders[mask_dev]
    n_kept = len(clean_dev)
    n_attempted = len(dev_texts)
    print(f"[s3] kept {n_kept}/{n_attempted} same-length dev pairs", flush=True)
    payload["stage3"]["n_pairs_attempted"] = int(n_attempted)
    payload["stage3"]["n_pairs_kept"] = int(n_kept)

    if args.n_eval < n_kept:
        rng = np.random.default_rng(args.seed)
        sub_idx = rng.permutation(n_kept)[:args.n_eval]
        clean_dev = [clean_dev[i] for i in sub_idx]
        corrupt_dev = [corrupt_dev[i] for i in sub_idx]
        g_dev = g_dev[sub_idx]
        n_kept = len(clean_dev)
    payload["stage3"]["probe_eval_n"] = int(n_kept)
    print(f"[s3] using {n_kept} dev pairs for sweep", flush=True)

    train_texts = list(train_sub["hard_text"])
    g_train = np.array(train_sub["gender"], dtype=np.int64)
    payload["stage3"]["probe_train_n"] = int(len(train_texts))

    # ------------------------------------------------------------------
    # Stage C: cache layer-1 [CLS] reps
    # ------------------------------------------------------------------
    print("[s3] caching layer-1 [CLS] reps for clean train + clean dev + corrupt dev...", flush=True)
    t0 = time.time()
    train_clean_cls = cache_layer1_cls(model, tokenizer, train_texts,
                                       max_length=args.max_length,
                                       batch_size=args.batch_size, device=device)
    print(f"[s3] train cached in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    eval_clean_cls = cache_layer1_cls(model, tokenizer, clean_dev,
                                      max_length=args.max_length,
                                      batch_size=args.batch_size, device=device)
    print(f"[s3] dev clean cached in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    eval_corrupt_cls = cache_layer1_cls(model, tokenizer, corrupt_dev,
                                        max_length=args.max_length,
                                        batch_size=args.batch_size, device=device)
    print(f"[s3] dev corrupt cached in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Stage D: train probe + baselines
    # ------------------------------------------------------------------
    print("[s3] training sklearn LR probe on 50K clean train reps...", flush=True)
    from sklearn.linear_model import LogisticRegression
    probe = LogisticRegression(class_weight="balanced", max_iter=2000)
    probe.fit(train_clean_cls.numpy(), g_train)

    eval_clean_acc = float(probe.score(eval_clean_cls.numpy(), g_dev))
    eval_corrupt_acc = float(probe.score(eval_corrupt_cls.numpy(), g_dev))
    eval_clean_logits = probe.decision_function(eval_clean_cls.numpy())
    eval_corrupt_logits = probe.decision_function(eval_corrupt_cls.numpy())
    eval_clean_pred = probe.predict(eval_clean_cls.numpy())
    eval_corrupt_pred = probe.predict(eval_corrupt_cls.numpy())
    mean_clean = float(eval_clean_logits.mean())
    mean_corrupt = float(eval_corrupt_logits.mean())
    spread = max(abs(mean_clean - mean_corrupt), 1e-9)
    payload["stage3"]["baseline_probe"] = {
        "eval_clean_acc": eval_clean_acc,
        "eval_corrupt_acc": eval_corrupt_acc,
        "mean_logit_clean": mean_clean,
        "mean_logit_corrupt": mean_corrupt,
        "spread_abs": spread,
    }
    print(f"[s3] probe — clean acc: {eval_clean_acc:.3f} corrupt acc: {eval_corrupt_acc:.3f}", flush=True)
    print(f"[s3] mean_logit clean: {mean_clean:+.4f} corrupt: {mean_corrupt:+.4f} spread: {spread:.4f}", flush=True)

    pop_r2_vanilla = population_linear_r2(eval_clean_cls, torch.tensor(g_dev))
    pop_r2_corrupt = population_linear_r2(eval_corrupt_cls, torch.tensor(g_dev))
    payload["stage3"]["baseline_population_R2"] = {
        "vanilla": pop_r2_vanilla,
        "corrupt": pop_r2_corrupt,
    }
    print(f"[s3] population R² — vanilla: {pop_r2_vanilla:.3f} corrupt: {pop_r2_corrupt:.3f}", flush=True)

    # corrupt-wrong mask for flip rate
    g_dev_t = torch.tensor(g_dev)
    corrupt_wrong = eval_corrupt_pred != g_dev
    n_corrupt_wrong = int(corrupt_wrong.sum())
    payload["stage3"]["baseline_probe"]["n_corrupt_wrong"] = n_corrupt_wrong

    json_path.write_text(json.dumps(payload, indent=2, default=float))
    if s3_prefix:
        s3_put(json_path, f"{s3_prefix}/mech_interp_full_results.json")

    # ------------------------------------------------------------------
    # Stage E: 12×12 patching sweep
    # ------------------------------------------------------------------
    print(f"[s3] starting 12×12 patching sweep on n={n_kept} dev pairs...", flush=True)
    sweep_t0 = time.time()
    patched = sweep_patching(
        model, tokenizer, clean_dev, corrupt_dev,
        max_length=args.max_length, batch_size=args.batch_size, device=device,
    )  # (12, 12, n_kept, 768)
    print(f"[s3] sweep done in {(time.time()-sweep_t0)/60:.1f} min", flush=True)

    # Compute per-cell metrics
    payload["stage3"]["per_head"] = {}
    for L in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            patched_cls = patched[L, h]  # (n_kept, 768)
            patched_logits = probe.decision_function(patched_cls.numpy())
            patched_pred = probe.predict(patched_cls.numpy())
            mean_patched = float(patched_logits.mean())
            delta = mean_patched - mean_corrupt
            norm_delta = delta / spread
            patched_correct = patched_pred == g_dev
            recovered = patched_correct & corrupt_wrong
            flip_rate = (float(recovered.sum()) / float(n_corrupt_wrong)
                         if n_corrupt_wrong > 0 else float("nan"))
            pop_r2 = population_linear_r2(patched_cls, g_dev_t)
            payload["stage3"]["per_head"][f"L{L}_h{h}"] = {
                "delta_logit": delta,
                "norm_delta": norm_delta,
                "flip_rate": flip_rate,
                "mean_logit_patched": mean_patched,
                "population_R2": pop_r2,
            }
        # Save progress per layer
        payload["stage3"]["wall_time_seconds"] = float(time.time() - t_start)
        payload["stage3"]["aws_cost_estimate_usd"] = (
            (time.time() - t_start) / 3600.0 * args.cost_per_hour
        )
        json_path.write_text(json.dumps(payload, indent=2, default=float))
        write_headline(headline_path, payload)
        if s3_prefix:
            s3_put(json_path, f"{s3_prefix}/mech_interp_full_results.json")
            s3_put(headline_path, f"{s3_prefix}/mech_interp_full_HEADLINE.txt")
        print(f"[s3] layer {L} cells written", flush=True)

    # ------------------------------------------------------------------
    # Stage F: top-K causal ablation
    # ------------------------------------------------------------------
    print("[s3] computing top-K causal ablation...", flush=True)
    cells = [(L, h, payload["stage3"]["per_head"][f"L{L}_h{h}"]["delta_logit"])
             for L in range(12) for h in range(12)]
    cells.sort(key=lambda x: abs(x[2]), reverse=True)
    payload["stage3"]["ablation"] = {}
    for K in (2, 3, 5):
        top = [(L, h) for L, h, _ in cells[:K]]
        try:
            t_a = time.time()
            ablated_cls = ablate_topk(
                model, tokenizer, corrupt_dev, top,
                max_length=args.max_length, batch_size=args.batch_size, device=device,
            )
            r2 = population_linear_r2(ablated_cls, g_dev_t)
            ablated_logits = probe.decision_function(ablated_cls.numpy())
            mean_ablated = float(ablated_logits.mean())
            ablated_pred = probe.predict(ablated_cls.numpy())
            ablated_acc = float((ablated_pred == g_dev).mean())
            payload["stage3"]["ablation"][f"top_{K}"] = {
                "heads": top,
                "population_R2": r2,
                "delta_vs_corrupt": r2 - pop_r2_corrupt,
                "delta_vs_vanilla": r2 - pop_r2_vanilla,
                "mean_logit_ablated": mean_ablated,
                "probe_acc_ablated": ablated_acc,
                "wall_seconds": float(time.time() - t_a),
            }
            print(f"[s3] top-{K} ablate {top}: R²={r2:.3f} acc={ablated_acc:.3f} "
                  f"({time.time()-t_a:.1f}s)", flush=True)
        except Exception as e:
            print(f"[s3] ablation top-{K} ERROR: {e}\n{traceback.format_exc()}", flush=True)
            payload["stage3"]["ablation"][f"top_{K}"] = {"error": str(e), "heads": top}
        payload["stage3"]["wall_time_seconds"] = float(time.time() - t_start)
        payload["stage3"]["aws_cost_estimate_usd"] = (
            (time.time() - t_start) / 3600.0 * args.cost_per_hour
        )
        json_path.write_text(json.dumps(payload, indent=2, default=float))
        write_headline(headline_path, payload)
        if s3_prefix:
            s3_put(json_path, f"{s3_prefix}/mech_interp_full_results.json")
            s3_put(headline_path, f"{s3_prefix}/mech_interp_full_HEADLINE.txt")

    # ------------------------------------------------------------------
    # Stage G: residual-stream decomposition
    # ------------------------------------------------------------------
    print("[s3] computing per-head OV-circuit CLS contribution...", flush=True)
    try:
        # Use a subsample for OV cache (mean over 1000 examples is plenty)
        ov_n = min(1000, len(clean_dev))
        ov_means = per_head_ov_at_cls(
            model, tokenizer, clean_dev[:ov_n],
            max_length=args.max_length, batch_size=args.batch_size, device=device,
        )  # (n_layers, n_heads, d_model) tensor

        # Direction 1: LEACE rank-1 direction in d_model space
        u_leace = leace_direction(train_clean_cls, torch.tensor(g_train))
        # Direction 2: class-mean gender direction
        train_F = train_clean_cls[g_train == 1].mean(0)
        train_M = train_clean_cls[g_train == 0].mean(0)
        u_means = train_F - train_M

        decomp = {}
        for L in range(model.cfg.n_layers):
            for h in range(model.cfg.n_heads):
                v = ov_means[L, h].float()  # (d_model,)
                decomp[f"L{L}_h{h}"] = {
                    "ov_norm": float(torch.linalg.norm(v)),
                    "proj_along_LEACE_dir": project_onto_unit(v, u_leace),
                    "proj_along_classmean_dir": project_onto_unit(v, u_means),
                }
        payload["stage3"]["residual_decomposition"] = decomp
        payload["stage3"]["residual_decomposition_meta"] = {
            "n_examples_for_ov_mean": ov_n,
            "leace_direction_norm": float(torch.linalg.norm(u_leace)),
            "classmean_direction_norm": float(torch.linalg.norm(u_means)),
        }
        print("[s3] residual decomposition done", flush=True)
    except Exception as e:
        print(f"[s3] residual decomposition ERROR: {e}\n{traceback.format_exc()}", flush=True)
        payload["stage3"]["residual_decomposition_error"] = str(e)

    # ------------------------------------------------------------------
    # Stage H: Spearman vs Stage 2
    # ------------------------------------------------------------------
    if args.stage2_results_uri:
        try:
            print(f"[s3] downloading Stage 2 results from {args.stage2_results_uri}...", flush=True)
            stage2_local = ROOT / "tmp_stage2_results.json"
            subprocess.run(["aws", "s3", "cp", args.stage2_results_uri, str(stage2_local)],
                           check=True)
            with open(stage2_local) as f:
                stage2 = json.load(f)
            stage2_deltas = []
            stage3_deltas_block1 = []
            for h in range(12):
                key = str(h)
                if key in stage2["stage2"]["per_head"]:
                    stage2_deltas.append(stage2["stage2"]["per_head"][key]["delta_probe_logit"])
                    stage3_deltas_block1.append(payload["stage3"]["per_head"][f"L1_h{h}"]["delta_logit"])
            from scipy.stats import spearmanr
            rho, p = spearmanr(stage2_deltas, stage3_deltas_block1)
            payload["stage3"]["spearman_block1_vs_stage2"] = {
                "rho": float(rho),
                "pvalue": float(p),
                "n_heads": len(stage2_deltas),
                "stage2_deltas": stage2_deltas,
                "stage3_block1_deltas": stage3_deltas_block1,
            }
            print(f"[s3] Spearman block-1 (Stage 2 LR vs Stage 3 R²): rho={rho:.3f} p={p:.3g}", flush=True)
        except Exception as e:
            print(f"[s3] Spearman ERROR: {e}\n{traceback.format_exc()}", flush=True)
            payload["stage3"]["spearman_error"] = str(e)

    # ------------------------------------------------------------------
    # Stage I: figures
    # ------------------------------------------------------------------
    print("[s3] generating figures...", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. heatmap
        H = np.zeros((12, 12))
        for L in range(12):
            for h in range(12):
                H[L, h] = payload["stage3"]["per_head"][f"L{L}_h{h}"]["delta_logit"]
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(H, cmap="RdBu_r", aspect="auto",
                       vmin=-np.max(np.abs(H)), vmax=np.max(np.abs(H)))
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer (block index)")
        ax.set_title("BIOS BERT — per-head Δ_logit on gender (clean→corrupt patching)")
        ax.set_xticks(range(12))
        ax.set_yticks(range(12))
        plt.colorbar(im, ax=ax, label="Δ probe logit")
        # Annotate top 3
        flat = [(L, h, abs(H[L, h])) for L in range(12) for h in range(12)]
        flat.sort(key=lambda x: x[2], reverse=True)
        for L, h, _ in flat[:3]:
            ax.add_patch(plt.Rectangle((h - 0.5, L - 0.5), 1, 1,
                                       fill=False, edgecolor="black", lw=2))
        fig.tight_layout()
        fig_path = out_dir / "mech_interp_heatmap.pdf"
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"[s3] saved {fig_path.name}", flush=True)
        if s3_prefix:
            s3_put(fig_path, f"{s3_prefix}/mech_interp_heatmap.pdf")

        # 2. ablation bar
        labels = ["vanilla", "corrupt"]
        values = [pop_r2_vanilla, pop_r2_corrupt]
        for K in (2, 3, 5):
            a = payload["stage3"]["ablation"].get(f"top_{K}", {})
            r = a.get("population_R2")
            if r is not None:
                labels.append(f"top-{K} ablate")
                values.append(r)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values, color=["steelblue"] * 2 + ["firebrick"] * (len(labels) - 2))
        ax.set_ylabel("layer-1 population R²(gender)")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
        ax.set_title("Top-K head ablation effect on layer-1 R²(gender)")
        ax.legend()
        fig.tight_layout()
        fig_path = out_dir / "mech_interp_ablation_bar.pdf"
        fig.savefig(fig_path)
        plt.close(fig)
        if s3_prefix:
            s3_put(fig_path, f"{s3_prefix}/mech_interp_ablation_bar.pdf")

        # 3. residual decomposition scatter
        if "residual_decomposition" in payload["stage3"]:
            xs, ys, labels_s = [], [], []
            for L in range(12):
                for h in range(12):
                    d = payload["stage3"]["residual_decomposition"][f"L{L}_h{h}"]
                    xs.append(d["proj_along_LEACE_dir"])
                    ys.append(d["proj_along_classmean_dir"])
                    labels_s.append(f"L{L}h{h}")
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.scatter(xs, ys, alpha=0.5)
            for L, h, _ in flat[:5]:
                idx = L * 12 + h
                ax.scatter([xs[idx]], [ys[idx]], color="red", s=80, zorder=5)
                ax.annotate(f"L{L}h{h}", (xs[idx], ys[idx]),
                            xytext=(5, 5), textcoords="offset points")
            ax.set_xlabel("proj. of OV-CLS along LEACE direction")
            ax.set_ylabel("proj. of OV-CLS along class-mean direction")
            ax.set_title("Per-head OV-circuit gender projection (top-5 patching heads in red)")
            fig.tight_layout()
            fig_path = out_dir / "mech_interp_decomposition.pdf"
            fig.savefig(fig_path)
            plt.close(fig)
            if s3_prefix:
                s3_put(fig_path, f"{s3_prefix}/mech_interp_decomposition.pdf")

    except Exception as e:
        print(f"[s3] figures ERROR: {e}\n{traceback.format_exc()}", flush=True)
        payload["stage3"]["figures_error"] = str(e)

    # ------------------------------------------------------------------
    # Stage J: write DONE
    # ------------------------------------------------------------------
    payload["stage3"]["wall_time_seconds"] = float(time.time() - t_start)
    payload["stage3"]["aws_cost_estimate_usd"] = (
        (time.time() - t_start) / 3600.0 * args.cost_per_hour
    )
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    write_headline(headline_path, payload)
    done_path.write_text("done\n")
    if s3_prefix:
        s3_put(json_path, f"{s3_prefix}/mech_interp_full_results.json")
        s3_put(headline_path, f"{s3_prefix}/mech_interp_full_HEADLINE.txt")
        s3_put(done_path, f"{s3_prefix}/mech_interp_STAGE3_DONE")

    print(f"[s3] DONE @ {(time.time()-t_start)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
        sys.exit(rc)
    except Exception as e:
        print(f"[fatal] {e}\n{traceback.format_exc()}", flush=True)
        sys.exit(1)
