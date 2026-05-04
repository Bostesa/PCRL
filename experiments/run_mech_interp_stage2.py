"""Stage 2 mech-interp go/no-go on BIOS BERT.

Block-1 attention head activation patching. Identifies which of the 12 heads
in BERT-base block 1 mediate gender re-derivation when the input pronoun is
flipped.

Pipeline:
  1. Load BIOS top-10 dev. Take 500 train + 200 eval (small for Stage 2).
  2. Build (clean, corrupt) pairs via cda_invariance.swap_gender_pronouns.
     Filter to pairs that tokenize to identical lengths (the vast majority).
  3. Forward clean + corrupt; cache activations.
  4. Train sklearn LogisticRegression probe on clean train layer-1 [CLS] reps
     (= TL blocks.1.hook_resid_post = HF hidden_states[2]).
  5. For each head h in 0..11:
       hook blocks.1.attn.hook_z, replace head h's output with clean cache,
       run on corrupt, score patched layer-1 dev reps with the probe.
       Δ_h = mean(probe.decision_function(patched)) - mean(probe.decision_function(corrupt))
  6. Also report population_linear_r2 on (vanilla / corrupt / per-head-patched)
     dev reps — this is the apples-to-apples-with-rank_k_sweep number.
  7. Save JSON + HEADLINE + DONE marker; push to S3.

Layer convention for the probe target ("layer 1" = block-1 output = HF
hidden_states[2]) matches rank_k_sweep.json.

Probe metric is documented in JSON metadata; not directly comparable to
rank_k_sweep's population_linear_r2, so we also report population R² for
direct comparison.
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
# Build (clean, corrupt) pairs with same token length
# ---------------------------------------------------------------------------

def build_counterfactual_pairs(
    texts: list[str], tokenizer, *, max_length: int = 128
) -> tuple[list[str], list[str], np.ndarray]:
    """Return (clean_texts, corrupt_texts, keep_mask) for items where the
    pronoun-swapped text tokenizes to the same length."""
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
        # Compare non-pad token counts
        a_len = int((tk_a[0] != tokenizer.pad_token_id).sum())
        b_len = int((tk_b[0] != tokenizer.pad_token_id).sum())
        if a_len != b_len:
            mask.append(False)
            continue
        clean_keep.append(t)
        corrupt_keep.append(s)
        mask.append(True)
    return clean_keep, corrupt_keep, np.array(mask, dtype=bool)


# ---------------------------------------------------------------------------
# Forward + cache layer-1 [CLS] reps via TL HookedEncoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_and_cache_layer1(
    model, tokenizer, texts: list[str], *,
    max_length: int = 128, batch_size: int = 32, device: str = "cuda",
) -> dict:
    """Forward `texts` through HookedEncoder; return layer-1 [CLS] reps
    (= TL blocks.1.hook_resid_post[:, 0, :]), and full caches if requested."""
    n = len(texts)
    layer1_cls = torch.empty((n, model.cfg.d_model), dtype=torch.float32)
    for i in range(0, n, batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, max_length=max_length, truncation=True,
                        padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].to(device)
        _, cache = model.run_with_cache(ids,
                                        names_filter="blocks.1.hook_resid_post")
        layer1_cls[i:i + batch_size] = cache["blocks.1.hook_resid_post"][:, 0, :].float().cpu()
    return {"layer1_cls": layer1_cls}


@torch.no_grad()
def cache_block1_attn_z(
    model, tokenizer, texts: list[str], *,
    max_length: int = 128, batch_size: int = 32, device: str = "cuda",
) -> dict:
    """For patching: cache blocks.1.attn.hook_z and the input token ids
    so we can re-run the same inputs with hooks."""
    n = len(texts)
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    z_all = torch.empty((n, max_length, n_heads, d_head), dtype=torch.float32)
    ids_all = torch.empty((n, max_length), dtype=torch.long)
    for i in range(0, n, batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, max_length=max_length, truncation=True,
                        padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].to(device)
        _, cache = model.run_with_cache(ids,
                                        names_filter="blocks.1.attn.hook_z")
        z = cache["blocks.1.attn.hook_z"]  # (B, T, H, D_head)
        z_all[i:i + batch_size] = z.float().cpu()
        ids_all[i:i + batch_size] = ids.cpu()
    return {"z": z_all, "ids": ids_all}


# ---------------------------------------------------------------------------
# Patch one head, re-forward, return layer-1 [CLS]
# ---------------------------------------------------------------------------

@torch.no_grad()
def patched_layer1_cls(
    model, ids_corrupt: torch.Tensor, clean_z: torch.Tensor, head_idx: int,
    *, batch_size: int = 32, device: str = "cuda",
) -> torch.Tensor:
    """For each example, run on ids_corrupt with blocks.1.attn.hook_z's
    head `head_idx` replaced with clean_z's same-head slice. Return
    layer-1 [CLS] reps (= blocks.1.hook_resid_post[:, 0, :])."""
    n = ids_corrupt.shape[0]
    out = torch.empty((n, model.cfg.d_model), dtype=torch.float32)
    for i in range(0, n, batch_size):
        ids_b = ids_corrupt[i:i + batch_size].to(device)
        clean_z_b = clean_z[i:i + batch_size].to(device)
        captured: dict[str, torch.Tensor] = {}

        def patch_z(z, hook, _ref=clean_z_b, _h=head_idx):
            z[..., _h, :] = _ref[..., _h, :]
            return z

        def cap_resid(act, hook, _store=captured):
            _store["v"] = act.detach().float().cpu()

        model.run_with_hooks(
            ids_b,
            fwd_hooks=[
                ("blocks.1.attn.hook_z", patch_z),
                ("blocks.1.hook_resid_post", cap_resid),
            ],
        )
        out[i:i + batch_size] = captured["v"][:, 0, :]
    return out


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def train_lr_probe(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X_train, y_train)
    return clf


def probe_logits(probe, X: np.ndarray) -> np.ndarray:
    return probe.decision_function(X)


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------

def write_headline(path: Path, payload: dict) -> None:
    s = payload["stage2"]
    rows = []
    for h in range(12):
        d = s["per_head"].get(str(h), {})
        delta = d.get("delta_probe_logit", float("nan"))
        sign = "+" if (np.isfinite(delta) and delta > 0) else ("-" if np.isfinite(delta) and delta < 0 else "?")
        rows.append(f"{h:>4} | {delta:+.4f}      | {sign}")

    deltas = [s["per_head"][str(h)].get("delta_probe_logit", 0.0) for h in range(12)
              if str(h) in s["per_head"]]
    abs_deltas = [abs(d) for d in deltas]
    sum_abs = sum(abs_deltas)
    top3_idx = sorted(range(len(deltas)), key=lambda i: abs_deltas[i], reverse=True)[:3]
    top3_sum = sum(abs_deltas[i] for i in top3_idx)
    diffuseness = (top3_sum / sum_abs) if sum_abs > 0 else float("nan")

    sanity_one_head = "Y" if any(d > 0.05 for d in abs_deltas) else "N"
    sanity_top3 = "Y" if (np.isfinite(diffuseness) and diffuseness > 0.30) else "N"
    signs = [1 if d > 0 else -1 for d in deltas]
    sanity_consistent = "Y" if abs(sum(signs)) >= 8 else "N"

    pop_r2 = s.get("population_R2", {})
    overall = "GO STAGE 3" if (sanity_one_head == "Y") else "NO-GO"

    text = (
        "STAGE 2 GO/NO-GO RESULTS\n"
        "========================\n"
        f"Wall time: {s.get('wall_time_seconds', 0)/60:.1f} min | "
        f"Cost: ~${s.get('aws_cost_estimate_usd', 0):.2f} | "
        f"Instance: {s.get('aws_instance_id', '?')}\n\n"
        f"Probe: {s.get('probe_type', '?')}\n"
        f"Probe train n: {s.get('probe_train_n', '?')} | "
        f"Eval n: {s.get('probe_eval_n', '?')}\n"
        f"Pairs kept (same-length tokenization): "
        f"{s.get('n_pairs_kept', '?')}/{s.get('n_pairs_attempted', '?')}\n\n"
        f"Vanilla layer-1 R² (population OLS, dev): "
        f"{pop_r2.get('vanilla', float('nan')):.3f}  "
        f"(rank_k_sweep baseline: 0.821)\n"
        f"Corrupt layer-1 R² (population OLS, dev): "
        f"{pop_r2.get('corrupt', float('nan')):.3f}\n\n"
        "Block-1 head patching (clean → corrupt run):\n"
        "Head | Δ_probe_logit | sign\n"
        "-----+---------------+-----\n"
        + "\n".join(rows) + "\n\n"
        f"Top 3 heads (by |Δ|): {[(int(i), f'{deltas[i]:+.4f}') for i in top3_idx]}\n"
        f"Diffuseness ratio (top-3 |Δ| / total |Δ|): {diffuseness:.3f}\n\n"
        "Sanity:\n"
        f"  - At least one head |Δ| > 0.05? [{sanity_one_head}]\n"
        f"  - Top-3 sum > 30% of total |Δ|? [{sanity_top3}]   "
        "(sharp circuit signal)\n"
        f"  - All same sign? [{sanity_consistent}]\n\n"
        f"OVERALL: {overall}\n"
    )
    path.write_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/v2_bios_FINAL")
    ap.add_argument("--s3-uri", default="")
    ap.add_argument("--instance-id", default="")
    ap.add_argument("--cost-per-hour", type=float, default=0.526)
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "mech_interp_stage2_results.json"
    headline_path = out_dir / "mech_interp_stage2_HEADLINE.txt"
    done_path = out_dir / "mech_interp_STAGE2_DONE"
    s3_prefix = args.s3_uri.rstrip("/")

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; falling back to cpu", flush=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    payload = {
        "stage2": {
            "implementation_source": "transformer_lens HookedEncoder bert-base-uncased",
            "aws_instance_id": args.instance_id,
            "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "probe_type": "sklearn_LogisticRegression(class_weight=balanced, max_iter=2000)",
            "rank_k_sweep_baseline_metric": "population_linear_r2 (closed-form OLS), not sklearn LR",
            "expected_apples_to_apples": "Stage 2 also reports population_R2 for vanilla; should reproduce ~0.821",
            "probe_train_split": f"stage2_mini_train_n={args.n_train}_seed={args.seed}",
            "layer_convention": "blocks.1.hook_resid_post = HF hidden_states[2] = block-1 output (matches rank_k_sweep)",
            "patching_target": "blocks.1.attn.hook_z[..., head, :]",
        }
    }
    t_start = time.time()

    # ------------------------------------------------------------------
    # Stage A: load model + tokenizer + BIOS dev
    # ------------------------------------------------------------------
    print("[s2] loading HookedEncoder bert-base-uncased...", flush=True)
    from transformer_lens import HookedEncoder
    model = HookedEncoder.from_pretrained("bert-base-uncased")
    model = model.to(device).eval()
    tokenizer = model.tokenizer
    print(f"[s2] model on {device}; n_layers={model.cfg.n_layers} n_heads={model.cfg.n_heads}", flush=True)

    print("[s2] loading BIOS top-10 dev split...", flush=True)
    from datasets import load_dataset
    from pcrl.language.bios_dataset import _filter_top10, _LABEL_TO_LOCAL
    dev = _filter_top10(load_dataset("LabHC/bias_in_bios", split="dev"))
    print(f"[s2] dev n={len(dev)}", flush=True)

    rng = np.random.default_rng(args.seed)
    n_total = len(dev)
    take = min(args.n_train + args.n_eval + 200, n_total)  # margin for filter dropouts
    idx = rng.choice(n_total, size=take, replace=False)
    sub = dev.select(sorted(idx.tolist()))
    texts = list(sub["hard_text"])
    genders = np.array(sub["gender"], dtype=np.int64)
    occs = np.array([_LABEL_TO_LOCAL[p] for p in sub["profession"]], dtype=np.int64)

    # ------------------------------------------------------------------
    # Stage B: build same-length counterfactual pairs
    # ------------------------------------------------------------------
    print("[s2] building counterfactual pairs (pronoun + honorific swap)...", flush=True)
    clean_t, corrupt_t, mask = build_counterfactual_pairs(
        texts, tokenizer, max_length=args.max_length,
    )
    g_kept = genders[mask]
    o_kept = occs[mask]
    n_attempted = len(texts)
    n_kept = len(clean_t)
    print(f"[s2] kept {n_kept}/{n_attempted} same-length pairs", flush=True)
    payload["stage2"]["n_pairs_attempted"] = int(n_attempted)
    payload["stage2"]["n_pairs_kept"] = int(n_kept)

    if n_kept < args.n_train + args.n_eval:
        print(f"[s2] WARN: kept ({n_kept}) < requested ({args.n_train + args.n_eval}); "
              f"resizing", flush=True)

    n_train = min(args.n_train, n_kept // 2)
    n_eval = min(args.n_eval, n_kept - n_train)
    perm = rng.permutation(n_kept)
    tr_idx = perm[:n_train].tolist()
    te_idx = perm[n_train:n_train + n_eval].tolist()
    payload["stage2"]["probe_train_n"] = int(n_train)
    payload["stage2"]["probe_eval_n"] = int(n_eval)
    print(f"[s2] split: train={n_train}, eval={n_eval}", flush=True)

    # ------------------------------------------------------------------
    # Stage C: forward clean + corrupt; cache reps
    # ------------------------------------------------------------------
    print("[s2] forward clean train + eval (caching layer-1 [CLS])...", flush=True)
    train_clean_texts = [clean_t[i] for i in tr_idx]
    eval_clean_texts = [clean_t[i] for i in te_idx]
    eval_corrupt_texts = [corrupt_t[i] for i in te_idx]
    g_train = g_kept[tr_idx]
    g_eval = g_kept[te_idx]

    train_clean_cls = forward_and_cache_layer1(
        model, tokenizer, train_clean_texts,
        max_length=args.max_length, batch_size=args.batch_size, device=device,
    )["layer1_cls"]
    eval_clean_cls = forward_and_cache_layer1(
        model, tokenizer, eval_clean_texts,
        max_length=args.max_length, batch_size=args.batch_size, device=device,
    )["layer1_cls"]
    eval_corrupt_cls = forward_and_cache_layer1(
        model, tokenizer, eval_corrupt_texts,
        max_length=args.max_length, batch_size=args.batch_size, device=device,
    )["layer1_cls"]
    print(f"[s2] cached: train_clean {tuple(train_clean_cls.shape)} "
          f"eval_clean {tuple(eval_clean_cls.shape)} "
          f"eval_corrupt {tuple(eval_corrupt_cls.shape)}", flush=True)

    # ------------------------------------------------------------------
    # Stage D: train probe
    # ------------------------------------------------------------------
    print("[s2] training sklearn LR probe...", flush=True)
    probe = train_lr_probe(train_clean_cls.numpy(), g_train)
    train_acc = float(probe.score(train_clean_cls.numpy(), g_train))
    eval_clean_acc = float(probe.score(eval_clean_cls.numpy(), g_eval))
    eval_corrupt_acc = float(probe.score(eval_corrupt_cls.numpy(), g_eval))
    payload["stage2"]["probe"] = {
        "train_acc": train_acc,
        "eval_clean_acc": eval_clean_acc,
        "eval_corrupt_acc": eval_corrupt_acc,
    }
    print(f"[s2] probe: train_acc={train_acc:.3f} eval_clean={eval_clean_acc:.3f} "
          f"eval_corrupt={eval_corrupt_acc:.3f}", flush=True)

    eval_clean_logits = probe_logits(probe, eval_clean_cls.numpy())
    eval_corrupt_logits = probe_logits(probe, eval_corrupt_cls.numpy())
    mean_clean = float(eval_clean_logits.mean())
    mean_corrupt = float(eval_corrupt_logits.mean())
    print(f"[s2] mean probe logit (he-she sign): clean={mean_clean:+.4f} "
          f"corrupt={mean_corrupt:+.4f}", flush=True)
    payload["stage2"]["probe"]["mean_logit_clean"] = mean_clean
    payload["stage2"]["probe"]["mean_logit_corrupt"] = mean_corrupt

    # Population R² baselines
    pop_r2_vanilla = population_linear_r2(eval_clean_cls, torch.tensor(g_eval))
    pop_r2_corrupt = population_linear_r2(eval_corrupt_cls, torch.tensor(g_eval))
    payload["stage2"]["population_R2"] = {
        "vanilla": pop_r2_vanilla,
        "corrupt": pop_r2_corrupt,
    }
    print(f"[s2] population R²: vanilla={pop_r2_vanilla:.3f} corrupt={pop_r2_corrupt:.3f} "
          f"(rank_k_sweep baseline: 0.821)", flush=True)

    # ------------------------------------------------------------------
    # Stage E: cache clean's blocks.1.attn.hook_z for eval
    # ------------------------------------------------------------------
    print("[s2] caching clean blocks.1.attn.hook_z for eval...", flush=True)
    clean_eval_z_cache = cache_block1_attn_z(
        model, tokenizer, eval_clean_texts,
        max_length=args.max_length, batch_size=args.batch_size, device=device,
    )
    clean_z = clean_eval_z_cache["z"]
    print(f"[s2] clean_z shape: {tuple(clean_z.shape)}", flush=True)

    # Tokenize corrupt eval to get input ids for re-forward
    print("[s2] tokenizing corrupt eval...", flush=True)
    corrupt_ids_list = []
    for i in range(0, len(eval_corrupt_texts), args.batch_size):
        chunk = eval_corrupt_texts[i:i + args.batch_size]
        enc = tokenizer(chunk, max_length=args.max_length, truncation=True,
                        padding="max_length", return_tensors="pt")
        corrupt_ids_list.append(enc["input_ids"])
    corrupt_ids = torch.cat(corrupt_ids_list)
    print(f"[s2] corrupt_ids shape: {tuple(corrupt_ids.shape)}", flush=True)

    # ------------------------------------------------------------------
    # Stage F: per-head patching
    # ------------------------------------------------------------------
    print("[s2] per-head patching at block-1 (12 heads)...", flush=True)
    payload["stage2"]["per_head"] = {}
    for h in range(model.cfg.n_heads):
        t_h = time.time()
        patched_cls = patched_layer1_cls(
            model, corrupt_ids, clean_z, head_idx=h,
            batch_size=args.batch_size, device=device,
        )
        patched_logits = probe_logits(probe, patched_cls.numpy())
        mean_patched = float(patched_logits.mean())
        delta = mean_patched - mean_corrupt
        delta_clean = mean_patched - mean_clean
        pop_r2_patched = population_linear_r2(patched_cls, torch.tensor(g_eval))
        payload["stage2"]["per_head"][str(h)] = {
            "delta_probe_logit": delta,
            "delta_from_clean": delta_clean,
            "mean_logit_patched": mean_patched,
            "population_R2_patched": pop_r2_patched,
            "wall_seconds": float(time.time() - t_h),
        }
        print(f"[s2] head {h:>2}: Δ_probe_logit={delta:+.4f}  "
              f"R²_patched={pop_r2_patched:.3f}  ({time.time()-t_h:.1f}s)", flush=True)

        payload["stage2"]["wall_time_seconds"] = float(time.time() - t_start)
        payload["stage2"]["aws_cost_estimate_usd"] = (
            (time.time() - t_start) / 3600.0 * args.cost_per_hour
        )
        json_path.write_text(json.dumps(payload, indent=2, default=float))
        write_headline(headline_path, payload)
        if s3_prefix:
            s3_put(json_path, f"{s3_prefix}/mech_interp_stage2_results.json")
            s3_put(headline_path, f"{s3_prefix}/mech_interp_stage2_HEADLINE.txt")

    # ------------------------------------------------------------------
    # Stage G: write DONE
    # ------------------------------------------------------------------
    done_path.write_text("done\n")
    if s3_prefix:
        s3_put(done_path, f"{s3_prefix}/mech_interp_STAGE2_DONE")
        s3_put(json_path, f"{s3_prefix}/mech_interp_stage2_results.json")
        s3_put(headline_path, f"{s3_prefix}/mech_interp_stage2_HEADLINE.txt")

    print(f"[s2] DONE @ {time.time()-t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
        sys.exit(rc)
    except Exception as e:
        print(f"[fatal] {e}\n{traceback.format_exc()}", flush=True)
        sys.exit(1)
