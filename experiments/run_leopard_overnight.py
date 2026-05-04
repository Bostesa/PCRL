"""experiments/run_leopard_overnight.py

LEOPARD adapter (Saillenfest & Lemberger 2025, ECAI 2025, arXiv:2507.12341).
Uses the official non-LEOPARD reference implementation
(github.com/toinesayan/non-LEOPARD) — code path appended to sys.path; we feed
cached BERT layer-N reps directly via init_model + train.

Pipeline:
  1. Cache BIOS top-10 BERT-base reps at layers {0, 1, 6, 12} (train + dev).
  2. For each layer, train cascade_projection (LEACE oblique→orthogonal +
     orthogonal projection trained with MMD density-matching loss).
     Hyperparameters from main.py biasbios config:
       projection_rank=50 (primary), 128 (rank-bump if time), 250 (final),
       batch_size=8192, num_epochs=100, scheduler_milestones=[50],
       loss=MMD, gamma=100, rbf_n_kernels=5,
       learning_rate=5e-4, weight_decay=5e-4.
  3. Population linear R² + sklearn LR balanced accuracy on projected dev.
  4. Push leopard_results.json + leopard_HEADLINE.txt to S3 after each
     (layer, rank) pair.

Layer convention matches v2_bios_FINAL: hidden_states[N].
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

LEOPARD_PATH = os.environ.get("LEOPARD_PATH", "/home/ubuntu/non-LEOPARD")
if LEOPARD_PATH not in sys.path:
    sys.path.insert(0, LEOPARD_PATH)


# ---------------------------------------------------------------------------
# Population R² (matches v2_bios_FINAL convention)
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


def sklearn_lr_balanced_acc(X_tr, Z_tr, X_te, Z_te) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced")
    clf.fit(X_tr.numpy(), Z_tr.numpy())
    pred = clf.predict(X_te.numpy())
    return float(balanced_accuracy_score(Z_te.numpy(), pred))


# ---------------------------------------------------------------------------
# Cache BERT reps (identical convention to QLEACE script)
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_bios_reps(*, n_train: int, seed: int, batch_size: int, max_length: int,
                    layers: list[int], device: torch.device,
                    cache_dir: str | None = None) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel

    from pcrl.language.bios_dataset import (
        BIOS_TOP10_IDS, _LABEL_TO_LOCAL, _filter_top10, _stratified_subsample,
    )

    print(f"[cache] loading BIOS train + dev splits...", flush=True)
    full_train = load_dataset("LabHC/bias_in_bios", split="train", cache_dir=cache_dir)
    full_dev = load_dataset("LabHC/bias_in_bios", split="dev", cache_dir=cache_dir)

    train_filtered = _filter_top10(full_train)
    occ_int_tr = np.array(train_filtered["profession"])
    gen_int_tr = np.array(train_filtered["gender"])
    occ_local_tr = np.array([_LABEL_TO_LOCAL[p] for p in occ_int_tr])
    keep_idx = _stratified_subsample(occ_local_tr, gen_int_tr, n_train, seed)
    train_sub = train_filtered.select(keep_idx.tolist())

    dev_filtered = _filter_top10(full_dev)
    print(f"[cache] train_sub={len(train_sub)} dev={len(dev_filtered)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()

    def encode_split(rows):
        gen = torch.tensor(rows["gender"], dtype=torch.long)
        occ = torch.tensor([_LABEL_TO_LOCAL[p] for p in rows["profession"]], dtype=torch.long)
        n = len(rows["hard_text"])
        outs = {L: torch.empty((n, 768), dtype=torch.float32) for L in layers}
        for i in range(0, n, batch_size):
            texts = rows["hard_text"][i:i + batch_size]
            enc = tokenizer(texts, max_length=max_length, truncation=True,
                            padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=mask,
                        output_hidden_states=True, return_dict=True)
            for L in layers:
                outs[L][i:i + batch_size] = out.hidden_states[L][:, 0, :].float().cpu()
            if (i // batch_size) % 50 == 0:
                print(f"[cache] {i}/{n}", flush=True)
        return outs, gen, occ

    print(f"[cache] forward train...", flush=True)
    t0 = time.time()
    train_reps, train_gen, train_occ = encode_split(train_sub[:])
    print(f"[cache] train done @ {time.time()-t0:.1f}s", flush=True)

    print(f"[cache] forward dev...", flush=True)
    t0 = time.time()
    dev_reps, dev_gen, dev_occ = encode_split(dev_filtered[:])
    print(f"[cache] dev done @ {time.time()-t0:.1f}s", flush=True)

    return {
        "train_reps": train_reps, "train_gen": train_gen, "train_occ": train_occ,
        "dev_reps": dev_reps, "dev_gen": dev_gen, "dev_occ": dev_occ,
    }


# ---------------------------------------------------------------------------
# LEOPARD train + apply
# ---------------------------------------------------------------------------

def fit_leopard_one_rank(X_train: torch.Tensor, Z_train: torch.Tensor, *,
                         rank: int, num_epochs: int = 100,
                         batch_size: int = 8192, learning_rate: float = 5e-4,
                         weight_decay: float = 5e-4, gamma: float = 100.0,
                         scheduler_milestones: tuple = (50,),
                         scheduler_gamma: float = 0.1,
                         rbf_n_kernels: int = 5, seed: int = 0,
                         device: torch.device = torch.device("cuda")) -> dict:
    """Train cascade_projection LEOPARD eraser; return projector P as torch.Tensor."""
    from eraser import init_model
    from loss import init_loss
    from training import train as leopard_train

    Xt = X_train.float()
    Zt = Z_train.long()
    cfg = dict(
        eraser_model="cascade_projection",
        loss_name="MMD",
        projection_rank=rank,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gamma=gamma,
        scheduler_milestones=list(scheduler_milestones),
        scheduler_gamma=scheduler_gamma,
        rbf_n_kernels=rbf_n_kernels,
        seed=seed,
    )
    print(f"[leopard] init_model rank={rank} d={Xt.shape[1]}", flush=True)
    model = init_model(X=Xt, Z=Zt, **cfg)
    model = model.to(device)

    criterion = init_loss(**cfg)
    print(f"[leopard] training {num_epochs} epochs, batch_size={batch_size}, "
          f"n_train={Xt.shape[0]}...", flush=True)
    t0 = time.time()
    leopard_train(
        model, criterion, Xt, Zt,
        device=device, gradient_acumulation_steps=1, **cfg,
    )
    train_seconds = time.time() - t0
    print(f"[leopard] train done @ {train_seconds:.1f}s", flush=True)

    model.eval()
    P = model.get_projector().detach().cpu()
    return {"P": P, "wall_seconds": train_seconds, "rank": rank}


def apply_leopard(P: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return X @ P.T.to(X.dtype)


# ---------------------------------------------------------------------------
# S3 push
# ---------------------------------------------------------------------------

def s3_put(local_path: Path, s3_uri: str) -> None:
    if not s3_uri:
        return
    cmd = ["aws", "s3", "cp", str(local_path), s3_uri]
    rc = subprocess.run(cmd, capture_output=True)
    print(f"[s3] {s3_uri} rc={rc.returncode}", flush=True)
    if rc.stderr:
        print(f"[s3-err] {rc.stderr.decode(errors='ignore')[:500]}", flush=True)


# ---------------------------------------------------------------------------
# Headline writer
# ---------------------------------------------------------------------------

def write_headline(path: Path, payload: dict) -> None:
    L = payload["leopard"]
    rows_primary = []
    for layer in (0, 1, 6, 12):
        key = f"layer_{layer}"
        if key in L:
            r = L[key]
            primary = r.get("rank_50", {})
            rows_primary.append(
                f"{layer:>5}  | {r.get('vanilla_R2', float('nan')):.3f}   | "
                f"{primary.get('leopard_R2', float('nan')):.3f}"
            )
        else:
            rows_primary.append(f"{layer:>5}  | (skipped)")

    layer0 = L.get("layer_0", {})
    primary0 = layer0.get("rank_50", {})
    leopard_l0 = primary0.get("leopard_R2", float("nan"))
    delta = layer0.get("vanilla_R2", float("nan")) - leopard_l0
    headline = (
        f"LEOPARD layer 0 (rank 50): {layer0.get('vanilla_R2', float('nan')):.3f} → "
        f"{leopard_l0:.3f} (delta = {delta:+.3f})"
    )
    if not np.isfinite(leopard_l0):
        branch = "TRAINING_FAILED" if "error" in primary0 else "INCOMPLETE"
    elif leopard_l0 < 0.5:
        branch = "DROP_BELOW_0.5"
    elif (layer0.get("vanilla_R2", 0) - leopard_l0) > 0.1:
        branch = "PARTIAL_DROP"
    elif leopard_l0 > layer0.get("vanilla_R2", 1):
        branch = "BACKFIRE"
    else:
        branch = "PLATEAU"

    rank_sweep_lines = []
    for ranks_run in (50, 128, 250):
        rkey = f"rank_{ranks_run}"
        if rkey in layer0:
            rank_sweep_lines.append(
                f"  rank {ranks_run:>3}: layer 0 = {layer0[rkey].get('leopard_R2', float('nan')):.3f}"
            )

    text = (
        "LEOPARD BIOS Results\n"
        "====================\n"
        f"Wall time: {L.get('wall_time_seconds', float('nan'))/60:.1f} min | "
        f"Cost: ~${L.get('aws_cost_estimate_usd', float('nan')):.2f} | "
        f"Instance: {L.get('aws_instance_id', 'unknown')}\n"
        f"Code: {L.get('code_source', 'unknown')}\n\n"
        "Primary table (rank=50):\n"
        "Layer  | vanilla | LEOPARD\n"
        "-------+---------+--------\n"
        + "\n".join(rows_primary)
        + "\n\n"
        f"HEADLINE: {headline}\n"
        f"BRANCH: {branch}\n"
    )
    if rank_sweep_lines:
        text += "\nLayer-0 rank sweep:\n" + "\n".join(rank_sweep_lines) + "\n"
    path.write_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/v2_bios_FINAL")
    ap.add_argument("--s3-uri", default="")
    ap.add_argument("--instance-id", default="")
    ap.add_argument("--cost-per-hour", type=float, default=0.20)
    ap.add_argument("--leopard-num-epochs", type=int, default=100)
    ap.add_argument("--leopard-batch-size", type=int, default=8192)
    ap.add_argument("--leopard-time-budget-sec", type=int, default=10800,
                    help="Stop launching new (layer, rank) trainings after this elapses")
    ap.add_argument("--ranks-extra", nargs="*", type=int, default=[],
                    help="Additional ranks to try at layer 0 only if rank=50 plateaus")
    ap.add_argument("--code-commit", default="")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    headline_path = out_dir / "leopard_HEADLINE.txt"
    json_path = out_dir / "leopard_results.json"
    done_path = out_dir / "LEOPARD_DONE"

    s3_prefix = args.s3_uri.rstrip("/")

    device = torch.device(args.device if torch.cuda.is_available()
                          else ("cuda" if args.device != "cpu" else "cpu"))
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda requested but unavailable, falling back to cpu", flush=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    payload = {
        "leopard": {
            "code_source": f"github.com/toinesayan/non-LEOPARD@{args.code_commit or 'unknown'}",
            "aws_instance_id": args.instance_id,
            "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "leopard_num_epochs": args.leopard_num_epochs,
            "leopard_batch_size": args.leopard_batch_size,
        }
    }
    t_start = time.time()

    # ------------------------------------------------------------------
    # Stage 1: cache reps
    # ------------------------------------------------------------------
    layers = [0, 1, 6, 12]
    print(f"[main] caching BIOS reps for layers {layers}...", flush=True)
    cache = cache_bios_reps(
        n_train=args.n_train, seed=args.seed,
        batch_size=args.batch_size, max_length=args.max_length,
        layers=layers, device=device,
    )
    payload["leopard"]["n_train"] = int(cache["train_gen"].shape[0])
    payload["leopard"]["n_eval"] = int(cache["dev_gen"].shape[0])

    train_gen = cache["train_gen"].long()
    dev_gen = cache["dev_gen"].long()

    # ------------------------------------------------------------------
    # Stage 2: per-layer rank-50 LEOPARD
    # ------------------------------------------------------------------
    primary_rank = 50

    def fit_and_eval_one(L: int, rank: int) -> dict:
        X_tr = cache["train_reps"][L]
        X_te = cache["dev_reps"][L]
        try:
            t_lay = time.time()
            res = fit_leopard_one_rank(
                X_tr, train_gen,
                rank=rank,
                num_epochs=args.leopard_num_epochs,
                batch_size=args.leopard_batch_size,
                seed=args.seed,
                device=device,
            )
            P = res["P"]
            X_te_proj = apply_leopard(P, X_te)
            r2_lp = population_linear_r2(X_te_proj, dev_gen)
            try:
                X_tr_proj = apply_leopard(P, X_tr)
                bal_acc = sklearn_lr_balanced_acc(X_tr_proj, train_gen, X_te_proj, dev_gen)
            except Exception as e:
                print(f"[probe-acc-error L={L} r={rank}] {e}", flush=True)
                bal_acc = float("nan")
            return {
                "rank": rank,
                "leopard_R2": r2_lp,
                "leopard_balanced_acc": bal_acc,
                "training_seconds": float(res["wall_seconds"]),
                "wall_seconds": float(time.time() - t_lay),
            }
        except Exception as e:
            print(f"[fit-error L={L} r={rank}] {e}\n{traceback.format_exc()}", flush=True)
            return {"rank": rank, "error": str(e),
                    "leopard_R2": float("nan"), "leopard_balanced_acc": float("nan")}

    for L in layers:
        if (time.time() - t_start) > args.leopard_time_budget_sec:
            print(f"[main] time budget {args.leopard_time_budget_sec}s exceeded — "
                  f"skipping remaining layers", flush=True)
            break
        print(f"[main] layer {L} rank={primary_rank}", flush=True)
        X_te = cache["dev_reps"][L]
        r2_vanilla = population_linear_r2(X_te, dev_gen)
        layer_payload = {"vanilla_R2": r2_vanilla}
        layer_payload[f"rank_{primary_rank}"] = fit_and_eval_one(L, primary_rank)
        payload["leopard"][f"layer_{L}"] = layer_payload

        payload["leopard"]["wall_time_seconds"] = float(time.time() - t_start)
        payload["leopard"]["aws_cost_estimate_usd"] = (
            (time.time() - t_start) / 3600.0 * args.cost_per_hour
        )
        json_path.write_text(json.dumps(payload, indent=2, default=float))
        write_headline(headline_path, payload)
        if s3_prefix:
            s3_put(json_path, f"{s3_prefix}/leopard_results.json")
            s3_put(headline_path, f"{s3_prefix}/leopard_HEADLINE.txt")

    # ------------------------------------------------------------------
    # Stage 3: rank-bump sweep at layer 0 only (time permitting)
    # ------------------------------------------------------------------
    layer0 = payload["leopard"].get("layer_0", {})
    primary0 = layer0.get(f"rank_{primary_rank}", {})
    primary_R2 = primary0.get("leopard_R2", float("nan"))
    plateau = (np.isfinite(primary_R2) and primary_R2 >= 0.5)

    if plateau and args.ranks_extra:
        print(f"[main] rank={primary_rank} plateau at layer 0 "
              f"(R²={primary_R2:.3f}) — sweeping extra ranks {args.ranks_extra}",
              flush=True)
        for r in args.ranks_extra:
            if (time.time() - t_start) > args.leopard_time_budget_sec:
                print(f"[main] time budget exceeded, skipping rank={r}", flush=True)
                break
            print(f"[main] rank-bump layer 0 rank={r}", flush=True)
            res_r = fit_and_eval_one(0, r)
            payload["leopard"]["layer_0"][f"rank_{r}"] = res_r
            payload["leopard"]["wall_time_seconds"] = float(time.time() - t_start)
            payload["leopard"]["aws_cost_estimate_usd"] = (
                (time.time() - t_start) / 3600.0 * args.cost_per_hour
            )
            json_path.write_text(json.dumps(payload, indent=2, default=float))
            write_headline(headline_path, payload)
            if s3_prefix:
                s3_put(json_path, f"{s3_prefix}/leopard_results.json")
                s3_put(headline_path, f"{s3_prefix}/leopard_HEADLINE.txt")

    # ------------------------------------------------------------------
    # Stage 4: write DONE
    # ------------------------------------------------------------------
    done_path.write_text("done\n")
    if s3_prefix:
        s3_put(done_path, f"{s3_prefix}/LEOPARD_DONE")
        s3_put(json_path, f"{s3_prefix}/leopard_results.json")
        s3_put(headline_path, f"{s3_prefix}/leopard_HEADLINE.txt")

    print(f"[main] DONE @ {time.time()-t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
        sys.exit(rc)
    except Exception as e:
        print(f"[fatal] {e}\n{traceback.format_exc()}", flush=True)
        sys.exit(1)
