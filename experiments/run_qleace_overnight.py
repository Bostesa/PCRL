"""experiments/run_qleace_overnight.py

QLEACE (Quirke & Belrose 2025, arXiv:2502.02820, Theorem 2.6) implemented from
the paper. Wasserstein-2 barycenter erasure: optimal-transport scrubbing of a
binary protected attribute with both first- AND second-moment alignment.

Pipeline:
  1. Toy sanity check on synthetic d=64 Gaussians with different means+covs.
     Vanilla LR ≥ 0.95; post-QLEACE linear AND quadratic both < 0.55.
     If sanity FAILS, write FAILED tag, skip BIOS, exit.
  2. Cache BIOS top-10 BERT-base reps at layers {0, 1, 6, 12} (train + dev).
  3. Per layer: vanilla R², LEACE rank-1 R² (sanity), QLEACE R².
  4. Push qleace_results.json + qleace_HEADLINE.txt to S3 after each layer.

Layer convention (matches v2_bios_FINAL): layer N = hidden_states[N], where
hidden_states[0] is post-embedding+LayerNorm and hidden_states[N] for N≥1 is
the output of transformer block N-1.
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
# Matrix square root utilities (eigendecomp on symmetric PSD)
# ---------------------------------------------------------------------------

def _symm(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.T)


def matrix_sqrt(M: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    w, V = torch.linalg.eigh(_symm(M))
    w = torch.clamp(w, min=eps)
    return V @ torch.diag(torch.sqrt(w)) @ V.T


def matrix_inv_sqrt(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w, V = torch.linalg.eigh(_symm(M))
    w = torch.clamp(w, min=eps)
    return V @ torch.diag(1.0 / torch.sqrt(w)) @ V.T


def shrink(Sigma: torch.Tensor, alpha: float) -> torch.Tensor:
    d = Sigma.shape[0]
    tr = torch.trace(Sigma)
    I = torch.eye(d, dtype=Sigma.dtype, device=Sigma.device)
    return (1 - alpha) * Sigma + alpha * (tr / d) * I


# ---------------------------------------------------------------------------
# Wasserstein-2 covariance barycenter (Álvarez-Esteban 2016)
# ---------------------------------------------------------------------------

def w2_barycenter(S0: torch.Tensor, S1: torch.Tensor,
                  *, max_iter: int = 200, tol: float = 1e-6) -> tuple[torch.Tensor, int]:
    St = 0.5 * (S0 + S1)
    for it in range(max_iter):
        St_root = matrix_sqrt(St)
        a = matrix_sqrt(St_root @ S0 @ St_root)
        b = matrix_sqrt(St_root @ S1 @ St_root)
        St_new = 0.5 * (a + b)
        denom = max(float(torch.linalg.norm(St)), 1e-12)
        rel = float(torch.linalg.norm(St_new - St) / denom)
        St = St_new
        if rel < tol:
            return St, it + 1
    return St, max_iter


# ---------------------------------------------------------------------------
# QLEACE fit + apply (binary case)
# ---------------------------------------------------------------------------

def fit_qleace(X: torch.Tensor, Z: torch.Tensor, *, alpha: float = 1e-3) -> dict:
    X = X.to(torch.float64)
    Z = Z.to(torch.long)
    m_0 = X[Z == 0].mean(0)
    m_1 = X[Z == 1].mean(0)
    n0 = float((Z == 0).sum())
    n1 = float((Z == 1).sum())
    Xc0 = X[Z == 0] - m_0
    Xc1 = X[Z == 1] - m_1
    S0 = (Xc0.T @ Xc0) / max(n0 - 1, 1.0)
    S1 = (Xc1.T @ Xc1) / max(n1 - 1, 1.0)
    S0 = shrink(_symm(S0), alpha)
    S1 = shrink(_symm(S1), alpha)
    S_bar, n_iters = w2_barycenter(S0, S1)
    m_bar = 0.5 * (m_0 + m_1)
    def transport(S_i: torch.Tensor) -> torch.Tensor:
        Si_h = matrix_sqrt(S_i)
        Si_ih = matrix_inv_sqrt(S_i)
        inner = matrix_sqrt(Si_h @ S_bar @ Si_h)
        return Si_ih @ inner @ Si_ih
    A_0 = transport(S0)
    A_1 = transport(S1)
    return {
        "m_0": m_0, "m_1": m_1, "m_bar": m_bar,
        "A_0": A_0, "A_1": A_1, "n_bar_iters": int(n_iters),
    }


def apply_qleace(model: dict, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    Xd = X.to(torch.float64)
    out = torch.empty_like(Xd)
    z0 = (Z == 0)
    z1 = (Z == 1)
    out[z0] = (Xd[z0] - model["m_0"]) @ model["A_0"].T + model["m_bar"]
    out[z1] = (Xd[z1] - model["m_1"]) @ model["A_1"].T + model["m_bar"]
    return out.to(torch.float32)


# ---------------------------------------------------------------------------
# Probes
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
# LEACE rank-1 (concept_erasure library) — for sanity comparison
# ---------------------------------------------------------------------------

def fit_leace_rank1(X: torch.Tensor, Z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (P, mu) such that erased = mu + (X - mu) @ P.T."""
    from concept_erasure import LeaceEraser
    Z_oh = torch.nn.functional.one_hot(Z.long(), num_classes=2).float()
    eraser = LeaceEraser.fit(X.float(), Z_oh)
    return eraser.P.detach(), eraser.bias.detach()


def apply_leace(P: torch.Tensor, mu: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    P = P.to(X.dtype)
    mu = mu.to(X.dtype)
    return mu + (X - mu) @ P.T


# ---------------------------------------------------------------------------
# Toy sanity check
# ---------------------------------------------------------------------------

def toy_sanity(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = 64
    n = 5000
    m0 = np.zeros(d); m0[0] = +1.0
    m1 = np.zeros(d); m1[0] = -1.0
    sig0 = np.ones(d); sig0[0] = 2.0
    sig1 = np.ones(d); sig1[1] = 2.0
    X0 = rng.standard_normal((n, d)) * sig0 + m0
    X1 = rng.standard_normal((n, d)) * sig1 + m1
    X = np.vstack([X0, X1]).astype(np.float32)
    Z = np.array([0] * n + [1] * n, dtype=np.int64)
    perm = rng.permutation(2 * n)
    X = X[perm]; Z = Z[perm]
    X_tr, X_te = X[:n], X[n:]
    Z_tr, Z_te = Z[:n], Z[n:]
    Xt_tr, Xt_te = torch.tensor(X_tr), torch.tensor(X_te)
    Zt_tr, Zt_te = torch.tensor(Z_tr), torch.tensor(Z_te)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline

    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_tr, Z_tr)
    vanilla_lin = float(lr.score(X_te, Z_te))

    poly = Pipeline([
        ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
        ("lr", LogisticRegression(max_iter=4000)),
    ])
    poly.fit(X_tr, Z_tr)
    vanilla_quad = float(poly.score(X_te, Z_te))

    qleace = fit_qleace(Xt_tr, Zt_tr, alpha=1e-3)
    Xq_tr = apply_qleace(qleace, Xt_tr, Zt_tr).numpy()
    Xq_te = apply_qleace(qleace, Xt_te, Zt_te).numpy()

    lr2 = LogisticRegression(max_iter=2000); lr2.fit(Xq_tr, Z_tr)
    qleace_lin = float(lr2.score(Xq_te, Z_te))

    poly2 = Pipeline([
        ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
        ("lr", LogisticRegression(max_iter=4000)),
    ])
    poly2.fit(Xq_tr, Z_tr)
    qleace_quad = float(poly2.score(Xq_te, Z_te))

    passed = (qleace_lin < 0.55) and (qleace_quad < 0.55)
    return {
        "d": d, "n_per_class": n,
        "vanilla_lin_acc": vanilla_lin, "vanilla_quad_acc": vanilla_quad,
        "qleace_lin_acc": qleace_lin, "qleace_quad_acc": qleace_quad,
        "n_bar_iters": qleace["n_bar_iters"],
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Cache BERT layer-N reps for BIOS top-10 train + dev
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
# S3 push helpers
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
    q = payload["qleace"]
    toy = q.get("toy_sanity", {})
    rows = []
    for L in (0, 1, 6, 12):
        key = f"layer_{L}"
        if key in q:
            r = q[key]
            rows.append(f"{L:>5}  | {r.get('vanilla_R2', float('nan')):.3f}   | "
                        f"{r.get('leace_rank1_R2', float('nan')):.3f}   | "
                        f"{r.get('qleace_R2', float('nan')):.3f}")
        else:
            rows.append(f"{L:>5}  | (skipped)")

    layer0 = q.get("layer_0", {})
    delta = layer0.get("vanilla_R2", float("nan")) - layer0.get("qleace_R2", float("nan"))
    qleace_l0 = layer0.get("qleace_R2", float("nan"))
    headline = (
        f"QLEACE layer 0: {layer0.get('vanilla_R2', float('nan')):.3f} → "
        f"{qleace_l0:.3f} (delta = {delta:+.3f})"
    )
    if not np.isfinite(qleace_l0):
        branch = "INCOMPLETE"
    elif qleace_l0 < 0.5:
        branch = "DROP_BELOW_0.5"
    elif (layer0.get("vanilla_R2", 0) - qleace_l0) > 0.1:
        branch = "PARTIAL_DROP"
    elif qleace_l0 > layer0.get("vanilla_R2", 1):
        branch = "BACKFIRE"
    else:
        branch = "PLATEAU"

    sanity_line = (
        f"Toy sanity: linear={toy.get('qleace_lin_acc', float('nan')):.2f}, "
        f"quad={toy.get('qleace_quad_acc', float('nan')):.2f}, "
        f"[{'PASSED' if toy.get('passed') else 'FAILED'}]"
    )
    leace_l0 = layer0.get("leace_rank1_R2", float("nan"))
    leace_l0_check = "Y" if (np.isfinite(leace_l0) and abs(leace_l0 - 0.86) < 0.05) else "N"
    layer12 = q.get("layer_12", {})
    l12_check = "Y" if (np.isfinite(layer12.get("vanilla_R2", float("nan"))) and
                        abs(layer12.get("vanilla_R2", 0) - 0.94) < 0.10) else "N"

    text = (
        "QLEACE BIOS Results\n"
        "===================\n"
        f"Wall time: {q.get('wall_time_seconds', float('nan'))/60:.1f} min | "
        f"Cost: ~${q.get('aws_cost_estimate_usd', float('nan')):.2f} | "
        f"Instance: {q.get('aws_instance_id', 'unknown')}\n\n"
        f"{sanity_line}\n\n"
        "Layer  | vanilla | LEACE-1 | QLEACE\n"
        "-------+---------+---------+--------\n"
        + "\n".join(rows)
        + "\n\n"
        f"HEADLINE: {headline}\n"
        f"BRANCH: {branch}\n\n"
        f"Sanity: LEACE-1 layer 0 ({leace_l0:.3f}) reproduces existing 0.86 within 0.05? [{leace_l0_check}]\n"
        f"Sanity: layer 12 vanilla R² ({layer12.get('vanilla_R2', float('nan')):.3f}) near 0.94? [{l12_check}]\n"
    )
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
    ap.add_argument("--s3-uri", default="",
                    help="s3://bucket prefix; if set, push artifacts after each layer")
    ap.add_argument("--instance-id", default="")
    ap.add_argument("--cost-per-hour", type=float, default=0.20)
    ap.add_argument("--shrinkage-alpha", type=float, default=1e-3)
    ap.add_argument("--layers", nargs="*", type=int, default=[0, 1, 6, 12],
                    help="BIOS hidden_states layers to probe")
    ap.add_argument("--skip-toy-sanity", action="store_true",
                    help="Skip toy sanity check (only for smoke after sanity has passed)")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    headline_path = out_dir / "qleace_HEADLINE.txt"
    json_path = out_dir / "qleace_results.json"
    done_path = out_dir / "QLEACE_DONE"

    s3_prefix = args.s3_uri.rstrip("/")

    device = torch.device(args.device if torch.cuda.is_available()
                          else ("cuda" if args.device != "cpu" else "cpu"))
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[warn] cuda requested but unavailable, falling back to cpu", flush=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    payload = {
        "qleace": {
            "implementation_source": "from-paper-arxiv-2502.02820-Theorem-2.6",
            "aws_instance_id": args.instance_id,
            "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "shrinkage_alpha": args.shrinkage_alpha,
        }
    }
    t_start = time.time()

    # ------------------------------------------------------------------
    # Stage 1: toy sanity
    # ------------------------------------------------------------------
    if args.skip_toy_sanity:
        print("[main] skipping toy sanity (--skip-toy-sanity)", flush=True)
        toy = {"skipped": True, "passed": True}
    else:
        print("[main] toy sanity check...", flush=True)
        try:
            toy = toy_sanity(seed=args.seed)
        except Exception as e:
            print(f"[toy-sanity-error] {e}\n{traceback.format_exc()}", flush=True)
            toy = {"passed": False, "error": str(e)}
    payload["qleace"]["toy_sanity"] = toy
    if not toy.get("skipped"):
        print(f"[main] toy: vanilla_lin={toy.get('vanilla_lin_acc'):.3f} "
              f"quad={toy.get('vanilla_quad_acc'):.3f} qleace_lin={toy.get('qleace_lin_acc'):.3f} "
              f"qleace_quad={toy.get('qleace_quad_acc'):.3f} passed={toy.get('passed')}", flush=True)
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    if s3_prefix:
        s3_put(json_path, f"{s3_prefix}/qleace_results.json")

    if not toy.get("passed", False):
        write_failed_marker(out_dir, "QLEACE toy sanity FAILED — skipping BIOS")
        if s3_prefix:
            s3_put(out_dir / "QLEACE_FAILED", f"{s3_prefix}/QLEACE_FAILED")
            s3_put(json_path, f"{s3_prefix}/qleace_results.json")
        return 2

    # ------------------------------------------------------------------
    # Stage 2: cache BIOS reps
    # ------------------------------------------------------------------
    layers = sorted(set(int(L) for L in args.layers))
    print(f"[main] caching BIOS reps for layers {layers}...", flush=True)
    cache = cache_bios_reps(
        n_train=args.n_train, seed=args.seed,
        batch_size=args.batch_size, max_length=args.max_length,
        layers=layers, device=device,
    )
    payload["qleace"]["n_train"] = int(cache["train_gen"].shape[0])
    payload["qleace"]["n_eval"] = int(cache["dev_gen"].shape[0])

    # ------------------------------------------------------------------
    # Stage 3: per-layer probes
    # ------------------------------------------------------------------
    train_gen = cache["train_gen"].long()
    dev_gen = cache["dev_gen"].long()
    for L in layers:
        print(f"[main] layer {L}: probing...", flush=True)
        X_tr = cache["train_reps"][L]
        X_te = cache["dev_reps"][L]
        try:
            t0 = time.time()
            r2_vanilla = population_linear_r2(X_te, dev_gen)
            P, mu = fit_leace_rank1(X_tr.to(device), train_gen.to(device))
            P = P.cpu(); mu = mu.cpu()
            X_te_leace = apply_leace(P, mu, X_te)
            r2_leace = population_linear_r2(X_te_leace, dev_gen)

            qleace_model = fit_qleace(X_tr, train_gen, alpha=args.shrinkage_alpha)
            X_te_qleace = apply_qleace(qleace_model, X_te, dev_gen)
            r2_qleace = population_linear_r2(X_te_qleace, dev_gen)

            try:
                bal_acc_qleace = sklearn_lr_balanced_acc(
                    apply_qleace(qleace_model, X_tr, train_gen), train_gen,
                    X_te_qleace, dev_gen,
                )
            except Exception as e:
                print(f"[probe-acc-error] {e}", flush=True)
                bal_acc_qleace = float("nan")

            payload["qleace"][f"layer_{L}"] = {
                "vanilla_R2": r2_vanilla,
                "leace_rank1_R2": r2_leace,
                "qleace_R2": r2_qleace,
                "qleace_balanced_acc": bal_acc_qleace,
                "n_bar_iters": qleace_model["n_bar_iters"],
                "n_train": int(X_tr.shape[0]),
                "n_eval": int(X_te.shape[0]),
                "wall_seconds": float(time.time() - t0),
            }
            print(f"[main] layer {L}: vanilla={r2_vanilla:.3f} leace1={r2_leace:.3f} "
                  f"qleace={r2_qleace:.3f} bal_acc={bal_acc_qleace:.3f} "
                  f"@ {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[layer-error L={L}] {e}\n{traceback.format_exc()}", flush=True)
            payload["qleace"][f"layer_{L}"] = {
                "error": str(e),
                "vanilla_R2": float("nan"),
                "leace_rank1_R2": float("nan"),
                "qleace_R2": float("nan"),
            }

        # Persist + push after each layer
        payload["qleace"]["wall_time_seconds"] = float(time.time() - t_start)
        payload["qleace"]["aws_cost_estimate_usd"] = (
            (time.time() - t_start) / 3600.0 * args.cost_per_hour
        )
        json_path.write_text(json.dumps(payload, indent=2, default=float))
        write_headline(headline_path, payload)
        if s3_prefix:
            s3_put(json_path, f"{s3_prefix}/qleace_results.json")
            s3_put(headline_path, f"{s3_prefix}/qleace_HEADLINE.txt")

    # ------------------------------------------------------------------
    # Stage 4: write DONE
    # ------------------------------------------------------------------
    done_path.write_text("done\n")
    if s3_prefix:
        s3_put(done_path, f"{s3_prefix}/QLEACE_DONE")
        s3_put(json_path, f"{s3_prefix}/qleace_results.json")
        s3_put(headline_path, f"{s3_prefix}/qleace_HEADLINE.txt")

    print(f"[main] DONE @ {time.time()-t_start:.1f}s", flush=True)
    return 0


def write_failed_marker(out_dir: Path, msg: str) -> None:
    fp = out_dir / "QLEACE_FAILED"
    fp.write_text(msg + "\n")
    print(f"[FAILED] {msg}", flush=True)


if __name__ == "__main__":
    try:
        rc = main()
        sys.exit(rc)
    except Exception as e:
        print(f"[fatal] {e}\n{traceback.format_exc()}", flush=True)
        sys.exit(1)
