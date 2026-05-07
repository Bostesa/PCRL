"""Cross-purpose constraint eval: load each seed's checkpoint, extract
per-purpose reprs, compute per-pair R² + h_concat R² + cross-purpose
attack (LR/MLP/XGB) on Adult test split. Compare to baseline R5."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.adult import AdultDataset, get_adult_purposes  # noqa: E402
from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.models.encoder import StandardEncoder  # noqa: E402
from pcrl.models.lora import PerPurposeLoRAEncoder  # noqa: E402

CKPT_BASE = ROOT / "checkpoints"
SEEDS = [0, 1, 2]
ATTRS = ["race", "sex", "age_group", "marital_status", "income"]
CROSS_ATTRS = ["race", "sex", "age_group"]
TAU = 0.05
TAU_CROSS = 0.10


def load_encoder(ckpt_path: Path, input_dim: int, n_purposes: int) -> torch.nn.Module:
    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
    )
    encoder = PerPurposeLoRAEncoder(
        backbone, n_purposes=n_purposes,
        rank=8, alpha=16.0, dropout=0.0,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder.backbone.load_state_dict(ckpt["backbone"])
    encoder.adapters.load_state_dict(ckpt["lora_adapters"])
    enc_buf = ckpt.get("encoder_buffers", {}) or {}
    for p_idx in range(n_purposes):
        P_key = f"leace_P_p{p_idx}"
        mu_key = f"leace_mu_p{p_idx}"
        if P_key in enc_buf and mu_key in enc_buf:
            encoder.set_leace_projection(p_idx, enc_buf[P_key], enc_buf[mu_key])
    encoder.eval()
    return encoder


def extract_reps(encoder, loader, purpose_idx: int) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"]
            h = encoder(x, purpose_idx)
            chunks.append(h.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def extract_attr(loader, attr_name: str) -> np.ndarray:
    out = []
    for batch in loader:
        out.append(batch["sensitive_attrs"][attr_name].numpy())
    return np.concatenate(out, axis=0).astype(np.int64)


def linear_r2_test(H_tr, y_tr, H_te, y_te, num_classes):
    """Closed-form Tikhonov OLS R² on test split (auditor metric)."""
    y_tr_oh = np.eye(num_classes)[y_tr].astype(np.float32)
    y_te_oh = np.eye(num_classes)[y_te].astype(np.float32)
    H_tr_c = H_tr - H_tr.mean(0, keepdims=True)
    H_te_c = H_te - H_te.mean(0, keepdims=True)
    Z_tr_c = y_tr_oh - y_tr_oh.mean(0, keepdims=True)
    Z_te_c = y_te_oh - y_te_oh.mean(0, keepdims=True)
    d = H_tr.shape[1]
    # Bump reg to handle rank-deficient h_concat (cross-purpose constraint
    # may collapse some dims). Fall back to lstsq if still singular.
    gram = H_tr_c.T @ H_tr_c + 1e-4 * np.eye(d, dtype=np.float32)
    rhs = H_tr_c.T @ Z_tr_c
    try:
        W = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        W, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
    Z_pred = H_te_c @ W
    ss_res = ((Z_te_c - Z_pred) ** 2).sum()
    ss_tot = (Z_te_c ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def attack_acc(X_tr, y_tr, X_te, y_te, arch: str, seed: int) -> float:
    if arch == "LR":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
        clf.fit(X_tr, y_tr); return float(clf.score(X_te, y_te))
    if arch == "MLP":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 256), activation="relu",
            alpha=1e-4, solver="adam", learning_rate_init=1e-3,
            max_iter=100, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=8, random_state=seed, batch_size=256,
        )
        clf.fit(X_tr, y_tr); return float(clf.score(X_te, y_te))
    if arch == "XGB":
        from xgboost import XGBClassifier
        n_classes = int(max(y_tr.max(), y_te.max()) + 1)
        obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
        clf = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.3,
            objective=obj, tree_method="hist",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=seed, n_jobs=2, verbosity=0,
            use_label_encoder=False,
        )
        clf.fit(X_tr, y_tr); return float(clf.score(X_te, y_te))
    raise ValueError(arch)


def best_attack(X_tr, y_tr, X_te, y_te, arch: str) -> float:
    return max(attack_acc(X_tr, y_tr, X_te, y_te, arch, s) for s in [11, 22, 33])


def main():
    purposes = get_adult_purposes()
    train_ds = AdultDataset(purposes=purposes, root=str(ROOT / "data"),
                            split="train", download=False)
    test_ds = AdultDataset(purposes=purposes, root=str(ROOT / "data"),
                           split="test", download=False,
                           norm_stats=train_ds.norm_stats)
    print(f"N_train={len(train_ds)} N_test={len(test_ds)} D={train_ds.info.num_features}")

    BS = 512
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=False,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)
    purpose_names = [p.name for p in purposes]

    train_attrs = {a: extract_attr(train_loader, a) for a in ATTRS}
    test_attrs = {a: extract_attr(test_loader, a) for a in ATTRS}
    majority = {}
    for a in ATTRS:
        _, c = np.unique(test_attrs[a], return_counts=True)
        majority[a] = float(c.max() / len(test_attrs[a]))

    results = {"per_seed": [], "summary": {}}
    for seed in SEEDS:
        ckpt_dir = CKPT_BASE / f"v2_adult_CROSSPURP_s{seed}"
        candidates = ["canonical_iterate.pt", "best.pt", "final.pt"]
        ckpt_path = None
        for name in candidates:
            if (ckpt_dir / name).exists():
                ckpt_path = ckpt_dir / name
                break
        if ckpt_path is None:
            print(f"[seed {seed}] no checkpoint at {ckpt_dir}, skipping")
            continue
        print(f"[seed {seed}] loading {ckpt_path.name}")
        encoder = load_encoder(ckpt_path, train_ds.info.num_features, len(purposes))

        H_tr_per = {}
        H_te_per = {}
        for idx, pname in enumerate(purpose_names):
            H_tr_per[pname] = extract_reps(encoder, train_loader, idx)
            H_te_per[pname] = extract_reps(encoder, test_loader, idx)
        H_tr_concat = np.concatenate([H_tr_per[p] for p in purpose_names], axis=1)
        H_te_concat = np.concatenate([H_te_per[p] for p in purpose_names], axis=1)
        print(f"  concat shape: train {H_tr_concat.shape} test {H_te_concat.shape}")

        seed_rec = {"seed": seed, "checkpoint": ckpt_path.name,
                    "per_pair": {}, "concat": {}, "attack": {}}

        for purpose, attr in [(p.name, a) for p in purposes for a in p.disallowed_attrs]:
            K = int(max(train_attrs[attr].max(), test_attrs[attr].max())) + 1
            r2 = linear_r2_test(H_tr_per[purpose], train_attrs[attr],
                                H_te_per[purpose], test_attrs[attr], K)
            seed_rec["per_pair"][f"{purpose}/{attr}"] = {
                "r2_onehot_test": r2, "tau": TAU, "pass": r2 <= TAU,
            }
            print(f"  per-pair {purpose}/{attr}: R²={r2:.4f} {'PASS' if r2<=TAU else 'FAIL'}")

        for attr in CROSS_ATTRS:
            K = int(max(train_attrs[attr].max(), test_attrs[attr].max())) + 1
            r2 = linear_r2_test(H_tr_concat, train_attrs[attr],
                                H_te_concat, test_attrs[attr], K)
            seed_rec["concat"][attr] = {
                "r2_onehot_test": r2, "tau_cross": TAU_CROSS, "pass": r2 <= TAU_CROSS,
            }
            print(f"  concat[{attr}]: R²={r2:.4f} {'PASS' if r2<=TAU_CROSS else 'FAIL'}")

        for arch in ["LR", "MLP", "XGB"]:
            seed_rec["attack"][arch] = {}
            for attr in ATTRS:
                t0 = time.time()
                acc = best_attack(
                    H_tr_concat, train_attrs[attr],
                    H_te_concat, test_attrs[attr], arch,
                )
                d = (acc - majority[attr]) * 100
                seed_rec["attack"][arch][attr] = {
                    "attack_acc": acc, "majority": majority[attr],
                    "delta_pp": d, "flag_1pp": d > 1.0,
                    "elapsed_s": time.time() - t0,
                }
                print(f"  attack {arch:>3} {attr:<15}: acc={acc:.4f} (Δ{d:+.2f}pp) "
                      f"{'FLAG' if d>1.0 else 'ok'}")

        results["per_seed"].append(seed_rec)

    n_seeds = len(results["per_seed"])
    if n_seeds > 0:
        agg_concat = {}
        for attr in CROSS_ATTRS:
            r2s = [s["concat"][attr]["r2_onehot_test"] for s in results["per_seed"]
                   if attr in s["concat"]]
            agg_concat[attr] = {
                "mean_r2": float(np.mean(r2s)), "max_r2": float(np.max(r2s)),
                "n_pass": sum(1 for r in r2s if r <= TAU_CROSS), "n": len(r2s),
            }
        agg_attack = {}
        for arch in ["LR", "MLP", "XGB"]:
            agg_attack[arch] = {}
            for attr in ATTRS:
                deltas = [s["attack"][arch][attr]["delta_pp"]
                          for s in results["per_seed"]
                          if attr in s.get("attack", {}).get(arch, {})]
                if deltas:
                    agg_attack[arch][attr] = {
                        "mean_delta_pp": float(np.mean(deltas)),
                        "flag_1pp_mean": float(np.mean(deltas)) > 1.0,
                    }
        flagged = sum(1 for arch in agg_attack for a in agg_attack[arch]
                      if agg_attack[arch][a]["flag_1pp_mean"])
        total = sum(1 for arch in agg_attack for a in agg_attack[arch])
        per_pair_pass = sum(1 for s in results["per_seed"]
                            for v in s["per_pair"].values() if v["pass"])
        per_pair_total = sum(len(s["per_pair"]) for s in results["per_seed"])
        results["summary"] = {
            "n_seeds": n_seeds,
            "concat_r2_mean": agg_concat,
            "attack_flagged_above_1pp": f"{flagged}/{total}",
            "attack_per_arch_per_attr": agg_attack,
            "per_pair_pass": f"{per_pair_pass}/{per_pair_total}",
        }

    out = ROOT / "results" / "v2_adult_CROSSPURP" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2,
                  default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
