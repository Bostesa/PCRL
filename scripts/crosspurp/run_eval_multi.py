"""Cross-purpose constraint eval — multi-dataset.

Generalises the Adult-only ``scripts/crosspurp/run_eval.py`` to all three
tabular datasets (adult, hmda, diabetes). For each checkpoint:
  - per-pair R²(h_p, A) on test split (auditor metric, τ=0.05)
  - h_concat R²(h_concat, A) on test (cross-purpose constraint metric, τ=0.10)
  - cross-purpose attack accuracy on h_concat via LR / MLP / XGB
    (matches §5.4 audit pipeline; majority-baseline delta in pp)

Outputs:
  results/v2_<dataset>_<TAG>/results.json
  results/v2_<dataset>_<TAG>/HEADLINE.txt

The cross-purpose attribute list is dataset-specific; defaults match the
rebuttal pilot plan. Override with --cross-attrs.

Usage:
  python scripts/crosspurp/run_eval_multi.py --dataset adult --tag CROSSPURP
  python scripts/crosspurp/run_eval_multi.py --dataset hmda --tag CROSSPURP_ERASE \\
      --cross-attrs ethnicity race sex
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.models.encoder import StandardEncoder  # noqa: E402
from pcrl.models.lora import PerPurposeLoRAEncoder  # noqa: E402

DATASET_CONFIG: dict[str, dict] = {
    "adult": {
        "loader_module": "pcrl.data.adult",
        "dataset_class": "AdultDataset",
        "purposes_fn": "get_adult_purposes",
        "lora_rank": 8, "lora_alpha": 16.0,
        "default_cross_attrs": ["race", "sex", "age_group"],
        "data_root": "data",
    },
    "hmda": {
        "loader_module": "pcrl.data.hmda",
        "dataset_class": "HMDADataset",
        "purposes_fn": "get_hmda_purposes",
        "lora_rank": 8, "lora_alpha": 16.0,
        "default_cross_attrs": ["ethnicity", "race", "sex"],
        "data_root": "data",
    },
    "diabetes": {
        "loader_module": "pcrl.data.diabetes",
        "dataset_class": "DiabetesDataset",
        "purposes_fn": "get_diabetes_purposes",
        "lora_rank": 24, "lora_alpha": 48.0,
        "default_cross_attrs": ["race", "gender", "age_bucket"],
        "data_root": "data",
    },
}

TAU = 0.05
TAU_CROSS = 0.10
SEEDS_DEFAULT = [0, 1, 2]


def import_dataset_pieces(dataset: str):
    cfg = DATASET_CONFIG[dataset]
    mod = __import__(cfg["loader_module"], fromlist=[cfg["dataset_class"], cfg["purposes_fn"]])
    ds_class = getattr(mod, cfg["dataset_class"])
    purposes_fn = getattr(mod, cfg["purposes_fn"])
    return ds_class, purposes_fn, cfg


def build_loaders(dataset: str, batch_size: int = 512):
    DSCls, purposes_fn, cfg = import_dataset_pieces(dataset)
    purposes = purposes_fn()
    root = str(ROOT / cfg["data_root"])
    train_ds = DSCls(purposes=purposes, root=root, split="train", download=False)
    test_ds = DSCls(
        purposes=purposes, root=root, split="test", download=False,
        norm_stats=train_ds.norm_stats,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)
    return purposes, train_ds, test_ds, train_loader, test_loader, cfg


def _detect_lora_target(lora_adapters_sd: dict) -> str:
    """Infer lora_target from a checkpoint's lora_adapters state_dict.

    Auto-detection over an explicit metadata file because (a) it works
    against existing checkpoints with no trainer changes, and (b) the
    state_dict key shape is structurally invariant under the lora_target
    choice — it's a property of the encoder construction, not of training.

    The PerPurposeLoRAEncoder.adapters is a ModuleList[purpose] of
    ModuleList[per-Linear-in-backbone]. Keys look like:
      <purpose_idx>.<linear_idx>.{A.weight,B.weight,bias}
    With ``lora_target="all_linear"`` the StandardEncoder backbone has 3
    Linear layers (in→128, 128→128, 128→64), so per-purpose linear_idx
    ranges over {0,1,2}. With ``lora_target="repr_proj_only"`` only the
    final Linear gets an adapter, so linear_idx is always 0.

    Detection: presence of any key with linear_idx ∈ {1,2} → "all_linear";
    otherwise → "repr_proj_only". Failure mode if wrong is loud (size
    mismatch or missing-key RuntimeError from load_state_dict), not silent
    corruption — so an incorrect inference cannot quietly produce bad
    eval numbers.

    Yesterday (2026-05-30) the AB eval crashed exactly here because the
    eval script always built the encoder with the default ``all_linear``
    while training used ``repr_proj_only``; this helper closes that gap.
    """
    for key in lora_adapters_sd.keys():
        m = re.match(r"^\d+\.([12])\.", key)
        if m is not None:
            return "all_linear"
    return "repr_proj_only"


def load_encoder(ckpt_path: Path, input_dim: int, n_purposes: int, cfg: dict) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lora_target = _detect_lora_target(ckpt["lora_adapters"])
    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
        use_erase_layer=True,  # tolerated even if checkpoint was trained without
    )
    encoder = PerPurposeLoRAEncoder(
        backbone, n_purposes=n_purposes,
        rank=cfg["lora_rank"], alpha=cfg["lora_alpha"], dropout=0.0,
        lora_target=lora_target,
    )
    encoder.backbone.load_state_dict(ckpt["backbone"], strict=False)
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
            h = encoder(batch["features"], purpose_idx)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG))
    ap.add_argument("--tag", required=True,
                    help="Checkpoint dir tag, e.g. 'CROSSPURP' for v2_<dataset>_CROSSPURP_s{seed}/")
    ap.add_argument("--cross-attrs", nargs="*", default=None,
                    help="Override the dataset's default cross-purpose attribute list")
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS_DEFAULT)
    ap.add_argument("--out-tag", default=None,
                    help="Output dir tag (default: same as --tag)")
    args = ap.parse_args()

    out_tag = args.out_tag or args.tag
    purposes, train_ds, test_ds, train_loader, test_loader, cfg = build_loaders(args.dataset)
    purpose_names = [p.name for p in purposes]
    # ATTRS = union of all disallowed_attrs across purposes (per-pair audit pool)
    ATTRS = sorted({a for p in purposes for a in p.disallowed_attrs})
    cross_attrs = args.cross_attrs or cfg["default_cross_attrs"]
    print(f"DATASET={args.dataset}  N_train={len(train_ds)}  N_test={len(test_ds)}  D={train_ds.info.num_features}")
    print(f"purposes: {purpose_names}")
    print(f"ATTRS (per-pair audit pool): {ATTRS}")
    print(f"cross-purpose attrs (h_concat audit): {cross_attrs}")

    train_attrs = {a: extract_attr(train_loader, a) for a in ATTRS}
    test_attrs = {a: extract_attr(test_loader, a) for a in ATTRS}
    majority = {a: float(np.unique(test_attrs[a], return_counts=True)[1].max() / len(test_attrs[a]))
                for a in ATTRS}

    results = {
        "dataset": args.dataset, "tag": args.tag, "tau": TAU, "tau_cross": TAU_CROSS,
        "purpose_names": purpose_names, "attrs": ATTRS, "cross_attrs": cross_attrs,
        "per_seed": [], "summary": {},
    }
    CKPT_BASE = ROOT / "checkpoints"

    for seed in args.seeds:
        ckpt_dir = CKPT_BASE / f"v2_{args.dataset}_{args.tag}_s{seed}"
        candidates = ["canonical_iterate.pt", "best.pt", "final.pt"]
        ckpt_path = next((ckpt_dir / n for n in candidates if (ckpt_dir / n).exists()), None)
        if ckpt_path is None:
            print(f"[seed {seed}] no checkpoint at {ckpt_dir}, skipping")
            continue
        print(f"\n[seed {seed}] loading {ckpt_path.name}")
        encoder = load_encoder(ckpt_path, train_ds.info.num_features, len(purposes), cfg)

        H_tr_per = {p: extract_reps(encoder, train_loader, i) for i, p in enumerate(purpose_names)}
        H_te_per = {p: extract_reps(encoder, test_loader, i) for i, p in enumerate(purpose_names)}
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
            print(f"  per-pair {purpose}/{attr:<16}: R²={r2:.4f} {'PASS' if r2<=TAU else 'FAIL'}")

        for attr in cross_attrs:
            K = int(max(train_attrs[attr].max(), test_attrs[attr].max())) + 1
            r2 = linear_r2_test(H_tr_concat, train_attrs[attr],
                                H_te_concat, test_attrs[attr], K)
            seed_rec["concat"][attr] = {
                "r2_onehot_test": r2, "tau_cross": TAU_CROSS, "pass": r2 <= TAU_CROSS,
            }
            print(f"  concat[{attr:<14}]: R²={r2:.4f} {'PASS' if r2<=TAU_CROSS else 'FAIL'}")

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
                print(f"  attack {arch:>3} {attr:<16}: acc={acc:.4f} (Δ{d:+.2f}pp) "
                      f"{'FLAG' if d>1.0 else 'ok'}")

        results["per_seed"].append(seed_rec)

    n_seeds = len(results["per_seed"])
    if n_seeds > 0:
        agg_concat = {}
        for attr in cross_attrs:
            r2s = [s["concat"][attr]["r2_onehot_test"]
                   for s in results["per_seed"] if attr in s["concat"]]
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

    out_dir = ROOT / "results" / f"v2_{args.dataset}_{out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "results.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2,
                  default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"\nWrote {out}")
    # Compact HEADLINE.txt
    s = results.get("summary", {})
    lines = [
        f"CROSSPURP_EVAL — {args.dataset.upper()} tag={args.tag} n_seeds={s.get('n_seeds', 0)}",
        f"per-pair R² ≤ {TAU}: {s.get('per_pair_pass', '?')}",
    ]
    for a, v in (s.get("concat_r2_mean") or {}).items():
        lines.append(f"h_concat R²[{a}]: mean={v['mean_r2']:.4f} max={v['max_r2']:.4f} pass(≤{TAU_CROSS}) {v['n_pass']}/{v['n']}")
    lines.append(f"attack flags >1pp (mean Δ over seeds): {s.get('attack_flagged_above_1pp', '?')}")
    (out_dir / "HEADLINE.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_dir / 'HEADLINE.txt'}")


if __name__ == "__main__":
    main()
