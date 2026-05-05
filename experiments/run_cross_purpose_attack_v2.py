#!/usr/bin/env python3
"""Cross-purpose concatenation attack on canonical V2 checkpoints.

For each dataset (Adult/HMDA/Diabetes), each PCRL seed (0/1/2):

  1. Load the canonical final.pt checkpoint (StandardEncoder backbone +
     PerPurposeLoRAEncoder adapters trained on the v2 pipeline).
  2. Extract per-purpose representations h_p1(x), h_p2(x), h_p3(x) on
     train+test splits.
  3. Build the concatenated representation
     ``h_concat(x) = [h_p1(x) | h_p2(x) | h_p3(x)]``  (R^{3*64} = R^{192}).
  4. For every disallowed attribute that appears in any purpose's
     ``disallowed_attrs`` set, train three auditor architectures
     (LogisticRegression, MLP, XGBoost) on the concatenation -> attribute.
     Each auditor architecture is retrained under three random seeds and
     we report the **best** of the three auditor seeds (worst-case
     attacker for the defender).
  5. As a baseline, run the same three auditors on each *single*
     purpose's representation, take the per-architecture max accuracy
     across the three purposes.

Aggregation is done per (dataset, attribute, architecture, source) by
mean across the three PCRL seeds. A per-row verdict is added: PASS if
``concat - best_single <= 1pp``, otherwise FLAG.

Outputs:
  - results/v2_cross_purpose/per_seed_results.json    (raw per-PCRL-seed)
  - results/v2_cross_purpose/aggregate.json           (mean+std)
  - results/V2_CROSS_PURPOSE_ATTACK.md                (human-readable)
  - paper-body/tables/cross_purpose_attack.tex        (LaTeX snippet)
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

from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.models.encoder import StandardEncoder  # noqa: E402
from pcrl.models.lora import PerPurposeLoRAEncoder  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("cross_purpose_v2")

# Silence noisy sklearn / xgboost convergence warnings
import warnings  # noqa: E402
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
import os  # noqa: E402
os.environ["PYTHONWARNINGS"] = "ignore"
try:
    from sklearn.exceptions import ConvergenceWarning  # noqa: E402
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
PCRL_SEEDS = [0, 1, 2]
AUDITOR_SEEDS = [11, 22, 33]
REPR_DIM = 64
HIDDEN_DIMS = [128, 128]
DROPOUT = 0.3
BATCH_SIZE = 512

DATASETS = {
    "adult": {
        "ckpt_pattern": "checkpoints/v2_adult_ROUND5_s{seed}/final.pt",
        "lora_rank": 8,
        "lora_alpha": 16.0,
    },
    "hmda": {
        "ckpt_pattern": "checkpoints/v2_hmda_ROUND5_s{seed}/final.pt",
        "lora_rank": 8,
        "lora_alpha": 16.0,
    },
    "diabetes": {
        "ckpt_pattern": "checkpoints/v2_diabetes_ROUND7_s{seed}/final.pt",
        "lora_rank": 24,
        "lora_alpha": 48.0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data and model loading
# ─────────────────────────────────────────────────────────────────────────────
def build_datasets(name: str):
    if name == "adult":
        from pcrl.data.adult import AdultDataset, get_adult_purposes
        purposes = get_adult_purposes()
        train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
        test_ds = AdultDataset(
            purposes=purposes, root="data", split="test", download=False,
            norm_stats=train_ds.norm_stats,
        )
    elif name == "diabetes":
        from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
        purposes = get_diabetes_purposes()
        train_ds = DiabetesDataset(purposes=purposes, split="train")
        test_ds = DiabetesDataset(purposes=purposes, split="test")
    elif name == "hmda":
        from pcrl.data.hmda import HMDADataset, get_hmda_purposes
        purposes = get_hmda_purposes()
        train_ds = HMDADataset(purposes=purposes, root="data", split="train")
        test_ds = HMDADataset(purposes=purposes, root="data", split="test")
    else:
        raise ValueError(f"unknown dataset {name}")
    return purposes, train_ds, test_ds


def load_encoder(ckpt_path: Path, n_purposes: int, input_dim: int,
                 lora_rank: int, lora_alpha: float, device: str) -> PerPurposeLoRAEncoder:
    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM, dropout=DROPOUT,
    )
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=n_purposes,
        rank=lora_rank, alpha=lora_alpha, dropout=0.0,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.backbone.load_state_dict(ckpt["backbone"])
    encoder.adapters.load_state_dict(ckpt["lora_adapters"])
    encoder.to(device).eval()
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# Representation + label extraction
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_reps(encoder: PerPurposeLoRAEncoder, loader: DataLoader,
                 purpose_idx: int, device: str) -> np.ndarray:
    out = []
    for batch in loader:
        h = encoder(batch["features"].to(device), purpose_idx)
        out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0)


def extract_attr(loader: DataLoader, attr_name: str) -> np.ndarray:
    out = []
    for batch in loader:
        out.append(batch["sensitive_attrs"][attr_name].numpy())
    return np.concatenate(out, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Auditor training
# ─────────────────────────────────────────────────────────────────────────────
def train_lr(X_tr, y_tr, X_te, y_te, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=2000, random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def train_mlp(X_tr, y_tr, X_te, y_te, seed: int) -> float:
    """sklearn MLP (the spec's MLP w/ dropout 0.3 isn't available in
    sklearn; we use early_stopping as a regulariser substitute and cap at
    100 max_iter which the spec mandates)."""
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 256),
        activation="relu",
        alpha=1e-4,
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=seed,
        batch_size=256,
    )
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def train_xgb(X_tr, y_tr, X_te, y_te, seed: int) -> float:
    from xgboost import XGBClassifier
    n_classes = int(max(y_tr.max(), y_te.max()) + 1)
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,  # XGBoost default
        objective=objective,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        use_label_encoder=False,
    )
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


AUDITORS = {"LR": train_lr, "MLP": train_mlp, "XGB": train_xgb}


# ─────────────────────────────────────────────────────────────────────────────
# Extra auditor architectures (defensive depth: RF / RBF SVM / Deep MLP)
# Enabled with --extra-architectures.  Same protocol as AUDITORS — best of
# AUDITOR_SEEDS auditor seeds, mean ± std across PCRL seeds.
# ─────────────────────────────────────────────────────────────────────────────
def train_rf(X_tr, y_tr, X_te, y_te, seed: int) -> float:
    """Random Forest: 200 trees, max depth 10, all CPU cores."""
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, n_jobs=-1, random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def train_svm_rbf(X_tr, y_tr, X_te, y_te, seed: int) -> float:
    """RBF SVM: C=1.0, gamma='scale'.  Subsample train to 10k rows when
    N>10k since SVC is O(n^2) in memory and O(n^2)–O(n^3) in time."""
    from sklearn.svm import SVC
    if len(X_tr) > 10_000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_tr), size=10_000, replace=False)
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]
    clf = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


class _DeepMLP(torch.nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_d = input_dim
        for h in (256, 256, 128, 128):
            layers.append(torch.nn.Linear(in_d, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.3))
            in_d = h
        layers.append(torch.nn.Linear(in_d, n_classes))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_deep_mlp(X_tr, y_tr, X_te, y_te, seed: int) -> float:
    """Deep MLP: hidden [256,256,128,128], ReLU, dropout 0.3, Adam lr 1e-3,
    batch 256, ≤100 epochs, early-stop patience 10 on a 10% held-out
    validation split (deterministic via seeded permutation).  Returns
    test accuracy at the best-val-acc epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = int(max(int(y_tr.max()), int(y_te.max())) + 1)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_tr))
    n_val = max(1, int(round(0.1 * len(X_tr))))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xt = torch.from_numpy(X_tr[tr_idx]).float()
    yt = torch.from_numpy(y_tr[tr_idx]).long()
    Xv = torch.from_numpy(X_tr[val_idx]).float().to(device)
    yv = torch.from_numpy(y_tr[val_idx]).long().to(device)
    Xe = torch.from_numpy(X_te).float().to(device)
    ye = torch.from_numpy(y_te).long().to(device)

    model = _DeepMLP(X_tr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    g = torch.Generator()
    g.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt),
        batch_size=256, shuffle=True, generator=g,
    )

    best_val = -1.0
    best_state: dict | None = None
    bad = 0
    PATIENCE = 10
    MAX_EPOCHS = 100
    for _epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_acc = float((model(Xv).argmax(dim=1) == yv).float().mean().item())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        test_acc = float((model(Xe).argmax(dim=1) == ye).float().mean().item())
    return test_acc


AUDITORS_EXTRA = {"RF": train_rf, "SVM": train_svm_rbf, "DeepMLP": train_deep_mlp}


def best_across_auditor_seeds(fn, X_tr, y_tr, X_te, y_te) -> float:
    best = 0.0
    for s in AUDITOR_SEEDS:
        try:
            acc = fn(X_tr, y_tr, X_te, y_te, s)
        except Exception as exc:
            log.warning(f"auditor {fn.__name__} seed={s} raised {exc!r}; treating as 0.0")
            acc = 0.0
        if acc > best:
            best = acc
    return best


# ─────────────────────────────────────────────────────────────────────────────
# One PCRL seed
# ─────────────────────────────────────────────────────────────────────────────
def run_one_seed(name: str, seed: int, device: str,
                 auditors: dict = AUDITORS) -> dict:
    cfg = DATASETS[name]
    ckpt_path = ROOT / cfg["ckpt_pattern"].format(seed=seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    purposes, train_ds, test_ds = build_datasets(name)
    purpose_names = [p.name for p in purposes]
    input_dim = train_ds.info.num_features

    log.info(f"[{name}/s{seed}] N_train={len(train_ds)} N_test={len(test_ds)} D={input_dim}")
    log.info(f"[{name}/s{seed}] purposes={purpose_names}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)

    encoder = load_encoder(
        ckpt_path, n_purposes=len(purposes), input_dim=input_dim,
        lora_rank=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        device=device,
    )

    # Extract per-purpose representations
    h_train: dict[str, np.ndarray] = {}
    h_test: dict[str, np.ndarray] = {}
    for idx, pname in enumerate(purpose_names):
        h_train[pname] = extract_reps(encoder, train_loader, idx, device)
        h_test[pname] = extract_reps(encoder, test_loader, idx, device)
    log.info(f"[{name}/s{seed}] per-purpose train shape: {h_train[purpose_names[0]].shape}")

    # Concat
    h_train_concat = np.concatenate([h_train[p] for p in purpose_names], axis=1)
    h_test_concat = np.concatenate([h_test[p] for p in purpose_names], axis=1)
    log.info(f"[{name}/s{seed}] concat shape: train {h_train_concat.shape} test {h_test_concat.shape}")

    # Union of disallowed attrs
    all_attrs = sorted({a for p in purposes for a in p.disallowed_attrs})

    # Labels + majority baseline (computed on test split — defines the
    # threshold the attacker must beat)
    y_train: dict[str, np.ndarray] = {}
    y_test: dict[str, np.ndarray] = {}
    majority: dict[str, float] = {}
    for attr in all_attrs:
        y_train[attr] = extract_attr(train_loader, attr).astype(np.int64)
        y_test[attr] = extract_attr(test_loader, attr).astype(np.int64)
        _, counts = np.unique(y_test[attr], return_counts=True)
        majority[attr] = float(counts.max() / len(y_test[attr]))

    log.info(f"[{name}/s{seed}] auditing attributes: {all_attrs}")

    # Per-attribute, per-architecture: concat acc + best single-purpose acc
    rows = []
    for attr in all_attrs:
        log.info(f"[{name}/s{seed}] attr={attr} (majority={majority[attr]:.3f})")
        for arch_name, fn in auditors.items():
            t0 = time.time()
            # Concat
            concat_acc = best_across_auditor_seeds(
                fn, h_train_concat, y_train[attr], h_test_concat, y_test[attr],
            )
            # Best single purpose (per architecture)
            single_per_purpose = {}
            for pname in purpose_names:
                acc = best_across_auditor_seeds(
                    fn, h_train[pname], y_train[attr], h_test[pname], y_test[attr],
                )
                single_per_purpose[pname] = acc
            best_single_purpose = max(single_per_purpose, key=single_per_purpose.get)
            best_single_acc = single_per_purpose[best_single_purpose]

            elapsed = time.time() - t0
            log.info(
                f"  {arch_name}: concat={concat_acc:.3f} (Δ{concat_acc-majority[attr]:+.3f}) "
                f"best_single={best_single_acc:.3f} ({best_single_purpose}, "
                f"Δ{best_single_acc-majority[attr]:+.3f}) "
                f"gain={concat_acc-best_single_acc:+.3f} [{elapsed:.1f}s]"
            )

            rows.append({
                "attribute": attr,
                "arch": arch_name,
                "majority": majority[attr],
                "concat_acc": concat_acc,
                "concat_delta": concat_acc - majority[attr],
                "best_single_acc": best_single_acc,
                "best_single_delta": best_single_acc - majority[attr],
                "best_single_purpose": best_single_purpose,
                "concat_gain_over_best_single": concat_acc - best_single_acc,
                "single_per_purpose": single_per_purpose,
            })

    return {
        "dataset": name,
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "purposes": purpose_names,
        "n_train": int(len(train_ds)),
        "n_test": int(len(test_ds)),
        "concat_dim": int(h_train_concat.shape[1]),
        "rows": rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation across PCRL seeds
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(per_seed: list[dict]) -> dict:
    """Group rows by (dataset, attribute, arch); compute mean+std across seeds."""
    by_key: dict[tuple, list[dict]] = {}
    for entry in per_seed:
        for r in entry["rows"]:
            key = (entry["dataset"], r["attribute"], r["arch"])
            by_key.setdefault(key, []).append({**r, "seed": entry["seed"]})

    out: dict[str, list] = {}
    for (dataset, attr, arch), rs in sorted(by_key.items()):
        majority = float(np.mean([x["majority"] for x in rs]))
        concat_accs = np.array([x["concat_acc"] for x in rs])
        single_accs = np.array([x["best_single_acc"] for x in rs])
        gain_pp = (concat_accs - single_accs) * 100.0  # percentage points

        agg_row = {
            "dataset": dataset,
            "attribute": attr,
            "arch": arch,
            "majority": round(majority, 4),
            "concat_acc_mean": float(concat_accs.mean()),
            "concat_acc_std": float(concat_accs.std(ddof=0)),
            "concat_delta_mean_pp": float(((concat_accs - majority).mean()) * 100),
            "concat_delta_std_pp": float(((concat_accs - majority).std(ddof=0)) * 100),
            "single_acc_mean": float(single_accs.mean()),
            "single_acc_std": float(single_accs.std(ddof=0)),
            "single_delta_mean_pp": float(((single_accs - majority).mean()) * 100),
            "single_delta_std_pp": float(((single_accs - majority).std(ddof=0)) * 100),
            "gain_mean_pp": float(gain_pp.mean()),
            "gain_std_pp": float(gain_pp.std(ddof=0)),
            "verdict": "PASS" if gain_pp.mean() <= 1.0 else "FLAG",
        }
        out.setdefault(dataset, []).append(agg_row)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Markdown + LaTeX writers
# ─────────────────────────────────────────────────────────────────────────────
def write_markdown(agg: dict, out_path: Path, per_seed: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# V2 Cross-Purpose Concatenation Attack")
    lines.append("")
    lines.append("Auditors trained on the concatenation of all three purpose-specific")
    lines.append("representations (`h_concat = [h_p1 | h_p2 | h_p3]`) versus the best")
    lines.append("single-purpose baseline.  Numbers are mean ± std across 3 PCRL seeds.")
    lines.append("Each cell reports the **best of 3 auditor seeds** (worst case for the")
    lines.append("defender). Δ values are in percentage points over the test-split")
    lines.append("majority baseline. **Verdict** = PASS if `concat − best_single ≤ 1 pp`,")
    lines.append("FLAG otherwise.")
    lines.append("")
    lines.append("Canonical checkpoints:")
    for entry in per_seed:
        if entry["seed"] == 0:
            ckpt = entry["checkpoint"]
            try:
                ckpt = str(Path(ckpt).relative_to(ROOT))
            except ValueError:
                pass
            lines.append(f"- **{entry['dataset']}**: `{ckpt}` (+ s1, s2 in same family)")
    lines.append("")

    # ── Headline findings ───────────────────────────────────────────────────
    all_rows = [r for rs in agg.values() for r in rs]
    flags = [r for r in all_rows if r["verdict"] == "FLAG"]
    biggest = max(all_rows, key=lambda r: r["gain_mean_pp"])
    lines.append("## Headline findings")
    lines.append("")
    lines.append(
        f"- **{len(flags)}/{len(all_rows)} (dataset, attribute, auditor) "
        f"triples flag** under the 1 pp gain threshold."
    )
    lines.append(
        f"- **Largest concat gain**: `{biggest['dataset']}/{biggest['attribute']}/"
        f"{biggest['arch']}` at **{biggest['gain_mean_pp']:+.2f} pp** "
        f"(concat Δ {biggest['concat_delta_mean_pp']:+.2f} pp vs single "
        f"Δ {biggest['single_delta_mean_pp']:+.2f} pp)."
    )
    # Per-attribute headline (worst across archs, per dataset)
    by_attr: dict[tuple, dict] = {}
    for r in all_rows:
        key = (r["dataset"], r["attribute"])
        cur = by_attr.get(key)
        if cur is None or r["gain_mean_pp"] > cur["gain_mean_pp"]:
            by_attr[key] = r
    lines.append("- **Worst-case gain per (dataset, attribute)** (max across LR/MLP/XGB):")
    for (dataset, attr), r in sorted(by_attr.items()):
        marker = "FLAG" if r["verdict"] == "FLAG" else "PASS"
        lines.append(
            f"  - `{dataset}/{attr}` — gain **{r['gain_mean_pp']:+.2f} ± "
            f"{r['gain_std_pp']:.2f} pp** ({r['arch']}) — **{marker}**"
        )
    lines.append("")

    for dataset, rows in agg.items():
        lines.append(f"## {dataset.upper()}")
        lines.append("")
        # Group rows by attribute
        attrs = sorted({r["attribute"] for r in rows})
        lines.append(
            "| attribute | arch | majority | best_single (Δ pp) | concat (Δ pp) | "
            "gain (pp) | verdict |"
        )
        lines.append(
            "|-----------|------|----------|--------------------|---------------|"
            "-----------|---------|"
        )
        for attr in attrs:
            for arch in ("LR", "MLP", "XGB"):
                r = next((x for x in rows if x["attribute"] == attr and x["arch"] == arch), None)
                if r is None:
                    continue
                lines.append(
                    f"| {attr} | {arch} | {r['majority']*100:.2f}% | "
                    f"{r['single_delta_mean_pp']:+.2f} ± {r['single_delta_std_pp']:.2f} | "
                    f"{r['concat_delta_mean_pp']:+.2f} ± {r['concat_delta_std_pp']:.2f} | "
                    f"{r['gain_mean_pp']:+.2f} ± {r['gain_std_pp']:.2f} | "
                    f"{r['verdict']} |"
                )
        lines.append("")

        # Per-attribute summary across architectures
        lines.append("### Worst-case per attribute")
        lines.append("")
        lines.append("| attribute | worst arch | worst gain (pp) | verdict |")
        lines.append("|-----------|-----------|-----------------|---------|")
        for attr in attrs:
            attr_rows = [r for r in rows if r["attribute"] == attr]
            worst = max(attr_rows, key=lambda r: r["gain_mean_pp"])
            lines.append(
                f"| {attr} | {worst['arch']} | {worst['gain_mean_pp']:+.2f} ± "
                f"{worst['gain_std_pp']:.2f} | {worst['verdict']} |"
            )
        lines.append("")

    # Overall conclusion
    all_rows = [r for rs in agg.values() for r in rs]
    flags = [r for r in all_rows if r["verdict"] == "FLAG"]
    lines.append("## Verdict summary")
    lines.append("")
    lines.append(f"- Total (dataset, attribute, arch) triples evaluated: **{len(all_rows)}**")
    lines.append(f"- PASS: **{len(all_rows)-len(flags)}**")
    lines.append(f"- FLAG (concat gain > 1 pp over best single purpose): **{len(flags)}**")
    if flags:
        lines.append("")
        lines.append("Flagged rows (sorted by gain):")
        flags_sorted = sorted(flags, key=lambda r: -r["gain_mean_pp"])
        for r in flags_sorted:
            lines.append(
                f"- `{r['dataset']}/{r['attribute']}/{r['arch']}` — "
                f"concat Δ={r['concat_delta_mean_pp']:+.2f} pp, "
                f"single Δ={r['single_delta_mean_pp']:+.2f} pp, "
                f"gain {r['gain_mean_pp']:+.2f} ± {r['gain_std_pp']:.2f} pp"
            )
    else:
        lines.append("")
        lines.append("**No flags.** Concatenation does not recover meaningfully more")
        lines.append("information than the best single-purpose representation across")
        lines.append("any (dataset, attribute, auditor) triple.")
    lines.append("")

    out_path.write_text("\n".join(lines))


def write_latex(agg: dict, out_path: Path) -> None:
    """LaTeX snippet — emits TWO tables stacked in the same .tex file.

    1. ``tab:cross`` (drop-in replacement for the existing Adult-only
       table in S5-experiments.tex): rows = attributes, columns =
       Best-Single | Concat | Gain. Reports raw accuracy (%) like the
       paper's existing format. The 'worst' auditor across LR/MLP/XGB is
       used so the table conservatively reports the largest concat
       advantage seen.
    2. ``tab:cross_multi`` (new multi-dataset table per the user's
       request): rows = race / sex / age_group, columns = Adult / HMDA /
       Diabetes; entries = accuracy gain over majority baseline (pp,
       worst across LR/MLP/XGB), mean (std) across 3 PCRL seeds.

    Diabetes attribute aliases: sex -> gender, age_group -> age_bucket.
    '--' means the attribute does not appear in any purpose for the
    dataset and is therefore not audited.
    """
    aliases = {
        ("diabetes", "sex"): "gender",
        ("diabetes", "age_group"): "age_bucket",
    }
    archs = ("LR", "MLP", "XGB")

    def find(dataset: str, attr: str, arch: str) -> dict | None:
        canonical = aliases.get((dataset, attr), attr)
        for r in agg.get(dataset, []):
            if r["attribute"] == canonical and r["arch"] == arch:
                return r
        return None

    def worst_arch_row(dataset: str, attr: str) -> tuple[dict | None, str]:
        rs = [(arch, find(dataset, attr, arch)) for arch in archs]
        rs = [(a, r) for a, r in rs if r is not None]
        if not rs:
            return None, ""
        a, r = max(rs, key=lambda ar: ar[1]["gain_mean_pp"])
        return r, a

    lines: list[str] = []
    lines.append("% Cross-purpose concatenation attack — auto-generated by")
    lines.append("% experiments/run_cross_purpose_attack_v2.py.  Two tables:")
    lines.append("%   1) tab:cross         drop-in replacement for the existing Adult-only table")
    lines.append("%   2) tab:cross_multi   multi-dataset summary the user asked for")
    lines.append("% In both, the worst-case (highest-gain) auditor across LR / MLP /")
    lines.append("% XGB is reported so the numbers conservatively bound the attack.")
    lines.append("")

    # ── Table 1: Adult-only drop-in replacement ─────────────────────────────
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Cross-purpose attack on Adult. Best of three auditor "
                 "architectures (LR / MLP / XGBoost), max across three auditor "
                 "seeds; mean over three PCRL seeds, std in parentheses.}")
    lines.append("\\label{tab:cross}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Attribute} & \\textbf{Auditor} & "
                 "\\textbf{Best Single} & \\textbf{Concatenated} & \\textbf{Gain} \\\\")
    lines.append("\\midrule")

    adult_attrs = sorted({r["attribute"] for r in agg.get("adult", [])})
    for attr in adult_attrs:
        r, arch = worst_arch_row("adult", attr)
        if r is None:
            continue
        single_pct = r["single_acc_mean"] * 100
        single_std = r["single_acc_std"] * 100
        concat_pct = r["concat_acc_mean"] * 100
        concat_std = r["concat_acc_std"] * 100
        gain = r["gain_mean_pp"]
        gain_std = r["gain_std_pp"]
        lines.append(
            f"{attr.replace('_', ' ').capitalize()} & {arch} & "
            f"{single_pct:.1f}\\% ({single_std:.1f}) & "
            f"{concat_pct:.1f}\\% ({concat_std:.1f}) & "
            f"{gain:+.2f}\\,pp ({gain_std:.2f}) \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # ── Table 2: multi-dataset overview ─────────────────────────────────────
    multi_rows = ("race", "sex", "age_group")
    multi_cols = ("adult", "hmda", "diabetes")

    def multi_cell(dataset: str, attr: str, kind: str) -> str:
        r, _arch = worst_arch_row(dataset, attr)
        if r is None:
            return "--"
        if kind == "single":
            mean = r["single_delta_mean_pp"]
            std = r["single_delta_std_pp"]
        elif kind == "concat":
            mean = r["concat_delta_mean_pp"]
            std = r["concat_delta_std_pp"]
        elif kind == "gain":
            mean = r["gain_mean_pp"]
            std = r["gain_std_pp"]
        else:
            raise ValueError(kind)
        return f"{mean:+.2f} ({std:.2f})"

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Cross-purpose concatenation attack across all three "
                 "datasets. Rows: protected attribute "
                 "(Diabetes uses {\\em gender} for `sex' and "
                 "{\\em age\\_bucket} for `age\\_group'). Numbers are auditor "
                 "accuracy gain over the majority baseline in percentage points "
                 "(mean across 3 PCRL seeds, std in parentheses); worst across "
                 "LR / MLP / XGBoost. `--' = attribute not in any purpose's "
                 "disallowed set on that dataset.}")
    lines.append("\\label{tab:cross_multi}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append(" & \\textbf{Adult} & \\textbf{HMDA} & \\textbf{Diabetes} \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{4}{l}{\\emph{Best single-purpose auditor "
                 "($\\Delta$ over majority, pp)}} \\\\")
    for attr in multi_rows:
        cells = [multi_cell(d, attr, "single") for d in multi_cols]
        lines.append(f"\\quad {attr.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{4}{l}{\\emph{Concatenation $h_{\\text{concat}} = "
                 "[h_{p_1}\\,|\\,h_{p_2}\\,|\\,h_{p_3}]$ "
                 "($\\Delta$ over majority, pp)}} \\\\")
    for attr in multi_rows:
        cells = [multi_cell(d, attr, "concat") for d in multi_cols]
        lines.append(f"\\quad {attr.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{4}{l}{\\emph{Gain (concat $-$ best single, pp)}} \\\\")
    for attr in multi_rows:
        cells = [multi_cell(d, attr, "gain") for d in multi_cols]
        lines.append(f"\\quad {attr.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    out_path.write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_from_dir(raw_dir: Path, extra: bool = False) -> tuple[dict, list[dict]]:
    """Aggregate per-seed shards.  Base run reads ``{ds}_s{seed}.json``;
    extra-architectures run reads ``{ds}_s{seed}_extra.json``."""
    per_seed: list[dict] = []
    for path in sorted(raw_dir.glob("*.json")):
        is_extra = path.stem.endswith("_extra")
        if extra and not is_extra:
            continue
        if not extra and is_extra:
            continue
        with path.open() as f:
            per_seed.append(json.load(f))
    return aggregate(per_seed), per_seed


def print_combined_stdout_table(agg_base: dict, agg_extra: dict) -> None:
    """Print a 6-architecture comparison table to stdout: rows = (dataset,
    attribute), columns = LR / MLP / XGB / RF / SVM / DeepMLP, cells =
    gain (pp) over best single-purpose, mean only (the headline number)."""
    archs = ("LR", "MLP", "XGB", "RF", "SVM", "DeepMLP")
    print()
    print("=" * 100)
    print("COMBINED 6-ARCHITECTURE CROSS-PURPOSE ATTACK")
    print("(cells: gain pp = mean(concat - best_single) over 3 PCRL seeds; '--' = arch absent)")
    print("=" * 100)
    header = f"{'Dataset':<10}{'Attribute':<22}" + "".join(f"{a:>10}" for a in archs)
    print(header)
    print("-" * len(header))

    def find(agg: dict, dataset: str, attr: str, arch: str) -> dict | None:
        for r in agg.get(dataset, []):
            if r["attribute"] == attr and r["arch"] == arch:
                return r
        return None

    for dataset in ("adult", "hmda", "diabetes"):
        attrs = sorted(
            {r["attribute"] for r in agg_base.get(dataset, [])}
            | {r["attribute"] for r in agg_extra.get(dataset, [])}
        )
        for attr in attrs:
            row = f"{dataset.capitalize():<10}{attr:<22}"
            for arch in archs:
                src = agg_base if arch in ("LR", "MLP", "XGB") else agg_extra
                r = find(src, dataset, attr, arch)
                row += f"{'--':>10}" if r is None else f"{r['gain_mean_pp']:+10.2f}"
            print(row)
    print()


def write_latex_combined(agg_base: dict, agg_extra: dict, out_path: Path) -> None:
    """Combined 6-architecture LaTeX table.  Rows: (dataset, attribute);
    columns: LR / MLP / XGB / RF / SVM / DeepMLP; cells: gain (pp) over
    best single-purpose, mean (std) across 3 PCRL seeds.  '--' when an
    architecture has no shard for that (dataset, attribute)."""
    archs = ("LR", "MLP", "XGB", "RF", "SVM", "DeepMLP")

    def find(agg: dict, dataset: str, attr: str, arch: str) -> dict | None:
        for r in agg.get(dataset, []):
            if r["attribute"] == attr and r["arch"] == arch:
                return r
        return None

    lines: list[str] = []
    lines.append("% Auto-generated by experiments/run_cross_purpose_attack_v2.py "
                 "--extra-architectures.")
    lines.append("% Six-auditor cross-purpose attack: original LR/MLP/XGB plus "
                 "RF/SVM/DeepMLP as defensive depth.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Cross-purpose concatenation attack across six auditor "
                 "architectures. Cells: gain over best single-purpose auditor "
                 "in percentage points (mean across 3 PCRL seeds, std in "
                 "parentheses); each cell is the best of 3 auditor seeds. "
                 "RF: 200 trees, max-depth 10; SVM: RBF, $C{=}1.0$, "
                 "$\\gamma{=}\\text{scale}$, train subsampled to 10k when "
                 "$N{>}10\\text{k}$; DeepMLP: $[256,256,128,128]$, ReLU, "
                 "dropout 0.3, Adam $10^{-3}$, batch 256, $\\le$100 epochs, "
                 "early stop patience 10.}")
    lines.append("\\label{tab:cross_extra}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{ll" + "c" * len(archs) + "}")
    lines.append("\\toprule")
    lines.append("\\textbf{Dataset} & \\textbf{Attribute} & "
                 + " & ".join(f"\\textbf{{{a}}}" for a in archs) + " \\\\")
    lines.append("\\midrule")

    for dataset in ("adult", "hmda", "diabetes"):
        attrs = sorted(
            {r["attribute"] for r in agg_base.get(dataset, [])}
            | {r["attribute"] for r in agg_extra.get(dataset, [])}
        )
        for attr in attrs:
            cells: list[str] = []
            for arch in archs:
                src = agg_base if arch in ("LR", "MLP", "XGB") else agg_extra
                r = find(src, dataset, attr, arch)
                if r is None:
                    cells.append("--")
                else:
                    cells.append(f"{r['gain_mean_pp']:+.2f} ({r['gain_std_pp']:.2f})")
            lines.append(
                f"{dataset.capitalize()} & {attr.replace('_', ' ')} & "
                + " & ".join(cells) + " \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    parser.add_argument("--seeds", type=int, nargs="*", default=PCRL_SEEDS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default="results/v2_cross_purpose")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip auditor runs; aggregate existing raw/*.json files.")
    parser.add_argument(
        "--extra-architectures", action="store_true",
        help="Run RF/SVM/DeepMLP instead of LR/MLP/XGB.  Writes shards to "
             "raw/{ds}_s{seed}_extra.json and aggregate_extra.json.  Does NOT "
             "overwrite the base aggregate.json.  When the base aggregate "
             "exists, also emits cross_purpose_attack_extra.tex (6-col table) "
             "and prints a combined stdout comparison.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.out_dir
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    auditors = AUDITORS_EXTRA if args.extra_architectures else AUDITORS
    shard_suffix = "_extra" if args.extra_architectures else ""
    agg_filename = "aggregate_extra.json" if args.extra_architectures else "aggregate.json"
    per_seed_filename = (
        "per_seed_results_extra.json" if args.extra_architectures
        else "per_seed_results.json"
    )

    log.info(f"device={device}, datasets={args.datasets}, seeds={args.seeds}")
    log.info(f"AUDITOR_SEEDS={AUDITOR_SEEDS}")
    log.info(f"auditors={list(auditors.keys())}")

    if not args.aggregate_only:
        for dataset in args.datasets:
            for seed in args.seeds:
                t0 = time.time()
                try:
                    entry = run_one_seed(dataset, seed, device, auditors=auditors)
                except FileNotFoundError as e:
                    log.error(f"skipping {dataset}/s{seed}: {e}")
                    continue
                # Save individual shard
                shard_path = raw_dir / f"{dataset}_s{seed}{shard_suffix}.json"
                shard_path.write_text(json.dumps(entry, indent=2))
                log.info(f"[{dataset}/s{seed}] DONE in {time.time()-t0:.0f}s -> {shard_path}")

    # Aggregate from all shards on disk (so parallel runs combine cleanly)
    agg, per_seed = aggregate_from_dir(raw_dir, extra=args.extra_architectures)

    (out_dir / per_seed_filename).write_text(
        json.dumps({"per_seed": per_seed}, indent=2)
    )
    (out_dir / agg_filename).write_text(json.dumps(agg, indent=2))

    if args.extra_architectures:
        # Emit combined LaTeX + stdout if the base aggregate exists.
        base_path = out_dir / "aggregate.json"
        if base_path.exists():
            agg_base = json.loads(base_path.read_text())
            latex_path = ROOT / "paper-body" / "tables" / "cross_purpose_attack_extra.tex"
            latex_path.parent.mkdir(parents=True, exist_ok=True)
            write_latex_combined(agg_base, agg, latex_path)
            print_combined_stdout_table(agg_base, agg)
            print(f"Saved:  {out_dir / per_seed_filename}")
            print(f"        {out_dir / agg_filename}")
            print(f"        {latex_path}")
            print()
        else:
            log.warning(
                f"base aggregate.json not found at {base_path} — "
                "skipping combined LaTeX/stdout. Run without "
                "--extra-architectures first to produce it."
            )
            print(f"Saved:  {out_dir / per_seed_filename}")
            print(f"        {out_dir / agg_filename}")
        return

    md_path = ROOT / "results" / "V2_CROSS_PURPOSE_ATTACK.md"
    write_markdown(agg, md_path, per_seed)

    latex_path = ROOT / "paper-body" / "tables" / "cross_purpose_attack.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    write_latex(agg, latex_path)

    # Print final summary
    print()
    print("=" * 90)
    print("CROSS-PURPOSE ATTACK SUMMARY  (mean Δ over majority, pp; gain over best single, pp)")
    print("=" * 90)
    for dataset, rows in agg.items():
        print()
        print(f"--- {dataset.upper()} ---")
        print(f"{'attribute':<18}{'arch':<6}{'maj':<8}{'single Δ pp':<18}"
              f"{'concat Δ pp':<18}{'gain pp':<16}{'verdict':<8}")
        for r in sorted(rows, key=lambda x: (x["attribute"], x["arch"])):
            print(
                f"{r['attribute']:<18}{r['arch']:<6}"
                f"{r['majority']*100:6.2f}%  "
                f"{r['single_delta_mean_pp']:+6.2f} ± {r['single_delta_std_pp']:.2f}     "
                f"{r['concat_delta_mean_pp']:+6.2f} ± {r['concat_delta_std_pp']:.2f}     "
                f"{r['gain_mean_pp']:+6.2f} ± {r['gain_std_pp']:.2f}   "
                f"{r['verdict']:<8}"
            )
    print()
    print(f"Saved:  {out_dir / 'per_seed_results.json'}")
    print(f"        {out_dir / 'aggregate.json'}")
    print(f"        {md_path}")
    print(f"        {latex_path}")
    print()


if __name__ == "__main__":
    main()
