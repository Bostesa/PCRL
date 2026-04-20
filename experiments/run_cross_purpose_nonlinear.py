#!/usr/bin/env python3
"""Nonlinear cross-purpose attack: MLP & XGBoost on concatenated reps.

Loads pre-trained Adult PCRL encoder from checkpoints/adult/best.pt.
Extracts h_p1 || h_p2 || h_p3 on train+test. Trains MLP (2 hidden, 256,
ReLU, dropout 0.3) and XGBoost auditors to predict race/sex/age_group/
marital_status. Max accuracy over 3 seeds each. Writes
results/adult/cross_purpose_nonlinear.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.linear_model import LogisticRegression

from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.models.encoder import PurposeConditionedEncoder

CKPT_PATH = project_root / "checkpoints" / "adult" / "best.pt"
LINEAR_CSV = project_root / "results" / "adult" / "cross_purpose_attack.csv"
OUT_CSV = project_root / "results" / "adult" / "cross_purpose_nonlinear.csv"

ATTRIBUTES = ["race", "sex", "age_group", "marital_status"]
ATTR_DIMS = {"race": 5, "sex": 2, "age_group": 4, "marital_status": 2}
SEEDS = [0, 1, 2]

# Encoder architecture (must match training config)
HIDDEN_DIMS = [128, 128]
REPR_DIM = 64
PURPOSE_EMB_DIM = 32
DROPOUT = 0.3
BATCH_SIZE = 256

# Attack MLP config
MLP_HIDDEN = 256
MLP_DROPOUT = 0.3
MLP_EPOCHS = 80
MLP_LR = 1e-3
MLP_WD = 1e-4
MLP_PATIENCE = 10


def extract_per_purpose(
    encoder: torch.nn.Module,
    loader: DataLoader,
    num_purposes: int,
    device: str,
) -> list[np.ndarray]:
    """Returns list of (N, repr_dim) arrays, one per purpose."""
    encoder.eval()
    per_purpose: list[list[np.ndarray]] = [[] for _ in range(num_purposes)]
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            for pidx in range(num_purposes):
                h = encoder(x, pidx)
                per_purpose[pidx].append(h.cpu().numpy())
    return [np.concatenate(per_purpose[i], axis=0) for i in range(num_purposes)]


def extract_labels(loader: DataLoader, attr: str) -> np.ndarray:
    chunks = []
    for batch in loader:
        chunks.append(batch["sensitive_attrs"][attr].numpy())
    return np.concatenate(chunks, axis=0)


class AttackMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(MLP_HIDDEN, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_attacker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    seed: int,
    device: str,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = X_train.shape[0]
    val_size = n // 10
    perm = np.random.permutation(n)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    X_tr_t = torch.from_numpy(X_tr).float().to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)
    X_te_t = torch.from_numpy(X_test).float().to(device)
    y_te_t = torch.from_numpy(y_test).long().to(device)

    model = AttackMLP(X_train.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state: dict | None = None
    stale = 0

    for epoch in range(MLP_EPOCHS):
        model.train()
        perm = torch.randperm(X_tr_t.size(0), device=device)
        for i in range(0, X_tr_t.size(0), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            logits = model(X_tr_t[idx])
            loss = loss_fn(logits, y_tr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= MLP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).argmax(dim=-1)
    return (preds == y_te_t).float().mean().item()


def train_xgb_attacker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    seed: int,
) -> float:
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=100,
        random_state=seed,
        use_label_encoder=False,
        eval_metric="mlogloss" if num_classes > 2 else "logloss",
        objective="multi:softprob" if num_classes > 2 else "binary:logistic",
        num_class=num_classes if num_classes > 2 else None,
        verbosity=0,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return float((preds == y_test).mean())


def best_single_linear_same_encoder(
    train_reps: list[np.ndarray],
    test_reps: list[np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Max LogisticRegression accuracy across single-purpose reps, same encoder."""
    best = 0.0
    for Xtr, Xte in zip(train_reps, test_reps):
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(Xtr, y_train)
        acc = float((clf.predict(Xte) == y_test).mean())
        if acc > best:
            best = acc
    return best


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────────
    purposes = get_adult_purposes()
    train_ds = AdultDataset(
        purposes=purposes, root="data", split="train", download=True,
    )
    test_ds = AdultDataset(
        purposes=purposes, root="data", split="test", download=False,
        norm_stats=train_ds.norm_stats,
    )
    input_dim = train_ds.info.num_features
    print(f"Train={len(train_ds)}  Test={len(test_ds)}  Features={input_dim}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_pcrl_batch,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_pcrl_batch,
    )

    # ── Load encoder from checkpoint ─────────────────────────────────────
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        repr_dim=REPR_DIM,
        num_purposes=len(purposes),
        purpose_emb_dim=PURPOSE_EMB_DIM,
        conditioning="film",
        dropout=DROPOUT,
    )
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.to(device).eval()
    print(f"Loaded encoder from {CKPT_PATH}")

    # ── Extract representations ──────────────────────────────────────────
    print("Extracting per-purpose representations...")
    train_reps = extract_per_purpose(encoder, train_loader, len(purposes), device)
    test_reps = extract_per_purpose(encoder, test_loader, len(purposes), device)
    X_train = np.concatenate(train_reps, axis=1)
    X_test = np.concatenate(test_reps, axis=1)
    print(f"  concat X_train: {X_train.shape}  X_test: {X_test.shape}")

    # ── Attack each attribute ────────────────────────────────────────────
    rows = []
    for attr in ATTRIBUTES:
        num_classes = ATTR_DIMS[attr]
        y_train = extract_labels(train_loader, attr)
        y_test = extract_labels(test_loader, attr)
        maj = float(np.bincount(y_test).max() / len(y_test))
        print(f"\n--- {attr} (C={num_classes}, majority={maj:.4f}) ---")

        lin_max = best_single_linear_same_encoder(train_reps, test_reps, y_train, y_test)
        print(f"  Linear (single, same ckpt) max={lin_max:.4f}")

        mlp_accs = [
            train_mlp_attacker(X_train, y_train, X_test, y_test, num_classes, s, device)
            for s in SEEDS
        ]
        mlp_max = max(mlp_accs)
        print(f"  MLP (concat)  seeds={[f'{a:.4f}' for a in mlp_accs]} max={mlp_max:.4f}")

        xgb_accs = [
            train_xgb_attacker(X_train, y_train, X_test, y_test, num_classes, s)
            for s in SEEDS
        ]
        xgb_max = max(xgb_accs)
        print(f"  XGBoost (concat) seeds={[f'{a:.4f}' for a in xgb_accs]} max={xgb_max:.4f}")

        rows.append({
            "attribute": attr,
            "best_single_linear": round(lin_max, 4),
            "concatenated_MLP_max": round(mlp_max, 4),
            "concatenated_XGB_max": round(xgb_max, 4),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["attribute", "best_single_linear",
                        "concatenated_MLP_max", "concatenated_XGB_max"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {OUT_CSV}")
    print("\nSummary:")
    for r in rows:
        print(f"  {r['attribute']:16s} "
              f"lin={r['best_single_linear']:.4f}  "
              f"MLP={r['concatenated_MLP_max']:.4f}  "
              f"XGB={r['concatenated_XGB_max']:.4f}")


if __name__ == "__main__":
    main()
