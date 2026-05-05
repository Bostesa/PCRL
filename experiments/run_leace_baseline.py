"""LEACE-on-raw baseline across Adult / Diabetes / HMDA.

Threat experiment 1: closed-form linear concept erasure (Belrose et al. 2023)
applied directly to raw features. Tests whether the paper's compliance
criterion (linear R^2 < 0.05 AND post-hoc MLP delta < 2pp) is achievable
without learning a representation at all.

Pipeline per dataset:
  1. Load raw features and per-purpose disallowed-attribute labels.
  2. For each purpose, iteratively LEACE-erase every disallowed attribute on
     train features, then transform train+test with the composed eraser.
  3. Score erased reps against fresh post-hoc auditors (LogReg / MLP / RF /
     XGB) and a closed-form linear R^2 audit.
  4. Train task heads on erased reps and report task accuracy.
  5. Dump per-pair / per-task / pass-count to JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from concept_erasure import LeaceFitter

from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
from pcrl.data.hmda import HMDADataset, get_hmda_purposes
from pcrl.purposes.verification import LinearComplianceCertificate

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as e:  # pragma: no cover
    print(f"xgboost unavailable ({e}); falling back to sklearn GBM")
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier

FAILURE_LOG = Path("/tmp/leace_failures.log")


def log_failure(msg: str) -> None:
    FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_LOG, "a") as f:
        f.write(msg.rstrip() + "\n")
    print(f"[FAIL] {msg}")


def load_dataset(name: str):
    if name == "adult":
        purposes = get_adult_purposes()
        train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=False)
        test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False)
    elif name == "diabetes":
        purposes = get_diabetes_purposes()
        train_ds = DiabetesDataset(purposes=purposes, root="data/diabetes_processed", split="train")
        test_ds = DiabetesDataset(purposes=purposes, root="data/diabetes_processed", split="test")
    elif name == "hmda":
        purposes = get_hmda_purposes()
        train_ds = HMDADataset(purposes=purposes, root="data", split="train")
        test_ds = HMDADataset(purposes=purposes, root="data", split="test")
    else:
        raise ValueError(name)
    return purposes, train_ds, test_ds


def majority_baseline(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())


class TaskMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, int], out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AuditorMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_torch_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
) -> float:
    device = torch.device(device)
    model = model.to(device)
    Xt = torch.from_numpy(X_train.astype(np.float32))
    yt = torch.from_numpy(y_train.astype(np.int64))
    Xv = torch.from_numpy(X_test.astype(np.float32)).to(device)
    yv = torch.from_numpy(y_test.astype(np.int64)).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        # Chunked eval to avoid OOM on large test sets
        preds = []
        for i in range(0, len(Xv), 4096):
            preds.append(model(Xv[i:i+4096]).argmax(dim=-1).cpu().numpy())
        preds = np.concatenate(preds)
    return float((preds == y_test).mean())


def linear_r2_multiclass(X: np.ndarray, y: np.ndarray) -> float:
    cert = LinearComplianceCertificate(epsilon=0.05)
    res = cert.check(X.astype(np.float64), y.astype(np.int64))
    return float(res.r_squared)


def fit_logreg(X_train, y_train, X_test, y_test) -> float:
    n_classes = int(max(y_train.max(), y_test.max())) + 1
    if n_classes == 1:
        return majority_baseline(y_test)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def fit_rf(X_train, y_train, X_test, y_test) -> float:
    clf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=0)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def fit_xgb(X_train, y_train, X_test, y_test) -> float:
    n_classes = int(max(y_train.max(), y_test.max())) + 1
    if HAS_XGB:
        if n_classes == 2:
            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                eval_metric="logloss", n_jobs=-1, verbosity=0,
            )
        else:
            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                objective="multi:softprob", num_class=n_classes,
                eval_metric="mlogloss", n_jobs=-1, verbosity=0,
            )
        clf.fit(X_train, y_train)
        return float(clf.score(X_test, y_test))
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=0)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def apply_leace_chain(
    X_train: np.ndarray,
    X_test: np.ndarray,
    label_train_per_attr: dict[str, np.ndarray],
    attrs_to_erase: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively LEACE-erase each disallowed attribute. Erasers compose."""
    Xt = torch.from_numpy(X_train.astype(np.float32))
    Xv = torch.from_numpy(X_test.astype(np.float32))
    for attr in attrs_to_erase:
        labels = torch.from_numpy(label_train_per_attr[attr].astype(np.int64))
        n_classes = int(labels.max().item()) + 1
        if n_classes < 2:
            continue
        Y = F.one_hot(labels, num_classes=n_classes).float()
        fitter = LeaceFitter.fit(Xt, Y)
        eraser = fitter.eraser
        Xt = eraser(Xt)
        Xv = eraser(Xv)
    return Xt.numpy().astype(np.float32), Xv.numpy().astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["adult", "diabetes", "hmda"], required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    t0 = time.time()
    print(f"=== LEACE-on-raw baseline: {args.dataset} ===")
    purposes, train_ds, test_ds = load_dataset(args.dataset)

    X_train = train_ds.features.numpy().astype(np.float32)
    X_test = test_ds.features.numpy().astype(np.float32)
    print(f"Train {X_train.shape}  Test {X_test.shape}")

    sens_train = {a: t.numpy() for a, t in train_ds.sensitive_attrs.items()}
    sens_test = {a: t.numpy() for a, t in test_ds.sensitive_attrs.items()}
    task_train = {t: lab.numpy() for t, lab in train_ds.task_labels.items()}
    task_test = {t: lab.numpy() for t, lab in test_ds.task_labels.items()}

    out_dir = ROOT / "results" / f"{args.dataset}_LEACE"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_pair: dict[str, dict] = {}
    per_task: dict[str, dict] = {}
    erased_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for p in purposes:
        attrs = list(p.disallowed_attrs)
        print(f"\n[{p.name}] erasing {attrs}")
        try:
            Xtr_e, Xte_e = apply_leace_chain(
                X_train, X_test, sens_train, attrs,
            )
        except Exception as e:
            log_failure(f"{args.dataset}/{p.name}: LEACE fit failed: {e}")
            continue

        torch.save(
            {"train": torch.from_numpy(Xtr_e), "test": torch.from_numpy(Xte_e)},
            out_dir / f"erased_{p.name}.pt",
        )
        erased_cache[p.name] = (Xtr_e, Xte_e)

        for attr in attrs:
            if attr not in sens_train:
                continue
            y_tr = sens_train[attr]
            y_te = sens_test[attr]
            n_classes = int(max(y_tr.max(), y_te.max())) + 1
            try:
                r2 = linear_r2_multiclass(Xte_e, y_te)
            except Exception as e:
                log_failure(f"{args.dataset}/{p.name}/{attr}: linear R2 failed: {e}")
                r2 = float("nan")

            try:
                lin_acc = fit_logreg(Xtr_e, y_tr, Xte_e, y_te)
            except Exception as e:
                log_failure(f"{args.dataset}/{p.name}/{attr}: logreg failed: {e}")
                lin_acc = majority_baseline(y_te)

            try:
                mlp = AuditorMLP(in_dim=Xtr_e.shape[1], out_dim=n_classes)
                mlp_acc = train_torch_classifier(
                    mlp, Xtr_e, y_tr, Xte_e, y_te,
                    epochs=50, batch_size=256, lr=1e-3, device=device,
                )
            except Exception as e:
                log_failure(f"{args.dataset}/{p.name}/{attr}: mlp failed: {e}")
                mlp_acc = majority_baseline(y_te)

            try:
                rf_acc = fit_rf(Xtr_e, y_tr, Xte_e, y_te)
            except Exception as e:
                log_failure(f"{args.dataset}/{p.name}/{attr}: rf failed: {e}")
                rf_acc = majority_baseline(y_te)

            try:
                xgb_acc = fit_xgb(Xtr_e, y_tr, Xte_e, y_te)
            except Exception as e:
                log_failure(f"{args.dataset}/{p.name}/{attr}: xgb failed: {e}")
                xgb_acc = majority_baseline(y_te)

            maj = majority_baseline(y_te)
            max_acc = max(lin_acc, mlp_acc, rf_acc, xgb_acc)
            delta = max_acc - maj
            adj_pass = bool((not np.isnan(r2)) and r2 < 0.05 and delta < 0.02)

            key = f"{p.name}/{attr}"
            per_pair[key] = {
                "linear_r2": r2,
                "linear_acc": lin_acc,
                "mlp_acc": mlp_acc,
                "rf_acc": rf_acc,
                "xgb_acc": xgb_acc,
                "max_acc": max_acc,
                "majority": maj,
                "delta_vs_majority": delta,
                "pass": adj_pass,
            }
            print(f"  {key}: r2={r2:.4f} max={max_acc:.4f} maj={maj:.4f} "
                  f"delta={delta:+.4f} pass={adj_pass}")

        for task in p.allowed_tasks:
            if task not in task_train:
                continue
            y_tr = task_train[task]
            y_te = task_test[task]
            n_classes = int(max(y_tr.max(), y_te.max())) + 1
            try:
                head = TaskMLP(in_dim=Xtr_e.shape[1], hidden=(128, 128), out_dim=n_classes)
                acc = train_torch_classifier(
                    head, Xtr_e, y_tr, Xte_e, y_te,
                    epochs=50, batch_size=256, lr=1e-3, device=device,
                )
            except Exception as e:
                log_failure(f"{args.dataset}/{p.name}/{task}: task head failed: {e}")
                acc = majority_baseline(y_te)
            maj = majority_baseline(y_te)
            per_task[f"{p.name}/{task}"] = {
                "acc": acc,
                "majority": maj,
                "delta": acc - maj,
            }
            print(f"  task {p.name}/{task}: acc={acc:.4f} maj={maj:.4f} delta={acc-maj:+.4f}")

    pass_count = sum(1 for v in per_pair.values() if v["pass"])
    total_pairs = len(per_pair)
    out = {
        "dataset": args.dataset,
        "method": "LEACE_on_raw",
        "wall_time_s": time.time() - t0,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features_raw": int(X_train.shape[1]),
        "per_pair": per_pair,
        "per_task": per_task,
        "adjusted_pass_count": pass_count,
        "total_pairs": total_pairs,
    }
    out_path = out_dir / "leace_baseline.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"adjusted_pass_count = {pass_count}/{total_pairs}")
    print(f"wall time = {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
