"""Standalone helpers for the E/F/G-AAAI recounts (verification role, 2026-10-01).

No durable-guarantees or PCRL code is imported. Only numpy/sklearn/json/hashlib.
Paths:
  DG    = read-only clone of Bostesa/durable-guarantees @956f5c8 (git-tracked results/)
  DRIVE = single members extracted from
          /Volumes/YOTUO/NathanSamson-Mac-relocated-2026-09-30/archives/tree-durable-guarantees.tar
          (analysis/ score arrays, gitignored in the repo)
"""
from __future__ import annotations
import hashlib, json, os
from itertools import combinations
import numpy as np
from sklearn.metrics import roc_auc_score

SCR = "/private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad"
DG = SCR + "/dg"
DRIVE = SCR + "/recon/verification/dg_drive/durable-guarantees"
OUTDIR = ("/Users/nathansamson/PCRL/.worktrees/combined-evidence-reconciliation-v1/results/"
          "combined_evidence_reconciliation_v1/notes/verification/recount_outputs")
INV = "/Volumes/YOTUO/NathanSamson-Mac-relocated-2026-09-30/inventories/tree-durable-guarantees.json.gz"
BARS = (0.52, 0.55, 0.60)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def dg_json(rel):
    p = os.path.join(DG, rel)
    return json.load(open(p)), {"location": f"durable-guarantees@956f5c8:{rel}", "sha256": sha(p)}


def drive_npz(rel):
    p = os.path.join(DRIVE, rel)
    return np.load(p), {"location": "tree-durable-guarantees.tar::durable-guarantees/" + rel,
                        "sha256": sha(p)}


def macro_ovr(y, p):
    """Macro one-vs-rest AUC (binary: AUC of column 1). Mirrors the paper's metric definition."""
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    present = sorted(set(int(v) for v in y))
    if len(present) < p.shape[1]:
        # classes absent from this fold: score present classes only and say so
        return float(np.mean([roc_auc_score((y == k).astype(int), p[:, k]) for k in present]))
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))


def per_class_ovr(y, p):
    out = {}
    for k in range(p.shape[1]):
        yk = (y == k).astype(int)
        if 0 < yk.sum() < len(yk):
            out[k] = float(roc_auc_score(yk, p[:, k]))
        else:
            out[k] = None
    return out


def worst_pair(y, p, keep=None, min_rows=10, orientation_free=True):
    """Max over class pairs of pairwise AUC scored by p_j/(p_i+p_j).
    keep: restrict to these classes (support rule). Pairs with < min_rows rows skipped."""
    k = p.shape[1]
    classes = range(k) if keep is None else keep
    best, arg = 0.0, None
    for i, j in combinations(classes, 2):
        m = (y == i) | (y == j)
        if m.sum() < min_rows or len(set(y[m].tolist())) < 2:
            continue
        den = p[m, i] + p[m, j]
        s = np.where(den > 0, p[m, j] / np.maximum(den, 1e-12), 0.5)
        a = float(roc_auc_score((y[m] == j).astype(int), s))
        if orientation_free:
            a = max(a, 1.0 - a)
        if a > best:
            best, arg = a, (int(i), int(j))
    return best, arg


def save(name, obj):
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, name)
    json.dump(obj, open(p, "w"), indent=1, default=float)
    return p
