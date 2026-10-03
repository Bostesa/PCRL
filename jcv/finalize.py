"""Deployment finalisation, identical for every neural arm (and the head part for FARE arms).

1. Erasure arms: refit the official LEACE map (float64, package defaults) on defense_train representations of the
   final encoders; release r_i = map_i(g_i(X)) (official LeaceEraser, float64). U: r_i = g_i(X).
2. Deployed head: affine = StandardScaler (fit on defense_train) + multinomial LogisticRegression, C in HEAD_C fitted on
   defense_train, selected by defense_val log loss (ties -> smaller C). Never fitted on attacker or assessment roles.
3. Released outputs: centred logits (= log-probabilities minus their row mean; exact for softmax heads), probabilities,
   hard decision. The primary recipient view is [r_i, centred logits_i].
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HEAD_C = [0.01, 0.1, 1.0, 10.0, 100.0]


def fit_head(R, y, tr, va, K):
    best = None
    table = []
    for C in HEAD_C:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        m.fit(R[tr], y[tr])
        ll = log_loss(y[va], m.predict_proba(R[va]), labels=list(range(K)))
        table.append({"C": C, "defense_val_log_loss": float(ll)})
        if best is None or ll < best[0] - 1e-12:
            best = (ll, C, m)
    return best[2], {"selected_C": best[1], "table": table}


def outputs(head, R):
    """Centred logits from the head's decision function (finite even when a probability underflows to 0), the
    head's probabilities and the hard decision. Amendment A1 (2026-10-03): the original log(predict_proba) form gave
    -inf for underflowing probabilities (U arm, income)."""
    z = np.asarray(head.decision_function(R), dtype=np.float64)
    if z.ndim == 1:                       # binary sklearn head: one margin d -> logits (0, d)
        z = np.stack([np.zeros_like(z), z], 1)
    cen = z - z.mean(1, keepdims=True)
    P = head.predict_proba(R)
    return cen, P, P.argmax(1)


def finalize_neural(model, arm_erasure, Xall, D, Ks):
    from stored_model_eval.defenses import fit_leace
    tr, va = D["idx"]["defense_train"], D["idx"]["defense_val"]
    S = D["sex"]
    out, maps, heads, meta = {}, {}, {}, {"heads": {}, "leace": {}}
    with torch.no_grad():
        Hs = [model.encode(i, torch.from_numpy(Xall)).double().numpy() for i in (0, 1)]
    for i in (0, 1):
        H = Hs[i]
        if arm_erasure:
            m = fit_leace(H[tr], np.eye(2)[S[tr]], fit_row_ids=D["row_id"][tr])
            R = m.transform(H)
            maps[i] = m
            nc = m.native_check(H[tr], np.eye(2)[S[tr]])
            meta["leace"][i] = {"native_check_fit_rows": nc, "metadata": m.metadata}
        else:
            R = H
        y = D["y"]["income" if i == 0 else "occupation_group"]
        head, hm = fit_head(R, y, tr, va, Ks[i])
        heads[i] = head
        meta["heads"][i] = hm
        cen, P, hard = outputs(head, R)
        out[f"r{i + 1}"] = R
        out[f"c{i + 1}"] = cen
        out[f"p{i + 1}"] = P
        out[f"hard{i + 1}"] = hard
    return out, maps, heads, meta


def save_unit(d: Path, files: dict, record: dict):
    """Atomic unit write: files dict name -> callable(path) ; COMPLETE.json with sha256 of every file."""
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name, writer in files.items():
        p = tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        writer(p)
    (tmp / "record.json").write_text(json.dumps(record, indent=1, default=str))
    hashes = {str(p.relative_to(tmp)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(tmp.rglob("*")) if p.is_file()}
    (tmp / "COMPLETE.json").write_text(json.dumps({"id": d.name, "files": hashes}, indent=1))
    if d.exists():
        d.rename(d.with_name(d.name + ".quarantined"))
    tmp.rename(d)


def unit_complete(d: Path) -> bool:
    c = d / "COMPLETE.json"
    if not c.exists():
        return False
    for f, h in json.loads(c.read_text())["files"].items():
        p = d / f
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            return False
    return True
