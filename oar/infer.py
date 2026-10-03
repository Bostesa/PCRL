"""Inference for the output-aware removal study, from saved per-row predictions only.

Recovery = supported-class macro one-vs-rest AUC (the benchmark's estimator: bench_infer.class_auc on all assessment
rows, mean over supported classes), averaged over attacker seeds (as0-as2), then over release seeds, then over
encoder seeds. Utility = U2 probe accuracy. Group bootstrap over assessment groups with fixed fitted predictors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stored_model_eval.bench_infer import Graph, _q, class_auc, pair_auc, run
from stored_model_eval.pilot_infer import UnitBootstrap

from .study import RUN, unit_dir

ATT = (0, 1, 2)


def load(uid):
    d = unit_dir(uid)
    with np.load(d / "preds.npz") as z:
        p = {k: z[k] for k in z.files}
    return p, json.loads((d / "record.json").read_text())


def acc_stat(correct):
    c = np.asarray(correct, float)

    def f(WT):
        n = WT.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, (c @ WT) / n, np.nan)
    return f


class CellGraph:
    """Builds recovery / utility ids for one cell. All units must share identical assessment rows."""

    def __init__(self, classes, pairs):
        self.g = Graph()
        self.classes, self.pairs = list(classes), [tuple(p) for p in pairs]
        self.rows = None
        self.membership = {}

    def _check_rows(self, p):
        r = (p["assess_row_id"], p["assess_unit"])
        if self.rows is None:
            self.rows = r
        elif not (np.array_equal(r[0], self.rows[0]) and np.array_equal(r[1], self.rows[1])):
            raise ValueError("assessment rows differ between units: pairing impossible")

    def recovery_of(self, uid, recipe="NL", resolve=True):
        """Unit-level macro AUC id, averaged over attacker seeds. Plus-surface units resolve their validation-selected
        ignore-candidate to the aliased component unit (benchmark rule)."""
        p, rec = load(uid)
        self._check_rows(p)
        if resolve and recipe == "NL" and rec.get("plus_selection", {}).get("alias_source"):
            src = rec["plus_selection"]["alias_source"]
            return self.recovery_of(src, "NL", resolve=True)
        y = p["y_s"]
        keys = [f"P__NL__as{a}" for a in ATT] if recipe == "NL" else [f"P__{recipe}"]
        ids = []
        for k in keys:
            P = p[k]
            cls = [self.g.base_once(f"{uid}|{k}|c{c}", f"{uid}|{k}|cls{c}", class_auc(y, P, c)) for c in self.classes]
            ids.append(self.g.add(f"{uid}|{k}|macro", "mean", cls))
        return self.g.add(f"{uid}|{recipe}|macro_auc", "mean", ids)

    def worst_parts(self, uid, recipe="NL"):
        """Per-class and per-pair AUC ids (averaged over attacker seeds) for worst-class / worst-pair diagnostics."""
        p, rec = load(uid)
        if recipe == "NL" and rec.get("plus_selection", {}).get("alias_source"):
            return self.worst_parts(rec["plus_selection"]["alias_source"], recipe)
        y = p["y_s"]
        keys = [f"P__NL__as{a}" for a in ATT] if recipe == "NL" else [f"P__{recipe}"]
        out = {}
        for c in self.classes:
            out[f"cls{c}"] = self.g.add(f"{uid}|{recipe}|cls{c}", "mean",
                                        [self.g.base_once(f"{uid}|{k}|c{c}", f"{uid}|{k}|cls{c}", class_auc(y, p[k], c))
                                         for k in keys])
        for i, j in self.pairs:
            out[f"pair{i}-{j}"] = self.g.add(f"{uid}|{recipe}|pair{i}-{j}", "mean",
                                             [self.g.base_once(f"{uid}|{k}|p{i}{j}", f"{uid}|{k}|pair{i}-{j}",
                                                               pair_auc(y, p[k], i, j)) for k in keys])
        return out

    def accuracy_of(self, uid):
        p, _ = load(uid)
        self._check_rows(p)
        return self.g.base_once(f"{uid}|acc", f"{uid}|U2|accuracy", acc_stat(p["U2_P"].argmax(1) == p["y_t"]))

    def group(self, name, ids, members):
        self.membership[name] = list(members)
        return self.g.add(name, "mean", ids)

    def diff(self, name, a, b):
        return self.g.add(name, "diff", [a, b])


def bootstrap(cg: CellGraph, ids, B, seed, chunk=500):
    boot = UnitBootstrap(cg.rows[1], B, seed, chunk)
    pts, reps = run(cg.g, boot, ids)
    return pts, reps, boot


def bound(r, q_lo, q_hi):
    iv, n_ne = _q(r, q_lo, q_hi, "linear", q_lo)
    return (iv if iv else (None, None)), n_ne
