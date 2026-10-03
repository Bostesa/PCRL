"""Inference for the output-leak diagnosis study, from saved per-row predictions only.

Point: mean over encoder seeds of per-seed statistics (recovery: mean over attacker seeds as0-as2 first).
SE: sd (ddof 1) over B paired group-bootstrap replicates (multinomial counts per assessment record group; the same
draws for every unit of a dataset). Interval: est +/- z * SE, z = Bonferroni normal critical value of the family.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stored_model_eval.bench_infer import Graph, class_auc, pair_auc, run
from stored_model_eval.pilot_infer import UnitBootstrap

HOME = Path.home()
ODX_UNITS = HOME / "PCRL_eval_cache_private" / "odx_v1" / "run" / "units"
ATT = (0, 1, 2)


def resolve(uid: str) -> Path:
    """Unit directory holding the predictions for uid (follows bank selections and alias records)."""
    d = ODX_UNITS / uid
    rec = json.loads((d / "record.json").read_text())
    if rec.get("kind") == "bank":
        return resolve(rec["bank_selected"])
    if (d / "ALIAS.json").exists():
        return HOME / json.loads((d / "ALIAS.json").read_text())["source"].lstrip("~/")
    return d


def load_preds(uid):
    with np.load(resolve(uid) / "preds.npz") as z:
        return {k: z[k] for k in z.files}


def acc_stat(correct):
    c = np.asarray(correct, float)

    def f(WT):
        n = WT.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, (c @ WT) / n, np.nan)
    return f


class DS:
    """Statistic graph for one dataset (all units share the assessment rows)."""

    def __init__(self, assess_row_id, assess_unit):
        self.g = Graph()
        self.rows, self.units = np.asarray(assess_row_id), np.asarray(assess_unit)

    def _check(self, p):
        if not (np.array_equal(p["assess_row_id"], self.rows) and np.array_equal(p["assess_unit"], self.units)):
            raise ValueError("assessment rows differ: pairing impossible")

    def recovery(self, uid, classes, recipe="NL"):
        p = load_preds(uid)
        self._check(p)
        keys = [f"P__NL__as{a}" for a in ATT] if recipe == "NL" else [f"P__{recipe}"]
        ids = []
        for k in keys:
            cls = [self.g.base_once(f"{resolve(uid)}|{k}|c{c}", f"{uid}|{k}|cls{c}", class_auc(p["y_s"], p[k], c))
                   for c in classes]
            ids.append(self.g.add(f"{uid}|{k}|macro", "mean", cls))
        return self.g.add(f"{uid}|{recipe}|macro", "mean", ids)

    def pair(self, uid, i, j, recipe="NL"):
        p = load_preds(uid)
        self._check(p)
        keys = [f"P__NL__as{a}" for a in ATT] if recipe == "NL" else [f"P__{recipe}"]
        ids = [self.g.base_once(f"{resolve(uid)}|{k}|p{i}-{j}", f"{uid}|{k}|pair{i}-{j}", pair_auc(p["y_s"], p[k], i, j))
               for k in keys]
        return self.g.add(f"{uid}|{recipe}|pair{i}-{j}", "mean", ids)

    def accuracy(self, name, correct):
        if len(correct) != len(self.rows):
            raise ValueError("accuracy vector length mismatch")
        return self.g.base_once(name, name, acc_stat(correct))

    def mean(self, name, ids):
        return self.g.add(name, "mean", list(ids))

    def diff(self, name, a, b):
        return self.g.add(name, "diff", [a, b])


def estimate(ds: DS, ids, B, seed, z):
    boot = UnitBootstrap(ds.units, B, seed, 500)
    pts, reps = run(ds.g, boot, list(dict.fromkeys(ids)))
    out = {}
    for s in ids:
        r = reps[s][np.isfinite(reps[s])]
        se = float(np.std(r, ddof=1)) if len(r) > 1 else float("nan")
        out[s] = {"point": pts[s], "se": se, "lower": pts[s] - z * se, "upper": pts[s] + z * se,
                  "n_finite_replicates": int(len(r)), "B": B, "seed": seed, "z": z}
    return out
