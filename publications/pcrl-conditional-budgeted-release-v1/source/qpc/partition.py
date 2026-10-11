"""Stage B fine partitions: task-only KL/Bregman k-means per teacher-predicted class on DEFENSE_FIT probabilities.

Income: at most 32 fine cells per predicted class; occupation: at most 128. The starts (source, kpp:20261006,
kpp:20261007), the <= 200-round convergence rule, best-coherent-iterate retention, empty-cell handling, per-class
start selection and the reserved fallback cell for an absent class are exactly qpc.kmeans (METHOD_CARD section 3).
No task label or SEX enters. Fine-cell IDs are private fitting state; they are never released (the deployed release
carries only the coarse token, its decoded vector and the decision).

Runner interface:
    fine_unit(T, tr, caps={1: 32, 2: 128}) -> (record, {"fine.json": writer, "assign.npz": arrays})
    load_fine(fine_dict) -> (fine1, fine2)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qpc import kmeans as KM

CAPS = {1: 32, 2: 128}
KS = {1: 2, 2: 6}


def support_receipt(fine: KM.FinePartition):
    """Per-class support of a fine partition (fitting rows)."""
    out = []
    for c in range(fine.K):
        idx = fine.cells_of(c)
        n = fine.n[idx]
        fb = bool(fine.fallback[idx].any())
        out.append({"class": c, "fallback": fb, "cells": int(idx.size), "rows": int(n.sum()),
                    "min_n": None if fb else int(n.min()), "median_n": None if fb else float(np.median(n)),
                    "singletons": 0 if fb else int(np.sum(n == 1)),
                    "sparse_lt5": 0 if fb else int(np.sum(n < KM.SPARSE_N))})
    return out


def fit_fine_pair(P1, d1, P2, d2, *, caps=None, starts=KM.STARTS, rounds=KM.ROUNDS, tag=""):
    """Fine partitions for both recipients on aligned fitting rows. Returns (fine1, fine2, receipts)."""
    caps = dict(CAPS if caps is None else caps)
    t0, c0 = time.perf_counter(), time.process_time()
    if np.asarray(P1).shape[0] != np.asarray(P2).shape[0]:
        raise ValueError("P1 and P2 must be aligned fitting rows")
    fits = {}
    for r, P, d in ((1, P1, d1), (2, P2, d2)):
        fits[r] = KM.fit_recipient(P, d, np.asarray(P).shape[1], int(caps[r]), starts=starts, rounds=rounds,
                                   tag=f"{tag}|fine|r{r}")
    rec = {"caps": {1: int(caps[1]), 2: int(caps[2])}, "r1": fits[1].receipt, "r2": fits[2].receipt,
           "support": {1: support_receipt(fits[1].partition), 2: support_receipt(fits[2].partition)},
           "F": [fits[1].partition.F, fits[2].partition.F],
           "fingerprints": [fits[1].partition.fingerprint(), fits[2].partition.fingerprint()],
           "wall_seconds": time.perf_counter() - t0, "cpu_seconds": time.process_time() - c0}
    return fits[1].partition, fits[2].partition, rec


def fine_unit(T, tr, caps=None):
    """Fine partitions of one teacher/seed on DEFENSE_FIT rows ``tr``; assignments of ALL rows (private)."""
    tr = np.asarray(tr)
    if np.asarray(T["p1"]).shape[1] != KS[1] or np.asarray(T["p2"]).shape[1] != KS[2]:
        raise ValueError("recipient 1 is income (K=2), recipient 2 is occupation (K=6)")
    f1, f2, rec = fit_fine_pair(T["p1"][tr], T["d1"][tr], T["p2"][tr], T["d2"][tr], caps=caps)
    a1 = KM.assign_fine(T["p1"], T["d1"], f1)
    a2 = KM.assign_fine(T["p2"], T["d2"], f2)
    for a, f in ((a1, f1), (a2, f2)):
        n = np.bincount(a[tr], minlength=f.F)
        if not np.array_equal(n, f.n):
            raise AssertionError("fine statistics differ from the deployed assignment on the fitting rows")
    rec["fit_rows"] = int(tr.size)
    KM.json_safe(rec)
    s = json.dumps({"fine1": f1.to_dict(), "fine2": f2.to_dict()}, allow_nan=False)
    return rec, {"fine.json": lambda p: Path(p).write_text(s),
                 "assign.npz": {"row_id": np.asarray(T["row_id"]), "f1": a1, "f2": a2}}


def load_fine(fine_dict):
    return KM.FinePartition.from_dict(fine_dict["fine1"]), KM.FinePartition.from_dict(fine_dict["fine2"])
