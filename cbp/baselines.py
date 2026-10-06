"""Continuous and reference baselines for the confidence-budgeted privacy study (cbp; role D; prompt sections 6, 10).

PROVENANCE. A COPY of qpc/baselines.py at the source handoff tip d0c8a45c879d01fb8b736ccc091ec3e2c3e9b351 (itself
adapted from dpc/baselines.py at 0a7b05a5). qpc/ and dpc/ are never edited and nothing in them is monkeypatched at
runtime. The documented diff against qpc/baselines.py is exactly:
  D1  docstrings only: this provenance header, the admission sentence below (cbp.admit) and three one-line
      docstrings that name the cbp store / cbp.run instead of qpc's;
  D2  units_dir(): the default unit store is cbp.run.UNITS (<PRIVATE_CACHE>/cbp_v1/run/units) instead of
      qpc.run.UNITS (the lazy import is bound to cbp.run). This is the only code change.
Everything else (view families, OSF_CHOICES record, finite-by-value FARE score views, teacher / reference loaders and
the hash-complete unit reader) is byte-for-byte the qpc logic. Nothing is refit or reselected here.

Baseline releases (every one faces the SAME FINAL slate and the SAME utility contract as every code):
  SRC|U            U continuous release (teacher.npz): interface [c_i, p_i] (PRIMARY), complete [r_i, c_i], scores c_i,
                   probs p_i, decisions one-hot(d_i). The decisions family is the "decisions alone" contract; its
                   recovery is reported, and as a protected release it carries no probability vector, so it is never a
                   confidence-eligible arm (the CLASS code U|CLASS|i1o1 is the decision-only CODE with decoded class
                   means, gated like every code).
  SRC|RAW-J_b0.3   RAW-J beta 0.3 score-only release: the same five families of its teacher.npz (incumbent comparator
                   only; no codes of RAW-J exist in this study, so its composed bank equals its own bank).
  REF|E            official LEACE score output (interface primary; scores, probs, decisions; complete descriptive).
  REF|F, REF|F0    official FARE / no-fairness FARE score outputs (interface primary; scores, probs, decisions; cells =
                   the FARE cell code as a finite view). Score views of F / F0 are finitely valued and keyed on the
                   exact float64 value identity, so the cell readers apply to them.
Admission status is the custody role's: a reference whose unit is missing or not hash-complete is MISSING (never
silently replaced); its provenance validity is recorded by cbp.admit (verified copies of the qpc units), not here.

API
  source_view_sets(t, D)                      -> {family: views} of a teacher dict (p, c, r, d)
  source_outputs(t)                           -> {p1, p2, hard1, hard2} (the continuous release's outputs)
  reference_view_sets(label, k, D, units_dir=None, with_outputs=False, arrays=None)
  load_reference(label, k, D, units_dir=None) -> (arrays, provenance)
  load_teacher(t, k, D=None, units_dir=None)  -> (arrays, provenance)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from dpc import audit as DA
from dpc.baselines import OSF_CHOICES  # noqa: F401  (inherited reference record, unchanged)

REFS = ("E", "F", "F0")
TEACHERS = ("U", "RAW-J_b0.3")
SOURCE_FAMILIES = ("interface", "complete", "scores", "probs", "decisions")
PRIMARY_FAMILY = "interface"
SCORE_FAMILIES = ("interface", "scores", "probs", "decisions")
TEACHER_KEYS = ("row_id", "p1", "p2", "d1", "d2", "c1", "c2", "r1", "r2")


def units_dir(units_dir=None):
    if units_dir is not None:
        return Path(units_dir)
    from cbp import run as R                     # D2: cbp unit store
    return R.UNITS


def load_unit_npz(unit, fname, units_dir_=None):
    """Arrays of a hash-complete unit file (jcv.finalize.unit_complete, as cbp.run writes them) + COMPLETE sha256."""
    from jcv.finalize import unit_complete
    d = units_dir(units_dir_) / unit
    if not unit_complete(d):
        raise SystemExit(f"REFUSED: {unit} is missing or not hash-complete")
    z = np.load(d / fname, allow_pickle=False)
    return {k: z[k] for k in z.files}, hashlib.sha256((d / "COMPLETE.json").read_bytes()).hexdigest()


def _aligned(z, D, unit):
    if D is not None and not np.array_equal(np.asarray(z["row_id"]), np.asarray(D["row_id"])):
        raise ValueError(f"REFUSED: {unit} rows are not aligned with D")
    return z


def load_teacher(t, k, D=None, units_dir_=None):
    unit = f"tea__s{k}__{t}"
    z, sha = load_unit_npz(unit, "teacher.npz", units_dir_)
    miss = [x for x in ("row_id", "p1", "p2", "d1", "d2") if x not in z]
    if miss:
        raise ValueError(f"REFUSED: {unit} lacks {miss}")
    return _aligned(z, D, unit), {"unit": unit, "complete_sha256": sha}


def load_reference(label, k, D=None, units_dir_=None):
    if label not in REFS:
        raise KeyError(label)
    unit = f"ref__s{k}__{label}"
    z, sha = load_unit_npz(unit, "reference.npz", units_dir_)
    return _aligned(z, D, unit), {"unit": unit, "complete_sha256": sha}


# ------------------------------------------------------------------ continuous sources (U, RAW-J beta 0.3)
def source_outputs(t):
    return {"p1": np.asarray(t["p1"]), "p2": np.asarray(t["p2"]), "hard1": np.asarray(t["d1"]),
            "hard2": np.asarray(t["d2"])}


def source_view_sets(t, D, meta=None, families=SOURCE_FAMILIES):
    """{family: views} of a continuous teacher release (dpc.audit.source_views, unchanged)."""
    return {fam: DA.source_views(t, D, fam, meta={**(meta or {})}) for fam in families}


# ------------------------------------------------------------------ references (E, F, F0)
def reference_view_sets(label, k, D, units_dir=None, with_outputs=False, arrays=None):
    """{family: views} of one admitted reference (dpc.baselines.reference_view_sets semantics, cbp unit store)."""
    if label not in REFS:
        raise KeyError(label)
    if arrays is None:
        z, prov = load_reference(label, k, D, units_dir)
    else:
        z, prov = {kk: np.asarray(v) for kk, v in arrays.items()}, {"unit": f"ref__s{k}__{label}",
                                                                    "complete_sha256": None}
        _aligned(z, D, prov["unit"])
    meta = {"kind": "reference", "label": label, "seed": k, "unit": prov["unit"], "osf_choice": OSF_CHOICES[label]}
    fams = list(SCORE_FAMILIES) + (["complete"] if label == "E" else ["cells"])
    sets = {}
    for fam in fams:
        if fam == "cells":
            X, T, chk = {}, {}, {}
            for i, K in ((1, DA.KS[0]), (2, DA.KS[1])):
                cells = np.asarray(z[f"cells{i}"], dtype=np.int64)
                r = np.asarray(z[f"r{i}"])
                ncell = int(r.shape[1])
                if not np.array_equal(r.argmax(1), cells) or not np.allclose(r.sum(1), 1.0):
                    raise ValueError(f"REFUSED: {label} r{i} is not the one-hot cell code")
                c = DA.token_decoder_check(cells, z[f"p{i}"], z[f"d{i}"])
                chk[f"v{i}"] = {**c, "alphabet": ncell}
                X[f"v{i}"] = DA.code_view(cells, z[f"p{i}"], z[f"d{i}"], ncell, K, D)
                T[f"v{i}"] = cells
            X["pair"] = np.hstack([X["v1"], X["v2"]])
            sets[fam] = {"family": "cells", "X": X, "tokens": T, "meta": {**meta, "family": "cells", "interface": chk}}
            continue
        V = DA.source_views(z, D, fam, meta={**meta})
        if label in ("F", "F0") and fam != "decisions":
            V = DA.finite_by_value(V)
        sets[fam] = V
    if not with_outputs:
        return sets
    out = {"p1": z["p1"], "p2": z["p2"], "hard1": z["d1"], "hard2": z["d2"]}
    return sets, out, prov


def baseline_ids():
    """Configuration ids of the continuous and reference baselines (cbp.run naming = qpc.run naming)."""
    return [f"SRC|{t}" for t in TEACHERS] + [f"REF|{r}" for r in REFS]


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
