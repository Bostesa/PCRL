"""Official reference score releases E (LEACE), F (FARE) and F0 (no-fairness FARE) for the decision-preserving
compression study (audit/baseline owner).

Nothing is refit or reselected here. The admitted osf references are used exactly as dpc.admit admits them
(dpc.admit.reference(label, k), stored by the lead's admission stage as unit ref__s{k}__<label>/reference.npz with
row_id, c1, c2, p1, p2, d1, d2, r1, r2 and, for F / F0, cells1, cells2). The osf configuration choices are inherited
and recorded (OSF_CHOICES): E = official LEACE on U's encoder features (osf status INFEASIBLE_CONTROL on the osf gates);
F = FARE configuration 1 for BOTH purposes, where the income purpose was SELECTED by the osf rule and the OCCUPATION
purpose had NO_FEASIBLE_CONFIGURATION (configuration 1 kept as the closest, DESCRIPTIVE point; osf arm status
NO_FEASIBLE_NOMINEE); F0 = the zero-fairness twin at F's configurations (osf status INFEASIBLE_CONTROL). In this study
eligibility is decided only by the fixed dpc task gates (dpc.utility.gate) against U on INNER_SELECTION; no reference
is excluded by name, and an osf status never makes a reference eligible or ineligible here.

View families (`reference_view_sets`), every one audited with the SAME FINAL slate as every other contract:
  interface  [c_i, p_i] (centred logits + probabilities; the score interface; PRIMARY)
  scores     c_i              probs   p_i              decisions   one-hot(d_i) (finite, tokens d_i)
  cells      F / F0 only: the FARE cell code as a finite view [one-hot(cell over the n_cells alphabet, occurrence-ordered
             columns), p_i, one-hot(d_i)], tokens = exact cell ids (cell-conditional readers on the cell identity)
  complete   E only: [r_i, c_i] (LEACE features + centred logits; the DESCRIPTIVE full-feature contract view). For F /
             F0 the full-feature view r_i IS the one-hot cell code, so `cells` is that view.
F / F0 score views are functions of the cell and take finitely many values; they are audited as finite views keyed
on the exact float64 value identity (dpc.audit.finite_by_value), so the cell-conditional readers apply to them too.
E's score views are continuous (no cell readers).

API for the lead
  reference_view_sets(label, k, D, units_dir=None, with_outputs=False) -> {family: views}
      (with_outputs: (sets, {p1, p2, hard1, hard2}, provenance))
  reference_candidates(k, units_dir=None) -> {"E": {...}, "F": {...}, "F0": {...}}: per reference the inner AUCs of
      every family {family: {v1, v2, pair}} from the inner units the lead ran (inner__ref__s{k}__<label>), the primary
      family, INNER_SELECTION utility, the dpc gate vs U at seed k (needs inner__tea__s{k}__U), the inherited osf
      choice, and the families eligible to enter C_global / T* as score releases (all score-only families; `complete`
      is descriptive).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dpc import audit as AU

REFS = ("E", "F", "F0")
PRIMARY_FAMILY = "interface"
SCORE_FAMILIES = ("interface", "scores", "probs", "decisions")
OSF_CHOICES = {
    "E": {"method": "official LEACE (EleutherAI concept-erasure, pinned wrapper) on U's encoder features, concept "
                    "one-hot SEX, fit on OSF_DEFENSE_FIT; heads refit (osf)", "osf_status": "INFEASIBLE_CONTROL",
          "configs": None},
    "F": {"method": "official FARE (eth-sri/fare, pinned), configuration 1 per purpose", "osf_status":
          "NO_FEASIBLE_NOMINEE", "configs": [1, 1],
          "per_purpose": {"income": "SELECTED (osf rule: gates on every seed, lowest seed-mean inner local AUC)",
                          "occupation_group": "NO_FEASIBLE_CONFIGURATION; configuration 1 kept as the closest "
                                              "(largest min-over-seeds worst-gate margin), DESCRIPTIVE in osf"}},
    "F0": {"method": "official FARE zero-fairness twin (gamma = 0) at F's configurations", "osf_status":
           "INFEASIBLE_CONTROL", "configs": [1, 1]},
}


def _ref_arrays(label, k, D, units_dir=None):
    """Arrays of ref__s{k}__<label> (hash-complete unit written by the lead's admission stage)."""
    unit = f"ref__s{k}__{label}"
    z, sha = AU._load_npz(unit, "reference.npz", units_dir)
    if not np.array_equal(z["row_id"], D["row_id"]):
        raise ValueError(f"REFUSED: {unit} rows are not aligned with D")
    return z, {"unit": unit, "complete_sha256": sha}


def reference_view_sets(label, k, D, units_dir=None, with_outputs=False, arrays=None):
    """{family: views} of one reference (see module docstring). arrays: optional in-memory reference dict (tests)."""
    if label not in REFS:
        raise KeyError(label)
    if arrays is None:
        z, prov = _ref_arrays(label, k, D, units_dir)
    else:
        z, prov = {kk: np.asarray(v) for kk, v in arrays.items()}, {"unit": f"ref__s{k}__{label}",
                                                                    "complete_sha256": None}
    meta = {"kind": "reference", "label": label, "seed": k, "unit": prov["unit"], "osf_choice": OSF_CHOICES[label]}
    fams = list(SCORE_FAMILIES) + (["complete"] if label == "E" else ["cells"])
    sets = {}
    for fam in fams:
        if fam == "cells":
            X, T, chk = {}, {}, {}
            for i, K in ((1, AU.KS[0]), (2, AU.KS[1])):
                cells = np.asarray(z[f"cells{i}"], dtype=np.int64)
                ncell = int(np.asarray(z[f"r{i}"]).shape[1])
                r = np.asarray(z[f"r{i}"])
                if not np.array_equal(r.argmax(1), cells) or not np.allclose(r.sum(1), 1.0):
                    raise ValueError(f"REFUSED: {label} r{i} is not the one-hot cell code")
                c = AU.token_decoder_check(cells, z[f"p{i}"], z[f"d{i}"])
                chk[f"v{i}"] = {**c, "alphabet": ncell}
                X[f"v{i}"] = AU.code_view(cells, z[f"p{i}"], z[f"d{i}"], ncell, K, D)
                T[f"v{i}"] = cells
            X["pair"] = np.hstack([X["v1"], X["v2"]])
            sets[fam] = {"family": "cells", "X": X, "tokens": T, "meta": {**meta, "family": "cells", "interface": chk}}
            continue
        V = AU.source_views(z, D, fam, meta={**meta})
        if label in ("F", "F0") and fam != "decisions":
            V = AU.finite_by_value(V)
        sets[fam] = V
    if not with_outputs:
        return sets
    out = {"p1": z["p1"], "p2": z["p2"], "hard1": z["d1"], "hard2": z["d2"]}
    return sets, out, prov


def _inner_rec(name, units_dir=None):
    d = AU.udir(name, units_dir)
    from rgj import finalize as FN
    if not FN.unit_complete(d):
        return None
    return json.loads((d / "record.json").read_text())


def reference_candidates(k, units_dir=None):
    """Selection-facing summary of E, F, F0 at seed k from the inner units the lead ran (inner roles only)."""
    from dpc import utility as UT
    u = _inner_rec(f"inner__tea__s{k}__U", units_dir)
    out = {}
    for lab in REFS:
        r = _inner_rec(f"inner__ref__s{k}__{lab}", units_dir)
        e = {"unit": f"ref__s{k}__{lab}", "inner_unit": f"inner__ref__s{k}__{lab}", "osf_choice": OSF_CHOICES[lab],
             "primary_family": PRIMARY_FAMILY, "score_release_families": list(SCORE_FAMILIES) +
             (["cells"] if lab != "E" else []), "descriptive_families": ["complete"] if lab == "E" else [],
             "status": "MISSING_INNER_UNIT" if r is None else "AUDITED"}
        if r is not None:
            fams = {r["primary_family"]: r["recovery"], **r.get("families", {})}
            e["inner"] = {f: dict(v["auc"]) for f, v in fams.items()}
            e["inner_ce"] = {f: dict(v["ce"]) for f, v in fams.items()}
            e["selected"] = {f: dict(v["selected"]) for f, v in fams.items()}
            e["coverage"] = {f: v.get("coverage") for f, v in fams.items()}
            e["utility"] = r["utility"]
            if u is not None:
                g = UT.gate(r["utility"], u["utility"])
                e["gate"] = g
                e["task_feasible"] = g["pass"]
            else:
                e["gate"], e["task_feasible"] = None, None
                e["status"] = "AUDITED (U inner unit missing: gate not computed)"
        out[lab] = e
    return out


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
