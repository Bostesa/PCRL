"""Generic cross-fitting and strength-selection driver of the family modules (PCRL strong-anchor release campaign; label-free code:
the labels arrive as arguments from the family entry point that read them through `rfa.common.load`).

A family supplies plain functions (the designers' interface):
    fit(Z, y_inc, y_occ, sex, unit, version, strength, **kw) -> state        (sex may be None for a SEX-blind configuration)
    release(state, Z) -> (P_inc (n, 2), P_occ (n, 5))                        (deterministic)
and, optionally, `fit_shared(...)` + `release_at(state, Z, strength)` when the strength acts on the release only (one fit per fold
serves every strength).

`crossfit` produces MODEL_FIT out-of-fold releases on household folds (tag `RFA_xf_<family>_S<v>`, 5 folds): every strength of a
path and the twin are released on identical held-out people from fits on the other folds (equal effort for candidate and controls).
`path_spends` turns them into paired spends over the twin (famkit.spend, household-clustered SEs, men's / women's occupation Brier
when SEX was read); `famkit.select_strength` applies the predeclared rule. Nothing here reads MODEL_SELECT or any label file.
"""
from __future__ import annotations
import time

import numpy as np

from . import famkit as FK

N_XF = 5


def folds(unit, family, version, k=N_XF):
    return FK.household_folds(unit, k, f'RFA_xf_{family}_S{int(version)}')


def crossfit(fit, release, Z, lab, unit, version, family, strengths, use_sex=True, fit_kw=None, shared=None, log=None):
    """{strength: (P_inc_oof, P_occ_oof)} on MODEL_FIT. `shared` = (fit_shared, release_at): one fit per fold for all strengths."""
    f = folds(unit, family, version)
    n = len(unit)
    out = {s: (np.zeros((n, 2)), np.zeros((n, 5))) for s in strengths}
    sex = lab.get('sex') if use_sex else None
    nfits = 0
    t0 = time.process_time()
    for j in range(N_XF):
        tr, te = f != j, f == j
        if np.intersect1d(unit[tr], unit[te]).size:
            raise AssertionError('household in both parts of a cross-fitting fold')
        args = (Z[tr], lab['income'][tr], lab['occupation'][tr], None if sex is None else sex[tr], unit[tr], version)
        if shared is not None:
            st = shared[0](*args, **(fit_kw or {}))
            nfits += 1
            for s in strengths:
                pi, po = shared[1](st, Z[te], s)
                out[s][0][te], out[s][1][te] = pi, po
        else:
            for s in strengths:
                st = fit(*args, s, **(fit_kw or {}))
                nfits += 1
                pi, po = release(st, Z[te])
                out[s][0][te], out[s][1][te] = pi, po
        if log:
            log(f'crossfit fold {j} done ({time.process_time() - t0:.1f} CPU-s)')
    return out, {'n_crossfit_fits': nfits, 'cpu_s': time.process_time() - t0, 'fold_tag': f'RFA_xf_{family}_S{int(version)}'}


def path_spends(oof, twin_key, lab, unit, with_sex=True):
    """Paired MODEL_FIT out-of-fold spends of every entry over the twin entry (famkit.spend)."""
    tw = FK.person_losses(*oof[twin_key], lab['income'], lab['occupation'])
    sex = lab.get('sex') if with_sex else None
    out = {}
    for k, (pi, po) in oof.items():
        out[k] = FK.spend(FK.person_losses(pi, po, lab['income'], lab['occupation']), tw, unit, sex)
    return out


def describe(oof, lab):
    """Absolute out-of-fold task losses per entry (descriptive)."""
    res = {}
    for k, (pi, po) in oof.items():
        L = FK.person_losses(pi, po, lab['income'], lab['occupation'])
        res[str(k)] = {m: float(v.mean()) for m, v in L.items()}
    return res
