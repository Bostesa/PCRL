"""Evidence on within-bin homogeneity T ⟂ H | (S, B) from 2018 aggregate count tables.

Input: Terminal 3's private fine-partition tables (/tmp/pcrl_objdiag_private/fineC_anchor_{a}.joblib;
SHA-256 values in its REUSABLE_INPUTS.json). These hold aggregate counts n(s, c, t) only, no person rows.
Only test statistics are printed.

Logic: C_fine = f(H_A). If the fine cells nest inside the coarse cells B, then T ⟂ H | (S,B) implies
T ⟂ C_fine | (S,B). A likelihood-ratio (G) test that rejects the latter is therefore evidence against
within-bin homogeneity at H_A level. A non-rejection establishes nothing.

Caveats: persons are clustered in households and the unweighted counts ignore survey weights, so the
asymptotic chi-square p-value is approximate (anti-conservative under positive intra-household
correlation). A design effect is reported for scale. The 2018 rows are development data.
"""
import itertools
import json
import math
import sys

import joblib
import numpy as np
from joblib import numpy_pickle as npk
from scipy.stats import chi2

_orig = npk.NumpyUnpickler.find_class


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self.__dict__["_state"] = s


def _fc(self, module, name):
    if module.startswith("experiments."):
        return type(name, (_Stub,), {})
    return _orig(self, module, name)


npk.NumpyUnpickler.find_class = _fc


def nesting(coarse, fine):
    """Return a fine->coarse map if summing fine cells reproduces coarse counts exactly."""
    nc, nf = coarse.shape[1], fine.shape[1]
    for assign in itertools.product(range(nc), repeat=nf):
        agg = np.zeros_like(coarse)
        for f, c in enumerate(assign):
            agg[:, c, :] += fine[:, f, :]
        if np.array_equal(agg, coarse):
            return list(assign)
    return None


def g_test(fine, assign):
    """G statistic for T ⟂ C_fine | (S, B) with counts n[s, f, t]."""
    G, df = 0.0, 0
    ns = fine.shape[0]
    for s in range(ns):
        for b in set(assign):
            cells = [f for f, c in enumerate(assign) if c == b]
            tab = fine[s][cells, :]  # (fine cells in b) x T
            n = tab.sum()
            if n == 0:
                continue
            rows, cols = tab.sum(1), tab.sum(0)
            exp = np.outer(rows, cols) / n
            m = tab > 0
            G += 2 * float(np.sum(tab[m] * np.log(tab[m] / exp[m])))
            df += (int((rows > 0).sum()) - 1) * (int((cols > 0).sum()) - 1)
    return G, df


def main():
    out = {}
    for a in (0, 1, 2):
        d = joblib.load(f"/tmp/pcrl_objdiag_private/fineC_anchor_{a}.joblib")
        for code in ("T0", "Ttask"):
            tabs = d["tables"][code]["support"]
            for role, attr in (("A", "SEX"), ("A", "RAC1P"), ("AB", "SEX")):
                coarse = tabs[f"{role}/{attr}"]["count"]
                fine = tabs[f"{role}/{attr}/fineC"]["count"]
                assign = nesting(coarse, fine)
                key = f"{a}/{code}/{role}/{attr}"
                if assign is None:
                    out[key] = {"nested": False}
                    continue
                G, df = g_test(fine, assign)
                n = int(fine.sum())
                out[key] = {"nested": True, "fine_to_coarse": assign, "N": n, "G": G, "df": df,
                            "p_value_asymptotic": float(chi2.sf(G, df)) if df > 0 else None,
                            "plugin_I_T_Cfine_given_S_B_nats": G / (2 * n),
                            "design_effect_needed_to_remove_significance": (G / chi2.isf(0.05, df)) if df > 0 else None}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
