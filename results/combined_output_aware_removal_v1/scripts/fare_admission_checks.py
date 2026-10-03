"""FARE admission checks (method-admission owner, 2026-10-03). Runs in the FARE environment:

    cd <worktree> && OAR_FARE_ROOT=... ~/PCRL_eval_cache_private/oar_v1/env_fare/bin/python \
        results/combined_output_aware_removal_v1/scripts/fare_admission_checks.py

Uses only (1) the official reproduction-gate output produced by the unmodified official entry point
``python -m src.tree.main --dataset ACSIncome-CA-2014 --max-k 50 --min-ni 100 --alpha 0.9 --val-split 0.3``
(public ACS data, written under ~/PCRL_eval_cache_private/oar_v1/gate/) and (2) synthetic data. No study data.

Writes notes/fare/FARE_ADMISSION_CHECKS.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from oar import fare_official as F  # noqa: E402

OUT = HERE.parents[1] / "notes" / "fare" / "FARE_ADMISSION_CHECKS.json"
SHIPPED_DP_UB = 0.15712399439471292  # fare/code/result/_eval/ACSIncome-CA-2014/tree.npy, k=50,ni=100,a=0.9,s=0.3
GATE_DIRS = {"official_build": "~/PCRL_eval_cache_private/oar_v1/gate/result",
             "fixed_build": "~/PCRL_eval_cache_private/oar_v1/gate/result_fixed"}
GATE_REL = "ACSIncome-CA-2014/tree/k=50,ni=100,a=0.9,s=0.3/embeddings.npy"


def gate_checks():
    AB = F._load_alphabeta()
    out = {"shipped_dp_ub": SHIPPED_DP_UB}
    for tag, d in GATE_DIRS.items():
        e = np.load(Path(d).expanduser() / GATE_REL, allow_pickle=True).item()
        out[f"{tag}_dp_ub"] = float(e["dp_ub"])
        out[f"{tag}_bit_exact"] = float(e["dp_ub"]) == SHIPPED_DP_UB
    # Re-derive the official binary bound through the wrapper core from the saved official embeddings.
    # main.py saves z_train = vstack(train, val) with n_val = int(0.3 * n_trainval) taken from the END.
    e = np.load(Path(GATE_DIRS["fixed_build"]).expanduser() / GATE_REL, allow_pickle=True).item()
    zt, ct = e["z_train"], e["c_train"]
    n_val = int(0.3 * len(ct))
    z_tr, c_tr, z_va, c_va = zt[:-n_val], ct[:-n_val], zt[-n_val:], ct[-n_val:]
    k = len(np.unique(zt, axis=0))
    med = F.official_pair_bound(AB, k, z_tr, c_tr, z_va, c_va, e["z_test"], e["c_test"], 0.05, 0.05 / 10.0,
                                0.05 / 10.0)
    allz = np.vstack([z_tr, z_va, e["z_test"]])
    _, inv = np.unique(allz, axis=0, return_inverse=True)
    inv = inv.reshape(-1, 1).astype(np.float64)
    a, b = len(z_tr), len(z_tr) + len(z_va)
    ids = F.official_pair_bound(AB, k, inv[:a], c_tr, inv[a:b], c_va, inv[b:], e["c_test"], 0.05, 0.05 / 10.0,
                                0.05 / 10.0)
    out.update(n_cells=k, wrapper_core_on_medians=med["ub"], wrapper_core_on_cell_ids=ids["ub"],
               wrapper_core_bit_exact=(med["ub"] == SHIPPED_DP_UB and ids["ub"] == SHIPPED_DP_UB),
               empirical_test=med["empirical_test"],
               note="medians and cell ids give the identical bound: the adversary only uses np.unique(z) cells")
    return out


def official_multigroup_budget_pitfall():
    """main.py:404-430 divides err_budget by #pairs but keeps eps_glob = eps_ab = 0.005; for >= 4 groups the
    Lemma 5.2 budget is negative, Clopper-Pearson returns NaN and max(total, nan) keeps the old total."""
    AB = F._load_alphabeta()
    rng = np.random.RandomState(0)
    k = 4
    rows = {}
    for nm, n in (("train", 4000), ("val", 4000), ("test", 4000)):
        z = rng.randint(0, k, n).astype(float).reshape(-1, 1)
        c = (rng.rand(n) < 0.3 + 0.1 * z.ravel()).astype(int)
        rows[nm] = (z, c)
    res = {}
    for n_groups in (2, 3, 4, 5):
        n_pairs = n_groups * (n_groups - 1) // 2
        eps_pair = 0.05 / n_pairs
        official = F.official_pair_bound(AB, k, *rows["train"], *rows["val"], *rows["test"], eps_pair, 0.005, 0.005)
        scaled = F.official_pair_bound(AB, k, *rows["train"], *rows["val"], *rows["test"], eps_pair,
                                       eps_pair / 10.0, eps_pair / 10.0)
        res[str(n_groups)] = {"pairs": n_pairs, "eps_pair": eps_pair,
                              "official_constants_lemma52_budget": eps_pair - 0.01,
                              "official_constants_ub": official["ub"],
                              "official_main_py_max_accumulator": max(0, official["ub"]),
                              "wrapper_scaled_ub": scaled["ub"]}
    return res


def main():
    out = {"verify": F._verify_impl(), "gate": gate_checks(),
           "multigroup_budget_pitfall_same_synthetic_pair": official_multigroup_budget_pitfall()}
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, float) and o != o:
            return "NaN"
        return o
    out = clean(out)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
