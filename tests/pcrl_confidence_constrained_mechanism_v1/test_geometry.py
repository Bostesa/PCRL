"""Synthetic tests of ccm.geometry (role C; no real data, no labels, no SEX).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q tests/pcrl_confidence_constrained_mechanism_v1/test_geometry.py

Brute-force references below re-derive each metric with plain loops over ccm.guard's public predicates.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from ccm import geometry as GE
from ccm import guard as GD

D, B = GD.CONFIG["d"], GD.CONFIG["b"]


def clustered(n, K, seed, n_centres=6, scale=0.0015):
    """Synthetic reference vectors: a few centres plus small multiplicative noise (so that bins merge)."""
    rng = np.random.default_rng(seed)
    C = rng.dirichlet(np.ones(K) * 1.5, size=n_centres)
    P = C[rng.integers(0, n_centres, n)] * np.exp(rng.normal(0, scale, (n, K)))
    return P / P.sum(axis=1, keepdims=True)


def binary(n, seed, lo=0.55, hi=0.99):
    rng = np.random.default_rng(seed)
    a = np.concatenate([rng.uniform(lo, hi, n // 2), 1 - rng.uniform(lo, hi, n - n // 2)])
    return np.stack([a, 1 - a], axis=1)


def test_cover_order_is_decreasing_max_then_index():
    P = np.array([[0.6, 0.4], [0.9, 0.1], [0.4, 0.6], [0.1, 0.9], [0.6, 0.4]])
    assert GE.cover_order(P).tolist() == [1, 3, 0, 2, 4]
    assert GE.cover_rank(P)[GE.cover_order(P)].tolist() == list(range(5))


@pytest.mark.parametrize("contract,scale", [("G", 0.0015), ("G_exp", 0.02)])
def test_f1_cover_is_a_certified_partition(contract, scale):
    P = clustered(300, 3, 0, scale=scale)
    dec = GD.decisions(P)
    f1 = GE.f1_cover(P, contract=contract)
    seen = np.zeros(len(P), dtype=int)
    for bb in f1["bins"]:
        seen[bb["members"]] += 1
        assert np.all(dec[bb["members"]] == bb["class"])
        assert np.all(GD.accepts(bb["q"], P[bb["members"]], contract, dec=bb["class"]))
        assert np.all(GD.check(bb["q"], P[bb["members"]], contract, dec=bb["class"])["ok"])
    assert f1["summary"]["n_uncoverable"] == 0 and np.all(seen == 1)
    assert np.all(f1["bin_of_row"] >= 0)
    s = f1["summary"]
    assert s["n_bins"] == len(f1["bins"]) and s["compression_ratio"] == s["n_bins"] / len(P)
    assert s["n_bins"] < len(P)                                   # the clustered fixture does merge


def _bruteforce_greedy(P, dec, contract):
    bins = []
    for r in GE.cover_order(P).tolist():
        for bb in bins:
            if bb[0] != dec[r]:
                continue
            q, cert = GD.bin_representative(P[bb[1] + [r]], dec=int(dec[r]), contract=contract)
            if cert["status"] == "CERTIFIED":
                bb[1].append(r)
                break
        else:
            bins.append((int(dec[r]), [r]))
    return sorted(sorted(m) for _, m in bins)


@pytest.mark.parametrize("contract,n,scale", [("G", 150, 0.0015), ("G_exp", 70, 0.02)])
def test_f1_shortcuts_do_not_change_the_greedy_rule(contract, n, scale):
    """Prefilters, the sufficient check and the cheap start point give the same cover as the plain rule 'join the first
    bin of the class whose members + row bin_representative certifies'."""
    P = clustered(n, 3, 1, scale=scale)
    dec = GD.decisions(P)
    f1 = GE.f1_cover(P, contract=contract)
    mine = sorted(sorted(bb["members"].tolist()) for bb in f1["bins"])
    assert mine == _bruteforce_greedy(P, dec, contract)


@pytest.mark.parametrize("contract", ["G", "G_exp"])
def test_f2_packing_is_pairwise_infeasible_maximal_and_below_f1(contract):
    P = clustered(250, 3, 2, scale=0.004 if contract == "G" else 0.05)
    dec = GD.decisions(P)
    f2 = GE.f2_packing(P, contract=contract)
    f1 = GE.f1_cover(P, contract=contract)
    rank = GE.cover_rank(P)
    for c, rows in f2["packing"].items():
        rows = rows.tolist()
        for i, j in itertools.combinations(rows, 2):
            assert GE.pairwise_infeasible(P[i], P[j][None, :], contract)[0]
        for r in np.flatnonzero(dec == c).tolist():
            if r in rows:
                continue
            earlier = [s for s in rows if rank[s] < rank[r]]
            assert any(not GE.pairwise_infeasible(P[r], P[s][None, :], contract)[0] for s in earlier)
        assert len(rows) <= f1["summary"]["per_class"][c]["n_bins"]
    if contract == "G":
        for i, j in itertools.combinations(range(40), 2):           # the registered closed form, conservative margin
            assert GE.pairwise_infeasible(P[i], P[j][None, :])[0] == \
                (float(np.sum(np.maximum(P[i], P[j]))) > math.exp(D) * (1 + 1e-12))


def test_identical_rows_one_bin_per_class():
    P = np.vstack([np.repeat([[0.7, 0.2, 0.1]], 30, axis=0), np.repeat([[0.1, 0.3, 0.6]], 20, axis=0)])
    for contract in ("G", "G_exp"):
        f1 = GE.f1_cover(P, contract=contract)
        assert f1["summary"]["n_bins"] == 2
        assert GE.f2_packing(P, contract=contract)["summary"]["per_class"] == {0: 1, 2: 1}
        f3 = GE.f3_capacity_coverage(P, f1, cap=1)
        assert f3["summary"]["coverage"] == 1.0
        assert GE.f5_class_eligibility(P, contract=contract)["summary"]["eligible"]


def test_f3_greedy_capacity_and_canonical_representatives():
    P = clustered(300, 3, 3)
    dec = GD.decisions(P)
    f1 = GE.f1_cover(P)
    big = GE.f3_capacity_coverage(P, f1, cap=10 ** 6)
    assert big["summary"]["coverage"] == 1.0 and big["summary"]["n_canonical_not_certified"] == 0
    one = GE.f3_capacity_coverage(P, f1, cap=1)
    best = sum(max(bb["size"] for bb in f1["bins"] if bb["class"] == c) for c in set(dec.tolist()))
    assert one["summary"]["covered_rows"] == best
    for q, c, i in zip(one["reps"], one["rep_classes"], one["selected_ids"]):
        assert np.all(GD.accepts(q, P[f1["bins"][i]["members"]], "G", dec=c))
    assert list(one["selected_ids"]) == sorted(one["selected_ids"])     # registered cover (creation) order


def test_f3u_matches_bruteforce_and_bounds_every_code():
    for contract, scale in (("G", 0.003), ("G_exp", 0.04)):
        P = clustered(150, 3, 4, scale=scale)
        dec = GD.decisions(P)
        N = GE.neighbourhood_counts(P, contract=contract, chunk=7)
        for r in range(len(P)):
            same = np.flatnonzero(dec == dec[r])
            assert N[r] == sum(1 for s in same if not GE.pairwise_infeasible(P[r], P[s][None, :], contract)[0])
        for cap in (1, 2, 5):
            f3u = GE.f3u_coverage_bound(P, cap, contract=contract)["summary"]["F3u"]
            want = sum(np.sort(N[dec == c])[::-1][:cap].sum() for c in set(dec.tolist())) / len(P)
            assert f3u == min(1.0, want)
            greedy = GE.f3_capacity_coverage(P, GE.f1_cover(P, contract=contract), cap)["summary"]["coverage"]
            assert greedy <= f3u + 1e-12
            pb = GE.packing_coverage_bound(GE.f2_packing(P, contract=contract), cap, len(P))
            assert greedy <= pb + 1e-12
    x = (math.exp(D) - 1.0) / 2.0
    Pb = np.array([[0.6 + x, 0.4 - x], [0.6 - x, 0.4 + x]])            # sum max = e^d: counted (conservative)
    assert GE.neighbourhood_counts(Pb).tolist() == [2, 2]


def test_income_intervals_match_the_construction_check():
    """Each row's interval [lo, hi] of s = q_1 is exactly the set where (1 - s, s) meets the tightened targets."""
    rng = np.random.default_rng(5)
    P = binary(400, 5, 0.5, 0.999)
    dec = GD.decisions(P)
    lo, hi = GE.income_intervals(P, dec)
    for i in range(len(P)):
        for s in rng.uniform(P[i, 1] - 0.01, P[i, 1] + 0.01, 25):
            if min(abs(s - lo[i]), abs(s - hi[i])) < 1e-12:
                continue
            q = np.array([1.0 - s, s])
            assert bool(GD.check_construction(q, P[i], "G", dec=dec[i])) == bool(lo[i] <= s <= hi[i])


def _best_windows_bruteforce(lo, hi, cap):
    n = len(lo)
    feas = {(i, j) for i in range(n) for j in range(i, n) if max(lo[i:j + 1]) <= min(hi[i:j + 1])}
    best = 0
    for k in range(1, cap + 1):
        for combo in itertools.combinations(sorted(feas), k):
            cells = [x for (i, j) in combo for x in range(i, j + 1)]
            if len(cells) == len(set(cells)):
                best = max(best, len(cells))
    return best


def test_f3x_income_dp_is_exact_and_certified():
    rng = np.random.default_rng(6)
    for it in range(25):
        n = int(rng.integers(4, 11))
        a = np.sort(rng.choice([rng.uniform(0.6, 0.62, n), rng.uniform(0.9, 0.93, n)]))
        P = np.stack([a, 1 - a], axis=1)
        lo, hi = GE.income_intervals(P, np.zeros(n, dtype=int))
        rows = np.lexsort((np.arange(n), P[:, 1]))
        for cap in (1, 2, 3):
            val, wins = GE._windows_dp(lo[rows], hi[rows], cap)
            assert val == _best_windows_bruteforce(lo[rows], hi[rows], cap), (it, cap)
            assert sum(j - i + 1 for i, j in wins) == val and len(wins) <= cap
    P = binary(2000, 7)
    f1 = GE.f1_cover(P)
    for cap in (2, 8):
        fx = GE.f3x_income(P, cap)
        s = fx["summary"]
        assert s["n_bins_certified"] == s["n_bins"] and s["coverage"] == s["F3x"]
        assert s["F3x"] <= s["F3x_untightened"] + 1e-12
        assert GE.f3_capacity_coverage(P, f1, cap)["summary"]["coverage"] <= s["F3x"] + 1e-12
        assert s["F3x"] <= GE.f3u_coverage_bound(P, cap)["summary"]["F3u"] + 1e-12
        assert s["F3x"] <= GE.packing_coverage_bound(GE.f2_packing(P), cap, len(P)) + 1e-12
        assert np.all(np.diff(fx["first_rank"]) > 0)                    # registered cover order of representatives
        for q, c in zip(fx["reps"], fx["rep_classes"]):
            assert abs(q.sum() - 1.0) <= 1e-12 and int(np.argmax(q)) == c


def test_f4_first_registered_representative_fallback_and_ties():
    reps = np.array([[0.70, 0.30], [0.701, 0.299], [0.2, 0.8]])
    cls = np.array([0, 0, 1])
    held = np.array([[0.7005, 0.2995],      # served by both class-0 reps -> the FIRST
                     [0.95, 0.05],          # unseen, far: fallback
                     [0.5, 0.5],            # tied top: nudged fallback
                     [0.2002, 0.7998]])     # class 1, served
    f4 = GE.f4_heldout_fallback(held, reps, cls)
    ok = GD.check_release(reps[[0, 1]], held[0], dec=0)["ok"]
    assert ok.all() and f4["served_by"].tolist() == [0, -1, -1, 2]
    assert f4["fallback"].tolist() == [False, True, True, False]
    assert np.array_equal(f4["release"][1], held[1])                    # fallback = Ucal itself
    assert f4["release"][2, 0] > f4["release"][2, 1]                    # eta-nudge breaks the tie toward dec 0
    s = f4["summary"]
    assert s["fallback_rate"] == 0.5 and s["n_released_failing_contract"] == 0
    assert s["fallback_release"]["n_tied"] == 1 and sum(s["blocking"].values()) == 2
    f4b = GE.f4_heldout_fallback(np.array([[0.3, 0.7]]), reps[:2], cls[:2])
    assert f4b["summary"]["blocking"] == {"NO_TOKEN_FOR_CLASS": 1}


def test_f4_fitting_rows_are_served_when_every_bin_is_registered():
    P = clustered(200, 3, 8)
    f1 = GE.f1_cover(P)
    f3 = GE.f3_capacity_coverage(P, f1, cap=10 ** 6)
    f4 = GE.f4_heldout_fallback(P, f3["reps"], f3["rep_classes"])
    assert f4["summary"]["fallback"] == 0


def test_f5_class_and_decision_only():
    tight = np.repeat([[0.8, 0.15, 0.05]], 10, axis=0) * np.exp(np.random.default_rng(9).normal(0, 1e-4, (10, 3)))
    tight /= tight.sum(axis=1, keepdims=True)
    assert GE.f5_class_eligibility(tight)["summary"]["eligible"]
    spread = clustered(100, 3, 9, scale=0.05)
    f5 = GE.f5_class_eligibility(spread)
    assert not f5["summary"]["eligible"]
    assert "INFEASIBLE_NLL" in {v["status"] for v in f5["summary"]["per_class"].values()}
    assert GE.decision_only_record()["eligible"] is False


def test_run_geometry_schema_and_go_inputs():
    P2 = binary(600, 10)
    H2 = binary(200, 11)
    r2 = GE.run_geometry(P2, H2, GD.capacity_for(K=2))
    assert r2["F3"]["method"] == "income_exact_dp" and r2["go_inputs"]["F3x"] == r2["F3"]["F3x"]
    assert set(r2["go_inputs"]) == {"F3_coverage", "F4_fallback_rate", "F3u", "F3u_packing", "F3u_registered", "F3x"}
    g = r2["go_inputs"]
    assert g["F3u_registered"] == min(g["F3u"], g["F3u_packing"])
    assert g["F3_coverage"] <= g["F3u_registered"] + 1e-12 and g["F3x"] <= g["F3u_registered"] + 1e-12
    assert r2["go_inputs"]["F3_coverage"] == r2["F3"]["coverage"]
    assert r2["n_tied_fit"] == 0 and r2["n_tied_held"] == 0
    P3 = clustered(300, 3, 12)
    r3 = GE.run_geometry(P3, clustered(100, 3, 12), 4)
    assert r3["go_inputs"]["F3x"] is None and r3["F3"]["metric"] == "F3_capacity_coverage"
    assert r3["go_inputs"]["F3_coverage"] == r3["F3_greedy"]["coverage"]
    re = GE.run_geometry(P2, H2, 8, contract="G_exp")
    assert re["go_inputs"]["F3x"] is None and re["F3u"]["neighbourhood"].startswith("G_exp")
    for r in (r2, r3, re):
        json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, allow_nan=False)


def test_packing_bound_is_valid_against_exact_income_dp():
    # F3u_packing and F3u are upper bounds on any admissible code; the exact K = 2 DP coverage never exceeds them
    for seed in (21, 22, 23):
        P = binary(800, seed)
        r = GE.run_geometry(P, binary(100, seed + 100), GD.capacity_for(K=2))
        g = r["go_inputs"]
        assert g["F3x"] <= g["F3u_packing"] + 1e-12 and g["F3x"] <= g["F3u"] + 1e-12


def test_run_go_rule_classifies_failures():
    from ccm import run as R
    rows = {"s0|r1": {"go_inputs": {"F3_coverage": 0.2, "F4_fallback_rate": 0.7, "F3x": 0.21, "F3u": 1.0,
                                    "F3u_packing": 0.3, "F3u_registered": 0.3}},
            "s0|r2": {"go_inputs": {"F3_coverage": 0.5, "F4_fallback_rate": 0.01, "F3x": None, "F3u": 1.0,
                                    "F3u_packing": 0.99, "F3u_registered": 0.99}},
            "s1|r1": {"go_inputs": {"F3_coverage": 0.97, "F4_fallback_rate": 0.01, "F3x": 0.97, "F3u": 1.0,
                                    "F3u_packing": 1.0, "F3u_registered": 1.0}}}
    out = R.go_rule(rows)
    assert out["go"] is False
    assert out["rows"]["s0|r1"]["coverage_failure_intrinsic"] and out["rows"]["s0|r1"]["fallback_dominated"]
    assert out["rows"]["s0|r2"]["coverage_failure_incomplete"] and not out["rows"]["s0|r2"]["fallback_dominated"]
    assert out["rows"]["s1|r1"]["ok"] and out["any_incomplete"] and not out["all_coverage_failures_intrinsic"]
    assert R.go_rule({"s1|r1": rows["s1|r1"]})["go"] is True
