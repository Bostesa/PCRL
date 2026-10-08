"""Synthetic tests of ccm.oracle (role C; no real data, no labels, no SEX arrays from data).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q tests/pcrl_confidence_constrained_mechanism_v1/test_oracle.py

Every law here is a synthetic fixture built in this file, EXCEPT test_joint_headroom_reproduces_construction_values,
which (at role A's request) loads only the JOINT_HEADROOM entry of the pinned TOY_LAWS.json and compares the oracle with
role F's construction values (coalition MI 0.3230 nats for identity / local / seq12 / seq21, 0.1853 for joint, unique
joint argmin r1 {x0},{x1,x2},{x3}, r2 {x0,x1,x2},{x3}). Independent references use plain loops.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ccm import guard as GD
from ccm import oracle as OR

TOY = Path(__file__).resolve().parents[2] / "results" / "pcrl_confidence_constrained_mechanism_v1" / "TOY_LAWS.json"
TOL = 1e-12


def loop_mi_ba(tokens, Px, Ps1):
    """Independent plug-in MI (nats) and Bayes accuracy by dictionaries."""
    J = {}
    for t, px, s1 in zip(tokens, Px, Ps1):
        J.setdefault(t, [0.0, 0.0])
        J[t][0] += px * (1 - s1)
        J[t][1] += px * s1
    ps = [sum(v[0] for v in J.values()), sum(v[1] for v in J.values())]
    mi = 0.0
    for v in J.values():
        pt = v[0] + v[1]
        for s in (0, 1):
            if v[s] > 0:
                mi += v[s] * math.log(v[s] / (pt * ps[s]))
    return mi, sum(max(v) for v in J.values())


def random_law(rng, n=5, dup=True):
    a = 0.895 + np.cumsum(rng.uniform(0.0, 0.004, n))
    p1 = np.stack([a, 1 - a], axis=1)
    sh = rng.uniform(-0.004, 0.004, n)
    p2 = np.stack([0.8 + sh, 0.15 - sh, np.full(n, 0.05)], axis=1)
    if dup and n > 2:
        p2[2] = p2[0]
    return OR.validate_law(rng.dirichlet(np.ones(n)), rng.uniform(0.05, 0.95, n), p1, p2)


def test_set_partitions_are_bell_and_lexicographic():
    bell = [1, 1, 2, 5, 15, 52, 203, 877, 4140]
    for m in range(9):
        rgs = list(OR.set_partitions_rgs(m))
        assert len(rgs) == bell[m] and len(set(rgs)) == bell[m] and rgs == sorted(rgs)


def test_measures_match_independent_loops():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 9))
        Px = rng.dirichlet(np.ones(n))
        Ps1 = rng.uniform(0, 1, n)
        t = rng.integers(0, 4, n)
        Pxs = np.stack([Px * (1 - Ps1), Px * Ps1], axis=1)
        m = OR.measures_tokens(t, Pxs)
        mi, ba = loop_mi_ba(t.tolist(), Px, Ps1)
        assert abs(m["mi"] - mi) < 1e-12 and abs(m["bayes_acc"] - ba) < 1e-12
    Pxs = np.array([[0.25, 0.0], [0.0, 0.25], [0.25, 0.0], [0.0, 0.25]])
    assert abs(OR.measures_tokens([0, 1, 2, 3], Pxs)["mi"] - math.log(2)) < 1e-15
    assert OR.measures_tokens([0, 0, 0, 0], Pxs) == {"mi": 0.0, "bayes_acc": 0.5}


def test_equal_reference_vectors_always_share_a_token():
    rng = np.random.default_rng(1)
    law = random_law(rng, 5)
    rep = OR.run_oracle(law)
    assert rep["n_values"] == {1: 5, 2: 4}
    for name, arm in rep["arms"].items():
        if "rgs" in arm:
            blocks = arm["partitions"][2]
            assert any({"0", "2"} <= set(b) for b in blocks), name


def test_identity_parity_when_nothing_is_mergeable():
    """Zero privacy strength path: all values distinct and pairwise NLL-infeasible -> every deterministic arm is the
    identity partition and equals the identity release."""
    a = np.array([0.60, 0.70, 0.80, 0.95])
    p1 = np.stack([a, 1 - a], axis=1)
    p2 = np.array([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    law = OR.validate_law(np.full(4, 0.25), np.array([0.1, 0.9, 0.4, 0.6]), p1, p2)
    rep = OR.run_oracle(law)
    assert rep["n_admissible_partitions"] == {1: 1, 2: 1}
    for name in ("task_only", "local", "seq12_nonadaptive", "seq21_nonadaptive", "joint"):
        arm = rep["arms"][name]
        assert arm["rgs"] == {1: [0, 1, 2, 3], 2: [0, 1, 2, 3]}
        assert arm["measures"] == rep["arms"]["identity_release"]["measures"]


def test_mergeable_classes_give_one_token_per_class_and_class_is_eligible():
    p1 = np.array([[0.9, 0.1], [0.901, 0.099], [0.2, 0.8], [0.201, 0.799]])
    p2 = np.array([[0.8, 0.15, 0.05], [0.801, 0.149, 0.05], [0.1, 0.1, 0.8], [0.1, 0.101, 0.799]])
    law = OR.validate_law(np.full(4, 0.25), np.array([0.2, 0.7, 0.5, 0.9]), p1, p2)
    rep = OR.run_oracle(law)
    assert rep["arms"]["task_only"]["tokens"] == {1: 2, 2: 2}
    assert rep["arms"]["class"]["eligible"]
    assert rep["arms"]["task_only"]["measures"] == rep["arms"]["class"]["measures"]
    assert rep["arms"]["task_only"]["measures"] == rep["arms"]["decision_only"]["measures"]


def test_positive_coarsening_removes_an_unnecessary_distinction():
    p1 = np.array([[0.9, 0.1], [0.902, 0.098], [0.3, 0.7]])
    p2 = np.array([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])
    law = OR.validate_law(np.array([0.4, 0.4, 0.2]), np.array([0.9, 0.1, 0.5]), p1, p2)
    rep = OR.run_oracle(law)
    loc = rep["arms"]["local"]
    assert loc["partitions"][1] == [["0", "1"], ["2"]]
    assert loc["measures"]["t1"]["mi"] < rep["arms"]["identity_release"]["measures"]["t1"]["mi"] - 0.1
    assert abs(loc["measures"]["t1"]["mi"]) < 1e-12


def _arm_measures(rep):
    for name, arm in rep["arms"].items():
        if arm.get("measures") is not None and (arm.get("eligible") or name == "decision_only"):
            yield name, arm["measures"]


def test_coalition_dominates_each_recipient_and_decision_floor():
    """Readers that ignore either recipient: MI and BA of (t1, t2) >= each single view; every admissible arm leaks at
    least the decision-only floor (R9.2)."""
    rng = np.random.default_rng(2)
    for _ in range(25):
        rep = OR.run_oracle(random_law(rng, int(rng.integers(3, 7))))
        floor = rep["arms"]["decision_only"]["measures"]
        for name, m in _arm_measures(rep):
            for key in ("mi", "bayes_acc"):
                assert m["t12"][key] >= max(m["t1"][key], m["t2"][key]) - TOL, name
                for v in ("t1", "t2", "t12"):
                    assert m[v][key] >= floor[v][key] - TOL, (name, v)


def test_arm_orderings_hold_by_construction():
    rng = np.random.default_rng(3)
    for _ in range(25):
        A = OR.run_oracle(random_law(rng, int(rng.integers(3, 7))))["arms"]
        for i in ("t1", "t2"):
            assert A["local"]["measures"][i]["mi"] <= A["task_only"]["measures"][i]["mi"] + TOL
        j = A["joint"]["measures"]["t12"]["mi"]
        for other in ("seq12_nonadaptive", "seq21_nonadaptive", "local", "task_only"):
            assert j <= A[other]["measures"]["t12"]["mi"] + TOL
        assert A["seq12_nonadaptive"]["measures"]["t12"]["mi"] <= A["local"]["measures"]["t12"]["mi"] + TOL
        assert A["seq21_nonadaptive"]["measures"]["t12"]["mi"] <= A["local"]["measures"]["t12"]["mi"] + TOL
        assert A["seq12_nonadaptive"]["partitions"][1] == A["local"]["partitions"][1]
        assert A["seq21_nonadaptive"]["partitions"][2] == A["local"]["partitions"][2]


def test_joint_matches_bruteforce_over_admissible_pairs():
    rng = np.random.default_rng(4)
    for _ in range(10):
        law = random_law(rng, 5)
        rep = OR.run_oracle(law)
        R1 = OR.Recipient(law["p"][1], law["dec"][1], 8)
        R2 = OR.Recipient(law["p"][2], law["dec"][2], 64)
        best = min(loop_mi_ba(list(zip(t1.tolist(), t2.tolist())), law["P_x"], law["P_s1"])[0]
                   for t1 in R1.tokens for t2 in R2.tokens)
        assert abs(rep["arms"]["joint"]["measures"]["t12"]["mi"] - best) < 1e-12
        for a in range(R1.n_partitions):                               # every enumerated block is certified
            for r in R1.receipts(a):
                assert r["status"] == "CERTIFIED"
                assert np.all(GD.check_release(np.array(r["q"]), R1.V[r["values"]], dec=r["class"])["ok"])


def test_stochastic_outputs_all_satisfy_G_and_lp_is_rechecked():
    rng = np.random.default_rng(5)
    for _ in range(15):
        law = random_law(rng, int(rng.integers(3, 7)))
        rep = OR.run_oracle(law)
        st = rep["arms"]["stochastic_local"]
        assert st["eligible"]
        for i in (1, 2):
            pr = st["per_recipient"][i]
            assert pr["recheck_pass"] and pr["all_supported_outputs_satisfy_contract"] and pr["within_capacity"]
            R = OR.Recipient(law["p"][i], law["dec"][i], 8 if i == 1 else 64)
            W = np.array(st["channels"][i]["W_values_by_output"])
            Q = np.array(st["channels"][i]["outputs"])
            assert np.allclose(W.sum(axis=1), 1.0, atol=1e-12) and W.min() >= 0
            for v, t in zip(*np.nonzero(W > 0)):                      # independent re-check of every supported output
                ok, why = _literal_G(Q[t], R.V[v], int(R.vdec[v]))
                assert ok, why
            for name in ("local", "task_only", "joint"):              # deterministic channels are LP-feasible
                key = "t1" if i == 1 else "t2"
                assert pr["lp_optimum_bayes_acc"] <= rep["arms"][name]["measures"][key]["bayes_acc"] + 1e-9


def _literal_G(q, p, dec):
    D, B = GD.CONFIG["d"], GD.CONFIG["b"]
    bad = [k for k in range(len(p)) if not q[k] >= math.exp(-D) * p[k]]
    sq = sum(v * v for v in q) - sum(v * v for v in p)
    bad += [("y", y) for y in range(len(p)) if not sq - 2 * (q[y] - p[y]) <= B]
    bad += [("c", k) for k in range(len(p)) if k != dec and not q[dec] > q[k]]
    return (not bad and abs(sum(q) - 1) <= 1e-12), bad


def test_stochastic_refuses_when_capacity_could_bind_and_capacity_limits_partitions():
    a = np.array([0.900, 0.903, 0.906, 0.909])                        # chain: neighbours merge, two steps do not
    p = np.stack([a, 1 - a], axis=1)
    R = OR.Recipient(p, GD.decisions(p), cap=2)
    assert R.n_partitions > 0 and all(R.n_tokens <= 2)
    Pxs = np.stack([np.full(4, 0.125), np.full(4, 0.125)], axis=1)
    out = OR.stochastic_local_lp(R, Pxs)
    assert out["status"] == "REFUSED_CAPACITY_COULD_BIND" and out["maximal_blocks_per_class"] == {0: 3}
    assert OR.stochastic_local_lp(OR.Recipient(p, GD.decisions(p), cap=3), Pxs)["status"] == "OPTIMAL"
    p2 = np.repeat([[0.8, 0.15, 0.05]], 4, axis=0)
    law = OR.validate_law(np.full(4, 0.25), np.full(4, 0.5), p, p2)
    assert OR.run_oracle(law, cap1=1)["status"] == "NO_ADMISSIBLE_PARTITION"


def test_law_parsing_forms_and_validation():
    recs = [{"id": "x0", "P": "1/2", "sex1": "1/10", "p1": ["9/10", "1/10"], "p2": ["4/5", "3/20", "1/20"]},
            {"id": "x1", "P_float": 0.5, "sex1_float": 0.8, "p1_float": [0.903, 0.097], "p2_float": [0.8, 0.15, 0.05]}]
    law = OR.law_from_mapping({"inputs": recs})
    assert law["names"] == ["x0", "x1"] and law["P_x"].tolist() == [0.5, 0.5] and law["P_s1"].tolist() == [0.1, 0.8]
    col = OR.law_from_mapping({"P_x": [0.5, 0.5], "P_s1": [0.1, 0.8], "p1": [[0.9, 0.1], [0.903, 0.097]],
                               "p2": [[0.8, 0.15, 0.05], [0.8, 0.15, 0.05]]})
    assert np.array_equal(col["p"][1], law["p"][1])
    with pytest.raises(KeyError):
        OR.law_from_mapping({"P_x": [1.0], "p1": [[0.9, 0.1]], "p2": [[0.9, 0.1]]})
    with pytest.raises(ValueError):
        OR.validate_law([0.5, 0.6], [0.1, 0.1], [[0.9, 0.1]] * 2, [[0.9, 0.1]] * 2)
    with pytest.raises(ValueError):
        OR.validate_law(np.full(9, 1 / 9), np.full(9, 0.5), [[0.9, 0.1]] * 9, [[0.9, 0.1]] * 9)


def test_run_all_is_json_able_and_uses_registered_capacities():
    rng = np.random.default_rng(6)
    laws = {"laws": {}}
    for k in range(3):
        law = random_law(rng, 4)
        laws["laws"][f"L{k}"] = {"inputs": [{"id": f"x{i}", "P_float": float(law["P_x"][i]),
                                             "sex1_float": float(law["P_s1"][i]), "p1_float": law["p"][1][i].tolist(),
                                             "p2_float": law["p"][2][i].tolist()} for i in range(4)]}
    out = OR.run_all(laws)
    json.dumps(out, allow_nan=False)
    assert set(out["laws"]) == {"L0", "L1", "L2"}
    assert out["laws"]["L0"]["capacity"] == {"1": 8, "2": 64}


def test_g_exp_oracle_runs_with_certified_blocks():
    rng = np.random.default_rng(7)
    rep = OR.run_oracle(random_law(rng, 4), contract="G_exp")
    assert rep["status"] == "OK"
    for r in rep["arms"]["joint"]["receipts"][1] + rep["arms"]["joint"]["receipts"][2]:
        assert r["status"] == "CERTIFIED"


def test_joint_headroom_reproduces_construction_values():
    doc = json.loads(TOY.read_text())
    rep = OR.run_oracle(OR.law_from_mapping(doc["laws"]["JOINT_HEADROOM"]))
    A = rep["arms"]
    for name in ("identity_release", "local", "seq12_nonadaptive", "seq21_nonadaptive"):
        assert round(A[name]["measures"]["t12"]["mi"], 4) == 0.3230, name
    assert round(A["joint"]["measures"]["t12"]["mi"], 4) == 0.1853
    assert A["joint"]["n_tied_optima"] == 1
    assert A["joint"]["partitions"] == {1: [["x0"], ["x1", "x2"], ["x3"]], 2: [["x0", "x1", "x2"], ["x3"]]}
    assert A["local"]["n_tied_optima"] == {1: 1, 2: 1}
