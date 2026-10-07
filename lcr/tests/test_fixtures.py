"""Role B tests for lcr.fixtures. SYNTHETIC only. No algorithm runs on the REGISTERED laws here (FIXTURE_LOCK
governs that): the registered file is only checked statically; the machinery is exercised on separate TOY laws
that are not part of the registered bank.

    cd <WORKTREE> && OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lcr.sema --label B:test-fixtures -- \\
        env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python \\
        -m pytest -q lcr/tests/test_fixtures.py
"""
from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest

from lcr import fixtures as FX


# ----------------------------------------------------------------------------------------------- toy laws (not registered)
def toy_spec(kind="xor"):
    """A toy law with the registered SHAPE but different numbers (machinery tests only)."""
    r1 = FX._cells_binary([84, 92, 86, 94], [10, 11, 10, 11], 16)
    r2 = FX._cells_binary([100, 108, 102, 110], [12, 13, 12, 13], 16)
    for r in (r1, r2):
        for c in r:
            c["a"], c["b"] = c["j"] // 2, c["j"] % 2
    if kind == "xor":
        pc = [[64 for _ in r2] for _ in r1]
        sx = [[3 if a["a"] != b["b"] else 1 for b in r2] for a in r1]      # SEX = a1 XOR b2 (w.p. 3/4)
    else:
        pc = [[64 for _ in r2] for _ in r1]
        sx = [[2 + (1 if a["b"] else -1) * (1 if a["class"] == 0 else 0) for b in r2] for a in r1]
    return {"id": f"TOY_{kind.upper()}", "label_den": [16, 16], "sex_den": 4, "r1": r1, "r2": r2,
            "pair_counts": pc, "sex_num": sx, "purpose": "toy (tests only)", "design": "toy"}


@pytest.fixture(scope="module")
def toy():
    return FX.build_law(toy_spec("xor"))


@pytest.fixture(scope="module")
def toy_run(toy):
    res, tables, arms = FX.run_fixture(toy)
    return toy, res, tables, arms


# ----------------------------------------------------------------------------------------------- registered file (static)
def test_registered_laws_static_integrity():
    body = FX.load_laws()                                      # hash + atoms rebuilt from the explicit tables
    assert [f["id"] for f in body["families"]] == ["F1_CALIBRATED_NULL", "F2_MISCALIBRATED", "F3_COMPLEMENTARY_XOR",
                                                   "F4_REDUNDANT"]
    for fam in body["families"]:
        p = fam["properties"]
        assert p["rows"] == 4096 and p["mapping_pairs"] <= FX.MAX_PAIRS
        assert all(Fraction(a[5], 4096) * 4096 == a[5] for a in fam["atoms"])
        for r in ("r1", "r2"):
            assert p[r]["accuracy_gain"] >= 0.03 and p[r]["cell_counts_match_declared"]
        for r in ("1", "2"):
            R = fam["recipients"][r]
            for c in R["cells"]:
                t = c["teacher"]
                assert int(np.argmax(t)) == c["class"] and sorted(t)[-1] > sorted(t)[-2]
        X = FX.fixture_rows(fam)                               # rows deploy to their declared cells (no algorithm)
        assert X["N"] == 4096
    rule = json.loads((FX.PKG / "FIXTURE_GATE_RULE.json").read_text())
    assert rule["laws_sha256"] == body["laws_sha256"]
    assert rule["check_ids"] == list(rule["mandatory_correctness_checks"])
    assert rule["verdict_strings"] == ["GATE_MET", "GATE_NOT_MET"]


def test_registered_f1_is_exactly_calibrated():
    fam = FX.load_laws()["families"][0]
    for r in ("1", "2"):
        R = fam["recipients"][r]
        for c in R["cells"]:
            assert [Fraction(x, R["teacher_den"]) for x in c["teacher"]] == [Fraction(x, R["label_den"])
                                                                            for x in c["labels"]]
    X = FX.fixture_rows(fam)
    for i in (1, 2):
        f = X["f1"] if i == 1 else X["f2"]
        for cell in range(X["fine"][i].F):
            rows = f == cell
            assert np.array_equal(np.bincount(X["y"][i][rows], minlength=2).astype(float), X["P"][i][rows].sum(0))


def test_tampered_laws_refused(tmp_path):
    body = json.loads((FX.PKG / "FIXTURE_LAWS.json").read_text())
    body["families"][2]["atoms"][0][5] += 1
    p = tmp_path / "laws.json"
    p.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        FX.load_laws(p)
    body["laws_sha256"] = FX.laws_hash(body)
    p.write_text(json.dumps(body))
    with pytest.raises(ValueError):                            # atoms no longer rebuild from the explicit tables
        FX.load_laws(p)


# ----------------------------------------------------------------------------------------------- components
def test_enumeration_is_canonical_and_complete(toy):
    X = FX.fixture_rows(toy)
    fine = X["fine"][1]
    parts = FX.enumerate_partitions(fine, 2)
    assert len(parts) == FX._bell_cap(4, 2) ** 2 == 64
    keys = {tuple(p) for p in parts}
    assert len(keys) == 64
    for p in parts:
        for c in range(fine.K):
            cells = fine.cells_of(c)
            assert len({int(p[f]) for f in cells}) <= 2
            assert all(int(fine.cell_class[p[f]]) == c for f in cells)        # never crosses classes
            for f in cells:
                assert p[f] == min(g for g in cells if p[g] == p[f])          # label = lowest member
    assert FX._bell_cap(4, 3) == 14 and FX._bell_cap(5, 5) == 52 and FX._bell_cap(6, 2) == 32


def test_plugin_mi_matches_dpc():
    from dpc.compress import mi_plugin
    rng = np.random.default_rng(0)
    for _ in range(10):
        s = rng.integers(0, 2, 500)
        t = rng.integers(0, 7, 500)
        assert abs(FX.plugin_mi(s, t) - mi_plugin(s, t)) <= 1e-14
        t2 = rng.integers(0, 3, 500)
        assert abs(FX.plugin_mi(s, t, t2) - mi_plugin(s, t * 3 + t2)) <= 1e-14


def test_oracle_terms_match_row_level(toy):
    from lcr import decoder as DC
    from qpc import release as RL
    X = FX.fixture_rows(toy)
    parts1, o1, _ = FX.recipient_oracle(X, 1, 2)
    parts2, o2, _ = FX.recipient_oracle(X, 2, 2)
    I12 = FX.pair_mi_matrix(X, parts1[:12], parts2[:12])
    meta = FX.fixture_meta("TOY")
    rng = np.random.default_rng(3)
    for _ in range(6):
        a, b = int(rng.integers(12)), int(rng.integers(12))
        p1 = RL.make_policy(1, X["fine"][1], parts1[a])
        p2 = RL.make_policy(2, X["fine"][2], parts2[b])
        pair = RL.make_pair(p1, p2, "TOY", 2, 2, meta=meta)
        rel1, _, _ = FX.d1_of(pair, X, "U|FINE-TASK|i8o64|D1")
        m = FX.release_metrics(rel1, X)
        assert abs(m["I12"] - I12[a, b]) <= 1e-12
        assert abs(m["L1"] - o1[a]["L_D1"]) <= 1e-12 and abs(m["B2"] - o2[b]["B_D1"]) <= 1e-12
        assert abs(m["I1"] - o1[a]["I"]) <= 1e-12
        rel0 = RL.release_arrays(pair, X["T"]["row_id"], X["T"]["p1"], X["T"]["d1"], X["T"]["p2"], X["T"]["d2"])
        m0 = FX.release_metrics(rel0, X)
        assert abs(m0["L2"] - o2[b]["L_D0"]) <= 1e-12
        assert m0["I12"] == m["I12"] and m0["I1"] == m["I1"]                  # fixed-token information unchanged


# ----------------------------------------------------------------------------------------------- gate logic
def _fake(fid, checks_ok=True, triggered=True, helps=False):
    return {"fixture": fid, "checks": {"C1": {"pass": checks_ok}}, "law_integrity": True,
            "trigger": {"triggered": triggered, "route": {"assignment_search_helps": helps}}}


def test_gate_truth_table():
    ids = ["F1", "F2", "F3", "F4"]
    g = FX.gate_from_results([_fake(i, triggered=(i == "F3")) for i in ids])
    assert g["verdict"] == "GATE_MET" and g["route"] == "DECODER_ENABLED_ROUTE"
    g = FX.gate_from_results([_fake(i, triggered=(i == "F3"), helps=(i == "F3")) for i in ids])
    assert g["route"] == "ASSIGNMENT_SEARCH_ROUTE"
    g = FX.gate_from_results([_fake(i, triggered=False, helps=True) for i in ids])
    assert g["verdict"] == "GATE_NOT_MET" and g["reasons"] == ["NO_FIXTURE_TRIGGERED"] and g["route"] is None
    g = FX.gate_from_results([_fake(i, checks_ok=(i != "F2")) for i in ids])
    assert g["verdict"] == "GATE_NOT_MET" and g["reasons"][0].startswith("CORRECTNESS_FAILURE:F2:C1")
    g = FX.gate_from_results([_fake(i) for i in ids[:3]])
    assert g["verdict"] == "GATE_NOT_MET" and "INCOMPLETE_FIXTURE_SET" in g["reasons"]


def _arm(i12, feas=True, i1=0.0, i2=0.0, acc=(0.8, 0.8)):
    return {"feasible": feas, "metrics": {"I12": i12, "I1": i1, "I2": i2, "T": 1.0, "acc1": acc[0], "acc2": acc[1]}}


def test_trigger_rule_cases():
    ct = _arm(0.13)
    const = [0.5, 0.5]
    base = {"U|C-TASK|i8o64|D1": ct, "U|FINE-TASK|i8o64|D1": _arm(0.13)}
    # a weighted control 0.012 below T* qualifies; d1 fixed map not -> ASSIGNMENT route on this fixture
    arms = {**base, "U|W-JOINT|i8o64|l0.04|D1": _arm(0.118), "U|JOINT|i8o64|l0.04|D1": _arm(0.125),
            "U|JOINT|i8o64|l0.04": _arm(0.125)}
    t = FX.trigger(arms, ct, [0.3, 0.3], const)
    assert t["triggered"] and t["T_star"] == "U|C-TASK|i8o64|D1" and t["route"]["assignment_search_helps"]
    # reduction exactly below 0.01 -> not qualifying
    arms = {**base, "U|K-JOINT-PAIR|i8o64|D1": _arm(0.1201)}
    assert not FX.trigger(arms, ct, [0.3, 0.3], const)["triggered"]
    # infeasible, local-cap violation, or own accuracy gain < 0.03 -> not qualifying
    for bad in (_arm(0.0, feas=False), _arm(0.0, i1=0.01), _arm(0.0, acc=(0.52, 0.8))):
        assert not FX.trigger({**base, "U|K-JOINT-SINGLE|i8o64|D1": bad}, ct, [0.3, 0.3], const)["triggered"]
    # T* = the feasible task-only candidate with the SMALLEST I12 (FINE-TASK|D1 here)
    arms = {**base, "U|FINE-TASK|i8o64|D1": _arm(0.05), "U|K-LOCAL|i8o64|D1": _arm(0.045)}
    t = FX.trigger(arms, ct, [0.3, 0.3], const)
    assert t["T_star"] == "U|FINE-TASK|i8o64|D1" and not t["triggered"]
    # D1 fixed map qualifies, new arm only 0.005 better -> decoder-enabled
    arms = {**base, "U|LOCAL|i8o64|l0.1|D1": _arm(0.0), "U|LOCAL|i8o64|l0.1": _arm(0.0, feas=False),
            "U|W-LOCAL|i8o64|l0.1|D1": _arm(0.0)}
    t = FX.trigger(arms, ct, [0.3, 0.3], const)
    assert t["triggered"] and t["route"]["decoder_enabled"] and not t["route"]["assignment_search_helps"]
    assert t["route"]["d0_version_already_qualifies"] == []
    # trivial fixture (static gain < 0.03) cannot trigger
    assert not FX.trigger({**base, "U|W-LOCAL|i8o64|l0.1|D1": _arm(0.0)}, ct, [0.02, 0.3], const)["triggered"]


def test_stage_refuses_real_data():
    with pytest.raises(ValueError):
        FX.stage_fixture({"idx": {}}, None)


# ----------------------------------------------------------------------------------------------- end to end (toy)
def test_toy_end_to_end_checks_pass(toy_run):
    law, res, tables, arms = toy_run
    for k, v in res["checks"].items():
        assert v["pass"], (k, v)
    ids = set(res["arms"])
    from lcr import run as R
    expect = set(R.d0_ids()) | set(R.d1_fixed_ids()) | {R.d0_id("CLASS") + "|D1"} | set(R.new_fit_ids())
    assert expect <= ids
    lab = res["checks"]["C6_HEURISTIC_LABELLING"]["labels"]
    assert all(v["label"] in ("EXHAUSTIVE_OPTIMAL", "HEURISTIC", "NOT_A_SEARCH") for v in lab.values())
    t = res["trigger"]
    assert t["T_star"] in FX.TASK_ONLY_D1
    json.dumps(R._finite(res), allow_nan=False)


def test_toy_fixed_token_ablation_and_decisions(toy_run):
    law, res, tables, arms = toy_run
    for cid, a in arms.items():
        if a["kind"] in ("d1_fixed", "d1_task"):
            b = arms[a["d0_of"]]
            for i in (1, 2):
                assert np.array_equal(a["rel"][f"tok{i}"], b["rel"][f"tok{i}"])
                assert np.array_equal(a["rel"][f"hard{i}"], b["rel"][f"hard{i}"])
            assert a["metrics"]["I12"] == b["metrics"]["I12"]


def test_toy_stage_writes_gate(tmp_path, toy, monkeypatch):
    from lcr import run as R
    monkeypatch.setattr(R, "RUN", tmp_path / "run")
    monkeypatch.setattr(R, "UNITS", tmp_path / "run" / "units")
    laws = FX.build_laws()
    laws["families"] = [toy, FX.build_law(toy_spec("plain"))]
    for f in laws["families"]:
        f["id"] = f["id"]
    laws["laws_sha256"] = FX.laws_hash(laws)
    lp = tmp_path / "laws.json"
    lp.write_text(json.dumps(laws))
    body = FX.stage_fixture(None, None, out=tmp_path, laws_path=lp, save_units=True)
    for f in laws["families"]:
        assert R.done(f"fix__{f['id']}")
        rr = json.loads((R.U(f"fix__{f['id']}") / "result.json").read_text())
        assert rr["fixture"] == f["id"] and "trigger" in rr
    assert body["verdict"] == "GATE_NOT_MET" and "INCOMPLETE_FIXTURE_SET" in body["reasons"]
    z = json.loads((tmp_path / "FIXTURE_GATE.json").read_text())
    assert z["verdict"] in ("GATE_MET", "GATE_NOT_MET") and z["synthetic_only"]
    assert (tmp_path / "fixture_oracle" / "TOY_XOR_partitions_r1.csv").exists()


def test_calibrated_toy_null_check_runs_and_passes():
    """A calibrated TOY law (not the registered F1; different numbers) under the F1 id so the C2 path is exercised
    (D0 bank + D1 only; no mapper)."""
    r1 = FX._cells_binary([120, 112, 96, 72], [15, 14, 12, 9], 16)
    r2 = FX._cells_binary([104, 96, 80, 72], [13, 12, 10, 9], 16)
    sp = {"id": "F1_CALIBRATED_NULL", "label_den": [16, 16], "sex_den": 4, "r1": r1, "r2": r2,
          "pair_counts": [[64 for _ in r2] for _ in r1],
          "sex_num": [[1 if (a["j"] + b["j"]) % 2 else 3 for b in r2] for a in r1], "purpose": "toy", "design": "toy"}
    law = FX.build_law(sp)
    res, _, arms = FX.run_fixture(law, mapper=False)
    c2 = res["checks"]["C2_CALIBRATED_NULL"]
    assert c2["pass"] and c2["max_q_diff"] <= 1e-9 and c2["subsets_checked"] == 2 * 2 * 15
    assert res["trigger"]["triggered"] is False and res["trigger"].get("reason") == "NO_C_TASK"
    # a deliberately miscalibrated law under the same id FAILS C2 (the check has power)
    sp2 = dict(sp, r1=FX._cells_binary([120, 112, 96, 72], [12, 12, 12, 12], 16))
    res2, _, _ = FX.run_fixture(FX.build_law(sp2), mapper=False)
    assert not res2["checks"]["C2_CALIBRATED_NULL"]["pass"]


def test_descriptive_flags_registered_groups():
    ids = ["U|JOINT|i8o64|l0.01|D1", "U|SEQ-21|i8o64|l0.01|D1", "U|W-JOINT|i8o64|l0.01|D1",
           "U|W-SEQ-12|i8o64|l0.01|D1", "U|K-JOINT-PAIR|i8o64|D1", "U|K-SEQ-12|i8o64|D1", "U|CLASS|i1o1|D1"]
    arms = {c: {"feasible": n != 4, "metrics": {"I12": 0.1 * (n + 1), "I1": 0.0, "I2": 0.0}} for n, c in enumerate(ids)}
    d = FX.descriptive(arms, {"metrics": {"I1": 1.0, "I2": 1.0}})
    b = d["best_I12"]
    assert b["constrained"]["all"] == pytest.approx(0.5) and b["constrained"]["eligible"] == pytest.approx(0.6)
    assert b["weighted"]["all"] == pytest.approx(0.3) and b["d1_fixed_joint"]["all"] == pytest.approx(0.1)
    assert d["joint_minus_sequential"]["d1_fixed"]["all"] == pytest.approx(-0.1)
    assert d["joint_minus_sequential"]["constrained"]["eligible"] is None      # only infeasible joint arm
    assert sum(g["n"] for k, g in b.items() if k in ("constrained", "weighted")) == 4   # CLASS is not privacy-trained
    t = FX.trigger(arms, None, [0.3, 0.3], [0.5, 0.5])
    assert t["reason"] == "NO_C_TASK" and "descriptive" in t
