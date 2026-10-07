"""[lra port of lcr/tests/test_fixtures.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md]
Role B tests for lra.fixtures. SYNTHETIC only. No fixture algorithm or oracle runs on the PINNED laws here
(CORRECTNESS_LOCK governs that): the pinned file is only checked statically (hashes, atom rebuild, static
properties, row construction); the engine and the engineering gate are exercised on separate TOY laws that are not
part of the registered bank.

    cd <WORKTREE> && OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.sema --label B:test-fixtures -- \\
        env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python \\
        -m pytest -q lra/tests/test_fixtures.py
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from lra import fixtures as FX


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
    assert body["laws_sha256"] == FX.PINNED_LAWS_SHA256
    assert FX._file_sha(FX.PKG / "FIXTURE_LAWS.json") == FX.PINNED_LAWS_FILE_SHA256
    assert FX._file_sha(FX.PKG / FX.SOURCE_GATE_FILE) == FX.PINNED_SOURCE_GATE_SHA256
    assert json.loads((FX.PKG / FX.SOURCE_GATE_FILE).read_text())["verdict"] == "GATE_NOT_MET"   # historical only


def test_build_laws_reproduces_the_pinned_laws_statically():
    """Static law construction (no algorithm): SPECS + build_laws give the pinned laws_sha256 and the same families."""
    b = FX.build_laws()
    z = json.loads((FX.PKG / "FIXTURE_LAWS.json").read_text())
    assert b["laws_sha256"] == z["laws_sha256"] == FX.PINNED_LAWS_SHA256
    assert b["families"] == z["families"] and b["schema"] == FX.LAWS_SCHEMA == "lcr-fixture-laws-v1"


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
    from lra import decoder as DC
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


def test_predecessor_gate_truth_table_descriptive_only():
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


def test_stage_correctness_refusals(monkeypatch, tmp_path):
    """Refusal paths only (cheap; also run inside the stage's E11 subprocess): real data, a shard, and a registered
    run whose locked documents differ. Nothing runs on the pinned laws."""
    with pytest.raises(ValueError, match="no real data"):
        FX.stage_correctness({"idx": {}}, None)
    with pytest.raises(ValueError, match="one process"):
        FX.stage_correctness(None, "0/2")
    from lra import lock as LK
    called = []
    monkeypatch.setattr(FX, "run_fixture", lambda *a, **k: called.append(1))
    monkeypatch.setattr(LK, "latest", lambda: {"name": "CORRECTNESS_LOCK", "documents_sha256": {
        "FIXTURE_LAWS.json": "0" * 64}})
    with pytest.raises(ValueError, match="REFUSED"):
        FX.stage_correctness(None, None)
    assert not called


# ----------------------------------------------------------------------------------------------- end to end (toy)
def test_toy_end_to_end_checks_pass(toy_run):
    law, res, tables, arms = toy_run
    for k, v in res["checks"].items():
        assert v["pass"], (k, v)
    ids = set(res["arms"])
    from lra import run as R
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


def test_incremental_terms_no_search_state_only_for_all_infeasible_starts():
    seq_infeasible = {"status": "INFEASIBLE", "winner": {"start": "A", "kind": "unchanged_descriptive"},
                      "starts": [{"name": "A", "stage1": {"status": "INFEASIBLE_START"}},
                                 {"name": "B", "stage1": {"status": "INFEASIBLE_START"}}]}
    assert FX._incremental_terms(seq_infeasible) == FX.NO_SEARCH_STATE
    refined = {"status": "FEASIBLE", "winner": {"start": "A", "kind": "refined"},
               "starts": [{"name": "A", "final": {"terms": {"L1": 1.0}}}]}
    assert FX._incremental_terms(refined) == {"L1": 1.0}
    missing = {"status": "FEASIBLE", "winner": {"start": "A", "kind": "refined"}, "starts": [{"name": "A"}]}
    assert FX._incremental_terms(missing) is None                  # a refined winner without its state still fails
    partly = {"status": "INFEASIBLE", "winner": {"start": "A", "kind": "unchanged_descriptive"},
              "starts": [{"name": "A", "stage1": {}}, {"name": "B", "final": {"terms": {}}}]}
    assert FX._incremental_terms(partly) is None                   # some start was refined: no exemption


# ----------------------------------------------------------------------------------------------- engineering gate (lra)
def _all_pass():
    return {k: {"pass": True} for k in FX.engineering_checks_text()}


def test_engineering_verdict_truth_table():
    ids = list(FX.engineering_checks_text())
    assert len(ids) == 12 and FX.VERDICTS == ("ENGINEERING_READY", "ENGINEERING_BLOCKED")
    assert FX.engineering_verdict(_all_pass()) == ("ENGINEERING_READY", [])
    for k in ids:                                            # any single failing mandatory check blocks
        c = _all_pass()
        c[k] = {"pass": False}
        v, why = FX.engineering_verdict(c)
        assert v == "ENGINEERING_BLOCKED" and why == [f"CHECK_FAILED:{k}"]
        c[k] = {"pass": None}                                # not exactly True is a failure, never a pass
        assert FX.engineering_verdict(c)[0] == "ENGINEERING_BLOCKED"
        del c[k]
        assert FX.engineering_verdict(c) == ("ENGINEERING_BLOCKED", [f"MISSING_CHECK:{k}"])
    c = _all_pass()
    c["C9_OLD_TRIGGER"] = {"pass": True}
    assert FX.engineering_verdict(c)[0] == "ENGINEERING_BLOCKED"
    assert FX.engineering_verdict(_all_pass(), fixture_errors={"F2_MISCALIBRATED": "boom"})[0] == "ENGINEERING_BLOCKED"
    # outcomes that must NOT block: the verdict reads only the mandatory checks (old trigger / affordability /
    # ties / source GATE_NOT_MET are carried as extra fields and ignored)
    c = _all_pass()
    c["E10_DECISION_DISCLOSURE_FLOOR"].update({"affordability": {"CLASS|D1_feasible": True}, "I12_CLASS": 0.0})
    c["E09_EXHAUSTIVE_ORACLE"].update({"heuristic_gaps": {"U|W-JOINT|i8o64|l0.01|D1": 7e-4}})
    c["E01_LAW_COUNTS_ROUTING_HASHES"].update({"old_trigger": {"triggered": False}, "source_gate": "GATE_NOT_MET"})
    assert FX.engineering_verdict(c) == ("ENGINEERING_READY", [])
    nb = " ".join(FX.NOT_BLOCKING)
    for phrase in ("zero-leakage", "affordable", "superior privacy release", "weighted control", "sequential arm",
                   "GATE_NOT_MET"):
        assert phrase in nb


def test_engineering_rule_file_matches_code():
    rule = FX.engineering_gate_rule()
    json.dumps(rule, allow_nan=False)
    assert rule["verdict_strings"] == ["ENGINEERING_READY", "ENGINEERING_BLOCKED"]
    assert rule["check_ids"] == list(rule["mandatory_checks"]) and len(rule["check_ids"]) == 12
    assert list(rule["prompt_check_map"]) == [str(i) for i in range(1, 13)]
    assert rule["laws"]["file_sha256"] == FX.PINNED_LAWS_FILE_SHA256 and rule["laws"]["laws_sha256"] == FX.PINNED_LAWS_SHA256
    assert rule["source_gate"]["sha256"] == FX.PINNED_SOURCE_GATE_SHA256 and "GATE_MET" not in rule["verdict_strings"]
    assert rule["tolerances"]["TOL_BUDGET"] == 1e-12 and rule["tolerances"]["TOL_TERMS"] == 1e-10
    assert rule["tolerances"]["decoder"]["kappa"] == 32.0 and rule["tolerances"]["decoder"]["eps"] == 1e-12
    for g, nodes in rule["wiring_tests"].items():
        for n in nodes:
            assert (FX.WT / n.split("::")[0]).exists(), n
    p = FX.PKG / FX.ENG_RULE_FILE
    if p.exists():                                           # once written, the registered file IS the code's rule
        assert json.loads(p.read_text()) == json.loads(json.dumps(rule))


def test_canon_apply_move_and_trace_states():
    assert FX._canon([3, 3, 0, 0]).tolist() == [0, 0, 2, 2]
    lab = {1: np.array([0, 0, 2, 2]), 2: np.array([0, 1, 2, 3])}
    nxt = FX._apply_move(lab, [{"r": 1, "f": 0, "to_canon": 2}])              # cell 0 joins token {2, 3}
    assert nxt[1].tolist() == [0, 1, 0, 0] and nxt[2].tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        FX._apply_move(lab, [{"r": 1, "f": 0, "to_canon": 3}])               # 3 is not a canonical label
    tr = {"starts": [{"name": "A", "stages": [{"stage": "joint", "start_labels": {"1": [0, 0, 2, 2], "2": [0, 1, 2, 3]},
                                                "moves": [{"step": 0, "type": "pair", "parts": [
                                                    {"r": 1, "f": 0, "to_canon": 2}, {"r": 2, "f": 3, "to_canon": 2}],
                                                    "terms_after": None}]}]}]}
    states = list(FX._trace_states(tr))
    assert [s[3] for s in states] == ["start", "pair"]
    assert states[1][4][1].tolist() == [0, 1, 0, 0] and states[1][4][2].tolist() == [0, 1, 2, 2]


def test_independent_fw_certificate_agrees_with_decoder_and_has_power():
    from lra import decoder as DC
    rng = np.random.default_rng(4)
    for K in (2, 3, 6):
        for _ in range(20):
            n = int(rng.integers(1, 300))
            d = int(rng.integers(K))
            P = rng.dirichlet(np.ones(K), size=n)
            P[:, d] += 1.0
            P = P / P.sum(1, keepdims=True)
            s = P.sum(0)
            y = np.bincount(rng.integers(0, K, n), minlength=K).astype(float)
            sol = DC.solve_token(y, s, n, d)
            rel, _, _ = FX._fw_gap(y, s, n, d, sol.u, sol.q)
            assert rel <= FX.TOL_FW
            u2 = np.full(K, 1.0 / K)
            u2[d] += 0.0
            from qpc.kmeans import smooth
            q2 = smooth(u2[None], np.array([d]), check=False)[0]
            rel2, _, _ = FX._fw_gap(y, s, n, d, u2, q2)
            if np.max(np.abs(sol.u - u2)) > 1e-3:
                assert rel2 > FX.TOL_FW                       # a non-optimal feasible point is detected


def test_findings_parser(tmp_path):
    items = [{"source_ordinal": i, "disposition": "RESOLVED", "regression_tests": [f"lra/tests/x.py::t{i}"]}
             for i in range(1, 15)]
    (tmp_path / FX.FINDINGS_FILE).write_text(json.dumps({"findings": items}))
    f, bad = FX._findings(tmp_path)
    assert not bad and len(f) == 14 and f[0]["tests"] == ["lra/tests/x.py::t1"]
    items[3]["disposition"] = "UNRESOLVED"
    items[4]["regression_tests"] = []
    (tmp_path / FX.FINDINGS_FILE).write_text(json.dumps({"findings": items[:13] + items[3:4]}))
    f, bad = FX._findings(tmp_path)
    assert any("ordinals" in b for b in bad) and any("UNRESOLVED" in b for b in bad) and any("no regression" in b for b in bad)
    assert FX._findings(tmp_path / "absent")[1] == [f"{FX.FINDINGS_FILE} missing"]
    # D's semantics: affects_required_adult_execution says what the defect WOULD touch if unresolved; a RESOLVED
    # finding with affects = true is fine, an unresolved one is named as an Adult-execution blocker
    items = [{"source_ordinal": i, "disposition": "SUPERSEDED_RESOLVED" if i == 10 else "RESOLVED",
              "affects_required_adult_execution": True, "regression_tests": [f"lra/tests/x.py::t{i}"]}
             for i in range(1, 15)]
    (tmp_path / FX.FINDINGS_FILE).write_text(json.dumps({"findings": items}))
    assert FX._findings(tmp_path)[1] == []
    items[5]["disposition"] = "OPEN"
    (tmp_path / FX.FINDINGS_FILE).write_text(json.dumps({"findings": items}))
    bad = FX._findings(tmp_path)[1]
    assert len(bad) == 2 and "Adult execution" in bad[1]


def test_run_pytest_nodes_maps_results(tmp_path):
    ok = "lra/tests/test_fixtures.py::test_canon_apply_move_and_trace_states"
    missing = "lra/tests/test_fixtures.py::test_does_not_exist"
    res, summ = FX.run_pytest_nodes([ok], tmp_path)
    assert res[ok]["pass"] and summ["ran"] == 1
    res, _ = FX.run_pytest_nodes([ok, missing], tmp_path)
    assert not res[missing]["pass"]                           # an uncollected node never passes


def _fake_runner(nodes, workdir):
    return {n: {"pass": True, "cases": [n + ":ok"]} for n in nodes}, {"returncode": 0, "ran": len(nodes)}


@pytest.fixture(scope="module")
def toy_stage(tmp_path_factory):
    """The full engineering stage on TWO TOY laws (not the registered bank) in a temporary package."""
    import shutil
    from lra import run as R
    tmp = tmp_path_factory.mktemp("lracorr")
    mp = pytest.MonkeyPatch()
    mp.setattr(R, "RUN", tmp / "run")
    mp.setattr(R, "UNITS", tmp / "run" / "units")
    filler = "lra/tests/test_fixtures.py::test_canon_apply_move_and_trace_states"
    mp.setattr(FX, "WIRING_TESTS", {g: (v or [filler]) for g, v in FX.WIRING_TESTS.items()})   # toy: no empty group
    laws = FX.build_laws()
    laws["families"] = [FX.build_law(toy_spec("xor")), FX.build_law(toy_spec("plain"))]
    laws["laws_sha256"] = FX.laws_hash(laws)
    pkg = tmp / "pkg"
    pkg.mkdir()
    lp = pkg / "FIXTURE_LAWS.json"
    lp.write_text(json.dumps(laws))
    FX.write_engineering_rule(pkg)
    items = [{"source_ordinal": i, "disposition": "RESOLVED",
              "regression_tests": ["lra/tests/test_fixtures.py::test_canon_apply_move_and_trace_states"]}
             for i in range(1, 15)]
    (pkg / FX.FINDINGS_FILE).write_text(json.dumps({"findings": items}))
    shutil.copy(FX.PKG / FX.SOURCE_GATE_FILE, pkg / FX.SOURCE_GATE_FILE)
    # first pass writes the oracle tables; the second pass must reproduce them (E09) from scratch
    ref = tmp / "ref"
    b1 = FX.stage_correctness(None, None, out=ref, laws_path=lp, save_units=False, run_tests=_fake_runner,
                              expect_laws=(FX._file_sha(lp), laws["laws_sha256"]), pkg=pkg,
                              oracle_ref=(tmp / "nonexistent", None))
    body = FX.stage_correctness(None, None, out=pkg, laws_path=lp, save_units=True, run_tests=_fake_runner,
                                expect_laws=(FX._file_sha(lp), laws["laws_sha256"]), pkg=pkg,
                                oracle_ref=(ref / FX.ORACLE_DIR, None))
    yield {"tmp": tmp, "pkg": pkg, "laws": laws, "first": b1, "body": body, "R": R}
    mp.undo()


def test_toy_engineering_stage_ready_and_outputs(toy_stage):
    body, pkg, R = toy_stage["body"], toy_stage["pkg"], toy_stage["R"]
    bad = {k: v.get("failures", v)[:5] if isinstance(v.get("failures", v), list) else v
           for k, v in body["checks"].items() if not v["pass"]}
    assert body["verdict"] == "ENGINEERING_READY", (body["reasons"], bad)
    assert json.loads((pkg / FX.ENG_RESULT_FILE).read_text())["verdict"] == "ENGINEERING_READY"
    for fid in ("TOY_XOR", "TOY_PLAIN"):
        assert R.done(f"cor__{fid}")
        u = R.U(f"cor__{fid}")
        traces = json.loads((u / "mapper_traces.json").read_text())
        recs = json.loads((u / "mapper_records.json").read_text())
        assert set(traces) == set(recs) and len(recs) == 30 and all(t and t.get("starts") for t in traces.values())
        for k in ("policies.json", "decoders.json", "d0_bank_policies.json", "fine.json", "releases.npz"):
            assert (u / k).exists()
        assert (pkg / FX.ORACLE_DIR / f"{fid}_partitions_r1.csv").exists()
    e9 = body["checks"]["E09_EXHAUSTIVE_ORACLE"]["per_fixture"]["TOY_XOR"]
    assert all(v["byte_identical"] for v in e9["source_tables"].values())
    assert body["checks"]["E03_CALIBRATED_NULL"]["per_fixture"]["TOY_XOR"]["applicable"] is False
    e6 = body["checks"]["E06_ACCEPTED_STATE_BUDGETS"]["per_fixture"]
    assert e6["TOY_PLAIN"]["accepted_states"] >= 10            # the plain toy law moves cells in K- and W- arms
    e7 = body["checks"]["E07_SEQUENTIAL_TEMPORARY_PARTNER"]["per_fixture"]["TOY_PLAIN"]
    assert e7["seq_starts"] >= 2 and e7["partner_class_only_checked"] >= 2
    e8 = body["checks"]["E08_INCREMENTAL_REPLAY"]["per_fixture"]["TOY_PLAIN"]
    assert e8["replay_ok"] == 30 and e8["max_trace_vs_own"] <= FX.TOL_TERMS
    # the old trigger is carried as descriptive output only
    assert body["descriptive"]["TOY_XOR"]["descriptive_only"] is True and "old_trigger" in body["descriptive"]["TOY_XOR"]
    assert body["source_gate"]["verdict"] == "GATE_NOT_MET"


def test_toy_engineering_stage_first_pass_fails_e09_without_reference(toy_stage):
    b1 = toy_stage["first"]
    assert b1["verdict"] == "ENGINEERING_BLOCKED" and "CHECK_FAILED:E09_EXHAUSTIVE_ORACLE" in b1["reasons"]


@pytest.fixture(scope="module")
def toy_plain_run():
    law = FX.build_law(toy_spec("plain"))
    res, tables, arms = FX.run_fixture(law)
    return law, res, tables, arms


def test_engineering_checks_have_power(toy_plain_run):
    """Injected defects are caught by the individual checks (toy law whose arms accept moves; no stage run)."""
    law, res, tables, arms = toy_plain_run
    from lra import run as R
    X, U = tables["X"], tables["U"]
    assert FX._e10(X, arms, tables, U)["pass"]
    bad = {**arms, "FAKE": {**arms[R.d0_id("CLASS")], "metrics": {**arms[R.d0_id("CLASS")]["metrics"],
                                                                 "I12": arms[R.d0_id("CLASS")]["metrics"]["I12"] - 1e-6}}}
    assert not FX._e10(X, bad, tables, U)["pass"]
    assert FX._e04(arms, tables)["pass"]
    cid = R.ctask_id()
    body = json.loads(json.dumps(arms[cid]["decoder_body"]))
    body["r1"]["u"][0] = list(reversed(body["r1"]["u"][0]))          # violates the bitwise re-solve / class dominance
    assert not FX._e04({cid: {**arms[cid], "decoder_body": body}}, {"sub1": {}, "sub2": {}})["pass"]
    assert FX._e01(law, [], X, arms, res)["pass"]
    X2 = {**X, "y": {1: X["y"][1].copy(), 2: X["y"][2]}}
    X2["y"][1][0] = 1 - X2["y"][1][0]                                 # a wrong label count is caught
    assert not FX._e01(law, [], X2, arms, res)["pass"]
    assert not FX._e01(law, ["laws_sha256"], X, arms, res)["pass"]
    assert FX._e05(X, arms, res, tables["mrecs"])["pass"]
    # an accepted state that breaks the budget is caught by the own state-by-state rebuild
    kj = R.constrained_id("JOINT-SINGLE")
    files = json.loads(json.dumps(R._finite(tables["mfiles"][kj]["trace.json"])))
    ct = arms[cid]["metrics"]
    limL, limB, cap = FX._limits(U, ct)
    f_ok = FX._unit_states(kj, "K-JOINT-SINGLE", {"trace.json": files}, X, limL, limB, cap)[0]
    assert not f_ok
    tight = {r: -1.0 for r in (1, 2)}
    f_bad = FX._unit_states(kj, "K-JOINT-SINGLE", {"trace.json": files}, X, tight, limB, cap)[0]
    n_moves = sum(len(sg["moves"]) for st in files["starts"] for sg in st["stages"])
    assert n_moves > 0 and len(f_bad) > 0                     # every accepted state now violates the (tight) budget
    # a tampered incremental term in the persisted trace is caught by the from-scratch rebuild (E08 part)
    bad_tr = json.loads(json.dumps(files))
    mv = next(m for st in bad_tr["starts"] for sg in st["stages"] for m in sg["moves"])
    mv["terms_after"]["I12"] = float(mv["terms_after"]["I12"]) + 1e-6
    f_t = FX._unit_states(kj, "K-JOINT-SINGLE", {"trace.json": bad_tr}, X, limL, limB, cap)[0]
    assert any(":terms:I12" in f for f in f_t)
    # the full E06/E07/E08 computation passes on the honest records
    e6, e7, e8 = FX._e06_e07_e08("TOY_PLAIN", X, U, arms, tables["mrecs"], tables["mfiles"], tables["bank"])
    assert e6["pass"] and e7["pass"] and e8["pass"], (e6["failures"][:3], e7["failures"][:3], e8["failures"][:3])
    assert e6["accepted_states"] >= 10
    assert FX._unit_states(kj, "K-JOINT-SINGLE", {"trace.json": None}, X, limL, limB, cap)[0] == [f"{kj}:trace_missing"]


def test_e11_e12_fail_closed(tmp_path, monkeypatch):
    items = [{"source_ordinal": i, "disposition": "RESOLVED", "regression_tests": [f"lra/tests/t.py::f{i}"]}
             for i in range(1, 15)]
    (tmp_path / FX.FINDINGS_FILE).write_text(json.dumps({"findings": items}))
    monkeypatch.setattr(FX, "WIRING_TESTS", {"a": ["lra/tests/t.py::g"], "b": []})
    e11, e12 = FX._e11_e12(tmp_path, _fake_runner, tmp_path / "w", [])
    assert not e11["pass"] and any("no registered node id" in f for f in e11["failures"]) and e12["pass"]
    monkeypatch.setattr(FX, "WIRING_TESTS", {"a": ["lra/tests/t.py::g"]})

    def one_fails(nodes, wd):
        r, s_ = _fake_runner(nodes, wd)
        r["lra/tests/t.py::f7"]["pass"] = False
        r["lra/tests/t.py::g"]["pass"] = False
        return r, s_
    e11, e12 = FX._e11_e12(tmp_path, one_fails, tmp_path / "w", [])
    assert not e11["pass"] and not e12["pass"] and any("finding 7" in f for f in e12["failures"])
    e11, _ = FX._e11_e12(tmp_path, _fake_runner, tmp_path / "w", ["rule differs"])
    assert not e11["pass"]                                     # a rule file that is not the code's rule blocks
    e11, e12 = FX._e11_e12(tmp_path / "nofile", _fake_runner, tmp_path / "w", [])
    assert not e12["pass"] and e12["failures"][0].endswith("missing")


def test_public_text_strips_private_paths():
    from lra import run as R
    raw = json.dumps({"e": f"FileNotFoundError: {R.PRIV}/run/units/x and {FX.WT}/lra/y.py and {Path.home()}/z"})
    out = FX.public_text(raw)
    assert str(Path.home()) not in out and "<PRIVATE_CACHE>/lra_v1/run/units/x" in out and "<WORKTREE>/lra/y.py" in out
    json.loads(out)


def test_e09_without_reference_tables_fails(toy_run):
    law, res, tables, arms = toy_run
    out = FX._e09(law["id"], law, tables, res, {}, (None, None))
    assert not out["pass"] and out["failures"] == ["no_reference_oracle_tables"]
