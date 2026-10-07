"""Synthetic tests of hcal.controls (role D; no real data, no real labels, no unsealing).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m hcal.sema --label D:test -- env OMP_NUM_THREADS=1 \\
        PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q tests/pcrl_heldout_calibration_v1/test_controls.py

Controls run on the COMMON bank (legacy AUDIT_FIT pipeline on every original release + fresh ATTACK_FIT_NEW pipeline;
coordinator decision); per-pipeline verdicts are diagnostic. The pinned "inner" slate (LR, MLP, HGB, DA_LR) on small
synthetic roles; the realised-threshold equality check is a parameter and is disabled for the synthetic split (it
applies to the real split only). Nothing pinned is edited or monkeypatched.
"""
from __future__ import annotations

import copy
import json
import tempfile

import numpy as np
import pytest

from dpc import audit as DA
from hcal import bank as B
from hcal import controls as C
from hcal import data as HD
from hcal import ids as I
from lra import audit as LA
from smf import audit as SA

SMALL = dict(n_fit=2000, n_head=50, n_audit=2000, n_sel=1000, n_assess=300)
SLATE = "inner"
P0 = "U|DIRECT-TASK|i8o64"                         # registered legacy partition; synthetic shapes; 6 registered tables


@pytest.fixture(scope="module")
def DC():
    return B.synthetic_D(seed=2, n_new=1500, **SMALL)


@pytest.fixture(scope="module")
def TC(DC):
    return LA.synthetic_teacher(DC, seed=2)


@pytest.fixture(scope="module")
def BANKC(DC, TC):
    return B.synthetic_bank(DC, TC, m=(2, 3), n_tables=6, seed=2)       # pure noise: no SEX tilt


def _release(bank, tables, name):
    """lra-format original release (row_id, tok_i, q_i, hard_i, alpha_i) decoding with one lookup table."""
    z = {"row_id": np.asarray(bank["row_id"])}
    for i in (1, 2):
        z.update({f"tok{i}": bank[f"tok{i}"], f"q{i}": tables[name][i - 1][bank[f"tok{i}"]],
                  f"hard{i}": bank[f"hard{i}"], f"alpha{i}": np.asarray(bank[f"alpha{i}"])})
    return z


@pytest.fixture(scope="module")
def RELS(BANKC):
    bank, tables = BANKC
    return [(rid, _release(bank, tables, d)) for (rid, _), d in zip(B.legacy_units(0, P0), ("D0", "D1"))]


@pytest.fixture(scope="module")
def CTRL(DC, BANKC, RELS):
    return C.controls_for_partition(f"{P0}|s0", 0, P0, *BANKC, RELS, DC, slate=SLATE, expected=I.decoders_of(P0))


def test_control_plan_is_registered_structural_and_common():
    p = C.control_plan()
    assert [tuple(x) for x in p["codes"]] == [(0, "U|DIRECT-TASK|i8o64"), (0, "U|JOINT|i8o64|l0.1"),
                                              (0, "U|CLASS|i1o1"), (0, "U|K-JOINT-PAIR|i8o64")]
    assert all(c in I.partitions() for _, c in p["codes"])
    assert p["tables"] == {c: I.decoders_of(c) for _, c in p["codes"]}            # full registered table sets
    assert p["legacy_releases"]["U|JOINT|i8o64|l0.1"] == ["U|JOINT|i8o64|l0.1", "U|JOINT|i8o64|l0.1|D1"]
    assert p["legacy_releases"]["U|K-JOINT-PAIR|i8o64"] == ["U|K-JOINT-PAIR|i8o64|D1"]
    assert "common bank" in C.control_plan.__doc__.lower() or "COMMON" in p["required"]
    assert "diagnostic" in p["required"]
    assert tuple(p["null_calibration"]) == (0, "U|DIRECT-TASK|i8o64") and p["null_reps"] == 5
    assert [tuple(x) for x in p["rot"]] == [(0, "SRC|U")]
    assert C.control_jobs(p) == [("code", 0, c) for _, c in p["codes"]] + [("calibration", 0, P0), ("rot", 0, "SRC|U")]
    lim = C.CONTROL_LIMITS
    assert lim is LA.CONTROL_LIMITS and lim["null_z"] == 3.5 and lim["plant_min"] == 0.75
    assert C.SOURCE_NULL_THRESHOLD == 0.5654143765984265 and C.CONF_ETA == 0.05 and C.ROT_AMP == 1e-6
    assert C.CONTROL_SEED == SA.CONTROL_SEED
    json.dumps(p, allow_nan=False)


def test_one_S_star_for_both_pipelines_is_lra_permutation(DC):
    old, new, sel = C.control_roles(DC)
    halves = SA.null_split(DC)
    S = np.asarray(DC["sex"])
    Sp, sha = C.permuted_sex(DC, halves)
    ref, ref_sha = SA.frozen_permutation(S, DC, halves)                            # exactly as lra.audit draws it
    assert sha == ref_sha and np.array_equal(Sp, ref)
    for rows in (old, sel[halves[0]], sel[halves[1]]):
        assert np.array_equal(np.sort(Sp[rows]), np.sort(S[rows]))                 # within AUDIT_FIT, A, B
    assert np.isin(new, old).all() and not np.array_equal(Sp[new], S[new])        # fresh: S* restricted to its rows
    thr, _ = SA.null_threshold(Sp, DC, halves)
    assert thr == SA.null_threshold(S, DC, halves)[0]                              # depends only on B's counts


def test_one_plant_gives_the_same_tokens_in_every_bank(DC, BANKC, RELS):
    bank, tables = BANKC
    rng = np.random.default_rng(0)
    bit = rng.integers(0, 2, len(DC["row_id"]))
    bit[:5] = -1                                                                   # unlabelled rows -> bit 0 (A1)
    bk, tb = C.split_bank(bank, tables, 2, bit, collide=False)
    a = int(bank["alpha2"])
    assert int(bk["alpha2"]) == 2 * a and np.array_equal(bk["class2"], np.repeat(bank["class2"], 2))
    assert np.array_equal(bk["tok2"], 2 * bank["tok2"] + np.where(bit < 0, 0, bit))
    E = DA.onehot(bank["class2"], 6)
    for nm in tables:
        T, T2 = tables[nm][1], tb[nm][1]
        assert np.array_equal(T2[0::2], T)
        assert np.array_equal(T2[1::2], (1 - C.CONF_ETA) * T + C.CONF_ETA * E)    # more confident, same decision
        assert np.array_equal(tb[nm][0], tables[nm][0])
    for (rid, z), nm in zip(RELS, ("D0", "D1")):                                   # == dpc split_tokens on the release
        zp = DA.split_tokens(z, 2, bit, collide=False)
        assert np.array_equal(zp["tok2"], bk["tok2"]) and int(zp["alpha2"]) == int(bk["alpha2"])
        assert np.allclose(zp["q2"], tb[nm][1][bk["tok2"]], rtol=0, atol=1e-15)
    with tempfile.TemporaryDirectory() as tmp:
        banks, _, _ = C._bank_views(bank, tables, C.check_releases(0, P0, bank, RELS, DC), DC,
                                    [(1, bit, True)], tmp, "t")
    assert [n for n, *_ in banks] == [f"legacy:{rid}" for rid, _ in RELS] + ["fresh"]
    assert all(e for *_, e in banks)                                               # serialisation exact


def _res(a, b):
    return {w: {"selected_on_A": f"own:{w}:X", "max_auc_A": a[w], "heldout_auc_B": b[w]} for w in DA.PRIMARY_VIEWS}


def test_common_union_selects_on_half_A_and_evaluates_on_half_B():
    v = lambda x: {w: x for w in DA.PRIMARY_VIEWS}                                 # noqa: E731
    r0, r1, r2 = _res(v(0.60), v(0.51)), _res(v(0.60 + 5e-13), v(0.90)), _res(v(0.60 + 1e-9), v(0.40))
    com = C.common_union([("legacy:D0", r0), ("legacy:D1", r1), ("fresh", r2)])
    for w in DA.PRIMARY_VIEWS:
        assert com[w]["winner"] == "fresh" and com[w]["heldout_auc_B"] == 0.40     # A decides, B is reported
        assert com[w]["bank_heldout_auc_B"] == {"legacy:D0": 0.51, "legacy:D1": 0.90, "fresh": 0.40}
    tie = C.common_union([("legacy:D0", r0), ("legacy:D1", r1)])                   # 5e-13: the earlier bank keeps it
    assert {tie[w]["winner"] for w in DA.PRIMARY_VIEWS} == {"legacy:D0"}
    # a plant only one pipeline sees: the common verdict follows the A-winner, per-pipeline verdicts are diagnostic
    seen = _res({"v1": 0.55, "v2": 0.95, "pair": 0.95}, {"v1": 0.5, "v2": 0.9, "pair": 0.9})
    blind = _res(v(0.52), v(0.5))
    com = C.common_union([("legacy:D0", blind), ("fresh", seen)])
    assert C.verdict("CONF_r2", "v2", com, thr=0.57) and not C.verdict("CONF_r2", "v2", blind, thr=0.57)
    xor = _res({"v1": 0.52, "v2": 0.53, "pair": 0.9}, {"v1": 0.6, "v2": 0.5, "pair": 0.9})
    assert not C.verdict("XOR", "pair", xor, thr=0.57)                             # a common local winner > thr
    assert not C.verdict("CONF_r1", "v1", _res(v(0.9), v(0.9)), thr=0.57, exact=False)


def test_common_null_passes_and_every_plant_is_detected(DC, CTRL, RELS):
    ch = CTRL["checks"]
    assert CTRL["banks"] == [f"legacy:{rid}" for rid, _ in RELS] + ["fresh"] and CTRL["required"] == "common"
    assert CTRL["permutation_sha256"] == SA.frozen_permutation(np.asarray(DC["sex"]), DC, SA.null_split(DC))[1]
    assert ch["NULL"]["ok"], ch["NULL"]["common"]
    for p in ("CONF_r1", "CONF_r2", "COLL_r1", "COLL_r2", "XOR"):
        assert ch[p]["ok"], (p, {w: ch[p]["common"][w]["heldout_auc_B"] for w in DA.PRIMARY_VIEWS})
        assert ch[p]["serialisation_exact"]
        assert set(ch[p]["diagnostic"]) == set(CTRL["banks"])                      # per-pipeline verdicts reported
        assert all(ch[p]["common"][w]["winner"] in CTRL["banks"] for w in DA.PRIMARY_VIEWS)
    assert ch["CONF_r2"]["decisions_only_misses_it"] and ch["CONF_r1"]["decisions_only_misses_it"]
    assert ch["COLL_r2"]["table_values_only_misses_it"]
    assert ch["XOR"]["common_locals_null_ok"]
    assert CTRL["all_ok"] and CTRL["failures"] == [] and CTRL["tables"] == I.decoders_of(P0)
    assert CTRL["fit_roles"] == {"legacy": "AUDIT_FIT", "fresh": "ATTACK_FIT_NEW"}
    assert CTRL["null_threshold"] == pytest.approx(0.5 + 3.5 * CTRL["null_sd_B"], abs=1e-15)


def test_a_leaky_release_fails_the_common_null(DC, BANKC, RELS):
    bank, tables = BANKC
    _, new, _ = C.control_roles(DC)
    halves = SA.null_split(DC)
    Sp, _ = C.permuted_sex(DC, halves)
    leak = SA.noisy_sex(Sp, C.CONTROL_SEED)                                       # the releases carry S* itself
    bk, tb = C.split_bank(bank, tables, 1, leak, collide=True)
    rels = [(rid, DA.split_tokens(z, 1, leak, collide=True)) for rid, z in RELS]
    with tempfile.TemporaryDirectory() as tmp:
        banks, _, _ = C._bank_views(bk, tb, C.check_releases(0, P0, bk, rels, DC), DC, [], tmp, "leak")
    thr, _ = SA.null_threshold(Sp, DC, halves)
    e = C._check_entry("NULL", None, banks, C._audit_banks(banks, Sp, DC, new, halves, SLATE), thr)
    assert not e["ok"] and e["common"]["v1"]["heldout_auc_B"] > thr
    assert not any(d["ok"] for d in e["diagnostic"].values())


def test_run_controls_common_null_calibration_rot_and_summary(DC, TC, BANKC, RELS, CTRL):
    bank, tables = BANKC
    plan = {"codes": [[0, P0]], "tables": {P0: I.decoders_of(P0)}, "null_calibration": [0, P0], "null_reps": 5,
            "rot": [[0, "SRC|U"]]}
    calls = []
    rel = dict(RELS)

    def load_release(k, rid):
        calls.append(rid)
        return rel[rid]
    jobs = C.control_jobs(plan)[1:]                                                # the code job is CTRL (reused)
    parts = [{f"{P0}|s0": CTRL}] + C.run_controls(DC, plan, lambda k, p: bank, lambda k, p: tables,
                                                  load_release_fn=load_release, load_teacher_fn=lambda k: TC,
                                                  slate=SLATE, jobs=jobs)
    assert calls == [rid for rid, _ in RELS]
    cal = next(p["NULL_CALIBRATION"] for p in parts if "NULL_CALIBRATION" in p)
    assert cal["bank"] == "common" and cal["summary"]["reps"] == 5 and cal["summary"]["exceedances"] == 0
    assert cal["summary"]["banks"] == [f"legacy:{rid}" for rid, _ in RELS] + ["fresh"]
    assert [r["seed"] for r in cal["rows"][::3]] == [C.CONTROL_SEED + 100 + k for k in range(5)]
    assert all(set(r["bank_heldout_auc_B"]) == set(cal["summary"]["banks"]) for r in cal["rows"])
    rot = parts[-1]["SRC|U|s0|interface"]
    assert rot["checks"]["ROT_r1"]["ok"] and rot["checks"]["ROT_r2"]["ok"] and rot["checks"]["NULL"]["ok"]
    s = C.summarise_controls(parts, plan, SLATE, realised_threshold=None)
    assert s["verdict"]["all_ok"], s["verdict"]
    assert s["verdict"]["realised_threshold_matches_source"] is None
    assert s["verdict"]["common_checks"][f"{P0}|s0"]["XOR"]
    assert set(s["verdict"]["diagnostic_per_pipeline"][f"{P0}|s0"]["XOR"]) == set(CTRL["banks"])
    C.public_json(s)
    # a per-pipeline (diagnostic) failure never fails the verdict; a common failure does
    diag = copy.deepcopy(CTRL)
    diag["checks"]["XOR"]["diagnostic"]["fresh"]["ok"] = False
    diag["diagnostic_failures"] = ["XOR:fresh"]
    s1 = C.summarise_controls([{f"{P0}|s0": diag}] + parts[1:], plan, SLATE, realised_threshold=None)
    assert s1["verdict"]["all_ok"] and s1["verdict"]["diagnostic_failures"] == [f"{P0}|s0:XOR:fresh"]
    bad = copy.deepcopy(CTRL)
    bad["checks"]["XOR"]["ok"] = False
    bad["failures"] = ["XOR"]
    s4 = C.summarise_controls([{f"{P0}|s0": bad}] + parts[1:], plan, SLATE, realised_threshold=None)
    assert not s4["verdict"]["all_ok"] and f"{P0}|s0:XOR" in s4["verdict"]["failures"]
    # the real-split receipt: a synthetic split cannot reproduce the source threshold -> technical failure
    s2 = C.summarise_controls(parts, plan, SLATE)
    assert not s2["verdict"]["all_ok"] and s2["verdict"]["realised_threshold_matches_source"] is False
    assert any(f.startswith("REALISED_NULL_THRESHOLD") for f in s2["verdict"]["failures"])
    # a missing planned control or calibration is a failure, never silently dropped
    s3 = C.summarise_controls(parts[:1], plan, SLATE, realised_threshold=None)
    assert not s3["verdict"]["all_ok"] and not s3["verdict"]["null_calibration_present"]
    assert any(f.startswith("MISSING_CONTROLS") for f in s3["verdict"]["failures"])


def test_controls_refuse_unsealed_D_wrong_releases_tables_and_jobs(DC, BANKC, RELS):
    bank, tables = BANKC
    with pytest.raises(SystemExit, match="sealed"):
        C.controls_for_partition("x", 0, P0, bank, tables, RELS, {**DC, "sealed": False}, slate=SLATE)
    with pytest.raises(SystemExit, match="sealed"):
        C.run_controls({**DC, "sealed": False}, C.control_plan(), lambda k, p: bank, lambda k, p: tables)
    with pytest.raises(ValueError, match="tables_fn"):
        C.run_controls(DC, C.control_plan(), lambda k, p: bank, None)
    with pytest.raises(B.BankRefused, match="registered"):
        C.check_releases(0, P0, bank, RELS[::-1], DC)                              # legacy bank order
    with pytest.raises(B.BankRefused, match="registered"):
        C.check_releases(0, "U|K-JOINT-PAIR|i8o64", bank, RELS, DC)                # lra partition: D1 only
    z = dict(RELS[1][1])
    z["tok1"] = 2 * (z["tok1"] // 2) + (1 - z["tok1"] % 2)                         # different tokens, same classes
    with pytest.raises(B.BankRefused, match="tok1"):
        C.check_releases(0, P0, bank, [RELS[0], (RELS[1][0], z)], DC)
    with pytest.raises(B.BankRefused, match="registered list"):
        C.run_control_job(("calibration", 0, "U|K-JOINT-PAIR|i8o64"), DC, C.control_plan(), lambda k, p: bank,
                          lambda k, p: tables, load_release_fn=lambda k, rid: RELS[1][1], slate=SLATE)
    with pytest.raises(ValueError, match="SRC|U"):
        C.run_control_job(("rot", 0, "SRC|RAW-J_b0.3"), DC, C.control_plan(), lambda k, p: bank,
                          lambda k, p: tables, slate=SLATE)
    assert HD.role_rows(DC, "ATTACK_FIT_NEW").size == 1500


def test_amendment_a1_roundtrip_is_bitwise_with_nan_teacher_means(tmp_path):
    """AMENDMENT_A1 regression: a frozen bank with NaN teacher means (reserved empty tokens, as on the real banks)
    round-trips EXACTLY (bitwise); before A1 np.array_equal reported such a round trip as inexact."""
    import numpy as np
    from hcal import controls as CT
    rng = np.random.default_rng(0)
    mu = rng.dirichlet(np.ones(6), 5)
    mu[4] = np.nan                                              # reserved empty token
    bank = {"tok1": rng.integers(0, 3, 50), "mu2": mu, "q02": rng.dirichlet(np.ones(6), 5)}
    tables = {"D0": (rng.dirichlet(np.ones(2), 3), rng.dirichlet(np.ones(6), 5))}
    b2, t2, exact = CT._roundtrip(bank, tables, tmp_path, "nan_ok")
    assert exact and np.array_equal(b2["mu2"], mu, equal_nan=True)
    assert not np.array_equal(b2["mu2"], mu)                   # the pre-A1 comparison would have failed here
