"""Lead runner plumbing on synthetic data (temporary directories only): partition and fit stages for a small bank,
class preservation on every row, release format, resumability, bank ids, selection rules on synthetic inner records."""
import json

import numpy as np
import pytest

from dpc import run as R
from dpc import select as SEL
from rgj import finalize as FN
from smf.tests.test_pipeline import synth_D


def synth_teacher(n, rng):
    P1 = rng.dirichlet([1.0, 1.0], n)
    P2 = rng.dirichlet([0.6] * 6, n)
    P2[:, 5] *= 0.01                                  # class 5 never predicted (fallback path)
    P2 /= P2.sum(1, keepdims=True)
    P1[:5] = [[0.0, 1.0]] * 5                         # exact underflow zeros
    P1[5:8] = [[0.5, 0.5]] * 3                        # exact ties
    return {"p1": P1, "p2": P2, "d1": P1.argmax(1), "d2": P2.argmax(1), "c1": np.zeros((n, 2)),
            "c2": np.zeros((n, 6)), "r1": np.zeros((n, 16)), "r2": np.zeros((n, 16))}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    D = synth_D(n=1800, seed=4)
    D["sex"] = D["sex"].copy()
    D["feature_names"] = np.array([f"x{j}" for j in range(83)])
    monkeypatch.setattr(R, "UNITS", tmp_path / "units")
    monkeypatch.setattr(R, "RUN", tmp_path / "run")
    monkeypatch.setattr(R, "SEEDS", (0,))
    rng = np.random.default_rng(0)
    for t in R.TEACHERS:
        T = synth_teacher(len(D["row_id"]), rng)
        FN.save_unit(R.U(f"tea__s0__{t}"), {"teacher.npz": R.npz_writer({"row_id": D["row_id"], **T})},
                     {"seed": 0, "admission": {"complete_files_sha256": {"model.pt": "synthetic"}}})
    ids = ["U|FINE-TASK|m2", "U|DIRECT-TASK|m2", "U|LOCAL|m2|l1", "U|SEQ-12|m2|l1", "U|SEQ-21|m2|l1",
           "U|JOINT|m2|l1", "U|CLASS|m1", "RAW-J_b0.3|JOINT|m4|l0.1"]
    monkeypatch.setattr(R, "locked_bank_ids", lambda: ids + ["SRC|U", "SRC|RAW-J_b0.3", "REF|E", "REF|F", "REF|F0"])
    return D, ids


def test_partition_and_fit(env):
    D, ids = env
    R.stage_partition(D)
    R.stage_fit(D)
    for cid in ids:
        n = R.unit_for(0, cid)
        assert R.done(n)
        z = np.load(R.U(n) / "release.npz")
        T = R.teacher(0, R.parse_id(cid)["teacher"])
        assert np.array_equal(z["hard1"], T["d1"]) and np.array_equal(z["hard2"], T["d2"])
        assert np.array_equal(z["q1"].argmax(1), T["d1"]) and np.array_equal(z["q2"].argmax(1), T["d2"])
        assert np.allclose(z["q2"].sum(1), 1.0, atol=1e-12) and (z["q2"] >= 0).all()
        assert z["tok1"].max() < int(z["alpha1"]) and z["tok2"].max() < int(z["alpha2"])
        r = R.rec(n)
        assert all(r["class_preservation_all_rows"].values())
    a = R.rec(R.unit_for(0, "U|JOINT|m2|l1"))["receipts"]
    assert a is not None
    before = sorted(p.name for p in R.UNITS.iterdir())
    R.stage_fit(D)                                    # resume = no-op
    assert sorted(p.name for p in R.UNITS.iterdir()) == before


def test_bank_ids_and_counts():
    full = R.bank_ids("full")
    pol = [c for c in full if "CLASS" not in c]
    assert len(pol) == 2 * (3 * 2 + 3 * 3 * 4) == 84 and len(full) == 86      # x 3 seeds = 252 + 6 class-only
    red = R.bank_ids("reduced")
    assert len([c for c in red if "CLASS" not in c]) == 2 * (2 * 2 + 2 * 2 * 4)
    for c in full:
        assert R.parse_id(c)["kind"] == "policy" and R.config_id(**{k: v for k, v in R.parse_id(c).items()
                                                                     if k in ("teacher", "family", "m", "lam")}) == c


def _row(cid, pair, v1, v2, acc=(0.85, 0.48), ll=(0.34, 1.27), br=(0.22, 0.65), ok=True, states=10):
    seeds = {}
    for k in (0, 1, 2):
        u = {i: {"acc": acc[i], "logloss": ll[i], "brier": br[i], "const_acc": (0.75, 0.28)[i]} for i in (0, 1)}
        seeds[k] = {"auc": {"v1": v1, "v2": v2, "pair": pair}, "utility": u, "token_states": states}
    return seeds


def test_selection_gates_and_guards(monkeypatch):
    U = {i: {"acc": (0.85, 0.48)[i], "logloss": (0.34, 1.27)[i], "brier": (0.22, 0.65)[i], "const_acc": (0.75, 0.28)[i]}
         for i in (0, 1)}
    good = {i: dict(U[i]) for i in (0, 1)}
    gm = SEL.gate_margins(good, U)
    assert SEL.shortfall(gm) == 0.0
    bad = {0: dict(U[0]), 1: dict(U[1], logloss=U[1]["logloss"] + 0.0101)}
    assert SEL.shortfall(SEL.gate_margins(bad, U)) == pytest.approx(0.0001, abs=1e-9)
    bad2 = {0: dict(U[0], brier=U[0]["brier"] + 0.006), 1: dict(U[1])}
    assert SEL.shortfall(SEL.gate_margins(bad2, U)) == pytest.approx(0.001, abs=1e-9)
    mk = lambda cid, pair, v1, v2, elig=True, ll=0.8: {                                    # noqa: E731
        "config": cid, "family": SEL.family(cid), "teacher": SEL.teacher_of(cid), "m": SEL.rate_of(cid),
        "seeds": {k: {"auc": {"v1": v1, "v2": v2, "pair": pair}} for k in (0, 1, 2)}, "mean_pair": pair,
        "mean_v1": v1, "mean_v2": v2, "mean_logloss": ll, "token_states": 30, "task_eligible": elig,
        "gate_shortfall": 0.0 if elig else 0.01}
    T = mk("U|FINE-TASK|m4", 0.78, 0.76, 0.75)
    p_ok = mk("U|LOCAL|m4|l1", 0.75, 0.764, 0.754)
    p_bad = mk("U|JOINT|m4|l1", 0.70, 0.77, 0.74)               # v1 0.77 > 0.76 + 0.005: guard fails
    P = SEL.pick([p_ok, p_bad], guards={"T*": T})
    assert P["status"] == "NOMINEE" and P["config"] == "U|LOCAL|m4|l1"
    P2 = SEL.pick([p_bad], guards={"T*": T})
    assert P2["status"] == "NO_FEASIBLE_NOMINEE" and P2["descriptive_config"] == "U|JOINT|m4|l1"
    tie_a = mk("U|SEQ-12|m4|l1", 0.75, 0.70, 0.70, ll=0.81)
    tie_b = mk("U|SEQ-21|m4|l1", 0.75, 0.70, 0.70, ll=0.80)
    assert SEL.pick([tie_a, tie_b])["config"] == "U|SEQ-21|m4|l1"   # lower mean log loss breaks the AUC tie
