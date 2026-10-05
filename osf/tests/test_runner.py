"""Runner plumbing on synthetic data with fake admitted smf units (temporary directories only; seconds to a minute):
admission rebuild + bitwise check, parity, fidelity, instrumented replay bitwise against the admitted checkpoints,
timing, and a tiny locked bank with release units."""
import json
import sys
import types

import joblib
import numpy as np
import pytest
import torch

from osf import run as R
from osf import train as T
from rgj import finalize as FN
from rgj import train as RT
from smf.tests.test_pipeline import synth_D


def _smf_release(name, root, state, k, D, critics=None):
    out, heads, meta = FN.finalize_model(R.model_from(state, k), D)
    files = {"model.pt": lambda p: torch.save(state, p),
             "release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"][D["role"] != "DEVELOPMENT_ASSESSMENT"],
                                                          **{kk: np.asarray(v)[D["role"] != "DEVELOPMENT_ASSESSMENT"]
                                                             for kk, v in out.items()})}
    for i, h in heads.items():
        files[f"head_{i}.joblib"] = (lambda p, h=h: joblib.dump(h, p))
    if critics is not None:
        files["critics.pt"] = lambda p: torch.save(critics, p)
    FN.save_unit(root / name, files, {"seed": k})


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("osfrun")
    D = synth_D(n=1200, seed=3)
    data = RT.TData(D)
    k = 0
    torch.manual_seed(5)
    warm = T.Model(T.D_IN, T.KS, k).state_dict()
    smf = tmp / "smf"
    FN.save_unit(smf / f"warm__s{k}", {"warm.pt": lambda p: torch.save(warm, p)}, {"seed": k})
    tm, _, ck = RT.task_line(warm, data, k, 40, stage="B", save_every=20)
    _smf_release(f"tl__s{k}__e40", smf, ck[40]["model"], k, D)
    for t, arm in (("J", "J-O"), ("L", "L-O")):
        for b in (0.1, 0.3):
            m, d, ck, _, _ = RT.train_run(arm, b, warm, data, k, "B", n_epochs=40, critic_head=RT.head_of(warm),
                                          ckpt_epochs=(20, 40))
            for e in (20, 40):
                _smf_release(f"raw__s{k}__RAW-{t}__b{b:g}__e{e}", smf, ck[e]["model"], k, D,
                             critics={kk: ck[e][kk] for kk in ("critics", "transforms", "lam")})
    return tmp, D, smf


@pytest.fixture()
def wired(env, monkeypatch):
    tmp, D, smf = env
    monkeypatch.setattr(R, "UNITS", tmp / "units")
    monkeypatch.setattr(R, "RUN", tmp / "run")
    monkeypatch.setattr(R, "PKG", tmp / "pkg")
    monkeypatch.setattr(R, "SEEDS", (0,))
    fake = types.ModuleType("osf.admit")
    fake.admitted_path = lambda name: smf / name
    monkeypatch.setitem(sys.modules, "osf.admit", fake)
    return D


def test_admit_parity_fidelity_replay_timing_bank(wired, monkeypatch):
    D = wired
    R.stage_admit(D)
    for cid in R.admitted_ids():
        rec = R.rec(R.rel_name(0, cid))
        assert all(rec["admitted_release_equal_on_smf_rows"]["equal"].values())
        z = np.load(R.U(R.rel_name(0, cid)) / "release.npz")
        assert np.array_equal(z["row_id"], D["row_id"])          # every row, including the sealed pool
    R.stage_parity(D)
    assert R.parity_passed()
    R.stage_fidelity(D)
    for t in "JL":
        r = R.rec(f"fid__s0__RAW-{t}_b0.3")
        assert r["pass"] and r["bitwise_model"] and r["bitwise_critics"]
        assert not r["expected_failure_cap"]["equivalent"]
    R.stage_replay(D)
    for cid in R.admitted_ids():
        r = R.rec(R.run_name(0, cid))
        assert all(r["bitwise"].values()), (cid, r["bitwise"])
        s = np.load(R.U(R.run_name(0, cid)) / "steps.npz")
        assert len(s["step"]) == 40 * int(np.ceil(len(D["idx"]["DEFENSE_FIT"]) / 256))
    assert (R.U(R.run_name(0, "RAW-J|b0.3")) / "captures.pt").exists()
    R.stage_timing(D)
    monkeypatch.setattr(R, "locked_bank", lambda: ("full", ["U", "RAW-J|b0.3", "RAW-J|b0.6", "NORM-L|r3|a2"]))
    R.stage_bank(D)
    for cid in ("RAW-J|b0.6", "NORM-L|r3|a2"):
        assert R.done(R.rel_name(0, cid)) and R.done(R.run_name(0, cid))
        r = R.rec(R.run_name(0, cid))
        assert r["diag"]["encoder_updates"] == 40 * int(np.ceil(len(D["idx"]["DEFENSE_FIT"]) / 256))
    assert not R.done(R.run_name(0, "RAW-J|b0.3") + "x")
    jobs = R.bank_jobs()
    assert (0, "RAW-J|b0.3") not in jobs and (0, "U") not in jobs        # admitted slots are never refitted
    from osf import report as REP
    monkeypatch.setattr(REP.R, "PKG", R.PKG)
    R.PKG.mkdir(parents=True, exist_ok=True)
    assert REP.write_fidelity()["all_pass"]
    assert REP.write_strength() > 0
