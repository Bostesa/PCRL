"""Synthetic tests of hcal.deploy (role A): calibrated decoders deploy bitwise, reproduce from their registered inputs,
and every binding / schema / export violation is refused with exit code 2. Random jcv Model + fitted heads and
random labels (lra/tests/test_deploy.py fixture helpers, imported unchanged): no Adult row."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cbp import fit as FT
from lra import decoder as DC
from lra.tests import test_deploy as LT
from qpc import compress as CP
from qpc import partition as PT
from qpc import release as RL

from hcal import calib as C
from hcal import deploy as DP


def _bank(pair, T):
    b = {}
    for i, pol in ((1, pair.p1), (2, pair.p2)):
        tok, _, hard = RL.encode(pol, T[f"p{i}"], T[f"d{i}"])
        n = np.asarray(pol.token_n, dtype=np.int64)
        S = np.asarray(pol.token_S, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(n[:, None] > 0, S / np.where(n > 0, n, 1)[:, None], np.nan)
        b.update({f"tok{i}": tok, f"hard{i}": hard, f"class{i}": np.asarray(pol.token_class), f"n_fit{i}": n,
                  f"mu{i}": mu, f"q0{i}": np.asarray(pol.token_proto)})
    return b


@pytest.fixture(scope="module")
def dep(tmp_path_factory):
    from qpc import deploy as QD
    tmp = tmp_path_factory.mktemp("hcaldeploy")
    unit, X, names = LT._synthetic_teacher_unit(tmp)
    model, heads, msha = QD.load_teacher(unit)
    P1, P2 = QD.teacher_probs(model, heads, X)
    T = {"row_id": np.arange(len(X)), "p1": P1, "p2": P2, "d1": P1.argmax(1), "d2": P2.argmax(1)}
    tr = np.arange(800)
    rng = np.random.default_rng(5)
    S = (rng.random(800) < 0.4 + 0.2 * (T["d1"][tr] == 1)).astype(np.int64)
    _, ff = PT.fine_unit(T, tr, caps={1: 12, 2: 72})
    fine = LT._materialize(ff["fine.json"], tmp / "fine.json")
    meta = {"teacher": "U", "seed": 0, "teacher_model_sha256": msha, "feature_names_sha256": QD.schema_sha256(names)}
    _, files = FT.fit_unit("LOCAL", fine, T, tr, S, 0.04, dict(meta))
    pol = tmp / "policy.json"
    files["policy.json"](pol)
    pair = RL.load_policy(pol)
    b = _bank(pair, T)
    cal = np.arange(800, 1200)
    ys = {"income": np.array([rng.choice(2, p=0.6 * p + 0.2) for p in P1[cal]]),
          "occupation": np.array([rng.choice(6, p=0.6 * p + 0.4 / 6) for p in P2[cal]])}
    decs = {}
    for fam in ("H-TOKEN32", "H-GLOBAL-TEMP", "H-CLASS-TEMP"):
        tabs = C.calibrate_partition(b, cal, ys, fam)
        body = DP.decoder_pair_dict(f"U|LOCAL|i8o64|l0.04|{fam}", pair, C.decoder_table(fam, 1, tabs[1]),
                                    C.decoder_table(fam, 2, tabs[2]))
        path = tmp / f"dec_{fam}.json"
        path.write_text(json.dumps(body))
        decs[fam] = (path, body["decoder_sha256"], tabs)
    Y = {i: np.array([rng.choice(P.shape[1], p=0.6 * p + 0.4 / P.shape[1]) for p in P[tr]]) for i, P in ((1, P1),
                                                                                                         (2, P2))}
    d1, d2 = LT._decoders(pair, T, tr, Y)
    d1p = tmp / "d1.json"
    DC.save_decoder_pair(d1p, "U|LOCAL|i8o64|l0.04|D1", pair, d1, d2)
    _, fc = CP.fit_unit("CLASS", fine, T, tr, S, 1, 1, None, dict(meta))
    clp = tmp / "class_policy.json"
    fc["policy.json"](clp)
    unit2, _, _ = LT._synthetic_teacher_unit(tmp / "other", seed=1)
    unreg = tmp / "unreg.json"
    _, f2 = CP.fit_unit("LOCAL", fine, T, tr, S, 8, 64, 1.0, dict(meta))
    f2["policy.json"](unreg)
    np.savez(tmp / "schema.npz", feature_names=np.array(names))
    np.savez(tmp / "in.npz", X=X, feature_names=np.array(names))
    return dict(tmp=tmp, unit=unit, X=X, names=names, pol=pol, pair=pair, bank=b, decs=decs, d1p=d1p, clp=clp,
                unit2=unit2, unreg=unreg, rel_d0=files["release.npz"])


def _args(D, **kw):
    a = {"--unit": str(D["unit"]), "--policy": str(D["pol"]), "--decoder": str(D["decs"]["H-GLOBAL-TEMP"][0]),
         "--X": str(D["tmp"] / "in.npz"), "--schema": str(D["tmp"] / "schema.npz"), "--out": str(D["tmp"] / "out.npz")}
    a.update(kw)
    out = []
    for k, v in a.items():
        if v is None:
            continue
        out += [k] if v is True else [k, v]
    return out


def _read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _refused(capsys, args, needle):
    with pytest.raises(SystemExit) as e:
        DP.main(args)
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert needle in err, err


@pytest.mark.parametrize("fam", ["H-GLOBAL-TEMP", "H-CLASS-TEMP", "H-TOKEN32"])
def test_calibrated_release_deploys_bitwise(dep, fam, tmp_path, capsys):
    path, sha, tabs = dep["decs"][fam]
    DP.main(_args(dep, **{"--decoder": str(path), "--decoder-sha256": sha, "--out": str(tmp_path / "o.npz")}))
    info = json.loads(capsys.readouterr().out)
    assert info["decoder"] == fam and info["release_id"] == f"U|LOCAL|i8o64|l0.04|{fam}"
    z = _read(tmp_path / "o.npz")
    assert sorted(z) == sorted(DP.ALLOWED_OUTPUT)
    for i in (1, 2):
        assert np.array_equal(z[f"tokens_{i}"], dep["bank"][f"tok{i}"])
        assert np.array_equal(z[f"decision_{i}"], dep["bank"][f"hard{i}"])
        assert np.array_equal(z[f"probs_{i}"], tabs[i]["q"][dep["bank"][f"tok{i}"]])
        assert np.array_equal(z[f"probs_{i}"].argmax(1), z[f"decision_{i}"])


def test_mean_and_d1_paths(dep, tmp_path, capsys):
    DP.main(_args(dep, **{"--decoder": None, "--out": str(tmp_path / "m.npz")}))
    assert json.loads(capsys.readouterr().out)["decoder"] == "D0"
    z = _read(tmp_path / "m.npz")
    assert all(np.array_equal(z[f"probs_{i}"], dep["rel_d0"][f"q{i}"]) for i in (1, 2))
    DP.main(_args(dep, **{"--decoder": str(dep["d1p"]), "--out": str(tmp_path / "d1.npz")}))
    assert json.loads(capsys.readouterr().out)["decoder"] == "D1"


def test_tampered_table_refused(dep, tmp_path, capsys):
    path, sha, _ = dep["decs"]["H-GLOBAL-TEMP"]
    body = json.loads(path.read_text())
    body["r2"]["q"][0][0] += 1e-9
    body["decoder_sha256"] = DP.decoder_hash(body)                     # self-consistent hash, wrong table
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(body))
    _refused(capsys, _args(dep, **{"--decoder": str(bad)}), "does not reproduce")
    body2 = json.loads(path.read_text())
    body2["r1"]["alpha"] = 1.0
    bad2 = tmp_path / "bad2.json"
    bad2.write_text(json.dumps(body2))
    _refused(capsys, _args(dep, **{"--decoder": str(bad2)}), "sha256 mismatch")
    _refused(capsys, _args(dep, **{"--decoder-sha256": "0" * 64}), "differs from --decoder-sha256")


def test_binding_refusals(dep, capsys):
    _refused(capsys, _args(dep, **{"--policy": str(dep["clp"])}), "different policy pair")
    _refused(capsys, _args(dep, **{"--unit": str(dep["unit2"])}), "different teacher")
    _refused(capsys, _args(dep, **{"--policy": str(dep["unreg"]), "--decoder": None}), "not one of the 57")


def test_schema_and_export_refusals(dep, capsys):
    X, names = dep["X"], dep["names"]
    np.savez(dep["tmp"] / "extra.npz", X=np.hstack([X, X[:, :1]]), feature_names=np.array(names + ["SEX"]))
    _refused(capsys, _args(dep, **{"--X": str(dep["tmp"] / "extra.npz")}), "refused")
    perm = list(range(83))
    perm[0], perm[1] = 1, 0
    np.savez(dep["tmp"] / "reord.npz", X=X[:, perm], feature_names=np.array(names)[perm])
    _refused(capsys, _args(dep, **{"--X": str(dep["tmp"] / "reord.npz")}), "refused")
    for flag in ("--raw-scores", "--fine-ids", "--export-teacher", "--bogus"):
        _refused(capsys, _args(dep) + [flag], "refused")
