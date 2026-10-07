"""[lra port of lcr/tests/test_deploy.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md]
Role B tests for lra.deploy. SYNTHETIC only (random jcv Model + fitted heads; random labels): no Adult row.

    cd <WORKTREE> && OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.sema --label B:test-deploy -- \\
        env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python \\
        -m pytest -q lra/tests/test_deploy.py

Pattern: cbp/tests/test_fit.py deploy tests at 7f3ec67.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from cbp import fit as FT
from lra import decoder as DC
from qpc import compress as CP
from qpc import partition as PT
from qpc import release as RL


def _synthetic_teacher_unit(tmp_path, seed=0):
    """Copied from cbp/tests/test_fit.py ``_synthetic_teacher_unit`` at 7f3ec67."""
    import joblib
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from jcv.finalize import save_unit
    from jcv.train import Model
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(1200, 83)).astype(np.float64)
    names = [f"f{j:02d}" for j in range(83)]
    model = Model(83, [2, 6], seed)
    with torch.no_grad():
        R = [model.encode(i, torch.from_numpy(X.astype(np.float32))).double().numpy() for i in (0, 1)]
    y1 = (R[0][:, 0] > np.median(R[0][:, 0])).astype(int)
    y2 = np.digitize(R[1][:, 1], np.quantile(R[1][:, 1], [0.2, 0.4, 0.6, 0.8, 0.999]))
    heads = [make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(R[0], y1),
             make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(R[1], y2)]
    unit = tmp_path / "tea__s0__SYNTH"
    state = model.state_dict()
    save_unit(unit, {"model.pt": lambda p: torch.save(state, p),
                     "head_0.joblib": lambda p: joblib.dump(heads[0], p),
                     "head_1.joblib": lambda p: joblib.dump(heads[1], p)}, {"seed": seed})
    return unit, X, names


def _materialize(writer, path):
    writer(path)
    return json.loads(path.read_text())


def _decoders(pair, T, tr, Y):
    decs = []
    for i, pol in ((1, pair.p1), (2, pair.p2)):
        P = T[f"p{i}"][tr]
        tok, _, _ = RL.encode(pol, P, T[f"d{i}"][tr])
        decs.append(DC.decode_policy(pol, tok, P, Y[i]))
    return decs


@pytest.fixture(scope="module")
def deployed(tmp_path_factory):
    from qpc import deploy as QD
    tmp = tmp_path_factory.mktemp("lradeploy")
    unit, X, names = _synthetic_teacher_unit(tmp)
    model, heads, msha = QD.load_teacher(unit)
    P1, P2 = QD.teacher_probs(model, heads, X)
    T = {"row_id": np.arange(len(X)), "p1": P1, "p2": P2, "d1": P1.argmax(1), "d2": P2.argmax(1)}
    tr = np.arange(800)
    rng = np.random.default_rng(5)
    S = (rng.random(800) < 0.4 + 0.2 * (T["d1"][tr] == 1)).astype(np.int64)
    Y = {i: np.array([rng.choice(P.shape[1], p=0.6 * p + 0.4 / P.shape[1]) for p in P[tr]])
         for i, P in ((1, P1), (2, P2))}
    fine_path = tmp / "fine.json"
    _, ff = PT.fine_unit(T, tr, caps={1: 12, 2: 72})
    fine = _materialize(ff["fine.json"], fine_path)
    meta = {"teacher": "U", "seed": 0, "teacher_model_sha256": msha, "feature_names_sha256": QD.schema_sha256(names)}
    _, files = FT.fit_unit("LOCAL", fine, T, tr, S, 0.04, dict(meta))
    pol = tmp / "policy.json"
    files["policy.json"](pol)
    pair = RL.load_policy(pol)
    d1, d2 = _decoders(pair, T, tr, Y)
    dec = tmp / "decoder.json"
    sha = DC.save_decoder_pair(dec, "U|LOCAL|i8o64|l0.04|D1", pair, d1, d2)
    rel_d1 = DC.release_arrays_d1(pair, d1, d2, T["row_id"], P1, T["d1"], P2, T["d2"])
    # a new-fit style policy (config = the lra id) with its decoder: the same map relabelled as C-TASK
    p1 = RL.Policy.from_dict(pair.p1.to_dict())
    p2 = RL.Policy.from_dict(pair.p2.to_dict())
    ct = RL.make_pair(p1, p2, "C-TASK", 8, 64, None, {**meta, "config": "U|C-TASK|i8o64|D1"})
    ctp = tmp / "ctask_policy.json"
    RL.save_policy(ct, ctp)
    ctd = tmp / "ctask_decoder.json"
    ctsha = DC.save_decoder_pair(ctd, "U|C-TASK|i8o64|D1", ct, *_decoders(ct, T, tr, Y))
    # another map (CLASS) and its decoder: for mismatch tests
    _, fc = CP.fit_unit("CLASS", fine, T, tr, S, 1, 1, None, dict(meta))
    clp = tmp / "class_policy.json"
    fc["policy.json"](clp)
    cl = RL.load_policy(clp)
    cld = tmp / "class_decoder.json"
    cl_decs = _decoders(cl, T, tr, Y)
    DC.save_decoder_pair(cld, "U|CLASS|i1o1|D1", cl, *cl_decs)
    rel_cl = DC.release_arrays_d1(cl, *cl_decs, T["row_id"], P1, T["d1"], P2, T["d2"])
    unit2, _, _ = _synthetic_teacher_unit(tmp / "other", seed=1)          # a DIFFERENT teacher unit (other weights)
    unreg = tmp / "unreg.json"
    _, f2 = CP.fit_unit("LOCAL", fine, T, tr, S, 8, 64, 1.0, dict(meta))
    f2["policy.json"](unreg)
    np.savez(tmp / "schema.npz", feature_names=np.array(names))
    np.savez(tmp / "in.npz", X=X, feature_names=np.array(names))
    return dict(tmp=tmp, unit=unit, X=X, names=names, pol=pol, dec=dec, sha=sha, rel_d0=files["release.npz"],
                rel_d1=rel_d1, ctp=ctp, ctd=ctd, ctsha=ctsha, clp=clp, cld=cld, unreg=unreg, rel_cl=rel_cl,
                unit2=unit2)


def _args(D, **kw):
    a = {"--unit": str(D["unit"]), "--policy": str(D["pol"]), "--decoder": str(D["dec"]),
         "--X": str(D["tmp"] / "in.npz"), "--schema": str(D["tmp"] / "schema.npz"), "--out": str(D["tmp"] / "out.npz")}
    a.update(kw)
    out = []
    for k, v in a.items():
        if v is None:
            continue
        out += [k] if v is True else [k, v]
    return out


def _refused(capsys, args, needle):
    from lra import deploy as DP
    with pytest.raises(SystemExit) as e:
        DP.main(args)
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert needle in err, err


def _read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_deploy_d1_outputs_only_release_and_matches_stored(deployed, tmp_path, capsys):
    from lra import deploy as DP
    D = deployed
    DP.main(_args(D, **{"--decoder-sha256": D["sha"]}))
    info = json.loads(capsys.readouterr().out)
    assert info["registered"] and info["config"] == "U|LOCAL|i8o64|l0.04|D1" and info["decoder"] == "D1"
    assert info["binding"] == "BOUND" and info["decoder_sha256"] == D["sha"]
    z = _read(D["tmp"] / "out.npz")
    assert sorted(z) == sorted(DP.ALLOWED_OUTPUT) == sorted(
        ["tokens_1", "probs_1", "decision_1", "tokens_2", "probs_2", "decision_2"])
    for i in (1, 2):
        assert np.array_equal(z[f"tokens_{i}"], D["rel_d1"][f"tok{i}"])
        assert np.array_equal(z[f"probs_{i}"], D["rel_d1"][f"q{i}"])
        assert np.array_equal(z[f"decision_{i}"], D["rel_d1"][f"hard{i}"])
        assert np.array_equal(z[f"tokens_{i}"], D["rel_d0"][f"tok{i}"])      # same tokens as the D0 release
        assert np.array_equal(z[f"decision_{i}"], D["rel_d0"][f"hard{i}"])
    assert not np.array_equal(z["probs_2"], D["rel_d0"]["q2"])
    shutil.copy(D["dec"], tmp_path / "restored_decoder.json")                 # rebuild from restored copies
    shutil.copy(D["pol"], tmp_path / "restored_policy.json")
    DP.main(_args(D, **{"--policy": str(tmp_path / "restored_policy.json"),
                        "--decoder": str(tmp_path / "restored_decoder.json"), "--out": str(tmp_path / "o2.npz")}))
    capsys.readouterr()
    a, b = _read(tmp_path / "o2.npz"), _read(D["tmp"] / "out.npz")
    assert all(np.array_equal(a[k], b[k]) for k in DP.ALLOWED_OUTPUT)


def test_deploy_d0_without_decoder_matches_qpc(deployed, tmp_path, capsys):
    from lra import deploy as DP
    D = deployed
    DP.main(_args(D, **{"--decoder": None, "--out": str(tmp_path / "d0.npz")}))
    info = json.loads(capsys.readouterr().out)
    assert info["config"] == "U|LOCAL|i8o64|l0.04" and info["decoder"] == "D0"
    z = _read(tmp_path / "d0.npz")
    for i in (1, 2):
        assert np.array_equal(z[f"probs_{i}"], D["rel_d0"][f"q{i}"])


def test_deploy_new_fit_config_requires_its_decoder(deployed, tmp_path, capsys):
    from lra import deploy as DP
    D = deployed
    DP.main(_args(D, **{"--policy": str(D["ctp"]), "--decoder": str(D["ctd"]), "--out": str(tmp_path / "ct.npz")}))
    info = json.loads(capsys.readouterr().out)
    assert info["config"] == "U|C-TASK|i8o64|D1" and info["decoder_sha256"] == D["ctsha"]
    _refused(capsys, _args(D, **{"--policy": str(D["ctp"]), "--decoder": None}), "decoder.json is required")
    # the LOCAL fixed-map decoder (same map, other configuration) is not the C-TASK decoder
    _refused(capsys, _args(D, **{"--policy": str(D["ctp"])}), "differs from the policy")
    # and the C-TASK decoder cannot be attached to the D0 LOCAL policy
    _refused(capsys, _args(D, **{"--decoder": str(D["ctd"])}), "not the registered D1 control")


def test_deploy_refuses_schema_violations(deployed, capsys):
    D = deployed
    X, names = D["X"], D["names"]
    np.savez(D["tmp"] / "extra.npz", X=np.hstack([X, X[:, :1]]), feature_names=np.array(names + ["SEX"]))
    _refused(capsys, _args(D, **{"--X": str(D["tmp"] / "extra.npz")}), "refused")
    np.savez(D["tmp"] / "missing.npz", X=X[:, :82], feature_names=np.array(names[:82]))
    _refused(capsys, _args(D, **{"--X": str(D["tmp"] / "missing.npz")}), "refused")
    perm = list(range(83))
    perm[10], perm[11] = perm[11], perm[10]
    np.savez(D["tmp"] / "reorder.npz", X=X[:, perm], feature_names=np.array([names[j] for j in perm]))
    _refused(capsys, _args(D, **{"--X": str(D["tmp"] / "reorder.npz")}), "reordered")
    np.savez(D["tmp"] / "bundled.npz", X=X, feature_names=np.array(names), sex=np.zeros(len(X)))
    _refused(capsys, _args(D, **{"--X": str(D["tmp"] / "bundled.npz")}), "exactly X and feature_names")
    _refused(capsys, _args(D, **{"--schema-sha256": "0" * 64}), "schema hash mismatch")


def test_deploy_refuses_exports_and_unknown_flags(deployed, capsys, tmp_path):
    from lra import deploy as DP
    D = deployed
    for flag in ("--export-fine-ids", "--fine-cells", "--raw-scores", "--teacher-probs", "--logits", "--export=all",
                 "--debug", "--allow-unbound-policy", "--continuous", "--include-sex", "--dump-u", "--latent"):
        _refused(capsys, _args(D) + [flag], "refused")
    _refused(capsys, _args(D) + ["--verbose"], "unknown flag")
    _refused(capsys, _args(D) + ["--kappa=16"], "unknown flag")
    bad = {k: np.zeros(3) for k in DP.ALLOWED_OUTPUT}
    bad["fine_1"] = np.zeros(3, dtype=np.int64)
    with pytest.raises(DP.Refused):
        DP.write_release(tmp_path / "bad.npz", bad)


def test_deploy_refuses_mismatched_teacher_decoder_and_unregistered(deployed, capsys, tmp_path):
    D = deployed
    z = json.loads(D["pol"].read_text())
    # mismatched teacher binding on the policy
    pp = RL.PolicyPair.from_dict(z)
    pp.config["teacher_model_sha256"] = "0" * 64
    for p in pp:
        p.meta["teacher_model_sha256"] = "0" * 64
    RL.save_policy(pp, tmp_path / "wrong.json")
    _refused(capsys, _args(D, **{"--policy": str(tmp_path / "wrong.json"), "--decoder": None}), "different teacher")
    # the decoder was bound to the original policy binding -> refused with the re-bound policy
    _refused(capsys, _args(D, **{"--policy": str(tmp_path / "wrong.json")}), "decoder failed")
    # decoder of another map
    _refused(capsys, _args(D, **{"--decoder": str(D["cld"])}), "different policy pair")
    # decoder sha mismatch, tampered decoder, decoder-sha without decoder
    _refused(capsys, _args(D, **{"--decoder-sha256": "f" * 64}), "differs from --decoder-sha256")
    body = json.loads(D["dec"].read_text())
    body["r2"]["q"][0][0] += 1e-9
    (tmp_path / "tampered.json").write_text(json.dumps(body))
    _refused(capsys, _args(D, **{"--decoder": str(tmp_path / "tampered.json")}), "decoder failed")
    body = json.loads(D["dec"].read_text())                          # stale statistics, hash recomputed
    yy = np.array(body["r2"]["y"])
    t = int(np.flatnonzero(np.array(body["r2"]["n"]) > 1)[0])
    k = int(np.flatnonzero(yy[t] > 0)[0])
    yy[t, k] -= 1
    yy[t, (k + 1) % 6] += 1
    body["r2"]["y"] = yy.tolist()
    from lra.decoder import DecoderTable
    r2 = body["r2"]
    t2 = DecoderTable(recipient=2, K=6, token_class=np.array(r2["token_class"]), n=np.array(r2["n"]),
                      y=yy.astype(float), s=np.array(r2["s"]), u=np.array(r2["u"]), q=np.array(r2["q"]),
                      fallback=np.array(r2["fallback"]), certs=r2["certs"], policy_fingerprint=r2["policy_fingerprint"],
                      config=r2["config"])
    body["r2"] = {**r2, "y": yy.tolist(), "stats_hash": t2.stats_hash(), "content_hash": t2.content_hash()}
    body["decoder_sha256"] = DC.decoder_hash(body)
    (tmp_path / "stale.json").write_text(json.dumps(body))
    _refused(capsys, _args(D, **{"--decoder": str(tmp_path / "stale.json")}), "decoder failed")
    _refused(capsys, _args(D, **{"--decoder": None, "--decoder-sha256": D["sha"]}), "without --decoder")
    # unregistered configurations
    _refused(capsys, _args(D, **{"--policy": str(D["unreg"]), "--decoder": None}), "not a registered")
    # CLASS|D1 IS a registered fixed-map D1 control in lra (role F R-1, lra.run.d1_fixed_ids): it deploys (below);
    # the CLASS policy with the decoder of ANOTHER map is refused
    _refused(capsys, _args(D, **{"--policy": str(D["clp"]), "--decoder": str(D["dec"])}), "different policy pair")
    pp = RL.PolicyPair.from_dict(json.loads(D["ctp"].read_text()))
    pp.config["config"] = "U|K-JOINT|i8o64|D1"
    RL.save_policy(pp, tmp_path / "kjoint.json")
    _refused(capsys, _args(D, **{"--policy": str(tmp_path / "kjoint.json"), "--decoder": str(D["ctd"])}),
             "not a registered lra configuration")
    pp = RL.PolicyPair.from_dict(json.loads(D["ctp"].read_text()))
    pp.config["m2"] = 32
    RL.save_policy(pp, tmp_path / "rate.json")
    _refused(capsys, _args(D, **{"--policy": str(tmp_path / "rate.json"), "--decoder": str(D["ctd"])}),
             "inconsistent")


# ----------------------------------------------------------------------------------------------- lra additions
def test_deploy_class_d1_control_is_registered_and_bitwise(deployed, tmp_path, capsys):
    from lra import deploy as DP
    from lra import run as R
    D = deployed
    assert "U|CLASS|i1o1|D1" in R.d1_fixed_ids()
    DP.main(_args(D, **{"--policy": str(D["clp"]), "--decoder": str(D["cld"]), "--out": str(tmp_path / "cl.npz")}))
    info = json.loads(capsys.readouterr().out)
    assert info["config"] == "U|CLASS|i1o1|D1" and info["decoder"] == "D1" and info["binding"] == "BOUND"
    z = _read(tmp_path / "cl.npz")
    for i in (1, 2):
        assert np.array_equal(z[f"tokens_{i}"], D["rel_cl"][f"tok{i}"])
        assert np.array_equal(z[f"probs_{i}"], D["rel_cl"][f"q{i}"])
        assert np.array_equal(z[f"decision_{i}"], D["rel_cl"][f"hard{i}"])


def test_deploy_refuses_a_different_teacher_unit(deployed, capsys):
    """The policy (and its decoder) bound to teacher A is refused on teacher unit B (other weights), with and
    without the decoder; nothing is written."""
    D = deployed
    out = D["tmp"] / "never.npz"
    for dec in (None, str(D["dec"])):
        _refused(capsys, _args(D, **{"--unit": str(D["unit2"]), "--decoder": dec, "--out": str(out)}),
                 "different teacher")
    assert not out.exists()


def test_deploy_refuses_extra_renamed_and_reordered_columns(deployed, capsys):
    D = deployed
    X, names = D["X"], D["names"]
    out = D["tmp"] / "never2.npz"
    # an extra column even with the pinned names first (84 columns), an extra column replacing one (renamed), and a
    # full reversal of the column order
    np.savez(D["tmp"] / "x84.npz", X=np.hstack([X, X[:, :1]]), feature_names=np.array(names + ["f00_copy"]))
    _refused(capsys, _args(D, **{"--X": str(D["tmp"] / "x84.npz"), "--out": str(out)}), "83 permitted columns")
    ren = list(names)
    ren[5] = "sex"
    np.savez(D["tmp"] / "renamed.npz", X=X, feature_names=np.array(ren))
    _refused(capsys, _args(D, **{"--X": str(D["tmp"] / "renamed.npz"), "--out": str(out)}), "refused")
    np.savez(D["tmp"] / "reversed.npz", X=X[:, ::-1], feature_names=np.array(names[::-1]))
    _refused(capsys, _args(D, **{"--X": str(D["tmp"] / "reversed.npz"), "--out": str(out)}), "refused")
    np.savez(D["tmp"] / "nan.npz", X=np.where(np.arange(83) == 3, np.nan, X), feature_names=np.array(names))
    _refused(capsys, _args(D, **{"--X": str(D["tmp"] / "nan.npz"), "--out": str(out)}), "finite")
    assert not out.exists()


def test_deploy_refuses_more_export_flags_and_abbreviations(deployed, capsys):
    D = deployed
    out = D["tmp"] / "never3.npz"
    for flag in ("--raw-score", "--scores", "--teacher-scores", "--export-teacher", "--fine-ids", "--fine-id=1",
                 "--export-fine", "--logit", "--embedding", "--features", "--with-labels", "--sensitive"):
        _refused(capsys, _args(D, **{"--out": str(out)}) + [flag], "refused")
    for flag in ("--deco", "--pol", "--sch", "--decoder-sha", "--kappa", "--eps=1e-9", "--lam=0.1", "--no-check"):
        _refused(capsys, _args(D, **{"--out": str(out)}) + [flag], "refused")
    assert not out.exists()


def test_deploy_refuses_mismatched_decoder_configs(deployed, tmp_path, capsys):
    """A decoder relabelled to another configuration (hash recomputed) is still refused: the policy/decoder pair and
    registration must match exactly."""
    D = deployed
    body = json.loads(D["dec"].read_text())
    body["config"] = "U|LOCAL|i8o64|l0.1|D1"                      # same map, other (registered) configuration
    body["decoder_sha256"] = DC.decoder_hash(body)
    (tmp_path / "relabel.json").write_text(json.dumps(body))
    _refused(capsys, _args(D, **{"--decoder": str(tmp_path / "relabel.json")}), "not the registered D1 control")
    body = json.loads(D["ctd"].read_text())
    body["config"] = "U|W-JOINT|i8o64|l0.04|D1"                   # a new-fit decoder relabelled to another new fit
    body["decoder_sha256"] = DC.decoder_hash(body)
    (tmp_path / "relabel2.json").write_text(json.dumps(body))
    _refused(capsys, _args(D, **{"--policy": str(D["ctp"]), "--decoder": str(tmp_path / "relabel2.json")}),
             "differs from the policy")
