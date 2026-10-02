"""CELL-A pilot unit-level checks: unit selection from the 152 manifests, contract propagation, selection never sees
assessment rows, the 100/30/100 support rule in every role, native / fixed / relative quantities kept distinct,
vectorised bootstrap statistics against reference implementations, primary bounds reconstructible from the saved
predictions, and the lock blocking any changed code file or manifest."""
import copy
import json
import shutil

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from stored_model_eval import cli, pilot
from stored_model_eval.access import ReleaseContract
from stored_model_eval.admission import load_admitted
from stored_model_eval.effective import EFFECTIVE_PROTOCOL, REGISTERED_UNITS, UNTREATED_UNITS
from stored_model_eval.fixtures import make_pilot_world
from stored_model_eval.guards import FitAuthorization
from stored_model_eval.lock import build_lock, verify_lock
from stored_model_eval.metrics import brier_skill, logloss_reduction, r2_onehot_ridge
from stored_model_eval.pilot_infer import (UnitBootstrap, auc_class_stats, prob_skill_stats, r2_stat, rho_stat)
from stored_model_eval.pilot_inputs import build_inputs
from stored_model_eval.recipes import (fit_g2, fixed_ridge_fit, historical_native_r2, linear_predict,
                                       r2_against_prior)
from stored_model_eval.support import freeze_support

from .conftest import WT, branch_extra, run_shell


@pytest.fixture
def world(tmp_path):
    w = make_pilot_world(tmp_path, n=1200)
    build_inputs(w["orig"], tmp_path / "run_v1" / "inputs", w["source"], w["checkpoint"], log=lambda m: None)
    return w, tmp_path / "run_v1"


def _small_eff():
    e = copy.deepcopy(EFFECTIVE_PROTOCOL)
    e["attackers"]["L"]["C"] = [1.0]
    e["attackers"]["GBT"]["configs"] = e["attackers"]["GBT"]["configs"][:2]
    e["attackers"]["MLP"].update(hidden_layer_sizes=[[8, 8]], alpha=[1e-4], learning_rate_init=[1e-3])
    e["utility"]["U2"]["C"] = [1.0]
    return e


# ---- unit selection ------------------------------------------------------------------------------------------


def test_runner_selects_exactly_the_26_registered_units_from_152(tmp_path):
    w = make_pilot_world(tmp_path, n=600)
    sel = pilot.select_units(w["orig"])
    assert sel["n_available"] == 152 and sel["selected"] == list(REGISTERED_UNITS) and sel["match"]
    assert sel["n_unselected"] == 126
    assert all("__p1_" not in u and "__p2_" not in u for u in sel["selected"])
    (w["orig"] / "manifest_income_prediction__sex__p0_sigma2_seed1.json").unlink()
    with pytest.raises(SystemExit, match="income_prediction__sex__p0_sigma2_seed1"):
        pilot.plan(w["orig"])
    with pytest.raises(SystemExit, match="unregistered"):
        pilot.select_units(w["orig"], ["education_assessment__income__p2_sigma1_seed0"])


def test_worktree_must_be_this_package_root(tmp_path):
    with pytest.raises(SystemExit, match="not the tree this package runs from"):
        pilot.resolve_worktree(str(tmp_path))


# ---- release contract ------------------------------------------------------------------------------------------


def test_noise_contract_reaches_access_records(pilot_e2e):
    fr = json.loads((pilot_e2e["units"] / "income_prediction__sex__p0_sigma8_seed2" / "fit_records.json").read_text())
    assert fr["contract"] == {"noise": "persistent_token", "sigma": 8.0, "Sigma": None, "persistent": True,
                              "release_count": "one", "seed": 2}
    accs = [r["access"] for r in fr["recipes"] if r["surface"] != "label_only" and r["recipe"] != "U2"]
    assert accs and all(a["contract"]["noise"] == "persistent_token" and a["contract"]["sigma"] == 8.0 for a in accs)
    tags = {r["recipe"]: r["access"]["tag"] for r in fr["recipes"]}
    assert tags["LRT_A2"] == "A2" and tags["LRT_A4"] == "A4" and tags["NL"] == "A1"
    staged = [r for r in fr["recipes"] if r["recipe"] == "A3_repeated_query"][0]
    assert staged["status"] == "STAGED_NOT_RUN" and staged["access"]["effective_queries"] == 1
    assert not staged["access"]["valid"]
    fr0 = json.loads((pilot_e2e["units"] / "income_prediction__sex" / "fit_records.json").read_text())
    assert fr0["contract"]["noise"] == "none"


def test_legacy_fit_path_reads_manifest_contract_not_global_none(tmp_path):
    """The pinned pipeline stamped cfg release_contract (noise='none') on every record; it now reads the manifest."""
    from stored_model_eval.config import load_protocol
    from stored_model_eval.pipeline import fit_attackers
    w = make_pilot_world(tmp_path, n=600)
    arrays, _ = load_admitted(w["orig"] / "manifest_income_prediction__sex__p0_sigma4_seed1.json")
    cfg = load_protocol(None)
    cfg["roles"]["score"] = "assessment"
    cfg["attackers"]["linear"]["C"] = [1.0]
    res = fit_attackers(arrays, cfg, FitAuthorization(synthetic=True), attackers=("linear",), surfaces=["rep"])
    c = res["records"][0]["access"]["contract"]
    assert cfg["release_contract"]["noise"] == "none"
    assert c["noise"] == "persistent_token" and c["sigma"] == 4.0 and c["seed"] == 1
    assert ReleaseContract.from_manifest(None).noise == "none"
    with pytest.raises(ValueError):
        ReleaseContract.from_manifest({"kind": "gaussian_noise", "sigma_abs": 1.0, "release_count": "many"})


# ---- selection never sees assessment rows ----------------------------------------------------------------------


def test_fit_phase_never_receives_assessment_rows(world, monkeypatch):
    w, run = world
    seen = {}
    real = pilot.fit_phase

    def spy(view, ctx):
        seen["keys"] = sorted(view)
        seen["ids"] = np.concatenate([view["fit"]["row_id"], view["val"]["row_id"]])
        for part in view.values():
            for k, v in part.items():
                if v is not None:
                    assert len(v) == len(part["row_id"]), k
        return real(view, ctx)
    monkeypatch.setattr(pilot, "fit_phase", spy)
    u = "income_prediction__sex"
    pilot.run_unit(u, run / "inputs" / f"manifest_{u}.json", run / "units", FitAuthorization(synthetic=True),
                   eff=_small_eff(), log=lambda m: None)
    with np.load(run / "units" / u / "preds.npz") as z:
        assess = set(z["assess_row_id"].tolist())
    assert seen["keys"] == ["fit", "val"] and not (set(seen["ids"].tolist()) & assess)
    fr = json.loads((run / "units" / u / "fit_records.json").read_text())
    for r in fr["recipes"]:
        for row in r.get("selection_table") or []:
            assert set(row) <= {"hp", "attacker_val_log_loss", "n_iter"}
        if r["recipe"] == "NL":
            cand = r["nl_selection"]["candidates"]
            assert r["nl_selection"]["selected_family"] == min(cand, key=cand.get)
    with pytest.raises(ValueError, match="only fit/val"):
        real({"fit": {}, "val": {}, "assessment": {}}, {})


def test_unit_refuses_to_overwrite_and_checks_release_block(world):
    w, run = world
    u = "income_prediction__race"
    m = run / "inputs" / f"manifest_{u}.json"
    (run / "units" / u).mkdir(parents=True)
    with pytest.raises(SystemExit, match="exists"):
        pilot.run_unit(u, m, run / "units", FitAuthorization(synthetic=True), eff=_small_eff())
    with pytest.raises(SystemExit, match="manifest .* declares unit_id"):
        pilot.run_unit("income_prediction__sex", m, run / "units2", FitAuthorization(synthetic=True), eff=_small_eff())


# ---- support ---------------------------------------------------------------------------------------------------


def test_support_rule_applies_in_every_role():
    roles, y = [], []
    counts = {0: (150, 20, 150), 1: (90, 40, 150), 2: (150, 40, 99), 3: (150, 40, 150), 4: (100, 30, 100)}
    for k, (f, v, a) in counts.items():
        for r, n in (("attacker_fit", f), ("attacker_val", v), ("assessment", a)):
            roles += [r] * n
            y += [k] * n
    rule = EFFECTIVE_PROTOCOL["support"]
    s = freeze_support(np.array(y), np.array(roles), rule)
    assert s["supported_classes"] == [3, 4] and s["supported_pairs"] == [[3, 4]] and s["status"] == "ESTIMABLE"
    assert "attacker_val<30" in s["unsupported_classes"]["0"]["detail"][0]
    assert "attacker_fit<100" in s["unsupported_classes"]["1"]["detail"][0]
    assert "assessment<100" in s["unsupported_classes"]["2"]["detail"][0]
    s2 = freeze_support(np.array(y), np.array(roles), dict(rule, min_attacker_fit=151))
    assert s2["status"] == "NE" and s2["ne_reason"] == "fewer_than_2_supported_classes"
    # macro AUC averages the frozen supported classes only (unsupported rows remain negatives)
    rng = np.random.default_rng(0)
    yy = np.array(y)
    P = rng.dirichlet(np.ones(5), len(yy))
    f = auc_class_stats(yy, P, [3, 4], [(3, 4)])["macro_auc"]
    ref = np.mean([roc_auc_score(yy == k, P[:, k]) for k in (3, 4)])
    assert abs(f(np.ones((len(yy), 1)))[0] - ref) < 1e-12


# ---- native / fixed / relative quantities ------------------------------------------------------------------------


def test_native_fixed_and_relative_quantities_are_distinct():
    rng = np.random.default_rng(0)
    n, d = 4000, 16
    s = rng.integers(0, 3, n)
    Z = rng.normal(size=(n, d))
    Z[:, -1] = 1e-5 * (rng.normal(size=n) + (s - 1))      # signal at Gram eigenvalue ~ penalty scale
    H = (Z @ np.linalg.qr(rng.normal(size=(d, d)))[0].T).astype(np.float32)
    fi, vi, ei = np.arange(0, 2000), np.arange(2000, 2600), np.arange(2600, n)
    K = 3
    # N0/N1 estimator == the PCRL verification formula (float64 variant == metrics.r2_onehot_ridge in float64)
    n0 = historical_native_r2(H, s, 1e-6, "float64")
    assert abs(n0["clamped"] - r2_onehot_ridge(H.astype(np.float64), s, 1e-6)) < 1e-10
    mixed = historical_native_r2(H, s, 1e-6, "historical_mixed_precision")
    assert mixed["centring_dtype"] == "float32" and mixed["gram_dtype"] == "float64"
    out = {}
    for c in (0.5, 1.0, 4.0):
        Hc = (c * H).astype(np.float64)
        g1 = fixed_ridge_fit(Hc[fi], s[fi], K, 1e-6)
        r_g1 = r2_against_prior(s[ei], linear_predict(g1, Hc[ei]), g1["muY"], K)
        g2 = fit_g2(Hc[fi], s[fi], Hc[vi], s[vi], K, EFFECTIVE_PROTOCOL["linear_closed_form"]["G2"],
                    auth=FitAuthorization(synthetic=True), synthetic=True)
        r_g2 = r2_against_prior(s[ei], linear_predict(g2["model"], Hc[ei]), g2["model"]["muY"], K)
        n1 = historical_native_r2(Hc[ei], s[ei], 1e-6, "float64")["raw"]
        out[c] = (n1, r_g1, r_g2)
    assert max(v[2] for v in out.values()) - min(v[2] for v in out.values()) < 1e-6   # G2 scale-invariant
    assert out[4.0][1] - out[0.5][1] > 0.05                                              # G1 penalty-dependent
    assert all(abs(n1 - g1) > 1e-3 and abs(g1 - g2) > 1e-3 for n1, g1, g2 in out.values())


# ---- vectorised bootstrap statistics vs references -----------------------------------------------------------


def test_weighted_statistics_match_reference_implementations():
    rng = np.random.default_rng(1)
    n, K = 600, 3
    y = rng.integers(0, K, n)
    P = rng.dirichlet(np.ones(K), n)
    P[np.arange(n), y] += 0.3
    P /= P.sum(1, keepdims=True)
    P = np.round(P, 2)                                   # ties
    P /= P.sum(1, keepdims=True)
    prior = np.bincount(y, minlength=K) / n
    ones = np.ones((n, 1))
    st = auc_class_stats(y, P, [0, 1, 2], [(0, 1), (0, 2), (1, 2)])
    per = [roc_auc_score(y == k, P[:, k]) for k in range(K)]
    assert abs(st["macro_auc"](ones)[0] - np.mean(per)) < 1e-12
    assert abs(st["worst_class_auc"](ones)[0] - max(per)) < 1e-12
    # integer bootstrap weights == repeating rows
    w = rng.integers(0, 3, n).astype(float)
    rep = np.repeat(np.arange(n), w.astype(int))
    assert abs(st["macro_auc"](w[:, None])[0] -
               np.mean([roc_auc_score(y[rep] == k, P[rep, k]) for k in range(K)])) < 1e-12
    ps = prob_skill_stats(y, P, prior, 1e-12)
    assert abs(ps["LLR_nats"](ones)[0] - logloss_reduction(y, P, prior)) < 1e-12
    assert abs(ps["brier_skill"](ones)[0] - brier_skill(y, P, prior)) < 1e-12
    pred = rng.normal(size=(n, K))
    assert abs(r2_stat(y, pred, prior)(ones)[0] - r2_against_prior(y, pred, prior, K)) < 1e-12
    u, v = rng.normal(size=n), rng.normal(size=n) + 0.5 * y
    assert abs(rho_stat(u, v)(ones)[0] - np.corrcoef(u, v)[0, 1] ** 2) < 1e-12


def test_primary_bound_reconstructs_from_saved_predictions(pilot_e2e):
    """Re-derive one primary bound from preds.npz + supported.json with the documented draw and quantile rule."""
    inf = pilot_e2e["infer"]
    ep = [e for e in inf["primary"]["endpoints"] if e["id"] == "P1-income_prediction__race"][0]
    d = pilot_e2e["units"] / "income_prediction__race"
    with np.load(d / "preds.npz") as z:
        y, pred, prior, units = z["y_s"], z["G1_pred"], z["G1_prior"], z["assess_unit"]
    a = inf["primary"]["alpha_each"]
    boot = UnitBootstrap(units, inf["primary"]["B"], inf["primary"]["seed"], 500)
    f = r2_stat(y, pred, prior)
    reps = np.concatenate([f(W) for W in boot.chunks()])
    assert abs(np.quantile(reps, a, method="linear") - ep["lower"]) < 1e-12
    assert abs(np.quantile(reps, 1 - a, method="linear") - ep["upper"]) < 1e-12
    assert inf["primary"]["family_size"] == 16 and abs(a * 16 - 0.05) < 1e-15
    assert [e["unit"] for e in inf["primary"]["family"]] == list(UNTREATED_UNITS) * 2


# ---- lock --------------------------------------------------------------------------------------------------------


def _copy_tree(tmp_path):
    root = tmp_path / "wt"
    shutil.copytree(WT / "stored_model_eval", root / "stored_model_eval",
                    ignore=shutil.ignore_patterns("__pycache__", "tests"))
    shutil.copytree(WT / "results/combined_stored_model_pilot_v1/scripts",
                    root / "results/combined_stored_model_pilot_v1/scripts")
    for f in ("results/combined_evaluation_preparation_v1/ATTACKER_ACCESS_TABLE.csv",
              "results/combined_evaluation_preparation_v1/PILOT_LOCK.json"):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(WT / f, root / f)
    return root


def test_lock_blocks_changed_code_or_manifest(world, tmp_path):
    w, run = world
    root = _copy_tree(tmp_path)
    lock = tmp_path / "PILOT_LOCK_v2.json"
    build_lock(root, run / "inputs", lock, features=w["features"])
    assert verify_lock(lock, root, run / "inputs")["ok"]
    code = root / "stored_model_eval" / "recipes.py"
    code.write_text(code.read_text() + "\n# changed\n")
    v = verify_lock(lock, root, run / "inputs")
    assert not v["ok"] and "code file changed: stored_model_eval/recipes.py" in v["mismatches"]
    (root / "stored_model_eval" / "extra.py").write_text("x = 1\n")
    assert any("code file added" in m for m in verify_lock(lock, root, run / "inputs")["mismatches"])
    (root / "stored_model_eval" / "extra.py").unlink()
    code.write_text(code.read_text().replace("\n# changed\n", ""))
    assert verify_lock(lock, root, run / "inputs")["ok"]
    m = run / "inputs" / "manifest_income_prediction__sex__p0_sigma1_seed0.json"
    m.write_text(m.read_text() + " ")
    v = verify_lock(lock, root, run / "inputs")
    assert not v["ok"] and "manifest changed or missing: income_prediction__sex__p0_sigma1_seed0" in v["mismatches"]


def test_execute_refuses_on_lock_mismatch_through_cli_and_shell(world, tmp_path, monkeypatch):
    w, run = world
    lock = tmp_path / "PILOT_LOCK_v2.json"
    build_lock(WT, run / "inputs", lock, features=w["features"])
    data = json.loads(lock.read_text())
    data["code_files"]["stored_model_eval/pilot.py"] = "0" * 64      # == a changed consumed code file
    lock.write_text(json.dumps(data))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    args = ["pilot", "--execute-scientific-fits", "--synthetic", "--run-dir", str(run), "--lock", str(lock)]
    args += branch_extra().split()
    with pytest.raises(SystemExit, match="lock verification failed"):
        cli.main(args)
    assert not (run / "units").exists()
    # the shell runner refuses the same way, before any unit directory is created
    build_lock(WT, run / "inputs", lock, features=w["features"])
    m = run / "inputs" / "manifest_education_assessment__race.json"
    m.write_text(m.read_text() + " ")
    r = run_shell({"RUN_DIR": run, "LOCK": lock, "OUT": tmp_path, "EXECUTE": 1, "STAGE": "run",
                   "PILOT_EXTRA_ARGS": "--synthetic" + branch_extra()}, check=False)
    assert r.returncode != 0 and "lock verification failed" in r.stderr and not (run / "units").exists()


def test_execute_requires_single_thread(world, tmp_path, monkeypatch):
    w, run = world
    lock = tmp_path / "L.json"
    build_lock(WT, run / "inputs", lock)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(SystemExit, match="OMP_NUM_THREADS=1"):
        cli.main(["pilot", "--execute-scientific-fits", "--synthetic", "--run-dir", str(run), "--lock", str(lock)]
                 + branch_extra().split())
