"""Matched removal benchmark: one synthetic end-to-end run through the SAME shell runner / CLI path
(lock -> plan -> dry-run -> tier1 -> sigma-star -> tier2 (subset) -> infer -> report -> plots -> resume), and checks on
its outputs: exact unit lists, missing seeds reported, attacker-seed retraining saved, replicate summary computed within
each bootstrap replicate, sigma* from validation only, primary family of 24, budget/trigger gates, lock refusals."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stored_model_eval import bench
from stored_model_eval.bench_effective import (BENCH_BRANCH, BENCH_EFFECTIVE, parse_unit, primary_family,
                                               tier1_units, tier2_units)
from stored_model_eval.pilot_infer import UnitBootstrap, _auc_eval, _auc_prep

WT = Path(__file__).resolve().parents[2]
RUNNER = WT / "results/combined_matched_removal_benchmark_v1/scripts/run_benchmark.sh"
SMALL = {"attackers": {"L": {"C": [1.0]},
                       "GBT": {"configs": [{"learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 20,
                                            "l2_regularization": 0.0},
                                           {"learning_rate": 0.3, "max_leaf_nodes": 15, "min_samples_leaf": 50,
                                            "l2_regularization": 1.0}], "max_iter": 60},
                       "MLP": {"hidden_layer_sizes": [[16, 16]], "alpha": [1e-4], "learning_rate_init": [3e-3],
                               "max_iter": 80}},
         "utility": {"U2": {"C": [1.0]}},
         "inference": {"exploratory": {"B": 300}, "primary": {"B": 2400}}}


def branch_extra() -> list[str]:
    return [] if bench.git_branch(WT) == BENCH_BRANCH else ["--allow-other-branch-for-tests"]


def write_override(path: Path, extra: dict | None = None) -> Path:
    o = copy.deepcopy(SMALL)
    if extra:
        o = bench._deep_merge(o, extra)
    path.write_text(json.dumps(o))
    return path


def cli(tmp: Path, *args, override: Path | None = None, priv: Path | None = None) -> dict:
    """In-process call of the same CLI entry (`python -m stored_model_eval bench ...`)."""
    priv = priv or tmp / "priv"
    a = list(args) + ["--private-root", str(priv), "--lock", str(tmp / "LOCK.json"), "--package-dir",
                      str(tmp / "pkg"), "--synthetic", "--effective-override",
                      str(override or write_override(tmp / "override.json"))] + branch_extra()
    return bench.main(a)


def eff_of(override: Path) -> dict:
    return bench._deep_merge(BENCH_EFFECTIVE, json.loads(Path(override).read_text()))


def run_shell(env_extra: dict, check=True):
    env = dict(os.environ, **{k: str(v) for k, v in env_extra.items()})
    env["PY"] = sys.executable
    r = subprocess.run(["sh", str(RUNNER)], env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"runner failed ({r.returncode}):\nSTDOUT\n{r.stdout}\nSTDERR\n{r.stderr[-4000:]}")
    return r


E1_SUBSET = ["adult__s0__income_prediction__race__A", "adult__s0__income_prediction__race__B"]


@pytest.fixture(scope="session")
def e2e(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench_e2e")
    priv = root / "priv"
    bench.make_bench_world(priv, n_test={"adult": 1600, "hmda": 1600}, n_train={"adult": 2500, "hmda": 2500},
                           failed_seeds=(1,))
    ov = write_override(root / "override.json")
    env = {"PRIVATE": priv, "LOCK": root / "LOCK.json", "PKG": root / "pkg",
           "BENCH_EXTRA_ARGS": " ".join(["--synthetic", "--effective-override", str(ov)] + branch_extra())}
    logs = {}
    for st in ("lock", "plan", "dry-run"):
        logs[st] = run_shell({**env, "STAGE": st})
    logs["sanity"] = run_shell({**env, "STAGE": "sanity", "UNITS": "adult__s0__income_prediction__sex__A,"
                                                                  "hmda__s0__underwriting__race__B"})
    for st in ("tier1", "sigma-star"):
        logs[st] = run_shell({**env, "STAGE": st})
    logs["tier2"] = run_shell({**env, "STAGE": "tier2", "UNITS": ",".join(E1_SUBSET)})
    complete_before = {u: (priv / "units" / u / "COMPLETE.json").read_text() for u in tier1_units()
                       if (priv / "units" / u).exists()}
    logs["resume"] = run_shell({**env, "STAGE": "tier1", "RESUME": 1})
    for st in ("infer", "report", "plots"):
        logs[st] = run_shell({**env, "STAGE": st, "TIER": "all"})
    return {"root": root, "priv": priv, "units": priv / "units", "pkg": root / "pkg", "env": env, "ov": ov,
            "logs": logs, "complete_before": complete_before,
            "infer": json.loads((priv / "infer" / "BENCH_INFER.json").read_text())}


# ---- unit lists --------------------------------------------------------------------------------------------------


def test_exact_unit_lists_and_selection():
    t1 = tier1_units()
    assert len(t1) == 126 and len(set(t1)) == 126
    assert t1[:4] == ["adult__s0__income_prediction__sex__A", "adult__s0__income_prediction__sex__B",
                      "adult__s0__income_prediction__sex__C", "adult__s0__income_prediction__sex__D_sigma0.25_rs0"]
    assert t1[-1] == "hmda__s2__underwriting__race__D_sigma8_rs2"
    t2 = tier2_units()
    assert {k: len(v) for k, v in t2.items()} == {"E1": 72, "E2": 36, "E3": 648}
    assert all(parse_unit(u)["arm"] in ("A", "B") for u in t2["E1"]) and all(parse_unit(u)["arm"] == "C" for u in t2["E2"])
    assert bench.select("1", ["hmda__s1__underwriting__race__C"]) == ["hmda__s1__underwriting__race__C"]
    with pytest.raises(SystemExit, match="unregistered"):
        bench.select("1", ["adult__s0__income_prediction__sex__D_sigma3_rs0"])
    with pytest.raises(SystemExit, match="not in tier 1"):
        bench.select("1", E1_SUBSET)
    fam = primary_family()
    assert len(fam) == 24 and len({e["id"] for e in fam}) == 24
    assert [e["id"] for e in fam if e["dataset"] == "adult"] == (
        ["P-adult-G1-A"] + [f"P-adult-Rrep-{m}" for m in ("A", "B", "C", "Dstar")]
        + [f"P-adult-Rplus-{m}" for m in ("A", "B", "C", "Dstar")] + [f"P-adult-U2NI-{m}" for m in ("B", "C", "Dstar")])


def test_plan_reports_missing_seeds_with_reason_and_runs_available_units(e2e):
    plan = json.loads((e2e["priv"] / "logs" / "PLAN_T1.json").read_text())
    assert plan["n_expected"] == 126 and plan["n_runnable"] == 42
    assert "lineage FAILED" in plan["missing_seeds"]["adult__s1"]["reason"]
    assert "adult_s1.pt" in plan["missing_seeds"]["adult__s1"]["reason"]
    assert "not declared" in plan["missing_seeds"]["hmda__s2"]["reason"]
    done = sorted(p.name for p in e2e["units"].iterdir() if (p / "COMPLETE.json").exists())
    assert set(plan["runnable"]) | set(E1_SUBSET) == set(done)
    dry = json.loads((e2e["priv"] / "logs" / "DRY_RUN_T1.json").read_text())
    assert dry["fits_performed"] == 0 and dry["input_hash_mismatches"] == [] and dry["lock_check"]["ok"]
    assert dry["datasets"]["hmda"]["hmda_rule_agreement"] == 1.0
    assert dry["datasets"]["adult"]["ignored_role_values_dropped"]["dropped"] == {"excluded_dup": 5}


def test_resume_skips_hash_verified_units(e2e):
    for u, before in e2e["complete_before"].items():
        assert (e2e["units"] / u / "COMPLETE.json").read_text() == before
    logs = sorted((e2e["priv"] / "logs").glob("EXECUTE_T1_*.json"))
    last = json.loads(logs[-1].read_text())
    assert last["completed"] == [] and len(last["skipped_complete"]) == 42


# ---- per-unit contents -------------------------------------------------------------------------------------------


def test_attacker_seed_retraining_is_saved_and_deterministic_recipes_aliased(e2e):
    u = e2e["units"] / "adult__s0__income_prediction__sex__A"
    fr = json.loads((u / "fit_records.json").read_text())
    nl = next(r for r in fr["recipes"] if r["surface"] == "rep" and r["recipe"] == "NL")
    seeds = nl["seed_retrain"]["seeds"]
    assert seeds["0"]["status"] == "GRID_FIT" and seeds["1"]["status"] == seeds["2"]["status"] == "REFIT"
    assert seeds["1"]["random_state"] == 1 and seeds["2"]["random_state"] == 2
    lrec = next(r for r in fr["recipes"] if r["recipe"] == "L_seeds")["seed_retrain"]["seeds"]
    assert all(v["status"] == "ALIAS_DETERMINISTIC" for v in lrec.values())
    with np.load(u / "preds.npz") as z:
        assert all(f"P__rep__NL__as{k}" in z.files and f"P__repPLUSoutputs__NL__as{k}" in z.files for k in range(3))
        assert not np.array_equal(z["P__rep__NL__as0"], z["P__rep__NL__as1"])        # a real refit
        assert np.array_equal(z["P__rep__L__as0"], z["P__rep__L__as2"])              # deterministic alias
        assert np.array_equal(z["P__repPLUSoutputs__ignore_out"], z["P__rep__NL__as0"])
        assert np.array_equal(z["P__repPLUSoutputs__ignore_rep"], z["P__outputs__NL__as0"])
    for k in (1, 2):
        assert (u / "models" / f"rep__NL__as{k}.joblib").exists()
    assert fr["model_files"]["rep__NL|as0"]["alias_of"].endswith(".joblib")


def test_outputs_only_fitted_once_and_aliased_across_arms(e2e):
    recs = {}
    for arm in ("A", "B", "C", "D_sigma1_rs0", "D_sigma8_rs2"):
        fr = json.loads((e2e["units"] / f"hmda__s0__underwriting__race__{arm}" / "fit_records.json").read_text())
        recs[arm] = fr["shared"]["outputs_only"]
    assert len({r["preds_sha256"] for r in recs.values()}) == 1
    assert recs["A"]["reused"] is False and all(recs[a]["reused"] for a in recs if a != "A")
    with np.load(e2e["units"] / "hmda__s0__underwriting__race__A" / "preds.npz") as a, \
            np.load(e2e["units"] / "hmda__s0__underwriting__race__D_sigma4_rs1" / "preds.npz") as d:
        assert np.array_equal(a["P__outputs__NL__as2"], d["P__outputs__NL__as2"])
        assert np.array_equal(a["OUT_logits"], d["OUT_logits"]) and np.array_equal(a["assess_row_id"], d["assess_row_id"])


def test_native_checks_and_maps_pinned(e2e):
    a = json.loads((e2e["units"] / "adult__s0__income_prediction__sex__A" / "fit_records.json").read_text())
    assert a["native"]["reproduction"]["status"] == "REPRODUCED" and a["native"]["rows"] == "test_split"
    b = json.loads((e2e["units"] / "adult__s0__income_prediction__sex__B" / "fit_records.json").read_text())
    assert b["native"]["status"] == "PASS" and b["native"]["primary"]["holds"]
    assert b["release"]["map"]["fit_role"] == "defense_fit" and b["release"]["map"]["n_fit_rows"] == b["roles"]["defense_fit"]
    c = json.loads((e2e["units"] / "adult__s0__income_prediction__sex__C" / "fit_records.json").read_text())
    assert c["alias_check"]["alias"] is False and c["release"]["map"]["concept_spec"]["attributes"] == ["race", "sex"]
    d = json.loads((e2e["units"] / "adult__s0__income_prediction__sex__D_sigma2_rs1" / "fit_records.json").read_text())
    assert d["native"]["status"] == "NA" and d["contract"]["noise"] == "persistent_token"
    pins = json.loads((e2e["priv"] / "defenses" / "MAP_PINS.json").read_text())["maps"]
    assert set(pins) == {"adult__s0__income_prediction__B_sex", "adult__s0__income_prediction__C_race+sex",
                         "hmda__s0__underwriting__B_race", "hmda__s0__underwriting__C_race+ethnicity",
                         "adult__s0__income_prediction__B_race"}
    assert bench.MapStore(e2e["priv"]).verify() == []


def test_noise_draw_is_persistent_across_attributes_and_rerun(e2e):
    import stored_model_eval.defenses as defs
    inputs = bench.Inputs(e2e["priv"] / "inputs" / "INPUTS_INDEX.json")
    D = inputs.dataset("adult")
    H = inputs.forward("adult", 0)["rep_p0"]
    rows = bench.noise_matrix_rows(D)
    R = defs.noise_release(H[rows], 2.0, 1)
    fr = json.loads((e2e["units"] / "adult__s0__income_prediction__sex__D_sigma2_rs1" / "fit_records.json").read_text())
    assert fr["release"]["release_scored_rows_sha256"] == bench._sha_arr(R)
    R2 = defs.noise_release(H[rows], 2.0, 1)
    assert np.array_equal(R, R2)


# ---- sigma* and inference ------------------------------------------------------------------------------------------


def test_sigma_star_reads_validation_only(e2e, monkeypatch):
    opened = []
    real = np.load

    def spy(f, *a, **k):
        opened.append(Path(f).name)
        return real(f, *a, **k)
    monkeypatch.setattr(np, "load", spy)
    inputs = bench.Inputs(e2e["priv"] / "inputs" / "INPUTS_INDEX.json")
    res = bench.sigma_star(e2e["priv"], inputs, eff_of(e2e["ov"]))
    assert opened and set(opened) == {"val_preds.npz"}
    saved = json.loads((e2e["priv"] / "infer" / "SIGMA_STAR.json").read_text())
    assert {d: v["sigma_star"] for d, v in res["datasets"].items()} == {d: v["sigma_star"] for d, v in
                                                                      saved["datasets"].items()}
    for d, v in res["datasets"].items():
        curve = v["val_curve_mean"]
        chosen = v["sigma_star"]
        qual = [float(s) for s, x in curve.items() if x is not None and x <= 0.55]
        assert chosen == (min(qual) if qual else 8.0) and v["flagged_fallback"] == (not qual)


def test_primary_family_and_replicate_summary_within_bootstrap(e2e):
    inf = e2e["infer"]
    eps = {e["id"]: e for e in inf["primary"]["endpoints"]}
    assert len(eps) == 24 and inf["primary"]["family_size"] == 24
    assert abs(inf["primary"]["alpha_each"] * 24 - 0.05) < 1e-15
    e = eps["P-adult-Rrep-Dstar"]
    sstar = inf["sigma_star"]["values"]["adult"]
    assert e["arm_group"] == f"D_sigma{sstar:g}" and len(e["members"]) == 3
    # recompute: mean over release seeds x attacker seeds of per-unit macro AUC WITHIN each replicate
    preds = []
    for u in e["members"]:
        with np.load(e2e["units"] / u / "preds.npz") as z:
            preds.append({k: z[k] for k in z.files})
    sup = json.loads((e2e["units"] / e["members"][0] / "supported.json").read_text())["sensitive"]["supported_classes"]
    y = preds[0]["y_s"]
    preps = [_auc_prep(p[f"P__rep__NL__as{k}"][:, c], y == c, np.arange(len(y)), len(y))
             for p in preds for k in range(3) for c in sup]
    boot = UnitBootstrap(preds[0]["assess_unit"], inf["primary"]["B"], inf["primary"]["seed"], 500)
    reps = []
    for W in boot.chunks():
        per = np.stack([_auc_eval(pp, W) for pp in preps]).reshape(9, len(sup), -1).mean(1)   # macro per (unit, as)
        reps.append(per.mean(0))                                                            # mean within replicate
    reps = np.concatenate(reps)
    a = inf["primary"]["alpha_each"]
    assert abs(np.quantile(reps, a, method="linear") - e["lower"]) < 1e-12
    assert abs(np.quantile(reps, 1 - a, method="linear") - e["upper"]) < 1e-12
    # U2NI is a paired difference on identical people and uses the NI rule
    u2 = eps["P-adult-U2NI-B"]
    assert u2["decision"] in ("NONINFERIOR", "INFERIOR", "UNRESOLVED")
    if u2["lower"] >= -0.01:
        assert u2["decision"] == "NONINFERIOR"


def test_worst_class_bounds_are_simultaneous_max_of_component_bounds(e2e):
    rows = [w for w in e2e["infer"]["worst"] if w["status"] == "ESTIMATED"]
    assert rows
    for w in rows:
        assert abs(w["lower"] - max(w["per_component_lower"])) < 1e-15
        assert abs(w["upper"] - max(w["per_component_upper"])) < 1e-15
        assert abs(w["point"] - max(w["per_component_point"])) < 1e-15


def test_report_tables_and_plots_written_without_private_paths(e2e):
    pkg = e2e["pkg"]
    for f in ("PRIMARY_ENDPOINTS.csv", "NATIVE_AND_HELDOUT_CHECKS.csv", "RECOVERY.csv", "UTILITY.csv",
              "MATCHED_COMPARISONS.csv", "SUPPORT_COVERAGE.csv", "SEED_VARIATION.csv", "PRIMARY_FAMILY.json",
              "MODEL_MANIFEST.json", "SIGMA_STAR_SUMMARY.json", "EFFECTIVE_PROTOCOL.json"):
        assert (pkg / f).exists(), f
        assert str(e2e["priv"]) not in (pkg / f).read_text()
    plots = json.loads((pkg / "plots" / "PLOTS_INDEX.json").read_text())["plots"]
    assert len(plots) >= 4 and all((pkg / p).exists() for p in plots)
    mm = json.loads((pkg / "MODEL_MANIFEST.json").read_text())
    assert mm["by_kind"]["official_leace_map"] >= 8 and all(not f["path"].startswith("/") for f in mm["files"])
    cons = json.loads((pkg / "notes" / "evaluation" / "CONSUMPTION.json").read_text())["consumption"]
    assert cons["not_consumed_other"] == [], cons["not_consumed_other"]
    util = (pkg / "UTILITY.csv").read_text()
    assert "unchanged across methods" in util and "outside LEACE's scope" in util


# ---- gates --------------------------------------------------------------------------------------------------------


def test_lock_refuses_changed_code_input_or_map(e2e, tmp_path, monkeypatch):
    from stored_model_eval.bench_lock import verify_bench_lock
    lock = json.loads((e2e["root"] / "LOCK.json").read_text())
    idx = e2e["priv"] / "inputs" / "INPUTS_INDEX.json"
    eff = eff_of(e2e["ov"])
    assert verify_bench_lock(e2e["root"] / "LOCK.json", WT, idx, e2e["priv"], eff)["ok"]
    bad = copy.deepcopy(lock)
    bad["code_files"]["stored_model_eval/defenses.py"] = "0" * 64
    (tmp_path / "L1.json").write_text(json.dumps(bad))
    v = verify_bench_lock(tmp_path / "L1.json", WT, idx, e2e["priv"], eff)
    assert not v["ok"] and "code file changed: stored_model_eval/defenses.py" in v["mismatches"]
    bad = copy.deepcopy(lock)
    p = next(iter(bad["inputs"]["files_expected"]))
    bad["inputs"]["files_expected"][p] = "1" * 64
    (tmp_path / "L2.json").write_text(json.dumps(bad))
    v = verify_bench_lock(tmp_path / "L2.json", WT, idx, e2e["priv"], eff)
    assert not v["ok"] and any("input file changed" in m for m in v["mismatches"])
    bad = copy.deepcopy(lock)
    bad["units"]["tier1"] = bad["units"]["tier1"][:-1]
    (tmp_path / "L3.json").write_text(json.dumps(bad))
    assert "units changed" in verify_bench_lock(tmp_path / "L3.json", WT, idx, e2e["priv"], eff)["mismatches"]
    # the CLI refuses execution before creating any unit directory
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    args = ["--execute-scientific-fits", "--tier", "1", "--units", "hmda__s0__underwriting__race__A",
            "--private-root", str(e2e["priv"]), "--lock", str(tmp_path / "L1.json"), "--package-dir",
            str(tmp_path / "pkg"), "--synthetic", "--effective-override", str(e2e["ov"]), "--resume"] + branch_extra()
    with pytest.raises(SystemExit, match="lock verification failed"):
        bench.main(args)


def test_map_pin_tamper_and_unpinned_map_refused(e2e, tmp_path):
    import shutil
    priv = tmp_path / "priv"
    shutil.copytree(e2e["priv"] / "defenses", priv / "defenses")
    ms = bench.MapStore(priv)
    assert ms.verify() == []
    f = next((priv / "defenses" / "adult__s0__income_prediction__B_sex").rglob("*.npz"))
    f.write_bytes(f.read_bytes() + b"x")
    assert any("hash changed" in m for m in ms.verify())
    (priv / "defenses" / "rogue_map").mkdir()
    assert any("not pinned" in m for m in ms.verify())


def test_tier2_gate_and_budget_stop_rule(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    bench.make_bench_world(tmp_path / "priv", n_test={"adult": 1200, "hmda": 1200},
                           n_train={"adult": 2500, "hmda": 2500}, datasets=("adult", "hmda"))
    ov = write_override(tmp_path / "ov.json", {"budget": {"cpu_hours": 1e-9}})
    cli(tmp_path, "--lock-build", override=ov)
    with pytest.raises(SystemExit, match="Tier-2 trigger not met"):
        cli(tmp_path, "--execute-scientific-fits", "--tier", "2", override=ov)
    t1 = [u for u in tier1_units() if "__s0__" in u]
    cli(tmp_path, "--execute-scientific-fits", "--tier", "1", override=ov)
    with pytest.raises(SystemExit, match="SIGMA_STAR.json missing"):
        cli(tmp_path, "--execute-scientific-fits", "--tier", "2", override=ov)
    cli(tmp_path, "--sigma-star", override=ov)
    res = cli(tmp_path, "--execute-scientific-fits", "--tier", "2", override=ov)
    assert res["completed"] == [] and res["budget_stop"]["at_unit"] == tier2_units()["E1"][0]
    led = json.loads((tmp_path / "priv" / "logs" / "BUDGET_LEDGER.json").read_text())
    assert led["stops"] and sum(1 for e in led["entries"] if e["kind"] == "unit") == len(t1)
    assert not (tmp_path / "priv" / "units" / tier2_units()["E1"][0]).exists()
