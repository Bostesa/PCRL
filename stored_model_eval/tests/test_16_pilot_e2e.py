"""CELL-A pilot, end to end through the SAME shell runner and CLI path the real run uses (run_pilot.sh ->
python -m stored_model_eval lock / pilot --plan / pilot --execute-scientific-fits / infer / report), with the
frozen EFFECTIVE_PROTOCOL (grids, B = 2,000 / 20,000, seeds 20261002 / 20261003) on a synthetic world:

  positive control  income_prediction/sex: sex is directly encoded in the representation -> recovery and
                    held-out R2 established above their bars;
  null controls     income_prediction/race and education_assessment/income: the attribute is independent of the
                    representation -> never established above (race also exercises unsupported classes).
"""
import csv
import json

import numpy as np

from stored_model_eval.effective import PRIMARY_FAMILY, UNTREATED_UNITS, validate_effective
from stored_model_eval.pilot import verify_complete

from .conftest import E2E_UNITS


def _csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_runner_plan_dry_run_and_fit_counts(pilot_e2e):
    run = pilot_e2e["run"]
    plan = json.loads((run / "PLAN.json").read_text())
    assert plan["match"] and plan["selected"] == plan["expected"] and len(plan["selected"]) == 26
    dry = json.loads((run / "DRY_RUN.json").read_text())
    assert dry["fits_performed"] == 0 and dry["lock_check"]["ok"]
    for u in E2E_UNITS:
        ok, errs = verify_complete(run / "units" / u)
        assert ok, (u, errs)
        fr = json.loads((run / "units" / u / "fit_records.json").read_text())
        nf = fr["n_model_fits"]
        # untreated: 3 surfaces x (5 L + 20 GBT + 18 MLP) + 3 G2 + 5 U2 + G1 + RHO1 + LO; noise: 2 surfaces + 2 LRTs
        assert nf["total"] == (3 * 43 + 3 + 5 + 3 if "__p0_" not in u else 2 * 43 + 3 + 5 + 3 + 2), nf


def test_positive_and_null_controls(pilot_e2e):
    rows = {r["id"]: r for r in _csv(pilot_e2e["out"] / "PRIMARY_ENDPOINTS.csv")}
    assert rows["P2-income_prediction__sex"]["decision"] == "ESTABLISHED_ABOVE"
    assert rows["P1-income_prediction__sex"]["decision"] == "ESTABLISHED_ABOVE"
    for u in ("income_prediction__race", "education_assessment__income"):
        assert rows[f"P2-{u}"]["decision"] != "ESTABLISHED_ABOVE", rows[f"P2-{u}"]
        assert rows[f"P1-{u}"]["decision"] != "ESTABLISHED_ABOVE", rows[f"P1-{u}"]
    # units that were not run stay in the family as NE (family size fixed at 16)
    assert rows["P1-employment_analysis__race"]["decision"] == "NE"
    assert rows["P1-employment_analysis__race"]["reason"] == "unit outputs missing"


def test_primary_family_membership_and_adjustment_fields(pilot_e2e):
    rows = _csv(pilot_e2e["out"] / "PRIMARY_ENDPOINTS.csv")
    assert [r["id"] for r in rows] == [e["id"] for e in PRIMARY_FAMILY]
    assert [r["id"] for r in rows] == [f"P1-{u}" for u in UNTREATED_UNITS] + [f"P2-{u}" for u in UNTREATED_UNITS]
    for r in rows:
        assert float(r["alpha_each"]) == 0.05 / 16 and int(r["B"]) == 20000 and int(r["seed"]) == 20261003
        assert float(r["tail_count"]) == 62.5 and float(r["resolution"]) == 5e-5
    fam = json.loads((pilot_e2e["out"] / "PRIMARY_FAMILY.json").read_text())
    assert fam["family_size"] == 16 and len(fam["family"]) == 16 and fam["quantile_method"] == "linear"


def test_label_only_utility_and_decomposition_rows_exist(pilot_e2e):
    inf = pilot_e2e["infer"]
    ids = {r["id"] for r in inf["exploratory"]}
    for u in ("income_prediction__sex", "income_prediction__race", "education_assessment__income"):
        assert f"{u}|LO|macro_auc" in ids and f"{u}|outputs_NL_minus_LO|macro_auc" in ids
        for q in ("U1", "U2"):
            for m in ("accuracy", "log_loss", "macro_f1", "macro_auc"):
                assert f"{u}|{q}|{m}" in ids
    ut = _csv(pilot_e2e["out"] / "UTILITY.csv")
    noise_u1 = [r for r in ut if r["unit"] == "income_prediction__sex__p0_sigma8_seed0" and r["kind"] == "U1"
                and r["metric"] == "accuracy"]
    assert noise_u1 and noise_u1[0]["reference_unit"] == "income_prediction__sex" and noise_u1[0]["diff_point"]
    assert any(r["unit"] == "income_prediction__sex__p0_sigma8__seedmean" and r["kind"] == "U1_diff" for r in ut)
    dec = _csv(pilot_e2e["out"] / "DECOMPOSITION.csv")
    steps = [r["step"] for r in dec if r["unit"] == "income_prediction__sex"]
    assert steps == ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]
    nat = {r["unit"]: r for r in _csv(pilot_e2e["out"] / "NATIVE_CHECKS.csv")}
    assert nat["income_prediction__sex__p0_sigma8_seed0"]["N0_status"] == "unverified"
    assert nat["income_prediction__sex"]["N0_status"] == "REPRODUCED" and nat["income_prediction__sex"]["N1_mixed_raw"]
    sup = _csv(pilot_e2e["out"] / "SUPPORT_COVERAGE.csv")
    race = [r for r in sup if r["unit"] == "income_prediction__race" and r["what"] == "sensitive"]
    assert [r["supported"] for r in race] == ["True", "True", "True", "False", "False"]
    assert all(r["reason"] for r in race if r["supported"] == "False")


def test_heldout_r2_survives_save_load_and_gets_interval(pilot_e2e):
    inf = pilot_e2e["infer"]
    ex = {r["id"]: r for r in inf["exploratory"]}
    for u in ("income_prediction__race", "income_prediction__sex", "education_assessment__income"):
        fr = json.loads((pilot_e2e["units"] / u / "fit_records.json").read_text())
        for nm in ("G1", "G2"):
            mem = fr["closed_form"][nm]["r2_in_memory_before_save"]
            row = ex[f"{u}|{nm}|r2"]
            assert abs(row["point"] - mem) < 1e-12 and row["interval"] is not None
            assert set(row["decisions"]) == {"tau=0.01", "tau=0.02", "tau=0.05", "tau=0.1"}
    # the null unit's held-out value is negative for this fixture seed and must stay negative (no clamping)
    neg = ex["education_assessment__income|G1|r2"]
    assert neg["point"] < 0 and neg["interval"][0] < 0


def test_seed_aggregation_and_reuse(pilot_e2e):
    inf = pilot_e2e["infer"]
    ex = {r["id"]: r for r in inf["exploratory"]}
    agg = ex["income_prediction__sex__p0_sigma8__seedmean|rep|NL|macro_auc"]
    per = [ex[f"income_prediction__sex__p0_sigma8_seed{k}|rep|NL|macro_auc"]["point"] for k in range(3)]
    assert abs(agg["point"] - np.mean(per)) < 1e-12 and agg["seed_sd"] is not None
    assert any(n.get("status") == "REUSED" for n in inf["notes"])
    fr = json.loads((pilot_e2e["units"] / "income_prediction__sex__p0_sigma8_seed1" / "fit_records.json").read_text())
    out_recs = [r for r in fr["recipes"] if r["surface"] == "outputs"]
    assert out_recs and all(r["status"] == "REUSED" and r["reused_from"] == "income_prediction__sex"
                            for r in out_recs)
    with np.load(pilot_e2e["units"] / "income_prediction__sex" / "preds.npz") as a, \
            np.load(pilot_e2e["units"] / "income_prediction__sex__p0_sigma8_seed1" / "preds.npz") as b:
        assert np.array_equal(a["P__outputs__NL"], b["P__outputs__NL"])


def test_resume_skips_complete_units_without_rewriting(pilot_e2e):
    assert "skipped (--resume)" in pilot_e2e["logs"]["resume"].stderr
    for u, before in pilot_e2e["complete_before_resume"].items():
        assert (pilot_e2e["units"] / u / "COMPLETE.json").read_text() == before


def test_effective_protocol_valid_and_every_key_consumed(pilot_e2e):
    assert validate_effective() == []
    cons = pilot_e2e["infer"]["consumption"]
    assert cons["not_consumed"] == [], cons["not_consumed"]
    eff = json.loads((pilot_e2e["out"] / "EFFECTIVE_PROTOCOL.json").read_text())
    assert eff["attackers"]["L"]["C"] == [0.01, 0.1, 1.0, 10.0, 100.0] and len(eff["attackers"]["GBT"]["configs"]) == 20
