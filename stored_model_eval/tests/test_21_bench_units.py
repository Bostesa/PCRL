"""Matched removal benchmark: synthetic controls through the same CLI entry (bench.main): null, direct signal,
nonlinear-only, class contrast, absent class (NE, never a pass), B/C alias detection, scale behaviour, shuffled-label
sanity on fit/validation only, ignore-candidates reachable in the C_rep_plus_clean_out slate, and the fit phase never
receiving assessment rows (attackers, probes, outputs_only, LEACE fit)."""
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from stored_model_eval import bench
from stored_model_eval.bench_infer import infer_bench

from .test_21_bench_e2e import cli, eff_of, write_override


@pytest.fixture(autouse=True)
def _omp(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


def world_run(tmp, units, override_extra=None, tier="1", **world_kw):
    world_kw.setdefault("n_train", {"adult": 2500, "hmda": 2500})
    bench.make_bench_world(tmp / "priv", **world_kw)
    ov = write_override(tmp / "ov.json", override_extra)
    cli(tmp, "--lock-build", override=ov)
    res = cli(tmp, "--execute-scientific-fits", "--tier", tier, "--units", ",".join(units), override=ov)
    return ov, res


def preds(tmp, u):
    with np.load(tmp / "priv" / "units" / u / "preds.npz") as z:
        return {k: z[k] for k in z.files}


def frec(tmp, u):
    return json.loads((tmp / "priv" / "units" / u / "fit_records.json").read_text())


def macro(y, P, classes):
    return float(np.mean([roc_auc_score(y == c, P[:, c]) for c in classes]))


def r2(y, pred, prior):
    Y = np.eye(pred.shape[1])[y]
    return 1 - ((Y - pred) ** 2).sum() / ((Y - prior[None]) ** 2).sum()


A_SEX = "adult__s0__income_prediction__sex__A"
B_SEX = "adult__s0__income_prediction__sex__B"


def test_direct_signal_recovered_and_linear_signal_erased_by_leace(tmp_path):
    world_run(tmp_path, [A_SEX, B_SEX], datasets=("adult",), n_test={"adult": 2000})
    a, b = preds(tmp_path, A_SEX), preds(tmp_path, B_SEX)
    assert macro(a["y_s"], a["P__rep__L__as0"], [0, 1]) > 0.85 and macro(a["y_s"], a["P__rep__NL__as0"], [0, 1]) > 0.85
    assert abs(macro(b["y_s"], b["P__rep__L__as0"], [0, 1]) - 0.5) < 0.06
    assert r2(b["y_s"], b["G1_pred"], b["G1_prior"]) < 0.02 < r2(a["y_s"], a["G1_pred"], a["G1_prior"])
    nb = frec(tmp_path, B_SEX)["native"]
    assert nb["status"] == "PASS" and nb["exact_zero_crosscov"]["status"] == "WITHIN_TOLERANCE"
    # outputs are untouched by the representation defense (identical clean logits in both arms)
    assert np.array_equal(a["OUT_logits"], b["OUT_logits"])


def test_null_world_shows_no_recovery(tmp_path):
    world_run(tmp_path, [A_SEX], datasets=("adult",), n_test={"adult": 2000}, signal={})
    a = preds(tmp_path, A_SEX)
    for k in ("P__rep__NL__as0", "P__rep__L__as0", "P__repPLUSoutputs__NL__as0"):
        assert abs(macro(a["y_s"], a[k], [0, 1]) - 0.5) < 0.07, k
    assert r2(a["y_s"], a["G1_pred"], a["G1_prior"]) < 0.03


def test_nonlinear_only_signal_survives_leace_and_is_found_by_nl(tmp_path):
    world_run(tmp_path, [A_SEX, B_SEX], datasets=("adult",), n_test={"adult": 3000},
              signal={"adult": {"sex": "xor"}})
    a, b = preds(tmp_path, A_SEX), preds(tmp_path, B_SEX)
    assert abs(macro(a["y_s"], a["P__rep__L__as0"], [0, 1]) - 0.5) < 0.07          # no linear signal
    assert frec(tmp_path, B_SEX)["native"]["status"] == "PASS"                        # LEACE condition holds ...
    assert abs(macro(b["y_s"], b["P__rep__L__as0"], [0, 1]) - 0.5) < 0.07
    assert macro(b["y_s"], b["P__rep__NL__as0"], [0, 1]) > 0.65                      # ... and NL still recovers


def _sigma_stub(tmp):
    bench._write_json(tmp / "priv" / "infer" / "SIGMA_STAR.json",
                      {"datasets": {"adult": {"sigma_star": 8.0, "flagged_fallback": True},
                                    "hmda": {"sigma_star": 8.0, "flagged_fallback": True}}})


def test_class_contrast_shows_in_worst_pair_not_macro(tmp_path):
    u = "hmda__s0__underwriting__race__A"
    ov, _ = world_run(tmp_path, [u], datasets=("hmda",), n_test={"hmda": 6000}, n_train={"hmda": 3000},
                      signal={"hmda": {"race": "contrast"}})
    _sigma_stub(tmp_path)
    inf = infer_bench(tmp_path / "priv", eff=eff_of(ov), unit_ids=[u])
    sup = inf["support"]["hmda__underwriting__race"]["sensitive"]["supported_classes"]
    assert sup == [0, 1, 2, 3, 4]
    wp = next(w for w in inf["worst"] if w["arm_group"] == "A" and w["surface"] == "rep" and w["recipe"] == "NL"
              and w["statistic"] == "worst_pair_auc")
    mac = next(r for r in inf["exploratory"] if r["level"] == "group" and r["arm_group"] == "A" and
               r["surface"] == "rep" and r["recipe"] == "NL" and r["metric"] == "macro_auc")
    assert wp["argmax"] == "pair3-4" and wp["K"] == 10 and wp["point"] > mac["point"] + 0.15
    # simultaneous bound = max over per-pair bounds at alpha/K (wider than any unadjusted single interval)
    assert wp["lower"] == max(wp["per_component_lower"]) and wp["upper"] == max(wp["per_component_upper"])


def test_absent_class_makes_endpoints_not_estimable_never_pass(tmp_path):
    ov, _ = world_run(tmp_path, [A_SEX], datasets=("adult",), n_test={"adult": 1500},
                      absent_assessment_class=("adult", "sex", 1))
    fr = frec(tmp_path, A_SEX)
    assert fr["support_status"] == "NE"
    _sigma_stub(tmp_path)
    inf = infer_bench(tmp_path / "priv", eff=eff_of(ov), unit_ids=[A_SEX])
    for e in inf["primary"]["endpoints"]:
        if e["dataset"] == "adult" and e["statistic"] != "U2_accuracy_diff_vs_A":
            assert e["decision"] == "NE" and e["point"] is None, e["id"]
    assert not any(r.get("decisions") and "ESTABLISHED_BELOW" in r["decisions"].values()
                   for r in inf["exploratory"] if r.get("surface") in ("rep", "rep+outputs"))


def test_eraser_concept_below_defense_floor_is_ne(tmp_path):
    u = "adult__s0__income_prediction__sex__C"
    world_run(tmp_path, [u], datasets=("adult",), n_test={"adult": 1200}, n_train={"adult": 900})
    fr = frec(tmp_path, u)
    assert fr["status"] == "NE" and fr["concept_support"]["classes_below_floor"]


def test_b_c_alias_detected_and_scored_once(tmp_path):
    us = ["hmda__s0__underwriting__race__B", "hmda__s0__underwriting__race__C"]
    ov, _ = world_run(tmp_path, us, override_extra={"support": {"min_defense_fit_concept": 0}}, datasets=("hmda",),
                      n_test={"hmda": 1600}, race_binary_alias=True)
    c = frec(tmp_path, us[1])
    assert c["alias_of"] == us[0] and c["alias_check"]["alias"] and c["n_model_fits"]["total"] == 0
    assert c["alias_check"]["max_abs_P_diff"] <= 1e-10
    _sigma_stub(tmp_path)
    inf = infer_bench(tmp_path / "priv", eff=eff_of(ov), unit_ids=us + ["hmda__s0__underwriting__race__A"])
    assert inf["aliases"] == {us[1]: us[0]}
    ep = {e["id"]: e for e in inf["primary"]["endpoints"]}
    assert ep["P-hmda-Rrep-C"]["alias_of"] == us[0] and "alias row" in ep["P-hmda-Rrep-C"].get("note", "")
    assert len(inf["primary"]["endpoints"]) == 24


def test_scale_invariance_and_sigma_relative_scale(tmp_path):
    units = [A_SEX, B_SEX, "adult__s0__income_prediction__sex__D_sigma1_rs0"]
    out = {}
    for sc in (1.0, 10.0):
        t = tmp_path / f"s{sc:g}"
        t.mkdir()
        world_run(t, units, datasets=("adult",), n_test={"adult": 1500}, rep_scale=sc)
        out[sc] = {u: preds(t, u) for u in units} | {"D": frec(t, units[2])}
    for u in units[:2]:
        p1, p10 = out[1.0][u], out[10.0][u]
        assert abs(r2(p1["y_s"], p1["G2_pred"], p1["G2_prior"]) - r2(p10["y_s"], p10["G2_pred"], p10["G2_prior"])) < 1e-6
        assert abs(macro(p1["y_s"], p1["P__rep__L__as0"], [0, 1]) - macro(p10["y_s"], p10["P__rep__L__as0"], [0, 1])) < 1e-6
    s1 = out[1.0]["D"]["release"]["scale_report"]["sigmas"]
    s10 = out[10.0]["D"]["release"]["scale_report"]["sigmas"]
    r1 = next(x for x in s1 if x["sigma_abs"] == 1.0)["sigma_over_rms_sd"]
    r10 = next(x for x in s10 if x["sigma_abs"] == 1.0)["sigma_over_rms_sd"]
    assert abs(r1 / r10 - 10.0) < 1e-3          # equal absolute sigma is not an equal operating point


def test_shuffled_label_sanity_is_fit_validation_only(tmp_path, monkeypatch):
    bench.make_bench_world(tmp_path / "priv", datasets=("adult",), n_test={"adult": 2000}, n_train={"adult": 2500})
    ov = write_override(tmp_path / "ov.json")
    cli(tmp_path, "--lock-build", override=ov)
    seen = []
    real = bench.fit_slate

    def spy(Xf, yf, Xv, yv, *a, **k):
        seen.append((len(Xf), len(Xv)))
        return real(Xf, yf, Xv, yv, *a, **k)
    monkeypatch.setattr(bench, "fit_slate", spy)
    res = cli(tmp_path, "--shuffled-label-sanity", "--units", A_SEX, override=ov)
    r = res["units"][A_SEX]
    assert seen == [(r["rows"]["attacker_fit"], r["rows"]["attacker_val"])] and r["assessment_rows_used"] == 0
    assert abs(r["val_auc_shuffled"]["L"] - 0.5) < 0.1 and abs(r["val_auc_shuffled"]["NL"] - 0.5) < 0.12
    assert not (tmp_path / "priv" / "units").exists() or not any((tmp_path / "priv" / "units").iterdir())


def test_select_plus_reaches_ignore_candidates():
    assert bench.select_plus({"GBT": 0.6, "MLP": 0.61, "ignore_rep": 0.5, "ignore_out": 0.7}) == "ignore_rep"
    assert bench.select_plus({"GBT": 0.6, "MLP": 0.61, "ignore_rep": 0.65, "ignore_out": 0.55}) == "ignore_out"
    assert bench.select_plus({"L": 0.5, "ignore_rep": 0.5, "ignore_out": 0.6}) == "L"           # tie -> listed first
    assert bench.select_plus({"GBT": float("nan"), "MLP": 0.7, "ignore_rep": 0.69, "ignore_out": 0.9}) == "ignore_rep"


def test_selected_ignore_candidates_alias_their_source_predictions(tmp_path, monkeypatch):
    choice = iter(["ignore_rep", "ignore_out"])
    monkeypatch.setattr(bench, "select_plus", lambda c: next(choice))
    world_run(tmp_path, [A_SEX], datasets=("adult",), n_test={"adult": 1200})
    p, fr = preds(tmp_path, A_SEX), frec(tmp_path, A_SEX)
    for k in range(3):
        assert np.array_equal(p[f"P__repPLUSoutputs__NL__as{k}"], p[f"P__outputs__NL__as{k}"])
        assert np.array_equal(p[f"P__repPLUSoutputs__Lslate__as{k}"], p[f"P__rep__L__as{k}"])
    sel = next(r for r in fr["recipes"] if r["surface"] == "rep+outputs" and r["recipe"] == "NL")["plus_selection"]
    assert sel["NL"]["selected"] == "ignore_rep" and sel["Lslate"]["selected"] == "ignore_out"
    assert set(sel["NL"]["candidates_attacker_val_log_loss"]) == {"GBT", "MLP", "ignore_rep", "ignore_out"}
    assert all(v["status"] == "ALIAS_IGNORE_CANDIDATE" for v in sel["NL"]["seed_retrain"]["seeds"].values())


def test_fit_phase_and_every_selection_see_only_fit_and_validation_rows(tmp_path, monkeypatch):
    import stored_model_eval.defenses as defs
    bench.make_bench_world(tmp_path / "priv", datasets=("adult",), n_test={"adult": 1500}, n_train={"adult": 2500})
    ov = write_override(tmp_path / "ov.json")
    cli(tmp_path, "--lock-build", override=ov)
    D = bench.Inputs(tmp_path / "priv" / "inputs" / "INPUTS_INDEX.json").dataset("adult")
    n = {r: len(D["idx"][r]) for r in bench.ROLE_ORDER}
    sizes, leace, views = [], [], []
    real_ff, real_fl, real_fp = bench.fit_family, defs.fit_leace, bench.fit_phase

    def ff(fam, Xf, yf, Xv, yv, *a, **k):
        sizes.append((len(Xf), len(Xv)))
        return real_ff(fam, Xf, yf, Xv, yv, *a, **k)

    def fl(H, Z, **k):
        leace.append((len(H), sorted(np.asarray(k["fit_row_ids"]).tolist())))
        return real_fl(H, Z, **k)

    def fp(view, ctx):
        views.append(sorted(view))
        ids = np.concatenate([view["fit"]["row_id"], view["val"]["row_id"]])
        assert not set(ids.tolist()) & set(D["row_id"][D["idx"]["assessment"]].tolist())
        return real_fp(view, ctx)
    monkeypatch.setattr(bench, "fit_family", ff)
    monkeypatch.setattr(defs, "fit_leace", fl)
    monkeypatch.setattr(bench, "fit_phase", fp)
    cli(tmp_path, "--execute-scientific-fits", "--tier", "1", "--units",
        ",".join([A_SEX, B_SEX, "adult__s0__income_prediction__sex__D_sigma4_rs2"]), override=ov)
    assert views == [["fit", "val"]] * 3
    assert sizes and set(sizes) == {(n["attacker_fit"], n["attacker_val"])}
    assert leace == [(n["defense_fit"], sorted(D["row_id"][D["idx"]["defense_fit"]].tolist()))]
    for u in (A_SEX, B_SEX):
        for r in frec(tmp_path, u)["recipes"]:
            for row in r.get("selection_table") or []:
                assert set(row) <= {"hp", "attacker_val_log_loss", "n_iter", "rho", "attacker_val_heldout_r2_fitmean"}
    with pytest.raises(ValueError, match="only fit/val"):
        real_fp({"fit": {}, "val": {}, "assessment": {}}, {})
