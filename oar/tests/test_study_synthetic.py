"""Synthetic smoke tests for the study runner (no real data): units, plus-aliasing, heads, cell-conditional
attacker, controls, and the inference graph on identical assessment rows."""
import os

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")


@pytest.fixture()
def world(tmp_path, monkeypatch):
    import oar.study as S
    monkeypatch.setattr(S, "RUN", tmp_path / "run")
    rng = np.random.default_rng(0)
    n = 1600
    s = rng.integers(0, 2, n)
    t = (rng.random(n) < 0.3 + 0.4 * s).astype(int)
    H = rng.normal(size=(n, 6)) + 0.8 * s[:, None] * np.array([1, 0, 0, 0, 0, 0])
    role = np.array(["defense_fit"] * 600 + ["attacker_fit"] * 500 + ["attacker_val"] * 200 + ["assessment"] * 300)
    W = {"row_id": np.arange(n), "unit": np.arange(n), "role": role, "s": s, "t": t,
         "idx": {r: np.flatnonzero(role == r) for r in ("defense_fit", "attacker_fit", "attacker_val", "assessment")}}
    W["idx"]["head_fit"] = W["idx"]["defense_fit"][:480]
    W["idx"]["head_val"] = W["idx"]["defense_fit"][480:]
    O = np.stack([-(H[:, 0] + t), H[:, 0] + t + rng.normal(size=n)], 1)
    return S, W, H, O, s, t


def _small_effective(S):
    import copy
    from stored_model_eval.bench_effective import BENCH_EFFECTIVE, Tracked
    e = copy.deepcopy(BENCH_EFFECTIVE)
    e["attackers"]["GBT"]["configs"] = e["attackers"]["GBT"]["configs"][:2]
    e["attackers"]["MLP"]["hidden_layer_sizes"] = [[16]]
    e["attackers"]["MLP"]["alpha"] = [1e-4]
    e["attackers"]["MLP"]["learning_rate_init"] = [1e-3]
    e["attackers"]["MLP"]["max_iter"] = 50
    return Tracked(e)


def test_units_plus_heads_and_inference(world):
    from stored_model_eval.guards import FitAuthorization
    from oar import infer as I
    S, W, H, O, s, t = world
    E, auth = _small_effective(S), FitAuthorization.synthetic_only()
    var, hard = S.output_variants(O)
    assert np.array_equal(var["hard"].argmax(1), O.argmax(1))
    S.attack_unit("u__O_full", var["full"], W, s, 2, E, auth, True)
    S.attack_unit("u__O_hard", var["hard"], W, s, 2, E, auth, True, finite=True)
    S.attack_unit("u__A__rep", H, W, s, 2, E, auth, True)
    r = S.attack_unit("u__A__rep+clean", np.hstack([H, O]), W, s, 2, E, auth, True,
                      plus={"rep_unit": "u__A__rep", "out_unit": "u__O_full"})
    assert r["plus_selection"]["selected"] in ("GBT/MLP", "ignore_rep", "ignore_out")
    h = S.head_unit("u__HEAD__A", H, W, t, 2, E, auth, True)
    assert h["outputs"].shape == (len(s), 2) and h["record"]["inputs_at_runtime"] == "protected features only"
    S.u2_unit("u__U2__A", H, W, t, 2, E, auth, True)
    S.reference_unit("u__REF", W, s, t, 2, 2, E, auth, True)
    assert S.unit_complete("u__A__rep")
    cg = I.CellGraph([0, 1], [[0, 1]])
    a = cg.recovery_of("u__O_full")
    b = cg.recovery_of("u__O_hard")
    c = cg.recovery_of("u__A__rep+clean")
    d = cg.diff("d", a, b)
    acc = cg.accuracy_of("u__U2__A")
    pts, reps, _ = I.bootstrap(cg, [a, b, c, d, acc], 200, 1)
    assert 0.5 < pts[a] <= 1 and np.isfinite(reps[d]).all()
    # resume: a complete unit is skipped (record returned unchanged)
    assert S.attack_unit("u__A__rep", H, W, s, 2, E, auth, True)["id"] == "u__A__rep"


def test_cell_conditional_and_controls(world):
    from stored_model_eval.guards import FitAuthorization
    S, W, H, O, s, t = world
    E, auth = _small_effective(S), FitAuthorization.synthetic_only()
    cells = (H[:, 0] > 0).astype(int) * 2 + (H[:, 1] > 0).astype(int)
    X = S.onehot(cells, 4)
    cc = S.fit_cell_conditional(X[:500], s[:500], X[500:700], s[500:700], 2, 1e-12)
    P = cc["model"].predict_proba(X[700:])
    assert np.allclose(P.sum(1), 1)
    ctl = S.null_and_planted("u__ctl", X, W, s, 2, E, auth, True, finite=True)
    assert ctl["null_val_macro_auc"] < 0.6 and ctl["planted_detected_above_0.75"]


def test_alias_report_detects_logit_offset():
    import oar.study as S
    rng = np.random.default_rng(1)
    O = rng.normal(size=(50, 2))
    rep = S.alias_report(O)
    assert not rep["logit_sum_is_constant"] and rep["prob_determines_centred_logits_maxabs"] < 1e-9


def test_fare_stage_end_to_end_synthetic(world, tmp_path, monkeypatch):
    """FARE stage through the official wrapper on synthetic data: grid -> alias -> validation nominee -> views ->
    zero-fairness twin -> certificate (cert rows disjoint from fit rows)."""
    import json
    from stored_model_eval.guards import FitAuthorization
    from oar import fare_run as FR
    S, W, H, O, s, t = world
    monkeypatch.setenv("OAR_RUN_UNITS", str(tmp_path / "run" / "units"))
    E, auth = _small_effective(S), FitAuthorization.synthetic_only()
    # carve a cert role out of attacker_fit for the synthetic world
    W["idx"]["cert"] = W["idx"]["attacker_fit"][:150]
    W["idx"]["attacker_fit"] = W["idx"]["attacker_fit"][150:]
    S.u2_unit("P__U2__A", H, W, t, 2, E, auth, True)
    S.attack_unit("P__O_full", O, W, s, 2, E, auth, True)
    grid = [{"id": 1, "max_leaf_nodes": 8, "min_samples_leaf": 20, "gamma": 0.3, "criterion": "fair_gini_dp"},
            {"id": 2, "max_leaf_nodes": 8, "min_samples_leaf": 20, "gamma": 0.3, "criterion": "fair_gini_dp"},
            {"id": 3, "max_leaf_nodes": 3, "min_samples_leaf": 50, "gamma": 0.9, "criterion": "fair_gini_dp"}]
    fare = {"grid": grid, "seed_base": 0, "supported_classes": {"adult": [0, 1]}, "sensitive_for_fit": "full_declared",
            "certificate": {"delta": 0.05, "groups": None, "split_seed": 0, "val_fraction": 0.5,
                            "eps_b_fraction": 0.1, "eps_s_fraction": 0.1}}
    led = []

    def atk(uid, X, finite=False, plus=None, contract=None):
        return S.attack_unit(uid, X, W, s, 2, E, auth, True, finite=finite, plus=plus, contract=contract)

    def u2(uid, X):
        return S.u2_unit(uid, X, W, t, 2, E, auth, True)

    sel = FR.run_fare_seed("adult", 0, "P", W, H, O, {"full": O}, {}, {}, fare, E, auth, True, atk, u2,
                           lambda *a: led.append(a), print)
    assert sel["aliases"] == {"2": 1}                       # identical config -> alias, not refitted
    assert sel["nominee"] in (1, 3)
    for u in ("P__F__rep+clean", "P__F__rep+head", "P__FZ__rep", "P__FZ__rep+head", "P__U2__FZ", "P__HEAD__F"):
        assert S.unit_complete(u), u
    cert = json.loads((S.RUN / "certificates" / "P.json").read_text())
    assert cert["nominee"]["primary_all_groups"]["status"] in ("OK", "UNAVAILABLE")
    assert len(led) >= 3


def test_leace_map_ids_match_benchmark_declared_order():
    """Regression (2026-10-03 HMDA run failure): C-map ids must use the declared disallowed-attribute order of the
    benchmark (hmda: race+ethnicity), not sorted order; and must equal stored_model_eval.bench.map_id."""
    import oar.study as S
    from stored_model_eval.bench_effective import BENCH_EFFECTIVE
    for ds in ("adult", "hmda"):
        c = S.CELLS[ds]
        declared = BENCH_EFFECTIVE["datasets"][ds]["purposes"][c["purpose"]]["disallowed_attrs"]
        assert c["policy"] == list(declared)
        assert S.leace_map_id(ds, 0, "C").endswith("C_" + "+".join(declared))
    assert S.leace_map_id("hmda", 1, "C") == "hmda__s1__underwriting__C_race+ethnicity"
    root = S.BENCH / "defenses"
    if root.exists():
        for ds in ("adult", "hmda"):
            for kind in ("B", "C"):
                assert (root / S.leace_map_id(ds, 0, kind) / "map").exists()
