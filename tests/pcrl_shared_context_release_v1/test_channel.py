import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import privacy_first as ar_privacy_first
from experiments.pcrl_adaptive_release_v1.fit_b import fixed_bank_dual_prices
from experiments.pcrl_adaptive_release_v1.reference import calibrate_reference
from experiments.pcrl_task_aligned_cuts_v1.solver import solve_p1
from experiments.pcrl_shared_context_release_v1 import channel

FIXTURE = Path(__file__).parent / "fixtures" / "math_review_exact_laws.json"
ROLES = ("A/SEX", "A/RAC1P", "AB/SEX", "AB/RAC1P")


def _d17(rng):
    d17 = np.zeros((32, 17))
    d17[np.arange(32), rng.integers(0, 17, 32)] = 1.
    return d17


def synthetic_instance(seed=0, n=600, K=2, M=3, n_attacks=6):
    """Per-person losses with a genuine task/privacy trade-off (no ACS data)."""
    rng = np.random.default_rng(seed)
    d17 = _d17(rng)
    codes = rng.integers(0, 32, n)
    ctx = rng.integers(0, K, n)
    tokens = np.column_stack([np.argmax(d17[codes], 1)] +
                             [rng.integers(0, 17, n) for _ in range(M - 1)])
    pwgtp = rng.uniform(5, 50, n)
    signal = rng.normal(size=(n, 17))
    task = 0.5 + 0.2 * np.abs(signal) + 0.05 * rng.random((n, 17))
    task[np.arange(n), tokens[:, 0]] += 0.05       # D17 not task-optimal
    cost = channel.coefficient_pair_blocks(task, codes, ctx, tokens, pwgtp, n_contexts=K)
    bank = []
    for a in range(n_attacks):
        # attacks are strong where the task is good: a trade-off
        loss = 0.6 + 0.3 * np.abs(signal + 0.5 * rng.normal(size=(n, 17))) \
            - 0.1 * (task < np.median(task)) + 0.05 * rng.random((n, 17))
        loss = np.maximum(loss, 0.01)
        pair = channel.coefficient_pair_blocks(loss, codes, ctx, tokens, pwgtp, n_contexts=K)
        role = ROLES[a % 4]
        for v in ("U", "W"):
            bank.append({"id": f"atk{a}/{v}", "role": role, "weighting": v,
                         "coeff_B": pair[v]["B"], "coeff_A": pair[v]["A"],
                         "coefficient_pool_sha256": f"pool-{role}", "class_order": [0, 1],
                         "weight_normalization": "1/n" if v == "U" else "PWGTP/sum(PWGTP)"})
    return {"d17": d17, "codes": codes, "ctx": ctx, "tokens": tokens, "pwgtp": pwgtp,
            "task": task, "cost": cost, "bank": bank, "K": K, "M": M}


def _random_params(rng, K, M, eta):
    B = rng.random((32, 17)); B *= (1 - eta) / B.sum(1, keepdims=True)
    A = rng.random((K, M)); A *= eta / A.sum(1, keepdims=True)
    return B, A, eta


def test_person_law_is_row_stochastic_and_validated():
    inst = synthetic_instance()
    rng = np.random.default_rng(3)
    for eta in (0., .3, 1.):
        B, A, eta = _random_params(rng, inst["K"], inst["M"], eta)
        q = channel.person_law(B, A, eta, inst["codes"], inst["ctx"], inst["tokens"])
        assert q.shape == (len(inst["codes"]), 17)
        assert np.all(q >= 0) and np.max(np.abs(q.sum(1) - 1)) <= 1e-12
    bad_B = B.copy(); bad_B[0, 0] += 1e-6
    with pytest.raises(ValueError):
        channel.person_law(bad_B, A, eta, inst["codes"], inst["ctx"], inst["tokens"])
    with pytest.raises(ValueError):
        channel.validate_law(np.full((2, 17), 1 / 17 + 1e-8))
    with pytest.raises(ValueError):
        channel.validate_law(np.full((2, 17), np.nan))


def test_exact_d17_embedding_both_witnesses():
    inst = synthetic_instance()
    B, A, eta = channel.d17_params(inst["d17"], inst["K"], inst["M"])
    q = channel.person_law(B, A, eta, inst["codes"], inst["ctx"], inst["tokens"])
    assert np.array_equal(q, inst["d17"][inst["codes"]])
    A1 = np.zeros_like(A); A1[:, 0] = 1.
    q1 = channel.person_law(np.zeros_like(B), A1, 1., inst["codes"], inst["ctx"], inst["tokens"])
    assert np.array_equal(q1, inst["d17"][inst["codes"]])
    assert channel.deterministic_emission(q)["all_rows_one_hot"]


def test_block_replay_equals_direct_person_sum_to_1e12():
    inst = synthetic_instance(seed=5)
    rng = np.random.default_rng(9)
    for eta in (0., .4, 1.):
        B, A, eta = _random_params(rng, inst["K"], inst["M"], eta)
        q = channel.person_law(B, A, eta, inst["codes"], inst["ctx"], inst["tokens"])
        for v in ("U", "W"):
            w = channel.normalized_weights(inst["pwgtp"], v)
            direct = channel.direct_expected_loss(q, inst["task"], w)
            assert abs(channel.block_value(inst["cost"][v], B, A) - direct) <= 1e-12


def test_eta_zero_reproduces_inherited_t32_lp():
    inst = synthetic_instance(seed=11)
    calibrated = channel.calibrate_nested(inst["d17"], inst["bank"], .001)
    ours = channel.solve_utility(inst["cost"], calibrated["cuts"], inst["d17"], fix_eta_zero=True)
    assert ours["feasible"] and ours["params"]["eta"] == 0. and not ours["params"]["A"].any()
    # inherited AR/TAC path on the B blocks only
    b_bank = [{**{k: v for k, v in c.items() if k not in ("coeff_B", "coeff_A")}, "coeff": c["coeff_B"]}
              for c in inst["bank"]]
    ar_cal = calibrate_reference(inst["d17"], b_bank, .001)
    cost = 0.5 * (inst["cost"]["U"]["B"] + inst["cost"]["W"]["B"])
    ar = solve_p1(cost, ar_cal["cuts"])
    assert ar["feasible"]
    assert np.array_equal(ours["params"]["B"], ar["Q"])
    assert abs(ours["replay"]["objective"] - ar["objective"]) <= 1e-12
    duals = fixed_bank_dual_prices(cost, ar_cal["cuts"])
    assert abs(duals["objective"] - ours["replay"]["objective"]) <= 1e-8
    # rho values identical to the inherited calibration
    assert {g: r["value"] for g, r in calibrated["rho"].items()} == \
        {g: r["value"] for g, r in ar_cal["rho"].items()}


def test_nested_lp_is_no_worse_than_t32_and_witness_feasible():
    inst = synthetic_instance(seed=13)
    calibrated = channel.calibrate_nested(inst["d17"], inst["bank"], .001)
    assert calibrated["witness_eta0"]["maximum_cut_violation"] == 0.
    assert calibrated["witness_eta1_d17_column_max_abs_difference"] <= 1e-12
    nested = channel.solve_utility(inst["cost"], calibrated["cuts"], inst["d17"])
    t32 = channel.solve_utility(inst["cost"], calibrated["cuts"], inst["d17"], fix_eta_zero=True)
    assert nested["feasible"] and t32["feasible"]
    assert nested["replay"]["objective"] <= t32["replay"]["objective"] + 1e-9
    assert nested["replay"]["maximum_cut_violation"] <= 1e-7
    assert nested["dual_lower_bound"] <= nested["replay"]["objective"] + 1e-9
    assert nested["fixed_bank_gap"] >= -1e-7
    d17_cost = 0.5 * sum(np.sum(inst["cost"][v]["B"] * inst["d17"]) for v in ("U", "W"))
    assert nested["replay"]["objective"] <= d17_cost + 1e-12
    p = nested["params"]
    channel.validate_params(p["B"], p["A"], p["eta"])
    q = channel.person_law(p["B"], p["A"], p["eta"], inst["codes"], inst["ctx"], inst["tokens"])
    for cut in calibrated["cuts"][:4]:
        assert channel.block_value({"B": cut["coeff_B"], "A": cut["coeff_A"]}, p["B"], p["A"]) \
            >= cut["floor"] - 1e-7
    report = channel.deterministic_emission(q)
    assert set(report) >= {"all_rows_one_hot", "one_hot_fraction", "stochastic_rows"}
    sparsity = nested["sparsity"]
    assert sparsity["excess_B"] >= 0 and sparsity["excess_A"] >= 0


def test_privacy_first_eta_zero_matches_ar_privacy_first():
    inst = synthetic_instance(seed=17)
    calibrated = channel.calibrate_nested(inst["d17"], inst["bank"], .001)
    ours = channel.solve_privacy_first(inst["cost"], calibrated["cuts"], inst["d17"], fix_eta_zero=True)
    ar_cuts = [{**{k: v for k, v in c.items() if k not in ("coeff_B", "coeff_A")}, "coeff": c["coeff_B"]}
               for c in calibrated["cuts"]]
    ar = ar_privacy_first.solve_privacy_first({v: inst["cost"][v]["B"] for v in ("U", "W")},
                                              ar_cuts, inst["d17"])
    assert ours["feasible"] and ar["status"] == "OPTIMAL"
    assert abs(ours["tau"] - ar["tau"]) <= 1e-7
    nested = channel.solve_privacy_first(inst["cost"], calibrated["cuts"], inst["d17"])
    assert nested["feasible"] and nested["tau"] >= ours["tau"] - 1e-9
    for v in ("U", "W"):
        assert nested["replay"][f"objective_{v}"] <= nested["task_caps"][v] + 1e-7
    assert nested["dual_tau_upper_bound"] >= nested["tau"] - 1e-7


def test_math_review_shared_eta_counterexample():
    fx = json.loads(FIXTURE.read_text())["shared_eta_counterexample"]
    cost_x = np.array([[float(Fraction(c)) for c in row] for row in fx["frozen_cost_x_by_token"]])
    t = np.array(fx["T_state"]); w = np.array([float(Fraction(x)) for x in fx["weights"]])
    losses = np.full((4, 17), 5.); losses[:, :3] = cost_x
    d17 = np.zeros((32, 17)); d17[:, 0] = 1.

    def value(ctx, columns):
        K = int(ctx.max()) + 1
        tokens = np.column_stack(columns)
        blocks = channel.coefficient_blocks(losses, t, ctx, tokens, w, n_contexts=K)
        cost = {"U": blocks, "W": blocks}
        out = channel.solve_utility(cost, [], d17)
        return out["replay"]["objective"]

    d17_col, d_new = np.array(fx["D17"]), np.array(fx["d_new"])
    switched = np.array(fx["switched_D17_anchored"])
    expect = {k: float(Fraction(v)) for k, v in fx["optimal_value"].items()}
    assert value(np.zeros(4, int), [d17_col, d_new]) == pytest.approx(expect["nested_K1_bank_D17_dnew"], abs=1e-9)
    assert value(t.copy(), [d17_col, d_new]) == pytest.approx(expect["nested_K2_aligned"], abs=1e-9)
    assert value(np.array([0, 1, 0, 1]), [d17_col, d_new]) == pytest.approx(expect["nested_K2_crossing"], abs=1e-9)
    assert value(np.zeros(4, int), [d17_col, switched]) == pytest.approx(expect["nested_K1_with_switched_column"], abs=1e-9)


def test_nonalias_diagnostic_and_projection():
    inst = synthetic_instance(seed=19)
    hh = np.asarray([f"h{i // 2}" for i in range(len(inst["codes"]))])
    q0 = inst["d17"][inst["codes"]]
    diag, proj = channel.nonalias_diagnostic(q0, inst["codes"], inst["d17"], inst["pwgtp"], hh)
    assert diag["tv_to_d17"]["max"] == 0. and diag["within_t32"]["V_weighted"] == 0.
    assert diag["within_t32"]["affected_unique_households"] == 0
    A = np.zeros((inst["K"], inst["M"])); A[:, 1] = .5
    B = inst["d17"] * .5
    q = channel.person_law(B, A, .5, inst["codes"], inst["ctx"], inst["tokens"])
    diag, proj = channel.nonalias_diagnostic(q, inst["codes"], inst["d17"], inst["pwgtp"], hh)
    assert diag["within_t32"]["V_unweighted"] > 0 and diag["within_t32"]["max_pairwise_tv"] > 0
    assert diag["tv_to_d17"]["affected_unique_households"] > 0
    assert not diag["deterministic_emission"]["all_rows_one_hot"]
    calibrated = channel.calibrate_nested(inst["d17"], inst["bank"], .001)
    rescore = channel.projection_rescore(proj, inst["cost"], calibrated["cuts"])
    # re-scoring the projection with B blocks equals the direct person sum under the projected law
    w = channel.normalized_weights(inst["pwgtp"], "U")
    direct = channel.direct_expected_loss(proj["U"][inst["codes"]], inst["task"], w)
    assert rescore["task_U"] == pytest.approx(direct, abs=1e-12)
