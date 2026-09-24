import numpy as np
import pytest

from experiments.pcrl_shared_context_release_v1 import policies


def _inputs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return {"x": rng.normal(size=(n, 32)), "ha": rng.normal(size=(n, 4)),
            "token_codes": rng.integers(0, 32, n), "teacher_p": rng.uniform(.05, .95, n),
            "residual": rng.normal(size=n), "risk": rng.dirichlet(np.ones(11), n)}


@pytest.mark.parametrize("key", ["hb", "labels", "ids", "households", "weights", "loss", "role", "SEX"])
def test_forbidden_inputs_rejected_everywhere(key):
    inputs = {**_inputs(10), key: np.zeros(10)}
    with pytest.raises(PermissionError):
        policies.check_legal(inputs)
    with pytest.raises(PermissionError):
        policies.policy_features(inputs)
    d17 = policies.D17Policy("D17", np.arange(32) % 17)
    with pytest.raises(PermissionError):
        d17.predict(inputs)
    with pytest.raises(PermissionError):
        policies.check_legal({"unknown_field": 1})


def test_aliases_and_legal_pick():
    rows = {**_inputs(5), "hb": np.zeros((5, 2)), "labels": {}, "households": np.arange(5)}
    legal = policies.legal_inputs(rows)
    assert set(legal) == set(policies.LEGAL_INPUTS)
    canonical = policies.check_legal({"T0": legal["token_codes"], "p": legal["teacher_p"], "r": legal["residual"]})
    assert set(canonical) == {"token_codes", "teacher_p", "residual"}
    with pytest.raises(ValueError):
        policies.check_legal({"T0": legal["token_codes"], "token_codes": legal["token_codes"]})
    assert policies.policy_features(legal).shape == (5, 49)


def test_switched_argmin_is_d17_anchored_with_ties_to_d17():
    base = np.array([3, 3, 3, 3])
    g = np.zeros((4, 17))
    g[0, 5] = -0.001            # gain 0.001 <= tau -> stay on D17
    g[1, 5] = -0.003            # gain 0.003 > tau -> deviate
    g[2, 3] = -0.01             # D17 is the minimum -> stay
    g[3, [7, 9]] = -0.01        # tie among non-D17 minima -> lowest id
    out = policies.switched_argmin(g, base, policies.TAU)
    assert out.tolist() == [3, 5, 3, 7]
    assert policies.switched_argmin(np.zeros((1, 17)), np.array([11])).tolist() == [11]


def test_cost_regression_policy_learns_signal_and_is_frozen(tmp_path):
    inputs = _inputs(600, seed=1)
    z = policies.policy_features(inputs)
    std = policies.FeatureStandardizer.fit(z)
    target = np.where(inputs["x"][:, 0] > 0, 2, 9)
    g = np.ones((600, 17)); g[np.arange(600), target] = 0.
    hh = np.asarray([f"h{i}" for i in range(600)])
    d17_tokens = np.full(32, 4)
    oracle, record = policies.fit_paired_oracle(std.transform(z), g, d17_tokens[inputs["token_codes"]],
                                                np.ones(600), hh, seed=3)
    assert len(oracle.models) == 17 and record["validation_people"] > 0
    assert all(1 <= r["iterations"] <= 150 for r in record["per_token"])
    pol = policies.switched_policy_from_oracle("task_only", d17_tokens, std, oracle)
    delta = pol.predict_delta(inputs)
    assert np.all(delta[:, 4] == 0.)
    pred = pol.predict(inputs)
    assert np.mean(pred == target) > 0.9
    bank = policies.PolicyBank(("D17", "task_only"), (policies.D17Policy("D17", d17_tokens), pol))
    tokens = bank.predict(inputs)
    assert tokens.shape == (600, 2) and np.all(tokens[:, 0] == 4)
    saved = policies.save_bank(tmp_path / "private" / "policies", bank)
    loaded = policies.load_bank(tmp_path / "private" / "policies", saved["sha256"])
    assert np.array_equal(loaded.predict(inputs), tokens)
    with pytest.raises(ValueError):
        policies.load_bank(tmp_path / "private" / "policies", "0" * 64)
    with pytest.raises(FileExistsError):
        policies.save_bank(tmp_path / "private" / "policies", bank)


def test_price_groups_restriction_fallback_and_doubling():
    cuts = [{"id": f"{r}/{i}", "role": r} for r in ("A/SEX", "A/RAC1P", "AB/SEX", "AB/RAC1P") for i in range(2)]
    duals = {c["id"]: 0. for c in cuts}
    duals["AB/SEX/0"] = 0.4
    groups = policies.price_groups(cuts, duals)
    assert groups["task_only"]["multipliers"] == {}
    assert groups["local_priced"]["fallback_used"] is True
    assert groups["local_priced"]["multipliers"] == {f"{r}/{i}": 0.25 for r in ("A/SEX", "A/RAC1P") for i in range(2)}
    assert groups["coalition_priced"]["fallback_used"] is False
    assert groups["coalition_priced"]["multipliers"] == {"AB/SEX/0": 0.4}
    assert groups["all_priced_x2"]["multipliers"] == {"AB/SEX/0": 0.8}
    with pytest.raises(ValueError):
        policies.price_groups(cuts, {"x": 1.})


def test_priced_costs_sign_and_label_masks():
    mask = np.array([True, True, False])
    u = np.ones((2, 17))
    attack = {"id": "a", "valid_mask": np.array([True, False, True]), "losses": np.full((2, 17), 2.)}
    g = policies.priced_person_costs(u, [attack], {"a": .5}, mask)
    assert np.allclose(g[0], 0.) and np.allclose(g[1], 1.)
    with pytest.raises(ValueError):
        policies.priced_person_costs(u, [], {"a": .5}, mask)


def test_alias_ledger_and_disagreement():
    tokens = np.array([[1, 1, 2, 1], [3, 3, 3, 3], [5, 5, 6, 5]])
    ledger = policies.alias_ledger(tokens, ["D17", "task_only", "local_priced", "coalition_priced"])
    assert ledger["retained"] == ["D17", "local_priced"]
    assert {r["removed"] for r in ledger["removed"]} == {"task_only", "coalition_priced"}
    diag = policies.disagreement_diagnostics(tokens, ["D17", "a", "b", "c"], np.array([0, 0, 1]),
                                             np.ones(3), np.array(["h1", "h1", "h2"]))
    assert diag["b"]["fraction_differs_from_D17_unweighted"] == pytest.approx(2 / 3)
    assert diag["b"]["affected_unique_households"] == 2
    assert diag["D17"]["fraction_differs_from_D17_weighted"] == 0.


def test_paired_targets_are_zero_at_d17():
    g = np.arange(34, dtype=float).reshape(2, 17)
    d = policies.paired_targets(g, np.array([3, 16]))
    assert d[0, 3] == 0. and d[1, 16] == 0. and d[0, 5] == 2. and d[1, 0] == -16.


def test_null_token_costs_keep_policy_mostly_d17():
    """M2: large person-level noise shared by all tokens must not drive the argmin.

    The raw per-token oracle (pre-M2) deviates for most people; the paired oracle
    stays mostly on D17 when the token-specific part is tiny relative to tau.
    """
    rng = np.random.default_rng(0)
    n = 4000
    x = _inputs(n, seed=4)
    std = policies.FeatureStandardizer.fit(policies.policy_features(x))
    z = std.transform(policies.policy_features(x))
    d17_tokens = np.arange(32) % 17
    base = d17_tokens[x["token_codes"]]
    hh = np.asarray([f"h{i}" for i in range(n)])
    g = (rng.normal(size=(n, 1)) + 0.3 * np.abs(x["x"][:, :1])) * np.ones((1, 17)) \
        + 0.01 * rng.normal(size=(n, 17))
    raw, _ = policies.fit_cost_regressors(z, g, np.ones(n), hh, seed=1)
    raw_tokens = policies.switched_argmin(np.column_stack([m.predict(z) for m in raw]), base)
    oracle, _ = policies.fit_paired_oracle(z, g, base, np.ones(n), hh, seed=1)
    paired_tokens = policies.switched_policy_from_oracle("null", d17_tokens, std, oracle).predict(x)
    assert np.mean(raw_tokens != base) > 0.5
    assert np.mean(paired_tokens != base) < 0.2
