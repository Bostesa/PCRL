import numpy as np
import pytest

from experiments.pcrl_shared_context_release_v1 import contexts


class FakeNuisance:
    """Deterministic SEX risk from x[:,0] (picklable, no labels)."""

    def probabilities(self, inputs):
        p = 1 / (1 + np.exp(-np.asarray(inputs.x_a)[:, 0]))
        return {"SEX": np.column_stack((p, 1 - p)), "RAC1P": np.full((len(p), 9), 1 / 9)}


def _rows(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return {"x": rng.normal(size=(n, 32)), "ha": rng.normal(size=(n, 4)),
            "residual": rng.normal(size=n), "weights": rng.uniform(1, 30, n),
            "households": np.asarray([f"h{i // 2}" for i in range(n)])}


def test_weighted_median():
    assert contexts.weighted_median([3., 1., 2.], [1., 1., 1.]) == 2.
    assert contexts.weighted_median([1., 2., 3.], [10., 1., 1.]) == 1.
    with pytest.raises(ValueError):
        contexts.weighted_median([1.], [0.])


def test_rules_are_hard_and_fitted_on_given_rows(tmp_path):
    rows = _rows()
    rules = contexts.fit_context_rules(rows, FakeNuisance(), "f" * 64)
    legal = {k: rows[k] for k in ("x", "ha", "residual")}
    k1, k2, k4 = (rules[k].assign(legal) for k in (1, 2, 4))
    assert set(k1) == {0} and set(k2) <= {0, 1} and set(k4) <= {0, 1, 2, 3}
    assert np.array_equal(k4 % 2, k2)
    assert len(set(k4)) == 4
    s = contexts.sex_risk_max(FakeNuisance(), legal)
    assert np.array_equal(k2, (s > rules[2].sex_risk_median).astype(int))
    with pytest.raises(PermissionError):
        rules[4].assign({**legal, "hb": np.zeros((400, 2))})
    saved = contexts.save_rules(tmp_path / "private" / "ctx", rules)
    loaded = contexts.load_rules(tmp_path / "private" / "ctx", saved["sha256"])
    assert np.array_equal(loaded[4].assign(legal), k4)
    with pytest.raises(ValueError):
        contexts.load_rules(tmp_path / "private" / "ctx", "0" * 64)


def test_support_census_and_registered_fallback():
    rows = _rows(2000)
    k = np.arange(2000) % 4
    census = contexts.support_census(k, rows["households"], rows["weights"], 4)
    assert [c["people"] for c in census["per_context"]] == [500] * 4
    assert all(c["unique_households"] == 500 for c in census["per_context"])
    assert census["meets_registered_floor"] is True
    thin = contexts.support_census(np.minimum(k, 2), rows["households"], rows["weights"], 4)
    assert thin["per_context"][3]["unique_households"] == 0 and not thin["meets_registered_floor"]
    assert contexts.fallback_decision({"0": census, "1": census, "2": census})["nm4_K"] == 4
    decision = contexts.fallback_decision({"0": census, "1": thin, "2": census})
    assert decision["nm4_K"] == 2 and decision["failing_anchors"] == ["1"]
    with pytest.raises(ValueError):
        contexts.fallback_decision({"0": census})
