"""The locked assessment must match the preregistered five-pool census."""
import numpy as np
import pickle
import pytest

from experiments.pcrl_adaptive_release_v1 import outer_pool, roles


def test_object_identifiers_survive_independent_deserialization():
    original = np.array(["household-0000000000000001",
                         "household-0000000000000002"], dtype=object)
    restored = pickle.loads(pickle.dumps(original))
    assert original[0] is not restored[0]
    assert outer_pool._same_bytes(original, restored)
    restored[1] = "different-household"
    assert not outer_pool._same_bytes(original, restored)


def _pool(person, household):
    return {"x": np.zeros((1, 32)), "ha": np.zeros((1, 4)),
            "hb": np.zeros((1, 2)), "weights": np.ones(1),
            "ids": np.array([person]), "households": np.array([household]),
            "labels": {"same_residence": np.array([1]),
                       "SEX": np.array([0]), "RAC1P": np.array([2])}}


def _encoded():
    return {"codes": {"T0": np.array([0])}, "p": np.array([.5]),
            "r": np.array([0.]), "risk": np.zeros((1, 11))}


def _outer_household(prefix):
    return next(f"{prefix}-{i}" for i in range(10000)
                if roles.role_of(f"{prefix}-{i}") == "outer_assessment")


def test_locked_outer_includes_fifth_pool_and_matches_census():
    prepared = {"ctx": {"pools": {
        "downstream_fit": _pool("first", _outer_household("first")),
        "attacker_validation": _pool("fifth", _outer_household("fifth"))}},
        "encoded": {"downstream_fit": _encoded(),
                    "attacker_validation": _encoded()}}
    census = {"people": 2, "households": 2, "weight_sum": 2.}
    result = outer_pool.pooled_locked_outer(prepared, census)
    assert result["ids"].tolist() == ["first", "fifth"]
    assert result["labels"]["RAC1P"].tolist() == [2, 2]
    with pytest.raises(ValueError, match="census"):
        outer_pool.pooled_locked_outer(prepared, {**census, "people": 1})


def test_locked_outer_requires_original_fifth_pool_labels():
    prepared = {"ctx": {"pools": {
        "downstream_fit": _pool("first", _outer_household("first")),
        "attacker_validation": _pool("fifth", _outer_household("fifth"))}},
        "encoded": {"downstream_fit": _encoded(),
                    "attacker_validation": _encoded()}}
    del prepared["ctx"]["pools"]["attacker_validation"]["labels"]
    with pytest.raises(ValueError, match="original label-bearing"):
        outer_pool.pooled_locked_outer(prepared, {"people": 2, "households": 2,
                                                   "weight_sum": 2.})


def test_locked_outer_accepts_archived_extra_task_labels_without_scoring_them():
    prepared = {"ctx": {"pools": {
        "downstream_fit": _pool("first", _outer_household("first")),
        "attacker_validation": _pool("fifth", _outer_household("fifth"))}},
        "encoded": {"downstream_fit": _encoded(),
                    "attacker_validation": _encoded()}}
    prepared["ctx"]["pools"]["attacker_validation"]["labels"]["historical_other_task"] = np.array([1])
    result = outer_pool.pooled_locked_outer(
        prepared, {"people": 2, "households": 2, "weight_sum": 2.})
    assert set(result["labels"]) == {"same_residence", "SEX", "RAC1P"}
    assert result["labels"]["same_residence"].tolist() == [1, 1]
