"""Synthetic disclosure checks; no Adult files or fitted attacker are used."""

import numpy as np
import pytest

from attack_views import aligned_complete_views, flatten_view


def view(purpose, flag="public"):
    k = 2 if purpose == "income" else 6
    probs = np.full(k, 0.1 / (k - 1), dtype=np.float32)
    probs[0] = 0.9
    return {
        "features": np.arange(16, dtype=np.float32),
        "logits": np.arange(k, dtype=np.float32),
        "probabilities": probs,
        "decision": 0,
        "metadata": {"purpose": purpose, "fallback": False, "flag": flag, "code_length": 16},
    }


def test_pair_contains_every_recipient_field_and_extra_metadata_without_key():
    rows = [("private-person-1", {"income": view("income"), "occupation": view("occupation")})]
    output = aligned_complete_views(rows)
    assert set(output) == {"income", "occupation", "pair"}
    assert output["pair"][0]["income.features[0]"] == 0.0
    assert output["pair"][0]["occupation.logits[5]"] == 5.0
    assert output["pair"][0]["income.metadata.fallback"] == 0.0
    assert output["pair"][0]["occupation.metadata.flag"] == "public"
    assert "private-person-1" not in repr(output)


def test_new_fallback_field_is_not_silently_omitted():
    release = view("income")
    release["new_fallback_score"] = 0.25
    assert flatten_view("income", release)["new_fallback_score"] == 0.25


def test_missing_view_and_invalid_probability_fail_closed():
    with pytest.raises(ValueError, match="Both recipient"):
        aligned_complete_views([("person", {"income": view("income")})])
    release = view("income")
    release["probabilities"][0] = 0.8
    with pytest.raises(ValueError, match="probability"):
        flatten_view("income", release)
