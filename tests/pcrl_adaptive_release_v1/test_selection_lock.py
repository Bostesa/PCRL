import pytest

from experiments.pcrl_adaptive_release_v1 import selection_lock


def test_global_alias_requires_equal_three_anchor_release_vector():
    maps = {
        0: {"A_control_D17": "D", "A_selected": "A", "B_selected": "A",
            "control": "D"},
        1: {"A_control_D17": "D", "A_selected": "A", "B_selected": "B",
            "control": "D"},
        2: {"A_control_D17": "D", "A_selected": "A", "B_selected": "B",
            "control": "D"},
    }
    aliases = selection_lock.global_exact_aliases(
        maps, {"D17", "A_selected", "B_selected", "control"})
    assert aliases["control"] == "D17"
    assert "B_selected" not in aliases
    maps[1]["B_selected"] = "A"
    maps[2]["B_selected"] = "A"
    aliases = selection_lock.global_exact_aliases(
        maps, {"D17", "A_selected", "B_selected", "control"})
    assert aliases["B_selected"] == "A_selected"


def test_global_alias_fails_when_a_registered_name_is_absent():
    maps = {anchor: {"A_control_D17": "D"} for anchor in (0, 1, 2)}
    with pytest.raises(ValueError, match="absent"):
        selection_lock.global_exact_aliases(maps, {"D17", "A_selected"})
