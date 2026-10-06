"""Admission bookkeeping (no real data): unit lists, composition-only maps, finite JSON, id helpers."""
import json

from cbp import admit as AD
from cbp import run as R


def test_admitted_unit_lists():
    u = AD.units_for(0)
    assert len(u) == 28 and len(set(u)) == 28
    assert "pol__s0__U_JOINT_i8o64_l0.1" in u and "pol__s0__U_LOCAL_i8o64_l0.01" in u
    assert not any("outer__" in x for x in u)                       # no per-person assessment predictions
    assert sum(x.startswith("pol__") for x in u) == 3 + 8 + 11
    assert len(AD.admitted_dirs()) == 6


def test_bank_ids_and_composition():
    assert len(R.code_ids()) == 27 and len(R.scored_ids()) == 32
    assert len(R.new_ids()) == 16 and len(R.reused_ids()) == 11
    assert set(R.COMPOSED_EXTRA_IDS).isdisjoint(R.scored_ids())
    assert R.composition_ids()[:27] == R.code_ids() and len(R.composition_ids()) == 38
    for c in R.code_ids() + R.COMPOSED_EXTRA_IDS:
        assert R.unit_for(0, R.parse_id(c) and c).replace("pol__s0__", "") == R.safe(c)
    assert [R.parse_id(c)["lam"] for c in R.privacy_ids()][::4] == list(R.LAMS)


def test_finite_json():
    o = R._finite({"a": float("inf"), "b": [float("nan"), 1.5], "c": {"d": -float("inf")}, "e": "Infinity-text"})
    json.dumps(o, allow_nan=False)
    assert o == {"a": None, "b": [None, 1.5], "c": {"d": None}, "e": "Infinity-text"}
