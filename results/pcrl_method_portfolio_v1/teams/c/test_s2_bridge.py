"""C release to the actual S2 shared AttackView, using synthetic people."""

import os
import sys
from pathlib import Path

import numpy as np

shared_dir = Path(os.environ.get(
    "PCRL_SHARED_EVAL_DIR", str(Path(__file__).resolve().parents[2] / "shared")
))
if not (shared_dir / "attack_eval.py").is_file():
    raise FileNotFoundError("set PCRL_SHARED_EVAL_DIR to the pinned S2 shared evaluator")
sys.path.insert(0, str(shared_dir))
from attack_eval import AttackView as S2AttackView

from s2_bridge import to_s2_attack_views
from prototype import AdultGroupedSchema, IdentityCalibrator, PurposeRelease
from test_evaluation import BinaryHead, SixHead, alignment, sample_views


def public_releases():
    schema = AdultGroupedSchema()
    return {
        "income": PurposeRelease(
            schema, ("age",), BinaryHead(), "income", "head1", "cal1",
            IdentityCalibrator(),
        ),
        "occupation_group": PurposeRelease(
            schema, ("capital-gain",), SixHead(), "occupation_group",
            "head2", "cal2", IdentityCalibrator(),
        ),
    }


def test_actual_s2_views_bind_every_transmitted_field_and_private_alignment():
    release = sample_views()
    views = to_s2_attack_views(release, alignment(), public_releases(), S2AttackView)
    assert set(views) == {"income", "occupation_group", "coalition"}
    for name, view in views.items():
        assert isinstance(view, S2AttackView)
        view.validate()
        assert set(view.wire_components) == set(release[name])
        assert np.array_equal(view.row_id, alignment()[name]["row_id"])
        assert np.array_equal(view.unit, alignment()[name]["unit"])
        assert "selected_values" in view.components
        assert view.x_defense_aware.shape[0] == 4
        assert view.x_defense_aware.shape[1] > view.x.shape[1]
        assert view.x_metadata is None
        assert view.cell_key is None
        assert np.array_equal(view.x_raw_only, release[name]["selected_values"])
    coalition = views["coalition"]
    assert "income_probabilities" in coalition.components
    assert "occupation_group_probabilities" in coalition.components
    assert "income_groups" in {k for k, _ in coalition.wire_constants}


def test_s2_bridge_refuses_unregistered_person_field():
    release = sample_views()
    release["coalition"]["fallback_value"] = np.array([0, 1, 0, 1])
    try:
        to_s2_attack_views(release, alignment(), public_releases(), S2AttackView)
    except ValueError as exc:
        assert "wire" in str(exc).lower() or "unexpected" in str(exc).lower()
    else:
        raise AssertionError("unregistered per-person field reached S2 view")
