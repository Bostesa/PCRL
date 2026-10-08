"""Data-free cache identity checks for shared masked heads and full F."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from bank import _sha
from head_pool import _file_sha, _inherited_modules, make_head_cache_identity


def _mask(atoms=("age", "workclass"), columns=(0, 5, 6)):
    identity = {
        "source_schema_sha256": "a" * 64,
        "atoms": list(atoms), "column_indices": list(columns),
    }
    return {**identity, "mask_id": "synthetic-mask",
            "mask_sha256": _sha(identity)}


def _identity(mask=None, **changes):
    arguments = {
        "mask": mask or _mask(), "purpose": "income", "seed": 0,
        "campaign_lock_sha256": "b" * 64,
        "data_manifest_sha256": "c" * 64,
        "role_file_sha256": {
            "fit_features": "1" * 64, "fit_label": "2" * 64,
            "calibration_features": "3" * 64,
            "calibration_label": "4" * 64,
        },
        "fit_row_unit_sha256": "d" * 64,
        "calibration_row_unit_sha256": "e" * 64,
        "head_spec": {"hidden_width": 64, "max_epochs": 30},
        "source_code_sha256": {
            "head_pool.py": "0" * 64,
            "training.py": "5" * 64,
            "prototype.py": "6" * 64,
            "scorecard_builder.py": "7" * 64,
            "release_package.py": "8" * 64,
            "common_eval.py": "9" * 64,
            "safe_data_guard.py": "a" * 64,
        },
    }
    arguments.update(changes)
    return make_head_cache_identity(**arguments)


def test_cache_identity_is_order_invariant_and_binds_every_scientific_input():
    base = _identity()
    assert base.cache_key == _identity(role_file_sha256={
        "calibration_label": "4" * 64, "fit_label": "2" * 64,
        "calibration_features": "3" * 64, "fit_features": "1" * 64,
    }).cache_key
    assert base.model_id.startswith("FOCUSED:income:")
    variants = (
        {"seed": 1},
        {"purpose": "occupation_group"},
        {"fit_row_unit_sha256": "f" * 64},
        {"head_spec": {"hidden_width": 32, "max_epochs": 30}},
        {"source_code_sha256": {
            "head_pool.py": "0" * 64,
            "training.py": "f" * 64, "prototype.py": "6" * 64,
            "scorecard_builder.py": "7" * 64,
            "release_package.py": "8" * 64,
            "common_eval.py": "9" * 64,
            "safe_data_guard.py": "a" * 64,
        }},
    )
    assert all(_identity(**variant).cache_key != base.cache_key
               for variant in variants)


def test_equal_encoded_width_is_not_a_cache_alias():
    first = _identity(mask=_mask(("age", "workclass"), (0, 5, 6)))
    second = _identity(mask=_mask(("age", "relationship"), (0, 50, 51)))
    assert len(first.selected_columns) == len(second.selected_columns) == 3
    assert first.cache_key != second.cache_key


def test_full_f_mask_can_use_same_head_path_without_primary_capacity():
    full = _mask(
        ("age", "education_fact", "capital-gain", "capital-loss",
         "hours-per-week", "workclass", "marital-status", "relationship",
         "native-country"),
        tuple(range(83)),
    )
    identity = _identity(mask=full)
    assert identity.selected_columns == tuple(range(83))
    assert identity.model_id.startswith("FOCUSED:income:")


def test_malformed_mask_hash_or_role_hash_fails_closed():
    bad = copy.deepcopy(_mask())
    bad["mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mask identity"):
        _identity(mask=bad)
    with pytest.raises(ValueError, match="role file"):
        _identity(role_file_sha256={"fit_features": "1" * 64})


def test_inherited_fit_sources_resolve_to_current_worktree_bytes():
    project = Path(__file__).resolve().parents[3]
    c_dir = project / "results/pcrl_method_portfolio_v1/teams/c"
    shared = project / "results/pcrl_method_portfolio_v1/shared"
    sources = {
        "head_pool.py": _file_sha(Path(__file__).with_name("head_pool.py")),
        **{name: _file_sha(c_dir / name) for name in (
            "training.py", "prototype.py", "scorecard_builder.py",
            "release_package.py",
        )},
        **{name: _file_sha(shared / name) for name in (
            "common_eval.py", "safe_data_guard.py",
        )},
    }
    identity = _identity(source_code_sha256=sources)
    training, package, common = _inherited_modules(identity)
    assert Path(training.__file__).resolve() == c_dir / "training.py"
    assert Path(package.__file__).resolve() == c_dir / "release_package.py"
    assert Path(common.__file__).resolve() == shared / "common_eval.py"
