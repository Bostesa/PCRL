import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1.roles import (
    ROLE_NAMES, role_of, pooled_role, summarize_roles,
)


def _house(role):
    return next(f'synthetic-house-{n}' for n in range(10000) if role_of(f'synthetic-house-{n}') == role)


def _prepared():
    houses = [_house(role) for role in ROLE_NAMES]
    n = len(houses)
    pool = {
        'x': np.arange(n * 32, dtype=np.float32).reshape(n, 32),
        'ha': np.arange(n * 4, dtype=np.float64).reshape(n, 4),
        'hb': np.arange(n * 2, dtype=np.float64).reshape(n, 2),
        'weights': np.arange(1, n + 1, dtype=np.float64),
        'ids': np.array([f'person-{i}' for i in range(n)], dtype=object),
        'households': np.array(houses, dtype=object),
        'labels': {
            'same_residence': np.arange(n) % 2,
            'SEX': np.arange(n) % 2,
            'RAC1P': np.arange(n) % 9,
        },
    }
    return {'ctx': {'pools': {'downstream_fit': pool}},
            'encoded': {'downstream_fit': {'codes': {'T0': np.arange(n) % 32},
                                           'p': np.linspace(0, 1, n),
                                           'r': np.linspace(-1, 1, n),
                                           'risk': np.zeros((n, 11))}}}


def test_household_role_is_stable_exclusive_and_pool_filter_preserves_original_people():
    prepared = _prepared()
    seen = set()
    for role in ROLE_NAMES[:-1]:
        rows = pooled_role(prepared, role)
        assert len(rows['ids']) == 1
        assert len(rows['labels']['SEX']) == 1
        assert rows['token_codes'].shape == (1,)
        assert rows['households'][0] not in seen
        seen.add(rows['households'][0])
    assert len(seen) == len(ROLE_NAMES) - 1
    assert role_of(_house('coefficient_split')) == 'coefficient_split'


def test_prelock_outer_role_refuses_labels_even_if_loaded_object_contains_them():
    with pytest.raises(PermissionError, match='selection lock'):
        pooled_role(_prepared(), 'outer_assessment')


def test_role_summary_counts_households_once_across_anchors_without_outer_labels():
    prepared = _prepared()
    result = summarize_roles({0: prepared, 1: prepared, 2: prepared})
    assert all(result['global_households'][r] == 1 for r in ROLE_NAMES)
    assert result['global_role_overlap'] == 0
    assert result['anchors']['0']['outer_assessment']['people'] == 1
    assert 'class_support' not in result['anchors']['0']['outer_assessment']
