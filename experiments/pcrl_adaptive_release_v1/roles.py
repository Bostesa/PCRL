"""Globally disjoint 2018 household roles with a pre-lock outer label barrier."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import numpy as np

ROLE_NAMES = (
    'nuisance_train', 'audit_fit', 'coefficient_split',
    'inner_selection', 'inner_check', 'outer_assessment',
)
ROLE_UPPER = (.20, .40, .65, .75, .85, 1.0)
SALT = 'pcrl_adaptive_release_v1|'
CLASS_COUNT = {'same_residence': 2, 'SEX': 2, 'RAC1P': 9}


def role_of(household: object) -> str:
    """A person's global role depends only on its household identifier."""
    value = str(household)
    if not value:
        raise ValueError('empty household identifier')
    digest = hashlib.sha256((SALT + value).encode('utf-8')).digest()
    unit = int.from_bytes(digest[:8], 'big') / 2**64
    return ROLE_NAMES[next(j for j, bound in enumerate(ROLE_UPPER) if unit < bound)]


def _role_mask(households: np.ndarray, role: str) -> np.ndarray:
    return np.asarray([role_of(h) == role for h in households], dtype=bool)


def pooled_role(prepared: dict[str, Any], role: str) -> dict[str, Any]:
    """Return only prelock inner households; outer labels remain sealed."""
    return _pooled_role(prepared, role, allow_outer=False)


def _pooled_role(prepared: dict[str, Any], role: str, *,
                 allow_outer: bool) -> dict[str, Any]:
    """Return only allocated, label-bearing rows; refuse outer pre-lock.

    The archived attacker-validation labels are absent in the sanitized loader
    and are never consulted here.  The original person, not an expanded token,
    remains the row and eventual household bootstrap unit.
    """
    if role not in ROLE_NAMES:
        raise ValueError('unknown household role')
    if role == 'outer_assessment' and not allow_outer:
        raise PermissionError('outer labels require the selection lock')
    columns = ('x', 'ha', 'hb', 'weights', 'ids', 'households')
    gathered: dict[str, list[np.ndarray]] = {key: [] for key in columns}
    gathered.update({'token_codes': [], 'teacher_p': [], 'residual': [], 'risk': []})
    labels: dict[str, list[np.ndarray]] = {key: [] for key in CLASS_COUNT}
    pools = prepared['ctx']['pools']
    for name in ('representation_fit', 'downstream_fit',
                 'downstream_validation', 'attacker_fit'):
        if name not in pools:
            continue
        pool = pools[name]
        mask = _role_mask(np.asarray(pool['households']), role)
        if not np.any(mask):
            continue
        encoded = prepared['encoded'][name]
        for key in columns:
            gathered[key].append(np.asarray(pool[key])[mask])
        for key, values in (
            ('token_codes', encoded['codes']['T0']),
            ('teacher_p', encoded['p']),
            ('residual', encoded['r']),
            ('risk', encoded['risk']),
        ):
            gathered[key].append(np.asarray(values)[mask])
        for key in CLASS_COUNT:
            labels[key].append(np.asarray(pool['labels'][key])[mask])
    if not gathered['ids']:
        raise ValueError(f'no admitted rows for role {role}')
    result = {key: np.concatenate(value) for key, value in gathered.items()}
    result['labels'] = {key: np.concatenate(value) for key, value in labels.items()}
    if len({str(x) for x in result['ids']}) != len(result['ids']):
        raise ValueError('person appears twice in one pooled role')
    if not np.all(np.isfinite(result['weights'])) or np.any(result['weights'] < 0):
        raise ValueError('invalid original person weights')
    if result['ha'].shape[1] != 4 or result['hb'].shape[1] != 2:
        raise ValueError('historical service shape changed')
    if result['x'].shape[1] != 32 or result['risk'].shape[1] != 11:
        raise ValueError('historical allowed-input feature shape changed')
    if np.any(result['token_codes'] < 0) or np.any(result['token_codes'] >= 32):
        raise ValueError('historical T0 code out of range')
    return result


def summarize_roles(prepared_by_anchor: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate-only split census; outer labels are never read."""
    if set(prepared_by_anchor) != {0, 1, 2}:
        raise ValueError('all three anchors required for global-role census')
    global_houses: dict[str, set[str]] = {role: set() for role in ROLE_NAMES}
    anchors: dict[str, Any] = {}
    for anchor, prepared in sorted(prepared_by_anchor.items()):
        by_role = {}
        for role in ROLE_NAMES:
            houses: set[str] = set()
            people = 0
            weight_sum = 0.0
            support = {target: np.zeros(count, dtype=np.int64)
                       for target, count in CLASS_COUNT.items()}
            for name, pool in prepared['ctx']['pools'].items():
                if name == 'attacker_validation' and role != 'outer_assessment':
                    continue
                mask = _role_mask(np.asarray(pool['households']), role)
                people += int(mask.sum())
                houses.update(str(x) for x in np.asarray(pool['households'])[mask])
                weight_sum += float(np.asarray(pool['weights'])[mask].sum())
                if role != 'outer_assessment':
                    for target, count in CLASS_COUNT.items():
                        observed = np.asarray(pool['labels'][target])[mask]
                        bins = np.bincount(observed, minlength=count)
                        if len(bins) != count:
                            raise ValueError('class schema changed')
                        support[target] += bins
            global_houses[role].update(houses)
            row = {'people': people, 'households': len(houses),
                   'weight_sum': weight_sum}
            if role != 'outer_assessment':
                row['class_support'] = {key: values.tolist()
                                        for key, values in support.items()}
            by_role[role] = row
        anchors[str(anchor)] = by_role
    overlap = sum(len(global_houses[a] & global_houses[b])
                  for j, a in enumerate(ROLE_NAMES) for b in ROLE_NAMES[j+1:])
    if overlap:
        raise AssertionError('global household role overlap')
    return {'schema': 'pcrl-adaptive-role-summary-v1',
            'algorithm': 'first 64 bits SHA256(UTF8(pcrl_adaptive_release_v1|household_id)) / 2^64',
            'role_upper_bounds': dict(zip(ROLE_NAMES, ROLE_UPPER)),
            'global_households': {key: len(value) for key, value in global_houses.items()},
            'global_role_overlap': overlap, 'anchors': anchors,
            'outer_labels_accessed': False}
