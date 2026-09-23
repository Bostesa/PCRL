"""Cheap orchestration check; no historical person records or model fits."""
from __future__ import annotations

import numpy as np

import json

import pytest

from experiments.pcrl_task_aligned_cuts_v1 import audit, data, inner_panel


def _pool(prefix: str):
    n = 4
    return {
        'ha': np.zeros((n, 4)), 'hb': np.ones((n, 2)),
        'labels': {'same_residence': np.array([0, 1, 0, 1]),
                   'SEX': np.array([0, 1, 0, 1]),
                   'RAC1P': np.array([0, 1, 0, 1])},
        'weights': np.ones(n),
        'ids': np.array([f'{prefix}-person-{i}' for i in range(n)]),
        'households': np.array([f'{prefix}-house-{i}' for i in range(n)]),
    }


def test_inner_panel_all_roles_legal_ancestors_and_no_outer(monkeypatch, tmp_path):
    pools = {'downstream_fit': _pool('fit'),
             'downstream_validation': _pool('validation')}
    codes = np.array([0, 1, 0, 1])
    prepared = {'ctx': {'anchor': 0, 'pools': pools},
                'encoded': {name: {'codes': {'T0': codes}} for name in pools}}
    split = {'assignment_sha256':
             '9624c02a5dfc1797c602c0ab49b55f0853d2ca12a2a631ebbf2bf307253c27bf',
             'anchors': {0: {'rows': {'inner_selection': np.array([0, 1]),
                                    'inner_pilot': np.array([2, 3])}}}}
    q = np.zeros((32, 17))
    q[:, 0] = 1.
    fitted = []
    selected = []

    def fake_fit(fit_rows, fit_p, validation_rows, validation_p, role, output_dir,
                 seed, *, release_id, slate):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'own_registry.json').write_text(json.dumps({'role': role, 'release_id': release_id}))
        fitted.append((role, release_id, len(fit_rows['ids']),
                       tuple(validation_rows['ids']), fit_p.shape[1], slate, seed))
        _, view, target = audit.parse_role(role)
        return {'role': role, 'release_id': release_id, 'slate': slate,
                'fit_missing_classes': [3] if target == 'RAC1P' else [],
                'validation_missing_classes': [3] if target == 'RAC1P' else [],
                'models': {'toy': {'kind': 'model', 'target': target,
                                   'source_view': view,
                                   'wire': 'H' if release_id == 'H' else 'release',
                                   'source_release_id': release_id,
                                   'model_sha256': 'a' * 64,
                                   'model_directory': str(output_dir)}}}

    def fake_select(rows, token_probs, role, routes, *, release_id):
        selected.append((role, release_id, tuple(rows['ids']), tuple(routes)))
        key = sorted(routes)[0]
        return {'role': role, 'release_id': release_id,
                'selected': key, 'route': routes[key], 'candidate_count': len(routes),
                'scores': {key: {'U': 0.5, 'PWGTP': 0.5, 'balanced': 0.5}},
                'rule': 'toy inner-selection rule'}

    def fake_score(rows, token_probs, lock):
        n = len(rows['ids'])
        value = 0.4 if lock['release_id'] == 'candidate' else 0.5
        return {'ids': rows['ids'], 'households': rows['households'],
                'weights': rows['weights'], 'loss': np.full(n, value),
                'scores': {'U': value, 'PWGTP': value, 'balanced': value}}

    monkeypatch.setattr(audit, 'fit_role_slate', fake_fit)
    monkeypatch.setattr(audit, 'select_frozen_routes', fake_select)
    monkeypatch.setattr(audit, 'score_frozen_route', fake_score)
    out = tmp_path / 'private' / 'pilot'
    result = inner_panel.run_inner_panel(prepared, q, split, 'candidate', out)
    assert len(fitted) == 12  # five release roles plus seven H/ancestor roles
    assert {role for role, *_ in fitted if role in audit.ROLES} == set(audit.ROLES)
    assert all(n_fit == 4 and ids == ('validation-person-0', 'validation-person-1')
               and slate == 'standard' for _, _, n_fit, ids, _, slate, _ in fitted)
    assert all(ids == ('validation-person-0', 'validation-person-1')
               for _, _, ids, _ in selected)
    coalition = [routes for role, release, _, routes in selected
                 if role == 'attack:AB/RAC1P' and release == 'candidate'][0]
    assert any(name.startswith('A/') for name in coalition)
    assert any(name.startswith('B/') for name in coalition)
    assert result['outer_pool_opened'] is False
    assert set(result['roles']) == set(audit.ROLES)
    assert result['roles']['utility:A/same_residence']['H_minus_candidate']['U'] > 0
    with np.load(out / 'INNER_PILOT_CONTRIBUTIONS.npz', allow_pickle=False) as private:
        assert set(private['utility_A_same_residence_households']) == {
            'validation-house-2', 'validation-house-3'}


def test_shared_h_requires_complete_hash_pinned_source(tmp_path):
    source = tmp_path / 'private' / 'source_audit'
    h_root = source / 'H'
    (h_root / 'attack_A_SEX').mkdir(parents=True)
    model = h_root / 'attack_A_SEX' / 'own_registry.json'
    model.write_text('{"toy": true}\n')
    receipt = {'unit_id': 'source_audit',
               'artifacts': {'H/attack_A_SEX/own_registry.json': data.sha256_file(model)}}
    (source / 'COMPLETE.json').write_text(json.dumps(receipt))
    verified = inner_panel._verified_shared_h_receipt(h_root)
    assert verified['source_unit_id'] == 'source_audit'
    assert verified['artifact_count'] == 1
    model.write_text('{"toy": false}\n')
    with pytest.raises(ValueError, match='differs'):
        inner_panel._verified_shared_h_receipt(h_root)
