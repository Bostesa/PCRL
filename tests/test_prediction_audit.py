"""Targeted boundary tests for the final scalar-release audit."""
import json
import numpy as np
import pytest
import torch
from torch import nn

from experiments import run_prediction_audit as audit
from experiments import run_nonlinear_conflict as old


def evaluation(value=0.):
    return {'task': {'mlp': {name: {'r2': .995} for name in ('p1_U', 'p2_V')}},
            'attack': {kind: {name: {'r2': value} for name in audit.TARGET_NAMES}
                       for kind in audit.CONFIG['families']}}


def test_every_constraint_and_own_threshold_fail_closed():
    row = evaluation()
    assert audit.assess(row)['feasible']
    row['attack']['histgb']['combined_S']['r2'] = .09
    assert audit.assess(row)['feasible']
    row['attack']['histgb']['p1_V']['r2'] = .051
    assert not audit.assess(row)['protection_pass']
    row['attack']['histgb']['p1_V']['r2'] = float('nan')
    assert not audit.assess(row)['protection_pass']
    row = evaluation(-.01)
    row['task']['mlp']['p2_V']['r2'] = .98999
    assert not audit.assess(row)['utility_pass']


def test_fresh_test_requires_matching_saved_selection_and_disjoint_ids(tmp_path, monkeypatch):
    # Nonpilot seed and tiny arrays prevent consuming any predeclared pilot test.
    monkeypatch.setitem(audit.CONFIG, 'test_n', 17)
    data, manifest = old.make_data(91, include_test=False)
    selection = tmp_path/'selection.json'
    with pytest.raises(RuntimeError):
        audit.fresh_test(91, manifest, selection)
    selection.write_text(json.dumps({'test_rng_seed': 1109104, 'test_generated': False}))
    test = audit.fresh_test(91, manifest, selection)
    assert len(test['y']) == 17
    for row in data.values():
        assert not np.intersect1d(test['ids'], row['ids']).size
    selection.write_text(json.dumps({'test_rng_seed': 1, 'test_generated': False}))
    with pytest.raises(ValueError):
        audit.fresh_test(91, manifest, selection)


def test_fitting_roles_targets_and_dimensions_are_explicit(tmp_path, monkeypatch):
    from experiments import prediction_release_attackers as attacks
    rng = np.random.default_rng(2)
    data = {key: {'y': rng.normal(size=(n, 3))}
            for key, n in [('representation_train', 11), ('attacker_fit', 13), ('validation', 7)]}
    views = {key: [rng.normal(size=(len(row['y']), d)) for d in (1, 1, 2)] for key, row in data.items()}
    calls = []
    def spy(xf, yf, xv, yv, **kwargs):
        calls.append((xf, yf, xv, yv, kwargs))
        return {}
    monkeypatch.setattr(attacks, 'fit_attackers', spy)
    audit.fit_probes(views, data, 91, tmp_path)
    assert len(calls) == 5
    for p, targets in enumerate(((1, 2), (0, 2), (2,))):
        xf, yf, xv, yv, kw = calls[p]
        assert xf is views['attacker_fit'][p]
        assert xv is views['validation'][p]
        np.testing.assert_array_equal(yf, data['attacker_fit']['y'][:, targets])
        np.testing.assert_array_equal(yv, data['validation']['y'][:, targets])
        assert len(kw['target_names']) == len(targets)
    for p, call in enumerate(calls[3:]):
        assert call[0] is views['representation_train'][p]
        np.testing.assert_array_equal(call[1], data['representation_train']['y'][:, p:p+1])


def test_scalar_release_preserves_frozen_parameters_buffers_and_concat():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Dropout(.7))
        def forward(self, x, purpose):
            return self.net(x)+purpose
    model = {'encoder': audit.freeze(Encoder()), 'heads': audit.freeze(nn.ModuleList([nn.Linear(4, 1) for _ in range(2)]))}
    before = audit.scalar_integrity(model)
    data = {'validation': {'x': np.random.default_rng(3).normal(size=(17, 3))}}
    first = audit.prediction_views(model, data)['validation']
    second = audit.prediction_views(model, data)['validation']
    assert [v.shape[1] for v in first] == [1, 1, 2]
    np.testing.assert_array_equal(first[2], np.concatenate(first[:2], axis=1))
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert audit.scalar_integrity(model) == before
    assert not any(p.requires_grad for module in model.values() for p in module.parameters())


def test_validation_family_selection_uses_each_aligned_target():
    row = evaluation()
    row['attack']['histgb']['p1_V']['r2'] = .1
    row['attack']['mlp']['p2_U']['r2'] = .2
    selected = audit.selected_families(row)
    assert selected['p1_V'] == 'histgb'
    assert selected['p2_U'] == 'mlp'
    assert selected['combined_S'] == 'histgb'  # declared alphabetical tie
