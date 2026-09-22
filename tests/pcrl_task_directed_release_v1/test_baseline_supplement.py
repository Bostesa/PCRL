"""Prospective fairness supplements use exact frozen slices and immutable parents."""
import copy
import importlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from scipy.special import expit

from experiments.pcrl_task_directed_release_v1 import run, selection


def module():
    return importlib.import_module('experiments.pcrl_task_directed_release_v1.baseline_supplement')


@pytest.fixture
def frozen_parent(tmp_path, monkeypatch):
    out = tmp_path/'results'; monkeypatch.setattr(run, 'OUT', out)
    out.mkdir(); (out/'AMENDMENT_6_BASELINE_FAIRNESS.md').write_text('Prospective synthetic amendment.\n')
    run.atomic(out/'RESOURCE_SCHEDULE.json', {'config_hash': run.digest(run.configuration()),
                                            'frozen_before_comparative_outcomes': True})
    rng = np.random.default_rng(40); n = 500; x = rng.normal(size=(n, 32))
    rows = {'x': x, 'ha': rng.normal(size=(n, 4)), 'hb': rng.normal(size=(n, 2)),
        'ids': np.array([f'p{i}' for i in range(n)]), 'households': np.array([f'h{i//2}' for i in range(n)]),
        'weights': rng.uniform(1, 10, n), 'labels': {'SEX': rng.integers(0, 2, n),
        'RAC1P': rng.integers(0, 9, n), **{k: rng.integers(0, 2, n)
            for k in ('same_residence', 'income_binary', 'civilian_at_work')}}}
    prepared = {'ctx': {'anchor': 0, 'pools': {'representation_fit': rows}},
        'roles': {'teacher_fit': np.arange(240), 'teacher_internal_validation': np.arange(240, 300),
                  'mechanism': np.arange(300, 500)},
        'encoded': {'representation_fit': {'p': .1+.8*expit(x[:, 0]), 'actions': {17: np.ones((n, 1))*.5},
                                          'b': np.ones(n)*.5}}, 'erasers': {}, 'encoder': None}
    base = out/'private/run/anchor_0'; (base/'encoder').mkdir(parents=True)
    joblib.dump(prepared, base/'prepared.joblib'); (base/'encoder/encoder.joblib').write_bytes(b'frozen opaque source pin')
    run.atomic(base/'PREPARED.json', {**run._provenance('prepare'), 'anchor': 0,
        'cache_sha256': run.sha(base/'prepared.joblib'), 'artifact_hashes': {
            'prepared.joblib': run.sha(base/'prepared.joblib'),
            'encoder/encoder.joblib': run.sha(base/'encoder/encoder.joblib')}})
    return out, prepared


def register_and_schedule():
    mod = module(); registry = mod.register()
    schedule = mod.freeze_schedule(resource_decision={'registry_commit': 'a'*40, 'reason': 'prospective fairness repair'})
    return mod, registry, schedule


def test_registry_has_exact12_controls_and_schedule_requires_committed_registry(frozen_parent):
    mod = module(); record = mod.register()
    assert record['controls'] == mod.specs() and len(mod.specs()) == 12
    assert {s['scope'] for s in mod.specs()} == {'mechanism40', 'union88'}
    with pytest.raises(ValueError, match='commit'):
        mod.freeze_schedule(resource_decision={'registry_commit': 'not-a-commit'})
    schedule = mod.freeze_schedule(resource_decision={'registry_commit': 'a'*40})
    assert schedule['unit_ids'] == [s['id'] for s in mod.specs()]
    assert mod.register() == record
    with pytest.raises(FileExistsError):
        mod.freeze_schedule(resource_decision={'registry_commit': 'a'*40})


@pytest.mark.parametrize('scope,expected', [('mechanism40', np.arange(300, 500)),
    ('union88', np.r_[np.arange(240), np.arange(300, 500)])])
def test_new_pairs_use_exact_scope_and_preserve_parent_cache_and_runtime_H(frozen_parent, scope, expected, monkeypatch):
    from experiments.pcrl_task_directed_release_v1 import baselines, mechanisms
    out, prepared = frozen_parent; mod, _, _ = register_and_schedule()
    source_hash = run.sha(out/'private/run/anchor_0/prepared.joblib')
    name = 'leace_supervised_'+scope; spec = mod.lookup_release(name, 0)
    before = copy.deepcopy(prepared)
    augmented = mod.prepared_for_spec(0, prepared, spec)
    assert name in augmented['erasers'] and not prepared['erasers']
    path = out/'private/run/anchor_0/baseline_supplement'/scope
    with np.load(path/'slice.npz') as f:
        np.testing.assert_array_equal(f['rf_row_indices'], expected)
        assert not np.isin(f['rf_row_indices'], np.arange(240, 300)).any()
    fitted = augmented['erasers'][name]
    assert fitted.metadata['fit_scope'] == scope
    assert fitted.metadata['provided_rows'] == len(expected) and not fitted.metadata['fit_weighted']
    assert (path/'splince_supervised/diagnostics.json').exists()
    release = mechanisms.build_release(augmented['ctx'], None, augmented['encoded'], name, erasers=augmented['erasers'])
    assert release['representation_fit']['aux'].shape == (500, 33)
    np.testing.assert_array_equal(prepared['ctx']['pools']['representation_fit']['ha'], before['ctx']['pools']['representation_fit']['ha'])
    assert run.sha(out/'private/run/anchor_0/prepared.joblib') == source_hash
    accepted = mod.artifact_receipt(0, spec)['receipt_sha256']
    monkeypatch.setattr(baselines, 'fit_supervised_erasers', lambda *a, **k: pytest.fail('Cached supplement refit'))
    mod.prepared_for_spec(0, prepared, spec, allow_fit=False)
    assert mod.artifact_receipt(0, spec)['receipt_sha256'] == accepted


def test_fit_requires_schedule_and_disjoint_frozen_household_roles(frozen_parent):
    out, prepared = frozen_parent; mod = module(); mod.register()
    spec = mod.lookup_release('leace_supervised_mechanism40', 0)
    with pytest.raises((ValueError, FileNotFoundError)):
        mod.prepared_for_spec(0, prepared, spec)
    mod.freeze_schedule(resource_decision={'registry_commit': 'a'*40})
    bad = copy.deepcopy(prepared); bad['roles']['mechanism'][0] = 0
    with pytest.raises(ValueError, match='slice|role|prepared|household'):
        mod.validate_slice(bad, 'union88')


def test_supplement_settings_are_mandatory_in_existing_two_families(frozen_parent):
    mod, _, _ = register_and_schedule()
    controls = {s['configuration']: mod.lookup_release(s['configuration'], 0) for s in mod.specs() if s['anchor'] == 0}
    groups = selection.default_families(registered_controls=controls)
    assert set(groups['supervised_LEACE']['required_configurations']) == {
        'leace_supervised', 'leace_supervised_mechanism40', 'leace_supervised_union88'}
    assert set(groups['supervised_SPLINCE']['required_configurations']) == {
        'splince_supervised', 'splince_supervised_mechanism40', 'splince_supervised_union88'}


def test_runtime_release_routes_to_new_map_and_cached_evaluation_cannot_fit(frozen_parent, monkeypatch):
    from experiments.pcrl_task_directed_release_v1 import baselines
    _, prepared = frozen_parent; mod, _, _ = register_and_schedule()
    name = 'leace_supervised_mechanism40'
    release = run.release_for(name, prepared, 0)
    assert release['representation_fit']['aux'].shape == (500, 33)
    assert not prepared['erasers']
    assert run._extra_module(run._extra_release(name, 0)) is mod
    monkeypatch.setattr(baselines, 'fit_supervised_erasers', lambda *a, **k: pytest.fail('Replay refit'))
    replayed = run.release_for(name, prepared, 0, allow_fit=False)
    np.testing.assert_array_equal(replayed['representation_fit']['aux'], release['representation_fit']['aux'])


def test_restored_supplement_loads_only_owned_maps_and_reconstructs_moments(frozen_parent, tmp_path, monkeypatch):
    import shutil
    from experiments.pcrl_task_directed_release_v1 import verify
    out, prepared = frozen_parent; mod, _, _ = register_and_schedule()
    name = 'splince_supervised_union88'; spec = mod.lookup_release(name, 0)
    original = mod.prepared_for_spec(0, prepared, spec)['erasers'][name]
    restored = tmp_path/'restore'; target = restored/'results'/run.STUDY
    shutil.copytree(out, target); (target/'private/inputs').mkdir()
    shutil.rmtree(out/'private/run')
    provider = verify._FrozenArtifacts(verify._Layout(restored, tmp_path/'original'), {})
    restored_spec = provider.spec(name, 0)
    assert restored_spec == spec
    assert provider.branch_schedule(restored_spec) == run.sha(target/mod.SCHEDULE)
    release = provider.release(name, 0, prepared)
    from scipy.special import logit
    features = np.column_stack((prepared['ctx']['pools']['representation_fit']['x'],
        logit(prepared['encoded']['representation_fit']['p'])))
    np.testing.assert_array_equal(release['representation_fit']['aux'], original.transform(features))
    report = verify.verify_baseline_supplement(prepared, restored_spec, study_out=target)
    assert report['passed'] and report['provided_people'] == 440 and report['provided_households'] == 220
    assert report['guardedness_passed'] and report['task_preservation_passed']
    bad = copy.deepcopy(prepared); bad['ctx']['pools']['representation_fit']['labels']['SEX'][0] ^= 1
    with pytest.raises(ValueError, match='slice'):
        verify.verify_baseline_supplement(bad, restored_spec, study_out=target)
