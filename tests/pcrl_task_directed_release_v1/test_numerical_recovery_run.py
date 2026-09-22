"""Branch D orchestration tests use only tiny synthetic accepted artifacts."""
import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from experiments.pcrl_task_directed_release_v1 import finite, mechanisms, run


class Code:
    def n_states(self, family): return 2 if family == 'T0' else 4
    def parents(self, family): return np.arange(2) if family == 'T0' else np.repeat(np.arange(2), 2)


def module():
    return importlib.import_module('experiments.pcrl_task_directed_release_v1.numerical_recovery_run')


def save_map(root, anchor, spec, tables, encoder, ctx, encoded, coarse=None):
    name = spec['configuration']; family = spec['input']; table = tables[family]
    base = root/'private/run'/f'anchor_{anchor}'/'maps'/name
    base.mkdir(parents=True)
    q = np.full((encoder.code.n_states(family), 2), .5)
    parents = encoder.code.parents(family)
    selected = {k: p for k, p in table['roles'].items() if spec['policy'] == 'C' or k.startswith('A/')}
    result = finite.solve(table['cost'], selected, spec['budget'], parent=parents, state_mass=table['state_mass'])
    result.update(Q=q, feasible=True, optimal=family == 'T0', status='optimal' if family == 'T0' else 'feasible_witness',
        input=family, policy=spec['policy'], budget=spec['budget'], configuration=name, actions=17,
        constrained_roles=sorted(selected), aggregation=table['aggregation'], population=table['population'],
        constant_action=0, cost=table['cost'], objective=float(np.sum(table['cost']*q)))
    result['embedding'] = None if coarse is None else mechanisms._embedding(ctx, encoder, encoded, tables, family, coarse['Q'], 17)[1]
    arrays = {k: table[k] for k in ('cost_U', 'cost_W', 'cost', 'state_mass')}
    arrays.update({'joint/'+k: p for k, p in table['roles'].items()})
    arrays.update({'support/'+k+'/'+field: v for k, fields in table['support'].items() for field, v in fields.items()})
    np.savez_compressed(base/'tables.npz', **arrays)
    np.savez_compressed(base/'Q.npz', Q=q)
    run.atomic(base/'metadata.json', {k: v for k, v in result.items() if k not in ('Q', 'cost')})
    run.atomic_joblib(base/'solution.joblib', result)
    coarse_name = None if coarse is None else coarse['configuration']
    run.atomic(base/'ACCEPTED.json', {**run._provenance('map'), 'anchor': anchor, 'configuration': name,
        'sha256': run.sha(base/'solution.joblib'), 'spec_hash': run.digest(spec),
        'prepared_cache_sha256': run.sha(base.parents[1]/'prepared.joblib'),
        'coarse_configuration': coarse_name,
        'coarse_solution_sha256': None if coarse is None else run.sha(base.parent/coarse_name/'solution.joblib'),
        'artifact_hashes': run._artifact_hashes(base, [base])})
    return result


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    names = [(0, 'Ttask_C_0.0005_a17'), (0, 'Ttask_C_0.01_a17'), (0, 'Ttask_L_0.002_a17'), (2, 'Trisk_C_0.01_a17')]
    root = tmp_path
    for anchor in (0, 2):
        base = root/'private/run'/f'anchor_{anchor}';base.mkdir(parents=True)
        encoder = SimpleNamespace(code=Code(), dictionaries={17: {'zero_action': 0, 'offsets': np.array([0., 1.])}})
        role_keys = [f'{view}/{label}/{weight}' for view in ('A', 'AB') for label in ('SEX', 'RAC1P') for weight in ('U', 'W')]
        law = np.full((2, 1, 2), .25)
        coarse_table = {'cost_U': np.array([[.1, .8], [.7, .2]]), 'state_mass': np.array([2, 2]),
            'roles': {k: law.copy() for k in role_keys}, 'support': {}, 'actions': 17,
            'aggregation': {'maximum_error': 0., 'errors': {}}, 'population': {'mechanism_rows': 4}}
        coarse_table['cost_W'] = coarse_table['cost_U'].copy();coarse_table['cost'] = coarse_table['cost_U'].copy()
        tables = {'T0': coarse_table}
        for family in ('Ttask', 'Trisk'):
            table = {k: np.repeat(coarse_table[k], 2, axis=0)/2 for k in ('cost_U', 'cost_W', 'cost')}
            table.update(state_mass=np.ones(4), roles={k: np.repeat(law, 2, axis=2)/2 for k in role_keys},
                support={}, actions=17, aggregation={'maximum_error': 0., 'errors': {}}, population={'mechanism_rows': 4})
            tables[family] = table
        ctx = {'anchor': anchor, 'pools': {p: {'ha': np.zeros((4, 4))} for p in run.FIT_POOLS}}
        encoded = {p: {'codes': {'T0': np.array([0, 0, 1, 1]), 'Ttask': np.arange(4), 'Trisk': np.arange(4)},
            'actions': {17: np.tile([.3, .7], (4, 1))}} for p in run.FIT_POOLS}
        run.atomic_joblib(base/'prepared.joblib', {'ctx': ctx, 'encoder': encoder, 'encoded': encoded, 'tables': tables})
        run.atomic(base/'PREPARED.json', {**run._provenance('prepare'), 'anchor': anchor,
            'cache_sha256': run.sha(base/'prepared.joblib'), 'artifact_hashes': {'prepared.joblib': run.sha(base/'prepared.joblib')}})
        for a, name in names:
            if a != anchor: continue
            spec = run.map_spec(name, anchor)
            coarse_name = run.mechanism_id('T0', spec['policy'], spec['budget'])
            coarse_spec = run.map_spec(coarse_name, anchor)
            coarse = save_map(root, anchor, coarse_spec, tables, encoder, ctx, encoded)
            save_map(root, anchor, spec, tables, encoder, ctx, encoded, coarse)
            audit = base/'audits'/name;audit.mkdir(parents=True)
            (audit/'registry.joblib').write_bytes(b'synthetic registry; never unpickle')
            (audit/'summary.json').write_text('{"never_read":true}')
            run.atomic(audit/'COMPLETE.json', {**run._provenance('audit'), 'anchor': anchor, 'configuration': name,
                'map_solution_sha256': run.sha(base/'maps'/name/'solution.joblib'),
                'registry_sha256': run.sha(audit/'registry.joblib'), 'prepared_cache_sha256': run.sha(base/'prepared.joblib'),
                'artifact_hashes': run._artifact_hashes(audit, [audit])})
    run.atomic(root/'RESOURCE_SCHEDULE.json', {'config_hash': run.digest(run.configuration()), 'frozen_before_comparative_outcomes': True})
    run.atomic(root/'private/SCHEDULER.json', {'status': 'complete', 'active_jobs': []})
    return root, names


def test_registration_is_exact_immutable_four_slots_and_does_not_solve(fixture, monkeypatch):
    root, names = fixture
    monkeypatch.setattr(module().recovery, 'solve_normalized_retry', lambda *a, **k: pytest.fail('register solved'))
    registry = module().register()
    assert [(s['anchor'], s['configuration']) for s in registry['slots']] == names
    assert set(registry['recovery_source_hashes']) == {'numerical_recovery.py', 'numerical_recovery_run.py'}
    assert all(s['original']['receipt_sha256'] and s['audit']['receipt_sha256'] for s in registry['slots'])
    with pytest.raises(FileExistsError): module().register()


def test_execute_stages_identical_tables_standard_result_once_without_mutation(fixture):
    root, names = fixture;mod = module();mod.register()
    anchor, name = names[0]
    original = root/'private/run'/f'anchor_{anchor}'/'maps'/name
    before = mod._tree_hashes(original)
    result = mod.execute(anchor, name)
    assert result['accepted']
    stage = root/'private/numerical_recovery'/f'anchor_{anchor}'/name
    solution = joblib.load(stage/'map/solution.joblib')
    assert solution['configuration'] == name and solution['input'] == 'Ttask' and solution['actions'] == 17
    assert solution['constrained_roles'] == sorted(solution['independent_cmi'])
    assert run.sha(stage/'map/tables.npz') == run.sha(original/'tables.npz')
    assert mod._tree_hashes(original) == before
    assert not (stage/'map/ACCEPTED.json').exists()
    with pytest.raises(FileExistsError): mod.execute(anchor, name)


def test_changed_original_or_source_blocks_before_claim(fixture, monkeypatch):
    root, names = fixture;mod = module();mod.register();anchor, name = names[0]
    monkeypatch.setattr(mod, '_recovery_sources', lambda: {'changed': 'source'})
    with pytest.raises(ValueError, match='source'): mod.execute(anchor, name)
    assert not (root/'private/numerical_recovery'/f'anchor_{anchor}'/name).exists()


def test_resealed_prepared_inputs_cannot_change_registered_table(fixture):
    root, names = fixture;mod = module();mod.register();anchor, name = names[0]
    path = root/'private/run'/f'anchor_{anchor}'/'prepared.joblib'
    prepared = joblib.load(path);prepared['tables']['Ttask']['cost'][0, 0] += .01
    run.atomic_joblib(path, prepared)
    with pytest.raises(ValueError, match='hash|changed'): mod.execute(anchor, name)


def test_rejected_retry_keeps_original_and_install_does_not_archive(fixture, monkeypatch):
    root, names = fixture;mod = module();mod.register();anchor, name = names[0]
    original = root/'private/run'/f'anchor_{anchor}'/'maps'/name
    before = mod._tree_hashes(original)
    actual = mod.recovery.solve_normalized_retry
    def reject(*a, **k):
        r = actual(*a, **k);r.update(accepted=False, feasible=False, optimal=False, status='failed', Q=None, objective=None)
        return r
    monkeypatch.setattr(mod.recovery, 'solve_normalized_retry', reject)
    assert not mod.execute(anchor, name)['accepted']
    installed = mod.install(anchor, name, writers_closed=True)
    assert installed['decision'] == 'retained_original'
    assert mod._tree_hashes(original) == before
    assert not (root/'private/numerical_originals'/f'anchor_{anchor}'/name).exists()


def test_install_archives_exact_originals_and_invalidates_only_changed_audit(fixture):
    root, names = fixture;mod = module();mod.register();anchor, name = names[0]
    base = root/'private/run'/f'anchor_{anchor}'
    map_before = mod._tree_hashes(base/'maps'/name);audit_before = mod._tree_hashes(base/'audits'/name)
    mod.execute(anchor, name)
    installed = mod.install(anchor, name, writers_closed=True)
    assert installed['decision'] == 'installed_accepted_retry'
    backup = root/'private/numerical_originals'/f'anchor_{anchor}'/name
    assert mod._tree_hashes(backup/'map') == map_before
    assert mod._tree_hashes(backup/'audit') == audit_before
    assert not (base/'audits'/name).exists()
    assert (base/'audits'/names[1][1]/'COMPLETE.json').exists()
    record = run._read_marker(base/'maps'/name/'ACCEPTED.json', 'map', anchor=anchor, name=name)
    assert mod.verify_installed_retry(anchor, name, record)
    assert mod.install(anchor, name, writers_closed=True) == installed


def test_install_refuses_active_scheduler_locks_and_frozen_selection(fixture):
    root, names = fixture;mod = module();mod.register();anchor, name = names[0];mod.execute(anchor, name)
    with pytest.raises(RuntimeError, match='closed'): mod.install(anchor, name)
    run.atomic(root/'private/SCHEDULER.json', {'status': 'running', 'active_jobs': []})
    with pytest.raises(RuntimeError, match='scheduler'): mod.install(anchor, name, writers_closed=True)
    run.atomic(root/'private/SCHEDULER.json', {'status': 'complete', 'active_jobs': []})
    with run.unit_lock(f'audit/{anchor}/{name}'):
        with pytest.raises(RuntimeError, match='lock'): mod.install(anchor, name, writers_closed=True)
    with run.unit_lock('programme/selection_freeze'):
        with pytest.raises(RuntimeError, match='lock'): mod.install(anchor, name, writers_closed=True)
    run.atomic(root/'SELECTION.json', {'selection_frozen': True})
    with pytest.raises(RuntimeError, match='frozen'): mod.install(anchor, name, writers_closed=True)


def test_staging_crash_consumes_attempt_and_preserves_original(fixture, monkeypatch):
    root, names = fixture;mod = module();mod.register();anchor, name = names[0]
    def fail(*a, **k): raise RuntimeError('synthetic crash')
    monkeypatch.setattr(mod.recovery, 'solve_normalized_retry', fail)
    with pytest.raises(RuntimeError, match='crash'): mod.execute(anchor, name)
    with pytest.raises(FileExistsError): mod.execute(anchor, name)
    assert (root/'private/run'/f'anchor_{anchor}'/'maps'/name/'ACCEPTED.json').exists()


def test_restored_receipt_verification_never_falls_back_to_original_paths(fixture, monkeypatch):
    root, names = fixture;mod = module();mod.register();anchor, name = names[0]
    mod.execute(anchor, name);mod.install(anchor, name, writers_closed=True)
    restored = root.parent/(root.name+'_restored');out = restored/'results'/run.STUDY
    shutil.copytree(root, out)
    code = restored/'experiments'/run.STUDY;code.mkdir(parents=True)
    for source in (Path(run.__file__).parent).glob('*.py'):
        shutil.copy2(source, code/source.name)
    record = json.loads((out/'private/run'/f'anchor_{anchor}'/'maps'/name/'ACCEPTED.json').read_text())
    # Move the entire original study out of its old path before replay.
    root.rename(root.parent/(root.name+'_unavailable'))
    monkeypatch.setattr(run, 'source_fingerprint', lambda *a: pytest.fail('original source resolver used'))
    assert mod.verify_installed_retry(anchor, name, record, study_out=out, source_root=restored)
    (out/'private/numerical_originals'/f'anchor_{anchor}'/name/'audit/registry.joblib').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='originals'):
        mod.verify_installed_retry(anchor, name, record, study_out=out, source_root=restored)


def test_install_failure_rolls_back_original_bytes_without_refitting(fixture, monkeypatch):
    root, names = fixture;mod = module();mod.register();anchor, name = names[0];mod.execute(anchor, name)
    base = root/'private/run'/f'anchor_{anchor}'
    before_map = mod._tree_hashes(base/'maps'/name);before_audit = mod._tree_hashes(base/'audits'/name)
    rename = mod.os.rename
    def fail_candidate(source, destination):
        if Path(source).name == 'installation_candidate': raise OSError('synthetic install failure')
        return rename(source, destination)
    monkeypatch.setattr(mod.os, 'rename', fail_candidate)
    with pytest.raises(OSError, match='install failure'): mod.install(anchor, name, writers_closed=True)
    assert mod._tree_hashes(base/'maps'/name) == before_map
    assert mod._tree_hashes(base/'audits'/name) == before_audit
    with pytest.raises(RuntimeError, match='manual recovery'): mod.install(anchor, name, writers_closed=True)


def test_completion_acceptance_cannot_disagree_with_hashed_solver_metadata(fixture):
    root, names = fixture;mod = module();mod.register();anchor, name = names[0];mod.execute(anchor, name)
    path = mod._stage(anchor, name)/'COMPLETE.json'
    complete = json.loads(path.read_text());complete['optimal'] = not complete['optimal']
    run.atomic(path, complete)
    with pytest.raises(ValueError, match='solver metadata'): mod.install(anchor, name, writers_closed=True)


def test_prepared_table_mismatch_is_detected_even_when_both_artifacts_are_validly_pinned(fixture):
    root, names = fixture;mod = module();anchor, name = names[0]
    base = root/'private/run'/f'anchor_{anchor}'/'maps'/name
    with np.load(base/'tables.npz') as f: arrays = {k: f[k].copy() for k in f.files}
    arrays['cost'][0, 0] += .001
    np.savez_compressed(base/'tables.npz', **arrays)
    receipt = json.loads((base/'ACCEPTED.json').read_text())
    receipt['artifact_hashes']['tables.npz'] = run.sha(base/'tables.npz')
    run.atomic(base/'ACCEPTED.json', receipt)
    mod.register()
    with pytest.raises(ValueError, match='tables differ'): mod.execute(anchor, name)


def test_closed_writer_gate_creates_and_holds_first_selection_freeze_lock(fixture):
    root, _ = fixture;mod = module()
    path = root/'private/locks'/(mod.hashlib.sha256(b'programme/selection_freeze').hexdigest()+'.lock')
    assert not path.exists()
    with mod._closed_writers(True):
        assert path.exists()
        with pytest.raises(RuntimeError, match='locked'):
            with run.unit_lock('programme/selection_freeze'): pass


def test_archival_target_symlink_escape_is_rejected_before_any_move(fixture):
    root, names = fixture;mod = module();mod.register();anchor, name = names[0];mod.execute(anchor, name)
    outside = root.parent/(root.name+'_outside');outside.mkdir()
    (root/'private/numerical_originals').symlink_to(outside, target_is_directory=True)
    before = mod._tree_hashes(root/'private/run'/f'anchor_{anchor}'/'maps'/name)
    with pytest.raises(ValueError, match='escapes'): mod.install(anchor, name, writers_closed=True)
    assert not list(outside.iterdir())
    assert mod._tree_hashes(root/'private/run'/f'anchor_{anchor}'/'maps'/name) == before
