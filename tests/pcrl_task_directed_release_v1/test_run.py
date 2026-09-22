"""Runner integrity tests use synthetic artifacts and never survey inputs."""
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from experiments.pcrl_task_directed_release_v1 import run


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'configuration', lambda: {'maps': [
        {'configuration': 'T0_C_0.002_a17', 'anchor': 0, 'input': 'T0',
         'policy': 'C', 'budget': .002, 'max_actions': 17}]})
    monkeypatch.setattr(run, 'source_fingerprint', lambda stage=None: {'science': 'unchanged'})
    return tmp_path


def test_test_context_cannot_fit_a_missing_map(isolated, monkeypatch):
    called=[]
    monkeypatch.setattr(run, 'ensure_map', lambda *a, **k: called.append('fit'))
    with pytest.raises((FileNotFoundError, ValueError)):
        run.release_for('T0_C_0.002_a17', {'ctx': {'pools': {'test': {}}}}, 0, allow_fit=False)
    assert not called
    with pytest.raises(ValueError, match='evaluation|test'):
        run.release_for('T0_C_0.002_a17', {'ctx': {'pools': {'test': {}}}}, 0)


def test_evaluation_checks_frozen_artifacts_before_loading_test(isolated, monkeypatch):
    run.atomic(isolated/'SELECTION.json', {'selection_frozen': True})
    calls=[]
    monkeypatch.setattr(run, 'prepare', lambda *a, **k: calls.append(('prepare', k)) or {})
    monkeypatch.setattr(run, 'load_anchor', lambda *a, **k: pytest.fail('test loader opened'))
    with pytest.raises(FileNotFoundError):
        run.evaluate_unit(0, 'T0_C_0.002_a17')
    assert all(k.get('allow_fit') is False for _, k in calls)


def test_accepted_provenance_rejects_config_or_scientific_source_change(isolated):
    record=run._provenance('map')
    run._validate_provenance(record, 'map')
    for key in ('config_hash','source_hashes'):
        altered={**record, key: 'different'}
        with pytest.raises(ValueError):run._validate_provenance(altered,'map')
    with pytest.raises(ValueError):run._validate_provenance({},'map')


def test_atomic_writer_preserves_previous_record_after_serialization_error(isolated):
    file=isolated/'record.json';run.atomic(file, {'old':1})
    with pytest.raises(ValueError):run.atomic(file, {'nonfinite':float('nan')})
    assert json.loads(file.read_text())=={'old':1}
    assert not list(isolated.glob('*.tmp'))


def test_lock_blocks_duplicate_worker_and_releases_after_exception(isolated):
    with run.unit_lock('same-unit'):
        with pytest.raises(RuntimeError,match='locked'):
            with run.unit_lock('same-unit'):pass
    with run.unit_lock('same-unit'):pass


def test_unaccepted_partial_map_is_quarantined_before_identical_retry(isolated, monkeypatch):
    from experiments.pcrl_task_directed_release_v1 import mechanisms
    base=run.anchor_dir(0)/'maps'/'T0_C_0.002_a17';base.mkdir(parents=True)
    (base/'partial.txt').write_text('preserve this failed attempt')
    monkeypatch.setattr(run, '_prepared_receipt', lambda anchor: {'cache_sha256':'p'})
    p={'ctx':{'pools':{}},'encoder':None,'encoded':{},'tables':{}}
    def fit(ctx, encoder, encoded, tables, spec, out, **kw):
        assert not Path(out).exists()
        Path(out).mkdir(parents=True)
        return {'feasible':True,'Q':np.eye(2),'input':'T0'}
    monkeypatch.setattr(mechanisms,'fit_map',fit)
    result=run.ensure_map(0,'T0_C_0.002_a17',p)
    assert result['feasible']
    assert list((isolated/'private/quarantine').rglob('partial.txt'))
    monkeypatch.setattr(mechanisms,'fit_map',lambda *a,**k:pytest.fail('accepted map refitted'))
    assert run.ensure_map(0,'T0_C_0.002_a17',p)['feasible']
    marker=base/'ACCEPTED.json';record=json.loads(marker.read_text());record['source_hashes']={}
    run.atomic(marker,record)
    with pytest.raises(ValueError):run.ensure_map(0,'T0_C_0.002_a17',p)


def test_infeasible_solution_is_never_accepted(isolated, monkeypatch):
    from experiments.pcrl_task_directed_release_v1 import mechanisms
    monkeypatch.setattr(run,'_prepared_receipt',lambda anchor:{'cache_sha256':'p'})
    p={'ctx':{'pools':{}},'encoder':None,'encoded':{},'tables':{}}
    def fit(*a,**k):
        Path(a[5]).mkdir(parents=True)
        return {'feasible':False,'Q':None,'status':'failed'}
    monkeypatch.setattr(mechanisms,'fit_map',fit)
    with pytest.raises(ValueError,match='feasible'):run.ensure_map(0,'T0_C_0.002_a17',p)
    assert not (run.anchor_dir(0)/'maps/T0_C_0.002_a17/ACCEPTED.json').exists()


def test_benchmark_failure_returns_only_resource_and_integrity_record(isolated, monkeypatch):
    def fail(anchor):
        print('sensitive diagnostic must stay private')
        raise ValueError('comparative CE must not escape')
    monkeypatch.setattr(run,'prepare',fail)
    result=run.benchmark()
    assert result['status']=='failed' and result['error_type']=='ValueError'
    public=json.dumps(result)
    assert 'comparative CE' not in public and 'sensitive diagnostic' not in public
    assert (isolated/'BENCHMARK.json').exists()


def test_source_dependencies_ignore_document_and_report_edits(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'ROOT',tmp_path)
    base=tmp_path/'experiments'/run.STUDY;base.mkdir(parents=True)
    for name in ('run','config','data','encoding','audits','evaluation','baselines','finite','mechanisms'):
        (base/(name+'.py')).write_text(name)
    initial=run.source_fingerprint('prepare')
    (base/'report.py').write_text('new prose')
    (base/'PROTOCOL.md').write_text('clarification')
    assert run.source_fingerprint('prepare')==initial
    (base/'run.py').write_text('new orchestration helper')
    assert run.source_fingerprint('prepare')==initial


def test_successful_frozen_evaluation_pins_summary_and_private_artifacts(isolated,monkeypatch):
    registry=run.anchor_dir(0)/'audits/H/registry.joblib';registry.parent.mkdir(parents=True)
    registry.write_bytes(b'synthetic frozen registry')
    receipt=registry.parent/'COMPLETE.json';receipt.write_text('{}')
    run.atomic(isolated/'SELECTION.json',{'selection_frozen':True,'config_hash':run.digest(run.configuration()),
        'evaluation_configurations':['H'],'frozen_audits':{'H':{'0':{
            'registry_sha256':run.sha(registry),'receipt_sha256':run.sha(receipt)}}}})
    monkeypatch.setattr(run,'_audit_receipt',lambda *a:{'registry_sha256':run.sha(registry)})
    encoder=SimpleNamespace(encode=lambda inputs:{'p':np.full(len(inputs.x_a),.5)})
    def prep(anchor,*,allow_fit):
        assert allow_fit is False
        return {'encoder':encoder,'erasers':{}}
    monkeypatch.setattr(run,'prepare',prep)
    loads=[]
    def load(anchor,*,pools,evaluation_permit):
        assert pools==('test',);loads.append('test')
        return {'anchor':anchor,'pools':{'test':{'x':np.zeros((2,32)),'ha':np.zeros((2,4))}}}
    monkeypatch.setattr(run,'load_anchor',load)
    def release(name,p,anchor,*,allow_fit):
        assert not allow_fit and set(p['ctx']['pools'])=={'test'}
        return {'test':{}}
    monkeypatch.setattr(run,'release_for',release)
    def evaluate(reg,ctx,release,out):
        out.mkdir(parents=True);(out/'test').mkdir()
        file=out/'test/role.npz';file.write_bytes(b'synthetic private result')
        summary={'test':{'role':{'synthetic':True}}};run.atomic(out/'summary.json',summary)
        return {'summary':summary,'artifacts':{'test':{'role':str(file)}}}
    monkeypatch.setattr(run,'evaluate_frozen_audits',evaluate)
    first=run.evaluate_unit(0,'H')
    assert run.evaluate_unit(0,'H')==first and loads==['test']
    (run.anchor_dir(0)/'evaluation/H/test/role.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='artifact hash'):run.evaluate_unit(0,'H')
    assert loads==['test']


def test_resealed_audit_cannot_open_test_after_selection(isolated,monkeypatch):
    base=run.anchor_dir(0)/'audits/H';base.mkdir(parents=True)
    (base/'COMPLETE.json').write_text('{"changed":true}')
    run.atomic(isolated/'SELECTION.json',{'selection_frozen':True,
        'evaluation_configurations':['H'],'frozen_audits':{'H':{'0':{
            'registry_sha256':'current-registry','receipt_sha256':'original-receipt'}}}})
    monkeypatch.setattr(run,'_audit_receipt',lambda *a:{'registry_sha256':'current-registry'})
    monkeypatch.setattr(run,'load_anchor',lambda *a,**k:pytest.fail('test opened after reseal'))
    with pytest.raises(ValueError,match='changed after selection'):
        run.evaluate_unit(0,'H')


def test_accepted_immutable_dependencies_do_not_take_mutation_lock(tmp_path,monkeypatch):
    from contextlib import contextmanager
    monkeypatch.setattr(run,'OUT',tmp_path)
    base=run.anchor_dir(0);base.mkdir(parents=True)
    (base/'PREPARED.json').write_text('{}')
    (base/'maps'/'q').mkdir(parents=True);(base/'maps'/'q'/'ACCEPTED.json').write_text('{}')
    (base/'audits'/'H').mkdir(parents=True);(base/'audits'/'H'/'COMPLETE.json').write_text('{}')
    @contextmanager
    def denied(key):
        raise AssertionError('An immutable read must not contend for a mutation lock')
        yield
    monkeypatch.setattr(run,'unit_lock',denied)
    monkeypatch.setattr(run,'_prepare',lambda a,allow_fit:('prepared',allow_fit))
    monkeypatch.setattr(run,'_load_map',lambda a,n:('map',n))
    monkeypatch.setattr(run,'_load_audit',lambda a,n:('audit',n))
    assert run.prepare(0)==('prepared',False)
    assert run.ensure_map(0,'q')==('map','q')
    assert run.audit_unit(0,'H')==('audit','H')
