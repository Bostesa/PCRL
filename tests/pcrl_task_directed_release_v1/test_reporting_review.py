"""Independent integrity regressions, using only synthetic files and losses."""
import copy
import json

import numpy as np
import pytest

from experiments.pcrl_task_directed_release_v1 import reporting, run
from experiments.pcrl_task_directed_release_v1.config import configuration, digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def fixture_rows():
    return {'ids': np.array(['p1', 'p2']), 'households': np.array(['h1', 'h2']),
            'weights': np.array([1., 2.]), 'y': np.array([0, 1]), 'loss': np.array([.2, .4])}


def evaluation_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, 'OUT', tmp_path)
    role = 'attack:A/SEX'
    audit = tmp_path / 'private/run/anchor_0/audits/Q'
    audit.mkdir(parents=True)
    registry = audit / 'registry.joblib'; registry.write_bytes(b'synthetic frozen registry')
    audit_record = {**run._provenance('audit'), 'anchor': 0, 'configuration': 'Q',
                    'registry_sha256': reporting.sha(registry),
                    'artifact_hashes': {'registry.joblib': reporting.sha(registry)}}
    write(audit / 'COMPLETE.json', audit_record)
    selection = {'selection_frozen': True, 'configuration_digest': digest(configuration()),
                 'config_hash': digest(configuration()), 'source_hashes': run.source_fingerprint(),
                 'inference_source_hashes': reporting.inference_source_fingerprint()
                    if hasattr(reporting, 'inference_source_fingerprint') else {},
                 'evaluation_configurations': ['Q'],
                 'frozen_audits': {'Q': {'0': {'registry_sha256': reporting.sha(registry),
                                              'receipt_sha256': reporting.sha(audit / 'COMPLETE.json')}}}}
    write(tmp_path / 'SELECTION.json', selection)
    base = tmp_path / 'private/run/anchor_0/evaluation/Q'
    path = base / 'test' / 'attack__A__SEX.npz'; path.parent.mkdir(parents=True)
    np.savez_compressed(path, **fixture_rows())
    marker = {**run._provenance('evaluation'), 'anchor': 0, 'configuration': 'Q',
              'selection_sha256': reporting.sha(tmp_path / 'SELECTION.json'),
              'registry_sha256': reporting.sha(registry),
              'artifact_hashes': {'test/attack__A__SEX.npz': reporting.sha(path)}}
    write(base / 'COMPLETE.json', marker)
    return role, base, audit, selection


@pytest.mark.parametrize('mutation', ['selection', 'registry', 'sources', 'configuration'])
def test_evaluation_receipt_must_match_frozen_selection_and_provenance(tmp_path, monkeypatch, mutation):
    role, base, audit, selected = evaluation_fixture(tmp_path, monkeypatch)
    marker = json.loads((base / 'COMPLETE.json').read_text())
    if mutation == 'selection': marker['selection_sha256'] = 'stale'
    if mutation == 'registry': marker['registry_sha256'] = 'wrong'
    if mutation == 'sources': marker['source_hashes'] = {'wrong': 'source'}
    if mutation == 'configuration': marker['config_hash'] = 'wrong'
    write(base / 'COMPLETE.json', marker)
    with pytest.raises(ValueError):
        reporting.frozen_loss_loader()(0, 'Q', role)


def test_changed_audit_registry_receipt_cannot_be_resealed_after_selection(tmp_path, monkeypatch):
    role, base, audit, selected = evaluation_fixture(tmp_path, monkeypatch)
    registry = audit / 'registry.joblib'; registry.write_bytes(b'new model with same recipe ID')
    marker = json.loads((audit / 'COMPLETE.json').read_text())
    marker['registry_sha256'] = reporting.sha(registry)
    marker['artifact_hashes']['registry.joblib'] = reporting.sha(registry)
    write(audit / 'COMPLETE.json', marker)
    evaluation = json.loads((base / 'COMPLETE.json').read_text())
    evaluation['registry_sha256'] = reporting.sha(registry)
    write(base / 'COMPLETE.json', evaluation)
    with pytest.raises(ValueError):
        reporting.frozen_loss_loader()(0, 'Q', role)


def test_valid_frozen_rows_load_and_none_identity_is_rejected(tmp_path, monkeypatch):
    role, base, audit, selected = evaluation_fixture(tmp_path, monkeypatch)
    got = reporting.frozen_loss_loader()(0, 'Q', role)
    np.testing.assert_array_equal(got['loss'], fixture_rows()['loss'])
    broken = fixture_rows(); broken['households'] = np.array([None, 'h2'], dtype=object)
    with pytest.raises(ValueError, match='identity|identifier'):
        reporting._validate_rows(broken)


def test_paired_contrast_cannot_mix_targets_or_use_out_of_schema_labels():
    endpoint = {'id': 'schema', 'anchors': [0, 1, 2], 'weighting': 'unweighted',
                'terms': [{'configuration': 'Q', 'role': 'attack:A/SEX', 'coefficient': 1.},
                          {'configuration': 'J', 'role': 'attack:AB/RAC1P', 'coefficient': -1.}]}
    with pytest.raises(ValueError, match='target|class'):
        reporting.paired_endpoint(endpoint, lambda *args: fixture_rows())
    endpoint['terms'][1]['role'] = 'attack:AB/SEX'
    bad = fixture_rows(); bad['y'] = np.array([0, 2])
    with pytest.raises(ValueError, match='class|label'):
        reporting.paired_endpoint(endpoint, lambda *args: bad)


def test_all_main_and_attribution_formulas_use_adjusted_bounds():
    contrast = {'family_size': 1, 'endpoints': [{'id': 'e', 'checks': [
        {'name': 'NI', 'bound': 'upper', 'operator': '<=', 'threshold': .001},
        {'name': 'strict', 'bound': 'upper', 'operator': '<', 'threshold': 0.}]}],
        'claim_formulas': {'utility_first': {'competitive': {'endpoint': 'e', 'check': 'strict'}}},
        'attribution_claim_formulas': {'utility_first': {'risk_refinement': {'kind': 'all', 'clauses': [
            {'endpoint': 'e', 'check': 'NI'}, {'endpoint': 'e', 'check': 'strict'}]}}}}
    bounds = {'family_size': 1, 'bounds': {'e': {'estimate': -.02, 'lower': -.04, 'upper': .0005}}}
    result = reporting.evaluate_claims(contrast, bounds)
    assert result['checks']['e']['NI'] and not result['checks']['e']['strict']
    assert not result['claim_results']['utility_first']['competitive']['passed']
    assert not result['attribution_claim_results']['utility_first']['risk_refinement']['passed']


def test_bind_provenance_checks_all_three_validation_summaries_and_pins_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(reporting, 'OUT', tmp_path)
    role = 'attack:A/SEX'; choices = {}
    for anchor in (0,1,2):
        base = tmp_path / 'private/run' / f'anchor_{anchor}' / 'audits/Q'
        base.mkdir(parents=True)
        (base / 'registry.joblib').write_bytes(f'opaque synthetic registry {anchor}'.encode())
        summary = {role: {'selection': 'frozen-recipe', 'independent_selection': 'independent-recipe',
                          'validation': {'unweighted': .5, 'weighted': .75}}}
        write(base / 'summary.json', summary)
        receipt = {**run._provenance('audit'), 'anchor': anchor, 'configuration': 'Q',
                   'registry_sha256': reporting.sha(base / 'registry.joblib'),
                   'artifact_hashes': {n: reporting.sha(base / n) for n in ('registry.joblib','summary.json')}}
        write(base / 'COMPLETE.json', receipt)
        choices[str(anchor)] = {role: {'selection': 'frozen-recipe', 'independent_selection': 'independent-recipe'}}
    selection = {'validation_only': True, 'configuration_digest': digest(configuration()),
                 'descriptive_configurations': ['Q'], 'predictor_choices': {'Q': choices},
                 'aggregates': {'Q': {'roles': {role: {'unweighted': .5, 'PWGTP': .75}}}}}
    original = copy.deepcopy(selection)
    pinned = reporting.bind_frozen_provenance(selection)
    assert selection == original and set(pinned['frozen_audits']['Q']) == {'0','1','2'}
    assert pinned['inference_source_hashes'] == reporting.inference_source_fingerprint()
    selection['aggregates']['Q']['roles'][role]['PWGTP'] = .74
    with pytest.raises(ValueError, match='means changed'):
        reporting.bind_frozen_provenance(selection)
    selection = original
    selection['predictor_choices']['Q']['2'][role]['selection'] = 'new-recipe'
    with pytest.raises(ValueError, match='predictor differs'):
        reporting.bind_frozen_provenance(selection)


@pytest.mark.parametrize('field', ['configuration_digest','config_hash','source_hashes','inference_source_hashes'])
def test_frozen_selection_rejects_changed_config_or_source_before_losses(tmp_path,monkeypatch,field):
    role, base, audit, selection = evaluation_fixture(tmp_path,monkeypatch)
    selection[field] = 'stale'
    write(tmp_path / 'SELECTION.json', selection)
    with pytest.raises(ValueError): reporting.frozen_loss_loader()
