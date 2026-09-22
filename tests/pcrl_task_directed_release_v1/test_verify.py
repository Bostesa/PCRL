"""Frozen-artifact orchestration tests; no ACS inputs or live outcomes."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil

import pytest
import joblib
import numpy as np
from test_replay import artifacts

from experiments.pcrl_task_directed_release_v1 import config, reporting, run, selection


def module():
    return importlib.import_module('experiments.pcrl_task_directed_release_v1.verify')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def summaries(u, s):
    return {role: {'selection': 'prior', 'independent_selection': 'prior',
                  'validation': {'unweighted': u if role.startswith('utility:') else s,
                                 'weighted': u if role.startswith('utility:') else s,
                                 'balanced': u if role.startswith('utility:') else s}}
            for role in selection.REQUIRED_ROLES}


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path/'archive'; out = root/'results'/config.STUDY
    sources = set(run.source_fingerprint()) | set(reporting.inference_source_fingerprint())
    for name in sources:
        destination = root/'experiments'/config.STUDY/(name+'.py')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(selection.__file__).with_name(name+'.py'), destination)
    validation = {name: {a: summaries(u, s) for a in (0, 1, 2)}
                  for name, u, s in [('H', .6, .6), ('J', .5, .5),
                     ('Trisk_C_0.002_a17', .49, .501), ('continuous_task', .495, .501)]}
    pins = {}; grid_receipts = []
    for name, anchors in validation.items():
        pins[name] = {}
        for anchor, summary in anchors.items():
            base = out/'private/run'/f'anchor_{anchor}'/'audits'/name
            write(base/'summary.json', summary)
            (base/'registry.joblib').write_bytes(b'opaque synthetic registry receipt fixture')
            receipt = {**run._provenance('audit'), 'anchor': anchor, 'configuration': name,
                'registry_sha256': sha(base/'registry.joblib'),
                'artifact_hashes': {f: sha(base/f) for f in ('summary.json', 'registry.joblib')}}
            write(base/'COMPLETE.json', receipt)
            pins[name][str(anchor)] = {'registry_sha256': receipt['registry_sha256'],
                                       'receipt_sha256': sha(base/'COMPLETE.json')}
            grid_receipts.append({'anchor': anchor, 'configuration': name,
                'receipt_sha256': sha(base/'COMPLETE.json'), 'summary_sha256': sha(base/'summary.json')})
    write(out/'RESOURCE_SCHEDULE.json', {'config_hash': config.digest(config.configuration()),
                                       'frozen_before_comparative_outcomes': True})
    write(out/'VALIDATION_GRID.json', {'records': validation, 'receipts': grid_receipts,
                                     'evaluation_opened': False})
    chosen = selection.select_validation(validation, candidate_ids=['Trisk_C_0.002_a17'])
    chosen.update(frozen_audits=pins, config_hash=config.digest(config.configuration()),
                  source_hashes=run.source_fingerprint(),
                  inference_source_hashes=reporting.inference_source_fingerprint())
    selection.freeze_selection(chosen, out_dir=out)
    return root, out


def test_rebuilds_selection_and_exact_explicit_family_from_saved_summaries(frozen):
    root, out = frozen
    result = module().verify_frozen_selection(artifact_root=root, original_root='/unavailable/original')
    assert result['report']['passed']
    assert result['report']['complete_configurations'] == 4
    assert result['report']['validation_units'] == 12
    assert result['report']['family_size'] == len(result['contrasts']['endpoints'])
    assert result['report']['selection_recomputed_from_saved_summaries']
    assert result['selection']['routes']['utility_first']['nominee'] == 'Trisk_C_0.002_a17'


@pytest.mark.parametrize('kind', ['source', 'summary', 'nominee', 'contrast_count', 'pin', 'diagnostics'])
def test_tampered_provenance_summaries_or_selection_fail_before_model_loading(frozen, kind):
    root, out = frozen
    if kind == 'source':
        with (root/'experiments'/config.STUDY/'audits.py').open('a') as f:
            f.write('\n# changed archive scientific source\n')
    elif kind == 'summary':
        path = out/'private/run/anchor_0/audits/J/summary.json'
        value = json.loads(path.read_text()); value[selection.TASK_ROLE]['validation']['weighted'] = .9
        write(path, value)
    else:
        path = out/('CONTRASTS.json' if kind == 'contrast_count' else 'SELECTION.json')
        value = json.loads(path.read_text())
        if kind == 'nominee':
            value['routes']['utility_first']['nominee'] = 'J'
        elif kind == 'contrast_count':
            value['family_size'] += 1
        elif kind == 'diagnostics':
            value['diagnostic_claim_formulas'] = {}
        else:
            value['frozen_audits']['J']['0']['registry_sha256'] = 'wrong'
        write(path, value)
    with pytest.raises(ValueError):
        module().verify_frozen_selection(artifact_root=root)


def test_archive_owned_paths_reject_symlink_escape_with_no_original_fallback(frozen, tmp_path):
    root, out = frozen
    path = out/'VALIDATION_GRID.json'
    external = tmp_path/'external.json'; shutil.move(path, external); path.symlink_to(external)
    with pytest.raises(ValueError, match='root|escape|owned'):
        module().verify_frozen_selection(artifact_root=root, original_root='/unused/source')


def test_resource_gate_fails_before_any_validation_grid_read(frozen):
    root, out = frozen
    write(out/'RESOURCE_SCHEDULE.json', {'frozen_before_comparative_outcomes': False})
    (out/'VALIDATION_GRID.json').unlink()
    with pytest.raises(PermissionError):
        module().verify_frozen_selection(artifact_root=root)


def test_replay_unit_reloads_weights_and_runs_wire_and_ancestor_checks(artifacts):
    registry, rows, release, root = artifacts
    role = registry['roles']['attack:A/SEX']
    role['candidates']['H__prior'] = role['candidates'].pop('prior')
    role['validation_scores']['H__prior'] = role['validation_scores'].pop('prior')
    with np.load(role['validation_loss_path']) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays['candidate_ids'] = np.array(['model', 'H__prior'])
    np.savez_compressed(role['validation_loss_path'], **arrays)
    baseline = copy.deepcopy(registry); baseline['is_H_release'] = True; baseline['name'] = 'H'
    hrole = baseline['roles']['attack:A/SEX']
    hrole['candidates'] = {'prior': hrole['candidates']['H__prior']}
    hrole['selection'] = hrole['independent_selection'] = 'prior'
    hrelease = {'aux': None, 'token_probs': np.ones((4, 1))}
    ctx = {'anchor': 0, 'pools': {'attacker_validation': rows}}
    result = module().verify_replay_unit(registry=registry, h_registry=baseline, ctx=ctx,
        release={'attacker_validation': release}, h_release={'attacker_validation': hrelease},
        artifact_root=root, out_dir=root/'unit_verify',
        evaluation={'ctx': {'anchor': 0, 'pools': {'test': rows}}, 'release': {'test': release},
                    'directory': root/'evaluation'}, wire_repetitions=512)
    assert result['passed']
    assert result['validation']['roles']['attack:A/SEX']['selection_verified_on_all_saved_validation_rows']
    assert result['ancestor_parity']['passed']
    row = result['evaluation']['pools']['test']['attack:A/SEX']
    assert row['wire_check']['passed'] and row['original_households'] == 3


def test_registered_but_unfinished_branch_keeps_all_attempted_controls(frozen):
    from experiments.pcrl_task_directed_release_v1 import branches
    root, out = frozen
    record = {'primary_config_hash': config.digest(config.configuration()),
        'scientific_source_hashes': run.source_fingerprint(), 'registered': True,
        'branch_source_sha256': sha(branches.__file__), 'trigger': {'triggered': True},
        'maps': branches.action33_specs(), 'controls': branches.action33_controls()}
    record['registration_payload_hash'] = config.digest(record)
    write(out/'EXTRA_CONFIGS.json', record)
    shutil.copy(branches.__file__, root/'experiments'/config.STUDY/'branches.py')
    file_hash = sha(out/'EXTRA_CONFIGS.json')
    ext = {s['configuration']: {**s, 'extra_registration_sha256': file_hash,
             'branch_source_sha256': record['branch_source_sha256']} for s in record['maps'] if s['anchor'] == 0}
    controls = {s['configuration']: {**s, 'extra_registration_sha256': file_hash,
             'branch_source_sha256': record['branch_source_sha256']} for s in record['controls'] if s['anchor'] == 0}
    old = json.loads((out/'SELECTION.json').read_text())
    grid = json.loads((out/'VALIDATION_GRID.json').read_text())
    chosen = selection.select_validation(grid['records'], candidate_ids=old['candidate_ids'],
        registered_extensions=ext, registered_controls=controls)
    for key in ('frozen_audits', 'config_hash', 'source_hashes', 'inference_source_hashes'):
        chosen[key] = old[key]
    (out/'SELECTION.json').unlink(); (out/'CONTRASTS.json').unlink()
    selection.freeze_selection(chosen, out_dir=out)
    result = module().verify_frozen_selection(artifact_root=root)
    family = result['selection']['routes']['utility_first']['family_nominees']['withholding']
    assert len(family['attempted']) == 18 and len(family['unavailable']) == 18
    assert result['report']['registrations']['EXTRA_CONFIGS.json']['registered_configurations_used'] == 41


def test_verify_study_records_gate_failure_without_opening_prepared_or_evaluation(frozen):
    root, out = frozen
    (out/'CONTRASTS.json').unlink()
    report = module().verify_study(artifact_root=root, out_dir=out/'private/verify-failure', include_evaluation=True)
    assert not report['passed'] and report['failed_stage'] == 'selection_and_provenance'
    assert not report['evaluation_opened']
    assert (out/'private/verify-failure/verification.json').is_file()


def test_math_channel_replays_empirical_tables_and_detects_changed_objective():
    from test_math_replay import prepared
    from experiments.pcrl_task_directed_release_v1 import math_replay
    p = prepared(); table = math_replay.reconstruct_tables(p, 'T0', 17)
    q = np.zeros_like(table['cost']); q[:, 0] = 1.
    spec = {'input': 'T0', 'max_actions': 17, 'policy': 'U', 'budget': None}
    result = {'Q': q, 'input': 'T0', 'actions': 17, 'policy': 'U', 'budget': None,
              'constrained_roles': [], 'status': 'optimal', 'optimal': True, 'feasible': True,
              'objective': float(table['cost'][:, 0].sum())}
    good = module().verify_math_channel(p, spec, result, table)
    assert good['passed'] and good['table_reconstruction']['passed']
    bad = module().verify_math_channel(p, spec, {**result, 'objective': result['objective']+.1}, table)
    assert not bad['passed'] and 'objective' in bad['violations']


def test_cli_emits_nonzero_and_private_failed_report_on_invalid_frozen_archive(frozen, capsys):
    root, out = frozen
    (out/'CONTRASTS.json').unlink()
    code = module().main(['--artifact-root', str(root), '--original-root', '/unused/root',
                          '--out-dir', str(out/'private/cli')])
    assert code == 1
    message = json.loads(capsys.readouterr().out)
    assert not message['passed'] and not message['evaluation_opened']


def test_missing_restored_registry_never_reads_existing_original_copy(frozen, tmp_path):
    root, out = frozen
    original = tmp_path/'original'
    relative = Path('results')/config.STUDY/'private/run/anchor_0/audits/H/registry.joblib'
    (original/relative).parent.mkdir(parents=True)
    shutil.copy(root/relative, original/relative)
    (root/relative).unlink()
    with pytest.raises(ValueError, match='missing'):
        module().verify_frozen_selection(artifact_root=root, original_root=original)


class SyntheticEncoder:
    def encode(self, data):
        n = len(data.h_a)
        return {'p': np.full(n, .6), 'b': np.full(n, .6), 'r': np.zeros(n),
                'risk': np.zeros((n, 2)), 'actions': {17: np.full((n, 1), .6)},
                'global_offsets': np.full((n, 1), .6),
                'codes': {name: np.zeros(n, dtype=int) for name in config.INPUTS}}


def test_full_validation_command_replays_real_synthetic_saved_models_without_refitting(tmp_path, monkeypatch):
    from test_evaluation import tiny_context
    from experiments.pcrl_task_directed_release_v1 import audits, data, evaluation, mechanisms
    root = tmp_path/'root'; out = root/'results'/config.STUDY
    for name in set(run.source_fingerprint()) | set(reporting.inference_source_fingerprint()):
        destination = root/'experiments'/config.STUDY/(name+'.py')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(selection.__file__).with_name(name+'.py'), destination)

    def prior_fit(h, p, y, w, hv, pv, yv, wv, n_classes, seed, out_dir, **kwargs):
        probability = np.full(n_classes, .4/(n_classes-1)); probability[0] = .6
        model = audits.TokenCandidate('prior', probability, np.zeros(h.shape[1]), np.ones(h.shape[1]),
            p.shape[1], n_classes, False, {'class_order': list(range(n_classes)), 'probability_floor': 1e-9,
                                        'input_dim': h.shape[1]})
        model.save(Path(out_dir)/'prior')
        return {'candidates': {'prior': model}}

    monkeypatch.setattr(evaluation, 'fit_slate', prior_fit)
    validation = {'H': {}, 'J': {}}; pins = {'H': {}, 'J': {}}; grid_receipts = []
    for anchor in (0, 1, 2):
        ctx = tiny_context(); ctx['anchor'] = anchor; ctx['pools'].pop('test')
        for rows in ctx['pools'].values():
            rows.update(x=np.zeros((6, 32)), J=np.zeros((6, 1)))
        encoder = SyntheticEncoder()
        encoded = {p: encoder.encode(data.RuntimeInputs(d['x'], d['ha'])) for p, d in ctx['pools'].items()}
        base = out/'private/run'/f'anchor_{anchor}'; (base/'encoder').mkdir(parents=True)
        joblib.dump(encoder, base/'encoder/encoder.joblib')
        joblib.dump({'ctx': ctx, 'encoder': encoder, 'encoded': encoded, 'erasers': {}, 'roles': {}}, base/'prepared.joblib')
        write(base/'PREPARED.json', {**run._provenance('prepare'), 'anchor': anchor,
            'cache_sha256': sha(base/'prepared.joblib'), 'artifact_hashes': {
                'prepared.joblib': sha(base/'prepared.joblib'), 'encoder/encoder.joblib': sha(base/'encoder/encoder.joblib')}})
        original = out/'private/inputs/results/redesign_20260909_acs_fixed_predictions_v1'/f'seed_{anchor}'
        original.mkdir(parents=True)
        np.savez_compressed(original/'anchors.npz', **{p+'/'+role: d[key] for p, d in ctx['pools'].items()
                            for role, key in (('A', 'ha'), ('B', 'hb'))})
        baseline = None
        for name in ('H', 'J'):
            release = mechanisms.build_release(ctx, encoder, encoded, name)
            path = base/'audits'/name
            fitted = evaluation.fit_release_audits(name, ctx, release, path, h_baseline=baseline, b_baseline=baseline)
            if name == 'H':
                baseline = fitted
            write(path/'COMPLETE.json', {**run._provenance('audit'), 'anchor': anchor, 'configuration': name,
                'registry_sha256': sha(path/'registry.joblib'), 'prepared_cache_sha256': sha(base/'prepared.joblib'),
                'H_registry_sha256': None if name == 'H' else sha(base/'audits/H/registry.joblib'),
                'artifact_hashes': {str(p.relative_to(path)): sha(p) for p in path.rglob('*') if p.is_file()}})
            validation[name][anchor] = fitted['summary']
            pins[name][str(anchor)] = {'registry_sha256': sha(path/'registry.joblib'), 'receipt_sha256': sha(path/'COMPLETE.json')}
            grid_receipts.append({'anchor': anchor, 'configuration': name,
                'receipt_sha256': sha(path/'COMPLETE.json'), 'summary_sha256': sha(path/'summary.json')})
    write(out/'RESOURCE_SCHEDULE.json', {'config_hash': config.digest(config.configuration()), 'frozen_before_comparative_outcomes': True})
    write(out/'VALIDATION_GRID.json', {'records': validation, 'receipts': grid_receipts, 'evaluation_opened': False})
    chosen = selection.select_validation(validation, candidate_ids=[])
    chosen.update(frozen_audits=pins, config_hash=config.digest(config.configuration()),
                  source_hashes=run.source_fingerprint(), inference_source_hashes=reporting.inference_source_fingerprint())
    selection.freeze_selection(chosen, out_dir=out)
    monkeypatch.setattr(evaluation, 'fit_slate', lambda *a, **k: pytest.fail('Verification refit'))
    monkeypatch.setattr(run, 'prepare', lambda *a, **k: pytest.fail('Global preparation fallback'))
    monkeypatch.setattr(data, 'load_anchor', lambda *a, **k: pytest.fail('Validation-only verification accessed test/data loader'))
    report = module().verify_study(artifact_root=root, original_root='/unused/original', out_dir=out/'private/check')
    assert report['passed'], (out/'private/check/incident.json').read_text() if not report['passed'] else ''
    assert len(report['units']) == 6 and not report['evaluation_opened']
    assert all(u['validation_roles'] == 16 for u in report['units'].values())
    # Material evaluation outputs are generated once from the frozen synthetic
    # models, then the command must independently reconstruct every selected role.
    evaluation_contexts = {}
    for anchor in (0, 1, 2):
        rows = tiny_context()['pools']['test']; rows.update(x=np.zeros((6, 32)), J=np.zeros((6, 1)))
        ctx = {'anchor': anchor, 'pools': {'test': rows}}; evaluation_contexts[anchor] = ctx
        encoder = SyntheticEncoder(); encoded = {'test': encoder.encode(data.RuntimeInputs(rows['x'], rows['ha']))}
        source = out/'private/inputs/results/redesign_20260909_acs_fixed_predictions_v1'/f'seed_{anchor}/anchors.npz'
        with np.load(source) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        arrays.update({'test/A': rows['ha'], 'test/B': rows['hb']}); np.savez_compressed(source, **arrays)
        for name in ('H', 'J'):
            base = out/'private/run'/f'anchor_{anchor}'
            path = base/'evaluation'/name; registry = base/'audits'/name/'registry.joblib'
            evaluation.evaluate_frozen_audits(registry, ctx, mechanisms.build_release(ctx, encoder, encoded, name), path)
            write(path/'COMPLETE.json', {**run._provenance('evaluation'), 'anchor': anchor, 'configuration': name,
                'selection_sha256': sha(out/'SELECTION.json'), 'registry_sha256': sha(registry),
                'artifact_hashes': {str(p.relative_to(path)): sha(p) for p in path.rglob('*') if p.is_file()}})

    def frozen_synthetic_input(anchor, pools, *, inputs_root, evaluation_permit):
        assert pools == ('test',) and inputs_root == out/'private/inputs'
        assert evaluation_permit == out/'SELECTION.json'
        return evaluation_contexts[anchor]

    monkeypatch.setattr(data, 'load_anchor', frozen_synthetic_input)
    evaluated = module().verify_study(artifact_root=root, original_root='/unused/original',
                                     out_dir=out/'private/check-evaluation', include_evaluation=True)
    assert evaluated['passed'], (out/'private/check-evaluation/incident.json').read_text() if not evaluated['passed'] else ''
    assert evaluated['evaluation_opened'] and len(evaluated['units']) == 6
    assert all(len(u['evaluation_roles']['test']) == 16 for u in evaluated['units'].values())
    # Representative archive restore contains only anchor0, and the original
    # anchor0 is deleted to make any unremapped fallback fail visibly.
    restored = tmp_path/'restored'; shutil.copytree(root, restored)
    restored_out = restored/'results'/config.STUDY
    for anchor in (1, 2):
        shutil.rmtree(restored_out/'private/run'/f'anchor_{anchor}')
    shutil.rmtree(out/'private/run/anchor_0')

    def restored_synthetic_input(anchor, pools, *, inputs_root, evaluation_permit):
        assert anchor == 0 and pools == ('test',)
        assert inputs_root == restored_out/'private/inputs'
        assert evaluation_permit == restored_out/'SELECTION.json'
        return evaluation_contexts[anchor]

    monkeypatch.setattr(data, 'load_anchor', restored_synthetic_input)
    restored_report = module().verify_restored_unit('J', anchor=0, artifact_root=restored, original_root=root,
        out_dir=restored_out/'private/representative', include_evaluation=True,
        expected_selection_sha256=sha(out/'SELECTION.json'))
    assert restored_report['passed'] and restored_report['anchor'] == 0
    assert restored_report['scope'] == 'representative single-anchor restore; full selection reconstruction is separate'
    assert restored_report['selection_recomputed'] is False
