"""Independent frozen-artifact verification, including owned archive restores.

This module never calls a fitting, repair, ensure-map, or evaluation-writing API.
The explicit artifact root is the only source of data/artifacts. Source closure
must match both this running checkout and the restored source files. Pickled
artifacts are loaded only after their accepted manifests and selection pins pass.
Reports describe executed coverage; validation-only runs never claim test replay.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import re
import time

import joblib
import numpy as np

from . import math_replay, replay, reporting, run, selection
from .config import STUDY, configuration, digest, mechanism_id


class VerificationError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise VerificationError(message)


def _sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def _name(value):
    _require(isinstance(value, str) and re.fullmatch('[A-Za-z0-9_.-]+', value),
             'Invalid artifact configuration name')
    return value


class _Layout:
    def __init__(self, artifact_root, original_root=None):
        self.root = Path(artifact_root).resolve()
        _require(self.root.is_dir(), 'Explicit artifact_root must be an existing directory')
        self.original = Path(original_root).resolve() if original_root is not None else self.root
        self.out = self.root/'results'/STUDY

    def owned(self, path, *, directory=False):
        path = Path(path).resolve()
        _require(path.is_relative_to(self.root), 'Artifact path escapes the owned artifact_root')
        _require(path.is_dir() if directory else path.is_file(), 'Required owned artifact is missing')
        return path

    def file(self, *parts):
        return self.owned(self.out.joinpath(*map(str, parts)))

    def anchor(self, anchor):
        _require(anchor in (0, 1, 2) and not isinstance(anchor, bool), 'Unknown registered anchor')
        return self.out/'private/run'/f'anchor_{anchor}'

    def manifest(self, path, *, stage=None, anchor=None, name=None):
        path = self.owned(path); record = _json(path)
        if stage is not None:
            # This checks source/config metadata only; all artifact I/O stays here.
            run._validate_provenance(record, stage)
        if anchor is not None:
            _require(record.get('anchor') == anchor, 'Accepted artifact anchor changed')
        if name is not None:
            _require(record.get('configuration') == name, 'Accepted configuration changed')
        hashes = record.get('artifact_hashes')
        _require(isinstance(hashes, dict) and bool(hashes), 'Accepted artifact manifest missing')
        for relative, expected in hashes.items():
            child = self.owned(path.parent/relative)
            _require(child.is_relative_to(path.parent), 'Manifest member escapes its accepted unit')
            _require(_sha(child) == expected, 'Accepted artifact hash mismatch')
        return record


def _source_closure(layout, frozen):
    expected = {'source_hashes': run.source_fingerprint(),
                'inference_source_hashes': reporting.inference_source_fingerprint()}
    for field, hashes in expected.items():
        _require(frozen.get(field) == hashes, 'Frozen running scientific/inference source closure changed')
        for name, value in hashes.items():
            path = layout.owned(layout.root/'experiments'/STUDY/(name+'.py'))
            _require(_sha(path) == value, 'Restored scientific/inference source closure changed')
    return expected


def _registrations(layout, frozen):
    """Validate supplied extension inputs against prospectively frozen registries."""
    primary = {m['configuration'] for m in configuration()['maps']}
    extensions = {n: copy.deepcopy(s) for n, s in frozen['mechanism_specs'].items() if n not in primary}
    controls = copy.deepcopy(frozen.get('registered_controls', {}))
    groups = {}
    for filename, module_name, hash_key, branch, source_key in (
            ('EXTRA_CONFIGS.json', 'branches', 'extra_registration_sha256', 'A', 'branch_source_sha256'),
            ('ROBUSTNESS_CONFIGS.json', 'robustness', 'robustness_registration_sha256', 'C', 'robustness_source_sha256'),
            ('BASELINE_SUPPLEMENTS.json', 'baseline_supplement', 'baseline_registration_sha256',
             'baseline_supplement', 'baseline_source_sha256')):
        path = layout.out/filename
        wanted = {n: s for n, s in {**extensions, **controls}.items()
                  if s.get('branch') == branch}
        if not wanted and not path.exists():
            continue
        path = layout.owned(path); record = _json(path); file_hash = _sha(path)
        _require(record.get('primary_config_hash') == digest(configuration()), 'Extension primary configuration changed')
        _require(record.get('scientific_source_hashes') == run.source_fingerprint(), 'Extension scientific sources changed')
        if branch == 'baseline_supplement':
            from . import baseline_supplement
            baseline_supplement.registration(study_out=layout.out)
        _require(record.get('registered') is True and (record.get('trigger', {}).get('triggered') is True
                 or (branch == 'baseline_supplement' and record.get('trigger', {}).get('outcomes_used') is False)),
                 'Extension lacks its prospective registration/trigger')
        _require(record.get('registration_payload_hash') == digest(
            {k: v for k, v in record.items() if k != 'registration_payload_hash'}), 'Extension registration payload changed')
        source = Path(__file__).with_name(module_name+'.py')
        archived = layout.owned(layout.root/'experiments'/STUDY/(module_name+'.py'))
        _require(record.get(source_key) == _sha(source) == _sha(archived), 'Extension source provenance changed')
        raw_specs = record.get('maps', []) + record.get('controls', [])
        _require({s['configuration'] for s in raw_specs} == set(wanted),
                 'Selection omitted prospectively known extension maps/controls')
        for name, spec in wanted.items():
            raw = [s for s in raw_specs if s['configuration'] == name]
            _require({s['anchor'] for s in raw} == {0, 1, 2}, 'Extension lacks its prospectively registered three anchors')
            _require(spec.get(hash_key, spec.get('registration_id')) == file_hash,
                     'Selection extension registration pin differs from restored registration')
            for row in raw:
                for key, value in row.items():
                    if key in ('anchor', 'id'):
                        continue
                    if key in spec:
                        _require(spec[key] == value, 'Selection extension/control specification changed')
            if source_key in spec:
                _require(spec[source_key] == record[source_key], 'Selection branch source pin changed')
        groups[filename] = {'sha256': file_hash, 'registered_configurations_used': len(wanted)}
    return extensions, controls, groups


def verify_frozen_selection(*, artifact_root, original_root=None):
    """Recompute validation route/family choices and the complete explicit M.

    Candidate/family scope is the frozen prospective input. Scores come afresh
    from accepted summary files, including incomplete optional configurations;
    only complete three-anchor configurations can become nominees. No model or
    current-run evaluation artifact is opened by this function.
    """
    layout = _Layout(artifact_root, original_root)
    schedule = _json(layout.file('RESOURCE_SCHEDULE.json'))
    cfg_hash = digest(configuration())
    if not schedule.get('frozen_before_comparative_outcomes') or schedule.get('config_hash') != cfg_hash:
        raise PermissionError('Verification requires the frozen resource schedule')
    frozen_path = layout.file('SELECTION.json'); frozen = _json(frozen_path)
    _require(frozen.get('selection_frozen') is True and frozen.get('validation_only') is True,
             'Verification requires a frozen validation-only selection')
    _require(frozen.get('config_hash') == frozen.get('configuration_digest') == cfg_hash,
             'Frozen selection configuration changed')
    sources = _source_closure(layout, frozen)
    contrast_path = layout.file('CONTRASTS.json'); contrasts = _json(contrast_path)
    _require(_sha(contrast_path) == frozen.get('contrasts_sha256'), 'Frozen contrast family hash changed')
    extensions, controls, registrations = _registrations(layout, frozen)
    grid_path = layout.file('VALIDATION_GRID.json'); grid = _json(grid_path)
    _require(grid.get('evaluation_opened') is False, 'Validation grid is not a validation-only artifact')
    grid_receipts = {(r['configuration'], int(r['anchor'])): r for r in grid['receipts']}
    _require(len(grid_receipts) == len(grid['receipts']), 'Duplicate validation grid receipt')
    validation = {}; units = 0
    for name, anchors in grid['records'].items():
        _name(name); validation[name] = {}
        for anchor, saved_summary in anchors.items():
            anchor = int(anchor); base = layout.anchor(anchor)/'audits'/name
            receipt = layout.manifest(base/'COMPLETE.json', stage='audit', anchor=anchor, name=name)
            summary_path = layout.owned(base/'summary.json'); registry_path = layout.owned(base/'registry.joblib')
            summary = _json(summary_path)
            _require(summary == saved_summary, 'Accepted validation summary differs from frozen validation grid')
            _require(receipt.get('registry_sha256') == _sha(registry_path), 'Accepted registry hash changed')
            grid_pin = grid_receipts.get((name, anchor), {})
            _require(grid_pin.get('receipt_sha256') == _sha(base/'COMPLETE.json')
                     and grid_pin.get('summary_sha256') == _sha(summary_path), 'Validation grid receipt pin changed')
            if name in frozen['descriptive_configurations']:
                pin = frozen.get('frozen_audits', {}).get(name, {}).get(str(anchor), {})
                _require(pin.get('registry_sha256') == _sha(registry_path)
                         and pin.get('receipt_sha256') == _sha(base/'COMPLETE.json'), 'Selection-time audit pin changed')
            validation[name][anchor] = summary; units += 1
    _require(units == len(grid_receipts), 'Validation grid contains an orphan receipt')
    rebuilt = selection.select_validation(validation, candidate_ids=frozen['candidate_ids'],
        families=frozen['families'], registered_extensions=extensions, registered_controls=controls)
    for key, value in rebuilt.items():
        if key != 'selection_frozen':
            _require(frozen.get(key) == value, 'Frozen selection did not reconstruct: '+key)
    rebuilt_contrasts = selection.build_contrast_family(rebuilt)
    _require(contrasts == rebuilt_contrasts, 'Frozen contrast definitions/claim formulas did not reconstruct')
    _require(frozen.get('contrast_family_size') == contrasts['family_size'] == len(contrasts['endpoints']),
             'Frozen explicit contrast count differs')
    for key in ('evaluation_configurations', 'selected_contrast_configurations', 'attribution',
                'claim_formulas', 'attribution_claim_formulas', 'diagnostic_claim_formulas'):
        _require(frozen.get(key) == contrasts[key], 'Selection/contrast binding differs: '+key)
    report = {'passed': True, 'complete_configurations': len(rebuilt['aggregates']),
        'validation_units': units, 'family_size': len(contrasts['endpoints']),
        'selection_recomputed_from_saved_summaries': True, 'both_weightings_share_frozen_predictors': True,
        'selection_sha256': _sha(frozen_path), 'contrasts_sha256': _sha(contrast_path),
        'validation_grid_sha256': _sha(grid_path), 'registrations': registrations, **sources,
        'scope': 'recomputed scores, route/family nomination, endpoints and formulas from frozen declared scope; no evaluation read'}
    return {'report': report, 'selection': frozen, 'contrasts': contrasts}


def verify_replay_unit(*, registry, h_registry, ctx, release, h_release,
                       artifact_root, original_root=None, out_dir, evaluation=None,
                       evaluation_roles=None, wire_repetitions=512):
    """Replay one accepted unit without selecting or fitting anything.

    Caller verifies accepted file manifests first. This lower-level API also
    supports small synthetic bundles. All consumed model/loss paths are remapped
    to the declared owned root, even when verifying the original installation.
    """
    layout = _Layout(artifact_root, original_root)
    directory = Path(out_dir).resolve()
    _require(directory.is_relative_to(layout.root), 'Private replay output must be under artifact_root')
    _require(not directory.exists(), 'Verification output must be a new directory')
    directory.mkdir(parents=True)
    paths = {'artifact_root': layout.root, 'original_root': layout.original}
    validation = replay.verify_validation(registry, ctx, release, candidate_scope='selected',
                                          out_dir=directory/'validation', **paths)
    parity = replay.verify_ancestor_parity(h_registry, registry, ctx, h_release, release,
                                         out_dir=directory/'ancestors', **paths)
    result = {'passed': True, 'validation': validation, 'ancestor_parity': parity,
              'evaluation': {'status': 'not_requested', 'passed': None}}
    if evaluation is not None:
        result['evaluation'] = replay.verify_evaluation(registry, evaluation['ctx'], evaluation['release'],
            evaluation['directory'], out_dir=directory/'evaluation', role_keys=evaluation_roles,
            wire_repetitions=wire_repetitions, wire_seed=replay.WIRE_SEED, **paths)
    _write(directory/'unit.json', result)
    return result


def verify_math_channel(prepared, spec, result, tables, *, fine_partition=None):
    """Independent table, support, decoder, CMI and objective reconstruction."""
    family, actions = spec['input'], spec['max_actions']
    rebuilt = math_replay.reconstruct_tables(prepared, family, actions, fine_partition=fine_partition)
    comparison = math_replay.compare_tables(tables, rebuilt)
    expected_roles = sorted(role for role in rebuilt['roles'] if spec['policy'] == 'C'
                            or (spec['policy'] == 'L' and role.startswith('A/')))
    _require(sorted(result['constrained_roles']) == expected_roles and result['budget'] == spec['budget']
             and result['input'] == family and result['actions'] == actions,
             'Accepted channel specification/roles changed')
    _require(result.get('feasible') is True and result.get('Q') is not None, 'Accepted channel is infeasible')
    parents = prepared['encoder'].code.parents(family)
    inspection = math_replay.inspect_channel(tables, result, parents,
                                             prepared['encoder'].dictionaries[actions]['zero_action'])
    inspection.update(table_reconstruction=comparison, decoder_error=rebuilt['decoder_error'])
    inspection['deployment_support'] = {pool: math_replay.deployment_support(
        encoded['codes'][family], prepared['ctx']['pools'][pool]['weights'], rebuilt['state_mass'], parents)
        for pool, encoded in prepared['encoded'].items()}
    if family != 'T0':
        coarse = math_replay.reconstruct_tables(prepared, 'T0', actions, fine_partition=fine_partition)
        inspection['refinement'] = math_replay.refinement_errors(coarse, rebuilt)
    inspection['passed'] = (inspection['passed'] and comparison['passed']
        and rebuilt['decoder_error'] <= math_replay.TOLERANCES['encoder_probability']
        and inspection.get('refinement', {}).get('maximum_error', 0) <= math_replay.TOLERANCES['aggregation'])
    return inspection


def verify_baseline_supplement(prepared, spec, *, study_out):
    """Reconstruct the accepted slice/moments and verify a frozen affine map.

    No baseline fitter, cached moment builder or original-root fallback is used.
    The supplied preparation must be the fitting context, even during test replay.
    """
    from scipy.special import logit
    from . import baseline_supplement, baselines
    root = Path(study_out).resolve(); anchor = prepared['ctx']['anchor']
    _require(spec['anchor'] == anchor, 'Supplement anchor differs from fitting context')
    receipt = baseline_supplement.artifact_receipt(anchor, spec, study_out=root)
    fitted = baseline_supplement.load_fitted(anchor, spec, study_out=root)
    ix = baseline_supplement.validate_slice(prepared, spec['scope'])
    rf = prepared['ctx']['pools']['representation_fit']; p = np.asarray(prepared['encoded']['representation_fit']['p'])
    features = np.column_stack((rf['x'][ix], logit(np.clip(p[ix], 1e-5, 1-1e-5))))
    expected = {'rf_row_indices': ix, 'ids': np.asarray(rf['ids'])[ix],
        'households': np.asarray(rf['households'])[ix], 'weights': np.asarray(rf['weights'])[ix],
        'features': features, 'teacher_probabilities': p[ix],
        **{'label_'+k: np.asarray(rf['labels'][k])[ix] for k in (*baselines.PROTECTED, *baselines.TASKS)}}
    path = root/'private/run'/f'anchor_{anchor}'/'baseline_supplement'/spec['scope']/'slice.npz'
    _require(path.resolve().is_relative_to(root), 'Supplement slice escapes owned root')
    with np.load(path, allow_pickle=False) as bundle:
        _require(set(bundle.files) == set(expected), 'Supplement slice schema changed')
        for key, value in expected.items():
            actual = bundle[key]
            same = (actual.shape == value.shape and (np.allclose(actual, value, atol=1e-12, rtol=1e-12)
                    if key in ('features', 'teacher_probabilities') else np.array_equal(actual, value)))
            _require(same, 'Supplement slice differs from frozen RF inputs: '+key)
    mask = np.ones(len(ix), bool)
    for key in baselines.PROTECTED:
        mask &= expected['label_'+key] >= 0
    if spec['method'] == 'splince_supervised':
        for key in baselines.TASKS:
            mask &= expected['label_'+key] >= 0
    rows = np.flatnonzero(mask); arrays = fitted.arrays
    _require(len(rows) >= 2 and np.array_equal(rows, arrays['fit_row_indices']), 'Supplement complete-case mask changed')
    xf = features[rows]; mean = xf.mean(axis=0); xc = xf-mean
    z = np.column_stack([expected['label_'+key][rows] == value
        for key in baselines.PROTECTED for value in np.unique(expected['label_'+key][rows])]).astype(float)
    z -= z.mean(axis=0)
    y = (np.column_stack([expected['label_'+key][rows] for key in baselines.TASKS]).astype(float)
         if spec['method'] == 'splince_supervised' else np.empty((len(rows), 0)))
    if y.shape[1]:
        y -= y.mean(axis=0)
    moments = {'mean': mean, 'covariance_xx': xc.T@xc/(len(rows)-1),
        'covariance_xz': xc.T@z/(len(rows)-1), 'covariance_xy': xc.T@y/(len(rows)-1)}
    def maximum(value):
        return float(np.max(np.abs(value))) if value.size else 0.
    errors = {key: maximum(value-arrays[key]) for key, value in moments.items()}
    _require(all(np.allclose(value, arrays[key], atol=1e-10, rtol=1e-10)
                 for key, value in moments.items()), 'Supplement empirical moments differ from frozen slice')
    guarded = maximum(fitted.projection@moments['covariance_xz'])
    preserved = maximum(fitted.projection@moments['covariance_xy']-moments['covariance_xy'])
    tol = baselines.TOLERANCES
    bound = lambda value: tol['constraint_atol']+tol['constraint_rtol']*maximum(value)
    guarded_ok = guarded <= bound(moments['covariance_xz'])
    preserved_ok = preserved <= bound(moments['covariance_xy'])
    _require(guarded_ok and preserved_ok, 'Supplement frozen map failed empirical moment constraints')
    _require(receipt.get('H_byte_unchanged') is True and receipt.get('parent_cache_unchanged') is True,
             'Supplement does not preserve its parent/H')
    return {'passed': True, 'scope': spec['scope'], 'configuration': spec['configuration'],
        'provided_people': len(ix), 'provided_households': len(np.unique(expected['households'])),
        'complete_case_people': len(rows), 'complete_case_households': len(np.unique(expected['households'][rows])),
        'moment_maximum_errors': errors, 'guardedness_passed': guarded_ok,
        'task_preservation_passed': preserved_ok, 'guardedness_maximum_error': guarded,
        'task_preservation_maximum_error': preserved, 'fit_receipt_sha256': receipt['receipt_sha256'],
        'scope_limitation': 'Unweighted empirical affine moments; full H parity is verified separately.'}


class _FrozenArtifacts:
    """Read-only archive adapter. Deliberately does not call run.prepare/release_for."""
    def __init__(self, layout, frozen):
        self.layout, self.frozen = layout, frozen
        self.prepared, self.preparation_reports, self.maps, self.map_receipts = {}, {}, {}, {}
        self.inputs = layout.owned(layout.out/'private/inputs', directory=True)
        # APIs below receive explicit inputs_root; reject any nested symlink that
        # would nevertheless redirect a read to the original machine.
        for child in self.inputs.rglob('*'):
            if child.is_symlink():
                _require(child.resolve().is_relative_to(layout.root), 'Input symlink escapes artifact_root')

    def spec(self, name, anchor):
        primary = next((s for s in configuration()['maps']
                        if s['configuration'] == name and s['anchor'] == anchor), None)
        if primary is not None:
            return primary
        for filename, registration_key, source_key in (
                ('EXTRA_CONFIGS.json', 'extra_registration_sha256', 'branch_source_sha256'),
                ('ROBUSTNESS_CONFIGS.json', 'robustness_registration_sha256', 'robustness_source_sha256'),
                ('BASELINE_SUPPLEMENTS.json', 'baseline_registration_sha256', 'baseline_source_sha256')):
            path = self.layout.out/filename
            if not path.exists():
                continue
            record = _json(self.layout.owned(path))
            raw = next((s for s in record['maps']+record.get('controls', [])
                        if s['configuration'] == name and s['anchor'] == anchor), None)
            if raw is not None:
                return {**raw, registration_key: _sha(path), source_key: record[source_key]}
        return None

    def branch_schedule(self, spec):
        if spec['branch'] == 'baseline_supplement':
            from . import baseline_supplement
            return baseline_supplement.require_scheduled(spec, study_out=self.layout.out)
        filename = 'ROBUSTNESS_RESOURCE_SCHEDULE.json' if spec['branch'] == 'C' else 'EXTRA_RESOURCE_SCHEDULE.json'
        path = self.layout.file(filename); schedule = _json(path)
        registration_key = 'robustness_registration_sha256' if spec['branch'] == 'C' else 'extra_registration_sha256'
        _require(schedule.get('schedule_payload_hash') == digest(
            {k: v for k, v in schedule.items() if k != 'schedule_payload_hash'}), 'Branch schedule payload changed')
        _require(schedule.get(registration_key) == spec[registration_key]
                 and schedule.get('primary_resource_schedule_sha256') == _sha(self.layout.file('RESOURCE_SCHEDULE.json'))
                 and schedule.get('frozen_before_affected_outcomes') is True
                 and spec['id'] in schedule.get('unit_ids', []), 'Branch unit was not prospectively scheduled')
        return _sha(path)

    def preparation(self, anchor):
        if anchor in self.prepared:
            return self.prepared[anchor]
        base = self.layout.anchor(anchor)
        receipt = self.layout.manifest(base/'PREPARED.json', stage='prepare', anchor=anchor)
        cache = self.layout.owned(base/'prepared.joblib')
        _require(_sha(cache) == receipt.get('cache_sha256'), 'Prepared cache pin changed')
        prepared = joblib.load(cache)
        _require(prepared['ctx']['anchor'] == anchor and 'test' not in prepared['ctx']['pools']
                 and set(prepared['encoded']) == set(prepared['ctx']['pools']), 'Prepared fitting/validation pool identity changed')
        encoder = joblib.load(self.layout.owned(base/'encoder/encoder.joblib'))
        encoded, encoder_report = math_replay._encoder_replay(prepared, encoder)
        parity = math_replay.h_parity(prepared['ctx'], self.inputs)
        _require(all(row['passed'] for row in encoder_report.values()), 'Frozen encoder predictions/codes did not replay')
        _require(all(all(row.values()) for row in parity.values()), 'Original H columns differ bytewise')
        self.preparation_reports[anchor] = {'passed': True, 'encoder_replay': encoder_report,
            'H_byte_parity': parity, 'cache_sha256': _sha(cache)}
        self.prepared[anchor] = {**prepared, 'encoder': encoder, 'encoded': encoded}
        return self.prepared[anchor]

    def required_map(self, name, anchor):
        spec = self.spec(name, anchor)
        if spec is not None:
            return spec.get('required_map') if spec.get('kind') == 'control' else name
        if name == 'constant_best':
            return mechanism_id('T0', 'U', None)
        if re.fullmatch(r'(T0|Ttask|Trisk)_(withhold|rr)_[0-9.]+', name):
            return mechanism_id(name.split('_')[0], 'U', None)
        return None

    def map(self, name, anchor):
        key = (anchor, name)
        if key in self.maps:
            return self.maps[key]
        self.preparation(anchor)
        base = self.layout.anchor(anchor)/'maps'/_name(name); spec = self.spec(name, anchor)
        _require(spec is not None and spec.get('kind') != 'control', 'Unregistered frozen map')
        receipt = self.layout.manifest(base/'ACCEPTED.json', stage='map', anchor=anchor, name=name)
        path = self.layout.owned(base/'solution.joblib')
        _require(receipt.get('sha256') == _sha(path) and receipt.get('spec_hash') == digest(spec),
                 'Map solution/specification provenance changed')
        _require(receipt.get('prepared_cache_sha256') == self.preparation_reports[anchor]['cache_sha256'],
                 'Map preparation dependency changed')
        if spec.get('branch') in ('A', 'C'):
            _require(receipt.get('extra_resource_schedule_sha256') == self.branch_schedule(spec),
                     'Map branch schedule changed')
        coarse = receipt.get('coarse_configuration')
        if coarse is not None:
            _require(coarse != name and self.spec(coarse, anchor)['input'] == 'T0', 'Invalid coarse dependency')
            self.map(coarse, anchor)
            _require(receipt.get('coarse_solution_sha256') == self.map_receipts[(anchor, coarse)]['sha256'],
                     'Frozen coarse map dependency changed')
        result = joblib.load(path)
        _require(result.get('feasible') is True and result.get('Q') is not None, 'Frozen accepted map is infeasible')
        self.maps[key], self.map_receipts[key] = result, receipt
        return result

    def audit(self, name, anchor):
        self.preparation(anchor)
        base = self.layout.anchor(anchor)/'audits'/_name(name)
        receipt = self.layout.manifest(base/'COMPLETE.json', stage='audit', anchor=anchor, name=name)
        _require(receipt.get('prepared_cache_sha256') == self.preparation_reports[anchor]['cache_sha256'],
                 'Audit preparation dependency changed')
        required = self.required_map(name, anchor)
        if required is not None:
            self.map(required, anchor)
            _require(receipt.get('map_solution_sha256') == self.map_receipts[(anchor, required)]['sha256'],
                     'Audit channel dependency changed')
        if name != 'H':
            pin = self.frozen['frozen_audits']['H'][str(anchor)]
            _require(receipt.get('H_registry_sha256') == pin['registry_sha256'], 'Audit H ancestor dependency changed')
        spec = self.spec(name, anchor)
        if spec is not None and spec.get('branch') in ('A', 'C', 'baseline_supplement'):
            _require(receipt.get('branch_release_hash') == digest(spec)
                     and receipt.get('extra_resource_schedule_sha256') == self.branch_schedule(spec),
                     'Audit branch registration/schedule dependency changed')
            if spec['branch'] == 'baseline_supplement':
                from . import baseline_supplement
                _require(receipt.get('supplement_fit_sha256') == baseline_supplement.artifact_receipt(
                    anchor, spec, study_out=self.layout.out)['receipt_sha256'], 'Audit supplement fit dependency changed')
        return self.layout.owned(base/'registry.joblib')

    def release(self, name, anchor, prepared):
        from .mechanisms import build_release
        required = self.required_map(name, anchor)
        mechanism = None if required is None else self.map(required, anchor)
        spec = self.spec(name, anchor) or {}
        erasers = prepared['erasers']
        if spec.get('branch') == 'baseline_supplement':
            from . import baseline_supplement
            report = verify_baseline_supplement(self.prepared.get(anchor, prepared), spec, study_out=self.layout.out)
            self.preparation_reports.setdefault(anchor, {}).setdefault('baseline_supplements', {})[name] = report
            erasers = {**erasers, name: baseline_supplement.load_fitted(anchor, spec, study_out=self.layout.out)}
        return build_release(prepared['ctx'], prepared['encoder'], prepared['encoded'],
            spec.get('canonical_release', name), mechanism=mechanism, erasers=erasers,
            actions=spec.get('max_actions', 17), inputs_root=self.inputs)

    def math_maps(self, anchor):
        reports = {}
        for (a, name), result in sorted(self.maps.items()):
            if a != anchor:
                continue
            spec = self.spec(name, anchor); prepared = self.preparation(anchor); fine = None
            if spec.get('branch') == 'C':
                base = self.layout.anchor(anchor)/'branches/fineC'
                receipt = self.layout.manifest(base/'TABLES.json', anchor=anchor)
                _require(receipt.get('prepared_cache_sha256') == self.preparation_reports[anchor]['cache_sha256']
                         and receipt.get('robustness_registration_sha256') == spec['robustness_registration_sha256']
                         and receipt.get('robustness_source_sha256') == spec['robustness_source_sha256']
                         and receipt.get('robustness_resource_schedule_sha256') == self.branch_schedule(spec),
                         'Fine-conditioning dependency changed')
                fine = joblib.load(self.layout.owned(base/'tables.joblib'))['partitions']
            tables = math_replay._read_tables(self.layout.owned(
                self.layout.anchor(anchor)/'maps'/name/'tables.npz'))
            reports[name] = verify_math_channel(prepared, spec, result, tables, fine_partition=fine)
        return {'passed': all(r['passed'] for r in reports.values()), 'channels': reports,
                'status': 'checked' if reports else 'not_applicable_no_channel_dependencies',
                'scope': 'selected release map dependencies and coarse witnesses; independent empirical math, no new optimum certificate'}


def verify_restored_unit(name, *, artifact_root, original_root, out_dir, anchor=0,
                         include_evaluation=True, expected_selection_sha256=None, wire_repetitions=512):
    """Replay one representative restored anchor, without reading other anchors.

    Use the selection hash from the separately completed original-root full
    verification as ``expected_selection_sha256``. Global registration/source
    documents, owned inputs and this anchor's accepted dependencies must exist.
    This checks the frozen selection binding; it does not recompute three-anchor
    selection and clearly reports that narrower scope.
    """
    layout = _Layout(artifact_root, original_root); _name(name); layout.anchor(anchor)
    directory = Path(out_dir).resolve()
    _require(directory.is_relative_to(layout.out/'private') and not directory.exists(),
             'Representative replay needs a new task-owned private output directory')
    schedule = _json(layout.file('RESOURCE_SCHEDULE.json')); cfg_hash = digest(configuration())
    if not schedule.get('frozen_before_comparative_outcomes') or schedule.get('config_hash') != cfg_hash:
        raise PermissionError('Representative replay requires the frozen resource schedule')
    permit = layout.file('SELECTION.json'); frozen = _json(permit); selection_hash = _sha(permit)
    _require(frozen.get('selection_frozen') is True and frozen.get('validation_only') is True
             and frozen.get('config_hash') == frozen.get('configuration_digest') == cfg_hash,
             'Representative replay requires the unchanged frozen selection')
    if expected_selection_sha256 is not None:
        _require(selection_hash == expected_selection_sha256, 'Restored selection differs from original verification pin')
    _source_closure(layout, frozen); _registrations(layout, frozen)
    _require(_sha(layout.file('CONTRASTS.json')) == frozen.get('contrasts_sha256'), 'Restored contrast binding changed')
    _require(name in frozen['evaluation_configurations'], 'Representative configuration was not frozen')
    artifacts = _FrozenArtifacts(layout, frozen); prepared = artifacts.preparation(anchor)
    registry_paths = {}
    for configuration_name in {'H', name}:
        path = artifacts.audit(configuration_name, anchor)
        pin = frozen['frozen_audits'].get(configuration_name, {}).get(str(anchor), {})
        _require(pin.get('registry_sha256') == _sha(path)
                 and pin.get('receipt_sha256') == _sha(layout.owned(path.parent/'COMPLETE.json')),
                 'Representative restored audit differs from selection-time pin')
        registry_paths[configuration_name] = path
    release = artifacts.release(name, anchor, prepared)
    h_release = artifacts.release('H', anchor, prepared); evaluation = None; evaluation_parity = None
    if include_evaluation:
        base = layout.anchor(anchor)/'evaluation'/name
        receipt = layout.manifest(base/'COMPLETE.json', stage='evaluation', anchor=anchor, name=name)
        _require(receipt.get('selection_sha256') == selection_hash
                 and receipt.get('registry_sha256') == _sha(registry_paths[name]),
                 'Representative evaluation selection/registry binding changed')
        from .data import RuntimeInputs, load_anchor
        ctx = load_anchor(anchor, pools=('test',), inputs_root=artifacts.inputs, evaluation_permit=permit)
        encoded = {p: prepared['encoder'].encode(RuntimeInputs(d['x'], d['ha'])) for p, d in ctx['pools'].items()}
        evaluation_parity = math_replay.h_parity(ctx, artifacts.inputs)
        _require(all(all(row.values()) for row in evaluation_parity.values()), 'Restored evaluation H byte parity failed')
        evaluation = {'ctx': ctx, 'release': artifacts.release(name, anchor,
            {**prepared, 'ctx': ctx, 'encoded': encoded}), 'directory': base}
    unit = verify_replay_unit(registry=registry_paths[name], h_registry=registry_paths['H'],
        ctx=prepared['ctx'], release=release, h_release=h_release, artifact_root=layout.root,
        original_root=layout.original, out_dir=directory, evaluation=evaluation, wire_repetitions=wire_repetitions)
    math_report = artifacts.math_maps(anchor)
    _require(_sha(permit) == selection_hash, 'Restored selection changed during replay')
    result = {'passed': bool(unit['passed'] and math_report['passed']), 'anchor': anchor, 'configuration': name,
        'scope': 'representative single-anchor restore; full selection reconstruction is separate',
        'selection_recomputed': False, 'selection_sha256': selection_hash,
        'original_selection_pin_checked': expected_selection_sha256 is not None,
        'original_fallback_allowed': False, 'preparation': artifacts.preparation_reports[anchor],
        'evaluation_H_byte_parity': evaluation_parity, 'unit': unit, 'mathematical_replay': math_report}
    _write(directory/'restore_verification.json', result)
    return result


def verify_study(*, artifact_root, original_root=None, out_dir, include_evaluation=False,
                 evaluation_roles=None, wire_repetitions=512, additional_evaluation_configurations=()):
    """Verify frozen nominees/comparators; optionally replay accepted test outputs.

    All endpoint-bearing configurations receive all-role validation and ancestor
    replay. Evaluation replays every role for those configurations. Additional
    descriptive configurations may be explicitly nominated for material-role
    replay (default primary four plus residence). Route finalists receive the
    fixed independent sampled-wire check. A fresh private output directory is
    required. Failures are reported with a non-passing status and private incident.
    """
    layout = _Layout(artifact_root, original_root); directory = Path(out_dir).resolve()
    _require(directory.is_relative_to(layout.out/'private'), 'Verification output must stay task-owned private')
    _require(not directory.exists(), 'Verification output must be new')
    _require(isinstance(wire_repetitions, int) and not isinstance(wire_repetitions, bool)
             and wire_repetitions >= 1, 'Finalist sampled-wire repetitions must be positive')
    directory.mkdir(parents=True)
    report = {'schema': 1, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'passed': False, 'evaluation_requested': bool(include_evaluation), 'evaluation_opened': False,
        'units': {}, 'preparations': {}, 'mathematical_replay': {},
        'verifier_source_sha256': _sha(__file__), 'replay_source_sha256': _sha(replay.__file__),
        'math_replay_source_sha256': _sha(math_replay.__file__), 'original_fallback_allowed': False,
        'scope': 'frozen fitted artifacts only; no refits, repairs, evaluation selection or population guarantees'}
    stage = 'selection_and_provenance'; tick = time.monotonic()
    try:
        rebuilt = verify_frozen_selection(artifact_root=layout.root, original_root=layout.original)
        frozen = rebuilt['selection']; report['selection'] = rebuilt['report']
        selected = frozen['selected_contrast_configurations']
        additional = list(additional_evaluation_configurations)
        _require(len(set(additional)) == len(additional) and set(additional) <= set(frozen['evaluation_configurations']),
                 'Additional evaluation sample must be unique frozen configurations')
        _require(include_evaluation or not additional, 'Additional evaluation sample requires explicit evaluation mode')
        finalists = {r['nominee'] for r in frozen['routes'].values() if r['nominee'] is not None}
        report.update(validation_configurations=selected, evaluation_configurations=sorted(set(selected+additional))
                      if include_evaluation else [], finalist_wire_configurations=sorted(finalists)
                      if include_evaluation else [], wire_repetitions=wire_repetitions)
        stage = 'frozen_artifact_loading'; artifacts = _FrozenArtifacts(layout, frozen)
        from .data import RuntimeInputs, load_anchor
        for anchor in (0, 1, 2):
            stage = f'anchor_{anchor}/preparation'
            prepared = artifacts.preparation(anchor)
            report['preparations'][str(anchor)] = artifacts.preparation_reports[anchor]
            h_registry = artifacts.audit('H', anchor)
            h_release = artifacts.release('H', anchor, prepared)
            # Resolve every frozen fitting dependency before any test-array read.
            for name in sorted(set(selected+additional)):
                artifacts.audit(name, anchor)
            evaluation_prepared = None
            if include_evaluation:
                # All selection/source/summary/registry pins have already passed.
                report['evaluation_opened'] = True
                ctx = load_anchor(anchor, pools=('test',), inputs_root=artifacts.inputs,
                                  evaluation_permit=layout.file('SELECTION.json'))
                encoded = {p: prepared['encoder'].encode(RuntimeInputs(d['x'], d['ha'])) for p, d in ctx['pools'].items()}
                evaluation_prepared = {**prepared, 'ctx': ctx, 'encoded': encoded}
                parity = math_replay.h_parity(ctx, artifacts.inputs)
                _require(all(all(row.values()) for row in parity.values()), 'Evaluation H columns differ bytewise')
                report['preparations'][str(anchor)]['evaluation_H_byte_parity'] = parity
            for name in sorted(set(selected+additional)):
                stage = f'anchor_{anchor}/{name}'
                registry = artifacts.audit(name, anchor)
                release = artifacts.release(name, anchor, prepared)
                evaluation = None
                if include_evaluation:
                    base = layout.anchor(anchor)/'evaluation'/name
                    receipt = layout.manifest(base/'COMPLETE.json', stage='evaluation', anchor=anchor, name=name)
                    _require(receipt.get('selection_sha256') == report['selection']['selection_sha256']
                             and receipt.get('registry_sha256') == frozen['frozen_audits'][name][str(anchor)]['registry_sha256'],
                             'Evaluation receipt differs from frozen selection/registry')
                    evaluation = {'ctx': evaluation_prepared['ctx'],
                        'release': artifacts.release(name, anchor, evaluation_prepared), 'directory': base}
                roles = None if name in selected else (evaluation_roles or list(selection.REQUIRED_ROLES))
                unit = verify_replay_unit(registry=registry, h_registry=h_registry, ctx=prepared['ctx'],
                    release=release, h_release=h_release, artifact_root=layout.root, original_root=layout.original,
                    out_dir=directory/f'anchor_{anchor}'/name, evaluation=evaluation, evaluation_roles=roles,
                    wire_repetitions=wire_repetitions if name in finalists else 0)
                report['units'][f'{name}/anchor_{anchor}'] = {'passed': unit['passed'],
                    'validation_roles': len(unit['validation']['roles']), 'ancestor_parity': True,
                    'evaluation_roles': {p: list(v) for p, v in unit['evaluation'].get('pools', {}).items()},
                    'sampled_wire_executed': include_evaluation and name in finalists}
            stage = f'anchor_{anchor}/mathematical_replay'
            report['mathematical_replay'][str(anchor)] = artifacts.math_maps(anchor)
            _require(report['mathematical_replay'][str(anchor)]['passed'], 'Selected mathematical channel replay failed')
            # Bound resident memory to one prepared anchor after its replay finishes.
            del artifacts.prepared[anchor]
            artifacts.maps = {k: v for k, v in artifacts.maps.items() if k[0] != anchor}
        _require(_sha(layout.file('SELECTION.json')) == report['selection']['selection_sha256']
                 and _sha(layout.file('CONTRASTS.json')) == report['selection']['contrasts_sha256'],
                 'Frozen selection/contrasts changed during verification')
        _source_closure(layout, frozen)
        report['passed'] = True
    except Exception as error:
        report.update(failed_stage=stage, failure_type=type(error).__name__)
        _write(directory/'incident.json', {'stage': stage, 'type': type(error).__name__, 'message': str(error)})
    report['seconds'] = time.monotonic()-tick
    _write(directory/'verification.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', required=True, type=Path)
    parser.add_argument('--original-root', type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--include-evaluation', action='store_true')
    parser.add_argument('--wire-repetitions', type=int, default=512)
    parser.add_argument('--additional-evaluation-configuration', action='append', default=[])
    args = parser.parse_args(argv)
    report = verify_study(artifact_root=args.artifact_root, original_root=args.original_root,
        out_dir=args.out_dir, include_evaluation=args.include_evaluation, wire_repetitions=args.wire_repetitions,
        additional_evaluation_configurations=args.additional_evaluation_configuration)
    print(json.dumps({'passed': report['passed'], 'evaluation_opened': report['evaluation_opened'],
                      'report': str(args.out_dir/'verification.json')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
