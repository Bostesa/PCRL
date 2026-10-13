"""Supplementary all-applicable-reader assessment, fixed before INNER opens.

The original K108 base9 diagnostic family remains unchanged. This separate K18
family includes every frozen base and extended reader applicable to the actual
full/projection anchor wires, with exact local-ignore aliases. No reader fit or
orientation changes occur here. Imports are metadata-only until root admission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .reader_specs import BASE_READERS, EXTENDED_READERS
from .reporting import (BOOTSTRAP_SEED, REPLICATES, RAW_CONTROL, table_name,
                        control_dependencies, primary_keys)

ARMS = ('U0', 'U0_PROJ', RAW_CONTROL, RAW_CONTROL+'_PROJ',
        'JOINT_CONDITIONAL_R03_E5', 'JOINT_CONDITIONAL_R03_E5_PROJ')


def complete_keys(view):
    names = tuple(BASE_READERS)+tuple(EXTENDED_READERS)
    keys = [f'{view}__{name}' for name in names]
    if view == 'pair':
        keys += [f'pair__IGNORE_{other}__{name}' for other in ('r1', 'r2') for name in names]
    return sorted(keys)


def build_registry():
    """Exact K18 definitions from already fitted reader metadata only."""
    endpoints = []
    for arm in ARMS:
        for view in ('pair', 'r1', 'r2'):
            benefit = view == 'pair'; metric = 'pair_benefit' if benefit else f'{view}_worsening'
            parts = []
            for seed in (0, 1):
                candidate = table_name('scores', arm, seed); control = table_name('scores', RAW_CONTROL, seed)
                left, right = (control, candidate) if benefit else (candidate, control)
                parts.append({'source_seed': seed, 'truth_table': 'truth', 'truth_key': 'sex',
                              'left_table': left, 'left_keys': complete_keys(view),
                              'right_table': right, 'right_keys': complete_keys(view), 'sign': 1})
            endpoints.append({'endpoint_id': f'complete_view__{arm}__{metric}', 'arm_id': arm,
                              'view': view, 'metric': metric, 'control': RAW_CONTROL,
                              'kind': 'auc_envelope_difference', 'direction': 'at_least' if benefit else 'at_most',
                              'threshold': .020 if benefit else .010, 'paired_seeds': parts,
                              'failure_dependencies': [{'table': 'control_gates', 'key': f'{dependency}__S{seed}'}
                                                       for dependency in control_dependencies(arm) for seed in (0, 1)]})
    return {'schema': 'OB1-supplementary-complete-view-family-v1', 'family': 'complete_view',
            'configuration_count': 6, 'K': 18, 'replicates': REPLICATES, 'bootstrap_seed': BOOTSTRAP_SEED,
            'alpha': .05, 'expected_rows': 2235, 'expected_groups': 2234,
            'base_readers': list(BASE_READERS), 'extended_readers': list(EXTENDED_READERS),
            'local_reader_count': 18, 'pair_reader_count_including_ignore_aliases': 54,
            'envelope': 'all applicable frozen base9+extended9; exact local-ignore aliases in pair',
            'new_fits_or_orientation_changes': False, 'post_INNER_endpoint_reduction_allowed': False,
            'original_base9_K108_family_changed': False, 'exploratory_fixed_models_readers': True,
            'bootstrap_draws': 'default_rng(seed).integers(0,G,size=(B,G)); shared group row multiplicities across every arm and both seeds',
            'interval': 'point +/- Student-t(1-alpha/(2K),G-1) times paired-group bootstrap SD(ddof=1)',
            'endpoints': endpoints}


def _committed(path, sha):
    from .source import ROOT
    p = Path(path).resolve()
    relative = p.relative_to(ROOT).as_posix()
    result = subprocess.run(['git', 'show', 'HEAD:'+relative], cwd=ROOT, capture_output=True, check=False)
    if result.returncode or hashlib.sha256(result.stdout).hexdigest() != sha:
        raise PermissionError('supplementary complete-view freeze bytes must be committed at HEAD')


def audit_worker(spec_path):
    from .evaluation_worker import queue_config, require_reporting
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    seal, canonical = require_reporting(cfg, verify)
    registry_path = verify(cfg['registry']['path'], cfg['registry']['sha256'])
    registry = json.loads(registry_path.read_text())
    if registry != build_registry():
        raise PermissionError('supplementary complete-view endpoints differ from pre-INNER full definitions')
    _committed(registry_path, cfg['registry']['sha256'])
    supplement_path = verify(cfg['supplementary_lock']['path'], cfg['supplementary_lock']['sha256'])
    supplement = json.loads(supplement_path.read_text())
    _committed(supplement_path, cfg['supplementary_lock']['sha256'])
    code_path = Path(__file__).resolve()
    code_sha = supplement.get('code_sha256')
    verify(code_path, code_sha)
    _committed(code_path, code_sha)
    if (supplement.get('owner') != 'ROOT' or supplement.get('status') != 'FROZEN'
            or supplement.get('registry_sha256') != cfg['registry']['sha256']
            or supplement.get('reporting_lock_sha256') != seal['sha256']):
        raise PermissionError('supplementary ROOT lock code/registry/reporting binding differs')
    # All access and commitment checks precede numerical imports/person reads.
    import numpy as np
    from scipy.stats import t
    from .reporting import (_arrays, _aligned, bootstrap_weights, weighted_auc_draws,
                            endpoint_classification, _private_directory, _save_arrays,
                            _write_json, _csv)
    units = _arrays(cfg['unit_input'], verify)
    needed = {'truth', 'control_gates'} | {table_name('scores', arm, seed) for arm in ARMS for seed in (0, 1)}
    tables = {name: _arrays(cfg['arrays'][name], verify) for name in sorted(needed)}
    rows = len(units['unit']); groups = len(np.unique(units['unit']))
    if rows != 2235 or groups != 2234 or len(np.unique(units['row_id'])) != rows:
        raise PermissionError('supplementary assessment uses different people/groups')
    for table in tables.values():
        _aligned(table, units)
    truth = tables['truth']['sex']
    weights = np.vstack((np.ones((1, rows)), bootstrap_weights(units['unit'])))
    reader_cache = {}; envelope_cache = {}; audit_rows = []; curves = {}; endpoint_rows = []
    def reader(table, key):
        scores = tables[table][key]
        if scores.shape != (rows,) or not np.isfinite(scores).all():
            raise ValueError('complete-view frozen score shape/range differs')
        digest = hashlib.sha256(np.ascontiguousarray(scores, dtype=np.float64).tobytes()).hexdigest()
        if digest not in reader_cache:
            reader_cache[digest] = weighted_auc_draws(truth, scores, weights)
        return reader_cache[digest]
    def envelope(table, keys):
        cache_key = (table, tuple(keys))
        if cache_key not in envelope_cache:
            envelope_cache[cache_key] = np.maximum.reduce([reader(table, key) for key in keys])
            print(f'complete_view_cached_envelopes={len(envelope_cache)} oriented_curves={len(reader_cache)}', flush=True)
        return envelope_cache[cache_key]
    # Verify exact alias arithmetic before any endpoint is advertised complete.
    for arm in ARMS:
        for seed in (0, 1):
            name = table_name('scores', arm, seed); scores = tables[name]
            for local, other in (('r1', 'r2'), ('r2', 'r1')):
                for reader_name in tuple(BASE_READERS)+tuple(EXTENDED_READERS):
                    a = f'{local}__{reader_name}'; b = f'pair__IGNORE_{other}__{reader_name}'
                    if a not in scores or b not in scores or not np.array_equal(scores[a], scores[b]):
                        raise PermissionError('all18 complete-view pair omits or alters an exact local-ignore alias')
            for view in ('r1', 'r2', 'pair'):
                keys = complete_keys(view)
                complete = envelope(name, keys)
                base = envelope(name, primary_keys(view))
                if np.any(complete+1e-14 < base):
                    raise AssertionError('expanded frozen envelope fell below its base9 subset')
                point_readers = {key: float(reader(name, key)[0]) for key in keys}
                audit_rows.append({'arm_id': arm, 'seed': seed, 'view': view,
                                   'base9_auc': float(base[0]), 'complete18_auc': float(complete[0]),
                                   'expanded_minus_base9': float(complete[0]-base[0]), 'reader_count': len(keys),
                                   'strongest_reader': max(point_readers, key=point_readers.get),
                                   'envelope_reselected_inside_each_draw': True})
    critical = float(t.ppf(1-.05/(2*18), df=groups-1))
    for endpoint in registry['endpoints']:
        guard()
        identity = {key: endpoint[key] for key in ('endpoint_id','arm_id','metric','direction','threshold')}
        try:
            for dependency in endpoint['failure_dependencies']:
                gate = tables[dependency['table']][dependency['key']]
                if gate.shape != (rows,) or gate.dtype.kind != 'b' or not np.all(gate == gate[0]):
                    raise ValueError('supplementary control gate is not a constant person-aligned bool vector')
                if not gate[0]:
                    raise KeyError('failed required recovery control '+dependency['key'])
            parts = []
            for part in endpoint['paired_seeds']:
                parts.append(envelope(part['left_table'], part['left_keys'])-envelope(part['right_table'], part['right_keys']))
            values = np.mean(parts, axis=0); point = float(values[0]); boot = values[1:]
            se = float(boot.std(ddof=1)); lower = point-critical*se; upper = point+critical*se
            status, point_status, gap = endpoint_classification(point, lower, upper, endpoint['threshold'], endpoint['direction'])
            row = {**identity, 'point': point, 'bootstrap_se': se, 'lower': lower, 'upper': upper,
                   'status': status, 'point_status': point_status, 'gap_to_target': gap,
                   'K': 18, 'df': groups-1, 'critical': critical, 'replicates': REPLICATES}
            curves[endpoint['endpoint_id']] = boot
        except KeyError as error:
            row = {**identity, 'point': None, 'bootstrap_se': None, 'lower': None, 'upper': None,
                   'status': 'MISSING', 'point_status': 'MISSING', 'gap_to_target': None,
                   'K': 18, 'df': groups-1, 'replicates': REPLICATES, 'missing_dependency': str(error)}
            curves[endpoint['endpoint_id']] = np.full(REPLICATES, np.nan)
        endpoint_rows.append(row)
    root = _private_directory(cfg['private_output']); public = Path(cfg['public_output']); public.mkdir(parents=True, exist_ok=True)
    curve_pin = _save_arrays(root/'COMPLETE_VIEW_ENDPOINT_BOOTSTRAP_DRAWS.npz', curves)
    result = {'schema': 'OB1-supplementary-complete-view-assessment-v1', 'family': 'complete_view',
              'unit_id': active['unit_id'], 'K': 18, 'configuration_count': 6, 'rows': rows, 'groups': groups,
              'bootstrap_seed': BOOTSTRAP_SEED, 'replicates': REPLICATES,
              'reporting_lock_sha256': seal['sha256'], 'registry_sha256': cfg['registry']['sha256'],
              'supplementary_lock_sha256': cfg['supplementary_lock']['sha256'],
              'endpoints': endpoint_rows, 'all_applicable_envelope_audit': audit_rows,
              'status_counts': {status: sum(row['status'] == status for row in endpoint_rows) for status in ('PASS','UNRESOLVED','VIOLATION','SHORTFALL','MISSING')},
              'complete_family': all(row['status'] != 'MISSING' for row in endpoint_rows),
              'cached_envelopes': len(envelope_cache), 'cached_oriented_reader_curves': len(reader_cache),
              'approximate_exploratory_fixed_model_reader_coverage': True,
              'original_base9_K108_family_changed': False, 'new_fits_or_orientation_changes': False}
    _write_json(public/'COMPLETE_VIEW_AUDIT.json', result)
    _csv(public/'COMPLETE_VIEW_ENVELOPE_AUDIT.csv', audit_rows,
         ['arm_id','seed','view','base9_auc','complete18_auc','expanded_minus_base9','reader_count','strongest_reader','envelope_reselected_inside_each_draw'])
    _csv(public/'COMPLETE_VIEW_GAP_TO_TARGET.csv', endpoint_rows,
         ['endpoint_id','arm_id','metric','direction','threshold','point','lower','upper','bootstrap_se','status','point_status','gap_to_target','K','df','replicates'])
    return {'status': 'SUPPLEMENTARY_COMPLETE_VIEW_ASSESSED', 'bootstrap_draws': curve_pin,
            'K': 18, 'status_counts': result['status_counts'], 'complete_family': result['complete_family']}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=('registry','audit'), required=True)
    parser.add_argument('--spec'); parser.add_argument('--output'); args = parser.parse_args()
    if args.phase == 'registry':
        from .reporting import _write_json
        result = _write_json(args.output, build_registry())
    else:
        result = audit_worker(args.spec)
    print(json.dumps(result, sort_keys=True), flush=True)

if __name__ == '__main__':
    main()
