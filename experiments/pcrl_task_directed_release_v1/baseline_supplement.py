"""Prospective mechanism40/union88 affine controls without changing parent fits.

Only this wrapper's new artifacts receive corrected slice metadata. Canonical
LEACE/SPLINCE mathematics, the frozen teacher and all original caches are reused
unchanged. Registration is metadata-only. A published registration commit and
all twelve scheduled audits are required before fitting either paired scope.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import joblib
import numpy as np
from scipy.special import logit

from . import baselines
from .config import configuration, digest

BRANCH = 'baseline_supplement'
REGISTRY = 'BASELINE_SUPPLEMENTS.json'
SCHEDULE = 'BASELINE_SUPPLEMENT_SCHEDULE.json'
AMENDMENT = 'AMENDMENT_6_BASELINE_FAIRNESS.md'
SCOPES = ('mechanism40', 'union88')
METHODS = ('leace_supervised', 'splince_supervised')


def _run():
    from . import run
    return run


def _root(study_out=None):
    return Path(_run().OUT if study_out is None else study_out).resolve()


def _owned(path, root, *, directory=False):
    path = Path(path).resolve()
    if not path.is_relative_to(root) or not (path.is_dir() if directory else path.is_file()):
        raise ValueError('Missing or escaping owned supplement artifact')
    return path


def specs():
    return [{'id': f'{method}_{scope}/anchor_{anchor}', 'configuration': f'{method}_{scope}',
        'anchor': anchor, 'branch': BRANCH, 'kind': 'control', 'method': method, 'scope': scope,
        'baseline_family': 'supervised_LEACE' if method.startswith('leace') else 'supervised_SPLINCE',
        'required': True, 'label_matched': True, 'max_actions': 17, 'required_map': None,
        'canonical_release': f'{method}_{scope}'}
        for scope in SCOPES for method in METHODS for anchor in (0, 1, 2)]


def registration(*, study_out=None):
    root = _root(study_out); run = _run()
    record = json.loads(_owned(root/REGISTRY, root).read_text())
    expected = {'schema': 1, 'branch': BRANCH, 'registered': True, 'maps': [], 'controls': specs(),
        'primary_config_hash': digest(configuration()), 'scientific_source_hashes': run.source_fingerprint(),
        'baseline_source_sha256': run.sha(__file__), 'amendment_sha256': run.sha(_owned(root/AMENDMENT, root)),
        'primary_resource_schedule_sha256': run.sha(_owned(root/'RESOURCE_SCHEDULE.json', root)),
        'mandatory_audit_units': 12,
        'trigger': {'kind': 'prospective_source_review_fairness_repair', 'outcomes_used': False}}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen baseline supplement registration/source changed')
    if record.get('registration_payload_hash') != digest({k: v for k, v in record.items() if k != 'registration_payload_hash'}):
        raise ValueError('Baseline supplement registration payload changed')
    return record


def register():
    """Write source/scope metadata only, before Git publication and new fits."""
    run = _run(); root = _root()
    with run.unit_lock('baseline_supplement/registration'):
        if (root/REGISTRY).exists():
            return registration()
        if any((root/name).exists() for name in ('SELECTION.json', 'VALIDATION_GRID.json')):
            raise PermissionError('Fairness supplement must be registered before comparative validation collection')
        schedule = json.loads(_owned(root/'RESOURCE_SCHEDULE.json', root).read_text())
        if not schedule.get('frozen_before_comparative_outcomes') or schedule.get('config_hash') != digest(configuration()):
            raise PermissionError('Frozen primary resource schedule required')
        if any((root/'private/run'/f'anchor_{a}'/'baseline_supplement').exists() for a in (0, 1, 2)):
            raise ValueError('Supplement registration must precede new fits')
        record = {'schema': 1, 'branch': BRANCH, 'registered': True, 'created_utc': run.now(),
            'primary_config_hash': digest(configuration()), 'scientific_source_hashes': run.source_fingerprint(),
            'baseline_source_sha256': run.sha(__file__), 'amendment_sha256': run.sha(_owned(root/AMENDMENT, root)),
            'primary_resource_schedule_sha256': run.sha(root/'RESOURCE_SCHEDULE.json'),
            'maps': [], 'controls': specs(), 'mandatory_audit_units': 12,
            'trigger': {'kind': 'prospective_source_review_fairness_repair', 'outcomes_used': False},
            'fit_scopes': {'mechanism40': 'exact frozen mechanism rows',
                          'union88': 'sorted teacher_fit union mechanism; exclude internal validation'},
            'parent_fits_changed': False, 'moment_weighting': 'canonical unweighted moments',
            'publication_order': 'registry/source commit and remote-ref verification, then schedule, then fits'}
        record['registration_payload_hash'] = digest(record)
        run.atomic(root/REGISTRY, record)
        return record


def lookup_release(name, anchor, *, study_out=None):
    raw = next((s for s in specs() if s['configuration'] == name and s['anchor'] == anchor), None)
    if raw is None:
        return None
    root = _root(study_out); record = registration(study_out=root)
    return {**raw, 'baseline_registration_sha256': _run().sha(root/REGISTRY),
            'baseline_source_sha256': record['baseline_source_sha256']}


def lookup_spec(name, anchor):
    return None  # these continuous controls have no new Q solution


def freeze_schedule(*, resource_decision):
    run = _run(); root = _root(); record = registration()
    commit = resource_decision.get('registry_commit') if isinstance(resource_decision, dict) else None
    if not isinstance(commit, str) or re.fullmatch('[0-9a-f]{40}', commit) is None:
        raise ValueError('Explicit full published registry_commit is required')
    with run.unit_lock('baseline_supplement/schedule'):
        if (root/SCHEDULE).exists():
            raise FileExistsError('Baseline supplement schedule already frozen')
        if any((root/'private/run'/f'anchor_{a}'/'baseline_supplement').exists() for a in (0, 1, 2)):
            raise ValueError('Supplement schedule must precede fits')
        schedule = {'schema': 1, 'branch': BRANCH, 'created_utc': run.now(),
            'baseline_registration_sha256': run.sha(root/REGISTRY),
            'baseline_source_sha256': record['baseline_source_sha256'],
            'primary_resource_schedule_sha256': run.sha(root/'RESOURCE_SCHEDULE.json'),
            'frozen_before_affected_outcomes': True, 'unit_ids': [s['id'] for s in specs()],
            'resource_decision': copy.deepcopy(resource_decision), 'registry_commit': commit,
            'Git_publication_provenance': 'commit and remote ref checked by root orchestration; remote tar stage has no Git checkout'}
        schedule['schedule_payload_hash'] = digest(schedule)
        run.atomic(root/SCHEDULE, schedule)
        return schedule


def require_scheduled(spec, *, study_out=None):
    root = _root(study_out); run = _run()
    if spec != lookup_release(spec['configuration'], spec['anchor'], study_out=root):
        raise ValueError('Unchanged registered supplement specification required')
    record = json.loads(_owned(root/SCHEDULE, root).read_text())
    if (record.get('schedule_payload_hash') != digest({k: v for k, v in record.items() if k != 'schedule_payload_hash'})
            or record.get('baseline_registration_sha256') != spec['baseline_registration_sha256']
            or record.get('baseline_source_sha256') != spec['baseline_source_sha256']
            or record.get('primary_resource_schedule_sha256') != run.sha(root/'RESOURCE_SCHEDULE.json')
            or record.get('unit_ids') != [s['id'] for s in specs()]
            or record.get('frozen_before_affected_outcomes') is not True
            or not isinstance(record.get('registry_commit'), str)
            or re.fullmatch('[0-9a-f]{40}', record['registry_commit']) is None):
        raise ValueError('Supplement unit is not covered by the frozen published schedule')
    return run.sha(root/SCHEDULE)


def validate_slice(prepared, scope):
    if scope not in SCOPES or 'test' in prepared['ctx']['pools']:
        raise ValueError('Frozen fitting context and registered supplement scope required')
    rows = prepared['ctx']['pools']['representation_fit']; n = len(rows['x']); roles = {}
    if len(np.unique(rows['ids'])) != n:
        raise ValueError('Frozen RF person identity must be unique')
    for name in ('teacher_fit', 'teacher_internal_validation', 'mechanism'):
        ix = np.asarray(prepared['roles'][name])
        if ix.ndim != 1 or ix.dtype.kind not in 'iu' or not len(ix) or len(np.unique(ix)) != len(ix) or np.any((ix < 0) | (ix >= n)):
            raise ValueError('Invalid frozen RF role indices')
        roles[name] = ix
    union = np.concatenate(list(roles.values()))
    if not np.array_equal(np.sort(union), np.arange(n)):
        raise ValueError('Frozen RF roles must partition all rows without overlap')
    households = {name: set(np.asarray(rows['households'])[ix].tolist()) for name, ix in roles.items()}
    names = list(households)
    if any(households[a] & households[b] for i, a in enumerate(names) for b in names[i+1:]):
        raise ValueError('Frozen RF role households overlap')
    return np.sort(roles['mechanism'] if scope == 'mechanism40' else np.r_[roles['teacher_fit'], roles['mechanism']])


def _dependencies(anchor, spec, root):
    run = _run(); base = root/'private/run'/f'anchor_{anchor}'
    marker = _owned(base/'PREPARED.json', root)
    parent = run._read_marker(marker, 'prepare', anchor=anchor)
    cache = _owned(base/'prepared.joblib', root); encoder = _owned(base/'encoder/encoder.joblib', root)
    if parent.get('cache_sha256') != run.sha(cache):
        raise ValueError('Frozen parent cache hash changed')
    return {'anchor': anchor, 'scope': spec['scope'], 'branch': BRANCH,
        'primary_config_hash': digest(configuration()), 'scientific_source_hashes': run.source_fingerprint(),
        'baseline_registration_sha256': spec['baseline_registration_sha256'],
        'baseline_source_sha256': spec['baseline_source_sha256'],
        'baseline_schedule_sha256': require_scheduled(spec, study_out=root),
        'prepared_cache_sha256': run.sha(cache), 'prepared_receipt_sha256': run.sha(marker),
        'encoder_sha256': run.sha(encoder)}


def artifact_receipt(anchor, spec, *, study_out=None):
    root = _root(study_out); run = _run(); expected = _dependencies(anchor, spec, root)
    base = root/'private/run'/f'anchor_{anchor}'/'baseline_supplement'/spec['scope']
    marker = _owned(base/'FITTED.json', root); record = json.loads(marker.read_text())
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Accepted supplement fitting dependency changed')
    if record.get('H_byte_unchanged') is not True or record.get('parent_cache_unchanged') is not True:
        raise ValueError('Supplement parent/H preservation failed')
    run._verify_artifacts(base, record)
    return {**record, 'receipt_sha256': run.sha(marker)}


def load_fitted(anchor, spec, *, study_out=None):
    root = _root(study_out); artifact_receipt(anchor, spec, study_out=root)
    directory = root/'private/run'/f'anchor_{anchor}'/'baseline_supplement'/spec['scope']/spec['method']
    metadata = json.loads(_owned(directory/'diagnostics.json', root).read_text())
    if metadata.get('configuration') != spec['configuration'] or metadata.get('fit_scope') != spec['scope']:
        raise ValueError('Supplement map identity/scope changed')
    if metadata['status'] != 'fitted':
        raise ValueError('Supplement has no fitted map; retain its numerical/infeasibility diagnostic without fallback')
    return baselines.AffineEraser.load(directory)


def _fit_scope(anchor, spec):
    root = _root(); run = _run(); expected = _dependencies(anchor, spec, root)
    base = root/'private/run'/f'anchor_{anchor}'/'baseline_supplement'/spec['scope']
    if base.exists():
        raise FileExistsError('Refusing to overwrite a supplement fitting attempt')
    parent = joblib.load(_owned(root/'private/run'/f'anchor_{anchor}'/'prepared.joblib', root))
    if parent['ctx']['anchor'] != anchor:
        raise ValueError('Frozen prepared anchor changed')
    ix = validate_slice(parent, spec['scope']); rf = parent['ctx']['pools']['representation_fit']
    p = np.asarray(parent['encoded']['representation_fit']['p'], float)
    if p.shape != (len(rf['x']),) or not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError('Frozen teacher probabilities invalid')
    h_before = {key: np.asarray(rf[key]).tobytes() for key in ('ha', 'hb')}
    x = np.column_stack((rf['x'][ix], logit(np.clip(p[ix], 1e-5, 1-1e-5))))
    maps = baselines.fit_supervised_erasers(x, {k: rf['labels'][k][ix] for k in baselines.PROTECTED},
        {k: rf['labels'][k][ix] for k in baselines.TASKS}, out_dir=base)
    slice_path = base/'slice.npz'
    np.savez_compressed(slice_path, rf_row_indices=ix, ids=np.asarray(rf['ids'])[ix],
        households=np.asarray(rf['households'])[ix], weights=np.asarray(rf['weights'])[ix],
        features=x, teacher_probabilities=p[ix], **{'label_'+k: rf['labels'][k][ix]
            for k in (*baselines.PROTECTED, *baselines.TASKS)})
    summaries = {}
    for method, result in maps.items():
        directory = base/method; original = directory/'diagnostics.json'
        original.rename(directory/'canonical_diagnostics.json')
        metadata = result.metadata if isinstance(result, baselines.AffineEraser) else result
        metadata.update(configuration=method+'_'+spec['scope'], fit_scope=spec['scope'],
            fit_row_indices_are='indices relative to supplement slice; slice.npz maps these to original RF rows',
            rf_slice_sha256=run.sha(slice_path), original_RF_row_indices_sha256=baselines._array_hash(ix),
            supplied_households=int(len(np.unique(np.asarray(rf['households'])[ix]))),
            teacher_frozen_without_refit=True, teacher_internal_validation_excluded=True,
            canonical_diagnostics_sha256=run.sha(directory/'canonical_diagnostics.json'))
        run.atomic(original, metadata)
        summaries[method] = {'status': metadata['status'], 'fit_rows': metadata.get('fit_rows'),
                            'provided_rows': metadata.get('provided_rows')}
    if any(value != np.asarray(rf[key]).tobytes() for key, value in h_before.items()):
        raise AssertionError('Supplement fitting modified H')
    if run.sha(root/'private/run'/f'anchor_{anchor}'/'prepared.joblib') != expected['prepared_cache_sha256']:
        raise AssertionError('Supplement fitting modified parent cache')
    run.atomic(base/'FITTED.json', {**expected, 'schema': 1, 'created_utc': run.now(),
        'rf_slice_sha256': run.sha(slice_path), 'provided_people': len(ix),
        'provided_households': int(len(np.unique(np.asarray(rf['households'])[ix]))),
        'methods': summaries, 'H_byte_unchanged': True, 'parent_cache_unchanged':
            run.sha(root/'private/run'/f'anchor_{anchor}'/'prepared.joblib') == expected['prepared_cache_sha256'],
        'artifact_hashes': run._artifact_hashes(base, [base])})


def prepared_for_spec(anchor, prepared, spec, *, allow_fit=True):
    if prepared['ctx']['anchor'] != anchor or (allow_fit and 'test' in prepared['ctx']['pools']):
        raise ValueError('Supplement fitting cannot consume evaluation context or another anchor')
    require_scheduled(spec); root = _root(); marker = root/'private/run'/f'anchor_{anchor}'/'baseline_supplement'/spec['scope']/'FITTED.json'
    if not marker.exists():
        if not allow_fit:
            raise FileNotFoundError('Frozen supplement map is missing')
        with _run().unit_lock(f'baseline_supplement/{anchor}/{spec["scope"]}'):
            if not marker.exists():
                _fit_scope(anchor, spec)
    fitted = load_fitted(anchor, spec)
    return {**prepared, 'erasers': {**prepared['erasers'], spec['configuration']: fitted}}
