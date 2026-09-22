"""Registered once-only Branch D staging and closed-writer installation.

Registration hashes accepted artifacts without reading audit outcomes. Execution
uses only the pinned training preparation, original empirical tables and coarse
witness. It neither fits auditors nor opens evaluation. The four nominal map IDs
are unchanged. Accepted retries replace their original slot without consulting
validation; failed retries retain the original. All original map/audit bytes are
preserved when installation occurs. A crashed claimed solve is never retried.

Before live execution, commit the immutable NUMERICAL_RECOVERY.json registration.
The caller also installs the small run._map_receipt verification hook documented
in verify_installed_retry; this module does not edit frozen/core source files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import joblib
import numpy as np

from . import finite, mechanisms, numerical_recovery as recovery, run


SLOTS = ((0, 'Ttask_C_0.0005_a17'), (0, 'Ttask_C_0.01_a17'),
         (0, 'Ttask_L_0.002_a17'), (2, 'Trisk_C_0.01_a17'))
REGISTRY_NAME = 'NUMERICAL_RECOVERY.json'


def _new_json(path, value):
    path = Path(path)
    payload = json.dumps(value, default=run.clean, sort_keys=True, indent=2, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x') as handle:
        os.chmod(path, 0o600)
        handle.write(payload);handle.flush();os.fsync(handle.fileno())


def _tree_hashes(base):
    base = Path(base)
    if base.is_symlink() or not base.is_dir():
        raise ValueError('Artifact directory missing or symlinked')
    files = {}
    for path in sorted(base.rglob('*')):
        if path.is_symlink(): raise ValueError('Symlink inside immutable artifacts')
        if path.is_file(): files[str(path.relative_to(base))] = run.sha(path)
    if not files: raise ValueError('Empty artifact directory')
    return files


def _verify_files(base, expected):
    base = Path(base).resolve()
    for relative, digest in expected.items():
        path = base/relative
        if path.is_symlink() or not path.resolve().is_relative_to(base) or run.sha(path) != digest:
            raise ValueError(f'Pinned artifact hash changed: {relative}')


def _recovery_sources():
    return {Path(p).name: run.sha(p) for p in (__file__, recovery.__file__)}


def _owned(path, root):
    path = Path(path)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact/source path escapes the explicitly owned root')
    return path


def _source_context(study_out=None, source_root=None):
    out = run.OUT if study_out is None else Path(study_out).resolve()
    if source_root is None:
        science, recovery_sources, runner = run.source_fingerprint(), _recovery_sources(), Path(run.__file__)
    else:
        root = Path(source_root).resolve()
        if not out.is_relative_to(root): raise ValueError('Relocated study output escapes explicit source root')
        code = root/'experiments'/run.STUDY
        names = ('config', 'data', 'encoding', 'audits', 'baselines', 'finite', 'mechanisms', 'evaluation')
        science = {n: run.sha(_owned(code/(n+'.py'), root)) for n in names}
        recovery_sources = {n: run.sha(_owned(code/n, root)) for n in ('numerical_recovery.py', 'numerical_recovery_run.py')}
        runner = _owned(code/'run.py', root)
    return {'config_hash': run.digest(run.configuration()),
            'scientific_source_hashes': science,
            'recovery_source_hashes': recovery_sources,
            'runner_sha256': run.sha(runner),
            'resource_schedule_sha256': run.sha(_owned(out/'RESOURCE_SCHEDULE.json', out))}


def _unfrozen():
    selection = run.OUT/'SELECTION.json'
    if selection.exists() and json.loads(selection.read_text()).get('selection_frozen'):
        raise RuntimeError('Numerical recovery cannot modify a frozen selection')
    if any((run.OUT/'private/run'/f'anchor_{a}'/'evaluation').exists() for a in (0, 1, 2)):
        raise RuntimeError('Numerical recovery cannot run after evaluation directories exist')


def _spec(anchor, name):
    if (anchor, name) not in SLOTS: raise ValueError('Slot is outside the four registered numerical failures')
    matches = [s for s in run.configuration()['maps'] if s['anchor'] == anchor and s['configuration'] == name]
    if len(matches) != 1: raise ValueError('Exact primary map specification missing')
    return matches[0]


def _map_snapshot(anchor, name):
    base = _owned(run.anchor_dir(anchor)/'maps'/name, run.OUT)
    spec = next(s for s in run.configuration()['maps'] if s['anchor'] == anchor and s['configuration'] == name)
    record = run._read_marker(base/'ACCEPTED.json', 'map', anchor=anchor, name=name)
    if record.get('sha256') != run.sha(base/'solution.joblib') or record.get('spec_hash') != run.digest(spec):
        raise ValueError('Accepted map identity/hash changed')
    return {'receipt_sha256': run.sha(base/'ACCEPTED.json'), 'solution_sha256': record['sha256'],
            'files': _tree_hashes(base), 'receipt': record}


def register():
    """Pin exactly the four completed feasible-witness slots; never solve/read scores."""
    _unfrozen()
    destination = run.OUT/REGISTRY_NAME
    if destination.exists(): raise FileExistsError(destination)
    context = _source_context()
    slots = []
    for anchor, name in SLOTS:
        spec = _spec(anchor, name);base = _owned(run.anchor_dir(anchor), run.OUT)
        preparation = run._read_marker(base/'PREPARED.json', 'prepare', anchor=anchor)
        if preparation.get('cache_sha256') != run.sha(base/'prepared.joblib'):
            raise ValueError('Prepared cache hash changed')
        original = _map_snapshot(anchor, name)
        metadata = json.loads((base/'maps'/name/'metadata.json').read_text())
        if metadata.get('status') != 'feasible_witness' or not metadata.get('feasible') or metadata.get('optimal'):
            raise ValueError('Only the predetermined feasible_witness maps may be retried')
        coarse_name = run.mechanism_id('T0', spec['policy'], spec['budget'], spec.get('max_actions', 17))
        coarse = _map_snapshot(anchor, coarse_name)
        if (original['receipt'].get('prepared_cache_sha256') != preparation['cache_sha256']
                or original['receipt'].get('coarse_configuration') != coarse_name
                or original['receipt'].get('coarse_solution_sha256') != coarse['solution_sha256']):
            raise ValueError('Original preparation/coarse dependency changed')
        audit_base = _owned(base/'audits'/name, run.OUT)
        audit = run._read_marker(audit_base/'COMPLETE.json', 'audit', anchor=anchor, name=name)
        if (audit.get('map_solution_sha256') != original['solution_sha256']
                or audit.get('registry_sha256') != run.sha(audit_base/'registry.joblib')
                or audit.get('prepared_cache_sha256') != preparation['cache_sha256']):
            raise ValueError('Original completed audit dependencies changed')
        slots.append({'anchor': anchor, 'configuration': name, 'spec': spec, 'spec_hash': run.digest(spec),
            'original': original, 'coarse_configuration': coarse_name, 'coarse': coarse,
            'relocation': {'original_study_root': str(run.OUT.resolve()),
                'original_map_relative': f'private/run/anchor_{anchor}/maps/{name}',
                'original_audit_relative': f'private/run/anchor_{anchor}/audits/{name}',
                'archived_map_relative': f'private/numerical_originals/anchor_{anchor}/{name}/map',
                'archived_audit_relative': f'private/numerical_originals/anchor_{anchor}/{name}/audit',
                'registry_execution': 'original registry bytes preserve old model paths; explicit original-prefix relocation is required'},
            'prepared': {'receipt_sha256': run.sha(base/'PREPARED.json'), 'cache_sha256': preparation['cache_sha256'],
                         'files': {**preparation['artifact_hashes'], 'PREPARED.json': run.sha(base/'PREPARED.json')}},
            'audit': {'receipt_sha256': run.sha(audit_base/'COMPLETE.json'), 'registry_sha256': audit['registry_sha256'],
                      'files': _tree_hashes(audit_base)}})
    if _source_context() != context: raise ValueError('Sources changed during registration')
    result = {'schema': 1, 'branch': 'D', 'created_utc': run.now(), **context, 'slots': slots,
        'solver': 'CLARABEL', 'attempts_per_slot': 1,
        'replacement_rule': 'install every independently accepted retry; rejected retry retains original; no outcome selection',
        'original_preservation': 'complete byte-identical map and audit directories under private/numerical_originals',
        'evaluation_read': False, 'new_nominal_configurations': 0}
    _new_json(destination, result)
    return result


def _registration(anchor, name, *, study_out=None, source_root=None):
    _spec(anchor, name)
    out = run.OUT if study_out is None else Path(study_out).resolve()
    path = _owned(out/REGISTRY_NAME, out)
    registry = json.loads(path.read_text())
    if (registry.get('schema') != 1 or registry.get('branch') != 'D'
            or registry.get('attempts_per_slot') != 1
            or [(s['anchor'], s['configuration']) for s in registry.get('slots', [])] != list(SLOTS)):
        raise ValueError('Numerical registry differs from exact four-slot plan')
    for key, value in _source_context(study_out=out, source_root=source_root).items():
        if registry.get(key) != value: raise ValueError(f'Registered source/configuration changed: {key}')
    slot = next(s for s in registry['slots'] if (s['anchor'], s['configuration']) == (anchor, name))
    if slot['spec'] != _spec(anchor, name) or slot['spec_hash'] != run.digest(slot['spec']):
        raise ValueError('Registered map specification changed')
    return registry, slot, run.sha(path)


def _dependencies(slot, *, study_out=None):
    out = run.OUT if study_out is None else Path(study_out).resolve()
    anchor = slot['anchor'];base = _owned(out/'private/run'/f'anchor_{anchor}', out)
    _verify_files(base, slot['prepared']['files'])
    coarse = _owned(base/'maps'/slot['coarse_configuration'], out)
    if _tree_hashes(coarse) != slot['coarse']['files']: raise ValueError('Coarse witness artifact changed')


def _originals(slot):
    base = run.anchor_dir(slot['anchor']);name = slot['configuration']
    _dependencies(slot)
    if _tree_hashes(base/'maps'/name) != slot['original']['files']: raise ValueError('Original map artifact changed')
    if _tree_hashes(base/'audits'/name) != slot['audit']['files']: raise ValueError('Original audit artifact changed')


def _inputs(slot):
    """Reconstruct original solver arguments from pinned files, with exact equality."""
    base = run.anchor_dir(slot['anchor']);name = slot['configuration'];spec = slot['spec']
    prepared = joblib.load(base/'prepared.joblib')
    if set(prepared['ctx']['pools']) != set(run.FIT_POOLS):
        raise ValueError('Prepared context is not exactly the original five fitting pools')
    original = joblib.load(base/'maps'/name/'solution.joblib')
    if original.get('status') != 'feasible_witness' or not original.get('feasible') or original.get('optimal'):
        raise ValueError('Original solution is not the registered feasible witness')
    family = spec['input'];table = prepared['tables'][family];encoder = prepared['encoder']
    expected = {k: table[k] for k in ('cost_U', 'cost_W', 'cost', 'state_mass')}
    expected.update({'joint/'+k: p for k, p in table['roles'].items()})
    expected.update({'support/'+k+'/'+field: v for k, fields in table['support'].items() for field, v in fields.items()})
    with np.load(base/'maps'/name/'tables.npz', allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected): raise ValueError('Original table field set differs from preparation')
        if any(not np.array_equal(arrays[k], v) for k, v in expected.items()):
            raise ValueError('Original empirical tables differ from preparation')
    if not np.array_equal(original['cost'], table['cost']): raise ValueError('Original cost differs from saved table')
    selected = {k: p for k, p in table['roles'].items() if spec['policy'] == 'C' or k.startswith('A/')}
    required = {f'{view}/{label}/{weight}' for view in (('A', 'AB') if spec['policy'] == 'C' else ('A',))
                for label in ('SEX', 'RAC1P') for weight in ('U', 'W')}
    if set(selected) != required or sorted(selected) != original['constrained_roles']:
        raise ValueError('Selected role laws differ from exact original policy')
    parents = np.asarray(encoder.code.parents(family))
    if (not np.array_equal(parents, original['support']['parent'])
            or not np.array_equal(table['state_mass'], original['support']['state_mass'])):
        raise ValueError('Original parent/support rules changed')
    coarse = joblib.load(base/'maps'/slot['coarse_configuration']/'solution.joblib')
    if (coarse.get('input') != 'T0' or coarse.get('policy') != spec['policy']
            or coarse.get('budget') != spec['budget'] or coarse.get('actions') != spec.get('max_actions', 17)
            or not coarse.get('feasible')): raise ValueError('Coarse witness specification mismatch')
    witness, embedding = mechanisms._embedding(prepared['ctx'], encoder, prepared['encoded'],
        prepared['tables'], family, coarse['Q'], spec.get('max_actions', 17))
    if embedding != original['embedding']: raise ValueError('Original embedding check changed')
    for key, value in (('input', family), ('policy', spec['policy']), ('budget', spec['budget']),
                       ('configuration', name), ('actions', spec.get('max_actions', 17))):
        if original[key] != value: raise ValueError('Original map metadata changed')
    if original['aggregation'] != table['aggregation'] or original['population'] != table['population']:
        raise ValueError('Original aggregation/population metadata changed')
    return original, table, selected, parents, encoder.dictionaries[spec.get('max_actions', 17)]['zero_action'], witness


def _stage(anchor, name, *, study_out=None):
    out = run.OUT if study_out is None else Path(study_out).resolve()
    return _owned(out/'private/numerical_recovery'/f'anchor_{anchor}'/name, out)


def execute(anchor, name):
    """Claim one slot atomically and stage a single retry; never replace originals."""
    _unfrozen();registry, slot, registration_sha = _registration(anchor, name);_originals(slot)
    stage = _stage(anchor, name)
    stage.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage.mkdir(mode=0o700, exist_ok=False)  # persists even after a crash: no second solve
    _new_json(stage/'CLAIM.json', {'anchor': anchor, 'configuration': name, 'claimed_utc': run.now(),
        'pid': os.getpid(), 'registration_sha256': registration_sha, 'attempt': 1})
    tick = time.perf_counter()
    try:
        original, table, selected, parents, zero_action, witness = _inputs(slot)
        result = recovery.solve_normalized_retry(table['cost'], selected, slot['spec']['budget'],
            parent=parents, state_mass=table['state_mass'], zero_action=zero_action, embedded_q=witness)
        result.update({k: copy.deepcopy(original[k]) for k in ('input', 'policy', 'budget', 'configuration',
            'actions', 'constrained_roles', 'embedding', 'aggregation', 'population', 'constant_action', 'cost')})
        if result['accepted']:
            q = mechanisms._probabilities(result['Q'], table['cost'].shape[0])
            information = {k: finite.cmi(p, q) for k, p in table['roles'].items()}
            objective = float(np.sum(table['cost']*q))
            if (abs(objective-result['objective']) > 1e-12 or
                    any(information[k] > slot['spec']['budget']+finite.ACCEPTANCE_TOLERANCES['cmi'] for k in selected)):
                raise AssertionError('Staged retry failed independent map checks')
            result.update(independent_cmi=information, independent_objective=objective)
        destination = stage/'map';destination.mkdir(mode=0o700)
        shutil.copy2(run.anchor_dir(anchor)/'maps'/name/'tables.npz', destination/'tables.npz')
        if result['Q'] is not None: np.savez_compressed(destination/'Q.npz', Q=result['Q'])
        run.atomic_joblib(destination/'solution.joblib', result)
        _new_json(destination/'metadata.json', {k: v for k, v in result.items() if k not in ('Q', 'cost')})
        _originals(slot)
        _, _, after = _registration(anchor, name)
        if after != registration_sha: raise ValueError('Registration changed during retry')
        complete = {'schema': 1, 'anchor': anchor, 'configuration': name, 'registration_sha256': registration_sha,
            'accepted': bool(result['accepted']), 'feasible': bool(result['feasible']), 'optimal': bool(result['optimal']),
            'status': result['status'], 'solver_status': result['solver_status'], 'attempts': 1,
            'created_utc': run.now(), 'seconds': time.perf_counter()-tick,
            'original_accepted_sha256': slot['original']['receipt_sha256'],
            'original_solution_sha256': slot['original']['solution_sha256'],
            'recovery_source_hashes': registry['recovery_source_hashes'],
            'artifact_hashes': {'map/'+k: v for k, v in _tree_hashes(destination).items()},
            'originals_unchanged': True, 'evaluation_read': False}
        _new_json(stage/'COMPLETE.json', complete)
        return complete
    except BaseException as exc:
        _new_json(stage/'FAILED.json', {'created_utc': run.now(), 'error_type': type(exc).__name__,
            'message': str(exc), 'attempt_consumed': True, 'originals_replaced': False})
        raise


def _completion(anchor, name, registration_sha, *, study_out=None):
    stage = _stage(anchor, name, study_out=study_out);record = json.loads((stage/'COMPLETE.json').read_text())
    if (record.get('anchor') != anchor or record.get('configuration') != name
            or record.get('registration_sha256') != registration_sha or record.get('attempts') != 1):
        raise ValueError('Retry completion identity/registration mismatch')
    _verify_files(stage, record['artifact_hashes'])
    if _tree_hashes(stage/'map') != {k.removeprefix('map/'): v for k, v in record['artifact_hashes'].items()}:
        raise ValueError('Staged retry file set changed')
    metadata = json.loads((stage/'map/metadata.json').read_text())
    if any(metadata.get(key) != record.get(key) for key in ('configuration', 'accepted', 'feasible', 'optimal', 'status', 'solver_status')):
        raise ValueError('Retry completion disagrees with hashed solver metadata')
    if record['accepted'] and ('map/Q.npz' not in record['artifact_hashes'] or not record['feasible']):
        raise ValueError('Accepted completion lacks a feasible channel artifact')
    return record


@contextmanager
def _closed_writers(writers_closed):
    if writers_closed is not True: raise RuntimeError('Installation requires scientific writers closed')
    private = run.OUT/'private';private.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(run.unit_lock('programme/selection_freeze'))
        freeze_lock = private/'locks'/(hashlib.sha256(b'programme/selection_freeze').hexdigest()+'.lock')
        paths = [private/'scheduler.lock']
        directory = private/'locks';directory.mkdir(exist_ok=True)
        paths += [p for p in directory.glob('*.lock') if p != freeze_lock]
        for anchor, name in SLOTS:
            paths += [directory/(hashlib.sha256(f'{kind}/{anchor}/{name}'.encode()).hexdigest()+'.lock')
                      for kind in ('map', 'audit')]
        for path in sorted(set(paths)):
            handle = stack.enter_context(path.open('a+'))
            try: fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc: raise RuntimeError(f'Scientific writer lock held: {path.name}') from exc
        state_path = private/'SCHEDULER.json'
        if not state_path.exists(): raise RuntimeError('Closed scheduler receipt missing')
        state = json.loads(state_path.read_text())
        if state.get('status') not in ('complete', 'finished_with_incidents') or state.get('active_jobs'):
            raise RuntimeError('Scientific scheduler is not closed')
        _unfrozen()
        yield


def install(anchor, name, *, writers_closed=False):
    """Install an accepted retry unconditionally, archiving exact original directories.

    No validation loss is opened or compared. A failed retry leaves its original
    and audit in place. Crashed partial installation fails closed for manual
    recovery; caught exceptions attempt byte-preserving rollback, never refit.
    """
    registry, slot, registration_sha = _registration(anchor, name)
    stage = _stage(anchor, name)
    with _closed_writers(writers_closed):
        complete = _completion(anchor, name, registration_sha)
        receipt_path = stage/'INSTALLATION.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('decision') == 'installed_accepted_retry':
                record = json.loads((run.anchor_dir(anchor)/'maps'/name/'ACCEPTED.json').read_text())
                verify_installed_retry(anchor, name, record)
            elif receipt.get('decision') == 'retained_original': _originals(slot)
            else: raise RuntimeError('Partial installation requires manual recovery; no automatic retry')
            return receipt
        _originals(slot)
        if not complete['accepted']:
            result = {'anchor': anchor, 'configuration': name, 'decision': 'retained_original',
                'created_utc': run.now(), 'retry_receipt_sha256': run.sha(stage/'COMPLETE.json')}
            _new_json(receipt_path, result);return result
        if not complete['feasible']: raise ValueError('Accepted retry must also be feasible')
        base = run.anchor_dir(anchor);active_map = base/'maps'/name;active_audit = base/'audits'/name
        backup = _owned(run.OUT/'private/numerical_originals'/f'anchor_{anchor}'/name, run.OUT)
        candidate = stage/'installation_candidate'
        if backup.exists() or candidate.exists(): raise RuntimeError('Partial installation artifacts already exist')
        shutil.copytree(stage/'map', candidate)
        pin = {'registration_sha256': registration_sha, 'retry_receipt_sha256': run.sha(stage/'COMPLETE.json'),
            'original_accepted_sha256': slot['original']['receipt_sha256'],
            'original_solution_sha256': slot['original']['solution_sha256'],
            'recovery_source_hashes': registry['recovery_source_hashes']}
        accepted = {**copy.deepcopy(slot['original']['receipt']), **run._provenance('map'),
            'created_utc': run.now(), 'sha256': run.sha(candidate/'solution.joblib'),
            'seconds': complete['seconds'], 'artifact_hashes': _tree_hashes(candidate), 'numerical_retry': pin}
        _new_json(candidate/'ACCEPTED.json', accepted)
        backup.mkdir(parents=True, mode=0o700, exist_ok=False)
        journal = {'anchor': anchor, 'configuration': name, 'decision': 'installation_started',
                   'created_utc': run.now(), **pin}
        _new_json(receipt_path, journal)
        moved_map = moved_audit = installed = False
        try:
            os.rename(active_map, backup/'map');moved_map = True
            os.rename(active_audit, backup/'audit');moved_audit = True
            os.rename(candidate, active_map);installed = True
            verify_installed_retry(anchor, name, accepted)
            result = {**journal, 'decision': 'installed_accepted_retry', 'installed_utc': run.now(),
                'new_solution_sha256': accepted['sha256'], 'new_accepted_sha256': run.sha(active_map/'ACCEPTED.json'),
                'reaudit_required': True, 'original_map_files': slot['original']['files'],
                'original_audit_files': slot['audit']['files'], 'original_path_relocation': slot['relocation']}
            run.atomic(receipt_path, result)
            return result
        except BaseException as exc:
            # All destinations were checked absent while scheduler/unit locks are held.
            if installed: os.rename(active_map, candidate)
            if moved_audit: os.rename(backup/'audit', active_audit)
            if moved_map: os.rename(backup/'map', active_map)
            run.atomic(receipt_path, {**journal, 'decision': 'rolled_back_requires_manual_review',
                'error_type': type(exc).__name__, 'message': str(exc)})
            raise


def verify_installed_retry(anchor, name, record, *, study_out=None, source_root=None):
    """Receipt hook for run._map_receipt, after its normal artifact verification.

    Add only::

        if record.get('numerical_retry') is not None:
            from .numerical_recovery_run import verify_installed_retry
            verify_installed_retry(anchor, name, record)

    Restored adapters must pass both ``study_out=layout.out`` and
    ``source_root=layout.root``; every artifact/source path then uses these
    explicit roots without consulting original artifact directories.
    No recursion into run._map_receipt, source mutation, fitting or score reads.
    """
    if (study_out is None) != (source_root is None):
        raise ValueError('Restored verification requires both explicit study_out and source_root')
    out = run.OUT if study_out is None else Path(study_out).resolve()
    registry, slot, registration_sha = _registration(anchor, name, study_out=out, source_root=source_root)
    _dependencies(slot, study_out=out)
    complete = _completion(anchor, name, registration_sha, study_out=out)
    stage = _stage(anchor, name, study_out=out)
    expected = {'registration_sha256': registration_sha, 'retry_receipt_sha256': run.sha(stage/'COMPLETE.json'),
        'original_accepted_sha256': slot['original']['receipt_sha256'],
        'original_solution_sha256': slot['original']['solution_sha256'],
        'recovery_source_hashes': registry['recovery_source_hashes']}
    if record.get('numerical_retry') != expected or not complete['accepted'] or not complete['feasible']:
        raise ValueError('Installed numerical retry pin/acceptance mismatch')
    if record.get('sha256') != complete['artifact_hashes']['map/solution.joblib']:
        raise ValueError('Installed solution differs from accepted retry')
    backup = _owned(out/'private/numerical_originals'/f'anchor_{anchor}'/name, out)
    if _tree_hashes(backup/'map') != slot['original']['files'] or _tree_hashes(backup/'audit') != slot['audit']['files']:
        raise ValueError('Preserved numerical originals changed')
    active = _owned(out/'private/run'/f'anchor_{anchor}'/'maps'/name, out)
    expected_files = {k.removeprefix('map/'): v for k, v in complete['artifact_hashes'].items()}
    _verify_files(active, expected_files)
    if record.get('artifact_hashes') != expected_files: raise ValueError('Installed artifact manifest differs from staged retry')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('register', 'execute', 'install'))
    parser.add_argument('--anchor', type=int);parser.add_argument('--name')
    parser.add_argument('--writers-closed', action='store_true')
    args = parser.parse_args()
    if args.command == 'register': result = register()
    elif args.command == 'execute': result = execute(args.anchor, args.name)
    else: result = install(args.anchor, args.name, writers_closed=args.writers_closed)
    print(json.dumps({k: result[k] for k in ('branch', 'anchor', 'configuration', 'status', 'accepted', 'decision') if k in result}))
