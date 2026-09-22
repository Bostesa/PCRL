"""Public registered release inventory; never inspect fits, people or outcomes.

Scheduling flags describe existing frozen schedules. This module cannot add or
change a scheduled scientific unit. ``build_ledger`` is read-only; ``regenerate``
preserves any replaced public ledger before atomically publishing its successor.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from .config import OUT, PRIMARY, SECONDARY, UTILITY, configuration, digest, release_ledger

EXTENSIONS = (
    ('A', 'EXTRA_CONFIGS.json', 'EXTRA_RESOURCE_SCHEDULE.json', 'extra_registration_sha256', 'branch_source_sha256'),
    ('C', 'ROBUSTNESS_CONFIGS.json', 'ROBUSTNESS_RESOURCE_SCHEDULE.json', 'robustness_registration_sha256', 'robustness_source_sha256'),
    ('baseline_supplement', 'BASELINE_SUPPLEMENTS.json', 'BASELINE_SUPPLEMENT_SCHEDULE.json',
     'baseline_registration_sha256', 'baseline_source_sha256'),
)


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _read(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Required public inventory input missing or escaping its root: '+name)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Public inventory input must be an object: '+name)
    return value, file_hash(path)


def _payload(record, field):
    if record.get(field) != digest({k: v for k, v in record.items() if k != field}):
        raise ValueError('Frozen registration/schedule payload hash changed')


def _hash(value):
    if not isinstance(value, str) or re.fullmatch('[a-f0-9]{64}', value) is None:
        raise ValueError('Invalid registered source hash')
    return value


def build_ledger(*, study_out=None):
    """Read only public CONFIGS, registrations and schedules; return all units."""
    root = Path(OUT if study_out is None else study_out).resolve()
    cfg, config_sha = _read(root, 'CONFIGS.json')
    if digest(cfg) != digest(configuration()):
        raise ValueError('Primary configuration differs from its immutable definition')
    primary = release_ledger(cfg)
    schedule, schedule_sha = _read(root, 'RESOURCE_SCHEDULE.json')
    p = schedule.get('primary', {})
    if (schedule.get('config_hash') != digest(cfg) or schedule.get('frozen_before_comparative_outcomes') is not True
            or p.get('schedule') != 'complete' or p.get('nominal_maps') != len(cfg['maps'])
            or p.get('release_anchor_units') != len(primary['records'])):
        raise ValueError('Primary schedule does not cover the registered complete primary inventory')
    inputs = {'CONFIGS.json': config_sha, 'RESOURCE_SCHEDULE.json': schedule_sha}
    records = []; seen = set(); seen_ids = set()
    all_roles = ['attack:'+r for r in PRIMARY+SECONDARY]+['utility:'+r for r in UTILITY]

    def append(raw, kind, branch, registered_file, registered_sha, resource_file, resource_sha,
               scheduled, source_hashes, registration_source_sha):
        name, anchor, ident = raw.get('configuration'), raw.get('anchor'), raw.get('id')
        if (not isinstance(name, str) or re.fullmatch('[A-Za-z0-9_.-]+', name) is None
                or type(anchor) is not int or anchor not in (0, 1, 2) or ident != f'{name}/anchor_{anchor}'):
            raise ValueError('Invalid registered release identity')
        if (name, anchor) in seen or ident in seen_ids:
            raise ValueError('Duplicate registered release identity: '+ident)
        seen.add((name, anchor)); seen_ids.add(ident)
        records.append({**copy.deepcopy(raw), 'kind': kind, 'branch': branch, 'status': 'registered',
            'scheduled': scheduled, 'resource_status': 'scheduled' if scheduled else 'unscheduled_resource',
            'audit_roles': list(PRIMARY+SECONDARY), 'utility_roles': list(UTILITY), 'roles': all_roles.copy(),
            'registration_file': registered_file, 'registration_sha256': registered_sha,
            'registration_source_sha256': registration_source_sha, 'scientific_source_hashes': source_hashes,
            'schedule_file': resource_file, 'schedule_sha256': resource_sha})

    for raw in primary['records']:
        append(raw, raw['kind'], 'primary', 'CONFIGS.json', config_sha, 'RESOURCE_SCHEDULE.json',
               schedule_sha, True, {'config': file_hash(Path(__file__).with_name('config.py'))},
               file_hash(Path(__file__).with_name('config.py')))
    for branch, registry_name, schedule_name, pin_key, source_key in EXTENSIONS:
        registry_path, schedule_path = root/registry_name, root/schedule_name
        if not registry_path.exists():
            if schedule_path.exists() or (branch == 'baseline_supplement' and (root/'AMENDMENT_6_BASELINE_FAIRNESS.md').exists()):
                raise ValueError('Schedule/amendment has no corresponding public registration')
            continue
        registry, registry_sha = _read(root, registry_name); _payload(registry, 'registration_payload_hash')
        if (registry.get('registered') is not True or registry.get('branch') != branch
                or registry.get('primary_config_hash') != digest(cfg)
                or registry.get('primary_resource_schedule_sha256') != schedule_sha):
            raise ValueError('Extension registration differs from frozen primary provenance')
        source_sha = _hash(registry.get(source_key)); sources = registry.get('scientific_source_hashes')
        if not isinstance(sources, dict) or not sources:
            raise ValueError('Extension scientific source pins missing')
        for value in sources.values():
            _hash(value)
        rows = []
        for field, kind in (('maps', 'finite_map'), ('controls', 'control')):
            if not isinstance(registry.get(field), list):
                raise ValueError('Extension registry needs explicit maps and controls')
            rows += [(row, kind) for row in registry[field]]
        ids = [row.get('id') for row, _ in rows]
        if any(not isinstance(ident, str) for ident in ids) or len(ids) != len(set(ids)):
            raise ValueError('Duplicate/invalid extension registration identity')
        selected = set(); resource_sha = None
        if schedule_path.exists():
            resource, resource_sha = _read(root, schedule_name); _payload(resource, 'schedule_payload_hash')
            scheduled_ids = resource.get('unit_ids')
            if (not isinstance(scheduled_ids, list) or any(not isinstance(x, str) for x in scheduled_ids)
                    or len(scheduled_ids) != len(set(scheduled_ids)) or not set(scheduled_ids) <= set(ids)):
                raise ValueError('Unknown or duplicate scheduled release identity')
            if (resource.get('branch') != branch or resource.get(pin_key) != registry_sha
                    or resource.get('primary_resource_schedule_sha256') != schedule_sha
                    or resource.get('frozen_before_affected_outcomes') is not True
                    or (source_key in resource and resource[source_key] != source_sha)):
                raise ValueError('Extension schedule registration/source pin changed')
            selected = set(scheduled_ids); inputs[schedule_name] = resource_sha
        inputs[registry_name] = registry_sha
        for raw, kind in rows:
            if raw.get('branch') != branch or (raw.get('kind') is not None and raw['kind'] != kind):
                raise ValueError('Registered release branch/kind changed')
            append(raw, kind, branch, registry_name, registry_sha, schedule_name, resource_sha,
                   raw['id'] in selected, copy.deepcopy(sources), source_sha)
    result = {'schema': 2, 'inventory_only': True, 'schedule_changed': False,
        'nominal_primary_Q_maps': primary['nominal_primary_Q_maps'],
        'nominal_primary_release_anchor_records': len(primary['records']),
        'nominal_Q_maps': sum(r['kind'] == 'finite_map' for r in records),
        'scheduled_Q_maps': sum(r['kind'] == 'finite_map' and r['scheduled'] for r in records),
        'nominal_release_anchor_records': len(records),
        'scheduled_release_anchor_records': sum(r['scheduled'] for r in records),
        'resource_unscheduled_release_anchor_records': sum(not r['scheduled'] for r in records),
        'nominal_role_audit_units': len(records)*len(all_roles),
        'scheduled_role_audit_units': sum(r['scheduled'] for r in records)*len(all_roles),
        'audit_unit_definition': primary['audit_unit_definition'], 'records': records,
        'input_sha256': inputs, 'generator_source_sha256': file_hash(__file__),
        'scope': 'All prospectively registered units; scheduling status only, not completion or scientific success.'}
    result['inventory_payload_hash'] = digest(result)
    return result


def _atomic(path, data):
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def regenerate(*, study_out=None):
    """Atomically replace ALL_RELEASES, preserving predecessor bytes/provenance."""
    root = Path(OUT if study_out is None else study_out).resolve()
    if not root.is_dir():
        raise ValueError('Existing public study directory required')
    with (root/'.ALL_RELEASES.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = build_ledger(study_out=root); path = root/'ALL_RELEASES.json'
        if path.exists():
            previous, prior_sha = _read(root, path.name)
            if previous.get('inventory_payload_hash') == result['inventory_payload_hash']:
                core = {k: v for k, v in previous.items() if k not in ('previous_ledger_provenance', 'primary_ledger_provenance')}
                if core != result:
                    raise ValueError('Existing combined ledger content disagrees with its inventory hash')
                return previous
            if not isinstance(previous.get('records'), list) or previous.get('nominal_primary_Q_maps') != 81:
                raise ValueError('Refusing to replace an unrecognized public release ledger')
            saved = root/f'ALL_RELEASES.previous.{prior_sha}.json'
            data = path.read_bytes()
            if saved.exists():
                if file_hash(saved) != prior_sha:
                    raise ValueError('Preserved previous ledger hash changed')
            else:
                _atomic(saved, data)
            provenance = {'sha256': prior_sha, 'preserved_file': saved.name,
                'nominal_release_anchor_records': previous.get('nominal_release_anchor_records'),
                'input_sha256': previous.get('input_sha256', {})}
            result['previous_ledger_provenance'] = provenance
            result['primary_ledger_provenance'] = previous.get('primary_ledger_provenance', provenance)
        _atomic(path, (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-out', type=Path, default=OUT)
    parser.add_argument('--check-only', action='store_true', help='derive inventory without writing ALL_RELEASES')
    args = parser.parse_args(argv)
    result = build_ledger(study_out=args.study_out) if args.check_only else regenerate(study_out=args.study_out)
    print(json.dumps({k: result[k] for k in ('nominal_primary_Q_maps', 'nominal_Q_maps', 'scheduled_Q_maps',
        'nominal_release_anchor_records', 'scheduled_release_anchor_records', 'resource_unscheduled_release_anchor_records')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
