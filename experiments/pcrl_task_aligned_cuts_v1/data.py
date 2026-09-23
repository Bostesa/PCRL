"""Hash-verified 2018 archived input access for PCRL task-aligned cuts.

Only the completed diagnosis's pinned private objects are trusted.  No person
records or household identifiers are emitted by the admission command.  The
historical Linux x86 token codes are loaded as stored, never regenerated here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

INDEX_RELATIVE = Path('results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json')
POOLS = ('representation_fit', 'downstream_fit', 'downstream_validation',
         'attacker_fit', 'attacker_validation')
MAPS = {'Q': 'T0_L_0.01_a17', 'D17': 'T0_U_unconstrained_a17',
        'D33': 'T0_U_unconstrained_a33'}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def index(root_or_path: str | Path) -> dict[str, Any]:
    """Read the pinned diagnosis index from an explicit file or study root."""
    source = Path(root_or_path)
    path = source if source.is_file() else source / INDEX_RELATIVE
    value = json.loads(path.read_text())
    if value.get('source_commit') != 'f4bdf4cd5bf74c634feeec50aef78bff249667e4':
        raise ValueError('unexpected historical source pin')
    if value.get('completed_prospective_evidence_commit') != '5e154e5c4fdaeb23d327a0ebefe838525f1a19cb':
        raise ValueError('unexpected completed evidence pin')
    return value


def member_record(value: dict[str, Any], anchor: int, kind: str,
                  name: str | None = None, filename: str | None = None) -> dict[str, Any]:
    if anchor not in (0, 1, 2):
        raise ValueError('only frozen 2018 development anchors 0,1,2 are allowed')
    entry = value['anchors'][str(anchor)]
    if kind in ('encoder', 'prepared', 'historical_fineC_tables') and name is None:
        return entry[kind]
    if kind == 'map' and name in MAPS and filename in ('Q.npz', 'solution.joblib', 'tables.npz'):
        return entry['maps'][name][filename]
    raise ValueError('unknown pinned member')


def verified_member(record: dict[str, Any], *, restore_root: str | Path | None = None) -> Path:
    if restore_root is None:
        path = Path(record['path'])
    else:
        path = Path(restore_root) / record['archive']['member_path']
    if not path.is_file():
        raise FileNotFoundError(f'pinned member unavailable: {record["archive"]["member_path"]}')
    actual = sha256_file(path)
    if actual != record['sha256'] or actual != record['archive']['member_sha256']:
        raise ValueError(f'pinned private member SHA-256 mismatch: {record["archive"]["member_path"]}')
    return path


def verify_source_files(root: str | Path, value: dict[str, Any]) -> dict[str, Any]:
    bad = []
    for rel, expected in value['source_files'].items():
        path = Path(root) / rel
        if not path.is_file() or sha256_file(path) != expected:
            bad.append(rel)
    return {'checked': len(value['source_files']), 'mismatches': bad, 'valid': not bad}


def verify_local_members(value: dict[str, Any], *, restore_root: str | Path | None = None,
                         include_fineC: bool = False) -> dict[str, Any]:
    checked = 0
    missing: list[str] = []
    mismatch: list[str] = []
    for anchor in (0, 1, 2):
        labels = [(kind, None, None) for kind in ('encoder', 'prepared')]
        labels.extend(('map', name, filename) for name in MAPS
                      for filename in ('Q.npz', 'solution.joblib', 'tables.npz'))
        if include_fineC:
            labels.append(('historical_fineC_tables', None, None))
        for kind, name, filename in labels:
            record = member_record(value, anchor, kind, name, filename)
            label = f'{anchor}/{kind}/{name or ""}/{filename or ""}'
            try:
                verified_member(record, restore_root=restore_root)
            except FileNotFoundError:
                missing.append(label)
            except ValueError:
                mismatch.append(label)
            else:
                checked += 1
    return {'checked': checked, 'missing': missing, 'mismatch': mismatch,
            'valid': not missing and not mismatch}


def load_prepared(value: dict[str, Any], anchor: int, *,
                  restore_root: str | Path | None = None) -> dict[str, Any]:
    path = verified_member(member_record(value, anchor, 'prepared'), restore_root=restore_root)
    prepared = joblib.load(path)
    if prepared['ctx']['anchor'] != anchor or tuple(prepared['ctx']['pools']) != POOLS:
        raise ValueError('prepared anchor/pool schema mismatch')
    for name in POOLS:
        pool = prepared['ctx']['pools'][name]
        code = np.asarray(prepared['encoded'][name]['codes']['T0'])
        n = len(code)
        if code.ndim != 1 or n != len(pool['ha']) or n != len(pool['households']):
            raise ValueError(f'invalid frozen T0 code alignment: {name}')
        if np.min(code) < 0 or np.max(code) >= 32:
            raise ValueError(f'invalid frozen T0 support: {name}')
        if pool['ha'].shape != (n, 4) or pool['hb'].shape != (n, 2):
            raise ValueError(f'invalid H shape: {name}')
    n_rep = len(prepared['ctx']['pools']['representation_fit']['ha'])
    roles = prepared['roles']
    expected = {'teacher_fit', 'teacher_internal_validation', 'mechanism'}
    if set(roles) != expected:
        raise ValueError('historical representation roles changed')
    joined = np.concatenate([np.asarray(roles[name], dtype=np.int64)
                             for name in sorted(expected)])
    if not np.array_equal(np.sort(joined), np.arange(n_rep)):
        raise ValueError('teacher/codebook and mechanism roles do not partition representation_fit')
    # The archived pickle bundles the historically used 2018 outer labels with
    # inner arrays.  Its bytes must be deserialized to recover frozen codes,
    # but no pre-lock caller receives those outer labels through this loader.
    # A separately checked post-selection loader is required for final audit.
    outer = prepared['ctx']['pools']['attacker_validation']
    prepared['ctx']['pools']['attacker_validation'] = {
        name: part for name, part in outer.items() if name != 'labels'}
    return prepared


def load_map(value: dict[str, Any], anchor: int, name: str, *,
             restore_root: str | Path | None = None) -> np.ndarray:
    if name not in MAPS:
        raise ValueError('unknown frozen map')
    path = verified_member(member_record(value, anchor, 'map', name, 'Q.npz'),
                           restore_root=restore_root)
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive)
        if len(keys) != 1:
            raise ValueError('unexpected map archive contents')
        q = np.asarray(archive[keys[0]], dtype=np.float64)
    width = 33 if name == 'D33' else 17
    if q.shape != (32, width) or np.min(q) < -1e-8 or np.max(np.abs(q.sum(axis=1) - 1)) > 1e-7:
        raise ValueError('invalid frozen map simplex')
    return q


def token_pool(prepared: dict[str, Any], pool_name: str, q: np.ndarray,
               *, representation_role: str | None = None) -> dict[str, Any]:
    """Private audit input; no code row or Q row is returned to a predictor."""
    if pool_name not in POOLS:
        raise ValueError('unregistered 2018 pool')
    if pool_name == 'representation_fit':
        if representation_role not in ('teacher_fit', 'teacher_internal_validation', 'mechanism'):
            raise ValueError('representation_fit requires an explicit historical role')
        rows = np.asarray(prepared['roles'][representation_role], dtype=np.int64)
    elif representation_role is not None:
        raise ValueError('representation role applies only to representation_fit')
    else:
        rows = slice(None)
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != 32 or np.min(q) < -1e-8 or np.max(np.abs(q.sum(axis=1) - 1)) > 1e-7:
        raise ValueError('invalid channel')
    raw = prepared['ctx']['pools'][pool_name]
    code = np.asarray(prepared['encoded'][pool_name]['codes']['T0'], dtype=np.int64)
    return {'ha': raw['ha'][rows], 'hb': raw['hb'][rows], 'token_probs': q[code[rows]],
            'labels': {name: labels[rows] for name, labels in raw['labels'].items()},
            'weights': raw['weights'][rows], 'ids': raw['ids'][rows],
            'households': raw['households'][rows]}


def coefficient_pool(prepared: dict[str, Any], q: np.ndarray) -> dict[str, Any]:
    """The archived 4,131-row mechanism subset only; never teacher/codebook rows."""
    return token_pool(prepared, 'representation_fit', q,
                      representation_role='mechanism')


def subset_token_pool(pool: dict[str, Any], rows: np.ndarray) -> dict[str, Any]:
    """Slice one already verified audit pool by registered private row indices."""
    rows = np.asarray(rows)
    n = len(pool['ha'])
    if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or
            len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= n)):
        raise ValueError('invalid registered pool subset rows')
    return {'ha': pool['ha'][rows], 'hb': pool['hb'][rows],
            'token_probs': pool['token_probs'][rows],
            'labels': {name: values[rows] for name, values in pool['labels'].items()},
            'weights': pool['weights'][rows], 'ids': pool['ids'][rows],
            'households': pool['households'][rows]}


def household_disjointness(prepared: dict[str, Any]) -> dict[str, int]:
    pools = prepared['ctx']['pools']
    return {f'{a}|{b}': len(set(pools[a]['households']) & set(pools[b]['households']))
            for i, a in enumerate(POOLS) for b in POOLS[i + 1:]}


def global_validation_split(prepared_by_anchor: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Hash-sort household IDs once across all anchors, then split in half.

    Household ID alone is hashed: the same household receives one assignment
    even if present in several anchors.  No label, H coordinate, weight, loss,
    row index or anchor enters the assignment.  Returned row indices stay
    private; only aggregate counts and the assignment digest belong in logs.
    """
    if set(prepared_by_anchor) != {0, 1, 2}:
        raise ValueError('all three archived anchors required for one global split')
    pools = {anchor: prepared_by_anchor[anchor]['ctx']['pools']['downstream_validation']
             for anchor in (0, 1, 2)}
    identifiers = [str(h) for pool in pools.values() for h in pool['households']]
    if not identifiers or any(not h for h in identifiers):
        raise ValueError('missing downstream-validation household ID')
    unique = set(identifiers)
    ordered = sorted(unique, key=lambda h: (hashlib.sha256(h.encode('utf-8')).hexdigest(), h))
    if len(ordered) < 2:
        raise ValueError('at least two households required')
    selection = set(ordered[:len(ordered) // 2])
    pilot = set(ordered[len(ordered) // 2:])
    if selection & pilot:
        raise AssertionError('household split overlap')
    assignment_digest = hashlib.sha256('\n'.join(
        hashlib.sha256(h.encode('utf-8')).hexdigest() + ':' + ('selection' if h in selection else 'pilot')
        for h in ordered).encode('ascii')).hexdigest()
    result: dict[str, Any] = {
        'assignment_sha256': assignment_digest,
        'selection_households_global': len(selection),
        'pilot_households_global': len(pilot),
        'anchors': {},
        'algorithm': 'SHA256(UTF8(household_id)); sort digest then ID; first floor(H/2) selection; rest pilot',
    }
    class_schema = {'same_residence': 2, 'SEX': 2, 'RAC1P': 9}
    for anchor, pool in pools.items():
        households = [str(h) for h in pool['households']]
        choice = np.asarray([h in selection for h in households], dtype=bool)
        rows = {'inner_selection': np.flatnonzero(choice),
                'inner_pilot': np.flatnonzero(~choice)}
        if not len(rows['inner_selection']) or not len(rows['inner_pilot']):
            raise ValueError(f'empty anchor {anchor} validation half')
        support = {}
        for half, indices in rows.items():
            labels = pool['labels']
            support[half] = {}
            for target, n_classes in class_schema.items():
                observed = np.asarray(labels[target])[indices]
                counts = np.bincount(observed, minlength=n_classes)
                if len(counts) != n_classes:
                    raise ValueError(f'unexpected {target} class label')
                support[half][target] = {
                    'counts': counts.astype(int).tolist(),
                    'missing_classes': np.flatnonzero(counts == 0).astype(int).tolist(),
                    'full_schema_supported': bool(np.all(counts > 0)),
                }
        result['anchors'][anchor] = {
            'rows': rows,
            'row_counts': {name: len(indices) for name, indices in rows.items()},
            'household_counts': {'inner_selection': len(set(households) & selection),
                                 'inner_pilot': len(set(households) & pilot)},
            'class_support': support,
        }
    for house in unique:
        if (house in selection) == (house in pilot):
            raise AssertionError('global assignment is incomplete or overlapping')
    return result


def split_summary(split: dict[str, Any]) -> dict[str, Any]:
    """Aggregate-only record of the fixed global household assignment."""
    return {
        'algorithm': split['algorithm'],
        'assignment_sha256': split['assignment_sha256'],
        'selection_households_global': split['selection_households_global'],
        'pilot_households_global': split['pilot_households_global'],
        'anchors': {str(anchor): {
            'row_counts': record['row_counts'],
            'household_counts': record['household_counts'],
            'class_support': record['class_support'],
        } for anchor, record in split['anchors'].items()},
    }
