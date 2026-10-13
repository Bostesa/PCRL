"""Metadata admission and role-specific prefit input guards.

Importing this module opens no source artifact. Metadata inventory never hashes
or deserializes checkpoints or person arrays. Numerical workers must explicitly
call load_role or verify_pin; INNER labels have no prefit capability here.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

ROLES = {'OSF_DEFENSE_FIT': (15434, 15428), 'ATTACK_FIT_NEW': (4064, 4061),
         'INNER_SELECTION': (2235, 2234)}
FOLD_SALT = 'IVR1|FIT-crossfold|20261010|'
FEATURE_FIELDS = {'row_id', 'unit', 'X', 'feature_names'}
REFERENCE_FIELDS = {'row_id', 'unit', 'U0_1', 'U0_2', 'Ucal_1', 'Ucal_2', 'd_1', 'd_2'}
LABELS = {'income', 'occupation', 'sex'}

def inner_truth_path(path):
    p = Path(path)
    low = p.as_posix().lower()
    stem = p.stem.lower()
    return ('inner_selection' in low or 'inner_truth' in low) and any(
        stem == label or stem.endswith('_' + label) for label in LABELS)


def admitted_path(path):
    """Refuse forbidden paths and symlink ancestors before any byte access."""
    p = Path(os.path.abspath(os.path.expanduser(str(path))))
    low = p.as_posix().lower()
    if any(x in low for x in ('head_validation', 'old_assessment',
                             'osf_development_assessment', 'all_role', 'all-role')):
        raise PermissionError('barred source, including hashing')
    if p.name == 'release.npz' and ('teacher' in low or 'rel__s' in low):
        raise PermissionError('all-role source release barred')
    if inner_truth_path(p):
        raise PermissionError('all INNER labels remain closed, including hashing')
    for q in (p, *p.parents):
        if q.is_symlink():
            raise PermissionError('symlink path refused')
    return p


def metadata_stat(path):
    """File-presence information only; no file content is read."""
    p = admitted_path(path)
    try:
        st = p.stat()
    except FileNotFoundError:
        return {'exists': False, 'size_bytes': None}
    return {'exists': p.is_file(), 'size_bytes': st.st_size}


def read_metadata(path):
    p = admitted_path(path)
    if p.suffix.lower() != '.json':
        raise PermissionError('metadata reader accepts explicit JSON only')
    return json.loads(p.read_text())


def sha_metadata(path):
    p = admitted_path(path)
    if p.suffix.lower() not in {'.json', '.md', '.py', '.txt', '.csv'}:
        raise PermissionError('metadata hash refuses numerical artifacts')
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_pin(item):
    """Explicit numerical-worker operation; call only in the shared queue."""
    from .scheduler import assert_active_from_env
    assert_active_from_env()
    p = admitted_path(item['path'])
    with p.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != item['sha256']:
        raise PermissionError('source hash differs: ' + p.name)
    return p


def load_manifest(path, expected_sha256=None):
    """Read the new reduced manifest; reject INNER label entries outright."""
    if expected_sha256 and sha_metadata(path) != expected_sha256:
        raise PermissionError('manifest hash differs')
    m = read_metadata(path)
    if m.get('schema') != 'OB1-role-safe-private-source-v1':
        raise PermissionError('new campaign source schema required')
    if LABELS & m['roles']['INNER_SELECTION'].keys():
        raise PermissionError('prefit manifest must omit every INNER label')
    for role, fields in m['roles'].items():
        if role not in ROLES:
            raise PermissionError('unadmitted role')
        for name, item in fields.items():
            if name not in {'features', 'references_s0', 'references_s1', *LABELS}:
                raise PermissionError('unadmitted role field')
            admitted_path(item['path'])
    return m


def _read_exact(item, fields):
    import numpy as np
    p = verify_pin(item)
    with ZipFile(p) as archive:
        names = [x.filename for x in archive.infolist()]
        if len(names) != len(fields) or set(names) != {x + '.npy' for x in fields}:
            raise PermissionError('unexpected source fields')
    with np.load(p, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in fields}


def group_folds(unit):
    import numpy as np
    return np.array([int(hashlib.sha256((FOLD_SALT + str(x)).encode()).hexdigest(), 16) % 3
                     for x in unit], dtype=np.int64)


def load_role(manifest, seed, role, fields=('features',)):
    """Explicit queued role loader with no nomination-only INNER exception."""
    from .scheduler import assert_active_from_env
    assert_active_from_env()
    import numpy as np
    fields = set(fields)
    if seed not in (0, 1) or role not in ROLES:
        raise PermissionError('unadmitted role or seed')
    if not fields <= {'features', 'references', *LABELS}:
        raise PermissionError('unadmitted field')
    if role == 'INNER_SELECTION' and fields & LABELS:
        raise PermissionError('INNER labels require separate committed reporting admission')
    if role == 'ATTACK_FIT_NEW' and fields & {'income', 'occupation'}:
        raise PermissionError('ATT task labels are unavailable in inherited role export')
    f = _read_exact(manifest['roles'][role]['features'], FEATURE_FIELDS)
    n, groups = ROLES[role]
    if len(f['row_id']) != n or len(np.unique(f['row_id'])) != n or len(np.unique(f['unit'])) != groups:
        raise PermissionError('actual role counts differ from inherited counts')
    if f['X'].shape != (n, 83) or f['X'].dtype != np.float32 or not np.isfinite(f['X']).all():
        raise PermissionError('raw83 input differs')
    names = tuple(map(str, f['feature_names']))
    forbidden = {'sex', 'race', 'income', 'occupation', 'fnlwgt', 'row_id', 'unit', 'id', 'hidden_flag'}
    if len(names) != 83 or len(set(names)) != 83 or any(x.lower().split('=')[0] in forbidden for x in names):
        raise PermissionError('forbidden input schema')
    out = dict(f)
    out['fold'] = group_folds(f['unit']) if role == 'OSF_DEFENSE_FIT' else None
    for name in fields - {'features'}:
        key = f'references_s{seed}' if name == 'references' else name
        item = manifest['roles'][role][key]
        z = _read_exact(item, REFERENCE_FIELDS if name == 'references' else {'row_id', 'unit', name})
        if any(not np.array_equal(z[k], f[k]) for k in ('row_id', 'unit')):
            raise PermissionError('person pairing differs')
        if name == 'references':
            out[name] = z
        else:
            a = z[name]
            k = {'income': 2, 'occupation': 6, 'sex': 2}[name]
            if a.shape != (n,) or a.dtype.kind not in 'iu' or np.any(a < 0) or np.any(a >= k):
                raise PermissionError('label convention differs')
            out[name] = a
    return out
