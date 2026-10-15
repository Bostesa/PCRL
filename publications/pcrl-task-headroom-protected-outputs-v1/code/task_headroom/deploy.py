"""Standalone raw83 -> released probabilities + decisions for tree services (numpy only).

Package layout: MANIFEST.json + teacher_income.json + teacher_occupation.json
[+ theta.npz]. The manifest binds every file by SHA-256 and states the input
schema (83 ordered raw feature names; SEX and task labels are never inputs).
Output npz holds exactly: p_income (n,2), decision_income (n,), p_occupation
(n,6), decision_occupation (n,). It refuses: manifest or input hash mismatch,
missing/reordered/extra columns, nonfinite values, label- or SEX-bearing input
arrays, any request for hidden quantities (leaves, raw scores, teacher outputs),
and overwriting an existing output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HIDDEN = ('leaves', 'leaf', 'raw', 'scores', 'teacher', 'logits', 'theta', 'routing')
FORBIDDEN_INPUT_KEYS = ('sex', 'income', 'occupation', 'label', 'labels', 'y', 'row_id', 'unit')


def _sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Refusal(Exception):
    pass


def load_package(package, manifest_sha256):
    package = Path(package); mpath = package / 'MANIFEST.json'
    if _sha(mpath) != manifest_sha256:
        raise Refusal('manifest SHA-256 differs')
    m = json.loads(mpath.read_text())
    if m.get('schema') != 'THP1-tree-service-package-v1':
        raise Refusal('package schema differs')
    for name, digest in m['files'].items():
        if _sha(package / name) != digest:
            raise Refusal(f'package file hash differs: {name}')
    from .trees import TreeTeacher
    teachers = [TreeTeacher(json.loads((package / m['teachers'][t]).read_text())) for t in ('income', 'occupation')]
    theta = [None, None]
    if m.get('theta'):
        with np.load(package / m['theta'], allow_pickle=False) as z:
            theta = [z['theta_income'].copy(), z['theta_occupation'].copy()]
    return m, teachers, theta


def predict(m, teachers, theta, X, names):
    from .trees import release
    if tuple(map(str, names)) != tuple(m['feature_names']):
        raise Refusal('input schema differs (missing, reordered or extra columns)')
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 83 or not np.issubdtype(X.dtype, np.floating) or not np.isfinite(X).all():
        raise Refusal('finite float raw83 matrix required')
    out = {}
    for task, teacher, th in zip(('income', 'occupation'), teachers, theta):
        r = release(task, teacher.raw_from_leaves(teacher.route(X, names), th))
        out[f'p_{task}'] = r['probability']; out[f'decision_{task}'] = r['decision']
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', required=True); p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--input', required=True); p.add_argument('--input-sha256', required=True)
    p.add_argument('--output', required=True); p.add_argument('--emit', default='released', help='only "released" is allowed')
    a = p.parse_args(argv)
    try:
        if a.emit != 'released' or any(h in a.emit.lower() for h in HIDDEN):
            raise Refusal('only released probabilities and decisions can be emitted')
        if _sha(a.input) != a.input_sha256:
            raise Refusal('input SHA-256 differs')
        if Path(a.output).exists():
            raise Refusal('output exists')
        with np.load(a.input, allow_pickle=False) as z:
            keys = set(z.files)
            if keys != {'X', 'feature_names'}:
                bad = keys & set(FORBIDDEN_INPUT_KEYS)
                raise Refusal('input must contain exactly X and feature_names' + (f'; label/identity arrays present: {sorted(bad)}' if bad else ''))
            X, names = z['X'], z['feature_names']
        m, teachers, theta = load_package(a.package, a.manifest_sha256)
        out = predict(m, teachers, theta, X, names)
        with Path(a.output).open('xb') as f:
            np.savez(f, **out)
    except Refusal as e:
        print(json.dumps({'status': 'REFUSED', 'reason': str(e)})); return 2
    print(json.dumps({'status': 'OK', 'rows': int(len(X)), 'output_sha256': _sha(a.output)})); return 0


if __name__ == '__main__':
    sys.exit(main())
