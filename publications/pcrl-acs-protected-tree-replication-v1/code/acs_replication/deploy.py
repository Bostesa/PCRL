"""Deploy a frozen ACS service from RAW permitted ACS fields to released probabilities + decisions (no labels needed).

Package layout (private): PACKAGE.json (kind, encoder, file hashes), teacher JSONs, optional THETA.npz (leaf-score
corrections) or a neural state (NEURAL_RAW). Input: a CSV whose header is EXACTLY the permitted raw fields
AGEP, WKHP, COW, SCHL, MAR, POBP (any order is rejected; extra or forbidden columns are rejected). Output per person:
income probabilities (2) + decision, occupation probabilities (5) + decision. Requests for hidden outputs (leaf ids,
raw scores, teacher outputs, representations) are refused: the release contract has no such field.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from . import schema, trees

PERMITTED = list(schema.INPUTS['numeric'] + schema.INPUTS['categorical'])
ALLOWED_OUTPUTS = {'probability', 'decision'}


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def encode_raw(rows, enc):
    names = enc['feature_names']; col = {n: i for i, n in enumerate(names)}
    X = np.zeros((len(rows), len(names)), dtype=np.float64)
    for i, r in enumerate(rows):
        for c in schema.INPUTS['numeric']:
            v = float(r[c]); st = enc['numeric'][c]
            if not np.isfinite(v):
                raise ValueError(f'non-finite {c}')
            X[i, col[c]] = (v - st['mean']) / st['sd']
        for c in schema.INPUTS['categorical']:
            key = f'{c}={str(r[c]).strip()}'
            X[i, col[key] if key in col else col[f'{c}=__OTHER__']] = 1.0
    return X.astype(np.float32)


def read_raw_csv(path):
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames or []
        forbidden = [h for h in header if h.upper() in schema.FORBIDDEN]
        if forbidden:
            raise PermissionError(f'forbidden columns present (never accepted as inputs): {forbidden}')
        if header != PERMITTED:
            raise ValueError(f'incompatible schema: expected exactly {PERMITTED} in this order, got {header}')
        return list(rd)


def load_package(pkg_dir):
    pkg_dir = Path(pkg_dir)
    meta = json.loads((pkg_dir / 'PACKAGE.json').read_text())
    for name, h in meta['files'].items():
        if sha(pkg_dir / name) != h:
            raise PermissionError(f'package file hash differs: {name}')
    return meta


def release(pkg_dir, rows, requested=('probability', 'decision')):
    if set(requested) - ALLOWED_OUTPUTS:
        raise PermissionError(f'hidden-output request refused: {sorted(set(requested) - ALLOWED_OUTPUTS)}')
    pkg_dir = Path(pkg_dir); meta = load_package(pkg_dir)
    X = encode_raw(rows, meta['encoder']); names = tuple(meta['encoder']['feature_names'])
    out = {}
    if meta['kind'] == 'TREE':
        theta = None
        if meta.get('theta'):
            with np.load(pkg_dir / meta['theta']) as z:
                theta = {'income': z['theta_income'], 'occupation': z['theta_occupation']}
        for task in ('income', 'occupation'):
            t = trees.TreeTeacher(json.loads((pkg_dir / meta['teachers'][task]).read_text()))
            raw = t.raw_from_leaves(t.route(X, names), None if theta is None else theta[task])
            out[task] = trees.release(task, raw)
    elif meta['kind'] == 'NEURAL':
        import torch
        from . import neural
        payload = torch.load(pkg_dir / meta['neural_state'], map_location='cpu', weights_only=True)
        for task, k in (('income', 2), ('occupation', 5)):
            m = neural.network(k, X.shape[1]); m.load_state_dict(payload[task]); m.eval()
            p = neural.probabilities(m, X); out[task] = {'probability': p, 'decision': p.argmax(1).astype(np.int64)}
    else:
        raise ValueError('unregistered package kind')
    return {t: {k: v for k, v in r.items() if k in requested} for t, r in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', required=True); ap.add_argument('--input', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--request', default='probability,decision')
    a = ap.parse_args(argv)
    rows = read_raw_csv(a.input)
    res = release(a.package, rows, tuple(a.request.split(',')))
    arrays = {}
    for t, r in res.items():
        for k, v in r.items():
            arrays[f'{t}_{k}'] = v
    with open(a.output, 'xb') as f:
        np.savez(f, **arrays)
    print(json.dumps({'rows': len(rows), 'fields': sorted(arrays)}))


if __name__ == '__main__':
    main()
