"""Common release format and the development evaluation entry point used by every family.

Release file (private): runs/<FAMILY>/<config_id>/S<seed>/RELEASE.npz with keys '<ROLE>__income' (n, 2) and
'<ROLE>__occupation' (n, 5), float64 probability vectors in the admitted row order of each role, for every role the mechanism
can compute from the permitted inputs (at least MODEL_SELECT; a promoted finalist needs ATTACK_FIT, ATTACK_VALIDATION and
DEV_REPORT too). Decisions are always the argmax. RELEASE.json beside it records the mechanism, inputs used and fit details.

    python -m amd.devrun score --family A --version A1 --config CCLC_a1 --seed 0 --counts 1 \
        --hypothesis "..." --change "..." --training "..." --selection "..." [--notes "..."]

writes runs/<FAMILY>/<config>/S<seed>/EVAL_MS.json and appends one EXPERIMENT_LEDGER.csv row.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import evaluate as EV
from . import ledger as L


def release_dir(family, config_id, seed):
    return C.RUNS / family / config_id / f'S{seed}'


def save_release(family, config_id, seed, outs_by_role, record):
    d = release_dir(family, config_id, seed)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    arrays = {}
    for role, outs in outs_by_role.items():
        for t in C.TASKS:
            p = np.ascontiguousarray(np.asarray(outs[t], dtype=np.float64))
            if p.ndim != 2 or p.shape[1] != C.K[t] or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
                raise ValueError(f'{role} {t}: not a valid probability-vector release')
            arrays[f'{role}__{t}'] = p
    with open(d / 'RELEASE.npz', 'wb') as f:
        np.savez(f, **arrays)
    rec = {**record, 'family': family, 'config_id': config_id, 'seed': seed, 'utc': C.now(), 'roles': sorted(outs_by_role),
           'release_sha256': C.sha(d / 'RELEASE.npz'),
           'wire_sha256': {k: C.wire_sha(v) for k, v in arrays.items()}}
    (d / 'RELEASE.json').write_text(json.dumps(rec, indent=1, sort_keys=True, default=float) + '\n')
    return d


def load_release(family, config_id, seed, roles=None):
    d = release_dir(family, config_id, seed)
    rec = json.loads((d / 'RELEASE.json').read_text())
    if C.sha(d / 'RELEASE.npz') != rec['release_sha256']:
        raise PermissionError('release file differs from its record')
    out = {}
    with np.load(d / 'RELEASE.npz') as z:
        for k in z.files:
            role, t = k.split('__')
            if roles is None or role in roles:
                out.setdefault(role, {})[t] = z[k].copy()
    return out, rec


def evaluate(family, version, config_id, seed, counts, hypothesis, change, training, selection, notes='', code_paths=()):
    t0 = time.process_time()
    outs, rec = load_release(family, config_id, seed, roles=('MODEL_SELECT',))
    S, refs, _ = EV.ms_context(seed, purpose=f'development evaluation {family}/{config_id}/S{seed}')
    m = EV.measure(seed, outs['MODEL_SELECT'], S, refs)
    d = release_dir(family, config_id, seed)
    (d / 'EVAL_MS.json').write_text(json.dumps(m, indent=1, sort_keys=True, default=float) + '\n')
    paths = [Path(__file__)] + [Path(p) for p in code_paths]
    row = {'family': family, 'version': version, 'config_id': config_id, 'seed': seed, 'kind': 'fit' if counts else 'diagnostic',
           'counts_toward_36': int(counts), 'code_sha256': L.code_sha(*paths), 'unit_id': os.environ.get('AMD_UNIT_ID', ''),
           'hypothesis': hypothesis, 'change_from_previous': change,
           'roles_read': 'MODEL_FIT (fit; features, labels, SEX as recorded in RELEASE.json); MODEL_SELECT (features, labels, SEX: evaluation)',
           'training_effort': training, 'selection_effort': selection, 'cpu_s': round(float(rec.get('fit_cpu_s', 0.0)) + time.process_time() - t0, 1),
           'notes': notes, **L.summarize(m)}
    n = L.append(row)
    return m, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('score', aliases=['eval'])
    for a in ('--family', '--version', '--config', '--hypothesis', '--change', '--training', '--selection'):
        e.add_argument(a, required=True)
    e.add_argument('--seed', type=int, required=True); e.add_argument('--counts', type=int, required=True)
    e.add_argument('--notes', default=''); e.add_argument('--code', action='append', default=[])
    a = ap.parse_args(argv)
    m, n = evaluate(a.family, a.version, a.config, a.seed, a.counts, a.hypothesis, a.change, a.training, a.selection, a.notes, a.code)
    print(json.dumps({'ledger_entry': n, **L.summarize(m)}, default=float))


if __name__ == '__main__':
    main()
