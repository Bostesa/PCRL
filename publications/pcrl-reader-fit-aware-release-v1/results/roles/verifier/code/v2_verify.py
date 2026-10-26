#!/usr/bin/env python3
"""RFA1 independent verifier: later-wave entry point (PREPARED IN WAVE 1, NOT RUN ON REAL DATA IN WAVE 1).

Labels only through rfa.common.load(..., operation='verify') called from THIS __main__ module (the access layer authorizes the
repository-relative script path under roles/verifier/code/), and only inside a queue unit:
    python3 -m rfa.jobqueue run --unit RFA_VER_<name> --stage verify --rss-gib 2 -- \\
        <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v2_verify.py <cmd> ...

    usefulness --config C [--config C2 ...] [--family FAM] [--eval-tag T]   28 conditions per configuration (own release / bank
                                                                             readers with own hash checks; compared with EVAL_MS)
    grid --bank-id ID --version V [--pair CAND:COMP ...]                    recompute a saved grid (GRID.json, BOOT.npz, cache)
    refit --bank-id ID --version V --pick CELL:MEMBER:VIEW [...] [--max-fits N]   bounded sklearn refit of grid winners
Outputs: roles/verifier/out/V2_<CMD>__<tag>.json. The metric, selection, AUC, bootstrap and reader code is vlib / vcore (verifier's).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vcore  # noqa: E402
import vlib as V  # noqa: E402

OUTDIR = HERE.parent / 'out'
CAMPAIGN = 'pcrl_reader_fit_aware_release_v1'
RUNS = ROOT / '.private' / CAMPAIGN / 'runs'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def refuse_outside_queue():
    if not os.environ.get('RFA_UNIT_ID'):
        sys.exit('refused: the verifier reads labels only inside an rfa.jobqueue verify unit (RFA_UNIT_ID unset)')


def load_ms(fields, purpose):
    """MODEL_SELECT through the logged access layer, called from this __main__ module (entry point = this script's path)."""
    from rfa import common as C
    S = C.load('MODEL_SELECT', ('features',) + tuple(fields), 'verify', purpose)
    S.pop('X', None)
    S.pop('raw_inputs', None)
    return S


def own_release(family, config, version):
    """MODEL_SELECT arrays of a saved release of this campaign, hash-checked by the verifier (file SHA-256 and per-array SHA)."""
    d = RUNS / family / config / f'V{int(version)}'
    rec = json.loads((d / 'RELEASE.json').read_text())
    if sha256(d / 'RELEASE.npz') != rec['release_sha256']:
        raise PermissionError(f'{family}/{config}/V{version}: RELEASE.npz differs from its record')
    out = {}
    with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
        for t in V.TASKS:
            a = z[f'MODEL_SELECT__{t}'].copy()
            if V.arr_sha(a) != rec['wire_sha256'][f'MODEL_SELECT__{t}']:
                raise PermissionError(f'{family}/{config}/V{version} {t}: array hash differs from the record')
            out[t] = a
    return out, {'release_sha256': rec['release_sha256'], 'kind': rec.get('kind'), 'randomized': bool(rec.get('randomized'))}


def own_refs(version, row_id):
    """U0 / Ucal MODEL_SELECT outputs from the admitted frozen bank; the verifier checks the file against BOTH the source admission
    record and the frozen-hybrids manifest, and the row order against the MODEL_SELECT export."""
    from ohn import common as OC
    from scm_bench import common as SB
    rel = f'admitted/bank/FROZEN_BANK_S{int(version)}.npz'
    p = OC.MIRROR / rel
    got = sha256(p)
    adm = OC.admission()['copies'][rel]['sha256']
    man = json.loads((SB.HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())['bank_files_sha256'][f'S{int(version)}']
    if not (got == adm == man):
        raise PermissionError(f'frozen bank S{version}: hash {got} vs admission {adm} vs manifest {man}')
    with np.load(p, allow_pickle=False) as z:
        if not np.array_equal(z['row_id__MODEL_SELECT'], row_id):
            raise PermissionError('frozen bank MODEL_SELECT row order differs from the export')
        refs = {r: {t: z[f'{r}__MODEL_SELECT__{t}'].copy() for t in V.TASKS} for r in V.REFS}
    return refs, {'bank_sha256': got}


def cmd_usefulness(a):
    S = load_ms(('income', 'occupation', 'sex'), f'verify: independent recomputation of the 28 usefulness conditions of {a.config}')
    labels = {t: S[t] for t in V.TASKS}
    refs, prov = {}, {}
    for v in (0, 1):
        refs[v], prov[f'bank_S{v}'] = own_refs(v, S['row_id'])
    releases, evals = {}, {}
    for cfg in a.config:
        releases[cfg] = {}
        for v in (0, 1):
            releases[cfg][v], prov[f'{cfg}_V{v}'] = own_release(a.family, cfg, v)
        ev = RUNS / a.family / cfg / (f'EVAL_MS__{a.eval_tag}.json' if a.eval_tag else 'EVAL_MS.json')
        if ev.is_file():
            evals[cfg] = json.loads(ev.read_text())
    res = vcore.usefulness_core(labels, S['sex'], S['unit'], releases, refs, evals)
    return {'cmd': 'usefulness', 'family': a.family, 'configs': a.config, 'provenance': prov, 'result': res,
            'people': int(len(S['row_id'])), 'households': int(len(np.unique(S['unit'])))}


def grid_members(bank_id, version, row_id, n):
    from rfa import devreaders as DR
    gdir = RUNS / 'GRID' / bank_id / f'S{int(version)}'
    grid = json.loads((gdir / 'GRID.json').read_text())
    members, prov = {}, {}
    for name, spec in grid['spec']['members'].items():
        if spec.get('type') == 'rfa':
            rel, pv = own_release(spec['family'], spec['config'], version)
            if pv['randomized']:
                raise NotImplementedError('randomized members are not recomputed by this wave-1 code')
        else:
            rel, _ = DR.load_member(spec, version, n=n, row_id=row_id)          # campaign loader (admitted copies), hash-checked below
        want = grid['members'][name]['wire_sha256']
        got = {t: V.wire_sha(rel[t]) for t in V.TASKS}
        if got != want:
            raise PermissionError(f'{name}: released wire differs from the grid provenance')
        members[name] = {t: np.asarray(rel[t], dtype=np.float64) for t in V.TASKS}
        prov[name] = got
    return gdir, members, prov


def cmd_grid(a):
    S = load_ms(('sex',), f'verify: independent recomputation of grid {a.bank_id} S{a.version} (winners, AUCs, bootstrap, paired)')
    gdir, members, prov = grid_members(a.bank_id, a.version, S['row_id'], len(S['row_id']))
    pairs = [tuple(p.split(':')) for p in a.pair] if a.pair else None
    res = vcore.grid_core(gdir, RUNS / 'devgrid_cache', a.version, S['sex'], S['unit'], members, pairs)
    return {'cmd': 'grid', 'bank_id': a.bank_id, 'version': a.version, 'member_wires': prov, 'result': res}


def cmd_refit(a):
    S = load_ms(('sex',), f'verify: bounded refit of grid {a.bank_id} S{a.version} winners ({len(a.pick)} picks)')
    gdir, members, prov = grid_members(a.bank_id, a.version, S['row_id'], len(S['row_id']))
    picks = [tuple(p.split(':')) for p in a.pick]
    res = vcore.refit_core(gdir, RUNS / 'devgrid_cache', a.version, S['sex'], S['unit'], members, picks, a.max_fits)
    import sklearn
    import scipy
    return {'cmd': 'refit', 'bank_id': a.bank_id, 'version': a.version, 'picks': a.pick, 'result': res,
            'versions': {'sklearn': sklearn.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__}}


def main(argv=None):
    ap = argparse.ArgumentParser(prog='v2_verify')
    sub = ap.add_subparsers(dest='cmd', required=True)
    u = sub.add_parser('usefulness'); u.add_argument('--config', action='append', required=True); u.add_argument('--family', default='FAM')
    u.add_argument('--eval-tag', default=None); u.add_argument('--tag', default=None)
    g = sub.add_parser('grid'); g.add_argument('--bank-id', required=True); g.add_argument('--version', type=int, required=True)
    g.add_argument('--pair', action='append', default=[]); g.add_argument('--tag', default=None)
    r = sub.add_parser('refit'); r.add_argument('--bank-id', required=True); r.add_argument('--version', type=int, required=True)
    r.add_argument('--pick', action='append', required=True); r.add_argument('--max-fits', type=int, default=8); r.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    refuse_outside_queue()
    t0 = time.process_time()
    rec = {'usefulness': cmd_usefulness, 'grid': cmd_grid, 'refit': cmd_refit}[a.cmd](a)
    rec.update({'unit_id': os.environ.get('RFA_UNIT_ID'), 'cpu_s': time.process_time() - t0,
                'code_sha256': {f: sha256(HERE / f) for f in ('v2_verify.py', 'vcore.py', 'vlib.py')}})
    name = f"V2_{a.cmd.upper()}__{a.tag or (a.bank_id if a.cmd != 'usefulness' else '_'.join(a.config))}.json"
    out = OUTDIR / name
    if out.exists():
        raise PermissionError(f'{name} exists (write-once)')
    out.write_text(json.dumps(rec, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
    print(json.dumps({'out': str(out.relative_to(ROOT)), 'cpu_s': rec['cpu_s']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
