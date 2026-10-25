"""Primary simulation grid of EDG1 (PROTOCOL.md section 4): 48 independent-sampling cells (4 families x 2 class frequencies x
2 occupancies x 3 sample sizes) and one clustered household sensitivity cell, 500 repetitions each (prospective minimum 200).
Each repetition draws one study sample and one independent test sample from the cell's known law with its own seed
SeedSequence(BASE_SEED, spawn_key=(cell, rep)); every view of a repetition uses the SAME people (paired comparisons).
Writes one JSON line per repetition (gzip) plus a run record with CPU time; never reads any non-synthetic input."""
from __future__ import annotations

import argparse
import os
import gzip
import hashlib
import json
import resource
import sys
import time
from datetime import datetime, timezone

import numpy as np

from . import OUT, RESULTS
from . import estimators as E
from . import laws as L

BASE_SEED = 20261025
REPS = 500
MIN_REPS = 200
NS = (2000, 8000, 20000)
ESTIMATORS = ('C_same', 'C_nested', 'D_nested', 'D_bank', 'B_same', 'B_indep')


def scenarios():
    out = []
    for fam in L.FAMILIES:
        for pi in L.PIS:
            for occ in L.OCCS:
                for n in NS:
                    out.append({'cell': len(out), 'family': fam, 'pi': pi, 'occ': occ, 'n': n, 'clustered': False})
    out.append({'cell': len(out), 'family': 'REF', 'pi': 0.5, 'occ': 'skewed', 'n': 20000, 'clustered': True})
    return out


def rng_for(cell, rep):
    return np.random.default_rng(np.random.SeedSequence(BASE_SEED, spawn_key=(int(cell), int(rep))))


def _wire_keys(view):
    w = view['wires']
    z = np.column_stack(w) if isinstance(w, tuple) else w
    return [np.ascontiguousarray(r, dtype=np.float64).tobytes() for r in z]


def sample_rep(sc, law, rep):
    """The repetition's study sample, independent test sample, 5 nested folds and the 10-part FIT / VALIDATION / SCORE split
    (parts 0-4 / 5-6 / 7-9), all by unit (person, or household in the clustered cell). The reader-bank subset reuses it."""
    rng = rng_for(sc['cell'], rep)
    if sc['clustered']:
        c, s, unit = L.draw_households(law, sc['n'], rng)
        ct, st, unit_t = L.draw_households(law, sc['n'], rng)
    else:
        c, s, unit = L.draw(law, sc['n'], rng)
        ct, st, unit_t = L.draw(law, sc['n'], rng)
    G = int(unit.max()) + 1
    folds = (rng.permutation(G) % E.N_FOLDS)[unit]
    parts = (rng.permutation(G) % 10)[unit]
    return c, s, unit, ct, st, unit_t, folds, parts


def one_rep(sc, law, rep, keys=None):
    c, s, unit, ct, st, unit_t, folds, parts = sample_rep(sc, law, rep)
    G = int(unit.max()) + 1
    sc_mask = parts >= 7
    row = {'cell': sc['cell'], 'rep': rep, 'n': int(len(c)), 'n1': int(s.sum()), 'n_test': int(len(ct)), 'n_test1': int(st.sum()),
           'units': G, 'views': {}, 'cmp': {}, 'undefined': []}
    IFs = {}
    for name, V in law['views'].items():
        ids, idt, K = V['inv'][c], V['inv'][ct], V['K']
        m1p, m0p = V['m1'], V['m0']
        r, inf = {}, {}
        r['C_same'], inf['C_same'] = E.c_same(ids, s, K)
        r['C_nested'], inf['C_nested'] = E.c_nested(ids, s, folds, K)
        dn = E.d_nested(ids, s, folds, K, m1p, m0p)
        r['D_nested'], inf['D_nested'] = dn['value'], dn['IF']
        db = E.d_bank(ids, s, parts, K, m1p, m0p, wire_keys=(keys[name] if keys is not None else None))
        r['D_bank'], inf['D_bank'] = db['value'], db['IF']
        r['B_same'], inf['B_same'] = E.b_known(ids, s, V['post'])
        r['B_indep'], inf['B_indep'] = E.b_known(idt, st, V['post'])
        for k in ('pop_fold_mean', 'pop_pooled', 'unseen_test_share'):
            r[f'D_nested_{k}'] = dn.get(k)
        for k in ('pop', 'val_auc', 'sign', 'unseen_score_share', 'source_checked'):
            r[f'D_bank_{k}'] = db.get(k)
        r['occupied'] = int((np.bincount(ids, minlength=K) > 0).sum())
        for k, v in r.items():
            if isinstance(v, float) and not np.isfinite(v):
                row['undefined'].append(f'{name}:{k}')
        row['views'][name] = r
        IFs[name] = inf
    units = {'study': unit, 'score': unit[sc_mask], 'test': unit_t}
    unit_of = {'C_same': 'study', 'C_nested': 'study', 'D_nested': 'study', 'B_same': 'study', 'D_bank': 'score', 'B_indep': 'test'}
    for x, yv in law['comparisons']:
        cr = {}
        for e in ESTIMATORS:
            d = row['views'][x][e] - row['views'][yv][e]
            ix, iy = IFs[x][e], IFs[yv][e]
            if ix is None or iy is None:
                cr[e] = [NAN_, NAN_, NAN_]
                continue
            u = units[unit_of[e]]
            se_unit = E.cluster_se(ix - iy, u)
            se_person = E.cluster_se(ix - iy, np.arange(len(u))) if sc['clustered'] else se_unit
            cr[e] = [float(d), se_unit, se_person]
        row['cmp'][f'{x}-{yv}'] = cr
    return row


NAN_ = float('nan')


def run(cells, reps, out_path, rep_start=0):
    t0, w0 = time.process_time(), time.time()
    started = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    scs = [s for s in scenarios() if s['cell'] in set(cells)]
    n_rows = 0
    with gzip.open(out_path, 'wt') as fh:
        for sc in scs:
            law = L.build(sc['family'], sc['pi'], sc['occ'])
            keys = {k: _wire_keys(v) for k, v in law['views'].items()}
            for rep in range(rep_start, rep_start + reps):
                row = one_rep(sc, law, rep, keys=keys if rep == rep_start else None)
                fh.write(json.dumps(row, allow_nan=True) + '\n')
                n_rows += 1
            print(json.dumps({'cell': sc['cell'], 'done_reps': reps, 'cpu_s': round(time.process_time() - t0, 1),
                              'utc': datetime.now(timezone.utc).strftime('%H:%M:%SZ')}), flush=True)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rec = {'schema': 'EDG1-sim-run-v1', 'cells': sorted(cells), 'reps': reps, 'rep_start': rep_start, 'rows': n_rows,
           'out': os.path.relpath(out_path, RESULTS), 'started_utc': started,
           'finished_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           'cpu_s_process': round(time.process_time() - t0, 2), 'rusage_user_s': ru.ru_utime, 'rusage_sys_s': ru.ru_stime,
           'wall_s': round(time.time() - w0, 1), 'python': sys.version.split()[0], 'numpy': np.__version__}
    with open(str(out_path).replace('.jsonl.gz', '.run.json'), 'w') as fh:
        json.dump(rec, fh, indent=1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', required=True, help='e.g. 0-23 or 48')
    ap.add_argument('--reps', type=int, default=REPS)
    ap.add_argument('--rep-start', type=int, default=0)
    ap.add_argument('--label', required=True)
    a = ap.parse_args(argv)
    lo, _, hi = a.cells.partition('-')
    cells = list(range(int(lo), int(hi or lo) + 1))
    OUT.mkdir(parents=True, exist_ok=True)
    rec = run(cells, a.reps, OUT / f'sim_{a.label}.jsonl.gz', rep_start=a.rep_start)
    print(json.dumps(rec))


if __name__ == '__main__':
    main()
