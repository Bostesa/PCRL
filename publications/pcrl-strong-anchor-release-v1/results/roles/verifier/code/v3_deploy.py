"""Verifier wave 3 (label-free; run under the torch venv, the interpreter all 13 releases were fitted with): deployment packages.

Per package .private/pcrl_strong_anchor_release_v1/deploy/<CONFIG>_V<v>/PACKAGE.json:
 (a) every depends_sha256 entry re-hashed with the verifier's own key resolution (CODE:: module file, SAR:: under runs/, REPO:: under
     the repository root);
 (b) the stored deployment wire runs/STORED/<FAMILY>__<CONFIG>__V<v>.npz (keys <task>__<ROLE>) equals the evaluated
     runs/<FAMILY>/<CONFIG>/V<v>/RELEASE.npz (keys <ROLE>__<task>) bitwise, both tasks, all six roles; the release hash chain;
 (c) the MODEL_SELECT wire hashes (vlib.wire_hash) equal wire_sha256_eval of the EVAL_MS record that evaluated this release;
 (d) probability validity of every stored array (finite, in [0, 1], rows sum to 1 within 1e-9);
 (e) black-box equivalence: sar.deploy.release on 50 DEV_REPORT raw rows (dicts of the six permitted fields, sample drawn with
     numpy.random.default_rng(20261026), the same rows for every package) equals the stored DEV_REPORT wire bitwise, decisions too;
 (f) three refusal probes per package: an extra SEX field, a missing POBP field, a 'raw_score' request.
No label is read (DEV_REPORT raw inputs are label-free features). The coordinator's DEPLOYMENT_CHECK / DEPLOYMENT_VALIDITY files are not
read here (compared afterwards in v3_compare.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

sys.path.insert(0, str(V.ROOT))
import numpy as np  # noqa: E402

PERMITTED = ('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
SEED = 20261026
N_ROWS = 50


def resolve(key):
    kind, rest = key.split('::', 1)
    if kind == 'SAR':
        return V.RUNS / rest
    if kind == 'REPO':
        return V.ROOT / rest
    if kind == 'CODE':
        p = V.ROOT / (rest.replace('.', '/') + '.py')
        return p if p.is_file() else V.ROOT / rest.replace('.', '/') / '__init__.py'
    raise KeyError(key)


def eval_record_for(release_sha):
    """The EVAL_MS record(s) that evaluated this release (matching release_sha256 per version)."""
    hits = []
    for p in sorted(V.RUNS.glob('**/EVAL_MS*.json')):
        if 'GRID' in p.parts or 'devgrid_cache' in p.parts:
            continue
        rec = json.loads(p.read_text())
        for v, e in (rec.get('versions') or {}).items():
            if isinstance(e, dict) and e.get('release_sha256') == release_sha:
                hits.append((p, v, e))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.process_time()
    from sar import common as C
    from sar import deploy as SD
    f = C.features('DEV_REPORT', raw=True)
    raw = f['raw_inputs']
    if not np.array_equal(raw['row_id'], f['row_id']):
        raise AssertionError('raw inputs and features are in different row orders')
    idx = np.sort(np.random.default_rng(SEED).choice(len(f['row_id']), size=N_ROWS, replace=False))
    rows = [{k: str(raw[k][i]) for k in PERMITTED} for i in idx]
    out = {'schema': 'SAR1-verifier-deploy-v1', 'utc_start': C.now(), 'interpreter': sys.executable,
           'python': sys.version.split()[0], 'numpy': np.__version__, 'sample': {'seed': SEED, 'n': N_ROWS, 'indices': idx.tolist()},
           'packages': {}}
    for pkg in sorted((V.PRIV / 'deploy').iterdir()):
        if not (pkg / 'PACKAGE.json').is_file():
            continue
        meta = json.loads((pkg / 'PACKAGE.json').read_text())
        fam, cfg = meta['name'].split('/')
        v = int(meta['version'])
        r = {'package_sha256': V.fsha(pkg / 'PACKAGE.json'), 'name': meta['name'], 'version': v}
        # (a)
        bad = {k: {'recorded': h, 'now': (V.fsha(resolve(k)) if resolve(k).is_file() else None)}
               for k, h in meta['depends_sha256'].items() if not resolve(k).is_file() or V.fsha(resolve(k)) != h}
        r['a_depends'] = {'n': len(meta['depends_sha256']), 'mismatched_or_missing': bad, 'pass': not bad}
        # (b)
        sp = V.RUNS / 'STORED' / f'{fam}__{cfg}__V{v}.npz'
        rd = V.RUNS / fam / cfg / f'V{v}'
        rrec = json.loads((rd / 'RELEASE.json').read_text())
        chain = {'release_npz_equals_record': V.fsha(rd / 'RELEASE.npz') == rrec['release_sha256'],
                 'package_release_sha_equals_record': meta.get('release_sha256') == rrec['release_sha256'],
                 'stored_in_depends': f'SAR::STORED/{fam}__{cfg}__V{v}.npz' in meta['depends_sha256']}
        diffs, valid = {}, {}
        with np.load(sp, allow_pickle=False) as zs, np.load(rd / 'RELEASE.npz', allow_pickle=False) as zr:
            for role in ROLES6:
                for t in V.TASKS:
                    s_, e_ = zs[f'{t}__{role}'], zr[f'{role}__{t}']
                    if not (s_.dtype == e_.dtype and s_.shape == e_.shape and np.array_equal(s_, e_)):
                        diffs[f'{t}__{role}'] = {'shape': [list(s_.shape), list(e_.shape)],
                                                 'max_abs': float(np.abs(s_ - e_).max()) if s_.shape == e_.shape else None}
                    fin = bool(np.isfinite(s_).all())
                    valid[f'{t}__{role}'] = {'finite': fin, 'min': float(s_.min()), 'max': float(s_.max()),
                                             'max_row_sum_dev': float(np.abs(s_.sum(-1) - 1).max()),
                                             'pass': fin and s_.min() >= 0 and s_.max() <= 1 and float(np.abs(s_.sum(-1) - 1).max()) <= 1e-9}
            stored_dev = {t: zs[f'{t}__DEV_REPORT'].copy() for t in V.TASKS}
            ms = {t: zr[f'MODEL_SELECT__{t}'].copy() for t in V.TASKS}
        r['b_stored_equals_release'] = {'differences': diffs, 'chain': chain, 'pass': not diffs and all(chain.values())}
        # (c)
        hits = eval_record_for(rrec['release_sha256'])
        mine = {t: V.wire_hash(ms[t]) for t in V.TASKS}
        cmp_ = [{'record': str(p.relative_to(V.RUNS)), 'version': vv, 'equal': all(e['wire_sha256_eval'][t] == mine[t] for t in V.TASKS)}
                for p, vv, e in hits]
        r['c_model_select_wire'] = {'mine': mine, 'records': cmp_, 'pass': bool(cmp_) and all(x['equal'] for x in cmp_)}
        # (d)
        r['d_validity'] = {'per_array': valid, 'tolerance_row_sum': 1e-9, 'pass': all(x['pass'] for x in valid.values())}
        # (e)
        t1 = time.process_time()
        try:
            res = SD.release(str(pkg), rows)
            eq = {t: {'probability_bitwise': bool(np.array_equal(res[t]['probability'], stored_dev[t][idx])),
                      'decision_equal_stored_argmax': bool(np.array_equal(res[t]['decision'], stored_dev[t][idx].argmax(1))),
                      'decision_equal_own_argmax': bool(np.array_equal(res[t]['decision'], res[t]['probability'].argmax(1))),
                      'max_abs': float(np.abs(res[t]['probability'] - stored_dev[t][idx]).max()),
                      'outputs': sorted(res[t].keys())} for t in V.TASKS}
            r['e_blackbox'] = {'per_task': eq, 'returned_tasks': sorted(res.keys()), 'cpu_s': time.process_time() - t1,
                               'pass': all(x['probability_bitwise'] and x['decision_equal_stored_argmax'] and x['decision_equal_own_argmax']
                                           and x['outputs'] == ['decision', 'probability'] for x in eq.values()) and sorted(res.keys()) == list(V.TASKS)}
        except Exception as ex:                                   # recorded, never hidden
            r['e_blackbox'] = {'error': f'{type(ex).__name__}: {ex}', 'pass': False}
        # (f)
        probes = {}
        for name, call in (('extra_SEX_field', lambda: SD.release(str(pkg), [{**rows[0], 'SEX': '2'}])),
                           ('missing_POBP_field', lambda: SD.release(str(pkg), [{k: rows[0][k] for k in PERMITTED if k != 'POBP'}])),
                           ('raw_score_request', lambda: SD.release(str(pkg), rows[:1], requested=('probability', 'decision', 'raw_score')))):
            try:
                got = call()
                probes[name] = {'refused': False, 'returned': sorted(got.keys()) if isinstance(got, dict) else str(type(got))}
            except Exception as ex:
                probes[name] = {'refused': True, 'exception': type(ex).__name__, 'message': str(ex)[:200]}
        r['f_refusals'] = {'probes': probes, 'pass': all(p['refused'] for p in probes.values())}
        r['pass'] = all(r[k]['pass'] for k in ('a_depends', 'b_stored_equals_release', 'c_model_select_wire', 'd_validity', 'e_blackbox', 'f_refusals'))
        out['packages'][pkg.name] = r
        print(json.dumps({'package': pkg.name, 'pass': r['pass'], **{k: r[k]['pass'] for k in r if isinstance(r[k], dict) and 'pass' in r[k]}}), flush=True)
    out['cpu_s'] = time.process_time() - t0
    out['utc_end'] = C.now()
    V.jdump(Path(a.out), out)
    print(json.dumps({'packages': len(out['packages']), 'all_pass': all(p['pass'] for p in out['packages'].values()), 'cpu_s': out['cpu_s']}))


if __name__ == '__main__':
    main()
