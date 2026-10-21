"""Verifier check 1 (label-free): release integrity and deployment recomputation; plus the MODEL_FIT strength selection and the
control bracket points re-derived from the recorded MODEL_FIT spends (label-free arithmetic on the fit records).

For every row of runs/CONFIG_LEDGER.jsonl: RELEASE.npz SHA-256 == RELEASE.json release_sha256 == ledger release_sha256; every
per-role wire hash in RELEASE.json == own array hash; MODEL.pkl SHA-256 == MODEL.json model_sha256 == RELEASE.json model_sha256;
the card SHA-256 equals the card blob at its recorded commit and the commit precedes the configuration's save time; the family's
deployment function (ddr.da.predict / ddr.db.predict; loads only the configuration's MODEL.pkl) recomputes the MODEL_SELECT arrays
bitwise from the admitted MODEL_SELECT features (all six roles for DA_TWIN, DA_STD and DB_PTC).

    python3 -m ddr.jobqueue run --unit DDR_VER_INTEGRITY_V1 --stage verify --rss-gib 2 -- python3 <this file>
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vlib as V  # noqa: E402

FULL_ROLES = {'DA_TWIN', 'DA_STD', 'DB_PTC'}


def git(*a):
    r = subprocess.run(['git', '-C', str(V.ROOT), *a], capture_output=True)
    return r.returncode, r.stdout


def main():
    t0 = time.process_time()
    from ddr import common as C              # label-free features only
    import ddr.da as DA
    import ddr.db as DB
    rows = [json.loads(l) for l in V.CONFIG_LEDGER.read_text().splitlines() if l.strip()]
    feats = {}

    def F(role):
        if role not in feats:
            feats[role] = C.features(role)
        return feats[role]

    out = {'unit_id': V.unit_id(), 'utc_start': V.utc(), 'configs': [], 'errors': []}
    card_blob_sha = {}
    for r in rows:
        fam, cfg, v = r['family'], r['config_id'], int(r['version'])
        rec = {'family': fam, 'config': cfg, 'version': v}
        try:
            npz, js = V.release_paths(fam, cfg, v)
            R = json.loads(js.read_text())
            fsha = V.file_sha(npz)
            rec['npz_sha_eq_json'] = fsha == R['release_sha256']
            rec['npz_sha_eq_ledger'] = fsha == r['release_sha256']
            arrays = {}
            with np.load(npz, allow_pickle=False) as z:
                for k in z.files:
                    arrays[k] = z[k].copy()
            wires = R.get('wire_sha256', {})
            rec['wire_keys_match'] = sorted(wires) == sorted(arrays)
            rec['wire_hashes_match'] = all(V.arr_sha(arrays[k]) == wires.get(k) for k in arrays)
            rec['n_roles'] = len({k.split('__')[0] for k in arrays})
            # model files
            d = npz.parent
            meta = json.loads((d / 'MODEL.json').read_text())
            msha = V.file_sha(d / 'MODEL.pkl')
            rec['model_sha_eq_meta'] = msha == meta['model_sha256']
            rec['model_sha_eq_release_json'] = msha == R.get('model_sha256')
            # card at its recorded commit
            key = (R['card'], R['card_commit'])
            if key not in card_blob_sha:
                rc, blob = git('show', f"{R['card_commit']}:{R['card']}")
                card_blob_sha[key] = V.hashlib.sha256(blob).hexdigest() if rc == 0 else None
            rec['card_sha_eq_commit_blob'] = card_blob_sha[key] == R['card_sha256'] == r['card_sha256']
            rec['card_sha_eq_current_file'] = V.file_sha(V.ROOT / R['card']) == R['card_sha256']
            rec['card_commit_before_save'] = V.parse_utc(R['card_commit_utc']) <= V.parse_utc(R['utc'])
            rec['card_commit_utc'] = R['card_commit_utc']
            rec['save_utc'] = R['utc']
            rec['randomized'] = bool(R.get('randomized'))
            # deployment recomputation
            pred = DA.predict if fam == 'DA' else DB.predict
            roles = C.ROLES6 if cfg in FULL_ROLES else ('MODEL_SELECT',)
            eq = {}
            for role in roles:
                f = F(role)
                got = pred(cfg, v, f['X'], f['feature_names'])
                for task in V.TASKS:
                    saved = arrays[f'{role}__{task}']
                    a = np.ascontiguousarray(np.asarray(got[task], dtype=np.float64))
                    eq[f'{role}__{task}'] = bool(a.shape == saved.shape and a.dtype == saved.dtype and a.tobytes() == saved.tobytes())
            rec['deploy_bitwise'] = eq
            rec['deploy_all_bitwise'] = all(eq.values())
            rec['ms_wire_sha'] = {t: V.wire_sha(arrays[f'MODEL_SELECT__{t}']) for t in V.TASKS}
            rec['strength'] = {k: R.get(k) for k in ('lambda', 'alpha', 'w', 'delta', 'op', 'target', 'scale_m') if k in R}
        except Exception as e:  # recorded, never hidden
            rec['error'] = f'{type(e).__name__}: {e}'
            out['errors'].append({'config': f'{fam}/{cfg}/V{v}', 'traceback': traceback.format_exc()[-2000:]})
        out['configs'].append(rec)

    # budget
    keys = {(r['family'], r['config_id'], int(r['version'])) for r in rows}
    out['budget'] = {'distinct_configurations': len(keys), 'limit': 48,
                     'per_family': {f: sum(1 for k in keys if k[0] == f) for f in ('DA', 'DB')},
                     'ledger_rows': len(rows), 'duplicate_rows': len(rows) - len(keys)}

    # exact aliases among MODEL_SELECT wires per version and task
    al = {}
    for v in (0, 1):
        for task in V.TASKS:
            by = {}
            for c in out['configs']:
                if c['version'] == v and 'ms_wire_sha' in c:
                    by.setdefault(c['ms_wire_sha'][task], []).append(c['config'])
            al[f'V{v}|{task}'] = [sorted(x) for x in by.values() if len(x) > 1]
    out['ms_exact_aliases'] = al

    # MODEL_FIT selection rule and bracket points from the fit records (label-free arithmetic)
    sel = {}
    for fam in ('da', 'db'):
        for v in (0, 1):
            d = json.loads((V.RESULTS / fam / f'FIT_V{v}.json').read_text())
            s = d['selection']
            budget = s['budget'] if isinstance(s['budget'], dict) else s['budget_limits']
            path = s['path']
            order = sorted(path, key=float)
            pick = None
            for g in order:
                okg = all(path[g][m]['estimate'] + 2 * path[g][m]['se'] <= lim for m, lim in budget.items())
                if okg:
                    pick = g
                else:
                    break
            star = s.get('lambda_star', s.get('alpha_star'))
            ent = {'budget': budget, 'recomputed_star': None if pick is None else float(pick), 'recorded_star': star,
                   'star_matches': (pick is not None and star is not None and abs(float(pick) - float(star)) < 1e-12),
                   'binding_at_next': None}
            nxt = [g for g in order if pick is not None and float(g) > float(pick)]
            if nxt:
                g = nxt[0]
                ent['binding_at_next'] = {m: path[g][m]['estimate'] + 2 * path[g][m]['se'] - lim for m, lim in budget.items()
                                          if path[g][m]['estimate'] + 2 * path[g][m]['se'] > lim}
            # brackets (PROTOCOL 4): lo = largest grid point with spend <= cand; hi = smallest with diff >= 2 se_diff
            def bracket(b):
                grid = b['grid']
                gs = sorted(grid, key=float)
                lo = [g for g in gs if grid[g]['spend'] <= b['cand_spend']]
                hi = [g for g in gs if grid[g]['diff_vs_cand'] >= 2 * grid[g]['se_diff']]
                return (float(lo[-1]) if lo else None, float(hi[0]) if hi else float(gs[-1]), 'CONTROL_PATH_MAY_NOT_BRACKET' if not hi else None)
            spb = d['spend_path_bracket']
            lo, hi, flag = bracket(spb)
            ent['spend_path'] = {'recomputed': [lo, hi, flag], 'recorded': [spb['lo'], spb['hi'], spb['flag']],
                                 'match': [lo, hi, flag] == [spb['lo'], spb['hi'], spb['flag']]}
            tries = d['sbc_tries']
            used = d.get('sbc_attribute', d.get('sbc_target'))
            tr = [t for t in tries if t.get('attribute', t.get('target')) == used][-1]
            lo, hi, flag = bracket(tr['bracket'])
            b = tr['bracket']
            ent['sbc_path'] = {'attribute': used, 'recomputed': [lo, hi, flag], 'recorded': [b['lo'], b['hi'], b['flag']],
                               'match': [lo, hi, flag] == [b['lo'], b['hi'], b['flag']],
                               'tries': [t.get('attribute', t.get('target')) for t in tries]}
            # saved control strengths equal the recorded bracket points
            fam_u = fam.upper()
            saved = {}
            for c in out['configs']:
                if c['family'] == fam_u and c['version'] == v and 'strength' in c:
                    saved[c['config']] = c['strength']
            ent['saved_strengths'] = saved
            sel[f'{fam_u}_V{v}'] = ent
    out['selection'] = sel
    out['cpu_s'] = round(time.process_time() - t0, 2)
    out['utc_end'] = V.utc()
    p = V.write_record(f'V1_INTEGRITY__{V.unit_id() or "NOQUEUE"}.json', out)
    summ = {'configs': len(out['configs']), 'errors': len(out['errors']),
            'all_npz_json_ledger': all(c.get('npz_sha_eq_json') and c.get('npz_sha_eq_ledger') for c in out['configs']),
            'all_wires': all(c.get('wire_hashes_match') and c.get('wire_keys_match') for c in out['configs']),
            'all_models': all(c.get('model_sha_eq_meta') and c.get('model_sha_eq_release_json') for c in out['configs']),
            'all_cards': all(c.get('card_sha_eq_commit_blob') and c.get('card_commit_before_save') for c in out['configs']),
            'all_deploy_bitwise': all(c.get('deploy_all_bitwise') for c in out['configs']),
            'selection_matches': {k: e['star_matches'] for k, e in sel.items()},
            'brackets_match': {k: (e['spend_path']['match'], e['sbc_path']['match']) for k, e in sel.items()},
            'record': str(p)}
    print(json.dumps(summ, default=V._default))
    return 0 if not out['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
