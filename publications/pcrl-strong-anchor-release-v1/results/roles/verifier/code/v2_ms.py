"""Verifier wave 2 unit (label read): FA_R2 MODEL_SELECT utility and paired costs, and the full reader-grid recomputation (winner
re-pick from stored VALIDATION AUCs, SCORE AUCs and household-bootstrap draws from the stored scores, paired cells and SEs) for the
banks given on the command line (SAR1_FA_R2; SAR1_FB_R1_ROB with its R3 / R4 cells and k2 / k3 seeds; later SAR1_FA_R2_ROB).
Own metric / AUC / bootstrap code (vlib); loaders only from sar.common, sar.release, ohn.common, ohn.devrun, tsz.evaluate (via v_ms).

    python3 .../roles/verifier/code/v2_ms.py --out .../work/ms2_<tag>.json [--util] --bank SAR1_FA_R2 [--bank ...]
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
import v_ms as W1  # noqa: E402  (wave-1 loaders: load_reference, load_sar, note_input; no label read at import)

import numpy as np  # noqa: E402
from sar import common as C  # noqa: E402

TOL_UTIL, TOL_AUC = 1e-9, 1e-12
UTIL2 = {   # name -> (record path under runs, per-version loader spec)
    'FA_TFRB_R2': ('FA/FA_TFRB_R2/EVAL_MS.json', {0: ('sar', 'FA', 'FA_TFRB_R2'), 1: ('sar', 'FA', 'FA_TFRB_R2')}),
    'FA_R2_CTRL_LO': ('ROLES/FA_R2_CTRL_LO/EVAL_MS.json', {0: ('sar', 'FA', 'FA_TFRB_R2_TASK_LO'), 1: ('sar', 'FA', 'FA_TFRB_R2_TASK_LO')}),
    'FA_R2_CTRL_HI': ('ROLES/FA_R2_CTRL_HI/EVAL_MS.json', {0: ('sar', 'FA', 'FA_TFRB_R2_TASK_HI'), 1: ('ref', 'SUCC')}),
    'FA_TFRO_R1_CONT': ('FA/FA_TFRO_R1_CONT/EVAL_MS.json', {0: ('sar', 'FA', 'FA_TFRO_R1_CONT'), 1: ('sar', 'FA', 'FA_TFRO_R1_CONT')}),
}
EXTRA = {'SUCC': {0: ('ref', 'SUCC'), 1: ('ref', 'SUCC')},
         'FA_R1_CTRL_LO': {0: ('sar', 'FA', 'FA_TFRO_R1_TASK_LO'), 1: ('sar', 'FA', 'FA_TFRO_R1_TASK_LO')}}
SPEC = {'FA/FA_TFRO_R1_CONT': 'FA_TFRO_R1_CONT', 'REF:SUCC': 'SUCC', 'ROLE:FA_R2_CTRL_HI': 'FA_R2_CTRL_HI',
        'ROLE:FA_R2_CTRL_LO': 'FA_R2_CTRL_LO', 'ROLE:FA_R1_CTRL_LO': 'FA_R1_CTRL_LO', 'FA/FA_TFRB_R2': 'FA_TFRB_R2'}


def load_spec(spec, v):
    if spec[0] == 'ref':
        return W1.load_reference(spec[1], v)
    return W1.load_sar(spec[1], spec[2], v)


def array_member(per_seed):
    out = {}
    for task in V.TASKS:
        src = per_seed[task]
        if 'frozen_reference' in src:
            out[task] = W1.load_reference(src['frozen_reference'], VCUR[0])[task]
            continue
        p = V.ROOT / src['path']
        if V.fsha(p) != src['sha256']:
            raise PermissionError(f'{p}: hash differs from the bank spec')
        W1.note_input(p)
        with np.load(p, allow_pickle=False) as z:
            out[task] = z[src['key']].copy()
    return out


VCUR = [0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--util', action='store_true')
    ap.add_argument('--bank', action='append', default=[])
    a = ap.parse_args()
    t0 = time.process_time()
    S = C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'verify',
               'independent verifier wave 2: FA_R2 MODEL_SELECT utility / paired costs and the reader-grid recomputation of '
               + ', '.join(a.bank) + ' with separate code')
    labels = {'income': np.asarray(S['income']), 'occupation': np.asarray(S['occupation'])}
    sex, unit, row_id = np.asarray(S['sex']), np.asarray(S['unit']), np.asarray(S['row_id'])
    del S
    from ohn import common as OC
    out = {'schema': 'SAR1-verifier-ms2-v1', 'utc_start': C.now(), 'n_households': int(len(np.unique(unit))), 'utility': {},
           'paired_costs': {}, 'grid': {}, 'checks': []}

    def check(cid, what, rec_val, mine, tol):
        ok = V.close(rec_val, mine, tol) if not isinstance(rec_val, (str, bool)) else rec_val == mine
        out['checks'].append({'id': cid, 'what': what, 'record': rec_val, 'recomputed': mine, 'tolerance': tol,
                              'status': 'PASS' if ok else 'FAIL'})

    if a.util:
        rel, pp, refs_by_v = {}, {}, {}
        for v in (0, 1):
            bank, ids = OC.frozen_bank(v, roles=('MODEL_SELECT',))
            if not np.array_equal(ids['row_id__MODEL_SELECT'], row_id):
                raise AssertionError('frozen bank row order differs')
            refs_by_v[v] = {r: {t: bank[r]['MODEL_SELECT'][t] for t in V.TASKS} for r in ('U0', 'Ucal')}
            for name, (_, specs) in UTIL2.items():
                rel[(name, v)] = load_spec(specs[v], v)
            for name, specs in EXTRA.items():
                rel[(name, v)] = load_spec(specs[v], v)
        for (name, v), r in rel.items():
            u, own = V.utility(labels, r, refs_by_v[v], sex, unit)
            out['utility'].setdefault(name, {})[str(v)] = u
            pp[(name, v)] = own
        for name, (path, _) in UTIL2.items():
            p = V.RUNS / path
            W1.note_input(p)
            rec = json.loads(p.read_text())
            for v in (0, 1):
                ev = rec['versions'][str(v)]
                u, mine = ev['utility'], out['utility'][name][str(v)]
                for c in u['clauses']:
                    n = f"{c['task']}|{c['reference']}|{c['metric']}"
                    check(f'U_{name}_v{v}_{n}_value', f'{name} v{v} {n} value', c['value'], mine[n]['value'], TOL_UTIL)
                    check(f'U_{name}_v{v}_{n}_se', f'{name} v{v} {n} SE', c['se'], mine[n]['se'], TOL_UTIL)
                for g in u['guards']:
                    n = f"guard|{g['group']}"
                    check(f'U_{name}_v{v}_{n}_value', f'{name} v{v} {n} value', g['value'], mine[n]['value'], TOL_UTIL)
                    check(f'U_{name}_v{v}_{n}_se', f'{name} v{v} {n} SE', g['se'], mine[n]['se'], TOL_UTIL)
                for t in V.TASKS:
                    check(f'U_{name}_v{v}_wire_{t}', f'{name} v{v} {t} wire = evaluated record', ev['wire_sha256_eval'][t],
                          V.wire_hash(rel[(name, v)][t]), 0)
                for spec, cm in (ev.get('paired_costs') or {}).items():
                    if cm.get('missing'):
                        continue
                    other = SPEC[spec]
                    mc = V.paired_costs(pp[(name, v)], pp[(other, v)], sex, unit)
                    out['paired_costs'].setdefault(name, {}).setdefault(other, {})[str(v)] = mc
                    for m, x in cm['costs'].items():
                        check(f'PC_{name}_vs_{other}_v{v}_{m}_value', f'paired cost {name} - {other} v{v} {m}', x['value'], mc[m]['value'], TOL_UTIL)
                        check(f'PC_{name}_vs_{other}_v{v}_{m}_se', f'paired cost {name} - {other} v{v} {m} SE', x['se'], mc[m]['se'], TOL_UTIL)
        # the FA_R2 v1 CTRL_HI role is the frozen SUCC reference: its wire equals SUCC's
        out['ctrl_hi_v1_is_succ'] = all(np.array_equal(rel[('FA_R2_CTRL_HI', 1)][t], rel[('SUCC', 1)][t]) for t in V.TASKS)
        # CAND's income equals SUCC's income bitwise (card FA_TFRB_R2 section 1)
        out['fa_r2_income_vs_succ'] = {str(v): {'bitwise_equal': bool(np.array_equal(rel[('FA_TFRB_R2', v)]['income'], rel[('SUCC', v)]['income'])),
                                                'max_abs_diff': float(np.abs(rel[('FA_TFRB_R2', v)]['income'] - rel[('SUCC', v)]['income']).max())}
                                       for v in (0, 1)}

    sex_hash = V.ahash(sex)
    cache = V.RUNS / 'devgrid_cache'
    for bank in a.bank:
        sp = V.RES / 'grid_specs' / f'{bank}.json'
        W1.note_input(sp)
        spec = json.loads(sp.read_text())
        out['grid'][bank] = {}
        for v in (0, 1):
            VCUR[0] = v
            gdir = V.RUNS / 'GRID' / bank / f'S{v}'
            for f in ('GRID.json', 'BOOT.npz', 'BOOT_EXTRAS.npz'):
                W1.note_input(gdir / f)
            g = json.loads((gdir / 'GRID.json').read_text())
            boot, bx = np.load(gdir / 'BOOT.npz'), np.load(gdir / 'BOOT_EXTRAS.npz')
            members = spec['members']
            if isinstance(members, list):          # a robustness bank names members of the main bank
                main = json.loads((V.RES / 'grid_specs' / f"{bank.replace('_ROB', '')}.json").read_text())['members']
                members = {n: main[n] for n in members}
            mem = {}
            for name, m in members.items():
                t = m['type']
                if t == 'sar':
                    r = W1.load_sar(m['family'], m['config'], v)
                elif t == 'arrays':
                    r = array_member(m['seeds'][str(v)])
                elif t == 'reference':
                    r = W1.load_reference({'H3+TASK_REFIT': 'TR100', 'H3+T_TREE': 'TT100'}.get(m['name'], m['name']), v)
                elif t == 'witness':
                    r = W1.load_reference('W', v)
                else:
                    raise ValueError(t)
                mem[name] = r
                for task in V.TASKS:
                    check(f'G_{bank}_v{v}_{name}_wire_{task}', f'{bank} v{v} member {name} {task} wire = grid record',
                          g['members'][name]['wire_sha256'][task], V.wire_hash(r[task]), 0)
            bm = {}
            cells = sorted(g['cells'])
            for cell in cells:
                rep, k = cell[2:4], int(cell[5:])
                sc = V.split_roles(unit, rep)['SCORE']
                y = sex[sc]
                W = V.boot_weights(unit[sc], v, rep)
                rs = V.reader_seed(v, k)
                for name, r in mem.items():
                    rd = {}
                    for view in ('income', 'occupation', 'pair'):
                        key = V.cache_key(rep, rs, 1.0, V.view_hash(view, r['income'], r['occupation']), sex_hash)
                        with np.load(cache / f'{key}.npz', allow_pickle=False) as z:
                            info = json.loads(str(z['info']))
                            rd[view] = {n: (z[f'score__{n}'].copy(), info[n]) for n in V.RECIPES}
                    for view in ('income', 'occupation', 'pair'):
                        for which, allowed in (('registered', V.RECIPES), ('nolookup', tuple(x for x in V.RECIPES if x != 'LOOKUP'))):
                            if view == 'pair':
                                cands = []
                                for grp in (V.SRC8, V.NEW8):
                                    cands += [(n, rd['pair'][n][1]['val_auc'], ('pair', n)) for n in grp if n in allowed]
                                    cands += [(f'IGNORE_occupation__{n}', rd['income'][n][1]['val_auc'], ('income', n)) for n in grp if n in allowed]
                                    cands += [(f'IGNORE_income__{n}', rd['occupation'][n][1]['val_auc'], ('occupation', n)) for n in grp if n in allowed]
                            else:
                                cands = [(n, rd[view][n][1]['val_auc'], (view, n)) for n in allowed]
                            win = max(range(len(cands)), key=lambda i: (cands[i][1], -i))
                            wname, wval, (wv, wn) = cands[win]
                            s = rd[wv][wn][0]
                            sa, draws = V.auc_weighted(y, s), V.auc_draws(y, s, W)
                            e = g['cells'][cell][name][view]
                            if which == 'registered':
                                recw, reca, recb, recv = e['winner'], e['score_auc'], boot[f'{cell}__{name}__{view}'], e['val_auc']
                            else:
                                nl = e['extras']['nolookup']
                                recw, reca, recb, recv = nl['winner'], nl['score_auc'], bx[f'{cell}__{name}__{view}__NOLOOKUP'], nl['val_auc']
                            tag = f'{bank}_{cell}_{name}_{view}_{which}'
                            check(f'GW_{tag}', f'{which} winner re-picked {tag}', recw, wname, 0)
                            check(f'GV_{tag}', f'{which} winner VALIDATION AUC {tag}', recv, float(wval), TOL_AUC)
                            check(f'GA_{tag}', f'{which} winner SCORE AUC {tag}', reca, sa, TOL_AUC)
                            check(f'GB_{tag}', f'{which} winner bootstrap draws (max abs diff) {tag}', 0.0, float(np.abs(np.asarray(recb) - draws).max()), TOL_AUC)
                            bm[(cell, name, view, which)] = (sa, draws)
            cand = spec['candidates'][0]
            pr, nl, enh = {}, {}, {}
            for comp in g['paired'][cand]:
                for view in ('pair', 'income', 'occupation'):
                    for cell in g['paired'][cand][comp][view]['cells']:
                        a_sa, a_d = bm[(cell, cand, view, 'registered')]
                        b_sa, b_d = bm[(cell, comp, view, 'registered')]
                        d, db = (b_sa - a_sa, b_d - a_d) if view == 'pair' else (a_sa - b_sa, a_d - b_d)
                        se = float(np.std(db, ddof=1))
                        pr.setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                        pr[comp][view]['cells'][cell], pr[comp][view]['cell_se'][cell] = float(d), se
                        rv = g['paired'][cand][comp][view]
                        check(f'GP_{bank}_{cell}_{cand}_vs_{comp}_{view}_value', f'paired cell {view} {cand} vs {comp} {cell}', rv['cells'][cell], float(d), TOL_AUC)
                        check(f'GP_{bank}_{cell}_{cand}_vs_{comp}_{view}_se', f'paired cell SE {view} {cand} vs {comp} {cell}', rv['cell_se'][cell], se, TOL_AUC)
                for cell in g['paired'][cand][comp]['pair']['cells']:
                    nl.setdefault(comp, {})[cell] = float(bm[(cell, comp, 'pair', 'nolookup')][0] - bm[(cell, cand, 'pair', 'nolookup')][0])
                    enh.setdefault(comp, {})[cell] = float(g['cells'][cell][comp]['pair']['extras']['enhanced']['score_auc']
                                                           - g['cells'][cell][cand]['pair']['extras']['enhanced']['score_auc'])
            out['grid'][bank][str(v)] = {'paired': pr, 'nolookup_pair_benefit': nl, 'enhanced_pair_benefit_from_record': enh,
                                         'cells': cells}
    out['inputs_sha256'] = W1.INPUTS
    out['cpu_s'] = time.process_time() - t0
    out['utc_end'] = C.now()
    out['summary'] = {'checks': len(out['checks']), 'fail': sum(c['status'] != 'PASS' for c in out['checks'])}
    V.jdump(Path(a.out), out)
    print(json.dumps({**out['summary'], 'cpu_s': out['cpu_s']}))


if __name__ == '__main__':
    main()
