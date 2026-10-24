"""Verifier unit (label read): MODEL_SELECT utility, paired costs and the reader-grid winners / SCORE AUCs / bootstrap draws,
recomputed with the verifier's own metric, AUC and bootstrap code (vlib) from the saved MODEL_SELECT arrays, the frozen U0 / Ucal
references, the stored reader scores (reader cache) and the MODEL_SELECT labels read through
sar.common.load('MODEL_SELECT', ..., operation='verify').

Imports used only as loaders: sar.common (label path, paths), sar.release.load (hash-checked saved releases), ohn.common.frozen_bank
(hash-checked frozen bank: U0, Ucal), ohn.devrun.reference_release and tsz.evaluate (frozen reference arrays). No sar.rules,
sar.claims, nrm.rules or sar.devreaders function computes any number here.

    python3 results/pcrl_strong_anchor_release_v1/roles/verifier/code/v_ms.py --out results/.../roles/verifier/work/ms_recomputed.json
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

from sar import common as C  # noqa: E402
from sar import release as RL  # noqa: E402

TOL_UTIL = 1e-9
TOL_AUC = 1e-12

UTIL_RECORDS = {   # name -> (eval record path relative to runs, key in the record or None, loader spec)
    'FB_PFO': ('FB/FB_PFO/EVAL_MS.json', None, ('sar', 'FB', 'FB_PFO')),
    'FB_PFO_TASK': ('FB/FB_PFO_TASK/EVAL_MS.json', None, ('sar', 'FB', 'FB_PFO_TASK')),
    'FB_R1_CTRL_HI': ('ROLES/FB_R1_CTRL_HI/EVAL_MS.json', None, ('sar', 'FB', 'FB_PFO_DQ_HI')),
    'FB_R1_ALT_HI': ('ROLES/FB_R1_ALT_HI/EVAL_MS.json', None, ('sar', 'FB', 'FB_PFO_RAWJ_HI')),
    'FA_TFRO_R1': ('FA/FA_TFRO_R1/EVAL_MS.json', None, ('sar', 'FA', 'FA_TFRO_R1')),
    'FA_TFRO_R1_CONT': ('FA/FA_TFRO_R1_CONT/EVAL_MS.json', None, ('sar', 'FA', 'FA_TFRO_R1_CONT')),
    'FA_R1_CTRL_LO': ('ROLES/FA_R1_CTRL_LO/EVAL_MS.json', None, ('sar', 'FA', 'FA_TFRO_R1_TASK_LO')),
}
REF_NAMES = {'NEURAL_RAW': 'REF:NEURAL_RAW', 'H_LJ': 'REF:H_LJ', 'SUCC': 'REF:SUCC', 'B2TWIN': 'REF:B2TWIN',
             'RETIRED_B3CTL': 'REF:RETIRED_B3CTL', 'TWJ': 'REF:TWJ', 'TR100': 'REF:H3+TASK_REFIT', 'TT100': 'REF:H3+T_TREE',
             'W': 'W', 'U0': 'REF:U0', 'Ucal': 'REF:Ucal'}
COMPARE_SPECS = {'FB/FB_PFO_TASK': 'FB_PFO_TASK', 'REF:NEURAL_RAW': 'NEURAL_RAW', 'ROLE:FB_R1_ALT_HI': 'FB_R1_ALT_HI',
                 'ROLE:FB_R1_CTRL_HI': 'FB_R1_CTRL_HI', 'FA/FA_TFRO_R1_CONT': 'FA_TFRO_R1_CONT', 'REF:SUCC': 'SUCC',
                 'ROLE:FA_R1_CTRL_LO': 'FA_R1_CTRL_LO', 'FB/FB_PFO': 'FB_PFO', 'FA/FA_TFRO_R1': 'FA_TFRO_R1'}
BANKS = ('SAR1_FB_R1', 'SAR1_FA_R1')
INPUTS = {}


def note_input(p):
    p = Path(p)
    INPUTS[str(p.relative_to(V.ROOT)) if p.is_relative_to(V.ROOT) else str(p)] = V.fsha(p)


_REF_CACHE = {}


def load_reference(name, v):
    if (name, v) in _REF_CACHE:
        return _REF_CACHE[(name, v)]
    spec = REF_NAMES[name]
    if spec == 'W':
        from tsz.evaluate import source_a1, h3_income
        rel = {'income': h3_income(v), 'occupation': source_a1(v)}
    else:
        from ohn.devrun import reference_release
        rel = reference_release(spec[4:], v, 'MODEL_SELECT')
    rel = {t: np.ascontiguousarray(np.asarray(rel[t], dtype=np.float64)) for t in V.TASKS}
    _REF_CACHE[(name, v)] = rel
    return rel


def load_sar(fam, cfg, v):
    arr, rec = RL.load(fam, cfg, v, roles=('MODEL_SELECT',))
    note_input(RL.rdir(fam, cfg, v) / 'RELEASE.npz')
    return arr['MODEL_SELECT']


def load_array_member(per_seed):
    out = {}
    for task in V.TASKS:
        src = per_seed[task]
        p = V.ROOT / src['path']
        got = V.fsha(p)
        if got != src['sha256']:
            raise PermissionError(f'{p}: hash differs from the bank spec')
        note_input(p)
        with np.load(p, allow_pickle=False) as z:
            out[task] = z[src['key']].copy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.process_time()
    S = C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'verify',
               'independent verifier replay: MODEL_SELECT utility, paired costs and reader-grid SCORE AUCs / bootstrap SEs '
               'of FB_R1 and FA_R1 recomputed with separate code (verifier wave 1)')
    labels = {'income': np.asarray(S['income']), 'occupation': np.asarray(S['occupation'])}
    sex, unit, row_id = np.asarray(S['sex']), np.asarray(S['unit']), np.asarray(S['row_id'])
    del S
    from ohn import common as OC
    out = {'schema': 'SAR1-verifier-ms-v1', 'utc_start': C.now(), 'n_people': int(len(sex)),
           'n_households': int(len(np.unique(unit))), 'utility': {}, 'paired_costs': {}, 'grid': {}, 'checks': []}

    def check(cid, what, rec_val, mine, tol, extra=None):
        ok = V.close(rec_val, mine, tol) if not isinstance(rec_val, (str, bool)) else rec_val == mine
        out['checks'].append({'id': cid, 'what': what, 'record': rec_val, 'recomputed': mine, 'tolerance': tol,
                              'status': 'PASS' if ok else 'FAIL', **(extra or {})})
        return ok

    releases = {}            # (name, v) -> {'income', 'occupation'} on MODEL_SELECT
    refs_by_v = {}
    for v in (0, 1):
        bank, ids = OC.frozen_bank(v, roles=('MODEL_SELECT',))
        if not np.array_equal(ids['row_id__MODEL_SELECT'], row_id):
            raise AssertionError('frozen bank MODEL_SELECT row order differs from the label export')
        refs_by_v[v] = {r: {t: bank[r]['MODEL_SELECT'][t] for t in V.TASKS} for r in ('U0', 'Ucal')}
        for name in REF_NAMES:
            releases[(name, v)] = load_reference(name, v)
        for name, (_, _, (_, fam, cfg)) in UTIL_RECORDS.items():
            releases[(name, v)] = load_sar(fam, cfg, v)

    # ---------------- check 1: utility (14 conditions per version) and paired costs
    pp = {}
    for (name, v), rel in releases.items():
        u, own = V.utility(labels, rel, refs_by_v[v], sex, unit)
        out['utility'].setdefault(name, {})[str(v)] = u
        pp[(name, v)] = own
    note_input(V.RUNS / 'REFERENCES' / 'EVAL_MS.json')
    refrec = json.loads((V.RUNS / 'REFERENCES' / 'EVAL_MS.json').read_text())
    for name, spec in REF_NAMES.items():
        for v in (0, 1):
            u = refrec['references'][spec][str(v)]['utility']
            mine = out['utility'][name][str(v)]
            for c in u['clauses']:
                n = f"{c['task']}|{c['reference']}|{c['metric']}"
                check(f'U_REF_{name}_v{v}_{n}_value', f'reference {name} v{v} {n} value', c['value'], mine[n]['value'], TOL_UTIL)
                check(f'U_REF_{name}_v{v}_{n}_se', f'reference {name} v{v} {n} household-clustered SE', c['se'], mine[n]['se'], TOL_UTIL)
            for g in u['guards']:
                n = f"guard|{g['group']}"
                check(f'U_REF_{name}_v{v}_{n}_value', f'reference {name} v{v} {n} value', g['value'], mine[n]['value'], TOL_UTIL)
                check(f'U_REF_{name}_v{v}_{n}_se', f'reference {name} v{v} {n} linearized SE', g['se'], mine[n]['se'], TOL_UTIL)
    for name, (path, _, _) in UTIL_RECORDS.items():
        p = V.RUNS / path
        note_input(p)
        rec = json.loads(p.read_text())
        for v in (0, 1):
            ev = rec['versions'][str(v)]
            if ev.get('missing'):
                out['checks'].append({'id': f'U_{name}_v{v}_missing', 'what': f'{name} v{v} record missing', 'status': 'FAIL'})
                continue
            u = ev['utility']
            mine = out['utility'][name][str(v)]
            for c in u['clauses']:
                n = f"{c['task']}|{c['reference']}|{c['metric']}"
                check(f'U_{name}_v{v}_{n}_value', f'{name} v{v} {n} value', c['value'], mine[n]['value'], TOL_UTIL)
                check(f'U_{name}_v{v}_{n}_se', f'{name} v{v} {n} household-clustered SE', c['se'], mine[n]['se'], TOL_UTIL)
            for g in u['guards']:
                n = f"guard|{g['group']}"
                check(f'U_{name}_v{v}_{n}_value', f'{name} v{v} {n} value', g['value'], mine[n]['value'], TOL_UTIL)
                check(f'U_{name}_v{v}_{n}_se', f'{name} v{v} {n} linearized SE', g['se'], mine[n]['se'], TOL_UTIL)
            # the release in the record is the release I evaluated
            rel = releases[(name, v)]
            for t in V.TASKS:
                check(f'U_{name}_v{v}_wire_{t}', f'{name} v{v} {t} wire hash equals the evaluated record',
                      ev['wire_sha256_eval'][t], V.wire_hash(rel[t]), 0)
            for spec, cm in (ev.get('paired_costs') or {}).items():
                if cm.get('missing'):
                    continue
                other = COMPARE_SPECS[spec]
                mine_c = V.paired_costs(pp[(name, v)], pp[(other, v)], sex, unit)
                out['paired_costs'].setdefault(name, {}).setdefault(other, {})[str(v)] = mine_c
                for m, x in cm['costs'].items():
                    check(f'PC_{name}_vs_{other}_v{v}_{m}_value', f'paired cost {name} - {other} v{v} {m}', x['value'],
                          mine_c[m]['value'], TOL_UTIL)
                    check(f'PC_{name}_vs_{other}_v{v}_{m}_se', f'paired cost {name} - {other} v{v} {m} SE', x['se'],
                          mine_c[m]['se'], TOL_UTIL)

    # paired costs needed by the decisions that the records may not carry (computed for completeness)
    for cand, others in (('FB_PFO', ('NEURAL_RAW', 'FB_PFO_TASK', 'FB_R1_CTRL_HI', 'FB_R1_ALT_HI')),
                         ('FA_TFRO_R1', ('SUCC', 'FA_TFRO_R1_CONT', 'FA_R1_CTRL_LO'))):
        for o in others:
            for v in (0, 1):
                out['paired_costs'].setdefault(cand, {}).setdefault(o, {}).setdefault(
                    str(v), V.paired_costs(pp[(cand, v)], pp[(o, v)], sex, unit))

    # FB_R1: is CAND's income bitwise START's (the frozen NEURAL_RAW reference) on MODEL_SELECT?
    out['fb_income_vs_neural_raw'] = {}
    for v in (0, 1):
        a_, b_ = releases[('FB_PFO', v)]['income'], releases[('NEURAL_RAW', v)]['income']
        out['fb_income_vs_neural_raw'][str(v)] = {'bitwise_equal': bool(np.array_equal(a_, b_)),
                                                  'max_abs_diff': float(np.abs(a_ - b_).max()),
                                                  'argmax_equal': bool(np.array_equal(a_.argmax(1), b_.argmax(1)))}

    # ---------------- check 3: grid winners, SCORE AUCs, bootstrap draws, paired cells (all cells, members, views)
    sex_hash = V.ahash(sex)
    cache = V.RUNS / 'devgrid_cache'
    for bank in BANKS:
        spec_p = V.RES / 'grid_specs' / f'{bank}.json'
        note_input(spec_p)
        spec = json.loads(spec_p.read_text())
        out['grid'][bank] = {}
        for v in (0, 1):
            gdir = V.RUNS / 'GRID' / bank / f'S{v}'
            for f in ('GRID.json', 'BOOT.npz', 'BOOT_EXTRAS.npz'):
                note_input(gdir / f)
            g = json.loads((gdir / 'GRID.json').read_text())
            boot = np.load(gdir / 'BOOT.npz')
            bx = np.load(gdir / 'BOOT_EXTRAS.npz')
            mem = {}
            for name, m in spec['members'].items():
                t = m['type']
                if t == 'sar':
                    rel = load_sar(m['family'], m['config'], v)
                elif t == 'arrays':
                    rel = load_array_member(m['seeds'][str(v)])
                elif t == 'reference':
                    rel = load_reference({'H3+TASK_REFIT': 'TR100', 'H3+T_TREE': 'TT100'}.get(m['name'], m['name']), v)
                elif t == 'witness':
                    rel = load_reference('W', v)
                else:
                    raise ValueError(t)
                mem[name] = rel
                prov = g['members'][name]['wire_sha256']
                for task in V.TASKS:
                    check(f'G_{bank}_v{v}_{name}_wire_{task}', f'{bank} v{v} member {name} {task} wire = grid record',
                          prov[task], V.wire_hash(rel[task]), 0)
            boot_mine = {}
            for rep in ('R1', 'R2'):
                sc = V.split_roles(unit, rep)['SCORE']
                y = sex[sc]
                W = V.boot_weights(unit[sc], v, rep)
                for k in (0, 1):
                    cell = f'S{v}{rep}k{k}'
                    rs = V.reader_seed(v, k)
                    for name, rel in mem.items():
                        rd = {}
                        for view in ('income', 'occupation', 'pair'):
                            key = V.cache_key(rep, rs, 1.0, V.view_hash(view, rel['income'], rel['occupation']), sex_hash)
                            p = cache / f'{key}.npz'
                            with np.load(p, allow_pickle=False) as z:
                                info = json.loads(str(z['info']))
                                rd[view] = {n: (z[f'score__{n}'].copy(), info[n]) for n in V.RECIPES}
                        for view in ('income', 'occupation', 'pair'):
                            for which, allowed in (('registered', V.RECIPES), ('nolookup', tuple(r for r in V.RECIPES if r != 'LOOKUP'))):
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
                                sa = V.auc_weighted(y, s)
                                draws = V.auc_draws(y, s, W)
                                entry = g['cells'][cell][name][view]
                                if which == 'registered':
                                    recw, reca, recb = entry['winner'], entry['score_auc'], boot[f'{cell}__{name}__{view}']
                                    rec_val = entry['val_auc']
                                else:
                                    recw, reca = entry['extras']['nolookup']['winner'], entry['extras']['nolookup']['score_auc']
                                    recb = bx[f'{cell}__{name}__{view}__NOLOOKUP']
                                    rec_val = entry['extras']['nolookup']['val_auc']
                                tag = f'{bank}_{cell}_{name}_{view}_{which}'
                                check(f'GW_{tag}', f'{which} winner (max VALIDATION AUC, ties to the earlier candidate) {bank} {cell} {name} {view}', recw, wname, 0)
                                check(f'GV_{tag}', f'{which} winner VALIDATION AUC (cache info) {bank} {cell} {name} {view}', rec_val, float(wval), TOL_AUC)
                                check(f'GA_{tag}', f'{which} winner SCORE AUC from stored scores {bank} {cell} {name} {view}', reca, sa, TOL_AUC)
                                check(f'GB_{tag}', f'{which} winner household-bootstrap draws (max abs diff) {bank} {cell} {name} {view}',
                                      0.0, float(np.abs(np.asarray(recb) - draws).max()), TOL_AUC)
                                boot_mine[(cell, name, view, which)] = (sa, draws)
            # paired cells and their bootstrap SEs (candidate vs every comparator)
            cand = spec['candidates'][0]
            pr = {}
            for comp in g['paired'][cand]:
                for view in ('pair', 'income', 'occupation'):
                    for cell in g['paired'][cand][comp][view]['cells']:
                        a_sa, a_d = boot_mine[(cell, cand, view, 'registered')]
                        b_sa, b_d = boot_mine[(cell, comp, view, 'registered')]
                        if view == 'pair':
                            d, db = b_sa - a_sa, b_d - a_d
                        else:
                            d, db = a_sa - b_sa, a_d - b_d
                        se = float(np.std(db, ddof=1))
                        pr.setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                        pr[comp][view]['cells'][cell] = float(d)
                        pr[comp][view]['cell_se'][cell] = se
                        rv = g['paired'][cand][comp][view]
                        check(f'GP_{bank}_{cell}_{cand}_vs_{comp}_{view}_value', f'paired cell {view} {cand} vs {comp} {cell}',
                              rv['cells'][cell], float(d), TOL_AUC)
                        check(f'GP_{bank}_{cell}_{cand}_vs_{comp}_{view}_se', f'paired cell bootstrap SE {view} {cand} vs {comp} {cell}',
                              rv['cell_se'][cell], se, TOL_AUC)
            # non-table (nolookup) and enhanced point benefits per cell (enhanced taken from the record: TABLE_CV scores not cached)
            nl, enh = {}, {}
            for comp in g['paired'][cand]:
                for cell in g['paired'][cand][comp]['pair']['cells']:
                    a_sa = boot_mine[(cell, cand, 'pair', 'nolookup')][0]
                    b_sa = boot_mine[(cell, comp, 'pair', 'nolookup')][0]
                    nl.setdefault(comp, {})[cell] = float(b_sa - a_sa)
                    ea = g['cells'][cell][cand]['pair']['extras']['enhanced']['score_auc']
                    eb = g['cells'][cell][comp]['pair']['extras']['enhanced']['score_auc']
                    enh.setdefault(comp, {})[cell] = float(eb - ea)
            out['grid'][bank][str(v)] = {'paired': pr, 'nolookup_pair_benefit': nl, 'enhanced_pair_benefit_from_record': enh,
                                         'exact_aliases_record': g.get('exact_aliases')}
    out['inputs_sha256'] = INPUTS
    out['cpu_s'] = time.process_time() - t0
    out['utc_end'] = C.now()
    n_fail = sum(c['status'] != 'PASS' for c in out['checks'])
    out['summary'] = {'checks': len(out['checks']), 'fail': n_fail}
    V.jdump(Path(a.out), out)
    print(json.dumps({'checks': len(out['checks']), 'fail': n_fail, 'cpu_s': out['cpu_s']}))


if __name__ == '__main__':
    main()
