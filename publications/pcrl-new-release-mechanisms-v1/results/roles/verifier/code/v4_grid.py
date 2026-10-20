"""V4 recovery and uncertainty: replay of a development reader grid (runs/GRID/<bank>/S<v>/GRID.json, BOOT.npz) from the SAVED
reader scores in this campaign's cache, with the verifier's own code: cache keys (view wire hash, reader seed, repetition, SEX
array hash), winners from the cached VALIDATION AUCs (own pick rule), SCORE AUCs (own Mann-Whitney), household-bootstrap draws (own
multiplicities and expanded-sample AUC), paired cell differences and their bootstrap SEs. Also checks that the wires the grid read
are the saved releases (member wire hashes recomputed from the release files).

Labels: MODEL_SELECT SEX only, through nrm.common.load(operation 'verify') called from THIS script (logged).

    python3 .../v4_grid.py --tag V1 --bank BANK_ID [--seeds 0 1] [--boot all|paired|none]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402
import vref  # noqa: E402

TOL = 1e-12
CACHE = V.RUNS / 'devgrid_cache'


def load_sex(purpose):
    sys.path.insert(0, str(V.ROOT))
    import nrm.common as C
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'verify', purpose)
    return np.asarray(S['sex']), np.asarray(S['unit']), np.asarray(S['row_id'])


REP_INSTANCE = {'R1': 0, 'R2': 1, 'R3': 6, 'R4': 7}   # declared instance map (DECLARATIONS_V1, reviewer's ruling; lock 1128afd67)


def draw_index(rule, rep):
    """Declared per-cell instance of a randomized release: R1 0, R2 1, R3 6, R4 7 (both reader seeds of a cell read the same
    instance) unless a fixed draw is given, plus an optional Monte Carlo offset (+2, +4) allowed on R1 / R2 only."""
    d = rule.get('draw', 'rep')
    off = int(rule.get('draw_offset', 0))
    if d == 'rep':
        if off and rep not in V.REG_REPS:
            raise ValueError('Monte Carlo offsets apply to R1 / R2 only')
        return REP_INSTANCE[rep] + off
    return int(d) + off


def member_arrays(m, v):
    t = m.get('type')
    if t == 'nrm':
        arr, rec, prov = vref.release(m['family'], m['config'], v)
        ms = arr['MODEL_SELECT']
        if rec.get('randomized'):
            rule = {'draw': m.get('draw', 'rep'), 'draw_offset': int(m.get('draw_offset', 0))}
            rel = {task: ms[task] for task in V.TASKS}
            return rel, {**prov, 'draw_rule': rule, 'randomized': True, 'kind': rec.get('kind')}
        return {task: ms[task] for task in V.TASKS}, {**prov, 'randomized': False, 'kind': rec.get('kind')}
    if t == 'witness':
        return vref.reference('W', v)
    if t == 'reference' and m['name'] in vref.FROZEN + ('B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ') + tuple(vref.COMPONENTS):
        return vref.reference('REF:' + m['name'], v)
    if t == 'arrays':
        rel, prov = {}, {}
        for task in V.TASKS:
            src = m['seeds'][str(v)][task]
            if 'path' in src:
                p = Path(src['path']); p = p if p.is_absolute() else V.ROOT / p
                h = V.file_sha(p)
                if h != src['sha256']:
                    raise PermissionError(f'{p}: hash differs from the spec')
                with np.load(p, allow_pickle=False) as z:
                    rel[task] = z[src['key']].copy()
                prov[task] = {'path': V.rel(p), 'sha256': h}
            else:
                sys.path.insert(0, str(V.ROOT))
                from ohn import devrun as D                       # source DATA loaders (admitted copies), no metric code
                if 'frozen_anchor' in src:
                    rel[task] = D.anchor_income(src['frozen_anchor'], v, 'MODEL_SELECT')
                elif 'frozen_reference' in src:
                    rel[task] = D.reference_release(src['frozen_reference'], v, 'MODEL_SELECT')[task]
                else:
                    f = src['frozen_occ']
                    rel[task] = D.load_occupation(f['family'], f['config'], v, roles=('MODEL_SELECT',))[0]['MODEL_SELECT']
                prov[task] = src
        return rel, prov
    sys.path.insert(0, str(V.ROOT))
    from ohn.devreaders import load_member                       # frozen source releases (DATA loader)
    return load_member(m, v), {'frozen_member': m}


def lock_status(g):
    """The grid unit ran the locked reader code: DEVGRID_CODE_LOCK.json committed, and the unit's launch hashes of every locked file
    equal the lock (queue CODE_AT_START / CODE_AT_LAUNCH)."""
    lock = V.RESULTS / 'DEVGRID_CODE_LOCK.json'
    out = {'lock_committed': lock.is_file() and V.committed_unchanged(lock), 'unit_id': g.get('unit_id')}
    if not lock.is_file():
        return out
    lk = json.loads(lock.read_text())
    out['lock_sha256'] = V.file_sha(lock)
    uid = g.get('unit_id')
    d = V.PRIV / 'compute' / 'units' / str(uid)
    p = d / 'CODE_AT_LAUNCH.json' if (d / 'CODE_AT_LAUNCH.json').is_file() else d / 'CODE_AT_START.json'
    if uid and p.is_file():
        files = json.loads(p.read_text())['files']
        out['unit_files_equal_lock'] = {f: files.get(f) == h for f, h in lk['files_sha256'].items()}
        out['all_equal'] = all(out['unit_files_equal_lock'].values())
        rc, t, _ = V.git('log', '-1', '--format=%cI', '--', str(lock.relative_to(V.ROOT)))
        out['lock_commit_utc'] = t
    return out


def read_cache(view, inc, occ, rep, rs, sex_sha, n_score):
    wh = V.view_hash(view, inc, occ)
    p = CACHE / f'{V.cache_key(rep, rs, 1.0, wh, sex_sha)}.npz'
    if not p.is_file():
        return None, p
    with np.load(p, allow_pickle=False) as z:
        info = json.loads(str(z['info']))
        sc = {n: z[f'score__{n}'].copy() for n in V.RECIPES if f'score__{n}' in z.files}
    bad = set(info) != set(V.RECIPES) or set(sc) != set(V.RECIPES) or any(s.shape != (n_score,) or not np.isfinite(s).all() for s in sc.values())
    return {'info': info, 'scores': sc, 'malformed': bool(bad), 'path': p}, p


def winner_scores(w, rd):
    view, name = w['reader_key'].split('__', 1)
    return rd[view]['scores'][name]


def replay_seed(bank_dir, sex, unit, boot_mode, members_filter=None):
    g = json.loads((bank_dir / 'GRID.json').read_text())
    boot = np.load(bank_dir / 'BOOT.npz')
    v = int(g['seed'])
    spec = g['spec']
    sx = V.arr_sha(sex)
    mm, info = [], {'cells': {}, 'members': {}}
    rels, rules = {}, {}
    for name, m in spec['members'].items():
        rel, prov = member_arrays(m, v)
        rel = {t: np.ascontiguousarray(np.asarray(rel[t], dtype=np.float64)) for t in V.TASKS}
        rels[name] = rel
        if prov.get('randomized'):
            rules[name] = prov['draw_rule']
        w = {t: ([V.wire_sha(x) for x in rel[t]] if rel[t].ndim == 3 else V.wire_sha(rel[t])) for t in V.TASKS}
        rec_w = (g.get('members') or {}).get(name, {}).get('wire_sha256')
        info['members'][name] = {'wire_sha256': w, 'recorded': rec_w, 'equal': rec_w == w, 'provenance': prov}
        if rec_w != w:
            mm.append({'what': f'member {name} wire differs from the grid record', 'mine': w, 'recorded': rec_w})
    max_auc = max_boot = max_pair = max_se = 0.0
    my_boot = {}
    pair_needed = set()
    for cand, comps in g.get('paired', {}).items():
        for comp in comps:
            pair_needed |= {cand, comp}
    for cell, cres in g['cells'].items():
        if '@' in cell:
            continue
        rep, k = cell[2:4], int(cell.split('k')[1])
        rs = V.reader_seed(v, k)
        roles = V.reader_roles(unit, rep)
        sc = roles['SCORE']
        y = sex[sc]
        n_sc = int(sc.sum())
        counts = V.boot_counts(unit[sc], V.boot_seed(v, rep)) if boot_mode != 'none' else None
        for name, ent in cres.items():
            if members_filter and name not in members_filter:
                continue
            rel = rels[name]
            inc, occ = rel['income'], rel['occupation']
            if inc.ndim == 3 or occ.ndim == 3:                     # randomized: this cell's ONE emitted draw per person
                di = draw_index(rules.get(name, {}), rep)
                inc = inc[di] if inc.ndim == 3 else inc
                occ = occ[di] if occ.ndim == 3 else occ
                info.setdefault('draws_read', {})[f'{cell}|{name}'] = di
            rd = {}
            for view in V.VIEWS:
                e, p = read_cache(view, inc, occ, rep, rs, sx, n_sc)
                if e is None:
                    mm.append({'what': f'{cell} {name} {view}: cache entry missing', 'path': V.rel(p)}); continue
                if e['malformed']:
                    mm.append({'what': f'{cell} {name} {view}: cache entry malformed'})
                rd[view] = e
            if len(rd) < 3:
                continue
            val = {view: {n: float(rd[view]['info'][n]['val_auc']) for n in V.RECIPES} for view in V.VIEWS}
            wins = {'income': V.pick(V.single_candidates(val, 'income')), 'occupation': V.pick(V.single_candidates(val, 'occupation')),
                    'pair': V.pick(V.pair_candidates(val))}
            for view in V.VIEWS:
                r = ent[view]
                w = wins[view]
                if w['candidate'] != r['winner'] or abs(w['val_auc'] - float(r['val_auc'])) > TOL or w['near_tie'] != bool(r['near_tie']):
                    mm.append({'what': f'{cell} {name} {view}: winner', 'mine': [w['candidate'], w['val_auc'], w['near_tie']],
                               'recorded': [r['winner'], r['val_auc'], r['near_tie']]})
                if r.get('gap_to_runner_up') is not None and abs(w['gap'] - float(r['gap_to_runner_up'])) > TOL:
                    mm.append({'what': f'{cell} {name} {view}: gap', 'mine': w['gap'], 'recorded': r['gap_to_runner_up']})
                s = winner_scores(w, rd)
                a = V.auc(y, s)
                max_auc = max(max_auc, abs(a - float(r['score_auc'])))
                info['cells'].setdefault(cell, {}).setdefault(name, {})[view] = {'winner': w['candidate'], 'score_auc': a,
                                                                                'recorded_score_auc': r['score_auc'], 'val_auc': w['val_auc']}
                do_boot = boot_mode == 'all' or (boot_mode == 'paired' and name in pair_needed)
                if do_boot:
                    bs = np.array([V.auc_expanded(y, s, counts[b]) for b in range(counts.shape[0])])
                    my_boot[(cell, name, view)] = bs
                    key = f'{cell}__{name}__{view}'
                    if key not in boot.files:
                        mm.append({'what': f'{key} missing in BOOT.npz'})
                    else:
                        max_boot = max(max_boot, float(np.abs(bs - boot[key]).max()))
    if max_auc > TOL:
        mm.append({'what': 'SCORE AUCs', 'max_abs_diff': max_auc})
    if max_boot > TOL:
        mm.append({'what': 'bootstrap draws', 'max_abs_diff': max_boot})
    pairs = {}
    for cand, comps in g.get('paired', {}).items():
        for comp, per in comps.items():
            for view in V.VIEWS:
                for cell, d_rec in per[view]['cells'].items():
                    a = info['cells'].get(cell, {}).get(cand, {}).get(view)
                    b = info['cells'].get(cell, {}).get(comp, {}).get(view)
                    if a is None or b is None:
                        continue
                    d = (b['score_auc'] - a['score_auc']) if view == 'pair' else (a['score_auc'] - b['score_auc'])
                    max_pair = max(max_pair, abs(d - float(d_rec)))
                    if (cell, cand, view) in my_boot and (cell, comp, view) in my_boot:
                        db = (my_boot[(cell, comp, view)] - my_boot[(cell, cand, view)]) if view == 'pair' else \
                             (my_boot[(cell, cand, view)] - my_boot[(cell, comp, view)])
                        se = float(np.std(db, ddof=1))
                        max_se = max(max_se, abs(se - float(per[view]['cell_se'][cell])))
                        pairs.setdefault(f'{cand}|{comp}|{view}', {})[cell] = {'d': d, 'se': se}
                    else:
                        pairs.setdefault(f'{cand}|{comp}|{view}', {})[cell] = {'d': d}
    if max_pair > TOL:
        mm.append({'what': 'paired cell differences', 'max_abs_diff': max_pair})
    if max_se > TOL:
        mm.append({'what': 'paired bootstrap SEs', 'max_abs_diff': max_se})
    summary = {'seed': v, 'n_cells': len([c for c in g['cells'] if '@' not in c]), 'n_members': len(spec['members']),
               'max_abs_score_auc_diff': max_auc, 'max_abs_boot_diff': max_boot, 'max_abs_pair_diff': max_pair,
               'max_abs_pair_se_diff': max_se, 'boot_replayed': len(my_boot), 'mismatches': mm}
    return summary, info, pairs, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--bank', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1]); ap.add_argument('--boot', default='paired', choices=('all', 'paired', 'none'))
    ap.add_argument('--members', nargs='*', default=None)
    ap.add_argument('--dir', action='append', default=[], help='explicit run folder per seed, V=FOLDER (e.g. 1=S1_V2)')
    a = ap.parse_args()
    explicit = {int(x.split('=')[0]): x.split('=')[1] for x in a.dir}
    sex, unit, _ = load_sex(f'verify: replay of reader grid {a.bank} from saved reader scores (own AUC, winners, bootstrap; SEX only)')
    checks, record = [], {'tag': a.tag, 'bank': a.bank, 'utc': V.utc(), 'seeds': {}}
    for v in a.seeds:
        d = V.RUNS / 'GRID' / a.bank / explicit.get(v, f'S{v}')
        if not (d / 'GRID.json').is_file():
            checks.append({'id': f'V4.GRID.{a.bank}.S{v}__{a.tag}', 'what': f'grid {a.bank} seed {v}: no GRID.json in {V.rel(d)}',
                           'inputs': {}, 'result': {'missing': V.rel(d)}, 'numerical_status': 'NOT_RUN', 'procedural_status': 'NOT_RUN'})
            continue
        summ, info, pairs, g = replay_seed(d, sex, unit, a.boot, set(a.members) if a.members else None)
        lk = lock_status(g)
        record['seeds'][str(v)] = {'summary': summ, 'cells': info['cells'], 'members': info['members'], 'pairs': pairs,
                                   'draws_read': info.get('draws_read'), 'lock': lk}
        wire_bad = [n for n, x in info['members'].items() if not x['equal']]
        proc_ok = not wire_bad and lk.get('lock_committed') and lk.get('all_equal')
        checks.append({'id': f'V4.GRID.{a.bank}.S{v}__{a.tag}',
                       'what': f'grid {a.bank} seed {v} replayed from saved reader scores: cache keys, winners (own pick on cached '
                               f'VALIDATION AUCs), SCORE AUCs, bootstrap draws ({a.boot}), paired differences and SEs; member wires = saved; '
                               'grid unit ran the locked reader code',
                       'inputs': V.inputs_sha([d / 'GRID.json', d / 'BOOT.npz', V.RESULTS / 'DEVGRID_CODE_LOCK.json']),
                       'result': {**{k: x for k, x in summ.items()}, 'members_wire_differs': wire_bad, 'lock': lk,
                                  'draws_read': info.get('draws_read')},
                       'numerical_status': 'PASS' if not summ['mismatches'] else 'FAIL',
                       'procedural_status': 'PASS' if proc_ok else 'FAIL'})
    V.write_record(f'V4_GRID__{a.bank}__{a.tag}.json', record)
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        print(c['id'], c['numerical_status'], c['procedural_status'], json.dumps({k: c['result'][k] for k in c['result'] if k != 'mismatches'}))
        for m in c['result']['mismatches'][:10]:
            print('  ', json.dumps(m, default=V._jd)[:300])
    return 0


if __name__ == '__main__':
    sys.exit(main())
