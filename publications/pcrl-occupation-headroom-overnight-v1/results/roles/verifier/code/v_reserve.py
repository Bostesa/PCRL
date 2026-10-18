"""VER check 4: whole-release reserve (R1, 28 conditions, k = 2, H_LJ floor rescaled by sqrt(5306 / households)), the stage-1
witness labels and ranking, and the income-branch trigger, recomputed with the verifier's own code from the verifier's own
utility tables (records VER_UTILITY_REFS.json and VER_UTILITY_CONFIGS.json; aggregate numbers only, no data array), compared with
the campaign's stored per-seed reserves, WITNESS_SELECTION.json (when present) and RULES.json constants. Light arithmetic.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcommon as V  # noqa: E402

RULES = V.load_rules()


def tables():
    out = {}
    refs = json.loads((V.RECORDS / 'VER_UTILITY_REFS.json').read_text())
    for name, per in refs['tables'].items():
        out[('REF', name, 'REF')] = {int(s): t for s, t in per.items()}
    p = V.RECORDS / 'VER_UTILITY_CONFIGS.json'
    if p.is_file():
        db = json.loads(p.read_text())
        for key, e in db['entries'].items():
            fam, cfg, s, anchor = key.split('/')
            out.setdefault((fam, cfg, anchor), {})[int(s[1:])] = e['table']
    return out


def _sex_read_units():
    p = V.PRIV / 'access' / 'TRAINING_LABEL_READS.jsonl'
    out = set()
    if p.is_file():
        for line in p.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                if 'sex' in e.get('fields', []):
                    out.add(e.get('unit_id'))
    return out


def task_only_of(fam, cfg):
    """Task-only status from the release records of both seeds: kind task_model, or kind control whose record declares no SEX
    (roles_read says 'no SEX' or task_only true); in both cases the fit unit must have no logged SEX read."""
    vals = []
    sex_units = _sex_read_units()
    for s in (0, 1):
        for name in ('OCC.json', 'RELEASE.json'):
            p = V.RUNS / fam / cfg / f'S{s}' / name
            if p.is_file():
                r = json.loads(p.read_text())
                rr = str(r.get('roles_read', '')).upper()
                declared = bool(r.get('task_only')) or ('NO SEX' in rr) or ('SEX' not in rr)
                ok = (r.get('kind') == 'task_model' or (r.get('kind') == 'control' and declared)) and r.get('unit_id') not in sex_units
                vals.append(ok)
    return bool(vals) and all(vals), vals


def n_components(fam, cfg):
    n = 0
    for s in (0, 1):
        p = V.RUNS / fam / cfg / f'S{s}' / 'OCC.json'
        if p.is_file():
            r = json.loads(p.read_text())
            n += int(r.get('n_model_fits', 0) or 0) + int(r.get('n_calibrator_fits', 0) or 0)
    return n


def first_ledger_entry(fam, cfg):
    rows = list(csv.DictReader(open(V.RES / 'EXPERIMENT_LEDGER.csv')))
    ent = [int(r['entry']) for r in rows if r['family'] == fam and r['config_id'] == cfg and r['seed'] == '0' and r.get('kind') != 'reader_grid']
    return min(ent) if ent else None


def leg_b():
    """Published DEV_REPORT values of the frozen anchor (no new read): margin >= r2 x max(SE_anchor, SE_H_LJ)."""
    rows = list(csv.DictReader(open(V.SRC_RESULTS / 'report' / 'PRIMARY_ENDPOINTS.csv')))
    r2 = 2.0 * math.sqrt(3955 / 5306)
    res = {}
    for s in (0, 1):
        for ref in ('U0', 'Ucal'):
            k_s = f'UTILITY|S|income|{ref}|accuracy_loss|s{s}'
            k_h = f'UTILITY|H_LJ|income|{ref}|accuracy_loss|s{s}'
            S = [r for r in rows if r['scorecard'] == 'ENHANCED' and r['key'] == k_s][0]
            H = [r for r in rows if r['scorecard'] == 'ENHANCED' and r['key'] == k_h][0]
            point, se_s, se_h = float(S['point']), float(S['se']), float(H['se'])
            margin = 0.010 - point
            se_used = max(se_s, se_h)
            reg = RULES['income_branch']['anchor_dev_record']['clauses'][f's{s}|income|{ref}|accuracy_loss']
            res[f's{s}|{ref}'] = {'point': point, 'se_anchor': se_s, 'se_h_lj': se_h, 'ratio': margin / se_used, 'fails': margin < r2 * se_used,
                                  'registered_values_equal': (abs(reg['point'] - point) < 1e-15 and abs(reg['se_anchor'] - se_s) < 1e-15
                                                              and abs(reg['se_h_lj'] - se_h) < 1e-15)}
    return {'r2_factor_recomputed': r2, 'r2_factor_registered': RULES['final_stage']['r2_factor'], 'clauses': res,
            'leg_B': any(v['fails'] for v in res.values())}


def ledger_consistency(T):
    """EXPERIMENT_LEDGER.csv utility columns (rounded by the ledger) against the verifier's own per-seed tables and reserves."""
    rows = list(csv.DictReader(open(V.RES / 'EXPERIMENT_LEDGER.csv')))
    cols = {f'occ_{m}_{r}': f'occupation|{r}|{m}' for r in ('U0', 'Ucal') for m in ('accuracy_loss', 'nll_excess', 'brier_excess')}
    out = {'n_rows_checked': 0, 'mismatches': []}
    for r in rows:
        fam, cfg, anchor = r['family'], r['config_id'], r['anchor']
        if fam in ('GRID', 'SELECT') or r.get('kind') in ('reader_grid', 'decision'):
            continue
        key = ('REF', cfg, 'REF') if fam == 'REF' else (fam, cfg, anchor or 'H3')
        try:
            s = int(r['seed'])
        except ValueError:
            continue
        if key not in T or s not in T[key]:
            continue
        t = T[key][s]
        out['n_rows_checked'] += 1
        for col, cond in cols.items():
            if r.get(col) not in (None, ''):
                if abs(float(r[col]) - t[cond]['value']) > 1.5e-6:
                    out['mismatches'].append(f"entry {r['entry']} {fam}/{cfg}/S{s} {col}: ledger {r[col]} vs {t[cond]['value']:.8f}")
            if r.get(col + '_se') not in (None, ''):
                if abs(float(r[col + '_se']) - t[cond]['se']) > 1.5e-6:
                    out['mismatches'].append(f"entry {r['entry']} {fam}/{cfg}/S{s} {col}_se: ledger {r[col + '_se']} vs {t[cond]['se']:.8f}")
        for col, cond in (('guard_men', 'guard|sex0'), ('guard_women', 'guard|sex1')):
            if r.get(col) not in (None, '') and abs(float(r[col]) - t[cond]['value']) > 1.5e-6:
                out['mismatches'].append(f"entry {r['entry']} {fam}/{cfg}/S{s} {col}: ledger {r[col]} vs {t[cond]['value']:.8f}")
        res = V.reserve_seed(t, s, RULES, 5306)
        mr = min(v['ratio'] for v in res.values())
        if r.get('reserve_min_ratio') not in (None, '') and abs(float(r['reserve_min_ratio']) - mr) > 1.5e-3:
            out['mismatches'].append(f"entry {r['entry']} {fam}/{cfg}/S{s} reserve_min_ratio: ledger {r['reserve_min_ratio']} vs {mr:.4f}")
        pp = all(t[c]['value'] <= t[c]['cap'] for c in t if not c.startswith('guard'))
        gp = all(t[c]['value'] <= t[c]['cap'] for c in t if c.startswith('guard'))
        if r.get('point_pass_12') not in (None, '') and (r['point_pass_12'] == 'True') != pp:
            out['mismatches'].append(f"entry {r['entry']} point_pass_12 differs")
        if r.get('guard_pass') not in (None, '') and (r['guard_pass'] == 'True') != gp:
            out['mismatches'].append(f"entry {r['entry']} guard_pass differs")
    return out


def main():
    T = tables()
    rec = {'check': 'VER_RESERVE', 'utc': V.utc(), 'configs': {}}
    for (fam, cfg, anchor), per in sorted(T.items()):
        row = {'seeds': sorted(per)}
        res = {s: V.reserve_seed(per[s], s, RULES, 5306) for s in per}
        met = {s: all(c['met'] for c in r.values()) for s, r in res.items()}
        row['status'] = V.reserve_status(met)
        ratios = [c['ratio'] for r in res.values() for c in r.values()]
        row['min_ratio'] = min(ratios)
        row['binding'] = min(((s, k, c['ratio']) for s, r in res.items() for k, c in r.items()), key=lambda x: x[2])[:2]
        row['ratio_vector_28'] = sorted(ratios) if len(per) == 2 else None
        occ_r = [c['ratio'] for r in res.values() for k, c in r.items() if k in V.OCC_SIDE]
        row['occ_side_min_ratio'] = min(occ_r)
        row['occ_side_vector_16'] = sorted(occ_r) if len(per) == 2 else None
        row['point_pass_all'] = all(c['point_pass'] for r in res.values() for c in r.values())
        row['failed_conditions'] = [f's{s}|{k}' for s, r in res.items() for k, c in r.items() if not c['met']]
        row['bonferroni_unresolved'] = sum(1 for s in per for k, c in per[s].items()
                                           if not (c['value'] + RULES['nomination']['bonferroni_z_28'] * c['se'] <= c['cap']))
        if fam in ('A', 'B', 'C') and anchor == 'H3':          # stage-1 occupation families only
            to, vals = task_only_of(fam, cfg)
            row['task_only'] = to
            row['witness_label'] = V.witness_label(to, res)
            row['n_components'] = n_components(fam, cfg)
            row['first_ledger_entry'] = first_ledger_entry(fam, cfg)
        if fam == 'REF' and cfg in ('H3+T_TREE', 'H3+TASK_REFIT'):
            row['witness_label_as_if_candidate'] = V.witness_label(True, res)
        rec['configs'][f'{fam}/{cfg}/{anchor}'] = row
    # witness ranking
    prom = [(k, r) for k, r in rec['configs'].items() if r.get('witness_label') == 'PROMISING_WITNESS']
    # leximin: larger minimum first -> compare ascending vectors lexicographically, larger first
    import functools

    def cmp(a, b):
        va, vb = a[1]['occ_side_vector_16'], b[1]['occ_side_vector_16']
        for x, y in zip(va, vb):
            if abs(x - y) > 1e-9:
                return -1 if x > y else 1
        if a[1]['n_components'] != b[1]['n_components']:
            return -1 if a[1]['n_components'] < b[1]['n_components'] else 1
        return -1 if (a[1]['first_ledger_entry'] or 10 ** 9) < (b[1]['first_ledger_entry'] or 10 ** 9) else 1
    prom = sorted(prom, key=functools.cmp_to_key(cmp))
    rec['witness_ranking'] = [k for k, _ in prom]
    rec['selected_witnesses_by_my_rule'] = [k for k, _ in prom[:RULES['witness']['max_witnesses']]]
    lb = leg_b()
    rec['income_branch'] = {'leg_B': lb}
    if prom:
        top = prom[0][1]
        failing = top['failed_conditions']
        trig = {'income|U0|accuracy_loss', 'income|Ucal|accuracy_loss'}
        leg_a = top['status'] != 'RESERVE_MET' and bool(failing) and all(f.split('|', 1)[1] in trig for f in failing)
        rec['income_branch'].update({'promising_witness_exists': True, 'leg_A': leg_a, 'trigger': leg_a or lb['leg_B']})
    else:
        rec['income_branch'].update({'promising_witness_exists': False, 'leg_A': None, 'trigger': False})
    ws = V.RES / 'WITNESS_SELECTION.json'
    if ws.is_file():
        rec['witness_selection_file_sha256'] = V.file_sha256(ws)
        W = json.loads(ws.read_text())
        sel = W.get('selection', {})
        theirs = {k: (v.get('label'), v.get('selection')) for k, v in sel.get('records', {}).items()}
        mine = {}
        ws_fams = set(W.get('families') or ['A', 'B', 'C'])
        ws_cands = set(W.get('candidates') or [])
        for k, r in rec['configs'].items():
            fam, cfg, anchor = k.split('/')
            if fam in ws_fams and anchor == 'H3' and (not ws_cands or f'{fam}/{cfg}' in ws_cands):
                key = f'{fam}/{cfg}'
                lab = r.get('witness_label')
                s = ('SELECTED_WITNESS' if k in rec['selected_witnesses_by_my_rule'] else
                     'PROMISING_NOT_SELECTED' if lab == 'PROMISING_WITNESS' else 'NOT_A_WITNESS')
                mine[key] = (lab, s)
        cmpw = {'my_selected': [x.rsplit('/', 1)[0] for x in rec['selected_witnesses_by_my_rule']], 'their_selected': sel.get('selected'),
                'my_ranking': [x.rsplit('/', 1)[0] for x in rec['witness_ranking']], 'their_ranking': sel.get('ranking'),
                'label_mismatches': {k: {'mine': mine.get(k), 'theirs': theirs.get(k)} for k in sorted(set(mine) | set(theirs))
                                     if (mine.get(k) or (None,))[0] != (theirs.get(k) or (None,))[0]},
                'selection_mismatches': {k: {'mine': mine.get(k), 'theirs': theirs.get(k)} for k in sorted(set(mine) | set(theirs))
                                         if (mine.get(k) or (None, None))[1] != (theirs.get(k) or (None, None))[1]},
                'their_trigger': {kk: vv for kk, vv in W.get('income_branch_trigger', {}).items() if kk != 'leg_B_detail'},
                'my_trigger': {kk: vv for kk, vv in rec['income_branch'].items() if kk != 'leg_B'} | {'leg_B': rec['income_branch']['leg_B']['leg_B']}}
        cmpw['agree'] = (cmpw['my_selected'] == cmpw['their_selected'] and not cmpw['label_mismatches'] and not cmpw['selection_mismatches'])
        rec['witness_selection_compared'] = cmpw
    rec['ledger_consistency'] = ledger_consistency(T)
    V.write_record('VER_RESERVE', rec)
    brief = {k: (r['status'], round(r['min_ratio'], 3), r['binding'], r.get('witness_label'), round(r['occ_side_min_ratio'], 3))
             for k, r in rec['configs'].items()}
    print(json.dumps({'configs': brief, 'witness_ranking': rec['witness_ranking'], 'income_branch': rec['income_branch']}, indent=1, default=str))


if __name__ == '__main__':
    main()
