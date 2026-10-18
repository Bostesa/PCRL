"""Verifier replay of a bank evaluation (runs/PAIRS/BANK_EVAL__<bank>.json): whole-release utility on MODEL_SELECT (12 clauses and 2
guards per seed) of the exact arrays the readers read, and the frozen source whole-release reserve (RULES.md section 11: margin >=
2 x max(own SE, floor SE x sqrt(5306 / G)); minimum ratio margin / se_used over the 28 conditions; reserve-qualified iff >= 2 on both
seeds), recomputed with the verifier's own metrics. Floor SEs: the source campaign's registered OHN1 RULES.json (constants only).

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).
MODEL_SELECT labels via cts.common.load(operation='verify'). Nothing person-level is printed or written.

    python3 .../v_bank.py --bank R_FRONTIER --tag V1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from cts import common as C  # noqa: E402

CAMP = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1'
GRID = CAMP / 'runs' / 'GRID'
OHN_RULES = ROOT / 'results' / 'pcrl_occupation_headroom_overnight_v1' / 'roles' / 'statistical_reviewer' / 'RULES.json'
CAP = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def ms():
    return C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'verify',
                  'independent verifier: replay of whole-release utility and reserve of a reader bank on MODEL_SELECT')


def metrics(y, p):
    y = np.asarray(y, dtype=np.int64); p = np.asarray(p, dtype=np.float64)
    idx = np.arange(len(y)); oh = np.zeros_like(p); oh[idx, y] = 1.0
    return {'accuracy': (np.argmax(p, 1) == y).astype(np.float64), 'nll': -np.log(np.maximum(p[idx, y], V.NLL_FLOOR)),
            'brier': np.sum((p - oh) ** 2, 1)}


def utility(S, refs, inc, occ):
    out = {}
    rel = {'income': inc, 'occupation': occ}
    for t in ('income', 'occupation'):
        own = metrics(S[t], rel[t])
        for ref in ('U0', 'Ucal'):
            r = metrics(S[t], refs[ref][t])
            for m, d in (('accuracy_loss', r['accuracy'] - own['accuracy']), ('nll_excess', own['nll'] - r['nll']),
                         ('brier_excess', own['brier'] - r['brier'])):
                out[f'{t}|{ref}|{m}'] = {'value': float(d.mean()), 'se': V.clustered_se(d, S['unit']), 'cap': CAP[m]}
    u0 = metrics(S['occupation'], refs['U0']['occupation']); own = metrics(S['occupation'], occ)
    d = own['brier'] - u0['brier']
    for g in (0, 1):
        msk = S['sex'] == g
        out[f'guard|sex{g}'] = {'value': float(d[msk].mean()), 'se': V.subgroup_se(d, msk, S['unit']), 'cap': 0.005}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bank', required=True); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    from ohn import common as OC, devrun as OD                    # admitted, hash-checked reference and anchor outputs (label-free)
    be = json.loads((CAMP / 'runs' / 'PAIRS' / f'BANK_EVAL__{a.bank}.json').read_text())
    rules = json.loads(OHN_RULES.read_text())
    fl, G0 = rules['reserve']['floor']['se'], int(rules['reserve']['floor']['households'])
    S = ms()
    G = int(len(np.unique(S['unit'])))
    scale = math.sqrt(G0 / G)
    grids = {int(p.name[1:]): json.loads((p / 'GRID.json').read_text()) for p in sorted((GRID / a.bank).glob('S*'))}
    refs, incs = {}, {}
    for s in grids:
        bank, ids = OC.frozen_bank(s, roles=('MODEL_SELECT',))
        if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
            raise AssertionError('row order')
        refs[s] = {r: {t: np.asarray(bank[r]['MODEL_SELECT'][t], dtype=np.float64) for t in ('income', 'occupation')} for r in ('U0', 'Ucal')}
        incs[s] = np.asarray(OD.anchor_income('H3', s, 'MODEL_SELECT'), dtype=np.float64)
    worst_v, worst_se, mism, rows = 0.0, 0.0, [], {}
    for name, rec in be['members'].items():
        per, ratios = {}, {}
        for s, g in grids.items():
            mem = g['members'].get(name)
            if not mem or 'path' not in (mem.get('occupation') or {}) or str(s) not in rec.get('per_seed', {}):
                continue
            src = mem['occupation']
            if V.file_sha(ROOT / src['path']) != src['sha256']:
                mism.append(f'{name} S{s}: occupation file hash'); continue
            occ = npz(ROOT / src['path'])[src['key']]
            isrc = mem.get('income') or {}
            inc = incs[s] if isrc.get('frozen_anchor') == 'H3' else npz(ROOT / isrc['path'])[isrc['key']]
            u = utility(S, refs[s], inc, occ)
            r = rec['per_seed'][str(s)]
            for c, x in u.items():
                rc = r['clauses'].get(c) if not c.startswith('guard') else r['guards'].get(c.split('|')[1])
                worst_v = max(worst_v, abs(x['value'] - rc['value'])); worst_se = max(worst_se, abs(x['se'] - rc['se']))
                se_used = max(x['se'], float(fl[str(s)][c]) * scale)
                ratios[f's{s}|{c}'] = (x['cap'] - x['value']) / se_used
            per[s] = True
        if not ratios:
            continue
        mn = min(ratios, key=ratios.get)
        row = {'min_ratio_mine': ratios[mn], 'binding_mine': mn, 'min_ratio_recorded': rec.get('min_ratio'),
               'binding_recorded': rec.get('binding_condition'), 'seeds': sorted(per)}
        if rec.get('min_ratio') is not None and (abs(ratios[mn] - rec['min_ratio']) > 1e-9 or mn != rec.get('binding_condition')):
            mism.append(f'{name}: min ratio / binding')
        row['reserve_qualified_mine'] = len(per) == 2 and all(min(v for k, v in ratios.items() if k.startswith(f's{s}|')) >= 2
                                                              for s in per)
        rows[name] = row
    chk = {'id': f'C6.4.BANK_EVAL.{a.bank}__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': f'{a.bank}: whole-release utility (12 clauses + 2 guards per seed: values and household-clustered SEs) and the frozen '
                   'source whole-release reserve (minimum ratio, binding condition) of every member, for the exact arrays the readers read',
           'method': 'own metrics and SEs against the frozen-bank U0 / Ucal outputs; own reserve: ratio = (cap - value) / max(own SE, '
                     'H_LJ floor SE x sqrt(5306 / G)) with the source OHN1 RULES.json floors; tolerances 1e-10 (values, SEs), 1e-9 (ratios)',
           'status': 'PASS' if (not mism and worst_v <= 1e-10 and worst_se <= 1e-10) else 'FAIL',
           'numbers': {'members': len(rows), 'max_abs_value_diff': worst_v, 'max_abs_se_diff': worst_se, 'mismatches': mism[:20],
                       'reserve_qualified_members_mine': sorted(k for k, v in rows.items() if v['reserve_qualified_mine']),
                       'per_member': rows}}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__)
    V.write_record(f'V_BANK__{a.bank}__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], 'n': len(rows), 'dv': worst_v, 'dse': worst_se, 'mism': mism[:5],
                      'qualified': chk['numbers']['reserve_qualified_members_mine']}))


if __name__ == '__main__':
    main()
