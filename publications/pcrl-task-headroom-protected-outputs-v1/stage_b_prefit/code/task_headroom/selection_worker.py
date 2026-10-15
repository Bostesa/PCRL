"""Apply the SCIENCE_LOCK nomination rules to both seeds' FIT2 tables (no ATT/INNER access)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

PROTECTED = ('J_TREE', 'L_TREE', 'L_STRONG', 'S12_TREE', 'S21_TREE')   # also the tie-break family order
STRENGTHS = ('1.0', '3.0', '10.0')
B_ROSTER = ('RAW_PROJ', 'N_EQ', 'T_TREE', 'H_JOINT_CODE', 'H_SEQ21_CODE', 'H_DPC_JOINT', 'FARE')
C_ROSTER = ('L_TREE', 'L_STRONG', 'S12_TREE', 'S21_TREE')


def labels(family):
    return [f'P2E{e}' for e in (2, 5, 10, 20)] if family.startswith('S') else [f'E{e}' for e in (2, 5, 10, 20)]


def epoch(label):
    return int(label.split('E')[-1])


def nominate(tabs, family):
    rows = []
    for rho in STRENGTHS:
        for lab in labels(family):
            key = f'{family}|{rho}|{lab}'
            r = [tabs[s][key] for s in (0, 1)]
            rows.append({'key': key, 'rho': float(rho), 'label': lab, 'feasible': all(x['internal_point_feasible'] for x in r),
                         'worst_pair_proxy': max(x['proxy']['pair']['value'] for x in r),
                         'worst_violation': max(x['max_normalized_violation'] for x in r),
                         'summed_nll': sum(x['summed_nll'] for x in r)})
    tie = lambda c: (c['summed_nll'], c['rho'], epoch(c['label']))
    feas = [c for c in rows if c['feasible']]
    if feas:
        chosen = min(feas, key=lambda c: (c['worst_pair_proxy'],) + tie(c)); status = 'POINT_FEASIBLE_NOMINEE'
    else:
        chosen = min(rows, key=lambda c: (c['worst_violation'],) + tie(c)); status = 'NO_FEASIBLE_FALLBACK_DIAGNOSTIC'
    return {'status': status, 'nominee': chosen, 'candidates': rows}


def run(spec_path):
    from .evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'verification')
    tabs = {s: json.loads(verify(cfg['fit2'][str(s)]['path'], cfg['fit2'][str(s)]['sha256']).read_text())['table'] for s in (0, 1)}
    fam = {f: nominate(tabs, f) for f in PROTECTED}
    jl = fam['J_TREE']['nominee']['label']
    t_refit = f'T_REFIT|none|{jl}'
    feasible = [f for f in PROTECTED if fam[f]['status'] == 'POINT_FEASIBLE_NOMINEE']
    order = {f: i for i, f in enumerate(PROTECTED)}
    primary = min(feasible, key=lambda f: (fam[f]['nominee']['worst_pair_proxy'], fam[f]['nominee']['summed_nll'], order[f])) if feasible else None
    elig = {}
    for c in B_ROSTER:
        point = all(tabs[s][c]['internal_point_feasible'] for s in (0, 1))
        elig[c] = {'internal_point_feasible': point, 'required': bool(c == 'RAW_PROJ' or point),
                   'failed_clauses': [x for s in (0, 1) for x in tabs[s][c]['internal'] if not x['point_pass']]}
    required = [c for c in B_ROSTER if elig[c]['required']]
    K = 24 + 3 * len(required) + 12 + (24 if primary != 'J_TREE' else 0)
    roster = {'T_TREE': 'T_TREE', 'T_REFIT': t_refit, **{f: fam[f]['nominee']['key'] for f in PROTECTED}, 'DIAG_PRIV': 'J_TREE|10.0|E20',
              'N_EQ': 'N_EQ', 'RAW_PROJ': 'RAW_PROJ', 'H_JOINT_CODE': 'H_JOINT_CODE', 'H_SEQ21_CODE': 'H_SEQ21_CODE',
              'H_DPC_JOINT': 'H_DPC_JOINT', 'FARE': 'FARE', 'FARE_Z1': 'FARE_Z1'}
    result = {'schema': 'THP1-selection-v1', 'queue_unit_id': active['unit_id'], 'families': fam, 'T_REFIT': t_refit,
              'primary_family': primary, 'primary_key': fam[primary]['nominee']['key'] if primary else None,
              'label_hint': None if primary else 'NO_POINT_FEASIBLE_PROTECTED_NOMINEE',
              'B_eligibility': elig, 'B_required': required, 'C_roster': list(C_ROSTER), 'K_primary_family': K,
              'attack_roster': roster, 'ATT_opened': False, 'INNER_opened': False,
              'fit2_proxy_by_service': {n: {str(s): tabs[s][k]['proxy'] for s in (0, 1)} for n, k in roster.items()}}
    with Path(cfg['output']).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
