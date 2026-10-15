"""Apply the COMPLETION_LOCK nomination rules to both seeds' FIT2 tables (no ATT/INNER access).

Neural RAW-J: identical to the tree-family rule -- among (rho, checkpoint) in {1,3,10} x {E2,E5,E10,E20}
point-feasible on all 12 internal clauses vs N_EQ in both seeds, the lowest worst-seed FIT2 coalition proxy;
ties: summed FIT2 NLL, lower rho, earlier checkpoint. If none is feasible: the minimum worst-seed normalized
clause violation (descriptive fallback). T_NEURAL is matched to the nominee's checkpoint.
FARE: among F1, F2, F3 (same rule, ties: summed NLL then setting order); fallback likewise; matched
zero-fairness control Z<id> of the chosen setting.
A new control is a REQUIRED comparator for L_TREE under the original criterion iff its nominee is
internally point-feasible in both seeds. Consistency check: T_REFIT E20 and L_TREE E20 FIT2 values are
recomputed and compared with the original FIT2 tables.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def nominate(tabs, keys, order_key):
    rows = []
    for key in keys:
        r = [tabs[s][key] for s in (0, 1)]
        rows.append({'key': key, 'feasible': all(x['internal_point_feasible'] for x in r),
                     'worst_pair_proxy': max(x['proxy']['pair']['value'] for x in r),
                     'worst_violation': max(x['max_normalized_violation'] for x in r),
                     'summed_nll': sum(x['summed_nll'] for x in r), 'order': order_key(key)})
    tie = lambda c: (c['summed_nll'],) + c['order']
    feas = [c for c in rows if c['feasible']]
    if feas:
        return {'status': 'POINT_FEASIBLE_NOMINEE', 'nominee': min(feas, key=lambda c: (c['worst_pair_proxy'],) + tie(c)), 'candidates': rows}
    return {'status': 'NO_FEASIBLE_FALLBACK_DIAGNOSTIC', 'nominee': min(rows, key=lambda c: (c['worst_violation'],) + tie(c)), 'candidates': rows}


def run(spec_path):
    from .evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'verification')
    tabs = {s: json.loads(verify(cfg['fit2'][str(s)]['path'], cfg['fit2'][str(s)]['sha256']).read_text())['table'] for s in (0, 1)}
    old = {s: json.loads(verify(cfg['old_fit2'][str(s)]['path'], cfg['old_fit2'][str(s)]['sha256']).read_text())['table'] for s in (0, 1)}
    rj_keys = [f'RAW_J|{rho}|E{e}' for rho in ('1.0', '3.0', '10.0') for e in (2, 5, 10, 20)]
    rj = nominate(tabs, rj_keys, lambda k: (float(k.split('|')[1]), int(k.split('|')[2][1:])))
    ep = rj['nominee']['key'].split('|')[2]
    fare = nominate(tabs, ['FARE|F1', 'FARE|F2', 'FARE|F3'], lambda k: (int(k[-1]),))
    fid = fare['nominee']['key'][-1]
    consistency = {}
    for key in ('T_REFIT|none|E20', 'L_TREE|1.0|E20'):
        for s in (0, 1):
            a, b = tabs[s][key], old[s][key]
            consistency[f'{key}|s{s}'] = {'clauses_equal': all(abs(x['value'] - y['value']) <= 1e-12 for x, y in zip(a['internal'], b['internal'])),
                                         'proxy_equal': abs(a['proxy']['pair']['value'] - b['proxy']['pair']['value']) <= 1e-12}
    req = {}
    for name, n in (('RAW_J_NEQ', rj), ('FARE_TC', fare)):
        req[name] = {'internal_point_feasible': n['status'] == 'POINT_FEASIBLE_NOMINEE', 'required_comparator_for_L_TREE': n['status'] == 'POINT_FEASIBLE_NOMINEE'}
    roster = {'T_REFIT_E20': 'T_REFIT|none|E20', 'RAW_J_NEQ': rj['nominee']['key'], 'T_NEURAL': f'T_NEURAL|none|{ep}',
              'FARE_TC': fare['nominee']['key'], 'FARE_TC_Z': f'FARE|Z{fid}'}
    result = {'schema': 'TC1-completion-selection-v1', 'queue_unit_id': active['unit_id'], 'RAW_J': rj, 'FARE': fare,
              'required_status': req, 'attack_roster': roster, 'consistency_with_original_fit2': consistency,
              'common_checkpoint_descriptive_FIT2_only': {k: {str(s): {'feasible': old[s][k]['internal_point_feasible'], 'pair_proxy': old[s][k]['proxy']['pair']['value']} for s in (0, 1)}
                                                          for k in ('J_TREE|1.0|E20', 'L_TREE|1.0|E20', 'J_TREE|1.0|E5', 'L_TREE|1.0|E5')},
              'fit2_proxy_by_attacked_service': {n: {str(s): tabs[s][k]['proxy'] for s in (0, 1)} for n, k in roster.items()},
              'ATT_opened': False, 'INNER_opened': False}
    with Path(cfg['output']).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
