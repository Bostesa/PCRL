"""Ordinary compression control at matched GUARD cost (PCRL correction-transfer campaign).

    python3 -m cts.compression fit --partition P --direction ID --rule RULE

A label-free K-means code of the witness's cross-fitted FIT output vectors (the source COMP recipe, ohn.protect.kmeans_code /
code_release): released vector = nearest centroid of the deployed witness vector; decision = argmax. K is the smallest value of the
source ladder whose GUARD utilization (max over the five statistics of incremental cost / the statistic's reserve slack) is at most
the GUARD utilization of the candidate (direction ID at the strength RULE selected) - the source's matched-utilization rule moved
from out-of-fold FIT costs to GUARD. If even the largest K exceeds the target it is flagged. CHECK / MODEL_SELECT outcomes are
computed later with the candidate's (cts.select realized; repair panels only after the committed REPAIR_FREEZE).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

from . import common as C
from . import rules as RL
from . import trackf as TF


def utilization_of(stats, sslack):
    u = 0.0
    for k in RL.STATS:
        b = sslack[k]
        u = max(u, stats[k] / b if b > 0 else (np.inf if stats[k] > b else 0.0))
    return float(u)


def fit(P, did, rule):
    """rule: a selection key 'RULE|full' (matched to the strength that rule selected for did) or 's<x>' (a fixed ladder strength,
    for the matched frontier)."""
    from ohn import protect as PR
    t0 = time.process_time()
    name = f'COMP__{did}__{rule.replace("|", "_")}'
    d = TF.pdir(P) / 'COMP' / name
    if (d / 'COMP.json').exists():
        raise PermissionError(f'{name} exists (append-only)')
    srec = json.loads((TF.pdir(P) / 'SELECT' / f'{did}.json').read_text())
    if rule.startswith('s'):
        s_sel = float(rule[1:])
    else:
        if '|' in rule and rule.split('|')[1] != 'full':
            raise PermissionError('compression twins are matched to the full deployment')
        s_sel = float(srec['selected'][rule if '|' in rule else f'{rule}|full'])
    if s_sel <= 0:
        raise PermissionError(f'{rule} selected identity for {did}: no compression twin to match')
    lad = json.loads((TF.pdir(P) / 'LADDER' / f'{did}.json').read_text())
    sslack = RL.stat_slack(srec['baseline'])
    wrec, W, O = TF.load_witness(P)
    drec, T, D = TF.load_direction(P, did)
    G = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'select', f'compression control {P}: GUARD costs of the K ladder matched '
               f'to {did} at the {rule} strength', partition=P, prole='GUARD')
    rows = G['rows']
    pw = W['dep__MODEL_FIT'][rows]
    cand = TF.cost_summary(TF.release_at(pw, D['full__MODEL_FIT'][rows], s_sel), pw, G['occupation'], G['sex'], G['unit'])
    target = utilization_of(cand['stats'], sslack)
    if not np.isfinite(target):
        raise ValueError(f'{name}: the candidate has a positive GUARD cost on a statistic whose reserve slack is nonpositive '
                         f'(utilization = inf; stat slack {sslack}); no code can be matched to it - the bisection would return the '
                         'smallest K labelled MATCHED')
    seen = {}

    def ev(i):
        if i not in seen:
            cents, info = PR.kmeans_code(O['oof'], PR.COMP_LADDER[i], f'{P}|{did}|{rule}')
            cs = TF.cost_summary(PR.code_release(pw, cents), pw, G['occupation'], G['sex'], G['unit'])
            seen[i] = {'K': PR.COMP_LADDER[i], 'cost': cs, 'utilization': utilization_of(cs['stats'], sslack), 'info': info, 'cents': cents}
        return seen[i]
    lo, hi = 0, len(PR.COMP_LADDER) - 1
    if ev(hi)['utilization'] > target:
        best, flag = hi, 'TARGET_NOT_REACHED_LARGEST_K'
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if ev(mid)['utilization'] <= target:
                hi = mid
            else:
                lo = mid + 1
        best, flag = lo, 'MATCHED'
    b = seen[best]
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    rel = {r: PR.code_release(W[f'dep__{r}'], b['cents']) for r in ('MODEL_FIT', 'MODEL_SELECT')}
    sha = TF._save_npz(d / 'CODE.npz', centroids=b['cents'], **{f'rel__{r}': v for r, v in rel.items()})
    rec = {'schema': 'CTS1-comp-v1', 'partition': P, 'name': name, 'matched_to': {'direction': did, 'rule': rule, 'strength': s_sel,
           'guard_utilization': target, 'guard_cost': cand['stats']}, 'K': b['K'], 'flag': flag, 'guard_cost': b['cost'],
           'guard_utilization': b['utilization'], 'ladder_path': [{'K': v['K'], 'utilization': v['utilization']} for _, v in sorted(seen.items())],
           'kmeans_info': b['info'], 'npz_sha256': sha, 'n_kmeans_fits': len(seen),
           'search': 'bisection assuming GUARD utilization falls with K (as ohn.protect.comp_select); unevaluated K are not checked',
           'operation': 'label-free K-means code of the cross-fitted FIT witness vectors; nearest centroid of the deployed witness vector',
           'cpu_s': time.process_time() - t0, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json(d / 'COMP.json', rec)
    C.unit_stats(n_calibrator_fits=len(seen))
    print(json.dumps({'partition': P, 'name': name, 'K': b['K'], 'flag': flag, 'target': round(target, 3), 'achieved': round(b['utilization'], 3)}))
    return rec


def realized(P):
    """Realized CHECK and MODEL_SELECT costs of every saved compression code of P (repair panels only after the REPAIR_FREEZE)."""
    t0 = time.process_time()
    kind = C.PARTITIONS[P]['kind']
    if kind == 'repair' and not C.repair_frozen():
        raise PermissionError('repair CHECK stays closed until the committed REPAIR_FREEZE')
    out = TF.pdir(P) / 'COMP' / 'COMP_REALIZED.json'
    if out.exists():
        raise PermissionError('COMP_REALIZED.json exists (append-only)')
    wrec, W, O = TF.load_witness(P)
    L = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'check_diag' if kind == 'diagnosis' else 'check_repair',
               f'realized CHECK costs of the compression codes of {P}', partition=P, prole='CHECK')
    S = C.load('MODEL_SELECT', ('features', 'occupation', 'sex'), 'benchmark', f'realized MODEL_SELECT costs of the compression codes of {P}')
    res = {}
    for d in sorted((TF.pdir(P) / 'COMP').iterdir()):
        if not (d / 'COMP.json').is_file():
            continue
        rec = json.loads((d / 'COMP.json').read_text())
        if C.sha(d / 'CODE.npz') != rec['npz_sha256']:
            raise PermissionError('compression record differs')
        with np.load(d / 'CODE.npz') as z:
            rf, rm = z['rel__MODEL_FIT'][L['rows']], z['rel__MODEL_SELECT']
        res[d.name] = {'K': rec['K'], 'matched_to': rec['matched_to'],
                       'CHECK': TF.cost_summary(rf, W['dep__MODEL_FIT'][L['rows']], L['occupation'], L['sex'], L['unit']),
                       'MS': TF.cost_summary(rm, W['dep__MODEL_SELECT'], S['occupation'], S['sex'], S['unit'])}
    C.write_json(out, {'schema': 'CTS1-comp-realized-v1', 'partition': P, 'codes': res, 'cpu_s': time.process_time() - t0,
                       'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)})
    print(json.dumps({'partition': P, 'codes': len(res)}))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    x = sub.add_parser('fit'); x.add_argument('--partition', required=True); x.add_argument('--direction', required=True)
    x.add_argument('--rule', required=True)
    y = sub.add_parser('realized'); y.add_argument('--partition', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'fit':
        fit(a.partition, a.direction, a.rule)
    else:
        realized(a.partition)
    return 0


if __name__ == '__main__':
    sys.exit(main())
