"""Original-benchmark evaluation on MODEL_SELECT for the PCRL correction-transfer campaign (operation 'benchmark', logged).

Unchanged definitions: the 12 point clauses per source seed (income and occupation x U0 and Ucal x accuracy loss <= 0.010, NLL
excess <= 0.010 nats with the inherited 1e-12 probability floor, summed Brier excess <= 0.005; household-clustered paired SEs) and
the men's and women's occupation summed-Brier excess vs U0 (<= 0.005), via ohn.evaluate.complete_eval / ihs.evaluate.utility; the
development reserve via the frozen source rule ohn.rules.reserve (margin >= 2 x max(own SE, floor SE)). A D or R "pair" is the
seed-0-associated partition (D0 / R0) with the seed-0 references plus the seed-1-associated one (D1 / R1) with the seed-1
references: 24 utility clauses + 4 guards. Repetitions are never pooled into extra seeds.

    python3 -m cts.evaluate witness --partition P          witness W_P with H3 income: EVAL_MS.json + baseline slack (logged
                                                           design input of the strength rules)
    python3 -m cts.evaluate release --partition P --name N --occ PATH.npz --key KEY
    python3 -m cts.evaluate pair --stage D|R --name N      combine the two seeds' evaluations (24 + 4) with the reserve
"""
from __future__ import annotations
import argparse
import json
import sys
import time

import numpy as np

from . import common as C

EVAL_DIR = 'EVAL'


def _ms(purpose):
    S = C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'benchmark', purpose)
    return S


def _refs(seed, S):
    from ohn import common as OC
    bank, ids = OC.frozen_bank(seed, roles=('MODEL_SELECT',))
    if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
        raise AssertionError('frozen bank MODEL_SELECT row order differs')
    refs = {r: {t: bank[r]['MODEL_SELECT'][t] for t in C.TASKS} for r in ('U0', 'Ucal')}
    others = {'T_TREE': bank['T_TREE']['MODEL_SELECT']['occupation'], 'TASK_REFIT': bank['TASK_REFIT']['MODEL_SELECT']['occupation'],
              'JOINT': bank['H_LJ']['MODEL_SELECT']['occupation']}
    return refs, others


def h3_income(seed, role='MODEL_SELECT'):
    from ohn import devrun as OD
    return OD.anchor_income('H3', seed, role)


def eval_release(seed, occ_ms, inc_ms, purpose, S=None):
    from ohn import evaluate as OE
    S = S if S is not None else _ms(purpose)
    refs, others = _refs(seed, S)
    return OE.complete_eval(seed, inc_ms, occ_ms, S, refs, others)


def slack_of(ev, seed):
    """Witness baseline slack per occupation-side condition (frozen source reserve rule, one seed): reserve slack =
    margin - 2 x se_used (se_used = max(own SE, floor SE)), point margin = cap - value."""
    from ohn import protect as PR
    slack, margin, res = PR.witness_slack(ev, seed)
    conds = res['per_seed'][seed]['conditions']
    return {'reserve_slack': slack, 'point_margin': margin,
            'se_used': {n: conds[n]['se_used'] for n in slack}, 'own_se': {n: conds[n]['own_se'] for n in slack},
            'witness_reserve_status_one_seed': res['per_seed'][seed]['status'], 'min_occupation_side_ratio': res['min_occupation_side_ratio']}


def witness(P):
    from . import trackf as TF
    t0 = time.process_time()
    s = TF.seed_of(P)
    out = TF.pdir(P) / EVAL_DIR / 'WITNESS_EVAL_MS.json'
    if out.exists():
        raise PermissionError('witness evaluation exists (append-only)')
    wrec, W, O = TF.load_witness(P)
    ev = eval_release(s, W['dep__MODEL_SELECT'], h3_income(s), f'benchmark: Track F {P} witness with H3 income on MODEL_SELECT '
                      '(24-clause / guard utility and baseline slack; a logged design input of the strength rules)')
    sl = slack_of(ev, s)
    rec = {'schema': 'CTS1-eval-v1', 'partition': P, 'seed': s, 'name': 'WITNESS', 'income': f'H3_STRONG_LJDIR_PB S{s}', 'eval': ev,
           'slack': sl, 'cpu_s': time.process_time() - t0, 'utc': C.now()}
    C.write_json(out, rec)
    print(json.dumps({'partition': P, 'point_pass_12': ev['utility']['point_pass_12'], 'guard_pass': ev['utility']['guard_pass'],
                      'reserve_slack': {k: round(v, 5) for k, v in sl['reserve_slack'].items()}}))
    return rec


def release(P, name, occ_ms, note=''):
    from . import trackf as TF
    s = TF.seed_of(P)
    out = TF.pdir(P) / EVAL_DIR / f'{name}_EVAL_MS.json'
    if out.exists():
        raise PermissionError(f'{out.name} exists (append-only)')
    ev = eval_release(s, occ_ms, h3_income(s), f'benchmark: Track F {P} {name} with H3 income on MODEL_SELECT {note}'.strip())
    rec = {'schema': 'CTS1-eval-v1', 'partition': P, 'seed': s, 'name': name, 'income': f'H3_STRONG_LJDIR_PB S{s}', 'eval': ev,
           'occ_sha256': C.arr_sha(occ_ms), 'utc': C.now()}
    C.write_json(out, rec)
    return rec


def pair(stage, name):
    from ohn import rules as OR
    from . import trackf as TF
    parts = {'D': ('D0', 'D1'), 'R': ('R0', 'R1')}[stage]
    evs = {}
    for P in parts:
        f = TF.pdir(P) / EVAL_DIR / (('WITNESS' if name == 'WITNESS' else name) + '_EVAL_MS.json')
        evs[TF.seed_of(P)] = json.loads(f.read_text())['eval']
    res = OR.reserve(evs)
    u = {s: e['utility'] for s, e in evs.items()}
    rec = {'schema': 'CTS1-pair-v1', 'stage': stage, 'name': name, 'partitions': parts,
           'point_pass_24': all(u[s]['point_pass_12'] for s in u), 'guards_pass_4': all(u[s]['guard_pass'] for s in u),
           'reserve_status': res['status'], 'min_ratio': res['min_ratio'], 'binding_condition': res['binding_condition'],
           'min_occupation_side_ratio': res['min_occupation_side_ratio'], 'min_income_ratio': res['min_income_ratio'],
           'failed': res['failed'],
           'clauses': {f's{s}|{c["task"]}|{c["reference"]}|{c["metric"]}': {k: c[k] for k in ('value', 'se', 'margin', 'point_pass')}
                       for s in u for c in u[s]['clauses']},
           'guards': {f's{s}|{g["group"]}': {k: g[k] for k in ('value', 'se', 'margin', 'point_pass')} for s in u for g in u[s]['guards']},
           'note': 'D and R pairs are reported separately; each pair is 24 utility clauses + 4 guards; no pooling of repetitions',
           'utc': C.now()}
    C.write_json(C.RUNS / 'PAIRS' / f'{stage}__{name}.json', rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    x = sub.add_parser('witness'); x.add_argument('--partition', required=True)
    x = sub.add_parser('release'); x.add_argument('--partition', required=True); x.add_argument('--name', required=True)
    x.add_argument('--occ', required=True); x.add_argument('--key', default='MODEL_SELECT')
    x = sub.add_parser('pair'); x.add_argument('--stage', required=True, choices=('D', 'R')); x.add_argument('--name', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'witness':
        witness(a.partition)
    elif a.cmd == 'release':
        with np.load(a.occ) as z:
            occ = z[a.key].copy()
        release(a.partition, a.name, occ)
    else:
        print(json.dumps(pair(a.stage, a.name), indent=1)[:4000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
