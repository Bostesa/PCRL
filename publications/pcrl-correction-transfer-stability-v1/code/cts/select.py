"""Strength selection by the frozen rules (OLD-B80, OLD-B50, SIMPLE, UNCERTAINTY) and their realized outcomes
(PCRL correction-transfer campaign; rules: roles/statistical_reviewer/RULES.md, implementation cts.rules).

    python3 -m cts.select run --partition P --direction ID          selections from FIT (OLD) and GUARD (SIMPLE, UNCERTAINTY) with the
                                                                    witness's MODEL_SELECT baseline slack, for both deployments (full:
                                                                    GUARD costs of the deployed release; ens: GUARD costs of the
                                                                    deployed ensemble); for D partitions then the realized CHECK and
                                                                    MODEL_SELECT outcomes
    python3 -m cts.select check --partition R0|R1 --direction ID    after the committed REPAIR_FREEZE: realized repair CHECK and
                                                                    MODEL_SELECT outcomes of the frozen selections

A rule never reads CHECK or MODEL_SELECT outcomes of the correction: the selections are computed and saved first (their own file),
then the realized costs at the selected strengths (identity = zero cost: a valid outcome that earns no protection claim).
OLD selects one continuous scale from out-of-fold FIT costs; it is applied to either deployment.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

from . import common as C
from . import rules as RL
from . import trackf as TF

RULES = ('OLD-B80', 'OLD-B50', 'SIMPLE', 'UNCERTAINTY')
DEPLOY = {'full': 'GUARD', 'ens': 'GUARD_ENS'}


def _ladder_maps(est):
    return {float(s): v['stats'] for s, v in est.items()}, {float(s): v['se'] for s, v in est.items()}


def baseline(P):
    from ohn import rules as OR
    ev = json.loads((TF.pdir(P) / 'EVAL' / 'WITNESS_EVAL_MS.json').read_text())
    s = ev['seed']
    conds = OR.conditions(ev['eval'], s)
    G = int(ev['eval']['utility']['households'])
    return RL.witness_slacks(conds, s, G), ev


def _ms_counts():
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'benchmark', 'SEX-group sizes of MODEL_SELECT for the predictive SE of the '
               'UNCERTAINTY rule (counts only)')
    return {'men': int((S['sex'] == 0).sum()), 'women': int((S['sex'] == 1).sum())}


def selections(P, did):
    t0 = time.process_time()
    out = TF.pdir(P) / 'SELECT' / f'{did}.json'
    if out.exists():
        raise PermissionError(f'{out.name} exists (append-only)')
    lad = json.loads((TF.pdir(P) / 'LADDER' / f'{did}.json').read_text())
    base, ev = baseline(P)
    wrec, W, O = TF.load_witness(P)
    drec, T, D = TF.load_direction(P, did)
    F = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'fit', f'strength rules {P} {did}: OLD projection on out-of-fold FIT '
               'costs (the source rule)', partition=P, prole='FIT')
    z0 = TF.logp(O['oof'])
    sel, selected = {}, {}
    for lvl in ('B80', 'B50'):
        r = RL.old_rule(z0, T['delta_oof'], F['occupation'], F['sex'], base, level=lvl)
        sel[f'OLD-{lvl}'] = {k: v for k, v in r.items() if k != 'ladder_oof'}
        for dep in DEPLOY:
            selected[f'OLD-{lvl}|{dep}'] = float(r['selected'])
    cnt = _ms_counts()
    ms_n = {'all': cnt['men'] + cnt['women'], 'men': cnt['men'], 'women': cnt['women']}
    for dep, est in DEPLOY.items():
        gc, gs = _ladder_maps(lad['estimators'][est])
        g = lad['estimators'][est]['0.0']
        n_guard = {'all': g['n'], 'men': g['n_men'], 'women': g['n_women']}
        sel[f'SIMPLE|{dep}'] = RL.simple_rule(gc, base)
        sel[f'UNCERTAINTY|{dep}'] = RL.uncertainty_rule(gc, gs, base, n_guard, ms_n)
        for rule in ('SIMPLE', 'UNCERTAINTY'):
            selected[f'{rule}|{dep}'] = float(sel[f'{rule}|{dep}']['selected'])
    rec = {'schema': 'CTS1-select-v2', 'partition': P, 'direction': did, 'selections': sel, 'selected': selected,
           'baseline': base, 'n_ms': ms_n, 'rules_sha256': C.sha(RL.RULES_PATH),
           'note': 'keys RULE|DEPLOYMENT; full = deployed full-FIT correction on the deployed witness; ens = mean of the 5 fold pipelines',
           'cpu_s': time.process_time() - t0, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json(out, rec)
    print(json.dumps({'partition': P, 'direction': did, 'selected': selected}))
    return rec


def realized(P, did):
    """Realized outcomes on CHECK and MODEL_SELECT of the saved selections (after selection; never feeds back)."""
    t0 = time.process_time()
    kind = C.PARTITIONS[P]['kind']
    out = TF.pdir(P) / 'SELECT' / f'{did}__REALIZED.json'
    if out.exists():
        raise PermissionError(f'{out.name} exists (append-only)')
    if kind == 'repair' and not C.repair_frozen():
        raise PermissionError('repair CHECK stays closed until the committed REPAIR_FREEZE')
    srec = json.loads((TF.pdir(P) / 'SELECT' / f'{did}.json').read_text())
    base = srec['baseline']
    wrec, W, O = TF.load_witness(P)
    drec, T, D = TF.load_direction(P, did)
    op = 'check_diag' if kind == 'diagnosis' else 'check_repair'
    L = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), op, f'realized CHECK outcomes of the frozen selections {P} {did}',
               partition=P, prole='CHECK')
    S = C.load('MODEL_SELECT', ('features', 'occupation', 'sex'), 'benchmark',
               f'realized MODEL_SELECT outcomes of the frozen selections {P} {did}')
    rows = L['rows']
    sets = {'CHECK': (W['dep__MODEL_FIT'][rows], D['full__MODEL_FIT'][rows], W['fold__MODEL_FIT'][:, rows], D['folds__MODEL_FIT'][:, rows], L),
            'MS': (W['dep__MODEL_SELECT'], D['full__MODEL_SELECT'], W['fold__MODEL_SELECT'], D['folds__MODEL_SELECT'], S)}
    res = {}
    for name, (pw, dl, pf, df, X) in sets.items():
        res[name] = {}
        e0 = TF.ensemble_at(pf, df, 0.0)
        for key, s in srec['selected'].items():
            dep = key.split('|')[1]
            if dep == 'full':
                cs = TF.cost_summary(TF.release_at(pw, dl, s), pw, X['occupation'], X['sex'], X['unit'])
            else:
                cs = TF.cost_summary(TF.ensemble_at(pf, df, s), e0, X['occupation'], X['sex'], X['unit'])
            res[name][key] = {'selected': s, 'cost': cs, 'outcome': RL.rule_outcome(s, cs['stats'], base)}
    rec = {'schema': 'CTS1-realized-v2', 'partition': P, 'direction': did, 'realized': res,
           'selection_file_sha256': C.sha(TF.pdir(P) / 'SELECT' / f'{did}.json'), 'cpu_s': time.process_time() - t0, 'utc': C.now(),
           'unit_id': os.environ.get(C.UNIT_ENV),
           'note': 'ens outcomes are increments of the deployed ensemble against the uncorrected ensemble on the same people'}
    C.write_json(out, rec)
    print(json.dumps({'partition': P, 'direction': did,
                      'reserve_crossing': {n: {k: v['outcome']['reserve_crossing'] for k, v in x.items()} for n, x in res.items()}}))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for c in ('run', 'check'):
        x = sub.add_parser(c); x.add_argument('--partition', required=True); x.add_argument('--direction', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        selections(a.partition, a.direction)
        if C.PARTITIONS[a.partition]['kind'] == 'diagnosis':
            realized(a.partition, a.direction)
    else:
        realized(a.partition, a.direction)
    return 0


if __name__ == '__main__':
    sys.exit(main())
