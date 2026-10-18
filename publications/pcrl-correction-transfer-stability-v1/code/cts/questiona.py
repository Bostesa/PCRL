"""Question A on one partition (RULES.md section 6): does the GUARD estimate predict the realized CHECK increment better than the
source's out-of-fold (OOF) estimate? Implemented with cts.rules.prediction_comparison (paired household bootstrap B = 2000).

    python3 -m cts.questiona run --partition P --direction ID [--deployment full|ens] [--oof OOF|NESTED]

Per-person increments (corrected minus uncorrected on the same people) at the 10 nonzero ladder strengths:
- OOF: fold pipeline k on fold k of FIT with the cross-fitted witness base (the source estimator) - or NESTED: every upstream layer
  refitted inside each outer fold (cts.nested; REFIT routing only);
- GUARD: the deployed release (full) or the deployed ensemble (ens) on GUARD;
- CHECK: the same deployment on CHECK (D partitions: diagnostic CHECK; R partitions: the reserved panel, only after the committed
  REPAIR_FREEZE).
Labels: FIT ('fit'), GUARD ('select'), CHECK ('check_diag' / 'check_repair'). Writes <partition>/QUESTION_A/<ID>__<dep>__<oof>.json.
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


def _incr(p_by_s, p0, y):
    db, dn, da = [], [], []
    for p in p_by_s:
        v = TF.cost_vectors(p, p0, y)
        db.append(v['brier']); dn.append(v['nll']); da.append(v['acc_loss'])
    return np.column_stack(db), np.column_stack(dn), np.column_stack(da)


def run(P, did, dep='full', oof='OOF'):
    t0 = time.process_time()
    kind = C.PARTITIONS[P]['kind']
    out = TF.pdir(P) / 'QUESTION_A' / f'{did}__{dep}__{oof}.json'
    if out.exists():
        raise PermissionError(f'{out.name} exists (append-only)')
    if kind == 'repair' and not C.repair_frozen():
        raise PermissionError('repair CHECK stays closed until the committed REPAIR_FREEZE')
    srec = json.loads((TF.pdir(P) / 'SELECT' / f'{did}.json').read_text())
    slack = srec['baseline']
    wrec, W, O = TF.load_witness(P)
    drec, T, D = TF.load_direction(P, did)
    S10 = TF.LADDER[1:]
    sets = {}
    F = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'fit', f'question A {P} {did}: per-person {oof} increments on FIT',
               partition=P, prole='FIT')
    if oof == 'OOF':
        z0 = TF.logp(O['oof']); p0 = TF.probs(z0)
        ps = [TF.probs(z0 + s * T['delta_oof']) for s in S10]
    else:
        nf = TF.pdir(P) / 'NESTED' / f'NESTED_{did.split("_", 1)[1]}.npz'
        if not did.startswith('REFIT_') or not nf.is_file():
            raise PermissionError('the nested variant exists only for REFIT directions with a cts.nested run')
        with np.load(nf) as z:
            pw, dn = z['witness_oof'].copy(), z['delta_oof'].copy()
        p0 = pw
        ps = [TF.release_at(pw, dn, s) for s in S10]
    db, dnll, da = _incr(ps, p0, F['occupation'])
    sets['OOF'] = {'db': db, 'dn': dnll, 'da': da, 'unit': F['unit'], 'sex': F['sex']}
    for name, op in (('GUARD', 'select'), ('CHECK', 'check_diag' if kind == 'diagnosis' else 'check_repair')):
        L = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), op, f'question A {P} {did}: per-person {dep} increments on {name}',
                   partition=P, prole=name)
        r = L['rows']
        if dep == 'full':
            pw, dl = W['dep__MODEL_FIT'][r], D['full__MODEL_FIT'][r]
            p0 = pw; ps = [TF.release_at(pw, dl, s) for s in S10]
        else:
            pf, df = W['fold__MODEL_FIT'][:, r], D['folds__MODEL_FIT'][:, r]
            p0 = TF.ensemble_at(pf, df, 0.0); ps = [TF.ensemble_at(pf, df, s) for s in S10]
        db, dnll, da = _incr(ps, p0, L['occupation'])
        sets[name] = {'db': db, 'dn': dnll, 'da': da, 'unit': L['unit'], 'sex': L['sex']}
    res = RL.prediction_comparison(sets, slack)
    rec = {'schema': 'CTS1-question-a-v1', 'partition': P, 'direction': did, 'deployment': dep, 'oof_estimator': oof,
           'result': res, 'A1_assumption': 'A1_NOT_SUPPORTED' if did.startswith('FROZEN_') else 'exchangeable households (REFIT routing)',
           'cpu_s': time.process_time() - t0, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json(out, rec)
    print(json.dumps({'partition': P, 'direction': did, 'dep': dep, 'oof': oof, 'D': round(res['D'], 4), 'q05': round(res['q05'], 4),
                      'label': res['label'], 'MAE_rel_GUARD': round(res['MAE_rel_GUARD'], 4), 'MAE_rel_OOF': round(res['MAE_rel_OOF'], 4)}))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    x = sub.add_parser('run'); x.add_argument('--partition', required=True); x.add_argument('--direction', required=True)
    x.add_argument('--deployment', default='full', choices=('full', 'ens')); x.add_argument('--oof', default='OOF', choices=('OOF', 'NESTED'))
    a = ap.parse_args(argv)
    run(a.partition, a.direction, a.deployment, a.oof)
    return 0


if __name__ == '__main__':
    sys.exit(main())
