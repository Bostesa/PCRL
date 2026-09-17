"""Recompute every Mode B 2017-validation score with confirmed predictions; report selection changes.

Reads only fitting/validation partitions. Written after the nondeterministic-prediction
finding (lock amendment 2) to test whether any pre-final selection was affected.
"""
from __future__ import annotations
import json, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from experiments import acs_spectral_transport_eval as ev
from experiments import run_acs_spectral_transport as run
from experiments.acs_transfer_heads import metrics


def unit(args):
    s, c = args
    run._init()
    rel, labels = run.fit_inputs(ev.OUT, s)
    rec = ev.read(ev.OUT/f'seed_{s}'/c/'audit_selection.json'); cache = ev.Loaded(); bad = []; changed = []; n = 0
    for b, roles in rec['candidates'].items():
        for role, cs in roles.items():
            v, t = role.split('/'); y = labels['attacker_validation'][t]; valid = y >= 0
            bundle = {'wire': rel['attacker_validation']['wire'][c][v][valid]}
            if rel['attacker_validation']['derived'][c] is not None: bundle['derived'] = rel['attacker_validation']['derived'][c][v][valid]
            scores = {}
            for cid, r in cs.items():
                p = cache.predict(r, bundle, 'attacker_validation'); n += 1
                try: scores[cid] = {'log_loss': metrics(y[valid], p, ev.old.CLASSES[t])['log_loss']}
                except ValueError:
                    bad.append({'budget': b, 'role': role, 'candidate': cid, 'invalid_prediction': True, 'shape': list(p.shape), 'finite': bool(np.isfinite(p).all()),
                                'min': float(np.nanmin(p)), 'max': float(np.nanmax(p)), 'max_rowsum_error': float(np.nanmax(abs(p.sum(1)-1)))})
                    scores[cid] = {'log_loss': float('inf')}; continue
                if scores[cid]['log_loss'] != rec['validation_scores'][b][role][cid]['log_loss']:
                    bad.append({'budget': b, 'role': role, 'candidate': cid, 'stored': rec['validation_scores'][b][role][cid]['log_loss'], 'recomputed': scores[cid]['log_loss']})
            sel = ev.select(cs, scores)
            for scope, cid in sel.items():
                if cid != rec['selections'][b][role][scope]: changed.append({'budget': b, 'role': role, 'scope': scope, 'stored': rec['selections'][b][role][scope], 'recomputed': cid})
    util = ev.read(ev.OUT/f'seed_{s}'/c/'utility_selection.json'); ubad = []
    for role, cs in util['candidates'].items():
        v, t = role.split('/'); y = labels['task_validation'][t]; valid = y >= 0
        for cid, cc in cs.items():
            from experiments.acs_transfer_heads import load_candidate
            p = ev.stable_predict(load_candidate(cc['dir']).predict_proba, rel['task_validation']['wire'][c][v][valid])
            if metrics(y[valid], p, 2)['log_loss'] != cc['validation_log_loss']: ubad.append({'role': role, 'candidate': cid})
    return {'seed': s, 'condition': c, 'predictions': n, 'score_mismatches': bad, 'selection_changes': changed,
            'utility_mismatches': ubad, 'events': list(ev.NONDETERMINISM_EVENTS)}


if __name__ == '__main__':
    jobs = [(s, c) for s in (0, 1, 2) for c in ev.INTERFACES]
    with ProcessPoolExecutor(8, initializer=run._init) as pool: out = list(pool.map(unit, jobs))
    (ev.OUT/'VALIDATION_SCORE_AUDIT.json').write_text(json.dumps(out, indent=2)+'\n')
    print('predictions', sum(r['predictions'] for r in out), 'score mismatches', sum(len(r['score_mismatches']) for r in out),
          'selection changes', sum(len(r['selection_changes']) for r in out), 'utility mismatches', sum(len(r['utility_mismatches']) for r in out),
          'events', sum(len(r['events']) for r in out))
    for r in out:
        if r['score_mismatches'] or r['selection_changes']: print(r['seed'], r['condition'], r['score_mismatches'][:3], r['selection_changes'])
