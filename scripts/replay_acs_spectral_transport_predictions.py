"""Replay every stored final prediction from the verified final releases (race-condition audit)."""
from __future__ import annotations
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from experiments import acs_spectral_transport_eval as ev
from experiments import run_acs_spectral_transport as run
from experiments import acs_spectral_audits as spec


def unit(args):
    s, c, mode = args[:3]; only = set(args[3]) if len(args) > 3 else None
    run._init()
    final, rel = run._final_inputs(ev.OUT, s); y = final['labels']
    d = ev.OUT/f'seed_{s}'/c/f'mode_{mode}'
    if not (d/'complete.json').exists(): return {'seed': s, 'condition': c, 'mode': mode, 'status': 'incomplete'}
    z = np.load(d/'predictions.npz'); ids = dict(zip(z['keys'].tolist(), z['ids'].tolist()))
    mism = 0; n = 0; keys = []
    if mode == 'B':
        rec = ev.read(ev.OUT/f'seed_{s}'/c/'audit_selection.json'); cache = ev.Loaded()
        items = [(f'audit/{b}/{role}/{cid}', role, lambda bundle, r=r: cache.predict(r, bundle, 'final')) for b, roles in rec['candidates'].items() for role, cs in roles.items() for cid, r in cs.items()]
    else:
        au = spec.load_audits(ev.DEV/f'seed_{s}'/c/'audits')
        def pa(bundle, cand):
            x = bundle[cand.space]; x = x[:, cand.columns] if cand.columns is not None else x
            return cand.base.predict_proba(np.ascontiguousarray(x))
        items = [(f'audit/{b}/{role}/{cid}', role, lambda bundle, cand=cand: pa(bundle, cand)) for b, roles in au['candidates'].items() for role, cs in roles.items() for cid, cand in cs.items()]
    for key, role, fn in items:
        if only is not None and key not in only: continue
        v, t = role.split('/'); valid = y[t] >= 0
        p = fn(run._bundle(rel, c, v, valid)); n += 1
        if not np.array_equal(np.ascontiguousarray(p, dtype=np.float64), z['p/'+ids[key]]): mism += 1; keys.append(key)
    return {'seed': s, 'condition': c, 'mode': mode, 'status': 'checked', 'predictions': n, 'mismatches': int(mism), 'keys': keys}


if __name__ == '__main__':
    run._init(); ev.verify_lock(); os.environ['PCRL_TRANSPORT_LOCK_VERIFIED'] = ev.sha_file(ev.lock_path())
    jobs = [(s, c, m) for s in (0, 1, 2) for c in ev.INTERFACES for m in ('A', 'B')]
    with ProcessPoolExecutor(4, initializer=run._init) as pool: out = list(pool.map(unit, jobs))
    # Resolve each disagreement with two further computations, each in a fresh process.
    for r in out:
        if r.get('mismatches'):
            votes = []
            for _ in range(2):
                with ProcessPoolExecutor(1, initializer=run._init) as pool:
                    votes.append(pool.submit(unit, (r['seed'], r['condition'], r['mode'], r['keys'])).result()['keys'])
            r['resolution'] = {k: 'stored_confirmed' if all(k not in v for v in votes) else 'stored_wrong' if all(k in v for v in votes) else 'unresolved' for k in r['keys']}
    (ev.OUT/'PREDICTION_REPLAY.json').write_text(json.dumps(out, indent=2)+'\n')
    print([r for r in out if r['status'] != 'checked' or r['mismatches']])
    print('stored_wrong', sum(v == 'stored_wrong' for r in out for v in r.get('resolution', {}).values()))
    print('checked', sum(r.get('predictions', 0) for r in out))
