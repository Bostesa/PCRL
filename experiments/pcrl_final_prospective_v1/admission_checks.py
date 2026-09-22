"""Pre-fit 2016 checks on fitting/validation pools only (final labels are never read here)."""
from __future__ import annotations

import numpy as np

from experiments.pcrl_task_directed_release_v1.evaluation import _wire
from .common import ANCHORS, OUT, PANEL, array_hash, atomic_json, now
from .prepare import load_pools
from .releases import FrozenReleases

POOLS = ('fit', 'attack_val', 'task_val')
SERVICES = {'income_binary': ('ha', slice(0, 2)), 'civilian_at_work': ('ha', slice(2, 4)), 'public_coverage': ('hb', slice(0, 2))}


def ce(y, p):
    m = y >= 0
    return float(np.mean(-np.log(np.clip(p[m][np.arange(m.sum()), y[m]], 1e-12, 1)))), float(np.mean(p[m].argmax(1) == y[m]))


def main(dataset='acs2016'):
    out = {'created_utc': now(), 'dataset': dataset, 'pools': POOLS, 'anchors': {}}
    for a in ANCHORS:
        data, enc = load_pools(dataset, a, POOLS)
        fr = FrozenReleases(a)
        rec = {'aliases': {}, 'h_parity': {}, 'support': {}, 'services': {}}
        for pool in POOLS:
            d = data[pool]
            prints, parity = {}, True
            for short in PANEL:
                rel = fr.release(short, {pool: {k: d[k] for k in ('x', 'ha', 'J')}}, {pool: enc[pool]})[pool]
                xa, p = _wire(d, rel, 'A', 'release' if short != 'H' else 'H')
                xab, _ = _wire(d, rel, 'AB', 'release' if short != 'H' else 'H')
                parity &= xa[:, :4].tobytes() == d['ha'].tobytes() and xab[:, :4].tobytes() == d['ha'].tobytes()
                key = (array_hash(rel['token_probs']), None if rel['aux'] is None else array_hash(np.asarray(rel['aux'])))
                prints.setdefault(key, []).append(short)
            rec['aliases'][pool] = sorted(prints.values())
            rec['h_parity'][pool] = bool(parity)
            if a == 0:
                lab = d['labels']
                rec['support'][pool] = {'people': int(len(d['ids'])), 'households': int(len(np.unique(d['households']))),
                                        **{t: {'valid': int((lab[t] >= 0).sum()), 'missing': int((lab[t] < 0).sum()),
                                               'class_counts': np.bincount(lab[t][lab[t] >= 0], minlength=9 if t == 'RAC1P' else 2).tolist()}
                                           for t in ('SEX', 'RAC1P', 'same_residence')}}
            rec['services'][pool] = {t: dict(zip(('log_loss', 'accuracy'), ce(d['labels'][t], d[k][:, s])))
                                     for t, (k, s) in SERVICES.items()}
        out['anchors'][a] = rec
    out['all_distinct_families'] = all(len(g) == len(PANEL) for r in out['anchors'].values() for g in r['aliases'].values())
    out['h_parity_all'] = all(v for r in out['anchors'].values() for v in r['h_parity'].values())
    atomic_json(OUT/'ADMISSION_CHECKS_2016.json', out)
    return out


if __name__ == '__main__':
    r = main()
    print(r['all_distinct_families'], r['h_parity_all'])
