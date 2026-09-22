"""Post-lock final-pool check of exact H preservation and service accuracy."""
from __future__ import annotations

import numpy as np

from experiments.pcrl_task_directed_release_v1.evaluation import _wire
from .admission_checks import SERVICES, ce
from .common import ANCHORS, OUT, PANEL, atomic_json, now
from .prepare import load_pools
from .releases import FrozenReleases


def main(dataset='acs2016'):
    result = {'created_utc': now(), 'dataset': dataset, 'pool': 'final', 'anchors': {}}
    for anchor in ANCHORS:
        data, encoded = load_pools(dataset, anchor, ('final',))
        rows = data['final']
        releases = FrozenReleases(anchor)
        parity = {}
        for short in PANEL:
            release = releases.release(short, {'final': {k: rows[k] for k in ('x', 'ha', 'J')}},
                                       {'final': encoded['final']})['final']
            a, _ = _wire(rows, release, 'A', 'H' if short == 'H' else 'release')
            b, _ = _wire(rows, release, 'B', 'H')
            ab, _ = _wire(rows, release, 'AB', 'H' if short == 'H' else 'release')
            parity[short] = bool(a[:, :4].tobytes() == rows['ha'].tobytes()
                                 and b.tobytes() == rows['hb'].tobytes()
                                 and ab[:, :6].tobytes() == np.column_stack((rows['ha'], rows['hb'])).tobytes())
        metrics = {target: dict(zip(('log_loss', 'accuracy'), ce(rows['labels'][target], rows[key][:, sl])))
                   for target, (key, sl) in SERVICES.items()}
        result['anchors'][anchor] = {'rows': int(len(rows['ids'])), 'service_parity': parity,
                                     'services': metrics}
    result['service_parity_all'] = all(value for anchor in result['anchors'].values()
                                        for value in anchor['service_parity'].values())
    atomic_json(OUT/'SERVICE_2016_FINAL.json', result)
    return result


if __name__ == '__main__':
    print(main()['service_parity_all'])
