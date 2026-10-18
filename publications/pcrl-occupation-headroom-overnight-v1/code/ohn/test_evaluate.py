"""Invented-data test of ohn.evaluate.complete_eval (no real data, no label): clause / guard / decomposition plumbing."""
import json

import numpy as np

from . import evaluate as EV
from . import common as C


def main():
    rng = np.random.default_rng(11)
    n = 3000
    unit = np.repeat(np.arange(n // 2), 2)
    sex = (rng.random(n) < 0.48).astype(np.int64)
    yi = (rng.random(n) < 0.4).astype(np.int64)
    yo = rng.integers(0, 5, n)

    def soft(z):
        e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
    zo = rng.normal(size=(n, 5)); zo[np.arange(n), yo] += 1.0
    zi = rng.normal(size=(n, 2)); zi[np.arange(n), yi] += 0.8
    refs = {'U0': {'income': soft(zi), 'occupation': soft(zo)}, 'Ucal': {'income': soft(0.9 * zi), 'occupation': soft(0.9 * zo)}}
    others = {'T_TREE': soft(1.1 * zo), 'TASK_REFIT': soft(1.05 * zo), 'JOINT': soft(0.8 * zo)}
    S = {'row_id': np.arange(n), 'unit': unit, 'sex': sex, 'income': yi, 'occupation': yo}
    m = EV.complete_eval(0, soft(zi + 0.05), soft(zo * 1.02), S, refs, others)
    u = m['utility']
    checks = {'n_clauses': len(u['clauses']), 'n_guards': len(u['guards']),
              'per_class_keys': len(m['occupation_per_class']), 'versus_keys': sorted(m['occupation_versus_task_only_references'])[:3],
              'contributions_sum_to_total': abs(sum(m['occupation_per_class'][f'U0|nll_excess|class{c}']['contribution'] for c in range(5))
                                                - [c['value'] for c in u['clauses'] if c['task'] == 'occupation' and c['reference'] == 'U0'
                                                   and c['metric'] == 'nll_excess'][0]) < 1e-12}
    checks['all_ok'] = checks['n_clauses'] == 12 and checks['n_guards'] == 2 and checks['contributions_sum_to_total']
    print(json.dumps(checks, indent=1))
    if not checks['all_ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
