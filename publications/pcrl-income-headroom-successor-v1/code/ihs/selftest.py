"""Unit IHS_SELFTEST: data-free correctness checks of the development evaluator on INVENTED arrays (no real data, no label).

Checks: (1) clause values equal the mean paired differences and the clustered SE equals a direct formula on a toy example;
(2) a release identical to the reference has zero clause values and zero SE; (3) the proxy grid runs on invented outputs, a
planted-SEX output reads near 1 and a SEX-independent output reads near 0.5; (4) caching returns identical values for an
identical released view; (5) the pair maximum includes the single-view readers.
"""
from __future__ import annotations
import json
import shutil

import numpy as np

from . import evaluate as EV
from . import common as C


def run():
    rng = np.random.default_rng(20261018)
    n, G = 1200, 800
    unit = np.sort(rng.integers(0, G, n)); sex = rng.integers(0, 2, n)
    y_inc = rng.integers(0, 2, n); y_occ = rng.integers(0, 5, n)

    def probs(k, logits):
        e = np.exp(logits - logits.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
    ref = {'income': probs(2, rng.normal(size=(n, 2))), 'occupation': probs(5, rng.normal(size=(n, 5)))}
    refs = {'U0': ref, 'Ucal': ref}
    labels = {'income': y_inc, 'occupation': y_occ}
    res = {}
    # (1)-(2)
    u_same = EV.utility(labels, ref, refs, sex, unit, 0)
    res['identical_release_zero'] = all(abs(c['value']) == 0 and c['se'] == 0 for c in u_same['clauses'])
    noisy = {'income': probs(2, np.log(ref['income']) + rng.normal(scale=0.3, size=(n, 2))), 'occupation': ref['occupation']}
    u = EV.utility(labels, noisy, refs, sex, unit, 0)
    c = [x for x in u['clauses'] if x['task'] == 'income' and x['reference'] == 'U0' and x['metric'] == 'brier_excess'][0]
    own = C.task_vectors(y_inc, noisy['income'], 2)['brier'] - C.task_vectors(y_inc, ref['income'], 2)['brier']
    sums = np.array([np.sum(own[unit == g] - own.mean()) for g in np.unique(unit)])
    Gs = len(sums); direct = np.sqrt(Gs / (Gs - 1) * np.sum(sums ** 2)) / n
    res['clustered_se_matches_direct'] = bool(abs(c['se'] - direct) < 1e-12 and abs(c['value'] - own.mean()) < 1e-12)
    # (3)-(5)
    C_old = EV.CACHE
    EV.CACHE = C.PRIV / 'selftest_proxy_cache'
    shutil.rmtree(EV.CACHE, ignore_errors=True)
    planted = {'income': probs(2, np.column_stack((np.zeros(n), 4.0 * (2 * sex - 1) + rng.normal(scale=0.2, size=n)))), 'occupation': ref['occupation']}
    p1 = EV.proxies(planted, sex, unit)
    p0 = EV.proxies(ref, sex, unit)
    p0b = EV.proxies(ref, sex, unit)
    res['planted_income_reads_high'] = p1['summary']['income']['min'] > 0.95
    res['planted_pair_max_includes_single'] = p1['summary']['pair']['min'] >= p1['summary']['income']['min'] - 1e-12
    res['independent_reads_near_half'] = abs(p0['summary']['pair']['mean'] - 0.5) < 0.08
    res['cache_identical'] = p0 == p0b
    res['proxy_cpu_example_s'] = sum(json.loads(f.read_text())['cpu_s'] for f in EV.CACHE.glob('*.json'))
    shutil.rmtree(EV.CACHE, ignore_errors=True)
    EV.CACHE = C_old
    res['status'] = 'PASS' if all(v for k, v in res.items() if k != 'proxy_cpu_example_s') else 'FAIL'
    C.write_json(C.RESULTS / 'checks' / 'EVALUATOR_SELFTEST.json', res)
    print(json.dumps(res))


if __name__ == '__main__':
    run()
