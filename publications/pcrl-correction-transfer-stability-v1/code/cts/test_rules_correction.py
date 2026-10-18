"""Data-free test of the recorded question A bootstrap correction (cts.rules.prediction_comparison_registered).

    python3 -m cts.test_rules_correction

Checks against a brute-force reference that resamples households explicitly, ONE resample per set and replicate shared by every
statistic and strength (RULES.md section 6): identical point estimates and D as the original function, identical bootstrap draws as
the reference (same generator order), and a larger SE than the superseded per-statistic scheme on positively correlated statistics.
"""
from __future__ import annotations
import hashlib
import json
import sys

import numpy as np

from . import rules as RL


def _sets(rng, n=300, G=120):
    out = {}
    for name in ('OOF', 'GUARD', 'CHECK'):
        unit = rng.integers(0, G, n)
        sex = rng.integers(0, 2, n)
        base = rng.normal(size=(n, 1)) * 1e-3 + np.linspace(0, 2e-3, 10)[None, :]
        db = base + rng.normal(scale=2e-4, size=(n, 10))
        dn = 2 * base + rng.normal(scale=2e-4, size=(n, 10))          # positively correlated with db on the same people
        da = base + rng.normal(scale=2e-4, size=(n, 10))
        out[name] = {'db': db, 'dn': dn, 'da': da, 'unit': unit, 'sex': sex}
    return out


def _reference(sets, slack, B, tag):
    ss = RL.stat_slack(slack); ks = [k for k in RL.STATS if ss[k] > 0]
    rng = np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:15], 16))
    boots = {}
    for name in ('OOF', 'GUARD', 'CHECK'):
        st = sets[name]
        hh, inv = np.unique(st['unit'], return_inverse=True)
        G = len(hh)
        W = rng.multinomial(G, np.full(G, 1.0 / G), size=B)
        for b in range(B):
            w = W[b][inv].astype(np.float64)                         # each person weighted by its household's draw count
            for k, (arr, mask) in RL._set_stats(st['db'], st['dn'], st['da'], st['sex']).items():
                m = np.ones(len(w), dtype=bool) if mask is None else mask
                boots.setdefault((name, k), []).append((w[m][:, None] * arr[m]).sum(0) / w[m].sum())
    Db = []
    for b in range(B):
        g = np.mean([np.abs(boots['CHECK', k][b] - boots['GUARD', k][b]) / ss[k] for k in ks], axis=0).mean()
        o = np.mean([np.abs(boots['CHECK', k][b] - boots['OOF', k][b]) / ss[k] for k in ks], axis=0).mean()
        Db.append(o - g)
    return np.asarray(Db)


def main():
    rng = np.random.default_rng(7)
    sets = _sets(rng)
    slack = {c: 0.002 for c in RL.OCC_CONDITIONS}
    new = RL.prediction_comparison_registered(sets, slack, B=200, seed_tag='CTS1_test')
    old = RL.prediction_comparison(sets, slack, B=200, seed_tag='CTS1_test')
    ref = _reference(sets, slack, 200, 'CTS1_test')
    checks = {'same_point_D': abs(new['D'] - old['D']) < 1e-15,
              'same_MAE': abs(new['MAE_rel_GUARD'] - old['MAE_rel_GUARD']) < 1e-15 and abs(new['MAE_rel_OOF'] - old['MAE_rel_OOF']) < 1e-15,
              'matches_bruteforce_se': abs(new['boot_se'] - float(ref.std(ddof=1))) < 1e-12,
              'matches_bruteforce_q05': abs(new['q05'] - float(np.quantile(ref, 0.05))) < 1e-12,
              'larger_se_than_superseded_scheme': new['boot_se'] > old['boot_se']}
    ok = all(checks.values())
    print(json.dumps({'ok': ok, **checks, 'se_registered': new['boot_se'], 'se_superseded': old['boot_se']}))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
