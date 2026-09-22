"""Secondary one-token replay fixture (runs after scoring; never feeds any scored number).

Persistent private tokens: u_i = HMAC-SHA256(secret, person_id)/2^256, token = inverse CDF of the
person's released token law. Repeated requests with the same secret return the same token. Averaging
the realized one-draw loss over independent secrets (different populations' coins, not redraws) must
converge to the exact expected loss.
"""
from __future__ import annotations
import hashlib
import hmac
import secrets

import numpy as np

from .audit_panel import unit_dir
from .common import OUT, atomic_json, now
from .pipeline import root_for


def tokens(secret, ids, law):
    u = np.array([int.from_bytes(hmac.new(secret, i.encode(), hashlib.sha256).digest(), 'big')/2**256 for i in ids])
    c = np.cumsum(law, 1); c[:, -1] = 1.
    return (u[:, None] >= c).sum(1)


def main(dataset='acs2016', units=(('Q', 0, 'attack:A/SEX'), ('Q', 0, 'utility:A/same_residence'), ('RR75', 1, 'attack:A/RAC1P')),
         n_secrets=200):
    out = []
    for short, a, role in units:
        d = unit_dir(root_for(dataset), short, a, role)
        with np.load(d/'score_final.npz') as s, np.load(d/'pred_final.npz') as p:
            ids, y, exact = s['ids'], s['y'], s['loss']
            q, law = p['probabilities'], p['token_probs']
        keyed = [secrets.token_bytes(32) for _ in range(n_secrets)]
        first = tokens(keyed[0], ids, law)
        repeat_identical = bool(np.array_equal(first, tokens(keyed[0], ids, law)))
        means = []
        for k in keyed:
            t = tokens(k, ids, law)
            means.append(float(np.mean(-np.log(np.maximum(q[np.arange(len(y)), t, y], 1e-9)))))
        means = np.array(means)
        mc_se = float(means.std(ddof=1)/np.sqrt(n_secrets))
        out.append({'unit': [short, a, role], 'people': int(len(ids)), 'tokens_per_person': int(law.shape[1]),
                    'exact_mean_loss': float(exact.mean()), 'mc_mean_loss': float(means.mean()), 'mc_se': mc_se,
                    'z': float((means.mean()-exact.mean())/mc_se) if mc_se > 0 else 0.,
                    'repeat_request_identical': repeat_identical, 'secrets': n_secrets,
                    'secrets_persisted': False})
    atomic_json(OUT/'private'/'TOKEN_FIXTURE.json', {'created_utc': now(), 'units': out})
    return out


if __name__ == '__main__':
    for r in main():
        print(r)
