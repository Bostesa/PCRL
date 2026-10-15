"""Nonparametric resampling check of the normal-model joint power (planning device, not new evidence).

Future studies of G groups are simulated by drawing G groups with replacement from the 2,234 INNER groups'
influence-function rows (private), centring at the scenario means, and applying each decision rule with
ESTIMATED per-clause standard errors (weighted sample variance / G). Compared with the known-variance
normal-model power at the same G.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def run(spec_path):
    from .evaluation_queue import queue_config
    from . import power
    active, cfg, verify, guard = queue_config(spec_path, 'verification')
    f, names, idx, theta, cov, Mpos, Mneg = power.load(verify(cfg['inputs']['path'], cfg['inputs']['sha256']))
    with np.load(verify(cfg['phi']['path'], cfg['phi']['sha256'])) as z:
        phi = z['phi'].copy(); pnames = [str(x) for x in z['names']]
    if pnames != names:
        raise PermissionError('influence rows and public inputs differ')
    rng = np.random.default_rng(20261016)
    out = []
    for cname, kind, req, b in (('B_narrow', 'B', power.ORIGINAL_REQUIRED, 'observed'), ('A_original_6', 'A', power.ORIGINAL_REQUIRED, 0.025)):
        cl = power.claim(idx, kind, req)
        A = np.array([c[1] for c in cl]); dirs = np.array([c[2] for c in cl], dtype=float); bounds = np.array([c[3] for c in cl])
        mu = A @ theta
        for j, c in enumerate(cl):
            if b != 'observed' and c[0].startswith('benefit|benefit_vs_') and c[0].split('_vs_')[1] in power.CODES:
                mu[j] = b
        Phi = A @ phi                                     # (clauses, groups)
        Phi = Phi - Phi.mean(1, keepdims=True)
        Sig = A @ cov @ A.T
        for dname, crit in (('IUT_alpha0.025', stats.norm.ppf(0.975)), ('SIM_claimK', stats.norm.ppf(1 - 0.05 / (2 * len(cl))))):
            gs, _, _ = power.gstar(mu, Sig, dirs, bounds, crit, 20000)
            for G in cfg['G_values'][cname][dname]:
                guard()
                passes = 0; reps = cfg['reps']
                for _ in range(reps):
                    w = rng.multinomial(G, np.full(Phi.shape[1], 1 / Phi.shape[1])).astype(np.float64)
                    m = Phi @ w / G
                    v = (Phi ** 2) @ w / G - m ** 2
                    se = np.sqrt(np.maximum(v, 0) / G)
                    t = mu + m
                    passes += bool(np.all(dirs * (t - bounds) + crit * se <= 0))
                p = passes / reps
                out.append({'claim': cname, 'design': dname, 'G': G, 'resampling_power': p, 'resampling_mc_se': float(np.sqrt(p * (1 - p) / reps)),
                            'normal_model_power': float(np.mean(gs <= G)), 'reps': reps})
    res = {'schema': 'TC1-resampling-check-v1', 'queue_unit_id': active['unit_id'], 'rows': out, 'definition': __doc__}
    with Path(cfg['output']).open('x') as fh:
        json.dump(res, fh, indent=2)
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
