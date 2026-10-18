"""Verifier replication (data-free) of the screen operating-characteristic law L1 (RULES.md section 10): per-cell recipient gate and
AGG1 pass rates under d_{s,R,k} = mu + A_{s,R} + B_{s,R,k}, A jointly normal across the two seeds with correlation rho_seed, B
independent, every cell SE = se_cell; two repetitions, two reader seeds. Own vectorized simulation (own seeds), compared with the
recorded SIM.json pass rates within 4 Monte Carlo standard errors of the difference. Invented numbers only; no data.

PCRL correction-transfer campaign (CTS1).

    python3 .../v_sim.py --tag V1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

SIM = V.ROOT / '.private' / 'pcrl_correction_transfer_stability_v1' / 'runs' / 'SCREEN' / 'SIM.json'


def simulate(mu, rho, sd_A, sd_k, se, phi_prior, n_sim, rng, allow=0.010, z=V.Z_REGISTERED):
    cov = sd_A ** 2 * np.array([[1.0, rho], [rho, 1.0]])
    A = rng.multivariate_normal(np.zeros(2), cov, size=(n_sim, 2))          # (n_sim, R, seed)
    Bn = rng.normal(0.0, sd_k, size=(n_sim, 2, 2, 2))                       # (n_sim, R, seed, k)
    d = mu + A[:, :, :, None] + Bn
    gate = np.all(d <= allow, axis=(1, 2, 3))
    dbar = d.mean(axis=(1, 2, 3))
    yR = d.mean(axis=(2, 3))                                                # (n_sim, R)
    v_samp = 2 * se ** 2 / 4.0
    phi_emp = ((yR - dbar[:, None]) ** 2).sum(1) / 1.0 / se ** 2
    phi = np.maximum(np.maximum(1.0, phi_prior), phi_emp)
    U = dbar + z * np.sqrt(phi * v_samp)
    return float(gate.mean()), float((U <= allow).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--n', type=int, default=20000)
    a = ap.parse_args()
    rec = json.loads(SIM.read_text())
    rows, bad = [], []
    rng = np.random.default_rng(20261018)
    for e in rec['L1_single']:
        if e['view'] != 'income' or e['n_R'] != 2 or e['mu'] not in (0.0, 0.01, -0.01, 0.02) or e['rho_seed'] not in (0.0, 0.76):
            continue
        g, ag = simulate(e['mu'], e['rho_seed'], e['sd_A'], e['sd_k'], e['se_cell'], e['phi_prior'], a.n, rng)
        se_g = math.sqrt(max(g * (1 - g), 1e-12) / a.n + e['old_mc_se'] ** 2)
        se_a = math.sqrt(max(ag * (1 - ag), 1e-12) / a.n + e['agg_mc_se'] ** 2)
        r = {'mu': e['mu'], 'rho_seed': e['rho_seed'], 'gate_mine': g, 'gate_recorded': e['old_pass_rate'], 'agg1_mine': ag,
             'agg1_recorded': e['agg_pass_rate'], 'gate_z': (g - e['old_pass_rate']) / se_g, 'agg1_z': (ag - e['agg_pass_rate']) / se_a}
        rows.append(r)
        if abs(r['gate_z']) > 4 or abs(r['agg1_z']) > 4:
            bad.append(r)
    chk = {'id': f'C6.5.SCREEN_STUDY_L1_INCOME__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': 'screen study (question C), law L1, income view, two repetitions: pass rates of the old per-cell gate and AGG1 at mu in '
                   '{-0.010, 0, +0.010, +0.020} and rho_seed in {0, 0.76}',
           'method': f'own vectorized simulation of law L1 from RULES.md section 10 with the recorded components, n = {a.n} per setting, own '
                     'seed; agreement within 4 Monte Carlo SEs of the difference',
           'status': 'PASS' if (rows and not bad) else ('NOT_RUN' if not rows else 'FAIL'),
           'numbers': {'settings': rows, 'outside_4_mc_se': bad,
                       'reading': 'simulated rates under that law only (not universal error rates); at mu = 0 the gate passes about '
                                  'one release in five and AGG1 fewer, so neither screen admits true zero income-view differences reliably '
                                  'at two repetitions'}}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__)
    V.write_record(f'V_SIM__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], 'rows': [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]},
                     indent=1))


if __name__ == '__main__':
    main()
