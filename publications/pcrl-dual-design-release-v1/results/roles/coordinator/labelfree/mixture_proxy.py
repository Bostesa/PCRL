"""Label-free design diagnostic (coordinator; no label file is read): model-based proxies for releases that replace the person's SEX
propensity inside the occupation posterior's SEX mixture by a shrunk value (the Pope-Sydnor / MMP family), on the cached MODEL_FIT
out-of-fold predictions. For shrink a in {0, .25, .5, .75, 1}: r_O = (1 - w) qO0 + w qO1 with w = (1 - a) pi + a pbar. Reports the
expected KL from the task posterior qO (a proxy of the NLL spend) and the expected AUC of E[pi | release] under S ~ Bernoulli(pi)
for the occupation view and the pair (task-only income). These are proxies under the fitted propensity model, not recoveries.

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/labelfree/mixture_proxy.py
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np

from propensity_structure import ROOT, cv_pred, expected_auc


def main():
    t0 = time.process_time()
    res = {}
    for v in (0, 1):
        T = pickle.load(open(ROOT / f'.private/pcrl_new_release_mechanisms_v1/fa_cache/V{v}/TASK.pkl', 'rb'))
        S = pickle.load(open(ROOT / f'.private/pcrl_finite_routing_comparison_v1/fa_cache/V{v}/SEXSIDE.pkl', 'rb'))
        qI, qO, pi = T['oof']['inc'], T['oof']['occ'], np.clip(S['oof']['sex'][:, 1], 0.01, 0.99)
        qO0, qO1 = S['oof']['occ_s'][0], S['oof']['occ_s'][1]
        lpi = np.log(pi / (1 - pi))
        lI = np.log(qI[:, 1:] / qI[:, :1])
        pbar = float(pi.mean())
        r = {}
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            w = (1 - a) * pi + a * pbar
            rO = (1 - w)[:, None] * qO0 + w[:, None] * qO1
            lO = np.log(rO[:, 1:] / rO[:, :1])
            kl = float(np.mean(np.sum(qO * np.log(qO / rO), 1)))
            eo = cv_pred(lO, lpi)
            ep = cv_pred(np.column_stack([lI, lO]), lpi)
            r[f'a{a}'] = {'kl_from_qO': kl, 'eauc_occ': expected_auc(eo, pi), 'eauc_pair': expected_auc(ep, pi)}
        res[f'V{v}'] = r
    res['cpu_s'] = time.process_time() - t0
    print(json.dumps(res, indent=1))
    (Path(__file__).parent / 'MIXTURE_PROXY.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
