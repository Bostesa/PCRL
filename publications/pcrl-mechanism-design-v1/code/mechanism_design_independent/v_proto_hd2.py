"""Robustness check of the registered P7 probe (V_PROTO_HD_V1): is the twin's lower fitted pair AUC a reader-capacity
artefact? Strong near-Bayes readers on the natural 3-dim coordinates of each release (logit of income, two occupation
log-ratios): HGB (500 iterations, 63 leaves) trained on 1,000,000 rows, tested on 200,000. MMP's exact Bayes value
Phi(||U'd||/sqrt 2) calibrates how close the strong reader gets to Bayes. Same law and draws as v_proto_hd.
"""
from mechanism_design import no_real_data
no_real_data.install()

import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from mechanism_design_independent.v_proto_hd import make, sample

OUT = Path(__file__).resolve().parent.parent / 'results' / 'pcrl_mechanism_design_v1' / 'roles' / 'math_verifier' / 'V_PROTO_HD_V2.json'
ND = NormalDist()


def coords(rel):
    p1, p2 = rel
    return np.column_stack([np.log(p1[:, 1] / p1[:, 0]), np.log(p2[:, 1] / p2[:, 0]), np.log(p2[:, 2] / p2[:, 0])])


def main():
    out = {}
    for draw in range(3):
        d, b, B = make(draw)
        U, _ = np.linalg.qr(np.stack([b, B[0], B[1]], 1))
        rng = np.random.default_rng(4242 + draw)
        tr = sample(1_000_000, rng, d, b, B)
        te = sample(200_000, rng, d, b, B)
        rec = {'mmp_bayes_exact': ND.cdf(np.linalg.norm(U.T @ d) / math.sqrt(2)), 'AUC_star_X_exact': ND.cdf(np.linalg.norm(d) / math.sqrt(2))}
        for arm, idx in (('twin', 2), ('mmp', 3)):
            m = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, early_stopping=True,
                                               validation_fraction=0.05, n_iter_no_change=30, random_state=draw)
            m.fit(coords(tr[idx]), tr[0])
            rec[f'{arm}_strong_hgb_1M'] = float(roc_auc_score(te[0], m.predict_proba(coords(te[idx]))[:, 1]))
        rec['twin_minus_mmp_strong'] = rec['twin_strong_hgb_1M'] - rec['mmp_strong_hgb_1M']
        rec['mmp_reader_gap_to_exact_bayes'] = rec['mmp_bayes_exact'] - rec['mmp_strong_hgb_1M']
        out[f'draw{draw}'] = rec
        print(f'draw {draw}:', json.dumps({k: round(v, 4) for k, v in rec.items()}), flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
