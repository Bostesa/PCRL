"""Investigator A label-free check (no label file opened, nothing computed from MODEL_SELECT labels).

1. SUCC identity: sar.fa.tfr._succ (ihs.fam_h.predict, H3_STRONG_LJDIR_PB) on admitted MODEL_SELECT FEATURES reproduces the frozen SUCC
   wire hashes pinned in results/pcrl_input_and_pair_release_v1/SOURCE_ADMISSION.json (outputs, not labels).
2. Teacher geometry on admitted MODEL_FIT features: how far SUCC's outputs sit from the admitted HGB task teachers (mean KL per person,
   share of changed decisions), i.e. the room the residual can move in; and the packed feature array the residual would see.
Run: OMP_NUM_THREADS=1 <repository>/.venv/bin/python results/pcrl_strong_anchor_release_v1/roles/investigator_a/check_teachers_labelfree.py
"""
import json
import os
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from sar import common as C, famkit as FK  # noqa: E402
from sar.fa import tfr  # noqa: E402
from nrm.fa import prc  # noqa: E402

t0 = time.process_time()
adm = json.loads((ROOT / 'results' / 'pcrl_input_and_pair_release_v1' / 'SOURCE_ADMISSION.json').read_text())
pins = adm['reference_releases_MODEL_SELECT_wire_sha256']
out = {'utc': C.now(), 'labels_read': 'none', 'versions': {}}


def kl(p, q):
    p, q = np.clip(p, 1e-300, None), np.clip(q, 1e-300, None)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), 1)))


for v in (0, 1):
    rec = {}
    F = C.features('MODEL_SELECT')
    t1 = time.process_time()
    s = tfr._succ(v, F['X'], F['feature_names'])
    rec['succ_predict_cpu_s_MS'] = round(time.process_time() - t1, 1)
    rec['succ_identity_MS'] = {t: C.wire_sha(s[t]) == pins[f'SUCC/S{v}'][t] for t in ('income', 'occupation')}
    M = C.features('MODEL_FIT')
    sm = tfr._succ(v, M['X'], M['feature_names'])
    T, tsha = tfr.task_cache(v)
    Zc = FK.compact(M['X'], M['feature_names'])
    hd = (prc._proba(T['models']['inc'], Zc), prc._proba(T['models']['occ'], Zc))
    ho = (T['oof']['inc'], T['oof']['occ'])
    A = tfr.blocks(Zc, (sm['income'], sm['occupation']), ho)
    ph = tfr.phi_fit(A)
    rec['task_cache_sha256'] = tsha
    rec['phi_width'] = int(tfr.phi_apply(ph, A).shape[1])
    rec['MF'] = {
        'kl_succ_to_hgb_oof': {'income': kl(sm['income'], ho[0]), 'occupation': kl(sm['occupation'], ho[1])},
        'kl_succ_to_hgb_deployed': {'income': kl(sm['income'], hd[0]), 'occupation': kl(sm['occupation'], hd[1])},
        'decision_disagreement_succ_vs_hgb_oof': {'income': float(np.mean(sm['income'].argmax(1) != ho[0].argmax(1))),
                                                   'occupation': float(np.mean(sm['occupation'].argmax(1) != ho[1].argmax(1)))},
        'distinct_vectors': {'succ_income': int(len(np.unique(np.round(sm['income'], 12), axis=0))),
                             'succ_occupation': int(len(np.unique(np.round(sm['occupation'], 12), axis=0)))},
        'hgb_oof_vs_deployed_kl': {'income': kl(hd[0], ho[0]), 'occupation': kl(hd[1], ho[1])},
    }
    s_ms = tfr._succ(v, F['X'], F['feature_names'])
    Zs = FK.compact(F['X'], F['feature_names'])
    hs = (prc._proba(T['models']['inc'], Zs), prc._proba(T['models']['occ'], Zs))
    rec['MS_features_only'] = {'kl_succ_to_hgb': {'income': kl(s_ms['income'], hs[0]), 'occupation': kl(s_ms['occupation'], hs[1])},
                               'decision_disagreement': {'income': float(np.mean(s_ms['income'].argmax(1) != hs[0].argmax(1))),
                                                         'occupation': float(np.mean(s_ms['occupation'].argmax(1) != hs[1].argmax(1)))}}
    out['versions'][f'V{v}'] = rec
out['cpu_s'] = round(time.process_time() - t0, 1)
p = Path(__file__).with_name('check_teachers_labelfree.json')
p.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
print(json.dumps(out, indent=1, sort_keys=True))
