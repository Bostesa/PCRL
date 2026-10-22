"""CTL: the shared task-only twin of the PCRL input-and-pair release campaign (`pcrl_input_and_pair_release_v1`; EXPLORATORY
DEVELOPMENT on previously used public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records; not confirmation).

The service maps six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding X) to an income vector (n, 2) and an
occupation vector (n, 5); each recipient receives only its own float64 vector and its argmax decision. SEX is never a runtime input.

Shared path points (card CTL_PATHS_v1; saved by the family fit units through `ipr.paths.save_shared`): CTL_DQ_<step> and
CTL_RO_<strength>, the SEX-blind spend path (DQ) and the generic release-only path (RO) of the shared twin, deployed here from the
twin's models plus the path payload, so that a point bracketing both families' candidates is saved and counted once.

CTL_TWIN (card CTL_TWIN_v1): the cached task teachers of the new-release-mechanisms campaign (`TASK.pkl`, sklearn HGB with the
A1_PO_16 settings, inner household-CV T*, full refit on the 20,000 MODEL_FIT people with income / occupation labels; pinned by hash in
SOURCE_ADMISSION.json), released unchanged: P_inc = HGB_inc(Z), P_occ = HGB_occ(Z), Z = ohn.fam_a.compact_po(X). It is bitwise the
same release law as the dual-design campaign's DA_TWIN (whose saved release is read as the beside reference DDR_TWIN). This unit reads
no label: the teachers were fitted by their source campaign.

    python3 -m ipr.ctl.twin selftest
    python3 -m ipr.ctl.twin fit --version V --card results/pcrl_input_and_pair_release_v1/cards/CTL_TWIN_v1.md
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

from .. import common as C
from .. import famkit as FK

FAMILY = 'CTL'
CONFIGS = {'CTL_TWIN': {'kind': 'twin', 'what': 'equally trained task-only twin shared by the families: the cached HGB task teachers '
                                                '(A1_PO_16 settings, MODEL_FIT income / occupation); never reads SEX'}}
CODE_MODULES = ('ipr.ctl.twin', 'ipr.famkit', 'nrm.fa.prc', 'ohn.fam_a')


def _prc():
    from nrm.fa import prc
    return prc


def load_task_cache(version, row_sha=None):
    """The cached task teachers (TASK.pkl), hash-checked against SOURCE_ADMISSION.json (and against the MODEL_FIT row set if given)."""
    adm = json.loads((C.RESULTS / 'SOURCE_ADMISSION.json').read_text())
    tp = C.NRM_FA_CACHE / f'V{int(version)}' / 'TASK.pkl'
    want = adm['source_private_artefacts'][f'fa_cache/V{int(version)}/TASK.pkl']['sha256']
    b = tp.read_bytes()
    if FK.sha_bytes(b) != want:
        raise PermissionError('the cached task teachers differ from their admission pin')
    T = pickle.loads(b)
    if row_sha is not None and T['row_id_sha256'] != row_sha:
        raise PermissionError('the cached task teachers were fitted on other rows')
    return T, want


def load_sex_cache(version, row_sha=None):
    """The cached SEX-side teachers (SEXSIDE.pkl), hash-checked against SOURCE_ADMISSION.json."""
    adm = json.loads((C.RESULTS / 'SOURCE_ADMISSION.json').read_text())
    sp = C.FRC_FA_CACHE / f'V{int(version)}' / 'SEXSIDE.pkl'
    want = adm['frc_private_artefacts'][f'frc/fa_cache/V{int(version)}/SEXSIDE.pkl']['sha256']
    b = sp.read_bytes()
    if FK.sha_bytes(b) != want:
        raise PermissionError('the cached SEX-side teachers differ from their admission pin')
    S = pickle.loads(b)
    if row_sha is not None and S['row_id_sha256'] != row_sha:
        raise PermissionError('the cached SEX-side teachers were fitted on other rows')
    return S, want


def twin_vectors(P, Z):
    prc = _prc()
    return prc._proba(P['twin_inc'], Z), prc._proba(P['twin_occ'], Z)


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function (ipr.deploy contract); deterministic."""
    if keys is not None:
        raise PermissionError('deterministic release: per-request keys are refused')
    if int(draw) != 0:
        raise PermissionError('deterministic release: only draw 0 exists')
    P = FK.load_model(FAMILY, config_id, version)
    X, names = FK.check_X(X, feature_names, P)
    Z = FK.compact(X, names)
    pi, po = twin_vectors(P, Z)
    if P.get('op', 'twin') in ('dq', 'ro'):                  # a shared SPEND / RO path point (card CTL_PATHS_v1, ipr.paths)
        from .. import paths as PA
        pi, po = PA.predict_path(P, pi, po, Z)
    return {'income': pi, 'occupation': po}


def model_paths(config_id, version):
    return FK.model_paths(FAMILY, config_id, version)


def stage_fit(version, card, save, roles=None):
    t0 = time.process_time()
    F = C.features('MODEL_FIT')
    prc = _prc()
    row_sha = prc.array_sha(np.asarray(F['row_id']))               # the teachers' own row-set convention (nrm.fa.prc._prep)
    T, tsha = load_task_cache(version, row_sha)
    names = tuple(map(str, F['feature_names']))
    meta = FK.write_model(FAMILY, 'CTL_TWIN', version, {'feature_names': names, 'twin_inc': T['models']['inc'],
                                                       'twin_occ': T['models']['occ'], 'task_cache_sha256': tsha})
    arrays = FK.release_all(predict, 'CTL_TWIN', version, roles)
    unit = np.asarray(F['unit'])
    rec = FK.record(card, 'twin', CONFIGS['CTL_TWIN']['what'], (0, 0), 'none by this unit (the cached teachers were fitted on '
                    'MODEL_FIT income and occupation labels by their source campaign)',
                    'MODEL_FIT income and occupation labels of all 20,000 MODEL_FIT people reached the cached task teachers (no SEX)',
                    len(unit), len(np.unique(unit)), time.process_time() - t0,
                    {'op': 'twin', 'task_cache_sha256': tsha, 'model_sha256': meta['model_sha256'],
                     'T': {k: int(v) for k, v in T['T'].items()} if isinstance(T.get('T'), dict) else None})
    save(FAMILY, 'CTL_TWIN', version, arrays, rec)
    return {'version': int(version), 'task_cache_sha256': tsha, 'cpu_s': time.process_time() - t0}


def selftest():
    """Invented-data check of the payload algebra (no file, no label)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(300, 6))
    yi, yo = rng.integers(0, 2, 300), rng.integers(0, 5, 300)
    mi = HistGradientBoostingClassifier(max_iter=5).fit(Z, yi)
    mo = HistGradientBoostingClassifier(max_iter=5).fit(Z, yo)
    pi, po = twin_vectors({'twin_inc': mi, 'twin_occ': mo}, Z)
    ok = {'shapes': pi.shape == (300, 2) and po.shape == (300, 5),
          'probabilities': bool(np.allclose(pi.sum(1), 1) and np.allclose(po.sum(1), 1))}
    ok['all_ok'] = all(ok.values())
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ipr.ctl.twin')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    f = sub.add_parser('fit')
    f.add_argument('--version', type=int, required=True)
    f.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        r = selftest(); print(json.dumps(r)); return 0 if r['all_ok'] else 1
    from .. import release as RL
    out = stage_fit(a.version, a.card, RL.save)
    print(json.dumps(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
