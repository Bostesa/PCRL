"""Restore check of the PCRL finite-output solver-adequacy campaign, run INSIDE a restored copy (backup alone; no label is read).

This campaign fitted no release configuration, so there is no release, deployment package or reader grid to re-run. What it
restores and re-runs instead:
- the admitted source mirror copies re-hash against the occupation-headroom admission record, and the read-only source artefacts
  this campaign pins (frc runs and caches, the NRM1 task caches) re-hash against SOURCE_ADMISSION.json;
- this campaign's private outputs re-hash against the records that name them: the P1 replay and nested per-person files (their
  records' per_person_file_sha256) and the source partitions / books (decisions/ROUTE_S_INPUT.json private_file_sha256);
- re-runs from the backup alone (fit interpreter pass): every saved source book re-routes the cached MODEL_FIT statistics to its
  saved partition bitwise; the fitted objectives of the original and SV1 PRC states at strength 8 equal the source record within
  1e-12; the P1 replay's per-person loss means equal the replayed record (label arrays are not read); the solver selftest passes;
- in every pass: the route S label and the advancement decision re-derived with the restored frozen rules from the restored
  decision inputs equal the recorded ones.
The keys models_rerun_bitwise / representative_readers_bitwise are empty by construction (fsa.custody aggregates them)."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from . import common as C


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(out):
    import sklearn
    here = sys.version.split()[0]
    fit_pass = here == '3.14.3'
    rec = {'python': here, 'sklearn': sklearn.__version__, 'labels_read': False, 'models_rerun_bitwise': {},
           'models_not_rerun_in_this_pass': {}, 'representative_readers_bitwise': {}, 'readers_judged_in_other_pass': {}}
    errs = []
    # 1. source pins
    from ohn import common as OC
    adm = OC.admission()['copies']
    bad = [r for r, v in adm.items() if C.sha(OC.MIRROR / r) != v['sha256']]
    rec['admitted_copies_checked'], rec['admitted_copies_mismatched'] = len(adm), bad[:20]
    sa = json.loads((C.RESULTS / 'SOURCE_ADMISSION.json').read_text())
    src_bad = []
    for key, v in sa['frc_private_artefacts'].items():
        if key.startswith('frc/runs/'):
            d = C.FRC_RUNS / '/'.join(key.split('/')[2:])
            if _sha(d / 'MODEL.pkl') != v['MODEL.pkl'] or _sha(d / 'RELEASE.npz') != v['RELEASE.npz']:
                src_bad.append(key)
        elif key.startswith('frc/fa_cache/'):
            if _sha(C.FRC_FA_CACHE / '/'.join(key.split('/')[2:])) != v['sha256']:
                src_bad.append(key)
        elif key.startswith('frc/results/'):
            if _sha(C.FRC_RESULTS / key[len('frc/results/'):]) != v['sha256']:
                src_bad.append(key)
    for key, v in sa['source_private_artefacts'].items():
        if key.startswith('fa_cache/'):
            if _sha(C.NRM_FA_CACHE / key[len('fa_cache/'):]) != v['sha256']:
                src_bad.append(key)
    rec['source_artefacts_checked'] = len(sa['frc_private_artefacts']) + len(sa['source_private_artefacts'])
    rec['source_artefacts_mismatched'] = src_bad
    # 2. this campaign's private outputs against their records
    from .fa import solve as SV
    priv_bad = {}
    for v in C.VERSIONS:
        for rel, field, name in ((f'replay/P1_REPLAY_V{v}.json', 'per_person_file_sha256', f'P1_REPLAY_V{v}.npz'),
                                 (f'nested/NESTED_SOLVERS_V{v}.json', 'per_person_file_sha256', f'NESTED_SOLVERS_V{v}.npz')):
            want = json.loads((C.RESULTS / rel).read_text())[field]
            if _sha(SV._priv(name)) != want:
                priv_bad[name] = 'hash differs from its record'
    rs = json.loads((C.RESULTS / 'decisions' / 'ROUTE_S_INPUT.json').read_text())
    for name, want in rs['private_file_sha256'].items():
        if _sha(SV._priv(name)) != want:
            priv_bad[name] = 'hash differs from ROUTE_S_INPUT.json'
    rec['private_outputs_checked'] = 4 + len(rs['private_file_sha256'])
    rec['private_outputs_mismatched'] = priv_bad
    # 3. re-runs from the backup alone (fit interpreter only: the cached teachers are pickled scikit-learn objects)
    if fit_pass:
        import pickle
        from frc.fa import route as FR
        from nrm.fa import prc as P
        reroute, objective_ok, replay_ok = {}, {}, {}
        mf = C.features('MODEL_FIT')
        for v in C.VERSIONS:
            E, Z, unit, names, row_sha = SV.statistics(v, mf)
            with open(SV._priv(f'SOURCE_BOOKS_V{v}.pkl'), 'rb') as f:
                books = pickle.load(f)
            with np.load(SV._priv(f'SOURCE_PARTITIONS_V{v}.npz')) as z:
                for key, bk in books.items():
                    k, l = FR.route_any(E, bk)
                    reroute[f'V{v}|{key}'] = bool(np.array_equal(k, z[f'{key}|k']) and np.array_equal(l, z[f'{key}|l']))
            src = json.loads((C.RESULTS / 'source' / f'SOURCE_CHECK_V{v}.json').read_text())
            for solver in ('original', 'SV1'):
                bk = books[f'PRC|{solver}|8.0']
                k, l = FR.route_any(E, bk)
                j = P.objective(E, bk, k, l)
                objective_ok[f'V{v}|PRC|{solver}|8.0'] = abs(j - src['runs'][f'PRC|{solver}']['per_strength']['8.0']['objective']) <= 1e-12
            rp = json.loads((C.RESULTS / 'replay' / f'P1_REPLAY_V{v}.json').read_text())
            with np.load(SV._priv(f'P1_REPLAY_V{v}.npz')) as z:
                worst = 0.0
                for nm, d in rp['nested_means'].items():
                    for key in ('occ_nll', 'inc_nll', 'occ_acc', 'inc_acc', 'occ_brier', 'inc_brier'):
                        if f'{nm}|{key}' in z.files and key in d:
                            worst = max(worst, abs(float(z[f'{nm}|{key}'].mean()) - d[key]))
                replay_ok[f'V{v}'] = worst
        from .fa import solver as SO
        rec['source_books_reroute_bitwise'] = reroute
        rec['fitted_objective_equal_record'] = objective_ok
        rec['replay_loss_means_max_abs_diff'] = replay_ok
        rec['solver_selftest_passed'] = bool(SO.selftest()['passed'])
        if not all(reroute.values()) or not all(objective_ok.values()) or max(replay_ok.values()) > 1e-12 \
                or not rec['solver_selftest_passed']:
            errs.append('a re-run from the backup differs')
    else:
        rec['reruns'] = 'skipped in this pass (the cached teachers are pickled under the fit interpreter); judged in the fit pass'
    # 4. decisions re-derived with the restored frozen rules
    from . import rules as RU
    adv = json.loads((C.RESULTS / 'decisions' / 'ADVANCEMENT.json').read_text())
    s_lab = RU.route_s(rs)['label']
    a_lab = RU.advancement(adv['input'])['label']
    rec['route_s_label_rederived'], rec['advancement_label_rederived'] = s_lab, a_lab
    if s_lab != 'S_PROXY_ONLY' or a_lab != adv['result']['label']:
        errs.append('a decision label differs when re-derived')
    if bad or src_bad or priv_bad:
        errs.append('hash mismatches')
    rec['errors'] = errs
    rec['all_ok'] = not errs
    Path(out).write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({'all_ok': rec['all_ok'], 'errors': errs}))
    return rec


if __name__ == '__main__':
    main(sys.argv[1])
