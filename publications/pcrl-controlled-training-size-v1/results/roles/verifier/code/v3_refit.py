"""Verifier stage V3: model identities by independent refit, deterministic aliases, and the controls' S100 identity claims.

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records).

  a1 --fits ID1,ID2,...   For each A1_PO_16 size fit: MODEL_FIT occupation labels of the registered subset only (tsz.common.load,
                          operation 'verify', subset = the fit's subset; rows compared with the verifier's own subset reconstruction);
                          the verifier's own FIT-only dictionary (or the admitted dictionary for __ADM fits), own compact_po, own inner
                          folds (SHA-256('OHN_A_cv_S<v>|<household>') over ALL MODEL_FIT units, dealt into 5, restricted to the
                          subset), own T* (first argmin of the pooled inner-CV NLL computed by own staged softmax / NLL code), a
                          scikit-learn HistGradientBoostingClassifier constructed by the verifier with the A1_PO_16 constants
                          (ohn.fam_a.CONFIGS values: lr 0.025, 16 leaves, min leaf 100, l2 3, max_iter 1000, max_bins 255, no early
                          stopping, COW / MAR categorical, random_state = v), refit at T*, own softmax on all six roles. Compared
                          bitwise with the production RELEASE.npz (and, for S100, the source A1_PO_16 S<v> OCC.npz), T* and the
                          production inner curve; the dictionary with DICT.json.
  aliases                 label-free: one S100 fit per version (no ordering-specific S100 directory), aliases O0..O3 in FIT.json.
  controls                label-free: the S100 controls' identity claims (T_TREE teachers vs the admitted teacher files, TASK_REFIT
                          theta E20 vs the admitted Stage B file, MODEL_SELECT outputs vs the frozen bank) from the saved artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (the logged label-access layer; operation 'verify')

PRIV = ROOT / '.private' / 'pcrl_controlled_training_size_v1'
SIZE = PRIV / 'runs' / 'SIZE'
CONTROL = PRIV / 'runs' / 'CONTROL'
MIRROR = ROOT / '.private' / 'pcrl_income_headroom_successor_v1'
OHN_RUNS = ROOT / '.private' / 'pcrl_occupation_headroom_overnight_v1' / 'runs'
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
CAT_MASK = np.array([False, False, True, False, True, False])        # compact_po: AGEP, WKHP, COW*, SCHL, MAR*, POBP


def npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def parse(fid):
    parts = fid.split('__')
    sid, v = parts[1], int(parts[2][1:])
    kind = 'admitted' if len(parts) == 4 and parts[3] == 'ADM' else 'fit_only'
    return sid, v, kind


def hgb(T, v):
    from ohn import fam_a as FA                                # recipe constants only (values, not code)
    cfg = FA.CONFIGS['A1_PO_16']
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(loss='log_loss', learning_rate=float(cfg['lr']), max_iter=int(T), max_leaf_nodes=int(cfg['leaves']),
                                          min_samples_leaf=int(cfg['msl']), l2_regularization=float(cfg['l2']), max_bins=255,
                                          early_stopping=False, categorical_features=CAT_MASK, random_state=int(v)), cfg


def fit(Xr, y, T, v):
    m, cfg = hgb(T, v)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(Xr, y)
    if m.n_iter_ != T or not np.array_equal(m.classes_, np.arange(5)):
        raise AssertionError('unexpected iteration count or classes')
    return m, cfg


def staged_nll_sums(m, Xr, y):
    """Own per-iteration summed NLL (softmax of the staged raw scores, floor 1e-12)."""
    out = []
    idx = np.arange(len(y))
    for raw in m.staged_decision_function(Xr):
        p = V.softmax(raw)
        out.append(float(np.sum(-np.log(np.maximum(p[idx, y], V.NLL_FLOOR)))))
    return np.asarray(out)


def refit_a1(fid, masks_mine, feats):
    t0 = time.process_time()
    sid, v, kind = parse(fid)
    d = SIZE / fid
    rec = json.loads((d / 'FIT.json').read_text())
    for f in ('MODEL.pkl', 'DICT.json', 'RELEASE.npz', 'CURVE.npz'):
        if V.file_sha(d / f) != rec['files_sha256'][f]:
            raise PermissionError(f'{fid}/{f} differs from FIT.json')
    F = C.load('MODEL_FIT', ('features', 'occupation'), 'verify',
               f'independent verifier V3: refit {fid} (A1_PO_16 on subset {sid}) with own dictionary, folds, T* and softmax code', subset=sid)
    names = [str(x) for x in F['feature_names']]
    rows_equal = bool(np.array_equal(F['rows'], masks_mine[sid]))
    y = np.asarray(F['occupation'], dtype=np.int64)
    Xs = np.asarray(F['X'])
    if kind == 'fit_only':
        collapsed = V.fit_only_dictionary(Xs, names, 10)
    else:
        collapsed = {b: [] for b in V.BLOCKS}
    dic_prod = json.loads((d / 'DICT.json').read_text())
    dict_equal = all(sorted(collapsed[b]) == sorted(dic_prod['collapsed'][b]) for b in V.BLOCKS)
    Xr = V.compact_po(V.apply_dictionary(Xs, names, collapsed), names)
    unit_all = np.asarray(feats['MODEL_FIT']['unit'])
    inner = V.folds(unit_all, 5, f'OHN_A_cv_S{v}')[masks_mine[sid]]
    T_max = 1000
    cur = np.zeros(T_max)
    support = {}
    for j in sorted(set(inner.tolist())):
        tr, va = inner != j, inner == j
        support[j] = int(np.bincount(y[tr], minlength=5).min())
        m, cfg = fit(Xr[tr], y[tr], T_max, v)
        cur += staged_nll_sums(m, Xr[va], y[va])
    cur /= float(len(y))
    tstar = int(np.argmin(cur)) + 1
    srt = np.sort(cur)
    gap = float(srt[1] - srt[0])
    curve_prod = npz(d / 'CURVE.npz')['curve'][0]
    dep, _ = fit(Xr, y, tstar, v)
    res_roles, src_roles = {}, {}
    rel_prod = npz(d / 'RELEASE.npz')
    src = None
    if sid == 'S100':
        rec_src = json.loads((OHN_RUNS / 'A' / 'A1_PO_16' / f'S{v}' / 'OCC.json').read_text())
        if V.file_sha(OHN_RUNS / 'A' / 'A1_PO_16' / f'S{v}' / 'OCC.npz') != rec_src['occ_sha256']:
            raise PermissionError('source OCC.npz differs from its record')
        src = {k.split('__')[0]: x for k, x in npz(OHN_RUNS / 'A' / 'A1_PO_16' / f'S{v}' / 'OCC.npz').items()}
    for r in ROLES6:
        Xrr = V.compact_po(V.apply_dictionary(np.asarray(feats[r]['X']), names, collapsed), names)
        p = V.softmax(np.asarray(dep.decision_function(Xrr), dtype=np.float64))
        res_roles[r] = {'bitwise_vs_production': bool(np.array_equal(p, rel_prod[r])), 'max_abs_vs_production': float(np.max(np.abs(p - rel_prod[r]))),
                        'decisions_identical_production': bool(np.array_equal(p.argmax(1), rel_prod[r].argmax(1))), 'arr_sha_mine': V.arr_sha(p)}
        if src is not None:
            src_roles[r] = {'bitwise_vs_source': bool(p.shape == src[r].shape and np.array_equal(p, src[r])),
                            'max_abs_vs_source': float(np.max(np.abs(p - src[r]))) if p.shape == src[r].shape else None}
    out = {'fit_id': fid, 'subset': sid, 'version': v, 'dictionary_kind': kind, 'access_rows_equal_own_subset': rows_equal,
           'n_people': int(len(y)), 'n_households': int(len(np.unique(F['unit']))), 'subset_min_class': int(np.bincount(y, minlength=5).min()),
           'inner_training_min_class': min(support.values()),
           'own_dictionary_equals_production': dict_equal, 'n_collapsed_mine': {b: len(collapsed[b]) for b in V.BLOCKS},
           'T_star_mine': tstar, 'T_star_production': rec['T_star'], 'T_star_source': (rec_src.get('T_star') if src is not None else None),
           'argmin_gap_to_second_best_nll': gap,
           'inner_curve_max_abs_diff_vs_production': float(np.max(np.abs(cur - curve_prod))),
           'inner_fold_people': {str(j): int((inner == j).sum()) for j in sorted(set(inner.tolist()))},
           'roles_vs_production': res_roles, 'roles_vs_source_A1_PO_16': src_roles,
           'rows_sha256_mine': V.arr_sha(np.asarray(F['row_id'])), 'rows_sha256_production': rec['rows_sha256'],
           'cpu_s': round(time.process_time() - t0, 1)}
    ok = (rows_equal and dict_equal and tstar == rec['T_star'] and all(x['bitwise_vs_production'] for x in res_roles.values())
          and out['rows_sha256_mine'] == out['rows_sha256_production']
          and (src is None or (all(x['bitwise_vs_source'] for x in src_roles.values()) and tstar == rec_src.get('T_star'))))
    near = (rows_equal and dict_equal and tstar == rec['T_star'] and all(x['max_abs_vs_production'] <= 1e-12 and x['decisions_identical_production']
                                                                         for x in res_roles.values()))
    chk = {'id': None, 'kind': 'NUMERICAL_REPLAY',
           'what': f'{fid}: independent refit of the A1_PO_16 fit on the verifier-reconstructed rows of {sid} (own dictionary, folds, T* and '
                   'softmax; learner constructed from the recipe constants) vs the production fit (T*, inner NLL curve, DICT.json, RELEASE.npz '
                   'bitwise on six roles)' + (' and the source A1_PO_16 S<v> release (bitwise, T*)' if src is not None else ''),
           'inputs': V.inputs_sha([d / 'FIT.json', d / 'RELEASE.npz', d / 'DICT.json', d / 'CURVE.npz'] +
                                  ([OHN_RUNS / 'A' / 'A1_PO_16' / f'S{v}' / 'OCC.npz'] if src is not None else [])),
           'expected': {'T_star': rec['T_star'], 'release_sha256': rec['release_sha256'], 'dictionary_n_collapsed': dic_prod['n_collapsed']},
           'observed': out, 'status': 'PASS' if ok else ('ATTENTION' if near else 'FAIL')}
    return chk


def aliases_check(tag):
    dirs = sorted(p.name for p in SIZE.iterdir() if p.is_dir())
    s100 = [d for d in dirs if '__S100' in d]
    recs = {d: json.loads((SIZE / d / 'FIT.json').read_text()) for d in s100 if (SIZE / d / 'FIT.json').is_file()}
    ok = sorted(s100) == ['A1__S100__V0', 'A1__S100__V1'] and all(r['aliases'] == list(V.ORDERINGS) for r in recs.values())
    return {'id': f'V3.ALIAS.S100_ONE_FIT_PER_VERSION__{tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
            'what': 'deterministic alias: exactly one S100 A1_PO_16 fit per version (no ordering-specific S100 fit directory) and its FIT.json '
                    'aliases it to all four orderings',
            'inputs': V.inputs_sha([SIZE / d / 'FIT.json' for d in recs]), 'expected': ['A1__S100__V0', 'A1__S100__V1'],
            'observed': {'size_dirs': dirs, 's100_dirs': s100, 'aliases': {d: r['aliases'] for d, r in recs.items()}},
            'status': 'PASS' if ok else 'FAIL'}


def controls_check(tag):
    from ohn import common as OC
    out, inputs, ok = {}, [], True
    for v in (0, 1):
        d = CONTROL / f'S100__V{v}'
        o = {}
        for t in ('income', 'occupation'):
            mine = d / 'TEACHERS' / f'TEACHER_{t}.json'
            adm = MIRROR / 'admitted' / 'stage_a' / f'S{v}' / f'TREE_TEACHER_{t}.json'
            inputs += [mine, adm]
            o[f'teacher_{t}_file_identical_to_admitted'] = V.file_sha(mine) == V.file_sha(adm)
            pm, pa = json.loads(mine.read_text()), json.loads(adm.read_text())
            o[f'teacher_{t}_payload_identical'] = pm == pa
        tb = npz(d / 'STAGE_B' / 'TASK_REFIT__E20.npz')
        ta = npz(MIRROR / 'admitted' / 'stage_b' / f'S{v}' / 'TASK_REFIT__E20.npz')
        inputs += [d / 'STAGE_B' / 'TASK_REFIT__E20.npz', MIRROR / 'admitted' / 'stage_b' / f'S{v}' / 'TASK_REFIT__E20.npz']
        o['task_refit_E20_keys'] = sorted(tb)
        o['task_refit_E20_theta_bitwise'] = bool(sorted(tb) == sorted(ta) and all(np.array_equal(tb[k], ta[k]) for k in ta))
        bank, ids = OC.frozen_bank(v, roles=('MODEL_SELECT',))
        outs = npz(d / 'STAGE_B' / 'STAGE_B_OUTPUTS_FINAL.npz')
        teach = npz(d / 'TEACHERS' / 'TEACHER_OUTPUTS.npz')
        inputs += [d / 'STAGE_B' / 'STAGE_B_OUTPUTS_FINAL.npz', d / 'TEACHERS' / 'TEACHER_OUTPUTS.npz']
        o['stage_b_output_keys_sample'] = sorted(outs)[:6]
        o['teacher_output_keys_sample'] = sorted(teach)[:6]
        cand_tr = [k for k in outs if 'MODEL_SELECT' in k and 'occupation' in k and 'TASK_REFIT' in k] or \
                  [k for k in outs if 'MODEL_SELECT' in k and 'occupation' in k]
        cand_tt = [k for k in teach if 'MODEL_SELECT' in k and 'occupation' in k]
        o['task_refit_ms_occ_key'] = cand_tr; o['t_tree_ms_occ_key'] = cand_tt
        o['task_refit_MODEL_SELECT_occupation_equals_bank'] = bool(len(cand_tr) == 1 and np.array_equal(outs[cand_tr[0]], bank['TASK_REFIT']['MODEL_SELECT']['occupation']))
        o['t_tree_MODEL_SELECT_occupation_equals_bank'] = bool(len(cand_tt) == 1 and np.array_equal(teach[cand_tt[0]], bank['T_TREE']['MODEL_SELECT']['occupation']))
        rep = json.loads((d / 'REPRO.json').read_text())
        o['production_repro_claims'] = {'reproduced': rep.get('reproduced'), 'teachers_identical': {t: rep['teachers'][t]['identical_payload'] for t in ('income', 'occupation')}}
        out[v] = o
        ok = ok and all(o[k] for k in o if isinstance(o[k], bool))
    return {'id': f'V3.CONTROLS.S100_IDENTITY__{tag}', 'kind': 'NUMERICAL_REPLAY',
            'what': 'S100 control identity claims (label-free, from the saved artifacts): the refitted T_TREE income / occupation teachers equal '
                    'the admitted teacher files (file hash and parsed payload), the refitted TASK_REFIT theta at E20 equals the admitted Stage B '
                    'file bitwise, and the saved MODEL_SELECT occupation outputs of TASK_REFIT and T_TREE equal the frozen bank arrays bitwise',
            'inputs': V.inputs_sha(inputs), 'expected': 'all identical', 'observed': out, 'status': 'PASS' if ok else 'FAIL'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('a1', 'aliases', 'controls')); ap.add_argument('--fits', default=''); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    code = V.code_hashes(__file__, V.__file__)
    checks = []
    if a.mode == 'a1':
        feats = {r: C.features(r) for r in ROLES6}
        masks = V.subset_masks(feats['MODEL_FIT']['unit'])
        for fid in [x for x in a.fits.split(',') if x]:
            c = refit_a1(fid, masks, feats)
            c['id'] = f'V3.REFIT.{fid}__{a.tag}'
            c['code_sha256'] = code
            checks.append(c)
            print(json.dumps({c['id']: c['status'], 'T': [c['observed']['T_star_mine'], c['observed']['T_star_production']],
                              'gap': c['observed']['argmin_gap_to_second_best_nll'], 'cpu': c['observed']['cpu_s']}), flush=True)
    elif a.mode == 'aliases':
        checks.append(aliases_check(a.tag))
    else:
        checks.append(controls_check(a.tag))
    for c in checks:
        c.setdefault('code_sha256', code)
    import os
    uid = os.environ.get('TSZ_UNIT_ID') or 'local'
    V.write_record(f'V3_{a.mode.upper()}__{a.tag}__{uid}.json', {'checks': checks, 'utc': V.utc()})
    print(json.dumps(V.append_checks(checks)))
    print(json.dumps({c['id']: c['status'] for c in checks}))


if __name__ == '__main__':
    main()
