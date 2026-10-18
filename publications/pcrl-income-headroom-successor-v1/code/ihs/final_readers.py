"""Units IHS_READERS_S{s}: the final evaluation readers over this campaign's compact final bank (PROTOCOL.md section 10).

Only after the committed FINALIST_LOCK, SERVICE_BANK_MANIFEST and READER_PLAN (the loader also refuses ATTACK SEX before the
FINALIST_LOCK). Readers estimate the recorded SEX column of public-use ACS 2019 Florida person records from the released
predictions only (each recipient's float64 probability vector and argmax decision; the aligned pair). Fit on ATTACK_FIT (8,001
people, SEX); orientation and every winner on ATTACK_VALIDATION (3,000 people, SEX); DEV_REPORT scored from released outputs
alone (no DEV_REPORT label is opened here). Model seed MODEL_SEED_BASE + s.

Recipes (the validation's enhanced common slate, READER_BANK_LOCK of research/pcrl-frozen-finalist-validation-v1):
- ORIGINAL bank (historical comparison): the source slate OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB
  (task_conditioned_representation.attacks.fit_reader on scm_bench.readers.kinds features) and the corrected exact-vector
  LOOKUP reader. Pair candidates: 8 pair readers + 16 ignore-one alternatives (the 8 income-view and 8 occupation-view readers).
- ENHANCED bank (the final evaluation standard): ORIGINAL plus ET5, ET25, HGB15, HGB31, MLP64, MLP6432 (released p, log p,
  one-hot decision) and CATI01, CATI1 (categorical-interaction design fitted label-free on ATTACK_FIT outputs), with their
  ignore-one alternatives; candidates in source-first order; maximum oriented ATTACK_VALIDATION AUC, exact ties to the earlier
  candidate; near ties (gap < 0.010) flagged.
- Single-view winners: ORIGINAL over the source 8; ENHANCED over source 8 then new 8.
- Identical single-recipient wires (exact hashes over ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT) are fitted once and aliased.
- Controls on the SUCCESSOR wires, per view: shuffled-SEX null (permutation seed NULL_SEED + s) and planted SEX bit, for
  OUT_HGB, ET5, HGB31, MLP6432, CATI1.
- Refit roster (frozen in READER_PLAN; never used for selection): the ENHANCED pair winners of SUCCESSOR, H_LJ and NEURAL_RAW
  refitted with 4 other model seeds when seed-dependent, and on 4 household-bootstrap resamples of ATTACK_FIT.
`--invented` runs the identical pipeline on invented arrays (no real data, no label) for the pre-lock dry run.
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

from scm_bench.readers import SLATE, NULL_SEED, kinds, lookup_keys, fit_lookup, predict_lookup
from task_conditioned_representation.attacks import MODEL_SEED_BASE, fit_reader, predict_reader, orientation, orient
from task_conditioned_representation.inference import auc
from ffv.readers import NEW8, CONTROL_READERS, SEED_DEPENDENT, make_model, params_of, versions, design_for, fit_new, predict_new, pick
from . import common as C

SRC8 = tuple(n for n, _ in SLATE) + ('LOOKUP',)
ROLES3 = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
VIEWS = ('income', 'occupation', 'pair')
VAR_J = (1, 2, 3, 4)
AUX_SEED = 2026101805
ROSTER_SERVICES = ('SUCCESSOR', 'H_LJ', 'NEURAL_RAW')


def load_bank(seed):
    man = json.loads((C.RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    p = C.RUNS / 'FINAL' / f'SERVICE_BANK_S{seed}.npz'
    if C.sha(p) != man['bank_files_sha256'][f'S{seed}']:
        raise PermissionError('service bank hash differs from SERVICE_BANK_MANIFEST')
    return C._bank_npz(p)


def invented_inputs(seed):
    rng = np.random.default_rng(77000 + seed)
    n = {'ATTACK_FIT': 1600, 'ATTACK_VALIDATION': 700, 'DEV_REPORT': 900}
    bank, sex, ids = {}, {}, {}
    for r, k in n.items():
        s = (rng.random(k) < 0.47).astype(np.int64); z = rng.normal(size=(k, 5)); z[:, 1] += 0.6 * s
        occ = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        inc_c = 1 / (1 + np.exp(-(z[:, 0] + 0.3 * s))); inc_c = np.column_stack((1 - inc_c, inc_c))
        levels = np.array([[0.8, 0.2], [0.55, 0.45], [0.3, 0.7]]); code = np.clip(np.round(1 + z[:, 0]), 0, 2).astype(int)
        sex[r] = s; ids[f'row_id__{r}'] = np.arange(k); ids[f'unit__{r}'] = np.arange(k) // 2
        for svc, inc, oc in (('SUCCESSOR', levels[code], occ), ('H_LJ', inc_c, occ), ('NEURAL_RAW', inc_c, occ ** 1.1 / (occ ** 1.1).sum(1, keepdims=True))):
            bank.setdefault(svc, {})[r] = {'income': inc, 'occupation': oc}
    return bank, ids, sex


def run(seed, out_dir, invented=False):
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh reader directory required')
    if not invented:
        for name in ('FINALIST_LOCK.json', 'SERVICE_BANK_MANIFEST.json', 'READER_PLAN.json'):
            if not C.committed(C.RESULTS / name):
                raise PermissionError(f'reader fits require the committed {name}')
        plan = json.loads((C.RESULTS / 'READER_PLAN.json').read_text())
        if plan['versions'] != versions():
            raise AssertionError(f'library versions differ from READER_PLAN: {versions()}')
        for name in NEW8:
            if plan['estimator_params'][name] != params_of(name):
                raise AssertionError(f'{name} parameters differ from READER_PLAN')
    out.mkdir(parents=True, mode=0o700)
    if invented:
        bank, ids, sex = invented_inputs(seed)
        unitA = ids['unit__ATTACK_FIT']
    else:
        A = C.load_role('ATTACK_FIT', ('features', 'sex'), purpose=f'IHS final reader fitting, seed {seed}')
        V = C.load_role('ATTACK_VALIDATION', ('features', 'sex'), purpose=f'IHS final reader orientation and selection, seed {seed}')
        R = C.load_role('DEV_REPORT', ('features',))
        for X in (A, V, R):                       # readers never see Census inputs: only ids are kept from the exports
            del X['X'], X['feature_names']
        bank, ids = load_bank(seed)
        for r, X in (('ATTACK_FIT', A), ('ATTACK_VALIDATION', V), ('DEV_REPORT', R)):
            if not (np.array_equal(ids[f'row_id__{r}'], X['row_id']) and np.array_equal(ids[f'unit__{r}'], X['unit'])):
                raise AssertionError(f'bank row order differs from the {r} export')
        if len(A['row_id']) != 8001 or len(V['row_id']) != 3000 or len(R['row_id']) != 5943:
            raise AssertionError('role sizes differ from the source budgets (8,001 / 3,000 / 5,943)')
        sex = {'ATTACK_FIT': A['sex'], 'ATTACK_VALIDATION': V['sex']}
        unitA = A['unit']
    roster = sorted(bank)
    yA, yV = sex['ATTACK_FIT'], sex['ATTACK_VALIDATION']
    seed_model = MODEL_SEED_BASE + seed
    rec = {'schema': 'IHS1-final-readers-v1', 'seed': seed, 'invented': invented, 'reader_model_seed': seed_model, 'versions': versions(),
           'roles': {'fit': 'ATTACK_FIT', 'select_and_orient': 'ATTACK_VALIDATION', 'score': 'DEV_REPORT (released outputs only; no label)'},
           'roster': roster, 'source_slate': list(SRC8), 'new_recipes': list(NEW8), 'marginal_banks': {}, 'pairs': {}, 'winners': {},
           'controls': {}, 'variation': {}, 'aux_sample_variation': {}, 'accounting': [], 'designs': {}}
    av, dev, models = {}, {}, {}

    def views3(svc):
        return [(bank[svc][r]['income'], bank[svc][r]['occupation']) for r in ROLES3]

    def save_model(rid, m):
        mp = out / f'MODEL__{rid}.joblib'; joblib.dump(m, mp, compress=3); mp.chmod(0o600); models[rid] = C.sha(mp)

    def src_tab(view, i, o, kind):
        a, b = kinds(i, 2)[kind], kinds(o, 5)[kind]
        return a if view == 'income' else b if view == 'occupation' else np.column_stack((a, b))

    def fit_src(name, view, trip, rid, seed_used=None, rows=None):
        t0 = time.process_time(); rows = np.arange(len(yA)) if rows is None else rows
        if name == 'LOOKUP':
            keys3 = [lookup_keys(i) if view == 'income' else lookup_keys(o) if view == 'occupation' else lookup_keys(i, o) for i, o in trip]
            k0 = [keys3[0][j] for j in rows]
            m = fit_lookup(k0, yA[rows]); raw_v = predict_lookup(m, keys3[1]); sign = orientation(yV, raw_v)
            sv, sd = orient(raw_v, sign), orient(predict_lookup(m, keys3[2]), sign)
        else:
            kind = dict(SLATE)[name]; X3 = [src_tab(view, i, o, kind) for i, o in trip]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = fit_reader(X3[0][rows], yA[rows], name, seed_model if seed_used is None else seed_used)
            raw_v = predict_reader(m, X3[1]); sign = orientation(yV, raw_v)
            sv, sd = orient(raw_v, sign), orient(predict_reader(m, X3[2]), sign)
        save_model(rid, m)
        rec['accounting'].append({'reader_id': rid, 'recipe': name, 'cpu_s': round(time.process_time() - t0, 3)})
        return sv, sd, {'attack_validation_auc': float(auc(yV, sv)), 'orientation': int(sign)}

    def fit_nw(name, view, trip, rid, seed_used=None, rows=None):
        t0 = time.process_time(); rows = np.arange(len(yA)) if rows is None else rows
        tf, desc, dobj = design_for(name, view, trip[0][0][rows], trip[0][1][rows])
        X3 = [tf(i, o) for i, o in trip]
        if name.startswith('CATI'):                                   # reviewer F2: keep the fitted (label-free) design for the verifier
            rec['designs'][rid] = {**desc, 'n_columns': int(X3[0].shape[1])}
        m, conv = fit_new(name, X3[0][rows], yA[rows], seed_model if seed_used is None else seed_used)
        raw_v = predict_new(m, X3[1]); sign = orientation(yV, raw_v)
        sv, sd = orient(raw_v, sign), orient(predict_new(m, X3[2]), sign)
        save_model(rid, m)
        rec['accounting'].append({'reader_id': rid, 'recipe': name, 'cpu_s': round(time.process_time() - t0, 3), **conv,
                                  'design_columns': int(X3[0].shape[1])})
        return sv, sd, {'attack_validation_auc': float(auc(yV, sv)), 'orientation': int(sign), **conv}

    # ---- unique single-recipient wires, then pairs (every service gets the same recipes and the same auxiliary sample)
    uniq = {}
    for svc in roster:
        for t in C.TASKS:
            h = tuple(C.wire_sha(bank[svc][r][t]) for r in ROLES3)
            uniq.setdefault((t, h), []).append(svc)
    wire_of = {}
    for (t, h), svcs in uniq.items():
        wid = f'{svcs[0]}|{t}'; mb = {'task': t, 'services_aliasing_this_wire': svcs, 'readers': {}}
        trip = views3(svcs[0])
        for name in SRC8:
            rid = f'MARG__{wid}__{name}'; av[rid], dev[rid], mb['readers'][name] = fit_src(name, t, trip, rid)
        for name in NEW8:
            rid = f'MARG__{wid}__{name}'; av[rid], dev[rid], mb['readers'][name] = fit_nw(name, t, trip, rid)
        rec['marginal_banks'][wid] = mb
        for svc in svcs:
            wire_of[(svc, t)] = wid
    for svc in roster:
        trip = views3(svc); pr = {}
        for name in SRC8:
            rid = f'PAIR__{svc}__{name}'; av[rid], dev[rid], pr[name] = fit_src(name, 'pair', trip, rid)
        for name in NEW8:
            rid = f'PAIR__{svc}__{name}'; av[rid], dev[rid], pr[name] = fit_nw(name, 'pair', trip, rid)
        rec['pairs'][svc] = pr

    # ---- winners (ATTACK_VALIDATION only)
    for svc in roster:
        w = {'original': {}, 'enhanced': {}}
        for view in ('income', 'occupation'):
            wid = wire_of[(svc, view)]; rd = rec['marginal_banks'][wid]['readers']
            src_c = [(n, rd[n]['attack_validation_auc'], f'MARG__{wid}__{n}', 'source') for n in SRC8]
            new_c = [(n, rd[n]['attack_validation_auc'], f'MARG__{wid}__{n}', 'new') for n in NEW8]
            w['original'][view] = pick(src_c); w['enhanced'][view] = pick(src_c + new_c)
        pr = rec['pairs'][svc]
        ig = {keep: rec['marginal_banks'][wire_of[(svc, keep)]]['readers'] for keep in ('income', 'occupation')}
        src_c = [(n, pr[n]['attack_validation_auc'], f'PAIR__{svc}__{n}', 'source') for n in SRC8]
        for drop, keep in (('occupation', 'income'), ('income', 'occupation')):
            src_c += [(f'IGNORE_{drop}__{n}', ig[keep][n]['attack_validation_auc'], f"MARG__{wire_of[(svc, keep)]}__{n}", 'source') for n in SRC8]
        new_c = [(n, pr[n]['attack_validation_auc'], f'PAIR__{svc}__{n}', 'new') for n in NEW8]
        for drop, keep in (('occupation', 'income'), ('income', 'occupation')):
            new_c += [(f'IGNORE_{drop}__{n}', ig[keep][n]['attack_validation_auc'], f"MARG__{wire_of[(svc, keep)]}__{n}", 'new') for n in NEW8]
        w['original']['pair'] = pick(src_c); w['enhanced']['pair'] = pick(src_c + new_c)
        for view in VIEWS:
            e, o = w['enhanced'][view], w['original'][view]
            e['same_as_original'] = e['reader_key'] == o['reader_key']
            e['gap_to_original_winner'] = float(e['attack_validation_auc'] - o['attack_validation_auc'])        # reviewer F1 (validation O4)
            e['AV_NEAR_TIE_TO_ORIGINAL'] = bool((not e['same_as_original']) and e['gap_to_original_winner'] < 0.010)
        rec['winners'][svc] = w

    # ---- real-data controls on the SUCCESSOR wires
    nominee = 'SUCCESSOR' if 'SUCCESSOR' in roster else roster[0]
    shuffled = np.random.default_rng(NULL_SEED + seed).permutation(yA)
    for view in VIEWS:
        trip = views3(nominee); crec = {}
        for name in CONTROL_READERS:
            t0 = time.process_time()
            if name == 'OUT_HGB':
                X3 = [src_tab(view, i, o, 'out') for i, o in trip]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rs = fit_reader(X3[0], shuffled, 'OUT_HGB', seed_model)
                    rp = fit_reader(np.column_stack((X3[0], yA)), yA, 'OUT_HGB', seed_model)
                pf = predict_reader
            else:
                tf, _, _ = design_for(name, view, trip[0][0], trip[0][1]); X3 = [tf(i, o) for i, o in trip]
                rs, _ = fit_new(name, X3[0], shuffled, seed_model)
                rp, _ = fit_new(name, np.column_stack((X3[0], yA)), yA, seed_model)
                pf = predict_new
            key = f'CONTROL__{view}__{name}'
            av[key + '__SHUFFLED'] = pf(rs, X3[1]); dev[key + '__SHUFFLED'] = pf(rs, X3[2])
            av[key + '__PLANTED'] = pf(rp, np.column_stack((X3[1], yV)))
            for bit in (0, 1):
                dev[f'{key}__PLANTED_{bit}'] = pf(rp, np.column_stack((X3[2], np.full(len(X3[2]), bit))))
            crec[name] = {'wire': nominee, 'shuffled_attack_validation_auc': float(auc(yV, av[key + '__SHUFFLED'])),
                          'planted_attack_validation_auc': float(auc(yV, av[key + '__PLANTED']))}
            rec['accounting'].append({'reader_id': key, 'recipe': name + ' x2 (control)', 'cpu_s': round(time.process_time() - t0, 3)})
        rec['controls'][view] = crec

    # ---- refit roster (variation diagnostics; never used for selection)
    groups = np.unique(unitA)
    for svc in [s for s in ROSTER_SERVICES if s in roster]:
        wv = rec['winners'][svc]['enhanced']['pair']; cand = wv['candidate']; name = cand.split('__')[-1]
        view_used = 'pair' if not cand.startswith('IGNORE_') else ('income' if cand.startswith('IGNORE_occupation') else 'occupation')
        trip = views3(svc); fitter = fit_src if name in SRC8 else fit_nw
        vrec = {'winner': cand, 'origin': wv['origin'], 'view_used': view_used, 'refits': {},
                'seed_dependence': 'seed-dependent' if name in SEED_DEPENDENT + ('OUT_MLP', 'LOG_MLP') else 'seed-invariant'}
        if vrec['seed_dependence'] == 'seed-dependent':
            for j in VAR_J:
                sj = seed_model + 1000 * j
                _, sd, info = fitter(name, view_used, trip, f'VAR__{svc}__j{j}', seed_used=sj)
                dev[f'VAR__{svc}__j{j}'] = sd; vrec['refits'][f'j{j}'] = {'model_seed': sj, **info}
        rec['variation'][svc] = vrec
        arec = {'winner': cand, 'origin': wv['origin'], 'refits': {}}
        for j in VAR_J:
            rng = np.random.default_rng(AUX_SEED + 10 * seed + j)
            pickg = rng.choice(groups, size=len(groups), replace=True)
            pos = {g: np.flatnonzero(unitA == g) for g in np.unique(pickg)}
            rows = np.concatenate([pos[g] for g in pickg])
            _, sd, info = fitter(name, view_used, trip, f'AUX__{svc}__j{j}', rows=rows)
            dev[f'AUX__{svc}__j{j}'] = sd; arec['refits'][f'j{j}'] = {'resample_seed': AUX_SEED + 10 * seed + j, **info}
        rec['aux_sample_variation'][svc] = arec

    for name, arrs in (('AV_SCORES.npz', av), ('DEV_REPORT_SCORES.npz', dev)):
        p = out / name
        with p.open('xb') as f:
            np.savez(f, row_id=ids[f"row_id__{'ATTACK_VALIDATION' if name.startswith('AV') else 'DEV_REPORT'}"], **arrs)
        p.chmod(0o600)
    rec['av_scores_sha256'] = C.sha(out / 'AV_SCORES.npz'); rec['dev_report_scores_sha256'] = C.sha(out / 'DEV_REPORT_SCORES.npz')
    rec['model_files_sha256'] = models
    rec['fit_counts'] = {'marginal_fits': sum(len(m['readers']) for m in rec['marginal_banks'].values()),
                         'pair_fits': sum(len(p) for p in rec['pairs'].values()), 'control_fits': 2 * len(CONTROL_READERS) * len(VIEWS),
                         'variation_fits': sum(len(v['refits']) for v in rec['variation'].values()),
                         'aux_sample_fits': sum(len(v['refits']) for v in rec['aux_sample_variation'].values())}
    rec['not_converged'] = [a['reader_id'] for a in rec['accounting'] if a.get('converged') is False]
    rec['DEV_REPORT_labels_opened'] = False
    with (out / 'READERS_SEED.json').open('x') as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--invented', action='store_true')
    a = ap.parse_args(); r = run(a.seed, a.out, a.invented)
    print(json.dumps({'seed': r['seed'], 'invented': r['invented'], 'fit_counts': r['fit_counts'], 'not_converged': r['not_converged'],
                      'winners': {s: {k: {v: (x['candidate'], round(x['attack_validation_auc'], 4), x['AV_NEAR_TIE']) for v, x in w[k].items()} for k in w}
                                  for s, w in r['winners'].items()},
                      'controls': r['controls']}, indent=1))
