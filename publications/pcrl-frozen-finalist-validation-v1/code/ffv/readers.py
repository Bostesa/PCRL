"""Units FFV_READERS_S{s}: the enhanced, access-matched reader bank for one model seed (PROTOCOL.md section 7, READER_BANK_LOCK).

Readers try to infer the recorded SEX column of public-use ACS 2019 Florida person records from the released predictions only:
each recipient's own float64 probability vector and argmax decision (income view, occupation view) and the two aligned outputs
(linked pair). Fit on ATTACK_FIT (8,001 people, SEX), orientation and selection on ATTACK_VALIDATION (3,000 people, SEX);
DEV_REPORT is scored from released outputs alone (no DEV_REPORT label is opened here).

- Source bank (retained, not refitted): the 8-reader slate and its winners, admitted by hash; source candidates enter selection
  with their STORED oriented ATTACK_VALIDATION AUCs and are scored with their STORED DEV_REPORT scores.
- New recipes (same for every release): ET5, ET25, HGB15, HGB31, MLP64, MLP6432 on `features.rich_view`; CATI01, CATI1 on
  `features.CatiDesign`. Estimators are built here (never through the source name-dispatching fit_reader); their full
  get_params() and the library versions are asserted against READER_BANK_LOCK. FiniteScaler fitted on ATTACK_FIT. Model seed
  MODEL_SEED_BASE + s. No early stopping. Convergence (n_iter_, warnings) recorded; a non-converged fit is used as fitted, flagged.
- Identical single-recipient wires (exact hashes over ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT) are fitted once and aliased.
- Enhanced winner (reviewer A1): single view = [source 8, new 8]; pair = [source 8 pair, source 8 IGNORE_occupation, source 8
  IGNORE_income, new 8 pair, new 8 IGNORE_occupation, new 8 IGNORE_income]; max oriented ATTACK_VALIDATION AUC, exact ties to the
  earlier candidate, so the enhanced winner equals the original unless a new candidate is strictly higher. Near-tie flags (O4).
- Controls on the FINALIST wire per view (source definitions): shuffled-SEX null and planted SEX bit for OUT_HGB (source
  features; compared bitwise with the source's stored controls) and ET5, HGB31, MLP6432, CATI1.
- Reproduction subset (source code, bitwise against stored scores); reader-seed variation of seed-dependent pair winners (ET, MLP;
  seed-invariant winners are labelled so); auxiliary-sample variation (O5): pair winners refitted on 4 household-bootstrap
  resamples of ATTACK_FIT. Variation fits are never used for selection.
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from scm_bench.readers import SLATE, NULL_SEED, kinds, lookup_keys, fit_lookup, predict_lookup
from task_conditioned_representation.attacks import MODEL_SEED_BASE, FiniteScaler, fit_reader, predict_reader, orientation, orient
from task_conditioned_representation.inference import auc
from . import common as C
from . import features as F

SRC8 = tuple(n for n, _ in SLATE) + ('LOOKUP',)
NEW8 = ('ET5', 'ET25', 'HGB15', 'HGB31', 'MLP64', 'MLP6432', 'CATI01', 'CATI1')
CONTROL_READERS = ('OUT_HGB', 'ET5', 'HGB31', 'MLP6432', 'CATI1')
SEED_DEPENDENT = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
VIEWS = ('income', 'occupation', 'pair')
VAR_J = (1, 2, 3, 4)
AUX_SEED = 2026101805
NEAR_TIE = 0.010


def make_model(name, seed):
    if name in ('ET5', 'ET25'):
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5 if name == 'ET5' else 25, max_features=1.0, n_jobs=1, random_state=seed)
    if name in ('HGB15', 'HGB31'):
        return HistGradientBoostingClassifier(max_leaf_nodes=15 if name == 'HGB15' else 31, max_iter=200, learning_rate=0.05, min_samples_leaf=30,
                                              l2_regularization=1.0, early_stopping=False, random_state=seed)
    if name in ('MLP64', 'MLP6432'):
        return MLPClassifier(hidden_layer_sizes=(64,) if name == 'MLP64' else (64, 32), alpha=0.01, batch_size=256, max_iter=300,
                             early_stopping=False, random_state=seed)
    if name in ('CATI01', 'CATI1'):
        return LogisticRegression(C=0.1 if name == 'CATI01' else 1.0, solver='lbfgs', max_iter=5000)
    raise ValueError(name)


def params_of(name):
    """Full get_params() of a recipe with the model seed shown symbolically (asserted against READER_BANK_LOCK)."""
    p = make_model(name, 0).get_params()
    return {k: ('MODEL_SEED_BASE+s' if k == 'random_state' and name[:3] in ('ET5', 'ET2', 'HGB', 'MLP') else repr(v)) for k, v in sorted(p.items())}


def versions():
    import scipy
    import sklearn
    return {'sklearn': sklearn.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__}


def design_for(name, view, fit_inc, fit_occ):
    """Fitted (on ATTACK_FIT outputs only) feature map for a new recipe: (transform(inc, occ), description, design object)."""
    if name.startswith('CATI'):
        d = F.CatiDesign(view).fit(fit_inc, fit_occ)
        return d.transform, d.describe(), d
    return (lambda inc, occ: F.rich_view(view, inc, occ)), {'view': view, 'features': 'rich: p, log max(p, 1e-300), one-hot argmax'}, None


def fit_new(name, X, y, seed):
    sc = FiniteScaler().fit(X); m = make_model(name, seed)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m.fit(sc.transform(X), y)
    names = sorted({type(x.message).__name__ for x in w})
    n_iter = int(np.max(getattr(m, 'n_iter_', [-1])))
    conv = {'n_iter': n_iter, 'warnings': names,
            'converged': not ('ConvergenceWarning' in names or (name.startswith('MLP') and n_iter >= 300) or (name.startswith('CATI') and n_iter >= 5000))}
    return {'model': m, 'scaler': sc}, conv


def predict_new(rec, X):
    if not np.array_equal(rec['model'].classes_, [0, 1]):
        raise ValueError('reader class order differs')
    return rec['model'].predict_proba(rec['scaler'].transform(X))[:, 1].astype(np.float64)


def pick(cands):
    """cands: [(candidate name, oriented ATTACK_VALIDATION AUC, score key, origin)]; max AUC, exact ties to the earlier candidate."""
    win = max(range(len(cands)), key=lambda i: (cands[i][1], -i))
    others = sorted((c[1] for i, c in enumerate(cands) if i != win), reverse=True)
    gap = float(cands[win][1] - others[0]) if others else None
    return {'candidate': cands[win][0], 'reader_key': cands[win][2], 'origin': cands[win][3], 'attack_validation_auc': float(cands[win][1]),
            'n_candidates': len(cands), 'candidate_order': [c[0] + ('' if c[3] == 'source' else ' [new]') for c in cands],
            'gap_to_runner_up': gap, 'AV_NEAR_TIE': bool(gap is not None and gap < NEAR_TIE)}


def invented_inputs(seed):
    """Invented arrays with the real shapes (no real data): a 6-level income code, continuous occupation vectors, invented SEX."""
    rng = np.random.default_rng(99000 + seed)
    n = {'ATTACK_FIT': 8001, 'ATTACK_VALIDATION': 3000, 'DEV_REPORT': 5943}
    sex, bank = {}, {}
    levels = np.array([[0.89, 0.11], [0.76, 0.24], [0.57, 0.43], [0.45, 0.55], [0.31, 0.69], [0.13, 0.87]])
    units = {}
    for r, k in n.items():
        z = rng.normal(size=(k, 5)); s = (rng.random(k) < 0.47).astype(np.int64)
        z[:, 1] += 0.6 * s; z[:, 3] -= 0.4 * s
        occ = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        code = np.clip(np.round(2.5 + z[:, 0] + 0.3 * s + rng.normal(scale=0.7, size=k)), 0, 5).astype(int)
        inc_f = levels[code]
        if r == 'DEV_REPORT':
            inc_f[:3] = [0.5, 0.5]                                   # an unseen income vector exercises the unseen-category rule
        inc_c = 1 / (1 + np.exp(-(z[:, 0] + 0.2 * s))); inc_c = np.column_stack((1 - inc_c, inc_c))
        sex[r] = s; units[r] = np.arange(k) // 2
        for svc, inc, oc in (('FINALIST', inc_f, occ), ('H_LJ', inc_c, occ), ('NEURAL_RAW', inc_c ** 1.1 / (inc_c ** 1.1).sum(1, keepdims=True), occ ** 1.2 / (occ ** 1.2).sum(1, keepdims=True))):
            bank.setdefault(svc, {})[r] = {'income': inc, 'occupation': oc}
    ids = {f'row_id__{r}': np.arange(k) for r, k in n.items()}
    src = {'services': {}, 'marginal_banks': {}, 'controls': {v: {'shuffled_attack_validation_auc': 0.5, 'planted_attack_validation_auc': 1.0} for v in VIEWS}}
    sc_src = {}
    for svc in bank:
        src['services'][svc] = {'income_wire': f'{svc}|income', 'occupation_wire': 'FINALIST|occupation' if svc != 'NEURAL_RAW' else 'NEURAL_RAW|occupation',
                                'pair_readers': {}, 'winner': {}}
    for svc in bank:
        for name in SRC8:
            src['services'][svc]['pair_readers'][name] = {'attack_validation_auc': 0.55 + 0.001 * SRC8.index(name)}
            sc_src[f'PAIR__{svc}__{name}'] = rng.random(n['DEV_REPORT'])
        for t in ('income', 'occupation'):
            wid = src['services'][svc][f'{t}_wire']
            mb = src['marginal_banks'].setdefault(wid, {'readers': {}})
            for name in SRC8:
                mb['readers'][name] = {'attack_validation_auc': 0.52 + 0.001 * SRC8.index(name)}
                sc_src[f'MARG__{wid}__{name}'] = rng.random(n['DEV_REPORT'])
        for view in VIEWS:
            src['services'][svc]['winner'][view] = {'candidate': 'LOG_MLP', 'attack_validation_auc': 0.55,
                                                    'reader_id': f'PAIR__{svc}__LOG_MLP' if view == 'pair' else f"MARG__{src['services'][svc][view + '_wire']}__LOG_MLP"}
    for view in VIEWS:
        for k in ('SHUFFLED', 'PLANTED_0', 'PLANTED_1'):
            sc_src[f'CONTROL__{view}__{k}'] = rng.random(n['DEV_REPORT'])
    return bank, ids, sex, src, sc_src, units


def run(seed, out_dir, invented=False, services=None):
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh reader directory required')
    if not invented:
        for name in ('READER_BANK_LOCK.json', 'PROTOCOL.md', 'FROZEN_CANDIDATE_MANIFEST.json', 'SOURCE_ADMISSION.json'):
            if not C.committed(C.RESULTS / name):
                raise PermissionError(f'reader fits require the committed {name}')
        lock = json.loads((C.RESULTS / 'READER_BANK_LOCK.json').read_text())
        if lock['versions'] != versions():
            raise AssertionError(f'library versions differ from READER_BANK_LOCK: {versions()}')
        for name in NEW8:
            if lock['estimator_params'][name] != params_of(name):
                raise AssertionError(f'{name} parameters differ from READER_BANK_LOCK')
    out.mkdir(parents=True, mode=0o700)
    if invented:
        bank, ids, sex, src, sc_src, units = invented_inputs(seed)
        roster = list(bank)
        unitA = units['ATTACK_FIT']
    else:
        A = C.load_role('ATTACK_FIT', ('features', 'sex'), purpose=f'FFV enhanced reader fitting, seed {seed}')
        V = C.load_role('ATTACK_VALIDATION', ('features', 'sex'), purpose=f'FFV reader orientation and selection, seed {seed}')
        R = C.load_role('DEV_REPORT', ('features',))
        for X in (A, V, R):                       # readers never see Census inputs: only ids are kept from the exports
            del X['X'], X['feature_names']
        bank, ids, man = C.service_bank(seed)
        for r, X in (('ATTACK_FIT', A), ('ATTACK_VALIDATION', V), ('DEV_REPORT', R)):
            if not (np.array_equal(ids[f'row_id__{r}'], X['row_id']) and np.array_equal(ids[f'unit__{r}'], X['unit'])):
                raise AssertionError(f'bank row order differs from the {r} export')
        if len(A['row_id']) != 8001 or len(V['row_id']) != 3000 or len(R['row_id']) != 5943:
            raise AssertionError('role sizes differ from the source budgets (8,001 / 3,000 / 5,943)')
        sex = {'ATTACK_FIT': A['sex'], 'ATTACK_VALIDATION': V['sex']}
        src, sc_src = C.source_readers(seed)
        roster = list(C.BANK)
        unitA = A['unit']
    if services:
        roster = [s for s in roster if s in services]
    yA, yV = sex['ATTACK_FIT'], sex['ATTACK_VALIDATION']
    seed_model = MODEL_SEED_BASE + seed
    rec = {'schema': 'FFV1-enhanced-readers-v1', 'seed': seed, 'invented': invented, 'reader_model_seed': seed_model, 'versions': versions(),
           'roles': {'fit': 'ATTACK_FIT', 'select_and_orient': 'ATTACK_VALIDATION', 'score': 'DEV_REPORT (released outputs only; no label)'},
           'roster': roster, 'new_recipes': list(NEW8), 'source_slate': list(SRC8), 'marginal_banks': {}, 'pairs': {}, 'winners': {},
           'controls': {}, 'controls_out_hgb_vs_source_bitwise': {}, 'reproduction': {}, 'variation': {}, 'aux_sample_variation': {},
           'accounting': [], 'designs': {}}
    av, dev, models = {}, {}, {}

    def fit_score(name, X3, rid, seed_used=seed_model, y_fit=None, x_fit=None, save=True):
        t0 = time.process_time()
        m, conv = fit_new(name, X3[0] if x_fit is None else x_fit, yA if y_fit is None else y_fit, seed_used)
        raw_v = predict_new(m, X3[1]); sign = orientation(yV, raw_v)
        sv = orient(raw_v, sign); sd = orient(predict_new(m, X3[2]), sign)
        if save:
            mp = out / f'MODEL__{rid}.joblib'; joblib.dump(m, mp, compress=3); mp.chmod(0o600); models[rid] = C.sha(mp)
        rec['accounting'].append({'reader_id': rid, 'recipe': name, 'cpu_s': round(time.process_time() - t0, 3), **conv})
        return sv, sd, {'attack_validation_auc': float(auc(yV, sv)), 'orientation': int(sign), **conv}

    def views3(svc):
        return [(bank[svc][r]['income'], bank[svc][r]['occupation']) for r in C.ROLES3]

    def record_design(rid, desc, dobj, X3, trip):
        d = {**desc, 'n_columns': int(X3[0].shape[1])}
        if dobj is not None:
            d['unseen_income_vectors'] = {r: dobj.unseen(trip[i][0]) for i, r in enumerate(C.ROLES3)}
        rec['designs'][rid] = d

    # ---- unique single-recipient wires (exact hashes over the three reader roles), then pairs
    uniq = {}
    for svc in roster:
        for t in C.TASKS:
            h = tuple(C.wire_sha(bank[svc][r][t]) for r in C.ROLES3)
            uniq.setdefault((t, h), []).append(svc)
    wire_of = {}
    for (t, h), svcs in uniq.items():
        wid = f'{svcs[0]}|{t}'; mb = {'task': t, 'services_aliasing_this_wire': svcs, 'readers': {}}
        trip = views3(svcs[0])
        for name in NEW8:
            tf, desc, dobj = design_for(name, t, trip[0][0], trip[0][1])
            X3 = [tf(i, o) for i, o in trip]
            rid = f'MARG__{wid}__{name}'
            av[rid], dev[rid], mb['readers'][name] = fit_score(name, X3, rid)
            record_design(rid, desc, dobj, X3, trip)
        rec['marginal_banks'][wid] = mb
        for svc in svcs:
            wire_of[(svc, t)] = wid
    for svc in roster:
        trip = views3(svc); pr = {}
        for name in NEW8:
            tf, desc, dobj = design_for(name, 'pair', trip[0][0], trip[0][1])
            X3 = [tf(i, o) for i, o in trip]
            rid = f'PAIR__{svc}__{name}'
            av[rid], dev[rid], pr[name] = fit_score(name, X3, rid)
            record_design(rid, desc, dobj, X3, trip)
        rec['pairs'][svc] = pr

    # ---- winners: original (source record) and enhanced (source-first order; ATTACK_VALIDATION only)
    def src_marg(svc, t):
        wid = src['services'][svc][f'{t}_wire']
        return wid, src['marginal_banks'][wid]['readers']
    for svc in roster:
        w = {'original': {}, 'enhanced': {}}
        for view in ('income', 'occupation'):
            swid, sread = src_marg(svc, view); nwid = wire_of[(svc, view)]
            cands = [(n, sread[n]['attack_validation_auc'], f'SRC::MARG__{swid}__{n}', 'source') for n in SRC8]
            cands += [(n, rec['marginal_banks'][nwid]['readers'][n]['attack_validation_auc'], f'NEW::MARG__{nwid}__{n}', 'new') for n in NEW8]
            w['enhanced'][view] = pick(cands)
            ow = src['services'][svc]['winner'][view]
            w['original'][view] = {'candidate': ow['candidate'], 'reader_key': f"SRC::{ow['reader_id']}", 'origin': 'source',
                                   'attack_validation_auc': ow.get('attack_validation_auc')}
        sp = src['services'][svc]['pair_readers']
        cands = [(n, sp[n]['attack_validation_auc'], f'SRC::PAIR__{svc}__{n}', 'source') for n in SRC8]
        for drop, keep in (('occupation', 'income'), ('income', 'occupation')):
            swid, sread = src_marg(svc, keep)
            cands += [(f'IGNORE_{drop}__{n}', sread[n]['attack_validation_auc'], f'SRC::MARG__{swid}__{n}', 'source') for n in SRC8]
        cands += [(n, rec['pairs'][svc][n]['attack_validation_auc'], f'NEW::PAIR__{svc}__{n}', 'new') for n in NEW8]
        for drop, keep in (('occupation', 'income'), ('income', 'occupation')):
            nwid = wire_of[(svc, keep)]
            cands += [(f'IGNORE_{drop}__{n}', rec['marginal_banks'][nwid]['readers'][n]['attack_validation_auc'], f'NEW::MARG__{nwid}__{n}', 'new') for n in NEW8]
        w['enhanced']['pair'] = pick(cands)
        ow = src['services'][svc]['winner']['pair']
        w['original']['pair'] = {'candidate': ow['candidate'], 'reader_key': f"SRC::{ow['reader_id']}", 'origin': 'source',
                                 'attack_validation_auc': ow.get('attack_validation_auc')}
        for view in VIEWS:
            e, o = w['enhanced'][view], w['original'][view]
            e['gap_to_original_winner'] = None if o['attack_validation_auc'] is None else float(e['attack_validation_auc'] - o['attack_validation_auc'])
            e['same_as_original'] = e['reader_key'] == o['reader_key']
        rec['winners'][svc] = w

    # ---- controls on the FINALIST wire (source definitions), source OUT_HGB features and new recipes
    nominee = 'FINALIST' if 'FINALIST' in roster else roster[0]
    shuffled = np.random.default_rng(NULL_SEED + seed).permutation(yA)
    for view in VIEWS:
        trip = views3(nominee); crec = {}
        for name in CONTROL_READERS:
            t0 = time.process_time()
            if name == 'OUT_HGB':
                def feats(i, o):
                    a, b = kinds(i, 2)['out'], kinds(o, 5)['out']
                    return a if view == 'income' else b if view == 'occupation' else np.column_stack((a, b))
                X3 = [feats(i, o) for i, o in trip]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rs = fit_reader(X3[0], shuffled, 'OUT_HGB', seed_model)
                    rp = fit_reader(np.column_stack((X3[0], yA)), yA, 'OUT_HGB', seed_model)
                pf_s, pf_p = predict_reader, predict_reader
            else:
                tf, _, _ = design_for(name, view, trip[0][0], trip[0][1]); X3 = [tf(i, o) for i, o in trip]
                rs, _ = fit_new(name, X3[0], shuffled, seed_model)
                rp, _ = fit_new(name, np.column_stack((X3[0], yA)), yA, seed_model)
                pf_s, pf_p = predict_new, predict_new
            key = f'CONTROL__{view}__{name}'
            av[key + '__SHUFFLED'] = pf_s(rs, X3[1]); dev[key + '__SHUFFLED'] = pf_s(rs, X3[2])
            av[key + '__PLANTED'] = pf_p(rp, np.column_stack((X3[1], yV)))
            for bit in (0, 1):
                dev[f'{key}__PLANTED_{bit}'] = pf_p(rp, np.column_stack((X3[2], np.full(len(X3[2]), bit))))
            crec[name] = {'wire': nominee, 'shuffled_attack_validation_auc': float(auc(yV, av[key + '__SHUFFLED'])),
                          'planted_attack_validation_auc': float(auc(yV, av[key + '__PLANTED']))}
            rec['accounting'].append({'reader_id': key, 'recipe': name + ' x2 (control)', 'cpu_s': round(time.process_time() - t0, 3)})
        rec['controls'][view] = crec
        rec['controls_out_hgb_vs_source_bitwise'][view] = {k: bool(np.array_equal(dev[f'CONTROL__{view}__OUT_HGB__{k}'], sc_src[f'CONTROL__{view}__{k}']))
                                                           for k in ('SHUFFLED', 'PLANTED_0', 'PLANTED_1')}

    def refit_any(name, origin, view_used, trip, seed_used, rows=None, tag=None):
        """Refit one winner recipe (new or source) with a model seed and optionally ATTACK_FIT rows; returns oriented DEV scores."""
        rows = np.arange(len(yA)) if rows is None else rows
        if origin == 'new':
            tf, _, _ = design_for(name, view_used, trip[0][0][rows], trip[0][1][rows]); X3 = [tf(i, o) for i, o in trip]
            _, sd, info = fit_score(name, X3, tag, seed_used=seed_used, y_fit=yA[rows], x_fit=X3[0][rows], save=True)
            return sd, info
        kind = dict(SLATE)[name]
        def tab(i, o):
            a, b = kinds(i, 2)[kind], kinds(o, 5)[kind]
            return a if view_used == 'income' else b if view_used == 'occupation' else np.column_stack((a, b))
        X3 = [tab(i, o) for i, o in trip]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = fit_reader(X3[0][rows], yA[rows], name, seed_used)
        mp = out / f'MODEL__{tag}.joblib'; joblib.dump(m, mp, compress=3); mp.chmod(0o600); models[tag] = C.sha(mp)
        raw_v = predict_reader(m, X3[1]); sign = orientation(yV, raw_v)
        return orient(predict_reader(m, X3[2]), sign), {'attack_validation_auc': float(auc(yV, orient(raw_v, sign))), 'orientation': int(sign)}

    # ---- reproduction subset: refit source winners with the source code and compare with the stored scores bitwise
    if not invented:
        subset = [(svc, 'pair') for svc in ('FINALIST', 'NEURAL_RAW', 'H_LJ')] + [('FINALIST', 'occupation')]
        for svc, view in subset:
            ow = src['services'][svc]['winner'][view]; cand, rid = ow['candidate'], ow['reader_id']
            name = cand.split('__')[-1]
            view_used = ('income' if cand.startswith('IGNORE_occupation') else 'occupation') if cand.startswith('IGNORE_') else view
            trip = views3(svc)
            if name == 'LOOKUP':
                keys3 = [lookup_keys(i) if view_used == 'income' else lookup_keys(o) if view_used == 'occupation' else lookup_keys(i, o) for i, o in trip]
                m = fit_lookup(keys3[0], yA); raw_v = predict_lookup(m, keys3[1]); sign = orientation(yV, raw_v)
                s_rep = orient(predict_lookup(m, keys3[2]), sign)
            else:
                s_rep, _ = refit_any(name, 'source', view_used, trip, seed_model, tag=f'REPRO__{svc}__{view}')
            dev[f'REPRO__{svc}__{view}'] = s_rep
            rec['reproduction'][f'{svc}|{view}'] = {'source_winner': cand, 'source_reader_id': rid,
                                                    'bitwise_equal_to_stored_score': bool(np.array_equal(s_rep, sc_src[rid])),
                                                    'max_abs_diff': float(np.max(np.abs(s_rep - sc_src[rid])))}

    # ---- variation diagnostics of the enhanced pair winners (never used for selection)
    groups = np.unique(unitA)
    for svc in [s for s in ('FINALIST', 'NEURAL_RAW', 'H_LJ') if s in roster]:
        w = rec['winners'][svc]['enhanced']['pair']; cand = w['candidate']; name = cand.split('__')[-1]
        view_used = 'pair' if not cand.startswith('IGNORE_') else ('income' if cand.startswith('IGNORE_occupation') else 'occupation')
        trip = views3(svc)
        vrec = {'winner': cand, 'origin': w['origin'], 'view_used': view_used, 'refits': {},
                'seed_dependence': 'seed-dependent (ET / MLP)' if name in SEED_DEPENDENT else 'seed-invariant (no RNG component: HGB without early stopping or subsampling, LR, CATI, LOOKUP)'}
        if name in SEED_DEPENDENT:
            for j in VAR_J:
                sj = seed_model + 1000 * j
                sd, info = refit_any(name, w['origin'], view_used, trip, sj, tag=f'VAR__{svc}__j{j}')
                dev[f'VAR__{svc}__j{j}'] = sd; vrec['refits'][f'j{j}'] = {'model_seed': sj, **info}
        rec['variation'][svc] = vrec
        arec = {'winner': cand, 'origin': w['origin'], 'refits': {},
                'note': 'pair winner refitted on household-bootstrap resamples of ATTACK_FIT (same model seed); measures part of the auxiliary-sample component'}
        if name != 'LOOKUP':
            for j in VAR_J:
                rng = np.random.default_rng(AUX_SEED + 10 * seed + j)
                pickg = rng.choice(groups, size=len(groups), replace=True)
                pos = {g: np.flatnonzero(unitA == g) for g in np.unique(pickg)}
                rows = np.concatenate([pos[g] for g in pickg])
                sd, info = refit_any(name, w['origin'], view_used, trip, seed_model, rows=rows, tag=f'AUX__{svc}__j{j}')
                dev[f'AUX__{svc}__j{j}'] = sd; arec['refits'][f'j{j}'] = {'resample_seed': AUX_SEED + 10 * seed + j, **info}
        else:
            arec['note'] = 'LOOKUP winner: not refitted (deterministic table); not measured'
        rec['aux_sample_variation'][svc] = arec

    for name, arrs in (('AV_SCORES.npz', av), ('DEV_REPORT_SCORES.npz', dev)):
        p = out / name
        with p.open('xb') as f:
            np.savez(f, row_id=ids[f"row_id__{'ATTACK_VALIDATION' if name.startswith('AV') else 'DEV_REPORT'}"], **arrs)
        p.chmod(0o600)
    rec['av_scores_sha256'] = C.sha(out / 'AV_SCORES.npz'); rec['dev_report_scores_sha256'] = C.sha(out / 'DEV_REPORT_SCORES.npz')
    rec['model_files_sha256'] = models
    rec['fit_counts'] = {'new_marginal_fits': sum(len(m['readers']) for m in rec['marginal_banks'].values()), 'new_pair_fits': sum(len(p) for p in rec['pairs'].values()),
                         'control_fits': 2 * len(CONTROL_READERS) * len(VIEWS), 'reproduction_fits': len(rec['reproduction']),
                         'variation_fits': sum(len(v['refits']) for v in rec['variation'].values()),
                         'aux_sample_fits': sum(len(v['refits']) for v in rec['aux_sample_variation'].values())}
    rec['not_converged'] = [a['reader_id'] for a in rec['accounting'] if a.get('converged') is False]
    rec['DEV_REPORT_labels_opened'] = False
    with (out / 'READERS_ENH_SEED.json').open('x') as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--invented', action='store_true'); ap.add_argument('--services', nargs='*')
    a = ap.parse_args(); r = run(a.seed, a.out, a.invented, a.services)
    print(json.dumps({'seed': r['seed'], 'invented': r['invented'], 'fit_counts': r['fit_counts'], 'not_converged': r['not_converged'],
                      'winners': {s: {k: {v: (x['candidate'], round(x['attack_validation_auc'], 4) if x['attack_validation_auc'] is not None else None)
                                          for v, x in w[k].items()} for k in w} for s, w in r['winners'].items()},
                      'controls': r['controls'], 'controls_out_hgb_vs_source_bitwise': r['controls_out_hgb_vs_source_bitwise'],
                      'reproduction': r['reproduction'],
                      'cpu_by_recipe': {n: round(sum(a['cpu_s'] for a in r['accounting'] if a['recipe'] == n), 1) for n in NEW8}}, indent=1))
