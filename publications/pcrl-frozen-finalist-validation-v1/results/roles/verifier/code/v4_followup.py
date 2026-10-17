"""Unit VER_FOLLOWUP_V1 (check 4 record checks, and label-free figures needed for check 9). No label is read.

Public-use ACS 2019 Florida person records; readers infer the recorded SEX column from released predictions.
- Check 4: reader design widths against the access rules, wire aliasing, source-candidate values, the SEX-read log.
- Check 9 support: joint power at the sizes quoted in the decision text (own bootstrap-rescaling and MVN methods), the
  reader-matched descriptive ranges, the H_LJ seed-1 income winner change, the union count of occupation vectors shared by
  distinct inputs, and an own recomputation of the Part 3 pair-trained reconstruction diagnostic.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import vcommon as V
from ffv import common as C        # admitted loaders only
from v8_power import conds_of, holds, G_DEV


def main():
    out = {'unit': 'VER_FOLLOWUP_V1', 'labels_read': 'none'}
    # ---------------------------------------------------------------- check 4: designs, aliasing, source values, SEX-read log
    lock = V.jload(V.RES / 'READER_BANK_LOCK.json')
    acc = {}
    for s in V.SEEDS:
        enh = V.jload(V.RUNS / f'READERS_ENH_S{s}' / 'READERS_ENH_SEED.json')
        src_rec, _ = C.source_readers(s)
        bad = []
        for rid, d in enh['designs'].items():
            name = rid.split('__')[-1]
            view = d['view']
            n = d['n_columns']
            if not name.startswith('CATI'):
                exp = {'income': 6, 'occupation': 15, 'pair': 21}[view]
            else:
                inc = d.get('income') or {}
                if view == 'occupation':
                    exp = 9 + sum(len(k) + 2 for k in d['occupation_log_contrast_spline_knots'] if len(k) >= 2) + 6 + 5
                elif view == 'income':
                    exp = inc['n_indicators'] if inc['kind'] == 'categorical' else 6 + 4 + (len(d['income_logit_spline_knots']) + 2 if len(d['income_logit_spline_knots']) >= 2 else 0)
                else:
                    k = inc['n_indicators']
                    exp = k + (0 if inc['kind'] == 'categorical' else 4) + 9 + 9 * k
            if n != exp:
                bad.append([rid, n, exp])
        occ_bank = enh['marginal_banks']['FINALIST|occupation']['services_aliasing_this_wire']
        same_key = {tag: len({enh['winners'][x][tag]['occupation']['reader_key'] for x in ('FINALIST', 'H_LJ', 'B2TWIN')}) == 1 for tag in ('original', 'enhanced')}
        src_vals = []
        for svc, w in enh['winners'].items():
            for view, e in w['enhanced'].items():
                if e['origin'] == 'source':
                    k = e['reader_key'].split('::', 1)[1]
                    if k.startswith('PAIR__'):
                        sv = src_rec['services'][svc]['pair_readers'][k.split('__')[-1]]['attack_validation_auc']
                    else:
                        wid = k.split('__')[1]
                        sv = src_rec['marginal_banks'][wid]['readers'][k.split('__')[-1]]['attack_validation_auc']
                    src_vals.append(e['attack_validation_auc'] == sv)
        unseen = [k for k, d in enh['designs'].items() if d.get('unseen_income_vectors') and any(v for v in d['unseen_income_vectors'].values() if v)]
        acc[f'S{s}'] = {'design_width_mismatches': bad, 'n_designs': len(enh['designs']), 'occupation_wire_alias': occ_bank,
                        'aliased_occupation_winners_share_one_reader': same_key, 'source_origin_winner_auc_equals_source_record': all(src_vals),
                        'n_source_origin_winners': len(src_vals), 'designs_with_unseen_income_vectors': unseen,
                        'roles_in_record': enh['roles'], 'DEV_REPORT_labels_opened': enh['DEV_REPORT_labels_opened']}
    logl = [json.loads(l) for l in C.LABEL_LOG.read_text().splitlines() if l.strip()]
    acc['sex_read_log'] = {'n': len(logl), 'roles': sorted({e['role'] for e in logl}), 'fields': sorted({e['field'] for e in logl}),
                           'units': sorted({e['unit_id'] for e in logl})}
    out['check4'] = acc
    # ---------------------------------------------------------------- check 9: joint power at the quoted sizes (plug-in)
    mine = V.jload(V.VPRIV / 'dev_results.json')
    with np.load(V.VPRIV / 'dev_endpoint_draws.npz') as z:
        keys = [str(k) for k in z['keys']]
        D_all = z['draws'].copy()
    rng = np.random.default_rng(61018)
    pw = {}
    for SC in ('ENHANCED', 'ORIGINAL'):
        eps = mine['scorecards'][SC]['endpoints']
        cds = conds_of(eps)
        idx = {k.split('::', 1)[1]: i for i, k in enumerate(keys) if k.startswith(SC + '::')}
        D = D_all[:, [idx[k] for k, _, _ in cds]]
        Dc = D - D.mean(0)
        th = np.array([eps[k]['point'] for k, _, _ in cds])
        se = np.array([eps[k]['se'] for k, _, _ in cds])
        kinds = np.array([kd for _, kd, _ in cds])
        caps = np.array([m for _, _, m in cds])
        X = np.vstack([(rng.standard_normal((5000, len(Dc))) @ Dc) / np.sqrt(len(Dc) - 1) for _ in range(4)])
        for n in (9853, 10000, 17510):
            f = np.sqrt(G_DEV / n)
            for dfn in ('D1', 'D2'):
                pw[f'{SC}|{dfn}|n={n}'] = {'boot': float(holds(th[None, :] + f * Dc, se * f, kinds, caps, dfn)[0].mean()),
                                           'mvn': float(holds(th[None, :] + f * X, se * f, kinds, caps, dfn)[0].mean())}
    out['power_at_quoted_sizes_plugin'] = pw
    # ---------------------------------------------------------------- check 9: reader-matched descriptive ranges (pair; NEURAL_RAW, H_LJ)
    rm = mine['reader_matched_benefits']
    sel = {k: v for k, v in rm.items() if k.split('|')[1] in ('NEURAL_RAW', 'H_LJ')}
    nonlookup = {k: v for k, v in sel.items() if k.split('|')[2] != 'LOOKUP'}
    pos = {k: v['benefit'] for k, v in nonlookup.items() if v['benefit'] > 0}
    neg = {k: v['benefit'] for k, v in nonlookup.items() if v['benefit'] <= 0}
    per_cell = defaultdict(int)
    for k in pos:
        s, comp, _ = k.split('|')
        per_cell[f'{s}|{comp}'] += 1
    out['reader_matched'] = {'n_cells': len(sel), 'positive_per_seed_reference_excluding_LOOKUP': dict(per_cell),
                             'positive_range': [min(pos.values()), max(pos.values())] if pos else None,
                             'nonpositive': neg, 'LOOKUP': {k: v['benefit'] for k, v in sel.items() if k.endswith('|LOOKUP')},
                             'CONF_HGB': {k: v['benefit'] for k, v in sel.items() if k.endswith('|CONF_HGB')}}
    # ---------------------------------------------------------------- check 9: the H_LJ seed-1 income winner change
    w = mine['winners']['H_LJ|S1|income']
    rec = mine['recovery']
    out['H_LJ_S1_income'] = {'original': w['orig'], 'original_av_auc': w['orig_av_auc'], 'enhanced': w['enh'], 'enhanced_av_auc': w['enh_av_auc'],
                             'gap_to_runner_up': w['gap_to_runner_up'], 'dev_auc_original': rec['H_LJ|S1|income|orig']['auc'],
                             'dev_auc_enhanced': rec['H_LJ|S1|income|enh']['auc'],
                             'finalist_income_S1': [mine['winners']['FINALIST|S1|income']['orig'], mine['winners']['FINALIST|S1|income']['enh']]}
    # ---------------------------------------------------------------- check 9: union occupation vectors shared by distinct encoded inputs
    man = V.jload(V.RES / 'FROZEN_CANDIDATE_MANIFEST.json')
    shared = {}
    for s in V.SEEDS:
        with np.load(V.ADM / 'runs' / 'B' / 'B3CTL_UNCOUPLED_MIX50_C6A' / f'S{s}' / 'RELEASE.npz') as z:
            rel = {k: z[k].copy() for k in z.files}
        inp_of = defaultdict(set)
        for r in V.ROLES6:
            X = np.ascontiguousarray(C.load_role(r, ('features',))['X'])
            occ = np.ascontiguousarray(rel[f'{r}__occupation'])
            for a, b in zip(occ, X):
                inp_of[a.tobytes()].add(b.tobytes())
        sizes = [len(v) for v in inp_of.values() if len(v) > 1]
        shared[f'S{s}'] = {'vectors_shared_by_distinct_inputs': len(sizes), 'excess_distinct_inputs': int(sum(x - 1 for x in sizes)),
                           'size_histogram': {str(k): sizes.count(k) for k in sorted(set(sizes))}}
    out['union_shared_vectors'] = shared
    # ---------------------------------------------------------------- check 9: Part 3 pair-trained reconstruction (own code; no SEX)
    from sklearn.ensemble import HistGradientBoostingClassifier
    rc = {}
    for s in V.SEEDS:
        bank, _, _ = C.service_bank(s, ('FINALIST',))
        lv = np.array([[float.fromhex(x) for x in row] for row in man['per_seed'][f'S{s}']['per_level_income_probability_hex']])
        lvb = {lv[j].tobytes(): j for j in range(6)}
        b = bank['FINALIST']
        Xr = {r: V.rich(b[r]['occupation']) for r in V.ROLES3}
        yr = {r: np.array([lvb[row.tobytes()] for row in np.ascontiguousarray(b[r]['income'])]) for r in V.ROLES3}
        m = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0,
                                           early_stopping=False, random_state=V.MODEL_SEED_BASE + s).fit(Xr['ATTACK_FIT'], yr['ATTACK_FIT'])
        maj = np.bincount(yr['ATTACK_FIT']).argmax()
        seen = {row.tobytes() for row in np.ascontiguousarray(b['ATTACK_FIT']['occupation'])}
        rc[f'S{s}'] = {r: {'level_accuracy': float((m.predict(Xr[r]) == yr[r]).mean()), 'majority_level_accuracy': float((yr[r] == maj).mean()),
                           'share_occupation_vectors_seen_in_attack_fit': float(np.mean([row.tobytes() in seen for row in np.ascontiguousarray(b[r]['occupation'])]))}
                       for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')}
    rep = V.jload(V.RES / 'OUTPUT_MAPPING_RESULTS.json')
    out['reconstruction'] = {'own': rc, 'reported': {s: {r: v['pair_trained_reconstruction'][r] for r in ('ATTACK_VALIDATION', 'DEV_REPORT')} | {'ATTACK_FIT_in_sample': v['pair_trained_reconstruction']['ATTACK_FIT_in_sample_level_accuracy']}
                                                     for s, v in rep['per_seed'].items()}}
    V.jdump(V.VPRIV / 'check4_9_followup.json', out)
    print(json.dumps(out, indent=1, default=str)[:9000])


if __name__ == '__main__':
    main()
