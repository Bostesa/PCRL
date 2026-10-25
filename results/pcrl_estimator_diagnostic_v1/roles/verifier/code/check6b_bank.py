"""CHECK6 (part b): the reader bank (edg.bank) never lets SCORE labels or the appended support rows into a fit, an orientation or
a selection.

Own synthetic samples (own sampler and seeds; n = 2,000: FIT 1,000 / VALIDATION 400 / SCORE 600 by an own 10-part split) from the
bank laws RED|0.5|skewed (income, occupation, pair views) and REF|0.5|skewed (L16, occupation recipe view with the constant income
vector). Reader seed MODEL_SEED_BASE + 7100 + 999999 (no study seed). edg.bank.run_bank is NOT called (it writes into the study's
out/ and refits 30,000-person class limits); its per-row logic is reviewed statically in CHECK6_code_review.json.
Tests:
  T1 SCORE-label perturbation: every recipe's scores on all study rows, VALIDATION AUC, orientation, convergence, population AUC and
     every view's winner are bitwise unchanged.
  T2 support-row non-influence: each recipe refitted through edg.bank._fit_one on the sample rows ONLY gives the same scores on the
     sample rows (bitwise) and the same VALIDATION AUC / orientation as fit_bank_view with the support rows appended.
  T3 population AUC of each fitted recipe: brute-force exact AUC of its support-row (float) scores under the own exact masses
     equals fit_bank_view's pop (1e-12).
  T4 winners: an independent re-implementation of DRF1's rule (max oriented VALIDATION AUC, exact ties to the earlier candidate,
     candidate order source pair / source IGNORE_occupation / source IGNORE_income / new pair / new IGNORE_occupation / new
     IGNORE_income) picks the same candidate as edg.bank._winners_view; every oriented VALIDATION AUC is >= 1/2.
  T5 power of T1: perturbing VALIDATION labels instead DOES change VALIDATION AUCs (so T1 would have detected their use).
  T6 source text: edg.bank._src_tab and _fit_one equal drf/devreaders.py's functions (AST source segments, docstrings included)."""
import ast
import time

import numpy as np

import vlib as V

V.study_imports()
from edg import laws as L             # noqa: E402
from edg import bank as BK            # noqa: E402
from task_conditioned_representation.attacks import MODEL_SEED_BASE    # noqa: E402

T0 = V.utc_now()
VSEED = 202610251900
RS = MODEL_SEED_BASE + BK.SEED_OFFSET + 999999
rec = {'check': 6, 'name': 'reader_bank_no_test_label_part_b', 'reader_seed': RS, 'cells': {}}
fails = []


def own_pick(cands):
    best = 0
    for i in range(1, len(cands)):
        if cands[i][1] > cands[best][1]:
            best = i
    return cands[best][0]


def own_candidates(view, rd, rd_inc=None, rd_occ=None):
    if view != 'pair':
        return [(n, rd[n][1]['val_auc']) for n in BK.RECIPES]
    out = []
    for group in (BK.SRC8, BK.NEW8):
        out += [(n, rd[n][1]['val_auc']) for n in group]
        out += [(f'IGNORE_occupation__{n}', rd_inc[n][1]['val_auc']) for n in group]
        out += [(f'IGNORE_income__{n}', rd_occ[n][1]['val_auc']) for n in group]
    return out


for ci, (fam, pi, occ, views) in enumerate((('RED', 0.5, 'skewed', {'income': ('income', 'income'), 'occupation': ('occupation', 'occupation'),
                                                                   'pair': ('pair', 'pair')}),
                                            ('REF', 0.5, 'skewed', {'L16': ('occupation', 'L16')}))):
    t_cell = time.process_time()
    law = L.build(fam, pi, occ)
    rng = np.random.default_rng(np.random.SeedSequence(VSEED, spawn_key=(ci,)))
    n = 2000
    c, s = V.draw_people(law['p'], law['q'], n, rng)
    parts = rng.permutation(n) % 10
    fit, val, scm = parts <= 4, (parts == 5) | (parts == 6), parts >= 7
    s_score = s.copy()
    s_score[scm] = rng.integers(0, 2, int(scm.sum()))
    s_val = s.copy()
    s_val[val] = rng.integers(0, 2, int(val.sum()))
    rds, popss, rds2, popss2 = {}, {}, {}, {}
    out = {'n': n, 'fit': int(fit.sum()), 'validation': int(val.sum()), 'score': int(scm.sum()), 'views': {}}
    for bview, (rview, lview) in views.items():
        rds[bview], popss[bview], _ = BK.fit_bank_view(law, c, s, fit, val, rview, lview, RS)
        rds2[bview], popss2[bview], _ = BK.fit_bank_view(law, c, s_score, fit, val, rview, lview, RS)
        # T1
        t1 = all(np.array_equal(rds[bview][r][0], rds2[bview][r][0]) and rds[bview][r][1] == rds2[bview][r][1]
                 and popss[bview][r] == popss2[bview][r] for r in BK.RECIPES)
        # T2 / T3: own calls of _fit_one with and without the appended support rows
        inc_a, occ_a, Vw = BK._arrays(law, c, rview, lview)
        K = Vw['K']
        wid, K2, m1, m0 = V.study_law_exact_view(law, lview)
        own2st = np.full(K2, -1)
        own2st[wid] = Vw['inv']
        m1x, m0x = [None] * K, [None] * K
        for w in range(K2):
            m1x[own2st[w]], m0x[own2st[w]] = m1[w], m0[w]
        pad = np.zeros(K, dtype=bool)
        sex_a = np.concatenate((s, np.zeros(K, dtype=s.dtype)))
        t2, t3_max, t2_maxdiff = True, 0.0, 0.0
        for r in BK.RECIPES:
            sc_app, info_app = BK._fit_one(r, rview, inc_a, occ_a, sex_a, np.concatenate((fit, pad)), np.concatenate((val, pad)), RS)
            sc_smp, info_smp = BK._fit_one(r, rview, inc_a[:n], occ_a[:n], s, fit, val, RS)
            same = np.array_equal(sc_smp, sc_app[:n]) and info_smp == info_app and np.array_equal(sc_app[:n], rds[bview][r][0])
            t2_maxdiff = max(t2_maxdiff, float(np.abs(sc_smp - sc_app[:n]).max()))
            t2 &= bool(same)
            pop_b = V.pop_auc_brute([float(x) for x in sc_app[n:]], m1x, m0x)
            t3_max = max(t3_max, abs(popss[bview][r] - float(pop_b)))
        out['views'][bview] = {'T1_score_label_perturbation_no_change': bool(t1), 'T2_support_rows_no_influence_bitwise': bool(t2),
                               'T2_max_abs_score_diff': t2_maxdiff, 'T3_max_abs_pop_vs_bruteforce': t3_max,
                               'oriented_val_auc_min': min(rds[bview][r][1]['val_auc'] for r in BK.RECIPES),
                               'K': K}
        if not (t1 and t2 and t3_max <= 1e-12 and out['views'][bview]['oriented_val_auc_min'] >= 0.5):
            fails.append(f'{fam} {bview} T1-T3')
    # T4 winners (VALIDATION only) and T1 for winners
    for bview, (rview, lview) in views.items():
        if rview == 'pair':
            w = BK._winners_view(rds[bview], 'pair', rds['income'], rds['occupation'])
            w2 = BK._winners_view(rds2[bview], 'pair', rds2['income'], rds2['occupation'])
            mine = own_pick(own_candidates('pair', rds[bview], rds['income'], rds['occupation']))
            src = {'pair': rds[bview], 'income': rds['income'], 'occupation': rds['occupation']}
        else:
            w = BK._winners_view(rds[bview], rview)
            w2 = BK._winners_view(rds2[bview], rview)
            mine = own_pick(own_candidates(rview, rds[bview]))
            src = {rview: rds[bview]}
        v_, nm = w['reader_key'].split('__', 1)
        s_win = src[v_][nm][0]
        from task_conditioned_representation.inference import auc as INF_AUC
        score_auc = INF_AUC(s[scm], s_win[scm])
        score_auc_brute = V.sample_auc_brute(s[scm], score=s_win[scm])
        out['views'][bview].update({'T4_winner': w['candidate'], 'T4_own_rule_winner': mine, 'T4_same': w['candidate'] == mine,
                                    'T1_winner_unchanged_under_score_perturbation': w2['candidate'] == w['candidate']
                                    and w2['attack_validation_auc'] == w['attack_validation_auc'],
                                    'winner_val_auc': w['attack_validation_auc'], 'winner_score_auc': score_auc,
                                    'winner_score_auc_bruteforce_minus_source': float(score_auc_brute) - score_auc,
                                    'n_candidates': w['n_candidates']})
        if not (out['views'][bview]['T4_same'] and out['views'][bview]['T1_winner_unchanged_under_score_perturbation']
                and abs(out['views'][bview]['winner_score_auc_bruteforce_minus_source']) <= 1e-12):
            fails.append(f'{fam} {bview} T4')
    # T5 power: VALIDATION-label perturbation changes VALIDATION AUCs (first bank view only)
    bview, (rview, lview) = next(iter(views.items()))
    rd3, _, _ = BK.fit_bank_view(law, c, s_val, fit, val, rview, lview, RS)
    changed = sum(rd3[r][1]['val_auc'] != rds[bview][r][1]['val_auc'] for r in BK.RECIPES)
    out['T5_validation_perturbation_changed_val_auc_recipes'] = int(changed)
    if changed == 0:
        fails.append(f'{fam} T5 no power')
    out['cpu_s_cell'] = round(time.process_time() - t_cell, 2)
    rec['cells'][f'{fam}|{pi}|{occ}'] = out
    print(fam, out)

# T6 source text equality with DRF1's devreaders
def seg(path, name):
    src = open(path).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


dev = V.W / 'publications' / 'pcrl-direct-recovery-frontier-v1' / 'code' / 'drf' / 'devreaders.py'
bk = V.D / 'code' / 'edg' / 'bank.py'
t6 = {f: seg(dev, f) is not None and seg(dev, f) == seg(bk, f) for f in ('_src_tab', '_fit_one')}
rec['T6_text_equal_to_drf_devreaders'] = t6
if not all(t6.values()):
    fails.append('T6 text')
rec['failures'] = fails
rec['status'] = 'PASS' if not fails else 'FAIL'
print(rec['status'], fails, t6)
V.write_out('CHECK6b_bank.json', rec, started_utc=T0)
