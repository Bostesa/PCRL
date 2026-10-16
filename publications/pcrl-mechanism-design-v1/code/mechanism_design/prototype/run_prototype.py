"""Run the data-free MMP prototype and write PROTOTYPE_RESULTS.json.

PREREGISTRATION (below) was written, committed and logged before the first run. Finite laws are exact (fractions;
exhaustive enumeration of all 41,209 calibrated partition pairs per law). Continuous laws use fitted readers and are
approximate: their AUCs are estimates for that law and that slate, not population floors or real-data evidence.

Usage: python -m mechanism_design.prototype.run_prototype --out results/pcrl_mechanism_design_v1/prototype/PROTOTYPE_RESULTS.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from mechanism_design import no_real_data
from mechanism_design.prototype import demonstrations, mmp

PREREGISTRATION = {
    'mechanism': 'MMP: release m_i = (1 - pibar) q_i^0 + pibar q_i^1 (designer 1 proposal sha256 2b7273ae...)',
    'reference': 'exact SEX-blind posterior eta (a perfect U0); budgets cap = (acc 0.010, NLL 0.010, summed Brier 0.005) and '
                 'cap_plus_headroom = (0.016, 0.020, 0.009)',
    'strongest_baseline_analogue': 'exact global frontier: minimum pair Bayes AUC over all calibrated partition pairs feasible '
                                   'at the budget (weakly dominates every deterministic release in Bayes terms)',
    'expected': {
        'identities': 'In all four finite laws and both tasks: Brier(m) - Brier(eta) equals E[(pi - pibar)^2 ||q^1 - q^0||^2] '
                      'exactly; NLL excess equals E KL(eta || m) and is <= I(S; X).',
        'L_NEG': 'MMP releases are constant (no task information); MMP infeasible at both budgets; the global frontier at the '
                 'cap stays at pair AUC >= 0.70 (required disclosure); twin pair AUC = AUC*(S; X) (about 0.81).',
        'L_PROXY': 'MMP pair AUC = 1/2 exactly; MMP feasible at the cap (no decision changes); global frontier at the cap = 1/2; '
                   'local, S12 and S21 designs also reach 1/2 (no joint component needed); twin pair AUC about 0.77.',
        'L_PROXY_FLIP': 'MMP view still has pair AUC 1/2 but MMP fails the strict-cap accuracy clause (decision changes); the '
                        'global frontier at the cap stays above 1/2.',
        'L_MIXED': 'MMP pair AUC = AUC*(S; h) = 0.65 exactly (residual through the task-relevant proxy); MMP feasible at the '
                   'cap; twin pair AUC > 0.65; global frontier at the cap within 0.02 of 0.65 (at least 0.63).',
        'L_NOPROXY': 'No pure proxy: MMP separates every cell, so its pair Bayes AUC equals the twin\'s (= AUC*(S; X)); MMP gives no '
                     'Bayes-level reduction here (an honest limit, not a failure of the code).',
        'L_XOR': 'Coalition required disclosure: MMP single-recipient AUCs are 1/2 but its pair AUC equals AUC*(S; X) (the tasks jointly '
                 'need a SEX-revealing combination); no release feasible at the cap lowers the pair below 0.80.',
        'subgroup': 'Within-SEX occupation Brier increases are reported for every law; in L_CONT laws (pibar = 0.47) the women\'s '
                    'increase is expected to exceed the men\'s (verifier M5, constant-gap approximation); a reversal is reported as such.',
        'masking_comparison': 'In L_CONT laws a critic-style local linear eraser on each recipient\'s own wire, at the same summed-Brier '
                              'price as MMP per task, lowers single-recipient fitted AUCs more than the pair AUC relative to the twin '
                              '(masking signature); MMP\'s pair AUC is below the eraser\'s pair AUC (deletion beats masking at equal '
                              'price). If not, MMP section 3.6 is not supported in that law.',
        'L_CONT_PROXY': 'Bayes: MMP pair AUC (P(S | H, E, A)) below the twin (AUC*(S; X)). Fitted 7-reader slate, mean over 3 '
                        'replications: MMP pair AUC below the twin by >= 0.005.',
        'L_CONT_NOPROXY': 'Bayes: MMP equals the twin (both functions of the same fields; no merging). Fitted: MMP pair AUC '
                          'below the twin by >= 0.005 (the learnability part of the rationale). If not, that part is not '
                          'supported in this law and any real-data benefit would have to come from genuine merging.',
        'demonstrations': 'D1 decision-preserving bound, D2 true labels not a floor, D3 fitted below Bayes: all checks pass.',
    },
    'what_would_contradict_the_rationale': [
        'Brier identity or NLL bound fails anywhere',
        'MMP feasible in L_NEG, or MMP pair AUC not 1/2 in L_PROXY, or not 0.65 in L_MIXED',
        'MMP far (> 0.02) above the global frontier at its own feasible budget in L_PROXY or L_MIXED',
        'In L_CONT_NOPROXY the fitted pair AUC of MMP is not below the twin by >= 0.005',
        'In an L_CONT law MMP\'s fitted pair AUC is not below the matched-price local eraser\'s',
    ],
    'added_before_first_run_per_selection_conditions': 'L_NOPROXY, L_XOR, the matched-price local linear eraser, subgroup and w* reporting '
                                                       '(verifier fixes (i), (ii), (iv); SELECTION_DECISION.md)',
}


def sha_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def run(n_reps=3):
    t0 = time.time()
    out = {'preregistration': PREREGISTRATION, 'preregistration_sha256': sha_text(json.dumps(PREREGISTRATION, sort_keys=True))}
    laws = mmp.finite_laws()
    out['finite'] = {name: mmp.evaluate_finite(name, cells, law) for name, (cells, law) in laws.items()}
    cont = {}
    for name, with_proxy in (('L_CONT_PROXY', True), ('L_CONT_NOPROXY', False)):
        law = mmp.ContinuousLaw(with_proxy=with_proxy)
        reps = [mmp.fitted_recovery(law, rng_seed=20261016 + 100 * r + (1 if with_proxy else 2), replication=r) for r in range(n_reps)]
        summ = {'local_linear_fractions': [rp['local_linear_fractions'] for rp in reps]}
        for arm in ('twin_eta', 'mmp', 'local_linear'):
            summ[arm] = {v: float(np.mean([rp[arm][v]['test_auc'] for rp in reps])) for v in ('income', 'occupation', 'pair')}
            summ[arm]['bayes_pair_auc_mc'] = float(np.mean([rp[arm]['bayes_pair_auc_mc'] for rp in reps]))
            summ[arm]['pair_winners'] = [rp[arm]['pair']['winner'] for rp in reps]
        summ['pair_auc_difference_twin_minus_mmp'] = summ['twin_eta']['pair'] - summ['mmp']['pair']
        summ['pair_auc_difference_local_linear_minus_mmp'] = summ['local_linear']['pair'] - summ['mmp']['pair']
        summ['per_replication_pair_difference'] = [rp['twin_eta']['pair']['test_auc'] - rp['mmp']['pair']['test_auc'] for rp in reps]
        summ['per_replication_local_linear_minus_mmp'] = [rp['local_linear']['pair']['test_auc'] - rp['mmp']['pair']['test_auc'] for rp in reps]
        summ['masking_signature_local_linear'] = {
            'single_drop_income': summ['twin_eta']['income'] - summ['local_linear']['income'],
            'single_drop_occupation': summ['twin_eta']['occupation'] - summ['local_linear']['occupation'],
            'pair_drop': summ['twin_eta']['pair'] - summ['local_linear']['pair']}
        keys = list(reps[0]['utility_vs_eta_mc'])
        summ['utility_vs_eta_mc'] = {k: {m: (float(np.mean([rp['utility_vs_eta_mc'][k][m] for rp in reps])) if m != 'within_sex_brier_increase'
                                             else {g: float(np.mean([rp['utility_vs_eta_mc'][k][m][g] for rp in reps])) for g in ('male', 'female')})
                                         for m in ('brier_excess', 'nll_excess', 'accuracy_loss', 'within_sex_brier_increase')} for k in keys}
        cont[name] = {'replications': reps, 'summary': summ}
    out['continuous'] = cont
    out['demonstrations'] = {'D1': demonstrations.d1(), 'D2': demonstrations.d2(), 'D3': demonstrations.d3()}
    out['checks'] = checks(out)
    out['provenance'] = {'wall_s': round(time.time() - t0, 1), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    return out


def checks(o):
    f, c = o['finite'], o['continuous']
    ids = all(v['identities'][t]['brier_identity_holds_exactly'] and v['identities'][t]['nll_bound_holds'] and v['identities'][t]['nll_equals_expected_kl']
              for v in f.values() for t in ('task1', 'task2'))
    neg, prox, flip, mix = f['L_NEG'], f['L_PROXY'], f['L_PROXY_FLIP'], f['L_MIXED']
    out = {
        'identities_exact_in_all_finite_laws': ids,
        'L_NEG_mmp_infeasible_both_budgets': not any(neg['mmp']['feasible'].values()),
        'L_NEG_frontier_at_cap_at_least_0.70': neg['frontier']['cap'].get('feasible', False) and neg['frontier']['cap']['global_min_pair_auc'] >= 0.70,
        'L_PROXY_mmp_pair_auc_is_half': prox['mmp']['pair_auc_exact'] == '1/2',
        'L_PROXY_mmp_feasible_at_cap': prox['mmp']['feasible']['cap'],
        'L_PROXY_frontier_at_cap_is_half': prox['frontier']['cap'].get('feasible', False) and abs(prox['frontier']['cap']['global_min_pair_auc'] - 0.5) < 1e-12,
        'L_PROXY_local_s12_s21_reach_half': all(abs(prox['frontier']['cap'][k] - 0.5) < 1e-12 for k in ('local_pair_auc', 's12_pair_auc', 's21_pair_auc')),
        'L_PROXY_FLIP_mmp_view_half': flip['mmp']['pair_auc_exact'] == '1/2',
        'L_PROXY_FLIP_mmp_fails_cap_accuracy': flip['mmp']['excess_vs_eta']['task2']['accuracy_loss'] > 0.010 or flip['mmp']['excess_vs_eta']['task1']['accuracy_loss'] > 0.010,
        'L_PROXY_FLIP_frontier_at_cap_above_half': flip['frontier']['cap'].get('feasible', False) and flip['frontier']['cap']['global_min_pair_auc'] > 0.5 + 1e-12,
        'L_MIXED_mmp_pair_auc_0.65': abs(mix['mmp']['pair_auc'] - 0.65) < 1e-12,
        'L_MIXED_mmp_feasible_at_cap': mix['mmp']['feasible']['cap'],
        'L_MIXED_twin_above_0.65': mix['twin_eta']['pair_auc'] > 0.65 + 1e-12,
        'L_MIXED_frontier_at_cap_at_least_0.63': mix['frontier']['cap'].get('feasible', False) and mix['frontier']['cap']['global_min_pair_auc'] >= 0.63 - 1e-12,
        'L_CONT_PROXY_bayes_mmp_below_twin': c['L_CONT_PROXY']['summary']['mmp']['bayes_pair_auc_mc'] < c['L_CONT_PROXY']['summary']['twin_eta']['bayes_pair_auc_mc'],
        'L_CONT_PROXY_fitted_mmp_below_twin_by_0.005': c['L_CONT_PROXY']['summary']['pair_auc_difference_twin_minus_mmp'] >= 0.005,
        'L_CONT_NOPROXY_bayes_equal': abs(c['L_CONT_NOPROXY']['summary']['mmp']['bayes_pair_auc_mc'] - c['L_CONT_NOPROXY']['summary']['twin_eta']['bayes_pair_auc_mc']) < 1e-12,
        'L_CONT_NOPROXY_fitted_mmp_below_twin_by_0.005': c['L_CONT_NOPROXY']['summary']['pair_auc_difference_twin_minus_mmp'] >= 0.005,
        'L_NOPROXY_mmp_equals_twin_bayes': abs(f['L_NOPROXY']['mmp']['pair_auc'] - f['L_NOPROXY']['twin_eta']['pair_auc']) < 1e-12,
        'L_XOR_mmp_singles_half': abs(f['L_XOR']['mmp']['income_auc'] - 0.5) < 1e-12 and abs(f['L_XOR']['mmp']['occupation_auc'] - 0.5) < 1e-12,
        'L_XOR_mmp_pair_equals_auc_star_X': abs(f['L_XOR']['mmp']['pair_auc'] - f['L_XOR']['auc_star_X']) < 1e-12,
        'L_XOR_frontier_at_cap_at_least_0.80': f['L_XOR']['frontier']['cap'].get('feasible', False) and f['L_XOR']['frontier']['cap']['global_min_pair_auc'] >= 0.80,
        'L_CONT_women_increase_exceeds_men_occupation': all(
            c[n]['summary']['utility_vs_eta_mc']['mmp|task2']['within_sex_brier_increase']['female'] >
            c[n]['summary']['utility_vs_eta_mc']['mmp|task2']['within_sex_brier_increase']['male'] for n in ('L_CONT_PROXY', 'L_CONT_NOPROXY')),
        'L_CONT_masking_signature_singles_drop_more_than_pair': all(
            max(c[n]['summary']['masking_signature_local_linear']['single_drop_income'], c[n]['summary']['masking_signature_local_linear']['single_drop_occupation'])
            > c[n]['summary']['masking_signature_local_linear']['pair_drop'] for n in ('L_CONT_PROXY', 'L_CONT_NOPROXY')),
        'L_CONT_PROXY_mmp_pair_below_local_linear': c['L_CONT_PROXY']['summary']['pair_auc_difference_local_linear_minus_mmp'] > 0,
        'L_CONT_NOPROXY_mmp_pair_below_local_linear': c['L_CONT_NOPROXY']['summary']['pair_auc_difference_local_linear_minus_mmp'] > 0,
    }
    d = o['demonstrations']
    out['demonstrations_all_pass'] = (d['D1']['D1a']['all_preserving_at_least_decision_only'] and d['D1']['D1a']['some_preserving_strictly_above']
                                      and d['D1']['D1a']['some_changing_strictly_below_decision_only'] and d['D1']['D1b']['all_preserving_at_least_decision_only']
                                      and d['D2']['D2a']['label_exceeds_every_release'] and d['D2']['D2b']['release_exceeds_label_only']
                                      and d['D3']['fitted_below_bayes_both'])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args(argv)
    no_real_data.install()
    res = run(a.reps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res['checks'], indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
