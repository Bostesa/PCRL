"""Assemble PROTOTYPE_VERIFICATION.json (key recomputed numbers + check verdicts) from this role's unit outputs."""
from mechanism_design import no_real_data
no_real_data.install()
import json
from pathlib import Path
R = Path(__file__).resolve().parent.parent / 'results' / 'pcrl_mechanism_design_v1' / 'roles' / 'math_verifier'
fin = json.loads((R / 'V_PROTO_FINITE_V1.json').read_text())
con = json.loads((R / 'V_PROTO_CONT_V1.json').read_text())
hd1 = json.loads((R / 'V_PROTO_HD_V1.json').read_text())
hd2 = json.loads((R / 'V_PROTO_HD_V2.json').read_text())
al = json.loads((R / 'V_PROTO_HD_ALIGN_V1.json').read_text())
out = {'schema': 'math_verifier-prototype-verification-v1', 'units': ['V_PROTO_FINITE_V1', 'V_PROTO_CONT_V1 (failed: own key bug)', 'V_PROTO_CONT_V2',
       'V_PROTO_HD_V1', 'V_PROTO_HD_V2', 'V_PROTO_HD_ALIGN_V1'], 'finite': {}, 'continuous': {}, 'post_hoc_probe_P7': {}}
for name, r in fin['laws'].items():
    c = fin['comparison_with_coordinator'][name]
    out['finite'][name] = {'auc_star_X': r['auc_star_X'], 'twin': r['twin'], 'mmp': {k: r['mmp'][k] for k in ('pair', 'income', 'occupation', 'feasible')},
                           'mmp_excess_vs_eta': {t: r['tasks'][t]['excess_vs_eta'] for t in r['tasks']},
                           'brier_identity_exact': all(r['tasks'][t]['brier_identity_exact'] for t in r['tasks']),
                           'nll_equals_E_KL_and_bound': all(r['tasks'][t]['nll_minus_E_KL'] < 1e-12 and r['tasks'][t]['nll_bound_holds'] for t in r['tasks']),
                           'within_sex_increase': {t: r['tasks'][t]['within_sex_increase'] for t in r['tasks']},
                           'w_star': {t: [r['tasks'][t]['w_star'], r['tasks'][t]['price_at_pibar'], r['tasks'][t]['price_at_w_star']] for t in r['tasks']},
                           'frontier': {b: {k: v for k, v in r['frontier'][b].items()} for b in ('cap', 'cap_plus_headroom')},
                           'max_abs_diff_vs_coordinator_scalars': c['max_abs_diff_scalars'], 'frontier_vs_coordinator': c['frontier']}
for name in ('L_CONT_PROXY', 'L_CONT_NOPROXY'):
    p, s = con[name]['population'], con[name]['summary']
    out['continuous'][name] = {
        'population_exact': {'twin_bayes_AUC_star_X': p['AUC_star_X_twin_bayes'], 'mmp_bayes': p['mmp_bayes'],
                             'occupation': {k: p['occupation'][k] for k in ('brier_price', 'nll_price', 'accuracy_loss', 'within_sex_increase', 'decomposition', 'corr_pi_g2', 'constant_gap_M5_prediction')},
                             'income': {k: p['income'][k] for k in ('brier_price', 'nll_price', 'accuracy_loss', 'within_sex_increase', 'corr_pi_g2', 'constant_gap_M5_prediction')}},
        'fitted_mean_3_reps': {arm: {k: v for k, v in s[arm].items() if k != 'utility_vs_eta'} for arm in ('twin_eta', 'mmp', 'erase_matched', 'erase_full_occ')},
        'twin_minus_mmp_pair': {k: s[f'twin_minus_mmp_pair|{k}'] for k in ('LR', 'HGB', 'MLP')},
        'matched_t': s['matched_t_mean'], 'matched_eraser_invertibility_max_abs_clr_error': s['invertibility_max_abs_clr_error'],
        'masking_matched_eraser_drop_vs_twin': s['masking_erase_matched'], 'masking_full_occupation_eraser_drop_vs_twin': s['masking_erase_full_occ'],
        'full_occupation_eraser_price': s['erase_full_occ']['utility_vs_eta']['occupation']}
for k in ('draw0', 'draw1', 'draw2'):
    out['post_hoc_probe_P7'][k] = {'mmp_bayes_exact': hd1[k]['mmp_bayes_exact'], 'AUC_star_X_exact': hd1[k]['AUC_star_X_exact'],
                                   'twin_minus_mmp_LR_8001': hd1[k]['twin_minus_mmp|LR|n=8001'], 'twin_minus_mmp_HGB_8001': hd1[k]['twin_minus_mmp|HGB|n=8001'],
                                   'twin_minus_mmp_strong_HGB_1M': hd2[k]['twin_minus_mmp_strong'], 'mmp_strong_reader_gap_to_exact_bayes': hd2[k]['mmp_reader_gap_to_exact_bayes'],
                                   'occupation_brier_price': hd1[k]['occupation_brier_price'], 'alignment': al[k]}
out['check_verdicts'] = {
    'a_L_CONT_NOPROXY_fitted_mmp_below_twin_by_0.005': 'FAILED as registered; reproduced (twin - MMP 0.0004-0.0007); not a bug; low-power law (injective 3-field release, readers within ~0.001 of Bayes, pair ~ occupation-only)',
    'b_women_increase_exceeds_men_L_CONT': 'FAILED as registered; reproduced exactly by quadrature; not a bug; the registered expectation used the constant-gap M5 approximation, invalid here (corr(pi, ||g||^2) = -0.35 / -0.45 for occupation)',
    'c_local_eraser_masking_signature': 'FAILED as registered; reproduced; MIS-SPECIFIED COMPARATOR: the matched-price eraser (t < 1) is an exactly invertible affine map of the log-probability wire, so refit readers undo it; a t = 1 eraser shows the signature at 6-10x MMP price',
    'P7_post_hoc_probe': 'FAILED in 2 of 3 draws, opposite sign: MMP raises fitted (and near-Bayes) pair recovery by ~0.018-0.020 when field-mediated and direct SEX effects are anti-aligned; lowers it by 0.033 when aligned'}
(R / 'PROTOTYPE_VERIFICATION.json').write_text(json.dumps(out, indent=1))
print('WROTE', R / 'PROTOTYPE_VERIFICATION.json')
