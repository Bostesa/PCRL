"""Encode the statistical reviewer's frozen design (roles/statistical_reviewer/CLAIMS_AND_ENDPOINTS_DRAFT.md, instantiated for the
registered-text finalist B3CTL_UNCOUPLED_MIX50_C6A) into ANALYSIS_LOCK.json. Write-once; no label read.

Bank roles: F = FINALIST; REF = NEURAL_RAW, H_LJ; TO = TASKONLY (matched task-only control, TASK_REFIT E20 bases);
SS = H_LJ (alias of a reference: sigma = 0); LOC = B2TWIN (B2CTL_UNCOUPLED_C6A, same coarsening with the LOCAL-only statistic:
iota = 1, J claimed); H_JJ and PUB_IBSI (COMPARATOR_INELIGIBLE, secondary only); T_TREE (marginal descriptive only).
"""
from __future__ import annotations

from . import analysis_lock as AL
from . import common as C

UTIL = [(t, r, m) for t in C.TASKS for r in ('U0', 'Ucal') for m in ('accuracy_loss', 'nll_excess', 'brier_excess')]
CAP = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}


def endpoints():
    E = []
    for s in C.SEEDS:
        P = 'P'
        for t, r, m in UTIL:
            E.append({'key': f'UTILITY|{t}|{r}|{m}|s{s}', 'type': 'utility', 'service': 'FINALIST', 'task': t, 'ref': r, 'metric': m, 'seed': s,
                      'kind': 'no_excess', 'margin': CAP[m], 'family': P, 'claim': 'E',
                      'flags': ['REPLAY_OF_PUBLISHED'] if t == 'occupation' else []})
        for g in (0, 1):
            E.append({'key': f'SUBGROUP|occupation_brier_excess_vs_U0|sex{g}|s{s}', 'type': 'guard', 'service': 'FINALIST', 'group': g, 'seed': s,
                      'kind': 'no_excess', 'margin': C.GUARD, 'family': P, 'claim': 'E', 'flags': ['REPLAY_OF_PUBLISHED']})
        for X in ('NEURAL_RAW', 'H_LJ'):
            E.append({'key': f'COMPETITIVE|pair_benefit_vs_{X}|s{s}', 'type': 'auc', 'a': X, 'b': 'FINALIST', 'view': 'pair', 'seed': s,
                      'kind': 'benefit', 'margin': 0.0, 'refs': [0.0, 0.020], 'family': P, 'claim': 'B', 'flags': []})
            for v in ('income', 'occupation'):
                E.append({'key': f'COMPETITIVE|{v}_worsening_vs_{X}|s{s}', 'type': 'auc', 'a': 'FINALIST', 'b': X, 'view': v, 'seed': s,
                          'kind': 'no_excess', 'margin': 0.010, 'family': P, 'claim': 'B',
                          'flags': (['IDENTICAL_RELEASE', 'REPLAY_OF_PUBLISHED'] if X == 'H_LJ' else ['REPLAY_OF_PUBLISHED']) if v == 'occupation' else []})
        for slot, X, claim in (('MECH_TO', 'TASKONLY', 'M'), ('JOINT', 'B2TWIN', 'J')):
            E.append({'key': f'{slot}|pair_benefit_vs_{X}|s{s}', 'type': 'auc', 'a': X, 'b': 'FINALIST', 'view': 'pair', 'seed': s,
                      'kind': 'benefit', 'margin': 0.0, 'refs': [0.0, 0.020], 'family': P, 'claim': claim, 'flags': []})
            for v in ('income', 'occupation'):
                E.append({'key': f'{slot}|{v}_worsening_vs_{X}|s{s}', 'type': 'auc', 'a': 'FINALIST', 'b': X, 'view': v, 'seed': s,
                          'kind': 'no_excess', 'margin': 0.010, 'family': P, 'claim': claim,
                          'flags': ['IDENTICAL_RELEASE'] if (X == 'B2TWIN' and v == 'occupation') else []})
        S = 'S'
        sec = [
            {'key': f'SEC|F_pair_minus_F_occupation|s{s}', 'type': 'auc_views', 'service': 'FINALIST', 'plus': 'pair', 'minus': 'occupation'},
            {'key': f'SEC|F_pair_minus_F_income|s{s}', 'type': 'auc_views', 'service': 'FINALIST', 'plus': 'pair', 'minus': 'income'},
            {'key': f'SEC|IBSI_NLLM_minus_F|pair|s{s}', 'type': 'auc', 'a': 'PUB_IBSI', 'b': 'FINALIST', 'view': 'pair', 'flags': ['COMPARATOR_INELIGIBLE']},
            {'key': f'SEC|income_nll_diff_vs_NEURAL_RAW|s{s}', 'type': 'loss_diff', 'a': 'FINALIST', 'b': 'NEURAL_RAW', 'task': 'income', 'metric': 'nll'},
            {'key': f'SEC|income_nll_diff_vs_H_LJ|s{s}', 'type': 'loss_diff', 'a': 'FINALIST', 'b': 'H_LJ', 'task': 'income', 'metric': 'nll'},
            {'key': f'SEC|occupation_nll_diff_vs_NEURAL_RAW|s{s}', 'type': 'loss_diff', 'a': 'FINALIST', 'b': 'NEURAL_RAW', 'task': 'occupation', 'metric': 'nll', 'flags': ['REPLAY_OF_PUBLISHED']},
            {'key': f'SEC|occupation_nll_diff_vs_H_LJ|s{s}', 'type': 'loss_diff', 'a': 'FINALIST', 'b': 'H_LJ', 'task': 'occupation', 'metric': 'nll', 'flags': ['IDENTICAL_RELEASE']},
            {'key': f'SEC|income_nll_diff_vs_TO|s{s}', 'type': 'loss_diff', 'a': 'FINALIST', 'b': 'TASKONLY', 'task': 'income', 'metric': 'nll'},
            {'key': f'SEC|income_nll_diff_vs_LOC|s{s}', 'type': 'loss_diff', 'a': 'FINALIST', 'b': 'B2TWIN', 'task': 'income', 'metric': 'nll'},
            {'key': f'SEC|H_JJ_minus_F|pair|s{s}', 'type': 'auc', 'a': 'H_JJ', 'b': 'FINALIST', 'view': 'pair', 'flags': ['COMPARATOR_INELIGIBLE']},
        ]
        for e in sec:
            E.append({**e, 'seed': s, 'kind': 'difference', 'margin': 0.0, 'family': S, 'claim': None, 'flags': e.get('flags', [])})
    return E


def claims():
    """Explicit conjunctions (draft section 2). Condition tests: 'point_le' (point <= margin: the E gates), 'clause' (L1 > m for
    benefit, U1 <= m for no_excess), 'clause_at_0020' (benefit L1 > 0.020), 'cv' (control validity), 'av' (availability)."""
    def keys(prefix):
        return [f'{prefix}|s{s}' for s in C.SEEDS]
    UG = [{'endpoint': f'UTILITY|{t}|{r}|{m}|s{s}', 'test': 'point_le', 'gate': 'UG'} for s in C.SEEDS for t, r, m in UTIL]
    SG = [{'endpoint': f'SUBGROUP|occupation_brier_excess_vs_U0|sex{g}|s{s}', 'test': 'point_le', 'gate': 'SG'} for s in C.SEEDS for g in (0, 1)]
    RG = [{'endpoint': f'COMPETITIVE|{v}_worsening_vs_{X}|s{s}', 'test': 'point_le', 'gate': 'RG'} for s in C.SEEDS for X in ('NEURAL_RAW', 'H_LJ') for v in ('income', 'occupation')]
    CV = [{'test': 'cv'}]
    BEN = lambda slot, X: [{'endpoint': f'{slot}|pair_benefit_vs_{X}|s{s}', 'test': 'clause'} for s in C.SEEDS]
    NW = lambda slot, X: [{'endpoint': f'{slot}|{v}_worsening_vs_{X}|s{s}', 'test': 'clause'} for s in C.SEEDS for v in ('income', 'occupation')]
    return [
        {'name': 'E', 'kind': 'gate', 'spends_alpha': False, 'conditions': UG + SG + RG,
         'success_label': 'DEV_POINT_ELIGIBLE', 'failure_label': 'DEV_POINT_INELIGIBLE', 'failure_order': ['UG', 'SG', 'RG']},
        {'name': 'B', 'kind': 'intersection_union', 'spends_alpha': True,
         'conditions': CV + [{'claim_ref': 'E'}] + BEN('COMPETITIVE', 'NEURAL_RAW') + BEN('COMPETITIVE', 'H_LJ') + NW('COMPETITIVE', 'NEURAL_RAW') + NW('COMPETITIVE', 'H_LJ'),
         'success_label': 'SUPERIOR_TO_ALL_ELIGIBLE',
         'failure_labels_in_order': ['CONTROL_FAILURE', 'FINALIST_FAILS_DEV_UTILITY_GATE', 'FINALIST_FAILS_DEV_SUBGROUP_POINT_GUARD',
                                     'FINALIST_FAILS_DEV_RECIPIENT_POINT_GATE', 'NOT_SUPERIOR_TO_ALL_ELIGIBLE'],
         'nested': {'name': 'MATERIAL_SUPERIORITY_ESTABLISHED', 'adds': [{'endpoint': f'COMPETITIVE|pair_benefit_vs_{X}|s{s}', 'test': 'clause_at_0020'}
                                                                        for s in C.SEEDS for X in ('NEURAL_RAW', 'H_LJ')]}},
        {'name': 'M', 'kind': 'fixed_sequence', 'spends_alpha': True,
         'steps': [{'name': 'M_TO', 'conditions': CV + [{'claim_ref': 'E'}, {'test': 'av', 'role': 'TASKONLY'}] + BEN('MECH_TO', 'TASKONLY') + NW('MECH_TO', 'TASKONLY')},
                   {'name': 'M_SS', 'alias_of': 'H_LJ', 'conditions': [{'test': 'av', 'role': 'H_LJ'}] + BEN('COMPETITIVE', 'H_LJ') + NW('COMPETITIVE', 'H_LJ')}],
         'labels': {'both': 'MECHANISM_ATTRIBUTION_ESTABLISHED', 'to_only': 'TASK_ONLY_ATTRIBUTION_ONLY', 'none': 'NOT_ESTABLISHED'}},
        {'name': 'J', 'kind': 'intersection_union', 'spends_alpha': True,
         'conditions': CV + [{'claim_ref': 'E'}, {'test': 'av', 'role': 'B2TWIN'}] + BEN('JOINT', 'B2TWIN') + NW('JOINT', 'B2TWIN'),
         'success_label': 'JOINT_ADVANTAGE_ESTABLISHED', 'failure_label': 'JOINT_ADVANTAGE_NOT_ESTABLISHED',
         'wording': 'At the same six-level coarsening and calibration, replacing half of the LOCAL_TREE income statistic by the replication\'s pair-aware JOINT_TREE statistic lowers the declared readers\' pair recovery. The joint objective is inherited, not designed here.'},
        {'name': 'N', 'kind': 'not_claimed', 'spends_alpha': False, 'conditions': [], 'label': 'NOT_CLAIMED'},
        {'name': 'IBSI', 'kind': 'descriptive', 'spends_alpha': False, 'conditions': [], 'label': 'DESCRIPTIVE (secondary key SEC|IBSI_NLLM_minus_F|pair; COMPARATOR_INELIGIBLE)'},
    ]


def main():
    E = endpoints()
    design = {
        'inference': {'bootstrap_B': 2048, 'bootstrap_seed': 2026101705, 'halt_unless': {'households': 3955, 'persons': 5943},
                      'alpha_family_one_sided': 0.025, 'n_claims': 3, 'claims_spending_alpha': ['B', 'M', 'J'],
                      'K': {'P': 52, 'S': 20}, 'sigma': 0, 'iota': 1,
                      'critical_values_expected': {'c1': 2.3949991, 'c_P': 3.3040310, 'c_S': 3.0252810, 'marginal': 1.9605641},
                      'df': 'G - 1 = 3954 throughout, including subgroups',
                      'weights': 'task_headroom.inference.group_weights(DEV_REPORT unit, seed=bootstrap_seed, draws=bootstrap_B), shared across releases, endpoints and seeds',
                      'auc_replicates': 'task_conditioned_representation.inference.bootstrap_auc(sex, score, W); AUC differences within replicate',
                      'loss_replicates': 'sum W_b,i d_i / sum W_b,i; subgroup weights restricted to the subgroup'},
        'roles': {'F': 'FINALIST', 'REF': ['NEURAL_RAW', 'H_LJ'], 'TO': 'TASKONLY', 'SS': 'H_LJ (alias of a reference; sigma = 0)', 'LOC': 'B2TWIN',
                  'COMPARATOR_INELIGIBLE': ['PUB_IBSI', 'H_JJ'], 'MARGINAL_DESCRIPTIVE_ONLY': ['T_TREE']},
        'availability_ms_point_eligible': {'NEURAL_RAW': True, 'H_LJ': True, 'TASKONLY': True, 'B2TWIN': True, 'PUB_IBSI': False, 'H_JJ': False,
                                           'note': 'from the MODEL_SELECT development table (12 clauses and 2 guards on both seeds); TO needs only presence'},
        'control_validity_bands': {'planted_bit_auc_min': 0.95, 'shuffled_null_auc_abs_minus_half_max': 0.05,
                                   'scope': 'every view (income, occupation, pair) and seed, on the FINALIST wire, DEV_REPORT and ATTACK_VALIDATION'},
        'endpoints': E,
        'claims': claims(),
        'refit_subset': [{'seed': s, 'reader_id': rid} for s in C.SEEDS for rid in
                         ('PAIR__FINALIST__OUT_HGB', 'PAIR__FINALIST__LOG_MLP', 'PAIR__NEURAL_RAW__OUT_MLP', 'PAIR__H_LJ__OUT_MLP',
                          'PAIR__TASKONLY__OUT_HGB', 'PAIR__B2TWIN__OUT_HGB', 'MARG__FINALIST|income__LOOKUP')],
        'refit_subset_note': 'independent verifier refits these reader ids from ATTACK_FIT with the source constructors and compares DEV_REPORT scores',
        'must_not_claim': 'roles/statistical_reviewer/CLAIMS_AND_ENDPOINTS_DRAFT.md section 6, verbatim in force',
        'in_force_by_reference': ('CLAIMS_AND_ENDPOINTS_DRAFT.md section 2 (E separate fields incl. the 28-endpoint simultaneous-confidence rule; '
                                  'M_SS = NOT_TESTED_SEQUENCE_STOPPED when M_TO fails), section 4, section 5 (statuses, failure-category order, flags, '
                                  'wording) and section 6; covered by frozen_design_sha256'),
        'marginal_descriptive': {'set': 'finalist minus every banked comparator for accuracy, NLL and Brier on both tasks; within-SEX occupation Brier and NLL; '
                                        'marginal 95% intervals (t 1.9605641); no status, no claim role',
                                 'mandatory_qualifiers': ['FINALIST - TASKONLY occupation NLL (for M)', 'FINALIST - B2TWIN income Brier (for J)']},
        'mandatory_qualifiers': ['EDGE_SELECTED', 'E fields', 'REPLAY_OF_PUBLISHED and IDENTICAL_RELEASE flags', 'each comparator DEV 12-clause point status',
                                 'exploratory development evidence on previously used records'],
        'notes': 'Encoded from the statistical reviewer draft (17:37Z) for the registered-text finalist; the bank adds T_TREE as a marginal-descriptive release only (8 effective per seed).'}
    AL.build(design)


if __name__ == '__main__':
    main()
