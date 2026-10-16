"""Write this campaign's locks: ANALYSIS_LOCK (before any reader is fit) and REPORTING_LOCK (after readers, before the
DEV_REPORT truth re-read). Each lock pins code, bank and inputs by sha256; nothing is edited after commit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from . import common as C

CODE = ('acs_hybrids/common.py', 'acs_hybrids/jobqueue.py', 'acs_hybrids/restore_admit.py', 'acs_hybrids/bank.py', 'acs_hybrids/readers.py',
        'acs_hybrids/report.py', 'acs_hybrids/hybrid_deploy.py', 'acs_hybrids/lock.py', 'task_conditioned_representation/attacks.py',
        'task_conditioned_representation/inference.py', 'task_headroom/inference.py', 'acs_replication/trees.py', 'acs_replication/deploy.py',
        'acs_replication/neural.py', 'acs_replication/schema.py')
UTIL = [(t, r, m) for t in C.TASKS for r in C.REFERENCES for m in C.CAPS]


def ad(view, plus, minus):
    return {'type': 'auc_diff', 'view': view, 'plus': plus, 'minus': minus}


REFS = {'coalition': [0.0, 0.020], 'worsening': [0.0, 0.010], 'interaction': [0.0]}
CRITICAL = {'df': 3954, 'clause_one_sided_alpha_over_3': 2.3950, 'primary_K38': 3.2148, 'secondary_K22': 3.0541, 'source_K40_reference': 3.2295, 'marginal_95': 1.9606}


def primary_roster():
    """Final K = 38 roster (STATISTICAL_REVIEW_PRELOCK (b)). value_source: P = exact recomputation of a published source value;
    p = point already public, SE new; N = new. source_counterpart locates the published record, if any."""
    out = []
    for s in C.SEEDS:
        out.append({'key': f'RECIPE|H_LJ|coalition_benefit_vs_TASK_REFIT|s{s}', 'claim': 'A', 'kind': 'benefit', 'margin': 0.020, 'refs': REFS['coalition'], 'seed': s,
                    'estimand': ad('pair', 'TASK_REFIT', C.PRIMARY), 'definition': 'AUC_pair(TASK_REFIT) - AUC_pair(H_LJ)', 'value_source': 'N', 'source_counterpart': None})
        for v, vs, cp in (('income', 'P', {'type': 'endpoint', 'key': f'RECIPE|income_recipient_worsening_vs_TASK_REFIT|s{s}', 'note': 'LOCAL income wire; P assumes the reader replay holds'}),
                          ('occupation', 'p', {'type': 'recovery_points', 'plus': 'JOINT_TREE', 'minus': 'TASK_REFIT', 'view': 'occupation'})):
            out.append({'key': f'RECIPE|H_LJ|{v}_recipient_worsening_vs_TASK_REFIT|s{s}', 'claim': 'A', 'kind': 'no_excess', 'margin': 0.010, 'refs': REFS['worsening'], 'seed': s,
                        'estimand': ad(v, C.PRIMARY, 'TASK_REFIT'), 'definition': f'AUC_{v}(H_LJ) - AUC_{v}(TASK_REFIT)', 'value_source': vs, 'source_counterpart': cp})
        for t, r, m in UTIL:
            comp = 'LOCAL_TREE' if t == 'income' else 'JOINT_TREE'
            out.append({'key': f'RECIPE|H_LJ|utility|{t}|{r}|{m}|s{s}', 'claim': 'A', 'kind': 'no_excess', 'margin': C.CAPS[m], 'refs': [C.CAPS[m]], 'seed': s,
                        'estimand': {'type': 'utility', 'service': C.PRIMARY, 'task': t, 'reference': r, 'metric': m}, 'value_source': 'P',
                        'source_counterpart': {'type': 'service_utility', 'service': comp, 'key': f'{t}|{r}|{m}'}})
        out.append({'key': f'COMPETITIVE|H_LJ|coalition_benefit_vs_NEURAL_RAW|s{s}', 'claim': 'B', 'kind': 'benefit', 'margin': 0.0, 'refs': REFS['coalition'], 'seed': s,
                    'estimand': ad('pair', 'NEURAL_RAW', C.PRIMARY), 'definition': 'AUC_pair(NEURAL_RAW) - AUC_pair(H_LJ); nested material margin 0.020', 'value_source': 'N', 'source_counterpart': None})
        for g in (0, 1):
            out.append({'key': f'SUBGROUP|H_LJ|occupation_brier_excess_vs_U0|sex{g}|s{s}', 'claim': 'C', 'kind': 'no_excess', 'margin': 0.005, 'refs': [0.005], 'seed': s,
                        'estimand': {'type': 'subgroup_occupation_brier_vs_U0', 'service': C.PRIMARY, 'group': g}, 'value_source': 'P',
                        'source_counterpart': {'type': 'service_subgroup', 'service': 'JOINT_TREE', 'group': f'sex{g}'}})
        out.append({'key': f'INCREMENT|H_LJ_over_H_LL|coalition_gain|s{s}', 'claim': None, 'kind': 'benefit', 'margin': None, 'refs': REFS['coalition'], 'seed': s,
                    'estimand': ad('pair', 'H_LL', C.PRIMARY), 'definition': 'AUC_pair(H_LL) - AUC_pair(H_LJ); no claim, no clause status, no gate role', 'value_source': 'N', 'source_counterpart': None})
    return out


def secondary_roster():
    """Final K = 22 descriptive roster (STATISTICAL_REVIEW_PRELOCK (c)): one entry per distinct wire statistic."""
    out = []
    for s in C.SEEDS:
        for h in ('H_LL', 'H_JL', 'H_JJ'):
            cp = {'type': 'endpoint', 'key': f'RECIPE|coalition_benefit_vs_TASK_REFIT|s{s}'} if h == 'H_LL' else None
            out.append({'key': f'SEC|{h}|coalition_benefit_vs_TASK_REFIT|s{s}', 'kind': 'benefit', 'margin': None, 'refs': REFS['coalition'], 'seed': s,
                        'estimand': ad('pair', 'TASK_REFIT', h), 'definition': f'AUC_pair(TASK_REFIT) - AUC_pair({h})', 'source_counterpart': cp})
        for h in ('H_LL', 'H_JL', 'H_JJ'):
            cp = {'type': 'endpoint', 'key': f'COMPETITIVE|coalition_benefit_vs_NEURAL_RAW|s{s}'} if h == 'H_LL' else None
            out.append({'key': f'SEC|{h}|coalition_benefit_vs_NEURAL_RAW|s{s}', 'kind': 'benefit', 'margin': None, 'refs': REFS['coalition'], 'seed': s,
                        'estimand': ad('pair', 'NEURAL_RAW', h), 'definition': f'AUC_pair(NEURAL_RAW) - AUC_pair({h})', 'source_counterpart': cp})
        out.append({'key': f'SEC|JOINT_income_wire|recipient_worsening_vs_TASK_REFIT|s{s}', 'kind': 'no_excess', 'margin': None, 'refs': REFS['worsening'], 'seed': s,
                    'estimand': ad('income', 'H_JJ', 'TASK_REFIT'), 'definition': 'AUC_income(JOINT) - AUC_income(TASK_REFIT) (shared by H_JL and H_JJ)', 'source_counterpart': None})
        out.append({'key': f'SEC|LOCAL_occupation_wire|recipient_worsening_vs_TASK_REFIT|s{s}', 'kind': 'no_excess', 'margin': None, 'refs': REFS['worsening'], 'seed': s,
                    'estimand': ad('occupation', 'H_LL', 'TASK_REFIT'), 'definition': 'AUC_occupation(LOCAL) - AUC_occupation(TASK_REFIT) (shared by H_LL and H_JL)',
                    'source_counterpart': {'type': 'endpoint', 'key': f'RECIPE|occupation_recipient_worsening_vs_TASK_REFIT|s{s}'}})
        out.append({'key': f'SEC|INCREMENT_J|H_JJ_over_H_JL|coalition_gain|s{s}', 'kind': 'benefit', 'margin': None, 'refs': REFS['coalition'], 'seed': s,
                    'estimand': ad('pair', 'H_JL', 'H_JJ'), 'definition': 'AUC_pair(H_JL) - AUC_pair(H_JJ): the occupation swap under JOINT income', 'source_counterpart': None})
        out.append({'key': f'SEC|INTERACTION|pair|s{s}', 'kind': 'difference', 'margin': None, 'refs': REFS['interaction'], 'seed': s,
                    'estimand': {'type': 'auc_interaction', 'view': 'pair'}, 'definition': '(AUC_LL - AUC_LJ) - (AUC_JL - AUC_JJ), pair view; descriptive, not a theorem about synergy',
                    'source_counterpart': None})
        out.append({'key': f'SEC|H_LJ|coalition_benefit_vs_T_TREE|s{s}', 'kind': 'benefit', 'margin': None, 'refs': REFS['coalition'], 'seed': s,
                    'estimand': ad('pair', 'T_TREE', C.PRIMARY), 'definition': 'AUC_pair(T_TREE) - AUC_pair(H_LJ); diagnostic, never a substitute for claim B', 'source_counterpart': None})
    return out


def claim_keys():
    k = lambda pref: [r['key'] for r in primary_roster() if r['key'].startswith(pref)]
    return {'A_coalition_benefit': k('RECIPE|H_LJ|coalition_benefit'), 'A_recipient_worsening': [x for x in k('RECIPE|H_LJ|') if '_recipient_worsening_' in x],
            'A_utility': k('RECIPE|H_LJ|utility|'), 'B_competitive': k('COMPETITIVE|H_LJ|'), 'C_subgroup': k('SUBGROUP|H_LJ|'), 'increment': k('INCREMENT|')}


def settled_before_readers():
    """Outcomes already determined by published source values (STATISTICAL_REVIEW_PRELOCK R1): recorded before any reader
    runs so that none is later reported as a finding. Intervals use the published point and SE with this campaign's
    critical values; the source K = 40 status is shown beside the K = 38 status."""
    src = json.loads((C.SRC_RESULTS / 'DEV_REPORT_RESULTS.json').read_text())
    def rec_of(cp, s):
        if cp['type'] == 'endpoint':
            return src['endpoints'][cp['key']]
        if cp['type'] == 'service_utility':
            return src['per_seed'][str(s)]['services'][cp['service']]['utility'][cp['key']]
        if cp['type'] == 'service_subgroup':
            return src['per_seed'][str(s)]['services'][cp['service']]['subgroup'][cp['group']]
        return None
    rows = {}
    for r in primary_roster():
        cp = r['source_counterpart']
        if not cp or cp['type'] == 'recovery_points':
            if cp:
                rv = src['per_seed'][str(r['seed'])]['services']
                rows[r['key']] = {'value_source': r['value_source'], 'published_point': rv[cp['plus']]['recovery'][cp['view']]['auc'] - rv[cp['minus']]['recovery'][cp['view']]['auc'],
                                  'note': 'point derivable from published AUCs; SE new (not yet computed)'}
            continue
        e = rec_of(cp, r['seed']); pt, se, m = e['point'], e['se'], r['margin']
        L1, U1 = pt - CRITICAL['clause_one_sided_alpha_over_3'] * se, pt + CRITICAL['clause_one_sided_alpha_over_3'] * se
        L, U = pt - CRITICAL['primary_K38'] * se, pt + CRITICAL['primary_K38'] * se
        st38 = 'PASS' if U <= m else ('ESTABLISHED_VIOLATION' if L > m else 'UNRESOLVED')
        rows[r['key']] = {'value_source': r['value_source'], 'published_point': pt, 'published_se': se, 'clause_bound_U1': U1, 'clause_pass': bool(U1 <= m),
                          'point_pass': bool(pt <= m), 'K38_interval': [L, U], 'K38_status': st38, 'source_K40_status': e.get('sim_status'),
                          'status_changed_by_multiplicity_only': bool(e.get('sim_status') is not None and e.get('sim_status') != st38),
                          'source_record': cp}
    ck = claim_keys()
    util = [rows[k] for k in ck['A_utility']]; sg = [rows[k] for k in ck['C_subgroup']]
    inc = [rows[k] for k in ck['A_recipient_worsening'] if 'income' in k]
    hh = src['subgroup_households_per_sex']
    return {'note': ('derived only from published source values (DEV_REPORT_RESULTS of publication f4ea199) before any reader of this campaign was fit; '
                     'not findings of this campaign; the income worsening entries assume the reader replay holds'),
            'per_key': rows,
            'dev_report_utility_point_gate_24': all(u['point_pass'] for u in util),
            'utility_clause_bounds_passing_of_24': int(sum(u['clause_pass'] for u in util)),
            'utility_K38_status_counts': {s_: sum(u['K38_status'] == s_ for u in util) for s_ in ('PASS', 'UNRESOLVED', 'ESTABLISHED_VIOLATION')},
            'subgroup_clause_passes': {k: rows[k]['clause_pass'] for k in ck['C_subgroup']},
            'subgroup_K38_statuses': {k: rows[k]['K38_status'] for k in ck['C_subgroup']},
            'claim_C_outcome_implied': ('SUBGROUP_QUALIFIED' if all(x['clause_pass'] for x in sg) else 'ESTABLISHED_SUBGROUP_HARM'
                                        if any(x['K38_status'] == 'ESTABLISHED_VIOLATION' for x in sg) else 'SUBGROUP_UNRESOLVED'),
            'income_worsening_clause_passes': {k: rows[k]['clause_pass'] for k in ck['A_recipient_worsening'] if 'income' in k},
            'subgroup_households_per_sex': hh, 'subgroup_estimable': all(v >= 200 for v in hh.values())}


def analysis_lock(review_sha, overrides):
    pr, sr = primary_roster(), secondary_roster()
    lock = {
        'schema': 'ACSH1-analysis-lock-v1', 'status': 'ANALYSIS_LOCKED', 'utc': C.now(), 'campaign': 'pcrl_acs_frozen_hybrids_v1',
        'branch': 'research/pcrl-acs-frozen-hybrids-v1',
        'registration_status': ('EXPLORATORY DEVELOPMENT FOLLOW-UP of the already-opened ACS DEV_REPORT results. Not confirmation, not an independent '
                                'replication, not a rescue of the source claim. Source labels stay unchanged: RECIPE_EFFECT_NOT_ESTABLISHED, '
                                'NOT_SUPERIOR_TO_ALL_ELIGIBLE (complete), SUBGROUP_UNRESOLVED, benchmark SUPPORTED. A public receipt does not make the '
                                'already-exposed data confirmatory.'),
        'question': ('Does the fixed composition H_LJ (LOCAL_TREE income output + JOINT_TREE occupation output, paired within each model seed) '
                     'retain the point utility expected from its components, reduce coalition SEX recovery materially below continued task fitting '
                     '(TASK_REFIT E20), and beat the eligible neural control (NEURAL_RAW rho1.0 E2)? Each answered separately, then the conjunction.'),
        'primary': {'service': C.PRIMARY, 'fixed': 'fixed before any reader fit; never replaced by another combination after results; a favourable alternative is a post-hoc observation, not a nominee'},
        'source': {'publication_commit': C.PUBLISHED_SOURCE_COMMIT, 'url': f'https://github.com/Bostesa/PCRL/tree/{C.PUBLISHED_SOURCE_COMMIT}/{C.PUBLISHED_SOURCE_PATH}',
                   'source_admission_sha256': C.sha(C.RESULTS / 'SOURCE_ADMISSION.json')},
        'bank': {'frozen_bank_manifest_sha256': C.sha(C.RESULTS / 'FROZEN_BANK_MANIFEST.json'),
                 'bank_files_sha256': json.loads((C.RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())['bank_files_sha256'],
                 'combinations': {h: {'income': a, 'occupation': b} for h, (a, b) in C.COMPONENTS.items()},
                 'controls': C.CONTROLS, 'references_for_utility_only': list(C.REFERENCES),
                 'ineligible_baselines_not_reattacked': {'FAGTB': 'INELIGIBLE_FALLBACK (source SELECTION_LOCK)', 'FARE': 'INELIGIBLE_FALLBACK (source SELECTION_LOCK)'}},
        'deployment_check_before_lock': {'deployment_results_sha256': C.sha(C.RESULTS / 'DEPLOYMENT_RESULTS.json'), 'note': 'label-free parity of the H_LJ package; no recovery result'},
        'adversary': ('auxiliary labelled releases (ATTACK_FIT with SEX) plus one target release (DEV_REPORT outputs); perfect linkage of the two recipients '
                      'for the coalition; no true task labels, no target raw inputs, no model internals'),
        'readers': {'slate': ['OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB'],
                    'transformations': {'OUT_*': 'probabilities + decision index', 'LOG_*': 'log-probabilities + one-hot decision', 'DEC_LR': 'one-hot decision', 'CONF_HGB': 'max probability'},
                    'definitions': 'task_conditioned_representation.attacks.fit_reader (MLP 64x32 adam alpha 1e-4 300 iter; HGB 15 leaves 200 iter lr 0.05; LR C=1 lbfgs 500 iter); frozen code hash below',
                    'views': ['income', 'occupation', 'pair'], 'rng_rule': 'reader model seed = MODEL_SEED_BASE (2026101205) + model seed s, for every service, view and reader',
                    'train': 'ATTACK_FIT (SEX)', 'select_and_orient': 'ATTACK_VALIDATION only', 'score': 'DEV_REPORT released outputs only (no labels)',
                    'marginal_banks': 'fit once per unique single-recipient wire (10 per seed x 7 readers = 70); sharing services alias the bank',
                    'pair_readers': 'fresh 7 per roster service (7 services x 7 = 49 per seed), including H_LL and H_JJ',
                    'pair_candidates': '7 pair readers + 14 ignore-one aliases (own income-only and occupation-only marginal readers)',
                    'winner_rule': 'max ATTACK_VALIDATION AUC with fixed orientation; ties to the earlier candidate (pair readers in slate order, then income-only, then occupation-only)',
                    'controls': 'per view per seed on the H_LJ wire: shuffled-SEX null (permutation seed 2026101602 + s; OUT_HGB) and planted SEX bit (OUT_HGB)',
                    'effective_fits': {'per_seed': 125, 'total': 250}, 'replay_check': 'label-free comparison with the source stored DEV_REPORT reader scores (recorded only)'},
        'scope_and_selection': [
            ('H_LJ was fixed after the DEV_REPORT results of its components were public (the source published, for LOCAL_TREE and JOINT_TREE, recovery on all '
             'three views, all 12 utility clauses per seed and both subgroup guards). The alpha guarantees assume a roster fixed independently of DEV_REPORT, '
             'so the error rates are nominal and every label is exploratory, not confirmation.'),
            ('Already public: the 24 utility clauses and the 4 subgroup guards are exact recomputations of published values (same outputs, labels and '
             'draws, hence the same points, SEs and alpha/3 bounds); the H_LJ income worsening endpoint is the source RECIPE income worsening endpoint '
             '(if the reader replay holds); the point of the H_LJ occupation worsening endpoint follows from published AUCs.'),
            ('New: only the H_LJ and H_JL pair-view recovery. Six primary endpoints carry new information: the two coalition benefits vs TASK_REFIT, the two '
             'COMPETITIVE benefits and the two increments. The outcomes already settled are written below (settled_before_readers) so that none is reported as a finding.'),
            ('Scope: all intervals and decisions are conditional on the frozen defense models and the fitted readers; the two seeds share people and are not '
             'independent evaluation populations ("both seeds" is a two-fit replication); the target is the unweighted DEV_REPORT household superpopulation, not '
             'ACS survey-design variance; PWGTP values are points only; the source labels are unchanged and the sealed assessment stays closed.')],
        'inference': {'alpha_family_one_sided': 0.025, 'n_claims': 3, 'alpha_per_claim_one_sided': 0.025 / 3, 'bootstrap_B': 2048, 'bootstrap_seed': 2026101603,
                      'bootstrap_unit': 'DEV_REPORT household (G = 3,955), one draw matrix shared across services, views, endpoints and both seeds',
                      'bootstrap_seed_note': 'reused from the source so that, with the same household order, the draw matrix and every recomputed source SE reproduce exactly',
                      'bounds': 'Wald-type: point +/- t(G-1) x bootstrap SE (not studentized)', 'K_primary': len(pr), 'K_secondary': len(sr),
                      'critical_values': CRITICAL, 'subgroup_min_households_per_sex': 200,
                      'distinct_statistics_note': 'Ucal is argmax-invariant, so the U0 and Ucal accuracy-loss rows are identical statistics; K = 38 covers 34 distinct statistics (Bonferroni conservative, as in the source)',
                      'monte_carlo_error_note': 'relative Monte Carlo error of a bootstrap SE at B = 2,048 is about 1.6% (1/sqrt(2(B-1)))',
                      'families': ('decision family: one-sided FWER 0.025 over the three claims (Bonferroni alpha/3; IUT within claims; clause bounds decide only positive '
                                   'claim clauses). Primary reporting family: two-sided 95% Bonferroni-t simultaneous intervals over the K = 38 keys; every failing-direction '
                                   'label comes only from these. Secondary descriptive family: two-sided 95% Bonferroni-t over the K = 22 keys; statuses labelled '
                                   '"descriptive (secondary family)" and never decision-bearing. Marginal 95% intervals (t = 1.9606) outside both families are labelled unadjusted. '
                                   'One statistic never carries two family intervals.'),
                      'multiplicity_vs_source': ('every status is recomputed under K = 38 (never inherited from K = 40); for endpoints with a source status the source K = 40 '
                                                 'status is shown beside it and any difference is labelled STATUS_CHANGED_BY_MULTIPLICITY_ONLY, never new evidence; source labels are not revised')},
        'primary_roster': pr, 'secondary_roster': sr, 'claim_keys': claim_keys(),
        'settled_before_readers': settled_before_readers(),
        'checks_before_truth': ['each hybrid output array equals its components\' frozen outputs bit for bit in every role (BANK)',
                                'each hybrid pair wire equals [income wire | occupation wire] of its components, row-aligned by row_id',
                                'all 250 reader fits complete and every DEV_REPORT score finite', 'replay against the source recorded',
                                'Stage-1 controls pass (ATTACK_VALIDATION)', 'any failure halts the run before truth is opened'],
        'checks_at_truth': ['per-person utility metric vectors equal those of the components',
                            'the bootstrap weight matrix reproduces the source: every endpoint recomputing a published source value has the published point and SE within 1e-12 (utility and subgroup always; income worsening only if the reader replay held bitwise)',
                            'exactly 38 primary and 22 secondary keys exist and all are finite', 'any failure halts the report and no decision is issued'],
        'inheritance_rule': ('exact, before rounding: H_LJ income per-person metric vectors equal LOCAL income\'s, H_LJ occupation equal JOINT occupation\'s, H_LL equal LOCAL, '
                             'H_JJ equal JOINT. An equality failure is a computation or alignment defect: it halts the run and is never a scientific outcome or a tolerance change.'),
        'claims': {
            'A': {'name': 'RECIPE_BENEFIT (H_LJ vs TASK_REFIT; one-sided alpha/3; IUT)', 'labels': ['RECIPE_BENEFIT_ESTABLISHED', 'RECIPE_BENEFIT_NOT_ESTABLISHED'],
                  'invalid_label': 'INVALID_CONTROL_FAILURE',
                  'nested_labels': ['RECIPE_BENEFIT_AND_UTILITY_BOUNDS_ESTABLISHED', 'RECIPE_BENEFIT_AND_UTILITY_BOUNDS_NOT_ESTABLISHED'],
                  'rule': ('If the controls are not valid: INVALID_CONTROL_FAILURE. Otherwise RECIPE_BENEFIT_ESTABLISHED iff keys RECIPE|H_LJ|coalition_benefit_vs_TASK_REFIT|s0 '
                           'and |s1 have clause lower bound L1 > 0.020, the four recipient worsening keys have clause upper bound U1 <= 0.010, and all 24 utility keys pass at '
                           'the point (deterministic gate, no error control); otherwise RECIPE_BENEFIT_NOT_ESTABLISHED listing every failed clause. Nested (no extra alpha): '
                           'RECIPE_BENEFIT_AND_UTILITY_BOUNDS_ESTABLISHED iff claim A is established and all 24 utility keys have U1 <= cap.'),
                  'scope': 'error-controlled for the 6 recovery clauses (approximate Wald, nominal given post-hoc composition, conditional on frozen models and fitted readers); utility gate: point only'},
            'B': {'name': 'COMPETITIVE (H_LJ vs eligible NEURAL_RAW; one-sided alpha/3)',
                  'labels': {'incomplete': 'INCOMPLETE', 'invalid': 'INVALID_CONTROL_FAILURE', 'no_eligible': 'NO_ELIGIBLE_BASELINE', 'fails_gate': 'H_LJ_FAILS_SAME_UTILITY_REQUIREMENTS',
                             'superior': 'SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL', 'not_superior': 'NOT_SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL'},
                  'nested_labels': ['MATERIALLY_SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL', 'NOT_ESTABLISHED'],
                  'rule': ('First rule that applies: (1) INCOMPLETE if NEURAL_RAW pair scores are missing or non-finite for either seed (never a win); (2) INVALID_CONTROL_FAILURE '
                           'if the controls are not valid; (3) NO_ELIGIBLE_BASELINE if SELECTION_LOCK.frozen.NEURAL_RAW.status != ELIGIBLE; (4) H_LJ_FAILS_SAME_UTILITY_REQUIREMENTS '
                           'if H_LJ fails MODEL_SELECT point feasibility (24 clauses, 12 per seed) or the DEV_REPORT 24-clause point gate; (5) SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL '
                           'if COMPETITIVE|H_LJ|coalition_benefit_vs_NEURAL_RAW|s0 and |s1 both have L1 > 0; (6) otherwise NOT_SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL. Nested (no extra '
                           'alpha): MATERIALLY_SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL iff rule 5 and both L1 > 0.020, else NOT_ESTABLISHED. The decision prints the eligible set '
                           '{NEURAL_RAW} and states that FAGTB and FARE had no MODEL_SELECT-feasible configuration and were not compared. T_TREE is never part of claim B. '
                           'The K = 38 status of both keys is reported at 0 and at 0.020.')},
            'C': {'name': 'SUBGROUP_SUITABILITY (H_LJ occupation = JOINT occupation; one-sided alpha/3; IUT)',
                  'labels': {'not_estimable': 'NOT_ESTIMABLE', 'qualified': 'SUBGROUP_QUALIFIED', 'harm': 'ESTABLISHED_SUBGROUP_HARM', 'unresolved': 'SUBGROUP_UNRESOLVED'},
                  'rule': ('First rule that applies: (1) NOT_ESTIMABLE if fewer than 200 DEV_REPORT households contain a member of either SEX group (counted from SEX alone); '
                           '(2) SUBGROUP_QUALIFIED if the four SUBGROUP keys all have U1 <= 0.005; (3) ESTABLISHED_SUBGROUP_HARM if any has K = 38 lower bound L > 0.005 '
                           '(guards named); (4) otherwise SUBGROUP_UNRESOLVED. Reader controls do not affect claim C.')}},
        'separate_status_fields': {
            'overall_point_feasibility': 'DEV_REPORT: POINT_FEASIBLE_24 or POINT_INFEASIBLE (failed keys listed); MODEL_SELECT: the admission result; no error control',
            'overall_confidence_status': ('from the K = 38 intervals of the 24 utility keys: CONFIDENCE_FEASIBLE_24 if all PASS; ESTABLISHED_UTILITY_VIOLATION if any '
                                          'ESTABLISHED_VIOLATION (keys named); otherwise CONFIDENCE_UNRESOLVED; plus PASS / UNRESOLVED / VIOLATION counts and the number of '
                                          'clause bounds passing out of 24 (descriptive except inside claim A\'s nested decision)'),
            'subgroup_status': 'the claim C decision plus the four K = 38 guard statuses',
            'rule': 'never merged with each other or into the conjunction'},
        'increment_rule': 'keys INCREMENT|H_LJ_over_H_LL|coalition_gain|s0/s1: point, SE, K = 38 interval, status at 0 and at 0.020, distance columns; no clause status, no claim, no gate role',
        'controls': {'planted_min_auc': 0.99, 'null_band': [0.45, 0.55], 'null_wide_band_disclosure_only': [0.47, 0.53],
                     'layout': ('H_LJ wire, each seed and view; OUT_HGB with model seed MODEL_SEED_BASE + s; no control is oriented. Shuffled null: trained on ATTACK_FIT with '
                                'SEX permuted by default_rng(2026101602 + s). Planted leak: trained on [wire, SEX] with true ATTACK_FIT SEX; scored with the role\'s SEX on '
                                'ATTACK_VALIDATION; on DEV_REPORT bit 1 for SEX = 1 and bit 0 for SEX = 0 persons. 12 null and 12 planted AUCs.'),
                     'acceptance': ('PLANTED_PASS iff all 12 planted AUCs are finite and >= 0.99; NULL_PASS iff all 12 null AUCs are finite and in [0.45, 0.55]; '
                                    'CONTROLS_VALID iff both; a missing, unfit, refused or non-finite control counts as a failure'),
                     'stage_1_before_truth': ('the 6 ATTACK_VALIDATION planted and 6 ATTACK_VALIDATION null values; on failure stop and do not open truth; a code, wire-construction '
                                              'or alignment defect may be fixed (slate, hyperparameters, seeds and thresholds may not change), then all 250 fits are refit and '
                                              'Stage 1 rerun, with disclosure; if no defect is found the campaign ends ATTACK_PIPELINE_INVALID and DEV_REPORT is never re-read'),
                     'stage_2_at_truth': ('the 6 DEV_REPORT planted and 6 DEV_REPORT null values; on failure claims A and B and their nested decisions become INVALID_CONTROL_FAILURE, '
                                          'the increment and every recovery status (primary and secondary) become INVALID (numbers still shown, marked invalid), the conjunction is '
                                          'NOT_SUPPORTED with CONTROL_FAILURE first; utility point feasibility, overall confidence status and claim C are unaffected; nothing is repaired, '
                                          'rerun or substituted'),
                     'interpretation': ('controls test pipeline validity only; a failed or refused control is never evidence of privacy; the DEV_REPORT planted check cannot '
                                        'detect score-label row misalignment, which the row_id and wire-construction checks cover'),
                     'disclosure': 'each DEV_REPORT null AUC with its household-bootstrap SE and (AUC - 0.5)/SE; NULL_WIDE flags a null outside [0.47, 0.53] but inside [0.45, 0.55] (no decision effect)',
                     'replay': ('readers on the LOCAL, JOINT, TASK_REFIT, NEURAL_RAW and T_TREE wires are compared with the source stored DEV_REPORT scores before truth; a mismatch '
                                'from a construction or alignment defect is fixed before truth with a full refit and disclosure; any other mismatch is recorded, the run continues '
                                'and statements that recovery values equal the source are withdrawn; a replay result never changes the slate, hyperparameters or winners')},
        'central_question': {'labels': ['H_LJ_FIXED_COMPOSITION_SUPPORTED', 'H_LJ_FIXED_COMPOSITION_NOT_SUPPORTED'],
                             'rule': ('reported last: SUPPORTED iff the controls are valid, claim A is RECIPE_BENEFIT_ESTABLISHED and claim B is SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL; '
                                      'otherwise NOT_SUPPORTED listing every unmet component in this order: controls; DEV_REPORT utility point gate (keys); MODEL_SELECT feasibility; '
                                      'coalition benefit s0 and s1; recipient worsening (4 keys); claim B completeness or eligibility; COMPETITIVE benefit s0 and s1. No extra alpha; '
                                      'it excludes subgroup status, overall confidence status, the nested 0.020 targets and the increment. Recommendations follow the source separation '
                                      'rule: unqualified only with SUBGROUP_QUALIFIED; with SUBGROUP_UNRESOLVED they must say that subgroup suitability is unresolved; with '
                                      'ESTABLISHED_SUBGROUP_HARM no recommendation for that group.'),
                             'scope_sentence': ('Exploratory development result on already-opened DEV_REPORT data; the composition was fixed after its components\' DEV_REPORT results '
                                                'were public, so error rates are nominal; conditional on the two fitted seed pairs (shared people, not independent samples), the frozen '
                                                'defense models and the fitted readers, in the unweighted DEV_REPORT household superpopulation; not confirmation; source labels unchanged.')},
        'distance_rule': ('one row per reference value m: coalition benefits (vs TASK_REFIT, NEURAL_RAW, T_TREE) and both increments m in {0, 0.020}; recipient worsening m in '
                          '{0, 0.010}; utility and subgroup m = cap; interaction m = 0. Columns point_minus_m, lower_minus_m, upper_minus_m (the endpoint\'s own family interval), '
                          'clause_bound_minus_m (claim clauses only: L1 - m or U1 - m), status_at_m; signed, positive = above m; unrounded in JSON, 4 decimals in tables; never '
                          'from a marginal interval for an endpoint in a family; absolute AUCs get only AUC - 0.5'),
        'sensitivity': 'for each claim clause a one-sided percentile bound at alpha/3 from the same draws; bootstrap skewness for all 38 primary keys; every disagreement with the Wald clause decision disclosed; never decisive',
        'descriptive_extras': ('DEV_REPORT AUC of every slate reader on the pair view for H_LJ, H_LL, TASK_REFIT and NEURAL_RAW, with the envelope (max over readers) labelled '
                               '"selected on report data"; absolute AUCs for the 17 distinct wires per seed with marginal intervals; per-recipient differences vs NEURAL_RAW and '
                               'T_TREE; utility and subgroup statistics of every service; PWGTP-weighted points'),
        'forecasts_sha256': C.sha(C.RESULTS / 'FORECASTS.json'),
        'prohibitions': ['no defense-model fitting, new teachers, leaf updates, calibration, strength or epoch changes, averaging, stochastic channels, per-person routing or new populations',
                         'no reader trained on MODEL_SELECT, DEV_REPORT or sealed people; no winner chosen by DEV_REPORT AUC',
                         'the Florida sealed assessment stays closed; the ACS extractor is not rerun and the raw census file is not read',
                         'no post-report calibration, model repair, strength search, new combination bank or additional population',
                         'no change of family, alpha or roster after results'],
        'resources': {'cpu_hours': 6, 'elapsed_hours': 6, 'paid_compute_usd': 0, 'heavy_jobs_at_once': 2, 'threads_per_job': 1, 'combined_rss_gib': 4},
        'source_disclosures': {'D1': 'transport-annotation failure in the source (wording of the ACS-CHANGE rule), not a numerical failure; source verification stays 10/11 PASS',
                               'D2': 'source extractor computed sealed features and labels transiently in memory before role separation; the holdout is not described as never accessed'},
        'statistical_review_prelock_sha256': review_sha,
        'code_sha256': {p: C.sha(C.ROOT / p) for p in CODE}}
    lock.update(overrides or {})
    return lock


def reporting_lock():
    """Written after the readers and before the DEV_REPORT truth re-read. Runs the label-free checks before truth
    (STATISTICAL_REVIEW_PRELOCK (d) 2) and Stage 1 of the controls (e); any failure refuses to write the lock."""
    import numpy as np
    from . import readers as RD
    alock_p = C.RESULTS / 'ANALYSIS_LOCK.json'
    if not C.committed(alock_p):
        raise PermissionError('ANALYSIS_LOCK must be committed first')
    alock = json.loads(alock_p.read_text()); cr = alock['controls']
    rec = {'schema': 'ACSH1-reporting-lock-v1', 'status': None, 'utc': C.now(), 'analysis_lock_sha256': C.sha(alock_p),
           'bank_files_sha256': json.loads((C.RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())['bank_files_sha256'],
           'DEV_REPORT_truth_reread_before_this_lock': False, 'sealed_assessment_touched': False, 'readers': {}, 'checks_before_truth': {}}
    st1 = {}
    for s in C.SEEDS:
        d = C.PRIV / 'runs' / f'READERS_{s}'
        rr = json.loads((d / 'READERS_SEED.json').read_text())
        if rr['analysis_lock_sha256'] != C.sha(alock_p):
            raise PermissionError('readers were not fit under the committed ANALYSIS_LOCK')
        with np.load(d / 'DEV_REPORT_SCORES.npz') as z:
            sc = {k: z[k].copy() for k in z.files}
        bank, ids = C.load_bank(s, roles=('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'))
        pair_ok = True
        for svc in C.ATTACK_ROSTER:
            for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'):
                a_, b_ = RD.kinds(bank[svc][r]['income'], 2), RD.kinds(bank[svc][r]['occupation'], 5)
                src_i, src_o = C.SOURCE_OF[svc]
                ref_i = bank['H_LL' if src_i == 'LOCAL_TREE' else 'H_JJ' if src_i == 'JOINT_TREE' else svc][r]['income']
                ref_o = bank['H_LL' if src_o == 'LOCAL_TREE' else 'H_JJ' if src_o == 'JOINT_TREE' else svc][r]['occupation']
                pair_ok &= bool(np.array_equal(np.column_stack((a_['out'], b_['out'])), np.column_stack((RD.kinds(ref_i, 2)['out'], RD.kinds(ref_o, 5)['out']))))
        n_scores = sum(1 for k in sc if k not in ('row_id', 'unit'))
        finite = all(np.isfinite(v).all() for k, v in sc.items() if k not in ('row_id', 'unit'))
        fits = rr['fit_counts']['effective_numerical_fits']
        rec['checks_before_truth'][f'S{s}'] = {'pair_wire_equals_component_concatenation': pair_ok, 'reader_fits': fits, 'fits_expected': 125,
                                                'dev_report_score_arrays': n_scores, 'all_scores_finite': finite,
                                                'score_rows_equal_bank_rows': bool(np.array_equal(sc['row_id'], ids['row_id__DEV_REPORT']))}
        for view, c in rr['controls'].items():
            st1[f'S{s}|{view}'] = {'planted_attack_validation_auc': c['planted_attack_validation_auc'], 'shuffled_attack_validation_auc': c['shuffled_attack_validation_auc'],
                                   'planted_ok': bool(np.isfinite(c['planted_attack_validation_auc']) and c['planted_attack_validation_auc'] >= cr['planted_min_auc']),
                                   'null_ok': bool(np.isfinite(c['shuffled_attack_validation_auc']) and cr['null_band'][0] <= c['shuffled_attack_validation_auc'] <= cr['null_band'][1])}
        rec['readers'][f'S{s}'] = {'readers_seed_json_sha256': C.sha(d / 'READERS_SEED.json'), 'dev_report_scores_sha256': C.sha(d / 'DEV_REPORT_SCORES.npz'),
                                   'model_files_sha256': rr['model_files_sha256'],
                                   'winners': {svc: {v: {'candidate': w['candidate'], 'reader_id': w['reader_id'], 'attack_validation_auc': w['attack_validation_auc']}
                                                     for v, w in x['winner'].items()} for svc, x in rr['services'].items()},
                                   'controls_attack_validation': rr['controls'], 'fit_counts': rr['fit_counts'], 'aliases': rr['aliases'],
                                   'replay_vs_source': {k: rr['replay_vs_source'][k] for k in ('compared', 'bitwise', 'max_abs_diff')},
                                   'replay_winner_agreement': {k: v['equal'] for k, v in rr['replay_vs_source']['winners'].items()}}
    valid1 = len(st1) == 6 and all(v['planted_ok'] and v['null_ok'] for v in st1.values())
    rec['stage_1_controls'] = {'per_view': st1, 'CONTROLS_STAGE1_VALID': valid1}
    pre_ok = all(v['pair_wire_equals_component_concatenation'] and v['reader_fits'] == v['fits_expected'] and v['all_scores_finite'] and v['score_rows_equal_bank_rows']
                 for v in rec['checks_before_truth'].values())
    if not (valid1 and pre_ok):
        raise SystemExit(f'HALT before truth: stage-1 controls valid={valid1}, pre-truth checks ok={pre_ok}; DEV_REPORT truth stays unread')
    rec['status'] = 'REPORTING_LOCKED'
    return rec


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('analysis'); a1.add_argument('--review', required=True); a1.add_argument('--overrides', default='')
    sub.add_parser('reporting')
    a = ap.parse_args()
    if a.cmd == 'analysis':
        ov = json.loads(Path(a.overrides).read_text()) if a.overrides else {}
        lock = analysis_lock(C.sha(a.review), ov)
        p = C.RESULTS / 'ANALYSIS_LOCK.json'
        with open(p, 'x') as f:
            json.dump(lock, f, indent=2, sort_keys=True)
        print(json.dumps({'K_primary': lock['inference']['K_primary'], 'K_secondary': lock['inference']['K_secondary']}))
    else:
        p = C.RESULTS / 'REPORTING_LOCK.json'
        with open(p, 'x') as f:
            json.dump(reporting_lock(), f, indent=2, sort_keys=True)
        print('reporting lock written')


if __name__ == '__main__':
    main()
