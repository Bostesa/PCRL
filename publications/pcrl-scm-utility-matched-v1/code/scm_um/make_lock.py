"""Write UNIT_INVENTORY.json (machine-readable registration) and the pre-fit ANALYSIS_LOCK.json (PROTOCOL.md).

ANALYSIS_LOCK hashes PROTOCOL.md, UNIT_INVENTORY.json, the reviewers' pre-fit reviews and every module of scm_um/ and
scm_bench/ present at lock time. Refuses to overwrite an existing lock.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from . import common as C
from .trajectory import NLL_GRID, CNT_GRID, TRAJ_NAMES, GAMMAS, N_SHUF

UTIL = [(t, r, m) for t in ('income', 'occupation') for r in ('U0', 'Ucal') for m in ('accuracy_loss', 'nll_excess', 'brier_excess')]
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
ROLE_ORDER = ('SCM', 'ORD_NLLM', 'ORD_CNT', 'IBSI_NLLM', 'IBSI_CNT', 'SHUF_NLLM', 'NEURAL_RAW', 'H_LJ', 'C0', 'HIST_SCM')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rosters():
    """Frozen design (statistical reviewer, PREFIT_STATISTICAL_REVIEW.md): 29 primary and 12 secondary keys per seed."""
    P, S = [], []
    for s in (0, 1):
        def ben(fam, comp, claim, refs=(0.0, 0.020)):
            P.append({'key': f'{fam}|coalition_benefit_vs_{comp}|s{s}', 'seed': s, 'claim': claim, 'kind': 'benefit', 'margin': 0.0,
                      'refs': list(refs), 'estimand': {'type': 'auc_diff', 'plus': comp, 'minus': 'SCM', 'view': 'pair'}})

        def wor(fam, comp, claim):
            for v in ('income', 'occupation'):
                P.append({'key': f'{fam}|{v}_worsening_vs_{comp}|s{s}', 'seed': s, 'claim': claim, 'kind': 'no_excess', 'margin': 0.010,
                          'refs': [0.010], 'estimand': {'type': 'auc_diff', 'plus': 'SCM', 'minus': comp, 'view': v}})
        ben('MATCH', 'ORD_NLLM', 'M'); wor('MATCH', 'ORD_NLLM', 'M')
        ben('MATCH_COMPONENT', 'IBSI_NLLM', 'MC', (0.0,)); ben('MATCH_COMPONENT', 'SHUF_NLLM', 'MC', (0.0,))
        ben('CAPACITY', 'ORD_CNT', 'K'); wor('CAPACITY', 'ORD_CNT', 'K'); ben('CAPACITY', 'IBSI_CNT', 'KC', (0.0,))
        ben('COMPETITIVE', 'NEURAL_RAW', 'B'); ben('COMPETITIVE', 'H_LJ', 'B'); wor('COMPETITIVE', 'NEURAL_RAW', 'B'); wor('COMPETITIVE', 'H_LJ', 'B')
        for t, r, m in UTIL:
            P.append({'key': f'UTILITY|{t}|{r}|{m}|s{s}', 'seed': s, 'claim': 'U', 'kind': 'no_excess', 'margin': CAPS[m], 'refs': [CAPS[m]],
                      'estimand': {'type': 'utility', 'service': 'SCM', 'task': t, 'reference': r, 'metric': m}})
        for g in (0, 1):
            P.append({'key': f'SUBGROUP|occupation_brier_excess_vs_U0|sex{g}|s{s}', 'seed': s, 'claim': 'C', 'kind': 'no_excess', 'margin': 0.005,
                      'refs': [0.005], 'estimand': {'type': 'subgroup_occupation_brier_vs_U0', 'service': 'SCM', 'group': g}})

        def sec(key, est, kind='difference', margin=0.0, refs=(0.0,)):
            S.append({'key': f'SEC|{key}|s{s}', 'seed': s, 'kind': kind, 'margin': margin, 'refs': list(refs), 'estimand': est})
        sec('C0_minus_SCM|pair', {'type': 'auc_diff', 'plus': 'C0', 'minus': 'SCM', 'view': 'pair'})
        sec('C0_minus_ORD_NLLM|pair', {'type': 'auc_diff', 'plus': 'C0', 'minus': 'ORD_NLLM', 'view': 'pair'})
        sec('HIST_SCM_minus_SCM|pair', {'type': 'auc_diff', 'plus': 'HIST_SCM', 'minus': 'SCM', 'view': 'pair'})
        sec('ORD_CNT_minus_ORD_NLLM|pair', {'type': 'auc_diff', 'plus': 'ORD_CNT', 'minus': 'ORD_NLLM', 'view': 'pair'})
        sec('ORD_NLLM_minus_SCM_slate7|pair', {'type': 'auc_diff_slate7', 'plus': 'ORD_NLLM', 'minus': 'SCM', 'view': 'pair'})
        sec('occupation_nll_difference_vs_ORD_CNT', {'type': 'nll_diff', 'plus': 'SCM', 'minus': 'ORD_CNT', 'task': 'occupation'})
        sec('occupation_nll_difference_vs_IBSI_CNT', {'type': 'nll_diff', 'plus': 'SCM', 'minus': 'IBSI_CNT', 'task': 'occupation'})
        sec('SCM_pair_minus_SCM_occupation', {'type': 'auc_view_diff', 'service': 'SCM', 'plus_view': 'pair', 'minus_view': 'occupation'})
        for comp in ('ORD_NLLM', 'IBSI_NLLM', 'SHUF_NLLM'):
            sec(f'occupation_nll_difference_vs_{comp}', {'type': 'nll_diff', 'plus': 'SCM', 'minus': comp, 'task': 'occupation'}, 'match', 0.001, (-0.001, 0.001))
        sec('income_nll_difference_vs_ORD_NLLM', {'type': 'nll_diff', 'plus': 'SCM', 'minus': 'ORD_NLLM', 'task': 'income'})
    return P, S


def inventory(P, S):
    keys = [p['key'] for p in P]; skeys = [x['key'] for x in S]
    return {
        'schema': 'SCMUM1-unit-inventory-v1', 'campaign': C.CAMPAIGN,
        'source': {'research_branch': 'research/pcrl-scm-engineering-benchmark-v1', 'recorded_tip': '2738cc30b', 'public_package': '6fc06021d'},
        'seeds': [0, 1], 'roles_access': {'MODEL_FIT': 'admitted source statistics only', 'MODEL_SELECT': 'selection and matching (labels, SEX)',
                                          'ATTACK_FIT': 'readers (8,001)', 'ATTACK_VALIDATION': 'readers (3,000)', 'DEV_REPORT': 'one logged evaluation + one verifier replay',
                                          'CALIBRATION': 'unused', 'SEALED_ASSESSMENT': 'closed'},
        'inputs': ['AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP'],
        'trajectories': {'names': list(TRAJ_NAMES), 'SCM': 'source SCM, beta 1.0, stop at no admissible pair', 'ORD': 'source TWIN rule at beta 1.0, no target, stop at no admissible pair',
                         'IBSI': {'gammas': list(GAMMAS), 'target_cells': 32}, 'SHUF': {'r': list(range(N_SHUF)), 'rule': 'source shuffled-signal SCM, beta 1.0, stop at no admissible pair'}},
        'checkpoints': {'nll_grid': list(NLL_GRID), 'nll_rule': 'last prefix m with F_m <= t + 1e-15 (F = engine running sum of occupation-NLL changes)',
                        'nll_reached': 'm*(t) < M or F_M >= t - 1e-15; else alias of TERM, missing coverage',
                        'count_grid': list(CNT_GRID), 'terminal': 'TERM', 'scm_count_companions': 'S{t} on ORD, IBSI_g*, SHUF_r*: prefix at SCM N{t} merge count',
                        'historical': 'HIST: prefix at the historical count (261 / 266)', 'dedupe': 'bit-identical per-initial-cell (v1[group], v2[cell]) tables'},
        'selection': {'scm_candidates': 'SCM N{t}, common t', 'nomination': {'clauses_24': True, 'occupation_nll_excess_max': 0.008, 'subgroup_guard_max': 0.005,
                                                                             'objective': 'min mean proxy over seeds', 'ties': 'smaller t'},
                      'fallback': 'min worst normalised violation over 32 conditions; ties smaller t; ineligible',
                      'match_tolerance_nats': 0.001, 'expenditure': 'mean MODEL_SELECT occupation NLL minus frozen C0, per seed',
                      'ORD_NLLM': 'per seed: match and 12 clauses; min (proxy, |delta|, m)', 'IBSI_NLLM': 'common gamma: admissible in both seeds; min mean proxy; grid ties',
                      'SHUF_NLLM': 'common r: as IBSI', 'nearest_diagnostic': '12 clauses; min (worst-seed |delta|, mean proxy, grid, m); labelled NEAREST_DIAGNOSTIC_NOT_MATCHED',
                      'ORD_CNT': 'ORD S{t*}; UNMATCHED_CAPACITY if unreached', 'IBSI_CNT': 'source rule 6 at S{t*}'},
        'final_bank_roles': list(ROLE_ORDER), 'max_releases_per_seed': 10,
        'readers': {'slate': ['OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP'], 'pair_candidates': '8 pair + 16 ignore-one',
                    'fit': 'ATTACK_FIT 8001', 'select': 'ATTACK_VALIDATION 3000', 'seed': 'MODEL_SEED_BASE + s', 'lookup_pseudo': 2.0,
                    'controls': {'null_band': [0.45, 0.55], 'planted_min_auc': 0.99, 'wire': 'SCM'},
                    'verifier_refits': 'pair winners of SCM, ORD_NLLM, IBSI_NLLM, NEURAL_RAW on both seeds'},
        'inference': {'bootstrap_B': 2048, 'bootstrap_seed': 2026101705, 'alpha_family_one_sided': 0.025, 'n_claims': 4, 'claim_slices': ['M', 'K', 'B', 'C'],
                      'K_primary': len(P), 'K_secondary': len(S), 'family': 'two-sided 95% Bonferroni-t', 'wald': 'point +/- t(G-1) x bootstrap SE'},
        'primary_roster': P, 'secondary_roster': S,
        'claim_keys': {'match_benefit': [k for k in keys if k.startswith('MATCH|coalition')],
                       'match_worsening': [k for k in keys if k.startswith('MATCH|') and '_worsening_' in k],
                       'match_nll': [k for k in skeys if k.startswith('SEC|occupation_nll_difference_vs_ORD_NLLM')],
                       'component_ibsi': [k for k in keys if k.startswith('MATCH_COMPONENT|') and 'IBSI' in k] + [k for k in skeys if k.startswith('SEC|occupation_nll_difference_vs_IBSI_NLLM')],
                       'component_shuf': [k for k in keys if k.startswith('MATCH_COMPONENT|') and 'SHUF' in k] + [k for k in skeys if k.startswith('SEC|occupation_nll_difference_vs_SHUF_NLLM')],
                       'capacity_benefit': [k for k in keys if k.startswith('CAPACITY|coalition_benefit_vs_ORD_CNT')],
                       'capacity_worsening': [k for k in keys if k.startswith('CAPACITY|') and '_worsening_' in k],
                       'capacity_ibsi': [k for k in keys if k.startswith('CAPACITY|coalition_benefit_vs_IBSI_CNT')],
                       'competitive_benefit': [k for k in keys if k.startswith('COMPETITIVE|coalition')],
                       'competitive_worsening': [k for k in keys if k.startswith('COMPETITIVE|') and '_worsening_' in k],
                       'utility_24': [k for k in keys if k.startswith('UTILITY|')], 'subgroup': [k for k in keys if k.startswith('SUBGROUP|')]},
        'labels': {'M': ['CONTROL_FAILURE', 'NOMINEE_INELIGIBLE', 'NOMINEE_IS_C0', 'MATCH_UNAVAILABLE', 'MATCH_DID_NOT_CARRY_OVER', 'NOMINEE_FAILS_DEV_UTILITY_GATE',
                         'NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD', 'MATCHED_ADVANTAGE_ESTABLISHED | MATCHED_ADVANTAGE_NOT_ESTABLISHED'],
                   'K': ['CONTROL_FAILURE', 'NOMINEE_INELIGIBLE', 'NOMINEE_IS_C0', 'UNMATCHED_CAPACITY', 'COMPARATOR_INELIGIBLE', 'NOMINEE_FAILS_DEV_UTILITY_GATE',
                         'NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD', 'SAME_CAPACITY_ADVANTAGE_ESTABLISHED | SAME_CAPACITY_ADVANTAGE_NOT_ESTABLISHED'],
                   'B': ['CONTROL_FAILURE', 'NOMINEE_INELIGIBLE', 'NOMINEE_IS_C0', 'NOMINEE_FAILS_DEV_UTILITY_GATE', 'NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD',
                         'SUPERIOR_TO_ALL_ELIGIBLE | NOT_SUPERIOR_TO_ALL_ELIGIBLE'],
                   'C': ['SUBGROUP_QUALIFIED', 'SUBGROUP_HARM', 'SUBGROUP_UNRESOLVED']},
        'units': {'UM_ADMIT_V1': 0.01, 'UM_TRAJ_S0': 0.25, 'UM_TRAJ_S1': 0.25, 'UM_SELECT': 0.3, 'UM_BANK': 0.02, 'UM_READERS_S0': 0.15,
                  'UM_READERS_S1': 0.15, 'UM_REPORT': 0.05, 'UM_DEPLOY': 0.05, 'verifier': 0.5, 'backup': 0.05, 'agents_and_light_scripts': 1.5,
                  'failed_attempt_margin': 1.5, 'total_estimate_cpu_h': 4.8},
        'limits': {'elapsed_h': 6, 'start_utc': '2026-10-17T04:38:00+00:00', 'cpu_h': 10, 'cloud_usd': 0, 'heavy_jobs': 2, 'threads_per_job': 1},
        'predictions': {'P1': 0.95, 'P2': 0.70, 'P3': 0.70, 'P4': 0.35, 'P5': 0.55, 'P6': 0.75, 'P7': 0.50, 'P8': 0.10},
    }


def main():
    lock = C.RESULTS / 'ANALYSIS_LOCK.json'
    if lock.exists():
        raise PermissionError('ANALYSIS_LOCK exists')
    P, S = rosters()
    assert len(P) == 58 and len(S) == 24, (len(P), len(S))
    inv = inventory(P, S)
    (C.RESULTS / 'UNIT_INVENTORY.json').write_text(json.dumps(inv, indent=1, sort_keys=True))
    reviews = sorted(str(p.relative_to(C.ROOT)) for p in (C.RESULTS / 'roles').rglob('*.md'))
    rec = {'schema': 'SCMUM1-analysis-lock-v1', 'utc': C.now(), 'status': 'PRE-FIT LOCK (no new trajectory, MODEL_SELECT label or reader fit yet)',
           'protocol_sha256': sha(C.RESULTS / 'PROTOCOL.md'), 'unit_inventory_sha256': sha(C.RESULTS / 'UNIT_INVENTORY.json'),
           'source_admission_sha256': sha(C.RESULTS / 'SOURCE_ADMISSION.json'),
           'prefit_reviews_sha256': {r: sha(C.ROOT / r) for r in reviews},
           'code_sha256': {str(p.relative_to(C.ROOT)): sha(p) for d in ('scm_um', 'scm_bench') for p in sorted((C.ROOT / d).glob('*.py'))},
           'inference': inv['inference'], 'claim_keys': inv['claim_keys'], 'primary_roster': P, 'secondary_roster': S,
           'note': 'bank, readers and report code are written after this lock and frozen in REPORTING_LOCK before any DEV_REPORT label; they implement the rosters and rules registered here'}
    lock.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({'K_primary': len(P), 'K_secondary': len(S), 'protocol_sha256': rec['protocol_sha256'][:12], 'inventory_sha256': rec['unit_inventory_sha256'][:12],
                      'code_files': len(rec['code_sha256']), 'reviews': reviews}, indent=1))


if __name__ == '__main__':
    main()
