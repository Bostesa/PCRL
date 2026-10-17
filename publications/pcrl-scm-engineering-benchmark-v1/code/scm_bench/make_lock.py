"""Write the machine-readable ANALYSIS_LOCK.json (pre-fit) for the SCM engineering benchmark (PROTOCOL.md)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from . import common as C

CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rosters():
    P, S = [], []
    for s in (0, 1):
        P.append({'key': f'RECIPE|coalition_benefit_vs_TWIN|s{s}', 'seed': s, 'claim': 'A', 'kind': 'benefit', 'margin': 0.020, 'refs': [0.0, 0.020],
                  'estimand': {'type': 'auc_diff', 'plus': 'TWIN', 'minus': 'SCM', 'view': 'pair'}})
        for v in ('income', 'occupation'):
            P.append({'key': f'RECIPE|{v}_worsening_vs_TWIN|s{s}', 'seed': s, 'claim': 'A', 'kind': 'no_excess', 'margin': 0.010, 'refs': [0.010],
                      'estimand': {'type': 'auc_diff', 'plus': 'SCM', 'minus': 'TWIN', 'view': v}})
        for comp, claim in (('IBSI', 'ATT'), ('SHUF', 'ATT'), ('NEURAL_RAW', 'B'), ('H_LJ', 'B')):
            fam = 'ATTRIBUTION' if claim == 'ATT' else 'COMPETITIVE'
            P.append({'key': f'{fam}|coalition_benefit_vs_{comp}|s{s}', 'seed': s, 'claim': claim, 'kind': 'benefit', 'margin': 0.0,
                      'refs': [0.0, 0.020], 'estimand': {'type': 'auc_diff', 'plus': comp, 'minus': 'SCM', 'view': 'pair'}})
        for t in ('income', 'occupation'):
            for r in ('U0', 'Ucal'):
                for m, cap in CAPS.items():
                    P.append({'key': f'UTILITY|{t}|{r}|{m}|s{s}', 'seed': s, 'claim': 'U', 'kind': 'no_excess', 'margin': cap, 'refs': [cap],
                              'estimand': {'type': 'utility', 'service': 'SCM', 'task': t, 'reference': r, 'metric': m}})
        for g in (0, 1):
            P.append({'key': f'SUBGROUP|occupation_brier_excess_vs_U0|sex{g}|s{s}', 'seed': s, 'claim': 'C', 'kind': 'no_excess', 'margin': 0.005,
                      'refs': [0.005], 'estimand': {'type': 'subgroup_occupation_brier_vs_U0', 'service': 'SCM', 'group': g}})
        S.append({'key': f'SEC|C0_minus_SCM|pair|s{s}', 'seed': s, 'kind': 'difference', 'margin': 0.0, 'refs': [0.0],
                  'estimand': {'type': 'auc_diff', 'plus': 'C0', 'minus': 'SCM', 'view': 'pair'}})
        S.append({'key': f'SEC|C0_minus_TWIN|pair|s{s}', 'seed': s, 'kind': 'difference', 'margin': 0.0, 'refs': [0.0],
                  'estimand': {'type': 'auc_diff', 'plus': 'C0', 'minus': 'TWIN', 'view': 'pair'}})
        for v in ('income', 'occupation'):
            S.append({'key': f'SEC|SCM_minus_NEURAL_RAW|{v}|s{s}', 'seed': s, 'kind': 'difference', 'margin': 0.0, 'refs': [0.0],
                      'estimand': {'type': 'auc_diff', 'plus': 'SCM', 'minus': 'NEURAL_RAW', 'view': v}})
        S.append({'key': f'SEC|TWIN_minus_SCM_slate7|pair|s{s}', 'seed': s, 'kind': 'difference', 'margin': 0.0, 'refs': [0.0],
                  'estimand': {'type': 'auc_diff_slate7', 'plus': 'TWIN', 'minus': 'SCM', 'view': 'pair'}})
    return P, S


def main():
    P, S = rosters()
    keys = [p['key'] for p in P]
    lock = {
        'schema': 'SCM1-analysis-lock-v1', 'utc': C.now(), 'status': 'PRE-FIT LOCK (no real fit of this benchmark has run)',
        'protocol_sha256': sha(C.RESULTS / 'PROTOCOL.md'),
        'code_sha256': {str(p.relative_to(C.ROOT)): sha(p) for p in sorted((C.ROOT / 'scm_bench').glob('*.py'))},
        'source': {'research_branch': 'research/pcrl-structural-mechanism-pilot-v1', 'tip': 'ec20c8bdf', 'public_package': '9b0f51c4e',
                   'scm_proposal_sha256': '6bc0617b286dccd2e7b428c51b15c6399696cacf5737415188d6db46f0aa4002'},
        'recipe': {'folds': 5, 'fine_cells_nominal': 512, 'income_groups': 32, 'kappa': 20, 'betas': [0.5, 1.0, 1.5], 'ibsi_gammas': [0.5, 1.0, 2.0],
                   'shuffled_controls': 5, 'arms_per_seed': 31, 'accuracy_budget': 0.005},
        'units': {'SCM_FIT_S0': 0.5, 'SCM_FIT_S1': 0.5, 'SCM_SELECT': 0.3, 'SCM_BANK': 0.02, 'SCM_READERS_S0': 0.3, 'SCM_READERS_S1': 0.3,
                  'SCM_REPORT': 0.05, 'SCM_DEPLOY': 0.05, 'verifier': 0.4, 'agents_and_light_scripts': 0.6, 'failed_attempt_margin': 1.0,
                  'total_estimate_cpu_h': 4.0, 'limit_cpu_h': 8.0, 'elapsed_limit_h': 6, 'start_utc': '2026-10-17T02:38:58+00:00',
                  'heavy_processes': 2, 'threads_per_job': 1, 'cloud_usd': 0},
        'selection': 'PROTOCOL.md section 7 (implemented in scm_bench/select_scm.py)',
        'readers': {'attack_roster_order': ['SCM', 'TWIN', 'IBSI', 'SHUF', 'NEURAL_RAW', 'H_LJ', 'C0'], 'frozen_attacked': ['NEURAL_RAW', 'H_LJ'],
                    'control_wire': 'SCM', 'slate': ['OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP'],
                    'lookup': {'key': 'exact float64 bytes of the released vector(s)', 'score': '(n_female + 2 pi) / (n + 2)', 'unseen': 'pi'},
                    'learning_curve': 'none', 'verifier_refit_subset': 'pair winner of SCM, TWIN and NEURAL_RAW, both seeds'},
        'inference': {'bootstrap_B': 2048, 'bootstrap_seed': 2026101705, 'alpha_family_one_sided': 0.025, 'n_claims': 3,
                      'K_primary': len(P), 'K_secondary': len(S)},
        'controls': {'null_band': [0.45, 0.55], 'planted_min_auc': 0.99},
        'claims': {'B': {'eligible_set': ['NEURAL_RAW', 'H_LJ']}},
        'primary_roster': P, 'secondary_roster': S,
        'claim_keys': {'recipe_coalition': [k for k in keys if k.startswith('RECIPE|coalition')],
                       'recipe_worsening': [k for k in keys if '_worsening_' in k],
                       'attribution': [k for k in keys if k.startswith('ATTRIBUTION')],
                       'competitive': [k for k in keys if k.startswith('COMPETITIVE')],
                       'utility_24': [k for k in keys if k.startswith('UTILITY')],
                       'subgroup': [k for k in keys if k.startswith('SUBGROUP')]},
        'access': {'MODEL_FIT': 'features, task labels, SEX (propensity and guard only)', 'CALIBRATION': 'unused by SCM',
                   'MODEL_SELECT': 'features, task labels, SEX (utility clauses and cross-fitted proxies)', 'ATTACK_FIT/ATTACK_VALIDATION': 'final readers only',
                   'DEV_REPORT': 'features; truth once after the committed REPORTING_LOCK, logged', 'SEALED_ASSESSMENT': 'closed'},
    }
    assert lock['inference']['K_primary'] == 42 and lock['inference']['K_secondary'] == 10
    (C.RESULTS / 'ANALYSIS_LOCK.json').write_text(json.dumps(lock, indent=1, sort_keys=True))
    print(json.dumps({'K_primary': len(P), 'K_secondary': len(S), 'protocol_sha256': lock['protocol_sha256']}))


if __name__ == '__main__':
    main()
