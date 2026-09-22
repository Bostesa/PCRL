"""Write the machine-readable pre-outcome registration (no 2016 data is read here)."""
from __future__ import annotations
import json

import numpy as np

from experiments.pcrl_task_directed_release_v1.audits import AUDIT_CONFIG, PROBABILITY_FLOOR, SELECTION_RULE
from . import inference_panel as ip
from .audit_panel import H_ROLES, RELEASE_ROLES, SEED_BASE, all_units, dependencies, role_seed
from .common import (ANCHORS, CANDIDATES, INPUTS, OUT, PANEL, PINNED, PRIMARY_ROLES, RESTORED, ROOT,
                     SECONDARY_COMPARATORS, TASK_ROLE, array_hash, atomic_json, now, read_json, sha)
from .releases import ACTIONS, ERASERS, MAPS, FrozenReleases, object_hashes

PURPOSE = {'H': 'service-only reference (immutable services)', 'J': 'historical operating-point comparator',
           'Q': 'predeclared operating-point hypothesis 1 (constrained stochastic release)',
           'D17': 'predeclared operating-point hypothesis 2 (same code, same 17 actions, unconstrained)',
           'D33': 'control: finer deterministic task control (33 actions)', 'C': 'control: direct continuous supervised task prediction',
           'E': 'control: label-matched LEACE adaptation (mechanism40)', 'S': 'control: label-matched SPLINCE adaptation (mechanism40)',
           'RR75': 'fixed simple-randomization control: randomized response over D17 actions, publish 0.75',
           'W75': 'fixed simple-randomization control: withhold D17 action with probability 0.25'}

FAIRNESS = {
    'H': {'training_labels': 'none new (historical services only)', 'training_households': 'n/a', 'actions': 1,
          'channel_width': 0, 'inference_inputs': 'H_A (4) to A, H_B (2) to B'},
    'J': {'training_labels': 'historical J training (source labels + protected-attribute adversarial terms); no residence supervision',
          'training_households': 'historical representation_fit (anchor-specific 2018 split)', 'actions': 1,
          'channel_width': 16, 'inference_inputs': 'standardized PCA32 -> frozen 64-16 mapper'},
    'Q': {'training_labels': 'residence (teacher, 48% of RF; costs on mechanism 40% of RF); SEX and RAC1P (A-view constraint tables on mechanism rows)',
          'training_households': 'anchor-specific 2018 representation_fit roles (teacher 48%, internal 12%, mechanism 40%)',
          'actions': 17, 'channel_width': '1 categorical token (17 values)', 'inference_inputs': 'PCA32 + H_A -> T0 code (32 cells) -> one sampled token'},
    'D17': {'training_labels': 'residence only (teacher + mechanism costs); no protected labels used by the channel',
            'training_households': 'same as Q', 'actions': 17, 'channel_width': '1 categorical token (17 values, deterministic)',
            'inference_inputs': 'same as Q'},
    'D33': {'training_labels': 'residence only', 'training_households': 'same as Q', 'actions': 33,
            'channel_width': '1 categorical token (33 values, deterministic)', 'inference_inputs': 'same as Q'},
    'C': {'training_labels': 'residence (teacher)', 'training_households': 'teacher 48% of RF', 'actions': 'continuous',
          'channel_width': 1, 'inference_inputs': 'PCA32 + H_A -> teacher probability p'},
    'E': {'training_labels': 'SEX and RAC1P (joint one-hot, complete cases) on mechanism 40% of RF; frozen teacher logit as input',
          'training_households': 'mechanism40 slice (anchor 0: 4131 people / 2800 households)', 'actions': 'continuous',
          'channel_width': 33, 'inference_inputs': '[PCA32, clipped teacher logit] -> affine eraser'},
    'S': {'training_labels': 'SEX and RAC1P as E, plus residence/income/employment preservation covariance',
          'training_households': 'mechanism40 slice (as E)', 'actions': 'continuous', 'channel_width': 33,
          'inference_inputs': 'as E'},
    'RR75': {'training_labels': 'as D17 (no new fit)', 'training_households': 'as D17', 'actions': 17,
             'channel_width': '1 categorical token (17 values, full support)', 'inference_inputs': 'as D17 plus private coin'},
    'W75': {'training_labels': 'as D17 (no new fit)', 'training_households': 'as D17', 'actions': '17 + withheld symbol',
            'channel_width': '1 categorical token (18 values)', 'inference_inputs': 'as D17 plus private coin'},
}


def alias_groups_2018():
    """Exact equivalence of released views on the 2018 test rows of each anchor (identical wire = alias)."""
    from .verify_historical import test_inputs
    groups = {}
    for a in ANCHORS:
        fr = FrozenReleases(a)
        d = test_inputs(a)
        enc = {'test': fr.encode(d['x'], d['ha'])}
        prints = {}
        for short in PANEL:
            rel = fr.release(short, {'test': {k: d[k] for k in ('x', 'ha', 'J')}}, enc)['test']
            key = (array_hash(rel['token_probs']), None if rel['aux'] is None else array_hash(np.asarray(rel['aux'])))
            prints.setdefault(key, []).append(short)
        groups[a] = sorted(prints.values())
    return groups


def main():
    staged = read_json(INPUTS/'STAGED_INPUTS.json')
    model_manifest = {'created_utc': now(), 'pinned': PINNED,
                      'historical_inputs': {r['path']: r['sha256'] for r in staged['files'] if 'acs_2016' not in r['path']},
                      'admitted_2016_files': {r['path']: r['sha256'] for r in staged['files'] if 'acs_2016' in r['path']},
                      'release_objects': {a: object_hashes(a) for a in ANCHORS},
                      'archive': {'bucket': 'pcrl-ux-archive-ed9d21fd',
                                  'manifest_key': 'pcrl_task_directed_release_v1/final-5a78a39412f7/MANIFEST.private.json',
                                  'manifest_sha256': '504944807f9f9b9f087836d34768440647b6276db0fca2827be3084213b3e667'},
                      'rule': 'no mapper, teacher, eraser, code, dictionary or Q is refitted; hashes are verified before use'}
    atomic_json(OUT/'MODEL_MANIFEST.json', model_manifest)
    aliases = alias_groups_2018()
    panel = {'created_utc': now(), 'families': {k: {'family': v, 'purpose': PURPOSE[k], 'map': MAPS.get(k),
                                                   'actions': ACTIONS.get(k, 17) if k in MAPS else None,
                                                   'eraser': ERASERS.get(k), 'fairness': FAIRNESS[k]} for k, v in PANEL.items()},
             'anchors': list(ANCHORS), 'anchor_rule': 'three frozen fits; common families across anchors; anchors are not independent datasets',
             'candidates': list(CANDIDATES), 'controls': [k for k in PANEL if k not in CANDIDATES],
             'secondary_comparators_of_Q': list(SECONDARY_COMPARATORS),
             'simple_randomization_controls': {'RR75': 'T0_rr_0.75 (fitted/archived as a fixed transform of T0_U_unconstrained_a17; complete for all three anchors)',
                                               'W75': 'T0_withhold_0.75 (same)'},
             'not_in_panel': {'OptNet16': 'no saved encoder weights (predecessor FRESH_YEAR_CONFIRMATION_PROTOCOL); no transport comparison exists',
                              'leace_A0/splince_A0': 'historical arrays only in this panel scope; not listed by the assignment',
                              'union88 erasers': 'assignment specifies mechanism40 label access'},
             'feature_schema': {'raw_covariates': ['AGEP', 'WKHP', 'SCHL', 'MAR', 'RELP', 'CIT', 'DIS', 'DEAR', 'DEYE', 'DREM'],
                                'preprocessing': 'frozen per-anchor CovariatePreprocessor (preprocessing.json) -> float32',
                                'pca': 'frozen per-anchor release_maps.joblib[pca] -> PCA32 float32',
                                'standardization': '(PCA32 float64 - J input_mean)/input_scale -> float32',
                                'H_A': 'income_binary (2) + civilian_at_work (2) frozen service heads on PCA32',
                                'H_B': 'public_coverage (2) frozen service head on PCA32',
                                'categorical_maps': 'unseen valid codes and missing/invalid map to separate declared columns (historical code)'},
             'service_fingerprints': {'2018_parity': 'PCA32, H_A, H_B and J recomputed from raw 2018 rows are byte-identical to the archived arrays (all pools, anchor 0 checked in Tier A; all anchors in HISTORICAL_VERIFICATION.json)'},
             'runtime_randomness_contract': 'one persistent token per person drawn from the Q row of its code with a private per-person coin; repeated requests return the same token; the coin and Q row are never released',
             'exact_equivalence_groups_2018_test': aliases,
             'roles': {'H': list(H_ROLES), 'other': list(RELEASE_ROLES)},
             'units': len(all_units())}
    atomic_json(OUT/'PANEL.json', panel)
    atomic_json(OUT/'ATTACK_RECIPES.json', {
        'slate': 'predecessor catchup slate, unchanged', 'config': AUDIT_CONFIG, 'probability_floor': PROBABILITY_FLOOR,
        'candidates': ['logistic (C=1, lbfgs, max_iter 2000; H + token onehot + H x onehot)',
                       'hist_gb_20, hist_gb_5 (exact person-token expansion; min_samples_leaf x max token support)',
                       'sampled_hist_gb_20, sampled_hist_gb_5 (one fixed sampled token per person; seed 20260921+role seed)',
                       'mlp_120, mlp_360 (64-32 ReLU, Adam 1e-3, wd 1e-4, batch 256 people, validation every 5 epochs, nested 120/360 selections of one trajectory)'],
        'fit_weighting': '0.5 + 0.5 * PWGTP/mean(PWGTP) per original person; token expansion conserves person weight',
        'ancestors': {'H': 'all same-role H-only model candidates (wire H) for every non-H release',
                      'AB': 'all candidates of the same release A role + H B role (B unchanged)',
                      'J': 'never an ancestor of a replacement release'},
        'selection_rule': SELECTION_RULE, 'validation_pool': 'validation/attacker_validation (2016 admission partition)',
        'fit_pool': 'fitting (attacker_fit U task_fit, 2016 admission partition)',
        'seeds': f'{SEED_BASE} + 10000*anchor + 37*role_index(predecessor ROLE_SPECS order); identical across releases (common random numbers)',
        'role_seeds': {f'{a}|{r}': role_seed(a, r) for a in ANCHORS for r in H_ROLES},
        'dependencies_example': {'Q/0/attack:AB/SEX': dependencies('Q', 0, 'attack:AB/SEX')},
        'full_schema': {'SEX': 2, 'RAC1P': 9, 'missing_fit_class': 'uniform full-schema floor (1-K*1e-9)q+1e-9; class never removed'},
        'stochastic_training': 'exact expected-risk person-token expansion with conserved weights (logistic, exact trees, MLP minibatches); sampled-token trees use one fixed categorical draw per person',
        'raw_input_diagnostic': 'not run (optional); absence of a stronger attacker is not a privacy certificate'})
    atomic_json(OUT/'UTILITY_RECIPES.json', {
        'role': TASK_ROLE, 'independent_probes': 'same catchup slate as attacks on [H_A, release]',
        'fixed_decoder': 'frozen prescribed decoder sigmoid(logit b(H_A)+a_Z) for token releases; p for C; b(H_A) for H; absent for J/E/S',
        'H_recalibrators': 'frozen global-offset decoders over the union 17/33 action dictionary applied to b(H_A) (legal for every release)',
        'H_ancestors': 'same-role H-only probes', 'selection_rule': SELECTION_RULE,
        'validation_pool': 'validation/task_validation', 'fit_pool': 'fitting',
        'reported_separately': ['validation-selected predictor (primary)', 'independent-slate selection', 'fixed decoder']})
    prim, sec, desc = ip.primary_endpoints(), ip.secondary_endpoints(), ip.descriptive_endpoints()
    counts = ip.family_counts()
    atomic_json(OUT/'PRIMARY_CLAIMS.json', {
        'status': 'registered_before_2016_outcomes', 'candidates': list(CANDIDATES),
        'alpha_family': ip.ALPHA_FAMILY, 'alpha_per_candidate': ip.ALPHA_PER_CANDIDATE,
        'z_one_sided': ip.z_primary(), 'task_margin': ip.TASK_MARGIN, 'privacy_cap': ip.PRIVACY_CAP,
        'clauses_per_candidate': counts['primary_per_candidate'], 'endpoints': prim,
        'decision_rule': 'intersection-union: candidate passes iff all ten one-sided 97.5% upper bounds meet their thresholds; invalid/missing = fail',
        'bound': 'estimate + z_.975 * household-bootstrap SE (pointwise component bounds; not simultaneous coverage); SE=0 -> bound = estimate',
        'error_control': 'nominal/asymptotic family-wise .05 over declaring either conjunction (Bonferroni .025 x 2), conditional on fitted releases, fitted attackers and their selection'})
    atomic_json(OUT/'SECONDARY_CONTRASTS.json', {
        'status': 'registered_before_2016_outcomes', 'family_size': counts['secondary'], 'arithmetic': counts['secondary_arithmetic'],
        'alpha': ip.SECONDARY_ALPHA, 'z_two_sided_simultaneous': ip.z_secondary(counts['secondary']),
        'correction': 'Bonferroni two-sided normal, z_(1-.05/(2*70))', 'endpoints': sec,
        'separation': 'separate family; no overall error rate across primary and secondary; no secondary result can become a primary success',
        'descriptive_endpoints': len(desc), 'descriptive_rule': 'unadjusted marginal 95% intervals, labeled descriptive'})
    atomic_json(OUT/'BOOTSTRAP_SPEC.json', {
        'implementation': 'experiments/pcrl_task_directed_release_v1/uncertainty.py::paired_household_bounds (unchanged); only estimate and bootstrap_se are consumed',
        'implementation_sha256': sha(ROOT/'experiments/pcrl_task_directed_release_v1/uncertainty.py'),
        'replicates': ip.BOOT['n_boot'], 'seed': ip.BOOT['seed'],
        'resampling': 'multinomial household multiplicities over the union of final households, one draw shared by every endpoint, interface and anchor',
        'estimator': 'equal mean of three anchor weighted ratios; numerator and denominator recomputed per draw',
        'zero_denominator': 'entire common draw rejected and counted (frozen predecessor rule)',
        'identical_pairs': 'exact zero estimate and zero variance',
        'verified_on_2018': 'reproduces archived PAIRED_BOUNDS estimates and SEs for 38 registered endpoints (max SE error 4.3e-19)',
        'scope': 'conditional on fitted releases, fitted attackers and selections; survey-weighted household bootstrap is not an official ACS design-based interval'})
    print(json.dumps({'aliases': aliases, 'counts': counts}, indent=1))


if __name__ == '__main__':
    main()
