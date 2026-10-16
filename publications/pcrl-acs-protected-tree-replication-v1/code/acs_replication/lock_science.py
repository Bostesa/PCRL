"""Build the SCIENCE_LOCK (all definitions, roles, recipes, selection rules, estimands, endpoint families, budgets and
forecasts) BEFORE any development label is opened. Writes results/pcrl_acs_protected_tree_replication_v1/SCIENCE_LOCK.json.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
PRIV = ROOT / '.private' / 'acs_replication'


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def git(*a):
    return subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def build(fagtb_grid, forecasts):
    from . import schema, trees, leafcorr, neural, neural_raw, selection, readers, report, fare_worker
    prov = json.loads((PRIV / 'admission' / 'PROVENANCE_AUDIT.json').read_text())
    decision = json.loads((RESULTS / 'POPULATION_DECISION.json').read_text())
    adm = json.loads((PRIV / 'admission' / 'METADATA_ADMISSION.json').read_text())
    occ = json.loads((PRIV / 'admission' / 'OCCUPATION_MAPPING.json').read_text())
    code = {p.name: sha(p) for p in sorted((ROOT / 'acs_replication').glob('*.py'))}
    roster = []
    for s in (0, 1):
        roster.append(f'RECIPE|coalition_benefit_vs_TASK_REFIT|s{s} (benefit; margin 0.020)')
        roster += [f'RECIPE|{v}_recipient_worsening_vs_TASK_REFIT|s{s} (no-excess; cap 0.010)' for v in ('income', 'occupation')]
        roster += [f'RECIPE|utility|{t}|{r}|{m}|s{s} (no-excess; cap {c})' for t in ('income', 'occupation') for r in ('U0', 'Ucal')
                   for m, c in (('accuracy_loss', 0.010), ('nll_excess', 0.010), ('brier_excess', 0.005))]
        roster += [f'COMPETITIVE|coalition_benefit_vs_{b}|s{s} (benefit; superiority margin 0; material 0.020 reported)' for b in ('NEURAL_RAW', 'FAGTB', 'FARE')]
        roster += [f'SUBGROUP|occupation_brier_excess_vs_U0|sex{g}|s{s} (no-excess; cap 0.005)' for g in (0, 1)]
    structured = []
    for line in roster:
        key, rest = line.split(' (', 1)
        structured.append({'key': key, 'claim': key.split('|')[0], 'kind': 'benefit' if rest.startswith('benefit') else 'no_excess',
                           'margin': (0.020 if 'margin 0.020' in rest else 0.0) if rest.startswith('benefit') else float(rest.split('cap ')[1].rstrip(')'))})
    lock = {
        'schema': 'ACSR1-science-lock-v1', 'status': 'SCIENCE_LOCKED', 'locked_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'campaign': 'pcrl_acs_protected_tree_replication_v1', 'branch': 'research/pcrl-acs-protected-tree-replication-v1',
        'base_commit': git('rev-parse', 'HEAD'),
        'estimand': 'RECIPE REPLICATION with new models, inputs and task definitions on one ACS 1-Year state-year; NOT confirmation of the frozen Adult packages; no Adult AUC, score map, U0/Ucal weight or 317,000-group requirement is transferred.',
        'source_publications': {'task_headroom': git('rev-parse', 'c8ae518'), 'frozen_tree_completion': git('rev-parse', '8975a23'),
                                'source_recipe': 'L_TREE rho=1 E20 (task_headroom/leafcorr.py arm L_TREE), transported in acs_replication/leafcorr.py with marked ACS-CHANGE lines only'},
        'population': {'admission_order': prov['admission_order'], 'admitted': decision['admitted'], 'decision_record': 'POPULATION_DECISION.json', 'decision_sha256': sha(RESULTS / 'POPULATION_DECISION.json'),
                       'provenance': {'classification': [c['candidate'] + ': ' + c['classification'] for c in prov['candidates_audited']],
                                      'unknown_scope': prov['unknown_scope'], 'receipt_sha256': sha(PRIV / 'admission' / 'PROVENANCE_AUDIT.json'),
                                      'wording': 'NO_USE_FOUND_IN_SEARCHED_RECORDS is bounded documented absence, not certified non-use or person-level independence'},
                       'metadata_admission': {k: adm[k] for k in ('zip_sha256', 'zip_bytes', 'csv_member_sha256', 'raw_person_records',
                                                                   'distinct_housing_unit_serials', 'distinct_group_quarters_serials', 'n_columns')},
                       'excluded_by_rule': prov['excluded_by_rule']},
        'schema': None,
        'benchmark': {**schema.frozen_definition(), 'occupation_mapping_sha256': occ['mapping_sha256'],
                      'occupation_mapping_counts': occ['per_group_code_counts'], 'occupation_mapping_excluded': occ['excluded'],
                      'occupation_mapping_rule': occ['source_rule']},
        'access': {'every_method': 'MODEL_FIT task labels and SEX; the same registered MODEL_SELECT information for selection',
                   'forbidden_for_fitting': 'ATTACK_FIT/ATTACK_VALIDATION labels (readers only), DEV_REPORT labels (opened once after REPORTING_LOCK), SEALED_ASSESSMENT (closed: no features, labels or predictions)',
                   'CALIBRATION': 'Ucal temperatures only'},
        'models': {
            'U0': {'spec': neural.SPEC, 'note': 'validated source per-task network with only input width and occupation output count changed; Recipe A on MODEL_FIT with a 15% household-disjoint internal fold, refit on all MODEL_FIT'},
            'Ucal': neural.UCAL,
            'T_TREE': {'registry': trees.REGISTRY, 'loss': trees.LOSS, 'classes': trees.CLASSES, 'catboost_version': trees.CATBOOST_VERSION,
                       'rule': 'internal fold early stopping: tree count = first argmin of internal loss + 1; refit on all MODEL_FIT with that count and seed; numpy teacher verified bitwise against CatBoost',
                       'headroom_gate': __import__('acs_replication.stage_a_report', fromlist=['x']).__doc__.strip()},
            'LOCAL_TREE': {'arm': 'L_TREE', 'rho': 1.0, 'checkpoint': 'E20', 'recipe': leafcorr.RECIPE, 'widths': leafcorr.WIDTHS,
                           'zero_witness': 'theta = 0 reproduces T_TREE probabilities <= 1e-12 and decisions exactly on every development role'},
            'TASK_REFIT': {'arm': 'T_REFIT', 'checkpoint': 'E20', 'note': 'identical loop, privacy terms disabled, all 20 stages'},
            'JOINT_TREE': {'arm': 'J_TREE', 'rho': 1.0, 'checkpoint': 'E20', 'status': 'OPTIONAL diagnostic only if budget remains; exact local coefficients/initialisation/updates plus the pair term; no strength search, no nomination'},
            'NEURAL_RAW': {'recipe': neural_raw.RECIPE, 'grid': {'strengths': neural_raw.RECIPE['strength_ratios'], 'checkpoints': ['E2', 'E5', 'E10', 'E20']},
                           'note': 'start from the new U0; joint privacy objective; same critics/calibration/batches as LOCAL_TREE'},
            'FAGTB': fagtb_grid,
            'FARE': {'settings': fare_worker.SETTINGS, 'twins': 'Z1-Z3 = gamma 0, same k-bar/n_i budget', 'head': 'jcv.finalize.fit_head on one-hot cells, C by internal-fold log loss',
                     'official': 'eth-sri/fare pinned commit via oar.fare_official (own Python 3.9 environment)', 'access': 'tree on MODEL_FIT with task labels + SEX'},
        },
        'release_contract': 'each recipient receives only its own probability vector and argmax decision; the coalition gets the aligned concatenation for the same person; no leaf ids, representations, teacher outputs, labels, raw features, ids, weights or flags',
        'selection': {'rule': selection.__doc__.strip(), 'common_across_seeds': True,
                      'comparator_eligibility': 'a tunable control is an ELIGIBLE strong comparator iff its frozen configuration came from the MODEL_SELECT-feasible set; a fallback is INELIGIBLE (descriptive only). FAGTB counts as a two-task baseline only if both FAGTB_BINARY (income) and FAGTB_MC_ADAPTATION (occupation) are valid; otherwise the competitive comparison is INCOMPLETE. A required family that cannot run makes it INCOMPLETE, never defeated.'},
        'adversary_and_readers': {'readers': readers.__doc__.strip(), 'threat': 'auxiliary labelled releases (ATTACK_FIT) plus one target release (DEV_REPORT); no target raw inputs, model internals, insiders or interaction; perfect linkage of the two recipients assumed; auxiliary-data recipient adversaries follow existing literature (not a new adversary)'},
        'inference': {'design': report.__doc__.strip(), 'bootstrap_B': report.B, 'bootstrap_seed': report.BOOT_SEED, 'alpha_family_one_sided': report.ALPHA, 'alpha_per_clause_one_sided': report.ALPHA / report.N_CLAIMS,
                      'claims': ['RECIPE_EFFECT', 'COMPETITIVE_COMPARISON', 'SUBGROUP_SUITABILITY'],
                      'claim_rules': {
                          'RECIPE_EFFECT': 'ESTABLISHED iff (IUT, one-sided alpha/3 per clause) coalition benefit vs TASK_REFIT has lower bound > 0.020 on both seeds AND each recipient worsening has upper bound <= 0.010 on both seeds, AND (point-feasibility gate; a deterministic condition on the estimates, no error control) all 24 LOCAL_TREE utility clauses pass at the point. Reported separately: the count of utility clauses whose one-sided alpha/3 upper bound <= cap, and the nested decision RECIPE_EFFECT_AND_UTILITY_BOUNDS (all 30 clauses at their bounds; same clause tests, no extra alpha). Conditional on the two fitted seed pairs and the frozen readers.',
                          'COMPETITIVE_COMPARISON': 'SUPERIOR_TO_ALL_ELIGIBLE (coalition recovery only) iff the comparison is complete, LOCAL_TREE meets the same utility requirements (the baselines\' eligibility rule: 24-clause MODEL_SELECT point feasibility on both seeds; plus the DEV_REPORT 24-clause point gate of RECIPE_EFFECT), at least one required baseline is ELIGIBLE, and coalition benefit vs every ELIGIBLE required baseline (NEURAL_RAW, FAGTB two-task, FARE) has one-sided alpha/3 lower bound > 0 on both seeds; INCOMPLETE if a required family is missing or invalid (never a win); LOCAL_TREE_FAILS_SAME_UTILITY_REQUIREMENTS if the LOCAL_TREE gate fails; the decision names the eligible set; ineligible baselines and recipient-level differences are descriptive. Nested (no extra alpha): MATERIALLY_SUPERIOR_TO_ALL_ELIGIBLE at margin 0.020.',
                          'SUBGROUP_SUITABILITY': 'SUBGROUP_QUALIFIED iff all four guards (occupation summed-Brier excess vs U0 within each SEX group, both seeds) have one-sided alpha/3 upper bound <= 0.005 (IUT); ESTABLISHED_SUBGROUP_HARM iff any guard has K = 40 simultaneous lower bound > 0.005; otherwise SUBGROUP_UNRESOLVED (crossing intervals are unresolved, never harm); NOT_ESTIMABLE if fewer than 200 DEV_REPORT households contain a member of a SEX group (counted from SEX alone, before any outcome is computed). All PASS/VIOLATION/SHORTFALL/HARM labels are read from the K = 40 simultaneous intervals; marginal alpha/3 bounds decide only the positive claim clauses.'},
                      'separation': 'the three conclusions are reported separately and never merged; an UNQUALIFIED recommendation requires SUBGROUP_QUALIFIED; under SUBGROUP_UNRESOLVED any recommendation must state that subgroup suitability is unresolved; under ESTABLISHED_SUBGROUP_HARM (simultaneous intervals) no recommendation for that group; crossing intervals are unresolved, never established harm',
                      'familywise': 'decision family: one-sided FWER 0.025 over the three positive claim decisions (Bonferroni, alpha/3 = 0.0083 per claim; IUT within claims; Wald clause tests, approximate); reporting family: K = 40 two-sided simultaneous 95% intervals (approximate simultaneous coverage); the two families carry separate guarantees and are labelled separately',
                      'scope': 'all intervals and decisions are conditional on the two fitted seed pairs and the frozen readers, in the unweighted DEV_REPORT household superpopulation; they do not quantify between-fit (training) variance or uncertainty about the recipe effect over refits',
                      'sensitivity': 'pre-registered: one-sided percentile bounds from the same 2,048 draws and bootstrap skewness for every roster endpoint; the primary Wald bound governs and every disagreement is disclosed',
                      'primary_roster': roster, 'primary_roster_structured': structured, 'K_primary': len(roster),
                      'weighted_secondary': 'PWGTP-weighted point summaries (ratio estimators) of AUCs and utility, targeting the person-weighted Florida 2019 household-resident cohort analogue for the same fitted models; no interval is attached to weighted values; the household-bootstrap intervals apply only to the unweighted primary estimands; ACS design-based variance (PWGTP1-80 replicate weights, successive-difference replication) is not computed',
                      'sealed_assessment': 'remains closed; a favourable DEV_REPORT does not authorise opening it'},
        'forecasts_registered_before_labels': forecasts,
        'forecast_note': 'FORECASTS.json was committed at the protocol seal (12cebc22f), before any label; after the pre-lock statistical review the ESTABLISHED_SUBGROUP_HARM label (F12) is read from the K = 40 simultaneous intervals and the decision alpha is one-sided 0.025 (family); probabilities are unchanged and scored as registered.',
        'statistical_review_prelock': {'file': 'STATISTICAL_REVIEW_PRELOCK.md', 'sha256': sha(RESULTS / 'STATISTICAL_REVIEW_PRELOCK.md'),
                                       'required_fixes': 'R1-R6 all applied (report.py claims/endpoint/planning/docstring; lock rule strings, familywise, scope, weighted wording)',
                                       'recommended_adopted': ['Rc1 alpha = 0.025 one-sided family', 'Rc3 percentile/skewness sensitivity (primary governs, disagreements disclosed)',
                                                               'Rc5 reader-slate AUC matrix for LOCAL_TREE and TASK_REFIT (descriptive)', 'Rc6 in_primary_family labels',
                                                               'Rc7 structured roster + roster-size assertion', 'Rc9 nested material superiority + descriptive per-recipient differences',
                                                               'Rc10 LOCAL_TREE minus TASK_REFIT utility (descriptive)', 'Rc11 DEV_REPORT row-alignment checks', 'Rc13 unused per-person store removed'],
                                       'recommended_not_adopted': ['Rc2 Holm (Bonferroni kept)', 'Rc4 Monte Carlo flip flags (MC SD of the SE ~1.6% disclosed instead)',
                                                                   'Rc8 domain degrees of freedom (negligible at expected size)', 'Rc12 label renames (decision_scope fields added instead)']},
        'budgets': {'cpu_hours': 20, 'elapsed_hours': 10, 'paid_usd': 0, 'heavy_jobs': 2, 'threads_per_job': 1, 'rss_gib_total': 6,
                    'queue': 'acs_replication/jobqueue.py (renamed from queue.py before the lock to avoid shadowing the standard library), ledger under .private/acs_replication/compute (failed attempts and agent work charged)'},
        'prohibitions': ['no retuning of the Adult candidate', 'no change to LOCAL_TREE after any baseline result', 'no post-report rescue, replacement, threshold change or new nominee',
                         'SEALED_ASSESSMENT closed throughout (features, labels and predictions)', 'no merging of classes or population switch after weak support'],
        'code_sha256_at_lock': code,
    }
    lock.pop('schema'); lock['schema'] = 'ACSR1-science-lock-v1'
    return lock


def main(fagtb_grid_path, forecasts_path):
    lock = build(json.loads(Path(fagtb_grid_path).read_text()), json.loads(Path(forecasts_path).read_text()))
    p = RESULTS / 'SCIENCE_LOCK.json'
    with open(p, 'x') as f:
        json.dump(lock, f, indent=2, sort_keys=True)
    print(json.dumps({'path': str(p), 'sha256': sha(p), 'K_primary': lock['inference']['K_primary']}))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--fagtb', required=True); ap.add_argument('--forecasts', required=True)
    a = ap.parse_args(); main(a.fagtb, a.forecasts)
