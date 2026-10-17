"""Writes READER_BANK_LOCK.json before any real-data reader fit (PROTOCOL.md section 7): recipes, feature maps, candidate order,
tie rule, seeds, controls, reproduction subset, variation diagnostic, verifier refit roster, code hashes, the invented-data dry run
and the statistical reviewer's decision on the single-view construction."""
from __future__ import annotations
import json
import sys

from . import common as C
from . import readers as RD


def cati_state():
    """Label-free fitted state of every CATI design (regime, categories, cuts, knots), per seed, unique wire and pair, from
    ATTACK_FIT released outputs only (recorded before any SEX-labelled fit; the reader run recomputes and must match)."""
    from . import features as F
    out = {}
    for s in C.SEEDS:
        bank, _, _ = C.service_bank(s)
        seen, st = set(), {}
        for svc in C.BANK:
            b = bank[svc]['ATTACK_FIT']
            for view in ('income', 'occupation', 'pair'):
                key = (view, C.wire_sha(b['income']) if view != 'occupation' else '', C.wire_sha(b['occupation']) if view != 'income' else '')
                if key in seen:
                    continue
                seen.add(key)
                st[f'{svc}|{view}'] = F.CatiDesign(view).fit(b['income'], b['occupation']).describe()
        out[f'S{s}'] = st
    return out


def main(dryrun_dir, review_path):
    dr = json.loads((C.ROOT / dryrun_dir / 'READERS_ENH_SEED.json').read_text())
    lock = {
        'schema': 'FFV1-reader-bank-lock-v1', 'utc': C.now(), 'campaign': C.CAMPAIGN,
        'protocol_sha256': C.sha(C.RESULTS / 'PROTOCOL.md'),
        'frozen_candidate_manifest_sha256': C.sha(C.RESULTS / 'FROZEN_CANDIDATE_MANIFEST.json'),
        'source_admission_sha256': C.sha(C.RESULTS / 'SOURCE_ADMISSION.json'),
        'statistical_review': {'path': str(review_path), 'sha256': C.sha(C.ROOT / review_path)},
        'roles': {'fit': 'ATTACK_FIT (8,001)', 'select_and_orient': 'ATTACK_VALIDATION (3,000)', 'score': 'DEV_REPORT released outputs only'},
        'bank': list(C.BANK),
        'source_slate': {'readers': list(RD.SRC8), 'status': 'retained with stored scores (admitted under the source REPORTING_LOCK); not refitted except the reproduction subset'},
        'new_recipes': {
            'ET5': 'ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, max_features=1.0, n_jobs=1, random_state=MODEL_SEED_BASE+s)',
            'ET25': 'ExtraTreesClassifier(n_estimators=300, min_samples_leaf=25, max_features=1.0, n_jobs=1, random_state=MODEL_SEED_BASE+s)',
            'HGB15': 'HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0, early_stopping=False, random_state=MODEL_SEED_BASE+s)',
            'HGB31': 'HistGradientBoostingClassifier(max_leaf_nodes=31, max_iter=200, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0, early_stopping=False, random_state=MODEL_SEED_BASE+s)',
            'MLP64': 'MLPClassifier(hidden_layer_sizes=(64,), alpha=0.01, batch_size=256, max_iter=300, early_stopping=False, random_state=MODEL_SEED_BASE+s); adam, relu',
            'MLP6432': 'MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01, batch_size=256, max_iter=300, early_stopping=False, random_state=MODEL_SEED_BASE+s); adam, relu',
            'CATI01': 'LogisticRegression(C=0.1, solver=lbfgs, max_iter=5000) on features.CatiDesign',
            'CATI1': 'LogisticRegression(C=1.0, solver=lbfgs, max_iter=5000) on features.CatiDesign'},
        'model_seed': 'MODEL_SEED_BASE + s = 2026101205 + s for every recipe, release and view',
        'features': {
            'ET_HGB_MLP': 'features.rich_view: per received output p, log max(p, 1e-300), one-hot argmax; pair = income then occupation',
            'CATI': 'features.CatiDesign (PROTOCOL.md 7.3, as agreed or amended by the statistical review recorded above)',
            'standardization': 'task_conditioned_representation.attacks.FiniteScaler fitted on ATTACK_FIT, every recipe',
            'fitted_state': 'categorical income levels, quantile cut points, spline knots and scalers are fitted on ATTACK_FIT released outputs only',
            'access': 'single views use only that recipient output; pairs use the two aligned outputs; no raw inputs, internals or labels other than ATTACK_FIT SEX (fit) and ATTACK_VALIDATION SEX (selection)'},
        'dedup': 'unique single-recipient wires by exact hashes over ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT; aliased services share fits and predictions',
        'orientation': 'score = predicted probability of the SEX value coded 1; sign fixed by the source rule (+1 iff ATTACK_VALIDATION AUC >= 0.5) before any DEV_REPORT read',
        'selection': {'single_view_candidates': 'source 8 (source order) then new 8 (recipe order above)',
                      'pair_candidates': 'source-first (reviewer A1): source 8 pair readers, source 8 IGNORE_occupation (income-view), source 8 IGNORE_income (occupation-view), then new 8 pair, new 8 IGNORE_occupation, new 8 IGNORE_income',
                      'source_candidate_values': 'stored oriented ATTACK_VALIDATION AUCs (admitted source READERS_SEED.json) and stored DEV_REPORT scores; a non-bitwise reproduction refit is a reported discrepancy, never a substitute',
                      'rule': 'max oriented ATTACK_VALIDATION AUC; exact ties go to the earlier candidate, so the enhanced winner equals the original unless a new candidate is strictly higher',
                      'original_winner': 'the source winner, retained and reported', 'near_tie_flag': 'AV_NEAR_TIE when the winner leads the runner-up by less than 0.010 (about one AUC SE on 3,000 people)'},
        'controls': {'definitions': 'source: shuffled-SEX null (fixed ATTACK_FIT permutation, seed 2026101602 + s) and planted SEX bit, on the FINALIST wire, per view and seed; bands planted >= 0.95, |shuffled - 0.5| <= 0.05 on ATTACK_VALIDATION and DEV_REPORT',
                     'ORIGINAL_CV': "the source run's stored OUT_HGB controls",
                     'ENHANCED_CV': 'all 5 control recipes (OUT_HGB, ET5, HGB31, MLP6432, CATI1) x 3 views x 2 seeds inside the bands on ATTACK_VALIDATION and DEV_REPORT (reviewer A5); the recomputed OUT_HGB controls are also compared bitwise with the stored source controls',
                     'no_retuning': 'no recipe is dropped, replaced or retuned after any control is seen; an ATTACK_VALIDATION control outside its band is a logged deviation before REPORTING_LOCK'},
        'reproduction_subset': 'per seed: source winners for the FINALIST, NEURAL_RAW and H_LJ pairs and the FINALIST occupation view, refitted with the source code; DEV_REPORT scores compared bitwise with the stored scores',
        'variation_diagnostic': 'per seed: enhanced pair winners of FINALIST, NEURAL_RAW, H_LJ; seed-dependent winners (ET, MLP) refitted with model seeds MODEL_SEED_BASE + s + 1000 j, j = 1..4; seed-invariant winners (HGB without early stopping or subsampling, LR, CATI, LOOKUP) labelled seed-invariant (no RNG component), not as measured zero variance; plus 4 household-bootstrap resamples of ATTACK_FIT (auxiliary-sample component, part only); never used for selection; a lower bound on reader-refit variation',
        'verifier_refit_roster': {'per_seed': ['PAIR__FINALIST__ET5', 'PAIR__FINALIST__HGB31', 'PAIR__FINALIST__MLP6432', 'PAIR__FINALIST__CATI1',
                                               'PAIR__NEURAL_RAW__ET5', 'PAIR__NEURAL_RAW__HGB31', 'PAIR__NEURAL_RAW__MLP6432', 'PAIR__NEURAL_RAW__CATI1',
                                               'PAIR__H_LJ__ET5', 'PAIR__H_LJ__HGB31', 'PAIR__H_LJ__MLP6432', 'PAIR__H_LJ__CATI1',
                                               'enhanced pair winners of FINALIST, NEURAL_RAW, H_LJ (resolved by rule after selection)',
                                               'source pair winners of FINALIST, NEURAL_RAW, H_LJ (bitwise against the stored scores)'],
                                  'seeds': [0, 1]},
        'no_additions_after_DEV_REPORT': True,
        'code_sha256': {f'ffv/{n}': C.sha(C.ROOT / 'ffv' / n) for n in ('common.py', 'features.py', 'readers.py', 'jobqueue.py', 'reader_lock.py')},
        'invented_data_dry_run': {'dir': str(dryrun_dir), 'record_sha256': C.sha(C.ROOT / dryrun_dir / 'READERS_ENH_SEED.json'),
                                  'fit_counts': dr['fit_counts'], 'controls_all_in_band': all(
                                      c['planted_attack_validation_auc'] >= 0.95 and abs(c['shuffled_attack_validation_auc'] - 0.5) <= 0.05
                                      for v in dr['controls'].values() for c in v.values()),
                                  'cpu_by_recipe_s': {n: round(sum(a['cpu_s'] for a in dr['accounting'] if a['recipe'] == n), 1) for n in RD.NEW8},
                                  'note': 'invented arrays with the real role shapes (6-level income code, an aliased occupation wire, an unseen DEV income vector); no real data or label'},
        'versions': RD.versions(),
        'estimator_params': {n: RD.params_of(n) for n in RD.NEW8},
        'cati_fitted_state': cati_state(),
        'statistical_review_changes_adopted': ['A1 source-first candidate order (pair: source 8 pair, 8 IGNORE_occupation, 8 IGNORE_income, then new 8 pair, 8 IGNORE_occupation, 8 IGNORE_income); stored source AV AUCs and DEV scores; exact ties to the earlier candidate',
                                               'A2 estimators built directly (never via the source name-dispatching fit_reader); full get_params and library versions asserted at fit time',
                                               'A3 / C1-C6 CATI formulas, explicit unique-quantile spline knots (block omitted below 2 knots), quantile groups, categorical income with unseen counts, fixed column order, convergence recorded and flagged',
                                               'A4 seed-dependent winners (ET, MLP) refitted with 4 extra seeds; seed-invariant winners labelled; RNG component only (lower bound)',
                                               'A5 no recipe dropped or retuned after any control; ENHANCED CV = all 5 control recipes x 3 views x 2 seeds on ATTACK_VALIDATION and DEV_REPORT; recomputed OUT_HGB controls compared bitwise with the stored source controls',
                                               'O3 reader-matched descriptive benefits (same recipe on FINALIST and comparator; DEV_REPORT-descriptive, never selection)',
                                               'O4 near-tie flags (AV gap to runner-up and to the original winner < 0.010)',
                                               'O5 auxiliary-sample variation: pair winners of FINALIST, NEURAL_RAW, H_LJ refitted on 4 household-bootstrap resamples of ATTACK_FIT (seed 2026101805 + 10 s + j)'],
        'statistical_review_optional_declined': {'O1': 'no clipping of contrasts to the ATTACK_FIT range (keeps the registered construction; AUC is rank-based)',
                                                 'O2': 'no continuous-income interaction block (the asymmetry is conservative for claim B; disclosed)',
                                                 'O6': 'class-0 log-contrasts kept (registered)'}}
    C.write_json_x(C.RESULTS / 'READER_BANK_LOCK.json', lock)
    print(json.dumps({k: lock[k] for k in ('protocol_sha256', 'code_sha256', 'invented_data_dry_run')}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
