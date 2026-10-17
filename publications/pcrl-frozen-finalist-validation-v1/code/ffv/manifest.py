"""Unit FFV_MANIFEST: FROZEN_CANDIDATE_MANIFEST.json from the admitted copies only (no label is read; nothing is refitted).

Records the frozen finalist exactly as deployed (routing files, stored corrections, thresholds, binning rule, CALIBRATION-fitted
per-level income probabilities, package and release hashes, evaluated-role wire hashes), the compact six-release bank with the
source eligibility labels, exact-wire aliasing between releases, and the source results this campaign must keep intact.
"""
from __future__ import annotations
import json

import numpy as np

from acs_replication import trees
from . import common as C

FIN = 'B3CTL_UNCOUPLED_MIX50_C6A'


def level_vectors(h, rows_per_level=1):
    """Per-level income probability vectors with the package's own arithmetic (amd.fam_b.package_release), evaluated in one
    batch holding `rows_per_level` rows of each level. The last bits depend on the batch size (matrix-vector kernel path):
    small batches can differ from the evaluated large-batch release by about 1e-16 on some levels; this is recorded, not repaired."""
    m = 6
    c = np.repeat(np.arange(m), rows_per_level)
    s = ((np.eye(m)[c][:, 1:] - h['mu']) / h['sd']) @ h['w'] + float(h['b'][0])
    p = trees.probabilities('income', s)
    return np.array([p[c == j][0] for j in range(m)])


def main():
    src = lambda n: json.loads((C.SRC_RESULTS / n).read_text())
    flock, sbm, alock = src('FINALIST_LOCK.json'), src('SERVICE_BANK_MANIFEST.json'), src('ANALYSIS_LOCK.json')
    dep, prom = src('DEPLOYMENT_RESULTS.json'), src('PROMOTION.json')
    rep = src('report/DEV_REPORT_RESULTS.json')
    per_seed, problems = {}, []
    for s in C.SEEDS:
        pkg = C.admitted(f'deploy/FINALIST_S{s}/PACKAGE.json'); meta = json.loads(pkg.read_text())
        with np.load(C.admitted(f'deploy/FINALIST_S{s}/H.npz')) as z:
            h = {k: z[k].copy() for k in z.files}
        rel = json.loads(C.admitted(f'runs/B/{FIN}/S{s}/RELEASE.json').read_text())
        with np.load(C.admitted(f'runs/B/{FIN}/S{s}/RELEASE.npz')) as z:
            arr = {k: z[k].copy() for k in z.files}
        lv = level_vectors(h, rows_per_level=1000)      # batch as large as the evaluated role batches
        lv1 = level_vectors(h, rows_per_level=1)        # one row per level (a 6-row request)
        if [float(x) for x in h['thresholds']] != [float(x) for x in rel['thresholds']]:
            problems.append(f'S{s}: package thresholds differ from the release record')
        observed = {}
        for r in C.ROLES6:
            pi = arr[f'{r}__income']; u = np.unique(pi, axis=0)
            match = [int(np.flatnonzero((lv == row).all(1))[0]) if (lv == row).all(1).any() else None for row in u]
            observed[r] = {'distinct_income_vectors': int(len(u)), 'levels_observed': sorted(m for m in match if m is not None),
                           'all_equal_a_package_level_vector_bitwise': all(m is not None for m in match)}
            if not observed[r]['all_equal_a_package_level_vector_bitwise']:
                problems.append(f'S{s} {r}: an income vector is not one of the package level vectors')
        per_seed[f'S{s}'] = {
            'thresholds_repr': [repr(float(x)) for x in h['thresholds']], 'thresholds_hex': [float(x).hex() for x in h['thresholds']],
            'binning_rule': 'level = numpy.searchsorted(thresholds, d, side="left"): the number of thresholds strictly below d; d equal to a threshold takes the lower level',
            'calibration_h': {k: [repr(float(x)) for x in np.ravel(v)] for k, v in h.items() if k != 'thresholds'},
            'per_level_income_probability': [[repr(float(x)) for x in row] for row in lv],
            'per_level_income_probability_hex': [[float(x).hex() for x in row] for row in lv],
            'per_level_decision': [int(x) for x in lv.argmax(1)],
            'small_batch_last_bit_observation': {
                'note': 'per-level vectors recomputed in a 6-row batch (one row per level) versus the 6,000-row batch above; equality is bitwise',
                'equal_bitwise_by_level': [bool((lv1[j] == lv[j]).all()) for j in range(6)],
                'max_abs_difference_by_level': [float(np.abs(lv1[j] - lv[j]).max()) for j in range(6)],
                'decisions_equal': bool((lv1.argmax(1) == lv.argmax(1)).all())},
            'package_json_sha256': C.sha(pkg), 'package_files_sha256': meta['files'], 'config': meta['config'],
            'release_sha256': rel['release_sha256'], 'release_sha256_equals_finalist_lock': rel['release_sha256'] == flock['release_sha256'][f'B/{FIN}'][f'S{s}'],
            'wire_sha256_by_role': {k: v for k, v in rel['wire_sha256'].items()},
            'observed_income_levels_by_role': observed,
            'deployment_record': {k: dep['packages'][f'FINALIST_S{s}'][k] for k in ('api_reproduces_evaluated_DEV_REPORT_wire_bitwise', 'csv_reproduces_evaluated_DEV_REPORT_wire_bitwise')}}
    bank = {}
    for s in C.SEEDS:
        ws = sbm['per_seed'][f'S{s}']['wire_sha256']
        groups = {}
        for t in C.TASKS:
            for svc in C.BANK:
                key = (t, tuple(ws[f'{svc}|{t}'][r] for r in ('MODEL_SELECT',) + C.ROLES3))
                groups.setdefault(key, []).append(svc)
        bank[f'S{s}'] = {'wire_sha256': {f'{svc}|{t}': ws[f'{svc}|{t}'] for svc in C.BANK for t in C.TASKS},
                         'identical_single_recipient_wires': [{'task': k[0], 'services': v} for k, v in groups.items() if len(v) > 1],
                         'unique_single_recipient_wires': {t: sum(1 for k in groups if k[0] == t) for t in C.TASKS},
                         'effective_releases': len(C.BANK)}
    elig = alock['availability_ms_point_eligible']
    roles = {'FINALIST': 'frozen candidate', 'NEURAL_RAW': 'eligible reference (claim B)', 'H_LJ': 'eligible reference (claim B; same occupation output)',
             'TASKONLY': 'matched coarse task-only control (claim M step 1)', 'B2TWIN': 'local-only coarse twin (claim J)',
             'PUB_IBSI': 'diagnostic comparator; source-ineligible (cannot be counted as beaten)'}
    keep = {k: {'point': rep['endpoints'][k]['point'], 'clause_L1': rep['endpoints'][k]['clause_L1'], 'se': rep['endpoints'][k]['se'],
                'category': rep['endpoints'][k]['category']} for k in rep['endpoints'] if k.startswith(('COMPETITIVE|pair', 'MECH_TO|pair', 'JOINT|pair'))}
    out = {'schema': 'FFV1-frozen-candidate-manifest-v1', 'utc': C.now(), 'campaign': C.CAMPAIGN,
           'candidate': {'config_id': FIN, 'family': 'B', 'source_selection_label': flock['finalist']['selection_label'],
                         'source_designation': 'control-designated configuration (coarse-only twin of the coupled B3), promoted under the statistical reviewer adjudication (source PROMOTION.json)',
                         'source_promotion_sha256': C.sha(C.SRC_RESULTS / 'PROMOTION.json'), 'source_promotion_adjudication': prom.get('adjudication'),
                         'definition': {
                             'occupation': 'the stored JOINT_TREE output: frozen CatBoost teacher (T_TREE) occupation trees routed on the frozen 145-column encoding, plus the stored JOINT_TREE E20 leaf-score corrections, softmax over the 5 groups; bitwise identical to H_LJ occupation output',
                             'income_statistic': 'd = 0.5 x LOCAL_TREE E20 income log-odds + 0.5 x JOINT_TREE E20 income log-odds (same frozen income teacher routing)',
                             'income_release': 'six decision-aligned levels of d (label-free MODEL_FIT thresholds: 3 levels on each side of 0); per-level probability from an L2 logistic regression on the one-hot level fitted on CALIBRATION income labels (stored H.npz); argmax decision',
                             'inputs': list(C.RAW_FIELDS), 'never_an_input': 'SEX'},
                         'not_changed_here': 'fitted parameters, thresholds, calibration, rounding, routing and deployed output are preserved exactly; nothing is refitted or improved'},
           'per_seed': per_seed,
           'output_contract': flock['output_contract'], 'release_contract': sbm['release_contract'],
           'bank': {'releases': {svc: {'role': roles[svc], 'source_ms_point_eligibility': elig.get(svc), 'source_bank_role': svc} for svc in C.BANK},
                    'excluded_from_this_bank': {'H_JJ': 'source secondary comparator (utility-ineligible); not in the brief bank', 'T_TREE': 'teacher; not in the brief bank'},
                    'per_seed': bank,
                    'neural_raw_seed1': 'admitted output by hash (source SERVICE_BANK); no raw-input package exists (deployment coverage gap)'},
           'source_results_kept_intact': {'claims': {k: v['label'] for k, v in rep['claims'].items()}, 'pair_benefit_endpoints': keep,
                                          'source_report_sha256': C.sha(C.SRC_RESULTS / 'report' / 'DEV_REPORT_RESULTS.json'),
                                          'summary': 'all task, subgroup and recipient conditions passed at the point; recovery comparisons against both eligible references were positive under the source scorecard; the 0.020 material margin and simultaneous utility preservation were not established'},
           'problems': problems}
    C.write_json_x(C.RESULTS / 'FROZEN_CANDIDATE_MANIFEST.json', out)
    print(json.dumps({'problems': problems, 'levels': {s: v['per_level_income_probability'] for s, v in per_seed.items()},
                      'observed': {s: {r: (o['distinct_income_vectors'], o['levels_observed']) for r, o in v['observed_income_levels_by_role'].items()} for s, v in per_seed.items()},
                      'aliases': {s: v['identical_single_recipient_wires'] for s, v in bank.items()}}, indent=1))


if __name__ == '__main__':
    main()
