"""Public tables from the one logged evaluation (aggregates only; no person-level values):
ORIGINAL_VS_ENHANCED_READERS.csv (recovery per release, view and seed under the original and the enhanced winners, with the paired
reader-class change), PRIMARY_ENDPOINTS.csv (every endpoint of both scorecards), READER_MATCHED_BENEFITS.csv (descriptive)."""
from __future__ import annotations
import csv
import json

from . import common as C


def main():
    rep = json.loads((C.RESULTS / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
    rows = []
    for s, ps in rep['per_seed'].items():
        for svc, v in ps['services'].items():
            for view in ('income', 'occupation', 'pair'):
                o, e, ch = v['recovery_original'][view], v['recovery_enhanced'][view], v['reader_class_change'][view]
                nt = ps['near_ties'][svc][view]
                rows.append({'seed': s, 'release': svc, 'view': view, 'original_winner': o['winner'], 'original_av_auc': o['attack_validation_auc'],
                             'original_dev_auc': o['auc'], 'original_dev_se': o['auc_se'], 'enhanced_winner': e['winner'], 'enhanced_origin': e['origin'],
                             'enhanced_av_auc': e['attack_validation_auc'], 'enhanced_dev_auc': e['auc'], 'enhanced_dev_se': e['auc_se'],
                             'enhanced_minus_original_dev_auc': ch['enhanced_minus_original'], 'change_se': ch['se'],
                             'change_marginal95_low': ch['marginal95'][0], 'change_marginal95_high': ch['marginal95'][1], 'same_reader': ch['same_reader'],
                             'av_gap_to_runner_up': nt['gap_to_runner_up'], 'av_gap_to_original_winner': nt['gap_to_original_winner'], 'AV_NEAR_TIE': nt['AV_NEAR_TIE'],
                             'source_eligibility': {'FINALIST': 'candidate', 'NEURAL_RAW': 'eligible reference', 'H_LJ': 'eligible reference',
                                                    'TASKONLY': 'control (presence)', 'B2TWIN': 'twin (eligible)', 'PUB_IBSI': 'INELIGIBLE (diagnostic)'}[svc]})
    with open(C.RESULTS / 'ORIGINAL_VS_ENHANCED_READERS.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    erows = []
    for sc, card in rep['scorecards'].items():
        for k, e in card['endpoints'].items():
            erows.append({'scorecard': sc, 'key': k, 'family': e['family'], 'claim': e.get('claim'), 'seed': e['spec'].get('seed'), 'kind': e['kind'],
                          'margin': e['margin'], 'point': e['point'], 'se': e['se'], 'clause_L1': e.get('clause_L1'), 'clause_U1': e.get('clause_U1'),
                          'family_lower': e['family_lower'], 'family_upper': e['family_upper'], 'category': e['category'], 'status': e['status'],
                          'flags': ';'.join(e['flags']), 'availability': e['availability']})
    with open(C.RESULTS / 'PRIMARY_ENDPOINTS.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(erows[0])); w.writeheader(); w.writerows(erows)
    mrows = []
    for s, ps in rep['per_seed'].items():
        for key, r in ps['reader_matched_benefits_descriptive'].items():
            comp, recipe = key.split('|')
            mrows.append({'seed': s, 'comparator': comp, 'recipe': recipe, 'origin': r['origin'], 'benefit_comparator_minus_finalist': r['benefit'],
                          'se': r['se'], 'marginal95_low': r['marginal95'][0], 'marginal95_high': r['marginal95'][1],
                          'note': 'DEV_REPORT-descriptive; same recipe on both releases; never used for selection'})
    with open(C.RESULTS / 'READER_MATCHED_BENEFITS.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(mrows[0])); w.writeheader(); w.writerows(mrows)
    print(len(rows), len(erows), len(mrows))


if __name__ == '__main__':
    main()
