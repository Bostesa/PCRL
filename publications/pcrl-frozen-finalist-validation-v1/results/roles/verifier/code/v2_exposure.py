"""Unit VER_EXPOSURE_V1 (check 2: exposure ledger). Label-free: only row_id and unit of the six label-free features exports are
used, plus public campaign records, admitted access-log snapshots and the lineage DEV_REPORT access log (events only).

Public-use ACS 2019 Florida person records. (a) Own household-overlap check across the six admitted development roles. (b) Spot
checks of EXPOSURE_AND_SELECTION_AUDIT.md entries against the source records they cite.
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import vcommon as V
from ffv import common as C        # admitted loaders only

AMD = V.SRC_RES
REP = V.ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
HYB = V.ROOT / 'results' / 'pcrl_acs_frozen_hybrids_v1'
AUDIT_TABLE = {'MODEL_FIT': (20000, 13337, 7), 'CALIBRATION': (2982, 1974, 5), 'MODEL_SELECT': (8001, 5306, 8), 'ATTACK_FIT': (8001, 5319, 6),
               'ATTACK_VALIDATION': (3000, 1974, 5), 'DEV_REPORT': (5943, 3955, 11)}


def spot(claim, record, found, expected):
    return {'claim': claim, 'record': record, 'found': found, 'expected_from_audit': expected, 'match': found == expected}


def main():
    out = {'unit': 'VER_EXPOSURE_V1', 'labels_read': 'none'}
    # ---------------------------------------------------------------- (a) household overlap, own code
    ids = {}
    for r in V.ROLES6:
        F = C.load_role(r, ('features',))
        ids[r] = (F['row_id'].copy(), F['unit'].copy())
        del F
    roles = {}
    for r, (rid, unit) in ids.items():
        u, c = np.unique(unit, return_counts=True)
        roles[r] = {'persons': int(len(rid)), 'households': int(len(u)), 'largest_household': int(c.max()),
                    'row_ids_unique': bool(len(np.unique(rid)) == len(rid)), 'audit_table_match': (int(len(rid)), int(len(u)), int(c.max())) == AUDIT_TABLE[r]}
    pairs = {}
    for a, b in itertools.combinations(V.ROLES6, 2):
        pairs[f'{a} x {b}'] = {'shared_households': int(len(np.intersect1d(ids[a][1], ids[b][1]))),
                               'shared_persons': int(len(np.intersect1d(ids[a][0], ids[b][0])))}
    allu = np.concatenate([ids[r][1] for r in V.ROLES6])
    allr = np.concatenate([ids[r][0] for r in V.ROLES6])
    es = V.jload(REP / 'EXTRACTION_SUMMARY.json')['roles']
    # a household is one unit id: check that no unit id spans two roles and that persons of one household share one role
    out['household_overlap'] = {'per_role': roles, 'pairs': pairs, 'n_pairs': len(pairs),
                                'all_pairs_zero': all(v['shared_households'] == 0 and v['shared_persons'] == 0 for v in pairs.values()),
                                'union_persons': int(len(allr)), 'union_distinct_persons': int(len(np.unique(allr))),
                                'union_distinct_households': int(len(np.unique(allu))),
                                'equals_extraction_summary': {r: (roles[r]['persons'], roles[r]['households']) == (es[r]['selected_persons'], es[r]['selected_households']) for r in V.ROLES6},
                                'sealed_assessment': 'UNKNOWN (ids not admitted; not opened)'}
    # ---------------------------------------------------------------- (b) spot checks of the ledger against cited records
    sc = []
    rj = V.jload(V.ADM / 'runs' / 'B' / 'B3CTL_UNCOUPLED_MIX50_C6A' / 'S0' / 'RELEASE.json')
    sc.append(spot('F8: seed-0 FINALIST bins hold 4,114 / 4,113 / 4,114 / 2,553 / 2,554 / 2,552 MODEL_FIT people',
                   'admitted runs/B/B3CTL_UNCOUPLED_MIX50_C6A/S0/RELEASE.json -> coarse_cell_counts.MODEL_FIT',
                   rj['coarse_cell_counts']['MODEL_FIT'], [4114, 4113, 4114, 2553, 2554, 2552]))
    fl = V.jload(AMD / 'FINALIST_LOCK.json')
    rr = fl['release_records']['B/B3CTL_UNCOUPLED_MIX50_C6A|S0']
    sc.append(spot('F9: calibration h fitted on all 2,982 CALIBRATION people, income labels only, no SEX',
                   'AMD/FINALIST_LOCK.json -> release_records[B/B3CTL_UNCOUPLED_MIX50_C6A|S0].h_fit_role, .labels_used_at_training.CALIBRATION',
                   [rr['h_fit_role'], rr['labels_used_at_training']['CALIBRATION']],
                   ['CALIBRATION (all 2982 rows; features -> base outputs; income labels; no SEX)', ['income']]))
    sc.append(spot('F9: CALIBRATION is 2,982 people in 1,974 households', 'CALIBRATION features export (row_id, unit)',
                   [roles['CALIBRATION']['persons'], roles['CALIBRATION']['households']], [2982, 1974]))
    pr = V.jload(AMD / 'PROMOTION.json')
    tr = {t['config_id']: t for t in pr['trials']}
    sc.append(spot('S7: 17 trials considered, 3 promotable (all EDGE, none robust); proxies B3CTL 0.60002, B2 0.61314, B2CTL 0.61533; PROMOTION 17:35:23Z; B3CTL control-designated',
                   'AMD/PROMOTION.json -> n_trials_considered, n_promotable, n_promotable_robust, trials[].mean_pair_proxy, utc, trials[B3CTL].designation',
                   [pr['n_trials_considered'], pr['n_promotable'], pr['n_promotable_robust'],
                    round(tr['B3CTL_UNCOUPLED_MIX50_C6A']['mean_pair_proxy'], 5), round(tr['B2_CSR_JT_C6A']['mean_pair_proxy'], 5),
                    round(tr['B2CTL_UNCOUPLED_C6A']['mean_pair_proxy'], 5), pr['utc'], tr['B3CTL_UNCOUPLED_MIX50_C6A']['designation'],
                    pr['finalist']['selection_label']],
                   [17, 3, 0, 0.60002, 0.61314, 0.61533, '2026-10-17T17:35:23+00:00', 'control', 'EDGE_SELECTED']))
    sc.append(spot('S8: superseded interim record named B2 at 17:23:54Z (never committed)', 'AMD/PROMOTION.json -> superseded_records[0]',
                   [pr['superseded_records'][0]['utc'], pr['superseded_records'][0]['finalist'].split(' ')[0]], ['2026-10-17T17:23:54+00:00', 'B2_CSR_JT_C6A']))
    al = V.jload(AMD / 'ANALYSIS_LOCK.json')
    led = {r['entry']: r for r in csv.DictReader(open(AMD / 'EXPERIMENT_LEDGER.csv'))}
    sc.append(spot('S9: eligibility NEURAL_RAW, H_LJ, B2TWIN, TASKONLY true, PUB_IBSI false; PUB_IBSI seed-1 men guard 0.00501 (ledger 22, guard fail)',
                   'AMD/ANALYSIS_LOCK.json -> availability_ms_point_eligible; AMD/EXPERIMENT_LEDGER.csv entry 22 -> ms_guard_pass, ms_guard_men',
                   [{k: al['availability_ms_point_eligible'][k] for k in ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'TASKONLY', 'PUB_IBSI')},
                    led['22']['config_id'], led['22']['seed'], led['22']['ms_guard_pass'], led['22']['ms_guard_men'], al['utc']],
                   [{'NEURAL_RAW': True, 'H_LJ': True, 'B2TWIN': True, 'TASKONLY': True, 'PUB_IBSI': False}, 'IBSI_NLLM', '1', 'False', '0.00501',
                    '2026-10-17T17:42:54+00:00']))
    sc.append(spot('S2: m screen on CALIBRATION, ledger 31 says no MODEL_SELECT use before this score; screen unit ended 17:13:21Z',
                   'AMD/EXPERIMENT_LEDGER.csv entry 31 -> selection_effort; AMD/RESOURCE_ACCOUNTING.json -> units[AMD_B_diag2_S0_V1].utc',
                   ['no MODEL_SELECT use before this score' in led['31']['selection_effort'],
                    next(u['utc'] for u in V.jload(AMD / 'RESOURCE_ACCOUNTING.json')['units'] if u['unit_id'] == 'AMD_B_diag2_S0_V1')],
                   [True, '2026-10-17T17:13:21+00:00']))
    sc.append(spot('S3: alpha screen on CALIBRATION, edge-seeking rule with the MS offset (ledger 40); screen unit ended 17:21:01Z; ledger 40 at 17:23:36Z',
                   'AMD/EXPERIMENT_LEDGER.csv entry 40 -> selection_effort, utc; AMD/RESOURCE_ACCOUNTING.json -> units[AMD_B_diag3_S0_V1].utc',
                   ['largest alpha whose screen point' in led['40']['selection_effort'] and 'MS offset' in led['40']['selection_effort'], led['40']['utc'],
                    next(u['utc'] for u in V.jload(AMD / 'RESOURCE_ACCOUNTING.json')['units'] if u['unit_id'] == 'AMD_B_diag3_S0_V1')],
                   [True, '2026-10-17T17:23:36+00:00', '2026-10-17T17:21:01+00:00']))
    sc.append(spot('S4: B3CTL run adaptively after B3 failed the seed-0 MODEL_SELECT income Brier clause (ledger 41) and seed 1 after seed 0 (ledger 42)',
                   'AMD/EXPERIMENT_LEDGER.csv entries 41, 42 -> selection_effort, utc',
                   ['after B3' in led['41']['selection_effort'] and 'Brier failure' in led['41']['selection_effort'], 'after seed 0' in led['42']['selection_effort'],
                    led['41']['utc'], led['42']['utc']],
                   [True, True, '2026-10-17T17:24:25+00:00', '2026-10-17T17:24:52+00:00']))
    sl = V.jload(REP / 'SELECTION_LOCK.json')
    sc.append(spot('F6: NEURAL_RAW selected on MODEL_SELECT before any DEV_REPORT read (SELECTION_LOCK 2026-10-16T00:18:37Z, DEV_REPORT_labels_opened false)',
                   'REP/SELECTION_LOCK.json -> utc, DEV_REPORT_labels_opened', [sl['utc'], sl['DEV_REPORT_labels_opened']], ['2026-10-16T00:18:37+00:00', False]))
    sc.append(spot('S10: H_LJ fixed as the hybrids primary at 2026-10-16T02:46:48Z', 'HYB/ANALYSIS_LOCK.json -> utc, primary.service',
                   [V.jload(HYB / 'ANALYSIS_LOCK.json')['utc'], V.jload(HYB / 'ANALYSIS_LOCK.json')['primary']['service']], ['2026-10-16T02:46:48+00:00', 'H_LJ']))
    sc.append(spot('Section 6: FINALIST_LOCK written 17:37:02Z; ANALYSIS_LOCK 17:42:54Z carries the claims draft sha256 8389e591...',
                   'AMD/FINALIST_LOCK.json -> utc; AMD/ANALYSIS_LOCK.json -> utc, frozen_design_sha256, frozen_design_source',
                   [fl['utc'], al['utc'], al['frozen_design_sha256'][:8], al['frozen_design_source']],
                   ['2026-10-17T17:37:02+00:00', '2026-10-17T17:42:54+00:00', '8389e591', 'roles/statistical_reviewer/CLAIMS_AND_ENDPOINTS_DRAFT.md']))
    msu = [json.loads(l) for l in (V.ADM / 'access' / 'MODEL_SELECT_USE.jsonl').read_text().splitlines() if l.strip()]
    before = [e for e in msu if e['utc'] < pr['utc']]
    after = [e for e in msu if e['utc'] >= pr['utc']]
    sc.append(spot('Section 4: 24 logged MODEL_SELECT label reads before promotion (2 baseline events + 22 development evaluations); a 25th by the verifier at 17:52:58Z',
                   'admitted access/MODEL_SELECT_USE.jsonl (snapshot) -> utc, purpose',
                   [len(before), sum(e['purpose'].startswith('baseline') for e in before), sum(e['purpose'].startswith('development evaluation') for e in before),
                    len(after), [e['utc'] for e in after]],
                   [24, 2, 22, 1, ['2026-10-17T17:52:58+00:00']]))
    log = [json.loads(l) for l in C.SRC_ACCESS_LOG.read_text().splitlines() if l.strip()]
    reads = [e for e in log if str(e.get('event', '')).startswith('DEV_REPORT_TRUTH') and e.get('campaign') != C.CAMPAIGN]
    sc.append(spot('Section 4: 10 lineage DEV_REPORT truth reads before this campaign; first opening 2026-10-16T00:20:20Z; AMD coordinator 17:45:30Z; AMD verifier 17:56:47Z',
                   'lineage DEV_REPORT access log (events only) -> event, utc',
                   [len(reads), reads[0].get('utc'), next(e['utc'] for e in log if e.get('event') == 'DEV_REPORT_TRUTH_REREAD_BY_AMD_COORDINATOR'),
                    next(e['utc'] for e in log if e.get('event') == 'DEV_REPORT_TRUTH_REREAD_BY_AMD_INDEPENDENT_VERIFIER')],
                   [10, '2026-10-16T00:20:20+00:00', '2026-10-17T17:45:30+00:00', '2026-10-17T17:56:47+00:00']))
    snap = [json.loads(l) for l in (V.ADM / 'access' / 'DEV_REPORT_REREAD.jsonl').read_text().splitlines() if l.strip()]
    sc.append(spot('Section 6: the one AMD coordinator DEV_REPORT read at 17:45:30Z (admitted access snapshot)', 'admitted access/DEV_REPORT_REREAD.jsonl -> event, utc',
                   [[e.get('event'), e.get('utc')] for e in snap][:1], [['DEV_REPORT_TRUTH_REREAD_BY_AMD_COORDINATOR', '2026-10-17T17:45:30+00:00']]))
    out['spot_checks'] = sc
    out['spot_check_summary'] = {'n': len(sc), 'matched': sum(x['match'] for x in sc), 'mismatched': [x['claim'] for x in sc if not x['match']]}
    out['lineage_log_event_names'] = [e.get('event') for e in log]
    V.jdump(V.VPRIV / 'check2_exposure.json', out)
    print(json.dumps({'overlap': {k: v for k, v in out['household_overlap'].items() if k not in ('pairs',)}, 'spot': out['spot_check_summary']}, indent=1, default=str))
    for x in sc:
        if not x['match']:
            print('MISMATCH', x['claim'], '| found', x['found'], '| expected', x['expected_from_audit'])


if __name__ == '__main__':
    main()
