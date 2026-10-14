"""Merge Stage B unit outputs (JSON only) + B3 label and queue accounting; write detail and public summary."""
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
PRIV = W / '.private/minimal_task_wire_adapter'
RES = W / 'results/pcrl_minimal_task_wire_adapter_v1'


def main(b1_units, b2_unit, stage_a_public):
    b1 = {s: json.loads((PRIV / f'independent/{u}.json').read_text()) for s, u in zip((0, 1), b1_units)}
    b2 = json.loads((PRIV / f'independent/{b2_unit}.json').read_text())
    sa = json.loads((RES / stage_a_public).read_text())
    checks = {}

    def add(name, ok, **nums):
        checks[name] = {'status': 'PASS' if ok else 'FAIL', **nums}
    fits = []
    for s, r in b1.items():
        assert 'error' not in r, r.get('error')
        for f in r['fits']:
            fits.append(dict(f, seed=s))
    add('B1_reader_refits', len(fits) <= 36 and all(f['orientation_mine'] == f['orientation_campaign'] and abs(f['att_val_auc_mine'] - f['att_val_auc_campaign']) <= 1e-12
                                                    and abs(f['inner_auc_mine'] - f['inner_auc_from_campaign_scores']) <= 1e-9 for f in fits),
        fits=len(fits), max_att_val_auc_diff=max(abs(f['att_val_auc_mine'] - f['att_val_auc_campaign']) for f in fits),
        max_inner_auc_diff=max(abs(f['inner_auc_mine'] - f['inner_auc_from_campaign_scores']) for f in fits),
        max_inner_score_diff=max(f['inner_score_maxabs_vs_campaign'] for f in fits),
        refits=[{k: f[k] for k in ('seed', 'service', 'view', 'reader', 'inner_auc_mine', 'inner_auc_from_campaign_scores')} for f in fits],
        tolerance='AUC 1e-9 (sklearn same seed; observed identical)')
    add('B1_own_wires_equal_campaign_INNER_release', all(f['wire_check']['r1_maxabs_vs_campaign_release'] <= 1e-12 and f['wire_check']['r2_maxabs_vs_campaign_release'] <= 1e-12
                                                         and f['wire_check']['r1_decisions_equal'] and f['wire_check']['r2_decisions_equal'] for f in fits),
        max_r1=max(f['wire_check']['r1_maxabs_vs_campaign_release'] for f in fits), max_r2=max(f['wire_check']['r2_maxabs_vs_campaign_release'] for f in fits),
        services=sorted({f['service'] for f in fits}))
    assert 'error' not in b2, b2.get('error')
    g = b2['control_gates']
    add('B2_control_gates', b2['control_gates_all_passed'] and all(v['passed'] == v['campaign_passed'] and v['maxabs_vs_campaign'] <= 1e-12 for v in g.values()),
        gates={k: v['values'] for k, v in g.items()})
    p = b2['primary']
    add('B2_primary_endpoints_K39', p['n_endpoints'] == 39 and p['endpoint_ids_match'] and not p['status_mismatches'] and max(p['maxabs_vs_campaign'].values()) <= 1e-12,
        status_counts=p['status_counts'], maxabs_vs_campaign=p['maxabs_vs_campaign'], critical=p['critical'])
    d = b2['diagnostic']
    add('B2_diagnostic_endpoints_K45', d['n_endpoints'] == 45 and not d['status_mismatches'] and d['maxabs_vs_campaign'] <= 1e-12,
        status_counts=d['status_counts'], maxabs_vs_campaign=d['maxabs_vs_campaign'], critical=d['critical'])
    add('B2_service_table_complete_AUCs', b2['service_table_complete_auc_maxabs'] <= 1e-12, maxabs=b2['service_table_complete_auc_maxabs'])
    # B3 label
    sel = json.loads((RES / 'SELECTION_LOCK.json').read_text())
    a_ok = sa['overall'] == 'PASS'
    nominee = sel['selection']['J_TRUST']['selected_epoch'] if 'selection' in sel and 'J_TRUST' in sel['selection'] else None
    privacy_driven = all(sa['checks'][f'A3_S{s}_privacy_changes_displacement_and_wire']['status'] == 'PASS' for s in (0, 1))
    eps = {r['endpoint_id']: r['status'] for r in p['endpoints']}
    priv = {k: v for k, v in eps.items() if k.startswith('privacy')}
    util = {k: v for k, v in eps.items() if k.startswith('utility')}
    verified = a_ok and all(c['status'] == 'PASS' for c in checks.values())
    if not verified:
        label = 'INCOMPLETE_EXECUTION_OR_VERIFICATION'
    elif not nominee or not privacy_driven:
        label = 'NO_POINT_FEASIBLE_NOMINEE_WITH_PRIVACY_DIAGNOSIS'
    elif all(v == 'PASS' for v in eps.values()):
        label = 'DEVELOPMENT_BANK_CRITERION_ESTABLISHED'
    elif all(v == 'PASS' for v in priv.values()) and not any(v == 'ESTABLISHED_VIOLATION' for v in util.values()):
        label = 'POINT_FEASIBLE_PRIVACY_LEAD_UTILITY_UNRESOLVED'
    else:
        label = 'POINT_FEASIBLE_NO_COMPETITIVE_CRITERION'
    decision = (RES / 'RESEARCH_DECISION.md').read_text()
    line = next((l for l in decision.splitlines() if l.startswith('**Scientific label:**')), '')
    campaign_label = next((x for x in ('INCOMPLETE_EXECUTION_OR_VERIFICATION', 'NO_POINT_FEASIBLE_NOMINEE_WITH_PRIVACY_DIAGNOSIS',
                                       'POINT_FEASIBLE_NO_COMPETITIVE_CRITERION', 'POINT_FEASIBLE_PRIVACY_LEAD_UTILITY_UNRESOLVED',
                                       'DEVELOPMENT_BANK_CRITERION_ESTABLISHED') if x in line), None)
    add('B3_scientific_label', label == campaign_label, independent_label=label, campaign_label=campaign_label, J_TRUST_nominee_epoch=nominee,
        privacy_driven_update_verified_each_seed=privacy_driven,
        privacy_endpoint_statuses=priv, utility_status_counts={k: sum(v == k for v in util.values()) for k in ('PASS', 'UNRESOLVED', 'ESTABLISHED_VIOLATION')},
        rule_reading='labels in SCIENCE_LOCK have no formal rule text; applied: verification failure->INCOMPLETE; no J_TRUST nominee or no verified privacy-driven update->NO_POINT_FEASIBLE; all 39 PASS->ESTABLISHED; all 15 privacy PASS and no utility violation->PRIVACY_LEAD; else NO_COMPETITIVE_CRITERION')
    # queue accounting
    st = json.loads((PRIV / 'compute/state.json').read_text())
    units = st['units']
    per_stage = {}
    for u in units.values():
        per_stage[u['spec']['stage']] = per_stage.get(u['spec']['stage'], 0.0) + float(u.get('charged_cpu_s', 0.0))
    charge_total = sum(float(c['cpu_s']) for c in st['charges'].values()) if st['charges'] else 0.0
    stage_ok = all(abs(per_stage.get(k, 0.0) - v) <= 0.0005 * len(units) for k, v in st['used_by_stage_s'].items() if k != 'channel' or v)
    running = [k for k, u in units.items() if u['status'] in ('running', 'queued')]
    failed = {k: u['status'] for k, u in units.items() if u['status'] not in ('done',)}
    add('B3_queue_accounting', abs(sum(per_stage.values()) + charge_total - st['used_cpu_s']) <= 0.0005 * len(units) and stage_ok and st['used_cpu_s'] <= st['cpu_limit_s'],
        tolerance='unit charges are recorded to 1 ms; sums compared within 0.5 ms per unit', units=len(units), used_cpu_s=st['used_cpu_s'], cpu_limit_s=st['cpu_limit_s'], per_stage_from_units=per_stage, used_by_stage_s=st['used_by_stage_s'],
        nondone_units=failed, running_or_queued_at_check=running,
        note='non-done units are superseded failed attempts (MTA_ADMISSION_V1, MTA_ENGINEERING_TESTS_V1) and MTAV_B2_V1 (my config-key collision, rerun as MTAV_B2_V2); no data outputs from failed units are used')
    import numpy, scipy, sklearn, torch, platform
    overall = 'PASS' if all(c['status'] == 'PASS' for c in checks.values()) else 'FAIL'
    public = {'schema': 'MTAV-stage-B-v1', 'campaign': 'pcrl_minimal_task_wire_adapter_v1', 'overall': overall, 'checks': checks,
              'queue_units': list(b1_units) + [b2_unit],
              'versions': {'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__, 'sklearn': sklearn.__version__},
              'scope': {'B1': '18 refits (<=36): per seed r1 strongest (R1 bank) + r2/pair strongest for J_TRUST E10, RAW_PROJ, TASK_TRUST E10 + pair strongest for PRIOR_GLOBAL_UPDATE_GUARD_E5 (hybrid) and _OWNPROJ; own raw83 wires for ATT(4064)/INNER(2235); own split/scaler/orientation/AUC; sklearn estimators from slate definitions, random_state=2026101205+seed',
                        'B2': 'from frozen INNER_SCORES/INNER_RELEASE (24 banks) + INNER labels/references: control gates (6), primary K=39, diagnostic K=45, service-table complete AUCs; own tie-aware weighted AUC, envelopes (18; pair 54 incl. IGNORE aliases) reselected per draw, duplicate-group bootstrap seed 2026101301 x2048, Bonferroni Student-t',
                        'B3': 'scientific label recomputed from independent A/B results; queue state accounting'},
              'private_detail': '$PRIV/independent/STAGE_B_DETAIL.json'}
    (PRIV / 'independent/STAGE_B_DETAIL.json').write_text(json.dumps({'public': public, 'b1': b1, 'b2': b2}, indent=1, sort_keys=True, default=float))
    (RES / 'independent/STAGE_B.json').write_text(json.dumps(public, indent=1, sort_keys=True, default=float))
    print(overall)
    for k, v in checks.items():
        if v['status'] != 'PASS':
            print('FAIL', k, json.dumps(v, default=float)[:800])


if __name__ == '__main__':
    main(sys.argv[1:3], sys.argv[3], 'independent/STAGE_A.json')
