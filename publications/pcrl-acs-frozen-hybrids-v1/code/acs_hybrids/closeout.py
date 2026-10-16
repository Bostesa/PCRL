"""Closeout records: forecast scores, reader accounting and resource inventory (aggregates only)."""
from __future__ import annotations
import argparse
import csv
import json

from . import common as C
from . import jobqueue


def forecast_scores():
    f = json.loads((C.RESULTS / 'FORECASTS.json').read_text())
    d = json.loads((C.RESULTS / 'DEV_REPORT_RESULTS.json').read_text())
    E, E2 = d['endpoints_primary'], d['endpoints_secondary']
    pair = {s: d['per_seed'][f'S{s}']['services'] for s in C.SEEDS}
    inc = [E[f'INCREMENT|H_LJ_over_H_LL|coalition_gain|s{s}'] for s in C.SEEDS]
    inter = [E2[f'SEC|INTERACTION|pair|s{s}'] for s in C.SEEDS]
    rl = json.loads((C.RESULTS / 'REPORTING_LOCK.json').read_text())
    outcomes = {
        'F1': d['claims']['A_RECIPE_BENEFIT']['decision'] == 'RECIPE_BENEFIT_ESTABLISHED',
        'F2': d['claims']['B_COMPETITIVE']['decision'] == 'SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL',
        'F3': d['claims']['B_COMPETITIVE']['nested_material_superiority_0.020'] == 'MATERIALLY_SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL',
        'F4': all(e['point'] > 0 for e in inc),
        'F5': all(e['family_lower'] > 0 for e in inc),
        'F8': inter[0]['point'] > 0,
        'F9': all(abs(e['point']) < 0.02 for e in inter),
        'F10': bool(d['controls']['CONTROLS_VALID'] and rl['stage_1_controls']['CONTROLS_STAGE1_VALID']),
        'F11': all(rl['readers'][f'S{s}']['replay_vs_source']['bitwise'] == rl['readers'][f'S{s}']['replay_vs_source']['compared'] for s in C.SEEDS),
        'F12': d['central_question']['overall'] == 'H_LJ_FIXED_COMPOSITION_SUPPORTED',
        'F13': d['claims']['C_SUBGROUP']['decision'] == 'SUBGROUP_UNRESOLVED'}
    rows, briers = [], []
    for fc in f['forecasts']:
        if 'probability' in fc:
            o = outcomes[fc['id']]; b = (fc['probability'] - float(o)) ** 2; briers.append(b)
            rows.append({'id': fc['id'], 'event': fc['event'], 'probability': fc['probability'], 'outcome': bool(o), 'brier': round(b, 4)})
        else:
            svc = 'H_LJ' if fc['id'] == 'F6' else 'H_JL'
            pts = [pair[s][svc]['recovery']['pair']['auc'] for s in C.SEEDS]
            lo, hi = fc['interval_80']
            rows.append({'id': fc['id'], 'event': fc['event'], 'median': fc['median'], 'interval_80': fc['interval_80'], 'observed_points': pts,
                         'covered_both_seeds': all(lo <= p <= hi for p in pts), 'abs_error_of_median': [round(abs(p - fc['median']), 4) for p in pts]})
    out = {'schema': 'ACSH1-forecast-scores-v1', 'utc': C.now(), 'forecasts_sha256': C.sha(C.RESULTS / 'FORECASTS.json'), 'scores': rows,
           'mean_brier_binary': round(sum(briers) / len(briers), 4), 'n_binary': len(briers),
           'note': 'F13 was already determined by published values (settled before readers) and is not informative'}
    (C.RESULTS / 'FORECAST_SCORES.json').write_text(json.dumps(out, indent=2))
    return out


def reader_accounting():
    rows, summary = [], {}
    for s in C.SEEDS:
        rr = json.loads((C.PRIV / 'runs' / f'READERS_{s}' / 'READERS_SEED.json').read_text())
        summary[f'S{s}'] = {'fit_counts': rr['fit_counts'], 'aliases': rr['aliases'], 'reader_model_seed': rr['reader_model_seed'],
                            'replay_vs_source': {k: rr['replay_vs_source'][k] for k in ('compared', 'bitwise', 'max_abs_diff')},
                            'replay_winner_agreement': sum(v['equal'] for v in rr['replay_vs_source']['winners'].values()),
                            'cpu_s_total': round(sum(a['cpu_s'] for a in rr['accounting']), 2)}
        for wid, mb in rr['marginal_banks'].items():
            for name, r in mb['readers'].items():
                rows.append([s, 'marginal', wid, '|'.join(mb['services_aliasing_this_bank']), name, round(r['attack_validation_auc'], 6), r['orientation'], 'fit once; aliased' if len(mb['services_aliasing_this_bank']) > 1 else 'fit once'])
        for svc, x in rr['services'].items():
            for name, r in x['pair_readers'].items():
                rows.append([s, 'pair', svc, svc, name, round(r['attack_validation_auc'], 6), r['orientation'], 'fresh pair fit'])
            for view, w in x['winner'].items():
                rows.append([s, f'winner:{view}', svc, svc, w['candidate'], round(w['attack_validation_auc'], 6), None, f"reader_id {w['reader_id']}"])
        for view, c in rr['controls'].items():
            rows.append([s, f'control:{view}', 'H_LJ wire', 'H_LJ', 'OUT_HGB shuffled-null / planted-bit', round(c['shuffled_attack_validation_auc'], 6), None,
                         f"planted ATTACK_VALIDATION AUC {c['planted_attack_validation_auc']:.4f}"])
    with open(C.RESULTS / 'READER_ACCOUNTING.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['seed', 'kind', 'wire_or_service', 'services', 'reader', 'attack_validation_auc', 'orientation', 'note']); w.writerows(rows)
    out = {'schema': 'ACSH1-reader-accounting-v1', 'per_seed': summary, 'effective_numerical_fits_total': sum(v['fit_counts']['effective_numerical_fits'] for v in summary.values()),
           'note': ('marginal readers were fitted once per unique single-recipient wire and aliased to every service sharing that wire; aliased arrays are not '
                    'independent experiments; selection and orientation used ATTACK_VALIDATION only')}
    (C.RESULTS / 'READER_ACCOUNTING.json').write_text(json.dumps(out, indent=2))
    return out


def resource_inventory():
    s = jobqueue.summary(); es = jobqueue.entries()
    ends = [e for e in es if e.get('event') == 'unit_end']
    peak = max((e.get('sampled_peak_rss_bytes') or 0) for e in ends) if ends else 0
    out = {'schema': 'ACSH1-resource-inventory-v1', 'utc': C.now(), 'campaign_start_utc': jobqueue.CAMPAIGN_START_UTC,
           'budget': {'cpu_hours': 6, 'elapsed_hours': 6, 'usd': 0, 'heavy_jobs_at_once': 2, 'threads_per_job': 1, 'combined_rss_gib': 4},
           'queue': {'units_done': s['units_done'], 'units_failed': s['units_failed'], 'failed_units': s['failed_units'], 'charged_cpu_h_queue_and_agent': s['charged_cpu_h']},
           'units': [{'unit': e['unit_id'], 'stage': e['stage'], 'status': e['status'], 'cpu_s': e['charged_cpu_s'], 'wall_s': e['wall_s'],
                      'peak_rss_mib': round((e.get('sampled_peak_rss_bytes') or 0) / 2 ** 20)} for e in ends],
           'peak_unit_rss_mib': round(peak / 2 ** 20), 'paid_compute_usd': 0, 'cloud': 'none'}
    (C.RESULTS / 'RESOURCE_INVENTORY.json').write_text(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('what', choices=['forecasts', 'readers', 'resources', 'all'])
    a = ap.parse_args()
    if a.what in ('forecasts', 'all'):
        print(json.dumps({k: v for k, v in forecast_scores().items() if k != 'scores'}))
    if a.what in ('readers', 'all'):
        print(json.dumps({'effective_fits': reader_accounting()['effective_numerical_fits_total']}))
    if a.what in ('resources', 'all'):
        r = resource_inventory(); print(json.dumps(r['queue']))
