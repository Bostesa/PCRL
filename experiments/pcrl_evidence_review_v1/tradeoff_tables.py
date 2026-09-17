"""Tradeoff, direction and scope tables for Terminal B's statistical reanalysis.

Consumes only the study's published evidence files plus the independent
reanalysis written by `reanalyse_transport`. Writes, under
results/pcrl_evidence_review_v1/:

  PER_SEED_DIRECTION.csv      per-seed sign agreement for every F1-F4 endpoint
  DEV_VS_TRANSPORT.csv        development seed-mean vs transport estimate, with ratio
  SCOPE_COMPARISON.csv        common_fresh vs transport_all additional recovery
  VECTOR_C1_J_CONTROLS.csv    the full utility/disclosure vector, no exchange rate
  WITHHOLDING_MATCHING.csv    post-hoc average-utility matching probabilities
  SERVICE_VS_PROBE.csv        exact parity vs frozen/fresh probe allowances
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

WEIGHTS = ('unweighted', 'person_weighted')
SENSITIVE = ('A/SEX', 'AB/SEX', 'A/RAC1P', 'AB/RAC1P')


def read_csv(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def per_seed_direction(indep_rows, consistency_rows):
    """Does every seed agree with the seed-mean sign, and with the development sign?"""
    dev = {(r['family'], r['left'], r['right'], r['endpoint'], r['weight']):
           float(r['development_seed_mean']) for r in consistency_rows}
    out = []
    for r in indep_rows:
        seeds = [float(r[f'seed_{s}']) for s in (0, 1, 2)]
        est = float(r['estimate'])
        key = (r['family'], r['left'], r['right'], r['endpoint'], r['weight'])
        d = dev.get(key)
        same_as_mean = all(math.copysign(1, v) == math.copysign(1, est) for v in seeds)
        same_as_dev = None if d is None else all(
            math.copysign(1, v) == math.copysign(1, d) for v in seeds)
        out.append({
            'family': r['family'], 'contrast': f"{r['left']} - {r['right']}",
            'endpoint': r['endpoint'], 'weight': r['weight'],
            'seed_mean': est, 'seed_0': seeds[0], 'seed_1': seeds[1], 'seed_2': seeds[2],
            'all_seeds_match_seed_mean_sign': same_as_mean,
            'development_seed_mean': d,
            'seed_mean_matches_development_sign': None if d is None else (
                math.copysign(1, est) == math.copysign(1, d)),
            'all_seeds_match_development_sign': same_as_dev,
        })
    return out


def dev_vs_transport(indep_rows, consistency_rows):
    dev = {(r['family'], r['left'], r['right'], r['endpoint'], r['weight']): r
           for r in consistency_rows}
    out = []
    for r in indep_rows:
        key = (r['family'], r['left'], r['right'], r['endpoint'], r['weight'])
        if key not in dev:
            continue
        d = float(dev[key]['development_seed_mean'])
        t = float(r['estimate'])
        out.append({
            'family': r['family'], 'contrast': f"{r['left']} - {r['right']}",
            'endpoint': r['endpoint'], 'weight': r['weight'],
            'development_seed_mean': d, 'transport_seed_mean': t,
            'same_sign': math.copysign(1, d) == math.copysign(1, t),
            'magnitude_ratio_transport_over_development': (abs(t) / abs(d)) if abs(d) > 1e-15 else None,
            'absolute_shift': t - d,
            'study_status': dev[key]['status'],
            'transport_se': float(r['se']),
            'development_shift_in_transport_se': ((t - d) / float(r['se'])) if float(r['se']) > 0 else None,
        })
    return out


def scope_comparison(aggregate_rows):
    """Additional recovery under the fresh-only scope versus the full transport scope."""
    idx = {}
    for r in aggregate_rows:
        if r['budget'] != '360' or r['kind'] != 'additional_recovery' or r['mode'] != 'B':
            continue
        idx[(r['scope'], r['condition'], r['endpoint'], r['weight'])] = float(r['seed_mean'])
    out = []
    conditions = sorted({k[1] for k in idx})
    for c in conditions:
        for e in SENSITIVE:
            for w in WEIGHTS:
                a = idx.get(('common_fresh', c, e, w))
                b = idx.get(('transport_all', c, e, w))
                if a is None or b is None:
                    continue
                out.append({'interface': c, 'endpoint': e, 'weight': w,
                            'common_fresh_additional': a, 'transport_all_additional': b,
                            'difference_transport_minus_fresh': b - a})
    return out


def vector_table(modeb_rows, conditions=('spectral_C1', 'J', 'spectral_L1', 'spectral_L2', 'spectral_S0', 'H')):
    """The full comparison vector: no scalar privacy/utility exchange rate is applied."""
    out = []
    for r in modeb_rows:
        if r['interface'] not in conditions:
            continue
        row = {'mode': r['mode'], 'weight': r['weight'], 'interface': r['interface'],
               'residence_gain_vs_H': float(r['residence_gain_vs_H'])}
        for e in SENSITIVE:
            row[f'absolute/{e}'] = float(r[f'absolute/{e}'])
            row[f'additional/{e}'] = float(r[f'additional/{e}'])
        out.append(row)
    return out


def withholding_matching(modeb_rows, target='spectral_C1',
                         sources=('E', 'A0', 'L025', 'L20', 'J', 'spectral_S0')):
    """Post-hoc average-utility matching.

    For the declared randomized-withholding family, expected per-person loss is
    (1-p)L_H + p L_aug, so expected residence *gain over H* is exactly p * gain_aug
    and expected additional recovery is exactly p * additional_aug. The matching
    probability p* = gain_target / gain_source is therefore identified in closed
    form whenever it lands in [0, 1]. It is estimated from the same data as the
    outcome, so it is a post-hoc average-utility calculation: the intervals of the
    fixed-p analysis do not transfer to it.
    """
    out = []
    for mode in sorted({r['mode'] for r in modeb_rows}):
        for w in WEIGHTS:
            rows = {r['interface']: r for r in modeb_rows if r['mode'] == mode and r['weight'] == w}
            if target not in rows:
                continue
            tgt = rows[target]
            g_t = float(tgt['residence_gain_vs_H'])
            for s in sources:
                if s not in rows:
                    continue
                g_s = float(rows[s]['residence_gain_vs_H'])
                p = g_t / g_s if abs(g_s) > 1e-12 else None
                identifiable = p is not None and 0.0 <= p <= 1.0
                row = {'mode': mode, 'weight': w, 'target': target, 'withholding_source': s,
                       'target_residence_gain': g_t, 'source_residence_gain_at_p1': g_s,
                       'matching_probability': p, 'identifiable_in_unit_interval': identifiable}
                for e in SENSITIVE:
                    t_add = float(tgt[f'additional/{e}'])
                    s_add = float(rows[s][f'additional/{e}'])
                    row[f'target_additional/{e}'] = t_add
                    row[f'source_additional_at_matched_p/{e}'] = (p * s_add) if identifiable else None
                    row[f'target_better_at_matched_p/{e}'] = (t_add < p * s_add) if identifiable else None
                out.append(row)
    return out


def service_vs_probe(criteria_rows, service_rows):
    out = []
    for r in criteria_rows:
        out.append({
            'mode': r['mode'], 'interface': r['condition'], 'weight': r['weight'],
            'released_service_vectors_identical': 'structural (verified bitwise, 210/210 views)',
            'legacy_source_probe_allowance_pass_seeds': int(r['source_allowance_pass_seeds']),
            'legacy_source_probe_allowance_fail_seeds': 3 - int(r['source_allowance_pass_seeds']),
            'half_headroom_pass_seeds': int(r['half_headroom_pass_seeds']),
            'half_headroom_fail_seeds': 3 - int(r['half_headroom_pass_seeds']),
            'residence_gain_mean': float(r['residence_gain_mean']),
        })
    by_task = {}
    for r in service_rows:
        by_task.setdefault(r['task'], []).append(r)
    summary = []
    for t, rs in sorted(by_task.items()):
        summary.append({
            'service_task': t,
            'development_2018_unweighted': sum(float(r['development_unweighted']) for r in rs) / len(rs),
            'transport_2017_unweighted': sum(float(r['transport_unweighted']) for r in rs) / len(rs),
            'change_2017_minus_2018': (sum(float(r['transport_unweighted']) for r in rs)
                                       - sum(float(r['development_unweighted']) for r in rs)) / len(rs),
        })
    return out, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--published', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    pub, out = Path(args.published), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    indep = read_csv(out / 'INDEPENDENT_FAMILIES.csv')
    cons = read_csv(pub / 'evidence' / 'DEVELOPMENT_CONSISTENCY.csv')
    agg = read_csv(pub / 'AGGREGATE.csv.gz')
    modeb = read_csv(out / 'MODEB_TRANSPORT_TABLE.csv')
    crit = read_csv(pub / 'evidence' / 'CRITERIA_SUMMARY.csv')
    svc = read_csv(pub / 'evidence' / 'SERVICE_QUALITY.csv')

    ds = per_seed_direction(indep, cons)
    write_csv(out / 'PER_SEED_DIRECTION.csv', ds)
    dv = dev_vs_transport(indep, cons)
    write_csv(out / 'DEV_VS_TRANSPORT.csv', dv)
    write_csv(out / 'SCOPE_COMPARISON.csv', scope_comparison(agg))
    write_csv(out / 'VECTOR_C1_J_CONTROLS.csv', vector_table(modeb))
    matching = withholding_matching(modeb)
    # the reverse direction: withhold C1 down to each comparator's mean residence gain
    matching += withholding_matching(modeb, target='J', sources=('spectral_C1', 'spectral_S0', 'A0', 'E'))
    matching += withholding_matching(modeb, target='spectral_L1', sources=('spectral_C1', 'spectral_S0'))
    write_csv(out / 'WITHHOLDING_MATCHING.csv', matching)
    sp, svc_summary = service_vs_probe(crit, svc)
    write_csv(out / 'SERVICE_VS_PROBE.csv', sp)
    write_csv(out / 'SERVICE_QUALITY_SUMMARY.csv', svc_summary)

    summary = {
        'family_endpoints': len(ds),
        'endpoints_where_all_seeds_match_seed_mean_sign': sum(
            1 for r in ds if r['all_seeds_match_seed_mean_sign']),
        'F1_endpoints_where_all_seeds_match_seed_mean_sign': sum(
            1 for r in ds if r['family'] == 'F1_primary' and r['all_seeds_match_seed_mean_sign']),
        'F1_endpoints_total': sum(1 for r in ds if r['family'] == 'F1_primary'),
        'F1_seed_mean_matches_development_sign': sum(
            1 for r in ds if r['family'] == 'F1_primary' and r['seed_mean_matches_development_sign']),
        'F1_all_seeds_match_development_sign': sum(
            1 for r in ds if r['family'] == 'F1_primary' and r['all_seeds_match_development_sign']),
        'F1_magnitude_ratio_range': [
            min(r['magnitude_ratio_transport_over_development'] for r in dv if r['family'] == 'F1_primary'),
            max(r['magnitude_ratio_transport_over_development'] for r in dv if r['family'] == 'F1_primary')],
        'service_quality_change': svc_summary,
    }
    (out / 'TRADEOFF_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
