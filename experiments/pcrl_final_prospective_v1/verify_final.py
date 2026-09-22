"""Independent post-lock replay of the complete evaluated panel."""
from __future__ import annotations

import argparse
import json

from . import inference_panel as ip
from .audit_panel import all_units
from .common import ANCHORS, OUT, atomic_json, now
from .independent_replay import TOL, bootstrap_se, point_estimates, replay
from .pipeline import root_for
from .prepare import load_pools


def final_inputs(dataset):
    features, labels = {}, None
    for anchor in ANCHORS:
        data, encoded = load_pools(dataset, anchor, ('final',))
        d, e = data['final'], encoded['final']
        if labels is None:
            labels = d['labels']
        features[anchor] = {k: d[k] for k in ('x', 'ha', 'hb', 'J', 'ids', 'households', 'weights')}
        features[anchor].update({'p': e['p'], 'b': e['b'], 'r': e['r'],
                                 'codes_T0': e['codes']['T0'], 'actions17': e['actions'][17],
                                 'actions33': e['actions'][33], 'global_offsets': e['global_offsets']})
    return features, labels


def main(dataset='acs2016', bootstrap_draws=500):
    root = root_for(dataset)
    features, labels = final_inputs(dataset)
    unit_rows = replay(root, features, labels, all_units())
    endpoints = ip.primary_endpoints()+ip.secondary_endpoints()+ip.descriptive_endpoints()
    points, cache = point_estimates(root, endpoints)
    inference = json.loads((root/'INFERENCE.json').read_text())
    reported = {row['id']: row['estimate'] for result in inference['primary'].values() for row in result['clauses']}
    reported.update({row['id']: row['estimate'] for row in inference['secondary']['rows']})
    reported.update({row['id']: row['estimate'] for row in inference['descriptive']})
    if set(points) != set(reported):
        raise ValueError('Independent replay endpoint IDs differ from registered inference')
    max_point_error = max(abs(points[key]['estimate']-reported[key]) for key in points)
    approximate_se = bootstrap_se(ip.primary_endpoints(), cache, n_boot=bootstrap_draws)
    primary_se = {row['id']: row['bootstrap_se'] for result in inference['primary'].values()
                  for row in result['clauses']}
    record = {'created_utc': now(), 'dataset': dataset, 'units': len(unit_rows),
              'rows_over_prediction_tolerance': sum(row['rows_over_tolerance'] for row in unit_rows),
              'max_abs_loss_error': max(row['max_abs_loss_error'] for row in unit_rows),
              'prediction_loss_tolerance': TOL,
              'all_unit_replays_within_tolerance': all(row['rows_over_tolerance'] == 0 for row in unit_rows),
              'endpoints': len(points), 'max_abs_point_estimate_error': max_point_error,
              'all_point_estimates_match_1e_10': max_point_error <= 1e-10,
              'primary_se_sanity': {key: {'registered_10000_draw_se': primary_se[key],
                                          'independent_se': value,
                                          'draws': bootstrap_draws}
                                    for key, value in approximate_se.items()},
              'unit_rows': unit_rows}
    atomic_json(OUT/'private/INDEPENDENT_REPLAY_2016.json', record)
    if not record['all_unit_replays_within_tolerance'] or not record['all_point_estimates_match_1e_10']:
        raise ValueError('Independent replay differs from stored panel')
    return record


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='acs2016')
    ap.add_argument('--bootstrap-draws', type=int, default=500)
    args = ap.parse_args()
    result = main(args.dataset, args.bootstrap_draws)
    print({k: result[k] for k in ('units', 'rows_over_prediction_tolerance',
                                  'max_abs_loss_error', 'endpoints', 'max_abs_point_estimate_error')})
