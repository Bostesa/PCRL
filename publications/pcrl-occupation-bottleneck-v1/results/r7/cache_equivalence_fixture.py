"""Queue-only synthetic check that envelope caching preserves assessment bytes."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path


def main():
    # ROOT registers this fixture as a verification unit. No numerical import
    # occurs before the inherited queue slot is authenticated.
    from occupation_bottleneck.scheduler import assert_active_from_env
    assert_active_from_env(stage='verification')
    import numpy as np
    from occupation_bottleneck_independent import replay
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    group = np.r_[np.arange(64), 63]
    rows = np.arange(65)
    y = group % 2
    table = {'row_id': rows, 'unit': group, 'sex': y,
             'raw1': .15+.3*y+.007*group,
             'raw2': .8-.5*y-.002*group,
             'candidate1': .2+.15*y+.005*group,
             'candidate2': .65-.3*y-.001*group}
    arrays = {'truth': {'path': 'synthetic-truth', 'sha256': '0'*64},
              'scores': {'path': 'synthetic-scores', 'sha256': '1'*64}}
    endpoints = []
    for index in range(4):
        parts = [{'truth_table': 'truth', 'truth_key': 'sex',
                  'left_table': 'scores', 'left_keys': ['raw1', 'raw2'],
                  'right_table': 'scores', 'right_keys': ['candidate1', 'candidate2'],
                  'sign': 1 if index % 2 == 0 else -1} for _ in range(2)]
        endpoints.append({'endpoint_id': 'e'+str(index),
                          'kind': 'auc_envelope_difference', 'threshold': .02,
                          'direction': 'at_least', 'paired_seeds': parts})
    registry = {'K': 4, 'replicates': 64, 'bootstrap_seed': 2026101301,
                'alpha': .05, 'expected_rows': 65, 'expected_groups': 64,
                'endpoints': endpoints}

    class SyntheticInputs:
        cfg = {'endpoint_registry': 'registry', 'unit_input': 'units', 'arrays': arrays}
        def seal(self):
            assert_active_from_env(stage='verification')
        def json(self, item):
            assert item == 'registry'
            return registry
        def arrays(self, item):
            return {'row_id': rows, 'unit': group} if item == 'units' else table

    source = inspect.getsource(replay.assessment)
    cache_line = 'if identity in envelope_cache:return envelope_cache[identity]'
    if cache_line not in source:
        raise AssertionError('pending assessment cache not applied')
    source = source.replace(cache_line, 'if False:return envelope_cache[identity]')
    namespace = dict(vars(replay))
    exec(source, namespace)
    original_auc = replay.weighted_auc
    calls = {'cached': 0, 'uncached': 0}
    phase = 'cached'
    def count_auc(*args, **kwargs):
        calls[phase] += 1
        return original_auc(*args, **kwargs)
    replay.weighted_auc = count_auc
    namespace['weighted_auc'] = count_auc
    try:
        cached = replay.assessment(SyntheticInputs())
        phase = 'uncached'
        uncached = namespace['assessment'](SyntheticInputs())
    finally:
        replay.weighted_auc = original_auc
    cached_json = json.dumps(cached, sort_keys=True, allow_nan=False)
    uncached_json = json.dumps(uncached, sort_keys=True, allow_nan=False)
    if cached_json != uncached_json or calls['cached'] >= calls['uncached']:
        raise AssertionError('cache changes output or does not reduce repeated AUC work')
    result = {'schema': 'OB1-independent-cache-equivalence-fixture-v1',
              'status': 'PASS', 'synthetic_only': True,
              'result_json_exact_equality': True, 'weighted_auc_calls': calls,
              'shared_group_draws_preserved': True, 'seed_sign_and_gate_arithmetic_preserved': True,
              'real_data_read': False, 'scientific_fits': 0}
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
