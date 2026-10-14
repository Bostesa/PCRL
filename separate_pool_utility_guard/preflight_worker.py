"""Queued real-role preflight before SCIENCE_LOCK: no model update, no metric reported.

Measures repeat-forward wire noise on the frozen UPDATE probe (sets the frozen
tolerance), times whole-pool GUARD/UPDATE gradient and metric passes, and checks
chunked-gradient determinism and equivalence on the actual admitted starts.
Pool loss values are not written; only timings and arithmetic differences.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path


def run(spec_path):
    from .evaluation_queue import queue_config
    active, spec, verify, guard = queue_config(spec_path, 'admission')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from task_conditioned_representation.models import load_checkpoint
    from . import guard as G
    from .wires import prepare_geometry
    from .train_worker import load_roles, probe_wire, _freeze, _params
    from .roles_worker import probe_key
    roles = load_roles(spec, verify)
    xU = torch.from_numpy(roles['UPDATE']['X']); yU = torch.from_numpy(roles['UPDATE']['occupation']).long()
    xG = torch.from_numpy(roles['GUARD']['X']); yG = torch.from_numpy(roles['GUARD']['occupation']).long()
    probe = sorted(range(len(xU)), key=lambda i: (probe_key(roles['UPDATE']['row_id'][i]), i))[:512]
    out = {'schema': 'SPG1-real-role-preflight-v1', 'queue_unit_id': active['unit_id'], 'seeds': {},
           'pool_loss_values_reported': False, 'model_updates': 0, 'VALIDATION_loaded': False}
    noise_max = {'representation': 0.0, 'centered_logits': 0.0, 'probabilities': 0.0}
    for seed in (0, 1):
        guard()
        c = spec['checkpoints'][str(seed)]; g = spec['geometry'][str(seed)]
        model, _ = load_checkpoint(verify(c['path'], c['sha256']), c['sha256'])
        geometry = prepare_geometry(json.loads(verify(g['path'], g['sha256']).read_text()), expected_sha256=g['binding_sha256'],
                                    source_seed=seed, checkpoint_sha256=c['sha256'])
        _freeze(model, geometry); params = _params(model)
        a = probe_wire(model, xU[probe], geometry); b = probe_wire(model, xU[probe], geometry)
        rep = {k: float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) for k in noise_max}
        rep['decision_changes'] = int((a['decision'] != b['decision']).sum())
        for k in noise_max:
            noise_max[k] = max(noise_max[k], rep[k])
        timings = {}
        for name, (x, y) in {'GUARD': (xG, yG), 'UPDATE': (xU, yU)}.items():
            t = time.process_time(); r1 = G.pool_pass(model, x, y, geometry, params); timings[f'{name}_gradient_pass_cpu_s'] = time.process_time() - t
            t = time.process_time(); G.pool_pass(model, x, y, geometry); timings[f'{name}_metric_pass_cpu_s'] = time.process_time() - t
            r2 = G.pool_pass(model, x, y, geometry, params)
            old = G.CHUNK; G.CHUNK = 10 ** 9
            try:
                r3 = G.pool_pass(model, x, y, geometry, params)
            finally:
                G.CHUNK = old
            timings[f'{name}_chunk_repeat_bitwise_equal'] = all(np.array_equal(r1['gradients'][k], r2['gradients'][k]) for k in ('nll', 'brier', 'ce'))
            timings[f'{name}_chunk_vs_single_max_relative_gradient_difference'] = max(
                float(np.max(np.abs(r1['gradients'][k] - r3['gradients'][k])) / np.max(np.abs(r3['gradients'][k]))) for k in ('nll', 'brier', 'ce'))
            timings[f'{name}_chunk_vs_single_metric_max_abs_difference'] = max(abs(r1[k] - r3[k]) for k in ('nll', 'brier', 'ce'))
            timings[f'{name}_correct_count_equal'] = r1['correct'] == r3['correct']
            timings[f'{name}_rows'] = r1['rows']
        out['seeds'][str(seed)] = {'repeat_forward_noise': rep, 'timing_and_equivalence': timings}
    out['wire_tolerance'] = {k: max(1e-10, 10 * v) for k, v in noise_max.items()}
    out['wire_tolerance_rule'] = 'max(1e-10, 10*max repeat-forward abs noise over both seeds), per coordinate block; any decision change counts'
    with Path(spec['output']).open('x') as stream:
        json.dump(out, stream, indent=2, sort_keys=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)
