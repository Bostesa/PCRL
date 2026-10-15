"""Queued source/role admission, two-recipient zero-residual witness and immutable bindings.

No model is updated and no label-dependent metric is computed; no label is a
source of this unit. Writes ADAPTER_BINDING_S{seed}.json (per-recipient scale,
rank, parameter count, head geometry facts) and ADMISSION.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def arr_sha(a):
    import numpy as np
    z = np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()


def _features(item, verify, n=None):
    import numpy as np
    from zipfile import ZipFile
    p = verify(item['path'], item['sha256'])
    with ZipFile(p) as archive:
        if {x.filename for x in archive.infolist()} != {'row_id.npy', 'unit.npy', 'X.npy', 'feature_names.npy'}:
            raise PermissionError('feature-only export required for admission')
    with np.load(p, allow_pickle=False) as z:
        out = {k: z[k].copy() for k in ('row_id', 'unit', 'X', 'feature_names')}
    if n is not None and len(out['row_id']) != n:
        raise PermissionError('role rows differ')
    if out['X'].dtype != np.float32 or out['X'].shape[1] != 83 or not np.isfinite(out['X']).all():
        raise PermissionError('raw83 schema differs')
    forbidden = {'sex', 'race', 'income', 'occupation', 'fnlwgt', 'row_id', 'unit', 'id', 'hidden_flag'}
    names = tuple(map(str, out['feature_names']))
    if len(set(names)) != 83 or any(x.lower().split('=')[0] in forbidden for x in names):
        raise PermissionError('forbidden input schema')
    return out


def _change(a, b):
    import numpy as np
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def run(spec_path):
    from .evaluation_queue import queue_config
    active, spec, verify, guard = queue_config(spec_path, 'admission')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from task_conditioned_representation.models import load_checkpoint
    from .wires import prepare_geometry, forward_role, head_sha, tensor_sha, probe_key
    from .geometry import audit_geometry, build_geometry
    from .admission import group_folds
    from .adapter import (RecipientAdapter, fit_scale, head_record, forward_raw, hidden64, z_from_hidden, canonical_sha,
                          ADAPTER_CONTRACT, HIDDEN_LAYER, RECIPIENTS)
    from .services import load_previous
    out = Path(spec['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private admission directory required')
    out.mkdir(parents=True, mode=0o700)
    recorded = json.loads(verify(spec['recorded_source_admission']['path'], spec['recorded_source_admission']['sha256']).read_text())
    expected_hash = {(g['seed'], g['role'], t['recipient']): t for g in recorded['geometry_audits']
                     if g['unit_id'].startswith('RAW_JOINT_R03_E5') for t in g['tasks']}
    split = json.loads(verify(spec['role_split']['path'], spec['role_split']['sha256']).read_text())
    prev_manifest = json.loads(verify(spec['previous_model_manifest']['path'], spec['previous_model_manifest']['sha256']).read_text())
    roles = {r: _features(spec['source_roles'][r], verify, n) for r, n in
             (('OSF_DEFENSE_FIT', 15434), ('ATTACK_FIT_NEW', 4064), ('INNER_SELECTION', 2235))}
    exports = {r: _features(spec['prior_role_features'][r], verify, n) for r, n in
               (('UPDATE', 8180), ('GUARD', 2043), ('VALIDATION', 5211))}
    names = roles['OSF_DEFENSE_FIT']['feature_names']
    checks = {}; facts = {}
    for r in list(roles.values()) + list(exports.values()):
        if not np.array_equal(r['feature_names'], names):
            raise PermissionError('raw83 feature order differs across roles')
    fit = roles['OSF_DEFENSE_FIT']; fold = group_folds(fit['unit'])
    index = {int(r): i for i, r in enumerate(fit['row_id'])}
    for name, want in (('UPDATE', (0, 1)), ('GUARD', (0, 1)), ('VALIDATION', (2,))):
        e = exports[name]; ix = np.array([index[int(r)] for r in e['row_id']])
        checks[f'{name}_rows_are_FIT_rows_with_identical_features_and_units'] = bool(
            np.array_equal(fit['X'][ix], e['X']) and np.array_equal(fit['unit'][ix], e['unit']) and np.isin(fold[ix], want).all())
    pool_ids = set(map(int, exports['UPDATE']['row_id'])) | set(map(int, exports['GUARD']['row_id']))
    checks['UPDATE_plus_GUARD_equals_FIT_folds01'] = pool_ids == set(map(int, fit['row_id'][fold != 2]))
    checks['VALIDATION_equals_FIT_fold2'] = set(map(int, exports['VALIDATION']['row_id'])) == set(map(int, fit['row_id'][fold == 2]))
    facts['fold_rows'] = {str(k): int((fold == k).sum()) for k in range(3)}
    units = {k: set(map(str, v['unit'])) for k, v in list(roles.items()) + list(exports.items())}
    overlap = {f'{a}__{b}': len(units[a] & units[b]) for a in units for b in units if a < b}
    checks['role_groups_disjoint'] = all(v == 0 for k, v in overlap.items()
                                         if not ({'OSF_DEFENSE_FIT'} & set(k.split('__')) and set(k.split('__')) & {'UPDATE', 'GUARD', 'VALIDATION'}))
    facts['group_overlap'] = overlap
    dup = {}
    for name, r in list(roles.items()) + list(exports.items()):
        u, inv, cnt = np.unique(r['unit'], return_inverse=True, return_counts=True)
        multi = np.flatnonzero(cnt > 1)
        same = all(np.array_equal(r['X'][inv == g], np.repeat(r['X'][inv == g][:1], cnt[g], 0)) for g in multi)
        dup[name] = {'rows': int(len(r['unit'])), 'groups': int(len(u)), 'duplicate_groups': int(len(multi)),
                     'duplicate_rows_identical_features': bool(same)}
    facts['duplicate_groups'] = dup
    checks['duplicate_rows_identical_features_all_roles'] = all(v['duplicate_rows_identical_features'] for v in dup.values())
    facts['split_rule'] = split['rule']; facts['role_support'] = split['support']
    all_roles = dict(roles, **exports)
    seeds = {}
    for seed in (0, 1):
        guard()
        c = spec['checkpoints'][str(seed)]; g = spec['geometry'][str(seed)]
        model, payload = load_checkpoint(verify(c['path'], c['sha256']), c['sha256'])
        if (payload.get('family'), payload.get('strength'), payload.get('epoch'), payload.get('seed')) != ('RAW_JOINT', 0.3, 5, seed):
            raise PermissionError('start package identity differs')
        binding = json.loads(verify(g['path'], g['sha256']).read_text())
        geometry = prepare_geometry(binding, expected_sha256=g['binding_sha256'], source_seed=seed, checkpoint_sha256=c['sha256'])
        rec = {'checkpoint_sha256': c['sha256'], 'geometry_binding_sha256': geometry.binding_sha256,
               'class_orders': model.class_orders, 'feature_schema_matches': tuple(map(str, names)) == tuple(model.names),
               'head_tensor_sha256_matches_binding': [head_sha(h) for h in model.head] == binding['head_tensor_sha256'],
               'head_dtypes': [str(h.weight.dtype) for h in model.head],
               'encoder_dtypes': sorted({str(v.dtype) for k, v in model.state_dict().items() if k.startswith('enc.')}),
               'checkpoint_has_model_pt_head_substitute': False,
               'heads_source': 'current folded own heads inside the admitted epoch-5 checkpoint (head.0/head.1), hash-matched to the geometry binding'}
        rec['architecture'] = [[type(m).__name__ + (f'({m.in_features}->{m.out_features})' if hasattr(m, 'in_features') else '') for m in model.enc[i]] for i in (0, 1)]
        rec['hidden_layer'] = HIDDEN_LAYER
        x_all = torch.from_numpy(fit['X'][:2048])
        with torch.no_grad():
            rec['hidden_then_final_equals_encoder_bitwise'] = [bool(torch.equal(z_from_hidden(model, i, hidden64(model, i, x_all)), model.encode(i, x_all))) for i in (0, 1)]
        heads = [head_record(model, geometry, i) for i in (0, 1)]
        rec['heads'] = heads
        # Immutable per-recipient scales on the authorized fitting pool (UPDATE + GUARD).
        pool = torch.from_numpy(np.concatenate((exports['UPDATE']['X'], exports['GUARD']['X'])))
        scales = []; norms = []
        for i in (0, 1):
            s, nr = fit_scale(model, i, pool); scales.append(float(s)); norms.append(nr)
        rec['scales'] = scales; rec['hidden_norms'] = norms
        adapters = [RecipientAdapter(model, geometry, i, scales[i]) for i in (0, 1)]
        rec['ranks'] = [a.rank for a in adapters]; rec['trainable_parameters'] = [a.n_parameters() for a in adapters]
        rec['trainable_parameters_total'] = sum(rec['trainable_parameters'])
        # Raw83 reproduction of recorded RAW_PROJ hashes and the zero-residual witness on every role.
        repro = {}
        for role, data in all_roles.items():
            proj = forward_role(model, data['X'], names, geometry, variant='RAW_PROJ')
            wz = forward_raw(model, geometry, scales, [None, None], data['X'], names)
            for i, task in enumerate(RECIPIENTS):
                r = {}
                if role in roles:
                    r['projected_matches_recorded'] = arr_sha(proj[i]['wire']) == expected_hash[(seed, role, task)]['projected_wire_sha256']
                r['zero_residual_raw_path_equals_RAW_PROJ_bitwise'] = bool(np.array_equal(wz[i]['wire'], proj[i]['wire']))
                r['zero_residual_decisions_equal'] = bool(np.array_equal(wz[i]['decision'], proj[i]['decision']))
                r['zero_residual_max_abs'] = _change(wz[i]['wire'], proj[i]['wire'])
                if role in roles:
                    with torch.no_grad():
                        z = model.encode(i, torch.from_numpy(data['X'])).double().numpy()
                    gg = build_geometry(model.head[i].weight.detach().numpy(), model.head[i].bias.detach().numpy(), model.class_orders[i])
                    r['geometry_audit_passed'] = bool(audit_geometry(z, gg)['passed'])
                    r['rebuilt_basis_equals_bound'] = bool(np.array_equal(gg['B'], geometry.bases[i].numpy()))
                repro[f'{role}|{task}'] = r
        rec['reproduction_and_witness'] = repro
        # Stored inherited RAW_PROJ release on VALIDATION (separate-pool deployment) reproduced exactly.
        st = spec['stored_raw_proj_validation'][str(seed)]
        with np.load(verify(st['path'], st['sha256']), allow_pickle=False) as z:
            stored = {k: z[k].copy() for k in z.files}
        vz = forward_raw(model, geometry, scales, [None, None], exports['VALIDATION']['X'], names)
        rec['stored_RAW_PROJ_validation_release_reproduced'] = bool(np.array_equal(vz[0]['wire'], stored['r1']) and np.array_equal(vz[1]['wire'], stored['r2']))
        # Previous occupation-only J_TRUST epoch 10 (O_PREVIOUS): exact manifest, scale, basis, stored INNER release.
        pv = spec['previous'][str(seed)]
        man_entry = prev_manifest['adapter_checkpoints'][str(seed)]['J_TRUST']['10']
        prev = load_previous(verify(pv['checkpoint']['path'], pv['checkpoint']['sha256']), pv['checkpoint']['sha256'], seed=seed, source_checkpoint_sha256=c['sha256'])
        pb = json.loads(verify(pv['binding']['path'], pv['binding']['sha256']).read_text())
        prec = {'checkpoint_sha256_matches_public_manifest': pv['checkpoint']['sha256'] == man_entry['sha256'],
                'R_sha256_matches_manifest': tensor_sha(prev['R']) == man_entry['R_sha256'],
                'b_res_sha256_matches_manifest': tensor_sha(prev['b_res']) == man_entry['b_res_sha256'],
                'binding_sha256_matches_manifest': pv['binding']['sha256'] == prev_manifest['adapter_bindings'][str(seed)],
                'previous_scale': float(prev['scale']), 'scale_equals_new_occupation_scale': float(prev['scale']) == scales[1],
                'occupation_basis_equals_bound': bool(torch.equal(prev['occupation_basis'].double(), geometry.bases[1].double())),
                'previous_binding_scale': float(pb['scale']), 'income_residual_used': 'zero (income frozen in previous campaign)'}
        inner = roles['INNER_SELECTION']
        live = {'J_TRUST': forward_raw(model, geometry, scales, [None, {'R': prev['R'], 'b': prev['b_res']}], inner['X'], names),
                'RAW_PROJ': forward_raw(model, geometry, scales, [None, None], inner['X'], names)}
        for label, key in (('J_TRUST', 'stored_inner_release_j_trust'), ('RAW_PROJ', 'stored_inner_release_raw_proj')):
            with np.load(verify(pv[key]['path'], pv[key]['sha256']), allow_pickle=False) as z:
                srel = {k: z[k].copy() for k in z.files}
            prec[f'stored_previous_{label}_INNER_release_reproduced_bitwise'] = bool(
                np.array_equal(live[label][0]['wire'], srel['r1']) and np.array_equal(live[label][1]['wire'], srel['r2']))
        rec['O_PREVIOUS'] = prec
        # Probe, cached-vs-raw path and per-recipient repeat-forward noise.
        xU = torch.from_numpy(exports['UPDATE']['X'])
        probe = sorted(range(len(xU)), key=lambda k: (probe_key(exports['UPDATE']['row_id'][k]), k))[:512]
        rec['probe_position_sha256_matches_split'] = hashlib.sha256(json.dumps(probe).encode()).hexdigest() == split['probe_position_sha256']
        raw_probe = forward_raw(model, geometry, scales, [None, None], exports['UPDATE']['X'][probe], names)
        noise = []; cached = []
        for i, ad in enumerate(adapters):
            h1, q1 = ad.frozen_features(xU); h2, q2 = ad.frozen_features(xU)
            with torch.no_grad():
                w1 = ad.wire(h1[probe], q1[probe]); w2 = ad.wire(h2[probe], q2[probe])
            noise.append({k: _change(w1[k], w2[k]) for k in ('representation', 'centered_logits', 'probabilities')})
            cached.append(bool(np.array_equal(w1['full'].numpy(), raw_probe[i]['wire'])))
        rec['repeat_forward_noise'] = noise; rec['cached_vs_raw_path_probe_bitwise'] = cached
        ab = {'schema': 'TRA1-adapter-binding-v1', 'campaign': ADAPTER_CONTRACT['campaign'], 'seed': seed,
              'source_checkpoint_sha256': c['sha256'], 'geometry_binding_sha256': geometry.binding_sha256,
              'basis_sha256': [tensor_sha(geometry.bases[i]) for i in (0, 1)], 'ranks': rec['ranks'],
              'trainable_parameters': rec['trainable_parameters'], 'scales': scales, 'scale_rule': norms[0]['rule'],
              'scale_pool_rows': norms[0]['rows'], 'hidden_layer': HIDDEN_LAYER, 'hidden_norms': norms, 'heads': heads,
              'adapter_contract_sha256': canonical_sha(ADAPTER_CONTRACT), 'bound_before_any_fit_or_FIT2_metric': True}
        path = out / f'ADAPTER_BINDING_S{seed}.json'
        path.write_text(json.dumps(ab, indent=2, sort_keys=True)); path.chmod(0o400)
        rec['adapter_binding_file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        seeds[str(seed)] = rec
    wire_tolerance = []
    for i in (0, 1):
        mx = {k: max(seeds[s]['repeat_forward_noise'][i][k] for s in ('0', '1')) for k in ('representation', 'centered_logits', 'probabilities')}
        wire_tolerance.append({k: max(1e-10, 10 * v) for k, v in mx.items()})
    flags = dict(checks)
    for s, rec in seeds.items():
        for key in ('feature_schema_matches', 'head_tensor_sha256_matches_binding', 'stored_RAW_PROJ_validation_release_reproduced',
                    'probe_position_sha256_matches_split'):
            flags[f'S{s}_{key}'] = bool(rec[key])
        flags[f'S{s}_hidden_then_final_bitwise_both'] = all(rec['hidden_then_final_equals_encoder_bitwise'])
        flags[f'S{s}_cached_vs_raw_probe_bitwise_both'] = all(rec['cached_vs_raw_path_probe_bitwise'])
        flags[f'S{s}_class_orders'] = rec['class_orders'] == [[0, 1], list(range(6))]
        flags[f'S{s}_A_Bt_B_equals_A_both'] = all(h['A_Bt_B_minus_A_max_abs'] <= 1e-10 for h in rec['heads'])
        flags[f'S{s}_orthonormal_bases'] = all(h['B_Bt_minus_I_max_abs'] <= 1e-10 for h in rec['heads'])
        flags[f'S{s}_measured_rank_equals_basis_rank'] = all(h['measured_centered_rank'] == h['basis_rank'] for h in rec['heads'])
        flags[f'S{s}_all_role_reproduction_and_zero_witness'] = all(all(bool(v) for k, v in r.items() if k != 'zero_residual_max_abs')
                                                                    for r in rec['reproduction_and_witness'].values())
        flags[f'S{s}_O_PREVIOUS_admitted'] = all(v for k, v in rec['O_PREVIOUS'].items() if isinstance(v, bool))
    result = {'schema': 'TRA1-admission-v1', 'queue_unit_id': active['unit_id'], 'status': 'PASS' if all(flags.values()) else 'FAIL',
              'flags': flags, 'facts': facts, 'seeds': seeds, 'wire_tolerance': wire_tolerance,
              'wire_tolerance_rule': 'per recipient: max(1e-10, 10*max repeat-forward abs noise over both seeds) per coordinate block; any decision change counts',
              'measured_ranks': {s: seeds[s]['ranks'] for s in seeds}, 'measured_trainable_parameters': {s: seeds[s]['trainable_parameters'] for s in seeds},
              'labels_loaded': False, 'INNER_labels_opened_or_hashed': False, 'FIT2_metrics_computed': False, 'model_updates': 0}
    for path in (out / 'ADMISSION.json', Path(spec['output'])):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, default=float)
    if result['status'] != 'PASS':
        raise AssertionError('admission failed: ' + json.dumps({k: v for k, v in flags.items() if not v}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)
