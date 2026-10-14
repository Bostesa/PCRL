"""Queued source/role admission, zero-residual witness and immutable bindings.

No model is updated and no label-dependent metric is computed. Labels are not
loaded (SEX/occupation are not sources of this unit). Writes:
  ADAPTER_BINDING_S{seed}.json   scale s, norms, rank, parameter count, hidden layer
  OWNPROJ_GLOBAL_UPDATE_GUARD_E5_S{seed}.json  own-head occupation geometry of the prior checkpoint
  ADMISSION.json                 observed facts and checks
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
    from .wires import prepare_geometry, forward_role, head_sha, tensor_sha
    from .geometry import audit_geometry
    from .wires import probe_key
    from .adapter import Adapter, fit_scale, head_norms, forward_raw, hidden64, z_from_hidden, canonical_sha, ADAPTER_CONTRACT, HIDDEN_LAYER
    from .services import build_ownproj_binding, service_forward
    out = Path(spec['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private admission directory required')
    out.mkdir(parents=True, mode=0o700)
    recorded = json.loads(verify(spec['recorded_source_admission']['path'], spec['recorded_source_admission']['sha256']).read_text())
    expected_hash = {(g['seed'], g['role'], t['recipient']): t for g in recorded['geometry_audits']
                     if g['unit_id'].startswith('RAW_JOINT_R03_E5') for t in g['tasks']}
    split = json.loads(verify(spec['role_split']['path'], spec['role_split']['sha256']).read_text())
    roles = {r: _features(spec['source_roles'][r], verify, n) for r, n in
             (('OSF_DEFENSE_FIT', 15434), ('ATTACK_FIT_NEW', 4064), ('INNER_SELECTION', 2235))}
    exports = {r: _features(spec['prior_role_features'][r], verify, n) for r, n in
               (('UPDATE', 8180), ('GUARD', 2043), ('VALIDATION', 5211))}
    names = roles['OSF_DEFENSE_FIT']['feature_names']
    checks = {}; facts = {}
    for r in list(roles.values()) + list(exports.values()):
        if not np.array_equal(r['feature_names'], names):
            raise PermissionError('raw83 feature order differs across roles')
    # Role membership: exports are exact subsets of FIT with inherited fold rule.
    from .admission import group_folds
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
    for name, r in roles.items():
        u, inv, cnt = np.unique(r['unit'], return_inverse=True, return_counts=True)
        multi = np.flatnonzero(cnt > 1)
        same = all(np.array_equal(r['X'][inv == g], np.repeat(r['X'][inv == g][:1], cnt[g], 0)) for g in multi)
        dup[name] = {'rows': int(len(r['unit'])), 'groups': int(len(u)), 'duplicate_groups': int(len(multi)),
                     'duplicate_rows_identical_features': bool(same)}
    facts['duplicate_groups'] = dup
    facts['split_rule'] = split['rule']; facts['role_support'] = split['support']
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
               'source_bindings_heads': model.source_bindings.get('heads') if isinstance(model.source_bindings, dict) else None}
        # Architecture: h64 is the input to enc.1.4 (Linear 64->16).
        seq = model.enc[1]
        rec['architecture'] = [type(m).__name__ + (f'({m.in_features}->{m.out_features})' if hasattr(m, 'in_features') else '') for m in seq]
        rec['hidden_layer'] = HIDDEN_LAYER
        x_all = torch.from_numpy(fit['X'][:2048])
        with torch.no_grad():
            h = hidden64(model, x_all)
            rec['hidden_then_final_equals_encoder_bitwise'] = bool(torch.equal(z_from_hidden(model, h), model.encode(1, x_all)))
            rec['hidden_dim'] = int(h.shape[1])
        # Folded own heads are the admitted source heads: binding hashes + exact reproduction of the
        # predecessor's recorded RAW_PROJ / RAW_HYBRID wires (which contain own probabilities) on all roles.
        repro = {}
        for role, data in roles.items():
            proj = forward_role(model, data['X'], names, geometry, variant='RAW_PROJ')
            hyb = forward_role(model, data['X'], names, geometry, variant='RAW_HYBRID')
            for i, task in enumerate(('income', 'occupation')):
                exp = expected_hash[(seed, role, task)]
                repro[f'{role}|{task}'] = {'projected_matches_recorded': arr_sha(proj[i]['wire']) == exp['projected_wire_sha256'],
                                           'hybrid_matches_recorded': arr_sha(hyb[i]['wire']) == exp['hybrid_wire_sha256']}
            # Zero-residual witness on every admitted role, raw83 deployment path.
            adapter = Adapter(model, geometry, 1.0)
            wz = forward_raw(adapter, data['X'], names)
            for i, task in enumerate(('income', 'occupation')):
                same = np.array_equal(wz[i]['wire'], proj[i]['wire'])
                repro[f'{role}|{task}']['zero_residual_raw_path_equals_RAW_PROJ_bitwise'] = bool(same)
                repro[f'{role}|{task}']['zero_residual_max_abs'] = _change(wz[i]['wire'], proj[i]['wire'])
            # Geometry audits on actual features (frozen income + occupation bases).
            with torch.no_grad():
                xx = torch.from_numpy(data['X'])
                zi = model.encode(0, xx).double().numpy(); zo = model.encode(1, xx).double().numpy()
            from .geometry import build_geometry
            for i, z in ((0, zi), (1, zo)):
                gg = build_geometry(model.head[i].weight.detach().numpy(), model.head[i].bias.detach().numpy(), model.class_orders[i])
                a = audit_geometry(z, gg)
                repro[f'{role}|{("income", "occupation")[i]}']['geometry_audit_passed'] = a['passed']
                repro[f'{role}|{("income", "occupation")[i]}']['rebuilt_basis_equals_bound'] = bool(np.array_equal(gg['B'], geometry.bases[i].numpy()))
        rec['reproduction_and_witness'] = repro
        # Stored previous-campaign RAW_PROJ release (VALIDATION) reproduced exactly.
        st = spec['stored_raw_proj_validation'][str(seed)]
        with np.load(verify(st['path'], st['sha256']), allow_pickle=False) as z:
            stored = {k: z[k].copy() for k in z.files}
        adapter = Adapter(model, geometry, 1.0)
        vz = forward_raw(adapter, exports['VALIDATION']['X'], names)
        rec['stored_previous_RAW_PROJ_validation_release_reproduced'] = bool(
            np.array_equal(vz[0]['wire'], stored['r1']) and np.array_equal(vz[1]['wire'], stored['r2']))
        # Immutable scale on the full authorized task pool (UPDATE + GUARD).
        pool = torch.from_numpy(np.concatenate((exports['UPDATE']['X'], exports['GUARD']['X'])))
        s, norms = fit_scale(model, pool)
        adapter = Adapter(model, geometry, float(s))
        rec['scale'] = s; rec['hidden_norms'] = norms; rec['head_norms'] = head_norms(model, geometry)
        rec['rank'] = adapter.rank; rec['trainable_parameters'] = adapter.n_parameters()
        A = model.head[1].weight.detach().double(); A = A - A.mean(0, keepdim=True)
        rec['A_Bt_B_equals_A_max_abs'] = float((A @ adapter.B.T @ adapter.B - A).abs().max())
        rec['A_Bt_full_column_rank'] = int(torch.linalg.matrix_rank(A @ adapter.B.T))
        # Cached path == raw path at zero residual with the bound scale; repeat-forward noise on the probe.
        xU = torch.from_numpy(exports['UPDATE']['X'])
        hU, qU, iU = adapter.frozen_features(xU); hU2, qU2, iU2 = adapter.frozen_features(xU)
        probe = sorted(range(len(xU)), key=lambda i: (probe_key(exports['UPDATE']['row_id'][i]), i))[:512]
        probe_sha = hashlib.sha256(json.dumps(probe).encode()).hexdigest()
        rec['probe_position_sha256_matches_split'] = probe_sha == split['probe_position_sha256']
        with torch.no_grad():
            w1 = adapter.wire(hU[probe], qU[probe]); w2 = adapter.wire(hU2[probe], qU2[probe])
        raw_probe = forward_raw(adapter, exports['UPDATE']['X'][probe], names)[1]['wire']
        noise = {'representation': _change(w1['representation'], w2['representation']),
                 'centered_logits': _change(w1['centered_logits'], w2['centered_logits']),
                 'probabilities': _change(w1['probabilities'], w2['probabilities'])}
        rec['repeat_forward_noise'] = noise
        rec['cached_vs_raw_path_probe_bitwise'] = bool(np.array_equal(w1['full'].numpy(), raw_probe))
        rec['income_cached_equals_RAW_PROJ_income'] = bool(np.array_equal(
            iU.numpy(), forward_role(model, exports['UPDATE']['X'], names, geometry, variant='RAW_PROJ')[0]['wire']))
        ab = {'schema': 'MTA1-adapter-binding-v1', 'campaign': ADAPTER_CONTRACT['campaign'], 'seed': seed,
              'source_checkpoint_sha256': c['sha256'], 'geometry_binding_sha256': geometry.binding_sha256,
              'occupation_basis_sha256': tensor_sha(geometry.bases[1]), 'income_basis_sha256': tensor_sha(geometry.bases[0]),
              'rank': adapter.rank, 'trainable_parameters': adapter.n_parameters(), 'scale': s, 'scale_rule': norms['rule'],
              'scale_pool_rows': norms['rows'], 'hidden_layer': HIDDEN_LAYER, 'hidden_norms': norms,
              'head_norms': rec['head_norms'], 'adapter_contract_sha256': canonical_sha(ADAPTER_CONTRACT),
              'bound_before_any_fit_or_FIT2_metric': True}
        path = out / f'ADAPTER_BINDING_S{seed}.json'
        path.write_text(json.dumps(ab, indent=2, sort_keys=True)); path.chmod(0o400)
        rec['adapter_binding_file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Prior guard diagnostics: identities, frozen income, own-head geometry for GLOBAL epoch 5.
        prior = {}
        for label in ('SEPARATE_GUARD', 'GLOBAL_UPDATE_GUARD'):
            pc = spec['prior_checkpoints'][str(seed)][label]
            pm, pp = load_checkpoint(verify(pc['path'], pc['sha256']), pc['sha256'])
            if (pp.get('seed'), pp.get('epoch'), pp.get('wire_variant')) != (seed, 5, label):
                raise PermissionError('prior checkpoint identity differs')
            same_income = all(torch.equal(pm.state_dict()[k], model.state_dict()[k]) for k in model.state_dict() if k.startswith(('enc.0.', 'head.0.')))
            changed_occ = any(not torch.equal(pm.state_dict()[k], model.state_dict()[k]) for k in model.state_dict() if k.startswith(('enc.1.', 'head.1.')))
            entry = {'checkpoint_sha256': pc['sha256'], 'income_tensors_bitwise_start': bool(same_income),
                     'occupation_tensors_changed_from_start': bool(changed_occ),
                     'occupation_head_changed_from_start': head_sha(pm.head[1]) != head_sha(model.head[1])}
            if label == 'GLOBAL_UPDATE_GUARD':
                ob = build_ownproj_binding(pm, checkpoint_sha256=pc['sha256'], source_seed=seed, start_geometry=geometry)
                opath = out / f'OWNPROJ_GLOBAL_UPDATE_GUARD_E5_S{seed}.json'
                opath.write_text(json.dumps(ob, indent=2, sort_keys=True)); opath.chmod(0o400)
                entry['ownproj_binding_sha256'] = ob['binding_sha256']
                entry['ownproj_file_sha256'] = hashlib.sha256(opath.read_bytes()).hexdigest()
                entry['ownproj_rank'] = ob['occupation_geometry']['rank']
                aud = {}
                for role, data in list(roles.items()):
                    hyb = service_forward('PRIOR_HYBRID', data['X'], names, model=pm, geometry=geometry)
                    own = service_forward('OWNPROJ', data['X'], names, model=pm, geometry=geometry, ownproj=ob)
                    aud[role] = {'income_wire_bitwise_equal': bool(np.array_equal(hyb[0]['wire'], own[0]['wire'])),
                                 'centered_logits_max_abs': _change(hyb[1]['centered_logits'], own[1]['centered_logits']),
                                 'probability_max_abs': _change(hyb[1]['probability'], own[1]['probability']),
                                 'decision_mismatch_rows': int((hyb[1]['decision'] != own[1]['decision']).sum()),
                                 'within_source_tolerance': bool(_change(hyb[1]['centered_logits'], own[1]['centered_logits']) <= 1e-10 + 1e-10 * float(np.abs(hyb[1]['centered_logits']).max())
                                                                 and _change(hyb[1]['probability'], own[1]['probability']) <= 2e-12
                                                                 and np.array_equal(hyb[1]['decision'], own[1]['decision']))}
                entry['ownproj_output_preservation'] = aud
            prior[label] = entry
        rec['prior_guard_diagnostics'] = prior
        seeds[str(seed)] = rec
    tol_noise = {k: max(seeds[s]['repeat_forward_noise'][k] for s in ('0', '1')) for k in ('representation', 'centered_logits', 'probabilities')}
    wire_tolerance = {k: max(1e-10, 10 * v) for k, v in tol_noise.items()}
    flags = dict(checks)
    for s, rec in seeds.items():
        for key in ('feature_schema_matches', 'head_tensor_sha256_matches_binding', 'hidden_then_final_equals_encoder_bitwise',
                    'stored_previous_RAW_PROJ_validation_release_reproduced', 'probe_position_sha256_matches_split',
                    'cached_vs_raw_path_probe_bitwise', 'income_cached_equals_RAW_PROJ_income'):
            flags[f'S{s}_{key}'] = bool(rec[key])
        flags[f'S{s}_class_orders'] = rec['class_orders'] == [[0, 1], list(range(6))]
        flags[f'S{s}_A_Bt_B_equals_A'] = rec['A_Bt_B_equals_A_max_abs'] <= 1e-10
        flags[f'S{s}_all_role_reproduction_and_zero_witness'] = all(all(bool(v) for k, v in r.items() if k != 'zero_residual_max_abs')
                                                                    for r in rec['reproduction_and_witness'].values())
        flags[f'S{s}_prior_income_frozen'] = all(v['income_tensors_bitwise_start'] for v in rec['prior_guard_diagnostics'].values())
        flags[f'S{s}_ownproj_preserves_outputs'] = all(v['within_source_tolerance'] and v['income_wire_bitwise_equal']
                                                       for v in rec['prior_guard_diagnostics']['GLOBAL_UPDATE_GUARD']['ownproj_output_preservation'].values())
    result = {'schema': 'MTA1-admission-v1', 'queue_unit_id': active['unit_id'], 'status': 'PASS' if all(flags.values()) else 'FAIL',
              'flags': flags, 'facts': facts, 'seeds': seeds, 'repeat_forward_noise_max': tol_noise, 'wire_tolerance': wire_tolerance,
              'wire_tolerance_rule': 'max(1e-10, 10*max repeat-forward abs noise over both seeds) per coordinate block; any decision change counts',
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
