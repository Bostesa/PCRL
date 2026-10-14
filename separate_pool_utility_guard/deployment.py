"""Standalone raw83 own-head runtime for the separate-pool trial (inherited CIS1 wire contract).

No source loader, training label, teacher, or private absolute path is required.
The runtime duplicates canonical arithmetic; it does not import trial workers.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch

SCHEMA = 'SPG1-raw83-own-head-deployment-v1'
CAMPAIGN = 'pcrl_separate_pool_utility_guard_v1'
# Per-seed geometry bindings are inherited byte-for-byte from the corrected trial.
GEOMETRY_CAMPAIGN = 'pcrl_compressed_income_safeguard_v1'
BATCH_SIZE = 1024
CLASS_ORDERS = [[0, 1], list(range(6))]
VARIANTS = {name: ['head_row_space', 'full'] for name in
            ('RAW_HYBRID', 'SEPARATE_GUARD', 'GLOBAL_UPDATE_GUARD', 'OLD_BATCH_GUARD', 'ORDINARY', 'TASK_ONLY')}
VARIANTS.update({name: ['full', 'full'] for name in ('RAW_FULL', 'U0_FULL')})
VARIANTS.update({name: ['head_row_space', 'head_row_space'] for name in ('RAW_PROJ', 'U0_PROJ')})
STATE_KEYS = {f'enc.{i}.{layer}.{kind}' for i in range(2) for layer in (0, 2, 4) for kind in ('weight', 'bias')} | {f'head.{i}.{kind}' for i in range(2) for kind in ('weight', 'bias')}
CONTRACT = {'recipient_full_dimensions': [21, 29], 'pair_full_dimensions': 50,
            'recipient_score_dimensions': [3, 7], 'pair_score_dimensions': 10,
            'deterministic': True, 'head_reads_released_representation': True,
            'role_forward_batch_size': BATCH_SIZE,
            'projection_arithmetic': 'torch_float64_q=z@B.T;released=q@B',
            'decision_rule': 'argmax probabilities; first class in fixed source order',
            'raw_z_or_q_co_release': False}
FIELDS = {'schema', 'campaign', 'service_id', 'seed', 'per_recipient_variants',
          'feature_names', 'feature_schema_sha256', 'class_orders', 'model_sha256',
          'source_checkpoint_sha256', 'provenance', 'contract', 'parameter_count', 'geometry'}


def sha_file(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def schema_hash(names):
    return hashlib.sha256(json.dumps(names, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def tensor_sha(value):
    a = value.detach().cpu().contiguous().numpy()
    h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def head_sha(head):
    return json_sha({'weight': tensor_sha(head.weight), 'bias': tensor_sha(head.bias)})


def validate_names(names):
    if not isinstance(names, (tuple, list)) or len(names) != 83 or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != 83:
        raise ValueError('exact ordered83 named columns required')
    forbidden = {'sex', 'race', 'income', 'occupation', 'fnlwgt', 'row_id', 'unit', 'id', 'person_id', 'sampling_id', 'group_id', 'hidden_flag', 'y', 'label', 'target', 'teacher', 'teacher_score'}
    if any(n.lower().split('=')[0] in forbidden for n in names):
        raise ValueError('forbidden deployment feature')
    return list(names)


class RuntimeModel(torch.nn.Module):
    def __init__(self, names, variants, bases):
        super().__init__(); self.names = tuple(validate_names(names)); self.variants = list(variants); self.bases = bases
        self.enc = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(83, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 16)) for _ in range(2)])
        self.head = torch.nn.ModuleList([torch.nn.Linear(16, k, dtype=torch.float64) for k in (2, 6)])

    def release(self, raw, names, **extra):
        if extra: raise ValueError('unregistered inputs or output channels refused')
        if tuple(validate_names(names)) != self.names: raise ValueError('wrong or reordered input columns')
        x = np.asarray(raw)
        if x.ndim != 2 or x.shape[1] != 83 or not len(x) or x.dtype.kind != 'f' or not np.isfinite(x).all():
            raise ValueError('finite nonempty floating raw83 required')
        with np.errstate(over='ignore'): x = np.ascontiguousarray(x, dtype=np.float32)
        if not np.isfinite(x).all(): raise ValueError('raw83 overflows encoder precision')
        collected = [{}, {}]
        with torch.inference_mode():
            for start in range(0, len(x), BATCH_SIZE):
                batch = torch.from_numpy(x[start:start+BATCH_SIZE])
                for i, (encoder, head) in enumerate(zip(self.enc, self.head)):
                    z = encoder(batch)
                    if self.variants[i] == 'head_row_space':
                        basis = self.bases[i]; z = (z.to(torch.float64) @ basis.T) @ basis
                    logits = head(z.to(torch.float64)); logits = logits - logits.mean(1, keepdim=True)
                    p = torch.softmax(logits, 1); order = torch.tensor(CLASS_ORDERS[i], dtype=torch.long)
                    d = order[p.argmax(1)]
                    full = torch.cat((z.to(logits.dtype), logits, p, d[:, None].to(logits.dtype)), 1)
                    values = {'representation': z, 'centered_logits': logits, 'probabilities': p,
                              'decision': d, 'full': full,
                              'output_only': torch.cat((p, d[:, None].to(p.dtype)), 1)}
                    if not torch.isfinite(full).all(): raise ValueError('nonfinite release refused')
                    for key, value in values.items(): collected[i].setdefault(key, []).append(value.numpy())
        return [{key: np.concatenate(value, axis=0) for key, value in c.items()} for c in collected]


def _geometry(binding, meta, model, seed, variants):
    b = dict(binding); digest = b.pop('binding_sha256', None)
    if (digest != json_sha(b) or digest != meta['binding_sha256'] or
        b.get('schema') != 'CIS1-per-seed-geometry-v1' or b.get('campaign') != GEOMETRY_CAMPAIGN or
        b.get('source_seed') != seed or b.get('checkpoint_sha256') != meta['checkpoint_sha256'] or
        b.get('head_tensor_sha256') != meta['head_tensor_sha256'] or
        len(b.get('heads', [])) != 2 or len(b.get('head_tensor_sha256', [])) != 2):
        raise ValueError('geometry source identity differs')
    bases = [None, None]
    for i, variant in enumerate(variants):
        if variant == 'full': continue
        g = b['heads'][i]
        if (head_sha(model.head[i]) != b['head_tensor_sha256'][i] or g.get('source_dimension') != 16 or
            g.get('class_order') != CLASS_ORDERS[i] or not isinstance(g.get('rank'), int) or
            not 0 <= g['rank'] <= len(CLASS_ORDERS[i])-1):
            raise ValueError('projected own head differs from frozen geometry')
        basis = torch.tensor(g['B'], dtype=torch.float64).reshape(g['rank'], 16)
        a = model.head[i].weight.detach(); a = a-a.mean(0, keepdim=True)
        if (not torch.isfinite(basis).all() or
            not torch.allclose(basis@basis.T, torch.eye(len(basis), dtype=torch.float64), atol=1e-10, rtol=1e-10) or
            not torch.allclose((a@basis.T)@basis, a, atol=1e-10, rtol=1e-10)):
            raise ValueError('invalid own-head basis')
        bases[i] = basis
    return bases


def load_package(directory, expected_manifest_sha256):
    root = Path(directory); mfile = root/'DEPLOYMENT.json'; weights = root/'model_state.pt'; geometry_file = root/'geometry.json'
    if not expected_manifest_sha256 or any(p.is_symlink() for p in (root, *root.parents, mfile, weights, geometry_file)) or sha_file(mfile) != expected_manifest_sha256:
        raise ValueError('manifest pin differs')
    m = json.loads(mfile.read_text())
    if (set(m) != FIELDS or m['schema'] != SCHEMA or m['campaign'] != CAMPAIGN or m['service_id'] not in VARIANTS or
        m['per_recipient_variants'] != VARIANTS[m['service_id']] or m['seed'] not in (0, 1) or
        m['class_orders'] != CLASS_ORDERS or m['parameter_count'] != 21288 or m['contract'] != CONTRACT):
        raise ValueError('deployment schema or contract differs')
    names = validate_names(m['feature_names'])
    if schema_hash(names) != m['feature_schema_sha256'] or sha_file(weights) != m['model_sha256']:
        raise ValueError('model or schema pin differs')
    state = torch.load(weights, map_location='cpu', weights_only=True)
    if set(state) != STATE_KEYS or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() or v.dtype != (torch.float32 if k.startswith('enc.') else torch.float64) for k, v in state.items()):
        raise ValueError('model state differs')
    model = RuntimeModel(names, m['per_recipient_variants'], [None, None]); model.load_state_dict(state, strict=True)
    meta = m['geometry']
    if set(meta) != {'file', 'sha256', 'binding_sha256', 'checkpoint_sha256', 'head_tensor_sha256'} or meta['file'] != 'geometry.json' or sha_file(geometry_file) != meta['sha256']:
        raise ValueError('geometry pin differs')
    model.bases = _geometry(json.loads(geometry_file.read_text()), meta, model, m['seed'], m['per_recipient_variants'])
    model.eval(); return model, m


def export_checkpoint(checkpoint, checkpoint_sha256, directory, *, service_id, geometry_binding, geometry_binding_sha256, seed):
    """Called only by admitted queued export worker; preserves admitted geometry."""
    if service_id not in VARIANTS or sha_file(checkpoint) != checkpoint_sha256: raise ValueError('checkpoint/service pin differs')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('schema') != 'TCR1-trainable-own-head-checkpoint-v1' or payload.get('class_orders') != CLASS_ORDERS or payload.get('seed') != seed:
        raise ValueError('checkpoint schema or source seed differs')
    if geometry_binding.get('binding_sha256') != geometry_binding_sha256: raise ValueError('geometry binding pin differs')
    if service_id in ('SEPARATE_GUARD', 'GLOBAL_UPDATE_GUARD', 'OLD_BATCH_GUARD', 'ORDINARY', 'TASK_ONLY'):
        if payload.get('wire_variant') != service_id or payload.get('wire_binding') != geometry_binding:
            raise ValueError('continuation checkpoint frozen basis identity differs')
    names = validate_names(payload['feature_names']); state = payload['model']
    if set(state) != STATE_KEYS or any(not isinstance(v, torch.Tensor) or v.dtype != (torch.float32 if k.startswith('enc.') else torch.float64) or not torch.isfinite(v).all() for k, v in state.items()):
        raise ValueError('encoder/head state differs')
    model = RuntimeModel(names, VARIANTS[service_id], [None, None]); model.load_state_dict(state, strict=True)
    meta = {'file': 'geometry.json', 'sha256': '', 'binding_sha256': geometry_binding_sha256,
            'checkpoint_sha256': geometry_binding['checkpoint_sha256'], 'head_tensor_sha256': geometry_binding['head_tensor_sha256']}
    _geometry(geometry_binding, meta, model, seed, VARIANTS[service_id])
    root = Path(directory)
    if any(p.is_symlink() for p in (root, *root.parents)): raise ValueError('package symlink refused')
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    weights = root/'model_state.pt'; torch.save(state, weights); weights.chmod(0o600)
    _write_json(root/'geometry.json', geometry_binding); meta['sha256'] = sha_file(root/'geometry.json')
    m = {'schema': SCHEMA, 'campaign': CAMPAIGN, 'service_id': service_id, 'seed': seed,
         'per_recipient_variants': VARIANTS[service_id], 'feature_names': names, 'feature_schema_sha256': schema_hash(names),
         'class_orders': payload['class_orders'], 'model_sha256': sha_file(weights), 'source_checkpoint_sha256': checkpoint_sha256,
         'provenance': {k: payload[k] for k in ('family', 'seed', 'strength', 'coefficient', 'epoch') if k in payload},
         'parameter_count': sum(v.numel() for v in state.values()), 'contract': CONTRACT, 'geometry': meta}
    _write_json(root/'DEPLOYMENT.json', m); return sha_file(root/'DEPLOYMENT.json')


def _write_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('package', 'manifest-sha256', 'input', 'input-sha256', 'output'): p.add_argument('--'+name, required=True)
    a = p.parse_args(); torch.set_num_threads(1); model, m = load_package(a.package, a.manifest_sha256)
    if sha_file(a.input) != a.input_sha256: raise ValueError('input pin differs')
    with np.load(a.input, allow_pickle=False) as z:
        if set(z.files) != {'X', 'feature_names'}: raise ValueError('feature-only raw83 input required')
        v = model.release(z['X'], z['feature_names'].tolist())
    with os.fdopen(os.open(a.output, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'wb') as f:
        np.savez(f, r1=v[0]['full'], r2=v[1]['full']); f.flush(); os.fsync(f.fileno())
    print(json.dumps({'schema': SCHEMA, 'rows': len(v[0]['full']), 'service_id': m['service_id'],
                      'output_sha256': sha_file(a.output), 'dimensions': [21, 29, 50]}))

if __name__ == '__main__': main()
