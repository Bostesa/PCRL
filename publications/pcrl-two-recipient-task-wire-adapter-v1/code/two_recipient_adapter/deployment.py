"""Standalone raw83 runtime for the two-recipient task-wire adapter services.

No source loader, label, teacher score or private absolute path is required.
Duplicates the canonical arithmetic; it does not import campaign workers.
Kinds: RAW_PROJ (both residuals exactly zero) and ADAPTER2 (saved per-recipient
R_i, b_i on each head row space). Emits only the contracted income21 and
occupation29 recipient wires.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch

SCHEMA = 'TRA1-raw83-two-recipient-adapter-deployment-v1'
CAMPAIGN = 'pcrl_two_recipient_task_wire_adapter_v1'
GEOMETRY_CAMPAIGN = 'pcrl_compressed_income_safeguard_v1'
BATCH_SIZE = 1024
CLASS_ORDERS = [[0, 1], list(range(6))]
KINDS = ('RAW_PROJ', 'ADAPTER2')
STATE_KEYS = {f'enc.{i}.{layer}.{kind}' for i in range(2) for layer in (0, 2, 4) for kind in ('weight', 'bias')} | {f'head.{i}.{kind}' for i in range(2) for kind in ('weight', 'bias')}
ADAPTER_KEYS = {'R_income', 'b_income', 'scale_income', 'R_occupation', 'b_occupation', 'scale_occupation'}
CONTRACT = {'recipient_full_dimensions': [21, 29], 'pair_full_dimensions': 50, 'deterministic': True,
            'per_recipient': 'z_i=((float64(z0_i)@B_i.T + float64(h64_i)/s_i@R_i.T) + b_i)@B_i; own frozen head i on z_i',
            'decision_rule': 'argmax probabilities; first class in fixed source order', 'role_forward_batch_size': BATCH_SIZE,
            'never_emitted': ['h64', 'z16', 'q', 'residual', 'teacher probabilities', 'another head output', 'labels', 'row ids', 'fine ids', 'original features']}
FIELDS = {'schema', 'campaign', 'service_id', 'kind', 'seed', 'feature_names', 'feature_schema_sha256', 'class_orders',
          'model_sha256', 'source_checkpoint_sha256', 'adapter', 'geometry', 'contract', 'scientific_status', 'provenance'}


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
        raise ValueError('exact ordered 83 named columns required')
    forbidden = {'sex', 'race', 'income', 'occupation', 'fnlwgt', 'row_id', 'unit', 'id', 'person_id', 'sampling_id', 'group_id',
                 'hidden_flag', 'y', 'label', 'target', 'teacher', 'teacher_score', 'fine_id'}
    if any(n.lower().split('=')[0] in forbidden for n in names):
        raise ValueError('forbidden deployment feature')
    return list(names)


class Runtime(torch.nn.Module):
    def __init__(self, names, kind):
        super().__init__(); self.names = tuple(validate_names(names)); self.kind = kind
        self.enc = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(83, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 16)) for _ in range(2)])
        self.head = torch.nn.ModuleList([torch.nn.Linear(16, k, dtype=torch.float64) for k in (2, 6)])
        self.B = [None, None]; self.R = [None, None]; self.b = [None, None]; self.scale = [1.0, 1.0]

    def release(self, raw, names, **extra):
        if extra:
            raise ValueError('unregistered inputs or output channels refused')
        if tuple(validate_names(names)) != self.names:
            raise ValueError('wrong or reordered input columns')
        x = np.asarray(raw)
        if x.ndim != 2 or x.shape[1] != 83 or not len(x) or x.dtype.kind != 'f' or not np.isfinite(x).all():
            raise ValueError('finite nonempty floating raw83 required')
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = [[], []]
        with torch.inference_mode():
            for start in range(0, len(x), BATCH_SIZE):
                batch = torch.from_numpy(x[start:start + BATCH_SIZE])
                for i in (0, 1):
                    seq = self.enc[i]; h = batch
                    for k in range(4):
                        h = seq[k](h)
                    z0 = seq[4](h)
                    q = (z0.to(torch.float64) @ self.B[i].T + (h.to(torch.float64) / self.scale[i]) @ self.R[i].T) + self.b[i]
                    z = q @ self.B[i]
                    logits = self.head[i](z); logits = logits - logits.mean(1, keepdim=True)
                    p = torch.softmax(logits, 1); d = torch.tensor(CLASS_ORDERS[i])[p.argmax(1)]
                    full = torch.cat((z, logits, p, d[:, None].to(torch.float64)), 1)
                    if not torch.isfinite(full).all():
                        raise ValueError('nonfinite release refused')
                    out[i].append(full.numpy())
        return [{'full': np.concatenate(o, 0)} for o in out]


def _basis(g, head, i):
    if g.get('source_dimension') != 16 or g.get('class_order') != CLASS_ORDERS[i] or not isinstance(g.get('rank'), int):
        raise ValueError('own-head geometry differs')
    B = torch.tensor(g['B'], dtype=torch.float64).reshape(g['rank'], 16)
    a = head.weight.detach(); a = a - a.mean(0, keepdim=True)
    if (not torch.allclose(B @ B.T, torch.eye(len(B), dtype=torch.float64), atol=1e-10, rtol=1e-10)
            or not torch.allclose((a @ B.T) @ B, a, atol=1e-10, rtol=1e-10)):
        raise ValueError('invalid own-head basis')
    return B


def load_package(directory, expected_manifest_sha256):
    root = Path(directory); mfile = root / 'DEPLOYMENT.json'
    if not expected_manifest_sha256 or any(p.is_symlink() for p in (root, *root.parents)) or not mfile.is_file() or sha_file(mfile) != expected_manifest_sha256:
        raise ValueError('manifest pin differs')
    m = json.loads(mfile.read_text())
    if set(m) != FIELDS or m['schema'] != SCHEMA or m['campaign'] != CAMPAIGN or m['kind'] not in KINDS or m['seed'] not in (0, 1) or m['class_orders'] != CLASS_ORDERS or m['contract'] != CONTRACT:
        raise ValueError('deployment schema or contract differs')
    names = validate_names(m['feature_names'])
    if schema_hash(names) != m['feature_schema_sha256'] or sha_file(root / 'model_state.pt') != m['model_sha256']:
        raise ValueError('model or schema pin differs')
    state = torch.load(root / 'model_state.pt', map_location='cpu', weights_only=True)
    if set(state) != STATE_KEYS or any(v.dtype != (torch.float32 if k.startswith('enc.') else torch.float64) or not torch.isfinite(v).all() for k, v in state.items()):
        raise ValueError('model state differs')
    rt = Runtime(names, m['kind']); rt.load_state_dict(state, strict=True)
    gm = m['geometry']
    if sha_file(root / 'geometry.json') != gm['sha256']:
        raise ValueError('geometry pin differs')
    g = json.loads((root / 'geometry.json').read_text()); digest = g.pop('binding_sha256')
    if digest != json_sha(g) or digest != gm['binding_sha256'] or g.get('campaign') != GEOMETRY_CAMPAIGN or g.get('source_seed') != m['seed'] or g.get('checkpoint_sha256') != gm['start_checkpoint_sha256']:
        raise ValueError('geometry identity differs')
    for i in (0, 1):
        if head_sha(rt.head[i]) != g['head_tensor_sha256'][i]:
            raise ValueError('head differs from frozen geometry')
        rt.B[i] = _basis(g['heads'][i], rt.head[i], i)
    am = m['adapter']
    if sha_file(root / 'adapter.pt') != am['sha256']:
        raise ValueError('adapter pin differs')
    ad = torch.load(root / 'adapter.pt', map_location='cpu', weights_only=True)
    if set(ad) != ADAPTER_KEYS:
        raise ValueError('adapter state differs')
    for i, name in enumerate(('income', 'occupation')):
        R, b, s = ad[f'R_{name}'], ad[f'b_{name}'], float(ad[f'scale_{name}'])
        if R.shape != (len(rt.B[i]), 64) or b.shape != (len(rt.B[i]),) or R.dtype != torch.float64 or b.dtype != torch.float64 or not torch.isfinite(R).all() or not torch.isfinite(b).all():
            raise ValueError('adapter state differs')
        if s < 1.0:
            raise ValueError('scale must be >= 1')
        if m['kind'] == 'RAW_PROJ' and (torch.any(R != 0) or torch.any(b != 0)):
            raise ValueError('RAW_PROJ package must carry exact zero residuals')
        rt.R[i], rt.b[i], rt.scale[i] = R, b, s
    rt.eval(); return rt, m


def _write_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def export(directory, *, service_id, kind, seed, source_payload, source_checkpoint_sha256, geometry_binding,
           states, scales, scientific_status, provenance):
    """Called only by the queued deployment worker. states[i] is {'R','b'} (zero for RAW_PROJ)."""
    if kind not in KINDS:
        raise ValueError('unregistered kind')
    names = validate_names(source_payload['feature_names']); state = {k: v.detach().clone() for k, v in source_payload['model'].items()}
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch.save(state, root / 'model_state.pt')
    _write_json(root / 'geometry.json', geometry_binding)
    payload = {}
    for i, name in enumerate(('income', 'occupation')):
        payload[f'R_{name}'] = states[i]['R'].detach().clone().to(torch.float64)
        payload[f'b_{name}'] = states[i]['b'].detach().clone().to(torch.float64)
        payload[f'scale_{name}'] = float(scales[i])
    torch.save(payload, root / 'adapter.pt')
    adapter = {'file': 'adapter.pt', 'sha256': sha_file(root / 'adapter.pt'),
               'ranks': [int(states[i]['R'].shape[0]) for i in (0, 1)],
               'trainable_parameters': [int(states[i]['R'].numel() + states[i]['b'].numel()) for i in (0, 1)]}
    m = {'schema': SCHEMA, 'campaign': CAMPAIGN, 'service_id': service_id, 'kind': kind, 'seed': seed,
         'feature_names': names, 'feature_schema_sha256': schema_hash(names), 'class_orders': CLASS_ORDERS,
         'model_sha256': sha_file(root / 'model_state.pt'), 'source_checkpoint_sha256': source_checkpoint_sha256,
         'adapter': adapter, 'contract': CONTRACT, 'scientific_status': scientific_status, 'provenance': provenance,
         'geometry': {'file': 'geometry.json', 'sha256': sha_file(root / 'geometry.json'), 'binding_sha256': geometry_binding['binding_sha256'],
                      'start_checkpoint_sha256': geometry_binding['checkpoint_sha256']}}
    _write_json(root / 'DEPLOYMENT.json', m)
    return sha_file(root / 'DEPLOYMENT.json')


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False, description='Emit only the contracted income21/occupation29 wires from raw83.')
    for name in ('package', 'manifest-sha256', 'input', 'input-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args(argv); torch.set_num_threads(1)
    rt, m = load_package(a.package, a.manifest_sha256)
    if sha_file(a.input) != a.input_sha256:
        raise ValueError('input pin differs')
    with np.load(a.input, allow_pickle=False) as z:
        if set(z.files) != {'X', 'feature_names'}:
            raise ValueError('feature-only raw83 input required (no ids, labels or teacher scores)')
        v = rt.release(z['X'], z['feature_names'].tolist())
    with os.fdopen(os.open(a.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as f:
        np.savez(f, r1=v[0]['full'], r2=v[1]['full'])
    print(json.dumps({'schema': SCHEMA, 'rows': len(v[0]['full']), 'service_id': m['service_id'], 'output_sha256': sha_file(a.output),
                      'dimensions': [21, 29, 50], 'scientific_status': m['scientific_status']}))


if __name__ == '__main__':
    main()
