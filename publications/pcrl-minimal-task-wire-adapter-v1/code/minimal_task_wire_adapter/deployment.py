"""Standalone raw83 runtime for the minimal task-wire adapter services.

No source loader, label, teacher score or private absolute path is required.
Duplicates the canonical arithmetic; it does not import campaign workers.
Services: RAW_PROJ (zero residual), ADAPTER (saved R, b_res on the occupation
head row space), OWNPROJ (prior checkpoint occupation projected on its own head).
Emits only the contracted income21/occupation29 recipient wires.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch

SCHEMA = 'MTA1-raw83-task-wire-adapter-deployment-v1'
CAMPAIGN = 'pcrl_minimal_task_wire_adapter_v1'
GEOMETRY_CAMPAIGN = 'pcrl_compressed_income_safeguard_v1'
BATCH_SIZE = 1024
CLASS_ORDERS = [[0, 1], list(range(6))]
KINDS = ('RAW_PROJ', 'ADAPTER', 'OWNPROJ')
STATE_KEYS = {f'enc.{i}.{layer}.{kind}' for i in range(2) for layer in (0, 2, 4) for kind in ('weight', 'bias')} | {f'head.{i}.{kind}' for i in range(2) for kind in ('weight', 'bias')}
CONTRACT = {'recipient_full_dimensions': [21, 29], 'pair_full_dimensions': 50, 'deterministic': True,
            'occupation': 'z=((float64(z0)@B.T + float64(h64)/s@R.T) + b_res)@B; own frozen head on z',
            'income': 'z=(float64(z_income)@Bi.T)@Bi; own frozen head',
            'decision_rule': 'argmax probabilities; first class in fixed source order', 'role_forward_batch_size': BATCH_SIZE,
            'never_emitted': ['h64', 'z16', 'q', 'residual', 'teacher probabilities', 'labels', 'row ids', 'fine ids', 'original features']}
FIELDS = {'schema', 'campaign', 'service_id', 'kind', 'seed', 'feature_names', 'feature_schema_sha256', 'class_orders',
          'model_sha256', 'source_checkpoint_sha256', 'adapter', 'geometry', 'ownproj', 'contract', 'scientific_status', 'provenance'}


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
        self.Bi = self.Bo = self.R = self.b = None; self.scale = 1.0

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
                zi = (self.enc[0](batch).to(torch.float64) @ self.Bi.T) @ self.Bi
                seq = self.enc[1]; h = batch
                for i in range(4):
                    h = seq[i](h)
                z0 = seq[4](h)
                if self.kind == 'OWNPROJ':
                    zo = (z0.to(torch.float64) @ self.Bo.T) @ self.Bo
                else:
                    q = (z0.to(torch.float64) @ self.Bo.T + (h.to(torch.float64) / self.scale) @ self.R.T) + self.b
                    zo = q @ self.Bo
                for i, z in enumerate((zi, zo)):
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
    if head_sha(rt.head[0]) != g['head_tensor_sha256'][0]:
        raise ValueError('income head differs from frozen geometry')
    rt.Bi = _basis(g['heads'][0], rt.head[0], 0)
    if m['kind'] == 'OWNPROJ':
        om = m['ownproj']
        if sha_file(root / 'ownproj.json') != om['sha256']:
            raise ValueError('own-head projection pin differs')
        o = json.loads((root / 'ownproj.json').read_text()); od = o.pop('binding_sha256')
        if od != json_sha(o) or od != om['binding_sha256'] or o['occupation_head_tensor_sha256'] != head_sha(rt.head[1]):
            raise ValueError('own-head projection identity differs')
        rt.Bo = _basis(o['occupation_geometry'], rt.head[1], 1)
    else:
        if head_sha(rt.head[1]) != g['head_tensor_sha256'][1]:
            raise ValueError('occupation head differs from frozen geometry')
        rt.Bo = _basis(g['heads'][1], rt.head[1], 1)
        am = m['adapter']
        if sha_file(root / 'adapter.pt') != am['sha256']:
            raise ValueError('adapter pin differs')
        ad = torch.load(root / 'adapter.pt', map_location='cpu', weights_only=True)
        if set(ad) != {'R', 'b_res', 'scale'} or ad['R'].shape != (len(rt.Bo), 64) or ad['b_res'].shape != (len(rt.Bo),) or ad['R'].dtype != torch.float64:
            raise ValueError('adapter state differs')
        if m['kind'] == 'RAW_PROJ' and (torch.any(ad['R'] != 0) or torch.any(ad['b_res'] != 0)):
            raise ValueError('RAW_PROJ package must carry an exact zero residual')
        rt.R, rt.b, rt.scale = ad['R'], ad['b_res'], float(ad['scale'])
        if rt.scale < 1.0:
            raise ValueError('scale must be >= 1')
    rt.eval(); return rt, m


def _write_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def export(directory, *, service_id, kind, seed, source_payload, source_checkpoint_sha256, geometry_binding,
           adapter_state=None, ownproj=None, scientific_status, provenance):
    """Called only by the queued deployment worker."""
    if kind not in KINDS:
        raise ValueError('unregistered kind')
    names = validate_names(source_payload['feature_names']); state = {k: v.detach().clone() for k, v in source_payload['model'].items()}
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch.save(state, root / 'model_state.pt')
    _write_json(root / 'geometry.json', geometry_binding)
    adapter = None; own = None
    if kind in ('RAW_PROJ', 'ADAPTER'):
        torch.save({'R': adapter_state['R'].detach().clone().to(torch.float64), 'b_res': adapter_state['b'].detach().clone().to(torch.float64),
                    'scale': float(adapter_state['scale'])}, root / 'adapter.pt')
        adapter = {'file': 'adapter.pt', 'sha256': sha_file(root / 'adapter.pt'), 'rank': int(adapter_state['R'].shape[0]),
                   'trainable_parameters': int(adapter_state['R'].numel() + adapter_state['b'].numel())}
    else:
        _write_json(root / 'ownproj.json', ownproj)
        own = {'file': 'ownproj.json', 'sha256': sha_file(root / 'ownproj.json'), 'binding_sha256': ownproj['binding_sha256']}
    m = {'schema': SCHEMA, 'campaign': CAMPAIGN, 'service_id': service_id, 'kind': kind, 'seed': seed,
         'feature_names': names, 'feature_schema_sha256': schema_hash(names), 'class_orders': CLASS_ORDERS,
         'model_sha256': sha_file(root / 'model_state.pt'), 'source_checkpoint_sha256': source_checkpoint_sha256,
         'adapter': adapter, 'ownproj': own, 'contract': CONTRACT, 'scientific_status': scientific_status, 'provenance': provenance,
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
