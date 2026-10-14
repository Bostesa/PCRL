"""Constrained occupation task-head adapter over the frozen RAW_JOINT trunk.

Imports perform no numerical work. Every numerical entrypoint requires the
active campaign queue. Arithmetic order (fixed, float64 after the trunk):

  h64     = enc[1][0:4](x)                (source float32 trunk, unchanged)
  z0      = enc[1][4](h64)                (source float32 final map to z16)
  q0      = float64(z0) @ B^T             (inherited factorized projection)
  hbar64  = float64(h64) / s              (one immutable scale s >= 1)
  q       = (q0 + hbar64 @ R^T) + b_res
  z_rel   = q @ B
  outputs = own_wire(z_rel, frozen folded own head, class order)

At R=0, b_res=0 the occupation wire equals the inherited RAW_PROJ wire. The
income recipient is the admitted factorized RAW income projection, unchanged.
No h64, z0, q0, q, teacher output or residual is ever transmitted.
"""
from __future__ import annotations
import hashlib
import json

CAMPAIGN = 'pcrl_minimal_task_wire_adapter_v1'
SCALE_RULE = 's = max(1, sqrt(mean_x ||h64(x)||^2 / 64)) over the authorized full task fitting pool (UPDATE+GUARD), float64'
HIDDEN_LAYER = 'enc.1 modules 0..3 (Linear 83->64, ReLU, Linear 64->64, ReLU): final 64-d activation before enc.1.4 (Linear 64->16)'
ADAPTER_CONTRACT = {
    'schema': 'MTA1-task-wire-adapter-contract-v1',
    'campaign': CAMPAIGN,
    'income': 'admitted factorized RAW income projection z Bi^T Bi with frozen own head (byte-identical RAW_PROJ income wire)',
    'occupation': 'z_release = (q0 + hbar64 R^T + b_res) B with q0 = float64(z0) B^T; frozen folded own head outputs',
    'hidden_layer': HIDDEN_LAYER,
    'scale_rule': SCALE_RULE,
    'widths': [21, 29, 50],
    'class_order': [[0, 1], list(range(6))],
    'nominal_layout': 'projected representation(16), own centered logits(K), own probabilities(K), own decision(1)',
    'decision_rule': 'argmax probabilities; first class in fixed source order',
    'never_transmitted': ['h64', 'original z16', 'discarded residual', 'q0 alongside q', 'teacher probabilities',
                          'unprotected task output', 'per-person adapter state', 'labels', 'row ids', 'fine ids'],
    'role_forward_batch_size': 1024,
}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _queued():
    from .scheduler import assert_active_from_env
    return assert_active_from_env()


def tensor_sha(value):
    a = value.detach().cpu().contiguous().numpy()
    h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def hidden64(model, x):
    """Final 64-d trunk activation immediately before enc[1][4] (source float32)."""
    seq = model.enc[1]
    h = x.to(dtype=__import__('torch').float32)
    for i in range(4):
        h = seq[i](h)
    return h


def z_from_hidden(model, h):
    return model.enc[1][4](h)


class Adapter:
    """Frozen per-seed source pieces plus the trainable residual (R, b_res)."""
    def __init__(self, model, geometry, scale, *, R=None, b=None):
        import torch
        _queued()
        self.model = model
        self.geometry = geometry
        self.B = geometry.bases[1].to(torch.float64)            # rank x 16, inherited bytes
        self.Bi = geometry.bases[0].to(torch.float64)
        self.rank = int(self.B.shape[0])
        self.head = model.head[1]
        self.order = model.class_orders[1]
        if not (isinstance(scale, float) and scale >= 1.0):
            raise ValueError('immutable scale s >= 1 (python float) required')
        self.scale = scale
        self.R = torch.zeros(self.rank, 64, dtype=torch.float64) if R is None else R.detach().clone().to(torch.float64)
        self.b = torch.zeros(self.rank, dtype=torch.float64) if b is None else b.detach().clone().to(torch.float64)
        if self.R.shape != (self.rank, 64) or self.b.shape != (self.rank,):
            raise ValueError('residual shape differs from measured head rank')

    def parameters(self):
        return [self.R, self.b]

    def n_parameters(self):
        return self.R.numel() + self.b.numel()

    # Cached frozen pieces -------------------------------------------------
    def frozen_features(self, x):
        """Return (hbar64, q0, income wire) for raw83 rows; no gradient."""
        import torch
        from task_conditioned_representation.models import own_wire
        with torch.no_grad():
            h = hidden64(self.model, x)
            z0 = z_from_hidden(self.model, h)
            q0 = z0.to(torch.float64) @ self.B.T
            hbar = h.to(torch.float64) / self.scale
            zi = self.model.encode(0, x)
            zi = (zi.to(torch.float64) @ self.Bi.T) @ self.Bi
            income = own_wire(zi, self.model.head[0], self.model.class_orders[0])['full']
        return hbar, q0, income

    def q(self, hbar, q0, R=None, b=None):
        R = self.R if R is None else R; b = self.b if b is None else b
        return (q0 + hbar @ R.T) + b

    def wire_from_q(self, q):
        from task_conditioned_representation.models import own_wire
        return own_wire(q @ self.B, self.head, self.order)

    def wire(self, hbar, q0, R=None, b=None):
        return self.wire_from_q(self.q(hbar, q0, R, b))

    def state(self):
        return {'R': self.R.detach().clone(), 'b': self.b.detach().clone()}


def fit_scale(model, x_pool):
    """Immutable s on the full authorized fitting pool; returns (s, norms record)."""
    import torch
    _queued()
    with torch.no_grad():
        h = hidden64(model, x_pool).to(torch.float64)
        sq = (h * h).sum(1)
        mean_sq = float(sq.mean())
        s = max(1.0, (mean_sq / 64.0) ** 0.5)
        record = {'rows': int(len(h)), 'mean_squared_norm': mean_sq, 'rms_per_coordinate': (mean_sq / 64.0) ** 0.5,
                  'median_norm': float(sq.sqrt().median()), 'max_norm': float(sq.sqrt().max()),
                  'min_norm': float(sq.sqrt().min()), 'fraction_zero_coordinates': float((h == 0).double().mean()),
                  'per_coordinate_std_min': float(h.std(0, correction=0).min()),
                  'per_coordinate_std_max': float(h.std(0, correction=0).max()),
                  'scale': s, 'rule': SCALE_RULE}
    return s, record


def head_norms(model, geometry):
    import torch
    W = model.head[1].weight.detach().to(torch.float64); b = model.head[1].bias.detach().to(torch.float64)
    A = W - W.mean(0, keepdim=True)
    return {'occupation_head_weight_frobenius': float(W.norm()), 'occupation_centered_weight_frobenius': float(A.norm()),
            'occupation_centered_bias_norm': float((b - b.mean()).norm()),
            'occupation_centered_singular_values': [float(v) for v in torch.linalg.svdvals(A)],
            'occupation_rank': int(geometry.bases[1].shape[0]), 'income_rank': int(geometry.bases[0].shape[0]),
            'final_map_64_to_16_frobenius': float(model.enc[1][4].weight.detach().double().norm())}


def forward_raw(adapter, raw, names):
    """Raw83 deployment-path forward with fixed batching (no cached features)."""
    _queued()
    import numpy as np
    import torch
    if tuple(map(str, names)) != tuple(adapter.model.names):
        raise ValueError('wrong or reordered raw83 schema')
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 83 or not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all():
        raise ValueError('finite raw83 matrix required')
    if not len(raw):
        raise ValueError('nonempty input required')
    out = [{}, {}]
    with torch.no_grad():
        for start in range(0, len(raw), ADAPTER_CONTRACT['role_forward_batch_size']):
            x = torch.from_numpy(np.ascontiguousarray(raw[start:start + 1024], dtype=np.float32))
            hbar, q0, income = adapter.frozen_features(x)
            occ = adapter.wire(hbar, q0)
            for i, full in enumerate((income, occ['full'])):
                out[i].setdefault('wire', []).append(full.numpy())
    return [{'wire': np.concatenate(o['wire'], 0)} for o in out]


def wire_contract_check(adapter, hbar, q0, full_wire):
    """Check a released occupation wire against the contract; returns failures.

    The representation must lie in row(B) (no discarded residual), and the
    logits/probabilities/decision must be the frozen own head recomputed from
    that representation; widths and decision layout are fixed.
    """
    import torch
    from task_conditioned_representation.models import own_wire
    failures = []
    w = torch.as_tensor(full_wire, dtype=torch.float64)
    if w.ndim != 2 or w.shape[1] != 29:
        return ['occupation wire width differs from 29']
    z = w[:, :16]
    off = z - (z @ adapter.B.T) @ adapter.B
    if float(off.abs().max()) > 1e-12 * max(1.0, float(z.abs().max())):
        failures.append('representation leaves the occupation head row space')
    expected = own_wire(z, adapter.head, adapter.order)
    if float((expected['centered_logits'] - w[:, 16:22]).abs().max()) > 1e-12:
        failures.append('released logits are not the frozen own head of the released representation')
    if float((expected['probabilities'] - w[:, 22:28]).abs().max()) > 1e-12:
        failures.append('released probabilities differ from own head')
    if not torch.equal(expected['decision'].to(torch.float64), w[:, 28]):
        failures.append('released decision differs from own argmax')
    canonical = adapter.wire(hbar, q0)['full']
    if not torch.equal(canonical, w):
        failures.append('released wire differs from canonical factorized adapter arithmetic')
    return failures
