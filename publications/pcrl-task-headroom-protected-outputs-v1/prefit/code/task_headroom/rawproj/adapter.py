"""Constrained two-recipient task-wire adapters over the frozen RAW_JOINT trunks.

Imports perform no numerical work. Every numerical entrypoint requires the
active campaign queue. Per recipient i (0 = income, 1 = occupation), fixed
arithmetic order (float64 after the source float32 trunk):

  h64_i   = enc[i][0:4](x)                (source float32 trunk, unchanged)
  z0_i    = enc[i][4](h64_i)              (source float32 final map to z16)
  q0_i    = float64(z0_i) @ B_i^T         (inherited factorized projection)
  hbar_i  = float64(h64_i) / s_i          (one immutable scale s_i >= 1)
  q_i     = (q0_i + hbar_i @ R_i^T) + b_i
  z_i     = q_i @ B_i
  outputs = own_wire(z_i, frozen folded own head i, class order i)

own_wire centers the head logits across classes, which equals A_i z + c_i with
A = W - mean_k W_k and c = b - mean(b); A_i B_i^T B_i = A_i is admitted. At
R_i = 0, b_i = 0 the wire equals the inherited RAW_PROJ wire for that recipient.
No h64, z0, q0, q, teacher output or residual is ever transmitted.
"""
from __future__ import annotations
import hashlib
import json

CAMPAIGN = 'pcrl_two_recipient_task_wire_adapter_v1'
RECIPIENTS = ('income', 'occupation')
WIDTHS = (21, 29)
KS = (2, 6)
SCALE_RULE = 's_i = max(1, sqrt(mean_x ||h64_i(x)||^2 / 64)) over the authorized fitting pool (UPDATE+GUARD, 10223 rows), float64'
HIDDEN_LAYER = 'enc.i modules 0..3 (Linear 83->64, ReLU, Linear 64->64, ReLU): final 64-d activation before enc.i.4 (Linear 64->16)'
ADAPTER_CONTRACT = {
    'schema': 'TRA1-two-recipient-task-wire-adapter-contract-v1',
    'campaign': CAMPAIGN,
    'recipients': list(RECIPIENTS),
    'per_recipient': 'z_i = (q0_i + hbar64_i R_i^T + b_i) B_i with q0_i = float64(z0_i) B_i^T; frozen folded own head i outputs',
    'hidden_layer': HIDDEN_LAYER,
    'scale_rule': SCALE_RULE,
    'widths': [21, 29, 50],
    'effective_ranks_expected': [1, 5],
    'class_order': [[0, 1], list(range(6))],
    'nominal_layout': 'projected representation(16), own centered logits(K), own probabilities(K), own decision(1)',
    'decision_rule': 'argmax probabilities; first class in fixed source order',
    'never_transmitted': ['h64', 'original z16', 'discarded residual', 'q0 alongside q', 'teacher probabilities',
                          'another head output', 'unprotected task output', 'per-person adapter or controller state',
                          'labels', 'row ids', 'fine ids'],
    'role_forward_batch_size': 1024,
}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _queued():
    from task_headroom.scheduler import assert_active_from_env
    return assert_active_from_env()


def tensor_sha(value):
    a = value.detach().cpu().contiguous().numpy()
    h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def hidden64(model, i, x):
    """Final 64-d activation of recipient i's trunk immediately before enc[i][4] (source float32)."""
    import torch
    seq = model.enc[i]
    h = x.to(dtype=torch.float32)
    for k in range(4):
        h = seq[k](h)
    return h


def z_from_hidden(model, i, h):
    return model.enc[i][4](h)


class RecipientAdapter:
    """Frozen per-seed source pieces of one recipient plus its trainable residual (R, b)."""
    def __init__(self, model, geometry, i, scale, *, R=None, b=None):
        import torch
        _queued()
        if i not in (0, 1):
            raise ValueError('recipient index 0 (income) or 1 (occupation)')
        self.i = i
        self.model = model
        self.geometry = geometry
        self.B = geometry.bases[i].to(torch.float64)            # rank x 16, inherited bytes
        self.rank = int(self.B.shape[0])
        self.head = model.head[i]
        self.order = model.class_orders[i]
        self.K = KS[i]
        if not (isinstance(scale, float) and scale >= 1.0):
            raise ValueError('immutable scale s >= 1 (python float) required')
        self.scale = scale
        self.R = torch.zeros(self.rank, 64, dtype=torch.float64) if R is None else R.detach().clone().to(torch.float64)
        self.b = torch.zeros(self.rank, dtype=torch.float64) if b is None else b.detach().clone().to(torch.float64)
        if self.R.shape != (self.rank, 64) or self.b.shape != (self.rank,):
            raise ValueError('residual shape differs from measured head rank')

    def n_parameters(self):
        return self.R.numel() + self.b.numel()

    def frozen_features(self, x):
        """Return (hbar64, q0) for raw83 rows; no gradient."""
        import torch
        with torch.no_grad():
            h = hidden64(self.model, self.i, x)
            z0 = z_from_hidden(self.model, self.i, h)
            q0 = z0.to(torch.float64) @ self.B.T
            hbar = h.to(torch.float64) / self.scale
        return hbar, q0

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

    def is_zero(self):
        import torch
        return bool(torch.all(self.R == 0) and torch.all(self.b == 0))


def fit_scale(model, i, x_pool):
    """Immutable s_i on the full authorized fitting pool; returns (s, norms record)."""
    import torch
    _queued()
    with torch.no_grad():
        h = hidden64(model, i, x_pool).to(torch.float64)
        sq = (h * h).sum(1)
        mean_sq = float(sq.mean())
        s = max(1.0, (mean_sq / 64.0) ** 0.5)
        record = {'recipient': RECIPIENTS[i], 'rows': int(len(h)), 'mean_squared_norm': mean_sq,
                  'rms_per_coordinate': (mean_sq / 64.0) ** 0.5, 'median_norm': float(sq.sqrt().median()),
                  'max_norm': float(sq.sqrt().max()), 'min_norm': float(sq.sqrt().min()),
                  'fraction_zero_coordinates': float((h == 0).double().mean()),
                  'per_coordinate_std_min': float(h.std(0, correction=0).min()),
                  'per_coordinate_std_max': float(h.std(0, correction=0).max()), 'scale': s, 'rule': SCALE_RULE}
    return s, record


def head_record(model, geometry, i):
    import torch
    W = model.head[i].weight.detach().to(torch.float64); b = model.head[i].bias.detach().to(torch.float64)
    A = W - W.mean(0, keepdim=True); c = b - b.mean()
    B = geometry.bases[i].to(torch.float64)
    return {'recipient': RECIPIENTS[i], 'head_weight_frobenius': float(W.norm()), 'centered_weight_frobenius': float(A.norm()),
            'centered_bias_norm': float(c.norm()), 'centered_singular_values': [float(v) for v in torch.linalg.svdvals(A)],
            'measured_centered_rank': int(torch.linalg.matrix_rank(A)), 'basis_rank': int(B.shape[0]),
            'A_Bt_B_minus_A_max_abs': float((A @ B.T @ B - A).abs().max()),
            'B_Bt_minus_I_max_abs': float((B @ B.T - torch.eye(len(B), dtype=torch.float64)).abs().max()),
            'final_map_64_to_16_frobenius': float(model.enc[i][4].weight.detach().double().norm())}


def forward_raw(model, geometry, scales, states, raw, names):
    """Raw83 deployment-path forward for both recipients with fixed batching (no cached features).

    states[i] is None (zero residual) or {'R','b'}; scales[i] is s_i.
    Returns [income dict, occupation dict] of numpy arrays.
    """
    _queued()
    import numpy as np
    import torch
    if tuple(map(str, names)) != tuple(model.names):
        raise ValueError('wrong or reordered raw83 schema')
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 83 or not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all():
        raise ValueError('finite raw83 matrix required')
    if not len(raw):
        raise ValueError('nonempty input required')
    adapters = [RecipientAdapter(model, geometry, i, float(scales[i]),
                                 R=None if states[i] is None else states[i]['R'],
                                 b=None if states[i] is None else states[i]['b']) for i in (0, 1)]
    out = [{}, {}]
    with torch.no_grad():
        for start in range(0, len(raw), ADAPTER_CONTRACT['role_forward_batch_size']):
            x = torch.from_numpy(np.ascontiguousarray(raw[start:start + 1024], dtype=np.float32))
            for i, ad in enumerate(adapters):
                hbar, q0 = ad.frozen_features(x)
                w = ad.wire(hbar, q0)
                for target, key in (('representation', 'representation'), ('centered_logits', 'centered_logits'),
                                    ('probability', 'probabilities'), ('decision', 'decision'), ('wire', 'full')):
                    out[i].setdefault(target, []).append(w[key].detach().cpu().numpy())
    return [{k: np.concatenate(v, 0) for k, v in o.items()} for o in out]


def wire_contract_check(adapter, hbar, q0, full_wire):
    """Check a released recipient wire against the contract; returns failures."""
    import torch
    from task_conditioned_representation.models import own_wire
    failures = []
    K = adapter.K; width = 16 + 2 * K + 1
    w = torch.as_tensor(full_wire, dtype=torch.float64)
    if w.ndim != 2 or w.shape[1] != width:
        return [f'{RECIPIENTS[adapter.i]} wire width differs from {width}']
    z = w[:, :16]
    off = z - (z @ adapter.B.T) @ adapter.B
    if float(off.abs().max()) > 1e-12 * max(1.0, float(z.abs().max())):
        failures.append('representation leaves the head row space')
    expected = own_wire(z, adapter.head, adapter.order)
    if float((expected['centered_logits'] - w[:, 16:16 + K]).abs().max()) > 1e-12:
        failures.append('released logits are not the frozen own head of the released representation')
    if float((expected['probabilities'] - w[:, 16 + K:16 + 2 * K]).abs().max()) > 1e-12:
        failures.append('released probabilities differ from own head')
    if not torch.equal(expected['decision'].to(torch.float64), w[:, -1]):
        failures.append('released decision differs from own argmax')
    canonical = adapter.wire(hbar, q0)['full']
    if not torch.equal(canonical, w):
        failures.append('released wire differs from canonical factorized adapter arithmetic')
    return failures
