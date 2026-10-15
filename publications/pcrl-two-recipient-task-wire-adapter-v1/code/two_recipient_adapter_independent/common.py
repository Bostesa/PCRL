"""Independent verifier: shared arithmetic written from the registered contract (no campaign numerics imported).

Only two_recipient_adapter.evaluation_queue.queue_config is used, for queue binding.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

W = Path(__file__).resolve().parents[1]
PRIV = W / '.private' / 'two_recipient_task_wire_adapter'
OUT = PRIV / 'independent'
RES = W / 'results' / 'pcrl_two_recipient_task_wire_adapter_v1'
KS = (2, 6)
WIDTHS = (21, 29)
NAMES = ('income', 'occupation')
BATCH = 1024
NLL_FLOOR = 1e-12


def bind(spec_path, inner=False):
    from two_recipient_adapter.evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'verification', allow_inner_truth=inner)
    import torch
    torch.set_num_threads(1)

    def get(item):
        return verify(item['path'], item['sha256'])
    return active, cfg, get, guard


def file_sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def arr_sha(a):
    import numpy as np
    z = np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()


def tsha(t):
    a = t.detach().cpu().contiguous().numpy()
    h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def npz(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def tokenize(obj):
    """Replace absolute paths/usernames by tokens for public copies."""
    s = json.dumps(obj, sort_keys=True, default=float)
    home = str(Path.home())
    reps = [(str(PRIV), '$PRIV'), (str(W), '$W'),
            (home + '/PCRL/.worktrees/pcrl-minimal-task-wire-adapter-v1', '$PREV'),
            (home + '/PCRL/.worktrees/pcrl-separate-pool-utility-guard-v1', '$SPG'),
            (home + '/PCRL/.worktrees/pcrl-compressed-income-safeguard-v1', '$OLD'),
            (home + '/PCRL/.worktrees/pcrl-task-conditioned-representations-v1', '$TCR'),
            (home + '/PCRL/.worktrees/pcrl-utility-reference-repair-v1', '$URR'),
            (home + '/PCRL', '$PCRL'), (home, '$HOME'), (os.environ.get('USER', ''), '$USER')]
    for a, b in reps:
        s = s.replace(a, b)
    return json.loads(s)


def write_out(cfg, result):
    out = Path(cfg['output'])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, default=float)
    return result


class Source:
    """Own forward of the frozen RAW_JOINT trunk/heads from the raw state dict."""
    def __init__(self, ckpt_path):
        import torch
        payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = payload['model']
        self.payload = payload
        self.names = tuple(map(str, payload['feature_names']))
        self.enc = []
        for i in (0, 1):
            layers = []
            for k, (din, dout) in zip((0, 2, 4), ((83, 64), (64, 64), (64, 16))):
                w = sd[f'enc.{i}.{k}.weight']; b = sd[f'enc.{i}.{k}.bias']
                assert w.dtype == torch.float32 and tuple(w.shape) == (dout, din) and tuple(b.shape) == (dout,)
                layers.append((w, b))
            self.enc.append(layers)
        self.head = []
        for i, K in enumerate(KS):
            w = sd[f'head.{i}.weight']; b = sd[f'head.{i}.bias']
            assert w.dtype == torch.float64 and tuple(w.shape) == (K, 16)
            self.head.append((w, b))
        self.state_sha = {k: tsha(v) for k, v in sd.items()}

    def h64(self, i, x):
        import torch
        import torch.nn.functional as F
        (w0, b0), (w2, b2), _ = self.enc[i]
        h = torch.relu(F.linear(x.to(torch.float32), w0, b0))
        return torch.relu(F.linear(h, w2, b2))

    def z0(self, i, h):
        import torch.nn.functional as F
        w4, b4 = self.enc[i][2]
        return F.linear(h, w4, b4)

    def wire(self, i, z):
        import torch
        import torch.nn.functional as F
        w, b = self.head[i]
        logits = F.linear(z.to(torch.float64), w, b)
        logits = logits - logits.mean(1, keepdim=True)
        p = torch.softmax(logits, 1)
        dec = p.argmax(1)
        full = torch.cat((z.to(torch.float64), logits, p, dec[:, None].to(torch.float64)), 1)
        return {'z': z, 'logits': logits, 'p': p, 'dec': dec, 'full': full}


def bases(geometry_path):
    import torch
    g = json.loads(Path(geometry_path).read_text())
    out = []
    for h in g['heads']:
        out.append(torch.tensor(h['B'], dtype=torch.float64).reshape(h['rank'], 16))
    return out, g


def role_forward(src, B, X, names, states=None, scales=(1.0, 1.0), mode='adapter'):
    """Raw83 -> [income full wire, occupation full wire] numpy, fixed 1024 batches.

    mode 'raw_proj': z = (float64(z0) B^T) B; mode 'adapter': z = (q0 + hbar R^T + b) B with q0 = float64(z0) B^T.
    states[i] None means exact zero residual.
    """
    import numpy as np
    import torch
    assert tuple(map(str, names)) == src.names
    X = np.asarray(X)
    outs = [[], []]
    with torch.no_grad():
        for s in range(0, len(X), BATCH):
            x = torch.from_numpy(np.ascontiguousarray(X[s:s + BATCH], dtype=np.float32))
            for i in (0, 1):
                h = src.h64(i, x); z0 = src.z0(i, h)
                if mode == 'raw_proj':
                    z = (z0.to(torch.float64) @ B[i].T) @ B[i]
                else:
                    q0 = z0.to(torch.float64) @ B[i].T
                    rank = B[i].shape[0]
                    R = torch.zeros(rank, 64, dtype=torch.float64) if states is None or states[i] is None else states[i]['R'].to(torch.float64)
                    b = torch.zeros(rank, dtype=torch.float64) if states is None or states[i] is None else states[i]['b'].to(torch.float64)
                    hbar = h.to(torch.float64) / float(scales[i])
                    z = ((q0 + hbar @ R.T) + b) @ B[i]
                outs[i].append(src.wire(i, z)['full'].numpy())
    return [np.concatenate(o, 0) for o in outs]


def task_metrics_np(y, p, K):
    """Per-row accuracy, NLL (1e-12 floor), summed Brier from a probability matrix (own implementation)."""
    import numpy as np
    y = np.asarray(y).astype(np.int64); p = np.asarray(p, dtype=np.float64)
    assert p.shape == (len(y), K)
    onehot = np.zeros_like(p); onehot[np.arange(len(y)), y] = 1.0
    acc = (p.argmax(1) == y).astype(np.float64)
    nll = -np.log(np.maximum(p[np.arange(len(y)), y], NLL_FLOOR))
    brier = ((p - onehot) ** 2).sum(1)
    return {'accuracy': acc, 'nll': nll, 'brier': brier}


def maxabs(a, b):
    import numpy as np
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float('inf')
    return float(np.max(np.abs(a - b))) if a.size else 0.0
