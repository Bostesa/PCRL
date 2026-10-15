"""Own full-pool task arithmetic on UPDATE+GUARD (10223 rows) for the residual adapters."""
from __future__ import annotations
import hashlib

CHUNK = 1024
PROBE_SALT = 'SPG1|UPDATE-wire-probe|20261013|'


def load_pool(cfg_train, get, src, B, scales):
    """Return per-recipient frozen features (full-batch per role, as registered) and labels, plus probe positions."""
    import numpy as np
    import torch
    from .common import npz
    roles = cfg_train['roles']
    data = {}
    for role in ('UPDATE', 'GUARD'):
        f = npz(get(roles[role]['features']))
        assert tuple(map(str, f['feature_names'])) == src.names
        d = {'X': f['X'], 'row_id': f['row_id'], 'unit': f['unit']}
        for t in ('income', 'occupation'):
            z = npz(get(roles[role][t]))
            assert np.array_equal(z['row_id'], f['row_id']) and np.array_equal(z['unit'], f['unit'])
            d[t] = z[t]
        data[role] = d
    assert len(data['UPDATE']['row_id']) == 8180 and len(data['GUARD']['row_id']) == 2043
    feats = []
    with torch.no_grad():
        for i, t in enumerate(('income', 'occupation')):
            out = {}
            for role, key in (('UPDATE', 'U'), ('GUARD', 'G')):
                x = torch.from_numpy(data[role]['X'])
                h = src.h64(i, x); z0 = src.z0(i, h)
                out['h' + key] = h.to(torch.float64) / float(scales[i])
                out['q' + key] = z0.to(torch.float64) @ B[i].T
                out['y' + key] = torch.from_numpy(data[role][t]).long()
            out['hP'] = torch.cat((out['hU'], out['hG'])); out['qP'] = torch.cat((out['qU'], out['qG'])); out['yP'] = torch.cat((out['yU'], out['yG']))
            feats.append(out)
    rid = data['UPDATE']['row_id']
    keys = [hashlib.sha256((PROBE_SALT + str(r)).encode()).hexdigest() for r in rid]
    probe = sorted(range(len(rid)), key=lambda k: (keys[k], k))[:512]
    return feats, probe, data


def logits_of(src, B, i, hbar, q0, R, b):
    import torch
    import torch.nn.functional as F
    z = ((q0 + hbar @ R.T) + b) @ B[i]
    w, bb = src.head[i]
    l = F.linear(z, w, bb)
    return l - l.mean(1, keepdim=True), z


def pool_metrics(src, B, i, f, R, b, lp0=None, grads=False):
    """Mean NLL (floor 1e-12), summed Brier, correct, KL(p0||p) over the 10223-row pool in 1024-row chunks; optional gradients."""
    import torch
    import torch.nn.functional as F
    n = len(f['yP']); K = (2, 6)[i]
    sums = {'nll': 0.0, 'brier': 0.0, 'kl': 0.0}; correct = 0
    g = {k: None for k in ('nll', 'brier', 'kl')}
    if grads:
        R = R.detach().clone().requires_grad_(True); b = b.detach().clone().requires_grad_(True)
    for s in range(0, n, CHUNK):
        sl = slice(s, s + CHUNK)
        with (torch.enable_grad() if grads else torch.no_grad()):
            l, _ = logits_of(src, B, i, f['hP'][sl], f['qP'][sl], R, b)
            p = torch.softmax(l, 1); y = f['yP'][sl]; idx = torch.arange(len(y))
            terms = {'nll': -p[idx, y].clamp_min(1e-12).log().sum(),
                     'brier': (p - F.one_hot(y, K).to(torch.float64)).square().sum()}
            if lp0 is not None:
                lp = F.log_softmax(l, 1); q = lp0[sl]
                terms['kl'] = (q.exp() * (q - lp)).sum()
            correct += int((p.argmax(1) == y).sum())
            for k, v in terms.items():
                sums[k] += float(v.detach())
                if grads:
                    gr = torch.autograd.grad(v, (R, b), retain_graph=True)
                    flat = torch.cat([gr[0].reshape(-1), gr[1].reshape(-1)]).numpy()
                    g[k] = flat if g[k] is None else g[k] + flat
    out = {k: v / n for k, v in sums.items()}
    out['correct'] = correct; out['rows'] = n
    if lp0 is None:
        out['kl'] = None
    if grads:
        out['gradients'] = {k: v / n for k, v in g.items() if v is not None}
    return out


def teacher(src, B, i, f):
    import torch
    import torch.nn.functional as F
    rank = B[i].shape[0]
    with torch.no_grad():
        l, _ = logits_of(src, B, i, f['hP'], f['qP'], torch.zeros(rank, 64, dtype=torch.float64), torch.zeros(rank, dtype=torch.float64))
        return F.log_softmax(l, 1)


def split(vec, rank):
    import torch
    v = torch.as_tensor(vec, dtype=torch.float64)
    return v[:rank * 64].reshape(rank, 64).clone(), v[rank * 64:].clone()


def probe_wire(src, B, i, f, probe, R, b):
    import torch
    with torch.no_grad():
        l, z = logits_of(src, B, i, f['hU'][probe], f['qU'][probe], R, b)
        p = torch.softmax(l, 1)
        return {'representation': z.numpy(), 'centered_logits': l.numpy(), 'probabilities': p.numpy(), 'decision': p.argmax(1).numpy()}
