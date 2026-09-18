"""Service-only baselines and residual-logit attackers.

For protected role `j` the service-only predictor `p0_j` sees that role's **`H` view
alone** -- `H_A` for a local role, `[H_A, H_B]` for a coalition role -- and is frozen
before the mapper moves. Every full-view attacker sees the same service view plus the
released channel `Z`, and is parameterised as a **correction to `p0_j`'s logits**:

    q_jk(s | H, Z) = softmax( logits_p0_j(H) + d_jk(H, Z) )

so the zero correction recovers `p0_j` exactly, and the empirical incremental gain

    g_jk = CE(p0_j) - CE(q_jk)

is zero at initialisation by construction rather than by luck. The attacker can learn
interactions between `H` and `Z`; conditioning is never replaced by label strata.

`g` is an **empirical incremental attack gain on a finite differentiable family**. It
is not a conditional mutual information estimate. Subtracting the fixed `CE(p0_j)`
adds no gradient of its own -- it changes which constraints are active and how the
quantity reads, nothing more.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .inputs import H_A_WIDTH, ROLE_CLASSES, ROLE_ORDER

# Differentiable correction families, in the fixed order that also breaks max ties.
FAMILIES = ('linear', 'mlp64', 'mlp64_32')
SINGLE_FAMILY = 'mlp64'

# Architectures held OUT of every training ensemble and reserved for the audit slate.
HELD_OUT_AUDIT_ARCHITECTURES = ('mlp[64,32] with independent audit seeds', 'HistGB trees',
                               'random-Fourier kernel ridge', 'multinomial logistic on the wire')

PROBABILITY_FLOOR = 1e-12       # fixed clipping rule for every REPORTED loss


def service_view(role: str, ha, hb):
    """`H_A` for a local role, `[H_A, H_B]` for a coalition role. Never `Z`."""
    if role.startswith('A/'):
        return ha
    if role.startswith('AB/'):
        return torch.cat((ha, hb), 1)
    raise ValueError(f'unknown role view in {role}')


def service_width(role: str, hb_width: int) -> int:
    return H_A_WIDTH if role.startswith('A/') else H_A_WIDTH + hb_width


def build_correction(family: str, input_dim: int, classes: int, seed: int) -> nn.Module:
    """Zero-initialised output layer, so the correction starts at exactly zero."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if family == 'linear':
            net = nn.Linear(input_dim, classes)
        elif family == 'mlp64':
            net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, classes))
        elif family == 'mlp64_32':
            net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32),
                                nn.ReLU(), nn.Linear(32, classes))
        else:
            raise ValueError(f'unknown correction family {family}')
    last = net if isinstance(net, nn.Linear) else net[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    return net


class ServiceBaseline(nn.Module):
    """`p0_j`: an MLP[64,32] on the role's `H` view. Fitted on `p0_fit`, then frozen."""

    def __init__(self, input_dim: int, classes: int, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, classes))

    def forward(self, x):
        return self.net(x)


def fit_service_baselines(views: dict, labels: dict, fold: np.ndarray, seed: int,
                          epochs: int = 60, batch: int = 256) -> dict:
    """Fit one frozen `p0_j` per role on the internal `p0_fit` household fold."""
    out = {}
    for index, role in enumerate(ROLE_ORDER):
        classes = ROLE_CLASSES[role]
        x = views[role][fold]
        y = labels[role][fold]
        valid = y >= 0
        x, y = x[valid], y[valid]
        model = ServiceBaseline(x.shape[1], classes, 20260918 + 100 * seed + index)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        rng = np.random.default_rng(20261018 + 100 * seed + index)
        history = []
        for epoch in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(order), batch):
                ix = torch.from_numpy(order[start:start + batch])
                optimiser.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[ix]), y[ix])
                if not torch.isfinite(loss):
                    raise AssertionError(f'non-finite p0 loss for {role}')
                loss.backward()
                optimiser.step()
            if (epoch + 1) % 20 == 0:
                with torch.no_grad():
                    history.append({'epoch': epoch + 1,
                                    'fit_cross_entropy': float(F.cross_entropy(model(x), y))})
        model.eval().requires_grad_(False)
        out[role] = {'model': model, 'rows': int(len(x)), 'history': history,
                     'input_dim': int(x.shape[1]), 'classes': classes}
    return out


@torch.no_grad()
def baseline_logits(baselines: dict, views: dict) -> dict:
    """`p0_j` logits on every row. They depend on `H` only, so they never change."""
    return {role: baselines[role]['model'](views[role]).detach() for role in baselines}


class RoleEnsemble:
    """The differentiable attacker slate for one role slot."""

    def __init__(self, role: str, slot: int, families, input_dim: int, seed: int):
        self.role = role
        self.slot = slot
        self.families = tuple(families)
        self.classes = ROLE_CLASSES[role]
        self.nets = {}
        self.seeds = {}
        for index, family in enumerate(self.families):
            s = (20262018 + 1000 * seed + 100 * slot + 10 * ROLE_ORDER.index(role) + index)
            self.nets[family] = build_correction(family, input_dim, self.classes, s)
            self.seeds[family] = s
        self.refreshes = []

    def parameters(self):
        for net in self.nets.values():
            yield from net.parameters()

    def logits(self, family, features, base):
        return base + self.nets[family](features)

    def replace(self, family: str, input_dim: int, seed: int):
        self.nets[family] = build_correction(family, input_dim, self.classes, seed)
        return self.nets[family]


def masked_cross_entropy(logits, y):
    valid = y >= 0
    if not bool(valid.any()):
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits[valid], y[valid]), int(valid.sum())


def clipped_log_loss(probabilities: np.ndarray, y: np.ndarray, classes: int) -> float:
    """Reported losses use the fixed `1e-12` floor and renormalise, as the scorer does."""
    p = np.clip(np.asarray(probabilities, np.float64), PROBABILITY_FLOOR, 1.0)
    p = p / p.sum(1, keepdims=True)
    valid = y >= 0
    return float(-np.log(p[np.arange(len(y))[valid], y[valid]]).mean())


__all__ = ['FAMILIES', 'SINGLE_FAMILY', 'HELD_OUT_AUDIT_ARCHITECTURES', 'PROBABILITY_FLOOR',
           'service_view', 'service_width', 'build_correction', 'ServiceBaseline',
           'fit_service_baselines', 'baseline_logits', 'RoleEnsemble', 'masked_cross_entropy',
           'clipped_log_loss']
