"""Constant-channel diagnostic for the training protection penalty.

`pcrl_utility_extension_v1.extension.role_gains` reports, per role,

    gain = CE(p0J) - CE(p0J + correction(H view, Z_J, R))

clamped below at zero and maximised over correction families. The frozen `p0J` is a finite-capacity
MLP fitted on one household fold, so it is not Bayes-optimal on its own inputs. A correction that
reads those same inputs therefore has headroom **before `R` contributes anything**: the reported
gain mixes baseline approximation slack with incremental disclosure.

METHOD section 3's claim that "the incremental gain is zero at initialisation by construction" is
true -- `build_correction` zero-initialises the output layer -- and holds only at initialisation.

This module measures the null level: run the identical slate, identical seeds and identical number
of optimiser steps against a **constant** extension, and report what `role_gains` says. Any
positive value is slack, because a constant `R` satisfies `I(S; R | H, Z_J) = 0` exactly.

The diagnostic calls the shipped `role_gains`; it does not reimplement it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from experiments.pcrl_direct_adversarial_v1 import inputs as dax
from experiments.pcrl_direct_adversarial_v1.attackers import (FAMILIES, RoleEnsemble,
                                                              ServiceBaseline, service_view,
                                                              service_width)
from experiments.pcrl_utility_extension_v1 import extension as ext

ROLE_GAINS = ext.role_gains          # pinned by test; the diagnostic must exercise shipped code
ROLE_ORDER = dax.ROLE_ORDER
ROLE_CLASSES = dax.ROLE_CLASSES


def is_constant(values) -> bool:
    """The pilot's `fit_record['extension_constant']` rule: zero column standard deviation."""
    a = values.detach().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    return bool(np.all(a.std(0) == 0))


def _synthetic_fold(rows: int, seed: int, informative: bool, r_width: int, baseline_steps: int):
    """A small deterministic stand-in for one household fold.

    Labels depend nonlinearly on `[H_A, Z_J]`, so a frozen under-fitted `p0J` leaves real headroom
    -- which is exactly the condition under which the defect shows up in the real study.
    """
    rng = np.random.default_rng(seed)
    ha = rng.normal(size=(rows, dax.H_A_WIDTH)).astype(np.float32)
    hb = rng.normal(size=(rows, dax.H_B_WIDTH)).astype(np.float32)
    zj = rng.normal(size=(rows, dax.A0_WIDTH)).astype(np.float32)

    drive = np.tanh(ha[:, 0] * 2.0) + zj[:, :4].sum(1) * 0.5 + np.sin(zj[:, 5] * 3.0)
    labels = {}
    for role in ROLE_ORDER:
        k = ROLE_CLASSES[role]
        cut = np.quantile(drive, np.linspace(0, 1, k + 1)[1:-1])
        labels[role] = torch.from_numpy(np.digitize(drive, cut).astype(np.int64))

    t_ha, t_hb, t_zj = torch.from_numpy(ha), torch.from_numpy(hb), torch.from_numpy(zj)
    views = {role: torch.cat((service_view(role, t_ha, t_hb), t_zj), 1) for role in ROLE_ORDER}

    # Deliberately under-fitted frozen baselines: the real p0J is finite-capacity and fold-fitted.
    base = {}
    for index, role in enumerate(ROLE_ORDER):
        model = ServiceBaseline(views[role].shape[1], ROLE_CLASSES[role], 900 + 10 * seed + index)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(baseline_steps):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(views[role]), labels[role]).backward()
            opt.step()
        base[role] = model.eval().requires_grad_(False)(views[role]).detach()

    if informative:
        sex = labels['A/SEX'].to(torch.float32).unsqueeze(1)
        r = torch.cat([sex * 2.0 - 1.0] + [torch.zeros(rows, 1)] * (r_width - 1), 1)
    else:
        r = torch.zeros(rows, r_width)

    fold = {'y': labels, 'base': base, 'views': views}
    return fold, r


def _run(rows, seed, steps, r_width, informative, policy, baseline_steps):
    """Train a fresh correction slate against `r`, then report the shipped `role_gains`."""
    fold, r = _synthetic_fold(rows, seed, informative, r_width, baseline_steps)
    ix = torch.arange(rows)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5000 + seed)
        ensembles = ext.new_slate(policy, r_width, seed)
        params = [p for ens in ensembles.values() for p in ens.parameters()]
        opt = torch.optim.Adam(params, lr=1e-3)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            total, count = ext.attacker_loss(r, fold, ix, ensembles)
            if count:
                (total / count).backward()
                opt.step()
        gains = ROLE_GAINS(r, fold, ix, ensembles)
    return {role: float(v.detach()) for role, v in gains.items()}, r


def constant_channel_gain(*, seed: int = 0, steps: int = 120, r_width: int = 2, rows: int = 512,
                          policy: str = 'C1', baseline_steps: int = 10) -> dict:
    """Measure `role_gains` when the extension is exactly constant, so disclosure is exactly zero."""
    gains, r = _run(rows, seed, steps, r_width, False, policy, baseline_steps)
    return {'gains': gains,
            'extension_is_constant': is_constant(r),
            'extension_std': r.std(0).tolist(),
            'positive_roles': sum(v > 0.0 for v in gains.values()),
            'clamped_at_zero_roles': sum(v == 0.0 for v in gains.values()),
            'steps': steps, 'policy': policy, 'seed': seed,
            'interpretation': ('a constant extension satisfies I(S; R | H, Z_J) = 0 exactly, so any '
                               'positive value here is frozen-baseline approximation slack, not '
                               'incremental disclosure')}


def null_corrected_gain(*, seed: int = 0, steps: int = 120, r_width: int = 2, rows: int = 512,
                        informative: bool = True, policy: str = 'C1',
                        baseline_steps: int = 10) -> dict:
    """`role_gains` minus the matched constant-channel run at identical seeds and step count.

    This is a *diagnostic*, not a replacement penalty: the two runs share an initialisation and a
    step budget but not a loss surface, so the subtraction is a calibration of scale, not an
    unbiased estimator of `I(S; R | view)`. It is reported alongside the raw gain so that a `beta`
    can be read against the null level instead of against zero.
    """
    raw, _ = _run(rows, seed, steps, r_width, informative, policy, baseline_steps)
    null, _ = _run(rows, seed, steps, r_width, False, policy, baseline_steps)
    return {'raw': raw, 'null': null,
            'corrected': {role: raw[role] - null[role] for role in raw},
            'steps': steps, 'policy': policy, 'seed': seed, 'informative': informative}
