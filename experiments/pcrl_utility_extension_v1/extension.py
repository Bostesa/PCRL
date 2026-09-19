"""Tier 2/3: a small extension channel R appended to the frozen J release.

Release  wire/A = [H_A, Z_J, R],  wire/B = H_B,  wire/AB = [H_A, Z_J, R, H_B].
H and Z_J are preserved byte-for-byte (asserted against the untouched J release on
every pool). R is computed from A-side inference inputs only: the frozen PCA_32
coordinates standardised with J's own frozen standardiser. No protected label, no row
identifier, no H_B at inference.

Inclusion fact (METHOD section 1): an unrestricted recipient of [H_A, Z_J, R] can ignore R,
so appending R can never lower optimal utility nor optimal sensitive recovery. The goal is
additional useful capability at a small measured disclosure increase, not removing
information already released in Z_J.

Utility proxy without residence labels:
  D0 : (H_A, Z_J) -> Z_A0          small MLP, frozen, household cross-fitted residuals
  T  = Z_A0 - D0(H_A, Z_J)         out-of-fold on the training rows
  D1 = D0 + E(H_A, Z_J, R)         E zero-initialised, so D1 starts at D0 exactly
  recon = mean ||T - E||^2 / var_T    var_T fixed from the training residuals
Protection (trainable roles, J-containing baseline):
  p0J_j(H view, Z_J) fitted on the p0_fit fold, frozen; attackers are zero-initialised
  corrections on [H view, Z_J, R]; gain = CE(p0J) - CE(p0J + correction).
Mapper objective: recon + beta * policy_penalty(G), policies L1/L2/C1 exactly as the
predecessor (train.policy_penalty, SLOT_SCHEDULE). Checkpoint selection: every checkpoint
(step 0 = zero extension included) scored with its OWN fresh attacker slate and its OWN
fresh decoder, both at fixed equal budgets, on the internal monitor fold.
"""
from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.pcrl_direct_adversarial_v1 import inputs as dax
from experiments.pcrl_direct_adversarial_v1.attackers import (FAMILIES, RoleEnsemble,
                                                              ServiceBaseline, masked_cross_entropy,
                                                              service_view, service_width)
from experiments.pcrl_direct_adversarial_v1.train import (REFRESH_FAMILY, SLOT_SCHEDULE,
                                                          policy_penalty, incumbent_input_dim,
                                                          _single_family_view)

ROLE_ORDER = dax.ROLE_ORDER
ROLE_CLASSES = dax.ROLE_CLASSES
Z_WIDTH = dax.A0_WIDTH
HA = dax.H_A_WIDTH
D0_FOLDS = 5
D0_SALT = 'pcrl_utility_extension_v1/d0_crossfit/v1'
DEGENERATE_RATIO = 1e-4          # var_T / var(Z_A0) below this -> degenerate residual target


@dataclass(frozen=True)
class Config:
    mapper_updates: int = 600
    checkpoint_every: int = 100
    attacker_updates_per_mapper: int = 5
    attacker_warmup: int = 100
    refresh_fractions: tuple = (0.25, 0.50, 0.75)
    refresh_catchup: int = 40
    batch: int = 256
    lr: float = 1e-3
    selection_probe_updates: int = 300
    selection_decoder_updates: int = 300
    d0_epochs: int = 60
    p0_epochs: int = 60
    validation_decoder_epochs: int = 40
    hidden: int = 64

    def as_dict(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


# ------------------------------------------------------------------ small networks
def mlp(n_in, n_out, hidden, seed, zero_last=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
    if zero_last:
        with torch.no_grad():
            net[-1].weight.zero_()
            net[-1].bias.zero_()
    return net


class Standardiser:
    def __init__(self, x: np.ndarray):
        self.mean = x.mean(0)
        self.scale = x.std(0)
        self.scale[self.scale < 1e-8] = 1.0

    def __call__(self, x):
        return torch.tensor((np.asarray(x, np.float64) - self.mean) / self.scale, dtype=torch.float32)


def household_fold(serials, salt=D0_SALT, k=D0_FOLDS) -> np.ndarray:
    return np.array([int(hashlib.sha256(f'{salt}|{s}'.encode()).hexdigest()[:8], 16) % k
                     for s in np.asarray(serials).astype(str)])


def fit_regressor(x: torch.Tensor, y: torch.Tensor, seed: int, epochs: int, hidden: int, batch=256,
                  lr=1e-3, zero_last=False) -> nn.Module:
    net = mlp(x.shape[1], y.shape[1], hidden, seed, zero_last=zero_last)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for s in range(0, len(order), batch):
            ix = torch.from_numpy(order[s:s + batch])
            opt.zero_grad(set_to_none=True)
            loss = ((net(x[ix]) - y[ix]) ** 2).mean()
            loss.backward()
            opt.step()
    return net.eval().requires_grad_(False)


# ------------------------------------------------------------------ seed context
def seed_context(seed: int, config: Config) -> dict:
    """Frozen state, J and A0 channels, labels, folds, D0 (cross-fitted), residual targets,
    J-containing service baselines. Everything here is outcome-free and shared by all units."""
    tick = time.perf_counter()
    registry = dax.Registry.new()
    state, j, portability = portable_state(seed, registry)
    jm = j['model']
    if not (np.array_equal(jm['input_mean'], state['a0']['input_mean'])
            and np.array_equal(jm['input_scale'], state['a0']['input_scale'])):
        raise AssertionError('J standardiser differs from A0 standardiser')
    labels = dax.representation_labels(seed, registry)
    folds = dax.household_folds(labels['serials'])
    rf = 'representation_fit'
    x_all = {p: dax.standardize(state['pca'][p], jm['input_mean'], jm['input_scale']) for p in dax.POOLS}
    ha = {p: np.asarray(state['anchors'][f'{p}/A'], np.float64) for p in dax.POOLS}
    hb = {p: np.asarray(state['anchors'][f'{p}/B'], np.float64) for p in dax.POOLS}
    zj = {p: j['channel'][p] for p in dax.POOLS}
    za0 = {p: state['channel'][p] for p in dax.POOLS}

    # D0 on (H_A, Z_J) -> Z_A0. Training rows = p0_fit U mapper_fit (monitor held out).
    train_rows = np.sort(np.concatenate([folds['p0_fit'], folds['mapper_fit']]))
    din = {p: np.column_stack([ha[p], zj[p]]) for p in dax.POOLS}
    std_d = Standardiser(din[rf][train_rows])
    ystd = Standardiser(za0[rf][train_rows])
    xd = std_d(din[rf])
    yd = ystd(za0[rf])
    hf = household_fold(labels['serials'])
    oof = np.full((len(xd), Z_WIDTH), np.nan)
    for f in range(D0_FOLDS):
        tr = train_rows[hf[train_rows] != f]
        te = train_rows[hf[train_rows] == f]
        net = fit_regressor(xd[tr], yd[tr], 31000 + 100 * seed + f, config.d0_epochs, config.hidden)
        with torch.no_grad():
            oof[te] = net(xd[te]).numpy() * ystd.scale + ystd.mean
    d0 = fit_regressor(xd[train_rows], yd[train_rows], 31000 + 100 * seed + 99, config.d0_epochs, config.hidden)

    def d0_apply(pool):
        with torch.no_grad():
            return d0(std_d(din[pool])).numpy().astype(np.float64) * ystd.scale + ystd.mean

    d0_out = {p: d0_apply(p) for p in dax.POOLS}
    t_train = za0[rf] - oof                        # cross-fitted residual on training rows
    t_final = {p: za0[p] - d0_out[p] for p in dax.POOLS}   # frozen-decoder residual (inference/eval)
    target = t_final[rf].copy()
    target[train_rows] = t_train[train_rows]       # monitor rows keep the frozen-decoder residual
    var_t = float(t_train[folds['mapper_fit']].var(0).mean())
    var_a0 = float(za0[rf][train_rows].var(0).mean())
    degenerate = (not np.isfinite(var_t)) or var_t < DEGENERATE_RATIO * var_a0

    # J-containing service baselines p0J_j on the p0_fit fold (H view + Z_J).
    base_in = {}
    for role in ROLE_ORDER:
        view = ha[rf] if role.startswith('A/') else np.column_stack([ha[rf], hb[rf]])
        base_in[role] = np.column_stack([view, zj[rf]])
    role_y = {role: np.asarray(labels['protected'][role.split('/')[1]], np.int64) for role in ROLE_ORDER}
    baselines = {}
    for i, role in enumerate(ROLE_ORDER):
        rows = folds['p0_fit'][role_y[role][folds['p0_fit']] >= 0]
        xt = torch.tensor(base_in[role][rows], dtype=torch.float32)
        yt = torch.from_numpy(role_y[role][rows])
        model = ServiceBaseline(xt.shape[1], ROLE_CLASSES[role], 32000 + 100 * seed + i)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        rng = np.random.default_rng(33000 + 100 * seed + i)
        for _ in range(config.p0_epochs):
            order = rng.permutation(len(xt))
            for s in range(0, len(order), 256):
                ix = torch.from_numpy(order[s:s + 256])
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(xt[ix]), yt[ix]).backward()
                opt.step()
        baselines[role] = model.eval().requires_grad_(False)

    def fold(name):
        rows = folds[name]
        t_ha = torch.tensor(ha[rf][rows], dtype=torch.float32)
        t_hb = torch.tensor(hb[rf][rows], dtype=torch.float32)
        t_zj = torch.tensor(zj[rf][rows], dtype=torch.float32)
        views = {r: torch.cat((service_view(r, t_ha, t_hb), t_zj), 1) for r in ROLE_ORDER}
        with torch.no_grad():
            base = {r: baselines[r](views[r]) for r in ROLE_ORDER}
        return {'x': x_all[rf][rows], 'dec_in': std_d(din[rf][rows]),
                'T': torch.tensor(target[rows], dtype=torch.float32),
                'views': views, 'base': base,
                'y': {r: torch.from_numpy(role_y[r][rows]) for r in ROLE_ORDER}, 'rows': rows}

    order_rng = np.random.default_rng(34000 + seed)
    n = len(folds['mapper_fit'])
    order = []
    while len(order) < 8000:
        perm = order_rng.permutation(n)
        order.extend(perm[s:s + config.batch] for s in range(0, n - config.batch + 1, config.batch))
    return {
        'seed': seed, 'state': state, 'j': j, 'labels': labels, 'folds': folds,
        'x_all': x_all, 'ha': ha, 'hb': hb, 'zj': zj, 'za0': za0, 'din': din, 'std_d': std_d,
        'd0': d0, 'd0_out': d0_out, 't_final': t_final, 't_target_rf': target, 'var_t': var_t, 'var_a0': var_a0,
        'degenerate_target': degenerate, 'train_rows': train_rows, 'baselines': baselines,
        'mapper_fit': fold('mapper_fit'), 'monitor': fold('monitor'), 'order': order,
        'context_seconds': time.perf_counter() - tick, 'portability': portability,
        'd0_record': {'train_rows': int(len(train_rows)), 'crossfit_folds': D0_FOLDS,
                      'var_T_mapper_fit_oof': var_t, 'var_Z_A0_train': var_a0,
                      'degenerate': degenerate,
                      'oof_vs_final_note': ('training uses OUT-OF-FOLD residuals of 5 household-'
                                            'cross-fitted D0 fits; inference/evaluation uses the single '
                                            'final D0 fitted on all p0_fit U mapper_fit rows'),
                      'final_d0_train_mse_norm': float(((t_final[rf][train_rows]) ** 2).mean() / var_a0),
                      'oof_mse_norm': float((t_train[train_rows] ** 2).mean() / var_a0)}}


# ------------------------------------------------------------------ portable frozen state
PORTABLE_TOL = 1e-5     # frozen before any new outcome: max |recomputed - stored| for A0/J mappers


def portable_state(seed, registry):
    """Stored release arrays are AUTHORITATIVE for Z_A0 and Z_J (byte-for-byte parity by
    construction). The frozen mappers are re-executed only as a floating-point PORTABILITY
    check: exact serialized identity (checkpoint sha256) and execution parity are different
    checks, and cross-platform float32 arithmetic may differ in the last bits."""
    from experiments.pcrl_competitive_method_v1.run_fit_e import load_mapper
    rows = dax.arrays(registry.resolve(dax.fixed_path(seed, 'split_rows.npz')))
    pca = dax.arrays(registry.resolve(dax.fixed_path(seed, 'pca.npz')))
    anchors = dax.arrays(registry.resolve(dax.fixed_path(seed, 'anchors.npz')))
    out, report = {}, {}
    for arm in ('A0', 'J'):
        model = load_mapper(seed, registry, arm)
        released = dax.arrays(registry.resolve(dax.fixed_path(seed, f'training/{arm}/releases.npz')))
        channel, diffs = {}, {}
        with torch.no_grad():
            for pool in dax.POOLS:
                stored = np.asarray(released[f'wire/A/{pool}'], np.float64)
                if not np.array_equal(stored[:, :HA], anchors[f'{pool}/A']):
                    raise AssertionError(f'{arm} H_A anchor mismatch at {pool}')
                value = model['mapper'](dax.standardize(pca[pool], model['input_mean'],
                                                        model['input_scale'])).numpy().astype(np.float64)
                diffs[pool] = float(np.abs(value - stored[:, HA:]).max())
                channel[pool] = stored[:, HA:].copy()
        worst = max(diffs.values())
        report[arm] = {'checkpoint_sha256': model['checkpoint_sha256'], 'max_abs_recompute_diff': worst,
                       'bitwise_on_this_platform': worst == 0.0, 'tolerance': PORTABLE_TOL,
                       'within_tolerance': worst <= PORTABLE_TOL}
        if worst > PORTABLE_TOL:
            raise AssertionError(f'{arm} recomputation differs from its release by {worst} > {PORTABLE_TOL}')
        out[arm] = {'channel': channel, 'model': model}
    state = {'seed': seed, 'rows': rows, 'pca': pca, 'anchors': anchors, 'a0': out['A0']['model'],
             'channel': out['A0']['channel']}
    return state, out['J'], report


# ------------------------------------------------------------------ extension model
class Extension(nn.Module):
    def __init__(self, r, hidden, seed):
        super().__init__()
        self.width = r
        self.g = mlp(dax.PCA_WIDTH, r, hidden, seed, zero_last=True)          # R = g(x), zero at step 0
        self.e = mlp(HA + Z_WIDTH + r, Z_WIDTH, hidden, seed + 1, zero_last=True)  # residual head

    def encode(self, x):
        return self.g(x)

    def decode(self, dec_in, r):
        return self.e(torch.cat((dec_in, r), 1))


def recon_loss(pred, t, var_t):
    return ((pred - t) ** 2).mean() / var_t


def role_gains(r, fold, ix, ensembles):
    per_role, cand = {}, {role: [] for role in ROLE_ORDER}
    for (role, slot), ens in ensembles.items():
        y = fold['y'][role][ix]
        valid = y >= 0
        if not bool(valid.any()):
            continue
        base = fold['base'][role][ix]
        ref = F.cross_entropy(base[valid], y[valid])
        feats = torch.cat((fold['views'][role][ix], r), 1)
        for fam in ens.families:
            cand[role].append(ref - F.cross_entropy((base + ens.nets[fam](feats))[valid], y[valid]))
    for role in ROLE_ORDER:
        if cand[role]:
            s = torch.stack(cand[role])
            per_role[role] = torch.clamp(s[int(torch.argmax(s.detach()))], min=0.0)
        else:
            per_role[role] = torch.zeros(())
    return per_role


def attacker_loss(r, fold, ix, ensembles):
    total, count = 0.0, 0
    for (role, _), ens in ensembles.items():
        y = fold['y'][role][ix]
        feats = torch.cat((fold['views'][role][ix], r), 1)
        for fam in ens.families:
            loss, known = masked_cross_entropy(fold['base'][role][ix] + ens.nets[fam](feats), y)
            if known:
                total, count = total + loss, count + 1
    return total, count


def new_slate(policy, r, seed_base):
    ens = {}
    for role, slot in SLOT_SCHEDULE[policy]:
        ens[(role, slot)] = RoleEnsemble(role, slot, FAMILIES, service_width(role, dax.H_B_WIDTH) + Z_WIDTH + r,
                                         seed_base)
    return ens


def fresh_scores(model, ctx, policy, beta, config, seed, step):
    """Equal-budget selection yardstick for one frozen checkpoint: its own fresh attacker slate
    and its own fresh residual decoder, fitted on mapper_fit, read on monitor."""
    tr, mo = ctx['mapper_fit'], ctx['monitor']
    with torch.no_grad():
        r_tr, r_mo = model.encode(tr['x']), model.encode(mo['x'])
    ens = new_slate(policy, model.width, 95000 + 100 * seed + step)
    params = [p for e in ens.values() for p in e.parameters()]
    opt = torch.optim.Adam(params, lr=config.lr)
    rng = np.random.default_rng(96000 + 1000 * seed + step)
    n = len(tr['x'])
    for _ in range(config.selection_probe_updates):
        ix = torch.from_numpy(rng.choice(n, size=min(config.batch, n), replace=False))
        opt.zero_grad(set_to_none=True)
        total, count = attacker_loss(r_tr[ix], tr, ix, ens)
        if count:
            total.backward()
            opt.step()
    dec = mlp(HA + Z_WIDTH + model.width, Z_WIDTH, config.hidden, 97000 + 100 * seed + step, zero_last=True)
    dopt = torch.optim.Adam(dec.parameters(), lr=config.lr)
    for _ in range(config.selection_decoder_updates):
        ix = torch.from_numpy(rng.choice(n, size=min(config.batch, n), replace=False))
        dopt.zero_grad(set_to_none=True)
        recon_loss(dec(torch.cat((tr['dec_in'][ix], r_tr[ix]), 1)), tr['T'][ix], ctx['var_t']).backward()
        dopt.step()
    with torch.no_grad():
        idx = torch.arange(len(mo['x']))
        gains = role_gains(r_mo, mo, idx, ens)
        pen = float(policy_penalty(gains, policy)) if policy else 0.0
        rec = float(recon_loss(dec(torch.cat((mo['dec_in'], r_mo), 1)), mo['T'], ctx['var_t']))
    return {'step': step, 'recon': rec, 'gains': {k: float(v) for k, v in gains.items()},
            'penalty': pen, 'score': rec + beta * pen}


def fit_unit(ctx, r, policy, beta, config: Config) -> dict:
    """One extension fit. beta = 0 is the unprotected learned extension (C1 attackers still
    train alongside, penalty weight zero)."""
    tick = time.perf_counter()
    seed = ctx['seed']
    pol_ix = ('L1', 'L2', 'C1').index(policy)
    torch.manual_seed(35000 + 1000 * seed + 100 * r + 10 * pol_ix)
    model = Extension(r, config.hidden, 36000 + 1000 * seed + 100 * r + 10 * pol_ix)
    tr, mo = ctx['mapper_fit'], ctx['monitor']
    ens = new_slate(policy, r, seed * 10)
    att_params = [p for e in ens.values() for p in e.parameters()]
    att_opt = torch.optim.Adam(att_params, lr=config.lr)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    order, cur = ctx['order'], {'i': 0}

    def batch():
        ix = order[cur['i'] % len(order)]
        cur['i'] += 1
        return torch.from_numpy(ix)

    def attacker_step(subset=None):
        ix = batch()
        with torch.no_grad():
            rr = model.encode(tr['x'][ix])
        att_opt.zero_grad(set_to_none=True)
        total, count = attacker_loss(rr, tr, ix, subset or ens)
        if count:
            if not torch.isfinite(total):
                raise AssertionError('non-finite attacker loss')
            total.backward()
            att_opt.step()

    for _ in range(config.attacker_warmup):
        attacker_step()
    refresh = {int(round(f * config.mapper_updates)): REFRESH_FAMILY[i] for i, f in enumerate(config.refresh_fractions)}
    checkpoints, trace, refresh_log = [], [], []

    def ce_by(ensembles):
        with torch.no_grad():
            rr = model.encode(mo['x'])
            out = {}
            for (role, slot), e in ensembles.items():
                feats = torch.cat((mo['views'][role], rr), 1)
                for fam in e.families:
                    loss, known = masked_cross_entropy(mo['base'][role] + e.nets[fam](feats), mo['y'][role])
                    out[(role, slot, fam)] = float(loss) if known else float('inf')
            return out

    checkpoints.append({'step': 0, 'state': copy.deepcopy(model.state_dict())})
    for step in range(1, config.mapper_updates + 1):
        if step in refresh:
            fam = refresh[step]
            before = ce_by(ens)
            fresh = {}
            for key, e in ens.items():
                role, slot = key
                inc = e.nets[fam]
                fresh[key] = (inc, e.replace(fam, incumbent_input_dim(inc),
                                             37000 + 1000 * seed + 100 * step + 10 * ROLE_ORDER.index(role) + slot))
            copt = torch.optim.Adam([p for _, n in fresh.values() for p in n.parameters()], lr=config.lr)
            subset = {k: _single_family_view(e, fam) for k, e in ens.items()}
            for _ in range(config.refresh_catchup):
                ix = batch()
                with torch.no_grad():
                    rr = model.encode(tr['x'][ix])
                copt.zero_grad(set_to_none=True)
                total, count = attacker_loss(rr, tr, ix, subset)
                if count:
                    total.backward()
                    copt.step()
            after = ce_by(ens)
            dec = {}
            for key, (inc, _new) in fresh.items():
                role, slot = key
                keep_new = after[(role, slot, fam)] < before[(role, slot, fam)]
                if not keep_new:
                    ens[key].nets[fam] = inc
                dec[f'{role}|{slot}'] = 'refreshed' if keep_new else 'incumbent'
            att_params = [p for e in ens.values() for p in e.parameters()]
            att_opt = torch.optim.Adam(att_params, lr=config.lr)
            refresh_log.append({'step': step, 'family': fam, 'decisions': dec})
        for _ in range(config.attacker_updates_per_mapper):
            attacker_step()
        ix = batch()
        for p in att_params:
            p.requires_grad_(False)
        opt.zero_grad(set_to_none=True)
        rr = model.encode(tr['x'][ix])
        rec = recon_loss(model.decode(tr['dec_in'][ix], rr), tr['T'][ix], ctx['var_t'])
        gains = role_gains(rr, tr, ix, ens)
        pen = policy_penalty(gains, policy)
        loss = rec + beta * pen
        if not torch.isfinite(loss):
            raise AssertionError('non-finite mapper loss')
        loss.backward()
        opt.step()
        for p in att_params:
            p.requires_grad_(True)
        if step % 50 == 0 or step == 1:
            trace.append({'step': step, 'recon': float(rec), 'penalty': float(pen),
                          'gains': {k: float(v) for k, v in gains.items()}})
        if step % config.checkpoint_every == 0:
            checkpoints.append({'step': step, 'state': copy.deepcopy(model.state_dict())})
    scores = []
    final = copy.deepcopy(model.state_dict())
    for c in checkpoints:
        model.load_state_dict(c['state'])
        scores.append(fresh_scores(model, ctx, policy, beta, config, seed, c['step']))
    model.load_state_dict(final)
    chosen = int(np.argmin([s['score'] for s in scores]))          # ties -> earliest step
    model.load_state_dict(checkpoints[chosen]['state'])
    model.eval()
    return {'model': model, 'selected_step': checkpoints[chosen]['step'], 'monitor_scores': scores,
            'trace': trace, 'refreshes': refresh_log, 'checkpoints': checkpoints,
            'runtime_seconds': time.perf_counter() - tick}


# ------------------------------------------------------------------ deployed functions
def learned_extension_values(ctx, model) -> dict:
    with torch.no_grad():
        return {p: model.encode(ctx['x_all'][p]).numpy().astype(np.float64) for p in dax.POOLS}


def pca_extension(ctx, r) -> tuple:
    """Deployable PCA residual control: R = (Z_A0(x) - D0(H_A, Z_J(x)) - mu) V_r, with mu, V_r
    fitted on the mapper_fit rows. Uses the FROZEN final D0 (a function of A-side inputs), never
    the row-specific cross-fitted residual targets."""
    rows = ctx['folds']['mapper_fit']
    resid = ctx['t_final']['representation_fit'][rows]
    mu = resid.mean(0)
    _, s, vt = np.linalg.svd(resid - mu, full_matrices=False)
    v = vt[:r].T
    values = {p: (ctx['t_final'][p] - mu) @ v for p in dax.POOLS}
    return values, {'mu': mu, 'components': v, 'singular_values': s.tolist()}


def build_release(ctx, r_values) -> dict:
    wires = {}
    for p in dax.POOLS:
        ha, hb, zj = ctx['ha'][p], ctx['hb'][p], ctx['zj'][p]
        rv = np.asarray(r_values[p], np.float64)
        if not np.isfinite(rv).all():
            raise AssertionError(f'non-finite extension values at {p}')
        wa = np.column_stack((ha, zj, rv))
        wires[f'wire/A/{p}'] = wa
        wires[f'wire/B/{p}'] = hb.copy()
        wires[f'wire/AB/{p}'] = np.column_stack((wa, hb))
    return wires


def assert_parity(ctx, wires):
    """H_A, H_B and Z_J byte-for-byte equal to the untouched J release on every pool."""
    released = dax.arrays(dax.Registry.new().resolve(dax.fixed_path(ctx['seed'], 'training/J/releases.npz')))
    for p in dax.POOLS:
        ref = np.asarray(released[f'wire/A/{p}'], np.float64)
        a = wires[f'wire/A/{p}']
        if not (a[:, :HA + Z_WIDTH].tobytes() == ref.tobytes()
                and a[:, :HA + Z_WIDTH].dtype == ref.dtype):
            raise AssertionError(f'[H_A, Z_J] not byte-identical to the J release at {p}')
        if not np.array_equal(wires[f'wire/B/{p}'], ctx['state']['anchors'][f'{p}/B']):
            raise AssertionError(f'H_B parity failed at {p}')
        if not np.array_equal(wires[f'wire/AB/{p}'][:, -dax.H_B_WIDTH:], ctx['state']['anchors'][f'{p}/B']):
            raise AssertionError(f'AB H_B parity failed at {p}')
    return True


def validation_reconstruction(ctx, r_values, config: Config, seed_offset=0) -> dict:
    """Validation-split capability proxy: fit a fresh residual decoder on representation-fit
    training rows (p0_fit U mapper_fit) and read MSE on the `source_validation` pool against the
    frozen-decoder residual. The same recipe with R removed is the no-extension baseline."""
    rows = ctx['train_rows']
    rf, pv = 'representation_fit', 'source_validation'
    t_tr = torch.tensor(ctx['t_target_rf'][rows], dtype=torch.float32)   # cross-fitted residuals
    t_va = torch.tensor(ctx['t_final'][pv], dtype=torch.float32)
    d_tr, d_va = ctx['std_d'](ctx['din'][rf][rows]), ctx['std_d'](ctx['din'][pv])
    out = {}
    for label, use_r in (('with_R', True), ('no_extension', False)):
        if use_r:
            rstd = Standardiser(np.asarray(r_values[rf][rows]))
            x_tr = torch.cat((d_tr, rstd(r_values[rf][rows])), 1)
            x_va = torch.cat((d_va, rstd(r_values[pv])), 1)
        else:
            x_tr, x_va = d_tr, d_va
        net = fit_regressor(x_tr, t_tr, 38000 + 100 * ctx['seed'] + seed_offset, config.validation_decoder_epochs,
                            config.hidden, zero_last=True)
        with torch.no_grad():
            out[label] = float(((net(x_va) - t_va) ** 2).mean() / ctx['var_t'])
    out['relative_reduction'] = 1 - out['with_R'] / out['no_extension']
    return out
