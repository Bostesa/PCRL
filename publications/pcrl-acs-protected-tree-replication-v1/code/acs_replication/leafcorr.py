"""ACS transport of task_headroom/leafcorr.py (validated L_TREE procedure); unchanged except lines marked ACS-CHANGE.

ACS role mapping: task pool = MODEL_FIT (task labels); privacy pool = MODEL_FIT with SEX (the user-authorised access:
every method may use MODEL_FIT task and SEX labels). The source split UPDATE (SEX) / GUARD (no SEX) does not exist here.

Privacy training of leaf-score corrections on fixed tree routing (Stage B).

theta_r has shape (T_r * L_r, dim_r); the released raw score of recipient r is
teacher_raw_r(x) + theta_r[leaf ids of x].sum(trees) / T_r, i.e. delta = theta/T.
Wires (the only things critics and readers see): income = (p0, p1, decision),
occupation = (p0..p5, decision), pair = their aligned concatenation.

Objective per active recipient r (gradient w.r.t. theta_r only):
    task CE_r on a uniform fitting-pool minibatch (UPDATE+GUARD)
  + lambda_r * privacy objective on an UPDATE minibatch with UPDATE SEX.
Privacy objective: risk(view) = 1 - min(critic CEs, H(SEX)) / H(SEX) over a
linear/MLP32/MLP64 bank per view (inherited validated online critics).
    local  = risk(income) + risk(occupation)
    joint  = local + risk(pair)
Minimizing it pushes every critic's SEX cross-entropy up toward the entropy.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import random
import statistics

import numpy as np

RECIPE = {
    'schema': 'ACSR1-leaf-score-correction-recipe-v1 (transport of THP1-leaf-score-correction-recipe-v1)',  # ACS-CHANGE
    'parametrization': 'delta[r,t,leaf] = theta_r[t,leaf]/T_r; theta_r = 0 at start (exact teacher)',
    'optimizer': 'AdamW per recipient, lr 0.01, betas (0.9, 0.999), eps 1e-8, decoupled weight decay 0.01 (shrink toward the teacher)',
    'learning_rate': 0.01, 'weight_decay': 0.01,
    'batch_size': 256, 'steps_per_epoch': 40, 'epochs_per_active_recipient': 20,
    'checkpoint_epochs': [0, 2, 5, 10, 20],
    'task_pool': 'MODEL_FIT (ACS-CHANGE: role name; capped at 20,000 people); uniform minibatches with replacement-free epoch order',
    'privacy_pool': 'MODEL_FIT rows with MODEL_FIT SEX (ACS-CHANGE: authorised access)',
    'critic_kinds': ['linear', 'mlp32', 'mlp64'], 'critic_learning_rate': 0.003, 'critic_optimizer': 'Adam',
    'critic_views': ['income(3)', 'occupation(6)', 'pair(9)'],  # ACS-CHANGE
    'critic_input_normalization': 'inherited DiagonalReference refreshed at every epoch start on a fixed 256-row UPDATE reference sample',
    'warm_critic_rounds': 80, 'critic_rounds_per_recipient_update': 5,
    'strength_ratios': [1.0, 3.0, 10.0],
    'calibration': 'FIT-only: at the exact teacher with warm critics, 16 fixed draws; lambda_J_r(rho) = rho * median||grad CE_r|| / median||grad joint_r||; lambda_L_r(rho) = rho * median||grad CE_r|| / median||grad local_r||',
    'arms': {
        'T_REFIT': {'privacy': None, 'coefficient': None, 'schedule': 'simultaneous'},
        'L_TREE': {'privacy': 'local', 'coefficient': 'lambda_J', 'schedule': 'simultaneous'},
        'L_STRONG': {'privacy': 'local', 'coefficient': 'lambda_L', 'schedule': 'simultaneous'},
        'J_TREE': {'privacy': 'joint', 'coefficient': 'lambda_J', 'schedule': 'simultaneous'},
        'S12_TREE': {'privacy': 'joint', 'coefficient': 'lambda_J', 'schedule': 'sequential', 'order': [0, 1]},
        'S21_TREE': {'privacy': 'joint', 'coefficient': 'lambda_J', 'schedule': 'sequential', 'order': [1, 0]},
    },
    'sequential_rule': 'first recipient trains 20 epochs while the other stays at its teacher; it is then frozen bitwise and the second trains 20 epochs with the joint objective restricted to it; critics continue',
    'no_sweeps': 'no KL trust region, no decision-preservation constraint, no temperature/calibration, no critic-refresh or gradient controller',
    'probe': 'fixed 512-row MODEL_FIT probe (label-free SHA256 order) for privacy-movement counterfactuals',  # ACS-CHANGE: role name
}
VIEWS = ('income', 'occupation', 'pair')
WIDTHS = (3, 6, 9)  # ACS-CHANGE: income(2 probs + decision), occupation(5 probs + decision), pair(9)


def seed_for(*items):
    return int(hashlib.sha256('|'.join(map(str, items)).encode()).hexdigest()[:15], 16)


def tensor_sha(t):
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


class Recipient:
    """Fixed routing + teacher raw scores for one role; theta lives in the trainer."""

    def __init__(self, task, teacher, leaves, raw):
        import torch
        self.task = task; self.T = teacher.T; self.L = teacher.L; self.dim = teacher.dim
        self.idx = torch.from_numpy((np.arange(self.T)[None, :] * self.L + leaves).astype(np.int64))
        self.raw = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float64))

    def logits(self, theta, rows):
        import torch
        raw = self.raw[rows] + theta[self.idx[rows]].sum(1) / self.T
        if self.task == 'income':
            d = raw[:, 0]
            return torch.stack((-d / 2, d / 2), 1)
        return raw - raw.mean(1, keepdim=True)

    def wire(self, theta, rows):
        import torch
        p = torch.softmax(self.logits(theta, rows), 1)
        return torch.cat((p, p.argmax(1, keepdim=True).to(p.dtype).detach()), 1)


class Critics:
    """Nine wire critics (income3, occupation6, pair9 x linear/mlp32/mlp64) on MODEL_FIT SEX (ACS-CHANGE: widths)."""

    def __init__(self, seed, sex):
        import torch
        from task_conditioned_representation.kernels import DiagonalReference  # noqa: F401 (validated normalizer)
        prior = torch.bincount(sex, minlength=2).double() / len(sex)
        self.entropy = float(-(prior * prior.log()).sum())
        self.banks, self.optimizers, self.references, self.rounds = [], [], None, 0
        for view, width in zip(VIEWS, WIDTHS):
            bank, opts = [], []
            for kind in RECIPE['critic_kinds']:
                torch.manual_seed(seed_for('THP1critic', seed, view, kind) % 2 ** 31)
                if kind == 'linear':
                    c = torch.nn.Linear(width, 2, dtype=torch.float64)
                elif kind == 'mlp32':
                    c = torch.nn.Sequential(torch.nn.Linear(width, 32, dtype=torch.float64), torch.nn.ReLU(), torch.nn.Linear(32, 2, dtype=torch.float64))
                else:
                    c = torch.nn.Sequential(torch.nn.Linear(width, 64, dtype=torch.float64), torch.nn.ReLU(),
                                            torch.nn.Linear(64, 64, dtype=torch.float64), torch.nn.ReLU(), torch.nn.Linear(64, 2, dtype=torch.float64))
                bank.append(c); opts.append(torch.optim.Adam(c.parameters(), lr=RECIPE['critic_learning_rate']))
            self.banks.append(bank); self.optimizers.append(opts)

    @staticmethod
    def raw_views(w1, w2):
        import torch
        return [w1, w2, torch.cat((w1, w2), 1)]

    def refresh(self, w1, w2):
        import torch
        from task_conditioned_representation.kernels import DiagonalReference
        with torch.no_grad():
            prior = self.references or [None] * 3
            self.references = [DiagonalReference.fit(v, old) for v, old in zip(self.raw_views(w1, w2), prior)]

    def views(self, w1, w2):
        return [r.normalize(v) for r, v in zip(self.references, self.raw_views(w1, w2))]

    def update(self, w1, w2, sex):
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            views = self.views(w1.detach(), w2.detach())
        for bank, opts, view in zip(self.banks, self.optimizers, views):
            for c, o in zip(bank, opts):
                for p in c.parameters():
                    p.requires_grad_(True)
                o.zero_grad(set_to_none=True)
                loss = F.cross_entropy(c(view), sex)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite critic loss')
                loss.backward(); o.step()
        self.rounds += 1

    def terms(self, w1, w2, sex):
        import torch
        import torch.nn.functional as F
        out = {}
        for name, bank, view in zip(VIEWS, self.banks, self.views(w1, w2)):
            for c in bank:
                for p in c.parameters():
                    p.requires_grad_(False)
            losses = torch.stack([F.cross_entropy(c(view), sex) for c in bank] + [view.sum() * 0 + self.entropy])
            out[name] = 1 - losses.min() / self.entropy
            out[name + '_ces'] = [float(x) for x in losses.detach()]
        return out

    def state_sha(self):
        return hashlib.sha256(json.dumps([[{k: tensor_sha(v) for k, v in c.state_dict().items()} for c in bank]
                                          for bank in self.banks], sort_keys=True).encode()).hexdigest()


def objective(terms, kind):
    if kind == 'joint':
        return terms['income'] + terms['occupation'] + terms['pair']
    if kind == 'local':
        return terms['income'] + terms['occupation']
    raise ValueError('unregistered privacy objective')


class Trainer:
    def __init__(self, seed, recipients, labels, update_rows, sex_update, probe_rows, reference_rows):
        import torch
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        self.seed = seed; self.rec = recipients; self.labels = [torch.from_numpy(labels[0]), torch.from_numpy(labels[1])]
        self.n = len(labels[0]); self.update_rows = np.asarray(update_rows); self.sex = torch.from_numpy(sex_update)
        self.probe = torch.from_numpy(np.asarray(probe_rows)); self.reference = torch.from_numpy(np.asarray(reference_rows))
        self.theta = [torch.zeros(r.T * r.L, r.dim, dtype=torch.float64, requires_grad=True) for r in recipients]

    def wires(self, rows):
        return [r.wire(t, rows) for r, t in zip(self.rec, self.theta)]

    def ce(self, i, rows):
        import torch.nn.functional as F
        return F.cross_entropy(self.rec[i].logits(self.theta[i], rows), self.labels[i][rows])

    def privacy_batch(self, rng):
        import torch
        pos = rng.sample(range(len(self.update_rows)), RECIPE['batch_size'])
        return torch.from_numpy(self.update_rows[pos]), self.sex[pos]

    def critic_round(self, critics, rng):
        import torch
        rows, s = self.privacy_batch(rng)
        with torch.no_grad():
            w = self.wires(rows)
        critics.update(w[0], w[1], s)

    def refresh(self, critics):
        import torch
        with torch.no_grad():
            w = self.wires(self.reference)
        critics.refresh(w[0], w[1])

    def warm(self):
        critics = Critics(self.seed, self.sex)
        self.refresh(critics)
        rng = random.Random(seed_for('THP1warm', self.seed))
        for _ in range(RECIPE['warm_critic_rounds']):
            self.critic_round(critics, rng)
        return critics

    def grad_norms(self, critics, task_rows, priv_rows, priv_sex):
        import torch
        out = {'task': [], 'joint': [], 'local': []}
        w = self.wires(priv_rows); terms = critics.terms(w[0], w[1], priv_sex)
        for i in range(2):
            g = torch.autograd.grad(self.ce(i, task_rows), self.theta[i])[0]; out['task'].append(float(g.norm()))
        for kind in ('joint', 'local'):
            gs = torch.autograd.grad(objective(terms, kind), self.theta, retain_graph=True, allow_unused=True)
            out[kind] = [0.0 if g is None else float(g.norm()) for g in gs]
        return out

    def calibrate(self, critics):
        rng = random.Random(seed_for('THP1calibration', self.seed))
        task_rng = np.random.default_rng(seed_for('THP1calibration-task', self.seed) % 2 ** 32)
        draws = []
        for _ in range(16):
            import torch
            task_rows = torch.from_numpy(task_rng.choice(self.n, RECIPE['batch_size'], replace=False))
            pr, ps = self.privacy_batch(rng)
            draws.append(self.grad_norms(critics, task_rows, pr, ps))
        med = {k: [statistics.median(d[k][i] for d in draws) for i in range(2)] for k in ('task', 'joint', 'local')}
        coeff = {}
        for rho in RECIPE['strength_ratios']:
            coeff[str(rho)] = {'lambda_J': [rho * med['task'][i] / med['joint'][i] for i in range(2)],
                               'lambda_L': [rho * med['task'][i] / med['local'][i] for i in range(2)]}
        for v in coeff.values():
            for lam in v.values():
                if not all(math.isfinite(x) and 0 < x < 1e6 for x in lam):
                    raise FloatingPointError('calibration coefficient outside registered range')
        return {'draws': draws, 'medians': med, 'coefficients': coeff}

    def snapshot(self):
        return [t.detach().clone().numpy() for t in self.theta]


def run_arm(trainer, arm, rho, coefficients, on_checkpoint, log_every=20, counterfactual_steps=0):
    """Train one arm; returns logs. on_checkpoint(label, thetas, info) persists checkpoints."""
    import torch
    rule = RECIPE['arms'][arm]
    critics = trainer.warm(); warm_sha = critics.state_sha()
    lam = [0.0, 0.0] if rule['privacy'] is None else coefficients[str(rho)][rule['coefficient']]
    phases = [[0, 1]] if rule['schedule'] == 'simultaneous' else [[rule['order'][0]], [rule['order'][1]]]
    opts = [torch.optim.AdamW([t], lr=RECIPE['learning_rate'], betas=(0.9, 0.999), eps=1e-8, weight_decay=RECIPE['weight_decay'])
            for t in trainer.theta]
    task_rng = np.random.default_rng(seed_for('THP1task', trainer.seed) % 2 ** 32)
    priv_rng = random.Random(seed_for('THP1privacy', trainer.seed)); critic_rng = random.Random(seed_for('THP1critic-rounds', trainer.seed))
    logs, counterfactuals, step, updates = [], [], 0, [0, 0]
    for pi, active in enumerate(phases):
        prefix = '' if len(phases) == 1 else f'P{pi + 1}'
        frozen = {i: trainer.theta[i].detach().clone() for i in range(2) if i not in active}
        for epoch in range(RECIPE['epochs_per_active_recipient'] + 1):
            trainer.refresh(critics)
            if epoch in RECIPE['checkpoint_epochs']:
                on_checkpoint(f'{prefix}E{epoch}', trainer.snapshot(), {'updates_per_recipient': updates[:], 'critic_rounds': critics.rounds,
                                                                       'phase_active': active, 'step': step})
            if epoch == RECIPE['epochs_per_active_recipient']:
                break
            order = task_rng.permutation(trainer.n)
            for b in range(RECIPE['steps_per_epoch']):
                task_rows = torch.from_numpy(order[b * RECIPE['batch_size']:(b + 1) * RECIPE['batch_size']])
                pr, ps = trainer.privacy_batch(priv_rng)
                grads, info = {}, {}
                if rule['privacy'] is not None:
                    w = trainer.wires(pr); terms = critics.terms(w[0], w[1], ps); P = objective(terms, rule['privacy'])
                    pg = torch.autograd.grad(P, [trainer.theta[i] for i in active], allow_unused=True)
                    pg = [torch.zeros_like(trainer.theta[i]) if g is None else g for g, i in zip(pg, active)]
                    info['privacy_objective'] = float(P.detach()); info['risk'] = {v: float(terms[v].detach()) for v in VIEWS}
                else:
                    pg = [torch.zeros_like(trainer.theta[i]) for i in active]
                for g_p, i in zip(pg, active):
                    g_t = torch.autograd.grad(trainer.ce(i, task_rows), trainer.theta[i])[0]
                    grads[i] = (g_t, g_p)
                if step % log_every == 0:
                    logs.append({'step': step, 'phase': pi, 'active': active, **info,
                                 'task_grad_norm': {i: float(grads[i][0].norm()) for i in active},
                                 'privacy_grad_norm': {i: float(grads[i][1].norm()) for i in active},
                                 'weighted_privacy_grad_norm': {i: float(lam[i] * grads[i][1].norm()) for i in active},
                                 'effective_ratio': {i: (float(lam[i] * grads[i][1].norm()) / max(float(grads[i][0].norm()), 1e-300)) for i in active}})
                record_cf = counterfactual_steps and rule['privacy'] is not None and step < counterfactual_steps
                if record_cf:
                    before = {i: (trainer.theta[i].detach().clone(), copy.deepcopy(opts[i].state_dict())) for i in active}
                for i in active:
                    g_t, g_p = grads[i]
                    opts[i].zero_grad(set_to_none=True)
                    trainer.theta[i].grad = g_t + lam[i] * g_p
                    opts[i].step(); updates[i] += 1
                if record_cf:
                    counterfactuals.append(_counterfactual(trainer, active, before, grads, lam, step))
                for i, t in frozen.items():
                    if not torch.equal(trainer.theta[i].detach(), t):
                        raise AssertionError('inactive recipient changed')
                for _ in range(RECIPE['critic_rounds_per_recipient_update'] * len(active)):
                    trainer.critic_round(critics, critic_rng)
                step += 1
    return {'logs': logs, 'counterfactuals': counterfactuals, 'warm_critic_sha256': warm_sha, 'critic_rounds': critics.rounds,
            'final_critic_sha256': critics.state_sha(), 'lambda': lam, 'steps': step, 'updates_per_recipient': updates}


def _counterfactual(trainer, active, before, grads, lam, step):
    """Re-solve the same AdamW step without the privacy gradient; compare transmitted probe wires."""
    import torch
    out = {'step': step, 'recipients': {}}
    for i in active:
        theta0, state = before[i]
        t = theta0.clone().requires_grad_(True)
        o = torch.optim.AdamW([t], lr=RECIPE['learning_rate'], betas=(0.9, 0.999), eps=1e-8, weight_decay=RECIPE['weight_decay'])
        if state['state']:
            o.load_state_dict(state)
        t.grad = grads[i][0].clone(); o.step()
        with torch.no_grad():
            actual = trainer.rec[i].wire(trainer.theta[i], trainer.probe)
            task_only = trainer.rec[i].wire(t, trainer.probe)
        diff = (actual - task_only).abs()
        out['recipients'][str(i)] = {'probe_probability_max_abs_change': float(diff[:, :-1].max()),
                                     'probe_decision_changes': int((diff[:, -1] > 0).sum()),
                                     'theta_max_abs_change': float((trainer.theta[i].detach() - t.detach()).abs().max()),
                                     'privacy_grad_norm': float(grads[i][1].norm()), 'lambda': float(lam[i])}
    return out
