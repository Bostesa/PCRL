"""Fresh, equally exposed neural RAW-J protection control and its task-only continuation.

Start: the admitted N_EQ task model (per-task Linear 83-64, ReLU, 64-64, ReLU, 64-16, 16-K networks, fit
on UPDATE+GUARD with the validated Recipe-A protocol). Released wire per recipient: its own probability
vector (softmax of centred float64 logits) and argmax decision -- the same contract as L_TREE.

Privacy objective, critics, critic schedule, reference refresh, row batches, calibration rule and strength
grid are IDENTICAL to the tree arms (task_headroom.leafcorr: same Critics class and seeds, joint objective
risk(income)+risk(occupation)+risk(pair), 80 warm rounds, 5 critic rounds per recipient update, 16-draw
FIT-only calibration lambda_J_r(rho) = rho * median||grad CE_r|| / median||grad joint_r||, rho in {1,3,10}).
What differs (recorded, not hidden): the trainable function (all N_EQ network parameters instead of
fixed-routing leaf scores) and the optimizer, which is the validated TCR1 neural continuation recipe
(Adam lr 1e-4, no weight decay, gradient-norm clip 5 per recipient network) instead of AdamW lr 0.01 on
theta/T. T_NEURAL is the identical loop with lambda = 0.
"""
from __future__ import annotations
import copy
import random
import statistics

import numpy as np

from task_headroom import leafcorr as lc

RECIPE = {
    'schema': 'TC1-neural-rawj-recipe-v1',
    'start': 'admitted N_EQ (equally exposed neural task reference, UPDATE+GUARD, Recipe A)',
    'optimizer': 'Adam per recipient network, lr 1e-4, betas (0.9, 0.999), eps 1e-8, no weight decay; clip_grad_norm 5 per recipient network (validated TCR1 continuation recipe)',
    'learning_rate': 1e-4, 'clip': 5.0,
    'batch_size': 256, 'steps_per_epoch': 40, 'epochs': 20, 'checkpoint_epochs': [0, 2, 5, 10, 20],
    'nomination_checkpoints': [2, 5, 10, 20],
    'privacy': 'joint (RAW-J): risk(income) + risk(occupation) + risk(pair); identical online critics/schedule/references to the tree arms',
    'strength_ratios': [1.0, 3.0, 10.0],
    'strength_rationale': 'identical nominal grid to the tree arms (not weaker); the source TCR1 RAW-J grid was {0.3, 1, 3} under its own objective, and 1 and 3 are shared',
    'calibration': lc.RECIPE['calibration'],
    'task_only_continuation': 'T_NEURAL: the identical loop, batches and checkpoints with lambda = 0',
    'rng': 'same seed_for names as the tree arms, so task, privacy, critic and warm batches are identical row draws',
}


class NeuralRecipient:
    def __init__(self, task, model, X):
        import torch
        self.task = task; self.model = model
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        for p in self.model.parameters():
            p.requires_grad_(True)

    def params(self):
        return list(self.model.parameters())

    def logits(self, rows):
        z = self.model(self.X[rows]).double()
        return z - z.mean(1, keepdim=True)

    def wire(self, rows):
        import torch
        p = torch.softmax(self.logits(rows), 1)
        return torch.cat((p, p.argmax(1, keepdim=True).to(p.dtype).detach()), 1)


class NeuralTrainer:
    def __init__(self, seed, recipients, labels, update_rows, sex_update, probe_rows, reference_rows):
        import torch
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        self.seed = seed; self.rec = recipients
        self.labels = [torch.from_numpy(labels[0]), torch.from_numpy(labels[1])]
        self.n = len(labels[0]); self.update_rows = np.asarray(update_rows); self.sex = torch.from_numpy(sex_update)
        self.probe = torch.from_numpy(np.asarray(probe_rows)); self.reference = torch.from_numpy(np.asarray(reference_rows))

    def wires(self, rows):
        return [r.wire(rows) for r in self.rec]

    def ce(self, i, rows):
        import torch.nn.functional as F
        return F.cross_entropy(self.rec[i].logits(rows), self.labels[i][rows])

    def privacy_batch(self, rng):
        import torch
        pos = rng.sample(range(len(self.update_rows)), lc.RECIPE['batch_size'])
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
        critics = lc.Critics(self.seed, self.sex)
        self.refresh(critics)
        rng = random.Random(lc.seed_for('THP1warm', self.seed))
        for _ in range(lc.RECIPE['warm_critic_rounds']):
            self.critic_round(critics, rng)
        return critics

    def calibrate(self, critics):
        import torch
        rng = random.Random(lc.seed_for('THP1calibration', self.seed))
        task_rng = np.random.default_rng(lc.seed_for('THP1calibration-task', self.seed) % 2 ** 32)
        draws = []
        for _ in range(16):
            task_rows = torch.from_numpy(task_rng.choice(self.n, lc.RECIPE['batch_size'], replace=False))
            pr, ps = self.privacy_batch(rng)
            w = self.wires(pr); terms = critics.terms(w[0], w[1], ps)
            d = {'task': [], 'joint': [], 'local': []}
            for i in range(2):
                g = torch.autograd.grad(self.ce(i, task_rows), self.rec[i].params())
                d['task'].append(float(torch.sqrt(sum((x.double() ** 2).sum() for x in g))))
            for kind in ('joint', 'local'):
                P = lc.objective(terms, kind)
                for i in range(2):
                    g = torch.autograd.grad(P, self.rec[i].params(), retain_graph=True, allow_unused=True)
                    d[kind].append(float(torch.sqrt(sum((x.double() ** 2).sum() for x in g if x is not None))) if any(x is not None for x in g) else 0.0)
            draws.append(d)
        med = {k: [statistics.median(d[k][i] for d in draws) for i in range(2)] for k in ('task', 'joint', 'local')}
        coeff = {str(rho): {'lambda_J': [rho * med['task'][i] / med['joint'][i] for i in range(2)]} for rho in RECIPE['strength_ratios']}
        for v in coeff.values():
            if not all(np.isfinite(x) and 0 < x < 1e6 for x in v['lambda_J']):
                raise FloatingPointError('calibration coefficient outside registered range')
        return {'draws': draws, 'medians': med, 'coefficients': coeff}

    def snapshot(self):
        return {t: copy.deepcopy(r.model.state_dict()) for t, r in zip(('income', 'occupation'), self.rec)}


def run_arm(trainer, arm, rho, coefficients, on_checkpoint, log_every=20, counterfactual_steps=0):
    import torch
    critics = trainer.warm(); warm_sha = critics.state_sha()
    privacy = arm == 'RAW_J'
    lam = coefficients[str(rho)]['lambda_J'] if privacy else [0.0, 0.0]
    opts = [torch.optim.Adam(r.params(), lr=RECIPE['learning_rate'], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0) for r in trainer.rec]
    task_rng = np.random.default_rng(lc.seed_for('THP1task', trainer.seed) % 2 ** 32)
    priv_rng = random.Random(lc.seed_for('THP1privacy', trainer.seed)); critic_rng = random.Random(lc.seed_for('THP1critic-rounds', trainer.seed))
    logs, counterfactuals, step = [], [], 0
    for epoch in range(RECIPE['epochs'] + 1):
        trainer.refresh(critics)
        if epoch in RECIPE['checkpoint_epochs']:
            on_checkpoint(f'E{epoch}', trainer.snapshot(), {'updates_per_recipient': [step, step], 'critic_rounds': critics.rounds})
        if epoch == RECIPE['epochs']:
            break
        order = task_rng.permutation(trainer.n)
        for b in range(RECIPE['steps_per_epoch']):
            task_rows = torch.from_numpy(order[b * RECIPE['batch_size']:(b + 1) * RECIPE['batch_size']])
            pr, ps = trainer.privacy_batch(priv_rng)
            info = {}
            if privacy:
                w = trainer.wires(pr); terms = critics.terms(w[0], w[1], ps); P = lc.objective(terms, 'joint')
                info['privacy_objective'] = float(P.detach()); info['risk'] = {v: float(terms[v].detach()) for v in lc.VIEWS}
                pgs = []
                for i in range(2):
                    g = torch.autograd.grad(P, trainer.rec[i].params(), retain_graph=True, allow_unused=True)
                    pgs.append([torch.zeros_like(p) if x is None else x for x, p in zip(g, trainer.rec[i].params())])
            else:
                pgs = [[torch.zeros_like(p) for p in r.params()] for r in trainer.rec]
            grads = []
            for i in range(2):
                gt = torch.autograd.grad(trainer.ce(i, task_rows), trainer.rec[i].params())
                grads.append((gt, pgs[i]))
            if step % log_every == 0:
                nrm = lambda gs: float(torch.sqrt(sum((x.double() ** 2).sum() for x in gs)))
                logs.append({'step': step, **info, 'task_grad_norm': [nrm(grads[i][0]) for i in range(2)],
                             'privacy_grad_norm': [nrm(grads[i][1]) for i in range(2)],
                             'weighted_privacy_grad_norm': [lam[i] * nrm(grads[i][1]) for i in range(2)]})
            record_cf = counterfactual_steps and privacy and step < counterfactual_steps
            if record_cf:
                before = [(copy.deepcopy(r.model.state_dict()), copy.deepcopy(opts[i].state_dict())) for i, r in enumerate(trainer.rec)]
            for i, r in enumerate(trainer.rec):
                gt, gp = grads[i]
                opts[i].zero_grad(set_to_none=True)
                for p, a, c in zip(r.params(), gt, gp):
                    p.grad = a + lam[i] * c
                torch.nn.utils.clip_grad_norm_(r.params(), RECIPE['clip'])
                opts[i].step()
            if record_cf:
                counterfactuals.append(_counterfactual(trainer, before, grads, step))
            for _ in range(lc.RECIPE['critic_rounds_per_recipient_update'] * 2):
                trainer.critic_round(critics, critic_rng)
            step += 1
    return {'logs': logs, 'counterfactuals': counterfactuals, 'warm_critic_sha256': warm_sha, 'critic_rounds': critics.rounds,
            'final_critic_sha256': critics.state_sha(), 'lambda': lam, 'steps': step, 'updates_per_recipient': [step, step]}


def _counterfactual(trainer, before, grads, step):
    """Same Adam step from the saved state without the privacy gradient; compare transmitted probe outputs."""
    import torch
    out = {'step': step, 'recipients': {}}
    for i, r in enumerate(trainer.rec):
        sd, osd = before[i]
        m = copy.deepcopy(r.model); m.load_state_dict(sd)
        o = torch.optim.Adam(m.parameters(), lr=RECIPE['learning_rate'], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        o.load_state_dict(osd)
        for p, a in zip(m.parameters(), grads[i][0]):
            p.grad = a.clone()
        torch.nn.utils.clip_grad_norm_(list(m.parameters()), RECIPE['clip']); o.step()
        with torch.no_grad():
            probe_x = r.X[trainer.probe]
            za = r.model(probe_x).double(); zb = m(probe_x).double()
            pa = torch.softmax(za - za.mean(1, keepdim=True), 1); pb = torch.softmax(zb - zb.mean(1, keepdim=True), 1)
        out['recipients'][str(i)] = {'probe_probability_max_abs_change': float((pa - pb).abs().max()),
                                     'probe_decision_changes': int((pa.argmax(1) != pb.argmax(1)).sum())}
    return out
