"""Stage 5d: a controlled OptNet-ARL adaptation.

OptNet-ARL (Sadeghi, Wang & Boddeti, arXiv 2109.05535, *Adversarial Representation
Learning With Closed-Form Solvers*) has three entities and exactly **one** optimised
player: a deep encoder trained by Adam, with target and adversary as **kernel ridge
regressors solved in closed form at every step**. There is no descent-ascent.

Lemma 1, the load-bearing result:

    J(Z) = (1/n)||Y~||_F^2  -  (1/n)|| P_M [Y~' ; 0_n] ||_F^2
    M    = [ K~ ; sqrt(n*gamma) I_n ],   K~ = D' K D

The stacked ``sqrt(n gamma) I`` block is the augmented-least-squares encoding of
Tikhonov regularisation and is what makes ``M'M`` invertible. Expanded, the explained
term is ``||P_M u||^2 + n*gamma*||P3 u||^2`` with ``P3 = (M'M + n gamma I)^{-1} M'``,
which is what the official code computes and what is implemented here. **Amos &
Kolter's OptNet is not cited anywhere in the paper**; "OptNet-ARL" is a name, not a QP
layer, and the gradient here is plain autodiff through the solve -- mathematically the
same gradient as the paper's Golub-Pereyra variable-projection derivative.

Every departure from the published single-view supervised formulation is declared in
``results/pcrl_invariant_baselines_v1/BASELINE_ADAPTATIONS.md`` section 5.2 and
repeated in :data:`DEPARTURES`. This is an **adaptation**, never described as a
verbatim reproduction.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pcrl_nonlinear_rank_v1.inputs import (Registry, array_hash, load_pools,
                                                       load_representation_labels, read_json,
                                                       sha_file, write_json)
from experiments.pcrl_nonlinear_rank_v1.maps import build_roles
from experiments.pcrl_nonlinear_rank_v1.run_fit import limit_threads, machine_state

from .objective import POLICIES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/pcrl_invariant_baselines_v1'
POOLS = ('representation_fit', 'source_validation', 'downstream_fit', 'downstream_validation',
         'attacker_fit', 'attacker_validation', 'test')

# ---- frozen before fitting (PROTOCOL.md section 3; BASELINE_ADAPTATIONS.md 5.3) ----
RANK = 16
HIDDEN = (64,)                 # encoder 128 -> 64 -> 16, ReLU
GAMMA = 1e-4                   # the paper's ridge value
LEARNING_RATE = 3e-4           # the paper's value
WEIGHT_DECAY = 2e-4            # the paper's value
BATCH = 512                    # mini-batch projector, the paper's own practice
STARTS = 2
ENCODER_SEED_BASE = 20260940
BATCH_SEED_BASE = 20260941
KERNEL_EPS = 1e-12

LOCAL_ROLES = ('A/SEX', 'A/RAC1P', 'A/public_coverage')
COALITION_ROLES = ('AB/SEX', 'AB/RAC1P')
ROLE_CLASSES = {'A/SEX': 2, 'A/RAC1P': 9, 'A/public_coverage': 2, 'AB/SEX': 2, 'AB/RAC1P': 9}
ROLE_ATTRIBUTE = {'A/SEX': 'SEX', 'A/RAC1P': 'RAC1P', 'A/public_coverage': 'public_coverage',
                  'AB/SEX': 'SEX', 'AB/RAC1P': 'RAC1P'}

DEPARTURES = [
    'Utility target is the residualised teacher R (32 columns), not a class label. '
    'Structurally free: Y enters Lemma 1 only as Y~ inside Y~\'Y~ and the lemma assumes no '
    'discreteness. rank(R~) = 32 > 16, so the Theorem 4.1 rank cap is not binding.',
    'Multiple lambda over multiple protected roles (the L1/L2/C1 policies). EXTRAPOLATION: the '
    'paper asserts multi-lambda generalisation in one sentence with no equation, no normalisation '
    'rule and no code. The form used is one independent J term per role carrying its policy '
    'weight, matching this study\'s existing aggregation. That specific form is not in the paper.',
    'Adversary players receive [H_c, Z] -- the actual service probabilities the recipient holds, '
    'plus the additional channel -- rather than Z alone. The original is single-view.',
    'Mini-batch exact Gaussian kernel at batch 512. This is the PAPER\'S OWN practice (it '
    'approximates the projector and gradient on one mini-batch, O(b^3) per step, justified by '
    'analogy with Nystrom). It is declared approximate and is NOT claimed exact for the '
    'infinite-dimensional kernel.',
    'Encoder is a fixed small MLP (128 -> 64 -> 16, ReLU) on the frozen whitened V, frozen before '
    'fitting; no architecture expansion is authorised.',
    'Rank fixed at 16 so comparisons are at equal realised rank. Theorem 4.1\'s negative-eigenvalue '
    'count is REPORTED alongside as the paper\'s upper bound, but is never used to select.',
    'Released Z is the raw encoder output and is NOT covariance-normalised, unlike the spectral '
    'arms where W\'W = I forces Cov(Z) = I. This does not bias any endpoint: recovery and '
    'reconstruction are both invariant under an invertible linear map of the channel.',
    'Player kernels use instance-normalised inputs with sigma = 1, which is what the PAPER states. '
    'The official code instead divides by the whole-batch Frobenius norm; the two differ and the '
    'paper\'s version is implemented.',
    'The official code trace-normalises each J term by the label energy (/||S~||^2, /||Y~||^2). '
    'That normalisation appears nowhere in the paper\'s equations. It is adopted here, because '
    'without it the continuous 32-column R and the one-hot protected targets are on '
    'incomparable scales, and it is declared as a code-not-paper choice.',
]


# ------------------------------------------------------------------ closed-form players
def gaussian_gram(x: torch.Tensor) -> torch.Tensor:
    """Exact Gaussian kernel on instance-normalised rows with sigma = 1 (the paper's setting)."""
    xn = x / x.norm(dim=1, keepdim=True).clamp_min(KERNEL_EPS)
    d2 = torch.cdist(xn, xn).pow(2)
    return torch.exp(-d2)


def centre(m: torch.Tensor) -> torch.Tensor:
    return m - m.mean(0, keepdim=True)


def explained_fraction(features: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
    """The closed-form kernel-ridge explained fraction of Lemma 1, expanded as in the code.

    ``M`` is the centred Gram; ``P3 = (M'M + reg I)^{-1} M'``; the explained energy is
    ``||M P3 u||^2 + reg ||P3 u||^2``, trace-normalised by ``||u||^2``. Differentiation
    is plain autodiff through the solve, which yields the same gradient as the paper's
    Golub-Pereyra variable-projection derivative.
    """
    n = features.shape[0]
    m = centre(gaussian_gram(features))
    u = centre(target)
    reg = n * gamma
    gram = m.T @ m + reg * torch.eye(n, dtype=m.dtype, device=m.device)
    p3 = torch.linalg.solve(gram, m.T @ u)
    explained = (m @ p3).pow(2).sum() + reg * p3.pow(2).sum()
    return explained / u.pow(2).sum().clamp_min(KERNEL_EPS)


# ------------------------------------------------------------------ encoder
class Encoder(torch.nn.Module):
    """Frozen architecture: 128 -> 64 -> 16, ReLU. Iteratively trained, per the method."""

    def __init__(self, width: int, rank: int = RANK, hidden=HIDDEN):
        super().__init__()
        layers, previous = [], width
        for size in hidden:
            layers += [torch.nn.Linear(previous, size), torch.nn.ReLU()]
            previous = size
        layers.append(torch.nn.Linear(previous, rank))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def theorem_4_1_rank(protected: dict, teacher: np.ndarray, weights: dict) -> dict:
    """The paper's Theorem 4.1 upper bound: #negative eigenvalues of B.

    Reported, never used to select. Its two stated assumptions (``z`` free and
    disconnected from the encoder; **linear** predictors) do not hold during training,
    which is why the paper itself frames it as an upper bound.
    """
    # B is n-by-n (n ~ 1.05e4, i.e. 884 MB) but has rank at most 56, so it is never
    # formed. Write B = M D M' with M = [S_1 ... S_a | Y] and D a diagonal sign/weight
    # matrix. The nonzero eigenvalues of M D M' are exactly those of the p-by-p
    # symmetric matrix G^{1/2} D G^{1/2} with G = M'M, which is what is decomposed here.
    blocks, diagonal = [], []
    for role, onehot in protected.items():
        weight = weights.get(role, 0.0)
        if weight:
            blocks.append(onehot - onehot.mean(0))
            diagonal.append(np.full(onehot.shape[1], weight))
    y = teacher - teacher.mean(0)
    blocks.append(y)
    diagonal.append(np.full(y.shape[1], -1.0))
    m = np.column_stack(blocks)
    d = np.concatenate(diagonal)
    gram = m.T @ m
    eigenvalues, vectors = np.linalg.eigh((gram + gram.T) / 2)
    root = vectors * np.sqrt(np.maximum(eigenvalues, 0.0)) @ vectors.T
    reduced = root @ np.diag(d) @ root
    values = np.linalg.eigvalsh((reduced + reduced.T) / 2)
    tolerance = 1e-10 * max(1.0, float(abs(values).max()))
    return {'negative_eigenvalue_count': int((values < -tolerance).sum()),
            'low_rank_factor_width': int(m.shape[1]),
            'matrix_rows': int(len(teacher)),
            'note': ('Theorem 4.1 upper bound only. Its assumptions (z free and disconnected '
                     'from the encoder, linear predictors) do not hold during training. '
                     'Reported, never used to select; rank is fixed at 16 for comparability.')}


# ------------------------------------------------------------------ fitting
def policy_weights(policy: str) -> dict:
    local_weight, coalition_weight = POLICIES[policy]
    weights = {r: local_weight / len(LOCAL_ROLES) for r in LOCAL_ROLES}
    if coalition_weight:
        weights.update({r: coalition_weight / len(COALITION_ROLES) for r in COALITION_ROLES})
    return weights


def fit_condition(state: dict, protected: dict, recipient: dict, policy: str, seed: int,
                  budget: int) -> dict:
    """Two deterministic starts; select on the TRAINING objective only."""
    v = torch.from_numpy(np.ascontiguousarray(state['V']))
    r = torch.from_numpy(np.ascontiguousarray(state['R']))
    weights = policy_weights(policy)
    tensors = {role: torch.from_numpy(np.ascontiguousarray(protected[role])) for role in weights}
    views = {role: torch.from_numpy(np.ascontiguousarray(recipient[role])) for role in weights}
    n = len(v)

    candidates = []
    for start in range(STARTS):
        torch.manual_seed(ENCODER_SEED_BASE + 100 * seed + 10 * start)
        encoder = Encoder(v.shape[1]).double()
        optimiser = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE,
                                     weight_decay=WEIGHT_DECAY)
        rng = np.random.default_rng(BATCH_SEED_BASE + 100 * seed + 10 * start)
        history = []
        for step in range(budget):
            index = torch.from_numpy(np.sort(rng.choice(n, min(BATCH, n), replace=False)))
            z = encoder(v[index])
            utility = explained_fraction(z, r[index], GAMMA)
            penalty = 0.0
            for role, weight in weights.items():
                features = torch.cat((views[role][index], z), dim=1)
                penalty = penalty + weight * explained_fraction(features, tensors[role][index],
                                                                GAMMA)
            loss = penalty - utility
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            if step % 50 == 0 or step == budget - 1:
                history.append({'step': step, 'loss': float(loss.detach()),
                                'penalty': float(penalty.detach() if torch.is_tensor(penalty)
                                                 else penalty),
                                'utility': float(utility.detach())})
        # Final training objective on a fixed held-in evaluation batch, identical across
        # starts, so selection compares like with like.
        evaluation = torch.from_numpy(
            np.sort(np.random.default_rng(BATCH_SEED_BASE).choice(n, min(BATCH, n), replace=False)))
        with torch.no_grad():
            z = encoder(v[evaluation])
            utility = explained_fraction(z, r[evaluation], GAMMA)
            penalty = sum(weight * explained_fraction(
                torch.cat((views[role][evaluation], z), dim=1), tensors[role][evaluation], GAMMA)
                for role, weight in weights.items())
            final = float(penalty - utility)
        candidates.append({'start': start, 'training_objective': final, 'encoder': encoder,
                           'history': history})

    candidates.sort(key=lambda c: (c['training_objective'], c['start']))
    best = candidates[0]
    with torch.no_grad():
        z_full = best['encoder'](v).numpy()
    return {
        'policy': policy, 'selected_start': best['start'],
        'training_objective': best['training_objective'],
        'start_objectives': [c['training_objective'] for c in candidates],
        'restart_spread': float(max(c['training_objective'] for c in candidates)
                                - min(c['training_objective'] for c in candidates)),
        'history': best['history'], 'encoder': best['encoder'],
        'released_covariance_eigenvalues': np.linalg.eigvalsh(
            np.cov(z_full, rowvar=False)).tolist(),
        'selection_rule': ('lowest training objective across two deterministic starts, on a '
                           'fixed evaluation batch; no attacker, residence, commute or '
                           'development outcome is consulted'),
    }


# ------------------------------------------------------------------ inputs and releases
def protected_and_views(state: dict, labels: dict):
    """One-hot protected targets and the recipient's ACTUAL service probabilities.

    OptNet-ARL is an unconditional method: its adversary regresses the raw one-hot, with
    no nuisance residualisation. Conditioning enters only by giving the adversary the
    service probabilities it genuinely holds, which is the declared adaptation.
    """
    ha, hb = state['anchors_A'], state['anchors_B']
    protected, views = {}, {}
    for role, classes in ROLE_CLASSES.items():
        y = labels[ROLE_ATTRIBUTE[role]]
        onehot = np.zeros((len(y), classes))
        known = y >= 0
        onehot[np.flatnonzero(known), y[known]] = 1.0
        protected[role] = onehot
        views[role] = ha if role.startswith('A/') else np.column_stack((ha, hb))
    return protected, views


def build_releases(name: str, seed: int, encoder, model, registry: Registry, out: Path) -> dict:
    """A = [hA | Z], B = hB, AB = [A | B]; `H_A` and `H_B` bitwise unchanged."""
    path = out / f'seed_{seed}' / 'releases' / name / 'releases.npz'
    if path.exists():
        return {'release_sha256': sha_file(path), 'reused': True, 'parity': []}
    _, teacher, anchors = load_pools(seed, registry)
    cache, parity = {}, []
    for pool in POOLS:
        t = np.asarray(teacher[pool], dtype=np.float64)
        ha = np.asarray(anchors[f'{pool}/A'], dtype=np.float64)
        hb = np.asarray(anchors[f'{pool}/B'], dtype=np.float64)
        with torch.no_grad():
            z = encoder(torch.from_numpy(
                np.ascontiguousarray(model.features(t, ha)))).numpy().astype(np.float64)
        wa = np.column_stack((ha, z))
        wab = np.column_stack((wa, hb))
        assert wa.dtype == np.float64 and np.array_equal(wa[:, :4], ha)
        assert np.array_equal(wab[:, -2:], hb) and np.array_equal(wab[:, :wa.shape[1]], wa)
        assert np.isfinite(wa).all() and np.isfinite(wab).all()
        cache[f'wire/A/{pool}'] = wa
        cache[f'wire/B/{pool}'] = hb
        cache[f'wire/AB/{pool}'] = wab
        parity.append({'arm': name, 'pool': pool, 'rank': int(z.shape[1]),
                       'anchor_A_exact': True, 'anchor_B_exact': True,
                       'A_width': int(wa.shape[1]), 'AB_width': int(wab.shape[1]),
                       'z_hash': array_hash(z)})
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.npz')
    os.close(fd)
    np.savez_compressed(tmp, **cache)
    os.replace(tmp, path)
    return {'release_sha256': sha_file(path), 'reused': False, 'parity': parity}


def calibrate(state: dict, protected: dict, views: dict) -> dict:
    """Training-only runtime calibration. Reads no outcome and no reserved label."""
    tick = time.perf_counter()
    probe = fit_condition(state, protected, views, 'C1', 0, budget=20)
    per_step = (time.perf_counter() - tick) / (20 * STARTS)
    return {'seconds_per_step': per_step, 'probe_objective': probe['training_objective'],
            'note': ('Training-only calibration: the objective and the wall clock, nothing else. '
                     'No attacker, residence, commute or development outcome is reachable here.')}


def run(out: Path = OUT, seeds=(0, 1, 2), budget: int = 1500, policies=tuple(POLICIES)) -> dict:
    limit_threads()
    torch.set_num_threads(1)
    out = Path(out)
    summary = {}
    for seed in seeds:
        marker = out / f'seed_{seed}' / 'optnet_complete.json'
        if marker.exists():
            summary[str(seed)] = read_json(marker)
            print('OPTNET_REUSED', seed, flush=True)
            continue
        tick = time.perf_counter()
        registry = Registry.new()
        state = build_roles(seed, registry)
        labels, _ = load_representation_labels(seed, registry)
        protected, views = protected_and_views(state, labels)

        # Reported before any encoder is built (BASELINE_ADAPTATIONS.md 5.4). This is the
        # OLS R^2 of the residualised teacher on the whitened features: the source review
        # warns that residualisation can annihilate the utility signal entirely, leaving
        # B with no negative eigenvalues and an empty encoder, silently. Because V is
        # whitened to V'V/n = I, the explained sum of squares is ||V'R||_F^2 / n, so the
        # 1/n is required for the ratio to read on the [0, 1] R^2 scale.
        centred = state['R'] - state['R'].mean(0)
        rows = len(state['V'])
        signal = float(np.sum((state['V'].T @ centred) ** 2)
                       / (rows * max(np.sum(centred ** 2), 1e-300)))
        conditions, encoders_by_arm = {}, {}
        for policy in policies:
            name = f'optnet16_{policy}'
            result = fit_condition(state, protected, views, policy, seed, budget)
            encoders_by_arm[name] = result['encoder']
            release = build_releases(name, seed, result['encoder'], state['model'], registry, out)
            rank_bound = theorem_4_1_rank(protected, state['R'], policy_weights(policy))
            conditions[name] = {
                **{k: v for k, v in result.items() if k != 'encoder'},
                **release, 'condition': name, 'rank': RANK,
                'theorem_4_1_upper_bound': rank_bound,
                'utility_signal_ratio': signal,
                'utility_signal_ratio_definition': (
                    "OLS R^2 of the residualised teacher R on the whitened features V, "
                    "= ||V'R||_F^2 / (n ||R||_F^2). A value near 0 would mean residualisation "
                    "annihilated the utility signal and the encoder would be empty."),
            }
            print('OPTNET', seed, name, 'obj', round(result['training_objective'], 6),
                  'spread', round(result['restart_spread'], 6), flush=True)
        # Persist the selected encoders so the exploratory 2017 transport can reuse them
        # without refitting. Added after the first production run, which therefore has no
        # encoder file; the stage is deterministic, so re-running reproduces them exactly.
        import joblib
        joblib.dump(encoders_by_arm, out / f'seed_{seed}' / 'optnet_encoders.joblib')

        record = {'seed': seed, 'runtime_seconds': time.perf_counter() - tick,
                  'budget_steps': budget, 'starts': STARTS, 'batch': BATCH,
                  'architecture': f'128 -> {HIDDEN} -> {RANK}, ReLU',
                  'gamma': GAMMA, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
                  'departures_from_published_formulation': DEPARTURES,
                  'conditions': conditions, 'inputs': registry.dump(),
                  'machine': machine_state()}
        write_json(marker, record)
        summary[str(seed)] = record
    write_json(out / 'OPTNET_SUMMARY.json', {
        'seeds': summary,
        'method': ('OptNet-ARL (Sadeghi, Wang & Boddeti 2109.05535) -- CONTROLLED ADAPTATION, '
                   'not a verbatim reproduction'),
        'departures_from_published_formulation': DEPARTURES,
        'unique_new_fits': sum(len(v.get('conditions', {})) for v in summary.values()),
    })
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--budget', type=int, default=1500)
    p.add_argument('--calibrate', action='store_true')
    a = p.parse_args()
    if a.calibrate:
        limit_threads()
        torch.set_num_threads(1)
        registry = Registry.new()
        state = build_roles(0, registry)
        labels, _ = load_representation_labels(0, registry)
        protected, views = protected_and_views(state, labels)
        print(calibrate(state, protected, views))
        return
    run(a.out, tuple(a.seeds), a.budget)


if __name__ == '__main__':
    main()
