"""Family B channel (numpy only): the SEX-targeted, Fisher-shaped, cross-recipient-coupled Gaussian logit channel ('FSC').

Census context: public-use ACS 2019 1-Year PUMS Florida person records. The service maps the six raw fields (AGEP, WKHP, COW, SCHL,
MAR, POBP), through the admitted 145-column encoding X, to an income probability vector (2) for one recipient and an occupation
probability vector (5) for another; decisions are the argmax of each EMITTED vector. SEX is never a runtime input or output; it
enters only the channel geometry fitted on MODEL_FIT (delta, Sigma_w below).

Coordinates. A person's deterministic logits (income l_I in R^2, occupation l_O in R^5) are mapped to linked centred coordinates
z = [H2 l_I ; H5 l_O] in R^5 (H_k: orthonormal Helmert rows orthogonal to the all-ones vector; softmax is unchanged by the map back).

Channel law (one release instance = one key; one draw per recipient per person):
    F_i      = blockdiag(H2 (diag p_I - p_I p_I^T) H2^T, H5 (diag p_O - p_O p_O^T) H5^T)   per-person Fisher (Hessian of the task
               log loss in z; label-free: computed from the deterministic probabilities)
    'sex'    d_i = (F_i + s^2 Sigma_w^{-1})^{-1} Sigma_w^{-1} delta,   delta = mean_z(women) - mean_z(men), Sigma_w = pooled
               within-SEX covariance of z (MODEL_FIT, SEX labels; frozen per configuration)
    'pca'    the same formula with delta := top principal axis of the total covariance of z (scaled by its sd), Sigma_w := total
               covariance (SEX-blind direction; the matched SEX-blind control)
    v_i      = d_i / sqrt(d_i^T F_i d_i)              (unit Fisher norm: second-order expected log-loss cost of noise = s^2 / 2)
    c_i      = min(s, cap / |v_i|)                    (cap bounds one person's noise in centred-logit units per unit draw)
    z~_i     = z_i + c_i * eps_i * v_i,  eps_i ~ N(0, 1) ONE scalar per person shared by both recipients (maximal coupling along
               the person's own direction: utility is per recipient, so the coupling costs nothing and aligns the linked view's noise)
    'iso'    z~_i = z_i + c_i * F_i^{-1/2} xi_i / sqrt(5), xi_i ~ N(0, I_5) (five independent draws; uncoupled; SEX-blind; second-
               order cost s^2 / 2; c_i = min(s, cap / sqrt(tr F_i^{-1} / 5)))
    'minf'   v_i = e_i / sqrt(lambda_min,i): the eigenvector of F_i with the smallest eigenvalue ("noise where the task is cheapest";
               SEX-blind; unit Fisher norm; sign: largest-|.| component positive), then the same c_i, shared scalar draw, cap, keys as
               'sex'. F_i is block diagonal, so the minimum-Fisher direction lies in one recipient's block (generically).
    emitted  p~_I = softmax(H2^T z~_I), p~_O = softmax(H5^T z~_O); decisions = argmax of the emitted vectors.
The optimal-direction formula is the rank-1 solution of: minimise the linked linear-Gaussian SEX discriminability
delta^T (Sigma_w + Sigma_n)^{-1} delta subject to the second-order Fisher cost tr(F Sigma_n)/2 = s^2/2 (gamma = 2c = s^2).

Seed handling. eps comes from keyed BLAKE2b (32-byte private key per (config, version) release; tag = role, repetition, slot) of the
person's row_id, then Box-Muller: a person's draw depends only on (key, role, repetition, row_id) - not on row order, batch or n.
Different repetitions use different tags (independent draws). Re-querying a person within one release instance returns the same
draw (no averaging by re-query inside an instance; repeated-query protection is NOT claimed across instances).

Deterministic constructions: identity (s = 0 or mode 'none': exactly softmax of the deterministic logits) and the mean vector
E_eps[p~] (Gauss-Hermite quadrature, rank-1 modes only) - the deterministic release with the same mean as the randomized one.

Commands:  python3 -m nrm.fb.channel selftest      (invented arrays only; no data access, no labels)
"""
from __future__ import annotations
import hashlib
import json
import sys
import time

import numpy as np

K = {'income': 2, 'occupation': 5}
DZ = 5
MODES = ('none', 'sex', 'pca', 'iso', 'minf')
CAP = 1.5                 # predeclared per-person noise cap (centred-logit units per unit draw)
GH_NODES = 48             # Gauss-Hermite nodes for the deterministic mean vector
SCHEMA = 'NRM-FB-channel-v1'


def helmert(k):
    """(k-1, k) orthonormal rows, each orthogonal to the all-ones vector."""
    H = np.zeros((k - 1, k))
    for i in range(1, k):
        H[i - 1, :i] = 1.0
        H[i - 1, i] = -float(i)
        H[i - 1] /= np.sqrt(i * (i + 1.0))
    return H


H2, H5 = helmert(2), helmert(5)


def to_z(l_inc, l_occ):
    return np.concatenate([np.asarray(l_inc, np.float64) @ H2.T, np.asarray(l_occ, np.float64) @ H5.T], axis=-1)


def from_z(z):
    z = np.asarray(z, np.float64)
    return z[..., :1] @ H2, z[..., 1:] @ H5


def softmax(l):
    l = np.asarray(l, np.float64)
    e = np.exp(l - l.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def probs(z):
    li, lo = from_z(z)
    return softmax(li), softmax(lo)


def z_from_probs(p_inc, p_occ, floor=1e-300):
    """Centred coordinates of emitted (or deterministic) probability vectors (log is defined up to a constant per block)."""
    return to_z(np.log(np.maximum(p_inc, floor)), np.log(np.maximum(p_occ, floor)))


def _block_fisher(H, p):
    hp = p @ H.T
    return np.einsum('ak,nk,bk->nab', H, p, H) - hp[:, :, None] * hp[:, None, :]


def fisher(z):
    """(n, 5, 5) per-person Fisher of the summed task log loss in z (block diagonal: income 1x1, occupation 4x4)."""
    pi, po = probs(np.atleast_2d(z))
    n = len(pi)
    F = np.zeros((n, DZ, DZ))
    F[:, :1, :1] = _block_fisher(H2, pi)
    F[:, 1:, 1:] = _block_fisher(H5, po)
    return F


# ------------------------------------------------------------------ geometry (fitted on MODEL_FIT; frozen per configuration)
def geometry(z, sex=None, mode='sex', ridge=1e-6):
    """Channel geometry from deterministic centred logits z (n, 5). 'sex' needs the MODEL_FIT SEX labels (0 = men, 1 = women)."""
    if mode not in MODES:
        raise ValueError(f'unknown mode {mode}')
    if mode in ('none', 'iso', 'minf'):
        return {'schema': SCHEMA, 'mode': mode}
    z = np.asarray(z, np.float64)
    if mode == 'sex':
        s = np.asarray(sex)
        if s.shape != (len(z),) or not set(np.unique(s).tolist()) <= {0, 1} or len(np.unique(s)) != 2:
            raise ValueError('SEX labels must be a 0/1 vector with both classes')
        z0, z1 = z[s == 0], z[s == 1]
        delta = z1.mean(0) - z0.mean(0)
        c0, c1 = z0 - z0.mean(0), z1 - z1.mean(0)
        Sw = (c0.T @ c0 + c1.T @ c1) / (len(z) - 2)
    else:
        Sw = np.cov(z.T)
        w, V = np.linalg.eigh(Sw)
        e = V[:, -1] * (1.0 if V[np.argmax(np.abs(V[:, -1])), -1] > 0 else -1.0)
        delta = e * np.sqrt(w[-1])
    Sw = Sw + ridge * np.trace(Sw) / DZ * np.eye(DZ)
    return {'schema': SCHEMA, 'mode': mode, 'delta': delta.tolist(), 'Sw': Sw.tolist(), 'n': int(len(z)),
            'lda_delta2': float(delta @ np.linalg.solve(Sw, delta)) if mode == 'sex' else None}


def scale_geometry(geom, t_inc, t_occ):
    """Geometry of z / T (per-block temperatures): delta and Sigma_w transform linearly."""
    if geom['mode'] in ('none', 'iso', 'minf'):
        return dict(geom)
    D = np.diag([1.0 / t_inc] + [1.0 / t_occ] * 4)
    d = D @ np.asarray(geom['delta']); S = D @ np.asarray(geom['Sw']) @ D
    return {**geom, 'delta': d.tolist(), 'Sw': S.tolist(), 'temperatures': [float(t_inc), float(t_occ)]}


def lda_delta2(z, sex):
    """Linked linear-Gaussian SEX discriminability delta^T Sigma_w^{-1} delta of coordinates z (diagnostic)."""
    return geometry(z, sex, 'sex')['lda_delta2']


# ------------------------------------------------------------------ per-person noise operators
def operators(z, chan):
    """Per-person noise operator: rank-1 modes -> (n, 5) vectors c_i v_i; 'iso' -> (n, 5, 5) matrices c_i F_i^{-1/2} / sqrt(5)."""
    z = np.atleast_2d(np.asarray(z, np.float64))
    mode, s = chan['mode'], float(chan['s'])
    cap = float(chan.get('cap', CAP))
    n = len(z)
    if mode == 'none' or s == 0.0:
        return np.zeros((n, DZ))
    F = fisher(z)
    if mode == 'minf':
        v = minf_directions(F)
        c = np.minimum(s, cap / np.maximum(np.linalg.norm(v, axis=1), 1e-300))
        return v * c[:, None]
    if mode == 'iso':
        w, V = np.linalg.eigh(F)
        w = np.maximum(w, 1e-12)
        Fi_half = np.einsum('nab,nb,ncb->nac', V, 1.0 / np.sqrt(w), V)
        sd = np.sqrt((1.0 / w).sum(1) / DZ)
        c = np.minimum(s, cap / sd)
        return Fi_half * (c / np.sqrt(DZ))[:, None, None]
    geom = chan['geom']
    if geom['mode'] != mode:
        raise ValueError('channel mode and geometry mode differ')
    Sw = np.asarray(geom['Sw']); delta = np.asarray(geom['delta'])
    Swi = np.linalg.inv(Sw)
    a = Swi @ delta
    A = F + (s * s) * Swi[None]
    d = np.linalg.solve(A, np.broadcast_to(a, (n, DZ))[..., None])[..., 0]
    q = np.einsum('ni,nij,nj->n', d, F, d)
    v = d / np.sqrt(np.maximum(q, 1e-300))[:, None]
    c = np.minimum(s, cap / np.maximum(np.linalg.norm(v, axis=1), 1e-300))
    return v * c[:, None]


def minf_directions(F):
    """(n, 5) unit-Fisher-norm directions along each person's smallest-eigenvalue eigenvector of F_i (deterministic sign)."""
    w, V = np.linalg.eigh(F)
    e = V[:, :, 0]
    big = np.argmax(np.abs(e), axis=1)
    sgn = np.sign(e[np.arange(len(e)), big])
    sgn[sgn == 0] = 1.0
    return e * sgn[:, None] / np.sqrt(np.maximum(w[:, 0], 1e-12))[:, None]


def perturb(z, op, eps):
    """z~ = z + op * eps for one draw: rank-1 op (n, 5) with eps (n,); 'iso' op (n, 5, 5) with eps (n, 5)."""
    if op.ndim == 2:
        return z + op * np.asarray(eps).reshape(-1, 1)
    return z + np.einsum('nab,nb->na', op, eps)


# ------------------------------------------------------------------ keyed one-draw-per-person normals
def keyed_normals(key, tag, row_id, m):
    """(n, m) standard normals; person i's values depend only on (key, tag, row_id[i])."""
    if not isinstance(key, (bytes, bytearray)) or not 16 <= len(key) <= 64:
        raise ValueError('a 16-64 byte private key is required')
    pairs = (m + 1) // 2
    nb = 16 * pairs
    if nb > 64:
        raise ValueError('at most 8 normals per person per tag')
    t = str(tag).encode()
    buf = b''.join(hashlib.blake2b(t + b'|' + str(r).encode(), key=bytes(key), digest_size=nb).digest() for r in row_id)
    U = (np.frombuffer(buf, dtype='<u8').reshape(len(row_id), 2 * pairs) >> np.uint64(11)).astype(np.float64) * 2.0 ** -53
    u1 = 1.0 - U[:, 0::2]
    u2 = U[:, 1::2]
    rad = np.sqrt(-2.0 * np.log(u1))
    Z = np.empty((len(row_id), 2 * pairs))
    Z[:, 0::2] = rad * np.cos(2 * np.pi * u2)
    Z[:, 1::2] = rad * np.sin(2 * np.pi * u2)
    return Z[:, :m]


def release_key(root, config_id, version):
    """Private per-(config, version) release key derived from a private 32-byte root (never published)."""
    if not isinstance(root, (bytes, bytearray)) or len(root) != 32:
        raise ValueError('a 32-byte private root is required')
    return hashlib.blake2b(f'NRM-FB|{config_id}|V{int(version)}'.encode(), key=bytes(root), digest_size=32).digest()


def emit(z, chan, key=None, role=None, row_id=None, reps=1, tag_prefix='NRM-FB'):
    """Emitted vectors. Deterministic channels (mode 'none' or s = 0) return {'income': (n, 2), 'occupation': (n, 5)}; randomized
    channels return (R, n, k) stacks: repetition r is an independent one-draw-per-person release instance."""
    z = np.atleast_2d(np.asarray(z, np.float64))
    if chan['mode'] == 'none' or float(chan['s']) == 0.0:
        pi, po = probs(z)
        return {'income': pi, 'occupation': po}
    if key is None or row_id is None or role is None:
        raise ValueError('a randomized release needs a key, a role and the row ids')
    if len(row_id) != len(z):
        raise ValueError('row ids and logits differ in length')
    op = operators(z, chan)
    out = {'income': np.empty((reps, len(z), 2)), 'occupation': np.empty((reps, len(z), 5))}
    for r in range(reps):
        out['income'][r], out['occupation'][r] = _instance(z, op, key, role, row_id, r, tag_prefix)
    return out


def _instance(z, op, key, role, row_id, r, tag_prefix='NRM-FB'):
    """Emitted vectors of release instance r (the single code path shared by emit and emit_one: bitwise-identical draws)."""
    m = DZ if op.ndim == 3 else 1
    eps = keyed_normals(key, f'{tag_prefix}|{role}|r{int(r)}', row_id, m)
    return probs(perturb(z, op, eps if m == DZ else eps[:, 0]))


def emit_one(z, chan, key, role, row_id, r, tag_prefix='NRM-FB'):
    """Instance r only (deployment): equals emit(...)[task][r] bitwise for the same inputs."""
    z = np.atleast_2d(np.asarray(z, np.float64))
    if chan['mode'] == 'none' or float(chan['s']) == 0.0:
        raise ValueError('emit_one is for randomized channels')
    if key is None or row_id is None or role is None or len(row_id) != len(z):
        raise ValueError('a randomized release needs a key, a role and one row id per row')
    pi, po = _instance(z, operators(z, chan), key, role, row_id, r, tag_prefix)
    return {'income': pi, 'occupation': po}


def mean_vector(z, chan, nodes=GH_NODES):
    """Deterministic construction: E_eps[emitted vector] (exact up to quadrature; rank-1 modes only)."""
    z = np.atleast_2d(np.asarray(z, np.float64))
    if chan['mode'] == 'iso':
        raise ValueError('mean vector is defined for the rank-1 modes only')
    op = operators(z, chan)
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    pi = np.zeros((len(z), 2)); po = np.zeros((len(z), 5))
    for xj, wj in zip(x, w):
        a, b = probs(perturb(z, op, np.full(len(z), xj)))
        pi += wj * a; po += wj * b
    return {'income': pi, 'occupation': po}


# ------------------------------------------------------------------ metrics of emitted vectors (labels supplied by the caller)
def task_metrics(P, y):
    """Accuracy, NLL and summed Brier of emitted vectors P ((n, k) or (R, n, k)) and their argmax decisions; per repetition, mean
    and Monte Carlo SE over repetitions (SE 0 for a deterministic release)."""
    P = np.asarray(P, np.float64)
    if P.ndim == 2:
        P = P[None]
    y = np.asarray(y, np.int64)
    R, n, k = P.shape
    py = P[:, np.arange(n), y]
    nll = -np.log(np.maximum(py, 1e-300)).mean(1)
    onehot = np.eye(k)[y]
    brier = ((P - onehot[None]) ** 2).sum(2).mean(1)
    acc = (P.argmax(2) == y[None]).mean(1)
    se = (lambda a: float(a.std(ddof=1) / np.sqrt(R)) if R > 1 else 0.0)
    return {'R': int(R), 'per_rep': {'acc': acc.tolist(), 'nll': nll.tolist(), 'brier': brier.tolist()},
            'acc': float(acc.mean()), 'nll': float(nll.mean()), 'brier': float(brier.mean()),
            'mc_se': {'acc': se(acc), 'nll': se(nll), 'brier': se(brier)}}


# ------------------------------------------------------------------ selftest (invented arrays only)
def _invented(rng, n):
    """Invented linked logits with a SEX shift concentrated in a low-variance, low-Fisher occupation direction (no real data)."""
    sex = rng.integers(0, 2, n)
    base = rng.normal(size=(n, DZ)) * np.array([1.2, 1.0, 0.8, 0.6, 0.15])
    base[:, 4] += 0.25 * (2 * sex - 1)              # small-variance coordinate carries the SEX shift
    base[:, 0] += 0.10 * (2 * sex - 1)
    return base, sex


def selftest():
    t0 = time.process_time()
    rng = np.random.default_rng(20261020)
    checks, info = {}, {}
    # 1. coordinates
    checks['helmert_orthonormal'] = all(np.allclose(H @ H.T, np.eye(len(H))) and np.allclose(H @ np.ones(H.shape[1]), 0) for H in (H2, H5))
    li, lo = rng.normal(size=(50, 2)), rng.normal(size=(50, 5))
    pi, po = probs(to_z(li, lo))
    checks['softmax_preserved'] = bool(np.allclose(pi, softmax(li)) and np.allclose(po, softmax(lo)))
    checks['z_from_probs_roundtrip'] = bool(np.allclose(z_from_probs(pi, po), to_z(li, lo)))
    # 2. Fisher = Hessian of the expected log loss (finite differences, one person)
    z1 = rng.normal(size=(1, DZ)); p1i, p1o = probs(z1)

    def exp_loss(zz):
        a, b = probs(zz)
        return -(p1i * np.log(a)).sum() - (p1o * np.log(b)).sum()
    h = 1e-4; Hn = np.zeros((DZ, DZ))
    for i in range(DZ):
        for j in range(DZ):
            e_i, e_j = np.eye(DZ)[i] * h, np.eye(DZ)[j] * h
            Hn[i, j] = (exp_loss(z1 + e_i + e_j) - exp_loss(z1 + e_i - e_j) - exp_loss(z1 - e_i + e_j) + exp_loss(z1 - e_i - e_j)) / (4 * h * h)
    checks['fisher_equals_hessian'] = bool(np.allclose(fisher(z1)[0], Hn, atol=1e-5))
    # 3. invented release
    n = 20000
    z, sex = _invented(rng, n)
    row_id = np.arange(100000, 100000 + n)
    key = hashlib.blake2b(b'selftest-invented-root', digest_size=32).digest()
    g_sex, g_pca = geometry(z, sex, 'sex'), geometry(z, None, 'pca')
    ident = emit(z, {'mode': 'sex', 's': 0.0, 'geom': g_sex})
    pi0, po0 = probs(z)
    checks['identity_channel_exact'] = bool(np.array_equal(ident['income'], pi0) and np.array_equal(ident['occupation'], po0))
    chan = {'mode': 'sex', 's': 0.2, 'geom': g_sex}
    E = emit(z, chan, key, 'MODEL_SELECT', row_id, reps=3)
    checks['shapes_one_draw_per_person'] = E['income'].shape == (3, n, 2) and E['occupation'].shape == (3, n, 5)
    checks['valid_probability_rows'] = bool(all(np.isfinite(E[t]).all() and np.allclose(E[t].sum(-1), 1) for t in E))
    perm = rng.permutation(n)
    Ep = emit(z[perm], chan, key, 'MODEL_SELECT', row_id[perm], reps=3)
    checks['draws_independent_of_row_order'] = bool(np.array_equal(Ep['occupation'], E['occupation'][:, perm]))
    checks['draws_reproducible'] = bool(np.array_equal(emit(z, chan, key, 'MODEL_SELECT', row_id, reps=3)['income'], E['income']))
    checks['repetitions_differ'] = bool(not np.allclose(E['occupation'][0], E['occupation'][1]))
    other = emit(z, chan, hashlib.blake2b(b'other-root', digest_size=32).digest(), 'MODEL_SELECT', row_id, reps=1)
    checks['keys_differ'] = bool(not np.allclose(other['occupation'][0], E['occupation'][0]))
    # coupling: z~ - z is the person's operator times ONE scalar shared by both recipients
    op = operators(z, chan)
    dz = z_from_probs(E['income'][0], E['occupation'][0]) - z
    c = (dz * op).sum(1) / np.maximum((op * op).sum(1), 1e-300)
    checks['rank1_coupled_draw'] = bool(np.allclose(dz, c[:, None] * op, atol=1e-6))
    eps = keyed_normals(key, 'NRM-FB|MODEL_SELECT|r0', row_id, 1)[:, 0]
    checks['coupled_scalar_is_the_keyed_draw'] = bool(np.allclose(c, eps, atol=1e-6))
    Z = keyed_normals(key, 'moments', np.arange(200000), 5)
    info['normal_moments'] = {'mean': float(Z.mean()), 'var': float(Z.var()), 'max_abs_corr': float(np.abs(np.corrcoef(Z.T) - np.eye(5)).max())}
    checks['keyed_normals_moments'] = abs(info['normal_moments']['mean']) < 0.01 and abs(info['normal_moments']['var'] - 1) < 0.01 and \
        info['normal_moments']['max_abs_corr'] < 0.01
    # 4. second-order cost: labels drawn from the deterministic probabilities (calibrated), small s, cap not binding
    yi = (rng.random(n) < pi0[:, 1]).astype(np.int64)
    yo = np.array([rng.choice(5, p=p) for p in po0])
    s_small = 0.1
    for mode, geom in (('sex', g_sex), ('pca', g_pca), ('iso', {'mode': 'iso'}), ('minf', {'mode': 'minf'})):
        ch = {'mode': mode, 's': s_small, 'geom': geom}
        Em = emit(z, ch, key, 'MODEL_FIT', row_id, reps=8)
        cost = (task_metrics(Em['income'], yi)['nll'] - task_metrics(pi0, yi)['nll']) + \
               (task_metrics(Em['occupation'], yo)['nll'] - task_metrics(po0, yo)['nll'])
        info[f'cost_{mode}'] = {'realized_nll_cost': cost, 'second_order': s_small ** 2 / 2}
    checks['second_order_cost_matches'] = all(abs(info[f'cost_{m}']['realized_nll_cost'] / (s_small ** 2 / 2) - 1) < 0.35 for m in ('sex', 'pca', 'iso', 'minf'))
    # MINF: unit Fisher norm, minimum-eigenvalue direction, cost s^2 / 2, cap share recorded, coupled shared scalar
    Fz = fisher(z)
    vm = minf_directions(Fz)
    lam = np.linalg.eigvalsh(Fz)[:, 0]
    checks['minf_unit_fisher_norm'] = bool(np.allclose(np.einsum('ni,nij,nj->n', vm, Fz, vm), 1.0, atol=1e-8))
    checks['minf_is_min_eigen_direction'] = bool(np.allclose(np.einsum('ni,ni->n', vm, vm) * lam, 1.0, atol=1e-6))
    info['minf_cap_binding_share_at_s0.1'] = float(np.mean(np.linalg.norm(vm, axis=1) * s_small > CAP))
    checks['minf_cost_s2_over_2'] = abs(info['cost_minf']['realized_nll_cost'] / (s_small ** 2 / 2) - 1) < 0.35
    Em = emit(z[:2000], {'mode': 'minf', 's': 0.2, 'geom': {'mode': 'minf'}}, key, 'MODEL_SELECT', row_id[:2000], reps=1)
    opm = operators(z[:2000], {'mode': 'minf', 's': 0.2, 'geom': {'mode': 'minf'}})
    dzm = z_from_probs(Em['income'][0], Em['occupation'][0]) - z[:2000]
    cm = (dzm * opm).sum(1) / np.maximum((opm * opm).sum(1), 1e-300)
    checks['minf_rank1_shared_keyed_draw'] = bool(np.allclose(dzm, cm[:, None] * opm, atol=1e-6) and
                                                  np.allclose(cm, keyed_normals(key, 'NRM-FB|MODEL_SELECT|r0', row_id[:2000], 1)[:, 0], atol=1e-6))
    one_q = np.mean(emit_one(z[:2000], {'mode': 'minf', 's': 0.2, 'geom': {'mode': 'minf'}}, key, 'MODEL_SELECT', row_id[:2000], 0)['occupation'] == Em['occupation'][0])
    checks['emit_one_equals_emit'] = bool(one_q == 1.0)
    # 5. mean vector (deterministic construction) equals the Monte Carlo average of draws
    mv = mean_vector(z[:500], chan)
    big = emit(z[:500], chan, key, 'MC', row_id[:500], reps=4000)
    info['mean_vector_vs_mc_max_abs'] = float(np.abs(big['occupation'].mean(0) - mv['occupation']).max())
    checks['mean_vector_matches_mc'] = info['mean_vector_vs_mc_max_abs'] < 0.01
    checks['mean_vector_identity_at_s0'] = bool(np.allclose(mean_vector(z[:50], {'mode': 'sex', 's': 0.0, 'geom': g_sex})['occupation'], po0[:50]))
    # 6. targeting: at the same second-order cost the SEX-targeted channel lowers the linear SEX discriminability more than the
    #    SEX-blind channels (one draw per person; invented data with the SEX shift in a low-variance direction)
    s_t = 0.3
    d2 = {'identity': lda_delta2(z, sex)}
    for mode, geom in (('sex', g_sex), ('pca', g_pca), ('iso', {'mode': 'iso'}), ('minf', {'mode': 'minf'})):
        Em = emit(z, {'mode': mode, 's': s_t, 'geom': geom}, key, 'MODEL_SELECT', row_id, reps=1)
        d2[mode] = lda_delta2(z_from_probs(Em['income'][0], Em['occupation'][0]), sex)
    info['lda_delta2_one_draw'] = d2
    checks['targeted_beats_blind_on_invented_data'] = d2['sex'] < min(d2['pca'], d2['iso']) and d2['sex'] < d2['identity']
    # 7. temperature scaling of the geometry is linear
    gs = scale_geometry(g_sex, 2.0, 0.5)
    zz = z / np.array([2.0] + [0.5] * 4)
    checks['scaled_geometry_equals_refit'] = bool(np.allclose(np.asarray(gs['delta']), np.asarray(geometry(zz, sex, 'sex')['delta'])))
    # 8. metrics: deterministic SE is 0; refusals
    checks['deterministic_mc_se_zero'] = task_metrics(po0, yo)['mc_se']['nll'] == 0.0
    ref = {}
    for k_, f in {'randomized_without_key': lambda: emit(z, chan, None, 'X', row_id, 1),
                  'short_key': lambda: keyed_normals(b'short', 't', row_id[:3], 1),
                  'mode_geometry_mismatch': lambda: operators(z[:5], {'mode': 'pca', 's': 0.2, 'geom': g_sex}),
                  'one_class_sex': lambda: geometry(z[:10], np.zeros(10, int), 'sex')}.items():
        try:
            f(); ref[k_] = False
        except ValueError:
            ref[k_] = True
    checks['refusals'] = all(ref.values())
    checks['no_data_access_module'] = 'nrm.common' not in sys.modules
    ok = all(bool(v) for v in checks.values())
    out = {'schema': SCHEMA + '-selftest', 'ok': ok, 'checks': {k_: bool(v) for k_, v in checks.items()}, 'refusals': ref, 'info': info,
           'invented_data_only': True, 'cpu_s': round(time.process_time() - t0, 2)}
    print(json.dumps(out, indent=1, default=float))
    return ok


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        sys.exit(0 if selftest() else 1)
    sys.exit('usage: python3 -m nrm.fb.channel selftest')
