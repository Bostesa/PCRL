"""Track E: coalition-conditioned partial moment projection of a frozen channel.

This is **post-processing of a fixed neural channel**, not another encoder fit. The
starting channel (`A0` or `J`) is frozen; everything here is an affine map derived from
representation-training rows, applied to `Z` alone, with `H_A` and `H_B` appended
bitwise unchanged.

## Conventions, stated once and tested

`z` is a **row vector** of width `d`. All data matrices are `(n, d)` with rows as
observations. `mu` is the training mean and `Sigma` the training covariance
(denominator `n`, i.e. the second moment of the centred rows).

A symmetric eigendecomposition `Sigma = Q diag(lam) Q'` with a **relative** numerical
rank tolerance `tol * lam.max()` defines the supported subspace. On it,

    R = Q_s diag(lam_s ** -0.5)          (d, r)     whitening
    S = diag(lam_s ** 0.5) Q_s'          (r, d)     unwhitening
    v = (z - mu) @ R                     (n, r)

`R @ S = Q_s Q_s'` is the orthogonal projector onto the supported subspace, so it is
**exactly the identity when the channel has full numerical rank**, and the release at
`k = 0` is then the input unchanged. There is no ridge and no silent regularisation:
unsupported directions are dropped, counted and reported, and the tolerance's effect is
measured rather than assumed.

## Moments

For each trainable role `j`, a service-only nuisance `p_j(s | h_j)` is fitted by
**household cross-fitting inside representation training**, giving out-of-fold
residuals `e_j = onehot(s) - p_j(h_j)`. Cross-fitting limits one form of self-fitting;
it does **not** prove the nuisance is correct or that any conditional independence
holds.

With `b_j(h_j)` the fixed polynomial basis and a per-role validity mask,

    G_j = mean_{valid j} [ v' (b_j(h_j) kron e_j)' ]          (r, B*K)

`v` is centred **globally**, not per role, so `v` may have a nonzero mean on a masked
subpopulation. That is the declared convention: changing it would change the
derivation, and the acceptance tests check the convention that is actually implemented.

    K_j = G_j G_j' / max(||G_j||_F ** 2, eps)                 (r, r), trace 1

with **one** scalar normalisation per role and `eps` locked relative to a fixed
training quantity, so an essentially empty role cannot be amplified into a
full-strength direction of numerical noise.

    K_local     = mean over the three A-roles
    K_coalition = mean over the two AB-roles
    policy L    ranks by K_local
    policy C    ranks by K_local + K_coalition

## Release

`U_k` holds the `k` leading orthonormal eigenvectors of the policy matrix,
`P_k = I - U_k U_k'`, and

    z_out = mu + v P_k S

Ambient width stays `d` for a common wire; the **realised numerical rank is `r - k`**
and is reported separately. This is a **partial** projection: it does not erase
anything unless the whole relevant span is removed, and the top-`k` eigensystem
optimises its declared finite trace criterion in the whitened metric — not attacker
recovery, not residence capability, and not source covariance.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------ locked constants
WHITEN_REL_TOL = 1e-10          # relative to the largest covariance eigenvalue
BASIS_REL_TOL = 1e-6            # relative; above float32 rounding of probability pairs (~1e-7), far below signal
EPS_REL = 1e-8                  # eps = EPS_REL * max_j ||G_j||_F**2, one per (channel, seed)
RFF_FEATURES = 10               # LX augmentation for SEX and race
RFF_SEED = 20269018
REMOVAL_RANKS = (1, 2, 4, 6, 8)

POLICIES = ('L', 'C', 'LX')
CONTROLS = ('marginal', 'pca', 'random')


# ------------------------------------------------------------------ whitening
@dataclass
class Whitening:
    mu: np.ndarray              # (d,)
    R: np.ndarray               # (d, r)
    S: np.ndarray               # (r, d)
    eigenvalues: np.ndarray     # (d,) descending
    rank: int
    tolerance: float
    dropped: int

    def forward(self, z: np.ndarray) -> np.ndarray:
        return (np.asarray(z, np.float64) - self.mu) @ self.R

    def inverse(self, v: np.ndarray) -> np.ndarray:
        return self.mu + np.asarray(v, np.float64) @ self.S

    def report(self) -> dict:
        return {'ambient_width': int(self.R.shape[0]), 'supported_rank': int(self.rank),
                'dropped_directions': int(self.dropped),
                'relative_tolerance': float(self.tolerance),
                'eigenvalues': [float(x) for x in self.eigenvalues],
                'condition_number_supported': float(self.eigenvalues[0]
                                                    / self.eigenvalues[self.rank - 1]),
                'round_trip_is_identity': bool(self.rank == self.R.shape[0]),
                'note': ('R @ S is the orthogonal projector onto the supported subspace. It is '
                         'exactly the identity at full numerical rank, so k=0 releases the '
                         'input unchanged. No ridge is added anywhere.')}


def fit_whitening(z_fit: np.ndarray, rel_tol: float = WHITEN_REL_TOL) -> Whitening:
    x = np.asarray(z_fit, np.float64)
    mu = x.mean(0)
    centred = x - mu
    sigma = (centred.T @ centred) / len(centred)
    sigma = 0.5 * (sigma + sigma.T)                       # exact symmetry before eigh
    lam, q = np.linalg.eigh(sigma)
    order = np.argsort(lam)[::-1]
    lam, q = lam[order], q[:, order]
    threshold = rel_tol * float(lam[0])
    keep = lam > threshold
    rank = int(keep.sum())
    lam_s, q_s = lam[:rank], q[:, :rank]
    R = q_s * (lam_s ** -0.5)                             # (d, r)
    S = (q_s * (lam_s ** 0.5)).T                          # (r, d)
    return Whitening(mu=mu, R=R, S=S, eigenvalues=lam, rank=rank,
                     tolerance=float(rel_tol), dropped=int(len(lam) - rank))


# ------------------------------------------------------------------ service basis
def nonredundant_columns(h_fit: np.ndarray, rel_tol: float = BASIS_REL_TOL) -> list:
    """Indices of `H` columns that are not affine functions of the constant and earlier ones.

    Outcome-free: the rule looks only at the training service array, in fixed column
    order, and never at a label, an outcome or a score. Binary service blocks are
    expected to give one nonredundant coordinate per task, but the count is **measured**
    rather than assumed.
    """
    x = np.asarray(h_fit, np.float64)
    accumulated = [np.ones(len(x))]
    kept = []
    for column in range(x.shape[1]):
        trial = np.column_stack(accumulated + [x[:, column]])
        singular = np.linalg.svd(trial, compute_uv=False)
        if singular[-1] > rel_tol * singular[0]:
            accumulated.append(x[:, column])
            kept.append(column)
    return kept


def polynomial_basis(h: np.ndarray, kept: list, scale: np.ndarray = None):
    """`[1] + degree-1 + degree-2 monomials` in the nonredundant coordinates.

    The constant is kept and the monomials are **not centred**: they are scaled to unit
    training standard deviation and left in place. Scaling is a genuine choice, because
    `G_j` is a raw cross-moment and a column rescaling reweights `G_j G_j'`; it is
    locked from training inputs before any outcome.
    """
    x = np.asarray(h, np.float64)[:, kept]
    columns = [np.ones(len(x))]
    for i in range(x.shape[1]):
        columns.append(x[:, i])
    for i in range(x.shape[1]):
        for j in range(i, x.shape[1]):
            columns.append(x[:, i] * x[:, j])
    basis = np.column_stack(columns)
    if scale is None:
        scale = basis.std(0)
        scale[0] = 1.0                                     # the constant keeps scale 1
        scale[scale <= 0] = 1.0
    return basis / scale, scale


def rff_features(h: np.ndarray, kept: list, weights: np.ndarray, offsets: np.ndarray,
                 scale: np.ndarray = None):
    """Deterministic random Fourier features of `H_A`, for the LX control only.

    LX sees `H_A` and no `H_B`. It is a **richer local conditioner**, not a function-class
    match to the joint-view polynomial basis of `C`: a polynomial in three coordinates
    and a cosine feature map of two coordinates are different families, and matching the
    nominal feature count does not isolate every cause of a `C`-versus-`LX` difference.
    """
    x = np.asarray(h, np.float64)[:, kept]
    projected = x @ weights + offsets
    features = np.sqrt(2.0 / weights.shape[1]) * np.cos(projected)
    if scale is None:
        scale = features.std(0)
        scale[scale <= 0] = 1.0
    return features / scale, scale


def rff_parameters(h_fit: np.ndarray, kept: list, count: int = RFF_FEATURES,
                   seed: int = RFF_SEED) -> dict:
    """Bandwidth from a training-only median heuristic; weights and offsets from a locked seed."""
    x = np.asarray(h_fit, np.float64)[:, kept]
    rng = np.random.default_rng(seed)
    sample = x[rng.choice(len(x), size=min(512, len(x)), replace=False)]
    difference = sample[:, None, :] - sample[None, :, :]
    distance = np.sqrt((difference ** 2).sum(-1))
    upper = distance[np.triu_indices(len(sample), 1)]
    bandwidth = float(np.median(upper[upper > 0])) if (upper > 0).any() else 1.0
    generator = np.random.default_rng(seed + 1)
    weights = generator.normal(0.0, 1.0 / bandwidth, size=(len(kept), count))
    offsets = generator.uniform(0.0, 2 * np.pi, size=count)
    return {'weights': weights, 'offsets': offsets, 'bandwidth': bandwidth,
            'count': int(count), 'seed': int(seed),
            'heuristic': 'median pairwise distance on <=512 training rows, fixed seed'}


# ------------------------------------------------------------------ cross-moments
def kron_rows(basis: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Row-wise Kronecker product: `(n, B) x (n, K) -> (n, B*K)`, basis-major."""
    n, b = basis.shape
    k = residual.shape[1]
    return (basis[:, :, None] * residual[:, None, :]).reshape(n, b * k)


def cross_moment(v: np.ndarray, basis: np.ndarray, residual: np.ndarray,
                 valid: np.ndarray) -> np.ndarray:
    """`G_j = mean_{valid} [ v' (b kron e)' ]`, shape `(r, B*K)`.

    `v` is the globally centred whitened channel and is **not** re-centred on the mask.
    """
    rows = np.flatnonzero(valid)
    if rows.size == 0:
        return np.zeros((v.shape[1], basis.shape[1] * residual.shape[1]))
    features = kron_rows(basis[rows], residual[rows])
    return (v[rows].T @ features) / rows.size


def role_kernel(G: np.ndarray, eps: float) -> tuple:
    """`K_j = G G' / max(||G||_F^2, eps)`; returns the kernel and the raw energy."""
    energy = float((G ** 2).sum())
    return (G @ G.T) / max(energy, eps), energy


def leading_directions(kernel: np.ndarray, k: int, degeneracy_tol: float = 1e-9) -> dict:
    """Top-`k` orthonormal eigenvectors of a symmetric PSD matrix, deterministically.

    `numpy.linalg.eigh` on a symmetric matrix returns an orthonormal basis in ascending
    eigenvalue order. Ties are resolved by that fixed order, and a tie **straddling the
    boundary** `k` is flagged: the removed subspace is then not uniquely determined by
    the spectrum, even though the projector this function returns is deterministic.
    """
    symmetric = 0.5 * (kernel + kernel.T)
    values, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    unstable = False
    if 0 < k < len(values):
        gap = float(values[k - 1] - values[k])
        unstable = bool(gap <= degeneracy_tol * max(float(values[0]), 1e-300))
    else:
        gap = float('nan')
    return {'U': np.ascontiguousarray(vectors[:, :k]), 'eigenvalues': values,
            'boundary_gap': gap, 'boundary_degenerate': unstable}


def projector(U: np.ndarray, dimension: int) -> np.ndarray:
    """`P = I - U U'`, symmetric and idempotent by construction.

    When `U` spans the whole space, `P` is analytically zero and is returned as EXACT
    zeros: the floating residue of `I - U U'` (~1e-16) is a deterministic function of the
    input and a standardising attacker could amplify it (RUN_STATUS amendment 2).
    """
    if U.size == 0:
        return np.eye(dimension)
    if U.shape[1] >= dimension:
        return np.zeros((dimension, dimension))
    return np.eye(dimension) - U @ U.T


def release(whitening: Whitening, z: np.ndarray, P: np.ndarray) -> np.ndarray:
    """`z_out = mu + (z - mu) R P S`, in the row-vector convention."""
    return whitening.inverse(whitening.forward(z) @ P)


# ------------------------------------------------------------------ controls
def pca_retaining(z_fit: np.ndarray, retained: int) -> dict:
    """Generic compression: keep the top-`retained` principal directions in the ORIGINAL metric.

    PCA is defined in the input metric on purpose. In the whitened metric every
    direction has unit variance, so "principal" would be meaningless there; the honest
    contrast is variance-ranked compression against moment-targeted removal.
    """
    x = np.asarray(z_fit, np.float64)
    mu = x.mean(0)
    centred = x - mu
    _u, singular, vt = np.linalg.svd(centred, full_matrices=False)
    basis = np.ascontiguousarray(vt[:retained].T)          # (d, retained)
    total = float((singular ** 2).sum())
    return {'mu': mu, 'basis': basis,
            'projector': basis @ basis.T,
            'singular_values': singular,
            'explained_variance_ratio': float((singular[:retained] ** 2).sum() / total)
            if total > 0 else None}


def random_subspace(dimension: int, retained: int, seed: int) -> dict:
    """One preregistered random `retained`-dimensional subspace in the whitened metric.

    Same algebra as the method with a random `U_k`, so the comparison isolates *which*
    directions are removed. **One draw is one draw**: this is a limited random-compression
    baseline and is never described as the best or the typical random subspace.
    """
    generator = np.random.default_rng(seed)
    gaussian = generator.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(gaussian)
    q = q * np.sign(np.diag(r))                            # fix the QR sign ambiguity
    basis = np.ascontiguousarray(q[:, :retained])
    return {'basis': basis, 'projector': basis @ basis.T, 'seed': int(seed)}


def config_id(*parts) -> str:
    return hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()[:16]


__all__ = ['WHITEN_REL_TOL', 'BASIS_REL_TOL', 'EPS_REL', 'RFF_FEATURES', 'RFF_SEED',
           'REMOVAL_RANKS', 'POLICIES', 'CONTROLS', 'Whitening', 'fit_whitening',
           'nonredundant_columns', 'polynomial_basis', 'rff_features', 'rff_parameters',
           'kron_rows', 'cross_moment', 'role_kernel', 'leading_directions', 'projector',
           'release', 'pca_retaining', 'random_subspace', 'config_id']
