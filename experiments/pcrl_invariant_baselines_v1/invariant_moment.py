"""Rotation-invariant conditional moment blocks: a finite objective, not a certificate.

The predecessor (``experiments.pcrl_nonlinear_rank_v1``) penalised a feature family
that was **coordinate dependent**: equal monomial weights, per-feature standardisation
and a finite Fourier bank together let the objective fall under rotations ``W -> W Q``
that leave the released information untouched. This module replaces that block with
two components that are exactly invariant for **every** orthogonal ``Q``, by algebra
rather than by Monte-Carlo convergence:

* :class:`QuadraticBlock` -- ``sum_{k,c} ||M_jkc||_F^2`` for the symmetric moment
  ``M_jkc = mean_i z_i z_i' b_k(H) e_c``. Rotation acts as ``M -> Q' M Q`` and the
  Frobenius norm is unitarily invariant.
* :class:`KernelBlock` -- an **exact** average of three Gaussian kernels on a frozen
  per-role subset of at most 512 representation-training rows. A radial kernel sees
  the data only through pairwise distances, which orthogonal maps preserve exactly.

Scope, unchanged from the predecessor. Every quantity here is a finite empirical
moment of a fixed feature family on a bounded row subset. Nothing establishes
marginal privacy, conditional privacy, a calibrated conditional-independence test,
generalisation, or a guarantee against a nonlinear downstream attacker. Rotation
invariance removes one provably inert direction of optimiser slack; it does **not**
imply the penalty captures all sensitive information. The objective is nonlinear in
``W`` and generally nonconvex: the closed-form global-optimality statement of the
original linear-moment family is **not** inherited.

The kernel block is exact **for the declared subset**, not for the ACS population. It
is never described as an exact full-data kernel objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np

# ------------------------------------------------------------------ frozen constants
# Hashed into results/pcrl_invariant_baselines_v1/PROTOCOL_FREEZE.json before any fit.
BANDS = (0.5, 1.0, 2.0)
KERNEL_SUBSET_MAX = 512
BANDWIDTH_SUBSET_MAX = 1024
NORMALIZER_FLOOR = 1e-12
BLOCK_WEIGHT = 0.5            # quadratic block / kernel block, equal weight
COMBINE_WEIGHT = 0.5          # original linear-moment term / nonlinear term
DEFAULT_CHUNK = 4096

BANDWIDTH_SEED_BASE = 20260918      # reused from the predecessor, unchanged
KERNEL_SUBSET_SEED_BASE = 20260930  # new; frozen in the protocol
PERTURB_SEED_BASE = 20260920        # reused from the predecessor, unchanged
PERTURB_MAGNITUDE = 0.05

# Optimiser: reused from the predecessor without change.
MAX_UPDATES = 200
INITIAL_STEP = 1.0
SHRINK = 0.5
MAX_BACKTRACK = 20
MIN_IMPROVEMENT = 1e-14
MIN_GRAD_NORM = 1e-12

# Fixed role order; ``role_index`` in the kernel-subset seed formula is a position here.
ROLE_ORDER = ('A/SEX', 'A/RAC1P', 'A/public_coverage', 'AB/SEX', 'AB/RAC1P')

SQRT2 = float(np.sqrt(2.0))


def bandwidth_seed(seed: int, r: int) -> int:
    return BANDWIDTH_SEED_BASE + 100 * seed + r


def kernel_subset_seed(seed: int, r: int, role_index: int) -> int:
    return KERNEL_SUBSET_SEED_BASE + 100 * seed + 10 * r + role_index


def perturb_seed(seed: int, r: int) -> int:
    return PERTURB_SEED_BASE + 100 * seed + r


def quadratic_pairs(r: int):
    """Degree-two monomial index pairs ``(a, b)`` with ``a <= b``, lexicographic."""
    return list(combinations_with_replacement(range(r), 2))


def packed_weights(pairs) -> np.ndarray:
    """``sqrt(2)`` off the diagonal, ``1`` on it.

    With these weights the packed squared sum equals
    ``sum_a M_aa^2 + 2 sum_{a<b} M_ab^2 = ||M||_F^2`` identically, which is the whole
    repair of the predecessor's quadratic block.
    """
    return np.array([1.0 if a == b else SQRT2 for a, b in pairs])


# ------------------------------------------------------------------ shared Z features
@dataclass
class QuadraticFeatures:
    """The packed degree-two monomial map of ``Z``, shared by every role.

    Depends only on ``W``, never on the protected role, so it is built once per
    objective evaluation. No centering and no per-coordinate scaling: a frozen
    non-isotropic centering matrix would **not** conjugate as ``Q' C Q`` and so would
    break the invariance the packed weights restore.
    """

    r: int
    pairs: list
    weights: np.ndarray           # (P,) packed sqrt(2) weights

    def __post_init__(self):
        p = len(self.pairs)
        self._a = np.fromiter((q[0] for q in self.pairs), dtype=np.intp, count=p)
        self._b = np.fromiter((q[1] for q in self.pairs), dtype=np.intp, count=p)
        # Dense 0/1 scatter matrices turn the backpropagation into two (n,P)@(P,r)
        # products. Adding both contributions unconditionally is correct on the
        # diagonal too: for a == b the two terms coincide and sum to 2 g z_a.
        self._select_a = np.zeros((p, self.r))
        self._select_b = np.zeros((p, self.r))
        self._select_a[np.arange(p), self._a] = 1.0
        self._select_b[np.arange(p), self._b] = 1.0

    @classmethod
    def build(cls, r: int) -> "QuadraticFeatures":
        pairs = quadratic_pairs(r)
        return cls(r=r, pairs=pairs, weights=packed_weights(pairs))

    @property
    def width(self) -> int:
        return len(self.pairs)

    def transform(self, z: np.ndarray) -> np.ndarray:
        return (z[:, self._a] * z[:, self._b]) * self.weights

    def backpropagate(self, g: np.ndarray, z: np.ndarray) -> np.ndarray:
        """``d(objective)/dZ`` given ``d(objective)/d(packed monomials)``."""
        gw = g * self.weights
        return (gw * z[:, self._b]) @ self._select_a + (gw * z[:, self._a]) @ self._select_b

    def to_full_moment(self, packed: np.ndarray) -> np.ndarray:
        """Unpack one packed moment vector into the symmetric ``(r, r)`` matrix.

        Reference path only: used by the fixture that checks the packed implementation
        against a direct ``||M||_F`` computation.
        """
        m = np.zeros((self.r, self.r))
        raw = packed / self.weights
        m[self._a, self._b] = raw
        m[self._b, self._a] = raw
        return m


@dataclass
class FeatureBundle:
    """``Z`` and its packed monomials for one ``W``, on the full representation rows."""

    z: np.ndarray                 # (n, r)
    quad: np.ndarray              # (n, P)
    chunk: int


def build_features(quadratic: QuadraticFeatures, v: np.ndarray, w: np.ndarray,
                   chunk: int = DEFAULT_CHUNK) -> FeatureBundle:
    n = len(v)
    z = np.empty((n, w.shape[1]))
    quad = np.empty((n, quadratic.width))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = v[start:stop] @ w
        z[start:stop] = block
        quad[start:stop] = quadratic.transform(block)
    return FeatureBundle(z=z, quad=quad, chunk=chunk)


# ------------------------------------------------------------------ bandwidth
def reference_bandwidth(z_reference: np.ndarray, seed: int, r: int) -> dict:
    """Median strictly positive pairwise distance of the common reference projection.

    Frozen once per (seed, rank) and then never recomputed from an optimised ``W`` --
    doing so would make the objective self-referential. Aborts loudly if no strictly
    positive distance exists; it never silently falls back.
    """
    from scipy.spatial.distance import pdist

    z = np.asarray(z_reference, dtype=np.float64)
    n = len(z)
    rng = np.random.default_rng(bandwidth_seed(seed, r))
    m = min(BANDWIDTH_SUBSET_MAX, n)
    subset = np.sort(rng.choice(n, m, replace=False))
    distances = pdist(z[subset])
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError('bandwidth diagnostic failure: no positive pairwise distances '
                         'in the reference projection subset')
    return {
        'sigma_reference': float(np.median(positive)),
        'bandwidth_subset_rows': int(m),
        'bandwidth_subset_seed': bandwidth_seed(seed, r),
        'positive_distance_pairs': int(len(positive)),
        'zero_distance_pairs': int(len(distances) - len(positive)),
        'reference_rows': int(n),
    }


# ------------------------------------------------------------------ quadratic block
def quadratic_moments(bundle: FeatureBundle, weights: np.ndarray, mask, n_valid: int):
    """Packed moment table ``Mp[p, (k,c)] = mean_i quad[i,p] * b_k(i) * e_c(i)``."""
    quad = bundle.quad if mask is None else bundle.quad[mask]
    if len(quad) != n_valid:
        raise ValueError('row count does not match the role mask')
    return (quad.T @ weights) / max(n_valid, 1)


def quadratic_value(bundle: FeatureBundle, weights: np.ndarray, mask, n_valid: int) -> float:
    mp = quadratic_moments(bundle, weights, mask, n_valid)
    return float(np.sum(mp * mp))


def quadratic_value_and_dz(quadratic: QuadraticFeatures, bundle: FeatureBundle,
                           weights: np.ndarray, mask, n_valid: int, scale: float):
    """``(scale * D_quad, d(scale * D_quad)/dZ)`` on the full row set."""
    mp = quadratic_moments(bundle, weights, mask, n_valid)
    value = scale * float(np.sum(mp * mp))
    # dD/d(quad)[i,p] = (2/n) sum_{kc} Mp[p,kc] weights[i,kc]
    g = (2.0 * scale / max(n_valid, 1)) * (weights @ mp.T)
    if mask is None:
        dz = quadratic.backpropagate(g, bundle.z)
    else:
        dz = np.zeros_like(bundle.z)
        dz[mask] = quadratic.backpropagate(g, bundle.z[mask])
    return value, dz


def quadratic_value_reference(quadratic: QuadraticFeatures, bundle: FeatureBundle,
                              weights: np.ndarray, mask, n_valid: int) -> float:
    """Independent reference path: build each symmetric ``M`` and take ``||M||_F^2``.

    Slow and allocation-heavy; used only by the validation fixtures to confirm the
    packed ``sqrt(2)`` implementation computes the Frobenius norm it claims to.
    """
    mp = quadratic_moments(bundle, weights, mask, n_valid)
    total = 0.0
    for column in range(mp.shape[1]):
        m = quadratic.to_full_moment(mp[:, column])
        total += float(np.sum(m * m))
    return total


# ------------------------------------------------------------------ exact kernel block
@dataclass
class KernelBlock:
    """Exact three-bandwidth Gaussian kernel block on one role's frozen row subset.

    ``L = (B B') * (E E')`` is fixed and precomputed. The block equals the squared
    norm of the empirical kernel-feature residual moment
    ``N_kc = (1/m) sum_i phi(z_i) b_k(i) e_c(i)``; it is **not** a calibrated
    conditional-independence test and carries no Type-I control.

    Estimator convention: the finite-sample diagonal ``i == l`` is **kept**, so this is
    a **V-statistic**. Declared before fitting and never swapped on an outcome.
    """

    role: str
    row_index: np.ndarray         # (m,) indices into the FULL representation rows
    subset_index: np.ndarray      # (m,) indices into the role's valid-row arrays
    l_matrix: np.ndarray          # (m, m) fixed
    inv_two_h2: np.ndarray        # (3,) 1 / (2 h_m^2)
    inv_h2: np.ndarray            # (3,) 1 / h_m^2
    sigma_reference: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def m(self) -> int:
        return len(self.row_index)

    @classmethod
    def build(cls, role_name: str, role, valid_rows: np.ndarray, sigma_reference: float,
              seed: int, r: int, role_index: int | None = None) -> "KernelBlock":
        """Frozen deterministic subset, then the fixed ``L`` matrix.

        Selection uses only the role's validity mask and the frozen RNG. No residence
        or commute label, and no outcome, enters it.

        ``role_index`` defaults to the role's position in the frozen :data:`ROLE_ORDER`,
        which is what every production fit uses; it is overridable only so that
        synthetic fixtures can use their own role names.
        """
        if role_index is None:
            role_index = ROLE_ORDER.index(role_name)
        rng = np.random.default_rng(kernel_subset_seed(seed, r, role_index))
        n_valid = role.n_valid
        m = min(KERNEL_SUBSET_MAX, n_valid)
        subset = np.sort(rng.choice(n_valid, m, replace=False))
        b = np.ascontiguousarray(role.basis[subset])
        e = np.ascontiguousarray(role.residual[subset])
        l_matrix = (b @ b.T) * (e @ e.T)
        h = np.asarray(BANDS, dtype=np.float64) * sigma_reference
        if not np.all(h > 0):
            raise ValueError(f'bandwidth diagnostic failure for {role_name}: h = {h}')

        # Realised per-class support on the subset is recorded by ``subset_class_support``
        # below, never repaired. Rare RAC1P classes can land at zero positive support;
        # such a class still contributes through its residual e_c = -m_c(H).
        return cls(role=role_name, row_index=valid_rows[subset], subset_index=subset,
                   l_matrix=l_matrix, inv_two_h2=1.0 / (2.0 * h ** 2), inv_h2=1.0 / h ** 2,
                   sigma_reference=float(sigma_reference),
                   diagnostics={
                       'role': role_name, 'role_index': role_index,
                       'subset_seed': kernel_subset_seed(seed, r, role_index),
                       'subset_rows': int(m), 'role_valid_rows': int(n_valid),
                       'bandwidths': h.tolist(), 'bands': list(BANDS),
                       'sigma_reference': float(sigma_reference),
                       'estimator': 'V-statistic (diagonal retained)',
                       'l_matrix_frobenius': float(np.linalg.norm(l_matrix)),
                   })

    # -------------------------------------------------------------- evaluation
    def _squared_distances(self, z_sub: np.ndarray) -> np.ndarray:
        sq = np.einsum('ij,ij->i', z_sub, z_sub)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (z_sub @ z_sub.T)
        np.maximum(d2, 0.0, out=d2)
        return d2

    def kernel(self, z_sub: np.ndarray) -> np.ndarray:
        d2 = self._squared_distances(z_sub)
        k = np.zeros_like(d2)
        for inv in self.inv_two_h2:
            k += np.exp(-d2 * inv)
        k /= len(self.inv_two_h2)
        return k

    def value(self, z_full: np.ndarray) -> float:
        z_sub = z_full[self.row_index]
        return float(np.sum(self.kernel(z_sub) * self.l_matrix)) / (self.m ** 2)

    def value_and_dz(self, z_full: np.ndarray, scale: float):
        """``(scale * D_kernel, d/dZ)`` scattered back to the full row set.

        With ``A(i,l) = -(1/3) sum_m exp(-d^2/(2 h_m^2)) / h_m^2`` and
        ``G = L * A`` (symmetric), ``dD/dZ = (2/m^2) (diag(G 1) - G) Z`` -- a
        graph-Laplacian contraction on at most 512 rows.
        """
        z_sub = z_full[self.row_index]
        d2 = self._squared_distances(z_sub)
        nb = len(self.inv_two_h2)
        k = np.zeros_like(d2)
        a = np.zeros_like(d2)
        for inv2, inv1 in zip(self.inv_two_h2, self.inv_h2):
            e = np.exp(-d2 * inv2)
            k += e
            a -= e * inv1
        k /= nb
        a /= nb
        value = scale * float(np.sum(k * self.l_matrix)) / (self.m ** 2)
        g = self.l_matrix * a
        coefficient = 2.0 * scale / (self.m ** 2)
        dz_sub = coefficient * (g.sum(axis=1)[:, None] * z_sub - g @ z_sub)
        dz = np.zeros_like(z_full)
        dz[self.row_index] = dz_sub
        return value, dz

    def value_dense_reference(self, z_full: np.ndarray, basis: np.ndarray,
                              residual: np.ndarray) -> float:
        """Independent reference: rebuild ``L`` from ``B`` and ``E`` and sum explicitly.

        Used only by the validation fixture, to confirm the precomputed ``L`` path and
        the elementwise product convention are the formula they claim to be.
        """
        z_sub = z_full[self.row_index]
        b = basis[self.subset_index]
        e = residual[self.subset_index]
        k = self.kernel(z_sub)
        total = 0.0
        for i in range(self.m):
            for l in range(self.m):
                total += k[i, l] * float(b[i] @ b[l]) * float(e[i] @ e[l])
        return total / (self.m ** 2)


def subset_class_support(role, block: KernelBlock, labels_valid: np.ndarray) -> dict:
    """Realised per-class support on the frozen subset. Recorded, never repaired."""
    y = labels_valid[block.subset_index]
    counts = np.bincount(y, minlength=role.classes)
    return {
        'subset_support': counts.tolist(),
        'subset_zero_support_classes': np.flatnonzero(counts == 0).tolist(),
        'population_support': role.support.tolist(),
        'population_unsupported_classes': list(role.unsupported),
        'note': ('A class with zero positive support on the subset still contributes '
                 'through its residual e_c = -m_c(H). Support is recorded, not '
                 'rebalanced, stratified or padded.'),
    }
