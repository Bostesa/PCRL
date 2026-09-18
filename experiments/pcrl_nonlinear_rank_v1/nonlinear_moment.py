"""Nonlinear moment refinement: a finite objective, not a privacy certificate.

The historical residual spectral baseline penalises a finite set of *first*
moments of the released channel ``Z = V W``, linear in ``Z`` over a degree-two
polynomial basis in the released service probabilities. Zero first moments are
compatible with perfect nonlinear recovery from ``Z`` (see the magnitude and XOR
fixtures in :mod:`tests.pcrl_nonlinear_rank_v1`). This module adds a penalty on
*nonlinear* functions of the actual released ``Z``, still conditioned on the
actual released service probabilities available to that recipient.

Scope. Every quantity here is a finite empirical moment of a fixed feature
family. Nothing in this module establishes marginal privacy, conditional
privacy, a conditional-independence test with calibrated level, generalisation,
or a guarantee against a nonlinear downstream attacker. The refined objective is
generally nonconvex: the global-optimality statement of the original baseline
applies only to its fixed linear-moment matrix and is *not* inherited.

Attribution. The ingredients are standard and separately attributable: random
Fourier features (Rahimi & Recht), residualised conditional moment criteria in
the spirit of kernel conditional-independence testing (Zhang et al. KCI 2012;
Strobl et al. RFF conditional tests 2017), spectral/closed-form adversarial
representation objectives (Sadeghi et al. 2019, 2021), and Stiefel-manifold
optimisation with QR retraction (Edelman/Absil et al.). The combination below is
an engineering adaptation named ``nonlinear_moment_refinement``; it is not a
reproduction of any of those methods and proves no new theorem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np

# ------------------------------------------------------------------ prospective constants
RFF_PER_BAND = 32
BANDS = (0.5, 1.0, 2.0)
RFF_TOTAL = RFF_PER_BAND * len(BANDS)
BANDWIDTH_SUBSET_MAX = 1024
NORMALIZER_FLOOR = 1e-12
BLOCK_WEIGHT = 0.5          # quadratic block / Fourier block, equal weight
COMBINE_WEIGHT = 0.5        # original linear-moment term / nonlinear term, equal weight
FEATURE_SCALE_FLOOR = 1e-12
DEFAULT_CHUNK = 4096

SUBSET_SEED_BASE = 20260918
OMEGA_SEED_BASE = 20260919
PERTURB_SEED_BASE = 20260920
PERTURB_MAGNITUDE = 0.05

# Optimiser (frozen after the synthetic checks and one training-only runtime calibration)
MAX_UPDATES = 200
INITIAL_STEP = 1.0
SHRINK = 0.5
MAX_BACKTRACK = 20
MIN_IMPROVEMENT = 1e-14
MIN_GRAD_NORM = 1e-12


def _seed(base: int, seed: int, r: int) -> int:
    return base + 100 * seed + r


def quadratic_pairs(r: int):
    """Degree-two monomial index pairs ``(a, b)`` with ``a <= b``, lexicographic."""
    return list(combinations_with_replacement(range(r), 2))


# ------------------------------------------------------------------ feature family
@dataclass
class ChiFeatures:
    """Frozen nonlinear feature family ``chi_r(Z)`` = [degree-2 monomials, RFF].

    ``omega_scaled`` already carries the per-band bandwidth division, so the
    Fourier argument is ``Z @ omega_scaled + phase``. Everything here is fitted
    once at the *reference* projection and is independent of the optimised ``W``:
    no bandwidth or feature-scale recalculation depending on the current ``W``.
    """

    r: int
    pairs: list
    omega_scaled: np.ndarray      # (r, 96)
    phase: np.ndarray             # (96,)
    amplitude: float              # sqrt(2 / RFF_PER_BAND)
    mean: np.ndarray              # (F,) frozen reference mean
    scale: np.ndarray             # (F,) frozen reference scale, floored
    sigma_reference: float
    subset_indices: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        """Cache the monomial index arrays and the two scatter matrices.

        ``select_a``/``select_b`` turn the quadratic-block backpropagation into two
        dense ``(n, n_quadratic) @ (n_quadratic, r)`` products instead of a Python
        loop over pairs. Adding both contributions unconditionally is exactly right
        on the diagonal too: for ``a == b`` the two terms coincide and sum to
        ``2 * g * z_a``, which is the correct derivative of ``z_a**2``.
        """
        r = self.r
        self._a = np.fromiter((p[0] for p in self.pairs), dtype=np.intp, count=len(self.pairs))
        self._b = np.fromiter((p[1] for p in self.pairs), dtype=np.intp, count=len(self.pairs))
        self._select_a = np.zeros((len(self.pairs), r))
        self._select_b = np.zeros((len(self.pairs), r))
        self._select_a[np.arange(len(self.pairs)), self._a] = 1.0
        self._select_b[np.arange(len(self.pairs)), self._b] = 1.0

    @property
    def n_quadratic(self) -> int:
        return len(self.pairs)

    @property
    def width(self) -> int:
        return self.n_quadratic + RFF_TOTAL

    # -------------------------------------------------------------- construction
    @classmethod
    def fit(cls, z_reference: np.ndarray, seed: int) -> "ChiFeatures":
        """Fit bandwidth, Fourier directions/phases and feature scaling at ``z_reference``."""
        z = np.asarray(z_reference, dtype=np.float64)
        if z.ndim != 2 or not np.isfinite(z).all():
            raise ValueError('reference projection must be finite two-dimensional')
        n, r = z.shape
        pairs = quadratic_pairs(r)

        subset_rng = np.random.default_rng(_seed(SUBSET_SEED_BASE, seed, r))
        m = min(BANDWIDTH_SUBSET_MAX, n)
        subset = np.sort(subset_rng.choice(n, m, replace=False))
        from scipy.spatial.distance import pdist
        distances = pdist(z[subset])
        positive = distances[distances > 0]
        if not len(positive):
            raise ValueError('bandwidth diagnostic failure: no positive pairwise distances '
                             'in the reference projection subset')
        sigma = float(np.median(positive))

        rng = np.random.default_rng(_seed(OMEGA_SEED_BASE, seed, r))
        omega = rng.normal(size=(r, RFF_TOTAL))
        phase = rng.uniform(0., 2 * np.pi, RFF_TOTAL)
        scaled = np.empty_like(omega)
        for j, band in enumerate(BANDS):
            block = slice(j * RFF_PER_BAND, (j + 1) * RFF_PER_BAND)
            scaled[:, block] = omega[:, block] / (band * sigma)

        obj = cls(r=r, pairs=pairs, omega_scaled=scaled, phase=phase,
                  amplitude=float(np.sqrt(2.0 / RFF_PER_BAND)),
                  mean=np.zeros(len(pairs) + RFF_TOTAL), scale=np.ones(len(pairs) + RFF_TOTAL),
                  sigma_reference=sigma, subset_indices=subset)
        raw = obj._raw(z[subset])
        mean = raw.mean(0)
        scale = raw.std(0)
        degenerate = scale <= FEATURE_SCALE_FLOOR
        obj.mean = mean
        obj.scale = np.where(degenerate, 1.0, np.maximum(scale, FEATURE_SCALE_FLOOR))
        obj.diagnostics = {
            'rank': int(r), 'n_quadratic': len(pairs), 'n_fourier': RFF_TOTAL,
            'feature_width': obj.width, 'bands': list(BANDS), 'rff_per_band': RFF_PER_BAND,
            'sigma_reference': sigma, 'bandwidth_subset_rows': int(m),
            'bandwidth_subset_seed': _seed(SUBSET_SEED_BASE, seed, r),
            'omega_phase_seed': _seed(OMEGA_SEED_BASE, seed, r),
            'reference_rows': int(n),
            'positive_distance_pairs': int(len(positive)),
            'zero_distance_pairs': int(len(distances) - len(positive)),
            # A constant monomial cannot arise from a whitened reference projection,
            # but a near-constant one would; such columns keep scale 1 and are flagged
            # rather than amplified (same convention as the historical basis code).
            'degenerate_scale_columns': np.flatnonzero(degenerate).tolist(),
            'degenerate_scale_count': int(degenerate.sum()),
            'raw_mean_abs_max': float(np.max(np.abs(mean))),
            'raw_scale_min': float(scale.min()), 'raw_scale_max': float(scale.max()),
        }
        return obj

    # -------------------------------------------------------------- evaluation
    def _raw(self, z: np.ndarray) -> np.ndarray:
        """Unstandardised feature block [quadratic monomials, Fourier features]."""
        quad = z[:, self._a] * z[:, self._b]
        arg = z @ self.omega_scaled + self.phase
        return np.concatenate((quad, self.amplitude * np.cos(arg)), axis=1)

    def transform(self, z: np.ndarray) -> np.ndarray:
        """Standardised ``chi_r(Z)`` using the frozen reference mean and scale."""
        return (self._raw(np.asarray(z, dtype=np.float64)) - self.mean) / self.scale

    def transform_with_state(self, z: np.ndarray):
        """``chi`` plus the intermediates the analytic gradient needs."""
        z = np.asarray(z, dtype=np.float64)
        arg = z @ self.omega_scaled + self.phase
        quad = z[:, self._a] * z[:, self._b]
        raw = np.concatenate((quad, self.amplitude * np.cos(arg)), axis=1)
        return (raw - self.mean) / self.scale, arg

    def backpropagate(self, g_chi: np.ndarray, z: np.ndarray, arg: np.ndarray) -> np.ndarray:
        """Map d(objective)/d(chi_standardised) back to d(objective)/dZ.

        ``g_chi`` is with respect to the *standardised* features, so the frozen
        scale is divided out here once.
        """
        g = g_chi / self.scale
        nq = self.n_quadratic
        # quadratic block: d(z_a z_b)/dz_m = delta_am z_b + delta_bm z_a, scattered by
        # two dense 0/1 selectors (correct on the diagonal, see __post_init__).
        gq = g[:, :nq]
        gz = (gq * z[:, self._b]) @ self._select_a + (gq * z[:, self._a]) @ self._select_b
        # Fourier block: d/dz cos(z.w + p) = -sin(z.w + p) * w
        tmp = -(g[:, nq:] * self.amplitude) * np.sin(arg)
        gz += tmp @ self.omega_scaled.T
        return gz

    def metadata(self) -> dict:
        return {**self.diagnostics,
                'quadratic_pairs': [list(p) for p in self.pairs],
                'amplitude': self.amplitude,
                'feature_mean_sha_input_len': int(self.mean.size)}


# ------------------------------------------------------------------ role moments
@dataclass
class ProtectedRole:
    """One protected (view, attribute) role with frozen nuisance residuals.

    ``basis`` is the training-only polynomial anchor basis ``b_k(H_c)`` for the
    recipient ``c`` (including its intercept, so conditional cancellation such as
    XOR is representable). ``residual`` is ``onehot(S_j) - m_j(H_c)`` from the
    original three-fold household out-of-fold nuisance predictions. Both are
    restricted to the same valid rows as that role in the original baseline.
    """

    name: str
    view: str
    basis: np.ndarray             # (n_j, B)
    residual: np.ndarray          # (n_j, K)
    linear_gram: np.ndarray       # (q, q) trace-normalised historical moment Gram
    n_valid: int
    classes: int
    support: np.ndarray
    unsupported: list
    raw_trace: float
    zero_trace: bool

    @property
    def basis_width(self) -> int:
        return self.basis.shape[1]

    def weights(self) -> np.ndarray:
        """``b_k(H) * e_c`` flattened to (n_j, B*K); the moment's row weights.

        Cached: this is a frozen function of the basis and the frozen nuisance
        residuals, and it is read once per chunk on every objective evaluation.
        """
        cached = getattr(self, '_weights', None)
        if cached is None:
            cached = np.ascontiguousarray(
                (self.basis[:, :, None] * self.residual[:, None, :]).reshape(len(self.basis), -1))
            self._weights = cached
        return cached


@dataclass
class FeatureBundle:
    """``Z``, ``chi_r(Z)`` and the Fourier arguments for one ``W``, on the full row set.

    ``chi_r(VW)`` depends only on ``W``, never on the protected role, so it is
    built once per objective evaluation and shared by every role. Rows are
    processed in chunks; no n-by-n kernel matrix is ever formed.
    """

    z: np.ndarray                 # (n, r)
    chi: np.ndarray               # (n, F) standardised features
    arg: np.ndarray               # (n, 96) Fourier arguments
    chunk: int


def build_features(chi: ChiFeatures, v: np.ndarray, w: np.ndarray,
                   chunk: int = DEFAULT_CHUNK) -> FeatureBundle:
    n = len(v)
    z = np.empty((n, w.shape[1]))
    features = np.empty((n, chi.width))
    arg = np.empty((n, RFF_TOTAL))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = v[start:stop] @ w
        z[start:stop] = block
        features[start:stop], arg[start:stop] = chi.transform_with_state(block)
    return FeatureBundle(z=z, chi=features, arg=arg, chunk=chunk)


def role_moments(role: ProtectedRole, bundle: FeatureBundle, mask: np.ndarray):
    """``M[f, k, c] = mean_i chi(V W)[i, f] * b[i, k] * e[i, c]`` over the role's valid rows.

    The divisor is the role's valid row count ``n_j``, matching the historical
    convention exactly.
    """
    features = bundle.chi if mask is None else bundle.chi[mask]
    if len(features) != role.n_valid:
        raise ValueError('row count does not match the role mask')
    total = features.T @ role.weights()
    total /= max(role.n_valid, 1)
    return total.reshape(bundle.chi.shape[1], role.basis_width, role.classes)


def role_moment_gradient(role: ProtectedRole, chi: ChiFeatures, bundle: FeatureBundle,
                         mask: np.ndarray, block_scale: np.ndarray):
    """Analytic ``d/dZ`` of ``sum_{f,k,c} block_scale[f] * M[f,k,c]^2``.

    Returns the value, the full-row gradient with respect to ``Z`` (so that a
    single ``V' dZ`` product finishes every role at once) and the moment tensor.
    ``block_scale`` lets the quadratic and Fourier blocks carry different fixed
    normalisers.
    """
    m = role_moments(role, bundle, mask)
    f, b, k = m.shape
    denominator = max(role.n_valid, 1)
    scaled = (m * block_scale[:, None, None]).reshape(f, b * k)
    # dD/dchi[i, f] = (2 / n) * sum_{k,c} scaled[f, (k,c)] * weights[i, (k,c)]
    g_chi = (2.0 / denominator) * (role.weights() @ scaled.T)
    if mask is None:
        gz = chi.backpropagate(g_chi, bundle.z, bundle.arg)
    else:
        gz = np.zeros_like(bundle.z)
        gz[mask] = chi.backpropagate(g_chi, bundle.z[mask], bundle.arg[mask])
    value = float(np.sum(scaled.reshape(f, b, k) * m))
    return value, gz, m


def block_squared_norms(m: np.ndarray, chi: ChiFeatures):
    """Separate squared Frobenius norms of the quadratic and Fourier moment blocks."""
    nq = chi.n_quadratic
    return float(np.sum(m[:nq] ** 2)), float(np.sum(m[nq:] ** 2))
