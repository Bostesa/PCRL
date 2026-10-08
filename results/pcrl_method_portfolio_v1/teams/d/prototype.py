"""Team D's post-hoc frozen-latent product-VQ release adapter.

This module contains no Adult loader and never reads task or sensitive labels.
Only a purpose's own frozen latent is accepted at deployment. Codebook fitting,
privacy objectives, calibrated head fitting, and endpoint evaluation remain
subject to the campaign lock and an approved role-specific data source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


def _latent(x: np.ndarray, d: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != d or not np.isfinite(a).all():
        raise ValueError(f"expected finite latent array with shape (n, {d})")
    return a


@dataclass(frozen=True)
class PurposeNormalizer:
    """Public fit-only per-purpose affine preprocessing; never SEX-routed."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if mean.ndim != 2 or mean.shape[0] != 2 or mean.shape != scale.shape:
            raise ValueError("mean and scale must have shape (2, latent_dim)")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("normalization parameters must be finite with positive scales")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "scale", scale.copy())

    @classmethod
    def fit(cls, latents: tuple[np.ndarray, np.ndarray]) -> "PurposeNormalizer":
        if len(latents) != 2:
            raise ValueError("exactly two purpose latent arrays are required")
        a = np.asarray(latents[0], dtype=np.float64)
        b = np.asarray(latents[1], dtype=np.float64)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not a.shape[0] or not b.shape[0]:
            raise ValueError("nonempty purpose arrays must have the same latent dimension")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("fit latents must be finite")
        mean = np.stack((a.mean(axis=0), b.mean(axis=0)))
        scale = np.stack((a.std(axis=0), b.std(axis=0)))
        scale = np.maximum(scale, 1e-6)
        return cls(mean, scale)

    @property
    def dim(self) -> int:
        return int(self.mean.shape[1])

    def transform(self, purpose: int, latent: np.ndarray) -> np.ndarray:
        if purpose not in (0, 1):
            raise ValueError("purpose must be 0 or 1")
        return (_latent(latent, self.dim) - self.mean[purpose]) / self.scale[purpose]

    def inverse(self, purpose: int, latent: np.ndarray) -> np.ndarray:
        if purpose not in (0, 1):
            raise ValueError("purpose must be 0 or 1")
        return _latent(latent, self.dim) * self.scale[purpose] + self.mean[purpose]


class ProductVQ:
    """Shared hard nearest-centroid map on normalized purpose latents.

    All blocks have the same size and the same number of centroids. Code IDs
    are transmitted as an integer tuple; ties use the first centroid index.
    """

    def __init__(self, codebooks: np.ndarray, normalizer: PurposeNormalizer):
        cb = np.asarray(codebooks, dtype=np.float64)
        if cb.ndim != 3 or min(cb.shape) < 1 or not np.isfinite(cb).all():
            raise ValueError("codebooks must be finite (blocks, codes, block_dim)")
        if cb.shape[0] * cb.shape[2] != normalizer.dim:
            raise ValueError("codebooks and normalizer have incompatible latent dimensions")
        if cb.shape[1] & (cb.shape[1] - 1):
            raise ValueError("codes per block must be a power of two")
        self.codebooks = cb.copy()
        self.normalizer = normalizer

    @property
    def code_length_bits(self) -> int:
        return self.codebooks.shape[0] * (self.codebooks.shape[1].bit_length() - 1)

    def encode(self, purpose: int, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        norm = self.normalizer.transform(purpose, latent)
        blocks, _, block_dim = self.codebooks.shape
        ids = np.empty((len(norm), blocks), dtype=np.int32)
        decoded = np.empty_like(norm)
        for block in range(blocks):
            sl = slice(block * block_dim, (block + 1) * block_dim)
            centers = self.codebooks[block]
            # Bounded temporary allocation; argmin resolves exact ties by index.
            for start in range(0, len(norm), 2048):
                stop = min(start + 2048, len(norm))
                distance = np.sum((norm[start:stop, None, sl] - centers[None, :, :]) ** 2, axis=2)
                ids[start:stop, block] = np.argmin(distance, axis=1)
            decoded[:, sl] = centers[ids[:, block]]
        # The public codebook and ID determine this shared normalized vector.
        # A head may not receive a clean or inverse-routed teacher output.
        return ids, decoded


@dataclass(frozen=True)
class SoftmaxHead:
    """Linear synthetic fixture head; the locked candidate uses MLPHead."""

    weight: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        w = np.asarray(self.weight, dtype=np.float64)
        b = np.asarray(self.bias, dtype=np.float64)
        if w.ndim != 2 or b.shape != (w.shape[0],) or w.shape[0] < 2:
            raise ValueError("head requires (classes, dim) weight and (classes,) bias")
        if not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError("head parameters must be finite")
        object.__setattr__(self, "weight", w.copy())
        object.__setattr__(self, "bias", b.copy())

    @property
    def input_dim(self) -> int:
        return int(self.weight.shape[1])

    def probabilities(self, decoded: np.ndarray) -> np.ndarray:
        x = _latent(decoded, self.weight.shape[1])
        logits = x @ self.weight.T + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class MLPHead:
    """Public protected head: Linear(d,hidden), exact GELU, Linear(hidden,K)."""

    weight1: np.ndarray
    bias1: np.ndarray
    weight2: np.ndarray
    bias2: np.ndarray

    def __post_init__(self) -> None:
        w1 = np.asarray(self.weight1, dtype=np.float64)
        b1 = np.asarray(self.bias1, dtype=np.float64)
        w2 = np.asarray(self.weight2, dtype=np.float64)
        b2 = np.asarray(self.bias2, dtype=np.float64)
        if (w1.ndim != 2 or w2.ndim != 2 or b1.shape != (w1.shape[0],)
                or w2.shape[1] != w1.shape[0] or b2.shape != (w2.shape[0],)
                or w2.shape[0] < 2):
            raise ValueError("incompatible MLP head weights and biases")
        if not all(np.isfinite(a).all() for a in (w1, b1, w2, b2)):
            raise ValueError("MLP head parameters must be finite")
        for name, value in (("weight1", w1), ("bias1", b1), ("weight2", w2), ("bias2", b2)):
            object.__setattr__(self, name, value.copy())

    @property
    def input_dim(self) -> int:
        return int(self.weight1.shape[1])

    def probabilities(self, decoded: np.ndarray) -> np.ndarray:
        x = _latent(decoded, self.input_dim)
        hidden = x @ self.weight1.T + self.bias1
        hidden = hidden * ndtr(hidden)
        logits = hidden @ self.weight2.T + self.bias2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)


class FrozenLatentRelease:
    """Purpose-local deployment: VQ tuple, decoded vector, protected head output."""

    def __init__(self, vq: ProductVQ, heads: tuple[SoftmaxHead | MLPHead, SoftmaxHead | MLPHead]):
        if len(heads) != 2 or any(h.input_dim != vq.normalizer.dim for h in heads):
            raise ValueError("one compatible protected head per purpose is required")
        self.vq = vq
        self.heads = heads

    def recipient(self, purpose: int, latent: np.ndarray) -> dict:
        if purpose not in (0, 1):
            raise ValueError("purpose must be 0 or 1")
        ids, decoded = self.vq.encode(purpose, latent)
        probabilities = self.heads[purpose].probabilities(decoded)
        return {
            "code_ids": ids,
            "decoded": decoded,
            "probabilities": probabilities,
            "decision": np.argmax(probabilities, axis=1),
            "metadata": {
                "purpose": purpose,
                "code_length_bits": self.vq.code_length_bits,
                "block_count": int(self.vq.codebooks.shape[0]),
                "fallback": False,
                "stochastic": False,
            },
        }


def _codes(codes: np.ndarray) -> np.ndarray:
    a = np.asarray(codes)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0):
        raise ValueError("code IDs must be a nonnegative integer matrix")
    return a.astype(np.int64, copy=False)


def occupancy_report(ids0: np.ndarray, ids1: np.ndarray) -> dict:
    """Aggregate code occupancy for each view and the aligned coalition."""
    a, b = _codes(ids0), _codes(ids1)
    if a.shape[0] != b.shape[0] or not len(a):
        raise ValueError("aligned, nonempty recipient code arrays are required")

    def one(codes: np.ndarray) -> dict:
        _, counts = np.unique(codes, axis=0, return_counts=True)
        singletons = int(np.sum(counts == 1))
        return {
            "rows": int(len(codes)),
            "distinct_tuples": int(len(counts)),
            "singleton_rows": singletons,
            "singleton_fraction": singletons / len(codes),
            "max_occupancy": int(counts.max()),
        }

    return {
        "recipient_0": one(a),
        "recipient_1": one(b),
        "coalition": one(np.column_stack((a, b))),
    }


def tuple_lookup_scores(
    fit_codes: np.ndarray,
    fit_sex: np.ndarray,
    query_codes: np.ndarray,
    smoothing: float = 1.0,
) -> np.ndarray:
    """Defense-aware full-code lookup diagnostic fitted on an allowed fit role.

    An unseen tuple receives the fit prevalence. For a seen tuple, the
    smoothing mass also uses that prevalence. This is an attack diagnostic,
    never a certified information bound or an inference-time defense input.
    """
    fit, query = _codes(fit_codes), _codes(query_codes)
    s = np.asarray(fit_sex)
    if fit.shape[0] == 0 or fit.shape[1] != query.shape[1] or s.shape != (len(fit),):
        raise ValueError("fitting codes, labels, and query dimensions disagree")
    if not np.isin(s, (0, 1)).all() or not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("binary fitting labels and positive smoothing required")
    prior = float(np.mean(s))
    counts: dict[tuple[int, ...], list[int]] = {}
    for row, label in zip(fit, s):
        entry = counts.setdefault(tuple(row), [0, 0])
        entry[0] += int(label)
        entry[1] += 1
    return np.asarray(
        [(counts[tuple(row)][0] + smoothing * prior) / (counts[tuple(row)][1] + smoothing)
         if tuple(row) in counts else prior for row in query],
        dtype=np.float64,
    )
