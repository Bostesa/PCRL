"""Counterfactual Data Augmentation + pair-invariance loss (Component 3).

The LabHC/bias_in_bios HF release ships only the gendered ``hard_text`` —
there is no pre-scrubbed field. We construct the counterfactual ``x'`` by
deterministic pronoun + honorific swap (Zmigrod 2019, arXiv:1906.04571).

The pair-invariance loss then penalizes ``||h(x) - h(x')||²`` on pooled
[CLS], teaching the encoder to produce equivalent representations whether
the surface form refers to a male or female subject — i.e. gender is no
longer a discriminative feature.

We do NOT swap names. BIOS bios begin with a (often gendered) first name;
swap-via-name-database is brittle and out of scope. The pronoun + honorific
swap covers the dominant gender signal; name-leak will be picked up by the
LEACE-projected residual stream (Component 2) instead.

Refs: Zmigrod CDA arXiv:1906.04571, Moyer invariance arXiv:1805.09458.
"""
from __future__ import annotations

import re

import torch


# Bidirectional gender-pronoun + honorific swap table.
# Single-pass regex with a callback dispatches via the dict so each token is
# matched once and swapped once — no risk of double-swapping (he → she → he).
# "her" is ambiguous (possessive vs object); we map "her" → "his" (possessive
# direction) and "him" → "her" (consistent with possessive). Imperfect on
# object-her cases like "I saw her" → "I saw his", but consistent enough for
# a representation-invariance constraint, which only needs the *distribution*
# of gendered tokens to flip — not perfect grammaticality.
_PRONOUN_SWAP: dict[str, str] = {
    "he": "she", "she": "he",
    "He": "She", "She": "He",
    "HE": "SHE", "SHE": "HE",
    "his": "her", "her": "his",
    "His": "Her", "Her": "His",
    "HIS": "HER", "HER": "HIS",
    "him": "her",
    "Him": "Her",
    "HIM": "HER",
    "himself": "herself", "herself": "himself",
    "Himself": "Herself", "Herself": "Himself",
    "hers": "his",
    "Hers": "His",
    "mr": "ms", "ms": "mr",
    "Mr": "Ms", "Ms": "Mr",
    "MR": "MS", "MS": "MR",
    "mr.": "ms.", "ms.": "mr.",
    "Mr.": "Ms.", "Ms.": "Mr.",
    "mrs": "mr", "mrs.": "mr.",
    "Mrs": "Mr", "Mrs.": "Mr.",
    "MRS": "MR", "MRS.": "MR.",
}

# Compile once. Word-boundary on both sides; honorifics with periods need
# special handling — we sort longest-first so "Mrs." matches before "Mrs",
# and the period is consumed in the match.
_swap_pattern = re.compile(
    r"\b(" + "|".join(
        re.escape(k) for k in sorted(_PRONOUN_SWAP, key=len, reverse=True)
    ) + r")(?=\b|\.)"
)


def swap_gender_pronouns(text: str) -> str:
    """Apply the deterministic pronoun + honorific swap.

    Idempotent under double-application: ``swap(swap(x)) == x`` for inputs
    whose only gendered tokens are in the table.
    """
    return _swap_pattern.sub(lambda m: _PRONOUN_SWAP[m.group(0)], text)


def pair_invariance_loss(
    cls_x: torch.Tensor,
    cls_x_prime: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """λ-free MSE between paired [CLS] vectors.

    Args:
        cls_x: ``(B, d)`` original-bio [CLS].
        cls_x_prime: ``(B, d)`` scrubbed-bio [CLS].
        reduction: ``"mean"`` averages over the batch and feature dims;
            ``"sum"`` sums over both. ``"mean"`` matches the convention in
            the spec: ``λ_inv · ||h(x) - h(x')||²``.

    Returns:
        Scalar tensor on the same device as the inputs. Weight by ``λ_inv``
        at the call site so the optimizer can tune the constraint strength.
    """
    if cls_x.shape != cls_x_prime.shape:
        raise ValueError(
            f"Shape mismatch: cls_x={tuple(cls_x.shape)} "
            f"cls_x_prime={tuple(cls_x_prime.shape)}"
        )
    sq = (cls_x - cls_x_prime).pow(2)
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    raise ValueError(f"unknown reduction {reduction!r}")
