"""Public-only defense-aware P* views for the H2 technical retry.

The original finite release wire is unchanged. Every added numerical column
is a deterministic function of its transmitted probability or code bits.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import (
    WireBatch, pair_batch, prepare_views,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.reference_wire import (
    ReferenceRoleRelease, prepare_reference_three_views,
    reference_recipient_batch,
)


def _local(batch: WireBatch) -> WireBatch:
    if batch.code_fields != ("code_id",) or batch.public_derived:
        raise ValueError("P* finite source view/code identity differs")
    probability = np.asarray(batch.fields["probability"].values)
    bits = np.asarray(batch.fields["code_bits"].values)
    if (probability.dtype != np.float64 or probability.ndim != 2 or
            not np.isfinite(probability).all() or np.any(probability < 0) or
            not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-8) or
            bits.ndim != 2 or bits.shape[0] != len(probability) or
            not np.isin(bits, (0, 1)).all()):
        raise ValueError("P* public probability/code bits differ")
    return replace(batch, public_derived={
        "public_log_probability": np.log(np.maximum(probability, 1e-12))})


def _pair(left: WireBatch, right: WireBatch) -> WireBatch:
    paired = pair_batch(left, right)
    bits1 = np.asarray(left.fields["code_bits"].values, dtype=np.float64)
    bits2 = np.asarray(right.fields["code_bits"].values, dtype=np.float64)
    outer = (bits1[:, :, None] * bits2[:, None, :]).reshape(
        len(bits1), bits1.shape[1] * bits2.shape[1])
    return replace(paired, public_derived={
        **paired.public_derived, "pair_code_bit_products": outer})


def prepare_pstar_three_views_h2(att: ReferenceRoleRelease,
                                 inner: ReferenceRoleRelease) -> dict:
    """Retain inherited identity validation and add public DA columns."""
    if (att.receipt.get("interface_kind") != "finite" or
            inner.receipt.get("interface_kind") != "finite" or
            att.receipt.get("arm_id") != "P_STAR" or
            inner.receipt.get("arm_id") != "P_STAR"):
        raise ValueError("H2 adapter accepts only the exact finite P* reference")
    # The predecessor checks both role receipts, release IDs and full wire
    # schemas. Its returned views have no DA columns and are never scored.
    prepare_reference_three_views(att, inner)
    batches = {}
    for name, package in (("att", att), ("inner", inner)):
        left = _local(reference_recipient_batch(package, 1))
        right = _local(reference_recipient_batch(package, 2))
        batches[name] = {"r1": left, "r2": right, "pair": _pair(left, right)}
    prepared = {view: prepare_views(batches["att"][view], batches["inner"][view])
                for view in ("r1", "r2", "pair")}
    for view in ("r1", "r2", "pair"):
        for role in ("att", "inner"):
            if getattr(prepared[view], role).x_defense_aware is None:
                raise AssertionError("H2 public defense-aware view was not prepared")
    return prepared
