"""Public-only defense-aware columns for corrected continuous references.

The original deterministic release and its complete wire remain unchanged.
This adapter is used only by the dated U0/Ucal retry worker.
"""

from dataclasses import replace

import numpy as np

from .reference_wire import ReferenceRoleRelease, reference_recipient_batch


def corrected_continuous_recipient_batch(
        package: ReferenceRoleRelease, recipient: int):
    if package.receipt.get("interface_kind") != "continuous":
        raise ValueError("corrected view requires a continuous reference")
    batch = reference_recipient_batch(package, recipient)
    if batch.code_fields or "code_id" in batch.fields or batch.public_derived:
        raise ValueError("continuous reference has unexpected public fields")
    q = np.asarray(batch.fields["probability"].values)
    if q.dtype != np.float64 or q.ndim != 2 or not np.isfinite(q).all():
        raise ValueError("continuous public probability differs")
    return replace(batch, public_derived={
        "log_probabilities": np.log(np.maximum(q, 1e-12)),
    })
