"""Lossless row-wise disclosure adapter for Team B's feature-track attacks.

This module does not fit or select an attacker. It flattens *every* transmitted
field of ``ReleaseAdapter`` into named numeric/categorical fields for S2's
common attacker preprocessing. Private provider keys determine row alignment
but never enter returned records.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

CLASSES = {"income": 2, "occupation": 6}
REQUIRED_VIEW_FIELDS = frozenset({"features", "logits", "probabilities", "decision", "metadata"})


def _flatten(prefix: str, value: object, output: dict[str, float | str]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ValueError("Transmitted mapping keys must be nonempty strings")
            _flatten(f"{prefix}.{key}", value[key], output)
    elif isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.ndim != 1:
            raise ValueError(f"Transmitted array must be one-dimensional: {prefix}")
        for index, item in enumerate(array.tolist()):
            _flatten(f"{prefix}[{index}]", item, output)
    elif isinstance(value, (bool, np.bool_)):
        output[prefix] = float(value)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Nonfinite transmitted value: {prefix}")
        output[prefix] = numeric
    elif isinstance(value, str):
        output[prefix] = value
    elif value is None:
        output[prefix] = "<NULL>"
    else:
        raise TypeError(f"Unsupported transmitted value type at {prefix}: {type(value).__name__}")


def flatten_view(purpose: str, view: Mapping[str, object]) -> dict[str, float | str]:
    """Flatten the complete observed recipient view, including metadata."""
    if purpose not in CLASSES:
        raise ValueError("Unknown purpose")
    if not isinstance(view, Mapping) or not REQUIRED_VIEW_FIELDS.issubset(view):
        raise ValueError("Missing transmitted recipient field")
    n_classes = CLASSES[purpose]
    z = np.asarray(view["features"], dtype=np.float64)
    logits = np.asarray(view["logits"], dtype=np.float64)
    probs = np.asarray(view["probabilities"], dtype=np.float64)
    if z.shape != (16,) or logits.shape != (n_classes,) or probs.shape != (n_classes,):
        raise ValueError("Wrong disclosed feature/logit/probability dimensions")
    if not all(np.isfinite(a).all() for a in (z, logits, probs)):
        raise ValueError("Nonfinite disclosed numeric field")
    if np.any(probs < 0) or not np.isclose(float(probs.sum()), 1.0, atol=1e-6):
        raise ValueError("Invalid disclosed probability vector")
    decision = view["decision"]
    if not isinstance(decision, (int, np.integer)) or int(decision) != int(probs.argmax()):
        raise ValueError("Decision must equal the disclosed probability argmax")
    if not isinstance(view["metadata"], Mapping):
        raise ValueError("Metadata must be a mapping")
    output: dict[str, float | str] = {}
    # Walk the entire supplied view rather than an allowlist, so an added
    # fallback flag or length cannot silently disappear from an attack.
    for key in sorted(view):
        if not isinstance(key, str) or not key:
            raise ValueError("Transmitted view keys must be nonempty strings")
        _flatten(key, view[key], output)
    # Decisions are categorical. Include a fixed one-hot representation as a
    # reader convenience; the original integer is still present above.
    for cls in range(n_classes):
        output[f"decision_onehot[{cls}]"] = float(cls == int(decision))
    return output


def aligned_complete_views(
    rows: Sequence[tuple[str, Mapping[str, Mapping[str, object]]]],
) -> dict[str, list[dict[str, float | str]]]:
    """Build both local and aligned pair rows without disclosing provider keys."""
    if not rows:
        raise ValueError("At least one aligned person is required")
    if len({key for key, _ in rows}) != len(rows):
        raise ValueError("Duplicate private provider key")
    result: dict[str, list[dict[str, float | str]]] = {
        "income": [], "occupation": [], "pair": [],
    }
    for person_key, recipient_views in rows:
        if not isinstance(person_key, str) or not person_key:
            raise ValueError("A private alignment key is required")
        if set(recipient_views) != set(CLASSES):
            raise ValueError("Both recipient views are required for each person")
        local = {purpose: flatten_view(purpose, recipient_views[purpose]) for purpose in CLASSES}
        result["income"].append(local["income"])
        result["occupation"].append(local["occupation"])
        pair = {f"income.{name}": value for name, value in local["income"].items()}
        pair.update({f"occupation.{name}": value for name, value in local["occupation"].items()})
        result["pair"].append(pair)
    return result
