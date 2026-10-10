"""Independent historical P* public-bank and H-CLASS-TEMP arithmetic."""

from __future__ import annotations

import numpy as np

from .independent_a_source import sha_array


def check_pstar_bank(bank: dict, policy: dict, fine: dict, decoder: dict,
                     binding: dict) -> dict:
    """Recompute every public vector from the fixed policy and class alpha."""
    probability = np.asarray(bank["probability"])
    ids = np.asarray(bank["token_id"])
    classes = np.asarray(bank["token_class"])
    input_classes = np.asarray(bank["input_class"])
    mapping = np.asarray(bank["fine_to_hcal_token"])
    width = int(fine["K"])
    if (probability.dtype != np.float64 or probability.shape != (len(ids), width)
            or any(v.dtype != np.int64 or v.ndim != 1 for v in
                   (ids, classes, input_classes, mapping))
            or len(ids) != len(classes) or len(input_classes) != len(mapping)
            or not np.array_equal(ids, np.arange(len(ids)))
            or not np.array_equal(mapping, np.asarray(policy["cell_token"], dtype=np.int64))
            or not np.array_equal(classes, np.asarray(policy["token_class"], dtype=np.int64))
            or not np.array_equal(input_classes, np.asarray(fine["cell_class"], dtype=np.int64))
            or not np.array_equal(input_classes, classes[mapping])
            or not np.array_equal(np.asarray(policy["fine"]["centroid"]),
                                  np.asarray(fine["centroid"]))
            or not np.array_equal(np.asarray(policy["fine"]["cell_class"]),
                                  input_classes)):
        raise ValueError("P* fixed fine/policy/bank identity differs")
    if (decoder["kind"] != "H-CLASS-TEMP" or decoder["target"] != "tokens"
            or int(decoder["T"]) != len(ids) or int(decoder["K"]) != width
            or not np.array_equal(probability, np.asarray(decoder["q"], dtype=np.float64))
            or not np.array_equal(classes, np.asarray(decoder["token_class"], dtype=np.int64))):
        raise ValueError("P* calibrated decoder record differs")
    alpha = np.asarray(decoder["alphas"], dtype=np.float64)
    proto = np.asarray(policy["token_proto"], dtype=np.float64)
    if (alpha.shape != (width,) or proto.shape != probability.shape
            or not np.isfinite(alpha).all() or np.any(alpha <= 0)
            or not np.isfinite(proto).all() or np.any(proto <= 0)):
        raise ValueError("P* temperature input dimensions invalid")
    logits = alpha[classes, None] * np.log(proto)
    z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    decoded = z / np.sum(z, axis=1, keepdims=True)
    difference = float(np.max(np.abs(decoded - probability)))
    if (difference > 2e-15 or not np.array_equal(np.argmax(probability, axis=1), classes)
            or np.any(probability < 0) or not np.isfinite(probability).all()
            or np.max(np.abs(np.sum(probability, axis=1) - 1)) > 1e-8):
        raise ValueError("P* class-temperature public vectors invalid")
    for field, bound_name in (("probability", "bank_probability_sha256"),
                              ("token_class", "token_class_sha256"),
                              ("input_class", "input_class_sha256"),
                              ("fine_to_hcal_token", "fine_to_hcal_token_sha256")):
        if sha_array(bank[field]) != binding[bound_name]:
            raise ValueError(f"P* public action hash differs: {field}")
    if len({(probability[i].tobytes(), int(classes[i])) for i in range(len(ids))}) != len(ids):
        raise ValueError("P* two public IDs name one wire symbol")
    return {"public_actions": int(len(ids)), "input_fine_cells": int(len(mapping)),
            "unique_public_actions": int(len(ids)), "max_temperature_difference": difference,
            "alphas": alpha.tolist()}
