"""Lossless public-code compilation of an admitted historical Q/FINE bank.

This is a representability witness only. The fitted score router and decoder
law stay historical; this module never retrains or searches bank parameters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from qpc import release as QREL


def tokens_to_bits(tokens: np.ndarray, width: int) -> np.ndarray:
    if width not in (10, 16):
        raise ValueError("only predeclared ten/sixteen-bit codes admitted")
    value = np.asarray(tokens)
    if (value.ndim != 1 or not np.issubdtype(value.dtype, np.integer)
            or np.any(value < 0) or np.any(value >= (1 << width))):
        raise ValueError("token is outside global public-code capacity")
    shift = np.arange(width, dtype=np.int64)
    return ((value.astype(np.int64)[:, None] >> shift) & 1).astype(np.uint8)


def bits_to_tokens(code: np.ndarray) -> np.ndarray:
    value = np.asarray(code)
    if (value.ndim != 2 or value.shape[1] not in (10, 16)
            or not np.isin(value, (0, 1)).all()):
        raise ValueError("hard public code must have ten/sixteen binary coordinates")
    return (value.astype(np.int64) *
            (1 << np.arange(value.shape[1], dtype=np.int64))).sum(axis=1)


def compile_release(tokens: np.ndarray, decoder_table: np.ndarray,
                    width: int) -> dict[str, np.ndarray]:
    table = np.asarray(decoder_table)
    value = np.asarray(tokens)
    if (table.ndim != 2 or table.shape[1] not in (2, 6) or len(table) < 1
            or not np.issubdtype(table.dtype, np.floating)
            or not np.isfinite(table).all() or np.any(table < 0)
            or not np.allclose(table.sum(axis=1), 1, rtol=0, atol=1e-10)):
        raise ValueError("fixed public decoder table invalid")
    if (value.ndim != 1 or not np.issubdtype(value.dtype, np.integer)
            or np.any(value < 0) or np.any(value >= len(table))):
        raise ValueError("token outside decoder table")
    code = tokens_to_bits(value, width)
    probabilities = table[value.astype(np.int64)].copy()
    return {"code": code, "probabilities": probabilities,
            "decision": probabilities.argmax(axis=1).astype(np.int64)}


def decode_from_code(code: np.ndarray, decoder_table: np.ndarray) -> dict[str, np.ndarray]:
    """Recipient computation from the sole per-person public identifier."""
    tokens = bits_to_tokens(code)
    out = compile_release(tokens, decoder_table, np.asarray(code).shape[1])
    return {"probabilities": out["probabilities"], "decision": out["decision"]}


def decoder_capacity(decoder_table: np.ndarray, token_class: np.ndarray,
                     width: int) -> dict:
    """Count globally distinct token states; decisions are already in their IDs."""
    table = np.asarray(decoder_table)
    classes = np.asarray(token_class)
    if (table.ndim != 2 or table.shape[1] not in (2, 6)
            or len(table) < 1 or len(table) > (1 << width)
            or classes.shape != (len(table),)
            or not np.issubdtype(classes.dtype, np.integer)
            or not np.isfinite(table).all() or np.any(table < 0)
            or not np.allclose(table.sum(axis=1), 1, rtol=0, atol=1e-10)
            or not np.array_equal(table.argmax(axis=1), classes)):
        raise ValueError("decoder states or classes do not fit public code")
    return {"global_states": int(len(table)),
            "capacity_bits_required": int((len(table) - 1).bit_length()),
            "class_state_counts": np.bincount(classes,
                                               minlength=table.shape[1]).astype(int).tolist(),
            "token_id_includes_class": True,
            "probability_and_decision_are_table_functions": True}


def admit_policy_pair(policy_path: Path, *, expected_sha256: str,
                      source_sha256: str, schema_sha256: str,
                      width: int = 10) -> tuple[QREL.PolicyPair, dict]:
    """Verify a frozen bank without opening any per-person array or label."""
    blob = Path(policy_path).read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected_sha256 or len(source_sha256) != 64 or len(schema_sha256) != 64:
        raise ValueError("policy/source/schema byte pin differs")
    pair = QREL.PolicyPair.from_dict(json.loads(blob))
    binding = QREL.check_bound(pair)
    if (binding["teacher_model_sha256"] != source_sha256
            or binding["feature_names_sha256"] != schema_sha256):
        raise ValueError("policy teacher or 83-column schema binding differs")
    recipients = []
    for purpose, policy in enumerate((pair.p1, pair.p2)):
        if (policy.recipient != purpose + 1
                or policy.K != (2, 6)[purpose]
                or policy.T > (1 << width)
                or not np.array_equal(np.asarray(policy.token_class),
                                      np.asarray(policy.token_proto).argmax(axis=1))):
            raise ValueError("bank alphabet, task classes or decisions incompatible")
        token = np.arange(policy.T, dtype=np.int64)
        code = tokens_to_bits(token, width)
        if not np.array_equal(bits_to_tokens(code), token):
            raise ValueError("bank token compilation is not lossless")
        recipients.append({"purpose": purpose, **decoder_capacity(
            policy.token_proto, policy.token_class, width)})
    return pair, {"schema": "code-utility-bank-static-admission-v1",
                  "policy_sha256": digest, "policy_fingerprint": pair.fingerprint(),
                  "source_model_sha256": source_sha256,
                  "source_feature_schema_sha256": schema_sha256,
                  "public_bit_width": width,
                  "recipients": recipients,
                  "real_role_utility_status": "NOT_YET_EVALUATED",
                  "scope": "frozen policy and public table only; no per-person arrays or labels"}


def route_scores(pair: QREL.PolicyPair, purpose: int,
                 source_probabilities: np.ndarray, width: int = 10,
                 decoder_table: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Historical score router followed by lossless public bit compilation."""
    if purpose not in (0, 1):
        raise ValueError("purpose must be income=0 or occupation=1")
    policy = (pair.p1, pair.p2)[purpose]
    token, probability, decision = QREL.encode(policy, source_probabilities)
    table = policy.token_proto if decoder_table is None else np.asarray(decoder_table)
    decoder_capacity(table, policy.token_class, width)
    out = compile_release(token, table, width)
    if (decoder_table is None and not np.array_equal(out["probabilities"], probability)
            or not np.array_equal(out["decision"], decision)):
        raise AssertionError("compiled public release differs from historical policy")
    return out
