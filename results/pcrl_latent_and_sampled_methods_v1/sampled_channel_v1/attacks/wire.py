"""Complete one-draw score-channel wires for recipient and coalition attacks.

Private ordered row/group IDs align releases and never become attack columns.
Every transmitted field is encoded through R02's common ATT-only view builder.
Public channel-derived features depend on observed code IDs and fixed tables,
never a person's untransmitted score, input cell or random coin.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import (
    Field, WireBatch, pair_batch,
)


SCHEMA = "SCV1-score-release-v1"
WIRE_KEYS = frozenset({"schema", "recipient", "code_id", "probability",
                       "decision", "release_id", "public_metadata"})
CLASSES = {1: 2, 2: 6}
METADATA_KEYS = frozenset({"model_id", "source_binding_sha256",
                           "canonical_bank_sha256", "mechanism_sha256",
                           "code_length_bits", "fallback", "sampling_contract"})
SAMPLING_CONTRACT = "cached_one_joint_draw_v1"


def canonical_bank_sha256(probability: np.ndarray) -> str:
    """Independently reproduce the fixed public bank digest from wire values."""
    p = np.asarray(probability)
    if (p.dtype != np.dtype("float64") or p.ndim != 2 or min(p.shape) < 1
            or not np.isfinite(p).all() or np.any(p < 0)
            or np.signbit(p[p == 0]).any()
            or not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("canonical bank requires finite little-endian float64 simplex rows")
    p = np.ascontiguousarray(p, dtype="<f8")
    decisions = np.argmax(p, axis=1).astype("<i8")
    if any(np.count_nonzero(row == row[winner]) != 1
           for row, winner in zip(p, decisions)):
        raise ValueError("canonical bank decision lacks strict argmax")
    digest = hashlib.sha256(b"SCV1-canonical-bank-v1\n")
    digest.update(json.dumps(list(p.shape), sort_keys=True,
                             separators=(",", ":")).encode() + b"\n")
    digest.update(p.tobytes())
    digest.update(decisions.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PublicBank:
    recipient: int
    probability: np.ndarray
    release_id: str
    public_metadata: Mapping[str, object]
    _metadata_json: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.recipient not in CLASSES or not isinstance(self.release_id, str) or not self.release_id:
            raise ValueError("public bank recipient/release identity is invalid")
        p = np.asarray(self.probability)
        if (p.ndim != 2 or p.shape[0] < 1 or p.shape[1] != CLASSES[self.recipient]
                or p.dtype != np.dtype("float64")
                or not np.isfinite(p).all() or np.any(p < 0)
                or not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-12)):
            raise ValueError("public bank probability rows are invalid")
        bank_sha256 = canonical_bank_sha256(p)
        winners = np.argmax(p, axis=1)
        if any(np.count_nonzero(row == row[winner]) != 1
               for row, winner in zip(p, winners)):
            raise ValueError("public bank decision lacks strict argmax")
        copied = np.ascontiguousarray(p.copy())
        if len({row.tobytes() for row in copied}) != len(copied):
            raise ValueError("duplicate public probability/decision vector in bank")
        copied.setflags(write=False)
        object.__setattr__(self, "probability", copied)
        if not isinstance(self.public_metadata, Mapping):
            raise ValueError("registered public metadata must be a mapping")
        if set(self.public_metadata) != METADATA_KEYS:
            raise ValueError("registered public metadata key set differs from exact contract")
        if self.public_metadata["model_id"] != self.release_id:
            raise ValueError("registered public metadata model_id differs from release_id")
        if (type(self.public_metadata["fallback"]) is not bool
                or self.public_metadata["fallback"] is not False
                or self.public_metadata["sampling_contract"] != SAMPLING_CONTRACT):
            raise ValueError("registered public metadata fallback/sampling contract differs")
        for name in ("source_binding_sha256", "canonical_bank_sha256",
                     "mechanism_sha256"):
            if not isinstance(self.public_metadata[name], str) or re.fullmatch(
                    r"[0-9a-f]{64}", self.public_metadata[name]) is None:
                raise ValueError(f"registered public metadata {name} is not SHA-256")
        if self.public_metadata["canonical_bank_sha256"] != bank_sha256:
            raise ValueError("registered canonical bank SHA differs from public vectors")
        if (type(self.public_metadata["code_length_bits"]) is not int
                or self.public_metadata["code_length_bits"] !=
                (len(self.probability) - 1).bit_length()):
            raise ValueError("registered public code length differs from canonical bank")
        flattened: dict[str, object] = {}
        _metadata_fields(self.public_metadata, "public_metadata", flattened)
        try:
            canonical = json.dumps(self.public_metadata, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("registered public metadata is not canonical JSON") from exc
        registered = json.loads(canonical)
        object.__setattr__(self, "_metadata_json", canonical)
        object.__setattr__(self, "public_metadata", MappingProxyType(registered))

    @property
    def bits(self) -> int:
        return max(1, (len(self.probability) - 1).bit_length())


@dataclass(frozen=True)
class CacheIdentity:
    """Private, externally verified custody of one complete joint draw table.

    The caller must independently hash and replay the named cache artifact.
    These fields align the two recipient views but never enter their wire.
    """

    source_seed: int
    role: str
    realization: int
    release_id: str
    source_binding_sha256: str
    cache_sha256: str

    def __post_init__(self) -> None:
        if (type(self.source_seed) is not int or self.source_seed not in (0, 1)
                or self.role not in ("ATTACK_FIT_NEW", "INNER_SELECTION")
                or type(self.realization) is not int or self.realization not in (0, 1, 2)
                or not isinstance(self.release_id, str) or not self.release_id):
            raise ValueError("private joint cache seed/role/realization identity differs")
        for value in (self.source_binding_sha256, self.cache_sha256):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("private joint cache receipt SHA-256 is invalid")


def _metadata_fields(value: object, prefix: str, output: dict[str, object]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ValueError("public metadata keys must be nonempty strings")
            _metadata_fields(value[key], f"{prefix}.{key}", output)
    elif isinstance(value, (list, tuple)):
        for j, item in enumerate(value):
            _metadata_fields(item, f"{prefix}[{j}]", output)
    elif value is None:
        output[prefix] = "<NULL>"
    elif isinstance(value, (bool, np.bool_)):
        output[prefix] = int(value)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite public metadata value")
        output[prefix] = number
    elif isinstance(value, str):
        output[prefix] = value
    else:
        raise ValueError("unsupported public metadata value")


def _row(wire: Mapping[str, object], bank: PublicBank) -> tuple[int, dict[str, object]]:
    if not isinstance(wire, Mapping) or set(wire) != WIRE_KEYS:
        raise ValueError("sampled release wire fields differ from exact contract")
    code = wire["code_id"]
    if (wire["schema"] != SCHEMA or type(wire["recipient"]) is not int
            or wire["recipient"] != bank.recipient or type(code) is not int
            or not 0 <= code < len(bank.probability)
            or wire["release_id"] != bank.release_id):
        raise ValueError("sampled release schema/recipient/code/version differs")
    raw_probability = wire["probability"]
    if (type(raw_probability) is not list or
            len(raw_probability) != CLASSES[bank.recipient] or
            any(type(value) is not float or not math.isfinite(value)
                for value in raw_probability)):
        raise ValueError("sampled probability must be native float64 JSON numbers")
    observed = np.asarray(raw_probability, dtype=np.float64)
    if (observed.shape != (CLASSES[bank.recipient],)
            or observed.tobytes() != bank.probability[code].tobytes()):
        raise ValueError("sampled probability differs from public bank row")
    decision = wire["decision"]
    if type(decision) is not int or decision != int(np.argmax(observed)):
        raise ValueError("sampled decision differs from public bank row")
    if not isinstance(wire["public_metadata"], Mapping):
        raise ValueError("public metadata must be a mapping")
    try:
        metadata_json = json.dumps(wire["public_metadata"], sort_keys=True,
                                   separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("sampled release public metadata is not canonical JSON") from exc
    if metadata_json != bank._metadata_json:
        raise ValueError("sampled release differs from registered public metadata")
    flattened: dict[str, object] = {}
    _metadata_fields(wire["public_metadata"], "public_metadata", flattened)
    flattened["public_metadata.canonical_json"] = metadata_json
    return code, flattened


def materialize_recipient_batch(
    wires: Sequence[Mapping[str, object]], row_id: np.ndarray, unit: np.ndarray,
    bank: PublicBank,
) -> WireBatch:
    """Build exactly one cached observed release per ordered private person."""
    row = np.asarray(row_id)
    groups = np.asarray(unit)
    if (row.dtype != np.int64 or groups.dtype != np.int64
            or row.ndim != 1 or groups.shape != row.shape
            or len(row) != len(wires) or not len(row) or len(np.unique(row)) != len(row)):
        raise ValueError("private release row/group alignment is invalid")
    codes, metadata, cached = [], [], {}
    for wire, group in zip(wires, groups):
        code, flattened = _row(wire, bank)
        try:
            canonical = json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("sampled release wire is not canonical JSON") from exc
        previous = cached.setdefault(int(group), canonical)
        if previous != canonical:
            raise ValueError("duplicate group received different cached releases")
        codes.append(code)
        metadata.append(flattened)
    metadata_keys = tuple(sorted(metadata[0]))
    if any(tuple(sorted(record)) != metadata_keys for record in metadata):
        raise ValueError("public metadata field schema differs across people")
    bit_values = np.asarray([[int(bool(code & (1 << shift)))
                              for shift in range(bank.bits)] for code in codes], dtype=np.float64)
    fields: dict[str, Field] = {
        "schema": Field("category", np.asarray([SCHEMA] * len(codes))),
        "recipient": Field("numeric", np.asarray([bank.recipient] * len(codes))),
        "code_id": Field("category", np.asarray(codes, dtype=np.int64)),
        "code_bits": Field("bits", bit_values),
        "probability": Field("probability", bank.probability[np.asarray(codes)]),
        "decision": Field("decision", np.argmax(bank.probability[np.asarray(codes)], axis=1)),
        "release_id": Field("category", np.asarray([bank.release_id] * len(codes))),
    }
    for key in metadata_keys:
        values = [record[key] for record in metadata]
        kinds = {"category" if isinstance(value, str) else "numeric" for value in values}
        if len(kinds) != 1:
            raise ValueError("public metadata field changes type across people")
        kind = "metadata_category" if "category" in kinds else "metadata"
        fields[key] = Field(kind, np.asarray(values))
    return WireBatch(row, groups, fields, code_fields=("code_id",))


def materialize_pair_batch(
    wires1: Sequence[Mapping[str, object]], wires2: Sequence[Mapping[str, object]],
    row_id: np.ndarray, unit: np.ndarray, bank1: PublicBank, bank2: PublicBank,
    *, left_cache: CacheIdentity, right_cache: CacheIdentity,
) -> WireBatch:
    """Join only two projections of the same externally verified joint cache."""
    if (bank1.recipient != 1 or bank2.recipient != 2
            or bank1.release_id != bank2.release_id
            or bank1.public_metadata["source_binding_sha256"] !=
            bank2.public_metadata["source_binding_sha256"]
            or bank1.public_metadata["mechanism_sha256"] !=
            bank2.public_metadata["mechanism_sha256"]):
        raise ValueError("recipient banks must belong to the same paired arm/source/mechanism")
    if left_cache != right_cache:
        raise ValueError("recipient views require the same verified joint cache receipt")
    if (left_cache.release_id != bank1.release_id
            or left_cache.source_binding_sha256 !=
            bank1.public_metadata["source_binding_sha256"]):
        raise ValueError("joint cache identity differs from paired public banks")
    left = materialize_recipient_batch(wires1, row_id, unit, bank1)
    right = materialize_recipient_batch(wires2, row_id, unit, bank2)
    return pair_batch(left, right)


def pair_posterior_lookup(prior: np.ndarray, public_q: np.ndarray,
                          output_shape: tuple[int, int]) -> np.ndarray:
    """Bayes features for each public output pair under a declared public law.

    The caller must pin the prior and Q table as public model artifacts. An
    unreachable output pair receives the public prior, without peeking at a
    person's hidden source state.
    """
    p = np.asarray(prior, dtype=np.float64)
    q = np.asarray(public_q, dtype=np.float64)
    if (p.ndim != 1 or not len(p) or q.shape != (len(p), math.prod(output_shape))
            or not np.isfinite(p).all() or not np.isfinite(q).all()
            or np.any(p < 0) or np.any(q < 0)
            or not np.isclose(p.sum(), 1.0, rtol=0, atol=1e-10)
            or not np.allclose(q.sum(axis=1), 1.0, rtol=0, atol=1e-10)):
        raise ValueError("public prior/channel law shape or stochasticity differs")
    mass = p[:, None] * q
    denominator = mass.sum(axis=0)
    posterior = np.divide(mass, denominator[None, :],
                          out=np.broadcast_to(p[:, None], mass.shape).copy(),
                          where=denominator[None, :] > 0)
    return posterior.T.reshape(*output_shape, len(p))
