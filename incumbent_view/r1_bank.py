"""Original-ID P0 decoder, bounded same-class support and exact public wire."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import symmetric_kl
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.reference_materialize import reconstruct_reference_wires


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_array(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def nearest_support(probability):
    """Identity then <=3 nearest same-class ORIGINAL global IDs; no medoid."""
    p = np.asarray(probability)
    if p.dtype != np.float64 or p.ndim != 2 or p.shape[1] not in (2, 6) or not len(p):
        raise ValueError("P0 table needs original float64 class vectors")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-12):
        raise ValueError("invalid P0 probability table")
    decision = p.argmax(1)
    if np.any(np.sort(p, axis=1)[:, -1] <= np.sort(p, axis=1)[:, -2]):
        raise ValueError("P0 table must have strict decisions")
    out = []
    for original_id, row in enumerate(p):
        others = [int(i) for i in np.flatnonzero(decision == decision[original_id]) if i != original_id]
        others.sort(key=lambda i: (symmetric_kl(row, p[i]), i))
        out.append((original_id, *others[:3]))
    return tuple(out)


def load_bank(path, expected_sha256=None):
    if expected_sha256 is not None and sha_file(path) != expected_sha256:
        raise ValueError("P0 public bank bytes differ")
    record = json.loads(Path(path).read_text())
    if record.get("schema") != "pcrl-incumbent-original-public-bank-v1" or record.get("seed") not in (0, 1):
        raise ValueError("P0 bank identity differs")
    for i in (1, 2):
        item = record["recipients"][str(i)]
        p = np.asarray(item["probability"], dtype=np.float64)
        if sha_array(p) != item["bank_sha256"] or p.shape[1] != (2, 6)[i-1]:
            raise ValueError("P0 bank array hash/order differs")
        if tuple(map(tuple, item["support"])) != nearest_support(p):
            raise ValueError("P0 fixed support differs")
        if item["code_length_bits"] != (len(p)-1).bit_length():
            raise ValueError("P0 public code length differs")
        if item["global_ids"] != list(range(len(p))):
            raise ValueError("P0 original IDs were changed")
    return record


def wires_for_codes(bank, recipient, code):
    if recipient not in (1, 2):
        raise ValueError("recipient must be 1 or 2")
    item = bank["recipients"][str(recipient)]
    p = np.asarray(item["probability"], dtype=np.float64)
    c = np.asarray(code)
    if c.dtype != np.int64 or c.ndim != 1 or np.any((c < 0) | (c >= len(p))):
        raise ValueError("P0 original public code is invalid")
    return reconstruct_reference_wires(
        p[c], p[c].argmax(1).astype(np.int64), c, recipient=recipient,
        interface_kind="finite", release_id=bank["release_id"],
        source_binding_sha256=bank["source_binding_sha256"],
        bank_sha256=item["bank_sha256"], bank_size=len(p))


def wire_hash(wires):
    digest = hashlib.sha256()
    for wire in wires:
        digest.update(json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()+b"\n")
    return digest.hexdigest()
