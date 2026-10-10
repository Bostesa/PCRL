"""R04 reconstruction of D's complete tabular attack views from public wire.

No R09 adapter, R02 complete-view encoder, or inherited AttackView is imported.
Categorical vocabularies are learned from ATT public wire only. INNER-only IDs
use an explicit unknown column, so no person is discarded.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np


def _fields(wire: Mapping, prefix: str = "") -> dict[str, tuple[str, np.ndarray]]:
    if set(wire) != {"code", "probabilities", "decision", "metadata"}:
        raise ValueError("D complete-view replay requires entire exact wire")
    code = np.asarray(wire["code"])
    prob = np.asarray(wire["probabilities"], dtype=np.float64)
    dec = np.asarray(wire["decision"])
    n = len(code)
    if (code.shape != (n, 10) or not np.issubdtype(code.dtype, np.integer)
            or not np.isin(code, (0, 1)).all() or prob.ndim != 2
            or prob.shape[0] != n or not np.isfinite(prob).all()
            or not np.allclose(prob.sum(1), 1, rtol=0, atol=1e-8)
            or dec.shape != (n,) or not np.issubdtype(dec.dtype, np.integer)
            or not np.array_equal(dec, np.argmax(prob, axis=1))):
        raise ValueError("invalid hard code/probability/decision wire")
    fields = {
        # The attacker derives a categorical ID from this transmitted bit
        # field and appends its ATT-vocabulary one-hot *after* the raw bits.
        # There is no independent code-ID field on the service wire.
        f"{prefix}code_bits": ("code_bits", code.astype(np.float64)),
        f"{prefix}probabilities": ("probability", prob),
        f"{prefix}decision": ("decision", dec.astype(np.float64)),
    }
    meta = wire["metadata"]
    if not isinstance(meta, Mapping) or not meta:
        raise ValueError("fixed metadata missing")
    for key in sorted(meta):
        value = meta[key]
        if type(value) in (bool, int, float):
            fields[f"{prefix}metadata.{key}"] = ("metadata", np.full(n, float(value)))
        elif value is None or isinstance(value, str):
            fields[f"{prefix}metadata.{key}"] = (
                "metadata_category", np.full(n, "null" if value is None else value)
            )
        else:
            raise ValueError("row-varying or invalid public metadata")
    return fields


def _build(att_fields: dict, inner_fields: dict, code_names: tuple[str, ...],
           att_derived: list[np.ndarray], inner_derived: list[np.ndarray]) -> tuple[dict, dict]:
    if tuple(att_fields) != tuple(inner_fields) or any(
        att_fields[name][0] != inner_fields[name][0] for name in att_fields
    ):
        raise ValueError("ATT and INNER wire schemas differ")
    vocab = {}
    for name, (kind, values) in att_fields.items():
        if kind in ("category", "metadata_category", "code_bits"):
            if kind == "code_bits":
                ids = np.asarray(values, dtype=np.int64) @ (1 << np.arange(10, dtype=np.int64))
                vocab[name] = tuple(sorted(set(str(value) for value in ids)))
            else:
                vocab[name] = tuple(sorted(set(str(value) for value in values)))

    def encode(fields, derived):
        n = len(next(iter(fields.values()))[1])
        blocks, metadata, bits = [], [], []
        for name, (kind, values) in fields.items():
            if name in vocab:
                lookup = {value: position for position, value in enumerate(vocab[name])}
                onehot = np.zeros((n, len(lookup) + 1), dtype=np.float64)
                if kind == "code_bits":
                    raw = np.asarray(values, dtype=np.float64)
                    ids = raw.astype(np.int64) @ (1 << np.arange(10, dtype=np.int64))
                    positions = [lookup.get(str(value), len(lookup)) for value in ids]
                    onehot[np.arange(n), positions] = 1
                    block = np.concatenate((raw, onehot), axis=1)
                else:
                    positions = [lookup.get(str(value), len(lookup)) for value in values]
                    onehot[np.arange(n), positions] = 1
                    block = onehot
            else:
                block = np.asarray(values, dtype=np.float64)
                if block.ndim == 1:
                    block = block[:, None]
                if block.shape[0] != n or not np.isfinite(block).all():
                    raise ValueError("numeric public wire field invalid")
            blocks.append(block)
            if kind in ("metadata", "metadata_category"):
                metadata.append(block)
            if kind == "code_bits":
                bits.append(np.asarray(values, dtype=np.float64))
        cell_columns = [
            [str(value) for value in (np.asarray(fields[name][1], dtype=np.int64)
                                      @ (1 << np.arange(10, dtype=np.int64)))]
            for name in code_names
        ]
        cell = np.asarray([json.dumps(list(values), separators=(",", ":"))
                           for values in zip(*cell_columns)], dtype=np.str_)
        x = np.concatenate(blocks, axis=1)
        return {
            "x": x,
            "x_defense_aware": np.concatenate([x, *derived], axis=1),
            "cell_key": cell,
            "x_metadata": np.concatenate(metadata, axis=1),
            "x_code_bits": np.concatenate(bits, axis=1),
        }

    return encode(att_fields, att_derived), encode(inner_fields, inner_derived)


def build_d_view(
    att_wire_1: Mapping, att_wire_2: Mapping,
    inner_wire_1: Mapping, inner_wire_2: Mapping,
    view: str,
) -> tuple[dict, dict]:
    """Return independently reconstructed ATT/INNER arrays for r1/r2/pair."""
    if view not in ("r1", "r2", "pair"):
        raise ValueError("unknown recipient/coalition view")
    if view == "r1":
        a, b = _fields(att_wire_1), _fields(inner_wire_1)
        return _build(a, b, ("code_bits",),
                      [np.log(np.maximum(att_wire_1["probabilities"], 1e-12))],
                      [np.log(np.maximum(inner_wire_1["probabilities"], 1e-12))])
    if view == "r2":
        a, b = _fields(att_wire_2), _fields(inner_wire_2)
        return _build(a, b, ("code_bits",),
                      [np.log(np.maximum(att_wire_2["probabilities"], 1e-12))],
                      [np.log(np.maximum(inner_wire_2["probabilities"], 1e-12))])
    a = {**_fields(att_wire_1, "r1."), **_fields(att_wire_2, "r2.")}
    b = {**_fields(inner_wire_1, "r1."), **_fields(inner_wire_2, "r2.")}

    def derived(first, second):
        bit1, bit2 = np.asarray(first["code"], dtype=np.float64), np.asarray(second["code"], dtype=np.float64)
        if len(bit1) != len(bit2):
            raise ValueError("unaligned public pair")
        return [np.log(np.maximum(first["probabilities"], 1e-12)),
                np.log(np.maximum(second["probabilities"], 1e-12)),
                (bit1[:, :, None] * bit2[:, None, :]).reshape(len(bit1), 100)]

    return _build(a, b, ("r1.code_bits", "r2.code_bits"),
                  derived(att_wire_1, att_wire_2),
                  derived(inner_wire_1, inner_wire_2))


def compare_saved_view(saved_path: str, reconstructed: dict,
                       row_id: np.ndarray, unit: np.ndarray) -> dict:
    """Byte-value comparison to the fitted reader's saved public-view arrays."""
    with np.load(saved_path, allow_pickle=False) as saved:
        if (not np.array_equal(saved["row_id"], row_id)
                or not np.array_equal(saved["unit"], unit)
                or set(saved.files) != set(reconstructed) | {"row_id", "unit"}):
            raise ValueError("saved attack view identity or field inventory differs")
        max_abs = {}
        for key, expected in reconstructed.items():
            observed = np.asarray(saved[key])
            if key == "cell_key":
                if not np.array_equal(observed, expected):
                    raise ValueError("saved categorical cell keys differ from public wire")
                max_abs[key] = 0.0
            else:
                if observed.shape != expected.shape:
                    raise ValueError(f"saved public view shape differs: {key}")
                mismatch = float(np.max(np.abs(observed - expected)))
                if mismatch > 1e-12:
                    raise ValueError(f"saved public view differs from actual wire: {key}")
                max_abs[key] = mismatch
    return {"status": "COMPLETE_VIEW_EXACT_REPLAY", "rows": len(row_id),
            "max_abs_difference": max_abs}
