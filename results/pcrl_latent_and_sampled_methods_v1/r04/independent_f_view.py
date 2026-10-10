"""Independent complete-view encoding of continuous F's transmitted wire."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _fields(wire: Mapping, prefix: str = "") -> dict[str, tuple[str, np.ndarray]]:
    if set(wire) != {"features", "probabilities", "decision", "metadata"}:
        raise ValueError("F view requires exact full continuous wire")
    feat = np.asarray(wire["features"], dtype=np.float64)
    prob = np.asarray(wire["probabilities"], dtype=np.float64)
    dec = np.asarray(wire["decision"])
    n = len(feat)
    if (feat.shape != (n, 16) or not np.isfinite(feat).all()
            or prob.shape not in ((n, 2), (n, 6))
            or not np.isfinite(prob).all() or np.any(prob < 0)
            or np.max(np.abs(prob.sum(1) - 1), initial=0) > 1e-8
            or dec.shape != (n,) or not np.issubdtype(dec.dtype, np.integer)
            or not np.array_equal(dec, np.argmax(prob, axis=1))):
        raise ValueError("F continuous feature or task output invalid")
    fields = {f"{prefix}features": ("numeric", feat),
              f"{prefix}probabilities": ("probability", prob),
              f"{prefix}decision": ("decision", dec.astype(np.float64))}
    metadata = wire["metadata"]
    if not isinstance(metadata, Mapping) or not metadata:
        raise ValueError("F public metadata absent")
    for key in sorted(metadata):
        value = metadata[key]
        if type(value) in (bool, int, float):
            fields[f"{prefix}metadata.{key}"] = ("metadata", np.full(n, float(value)))
        elif value is None or isinstance(value, str):
            fields[f"{prefix}metadata.{key}"] = (
                "metadata_category", np.full(n, "null" if value is None else value))
        else:
            raise ValueError("F metadata has invalid person-varying value")
    return fields


def _build(att_fields: dict, inner_fields: dict,
           att_derived: list[np.ndarray], inner_derived: list[np.ndarray]) -> tuple[dict, dict]:
    if (tuple(att_fields) != tuple(inner_fields)
            or any(att_fields[key][0] != inner_fields[key][0] for key in att_fields)):
        raise ValueError("F ATT/INNER wire field schemas differ")
    vocabulary = {}
    for key, (kind, value) in att_fields.items():
        if kind == "metadata_category":
            vocabulary[key] = tuple(sorted(set(str(x) for x in value)))

    def encode(fields: dict, derived: list[np.ndarray]) -> dict:
        n = len(next(iter(fields.values()))[1])
        blocks = []
        meta = []
        for key, (kind, value) in fields.items():
            if key in vocabulary:
                choices = vocabulary[key]
                index = {choice: j for j, choice in enumerate(choices)}
                onehot = np.zeros((n, len(choices) + 1), dtype=np.float64)
                onehot[np.arange(n), [index.get(str(x), len(choices)) for x in value]] = 1
                block = onehot
            else:
                block = np.asarray(value, dtype=np.float64)
                if block.ndim == 1:
                    block = block[:, None]
                if block.ndim != 2 or block.shape[0] != n or not np.isfinite(block).all():
                    raise ValueError("F numeric view field invalid")
            blocks.append(block)
            if kind in ("metadata", "metadata_category"):
                meta.append(block)
        x = np.concatenate(blocks, axis=1)
        return {"x": x, "x_defense_aware": np.concatenate([x, *derived], axis=1),
                "x_metadata": np.concatenate(meta, axis=1)}

    return encode(att_fields, att_derived), encode(inner_fields, inner_derived)


def build_f_view(att_wire_1: Mapping, att_wire_2: Mapping,
                 inner_wire_1: Mapping, inner_wire_2: Mapping,
                 view: str) -> tuple[dict, dict]:
    """Rebuild F r1, r2 or aligned pair without the R08/R02 adapters."""
    if view == "r1":
        return _build(_fields(att_wire_1), _fields(inner_wire_1),
                      [np.log(np.maximum(att_wire_1["probabilities"], 1e-12))],
                      [np.log(np.maximum(inner_wire_1["probabilities"], 1e-12))])
    if view == "r2":
        return _build(_fields(att_wire_2), _fields(inner_wire_2),
                      [np.log(np.maximum(att_wire_2["probabilities"], 1e-12))],
                      [np.log(np.maximum(inner_wire_2["probabilities"], 1e-12))])
    if view != "pair":
        raise ValueError("unknown F view")
    return _build({**_fields(att_wire_1, "r1."), **_fields(att_wire_2, "r2.")},
                  {**_fields(inner_wire_1, "r1."), **_fields(inner_wire_2, "r2.")},
                  [np.log(np.maximum(att_wire_1["probabilities"], 1e-12)),
                   np.log(np.maximum(att_wire_2["probabilities"], 1e-12))],
                  [np.log(np.maximum(inner_wire_1["probabilities"], 1e-12)),
                   np.log(np.maximum(inner_wire_2["probabilities"], 1e-12))])
