"""Independent reconstruction of Team C's disclosed local and coalition views.

This module imports no Team C or shared evaluator implementation. Adult use
requires a separately sealed S4 unit in the global S3 queue.
"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


ATOM_ORDER = (
    "age", "education_fact", "capital-gain", "capital-loss",
    "hours-per-week", "workclass", "marital-status", "relationship",
    "native-country",
)
LOCAL_FIELDS = frozenset((
    "selected_values", "feature_names", "selected_groups", "mask_id",
    "head_id", "calibrator_id", "flags", "probabilities", "decisions",
))
COALITION_FIELDS = frozenset((
    "selected_values", "feature_names", "selected_groups", "income_groups",
    "occupation_group_groups", "income_mask_id", "occupation_group_mask_id",
    "income_head_id", "occupation_group_head_id", "income_calibrator_id",
    "occupation_group_calibrator_id", "flags", "income_probabilities",
    "occupation_group_probabilities", "income_decisions",
    "occupation_group_decisions",
))
NUMERIC_FIELDS = {
    "income": ("selected_values", "probabilities", "decisions"),
    "occupation_group": ("selected_values", "probabilities", "decisions"),
    "coalition": (
        "selected_values", "income_probabilities",
        "occupation_group_probabilities", "income_decisions",
        "occupation_group_decisions",
    ),
}


def atom_indices(feature_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Resolve all nine whole Adult atoms from a pinned ordered 83-column list."""
    names = tuple(feature_names)
    if len(names) != 83 or len(set(names)) != 83:
        raise ValueError("C feature schema needs 83 distinct columns")
    fixed = ("age", "education-num", "capital-gain", "capital-loss",
             "hours-per-week")
    if names[:5] != fixed:
        raise ValueError("C fixed numeric feature order differs")
    starts = {
        "workclass": ("workclass=", 8),
        "education": ("education=", 16),
        "marital-status": ("marital-status=", 7),
        "relationship": ("relationship=", 6),
        "native-country": ("native-country=", 41),
    }
    indices = {key: tuple(i for i, name in enumerate(names)
                          if name.startswith(prefix))
               for key, (prefix, _) in starts.items()}
    if any(len(indices[key]) != count for key, (_, count) in starts.items()):
        raise ValueError("C categorical atom size differs")
    if set(range(5, 83)) != set(i for group in indices.values() for i in group):
        raise ValueError("C categorical atoms are not a complete partition")
    return {
        "age": (0,),
        "education_fact": (1,) + indices["education"],
        "capital-gain": (2,), "capital-loss": (3,),
        "hours-per-week": (4,),
        "workclass": indices["workclass"],
        "marital-status": indices["marital-status"],
        "relationship": indices["relationship"],
        "native-country": indices["native-country"],
    }


def _groups(value: object, atoms: Mapping[str, tuple[int, ...]]) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("C selected groups need an ordered sequence")
    groups = tuple(value)
    if (len(groups) != len(set(groups)) or
            any(group not in atoms for group in groups) or
            groups != tuple(group for group in ATOM_ORDER if group in groups)):
        raise ValueError("C selected groups differ from canonical atoms")
    return groups


def _columns(groups: tuple[str, ...], atoms: Mapping[str, tuple[int, ...]]) -> tuple[int, ...]:
    return tuple(sorted(i for group in groups for i in atoms[group]))


def _prob(value: object, rows: int, width: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if (arr.shape != (rows, width) or not np.isfinite(arr).all() or
            np.any((arr < 0) | (arr > 1)) or
            not np.allclose(arr.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("C disclosed probability simplex differs")
    return arr


def _decision(value: object, probability: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if (arr.shape != (len(probability),) or
            not np.issubdtype(arr.dtype, np.integer) or
            not np.array_equal(arr, probability.argmax(axis=1))):
        raise ValueError("C decision differs from disclosed probability")
    return arr.astype(np.int64, copy=False)


def rebuild_views(
    role_x: np.ndarray,
    feature_names: Sequence[str],
    wires: Mapping[str, Mapping[str, object]],
    raw_probabilities: Mapping[str, np.ndarray],
) -> dict[str, dict]:
    """Recompute every selected value and actual S2 reader matrix.

    `raw_probabilities` comes from a separate S4 replay of the saved public
    head. This function keeps the calibrated disclosed wire and pre-calibration
    defense-aware columns distinct.
    """
    names = tuple(feature_names)
    atoms = atom_indices(names)
    x_source = np.asarray(role_x)
    if (x_source.ndim != 2 or x_source.shape[1] != 83 or
            not np.issubdtype(x_source.dtype, np.number) or
            not np.isfinite(x_source).all() or
            set(wires) != {"income", "occupation_group", "coalition"} or
            set(raw_probabilities) != {"income", "occupation_group"}):
        raise ValueError("C source features or three-view declaration differ")
    rows = len(x_source)
    groups: dict[str, tuple[str, ...]] = {}
    selected: dict[str, np.ndarray] = {}
    for name, required in (("income", LOCAL_FIELDS),
                           ("occupation_group", LOCAL_FIELDS),
                           ("coalition", COALITION_FIELDS)):
        wire = wires[name]
        if set(wire) != required or wire["flags"] != ():
            raise ValueError("C wire fields or flags differ from release contract")
        groups[name] = _groups(wire["selected_groups"], atoms)
        cols = _columns(groups[name], atoms)
        selected[name] = np.asarray(x_source[:, cols], dtype=np.float64)
        if (tuple(wire["feature_names"]) != tuple(names[i] for i in cols) or
                not np.array_equal(np.asarray(wire["selected_values"],
                                              dtype=np.float64), selected[name])):
            raise ValueError("C selected raw columns differ from admitted features")
    if (set(groups["coalition"]) != set(groups["income"]) | set(groups["occupation_group"]) or
            tuple(wires["coalition"]["income_groups"]) != groups["income"] or
            tuple(wires["coalition"]["occupation_group_groups"]) != groups["occupation_group"]):
        raise ValueError("C coalition union differs from local atom selections")
    for name in ("income", "occupation_group"):
        wire = wires[name]
        if wire["mask_id"] != ("+".join(groups[name]) or "none"):
            raise ValueError("C public mask ID differs from selected atoms")
    union = wires["coalition"]
    for name, prefix in (("income", "income"),
                         ("occupation_group", "occupation_group")):
        local = wires[name]
        for key, local_key in (("mask_id", "mask_id"),
                               ("head_id", "head_id"),
                               ("calibrator_id", "calibrator_id")):
            if union[f"{prefix}_{key}"] != local[local_key]:
                raise ValueError("C coalition public identity differs from local wire")
    pi = _prob(wires["income"]["probabilities"], rows, 2)
    po = _prob(wires["occupation_group"]["probabilities"], rows, 6)
    di = _decision(wires["income"]["decisions"], pi)
    do = _decision(wires["occupation_group"]["decisions"], po)
    if (not np.array_equal(_prob(union["income_probabilities"], rows, 2), pi) or
            not np.array_equal(_prob(union["occupation_group_probabilities"], rows, 6), po) or
            not np.array_equal(_decision(union["income_decisions"], pi), di) or
            not np.array_equal(_decision(union["occupation_group_decisions"], po), do)):
        raise ValueError("C coalition output differs from recipient output")
    raw_pi = _prob(raw_probabilities["income"], rows, 2)
    raw_po = _prob(raw_probabilities["occupation_group"], rows, 6)
    out = {}
    for name in ("income", "occupation_group", "coalition"):
        wire = wires[name]
        numeric_fields = NUMERIC_FIELDS[name]
        blocks = []
        spans = []
        start = 0
        for field in numeric_fields:
            if field == "selected_values":
                block = selected[name]
            elif field.endswith("decisions") or field == "decisions":
                width = 2 if field in ("income_decisions",) or name == "income" else 6
                block = np.eye(width, dtype=np.float64)[np.asarray(wire[field], dtype=np.int64)]
            else:
                block = np.asarray(wire[field], dtype=np.float64)
            blocks.append(block)
            spans.append((field, start, start + block.shape[1]))
            start += block.shape[1]
        numeric = np.concatenate(blocks, axis=1)
        constants = tuple((key, json.dumps(wire[key], sort_keys=True,
                                           separators=(",", ":")))
                          for key in sorted(set(wire) - set(numeric_fields)))
        extra = ((raw_pi,) if name == "income" else
                 (raw_po,) if name == "occupation_group" else
                 (raw_pi, raw_po))
        out[name] = {
            "x": numeric,
            "x_defense_aware": np.concatenate((numeric, *extra), axis=1),
            "x_raw_only": selected[name],
            "components": numeric_fields,
            "wire_components": numeric_fields + tuple(key for key, _ in constants),
            "component_columns": tuple(spans),
            "wire_constants": constants,
        }
    return out


def load_head_checkpoint(path: str | Path, expected_sha256: str,
                         feature_names: Sequence[str]) -> dict:
    """Read a saved C weight archive as data, without importing C training."""
    source = Path(path)
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("C head checkpoint SHA differs")
    with np.load(source, allow_pickle=False) as archive:
        if "meta" not in archive.files:
            raise ValueError("C head checkpoint metadata is absent")
        meta = json.loads(str(archive["meta"].item()))
        keys = ("0.weight", "0.bias", "2.weight", "2.bias",
                "4.weight", "4.bias")
        if (meta.get("schema") != "team-c-head-checkpoint-v1" or
                meta.get("purpose") not in ("income", "occupation_group") or
                meta.get("state_keys") != list(keys) or
                set(archive.files) != {"meta", *(f"state_{i}" for i in range(6))}):
            raise ValueError("C head checkpoint schema or state keys differ")
        atom_map = atom_indices(feature_names)
        groups = _groups(meta.get("selected_atoms"), atom_map)
        width = len(_columns(groups, atom_map))
        hidden = meta.get("spec", {}).get("hidden_width")
        classes = 2 if meta["purpose"] == "income" else 6
        if (meta.get("selected_columns") != width or
                not isinstance(hidden, int) or hidden <= 0 or
                meta.get("receipt", {}).get("feature_schema_sha256") !=
                hashlib.sha256("\n".join(feature_names).encode()).hexdigest()):
            raise ValueError("C head checkpoint mask or feature schema differs")
        shapes = ((hidden, width), (hidden,), (hidden, hidden),
                  (hidden,), (classes, hidden), (classes,))
        state = []
        for index, shape in enumerate(shapes):
            array = np.asarray(archive[f"state_{index}"])
            if (array.shape != shape or array.dtype != np.float32 or
                    not np.isfinite(array).all()):
                raise ValueError("C head checkpoint weights differ")
            state.append(array.copy())
    return {"metadata": meta, "state": tuple(state), "selected_atoms": groups,
            "selected_columns": width, "classes": classes,
            "checkpoint_sha256": expected_sha256}


def head_probabilities(head: Mapping[str, object], selected: np.ndarray) -> np.ndarray:
    """Forward the six saved tensors through an independent two-ReLU head."""
    import torch

    x = np.asarray(selected)
    if (x.ndim != 2 or x.shape[1] != head["selected_columns"] or
            not np.isfinite(x).all()):
        raise ValueError("C head selected matrix differs")
    torch.set_num_threads(1)
    value = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    state = [torch.from_numpy(np.asarray(item)) for item in head["state"]]
    with torch.no_grad():
        for offset in (0, 2):
            value = torch.relu(torch.nn.functional.linear(
                value, state[offset], state[offset + 1]))
        logits = torch.nn.functional.linear(value, state[4], state[5])
        probability = torch.softmax(logits.to(dtype=torch.float64), dim=1).cpu().numpy()
    return _prob(probability, len(x), int(head["classes"]))


def temperature_apply(raw_probabilities: np.ndarray,
                      record: Mapping[str, object]) -> np.ndarray:
    """Recompute the saved global inverse-temperature map independently."""
    if (record.get("schema") != "pcrl-portfolio-global-temp-v1" or
            record.get("task") not in ("income", "occupation")):
        raise ValueError("C calibrator schema or task differs")
    width = 2 if record["task"] == "income" else 6
    raw = np.asarray(raw_probabilities, dtype=np.float64)
    p = _prob(raw, len(raw), width)
    alpha = float(record.get("alpha", float("nan")))
    if not np.isfinite(alpha) or not 0.25 <= alpha <= 4.0:
        raise ValueError("C calibrator inverse temperature differs")
    if alpha == 1.0:
        return p.copy()
    clipped = np.maximum(p, 1e-12)
    normalized = clipped / clipped.sum(axis=1, keepdims=True)
    scaled = alpha * np.log(normalized)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    result = exp / exp.sum(axis=1, keepdims=True)
    if not np.array_equal(result.argmax(axis=1), p.argmax(axis=1)):
        raise ValueError("C calibrator altered public decisions")
    return _prob(result, len(p), width)


def sha_array(value: object) -> str:
    """Hash dtype, shape and contiguous bytes in the frozen S2 wire format."""
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def view_content_hashes(view: Mapping[str, object]) -> dict[str, str | None]:
    """Recreate the fixed S2 content receipt from rebuilt C matrices."""
    schema = {key: view[key] for key in
              ("components", "wire_components", "component_columns",
               "wire_constants")}
    wire_schema = hashlib.sha256(json.dumps(
        schema, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return {
        "x": sha_array(np.asarray(view["x"], dtype=np.float64)),
        "x_defense_aware": sha_array(np.asarray(
            view["x_defense_aware"], dtype=np.float64)),
        "cell_key": None,
        "x_metadata": None,
        "x_code_bits": None,
        "x_raw_only": sha_array(np.asarray(view["x_raw_only"],
                                          dtype=np.float64)),
        "wire_schema": wire_schema,
    }


def load_saved_bundle(receipt_path: str | Path, npz_path: str | Path,
                      role_x: np.ndarray, feature_names: Sequence[str],
                      raw_probabilities: Mapping[str, np.ndarray],
                      expected_row_id: np.ndarray,
                      expected_unit: np.ndarray) -> dict:
    """Rebuild a complete ATT/INNER C wire from admitted features and saved bytes."""
    receipt = json.loads(Path(receipt_path).read_text())
    package_path = Path(npz_path)
    if (receipt.get("schema") != "team-c-private-complete-view-v1" or
            hashlib.sha256(package_path.read_bytes()).hexdigest() !=
            receipt.get("npz_sha256")):
        raise ValueError("C saved view bundle SHA or schema differs")
    with np.load(package_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    expected_keys = receipt.get("array_sha256")
    if not isinstance(expected_keys, dict) or set(arrays) != set(expected_keys):
        raise ValueError("C saved array member set differs")
    for key, value in arrays.items():
        if sha_array(value) != expected_keys[key]:
            raise ValueError(f"C saved array SHA differs: {key}")
    names = ("income", "occupation_group", "coalition")
    matrices = ("row_id", "unit", "x", "x_defense_aware", "x_raw_only")
    if (set(receipt.get("wire_constants", {})) != set(names) or
            set(receipt.get("view_schemas", {})) != set(names)):
        raise ValueError("C saved wire/view declaration differs")
    wires = {}
    for name in names:
        wire = dict(receipt["wire_constants"][name])
        for field in ("feature_names", "selected_groups", "income_groups",
                      "occupation_group_groups", "flags"):
            if field in wire:
                wire[field] = tuple(wire[field])
        wire.update({key[len(f"{name}__wire__"):]: value
                     for key, value in arrays.items()
                     if key.startswith(f"{name}__wire__")})
        wires[name] = wire
    rebuilt = rebuild_views(role_x, feature_names, wires, raw_probabilities)
    expected_array_keys = {
        *(f"{name}__view__{field}" for name in names for field in matrices),
        *(f"{name}__wire__{field}" for name in names
          for field in NUMERIC_FIELDS[name]),
    }
    if set(arrays) != expected_array_keys:
        raise ValueError("C saved view array members differ from complete wire")
    reference_id = arrays["income__view__row_id"]
    reference_unit = arrays["income__view__unit"]
    if (reference_id.ndim != 1 or reference_unit.shape != reference_id.shape or
            len(reference_id) != len(role_x) or
            len(np.unique(reference_id)) != len(reference_id)):
        raise ValueError("C saved row/unit geometry differs")
    if (not np.array_equal(reference_id, np.asarray(expected_row_id,
                                                    dtype=np.int64)) or
            not np.array_equal(reference_unit, np.asarray(expected_unit,
                                                          dtype=np.int64))):
        raise ValueError("C saved ordered row/unit differs from admitted role")
    for name in names:
        for field in matrices:
            actual = arrays[f"{name}__view__{field}"]
            if field in ("row_id", "unit"):
                expected = reference_id if field == "row_id" else reference_unit
            else:
                expected = rebuilt[name][field]
            if not np.array_equal(actual, expected):
                raise ValueError(f"C saved {name} {field} differs from independent view")
        schema = {key: rebuilt[name][key] for key in
                  ("components", "wire_components", "component_columns",
                   "wire_constants")}
        if json.loads(json.dumps(schema)) != receipt["view_schemas"][name]:
            raise ValueError(f"C saved {name} complete-view schema differs")
        rebuilt[name]["row_id"] = reference_id.copy()
        rebuilt[name]["unit"] = reference_unit.copy()
    return {"views": rebuilt, "wires": wires, "receipt": receipt,
            "receipt_sha256": hashlib.sha256(Path(receipt_path).read_bytes()).hexdigest()}
