"""Independent recipe and full-input task-gate arithmetic.

No R1, portfolio scoring, selection, or deployment implementation is imported.
The caller supplies admitted, aligned private arrays after queue admission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from results.pcrl_utility_reference_repair_v1.r3.independent_task import (
    gate_against_references, task_metrics,
)
from results.pcrl_utility_reference_repair_v1.r3.source_reference import (
    apply_inverse_temperature, file_sha, forward_u0, restore_source,
)


TASKS = (("income", 2, "income"),
         ("occupation_group", 6, "occupation"))


def _read_bound_json(path: Path, expected_sha: str) -> dict:
    if file_sha(path) != expected_sha:
        raise PermissionError("private JSON differs from frozen SHA")
    return json.loads(path.read_text())


def _read_bound_npz(path: Path, expected_sha: str,
                    expected_fields: set[str]) -> dict[str, np.ndarray]:
    if file_sha(path) != expected_sha:
        raise PermissionError("private predictions differ from frozen SHA")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_fields:
            raise PermissionError("private prediction schema differs")
        return {name: archive[name] for name in expected_fields}


def _aligned_people(arrays: dict[str, np.ndarray], expected_rows: int | None = None):
    row, unit = arrays["row_id"], arrays["unit"]
    if (row.dtype != np.int64 or unit.dtype != np.int64 or row.ndim != 1
            or unit.shape != row.shape or (expected_rows is not None and len(row) != expected_rows)
            or len(np.unique(row)) != len(row)):
        raise PermissionError("private predictions have invalid people or order")
    return row, unit


def screen_metrics(arrays: dict[str, np.ndarray]) -> dict:
    _aligned_people(arrays)
    out = {}
    for recipe in ("A", "B"):
        out[recipe] = {}
        for task, width, label_key in TASKS:
            value = arrays[f"{recipe}_{task}"]
            if value.shape != (len(arrays["row_id"]), width):
                raise ValueError("screen probability width differs")
            out[recipe][task] = task_metrics(value, arrays[label_key])
    return out


def select_recipe(screens: dict[int, dict[str, np.ndarray]]) -> dict:
    """Sum raw outer-FIT validation NLL over paired seeds and both tasks."""
    if set(screens) != {0, 1}:
        raise ValueError("both paired screen seeds are required")
    per_seed = {seed: screen_metrics(arrays) for seed, arrays in screens.items()}
    scores = {recipe: float(sum(per_seed[seed][recipe][task]["nll"]
                                for seed in (0, 1) for task, _, _ in TASKS))
              for recipe in ("A", "B")}
    return {"selected_recipe": "A" if scores["A"] <= scores["B"] else "B",
            "total_raw_nll": scores, "per_seed": per_seed}


def _network_a(input_width: int, output_width: int, hidden_width: int):
    return torch.nn.Sequential(
        torch.nn.Linear(input_width, hidden_width), torch.nn.ReLU(),
        torch.nn.Linear(hidden_width, hidden_width), torch.nn.ReLU(),
        torch.nn.Linear(hidden_width, output_width),
    )


def _a_forward(path: Path, expected_sha: str, x: np.ndarray,
               purpose: str, width: int) -> np.ndarray:
    if file_sha(path) != expected_sha:
        raise PermissionError("A checkpoint SHA differs")
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(archive["meta"].item())
        if (meta.get("schema") != "team-c-head-checkpoint-v1"
                or meta.get("purpose") != purpose
                or meta.get("selected_columns") != 83
                or meta.get("receipt", {}).get("selected_columns") != 83):
            raise PermissionError("A checkpoint identity or full-input width differs")
        spec = meta["spec"]
        if (spec.get("hidden_width") != 64
                or spec.get("max_epochs") != 30
                or len(meta["state_keys"]) != 6):
            raise PermissionError("A checkpoint architecture differs")
        model = _network_a(83, width, 64)
        state = {key: torch.from_numpy(archive[f"state_{i}"].copy())
                 for i, key in enumerate(meta["state_keys"])}
        model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_num_threads(1)
    with torch.no_grad():
        logits = model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
        return torch.softmax(logits.double(), dim=1).numpy()


def _resolve_private(root: Path, relative: str) -> Path:
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise PermissionError("F package path escapes private root")
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)):
        raise PermissionError("F package path escapes private root")
    return path


def _f_raw(receipt: dict, private_root: Path,
           x83: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    recipe, seed = receipt["selected_recipe"], receipt["seed"]
    if x83.dtype != np.float32 or x83.ndim != 2 or x83.shape[1] != 83:
        raise ValueError("F replay requires admitted 83-column float32 features")
    if recipe == "A":
        return tuple(_a_forward(
            _resolve_private(private_root, receipt["packages"][task]["relative_path"]),
            receipt["packages"][task]["sha256"], x83, task, width)
            for task, width, _ in TASKS)
    if recipe == "B":
        entry = receipt["packages"]["B_source_bundle"]
        folder = _resolve_private(private_root, entry["relative_directory"])
        names = ("encoder.pt", "head_0.joblib", "head_1.joblib")
        model, heads = restore_source(folder / names[0],
                                      (folder / names[1], folder / names[2]),
                                      tuple(entry["sha256"][name] for name in names))
        if any(not np.array_equal(heads[i].classes_, np.arange(k))
               for i, (_, k, _) in enumerate(TASKS)):
            raise PermissionError("B task head class order differs")
        return forward_u0(model, heads, x83)
    raise PermissionError("F recipe differs from frozen A/B set")


def restore_f_predictions(receipt: dict, private_root: Path,
                          x83: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct both full-input F probabilities from private weights."""
    raw = _f_raw(receipt, private_root, x83)
    result = []
    for i, (task, _, cal_task) in enumerate(TASKS):
        entry = receipt["packages"][f"calibrator_{task}"]
        cal = _read_bound_json(_resolve_private(private_root, entry["relative_path"]),
                               entry["sha256"])
        if (cal.get("schema") != "pcrl-portfolio-global-temp-v1"
                or cal.get("task") != cal_task
                or cal.get("model_seed") != receipt["seed"]
                or cal.get("fit_role") != "CALIBRATION_HELDOUT"):
            raise PermissionError("F calibrator identity differs")
        result.append(apply_inverse_temperature(raw[i], float(cal["alpha"])))
    return tuple(result)


def gate_metrics(arrays: dict[str, np.ndarray]) -> dict:
    _aligned_people(arrays)
    out = {arm: {} for arm in ("F", "U0", "Ucal")}
    for i, (task, width, label_key) in enumerate(TASKS, 1):
        for arm in out:
            key = f"F_{task}" if arm == "F" else f"{arm}_{i}"
            value = arrays[key]
            if value.shape != (len(arrays["row_id"]), width):
                raise ValueError("gate probability width differs")
            out[arm][task] = task_metrics(value, arrays[label_key])
    margins = {task: gate_against_references(
        out["F"][task], {reference: out[reference][task]
                           for reference in ("U0", "Ucal")})
               for task, _, _ in TASKS}
    return {"metrics": out, "margins": margins,
            "point_feasible": all(value["point_eligible"] for value in margins.values())}


def array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()
