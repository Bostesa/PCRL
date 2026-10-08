"""Hash-bound cache and inherited task-head fit for masked and full-F heads.

The caller is the M0 sealed, S3-queued worker.  It obtains FIT/CAL role-only
arrays from M1's admitted loader before invoking this module.  This module
never opens a role file and has no parameter for SEX, ATTACK_FIT_NEW or
INNER_SELECTION labels.  Both masked and full-83 F heads use the same
inherited 64x64 architecture and fixed H-GLOBAL calibration.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


ROLE_KEYS = frozenset((
    "fit_features", "fit_label", "calibration_features", "calibration_label",
))
SOURCE_KEYS = frozenset((
    "head_pool.py",
    "training.py", "prototype.py", "scorecard_builder.py",
    "release_package.py", "common_eval.py", "safe_data_guard.py",
))


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode() + str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _full_sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _validated_hash_map(value: Mapping[str, str], keys: frozenset[str],
                        label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} hashes require exact frozen keys")
    if not all(_full_sha(item) for item in value.values()):
        raise ValueError(f"{label} hashes require full SHA-256")
    return tuple(sorted(value.items()))


def _validated_spec(spec: object) -> tuple[tuple[str, object], ...]:
    raw = asdict(spec) if is_dataclass(spec) else dict(spec)
    if not raw or not all(isinstance(key, str) for key in raw):
        raise ValueError("head spec needs named fixed parameters")
    _sha(raw)  # Reject unsupported/nonfinite JSON values.
    return tuple(sorted(raw.items()))


@dataclass(frozen=True)
class HeadCacheIdentity:
    cache_key: str
    model_id: str
    mask_id: str
    mask_sha256: str
    purpose: str
    seed: int
    selected_atoms: tuple[str, ...]
    selected_columns: tuple[int, ...]
    source_schema_sha256: str
    campaign_lock_sha256: str
    data_manifest_sha256: str
    role_file_sha256: tuple[tuple[str, str], ...]
    fit_row_unit_sha256: str
    calibration_row_unit_sha256: str
    head_spec: tuple[tuple[str, object], ...]
    source_code_sha256: tuple[tuple[str, str], ...]

    def to_payload(self) -> dict:
        payload = dict(vars(self))
        payload["role_file_sha256"] = dict(self.role_file_sha256)
        payload["head_spec"] = dict(self.head_spec)
        payload["source_code_sha256"] = dict(self.source_code_sha256)
        return payload


def make_head_cache_identity(
    *, mask: Mapping[str, object], purpose: str, seed: int,
    campaign_lock_sha256: str, data_manifest_sha256: str,
    role_file_sha256: Mapping[str, str], fit_row_unit_sha256: str,
    calibration_row_unit_sha256: str, head_spec: object,
    source_code_sha256: Mapping[str, str],
) -> HeadCacheIdentity:
    """Bind exact data/code/mask/model/calibration identities, never width alone."""
    if purpose not in ("income", "occupation_group") or type(seed) is not int or seed < 0:
        raise ValueError("head purpose or seed is invalid")
    if not isinstance(mask, Mapping) or not {
        "mask_id", "mask_sha256", "source_schema_sha256", "atoms", "column_indices",
    } <= set(mask):
        raise ValueError("mask identity is incomplete")
    mask_id = mask["mask_id"]
    atoms = tuple(mask["atoms"])
    columns = tuple(mask["column_indices"])
    schema_sha = mask["source_schema_sha256"]
    if (not isinstance(mask_id, str) or not mask_id
            or not _full_sha(schema_sha) or not _full_sha(mask["mask_sha256"])
            or columns != tuple(sorted(set(columns)))
            or any(type(column) is not int or not 0 <= column < 83
                   for column in columns)
            or not columns or len(atoms) != len(set(atoms))):
        raise ValueError("mask identity or source columns are invalid")
    canonical_mask = {
        "source_schema_sha256": schema_sha,
        "atoms": atoms, "column_indices": columns,
    }
    if _sha(canonical_mask) != mask["mask_sha256"]:
        raise ValueError("mask identity SHA differs from source groups")
    if not all(_full_sha(value) for value in (
        campaign_lock_sha256, data_manifest_sha256,
        fit_row_unit_sha256, calibration_row_unit_sha256,
    )):
        raise ValueError("campaign or row/group identity SHA is invalid")
    roles = _validated_hash_map(role_file_sha256, ROLE_KEYS, "role file")
    sources = _validated_hash_map(source_code_sha256, SOURCE_KEYS, "source code")
    spec = _validated_spec(head_spec)
    key_payload = {
        "schema": "pcrl-focused-head-cache-key-v1",
        "mask_id": mask_id, "mask_sha256": mask["mask_sha256"],
        "atoms": atoms, "column_indices": columns,
        "source_schema_sha256": schema_sha,
        "purpose": purpose, "seed": seed,
        "campaign_lock_sha256": campaign_lock_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "role_file_sha256": dict(roles),
        "fit_row_unit_sha256": fit_row_unit_sha256,
        "calibration_row_unit_sha256": calibration_row_unit_sha256,
        "head_spec": dict(spec), "source_code_sha256": dict(sources),
    }
    cache_key = _sha(key_payload)
    return HeadCacheIdentity(
        cache_key, f"FOCUSED:{purpose}:{cache_key[:24]}", str(mask_id),
        str(mask["mask_sha256"]), purpose, seed, atoms, columns,
        str(schema_sha), campaign_lock_sha256, data_manifest_sha256,
        roles, fit_row_unit_sha256, calibration_row_unit_sha256,
        spec, sources,
    )


def _inherited_modules(identity: HeadCacheIdentity):
    """Verify exact inherited source bytes before loading fit code."""
    project = Path(__file__).resolve().parents[3]
    c_dir = project / "results" / "pcrl_method_portfolio_v1" / "teams" / "c"
    shared = project / "results" / "pcrl_method_portfolio_v1" / "shared"
    source_map = dict(identity.source_code_sha256)
    for name, expected in source_map.items():
        path = (Path(__file__).resolve() if name == "head_pool.py" else
                (shared if name in ("common_eval.py", "safe_data_guard.py")
                 else c_dir) / name)
        if _file_sha(path) != expected:
            raise PermissionError(f"inherited {name} source differs from cache identity")
    for directory in (str(c_dir), str(shared)):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    training = importlib.import_module("training")
    package = importlib.import_module("release_package")
    common = importlib.import_module("common_eval")
    for module, expected_path in (
        (training, c_dir / "training.py"),
        (package, c_dir / "release_package.py"),
        (common, shared / "common_eval.py"),
    ):
        if Path(module.__file__).resolve() != expected_path.resolve():
            raise PermissionError("inherited module resolved outside pinned source path")
    return training, package, common


def _validate_arrays(identity: HeadCacheIdentity, fit_x, fit_y, fit_row_id,
                     fit_unit, cal_x, cal_y, cal_row_id, cal_unit,
                     feature_names) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(item) for item in (
        fit_x, fit_y, fit_row_id, fit_unit,
        cal_x, cal_y, cal_row_id, cal_unit,
    ))
    x, y, row, unit, cx, cy, crow, cunit = arrays
    classes = 2 if identity.purpose == "income" else 6
    if (x.dtype != np.float32 or cx.dtype != np.float32
            or x.ndim != 2 or cx.ndim != 2
            or x.shape[1] != 83 or cx.shape[1] != 83
            or not np.isfinite(x).all() or not np.isfinite(cx).all()):
        raise ValueError("head fit needs admitted finite float32 83-column FIT/CAL features")
    if (y.dtype != np.int64 or cy.dtype != np.int64
            or y.shape != (len(x),) or cy.shape != (len(cx),)
            or np.any(y < 0) or np.any(y >= classes)
            or np.any(cy < 0) or np.any(cy >= classes)):
        raise ValueError("head fit needs aligned permitted task labels")
    if (row.dtype != np.int64 or unit.dtype != np.int64
            or crow.dtype != np.int64 or cunit.dtype != np.int64
            or row.shape != (len(x),) or unit.shape != row.shape
            or crow.shape != (len(cx),) or cunit.shape != crow.shape):
        raise ValueError("head fit row/group identity arrays are malformed")
    if _array_sha(np.column_stack((row, unit)).astype(np.int64)) != identity.fit_row_unit_sha256:
        raise ValueError("FIT row/group pair differs from admitted identity")
    if _array_sha(np.column_stack((crow, cunit)).astype(np.int64)) != identity.calibration_row_unit_sha256:
        raise ValueError("CAL row/group pair differs from admitted identity")
    if hashlib.sha256("\n".join(tuple(feature_names)).encode()).hexdigest() != identity.source_schema_sha256:
        raise ValueError("head feature schema differs from admitted columns")
    return arrays


def _write_once_json(path: Path, value: dict) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    sha = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("private head-pool receipt differs from existing bytes")
        return sha
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)
    return sha


def _check_existing_cache(home: Path, *, cache_key: str) -> dict | None:
    """Accept a resumed cache only when its completed receipt binds every file."""
    names = ("head.npz", "calibrator.json", "release_package.json")
    receipt_path = home / "HEAD_POOL_RECEIPT.json"
    if not receipt_path.exists():
        if any((home / name).exists() for name in names):
            raise PermissionError("checkpoint without immutable receipt requires a dated repair")
        return None
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError) as exc:
        raise PermissionError("immutable cache receipt cannot be read") from exc
    if (receipt.get("schema") != "pcrl-focused-head-pool-receipt-v1"
            or receipt.get("cache_identity", {}).get("cache_key") != cache_key):
        raise PermissionError("immutable cache receipt identity changed")
    for name, field in (("head.npz", "head_sha256"),
                        ("calibrator.json", "calibrator_sha256"),
                        ("release_package.json", "package_sha256")):
        path = home / name
        if not path.is_file() or _file_sha(path) != receipt.get(field):
            raise PermissionError(f"immutable cache {name} checkpoint hash changed")
    return receipt


@dataclass(frozen=True)
class FittedPackage:
    cache_key: str
    model_id: str
    head_sha256: str
    calibrator_sha256: str
    package_sha256: str
    receipt_sha256: str
    relative_package_path: str
    restored_release: object
    fitted_now: bool


def fit_or_restore_task_package(
    identity: HeadCacheIdentity, *,
    fit_x: np.ndarray, fit_y: np.ndarray, fit_row_id: np.ndarray,
    fit_unit: np.ndarray, calibration_x: np.ndarray,
    calibration_y: np.ndarray, calibration_row_id: np.ndarray,
    calibration_unit: np.ndarray, feature_names: tuple[str, ...],
    private_root: str | Path, head_spec: object,
    assert_queue_admitted: Callable[[], None],
) -> FittedPackage:
    """Fit or restore one exact purpose/mask/seed/F package under M0's gate.

    The queue gate must run before this worker loads role arrays.  It is called
    again here before touching them, and absence of a callable fails closed.
    The caller must supply arrays only from M1's verified SafeRoleStore.
    """
    if not callable(assert_queue_admitted):
        raise PermissionError("an effective S3 queue admission callback is required")
    assert_queue_admitted()
    if _validated_spec(head_spec) != identity.head_spec:
        raise PermissionError("head architecture differs from cache identity")
    arrays = _validate_arrays(
        identity, fit_x, fit_y, fit_row_id, fit_unit,
        calibration_x, calibration_y, calibration_row_id,
        calibration_unit, feature_names,
    )
    x, y, row, unit, cx, cy, crow, cunit = arrays
    root = Path(private_root).resolve(strict=True)
    training, package, common = _inherited_modules(identity)
    if type(head_spec) is not training.HeadSpec:
        raise ValueError("fit must use the inherited fixed HeadSpec object")
    home = root / identity.campaign_lock_sha256 / "head_pool" / identity.cache_key
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    existing_receipt = _check_existing_cache(home, cache_key=identity.cache_key)
    head_path = home / "head.npz"
    cal_path = home / "calibrator.json"
    package_path = home / "release_package.json"
    receipt_path = home / "HEAD_POOL_RECEIPT.json"
    fitted_now = False
    if existing_receipt is not None:
        head_sha = existing_receipt["head_sha256"]
        head = training.restore_head(head_path, expected_sha256=head_sha,
                                     private_root=root)
    else:
        head = training.fit_task_head(
            x, y, unit, tuple(feature_names), identity.selected_atoms,
            purpose=identity.purpose, model_id=identity.model_id,
            seed=identity.seed, spec=head_spec,
        )
        head_sha = training.save_head(head, head_path, private_root=root)
        fitted_now = True
    if (head.model_id != identity.model_id or head.purpose != identity.purpose
            or head.selected_atoms != identity.selected_atoms
            or head.selected_columns != len(identity.selected_columns)
            or head.receipt.get("seed") != identity.seed):
        raise ValueError("cached head differs from exact mask/purpose/seed identity")
    raw = head.predict_proba(cx[:, identity.selected_columns])
    task = "occupation" if identity.purpose == "occupation_group" else "income"
    if existing_receipt is not None:
        cal_sha = existing_receipt["calibrator_sha256"]
        cal_record = common.load_calibrator(cal_path, cal_sha)
    else:
        cal_record = common.fit_global_temperature(
            raw, cy, task=task, arm_id=identity.model_id,
            model_seed=identity.seed, row_id=crow, unit=cunit,
        )
        cal_sha = common.save_calibrator(cal_record, cal_path)
    if (cal_record.get("task") != task
            or cal_record.get("arm_id") != identity.model_id
            or cal_record.get("model_seed") != identity.seed
            or cal_record.get("fit_row_unit_pair_sha256") !=
            identity.calibration_row_unit_sha256):
        raise ValueError("cached calibrator differs from exact CAL identity")
    if existing_receipt is not None:
        package_sha = existing_receipt["package_sha256"]
    else:
        package_sha = package.save_release_package(
            head, head_path, head_sha, cal_record, cal_path, cal_sha,
            package_path=package_path, private_root=root,
            campaign_lock_sha256=identity.campaign_lock_sha256,
            data_manifest_sha256=identity.data_manifest_sha256,
            source_role_hashes=dict(identity.role_file_sha256),
        )
    restored = package.restore_release_package(
        package_path, expected_sha256=package_sha, private_root=root,
        expected_campaign_lock_sha256=identity.campaign_lock_sha256,
        expected_data_manifest_sha256=identity.data_manifest_sha256,
        expected_source_role_hashes=dict(identity.role_file_sha256),
        load_calibrator=common.load_calibrator,
        apply_global_temperature=common.apply_global_temperature,
    )
    if (restored.purpose != identity.purpose
            or restored.selected_atoms != identity.selected_atoms):
        raise ValueError("restored deployment differs from fitted head identity")
    receipt = {
        "schema": "pcrl-focused-head-pool-receipt-v1",
        "cache_identity": identity.to_payload(),
        "head_sha256": head_sha,
        "calibrator_sha256": cal_sha,
        "package_sha256": package_sha,
        "relative_package_path": package_path.relative_to(root).as_posix(),
        "fit_rows": len(x), "calibration_rows": len(cx),
        "fit_x_sha256": _array_sha(x), "fit_y_sha256": _array_sha(y),
        "calibration_x_sha256": _array_sha(cx),
        "calibration_y_sha256": _array_sha(cy),
    }
    receipt_sha = _write_once_json(receipt_path, receipt)
    return FittedPackage(
        identity.cache_key, identity.model_id, head_sha, cal_sha,
        package_sha, receipt_sha, package_path.relative_to(root).as_posix(),
        restored, fitted_now,
    )
