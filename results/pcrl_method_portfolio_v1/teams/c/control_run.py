"""Team C arm runner over S2-admitted role files and S3's heavy-job slot.

No inherited Adult loader is imported. Every source file is accessed through
SafeRoleStore. The caller must supply the prefit campaign seal, a fixed
H-GLOBAL-TEMP fitter, and the global heavy-process context manager. Raw views
and predictions returned here are private; S2 owns attacks and endpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
import subprocess
import tempfile
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np

from controls import ArmPlan, ControlPlan, build_simple_plan, build_task_only_plan
from prototype import ATOM_ORDER, AdultGroupedSchema, release_pair
from release_package import restore_release_package, save_release_package
from s2_bridge import to_s2_attack_views
from task_scorecard import build_task_scorecard
from training import HeadSpec, fit_task_head, restore_head, save_head
from view_custody import restore_view_bundle, save_view_bundle


ROLES = (
    "OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT", "ATTACK_FIT_NEW",
    "INNER_SELECTION",
)
FIELDS = {
    "OSF_DEFENSE_FIT": ("features", "income", "occupation", "sex"),
    "CALIBRATION_HELDOUT": ("features", "income", "occupation"),
    "ATTACK_FIT_NEW": ("features",),
    "INNER_SELECTION": ("features",),
}
PRIVACY_FIT_FIELDS = {
    role: FIELDS[role]
    for role in ("OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT")
}
TASK_ONLY_FIELDS = {
    "OSF_DEFENSE_FIT": ("features", "income", "occupation"),
    "CALIBRATION_HELDOUT": ("features", "income", "occupation"),
}
TASK_ONLY_FORWARD_FIELDS = {
    "ATTACK_FIT_NEW": ("features",),
    "INNER_SELECTION": ("features",),
}
PROCEDURE = {"income": "defense_fit", "occupation": "defense_fit", "sex": "defense_fit"}
SOURCE_TIP = "1baf5bbdaf59cfa6a664cae07d652a4712bdfdc5"
SOURCE_FILES = (
    "controls.py", "evaluation.py", "control_run.py", "prototype.py",
    "selector.py", "scorecard_builder.py", "training.py", "release_package.py",
    "s2_bridge.py", "task_scorecard.py",
    "view_custody.py",
)


class RoleStore(Protocol):
    manifest_path: Path
    manifest: Mapping[str, object]

    def read(self, role: str, field: str, procedure: str | None = None) -> dict: ...


@dataclass(frozen=True)
class RoleData:
    role: str
    X: np.ndarray
    feature_names: tuple[str, ...]
    row_id: np.ndarray
    unit: np.ndarray
    labels: Mapping[str, np.ndarray]
    file_hashes: Mapping[str, str]


@dataclass(frozen=True)
class Admission:
    roles: Mapping[str, RoleData]
    data_manifest_sha256: str
    source_file_sha256: str
    s4_audit_receipt_sha256: str
    role_file_hashes: Mapping[str, str]


@dataclass(frozen=True)
class ArmOutput:
    arm: ArmPlan
    status: str
    attack_release: Mapping[str, Mapping[str, object]] | None
    inner_release: Mapping[str, Mapping[str, object]] | None
    attack_views: Mapping[str, object] | None
    inner_views: Mapping[str, object] | None
    head_ids: tuple[str, str] | None
    calibrator_ids: tuple[str, str] | None
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class RunOutput:
    plan: ControlPlan
    arms: Mapping[str, ArmOutput]
    scorecard_receipt: Mapping[str, object]
    role_file_hashes: Mapping[str, str]
    campaign_lock_sha256: str
    head_spec_sha256: str


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_spec(spec: HeadSpec) -> str:
    return hashlib.sha256(
        json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha_arr(value: np.ndarray) -> str:
    """Match S2's public contiguous-array identity convention."""
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode() + str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _hex_digest(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _check_seal(
    lock_path: Path, lock_sha256: str, *, expected_manifest_sha256: str,
    expected_head_spec_sha256: str, local_seal_receipt_path: Path,
) -> str:
    """Reject any draft, stale source, or uncommitted science authorization."""
    if not _hex_digest(lock_sha256) or _sha_file(lock_path) != lock_sha256:
        raise PermissionError("campaign lock/local seal hash is absent or changed")
    if lock_path.name != "CAMPAIGN_LOCK.json":
        raise PermissionError("a CAMPAIGN_LOCK.json is required")
    lock = json.loads(lock_path.read_text())
    if (lock.get("schema") != "pcrl-method-portfolio-campaign-lock-v1"
            or lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"
            or lock.get("adult_fit_permission") is not True):
        raise PermissionError("campaign lock is not effective for Adult fits")
    if lock.get("source_tip") != SOURCE_TIP:
        raise PermissionError("campaign lock changed the pinned source tip")
    admission = lock.get("data_admission", {})
    if admission.get("manifest_sha256") != expected_manifest_sha256:
        raise PermissionError("campaign lock does not bind the admitted data manifest")
    audit_sha = admission.get("s4_audit_receipt_sha256")
    if not isinstance(audit_sha, str) or not _hex_digest(audit_sha):
        raise PermissionError("campaign lock lacks the independent S4 audit hash")
    audit_path = lock_path.with_name("S4_ROLE_EXPORT_VERIFICATION.json")
    if _sha_file(audit_path) != audit_sha:
        raise PermissionError("independent S4 audit artifact changed")
    audit = json.loads(audit_path.read_text())
    if (audit.get("schema") != "pcrl-s4-independent-role-export-audit-v1"
            or audit.get("status") != "PASS"
            or audit.get("source_sha256") !=
            "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
            or audit.get("export_npz_count") != 25
            or audit.get("barred_id_overlap") != 0
            or audit.get("export_group_overlap") != 0):
        raise PermissionError("independent S4 role admission did not pass")
    family = lock.get("families", {}).get("C", {})
    if family.get("head_spec_sha256") != expected_head_spec_sha256:
        raise PermissionError("campaign lock does not bind Team C head spec")
    method_path = Path(__file__).with_name("METHOD_SPEC.md")
    if family.get("method_spec_sha256") != _sha_file(method_path):
        raise PermissionError("campaign lock does not bind Team C method spec")
    source_map = family.get("source_code_sha256")
    if not isinstance(source_map, dict) or set(source_map) != set(SOURCE_FILES):
        raise PermissionError("campaign lock lacks the exact Team C source map")
    if any(source_map[name] != _sha_file(Path(__file__).with_name(name))
           for name in SOURCE_FILES):
        raise PermissionError("Team C source differs from campaign lock")
    if local_seal_receipt_path.name != "LOCAL_SEAL_RECEIPT.json":
        raise PermissionError("unrecognized local Git seal receipt")
    if local_seal_receipt_path.resolve().parent != lock_path.resolve().parent:
        raise PermissionError("local seal receipt is not adjacent to campaign lock")
    seal = json.loads(local_seal_receipt_path.read_text())
    commit_sha = seal.get("commit_sha", "")
    if (not isinstance(commit_sha, str) or len(commit_sha) != 40
            or any(ch not in "0123456789abcdef" for ch in commit_sha)
            or seal.get("campaign_lock_sha256") != lock_sha256):
        raise PermissionError("local Git seal does not bind this campaign lock")
    relative_lock = lock_path.resolve().relative_to(lock_path.parents[2].resolve())
    committed = subprocess.run(
        ["git", "-C", str(lock_path.parents[2]), "show", f"{commit_sha}:{relative_lock}"],
        capture_output=True, check=False,
    )
    if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != lock_sha256:
        raise PermissionError("campaign lock bytes are not in the sealed Git commit")
    return audit_sha


def admit_control_roles(
    store: RoleStore, *, expected_manifest_sha256: str,
    scope: str = "privacy",
) -> Admission:
    """Read only this worker's admitted fields through S2's verified store."""
    if scope not in ("privacy", "task_only", "task_only_forward"):
        raise ValueError("unrecognized Team C worker scope")
    allowed_fields = {
        "privacy": PRIVACY_FIT_FIELDS,
        "task_only": TASK_ONLY_FIELDS,
        "task_only_forward": TASK_ONLY_FORWARD_FIELDS,
    }[scope]
    selected_roles = tuple(role for role in ROLES if role in allowed_fields)
    if not _hex_digest(expected_manifest_sha256):
        raise PermissionError("data manifest hash must be a full SHA-256")
    manifest_path = Path(store.manifest_path)
    if _sha_file(manifest_path) != expected_manifest_sha256:
        raise PermissionError("admitted data manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest != store.manifest:
        raise PermissionError("role store holds a different data manifest revision")
    if manifest.get("schema") != "pcrl-method-portfolio-data-access-v1":
        raise PermissionError("unrecognized role manifest")
    if manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise PermissionError("role files are not independently admitted")
    s4_sha = manifest.get("custodian_export", {}).get("s4_report_sha256", "")
    if not _hex_digest(s4_sha):
        raise PermissionError("role manifest lacks the independent S4 report hash")
    source_hash = str(manifest.get("source", {}).get("source_npz_sha256", ""))
    if not _hex_digest(source_hash) or source_hash != (
        "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
    ):
        raise PermissionError("pinned Adult source identity is absent or changed")
    specs = manifest.get("real_file_specs", {})
    for role in selected_roles:
        for field in allowed_fields[role]:
            if f"{role}:{field}" not in specs:
                raise PermissionError(f"role-only file not admitted: {role}/{field}")
    roles = {}
    hashes = {}
    schema = AdultGroupedSchema()
    for role in selected_roles:
        features = store.read(role, "features")
        row_id = np.asarray(features["row_id"], dtype=np.int64)
        unit = np.asarray(features["unit"], dtype=np.int64)
        X = np.asarray(features["X"])
        names = tuple(str(n) for n in features["feature_names"])
        schema.validate_feature_names(names)
        if (
            X.shape != (len(row_id), 83) or X.dtype != np.float32
            or unit.shape != row_id.shape
            or not np.isfinite(X).all() or len(np.unique(row_id)) != len(row_id)
        ):
            raise PermissionError(f"role-only feature content invalid: {role}")
        labels = {}
        for field in allowed_fields[role]:
            spec = specs[f"{role}:{field}"]
            if not _hex_digest(str(spec.get("sha256", ""))):
                raise PermissionError("role file has no full hash")
            hashes[f"{role}:{field}"] = spec["sha256"]
            if field == "features":
                continue
            procedure = (
                "defense_fit" if role == "OSF_DEFENSE_FIT" else "calibration"
            )
            item = store.read(role, field, procedure)
            if not np.array_equal(item["row_id"], row_id) or not np.array_equal(item["unit"], unit):
                raise PermissionError(f"{role}/{field} label identities differ from features")
            labels[field] = np.asarray(item[field], dtype=np.int64)
        roles[role] = RoleData(role, X, names, row_id, unit, labels, {
            field: hashes[f"{role}:{field}"] for field in allowed_fields[role]
        })
    # S2's store checks pinned role ID/group hashes. Here we also reject any
    # cross-role group overlap before scientific fitting.
    for i, role_a in enumerate(selected_roles):
        for role_b in selected_roles[i + 1:]:
            if np.intersect1d(roles[role_a].unit, roles[role_b].unit).size:
                raise PermissionError("admitted scientific roles share a person group")
    return Admission(roles, expected_manifest_sha256, source_hash, s4_sha, hashes)


def _private_alignment(role: RoleData) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: {"row_id": role.row_id, "unit": role.unit}
        for name in ("income", "occupation_group", "coalition")
    }


def _write_json_once(path: Path, value: Mapping[str, object]) -> None:
    """Publish one private identity receipt without replacing prior bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise PermissionError("private receipt identity collision")
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _team_c_fit_lock(private_checkpoint_root: Path, campaign_lock_sha256: str):
    """Serialize exact head/calibrator/package identities across C workers."""
    path = private_checkpoint_root / "team_c" / campaign_lock_sha256 / "fit.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _publish_fit_handoff(
    arm: ArmOutput, admission: Admission, *, seed: int,
    worker_scope: str, scorecard_sha256: str,
    campaign_lock_sha256: str, head_spec_sha256: str,
    private_checkpoint_root: Path,
) -> tuple[Path, str]:
    """Freeze one mask and both deployment packages before screen-row reads."""
    if (arm.status != "FROZEN_PACKAGE_READY_FOR_FORWARD"
            or len(arm.receipt.get("package_units", ())) != 2):
        raise PermissionError("C fit did not freeze both release packages")
    payload = {
        "schema": "team-c-fit-handoff-v1",
        "worker_scope": worker_scope,
        "worker_pid": os.getpid(),
        "seed": seed,
        "arm_id": arm.arm.arm_id,
        "campaign_lock_sha256": campaign_lock_sha256,
        "data_manifest_sha256": admission.data_manifest_sha256,
        "head_spec_sha256": head_spec_sha256,
        "s4_audit_receipt_sha256": admission.s4_audit_receipt_sha256,
        "selected_atoms": {
            "income": list(arm.arm.income_atoms),
            "occupation_group": list(arm.arm.occupation_atoms),
        },
        "scorecard_sha256": scorecard_sha256,
        "fit_cal_role_file_hashes": dict(admission.role_file_hashes),
        "arm_receipt": dict(arm.receipt),
        "head_ids": arm.head_ids,
        "calibrator_ids": arm.calibrator_ids,
    }
    path = (
        private_checkpoint_root / "team_c" / campaign_lock_sha256 /
        "fit_handoff" / f"{arm.arm.arm_id}__s{seed}.json"
    )
    _write_json_once(path, payload)
    return path, _sha_file(path)


def _fit_or_restore_head(
    fitting: RoleData, *, seed: int, purpose: str, field: str,
    selected_atoms: tuple[str, ...], spec: HeadSpec, model_id: str,
    campaign_lock_sha256: str, data_manifest_sha256: str,
    private_checkpoint_root: Path,
) -> tuple[object, str, Path, str]:
    """Use an exact-identity checkpoint or fit once under a crash marker."""
    namespace = private_checkpoint_root / "team_c" / campaign_lock_sha256
    identity = {
        "schema": "team-c-head-fit-identity-v1",
        "campaign_lock_sha256": campaign_lock_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "fit_feature_sha256": fitting.file_hashes["features"],
        "fit_label_sha256": fitting.file_hashes[field],
        "head_spec_sha256": _sha_spec(spec),
        "seed": seed, "purpose": purpose,
        "selected_atoms": list(selected_atoms), "model_id": model_id,
    }
    receipt_path = namespace / "head_receipts" / f"{model_id}.json"
    pending_path = namespace / "head_pending" / f"{model_id}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("identity") != identity:
            raise PermissionError("existing head receipt has a different scientific identity")
        relative = receipt.get("checkpoint_relative_path")
        checksum = receipt.get("checkpoint_sha256")
        if not isinstance(relative, str) or not _hex_digest(str(checksum)):
            raise PermissionError("existing head receipt is incomplete")
        checkpoint_path = private_checkpoint_root / relative
        if not checkpoint_path.resolve().is_relative_to(private_checkpoint_root.resolve()):
            raise PermissionError("head receipt points outside private custody")
        head = restore_head(
            checkpoint_path, expected_sha256=checksum,
            private_root=private_checkpoint_root,
        )
        if (head.model_id != model_id or head.purpose != purpose
                or tuple(head.selected_atoms) != selected_atoms
                or _sha_spec(head.spec) != _sha_spec(spec)):
            raise PermissionError("restored head differs from its locked identity")
        return head, checksum, checkpoint_path, "RESTORED_EXACT_IDENTITY"
    if pending_path.exists():
        raise PermissionError("prior head fit is incomplete; S3 must adjudicate before retry")
    _write_json_once(pending_path, identity)
    head = fit_task_head(
        fitting.X, fitting.labels[field], fitting.unit,
        fitting.feature_names, selected_atoms, purpose=purpose,
        model_id=model_id, seed=seed, spec=spec,
    )
    checkpoint_path = namespace / "heads" / f"{model_id}.npz"
    checksum = save_head(head, checkpoint_path, private_root=private_checkpoint_root)
    receipt = {
        "identity": identity,
        "checkpoint_relative_path": str(checkpoint_path.relative_to(private_checkpoint_root)),
        "checkpoint_sha256": checksum,
        "fit_receipt": head.receipt,
    }
    _write_json_once(receipt_path, receipt)
    pending_path.unlink()
    return head, checksum, checkpoint_path, "COMPLETED"


def _fit_or_restore_calibrator(
    head: object, calibration: RoleData, *, field: str, seed: int,
    selected_atoms: tuple[str, ...], schema: AdultGroupedSchema,
    campaign_lock_sha256: str, data_manifest_sha256: str,
    private_checkpoint_root: Path,
    fit_calibrator: Callable[..., dict],
    save_calibrator: Callable[[dict, Path], str],
    load_calibrator: Callable[[Path, str], dict],
) -> tuple[dict, Path, str, str]:
    """Bind one S2 H-GLOBAL calibrator to exact ordered held-out people."""
    task = "occupation" if head.purpose == "occupation_group" else "income"
    if field != task:
        raise ValueError("calibration task and fitted head differ")
    raw = head.predict_proba(calibration.X[:, schema.column_indices(selected_atoms)])
    pair_sha = _sha_arr(np.column_stack((calibration.row_id, calibration.unit)))
    expected = {
        "schema": "pcrl-portfolio-global-temp-v1",
        "task": task, "arm_id": head.model_id, "model_seed": seed,
        "fit_role": "CALIBRATION_HELDOUT", "draw_id": "deterministic",
        "fit_row_unit_pair_sha256": pair_sha,
        "input_probability_sha256": _sha_arr(raw),
        "task_label_sha256": _sha_arr(calibration.labels[field]),
    }
    namespace = private_checkpoint_root / "team_c" / campaign_lock_sha256
    path = namespace / "calibrators" / f"{head.model_id}.json"
    receipt_path = namespace / "calibrator_receipts" / f"{head.model_id}.json"
    pending_path = namespace / "calibrator_pending" / f"{head.model_id}.json"
    identity = {
        "campaign_lock_sha256": campaign_lock_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "calibration_feature_sha256": calibration.file_hashes["features"],
        "calibration_label_sha256": calibration.file_hashes[field],
        "head_model_id": head.model_id,
        "head_spec_sha256": _sha_spec(head.spec),
        "selected_atoms": list(selected_atoms),
        "expected_record_identity": expected,
    }
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("identity") != identity:
            raise PermissionError("existing calibrator receipt has different identity")
        checksum = receipt.get("calibrator_sha256")
        if not _hex_digest(str(checksum)):
            raise PermissionError("calibrator receipt has no full hash")
        record = load_calibrator(path, checksum)
        status = "RESTORED_EXACT_IDENTITY"
    else:
        if pending_path.exists() or path.exists():
            raise PermissionError("prior calibrator fit is incomplete; S3 must adjudicate")
        _write_json_once(pending_path, identity)
        record = fit_calibrator(
            raw, calibration.labels[field], task=task, arm_id=head.model_id,
            model_seed=seed, row_id=calibration.row_id, unit=calibration.unit,
            draw_id="deterministic",
        )
        if any(record.get(key) != value for key, value in expected.items()):
            raise PermissionError("S2 calibrator record differs from admitted role/head")
        checksum = save_calibrator(record, path)
        if _sha_file(path) != checksum:
            raise PermissionError("saved S2 calibrator bytes differ from its hash")
        _write_json_once(receipt_path, {
            "identity": identity, "calibrator_sha256": checksum,
            "calibrator_relative_path": str(path.relative_to(private_checkpoint_root)),
        })
        pending_path.unlink()
        status = "COMPLETED"
    if any(record.get(key) != value for key, value in expected.items()):
        raise PermissionError("restored S2 calibrator differs from admitted role/head")
    if _sha_file(path) != checksum:
        raise PermissionError("calibrator bytes changed after fit")
    return record, path, checksum, status


def _fit_and_calibrate(
    admission: Admission, plan: ControlPlan, *, seed: int, spec: HeadSpec,
    fit_calibrator: Callable[..., dict],
    save_calibrator: Callable[[dict, Path], str],
    load_calibrator: Callable[[Path, str], dict],
    apply_global_temperature: Callable[[np.ndarray, dict], np.ndarray],
    attack_view_type: type,
    heavy_slot: Callable[[str], AbstractContextManager[object]],
    campaign_lock_sha256: str,
    private_checkpoint_root: Path,
    selected_arm_ids: tuple[str, ...] | None,
    forward: bool = False,
) -> dict[str, ArmOutput]:
    if forward:
        raise PermissionError("screen-row forwards require the postfit handoff worker")
    fitting = admission.roles["OSF_DEFENSE_FIT"]
    calibration = admission.roles["CALIBRATION_HELDOUT"]
    schema = AdultGroupedSchema()
    head_cache = {}
    calibration_cache = {}
    result = {}
    requested = set(arm.arm_id for arm in plan.arms) if selected_arm_ids is None else set(selected_arm_ids)
    if requested - {a.arm_id for a in plan.arms}:
        raise ValueError("requested arm is absent from predeclared Team C plan")
    head_spec_sha = _sha_spec(spec)
    for arm in plan.arms:
        if arm.arm_id not in requested:
            result[arm.arm_id] = ArmOutput(
                arm, "NOT_RUN_SCHEDULED_LATER", None, None, None, None, None,
                None, {"reason": "predeclared wave did not include this arm"},
            )
            continue
        if arm.capacity_status.startswith(("PREFIT", "SKIPPED", "NOT_RUN", "DEFERRED")):
            result[arm.arm_id] = ArmOutput(
                arm, arm.capacity_status, None, None, None, None, None,
                None, {"reason": arm.note or arm.capacity_status},
            )
            continue
        packages = {}
        receipt = {"arm_id": arm.arm_id, "seed": seed,
                   "capacity_status": arm.capacity_status,
                   "budget": {"semantic": arm.budget.semantic_counts,
                              "encoded": arm.budget.encoded_counts,
                              "raw_feature_payload_bits_per_person": tuple(
                                  32 * width for width in arm.budget.encoded_counts
                              ),
                              "raw_feature_dtype": "float32"},
                   "budget_match": asdict(arm.budget_match),
                   "campaign_lock_sha256": campaign_lock_sha256,
                   "data_manifest_sha256": admission.data_manifest_sha256,
                   "head_spec_sha256": head_spec_sha,
                   "head_units": [], "calibration_units": []}
        for purpose, field, selected_atoms in (
            ("income", "income", arm.income_atoms),
            ("occupation_group", "occupation", arm.occupation_atoms),
        ):
            key = (seed, purpose, selected_atoms, head_spec_sha,
                   fitting.file_hashes["features"], fitting.file_hashes[field])
            if key not in head_cache:
                mask_digest = hashlib.sha256("|".join(selected_atoms).encode()).hexdigest()[:16]
                model_id = f"team-c-s{seed}-{purpose}-{mask_digest}"
                unit_id = f"C:s{seed}:HEAD:{purpose}:{mask_digest}"
                with heavy_slot(unit_id):
                    head, checkpoint_sha, checkpoint_path, head_status = _fit_or_restore_head(
                        fitting, seed=seed, purpose=purpose, field=field,
                        selected_atoms=selected_atoms, spec=spec, model_id=model_id,
                        campaign_lock_sha256=campaign_lock_sha256,
                        data_manifest_sha256=admission.data_manifest_sha256,
                        private_checkpoint_root=private_checkpoint_root,
                    )
                head_cache[key] = (head, checkpoint_sha, checkpoint_path)
            else:
                head, checkpoint_sha, checkpoint_path = head_cache[key]
                head_status = "ALIASED"
            receipt["head_units"].append({
                "purpose": purpose, "status": head_status, "model_id": head.model_id,
                "source_feature_sha256": fitting.file_hashes["features"],
                "source_label_sha256": fitting.file_hashes[field],
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_relative_path": str(checkpoint_path.relative_to(private_checkpoint_root)),
                "head_receipt": head.receipt if head_status == "COMPLETED" else None,
            })
            cal_key = (key, calibration.file_hashes["features"],
                       calibration.file_hashes[field], "H-GLOBAL-TEMP")
            if cal_key not in calibration_cache:
                unit_id = f"C:s{seed}:CAL:{purpose}:{head.model_id}"
                with heavy_slot(unit_id):
                    cal_record, cal_path, cal_sha, cal_status = _fit_or_restore_calibrator(
                        head, calibration, field=field, seed=seed,
                        selected_atoms=selected_atoms, schema=schema,
                        campaign_lock_sha256=campaign_lock_sha256,
                        data_manifest_sha256=admission.data_manifest_sha256,
                        private_checkpoint_root=private_checkpoint_root,
                        fit_calibrator=fit_calibrator,
                        save_calibrator=save_calibrator,
                        load_calibrator=load_calibrator,
                    )
                calibration_cache[cal_key] = (cal_record, cal_path, cal_sha)
            else:
                cal_record, cal_path, cal_sha = calibration_cache[cal_key]
                cal_status = "ALIASED"
            receipt["calibration_units"].append({
                "purpose": purpose, "status": cal_status,
                "calibrator_id": f"H-GLOBAL-TEMP:{cal_sha[:16]}",
                "calibrator_sha256": cal_sha,
                "calibrator_relative_path": str(cal_path.relative_to(private_checkpoint_root)),
                "calibration_feature_sha256": calibration.file_hashes["features"],
                "calibration_label_sha256": calibration.file_hashes[field],
                "calibration_receipt": cal_record if cal_status == "COMPLETED" else None,
            })
            source_hashes = {
                "fit_features": fitting.file_hashes["features"],
                "fit_label": fitting.file_hashes[field],
                "calibration_features": calibration.file_hashes["features"],
                "calibration_label": calibration.file_hashes[field],
            }
            package_path = (
                private_checkpoint_root / "team_c" / campaign_lock_sha256 /
                "packages" / f"{head.model_id}.json"
            )
            package_unit = f"C:s{seed}:PACKAGE:{purpose}:{head.model_id}"
            with heavy_slot(package_unit):
                if package_path.exists():
                    package_sha = _sha_file(package_path)
                    package_status = "RESTORED_EXACT_IDENTITY"
                else:
                    package_sha = save_release_package(
                        head, checkpoint_path, checkpoint_sha,
                        cal_record, cal_path, cal_sha, package_path=package_path,
                        private_root=private_checkpoint_root,
                        campaign_lock_sha256=campaign_lock_sha256,
                        data_manifest_sha256=admission.data_manifest_sha256,
                        source_role_hashes=source_hashes,
                    )
                    package_status = "COMPLETED"
                packages[purpose] = restore_release_package(
                    package_path, expected_sha256=package_sha,
                    private_root=private_checkpoint_root,
                    expected_campaign_lock_sha256=campaign_lock_sha256,
                    expected_data_manifest_sha256=admission.data_manifest_sha256,
                    expected_source_role_hashes=source_hashes,
                    load_calibrator=load_calibrator,
                    apply_global_temperature=apply_global_temperature,
                )
            receipt.setdefault("package_units", []).append({
                "purpose": purpose, "status": package_status,
                "package_sha256": package_sha,
                "package_relative_path": str(package_path.relative_to(private_checkpoint_root)),
                "source_role_hashes": source_hashes,
            })
        income_release = packages["income"]
        occupation_release = packages["occupation_group"]
        result[arm.arm_id] = ArmOutput(
            arm, "FROZEN_PACKAGE_READY_FOR_FORWARD",
            None, None, None, None,
            (income_release.head_id, occupation_release.head_id),
            (income_release.calibrator_id,
             occupation_release.calibrator_id),
            receipt,
        )
    return result


def run_controls(
    store: RoleStore, *, seed: int, head_spec: HeadSpec,
    expected_head_spec_sha256: str,
    expected_manifest_sha256: str,
    campaign_lock_path: str | Path, campaign_lock_sha256: str,
    local_seal_receipt_path: str | Path,
    fit_calibrator: Callable[..., dict],
    save_calibrator: Callable[[dict, Path], str],
    load_calibrator: Callable[[Path, str], dict],
    apply_global_temperature: Callable[[np.ndarray, dict], np.ndarray],
    attack_view_type: type,
    heavy_slot: Callable[[str], AbstractContextManager[object]],
    private_checkpoint_root: str | Path,
    selected_arm_ids: tuple[str, ...] | None = None,
) -> RunOutput:
    """Fit privacy-card arms; task-only has a separate no-SEX worker.

    S2 must fit attackers/score endpoints separately and charge their units.
    `selected_arm_ids` is for S1's frozen execution waves, not outcome-based
    pruning. A skipped prior-art slot yields a receipt and no fit.
    """
    lock_path = Path(campaign_lock_path)
    sealed_s4_sha = _check_seal(
        lock_path, campaign_lock_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_head_spec_sha256=expected_head_spec_sha256,
        local_seal_receipt_path=Path(local_seal_receipt_path),
    )
    if not _hex_digest(expected_head_spec_sha256) or _sha_spec(head_spec) != expected_head_spec_sha256:
        raise PermissionError("head architecture/budget differ from campaign lock")
    if not all(callable(func) for func in (
        heavy_slot, fit_calibrator, save_calibrator,
        load_calibrator, apply_global_temperature, attack_view_type,
    )):
        raise TypeError("global heavy slot and four S2 calibration functions are required")
    private_checkpoint_root = Path(private_checkpoint_root)
    if private_checkpoint_root.name != ".private":
        raise PermissionError("Team C checkpoints must use the declared .private root")
    if selected_arm_ids is not None and "C-SIMPLE-TASK-SHARED1" in selected_arm_ids:
        raise PermissionError("task-only arm requires its separate no-SEX worker")
    with heavy_slot(f"C:s{seed}:ROLE_ADMISSION"):
        admission = admit_control_roles(
            store, expected_manifest_sha256=expected_manifest_sha256,
            scope="privacy",
        )
    if admission.s4_audit_receipt_sha256 != sealed_s4_sha:
        raise PermissionError("lock and role manifest bind different S4 audits")
    fit = admission.roles["OSF_DEFENSE_FIT"]
    from scorecard_builder import ScorecardProxyCache, build_scorecard
    cache = ScorecardProxyCache()
    with heavy_slot(f"C:s{seed}:SCORECARD:FULL"):
        full_scorecard = build_scorecard(
            fit.X, fit.feature_names, fit.labels["income"],
            fit.labels["occupation"], fit.labels["sex"], fit.unit,
            base_atoms=ATOM_ORDER, optional_atoms=ATOM_ORDER,
            seed=seed, cache=cache,
        )
    plan = build_simple_plan(full_scorecard.card)
    full_card_sha = hashlib.sha256(json.dumps(
        asdict(full_scorecard.card), sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    scorecard_receipt = {
        "seed": seed, "worker_scope": "PRIVACY_FIT_CAL",
        "worker_pid": os.getpid(),
        "full_fit_counts": full_scorecard.fit_counts,
        "full_split": full_scorecard.split_receipt,
        "full_card_sha256": full_card_sha,
        "task_only_status": "DEFERRED_TO_SEPARATE_NO_SEX_WORKER",
        "phase1_xor_premise": "FAILED_C-TEST-SCORECARD-BUILDER-016",
        "defense_feature_sha256": fit.file_hashes["features"],
        "defense_label_sha256": {field: fit.file_hashes[field]
                                  for field in ("income", "occupation", "sex")},
        "data_manifest_sha256": admission.data_manifest_sha256,
        "campaign_lock_sha256": campaign_lock_sha256,
    }
    with _team_c_fit_lock(private_checkpoint_root, campaign_lock_sha256):
        arms = _fit_and_calibrate(
            admission, plan, seed=seed, spec=head_spec,
            fit_calibrator=fit_calibrator,
            save_calibrator=save_calibrator,
            load_calibrator=load_calibrator,
            apply_global_temperature=apply_global_temperature,
            attack_view_type=attack_view_type,
            heavy_slot=heavy_slot,
            campaign_lock_sha256=campaign_lock_sha256,
            private_checkpoint_root=private_checkpoint_root,
            selected_arm_ids=selected_arm_ids,
            forward=False,
        )
    handoffs = {}
    for arm_id, arm in arms.items():
        if arm.status != "FROZEN_PACKAGE_READY_FOR_FORWARD":
            continue
        path, checksum = _publish_fit_handoff(
            arm, admission, seed=seed, worker_scope="PRIVACY_FIT_CAL_NO_SCREEN_ROWS",
            scorecard_sha256=full_card_sha,
            campaign_lock_sha256=campaign_lock_sha256,
            head_spec_sha256=expected_head_spec_sha256,
            private_checkpoint_root=private_checkpoint_root,
        )
        handoffs[arm_id] = {
            "relative_path": str(path.relative_to(private_checkpoint_root)),
            "sha256": checksum,
        }
    scorecard_receipt["fit_handoffs"] = handoffs
    return RunOutput(
        plan, arms, scorecard_receipt, dict(admission.role_file_hashes),
        campaign_lock_sha256, expected_head_spec_sha256,
    )


def run_task_only_control(
    store: RoleStore, *, seed: int, head_spec: HeadSpec,
    expected_head_spec_sha256: str,
    expected_manifest_sha256: str,
    campaign_lock_path: str | Path, campaign_lock_sha256: str,
    local_seal_receipt_path: str | Path,
    fit_calibrator: Callable[..., dict],
    save_calibrator: Callable[[dict, Path], str],
    load_calibrator: Callable[[Path, str], dict],
    apply_global_temperature: Callable[[np.ndarray, dict], np.ndarray],
    attack_view_type: type,
    heavy_slot: Callable[[str], AbstractContextManager[object]],
    private_checkpoint_root: str | Path,
) -> RunOutput:
    """Run TASK-SHARED1 in an OS worker that never requests SEX labels.

    The scheduler must invoke this entry point in a separate process from
    `run_controls`. It reads only FIT/CAL features and task labels, then
    freezes two release packages. The postfit forward worker later reads
    ATT/INNER features; S2 attack jobs later read SEX.
    """
    lock_path = Path(campaign_lock_path)
    sealed_s4_sha = _check_seal(
        lock_path, campaign_lock_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_head_spec_sha256=expected_head_spec_sha256,
        local_seal_receipt_path=Path(local_seal_receipt_path),
    )
    if not _hex_digest(expected_head_spec_sha256) or _sha_spec(head_spec) != expected_head_spec_sha256:
        raise PermissionError("head architecture/budget differ from campaign lock")
    if not all(callable(func) for func in (
        heavy_slot, fit_calibrator, save_calibrator,
        load_calibrator, apply_global_temperature, attack_view_type,
    )):
        raise TypeError("global heavy slot and four S2 calibration functions are required")
    private_checkpoint_root = Path(private_checkpoint_root)
    if private_checkpoint_root.name != ".private":
        raise PermissionError("Team C checkpoints must use the declared .private root")
    with heavy_slot(f"C:s{seed}:TASK_ONLY:ROLE_ADMISSION"):
        admission = admit_control_roles(
            store, expected_manifest_sha256=expected_manifest_sha256,
            scope="task_only",
        )
    if admission.s4_audit_receipt_sha256 != sealed_s4_sha:
        raise PermissionError("lock and role manifest bind different S4 audits")
    fitting = admission.roles["OSF_DEFENSE_FIT"]
    if "sex" in fitting.labels or "sex" in fitting.file_hashes or any(
        field.endswith(":sex") for field in admission.role_file_hashes
    ):
        raise PermissionError("task-only worker received a sensitive-label artifact")
    with heavy_slot(f"C:s{seed}:TASK_ONLY:SCORECARD"):
        card = build_task_scorecard(
            fitting.X, fitting.feature_names, fitting.labels["income"],
            fitting.labels["occupation"], fitting.unit, seed=seed,
        )
    plan = build_task_only_plan(card.task_deltas)
    card_payload = {
        task: {atom: asdict(card.task_deltas[task][atom])
               for atom in card.optional_atoms}
        for task in ("income", "occupation_group")
    }
    card_sha = hashlib.sha256(json.dumps(
        card_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    scorecard_receipt = {
        "seed": seed, "worker_scope": "TASK_ONLY_NO_SEX_READ",
        "worker_pid": os.getpid(),
        "fit_counts": card.fit_counts,
        "split": card.split_receipt,
        "task_only_card_sha256": card_sha,
        "selected_mask": plan.arms[0].mask_pair,
        "fit_feature_sha256": fitting.file_hashes["features"],
        "fit_task_label_sha256": {
            task: fitting.file_hashes[task]
            for task in ("income", "occupation")
        },
        "role_file_hashes": dict(admission.role_file_hashes),
        "data_manifest_sha256": admission.data_manifest_sha256,
        "campaign_lock_sha256": campaign_lock_sha256,
    }
    with _team_c_fit_lock(private_checkpoint_root, campaign_lock_sha256):
        arms = _fit_and_calibrate(
            admission, plan, seed=seed, spec=head_spec,
            fit_calibrator=fit_calibrator,
            save_calibrator=save_calibrator,
            load_calibrator=load_calibrator,
            apply_global_temperature=apply_global_temperature,
            attack_view_type=attack_view_type,
            heavy_slot=heavy_slot,
            campaign_lock_sha256=campaign_lock_sha256,
            private_checkpoint_root=private_checkpoint_root,
            selected_arm_ids=("C-SIMPLE-TASK-SHARED1",),
            forward=False,
        )
    fit_receipt_path, handoff_sha = _publish_fit_handoff(
        arms["C-SIMPLE-TASK-SHARED1"], admission, seed=seed,
        worker_scope="TASK_ONLY_FIT_CAL_NO_SEX_NO_SCREEN_ROWS",
        scorecard_sha256=card_sha,
        campaign_lock_sha256=campaign_lock_sha256,
        head_spec_sha256=expected_head_spec_sha256,
        private_checkpoint_root=private_checkpoint_root,
    )
    scorecard_receipt["fit_handoff_relative_path"] = str(
        fit_receipt_path.relative_to(private_checkpoint_root)
    )
    scorecard_receipt["fit_handoff_sha256"] = handoff_sha
    return RunOutput(
        plan, arms, scorecard_receipt, dict(admission.role_file_hashes),
        campaign_lock_sha256, expected_head_spec_sha256,
    )


def run_forward_from_handoff(
    store: RoleStore, *, arm_id: str, seed: int,
    fit_handoff_sha256: str,
    expected_head_spec_sha256: str,
    expected_manifest_sha256: str,
    campaign_lock_path: str | Path, campaign_lock_sha256: str,
    local_seal_receipt_path: str | Path,
    load_calibrator: Callable[[Path, str], dict],
    apply_global_temperature: Callable[[np.ndarray, dict], np.ndarray],
    attack_view_type: type,
    heavy_slot: Callable[[str], AbstractContextManager[object]],
    private_checkpoint_root: str | Path,
) -> dict:
    """After frozen fit handoff, release admitted ATT/INNER features only.

    This worker never reads SEX. S2's later attack job loads SEX and restores
    the complete private view bundles by their immutable receipt hashes.
    """
    lock_path = Path(campaign_lock_path)
    sealed_s4_sha = _check_seal(
        lock_path, campaign_lock_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_head_spec_sha256=expected_head_spec_sha256,
        local_seal_receipt_path=Path(local_seal_receipt_path),
    )
    if not all(callable(value) for value in (
        load_calibrator, apply_global_temperature, attack_view_type, heavy_slot,
    )):
        raise TypeError("S2 calibrator, view and queue callbacks are required")
    private_checkpoint_root = Path(private_checkpoint_root)
    if private_checkpoint_root.name != ".private":
        raise PermissionError("Team C view custody must use the declared .private root")
    if not _hex_digest(fit_handoff_sha256):
        raise PermissionError("full fit handoff hash is required")
    fit_path = (
        private_checkpoint_root / "team_c" / campaign_lock_sha256 /
        "fit_handoff" / f"{arm_id}__s{seed}.json"
    )
    if _sha_file(fit_path) != fit_handoff_sha256:
        raise PermissionError("frozen C fit handoff changed")
    handoff = json.loads(fit_path.read_text())
    if (handoff.get("schema") != "team-c-fit-handoff-v1"
            or handoff.get("arm_id") != arm_id
            or handoff.get("seed") != seed
            or handoff.get("campaign_lock_sha256") != campaign_lock_sha256
            or handoff.get("data_manifest_sha256") != expected_manifest_sha256
            or handoff.get("head_spec_sha256") != expected_head_spec_sha256
            or handoff.get("s4_audit_receipt_sha256") != sealed_s4_sha
            or not isinstance(handoff.get("worker_pid"), int)
            or handoff["worker_pid"] <= 0
            or handoff["worker_pid"] == os.getpid()
            or handoff.get("worker_scope") not in (
                "PRIVACY_FIT_CAL_NO_SCREEN_ROWS",
                "TASK_ONLY_FIT_CAL_NO_SEX_NO_SCREEN_ROWS",
            )):
        raise PermissionError("C fit handoff has wrong locked identity")
    if (arm_id == "C-SIMPLE-TASK-SHARED1") != handoff["worker_scope"].startswith("TASK_ONLY"):
        raise PermissionError("task-only arm used the wrong fitting worker")
    manifest = json.loads(Path(store.manifest_path).read_text())
    if (_sha_file(Path(store.manifest_path)) != expected_manifest_sha256
            or manifest != store.manifest):
        raise PermissionError("forward worker data manifest changed")
    role_hashes = handoff["fit_cal_role_file_hashes"]
    expected_fit_cal_keys = {
        f"{role}:{field}"
        for role, fields in (
            ("OSF_DEFENSE_FIT", ("features", "income", "occupation") +
             (() if arm_id == "C-SIMPLE-TASK-SHARED1" else ("sex",))),
            ("CALIBRATION_HELDOUT", ("features", "income", "occupation")),
        )
        for field in fields
    }
    specs = manifest.get("real_file_specs", {})
    if (set(role_hashes) != expected_fit_cal_keys
            or any(specs[key]["sha256"] != value for key, value in role_hashes.items())):
        raise PermissionError("fit handoff role hashes differ from admitted manifest")
    selected = handoff.get("selected_atoms", {})
    if set(selected) != {"income", "occupation_group"}:
        raise PermissionError("fit handoff lacks both purpose masks")
    schema = AdultGroupedSchema()
    for purpose in ("income", "occupation_group"):
        atoms = tuple(selected[purpose])
        if atoms != schema.normalize_atoms(atoms):
            raise PermissionError("fit handoff mask is not a canonical grouped mask")
    packages = {}
    package_hashes = {}
    package_units = handoff.get("arm_receipt", {}).get("package_units", ())
    if len(package_units) != 2:
        raise PermissionError("fit handoff lacks two release packages")
    for unit in package_units:
        purpose = unit.get("purpose")
        if purpose not in ("income", "occupation_group") or purpose in packages:
            raise PermissionError("duplicate or unknown release package purpose")
        relative = unit.get("package_relative_path")
        checksum = unit.get("package_sha256")
        sources = unit.get("source_role_hashes")
        if (not isinstance(relative, str) or not _hex_digest(str(checksum))
                or not isinstance(sources, dict)):
            raise PermissionError("release package handoff is incomplete")
        path = private_checkpoint_root / relative
        if not path.resolve().is_relative_to(private_checkpoint_root.resolve()):
            raise PermissionError("release package escapes private custody")
        field = "income" if purpose == "income" else "occupation"
        expected_sources = {
            "fit_features": role_hashes["OSF_DEFENSE_FIT:features"],
            "fit_label": role_hashes[f"OSF_DEFENSE_FIT:{field}"],
            "calibration_features": role_hashes["CALIBRATION_HELDOUT:features"],
            "calibration_label": role_hashes[f"CALIBRATION_HELDOUT:{field}"],
        }
        if sources != expected_sources:
            raise PermissionError("release package source hashes differ from FIT/CAL")
        packages[purpose] = restore_release_package(
            path, expected_sha256=checksum,
            private_root=private_checkpoint_root,
            expected_campaign_lock_sha256=campaign_lock_sha256,
            expected_data_manifest_sha256=expected_manifest_sha256,
            expected_source_role_hashes=expected_sources,
            load_calibrator=load_calibrator,
            apply_global_temperature=apply_global_temperature,
        )
        package_hashes[purpose] = checksum
        if packages[purpose].selected_atoms != tuple(selected[purpose]):
            raise PermissionError("restored package mask differs from frozen fit handoff")
    if (tuple(packages[p].head_id for p in ("income", "occupation_group")) !=
            tuple(handoff["head_ids"]) or
            tuple(packages[p].calibrator_id for p in ("income", "occupation_group")) !=
            tuple(handoff["calibrator_ids"])):
        raise PermissionError("restored package IDs differ from frozen fit handoff")

    with heavy_slot(f"C:s{seed}:FORWARD:ROLE_ADMISSION"):
        forward_admission = admit_control_roles(
            store, expected_manifest_sha256=expected_manifest_sha256,
            scope="task_only_forward",
        )
    if forward_admission.s4_audit_receipt_sha256 != sealed_s4_sha:
        raise PermissionError("forward role admission has a different S4 audit")
    receipts = {}
    for role_name in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        role = forward_admission.roles[role_name]
        unit_id = f"C:s{seed}:FORWARD:{arm_id}:{role_name}"
        with heavy_slot(unit_id):
            wire = release_pair(
                role.X, role.feature_names,
                packages["income"], packages["occupation_group"],
            )
            views = to_s2_attack_views(
                wire, _private_alignment(role), packages, attack_view_type,
            )
            view_path = (
                private_checkpoint_root / "team_c" / campaign_lock_sha256 /
                "views" / f"{arm_id}__s{seed}__{role_name}.npz"
            )
            receipt_path, checksum = save_view_bundle(
                wire, views, npz_path=view_path,
                private_root=private_checkpoint_root,
                role=role_name, arm_id=arm_id, seed=seed,
                campaign_lock_sha256=campaign_lock_sha256,
                data_manifest_sha256=expected_manifest_sha256,
                role_feature_sha256=role.file_hashes["features"],
                package_sha256_by_purpose=package_hashes,
            )
        receipts[role_name] = {
            "receipt_relative_path": str(receipt_path.relative_to(private_checkpoint_root)),
            "receipt_sha256": checksum,
            "role_feature_sha256": role.file_hashes["features"],
            "row_unit_pair_sha256": _sha_arr(np.column_stack((role.row_id, role.unit))),
        }
    forward_receipt = {
        "schema": "team-c-postfit-forward-handoff-v1",
        "arm_id": arm_id, "seed": seed,
        "worker_scope": "POSTFIT_FEATURE_FORWARD_NO_SEX_READ",
        "worker_pid": os.getpid(),
        "campaign_lock_sha256": campaign_lock_sha256,
        "data_manifest_sha256": expected_manifest_sha256,
        "fit_handoff_sha256": fit_handoff_sha256,
        "package_sha256_by_purpose": package_hashes,
        "roles": receipts,
    }
    forward_path = (
        private_checkpoint_root / "team_c" / campaign_lock_sha256 /
        "forward_handoff" / f"{arm_id}__s{seed}.json"
    )
    _write_json_once(forward_path, forward_receipt)
    return {
        "status": "PRIVATE_VIEWS_FROZEN_FOR_S2_ATTACK",
        "forward_receipt_relative_path": str(forward_path.relative_to(private_checkpoint_root)),
        "forward_receipt_sha256": _sha_file(forward_path),
        "roles": receipts,
    }


def restore_forward_for_s2(
    *, arm_id: str, seed: int,
    fit_handoff_sha256: str, forward_handoff_sha256: str,
    expected_head_spec_sha256: str,
    expected_manifest_sha256: str,
    manifest_path: str | Path,
    campaign_lock_path: str | Path, campaign_lock_sha256: str,
    local_seal_receipt_path: str | Path,
    load_calibrator: Callable[[Path, str], dict],
    apply_global_temperature: Callable[[np.ndarray, dict], np.ndarray],
    attack_view_type: type,
    private_checkpoint_root: str | Path,
) -> dict:
    """Restore both C full views and task outputs without reading role labels.

    S2 supplies ATT/INNER SEX labels separately and joins by these private
    ordered row/unit identities. No Team C-specific wire inference is needed.
    """
    sealed_s4_sha = _check_seal(
        Path(campaign_lock_path), campaign_lock_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_head_spec_sha256=expected_head_spec_sha256,
        local_seal_receipt_path=Path(local_seal_receipt_path),
    )
    if not all(map(_hex_digest, (
        fit_handoff_sha256, forward_handoff_sha256,
        expected_manifest_sha256, expected_head_spec_sha256,
    ))):
        raise PermissionError("full hash identities required for C replay")
    root = Path(private_checkpoint_root)
    if root.name != ".private":
        raise PermissionError("C replay requires private custody root")
    manifest_path = Path(manifest_path)
    if _sha_file(manifest_path) != expected_manifest_sha256:
        raise PermissionError("C replay manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise PermissionError("C replay role files lack S4 admission")
    specs = manifest.get("real_file_specs", {})
    namespace = root / "team_c" / campaign_lock_sha256
    fit_path = namespace / "fit_handoff" / f"{arm_id}__s{seed}.json"
    forward_path = namespace / "forward_handoff" / f"{arm_id}__s{seed}.json"
    if (_sha_file(fit_path) != fit_handoff_sha256
            or _sha_file(forward_path) != forward_handoff_sha256):
        raise PermissionError("C replay handoff bytes changed")
    fit = json.loads(fit_path.read_text())
    forward = json.loads(forward_path.read_text())
    if (fit.get("schema") != "team-c-fit-handoff-v1"
            or fit.get("arm_id") != arm_id or fit.get("seed") != seed
            or fit.get("campaign_lock_sha256") != campaign_lock_sha256
            or fit.get("data_manifest_sha256") != expected_manifest_sha256
            or fit.get("head_spec_sha256") != expected_head_spec_sha256
            or fit.get("s4_audit_receipt_sha256") != sealed_s4_sha
            or fit.get("worker_scope") not in (
                "PRIVACY_FIT_CAL_NO_SCREEN_ROWS",
                "TASK_ONLY_FIT_CAL_NO_SEX_NO_SCREEN_ROWS",
            )
            or not isinstance(fit.get("worker_pid"), int)
            or fit["worker_pid"] <= 0
            or (arm_id == "C-SIMPLE-TASK-SHARED1") !=
            (fit["worker_scope"] == "TASK_ONLY_FIT_CAL_NO_SEX_NO_SCREEN_ROWS")):
        raise PermissionError("C replay fit identity changed")
    if (forward.get("schema") != "team-c-postfit-forward-handoff-v1"
            or forward.get("arm_id") != arm_id or forward.get("seed") != seed
            or forward.get("campaign_lock_sha256") != campaign_lock_sha256
            or forward.get("data_manifest_sha256") != expected_manifest_sha256
            or forward.get("fit_handoff_sha256") != fit_handoff_sha256
            or forward.get("worker_scope") != "POSTFIT_FEATURE_FORWARD_NO_SEX_READ"
            or not isinstance(forward.get("worker_pid"), int)
            or forward["worker_pid"] <= 0
            or forward["worker_pid"] == fit["worker_pid"]
            or set(forward.get("roles", {})) != {"ATTACK_FIT_NEW", "INNER_SELECTION"}):
        raise PermissionError("C replay forward identity changed")
    role_hashes = fit.get("fit_cal_role_file_hashes", {})
    expected_fit_cal = {
        f"{role}:{field}" for role, fields in (
            ("OSF_DEFENSE_FIT", ("features", "income", "occupation") +
             (() if arm_id == "C-SIMPLE-TASK-SHARED1" else ("sex",))),
            ("CALIBRATION_HELDOUT", ("features", "income", "occupation")),
        ) for field in fields
    }
    if (set(role_hashes) != expected_fit_cal
            or any(specs[key]["sha256"] != value for key, value in role_hashes.items())):
        raise PermissionError("C replay FIT/CAL source hashes changed")
    package_units = fit.get("arm_receipt", {}).get("package_units", ())
    if len(package_units) != 2:
        raise PermissionError("C replay requires both fitted packages")
    packages = {}
    package_hashes = {}
    for unit in package_units:
        purpose = unit.get("purpose")
        if purpose not in ("income", "occupation_group") or purpose in packages:
            raise PermissionError("C replay has duplicate package purpose")
        field = "income" if purpose == "income" else "occupation"
        sources = {
            "fit_features": role_hashes["OSF_DEFENSE_FIT:features"],
            "fit_label": role_hashes[f"OSF_DEFENSE_FIT:{field}"],
            "calibration_features": role_hashes["CALIBRATION_HELDOUT:features"],
            "calibration_label": role_hashes[f"CALIBRATION_HELDOUT:{field}"],
        }
        if unit.get("source_role_hashes") != sources:
            raise PermissionError("C replay package source hashes changed")
        path = root / unit["package_relative_path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise PermissionError("C replay package escaped private custody")
        checksum = unit["package_sha256"]
        packages[purpose] = restore_release_package(
            path, expected_sha256=checksum, private_root=root,
            expected_campaign_lock_sha256=campaign_lock_sha256,
            expected_data_manifest_sha256=expected_manifest_sha256,
            expected_source_role_hashes=sources,
            load_calibrator=load_calibrator,
            apply_global_temperature=apply_global_temperature,
        )
        package_hashes[purpose] = checksum
        if packages[purpose].selected_atoms != tuple(fit["selected_atoms"][purpose]):
            raise PermissionError("C replay package mask differs from fit")
    if (forward.get("package_sha256_by_purpose") != package_hashes
            or tuple(packages[p].head_id for p in ("income", "occupation_group")) !=
            tuple(fit["head_ids"])
            or tuple(packages[p].calibrator_id for p in ("income", "occupation_group")) !=
            tuple(fit["calibrator_ids"])):
        raise PermissionError("C replay public release identity changed")
    output = {}
    for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        item = forward["roles"][role]
        expected_feature_sha = specs[f"{role}:features"]["sha256"]
        if item.get("role_feature_sha256") != expected_feature_sha:
            raise PermissionError("C replay forward feature identity changed")
        path = root / item["receipt_relative_path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise PermissionError("C replay view receipt escaped private custody")
        views, wire = restore_view_bundle(
            path, expected_receipt_sha256=item["receipt_sha256"],
            private_root=root,
            expected_campaign_lock_sha256=campaign_lock_sha256,
            expected_data_manifest_sha256=expected_manifest_sha256,
            expected_role=role, expected_arm_id=arm_id, expected_seed=seed,
            expected_role_feature_sha256=expected_feature_sha,
            expected_package_sha256_by_purpose=package_hashes,
            public_releases=packages, attack_view_type=attack_view_type,
            return_release=True,
        )
        row_id = views["coalition"].row_id
        unit = views["coalition"].unit
        if _sha_arr(np.column_stack((row_id, unit))) != item["row_unit_pair_sha256"]:
            raise PermissionError("C replay ordered people differ from forward receipt")
        output[role] = {
            "views": views, "wire": wire,
            "row_id": row_id, "unit": unit,
            "role_feature_sha256": expected_feature_sha,
        }
    return {
        "schema": "team-c-restored-s2-views-v1",
        "arm_id": arm_id, "seed": seed,
        "public_releases": packages,
        "package_sha256_by_purpose": package_hashes,
        "roles": output,
    }
