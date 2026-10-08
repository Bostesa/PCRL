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
import subprocess
import tempfile
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np

from controls import ArmPlan, ControlPlan, REL0_ATOMS, build_control_plan
from evaluation import AttackView, attack_views_from_release
from prototype import ATOM_ORDER, AdultGroupedSchema, release_pair
from release_package import restore_release_package, save_release_package
from scorecard_builder import ScorecardProxyCache, build_scorecard
from training import HeadSpec, fit_task_head, restore_head, save_head


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
PROCEDURE = {"income": "defense_fit", "occupation": "defense_fit", "sex": "defense_fit"}
SOURCE_TIP = "1baf5bbdaf59cfa6a664cae07d652a4712bdfdc5"
SOURCE_FILES = (
    "controls.py", "evaluation.py", "control_run.py", "prototype.py",
    "selector.py", "scorecard_builder.py", "training.py", "release_package.py",
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
    attack_views: Mapping[str, AttackView] | None
    inner_views: Mapping[str, AttackView] | None
    head_ids: tuple[str, str] | None
    calibrator_ids: tuple[str, str] | None
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class RunOutput:
    plan: ControlPlan
    arms: Mapping[str, ArmOutput]
    scorecard_receipt: Mapping[str, object]
    admission: Admission
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


def admit_control_roles(store: RoleStore, *, expected_manifest_sha256: str) -> Admission:
    """Read only Team C's authorized fields through S2's verified store."""
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
    for role in ROLES:
        for field in FIELDS[role]:
            if f"{role}:{field}" not in specs:
                raise PermissionError(f"role-only file not admitted: {role}/{field}")
    roles = {}
    hashes = {}
    schema = AdultGroupedSchema()
    for role in ROLES:
        features = store.read(role, "features")
        row_id = np.asarray(features["row_id"], dtype=np.int64)
        unit = np.asarray(features["unit"], dtype=np.int64)
        X = np.asarray(features["X"])
        names = tuple(str(n) for n in features["feature_names"])
        schema.validate_feature_names(names)
        if (
            X.shape != (len(row_id), 83) or unit.shape != row_id.shape
            or not np.isfinite(X).all() or len(np.unique(row_id)) != len(row_id)
        ):
            raise PermissionError(f"role-only feature content invalid: {role}")
        labels = {}
        for field in FIELDS[role]:
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
            field: hashes[f"{role}:{field}"] for field in FIELDS[role]
        })
    # S2's store checks pinned role ID/group hashes. Here we also reject any
    # cross-role group overlap before scientific fitting.
    for i, role_a in enumerate(ROLES):
        for role_b in ROLES[i + 1:]:
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
    heavy_slot: Callable[[str], AbstractContextManager[object]],
    campaign_lock_sha256: str,
    private_checkpoint_root: Path,
    selected_arm_ids: tuple[str, ...] | None,
) -> dict[str, ArmOutput]:
    fitting = admission.roles["OSF_DEFENSE_FIT"]
    calibration = admission.roles["CALIBRATION_HELDOUT"]
    to_release = {
        "ATTACK_FIT_NEW": admission.roles["ATTACK_FIT_NEW"],
        "INNER_SELECTION": admission.roles["INNER_SELECTION"],
    }
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
        if arm.capacity_status.startswith("SKIPPED") or arm.capacity_status.startswith("NOT_RUN"):
            result[arm.arm_id] = ArmOutput(
                arm, arm.capacity_status, None, None, None, None, None,
                None, {"reason": arm.note or arm.capacity_status},
            )
            continue
        packages = {}
        receipt = {"arm_id": arm.arm_id, "seed": seed,
                   "capacity_status": arm.capacity_status,
                   "budget": {"semantic": arm.budget.semantic_counts,
                              "encoded": arm.budget.encoded_counts},
                   "budget_match": asdict(arm.budget_match),
                   "campaign_lock_sha256": campaign_lock_sha256,
                   "data_manifest_sha256": admission.data_manifest_sha256,
                   "head_spec_sha256": head_spec_sha,
                   "head_units": [], "calibration_units": [], "forward_units": []}
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
        releases = {}
        attacks = {}
        for role_name, role in to_release.items():
            unit_id = f"C:s{seed}:FORWARD:{arm.arm_id}:{role_name}"
            with heavy_slot(unit_id):
                release = release_pair(
                    role.X, role.feature_names, income_release,
                    occupation_release,
                )
                attack_view = attack_views_from_release(
                    release, _private_alignment(role),
                )
            releases[role_name] = release
            attacks[role_name] = attack_view
            receipt["forward_units"].append({
                "role": role_name, "status": "COMPLETED",
                "feature_sha256": role.file_hashes["features"],
                "rows": len(role.row_id), "views": ("income", "occupation_group", "coalition"),
            })
        result[arm.arm_id] = ArmOutput(
            arm, "RELEASES_READY_FOR_S2_ATTACK_AND_ENDPOINTS",
            releases["ATTACK_FIT_NEW"], releases["INNER_SELECTION"],
            attacks["ATTACK_FIT_NEW"], attacks["INNER_SELECTION"],
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
    heavy_slot: Callable[[str], AbstractContextManager[object]],
    private_checkpoint_root: str | Path,
    selected_arm_ids: tuple[str, ...] | None = None,
) -> RunOutput:
    """Fit selected locked arms and construct private full-view attack inputs.

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
        load_calibrator, apply_global_temperature,
    )):
        raise TypeError("global heavy slot and four S2 calibration functions are required")
    private_checkpoint_root = Path(private_checkpoint_root)
    if private_checkpoint_root.name != ".private":
        raise PermissionError("Team C checkpoints must use the declared .private root")
    with heavy_slot(f"C:s{seed}:ROLE_ADMISSION"):
        admission = admit_control_roles(
            store, expected_manifest_sha256=expected_manifest_sha256,
        )
    if admission.s4_audit_receipt_sha256 != sealed_s4_sha:
        raise PermissionError("lock and role manifest bind different S4 audits")
    fit = admission.roles["OSF_DEFENSE_FIT"]
    cache = ScorecardProxyCache()
    scorecards = {}
    for basis, atoms in (("FULL", ATOM_ORDER), ("REL0", REL0_ATOMS)):
        with heavy_slot(f"C:s{seed}:SCORECARD:{basis}"):
            scorecards[basis] = build_scorecard(
                fit.X, fit.feature_names, fit.labels["income"],
                fit.labels["occupation"], fit.labels["sex"], fit.unit,
                base_atoms=atoms, optional_atoms=atoms, seed=seed, cache=cache,
            )
    plan = build_control_plan(scorecards["REL0"].card, scorecards["FULL"].card)
    scorecard_receipt = {
        "seed": seed, "full_fit_counts": scorecards["FULL"].fit_counts,
        "rel0_fit_counts": scorecards["REL0"].fit_counts,
        "rel0_alias_masks": scorecards["REL0"].alias_masks,
        "full_split": scorecards["FULL"].split_receipt,
        "rel0_split": scorecards["REL0"].split_receipt,
        "defense_feature_sha256": fit.file_hashes["features"],
        "defense_label_sha256": {field: fit.file_hashes[field]
                                  for field in ("income", "occupation", "sex")},
        "data_manifest_sha256": admission.data_manifest_sha256,
        "campaign_lock_sha256": campaign_lock_sha256,
    }
    arms = _fit_and_calibrate(
        admission, plan, seed=seed, spec=head_spec,
        fit_calibrator=fit_calibrator,
        save_calibrator=save_calibrator,
        load_calibrator=load_calibrator,
        apply_global_temperature=apply_global_temperature,
        heavy_slot=heavy_slot,
        campaign_lock_sha256=campaign_lock_sha256,
        private_checkpoint_root=private_checkpoint_root,
        selected_arm_ids=selected_arm_ids,
    )
    return RunOutput(
        plan, arms, scorecard_receipt, admission,
        campaign_lock_sha256, expected_head_spec_sha256,
    )
