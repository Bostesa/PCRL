"""Run one complete Team B arm inside an S3-admitted queue unit.

The caller supplies S2-admitted FitData and one common per-seed initialization.
Every checkpoint and first-stage person-level draw is written under a new
``.private`` attempt directory. This module loads no source data on its own.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np

from bottleneck import BottleneckPair, SexCritics, model_fingerprint, schema_fingerprint
from control_artifacts import (
    assemble_local_pair, file_sha256, paired_initial_copies,
    verify_sequential_first_artifacts,
)
from control_plan import ARMS, Stage, stages_for_arm
from control_runner import ControlHyperparameters, stage_config
from train_bottleneck import FitData, FrozenPurposeView, StageRunner


@dataclass
class ArmFitResult:
    arm: str
    seed: int
    model: BottleneckPair
    stage_receipts: tuple[dict, ...]
    private_artifact_sha256: dict[str, str]
    base_model_sha256: str
    base_critics_sha256: str


@dataclass(frozen=True)
class FitAdmissionSeal:
    """S2's preadmitted role-only export identity, supplied by the queue worker."""

    source_id: str
    role_id: str
    reference_seed: int
    admission_manifest_path: str
    role_only_export_sha256: str
    reference_export_sha256: str
    source_npz_sha256: str
    admission_manifest_sha256: str
    admitted_rows: int
    admitted_groups: int
    admitted_row_id_sha256: str
    admitted_group_id_sha256: str
    admitted_row_unit_pair_sha256: str
    data_sha256: str
    row_order_sha256: str
    group_order_sha256: str
    feature_schema_sha256: str
    teacher_identity: tuple[tuple[str, str], ...]

    @staticmethod
    def _sha_arr(value: np.ndarray) -> str:
        arr = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(arr.dtype).encode() + str(arr.shape).encode())
        digest.update(arr.tobytes())
        return digest.hexdigest()

    def check(self, data: FitData, *, source_id: str, role_id: str, seed: int,
              need_sex: bool) -> None:
        if self.source_id != source_id or self.role_id != role_id:
            raise ValueError("Fitting source or role differs from S2 admission seal")
        if self.reference_seed != seed or seed not in (0, 1, 2):
            raise ValueError("Fitting teacher reference seed differs from S2 admission seal")
        for field in ("role_only_export_sha256", "reference_export_sha256",
                      "source_npz_sha256", "admission_manifest_sha256",
                      "admitted_row_id_sha256", "admitted_group_id_sha256",
                      "admitted_row_unit_pair_sha256",
                      "data_sha256", "row_order_sha256", "group_order_sha256",
                      "feature_schema_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, field)):
                raise ValueError(f"Invalid S2 admission seal field: {field}")
        if (self.data_sha256 != data.sha256
                or self.row_order_sha256 != data.row_order_sha256
                or self.group_order_sha256 != data.group_order_sha256
                or self.feature_schema_sha256 != data.feature_schema_sha256
                or self.teacher_identity != tuple(sorted(data.teacher_identity.items()))):
            raise ValueError("Fitting data differs from S2 role-only admission seal")
        try:
            rows = np.asarray(data.row_ids, dtype=np.int64)
            groups = np.asarray(data.group_ids, dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError("S2 fitting membership IDs must be int64") from exc
        if (rows.size != self.admitted_rows
                or np.unique(groups).size != self.admitted_groups
                or self._sha_arr(np.sort(rows)) != self.admitted_row_id_sha256
                or self._sha_arr(np.unique(groups)) != self.admitted_group_id_sha256
                or self._sha_arr(np.column_stack((rows, groups)).astype(np.int64))
                != self.admitted_row_unit_pair_sha256):
            raise ValueError("Fitting membership differs from S2 admitted role")
        manifest_path = Path(self.admission_manifest_path).resolve()
        if not manifest_path.is_file() or file_sha256(manifest_path) != self.admission_manifest_sha256:
            raise ValueError("S2 admission manifest file differs from sealed bytes")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema") != "pcrl-method-portfolio-data-access-v1"
                or manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED"):
            raise ValueError("S2 role-only manifest is not independently admitted")
        if manifest["source"]["source_npz_sha256"] != self.source_npz_sha256:
            raise ValueError("Source identity differs from S2 admission")
        admitted_role = manifest["roles"][self.role_id]
        if (admitted_role["rows"] != self.admitted_rows
                or admitted_role["groups"] != self.admitted_groups
                or admitted_role["row_id_sha256"] != self.admitted_row_id_sha256
                or admitted_role["group_id_sha256"] != self.admitted_group_id_sha256):
            raise ValueError("Role membership hashes differ from S2 admission")
        names_hash = hashlib.sha256("\n".join(data.feature_names).encode()).hexdigest()
        if names_hash != manifest["source"]["feature_names_sha256_newline_join"]:
            raise ValueError("Feature-name identity differs from S2 admission")
        specs = manifest["real_file_specs"]
        required = (("features", "income", "occupation", "sex", f"references_s{seed}")
                    if need_sex else
                    ("features", "income", "occupation", f"references_s{seed}"))
        for field in required:
            spec = specs[f"{self.role_id}:{field}"]
            declared = Path(spec["path"])
            path = (declared if declared.is_absolute() else manifest_path.parent / declared).resolve()
            if spec["role"] != self.role_id or spec["field"] != field or not path.is_file():
                raise ValueError("S2 role-only field identity or file is missing")
            if file_sha256(path) != spec["sha256"]:
                raise ValueError("S2 role-only file bytes differ from admission")
            if spec["row_unit_pair_sha256"] != self.admitted_row_unit_pair_sha256:
                raise ValueError("S2 role-only field has different person/group pairing")
        if (specs[f"{self.role_id}:features"]["sha256"] != self.role_only_export_sha256
                or specs[f"{self.role_id}:references_s{seed}"]["sha256"]
                != self.reference_export_sha256):
            raise ValueError("Feature or teacher export differs from S2 admission seal")


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _fit_stage(stage: Stage, model: BottleneckPair, critics: SexCritics,
               data: FitData, config, *, source_id: str, role_id: str,
               directory: Path, frozen_views=None):
    target = directory / (stage.stage_id.replace("/", "__") + ".pt")
    failed = target.with_suffix(".failed.pt")
    if target.exists() or failed.exists():
        raise FileExistsError("An earlier attempt for this stage is retained")
    runner = StageRunner(model, critics, data, config, source_id=source_id,
                         role_id=role_id, frozen_views=frozen_views)
    started, wall0, cpu0 = _utc(), time.monotonic(), time.process_time()
    try:
        receipt = runner.run()
        runner.save_checkpoint(target)
    except Exception:
        runner.save_checkpoint(failed)
        raise
    receipt = dict(receipt)
    receipt.update(started_utc=started, ended_utc=_utc(),
                   elapsed_s=time.monotonic()-wall0,
                   process_cpu_s=time.process_time()-cpu0,
                   checkpoint_file_sha256=file_sha256(target))
    return runner.model, runner.critics, receipt, target


def _save_assembled(model: BottleneckPair, provenance: dict, target: Path) -> str:
    if target.exists():
        raise FileExistsError(target)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                                         prefix=".team-b-assembled-", delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            torch.save({"format": "TeamBAssembledLocalPairV1",
                        "model": model.state_dict(), "provenance": provenance}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return file_sha256(target)


def fit_arm(arm: str, seed: int, *, data: FitData,
            base_model: BottleneckPair, base_critics: SexCritics,
            hyper: ControlHyperparameters, source_id: str, role_id: str,
            admitted_fit: FitAdmissionSeal,
            private_attempt_dir: str | Path,
            expected_base_model_sha256: str,
            expected_base_critics_sha256: str,
            sampler_secret: bytes | None = None,
            sampler_plan_sha256: str | None = None) -> ArmFitResult:
    """Fit all stages of one registered arm, with no stage alias by default."""
    if arm not in ARMS:
        raise ValueError("Unknown Team B arm")
    if role_id != "OSF_DEFENSE_FIT" or not source_id:
        raise ValueError("An admitted defense-fitting role identity is required")
    need_sex = arm not in ("B-TASK", "B-RAW-TASK")
    if (need_sex and data.sex is None) or (not need_sex and data.sex is not None):
        raise ValueError("SEX fitting access differs from arm's locked task-only/privacy objective")
    admitted_fit.check(data, source_id=source_id, role_id=role_id, seed=seed,
                       need_sex=need_sex)
    if (base_model.feature_names != data.feature_names
            or schema_fingerprint(base_model.feature_names) != data.feature_schema_sha256):
        raise ValueError("Base model does not match S2's ordered permitted feature schema")
    directory = Path(private_attempt_dir)
    if ".private" not in directory.parts or directory.exists():
        raise ValueError("A new versioned .private attempt directory is required")
    stages = stages_for_arm(arm, seed, hyper.steps_per_encoder)
    if arm in ("B-SEQ12", "B-SEQ21") and (
        not isinstance(sampler_secret, bytes) or len(sampler_secret) < 32
        or not isinstance(sampler_plan_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", sampler_plan_sha256)
    ):
        raise ValueError("Sequential first draw needs a private sampler contract")
    base_sha = model_fingerprint(base_model)
    _, _, common = paired_initial_copies(base_model, base_critics)
    if (base_sha != expected_base_model_sha256
            or common["base_critics_sha256"] != expected_base_critics_sha256):
        raise ValueError("This arm lacks the sealed common per-seed initialization")
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if stat.S_IMODE(directory.parent.stat().st_mode) != 0o700:
        raise ValueError("Private parent directory must have mode 0700")
    directory.mkdir(mode=0o700)
    if stat.S_IMODE(directory.stat().st_mode) != 0o700:
        raise ValueError("Private attempt directory must have mode 0700")
    receipts, hashes = [], {}

    if arm == "B-LOCAL":
        models, checkpoints = {}, {}
        for stage in stages:
            model, critics, initial = paired_initial_copies(base_model, base_critics)
            if initial != common:
                raise AssertionError("LOCAL starts differ")
            model, _, receipt, checkpoint = _fit_stage(
                stage, model, critics, data, stage_config(stage, hyper),
                source_id=source_id, role_id=role_id, directory=directory)
            purpose = stage.train_purposes[0]
            models[purpose], checkpoints[purpose] = model, checkpoint
            receipts.append(receipt)
            hashes[stage.stage_id] = receipt["checkpoint_file_sha256"]
        final, provenance = assemble_local_pair(
            base_model, income_model=models["income"],
            occupation_model=models["occupation"],
            income_receipt=receipts[0], occupation_receipt=receipts[1],
            income_checkpoint=checkpoints["income"],
            occupation_checkpoint=checkpoints["occupation"],
            steps_per_encoder=hyper.steps_per_encoder)
        hashes["assembled_local_pair"] = _save_assembled(
            final, provenance, directory / "B-LOCAL__assembled.pt")
    elif arm in ("B-SEQ12", "B-SEQ21"):
        model, critics, initial = paired_initial_copies(base_model, base_critics)
        if initial != common:
            raise AssertionError("Sequential start differs")
        first, second = stages
        first_config = stage_config(first, hyper)
        model, critics, first_receipt, first_checkpoint = _fit_stage(
            first, model, critics, data, first_config,
            source_id=source_id, role_id=role_id, directory=directory)
        receipts.append(first_receipt)
        hashes[first.stage_id] = first_receipt["checkpoint_file_sha256"]
        frozen = FrozenPurposeView.from_model(
            model, data, purpose=first.train_purposes[0], source_id=source_id,
            first_checkpoint_sha256=file_sha256(first_checkpoint),
            sampler_plan_sha256=sampler_plan_sha256, sampler_secret=sampler_secret)
        frozen_path = directory / "first_fixed_draw.pt"
        frozen.save(frozen_path)
        hashes["first_fixed_draw_file"] = file_sha256(frozen_path)
        identity = verify_sequential_first_artifacts(
            checkpoint_path=first_checkpoint, first_model=model,
            first_receipt=first_receipt, fixed_first_view=frozen,
            fit_data=data, expected_sampler_plan_sha256=sampler_plan_sha256,
            expected_first_purpose=first.train_purposes[0],
            expected_steps=hyper.steps_per_encoder,
            expected_stage_id=first.stage_id,
            expected_config_sha256=first_config.sha256,
            expected_source_id=source_id, expected_role_id=role_id)
        second_config = stage_config(second, hyper,
            first_stage_receipt=identity, recomputed_first_stage_hashes=identity)
        first_snapshot = copy.deepcopy(model)
        final, _, second_receipt, second_checkpoint = _fit_stage(
            second, model, critics, data, second_config,
            source_id=source_id, role_id=role_id, directory=directory,
            frozen_views={first.train_purposes[0]: frozen})
        purpose = first.train_purposes[0]
        for module_name in ("encoders", "heads"):
            before = getattr(first_snapshot, module_name)[purpose].state_dict()
            after = getattr(final, module_name)[purpose].state_dict()
            if not all(torch.equal(before[name], after[name]) for name in before):
                raise AssertionError("Sequential stage 2 changed its frozen first purpose")
        receipts.append(second_receipt)
        hashes[second.stage_id] = file_sha256(second_checkpoint)
    else:
        model, critics, initial = paired_initial_copies(base_model, base_critics)
        if initial != common:
            raise AssertionError("Paired arm start differs")
        stage = stages[0]
        final, _, receipt, checkpoint = _fit_stage(
            stage, model, critics, data, stage_config(stage, hyper),
            source_id=source_id, role_id=role_id, directory=directory)
        receipts.append(receipt)
        hashes[stage.stage_id] = file_sha256(checkpoint)
    if model_fingerprint(base_model) != base_sha:
        raise AssertionError("Common initial model mutated")
    return ArmFitResult(arm, seed, final, tuple(receipts), hashes,
                        common["base_model_sha256"], common["base_critics_sha256"])
