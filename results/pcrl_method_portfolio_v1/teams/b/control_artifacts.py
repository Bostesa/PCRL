"""Provenance checks for paired starts, independent LOCAL, and sequential stages.

All paths here are private checkpoint/release paths supplied by an S3-admitted
worker. Nothing is opened at import time. Public receipts contain hashes only.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Mapping

import torch

from bottleneck import BottleneckPair, SexCritics, model_fingerprint, schema_fingerprint
from train_bottleneck import FitData, FrozenPurposeView

PURPOSES = ("income", "occupation")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def critic_fingerprint(critics: SexCritics) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(critics.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def paired_initial_copies(
    base_model: BottleneckPair, base_critics: SexCritics,
) -> tuple[BottleneckPair, SexCritics, dict[str, str]]:
    """Clone exactly common purpose-specific initial weights for one seed."""
    model, critics = copy.deepcopy(base_model), copy.deepcopy(base_critics)
    receipt = {
        "base_model_sha256": model_fingerprint(base_model),
        "base_critics_sha256": critic_fingerprint(base_critics),
        "feature_schema_sha256": schema_fingerprint(base_model.feature_names),
    }
    if model_fingerprint(model) != receipt["base_model_sha256"]:
        raise AssertionError("Paired model clone changed initialization")
    if critic_fingerprint(critics) != receipt["base_critics_sha256"]:
        raise AssertionError("Paired critic clone changed initialization")
    return model, critics, receipt


def _purpose_equal(a: BottleneckPair, b: BottleneckPair, purpose: str) -> bool:
    for module_name in ("encoders", "heads"):
        left = getattr(a, module_name)[purpose].state_dict()
        right = getattr(b, module_name)[purpose].state_dict()
        if set(left) != set(right) or not all(torch.equal(left[key], right[key]) for key in left):
            return False
    return True


def _verify_checkpoint_model(
    path: str | Path, supplied_model: BottleneckPair, receipt: Mapping[str, object],
) -> str:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if state.get("format") != "TeamBStageCheckpointV1":
        raise ValueError("Wrong Team B stage checkpoint format")
    for field in ("source_id", "role_id", "data_sha256", "row_order_sha256",
                  "group_order_sha256", "code_sha256", "config_sha256"):
        if state.get(field) != receipt.get(field):
            raise ValueError(f"Checkpoint/receipt mismatch: {field}")
    loaded = copy.deepcopy(supplied_model)
    loaded.load_state_dict(state["model"])
    if model_fingerprint(loaded) != model_fingerprint(supplied_model):
        raise ValueError("Supplied trained model differs from saved checkpoint")
    if model_fingerprint(supplied_model) != receipt.get("model_sha256"):
        raise ValueError("Trained model fingerprint differs from stage receipt")
    return file_sha256(path)


def assemble_local_pair(
    base_model: BottleneckPair,
    *,
    income_model: BottleneckPair,
    occupation_model: BottleneckPair,
    income_receipt: Mapping[str, object],
    occupation_receipt: Mapping[str, object],
    income_checkpoint: str | Path,
    occupation_checkpoint: str | Path,
    steps_per_encoder: int,
) -> tuple[BottleneckPair, dict[str, object]]:
    """Assemble two independent B-LOCAL fits into one attacked pair release."""
    base_sha = model_fingerprint(base_model)
    receipts = {"income": income_receipt, "occupation": occupation_receipt}
    models = {"income": income_model, "occupation": occupation_model}
    paths = {"income": income_checkpoint, "occupation": occupation_checkpoint}
    for purpose in PURPOSES:
        receipt = receipts[purpose]
        if receipt.get("stage_initial_model_sha256") != base_sha:
            raise ValueError(f"{purpose} LOCAL fit did not start from common paired weights")
        if receipt.get("status") != "COMPLETED" or receipt.get("encoder_steps", {}).get(purpose) != steps_per_encoder:
            raise ValueError(f"{purpose} LOCAL fit is incomplete")
        other = "occupation" if purpose == "income" else "income"
        if receipt.get("encoder_steps", {}).get(other) != 0 or not _purpose_equal(models[purpose], base_model, other):
            raise ValueError(f"{purpose} LOCAL fit changed the other purpose")
        if f"B-LOCAL/" not in str(receipt.get("stage_id", "")):
            raise ValueError(f"{purpose} checkpoint is not a B-LOCAL stage")
    for field in ("source_id", "role_id", "data_sha256", "row_order_sha256",
                  "group_order_sha256", "teacher_identity", "code_sha256"):
        if income_receipt.get(field) != occupation_receipt.get(field):
            raise ValueError(f"LOCAL purpose fits differ in {field}")
    checkpoint_shas = {
        purpose: _verify_checkpoint_model(paths[purpose], models[purpose], receipts[purpose])
        for purpose in PURPOSES
    }
    combined = copy.deepcopy(base_model)
    for purpose in PURPOSES:
        combined.encoders[purpose].load_state_dict(models[purpose].encoders[purpose].state_dict())
        combined.heads[purpose].load_state_dict(models[purpose].heads[purpose].state_dict())
    return combined, {
        "arm": "B-LOCAL",
        "base_model_sha256": base_sha,
        "income_stage_checkpoint_sha256": checkpoint_shas["income"],
        "occupation_stage_checkpoint_sha256": checkpoint_shas["occupation"],
        "income_model_sha256": income_receipt["model_sha256"],
        "occupation_model_sha256": occupation_receipt["model_sha256"],
        "combined_model_sha256": model_fingerprint(combined),
        "data_sha256": income_receipt["data_sha256"],
        "row_order_sha256": income_receipt["row_order_sha256"],
        "group_order_sha256": income_receipt["group_order_sha256"],
    }


def verify_sequential_first_artifacts(
    *,
    checkpoint_path: str | Path,
    first_model: BottleneckPair,
    first_receipt: Mapping[str, object],
    fixed_first_view: FrozenPurposeView,
    fit_data: FitData,
    expected_sampler_plan_sha256: str,
    expected_first_purpose: str,
    expected_steps: int,
    expected_stage_id: str,
    expected_config_sha256: str,
    expected_source_id: str,
    expected_role_id: str,
) -> dict[str, str]:
    """Read actual stage-1 artifacts before granting stage-2 pair access."""
    if expected_first_purpose not in PURPOSES or expected_steps <= 0:
        raise ValueError("Expected first purpose and positive complete step count required")
    for value in (expected_stage_id, expected_config_sha256, expected_source_id, expected_role_id):
        if not isinstance(value, str) or not value:
            raise ValueError("Expected sealed first-stage identities are required")
    if not expected_stage_id.startswith(("B-SEQ12/", "B-SEQ21/")):
        raise ValueError("First-stage ID must belong to a registered sequential arm")
    if (first_receipt.get("status") != "COMPLETED"
            or first_receipt.get("stage_id") != expected_stage_id
            or first_receipt.get("config_sha256") != expected_config_sha256
            or first_receipt.get("source_id") != expected_source_id
            or first_receipt.get("role_id") != expected_role_id
            or first_receipt.get("data_sha256") != fit_data.sha256
            or first_receipt.get("row_order_sha256") != fit_data.row_order_sha256
            or first_receipt.get("group_order_sha256") != fit_data.group_order_sha256
            or first_receipt.get("planned_steps") != expected_steps
            or first_receipt.get("completed_steps") != expected_steps
            or first_receipt.get("encoder_steps", {}).get(expected_first_purpose) != expected_steps
            or first_receipt.get("encoder_steps", {}).get(
                "occupation" if expected_first_purpose == "income" else "income"
            ) != 0
            or first_receipt.get("local_critic_steps", {}).get(expected_first_purpose) != expected_steps
            or first_receipt.get("pair_critic_steps") != 0):
        raise ValueError("First sequential stage is not a complete one-purpose fit")
    if fixed_first_view.purpose != expected_first_purpose or fixed_first_view.source_id != expected_source_id:
        raise ValueError("Frozen first view has wrong purpose")
    checkpoint_sha = _verify_checkpoint_model(checkpoint_path, first_model, first_receipt)
    if fixed_first_view.first_checkpoint_sha256 != checkpoint_sha:
        raise ValueError("Frozen first view cites a different checkpoint file")
    if fixed_first_view.model_sha256 != model_fingerprint(first_model):
        raise ValueError("Frozen first view cites a different trained model")
    if (fixed_first_view.row_order_sha256 != fit_data.row_order_sha256
            or fixed_first_view.group_order_sha256 != fit_data.group_order_sha256):
        raise ValueError("Frozen first view has wrong defense-fit people or order")
    if fixed_first_view.feature_schema_sha256 != schema_fingerprint(first_model.feature_names):
        raise ValueError("Frozen first view has wrong permitted schema")
    if fixed_first_view.sampler_plan_sha256 != expected_sampler_plan_sha256:
        raise ValueError("Frozen first view has wrong sampler plan")
    return {
        "stage_id": str(first_receipt["stage_id"]),
        "checkpoint_sha256": checkpoint_sha,
        "fixed_first_release_sha256": fixed_first_view.sha256,
        "defense_fit_row_ids_sha256": fit_data.row_order_sha256,
        "defense_fit_group_ids_sha256": fit_data.group_order_sha256,
        "feature_schema_sha256": fixed_first_view.feature_schema_sha256,
        "sampler_plan_sha256": fixed_first_view.sampler_plan_sha256,
    }
