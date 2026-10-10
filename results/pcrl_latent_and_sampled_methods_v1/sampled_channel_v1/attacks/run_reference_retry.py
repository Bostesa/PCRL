"""Dated retry of fixed ATT/INNER attacks after two prefit source defects.

This file is a new, source-pinned worker. The producer-pinned reference worker
and its failed attempts remain immutable. The scientific release, reader slate,
and ATT-before-INNER label boundary are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, sha_array,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import (
    pair_batch, prepare_views,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.protocol import (
    run_three_view_attack,
)
from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import (
    assert_active_from_env,
)
from .q_artifact import load_q_table_artifact
from .q_aware import fit_q_posterior_reader
from .run_q_preprocess import (
    build_fixed_public_policy, load_clean_public_centroids,
)
from .reference_wire import (
    load_public_fixed_map, load_reference_role_package,
    reference_recipient_batch, with_reference_q_features,
)
from .retry_reference_view import corrected_continuous_recipient_batch
from .run_downstream_attack import validate_registration
from .wire import canonical_bank_sha256


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HISTORICAL_FINITE = frozenset({"T_STAR", "P_STAR", "D0_LOCAL",
                               "D0_SEQ12", "D0_SEQ21"})
DIAGNOSTIC = frozenset({"A_IDENTITY", "A_CLASS_ONLY"})
FINITE = HISTORICAL_FINITE | DIAGNOSTIC
CONTINUOUS = frozenset({"U0", "Ucal"})
IDENTITY_KEYS = frozenset({
    "reference_id", "source_seed", "release_id", "interface_kind",
    "source_binding_sha256", "release_receipt_sha256_by_role",
    "fixed_map_identity", "q_table_identity",
})
READER_IDS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
              "CELL_A1", "CODE_BIT_LR_C1", "PROB_LR_C1",
              "PROB_HGB_15x200", "Q_POST_LR_C1")
INHERITED_SHARED = ("attack_eval.py", "attack_controls_v2.py",
                    "common_eval.py", "safe_data_guard.py")
RETRY_REFS = frozenset({"P_STAR", "D0_LOCAL", "D0_SEQ12", "D0_SEQ21",
                        "U0", "Ucal"})
CORRECTION_SCHEMA = "pcrl-a-fixed-attack-correction-v1"
CORRECTION_STATUS = "LOCALLY_SEALED_FIXED_ATTACK_CORRECTION"


def _fixed_map_expected_audit_sha(reference_id: str, amendment: dict) -> str:
    """Choose the source audit already bound by the clean fixed-map receipt."""
    key = ("r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"
           if reference_id in {"P_STAR", "D0_LOCAL", "D0_SEQ12", "D0_SEQ21"}
           else "r01/SOURCE_AUDIT.json")
    if reference_id not in FINITE:
        raise ValueError("continuous reference has no fixed-map audit")
    value = amendment.get("source_sha256", {}).get(key)
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise PermissionError("fixed-map source audit is not pinned")
    return value


def _checked_retry_command(spec: dict) -> None:
    """Tie this process to the exact queued module, flags and one-thread slot."""
    argv = spec.get("argv")
    expected_module = (
        "results.pcrl_latent_and_sampled_methods_v1."
        "sampled_channel_v1.attacks.run_reference_retry"
    )
    if (type(argv) is not list or len(argv) < 4 or
            type(spec.get("cwd")) is not str or
            type(spec.get("env")) is not dict or
            Path(argv[0]).resolve(strict=True) != Path(sys.executable).resolve(strict=True) or
            argv[1:3] != ["-m", expected_module] or
            argv[3:] != sys.argv[1:] or
            Path(spec.get("cwd", "")).resolve(strict=True) != Path.cwd().resolve() or
            any(spec.get("env", {}).get(name) != "1" for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))):
        raise PermissionError("corrected attack command differs from queue spec")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_map(spec: dict) -> dict[Path, str]:
    records = spec.get("sources")
    if type(records) is not list or any(type(record) is not dict or
            record.keys() != {"path", "sha256"} for record in records):
        raise PermissionError("reference attack source list differs")
    result = {}
    for record in records:
        supplied = Path(record["path"])
        if (supplied.is_symlink() or type(record["sha256"]) is not str or
                HEX64.fullmatch(record["sha256"]) is None):
            raise PermissionError("reference attack source path/SHA differs")
        path = supplied.resolve(strict=True)
        if path in result:
            raise PermissionError("reference attack source is repeated")
        result[path] = record["sha256"]
    return result


def _checked_source(path: Path, sources: dict[Path, str],
                    expected: str | None = None) -> None:
    path = path.resolve(strict=True)
    actual = _sha(path)
    if sources.get(path) != actual or (expected is not None and actual != expected):
        raise PermissionError("reference attack source differs from queue/lock")


def _fixed_q_mechanism(fixed: dict, receipt: dict) -> tuple[str, str]:
    """Rebuild R17's public deterministic law identity from R01's clean map."""
    descriptor = {
        "source_kind": "historical_finite_reference",
        "arm_id": receipt["arm_id"], "seed": receipt["seed"],
        "release_id": receipt["release_id"],
        "source_binding_sha256": receipt["source_binding_sha256"],
        "source_receipt_sha256": receipt["original_bank_receipt_sha256"],
        "canonical_bank_sha256_by_recipient": {
            f"r{i}": canonical_bank_sha256(fixed[f"bank_r{i}"])
            for i in (1, 2)},
        "full_bank_array_sha256_by_recipient": {
            f"r{i}": sha_array(fixed[f"bank_r{i}"]) for i in (1, 2)},
        "fine_to_code_sha256_by_recipient": {
            f"r{i}": sha_array(fixed[f"fine_to_code_r{i}"]) for i in (1, 2)},
    }
    canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")
    mechanism = hashlib.sha256(b"SCV1-fixed-public-Q-law-v1\n" + canonical).hexdigest()
    descriptor["mechanism_sha256"] = mechanism
    descriptor_sha = hashlib.sha256(json.dumps(
        descriptor, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    return mechanism, descriptor_sha


def _diagnostic_q_mechanism(fixed: dict, receipt: dict,
                            centroids: tuple[np.ndarray, np.ndarray], *,
                            ref: str, seed: int, release_id: str
                            ) -> tuple[str, str, dict, np.ndarray, np.ndarray]:
    """Replay R13's actual same-bank selector on the public centroid indices."""
    records = tuple({
        "probability": fixed[f"bank_r{i}"],
        "fine_to_hcal_token": fixed[f"fine_to_code_r{i}"],
        "token_id": np.arange(len(fixed[f"bank_r{i}"]), dtype=np.int64),
        "token_class": np.argmax(fixed[f"bank_r{i}"], axis=1).astype(np.int64),
        "input_class": np.argmax(centroids[i - 1], axis=1).astype(np.int64),
    } for i in (1, 2))
    bank1, bank2, _, descriptor = build_fixed_public_policy(
        arm_id=ref, seed=seed, source_kind="same_bank_diagnostic",
        records=records, release_id=release_id,
        source_binding_sha256=receipt["source_binding_sha256"],
        source_receipt_sha256=receipt["original_bank_receipt_sha256"])
    descriptor_sha = hashlib.sha256(json.dumps(
        descriptor, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    return (descriptor["mechanism_sha256"], descriptor_sha, descriptor,
            bank1.probability, bank2.probability)


def _checked_output(path: str | Path, spec: dict) -> Path:
    supplied = Path(path)
    if not supplied.is_absolute() or ".private" not in supplied.parts:
        raise PermissionError("reference attack output must have an absolute private root")
    private_index = supplied.parts.index(".private")
    private_base = Path(*supplied.parts[:private_index + 1])
    protected = [ancestor for ancestor in supplied.parents
                 if ancestor == private_base or private_base in ancestor.parents]
    if supplied.is_symlink() or any(ancestor.is_symlink() for ancestor in protected):
        raise PermissionError("reference attack output ancestry is a symlink")
    root = supplied.resolve()
    parent = root.parent
    if (root.exists() or not parent.is_dir() or
            any(not ancestor.is_dir() or ancestor.stat().st_uid != os.getuid()
                or stat.S_IMODE(ancestor.stat().st_mode) & 0o077
                for ancestor in protected) or
            spec.get("outputs") != [str(root / "ATTACK_ATTEMPT_INDEX.json")]):
        raise PermissionError("reference attack output custody differs")
    return root


def _checked_module_origins(campaign: Path, sources: dict[Path, str]) -> None:
    """Reject a shadowed imported evaluator/loader despite matching file pins."""
    prefix = "results.pcrl_latent_and_sampled_methods_v1"
    modules = {
        prefix + ".r01.role_store": campaign / "r01/role_store.py",
        prefix + ".r02.complete_view": campaign / "r02/complete_view.py",
        prefix + ".r02.protocol": campaign / "r02/protocol.py",
        prefix + ".r03.compute_queue": campaign / "r03/compute_queue.py",
        prefix + ".sampled_channel_v1.attacks.q_aware":
            campaign / "sampled_channel_v1/attacks/q_aware.py",
        prefix + ".sampled_channel_v1.attacks.q_artifact":
            campaign / "sampled_channel_v1/attacks/q_artifact.py",
        prefix + ".sampled_channel_v1.attacks.reference_wire":
            campaign / "sampled_channel_v1/attacks/reference_wire.py",
        prefix + ".sampled_channel_v1.attacks.retry_reference_view":
            campaign / "sampled_channel_v1/attacks/retry_reference_view.py",
        prefix + ".sampled_channel_v1.attacks.run_downstream_attack":
            campaign / "sampled_channel_v1/attacks/run_downstream_attack.py",
        prefix + ".sampled_channel_v1.attacks.run_q_preprocess":
            campaign / "sampled_channel_v1/attacks/run_q_preprocess.py",
        prefix + ".sampled_channel_v1.attacks.wire":
            campaign / "sampled_channel_v1/attacks/wire.py",
        prefix + ".sampled_channel_v1.impl.reference_materialize":
            campaign / "sampled_channel_v1/impl/reference_materialize.py",
        prefix + ".sampled_channel_v1.impl.bank":
            campaign / "sampled_channel_v1/impl/bank.py",
        prefix + ".sampled_channel_v1.controls.diagnostic_codes":
            campaign / "sampled_channel_v1/controls/diagnostic_codes.py",
    }
    shared = campaign.parent / "pcrl_method_portfolio_v1/shared"
    modules.update({name.removesuffix(".py"): shared / name
                    for name in INHERITED_SHARED})
    for name, expected in modules.items():
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", None)
        if origin is None or Path(origin).resolve(strict=True) != expected.resolve(strict=True):
            raise PermissionError("reference attack loaded module differs from source pin")
        _checked_source(expected, sources)


def _completed_dependency(state: dict, queue_root: Path, unit_id: str,
                          declared_path: Path, expected_sha: str) -> None:
    unit = state.get("units", {}).get(unit_id, {})
    receipt_path = queue_root / "units" / unit_id / "RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if (unit.get("status") != "done" or receipt.get("status") != "done" or
            receipt.get("returncode") != 0 or
            receipt.get("spec_sha256") != unit.get("spec_sha256") or
            unit.get("spec", {}).get("outputs") != [str(declared_path)] or
            receipt.get("output_sha256") != {str(declared_path): expected_sha} or
            _sha(declared_path) != expected_sha):
        raise PermissionError("reference attack parent dependency receipt differs")


def _check_fixed_q_parent(q_plan: dict, q_spec: dict, *, ref: str, seed: int,
                          fixed_identity: dict) -> None:
    source_kind = ("same_bank_diagnostic" if ref in DIAGNOSTIC
                   else "historical_finite_reference")
    if (q_plan.get("kind") != "q_preprocess" or
            q_plan.get("source_kind") != source_kind or
            q_plan.get("arm_id", q_plan.get("reference_id")) != ref or
            q_plan.get("seed") != seed or
            q_spec.get("source_kind") != source_kind or
            q_spec.get("arm_id") != ref or q_spec.get("seed") != seed or
            q_spec.get("fixed_map_receipt_sha256") !=
            fixed_identity["receipt_sha256"] or
            q_spec.get("fixed_map_npz_sha256") != fixed_identity["npz_sha256"]):
        raise PermissionError("reference Q parent/source kind differs")


def admit_reference_retry(*, unit_id: str, original_attack_unit_id: str,
                           correction_path: str | Path,
                           correction_sha256: str,
                           downstream_manifest_path: str | Path,
                           downstream_manifest_sha256: str,
                           downstream_registration_path: str | Path,
                           downstream_registration_sha256: str,
                           original_lock_path: str | Path,
                           original_lock_sha256: str,
                           a_amendment_path: str | Path,
                           a_amendment_sha256: str,
                           role_manifest_path: str | Path,
                           role_manifest_sha256: str,
                           att_release_dir: str | Path,
                           inner_release_dir: str | Path,
                           fixed_map_receipt: str | Path | None,
                           fixed_map_npz: str | Path | None,
                           q_table_dir: str | Path | None,
                           private_root: str | Path) -> tuple[dict, dict, dict, Path]:
    """Bind correction, original plan and completed parents before SEX access."""
    active = assert_active_from_env(unit_id=unit_id, stage="a")
    lock = Path(original_lock_path).resolve(strict=True)
    amendment_path = Path(a_amendment_path).resolve(strict=True)
    correction_path = Path(correction_path).resolve(strict=True)
    downstream_manifest_path = Path(downstream_manifest_path).resolve(strict=True)
    downstream_registration_path = Path(downstream_registration_path).resolve(strict=True)
    manifest_path = Path(role_manifest_path).resolve(strict=True)
    if (_sha(lock) != original_lock_sha256 or
            _sha(amendment_path) != a_amendment_sha256 or
            _sha(correction_path) != correction_sha256 or
            _sha(downstream_manifest_path) != downstream_manifest_sha256 or
            _sha(downstream_registration_path) != downstream_registration_sha256 or
            _sha(manifest_path) != role_manifest_sha256 or
            active["seal_sha256"] != original_lock_sha256):
        raise PermissionError("reference attack original/amended seal differs")
    original = json.loads(lock.read_text())
    amendment = json.loads(amendment_path.read_text())
    correction = json.loads(correction_path.read_text())
    downstream_manifest = json.loads(downstream_manifest_path.read_text())
    plan_sha = hashlib.sha256(json.dumps(
        amendment.get("unit_plan"), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if (original.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1" or
            original.get("status") != "LOCALLY_SEALED_D_STAGE" or
            original.get("source_files_sha256", {}).get("source_role_manifest") !=
            role_manifest_sha256 or
            amendment.get("schema") != "pcrl-latent-sampled-A-stage-amendment-v1" or
            amendment.get("status") != "LOCALLY_SEALED_A_STAGE" or
            amendment.get("original_lock_sha256") != original_lock_sha256):
        raise PermissionError("reference attack does not have effective A seal")
    if (correction_path.parent != lock.parent or
            downstream_manifest_path.parent != lock.parent or
            downstream_registration_path.parent != lock.parent or
            correction.get("schema") != CORRECTION_SCHEMA or
            correction.get("status") != CORRECTION_STATUS or
            correction.get("owner") != "R00" or
            correction.get("scientific_scope_unchanged") is not True or
            type(correction.get("scope")) is not str or
            not correction["scope"] or
            type(correction.get("dated_utc")) is not str or
            correction.get("original_lock_sha256") != original_lock_sha256 or
            correction.get("producer_amendment_sha256") != a_amendment_sha256 or
            correction.get("producer_unit_plan_sha256") != plan_sha or
            correction.get("downstream_manifest_sha256") !=
                downstream_manifest_sha256 or
            correction.get("downstream_registration_sha256") !=
                downstream_registration_sha256 or
            correction.get("resource_limits") != {
                "max_wall_s": 90, "cpu_reserve_s": 90, "max_rss_gib": 3.0,
                "numerical_threads": 1, "heavy_slots": 1}):
        raise PermissionError("fixed attack correction lineage differs")
    expected_physical = {
        f"A-REFERENCE-ATTACK-{ref}-S{seed}-R1"
        for ref in RETRY_REFS for seed in (0, 1)
    }
    if (type(correction.get("physical_attempts")) is not dict or
            set(correction["physical_attempts"]) != expected_physical):
        raise PermissionError("fixed attack correction map is incomplete")
    validate_registration(
        campaign=lock.parent, producer_sha256=a_amendment_sha256,
        manifest_path=downstream_manifest_path,
        manifest_sha256=downstream_manifest_sha256,
        manifest=downstream_manifest,
        registration_path=downstream_registration_path,
        registration_sha256=downstream_registration_sha256)
    queue_root = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    state = json.loads((queue_root / "state.json").read_text())
    stages = [row for row in state.get("amendments", [])
              if "a" in row.get("stages", [])]
    spec = state.get("units", {}).get(unit_id, {}).get("spec", {})
    plan = amendment.get("unit_plan", {}).get(original_attack_unit_id)
    retry = correction.get("physical_attempts", {}).get(unit_id)
    identity = spec.get("reference_identity")
    if (not stages or stages[-1].get("sha256") != correction_sha256 or
            Path(stages[-1].get("path", "")).resolve() != correction_path or
            spec.get("unit_id") != unit_id or
            spec.get("amendment_sha256") != correction_sha256 or
            spec.get("producer_amendment_sha256") != a_amendment_sha256 or
            spec.get("original_attack_unit_id") != original_attack_unit_id or
            spec.get("stage") != "a" or spec.get("mode") != "real_data" or
            spec.get("data_access") !=
                "ATTACK_FIT_NEW_SEX_THEN_INNER_SELECTION_SEX" or
            unit_id != original_attack_unit_id + "-R1" or
            type(plan) is not dict or type(retry) is not dict or
            retry.get("original_attack_unit_id") != original_attack_unit_id or
            retry.get("physical_unit_id") != unit_id or
            retry.get("output_id") != unit_id or
            type(identity) is not dict or
            identity.keys() != IDENTITY_KEYS):
        raise PermissionError("corrected reference attack unit/plan differs")
    _checked_retry_command(spec)
    ref, seed = identity["reference_id"], identity["source_seed"]
    finite = ref in FINITE
    diagnostic = ref in DIAGNOSTIC
    if (ref not in RETRY_REFS or seed not in (0, 1) or
            identity["interface_kind"] != ("finite" if finite else "continuous") or
            type(identity["release_id"]) is not str or not identity["release_id"] or
            type(identity["source_binding_sha256"]) is not str or
            HEX64.fullmatch(identity["source_binding_sha256"]) is None or
            type(identity["release_receipt_sha256_by_role"]) is not dict or
            identity["release_receipt_sha256_by_role"].keys() !=
            {"ATTACK_FIT_NEW", "INNER_SELECTION"} or
            any(type(value) is not str or HEX64.fullmatch(value) is None
                for value in identity["release_receipt_sha256_by_role"].values()) or
            finite != (identity["fixed_map_identity"] is not None) or
            finite != (identity["q_table_identity"] is not None)):
        raise PermissionError("reference attack finite/continuous identity differs")
    if finite:
        if (type(identity["fixed_map_identity"]) is not dict or
                identity["fixed_map_identity"].keys() != {"receipt_sha256", "npz_sha256"} or
                type(identity["q_table_identity"]) is not dict or
                identity["q_table_identity"].keys() != {"receipt_sha256", "content_sha256"} or
                any(type(value) is not str or HEX64.fullmatch(value) is None
                    for record in (identity["fixed_map_identity"],
                                   identity["q_table_identity"])
                    for value in record.values())):
            raise PermissionError("reference fixed public source identity differs")
    release_units = (plan.get("release_unit_ids") if not diagnostic else {
        "ATT": f"A-DIAG-RELEASE-{ref}-S{seed}-ATT",
        "INNER": f"A-DIAG-RELEASE-{ref}-S{seed}-INNER"})
    expected_deps = ([release_units.get("ATT"), release_units.get("INNER")]
                     if type(release_units) is dict else [])
    if finite:
        expected_deps.append(plan.get("q_unit_id"))
    if (diagnostic or plan.get("kind") != "reference_attack" or
            plan.get("output_id") != original_attack_unit_id or
            plan.get("arm_id" if diagnostic else "reference_id") != ref or
            plan.get("seed") != seed or
            plan.get("realization") != 0 or
            plan.get("trained_reader_and_probe_fits") != (33 if finite else 24) or
            type(release_units) is not dict or release_units.keys() != {"ATT", "INNER"} or
            any(type(value) is not str or not value for value in expected_deps) or
            plan.get("deps") != expected_deps or spec.get("deps") != expected_deps or
            (plan.get("q_unit_id") is not None) != finite or
            retry.get("reference_id") != ref or retry.get("seed") != seed or
            retry.get("interface_kind") != identity["interface_kind"] or
            retry.get("release_unit_ids") != release_units or
            retry.get("q_unit_id") != plan.get("q_unit_id") or
            retry.get("deps") != expected_deps or
            retry.get("frozen_reader_and_probe_fits") != (33 if finite else 24) or
            retry.get("cpu_reserve_s") != 90 or retry.get("max_wall_s") != 90 or
            retry.get("max_rss_gib") != 3.0 or
            spec.get("cpu_reserve_s") != 90 or spec.get("max_wall_s") != 90 or
            spec.get("max_rss_gib") != 3.0 or
            spec.get("output_id") != unit_id):
        raise PermissionError("reference attack differs from sealed unit/dependency plan")
    if (amendment.get("attack_plan", {}).get("reader_ids") != list(READER_IDS) or
            amendment.get("attack_plan", {}).get("reference_ids") !=
            ["T_STAR", "P_STAR", "U0", "Ucal", "D0_LOCAL", "D0_SEQ12", "D0_SEQ21"]):
        raise PermissionError("reference attack slate differs from sealed family")
    output = _checked_output(private_root, spec)
    sources = _source_map(spec)
    for path, digest in ((correction_path, correction_sha256),
                         (downstream_manifest_path, downstream_manifest_sha256),
                         (downstream_registration_path,
                          downstream_registration_sha256)):
        _checked_source(path, sources, digest)
    new_sources = correction.get("source_sha256")
    if (type(new_sources) is not dict or
            not {"sampled_channel_v1/attacks/run_reference_retry.py",
                 "sampled_channel_v1/attacks/retry_reference_view.py"} <=
                set(new_sources)):
        raise PermissionError("corrected worker source inventory differs")
    for relative, digest in new_sources.items():
        part = Path(relative)
        if (type(relative) is not str or part.is_absolute() or
                ".." in part.parts or type(digest) is not str or
                HEX64.fullmatch(digest) is None):
            raise PermissionError("corrected source path/hash differs")
        path = (lock.parent / part).resolve(strict=True)
        if not path.is_relative_to(lock.parent):
            raise PermissionError("corrected source escapes campaign")
        _checked_source(path, sources, digest)
    if (Path(__file__).resolve(strict=True) !=
            (lock.parent / "sampled_channel_v1/attacks/run_reference_retry.py")
            .resolve(strict=True)):
        raise PermissionError("corrected attack executing module differs")
    if (downstream_manifest.get("producer_amendment_sha256") !=
            a_amendment_sha256 or
            downstream_manifest.get("original_lock_sha256") !=
                original_lock_sha256 or
            downstream_manifest.get("unit_plan_sha256") != plan_sha):
        raise PermissionError("downstream manifest differs from original plan")
    for relative, digest in downstream_manifest.get(
            "new_source_sha256", {}).items():
        part = Path(relative)
        if (part.is_absolute() or ".." in part.parts or
                type(digest) is not str or HEX64.fullmatch(digest) is None):
            raise PermissionError("downstream source path/hash differs")
        _checked_source(lock.parent / part, sources, digest)
    shared_pins = downstream_manifest.get("attack_plan", {}).get(
        "inherited_shared_source_sha256")
    if type(shared_pins) is not dict or set(shared_pins) != set(INHERITED_SHARED):
        raise PermissionError("inherited reader source inventory differs")
    shared = lock.parent.parent / "pcrl_method_portfolio_v1" / "shared"
    for name, digest in shared_pins.items():
        _checked_source(shared / name, sources, digest)
    failed_map = correction.get("old_attempt_failed_queue_receipt_sha256")
    if (type(failed_map) is not dict or set(failed_map) != {
            "A-REFERENCE-ATTACK-U0-S0", "A-REFERENCE-ATTACK-P_STAR-S0"}):
        raise PermissionError("original failed attack custody is incomplete")
    if original_attack_unit_id in failed_map:
        old_state = state.get("units", {}).get(original_attack_unit_id, {})
        failed_receipt = queue_root / "units" / original_attack_unit_id / "RECEIPT.json"
        old_record = json.loads(failed_receipt.read_text())
        if (retry.get("original_attempt_status") !=
                "FAILED_TECHNICAL_PRESERVED" or
                old_state.get("status") != "failed" or
                old_record.get("status") != "failed" or
                old_record.get("returncode") != 1 or
                old_record.get("output_sha256") != {} or
                _sha(failed_receipt) != failed_map[original_attack_unit_id] or
                retry.get("original_failed_queue_receipt_sha256") !=
                    failed_map[original_attack_unit_id]):
            raise PermissionError("original failed attempt custody differs")
    elif (original_attack_unit_id in state.get("units", {}) or
          retry.get("original_attempt_status") !=
              "NOT_ENQUEUED_SOURCE_CORRECTION"):
        raise PermissionError("unstarted original attempt was already enqueued")
    implementation = amendment.get("implementation_source_sha256")
    if (type(implementation) is not dict or
            not {"sampled_channel_v1/attacks/run_reference_attack.py",
                 "sampled_channel_v1/attacks/reference_wire.py",
                 "sampled_channel_v1/attacks/q_aware.py",
                 "sampled_channel_v1/attacks/run_q_preprocess.py",
                 "r02/protocol.py", "r03/compute_queue.py"} <= set(implementation)):
        raise PermissionError("reference attack implementation map is incomplete")
    for relative, digest in implementation.items():
        part = Path(relative)
        if (type(relative) is not str or part.is_absolute() or
                ".." in part.parts or type(digest) is not str or
                HEX64.fullmatch(digest) is None):
            raise PermissionError("reference implementation map key differs")
        path = (lock.parent / part).resolve(strict=True)
        if not path.is_relative_to(lock.parent):
            raise PermissionError("reference implementation escapes campaign")
        _checked_source(path, sources, digest)
    for path, expected in ((lock, original_lock_sha256),
                           (amendment_path, a_amendment_sha256),
                           (manifest_path, role_manifest_sha256)):
        _checked_source(path, sources, expected)
    _checked_source(Path(__file__), sources)
    shared = lock.parent.parent / "pcrl_method_portfolio_v1" / "shared"
    for name in INHERITED_SHARED:
        _checked_source(shared / name, sources)
    _checked_module_origins(lock.parent, sources)
    original_audit = original.get("source_files_sha256", {}).get(
        "r01/SOURCE_AUDIT.json")
    admitted_sources = amendment.get("source_sha256", {})
    if admitted_sources.get("r01/SOURCE_AUDIT.json") != original_audit:
        raise PermissionError("reference original source audit pin differs")
    for relative, digest in (("r01/SOURCE_AUDIT.json", original_audit),
                             ("r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json",
                              admitted_sources.get(
                                  "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"))):
        _checked_source(lock.parent / relative, sources, digest)
    for role, directory, dependency in (
            ("ATTACK_FIT_NEW", Path(att_release_dir), release_units["ATT"]),
            ("INNER_SELECTION", Path(inner_release_dir), release_units["INNER"])):
        if directory.is_symlink():
            raise PermissionError("reference saved role directory is a symlink")
        directory = directory.resolve(strict=True)
        receipt = directory / "REFERENCE_RELEASE_RECEIPT.json"
        archive = directory / "reference_release.npz"
        for path in (receipt, archive):
            _checked_source(path, sources)
        expected_sha = identity["release_receipt_sha256_by_role"][role]
        _completed_dependency(state, queue_root, dependency, receipt, expected_sha)
    if finite:
        source_arm = "T_STAR" if diagnostic else ref
        fixed_relative = f"r01/A_FIXED_MAP_{source_arm}_S{seed}.json"
        archive_relative = f"r01/A_FIXED_MAP_{source_arm}_S{seed}.npz"
        expected_fixed = lock.parent / fixed_relative
        expected_archive = lock.parent / archive_relative
        if (fixed_map_receipt is None or fixed_map_npz is None or
                Path(fixed_map_receipt).resolve(strict=True) != expected_fixed.resolve(strict=True)
                or Path(fixed_map_npz).resolve(strict=True) !=
                expected_archive.resolve(strict=True)
                or admitted_sources.get(fixed_relative) !=
                identity["fixed_map_identity"]["receipt_sha256"] or
                admitted_sources.get(archive_relative) !=
                identity["fixed_map_identity"]["npz_sha256"]):
            raise PermissionError("reference public fixed-map source pin differs")
        for path, key in ((fixed_map_receipt, "receipt_sha256"),
                          (fixed_map_npz, "npz_sha256")):
            if path is None:
                raise PermissionError("finite reference public map is absent")
            _checked_source(Path(path), sources,
                            identity["fixed_map_identity"][key])
        q_plan = amendment["unit_plan"].get(plan["q_unit_id"], {})
        q_spec = state.get("units", {}).get(plan["q_unit_id"], {}).get("spec", {})
        _check_fixed_q_parent(q_plan, q_spec, ref=ref, seed=seed,
                              fixed_identity=identity["fixed_map_identity"])
        if diagnostic:
            centroid_path = lock.parent / f"r01/A_PUBLIC_FINE_CENTROIDS_S{seed}.npz"
            centroid_sha = amendment["attack_plan"][
                "public_centroid_npz_sha256_by_seed"][f"s{seed}"]
            _checked_source(centroid_path, sources, centroid_sha)
        if q_table_dir is None or Path(q_table_dir).is_symlink():
            raise PermissionError("finite reference Q directory is absent")
        q_dir = Path(q_table_dir).resolve(strict=True)
        for name in ("Q_TABLE_RECEIPT.json", "q_table.npz"):
            _checked_source(q_dir / name, sources)
        from .runner import _assert_q_unit_custody
        _assert_q_unit_custody(q_dir, spec=spec, amendment=amendment,
                               queue_unit_id=original_attack_unit_id,
                               q_identity=identity["q_table_identity"])
    elif any(value is not None for value in
             (fixed_map_receipt, fixed_map_npz, q_table_dir)):
        raise PermissionError("continuous reference cannot claim a finite bank or Q")
    return amendment, spec, identity, output


def run_reference_retry(*, unit_id: str, original_attack_unit_id: str,
                         correction_path: str | Path,
                         correction_sha256: str,
                         downstream_manifest_path: str | Path,
                         downstream_manifest_sha256: str,
                         downstream_registration_path: str | Path,
                         downstream_registration_sha256: str,
                         original_lock_path: str | Path,
                         original_lock_sha256: str,
                         a_amendment_path: str | Path,
                         a_amendment_sha256: str,
                         role_manifest_path: str | Path,
                         role_manifest_sha256: str,
                         att_release_dir: str | Path,
                         inner_release_dir: str | Path,
                         fixed_map_receipt: str | Path | None,
                         fixed_map_npz: str | Path | None,
                         q_table_dir: str | Path | None,
                         private_root: str | Path) -> dict:
    amendment, spec, identity, output = admit_reference_retry(
        unit_id=unit_id, original_lock_path=original_lock_path,
        original_attack_unit_id=original_attack_unit_id,
        correction_path=correction_path, correction_sha256=correction_sha256,
        downstream_manifest_path=downstream_manifest_path,
        downstream_manifest_sha256=downstream_manifest_sha256,
        downstream_registration_path=downstream_registration_path,
        downstream_registration_sha256=downstream_registration_sha256,
        original_lock_sha256=original_lock_sha256,
        a_amendment_path=a_amendment_path,
        a_amendment_sha256=a_amendment_sha256,
        role_manifest_path=role_manifest_path,
        role_manifest_sha256=role_manifest_sha256,
        att_release_dir=att_release_dir, inner_release_dir=inner_release_dir,
        fixed_map_receipt=fixed_map_receipt, fixed_map_npz=fixed_map_npz,
        q_table_dir=q_table_dir, private_root=private_root)
    finite = identity["interface_kind"] == "finite"
    bank1 = bank2 = map_receipt = q_table = None
    if finite:
        source_arm = "T_STAR" if identity["reference_id"] in DIAGNOSTIC else \
            identity["reference_id"]
        fixed, map_receipt = load_public_fixed_map(
            fixed_map_receipt, fixed_map_npz,
            expected_receipt_sha256=identity["fixed_map_identity"]["receipt_sha256"],
            expected_archive_sha256=identity["fixed_map_identity"]["npz_sha256"],
            arm_id=source_arm, seed=identity["source_seed"])
        bank1, bank2 = fixed["bank_r1"], fixed["bank_r2"]
        if (map_receipt["source_binding_sha256"] !=
                identity["source_binding_sha256"] or
                (identity["reference_id"] not in DIAGNOSTIC and
                 map_receipt["release_id"] != identity["release_id"]) or
                (identity["reference_id"] in DIAGNOSTIC and
                 identity["release_id"] !=
                 f"{identity['reference_id']}-s{identity['source_seed']}-v1") or
                map_receipt["source_audit_sha256"] !=
                    _fixed_map_expected_audit_sha(identity["reference_id"], amendment)):
            raise PermissionError("reference public fixed map differs from actual wire")
        q_table = load_q_table_artifact(
            q_table_dir,
            expected_receipt_sha256=identity["q_table_identity"]["receipt_sha256"],
            expected_content_sha256=identity["q_table_identity"]["content_sha256"],
            expected_policy_receipt_sha256=
                map_receipt["original_bank_receipt_sha256"])
        if identity["reference_id"] in DIAGNOSTIC:
            audit = json.loads((Path(original_lock_path).resolve(strict=True).parent /
                                "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json").read_text())
            seed = identity["source_seed"]
            centroid_path = (Path(original_lock_path).resolve(strict=True).parent /
                             f"r01/A_PUBLIC_FINE_CENTROIDS_S{seed}.npz")
            c1, c2 = load_clean_public_centroids(
                centroid_path, audit["public_centroids"][f"s{seed}"])
            if (amendment["attack_plan"]["public_centroid_npz_sha256_by_seed"][f"s{seed}"]
                    != _sha(centroid_path)):
                raise PermissionError("diagnostic Q public centroid pin differs")
            mechanism_sha, descriptor_sha, _, bank1, bank2 = _diagnostic_q_mechanism(
                fixed, map_receipt, (c1, c2),
                ref=identity["reference_id"], seed=seed,
                release_id=identity["release_id"])
        else:
            mechanism_sha, descriptor_sha = _fixed_q_mechanism(fixed, map_receipt)
        queue_root = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
        queue_state = json.loads((queue_root / "state.json").read_text())
        q_unit_id = amendment["unit_plan"][original_attack_unit_id]["q_unit_id"]
        q_spec = queue_state["units"][q_unit_id]["spec"]
        if (q_spec.get("source_descriptor_sha256") != descriptor_sha or
                q_spec.get("mechanism_sha256") != mechanism_sha or
                q_table.identity["mechanism_sha256"] != mechanism_sha):
            raise PermissionError("reference Q law differs from fixed public map")
    att = load_reference_role_package(
        att_release_dir,
        receipt_sha256=identity["release_receipt_sha256_by_role"]["ATTACK_FIT_NEW"],
        full_bank1=bank1, full_bank2=bank2)
    inner = load_reference_role_package(
        inner_release_dir,
        receipt_sha256=identity["release_receipt_sha256_by_role"]["INNER_SELECTION"],
        full_bank1=bank1, full_bank2=bank2)
    store = RoleOnlyStore(role_manifest_path)
    original_source_audit = json.loads((Path(original_lock_path).resolve(strict=True).parent /
                                        "r01/SOURCE_AUDIT.json").read_text())
    for role, package in (("ATTACK_FIT_NEW", att), ("INNER_SELECTION", inner)):
        historical_audit = amendment["source_sha256"][
            "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"]
        original_audit = amendment["source_sha256"]["r01/SOURCE_AUDIT.json"]
        expected_audit = (historical_audit if identity["reference_id"] in
                          {"P_STAR", "D0_LOCAL", "D0_SEQ12", "D0_SEQ21"}
                          else original_audit)
        if (package.receipt["role"] != role or
                package.receipt["arm_id"] != identity["reference_id"] or
                package.receipt["source_seed"] != identity["source_seed"] or
                package.receipt["release_id"] != identity["release_id"] or
                package.receipt["source_binding_sha256"] !=
                identity["source_binding_sha256"] or
                package.receipt["interface_kind"] != identity["interface_kind"] or
                package.receipt["source_identity"]["source_audit_sha256"] !=
                expected_audit or
                (finite and package.receipt["source_identity"]["bank_receipt_sha256"] !=
                 map_receipt["original_bank_receipt_sha256"])):
            raise PermissionError("reference release source identity differs")
        if identity["reference_id"] in DIAGNOSTIC:
            tstar = original_source_audit["tstar_banks"][str(identity["source_seed"])]
            role_key = f"{role}:references_s{identity['source_seed']}"
            if (package.receipt["source_identity"]["role_source_file_sha256"] !=
                    tstar["file_sha256"][f"tokens_{role}.npz"] or
                    package.receipt["source_identity"]["reference_file_sha256"] !=
                    store.manifest["role_file_sha256"][role_key] or
                    identity["source_binding_sha256"] !=
                    tstar["source_binding_sha256"] or
                    map_receipt["original_bank_receipt_sha256"] !=
                    tstar["receipt_sha256"]):
                raise PermissionError("diagnostic role source lineage differs")
    batches = {}
    for role, package in (("att", att), ("inner", inner)):
        adapter = (reference_recipient_batch if finite
                   else corrected_continuous_recipient_batch)
        left = adapter(package, 1)
        right = adapter(package, 2)
        views = {"r1": left, "r2": right, "pair": pair_batch(left, right)}
        if finite:
            views = {name: with_reference_q_features(
                batch, q_table, view=name, full_bank1=bank1,
                full_bank2=bank2, role_release=package,
                expected_mechanism_sha256=mechanism_sha)
                for name, batch in views.items()}
        batches[role] = views
    prepared = {name: prepare_views(batches["att"][name],
                                    batches["inner"][name])
                for name in ("r1", "r2", "pair")}
    for role, package in (("ATTACK_FIT_NEW", att), ("INNER_SELECTION", inner)):
        if (sha_array(np.column_stack((package.row_id, package.unit))) !=
                store.manifest["roles"][role]["row_unit_pair_sha256"]):
            raise PermissionError("reference released people differ from admitted role")
    guard = lambda: assert_active_from_env(unit_id=unit_id, stage="a")
    guard()
    sex_att = store.read("ATTACK_FIT_NEW", "sex", procedure="attack")
    output.mkdir(mode=0o700)
    specialists = None
    if finite:
        specialists = {}
        for name in ("r1", "r2", "pair"):
            specialists[name] = (fit_q_posterior_reader(
                batches["att"][name], batches["inner"][name],
                sex_att, q_table, view=name,
                attacker_seed=identity["source_seed"],
                private_dir=output / "q_readers" / name,
                lock_path=original_lock_path,
                lock_sha256=original_lock_sha256,
                expected_table_sha256=identity["q_table_identity"]["content_sha256"],
                queue_unit_id=unit_id, queue_guard=guard,
                prevalidated_reference=True),)

    def load_inner_sex() -> dict:
        guard()
        marker = output / "common" / "FROZEN_COMPLETE.json"
        if (not marker.is_file() or
                json.loads(marker.read_text()).get("status") != "FROZEN_COMPLETE"):
            raise PermissionError("INNER SEX is embargoed until ATT readers freeze")
        return store.read("INNER_SELECTION", "sex", procedure="screen")

    result = run_three_view_attack(
        prepared, sex_att, load_inner_sex=load_inner_sex,
        family="A" if finite else "D", seed=identity["source_seed"],
        private_root=output / "common", lock_path=original_lock_path,
        lock_sha256=original_lock_sha256, queue_guard=guard,
        specialists_by_view=specialists)
    controls = {name: result["public"]["controls"][name]["scores"]["status"]
                for name in ("r1", "r2", "pair")}
    index = {
        "schema": "SCV1-reference-attack-attempt-v1",
        "status": "COMPLETE" if all(value == "PASS" for value in controls.values())
                  else "CONTROL_FAIL",
        "unit_id": unit_id,
        "original_attack_unit_id": original_attack_unit_id,
        "original_lock_sha256": original_lock_sha256,
        "a_amendment_sha256": a_amendment_sha256,
        "correction_amendment_sha256": correction_sha256,
        "reference_identity": identity,
        "att_release_receipt_sha256": att.receipt_sha256,
        "inner_release_receipt_sha256": inner.receipt_sha256,
        "q_table_content_sha256": None if q_table is None else q_table.table_sha256,
        "view_content_sha256": {name: {
            "att": prepared[name].att.content_hashes(),
            "inner": prepared[name].inner.content_hashes()}
            for name in ("r1", "r2", "pair")},
        "reader_freeze_sha256": result["public"]["freeze_receipt_sha256"],
        "scores": result["public"]["scores"],
        "control_status": controls,
    }
    index_path = output / "ATTACK_ATTEMPT_INDEX.json"
    fd = os.open(index_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(index, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"status": index["status"], "index_sha256": _sha(index_path),
            "index_path": str(index_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Queued corrected fixed reference attack")
    for name in ("unit-id", "original-attack-unit-id",
                 "correction-path", "correction-sha256",
                 "downstream-manifest-path", "downstream-manifest-sha256",
                 "downstream-registration-path", "downstream-registration-sha256",
                 "original-lock-path", "original-lock-sha256",
                 "a-amendment-path", "a-amendment-sha256", "role-manifest-path",
                 "role-manifest-sha256", "att-release-dir", "inner-release-dir",
                 "private-root"):
        parser.add_argument("--" + name, required=True)
    for name in ("fixed-map-receipt", "fixed-map-npz", "q-table-dir"):
        parser.add_argument("--" + name)
    result = run_reference_retry(**vars(parser.parse_args(argv)))
    print(json.dumps({"status": result["status"],
                      "index_sha256": result["index_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
