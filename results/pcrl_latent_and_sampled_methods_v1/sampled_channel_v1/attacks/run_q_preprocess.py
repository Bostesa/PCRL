"""R03-only public-centroid Q preprocessing for sampled and finite score arms.

No per-person row, task label, SEX label, fitted source frequency, or old
assessment is read here. Admission precedes any source file load. The 64x641
uniform public grid is still a queued numerical job, never an inline attack fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_array
from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import (
    FixedBank, build_actions,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls import control_package
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.diagnostic_codes import select_codes
from .q_aware import _digest_arrays, compile_uniform_prototype_features
from .q_artifact import save_q_table_artifact
from .wire import PublicBank, canonical_bank_sha256


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SCHEMA = "pcrl-latent-sampled-A-stage-amendment-v1"
ZERO_SUPPORT_RULE = "all_zero_features_with_zero_support_v1"
FITTED = frozenset({
    "A_JOINT_STOCH", "A_TASK_STOCH", "A_TASK_DET", "A_JOINT_DET",
    "A_CENTRAL_PRODUCT", "A_LOCAL", "A_SEQ12", "A_SEQ21",
})
DIAGNOSTIC = frozenset({"A_IDENTITY", "A_CLASS_ONLY"})
HISTORICAL = frozenset({
    "T_STAR", "P_STAR", "D0_LOCAL", "D0_SEQ12", "D0_SEQ21",
})
IMPLEMENTATION = frozenset({
    "sampled_channel_v1/attacks/run_q_preprocess.py",
    "sampled_channel_v1/attacks/q_aware.py",
    "sampled_channel_v1/attacks/q_artifact.py",
    "sampled_channel_v1/attacks/wire.py",
    "sampled_channel_v1/controls/control_package.py",
    "sampled_channel_v1/controls/diagnostic_codes.py",
    "sampled_channel_v1/controls/deterministic_policy.py",
    "sampled_channel_v1/controls/factor_controls.py",
    "sampled_channel_v1/impl/bank.py",
    "sampled_channel_v1/impl/channel.py",
    "sampled_channel_v1/impl/fit.py",
    "sampled_channel_v1/impl/policy_package.py",
    "sampled_channel_v1/impl/release.py",
    "sampled_channel_v1/impl/source_bridge.py",
    "r01/role_store.py", "r02/complete_view.py", "r02/protocol.py",
    "r03/compute_queue.py",
})
CENTROID_FIELDS = frozenset({
    "centroid_r1", "centroid_r2", "cell_class_r1", "cell_class_r2",
})


def sha_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _hex(value: object) -> bool:
    return type(value) is str and HEX64.fullmatch(value) is not None


def _source(spec: dict, path: Path, expected: str) -> None:
    """Check a source without interpreting its contents."""
    rows = spec.get("sources")
    if type(rows) is not list or any(type(row) is not dict or
                                   set(row) != {"path", "sha256"} for row in rows):
        raise PermissionError("Q unit has no exact source inventory")
    listed = {Path(row["path"]).resolve(): row["sha256"] for row in rows}
    if len(listed) != len(rows):
        raise PermissionError("Q source inventory repeats a path")
    if (not _hex(expected) or listed.get(path.resolve()) != expected or
            path.is_symlink() or not path.is_file() or sha_file(path) != expected):
        raise PermissionError("Q source path or bytes differ from queued identity")


def _private_new_output(path: Path) -> Path:
    supplied = Path(path)
    chain = [supplied.parent, *supplied.parent.parents]
    if not supplied.is_absolute() or ".private" not in supplied.parts:
        raise PermissionError("Q output must be under private custody")
    private_index = supplied.parts.index(".private")
    private_root = Path(*supplied.parts[:private_index + 1])
    guarded = [part for part in chain if part == private_root or private_root in part.parents]
    if (supplied.is_symlink() or any(part.is_symlink() for part in guarded)
            or any(not part.is_dir() or part.stat().st_uid != os.getuid() or
                   stat.S_IMODE(part.stat().st_mode) & 0o077 for part in guarded)):
        raise PermissionError("Q output must be a fresh owner-only private directory")
    target = supplied.resolve(strict=False)
    if target.exists() or not target.parent.is_dir():
        raise PermissionError("Q output must be a fresh owner-only private directory")
    return target


def _private_existing_dir(path: Path) -> Path:
    supplied = Path(path)
    if not supplied.is_absolute() or ".private" not in supplied.parts:
        raise PermissionError("Q selected package must have an absolute private path")
    private_root = Path(*supplied.parts[:supplied.parts.index(".private") + 1])
    guarded = [supplied, *(parent for parent in supplied.parents
                           if parent == private_root or private_root in parent.parents)]
    if any(part.is_symlink() or not part.is_dir() or
           part.stat().st_uid != os.getuid() or
           stat.S_IMODE(part.stat().st_mode) & 0o077 for part in guarded):
        raise PermissionError("Q selected package ancestry is not owner-only")
    return supplied.resolve(strict=True)


def validate_q_unit_contract(unit_id: str, plan: dict, spec: dict, output: Path) -> str:
    """Require one exact arm/seed Q unit and its two-file output contract."""
    if type(plan) is not dict or type(spec) is not dict:
        raise PermissionError("Q unit plan is absent")
    arm = plan.get("arm_id", plan.get("reference_id"))
    seed = plan.get("seed")
    kind = plan.get("source_kind")
    expected_kind = ("selected_fit" if arm in FITTED else
                     "same_bank_diagnostic" if arm in DIAGNOSTIC else
                     "historical_finite_reference" if arm in HISTORICAL else None)
    expected_id = f"A-Q-PREPROCESS-{arm}-S{seed}"
    fit_id = plan.get("fit_unit_id")
    if kind == "selected_fit":
        fit_prefix = f"A-FIT-{arm}-S{seed}"
        if (type(fit_id) is not str or
                re.fullmatch(re.escape(fit_prefix) + r"(?:-R[1-9][0-9]*)?", fit_id)
                is None):
            raise PermissionError("Q selected FIT ID differs from its arm and seed")
    elif fit_id is not None:
        raise PermissionError("Q fixed public unit cannot depend on a FIT")
    deps = [fit_id] if fit_id else []
    outputs = [str(output / name) for name in ("Q_TABLE_RECEIPT.json", "q_table.npz")]
    for key in ("cpu_seconds", "wall_seconds"):
        if type(plan.get(key)) not in (int, float) or not 0 < plan[key] < float("inf"):
            raise PermissionError("Q unit plan has no finite resource reservation")
    if (type(arm) is not str or type(seed) is not int or seed not in (0, 1) or
            kind != expected_kind or unit_id != expected_id or
            plan.get("kind") != "q_preprocess" or plan.get("output_id") != unit_id or
            plan.get("expected_outputs") != ["Q_TABLE_RECEIPT.json", "q_table.npz"] or
            plan.get("deps") != deps or plan.get("fit_unit_id") != fit_id or
            spec.get("stage") != "a" or spec.get("mode") != "real_data" or
            spec.get("arm_id") != arm or spec.get("seed") != seed or
            spec.get("source_kind") != kind or spec.get("output_id") != unit_id or
            spec.get("fit_unit_id") != fit_id or spec.get("deps") != deps or
            spec.get("outputs") != outputs or
            spec.get("cpu_reserve_s") != plan["cpu_seconds"] or
            spec.get("max_wall_s") != plan["wall_seconds"] or
            not _hex(spec.get("source_descriptor_sha256")) or
            not _hex(spec.get("public_centroids_sha256")) or
            spec.get("zero_support_rule") != ZERO_SUPPORT_RULE or
            (plan.get("source_descriptor_sha256") is not None and
             plan["source_descriptor_sha256"] != spec["source_descriptor_sha256"]) or
            (plan.get("public_centroids_sha256") is not None and
             plan["public_centroids_sha256"] != spec["public_centroids_sha256"])):
        raise PermissionError("Q unit plan/spec identity or output contract differs")
    return arm


def _invocation_is_registered(spec: dict, args: argparse.Namespace) -> bool:
    """Bind this module entry point and every CLI value to the queued argv."""
    argv = spec.get("argv")
    if type(argv) is not list or len(argv) < 4 or not all(type(x) is str for x in argv):
        return False
    module = ("results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1"
              ".attacks.run_q_preprocess")
    expected = ["-m", module]
    for name, value in (("unit-id", args.unit_id),
                        ("original-lock", args.original_lock),
                        ("a-amendment", args.a_amendment),
                        ("a-amendment-sha256", args.a_amendment_sha256),
                        ("campaign-root", args.campaign_root),
                        ("private-output", args.private_output)):
        expected.extend(("--" + name, str(value)))
    return (Path(argv[0]).resolve() == Path(sys.executable).resolve() and
            argv[1:] == expected and Path(sys.argv[0]).resolve() ==
            Path(__file__).resolve())


def load_clean_public_centroids(path: Path, audit_record: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read only R01's four-member, hash-admitted public fine-centroid export."""
    path = Path(path)
    if (path.is_symlink() or audit_record.get("clean_relative_path") !=
            f"r01/{path.name}" or path.parent.name != "r01" or
            sha_file(path) != audit_record.get("clean_sha256")):
        raise ValueError("public centroid file differs from R01 audit")
    with ZipFile(path) as archive:
        members = archive.infolist()
        if (len(members) != 4 or {member.filename for member in members} !=
                {name + ".npy" for name in CENTROID_FIELDS}):
            raise ValueError("public centroid NPZ member set differs")
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in CENTROID_FIELDS}
    c1, c2 = values["centroid_r1"], values["centroid_r2"]
    classes = (values["cell_class_r1"], values["cell_class_r2"])
    if (c1.dtype != np.float64 or c1.shape != (64, 2) or
            c2.dtype != np.float64 or c2.shape != (641, 6) or
            any(value.dtype != np.int64 or value.shape != (count,) for value, count
                in zip(classes, (64, 641))) or
            not np.isfinite(c1).all() or not np.isfinite(c2).all() or
            not np.array_equal(np.argmax(c1, axis=1), classes[0]) or
            not np.array_equal(np.argmax(c2, axis=1), classes[1]) or
            sha_array(c1) != audit_record.get("centroid_r1_sha256") or
            sha_array(c2) != audit_record.get("centroid_r2_sha256") or
            _digest_arrays(c1, c2) != audit_record.get("prototype_array_pair_sha256")):
        raise ValueError("public centroid arrays/class/hash differ from R01 audit")
    return c1, c2


def build_fixed_public_policy(*, arm_id: str, seed: int, source_kind: str,
                              records: tuple[dict, dict], release_id: str,
                              source_binding_sha256: str,
                              source_receipt_sha256: str):
    """Build a deterministic public fine-index to full-bank code map only."""
    if (arm_id not in DIAGNOSTIC | HISTORICAL or seed not in (0, 1) or
            source_kind != ("same_bank_diagnostic" if arm_id in DIAGNOSTIC else
                            "historical_finite_reference") or
            type(release_id) is not str or not release_id or
            not _hex(source_binding_sha256) or not _hex(source_receipt_sha256) or
            len(records) != 2):
        raise ValueError("fixed public source identity differs")
    probabilities = []
    maps = []
    for record in records:
        source = np.asarray(record["probability"])
        fine = np.asarray(record["fine_to_hcal_token"])
        if arm_id in DIAGNOSTIC:
            bank = FixedBank.from_source(source, record["token_id"],
                                         record["token_class"])
            actions = build_actions(bank, input_class=record["input_class"],
                                    fine_to_old_token=fine)
            mapping = select_codes(bank, actions,
                                   np.arange(len(fine), dtype=np.int64),
                                   arm_id=arm_id)
            probabilities.append(bank.probability)
        else:
            if (source.dtype != np.float64 or fine.dtype != np.int64 or
                    fine.ndim != 1 or np.any((fine < 0) | (fine >= len(source)))):
                raise ValueError("historical public fine-to-code map differs")
            mapping = fine.copy()
            probabilities.append(source)
        maps.append(np.asarray(mapping, dtype=np.int64))
    if (probabilities[0].shape[1] != 2 or probabilities[1].shape[1] != 6 or
            maps[0].shape != (64,) or maps[1].shape != (641,)):
        raise ValueError("public fixed map is not on the admitted 64x641 grid")
    bank_hash = {f"r{i}": canonical_bank_sha256(probabilities[i - 1])
                 for i in (1, 2)}
    map_hash = {f"r{i}": sha_array(maps[i - 1]) for i in (1, 2)}
    descriptor = {"source_kind": source_kind, "arm_id": arm_id, "seed": seed,
                  "release_id": release_id,
                  "source_binding_sha256": source_binding_sha256,
                  "source_receipt_sha256": source_receipt_sha256,
                  "canonical_bank_sha256_by_recipient": bank_hash,
                  "full_bank_array_sha256_by_recipient": {
                      f"r{i}": sha_array(probabilities[i - 1]) for i in (1, 2)},
                  "fine_to_code_sha256_by_recipient": map_hash}
    mechanism = hashlib.sha256(b"SCV1-fixed-public-Q-law-v1\n" +
                               _canonical(descriptor)).hexdigest()
    descriptor["mechanism_sha256"] = mechanism
    banks = tuple(PublicBank(
        i, probabilities[i - 1], release_id,
        {"model_id": release_id,
         "source_binding_sha256": source_binding_sha256,
         "canonical_bank_sha256": bank_hash[f"r{i}"],
         "mechanism_sha256": mechanism,
         "code_length_bits": (len(probabilities[i - 1]) - 1).bit_length(),
         "fallback": False,
         "sampling_contract": "cached_one_joint_draw_v1"}) for i in (1, 2))

    def sparse_law(_score1, _score2, cell1, cell2):
        return ((int(maps[0][cell1]), int(maps[1][cell2]), 1.0),)

    return banks[0], banks[1], sparse_law, descriptor


def _read_json_source(spec: dict, path: Path, expected: str) -> dict:
    _source(spec, path, expected)
    value = json.loads(path.read_text())
    if type(value) is not dict:
        raise PermissionError("Q JSON source is not a record")
    return value


def _fit_dependency(state: dict, spec: dict, plan: dict,
                    amendment_sha256: str) -> dict:
    """Bind the selected private policy and FIT receipt to a completed unit."""
    fit_id = plan["fit_unit_id"]
    parent = state.get("units", {}).get(fit_id, {})
    policy_dir = _private_existing_dir(Path(spec.get("policy_dir", "")))
    fit_path = policy_dir / "FIT_RECEIPT.json"
    policy_path = policy_dir / "POLICY_RECEIPT.json"
    parameter_path = policy_dir / "policy.npz"
    fit_sha = spec.get("fit_receipt_sha256")
    policy_sha = spec.get("policy_receipt_sha256")
    if (parent.get("status") != "done" or
            parent.get("spec", {}).get("arm_id") != plan["arm_id"] or
            parent.get("spec", {}).get("seed") != plan["seed"] or
            parent.get("spec", {}).get("outputs") != [str(fit_path)] or
            parent.get("output_sha256", {}).get(str(fit_path)) != fit_sha or
            not _hex(fit_sha) or not _hex(policy_sha) or
            not _hex(spec.get("policy_file_sha256"))):
        raise PermissionError("Q selected FIT dependency lacks completed exact custody")
    for path, sha in ((fit_path, fit_sha), (policy_path, policy_sha),
                      (parameter_path, spec["policy_file_sha256"])):
        _source(spec, path, sha)
    receipt = json.loads(fit_path.read_text())
    if (receipt.get("status") != "SELECTED_FIT_POLICY" or
            receipt.get("arm_id") != plan["arm_id"] or
            receipt.get("source_seed") != plan["seed"] or
            receipt.get("unit_id") != fit_id or
            receipt.get("a_amendment_sha256") != amendment_sha256 or
            receipt.get("source_binding_sha256") !=
                spec.get("source_binding_sha256") or
            (policy_dir / "INCOMPLETE_RESOURCE.json").exists()):
        raise PermissionError("Q parent FIT is incomplete or source mismatched")
    return receipt


def _selected_parent_identity(fit: dict, policy: dict, plan: dict,
                              amendment_sha256: str) -> None:
    """Match the selected public law to the exact completed FIT outcome."""
    shared = ("arm_id", "source_seed", "source_binding_sha256",
              "row_unit_pair_sha256", "bank_sha256_by_recipient",
              "action_sha256_by_recipient")
    if (fit.get("status") != "SELECTED_FIT_POLICY" or
            policy.get("status") != "SELECTED_FIT_POLICY" or
            fit.get("unit_id") != plan["fit_unit_id"] or
            policy.get("fit_unit_id") != plan["fit_unit_id"] or
            fit.get("a_amendment_sha256") != amendment_sha256 or
            any(fit.get(key) != policy.get(key) for key in shared)):
        raise PermissionError("Q policy differs from completed FIT identity")
    if fit.get("schema") == "SCV1-private-stochastic-fit-v1":
        selected = fit.get("selected")
        expected = None if type(selected) is not dict else (
            "identity_biased" if selected.get("start") == 0 else
            "uniform_logit" if selected.get("start") == 1 else None,
            selected.get("update"))
    elif fit.get("schema") == "SCV1-private-control-fit-v1":
        details = fit.get("fit")
        if type(details) is not dict or details.get("complete") is not True or \
                details.get("success") is not True:
            raise PermissionError("Q control FIT was not complete and selected")
        arm = fit.get("arm_id")
        if arm in ("A_LOCAL", "A_SEQ12", "A_SEQ21"):
            expected = ("composite", 0)
        elif arm == "A_CENTRAL_PRODUCT":
            rank = details.get("diagnostic_best")
            expected = None if type(rank) not in (list, tuple) or len(rank) < 3 else (
                "identity_biased" if rank[1] == 0 else "uniform_logit"
                if rank[1] == 1 else None, rank[2])
        else:
            selected = details.get("selection")
            starts = [row for row in details.get("starts", [])
                      if row.get("terminal_metrics") == selected]
            expected = None if not starts else (
                starts[0].get("start"),
                300 if starts[0].get("chosen_checkpoint_update") is None else
                starts[0]["chosen_checkpoint_update"])
    else:
        raise PermissionError("Q FIT receipt schema differs")
    if (expected is None or expected[0] is None or
            policy.get("selected_start") != expected[0] or
            policy.get("selected_update") != expected[1]):
        raise PermissionError("Q policy checkpoint selection differs from FIT")


def admit_q_unit(args: argparse.Namespace) -> tuple[dict, dict, dict, dict, Path]:
    """Verify active slot, two seals, unit identity and source hashes first."""
    active = assert_active_from_env(unit_id=args.unit_id, stage="a")
    campaign = Path(args.campaign_root).resolve(strict=True)
    supplied_lock, supplied_amendment = Path(args.original_lock), Path(args.a_amendment)
    if supplied_lock.is_symlink() or supplied_amendment.is_symlink():
        raise PermissionError("Q seal path cannot be a symlink")
    lock = supplied_lock.resolve(strict=True)
    amendment_path = supplied_amendment.resolve(strict=True)
    output = _private_new_output(Path(args.private_output))
    amendment_sha = args.a_amendment_sha256
    if (lock.parent != campaign or not _hex(amendment_sha) or
            sha_file(lock) != active.get("seal_sha256") or
            sha_file(amendment_path) != amendment_sha):
        raise PermissionError("Q original or A-stage seal bytes differ")
    original = json.loads(lock.read_text())
    amendment = json.loads(amendment_path.read_text())
    if (original.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1" or
            original.get("status") != "LOCALLY_SEALED_D_STAGE" or
            amendment.get("schema") != SCHEMA or
            amendment.get("status") != "LOCALLY_SEALED_A_STAGE" or
            amendment.get("original_lock_sha256") != active["seal_sha256"] or
            amendment.get("role_bindings", {}).get("repair_manifest_sha256") !=
            original.get("source_files_sha256", {}).get("source_role_manifest") or
            amendment.get("role_bindings", {}).get("source_audit_sha256") !=
            original.get("source_files_sha256", {}).get("r01/SOURCE_AUDIT.json")):
        raise PermissionError("Q lacks the effective original/latest A seal")
    queue = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    state = json.loads((queue / "state.json").read_text())
    latest = [row for row in state.get("amendments", []) if "a" in row.get("stages", [])]
    unit = state.get("units", {}).get(args.unit_id, {})
    spec = unit.get("spec", {})
    plan = amendment.get("unit_plan", {}).get(args.unit_id)
    if (not latest or latest[-1].get("sha256") != amendment_sha or
            Path(latest[-1].get("path", "")).resolve() != amendment_path or
            spec.get("amendment_sha256") != amendment_sha or
            not _invocation_is_registered(spec, args)):
        raise PermissionError("Q unit does not bind the latest registered A amendment")
    arm = validate_q_unit_contract(args.unit_id, plan, spec, output)
    _source(spec, lock, active["seal_sha256"])
    _source(spec, amendment_path, amendment_sha)
    implementation = amendment.get("implementation_source_sha256", {})
    if not IMPLEMENTATION <= set(implementation):
        raise PermissionError("Q implementation source inventory is incomplete")
    for relative in IMPLEMENTATION:
        path = (campaign / relative).resolve(strict=True)
        if not path.is_relative_to(campaign):
            raise PermissionError("Q implementation source escapes campaign")
        module_name = ("results.pcrl_latent_and_sampled_methods_v1." +
                       relative[:-3].replace("/", "."))
        module = sys.modules.get(module_name)
        origin = Path(__file__) if relative.endswith("/run_q_preprocess.py") else (
            Path(module.__file__) if module is not None else None)
        if origin is None or origin.resolve(strict=True) != path:
            raise PermissionError("Q imported module origin differs from sealed source")
        _source(spec, path, implementation[relative])
    r01_audit = campaign / "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"
    if spec.get("historical_audit_sha256") != amendment.get(
            "source_sha256", {}).get("r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"):
        raise PermissionError("Q historical audit is not the latest A source")
    audit = _read_json_source(spec, r01_audit,
                              spec.get("historical_audit_sha256"))
    centroid = audit.get("public_centroids", {}).get(f"s{plan['seed']}", {})
    centroid_path = campaign / f"r01/A_PUBLIC_FINE_CENTROIDS_S{plan['seed']}.npz"
    if (spec.get("public_centroids_sha256") != centroid.get("clean_sha256") or
            centroid.get("clean_relative_path") !=
            f"r01/A_PUBLIC_FINE_CENTROIDS_S{plan['seed']}.npz"):
        raise PermissionError("Q public centroid identity differs from R01 audit")
    _source(spec, centroid_path, centroid["clean_sha256"])
    fit_receipt = None
    if arm in FITTED:
        fit_receipt = _fit_dependency(state, spec, plan, amendment_sha)
    else:
        source_arm = "T_STAR" if arm in DIAGNOSTIC else arm
        prefix = f"r01/A_FIXED_MAP_{source_arm}_S{plan['seed']}"
        receipt_path, npz_path = campaign / (prefix + ".json"), campaign / (prefix + ".npz")
        expected_source_audit = (amendment["role_bindings"]["source_audit_sha256"]
                                 if source_arm == "T_STAR" else
                                 spec["historical_audit_sha256"])
        if spec.get("source_audit_sha256") != expected_source_audit:
            raise PermissionError("Q fixed map source audit pin differs")
        for path, field in ((receipt_path, "fixed_map_receipt_sha256"),
                            (npz_path, "fixed_map_npz_sha256")):
            if spec.get(field) != amendment.get("source_sha256", {}).get(
                    str(path.relative_to(campaign))):
                raise PermissionError("Q public fixed map is not in latest A source inventory")
            _source(spec, path, spec[field])
        if not _hex(spec.get("source_receipt_sha256")):
            raise PermissionError("Q fixed source bank receipt is unregistered")
    return amendment, spec, plan, centroid, output, fit_receipt


def _fixed_source(spec: dict, campaign: Path, arm: str, seed: int,
                  centroids: tuple[np.ndarray, np.ndarray]):
    """Restore only R01's clean, label-free public finite-bank/map export."""
    source_arm = "T_STAR" if arm in DIAGNOSTIC else arm
    prefix = campaign / f"r01/A_FIXED_MAP_{source_arm}_S{seed}"
    receipt_path = prefix.with_suffix(".json")
    npz_path = prefix.with_suffix(".npz")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "pcrl-r01-public-fixed-fine-map-v1" or
            receipt.get("status") != "LABEL_FREE_FIXED_PUBLIC_SOURCE" or
            receipt.get("arm_id") != source_arm or receipt.get("seed") != seed or
            receipt.get("npz_sha256") != spec["fixed_map_npz_sha256"] or
            receipt.get("original_bank_receipt_sha256") !=
                spec["source_receipt_sha256"] or
            receipt.get("source_binding_sha256") !=
                spec.get("source_binding_sha256") or
            receipt.get("source_audit_sha256") !=
                spec.get("source_audit_sha256") or
            (arm in HISTORICAL and receipt.get("release_id") !=
                spec.get("release_id"))):
        raise PermissionError("Q fixed public map receipt/source identity differs")
    names = ("bank_r1", "bank_r2", "fine_to_code_r1", "fine_to_code_r2")
    with ZipFile(npz_path) as archive:
        members = archive.infolist()
        if len(members) != 4 or {row.filename for row in members} != {
                name + ".npy" for name in names}:
            raise PermissionError("Q fixed public map NPZ member set differs")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in names}
    for name, value in arrays.items():
        if (receipt.get("array_sha256", {}).get(name) != sha_array(value) or
                receipt.get("array_shape", {}).get(name) != list(value.shape)):
            raise PermissionError("Q fixed public array SHA or shape differs")
    records = []
    for recipient in (1, 2):
        probability = arrays[f"bank_r{recipient}"]
        mapping = arrays[f"fine_to_code_r{recipient}"]
        if (probability.dtype != np.float64 or probability.ndim != 2 or
                probability.shape[1] != (2 if recipient == 1 else 6) or
                mapping.dtype != np.int64 or
                mapping.shape != ((64,) if recipient == 1 else (641,)) or
                np.any((mapping < 0) | (mapping >= len(probability)))):
            raise PermissionError("Q fixed public bank/map dtype or range differs")
        records.append({"probability": probability,
                        "fine_to_hcal_token": mapping,
                        "token_id": np.arange(len(probability), dtype=np.int64),
                        "token_class": np.argmax(probability, axis=1).astype(np.int64),
                        "input_class": np.argmax(centroids[recipient - 1], axis=1)
                                       .astype(np.int64)})
    return receipt, (records[0], records[1])


def run(args: argparse.Namespace) -> dict:
    amendment, spec, plan, centroid_record, output, fit_receipt = admit_q_unit(args)
    campaign = Path(args.campaign_root).resolve(strict=True)
    centroid_path = campaign / f"r01/A_PUBLIC_FINE_CENTROIDS_S{plan['seed']}.npz"
    c1, c2 = load_clean_public_centroids(centroid_path, centroid_record)
    arm = plan.get("arm_id", plan.get("reference_id"))
    if arm in FITTED:
        policy, receipt = control_package.load_selected_policy_package(
            spec["policy_dir"], receipt_sha256=spec["policy_receipt_sha256"])
        _selected_parent_identity(fit_receipt, receipt, plan,
                                  args.a_amendment_sha256)
        if (receipt.get("arm_id") != arm or receipt.get("source_seed") != plan["seed"] or
                receipt.get("fit_receipt_sha256") != spec["fit_receipt_sha256"] or
                receipt.get("mechanism_sha256") != spec.get("mechanism_sha256") or
                receipt.get("source_binding_sha256") != spec.get("source_binding_sha256")):
            raise PermissionError("Q selected policy receipt differs from completed FIT")
        release_id = receipt["release_id"]
        banks = tuple(PublicBank(
            recipient, fixed.probability, release_id,
            {"model_id": release_id,
             "source_binding_sha256": receipt["source_binding_sha256"],
             "canonical_bank_sha256":
                 receipt["bank_sha256_by_recipient"][f"r{recipient}"],
             "mechanism_sha256": receipt["mechanism_sha256"],
             "code_length_bits": (len(fixed.probability) - 1).bit_length(),
             "fallback": False,
             "sampling_contract": "cached_one_joint_draw_v1"})
            for recipient, fixed in ((1, policy.bank1), (2, policy.bank2)))
        law = policy.sparse_law
        source_receipt_sha = spec["policy_receipt_sha256"]
        descriptor = {"source_kind": "selected_fit", "arm_id": arm,
                      "seed": plan["seed"], "release_id": release_id,
                      "source_binding_sha256": receipt["source_binding_sha256"],
                      "source_receipt_sha256": source_receipt_sha,
                      "mechanism_sha256": receipt["mechanism_sha256"],
                      "canonical_bank_sha256_by_recipient":
                          receipt["bank_sha256_by_recipient"]}
    else:
        source, records = _fixed_source(spec, campaign, arm, plan["seed"], (c1, c2))
        banks = build_fixed_public_policy(
            arm_id=arm, seed=plan["seed"], source_kind=plan["source_kind"],
            records=records, release_id=spec["release_id"],
            source_binding_sha256=source["source_binding_sha256"],
            source_receipt_sha256=spec["source_receipt_sha256"])
        banks, law, descriptor = banks[:2], banks[2], banks[3]
        source_receipt_sha = spec["source_receipt_sha256"]
        if (descriptor["fine_to_code_sha256_by_recipient"] !=
                spec.get("fine_to_code_sha256_by_recipient") or
                descriptor["full_bank_array_sha256_by_recipient"] !=
                spec.get("full_bank_array_sha256_by_recipient") or
                descriptor["mechanism_sha256"] != spec.get("mechanism_sha256")):
            raise PermissionError("Q fixed public map differs from sealed descriptor")
    if hashlib.sha256(_canonical(descriptor)).hexdigest() != spec["source_descriptor_sha256"]:
        raise PermissionError("Q actual public law descriptor differs from queued identity")
    if not callable(law):
        raise PermissionError("Q public mechanism lacks a sparse law")
    table = compile_uniform_prototype_features(
        banks[0], banks[1], c1, c2, law,
        public_centroids_sha256=spec["public_centroids_sha256"])
    if (table.identity["prototype_count"] != 64 * 641 or
            table.identity.get("zero_support_rule") != ZERO_SUPPORT_RULE or
            table.identity["centroid_array_sha256"] !=
            centroid_record["prototype_array_pair_sha256"]):
        raise PermissionError("Q table does not match the admitted uniform public grid")
    output.mkdir(mode=0o700)
    saved = save_q_table_artifact(table, output,
                                  policy_receipt_sha256=source_receipt_sha)
    return {"status": "FROZEN_PUBLIC_PROTOTYPE", "unit_id": args.unit_id,
            "source_kind": plan["source_kind"], "arm_id": arm,
            "seed": plan["seed"], "source_descriptor_sha256":
                spec["source_descriptor_sha256"],
            "zero_support_rule": ZERO_SUPPORT_RULE,
            "q_receipt_sha256": saved["receipt_sha256"],
            "q_table_content_sha256": table.table_sha256,
            "public_centroids_sha256": spec["public_centroids_sha256"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    for name in ("unit-id", "original-lock", "a-amendment",
                 "a-amendment-sha256", "campaign-root", "private-output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
