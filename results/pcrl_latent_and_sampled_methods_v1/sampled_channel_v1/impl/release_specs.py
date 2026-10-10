"""Read-only queue spec renderer for selected sampled-channel releases.

R03 owns enqueue and execution. This module only turns an already completed,
selected FIT parent into one immutable release spec; it never reads role rows.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from results.pcrl_latent_and_sampled_methods_v1.r03.a_downstream_specs import (
    _flag, release_spec,
)
from .run_release import _control_selected_provenance, _selected_parent_identity


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ROLES = frozenset({"ATTACK_FIT_NEW", "INNER_SELECTION"})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _private_directory(path: Path) -> Path:
    supplied = Path(path)
    if supplied.is_symlink():
        raise PermissionError("selected release directory cannot be a symlink")
    resolved = supplied.resolve(strict=True)
    if (".private" not in resolved.parts or not resolved.is_dir() or
            resolved.stat().st_uid != os.getuid() or
            stat.S_IMODE(resolved.stat().st_mode) & 0o077):
        raise PermissionError("selected release directory is not owner-only private")
    return resolved


def _selected_parent(fit_dir: Path, *, parent: dict, amendment_sha256: str,
                     fit_unit_id: str, arm_id: str, seed: int) -> tuple[dict, dict]:
    fit_path = fit_dir / "FIT_RECEIPT.json"
    policy_path = fit_dir / "POLICY_RECEIPT.json"
    policy_file = fit_dir / "policy.npz"
    if ((fit_dir / "INCOMPLETE_RESOURCE.json").exists() or
            parent.get("status") != "done" or
            parent.get("spec", {}).get("unit_id") != fit_unit_id or
            parent.get("spec", {}).get("stage") != "a" or
            parent.get("spec", {}).get("mode") != "real_data" or
            parent.get("spec", {}).get("arm_id") != arm_id or
            parent.get("spec", {}).get("seed") != seed or
            parent.get("spec", {}).get("amendment_sha256") != amendment_sha256 or
            parent.get("spec", {}).get("outputs") != [str(fit_path)] or
            parent.get("output_sha256", {}).get(str(fit_path)) != _sha(fit_path)):
        raise PermissionError("selected release lacks the exact completed FIT output")
    fit = json.loads(fit_path.read_text())
    if (fit.get("status") != "SELECTED_FIT_POLICY" or
            not policy_path.is_file() or not policy_file.is_file()):
        raise PermissionError("completed FIT has no selected policy")
    policy = json.loads(policy_path.read_text())
    _selected_parent_identity(fit, policy, unit_id=fit_unit_id,
                              arm_id=arm_id, seed=seed,
                              amendment_sha256=amendment_sha256)
    if fit.get("schema") == "SCV1-private-stochastic-fit-v1":
        selected = fit.get("selected")
        starts = fit.get("starts", {})
        selected_name = ({0: "identity_biased", 1: "uniform_logit"}
                         .get(selected.get("start")) if type(selected) is dict else None)
        if (set(starts) != {"identity_biased", "uniform_logit"} or
                any(starts[name].get("completed") is not True for name in starts) or
                selected_name is None or
                not any(checkpoint.get("update") == selected.get("update") and
                        checkpoint.get("feasible") is True
                        for checkpoint in starts[selected_name].get("checkpoints", []))):
            raise PermissionError("stochastic FIT lacks a feasible completed checkpoint")
    elif fit.get("schema") == "SCV1-private-control-fit-v1":
        detail = fit.get("fit")
        if (type(detail) is not dict or detail.get("complete") is not True or
                detail.get("success") is not True):
            raise PermissionError("control FIT lacks a complete selected policy")
        start, update = _control_selected_provenance(arm_id, detail)
        if (policy.get("selected_start") != start or
                policy.get("selected_update") != update):
            raise PermissionError("control policy differs from selected FIT checkpoint")
    if (policy.get("fit_receipt_sha256") != _sha(fit_path) or
            policy.get("policy_file_sha256") != _sha(policy_file) or
            type(policy.get("mechanism_sha256")) is not str or
            HEX64.fullmatch(policy["mechanism_sha256"]) is None):
        raise PermissionError("selected policy package differs from FIT receipt")
    return fit, policy


def selected_release_spec(*, unit_id: str, amendment_path: Path,
                          amendment_sha256: str, seal_path: Path,
                          seal_sha256: str, queue_state: Path,
                          private_parent: Path) -> dict:
    """Render one of the sealed release nodes, refusing unselected FIT parents."""
    amendment_path = Path(amendment_path).resolve(strict=True)
    queue_state = Path(queue_state).resolve(strict=True)
    amendment = json.loads(amendment_path.read_text())
    node = amendment.get("unit_plan", {}).get(unit_id)
    if (type(node) is not dict or node.get("kind") != "release" or
            node.get("role") not in ROLES or
            node.get("realization") not in (0, 1, 2) or
            node.get("cpu_seconds") != 30 or node.get("wall_seconds") != 30 or
            node.get("deps") != [node.get("fit_unit_id")]):
        raise ValueError("selected release differs from sealed role/draw/resource plan")
    parent_dir = _private_directory(private_parent)
    output = parent_dir / unit_id
    if output.exists() or output.is_symlink():
        raise FileExistsError("selected release output already exists")
    state = json.loads(queue_state.read_text())
    parent_id = node["fit_unit_id"]
    parent = state.get("units", {}).get(parent_id, {})
    parent_spec = parent.get("spec", {})
    parent_argv = parent_spec.get("argv", [])
    fit_dir = _private_directory(Path(_flag(parent_argv, "private_output")))
    if fit_dir != parent_dir / parent_id:
        raise PermissionError("selected FIT output directory differs from unit identity")
    fit, policy = _selected_parent(
        fit_dir, parent=parent, amendment_sha256=amendment_sha256,
        fit_unit_id=parent_id, arm_id=node["arm_id"], seed=node["seed"])
    identities = amendment["bank_action_sha256_by_seed"][f"s{node['seed']}"]
    if (fit.get("source_binding_sha256") !=
            amendment["role_bindings"]["source_binding_sha256_by_seed"][f"s{node['seed']}"] or
            fit.get("row_unit_pair_sha256") !=
            amendment["role_bindings"]["fit_row_unit_pair_sha256"] or
            fit.get("bank_sha256_by_recipient") !=
            {"r1": identities["bank_r1"], "r2": identities["bank_r2"]} or
            fit.get("action_sha256_by_recipient") !=
            {"r1": identities["action_r1"], "r2": identities["action_r2"]}):
        raise PermissionError("selected FIT identity differs from sealed FIT source")
    stream = amendment["realizations"]
    manifest_path = amendment_path.parent / "sampled_channel_v1/design/STREAM_ROOT_HASH_MANIFEST.json"
    if _sha(manifest_path) != stream.get("stream_root_hash_manifest_sha256"):
        raise PermissionError("public stream manifest differs from sealed commitment")
    manifest = json.loads(manifest_path.read_text())
    if (stream.get("ids") != [0, 1, 2] or
            stream.get("release_root_sha256_by_realization") !=
            manifest.get("release_realization_root_sha256") or
            stream.get("seq_fit_root_sha256") != manifest.get("seq_fit_root_sha256") or
            len(set(manifest["release_realization_root_sha256"].values()) |
                {manifest["seq_fit_root_sha256"]}) != 4):
        raise PermissionError("release streams are not three distinct sealed draws")
    private_stream = amendment_path.parent / "sampled_channel_v1/design/.private/stream_roots_v1.json"
    if (private_stream.is_symlink() or not private_stream.is_file() or
            stat.S_IMODE(private_stream.stat().st_mode) != 0o600 or
            _sha(private_stream) != manifest.get("private_file_sha256")):
        raise PermissionError("private release stream file differs from commitment")
    spec = release_spec(
        unit_id=unit_id, amendment_path=amendment_path,
        amendment_sha256=amendment_sha256, seal_path=seal_path,
        seal_sha256=seal_sha256, queue_state=queue_state,
        private_parent=parent_dir)
    sources = {entry["path"]: entry["sha256"] for entry in spec["sources"]}
    for path in (fit_dir / "FIT_RECEIPT.json", fit_dir / "POLICY_RECEIPT.json",
                 fit_dir / "policy.npz", private_stream, manifest_path):
        if sources.get(str(path)) != _sha(path):
            raise PermissionError("release spec omitted selected parent or stream bytes")
    if (spec.get("deps") != [parent_id] or
            spec.get("outputs") != [str(output / "RELEASE_RECEIPT.json")] or
            spec.get("cpu_reserve_s") != 30 or spec.get("max_wall_s") != 30 or
            spec.get("max_rss_gib", 0) > 3 or
            spec.get("argv", [None, None, None])[2] !=
            ("results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1."
             + ("impl.run_release" if node["arm_id"] in
                {"A_TASK_STOCH", "A_JOINT_STOCH"} else
                "controls.run_control_release"))):
        raise PermissionError("release spec differs from sealed execution contract")
    own_path = Path(__file__).resolve(strict=True)
    if str(own_path) not in sources:
        spec["sources"].append({"path": str(own_path), "sha256": _sha(own_path)})
        spec["sources"].sort(key=lambda row: row["path"])
    return spec


__all__ = ["selected_release_spec"]
