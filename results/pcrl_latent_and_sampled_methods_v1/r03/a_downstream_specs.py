"""Render A release queue specs only after their selected FIT parent is complete.

This stays separate from the frozen first-FIT renderer. It reads metadata and
hashes, never role arrays, and never enqueues or starts a worker.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .a_graph import compile_graph
from .a_specs import PYTHON, THREAD_ENV, _add_source, _owner_private_dir


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flag(argv: list[str], name: str) -> str:
    flag = "--" + name.replace("_", "-")
    if argv.count(flag) != 1:
        raise ValueError(f"parent argv lacks unique {flag}")
    return argv[argv.index(flag) + 1]


def release_spec(*, unit_id: str, amendment_path: Path, amendment_sha256: str,
                 seal_path: Path, seal_sha256: str, queue_state: Path,
                 private_parent: Path) -> dict:
    amendment_path = Path(amendment_path).resolve(strict=True)
    seal_path = Path(seal_path).resolve(strict=True)
    queue_state = Path(queue_state).resolve(strict=True)
    private_parent = Path(private_parent).resolve(strict=True)
    _owner_private_dir(private_parent)
    campaign = amendment_path.parent
    worktree = campaign.parents[1]
    compile_graph(amendment_path, amendment_sha256, seal_path, seal_sha256,
                  source_root=campaign, expected_units=220)
    amendment = json.loads(amendment_path.read_text())
    plan = amendment["unit_plan"].get(unit_id)
    if not isinstance(plan, dict) or plan.get("kind") != "release":
        raise ValueError("A unit is not a selected policy release")
    state = json.loads(queue_state.read_text())
    latest = [entry for entry in state["amendments"] if "a" in entry.get("stages", [])]
    if (not latest or latest[-1]["sha256"] != amendment_sha256 or
            Path(latest[-1]["path"]).resolve(strict=True) != amendment_path or
            state.get("occupancy") is not None):
        raise PermissionError("A release lacks latest idle amendment")
    fit_id = plan["fit_unit_id"]
    parent = state["units"].get(fit_id, {})
    if parent.get("status") != "done":
        raise ValueError("selected FIT dependency is not done")
    parent_spec = parent["spec"]
    parent_argv = parent_spec["argv"]
    policy_dir = Path(_flag(parent_argv, "private_output")).resolve(strict=True)
    fit_receipt = policy_dir / "FIT_RECEIPT.json"
    policy_receipt = policy_dir / "POLICY_RECEIPT.json"
    policy_file = policy_dir / "policy.npz"
    if (parent_spec.get("unit_id") != fit_id or
            parent_spec.get("amendment_sha256") != amendment_sha256 or
            parent_spec.get("arm_id") != plan["arm_id"] or
            parent_spec.get("seed") != plan["seed"] or
            parent_spec.get("outputs") != [str(fit_receipt)] or
            parent.get("output_sha256", {}).get(str(fit_receipt)) != _sha(fit_receipt) or
            json.loads(fit_receipt.read_text()).get("status") != "SELECTED_FIT_POLICY"):
        raise ValueError("selected FIT parent receipt/identity differs")
    sources = {row["path"]: row["sha256"] for row in parent_spec["sources"]}
    for path in (fit_receipt, policy_receipt, policy_file):
        _add_source(sources, path, _sha(path))
    repair_manifest = Path(_flag(parent_argv, "repair_manifest")).resolve(strict=True)
    portfolio = Path(_flag(parent_argv, "portfolio_package")).resolve(strict=True)
    seed, role = plan["seed"], plan["role"]
    bank = portfolio / f"shared/.private/tstar_score_bank_s{seed}_v1"
    bank_receipt = json.loads((bank / "BANK_ADMISSION_RECEIPT.json").read_text())
    token = bank / f"tokens_{role}.npz"
    _add_source(sources, token, bank_receipt["private_output_sha256"][token.name])
    repair = json.loads(repair_manifest.read_text())
    reference_key = f"{role}:references_s{seed}"
    reference = (repair_manifest.parent.parent /
                 repair["role_file_relative_path"][reference_key]).resolve(strict=True)
    _add_source(sources, reference, repair["role_file_sha256"][reference_key])
    stream = campaign / "sampled_channel_v1/design/.private/stream_roots_v1.json"
    _add_source(sources, stream, _sha(stream))
    output = private_parent / unit_id
    if output.exists() or output.is_symlink():
        raise FileExistsError("A release output already exists")
    arm = plan["arm_id"]
    module = ("results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1."
              + ("impl.run_release" if arm in {"A_TASK_STOCH", "A_JOINT_STOCH"}
                 else "controls.run_control_release"))
    argv = [str(PYTHON), "-m", module,
            "--unit-id", unit_id, "--arm-id", arm,
            "--role", role, "--seed", str(seed),
            "--realization", str(plan["realization"]),
            "--original-lock", str(seal_path),
            "--a-amendment", str(amendment_path),
            "--a-amendment-sha256", amendment_sha256,
            "--private-output", str(output),
            "--repair-manifest", str(repair_manifest),
            "--source-audit", str(campaign / "r01/SOURCE_AUDIT.json"),
            "--portfolio-package", str(portfolio),
            "--campaign-root", str(campaign),
            "--policy-dir", str(policy_dir),
            "--private-stream-roots", str(stream)]
    return {
        "unit_id": unit_id, "stage": "a", "mode": "real_data",
        "data_access": "ADMITTED_RELEASE_ROLE_ONLY",
        "amendment_sha256": amendment_sha256,
        "arm_id": arm, "seed": seed, "role": role,
        "realization": plan["realization"], "output_id": plan["output_id"],
        "fit_unit_id": fit_id, "fit_receipt_sha256": _sha(fit_receipt),
        "policy_receipt_sha256": _sha(policy_receipt),
        "policy_file_sha256": _sha(policy_file),
        "argv": argv, "cwd": str(worktree), "env": THREAD_ENV,
        "deps": [fit_id],
        "sources": [{"path": path, "sha256": digest}
                    for path, digest in sorted(sources.items())],
        "outputs": [str(output / "RELEASE_RECEIPT.json")],
        "cpu_reserve_s": plan["cpu_seconds"],
        "max_wall_s": plan["wall_seconds"],
        "max_rss_gib": 2.0, "projected_output_bytes": 200_000_000,
    }
