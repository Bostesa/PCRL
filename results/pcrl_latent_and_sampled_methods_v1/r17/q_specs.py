"""Read-only, receipt-bound queue spec rendering for the 30 A public-Q units.

R03 owns enqueue and execution. This renderer reads public fine centroids/maps
and completed FIT receipt metadata; it never opens a role array or starts Q.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from results.pcrl_latent_and_sampled_methods_v1.r03.a_graph import compile_graph
from results.pcrl_latent_and_sampled_methods_v1.r03.a_specs import (
    PYTHON, THREAD_ENV, _add_source, _latest_a_amendment, _owner_private_dir,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.downstream_sources import (
    verify_downstream_registration, verify_downstream_sidecar,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks import (
    run_q_preprocess as q,
)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _add(sources: dict[str, str], path: Path, digest: str) -> None:
    if path.is_symlink():
        raise PermissionError("Q source cannot be a symlink")
    _add_source(sources, path, digest)


def _flag(argv: list[str], name: str) -> str:
    flag = "--" + name.replace("_", "-")
    if type(argv) is not list or argv.count(flag) != 1:
        raise ValueError("Q FIT parent argv lacks a unique output flag")
    return argv[argv.index(flag) + 1]


def _selected_fields(*, plan: dict, state: dict, amendment_sha256: str,
                     sources: dict[str, str]) -> dict:
    """Bind one completed selected FIT to its private policy receipt bytes."""
    fit_id = plan["fit_unit_id"]
    parent = state.get("units", {}).get(fit_id, {})
    parent_spec = parent.get("spec", {})
    if (parent.get("status") != "done" or
            parent_spec.get("unit_id") != fit_id or
            parent_spec.get("stage") != "a" or
            parent_spec.get("amendment_sha256") != amendment_sha256 or
            parent_spec.get("arm_id") != plan["arm_id"] or
            parent_spec.get("seed") != plan["seed"]):
        raise ValueError("Q selected FIT parent is not completed under this amendment")
    policy_dir = q._private_existing_dir(Path(_flag(parent_spec.get("argv"),
                                                     "private_output")))
    fit_path = policy_dir / "FIT_RECEIPT.json"
    policy_path = policy_dir / "POLICY_RECEIPT.json"
    parameter_path = policy_dir / "policy.npz"
    for path in (fit_path, policy_path, parameter_path):
        if path.is_symlink() or not path.is_file():
            raise PermissionError("Q selected FIT package is incomplete or symlinked")
    fit_sha, policy_sha, parameter_sha = map(_sha, (fit_path, policy_path,
                                                    parameter_path))
    if (parent_spec.get("outputs") != [str(fit_path)] or
            parent.get("output_sha256", {}).get(str(fit_path)) != fit_sha):
        raise ValueError("Q selected FIT output differs from completed queue receipt")
    fit = json.loads(fit_path.read_text())
    policy = json.loads(policy_path.read_text())
    if (fit.get("status") != "SELECTED_FIT_POLICY" or
            policy.get("status") != "SELECTED_FIT_POLICY" or
            fit.get("unit_id") != fit_id or
            fit.get("a_amendment_sha256") != amendment_sha256 or
            policy.get("fit_receipt_sha256") != fit_sha or
            (policy_dir / "INCOMPLETE_RESOURCE.json").exists()):
        raise ValueError("Q selected FIT has no complete selected policy")
    q._selected_parent_identity(fit, policy, plan, amendment_sha256)
    for path, digest in ((fit_path, fit_sha), (policy_path, policy_sha),
                         (parameter_path, parameter_sha)):
        _add(sources, path, digest)
    descriptor = {
        "source_kind": "selected_fit", "arm_id": plan["arm_id"],
        "seed": plan["seed"], "release_id": policy["release_id"],
        "source_binding_sha256": policy["source_binding_sha256"],
        "source_receipt_sha256": policy_sha,
        "mechanism_sha256": policy["mechanism_sha256"],
        "canonical_bank_sha256_by_recipient": policy["bank_sha256_by_recipient"],
    }
    return {
        "policy_dir": str(policy_dir), "fit_receipt_sha256": fit_sha,
        "policy_receipt_sha256": policy_sha,
        "policy_file_sha256": parameter_sha,
        "source_binding_sha256": policy["source_binding_sha256"],
        "mechanism_sha256": policy["mechanism_sha256"],
        "source_descriptor_sha256": hashlib.sha256(q._canonical(descriptor)).hexdigest(),
    }


def _fixed_fields(*, plan: dict, campaign: Path, amendment: dict,
                  historical_audit_sha256: str, centroids: tuple,
                  sources: dict[str, str]) -> dict:
    """Construct the exact public fine-cell law without role rows."""
    arm = plan.get("arm_id", plan.get("reference_id"))
    seed = plan["seed"]
    source_arm = "T_STAR" if arm in q.DIAGNOSTIC else arm
    prefix = f"r01/A_FIXED_MAP_{source_arm}_S{seed}"
    receipt_path, npz_path = campaign / (prefix + ".json"), campaign / (prefix + ".npz")
    source_map = amendment["source_sha256"]
    receipt_sha = source_map.get(prefix + ".json")
    npz_sha = source_map.get(prefix + ".npz")
    if not q._hex(receipt_sha) or not q._hex(npz_sha):
        raise ValueError("Q fixed public map is absent from latest amendment")
    _add(sources, receipt_path, receipt_sha)
    _add(sources, npz_path, npz_sha)
    receipt = json.loads(receipt_path.read_text())
    source_audit_sha = (amendment["role_bindings"]["source_audit_sha256"]
                        if source_arm == "T_STAR" else historical_audit_sha256)
    release_id = (f"{arm}-s{seed}-v1" if arm in q.DIAGNOSTIC
                  else receipt.get("release_id"))
    preliminary = {
        "fixed_map_receipt_sha256": receipt_sha,
        "fixed_map_npz_sha256": npz_sha,
        "source_audit_sha256": source_audit_sha,
        "source_receipt_sha256": receipt.get("original_bank_receipt_sha256"),
        "source_binding_sha256": receipt.get("source_binding_sha256"),
        "release_id": release_id,
    }
    if (not q._hex(preliminary["source_receipt_sha256"]) or
            not q._hex(preliminary["source_binding_sha256"])):
        raise ValueError("Q fixed source receipt or binding is malformed")
    source, records = q._fixed_source(preliminary, campaign, arm, seed, centroids)
    _, _, _, descriptor = q.build_fixed_public_policy(
        arm_id=arm, seed=seed, source_kind=plan["source_kind"],
        records=records, release_id=release_id,
        source_binding_sha256=source["source_binding_sha256"],
        source_receipt_sha256=preliminary["source_receipt_sha256"])
    preliminary.update({
        "mechanism_sha256": descriptor["mechanism_sha256"],
        "fine_to_code_sha256_by_recipient":
            descriptor["fine_to_code_sha256_by_recipient"],
        "full_bank_array_sha256_by_recipient":
            descriptor["full_bank_array_sha256_by_recipient"],
        "source_descriptor_sha256": hashlib.sha256(q._canonical(descriptor)).hexdigest(),
    })
    return preliminary


def q_spec(*, unit_id: str, amendment_path: Path, amendment_sha256: str,
           seal_path: Path, seal_sha256: str, queue_state: Path,
           private_parent: Path, sidecar_path: Path, sidecar_sha256: str,
           registration_path: Path, registration_sha256: str) -> dict:
    """Render one immutable R03 Q spec after all applicable parents complete."""
    if Path(amendment_path).is_symlink() or Path(seal_path).is_symlink():
        raise PermissionError("Q seal paths cannot be symlinks")
    amendment_path = Path(amendment_path).resolve(strict=True)
    seal_path = Path(seal_path).resolve(strict=True)
    queue_state = Path(queue_state).resolve(strict=True)
    private_parent = Path(private_parent).resolve(strict=True)
    if Path(sidecar_path).is_symlink() or Path(registration_path).is_symlink():
        raise PermissionError("Q downstream source custody cannot be symlinked")
    sidecar_path = Path(sidecar_path).resolve(strict=True)
    registration_path = Path(registration_path).resolve(strict=True)
    _owner_private_dir(private_parent)
    campaign = amendment_path.parent
    worktree = campaign.parents[1]
    if seal_path.parent != campaign:
        raise ValueError("Q original lock is outside campaign")
    _latest_a_amendment(queue_state, amendment_path, amendment_sha256,
                        seal_path, seal_sha256)
    compile_graph(amendment_path, amendment_sha256, seal_path, seal_sha256,
                  source_root=campaign, expected_units=220)
    amendment = json.loads(amendment_path.read_text())
    plan = amendment["unit_plan"].get(unit_id)
    if type(plan) is not dict or plan.get("kind") != "q_preprocess":
        raise ValueError("requested unit is not a sealed Q node")
    arm = plan.get("arm_id", plan.get("reference_id"))
    seed = plan["seed"]
    output = q._private_new_output(private_parent / unit_id)
    sources: dict[str, str] = {}
    _add(sources, seal_path, seal_sha256)
    _add(sources, amendment_path, amendment_sha256)
    _add(sources, sidecar_path, sidecar_sha256)
    _add(sources, registration_path, registration_sha256)
    for relative in sorted(amendment["implementation_source_sha256"]):
        digest = amendment["implementation_source_sha256"].get(relative)
        if not q._hex(digest):
            raise ValueError("Q implementation is absent from latest amendment")
        _add(sources, campaign / relative, digest)
    sidecar = json.loads(sidecar_path.read_text())
    for relative, digest in sorted(sidecar.get("new_source_sha256", {}).items()):
        _add(sources, campaign / relative, digest)
    shared = (sidecar.get("attack_plan", {})
              .get("inherited_shared_source_sha256", {}))
    for basename, digest in sorted(shared.items()):
        _add(sources, campaign.parent / "pcrl_method_portfolio_v1/shared" /
             basename, digest)
    sidecar_fields = {
        "amendment_sha256": amendment_sha256,
        "downstream_manifest_path": str(sidecar_path),
        "downstream_manifest_sha256": sidecar_sha256,
        "downstream_registration_path": str(registration_path),
        "downstream_registration_sha256": registration_sha256,
        "sources": [{"path": path, "sha256": digest}
                    for path, digest in sorted(sources.items())],
    }
    verify_downstream_sidecar(
        sidecar_path=sidecar_path, sidecar_sha256=sidecar_sha256,
        producer_path=amendment_path, producer_sha256=amendment_sha256,
        original_lock_path=seal_path, original_lock_sha256=seal_sha256,
        campaign_root=campaign, spec=sidecar_fields,
        required_new_paths=frozenset({"r17/q_specs.py"}))
    verify_downstream_registration(
        registration_path=registration_path,
        registration_sha256=registration_sha256,
        sidecar_path=sidecar_path, sidecar_sha256=sidecar_sha256,
        producer_sha256=amendment_sha256, repo_root=worktree,
        spec=sidecar_fields)
    historical_rel = "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"
    historical_sha = amendment["source_sha256"].get(historical_rel)
    if not q._hex(historical_sha):
        raise ValueError("Q historical audit is absent from latest amendment")
    historical_path = campaign / historical_rel
    _add(sources, historical_path, historical_sha)
    historical = json.loads(historical_path.read_text())
    centroid_record = historical.get("public_centroids", {}).get(f"s{seed}", {})
    centroid_path = campaign / f"r01/A_PUBLIC_FINE_CENTROIDS_S{seed}.npz"
    centroid_sha = centroid_record.get("clean_sha256")
    if (not q._hex(centroid_sha) or
            amendment["source_sha256"].get(f"r01/{centroid_path.name}") != centroid_sha):
        raise ValueError("Q centroid source differs from latest amendment")
    _add(sources, centroid_path, centroid_sha)
    centroids = q.load_clean_public_centroids(centroid_path, centroid_record)
    fields = {"historical_audit_sha256": historical_sha,
              "public_centroids_sha256": centroid_sha,
              "zero_support_rule": q.ZERO_SUPPORT_RULE}
    if arm in q.FITTED:
        fields.update(_selected_fields(plan=plan, state=json.loads(queue_state.read_text()),
                                       amendment_sha256=amendment_sha256,
                                       sources=sources))
    else:
        fields.update(_fixed_fields(plan=plan, campaign=campaign,
                                    amendment=amendment,
                                    historical_audit_sha256=historical_sha,
                                    centroids=centroids, sources=sources))
    module = ("results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1."
              "attacks.run_q_preprocess")
    argv = [str(PYTHON), "-m", module,
            "--unit-id", unit_id,
            "--original-lock", str(seal_path),
            "--a-amendment", str(amendment_path),
            "--a-amendment-sha256", amendment_sha256,
            "--campaign-root", str(campaign),
            "--private-output", str(output)]
    spec = {
        "unit_id": unit_id, "stage": "a", "mode": "real_data",
        "data_access": "PUBLIC_Q_ONLY", "amendment_sha256": amendment_sha256,
        "downstream_manifest_path": str(sidecar_path),
        "downstream_manifest_sha256": sidecar_sha256,
        "downstream_registration_path": str(registration_path),
        "downstream_registration_sha256": registration_sha256,
        "arm_id": arm, "seed": seed, "source_kind": plan["source_kind"],
        "output_id": plan["output_id"], "fit_unit_id": plan.get("fit_unit_id"),
        "deps": list(plan.get("deps", [])), "argv": argv,
        "cwd": str(worktree), "env": THREAD_ENV,
        "sources": [{"path": path, "sha256": digest}
                    for path, digest in sorted(sources.items())],
        "outputs": [str(output / name) for name in plan["expected_outputs"]],
        "cpu_reserve_s": plan["cpu_seconds"],
        "max_wall_s": plan["wall_seconds"], "max_rss_gib": 1.0,
        "projected_output_bytes": 25_000_000,
        **fields,
    }
    q.validate_q_unit_contract(unit_id, plan, spec, output)
    return spec
