"""Render one immutable A ATT/INNER attack spec from completed parent receipts.

Only public protocol files, queue metadata and parent JSON receipts are read.
The renderer does not open role values, saved code arrays, caches, policy arrays,
Q arrays or SEX labels. R03 alone registers and runs the resulting spec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from results.pcrl_latent_and_sampled_methods_v1.r03.a_graph import compile_graph
from results.pcrl_latent_and_sampled_methods_v1.r03.a_specs import (
    PYTHON, THREAD_ENV, _owner_private_dir,
)


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
KINDS = frozenset({"attack", "reference_attack", "diagnostic_attack"})
FINITE_REFS = frozenset({"T_STAR", "P_STAR", "D0_LOCAL", "D0_SEQ12", "D0_SEQ21"})
CONTINUOUS_REFS = frozenset({"U0", "Ucal"})
DIAGNOSTICS = frozenset({"A_IDENTITY", "A_CLASS_ONLY"})
SHARED = ("attack_eval.py", "attack_controls_v2.py", "common_eval.py",
          "safe_data_guard.py")
ACCESS = "ATTACK_FIT_NEW_SEX_THEN_INNER_SELECTION_SEX"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check_sha(value: object) -> str:
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError("attack source SHA-256 is malformed")
    return value


def _add(sources: dict[str, str], path: Path, digest: str) -> None:
    """Register a hash without opening a private array/cache payload."""
    supplied = Path(path)
    if supplied.is_symlink():
        raise PermissionError("attack source path is a symlink")
    resolved = supplied.resolve(strict=True)
    digest = _check_sha(digest)
    prior = sources.setdefault(str(resolved), digest)
    if prior != digest:
        raise ValueError("attack source has conflicting parent hashes")


def _add_parent_sources(sources: dict[str, str], spec: dict) -> None:
    rows = spec.get("sources")
    if type(rows) is not list:
        raise ValueError("parent source list is missing")
    for row in rows:
        if type(row) is not dict or row.keys() != {"path", "sha256"}:
            raise ValueError("parent source row differs")
        _add(sources, Path(row["path"]), row["sha256"])


def _private_dir(path: Path) -> Path:
    supplied = Path(path)
    protected = []
    for ancestor in (supplied, *supplied.parents):
        if ancestor.is_symlink():
            raise PermissionError("parent artifact ancestry is a symlink")
        protected.append(ancestor)
        if ancestor.name == ".private":
            break
    else:
        raise PermissionError("parent artifact is outside private custody")
    root = supplied.resolve(strict=True)
    for ancestor in protected:
        _owner_private_dir(ancestor)
    return root


def _parent(state: dict, queue_root: Path, unit_id: str,
            names: tuple[str, ...]) -> tuple[dict, dict, Path, dict[str, dict]]:
    """Use only a completed R03 receipt and small JSON parent artifacts."""
    unit = state.get("units", {}).get(unit_id, {})
    spec = unit.get("spec", {})
    receipt_path = queue_root / "units" / unit_id / "RECEIPT.json"
    if not receipt_path.is_file():
        raise ValueError("attack parent queue receipt is absent")
    queue_receipt = json.loads(receipt_path.read_text())
    outputs = spec.get("outputs")
    if (unit.get("status") != "done" or spec.get("unit_id") != unit_id or
            queue_receipt.get("status") != "done" or
            queue_receipt.get("returncode") != 0 or
            queue_receipt.get("spec_sha256") != unit.get("spec_sha256") or
            type(outputs) is not list or len(outputs) != len(names) or
            tuple(Path(path).name for path in outputs) != names or
            queue_receipt.get("output_sha256") != unit.get("output_sha256") or
            set(queue_receipt.get("output_sha256", {})) != set(outputs)):
        raise ValueError("attack parent is not an exact completed R03 unit")
    roots = {Path(path).parent for path in outputs}
    if len(roots) != 1:
        raise ValueError("attack parent outputs do not share one directory")
    root = _private_dir(roots.pop())
    docs = {}
    for path_string in outputs:
        path = Path(path_string)
        if path.is_symlink() or not path.is_file():
            raise ValueError("attack parent output is absent or redirected")
        digest = _check_sha(queue_receipt["output_sha256"][path_string])
        if path.suffix == ".json":
            if _sha(path) != digest:
                raise ValueError("attack parent JSON changed after R03 completion")
            docs[path.name] = json.loads(path.read_text())
    return spec, queue_receipt, root, docs


def _expect_parent(spec: dict, *, amendment_sha256: str, arm: str, seed: int,
                   role: str | None = None, draw: int | None = None) -> None:
    if (spec.get("stage") != "a" or spec.get("mode") != "real_data" or
            spec.get("amendment_sha256") != amendment_sha256 or
            spec.get("arm_id", spec.get("reference_id")) != arm or
            spec.get("seed") != seed or
            (role is not None and spec.get("role") != role) or
            (draw is not None and spec.get("realization") != draw)):
        raise ValueError("attack parent arm/seed/role/draw differs")


def _add_artifact(sources: dict[str, str], root: Path, name: str,
                  digest: str) -> None:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("attack parent artifact is absent or redirected")
    _add(sources, path, digest)


def _role_parent(state: dict, queue_root: Path, unit_id: str, *, kind: str,
                 amendment_sha256: str, arm: str, seed: int, role: str,
                 draw: int, sources: dict[str, str]) -> tuple[Path, str, dict]:
    selected = kind == "attack"
    name = "RELEASE_RECEIPT.json" if selected else "REFERENCE_RELEASE_RECEIPT.json"
    spec, queue_receipt, root, docs = _parent(state, queue_root, unit_id, (name,))
    _expect_parent(spec, amendment_sha256=amendment_sha256, arm=arm,
                   seed=seed, role=role, draw=draw)
    _add_parent_sources(sources, spec)
    receipt_path = root / name
    digest = queue_receipt["output_sha256"][str(receipt_path)]
    _add(sources, receipt_path, digest)
    record = docs[name]
    if (record.get("status") != ("FROZEN_ONE_DRAW" if selected else
                                "FROZEN_DETERMINISTIC") or
            record.get("source_seed") != seed or record.get("role") != role or
            (selected and record.get("realization") != draw) or
            (not selected and record.get("arm_id") != arm)):
        raise ValueError("saved ATT/INNER release receipt identity differs")
    if selected:
        for filename, field in (("release_codes.npz", "codes_file_sha256"),
                                ("cache.sqlite3", "cache_file_sha256")):
            _add_artifact(sources, root, filename, record[field])
    else:
        _add_artifact(sources, root, "reference_release.npz",
                      record["codes_file_sha256"])
    return root, digest, record


def _q_parent(state: dict, queue_root: Path, unit_id: str, *,
              amendment_sha256: str, arm: str, seed: int,
              policy_receipt_sha256: str,
              sources: dict[str, str]) -> tuple[Path, dict]:
    names = ("Q_TABLE_RECEIPT.json", "q_table.npz")
    spec, queue_receipt, root, docs = _parent(state, queue_root, unit_id, names)
    _expect_parent(spec, amendment_sha256=amendment_sha256, arm=arm, seed=seed)
    _add_parent_sources(sources, spec)
    receipt = docs[names[0]]
    receipt_sha = queue_receipt["output_sha256"][str(root / names[0])]
    data_sha = queue_receipt["output_sha256"][str(root / names[1])]
    if (receipt.get("status") != "FROZEN_PUBLIC_PROTOTYPE" or
            receipt.get("policy_receipt_sha256") != policy_receipt_sha256 or
            receipt.get("q_table_file_sha256") != data_sha):
        raise ValueError("attack Q receipt differs from selected public source")
    _add(sources, root / names[0], receipt_sha)
    _add(sources, root / names[1], data_sha)
    return root, {"receipt_sha256": receipt_sha,
                  "content_sha256": _check_sha(receipt["table_content_sha256"])}


def _selected_fit(record: dict, *, arm: str, seed: int,
                  fit_id: str, directory: Path) -> None:
    """Recognize both frozen FIT receipt schemas without a fallback choice."""
    if (record.get("status") != "SELECTED_FIT_POLICY" or
            record.get("unit_id") != fit_id or
            record.get("arm_id") != arm or record.get("source_seed") != seed or
            (directory / "INCOMPLETE_RESOURCE.json").exists()):
        raise ValueError("selected A FIT has no eligible frozen policy")
    if record.get("schema") == "SCV1-private-stochastic-fit-v1":
        valid = arm in {"A_JOINT_STOCH", "A_TASK_STOCH"} and record.get("selected") is not None
    elif record.get("schema") == "SCV1-private-control-fit-v1":
        fit = record.get("fit")
        valid = (arm not in {"A_JOINT_STOCH", "A_TASK_STOCH"} and
                 type(fit) is dict and fit.get("complete") is True and
                 fit.get("success") is True)
    else:
        valid = False
    if not valid:
        raise ValueError("selected A FIT schema has no frozen feasible policy")


def _locked_shared_sources(manifest: dict, shared: Path,
                           sources: dict[str, str]) -> None:
    """Bind inherited readers to S1's pre-attack source freeze."""
    plan = manifest.get("attack_plan")
    pins = (plan.get("inherited_shared_source_sha256")
            if type(plan) is dict else None)
    if type(pins) is not dict or set(pins) != set(SHARED):
        raise ValueError("A amendment lacks the exact inherited reader source freeze")
    for name in SHARED:
        expected = _check_sha(pins[name])
        path = shared / name
        if _sha(path) != expected:
            raise ValueError("inherited reader changed after the A source freeze")
        _add(sources, path, expected)


def _base(*, unit_id: str, amendment_path: Path, amendment_sha256: str,
          seal_path: Path, seal_sha256: str, role_manifest: Path,
          queue_state: Path, private_parent: Path,
          downstream_manifest_path: Path,
          downstream_manifest_sha256: str,
          downstream_registration_path: Path,
          downstream_registration_sha256: str):
    amendment_path = Path(amendment_path).resolve(strict=True)
    seal_path = Path(seal_path).resolve(strict=True)
    role_manifest = Path(role_manifest).resolve(strict=True)
    queue_state = Path(queue_state).resolve(strict=True)
    private_parent = _private_dir(Path(private_parent))
    if Path(downstream_manifest_path).is_symlink():
        raise PermissionError("downstream source manifest is a symlink")
    downstream_manifest_path = Path(downstream_manifest_path).resolve(strict=True)
    if (_sha(amendment_path) != _check_sha(amendment_sha256) or
            _sha(seal_path) != _check_sha(seal_sha256)):
        raise ValueError("attack amendment or original seal bytes differ")
    campaign = amendment_path.parent
    if downstream_manifest_path.parent != campaign or \
            _sha(downstream_manifest_path) != _check_sha(downstream_manifest_sha256):
        raise ValueError("downstream source manifest location or bytes differ")
    manifest = json.loads(downstream_manifest_path.read_text())
    from .run_downstream_attack import (MANIFEST_KEYS, SCHEMA, STATUS,
                                        THIS_MODULE, RENDERER_MODULE,
                                        validate_registration)
    amendment = json.loads(amendment_path.read_text())
    plan_digest = hashlib.sha256(json.dumps(
        amendment.get("unit_plan"), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    if (set(manifest) != MANIFEST_KEYS or manifest.get("schema") != SCHEMA or
            manifest.get("status") != STATUS or
            manifest.get("producer_amendment_sha256") != amendment_sha256 or
            manifest.get("original_lock_sha256") != seal_sha256 or
            manifest.get("unit_plan_sha256") != plan_digest or
            manifest.get("stage_scope") !=
            ["attack", "diagnostic", "q", "reference", "release"] or
            manifest.get("scientific_scope_unchanged") is not True):
        raise ValueError("downstream manifest is not bound to producer graph")
    if (type(manifest.get("attack_plan")) is not dict or
            set(manifest["attack_plan"]) != {
                "inherited_shared_source_sha256"}):
        raise ValueError("downstream reader source plan differs")
    if Path(__file__).resolve(strict=True) != \
            (campaign / RENDERER_MODULE).resolve(strict=True):
        raise PermissionError("executing downstream renderer origin differs")
    validate_registration(
        campaign=campaign, producer_sha256=amendment_sha256,
        manifest_path=downstream_manifest_path,
        manifest_sha256=downstream_manifest_sha256, manifest=manifest,
        registration_path=Path(downstream_registration_path),
        registration_sha256=downstream_registration_sha256)
    compile_graph(amendment_path, amendment_sha256, seal_path, seal_sha256,
                  source_root=campaign, expected_units=220)
    plan = amendment.get("unit_plan", {}).get(unit_id)
    if (type(plan) is not dict or plan.get("kind") not in KINDS or
            plan.get("output_id") != unit_id or
            plan.get("cpu_seconds") != 90 or plan.get("wall_seconds") != 90 or
            plan.get("realization") not in (0, 1, 2)):
        raise ValueError("unit is not a bounded A attack in the sealed 220 graph")
    state = json.loads(queue_state.read_text())
    latest = [entry for entry in state.get("amendments", [])
              if "a" in entry.get("stages", [])]
    if (not latest or latest[-1].get("sha256") != amendment_sha256 or
            Path(latest[-1].get("path", "")).resolve(strict=True) != amendment_path or
            state.get("seal", {}).get("sha256") != seal_sha256):
        raise PermissionError("attack spec lacks latest effective A seal")
    if (_sha(role_manifest) != amendment["role_bindings"]["repair_manifest_sha256"]):
        raise ValueError("attack role-only manifest differs from sealed reference")
    output = private_parent / unit_id
    if output.exists() or output.is_symlink():
        raise FileExistsError("attack output already exists")
    sources: dict[str, str] = {}
    _add(sources, seal_path, seal_sha256)
    _add(sources, amendment_path, amendment_sha256)
    _add(sources, downstream_manifest_path, downstream_manifest_sha256)
    _add(sources, Path(downstream_registration_path),
         downstream_registration_sha256)
    _add(sources, role_manifest,
         amendment["role_bindings"]["repair_manifest_sha256"])
    for relative, digest in amendment["implementation_source_sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("attack implementation source escapes campaign")
        _add(sources, campaign / path, digest)
    for relative, digest in amendment["source_sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("attack source inventory escapes campaign")
        _add(sources, campaign / path, digest)
    additions = manifest.get("new_source_sha256")
    if type(additions) is not dict or not {THIS_MODULE, RENDERER_MODULE} <= set(additions):
        raise ValueError("downstream wrapper or renderer is not source-pinned")
    for relative, digest in additions.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("downstream source escapes campaign")
        target = campaign / path
        if _sha(target) != _check_sha(digest):
            raise ValueError("downstream source differs from sidecar")
        _add(sources, target, digest)
    shared = campaign.parent / "pcrl_method_portfolio_v1" / "shared"
    _locked_shared_sources(manifest, shared, sources)
    return campaign, amendment, plan, state, queue_state.parent, output, sources


def attack_spec(*, unit_id: str, amendment_path: Path, amendment_sha256: str,
                seal_path: Path, seal_sha256: str, role_manifest: Path,
                queue_state: Path, private_parent: Path,
                downstream_manifest_path: Path,
                downstream_manifest_sha256: str,
                downstream_registration_path: Path,
                downstream_registration_sha256: str) -> dict:
    """Construct a selected, historical or diagnostic attack; never enqueue."""
    campaign, amendment, plan, state, queue_root, output, sources = _base(
        unit_id=unit_id, amendment_path=amendment_path,
        amendment_sha256=amendment_sha256, seal_path=seal_path,
        seal_sha256=seal_sha256, role_manifest=role_manifest,
        queue_state=queue_state, private_parent=private_parent,
        downstream_manifest_path=downstream_manifest_path,
        downstream_manifest_sha256=downstream_manifest_sha256,
        downstream_registration_path=downstream_registration_path,
        downstream_registration_sha256=downstream_registration_sha256)
    arm = plan.get("arm_id", plan.get("reference_id"))
    seed, draw, kind = plan["seed"], plan["realization"], plan["kind"]
    if type(arm) is not str or seed not in (0, 1):
        raise ValueError("attack arm or source seed differs")
    if kind == "diagnostic_attack":
        release_units = {role: f"A-DIAG-RELEASE-{arm}-S{seed}-{suffix}"
                         for role, suffix in (("ATTACK_FIT_NEW", "ATT"),
                                              ("INNER_SELECTION", "INNER"))}
    else:
        planned = plan.get("release_unit_ids")
        if type(planned) is not dict or planned.keys() != {"ATT", "INNER"}:
            raise ValueError("attack release-unit plan is incomplete")
        release_units = {"ATTACK_FIT_NEW": planned["ATT"],
                         "INNER_SELECTION": planned["INNER"]}
    expected_deps = ([plan["fit_unit_id"]] if kind == "attack" else []) + [
        release_units["ATTACK_FIT_NEW"], release_units["INNER_SELECTION"]]
    finite = kind == "attack" or arm in FINITE_REFS | DIAGNOSTICS
    if finite:
        expected_deps.append(plan["q_unit_id"])
    if (plan.get("deps") != expected_deps or
            (plan.get("q_unit_id") is not None) != finite or
            plan["trained_reader_and_probe_fits"] != (33 if finite else 24) or
            (kind == "diagnostic_attack" and arm not in DIAGNOSTICS) or
            (kind == "reference_attack" and
             arm not in FINITE_REFS | CONTINUOUS_REFS)):
        raise ValueError("attack parents or fit count differ from sealed graph")
    fit_parent = None
    if kind == "attack":
        fit_id = plan["fit_unit_id"]
        fit_parent = _parent(state, queue_root, fit_id, ("FIT_RECEIPT.json",))
        fit_spec, _, fit_dir, fit_docs = fit_parent
        _expect_parent(fit_spec, amendment_sha256=amendment_sha256,
                       arm=arm, seed=seed)
        _selected_fit(fit_docs["FIT_RECEIPT.json"], arm=arm, seed=seed,
                      fit_id=fit_id, directory=fit_dir)
    parent_records = {}
    for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        parent_records[role] = _role_parent(
            state, queue_root, release_units[role], kind=kind,
            amendment_sha256=amendment_sha256, arm=arm, seed=seed,
            role=role, draw=draw, sources=sources)
    att_dir, att_sha, att = parent_records["ATTACK_FIT_NEW"]
    inner_dir, inner_sha, inner = parent_records["INNER_SELECTION"]
    if (att.get("release_id") != inner.get("release_id") or
            att.get("source_binding_sha256") != inner.get("source_binding_sha256") or
            att.get("row_unit_pair_sha256") == inner.get("row_unit_pair_sha256")):
        raise ValueError("ATT/INNER saved release identity or disjointness differs")
    q_dir = None
    q_identity = None
    if kind == "attack":
        fit_id = plan["fit_unit_id"]
        fit_spec, fit_queue, fit_dir, fit_docs = fit_parent
        _add_parent_sources(sources, fit_spec)
        fit_sha = fit_queue["output_sha256"][str(fit_dir / "FIT_RECEIPT.json")]
        _add(sources, fit_dir / "FIT_RECEIPT.json", fit_sha)
        fit_receipt = fit_docs["FIT_RECEIPT.json"]
        policy_path = fit_dir / "POLICY_RECEIPT.json"
        if policy_path.is_symlink() or not policy_path.is_file():
            raise ValueError("selected policy receipt is absent")
        policy_sha = _sha(policy_path)
        policy = json.loads(policy_path.read_text())
        if (policy.get("status") != "SELECTED_FIT_POLICY" or
                policy.get("fit_unit_id") != fit_id or
                policy.get("fit_receipt_sha256") != fit_sha or
                policy.get("arm_id") != arm or policy.get("source_seed") != seed or
                policy.get("source_binding_sha256") !=
                att["source_binding_sha256"] or
                policy.get("release_id") != att["release_id"] or
                policy.get("mechanism_sha256") != att["mechanism_sha256"] or
                policy.get("bank_sha256_by_recipient") !=
                att["canonical_bank_sha256_by_recipient"]):
            raise ValueError("selected policy and actual release differ")
        _add(sources, policy_path, policy_sha)
        _add_artifact(sources, fit_dir, "policy.npz", policy["policy_file_sha256"])
        stream_path = campaign / "sampled_channel_v1/design/STREAM_ROOT_HASH_MANIFEST.json"
        streams = json.loads(stream_path.read_text())
        root_sha = streams["release_realization_root_sha256"][str(draw)]
        if (att.get("private_stream_root_sha256") != root_sha or
                inner.get("private_stream_root_sha256") != root_sha or
                att.get("mechanism_sha256") != inner.get("mechanism_sha256") or
                att.get("canonical_bank_sha256_by_recipient") !=
                inner.get("canonical_bank_sha256_by_recipient")):
            raise ValueError("sampled ATT/INNER draw or bank identity differs")
        q_dir, q_identity = _q_parent(
            state, queue_root, plan["q_unit_id"],
            amendment_sha256=amendment_sha256, arm=arm, seed=seed,
            policy_receipt_sha256=policy_sha, sources=sources)
        identity = {
            "arm_id": arm, "source_seed": seed, "realization": draw,
            "release_id": policy["release_id"],
            "source_binding_sha256": policy["source_binding_sha256"],
            "mechanism_sha256": policy["mechanism_sha256"],
            "canonical_bank_sha256_by_recipient":
                policy["bank_sha256_by_recipient"],
            "release_receipt_sha256_by_role": {
                "ATTACK_FIT_NEW": att_sha, "INNER_SELECTION": inner_sha},
            "private_stream_root_sha256_by_role": {
                "ATTACK_FIT_NEW": root_sha, "INNER_SELECTION": root_sha},
            "policy_receipt_sha256": policy_sha,
        }
        extra = {"attack_identity": identity, "q_table_identity": q_identity}
    else:
        finite_reference = arm in FINITE_REFS | DIAGNOSTICS
        map_receipt = map_npz = None
        if finite_reference:
            source_arm = "T_STAR" if arm in DIAGNOSTICS else arm
            stem = campaign / f"r01/A_FIXED_MAP_{source_arm}_S{seed}"
            map_receipt, map_npz = stem.with_suffix(".json"), stem.with_suffix(".npz")
            map_sha = amendment["source_sha256"][str(map_receipt.relative_to(campaign))]
            npz_sha = amendment["source_sha256"][str(map_npz.relative_to(campaign))]
            fixed_record = json.loads(map_receipt.read_text())
            if (fixed_record.get("source_binding_sha256") !=
                    att["source_binding_sha256"] or
                    fixed_record.get("original_bank_receipt_sha256") !=
                    att.get("source_identity", {}).get("bank_receipt_sha256")):
                raise ValueError("reference full bank or source binding differs")
            fixed_identity = {"receipt_sha256": map_sha, "npz_sha256": npz_sha}
            q_dir, q_identity = _q_parent(
                state, queue_root, plan["q_unit_id"],
                amendment_sha256=amendment_sha256, arm=arm, seed=seed,
                policy_receipt_sha256=fixed_record["original_bank_receipt_sha256"],
                sources=sources)
        else:
            fixed_identity = None
        identity = {
            "reference_id": arm, "source_seed": seed,
            "release_id": att["release_id"],
            "interface_kind": "finite" if finite_reference else "continuous",
            "source_binding_sha256": att["source_binding_sha256"],
            "release_receipt_sha256_by_role": {
                "ATTACK_FIT_NEW": att_sha, "INNER_SELECTION": inner_sha},
            "fixed_map_identity": fixed_identity,
            "q_table_identity": q_identity,
        }
        extra = {"reference_identity": identity}
    from .run_downstream_attack import MODULE_ID
    argv = [str(PYTHON), "-m", MODULE_ID,
            "--unit-id", unit_id, "--kind", kind,
            "--producer-amendment-path", str(amendment_path),
            "--producer-amendment-sha256", amendment_sha256,
            "--downstream-manifest-path", str(downstream_manifest_path),
            "--downstream-manifest-sha256", downstream_manifest_sha256,
            "--downstream-registration-path", str(downstream_registration_path),
            "--downstream-registration-sha256", downstream_registration_sha256,
            "--original-lock-path", str(seal_path),
            "--original-lock-sha256", seal_sha256,
            "--role-manifest-path", str(role_manifest),
            "--role-manifest-sha256", _sha(role_manifest),
            "--att-release-dir", str(att_dir),
            "--inner-release-dir", str(inner_dir),
            "--private-root", str(output)]
    if kind == "attack":
        argv.extend(("--policy-package-dir", str(fit_dir),
                     "--q-table-dir", str(q_dir)))
    elif finite:
        argv.extend(("--fixed-map-receipt", str(map_receipt),
                     "--fixed-map-npz", str(map_npz),
                     "--q-table-dir", str(q_dir)))
    worktree = campaign.parents[1]
    return {
        "unit_id": unit_id, "stage": "a", "mode": "real_data",
        "data_access": ACCESS, "amendment_sha256": amendment_sha256,
        "downstream_manifest_sha256": downstream_manifest_sha256,
        "downstream_manifest_path": str(Path(downstream_manifest_path).resolve(strict=True)),
        "downstream_registration_sha256": downstream_registration_sha256,
        "downstream_registration_path": str(Path(downstream_registration_path).resolve(strict=True)),
        "arm_id": arm, "seed": seed, "realization": draw,
        "output_id": plan["output_id"], "argv": argv,
        "cwd": str(worktree), "env": THREAD_ENV,
        "deps": expected_deps,
        "sources": [{"path": path, "sha256": digest}
                    for path, digest in sorted(sources.items())],
        "outputs": [str(output / "ATTACK_ATTEMPT_INDEX.json")],
        "cpu_reserve_s": plan["cpu_seconds"],
        "max_wall_s": plan["wall_seconds"],
        "max_rss_gib": 3.0, "projected_output_bytes": 200_000_000,
        **extra,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    for flag in ("unit-id", "amendment", "amendment-sha256", "seal",
                 "seal-sha256", "role-manifest", "queue-state",
                 "private-parent", "downstream-manifest",
                 "downstream-manifest-sha256", "downstream-registration",
                 "downstream-registration-sha256", "spec-output"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args(argv)
    spec = attack_spec(
        unit_id=args.unit_id, amendment_path=Path(args.amendment),
        amendment_sha256=args.amendment_sha256, seal_path=Path(args.seal),
        seal_sha256=args.seal_sha256, role_manifest=Path(args.role_manifest),
        queue_state=Path(args.queue_state),
        private_parent=Path(args.private_parent),
        downstream_manifest_path=Path(args.downstream_manifest),
        downstream_manifest_sha256=args.downstream_manifest_sha256,
        downstream_registration_path=Path(args.downstream_registration),
        downstream_registration_sha256=args.downstream_registration_sha256)
    target = Path(args.spec_output)
    _owner_private_dir(target.parent)
    if target.exists() or target.is_symlink():
        raise FileExistsError("attack queue spec is write-once")
    data = (json.dumps(spec, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"unit_id": spec["unit_id"],
                      "source_spec_sha256": hashlib.sha256(data).hexdigest(),
                      "source_count": len(spec["sources"])}, sort_keys=True))


if __name__ == "__main__":
    main()
