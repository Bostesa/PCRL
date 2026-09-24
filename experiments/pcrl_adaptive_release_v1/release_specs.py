"""Read-only, hash-pinned adapters from fitted units to common audit specs.

No fit or audit starts here. The returned Python specs are intended for one
coordinator-dispatched `evaluate.audit_panel` call on the verified Linux host.
Named exact aliases share a fitted audit while their original labels persist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import data
from . import fit_a, fit_b, fit_controls, refinement, task_baselines


def _file_sha(path: str | Path) -> str:
    return data.sha256_file(path)


def _private_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    if "private" not in root.parts:
        raise ValueError("fitted release source must stay private")
    return root


def _member(root: Path, relative: str) -> Path:
    part = Path(relative)
    target = (root / part).resolve()
    if part.is_absolute() or not target.is_relative_to(root) or not target.is_file():
        raise ValueError("unsafe or absent fitted release artifact")
    return target


def _load_q(path: Path, expected_sha: str | None = None) -> np.ndarray:
    if expected_sha is not None and _file_sha(path) != expected_sha:
        raise ValueError("channel artifact file hash differs")
    with np.load(path, allow_pickle=False) as archive:
        if list(archive.files) != ["Q"]:
            raise ValueError("channel archive must contain only Q")
        q = np.asarray(archive["Q"], dtype=np.float64).copy()
    if (q.ndim != 2 or q.shape[1] != 17 or len(q) < 1 or
            not np.isfinite(q).all() or np.any((q < 0) | (q > 1)) or
            np.max(np.abs(q.sum(axis=1)-1)) > 1e-10):
        raise ValueError("invalid 17-token saved channel")
    return q


def _array_identity(q: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(q, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _t0_collapse(q: np.ndarray, parent_of_leaf: np.ndarray) -> np.ndarray | None:
    parent = np.asarray(parent_of_leaf, dtype=np.int64)
    if parent.shape != (len(q),) or np.any(parent < 0) or np.any(parent >= 32):
        raise ValueError("invalid B child-parent identity")
    if set(parent.tolist()) != set(range(32)):
        raise ValueError("B child partition omits a historical T32 parent")
    collapsed = np.empty((32, 17), dtype=np.float64)
    for t in range(32):
        rows = q[parent == t]
        if not all(np.array_equal(row, rows[0]) for row in rows[1:]):
            return None
        collapsed[t] = rows[0]
    return collapsed


def _channel_identity(q: np.ndarray, partition: refinement.NestedPartition | None) -> str:
    if partition is None:
        if q.shape != (32, 17):
            raise ValueError("T0 audit spec requires 32 channel rows")
        return "T0:" + _array_identity(q)
    collapsed = _t0_collapse(q, partition.parent_of_leaf)
    if collapsed is not None:
        return "T0:" + _array_identity(collapsed)
    partition_record = json.dumps(partition.to_record(), sort_keys=True,
                                  separators=(",", ":"), allow_nan=False).encode()
    return "CHILD:" + hashlib.sha256(partition_record).hexdigest() + ":" + _array_identity(q)


class PinnedBRouter:
    """Private child router over only the five declared local/stored fields."""

    def __init__(self, partition: refinement.NestedPartition, frozen_nuisance,
                 pins: Mapping[str, tuple[str | Path, str]]):
        self._partition = partition
        self._nuisance = frozen_nuisance
        self._pins = {name: (Path(path), sha) for name, (path, sha) in pins.items()}

    def __call__(self, legal: Mapping[str, Any]) -> np.ndarray:
        if set(legal) != {"x", "ha", "token_codes", "teacher_p", "residual"}:
            raise ValueError("B router received forbidden or missing legal local fields")
        for name, (path, expected) in self._pins.items():
            if not path.is_file() or _file_sha(path) != expected:
                raise ValueError(f"B router {name} artifact hash differs")
        t, features = fit_b.stored_deployable_features(legal, self._nuisance)
        return self._partition.route(t, features)


def load_refined_b_spec(b_center_dir: str | Path,
                        parity_receipt_path: str | Path, *, anchor: int,
                        delta: float, a_center_dir: str | Path | None = None,
                        require_current_linux: bool = True) -> dict:
    """Verify selected B Q and its Linux runtime parity before audit routing."""
    from . import nuisance

    root = _private_root(b_center_dir)
    parity_path = _private_root(parity_receipt_path)
    if require_current_linux and (platform.system() != "Linux" or
                                  platform.machine() not in ("x86_64", "AMD64")):
        raise RuntimeError("refined B audit routing requires current Linux x86 host")
    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    if (complete.get("schema") != "pcrl-adaptive-B-center-v1" or
            complete.get("anchor") != anchor or complete.get("delta") != delta or
            complete.get("status") not in ("COMPLETE", "SUPPORT_LIMITED_ALIAS_A",
                                           "NO_ACCEPTED_SPLIT_ALIAS_A") or
            complete.get("artifact_sha256") != fit_a._inventory(root)):
        raise ValueError("B fitted center input or artifact hash inventory differs")
    selected_path = _member(root, complete["selected_channel_relative"])
    q = _load_q(selected_path)
    if complete["status"] == "COMPLETE":
        problem = fit_controls.load_final_problem("B", root,
                                                 anchor=anchor, delta=delta)
        if not np.array_equal(problem["selected_channel"], q):
            raise ValueError("B selected channel differs from final-bank replay")
    else:
        if a_center_dir is None:
            raise ValueError("B alias needs its completed frozen A center")
        selected_a = fit_b.load_a_selected(a_center_dir, anchor=anchor, delta=delta)
        if (complete.get("a_complete_receipt_sha256") !=
                selected_a["complete_receipt_sha256"] or
                not np.array_equal(q, selected_a["Q"])):
            raise ValueError("B alias does not reproduce frozen selected A")
    partition_path = root / "PARTITION.json"
    # Support-limited and rejected-split aliases predate the explicit field;
    # their immutable COMPLETE inventory still pins the partition bytes.
    partition_sha = complete.get(
        "partition_sha256", complete["artifact_sha256"].get("PARTITION.json"))
    if _file_sha(partition_path) != partition_sha:
        raise ValueError("B selected partition hash differs")
    partition = refinement.NestedPartition.from_record(
        json.loads(partition_path.read_text()))
    if q.shape != (len(partition.parents), 17):
        raise ValueError("selected B Q shape differs from partition")
    nuisance_model, nuisance_record = nuisance.load_frozen_nuisance(root / "nuisance")
    if nuisance_record["model_sha256"] != complete.get("nuisance_model_sha256"):
        raise ValueError("selected B nuisance model differs")
    parity = json.loads(parity_path.read_text())
    if (parity.get("schema") != "pcrl-refined-runtime-parity-v1" or
            parity.get("system") != "Linux" or
            parity.get("machine") not in ("x86_64", "AMD64") or
            parity.get("linux_x86_parity_verified") is not True or
            parity.get("t32_bitwise_equal") is not True or
            parity.get("child_bitwise_equal") is not True or
            parity.get("service_byte_equal") is not True or
            parity.get("center_complete_sha256") != _file_sha(complete_path) or
            parity.get("selected_channel_relative") != complete["selected_channel_relative"] or
            parity.get("channel_array_sha256") != fit_b._array_sha256(q) or
            parity.get("selected_channel_array_sha256") != fit_b._array_sha256(q) or
            parity.get("partition_record") != partition.to_record() or
            parity.get("encoder_sha256") != complete.get("encoder_sha256") or
            parity.get("nuisance_sha256") != nuisance_record["model_sha256"] or
            len(str(parity.get("fixture_input_sha256", ""))) != 64 or
            len(str(parity.get("fixture_h_a_sha256", ""))) != 64 or
            not np.isfinite(parity.get("task_posterior_max_absolute_difference", np.nan)) or
            parity["task_posterior_max_absolute_difference"] > 1e-7 or
            not np.isfinite(parity.get("residual_max_absolute_difference", np.nan)) or
            parity["residual_max_absolute_difference"] > 1e-7 or
            parity.get("checked_original_people", 0) < 1):
        raise ValueError("B Linux x86 runtime parity receipt differs from selected artifacts")
    pins = {"complete": (complete_path, _file_sha(complete_path)),
            "partition": (partition_path, _file_sha(partition_path)),
            "nuisance_model": (root / "nuisance/nuisance.joblib",
                               nuisance_record["model_sha256"]),
            "nuisance_receipt": (root / "nuisance/NUISANCE.json",
                                 _file_sha(root / "nuisance/NUISANCE.json")),
            "parity": (parity_path, _file_sha(parity_path))}
    router = PinnedBRouter(partition, nuisance_model, pins)
    return {"Q": q, "channel_artifact_path": str(selected_path),
            "channel_artifact_sha256": _file_sha(selected_path),
            "router": router, "router_artifact_path": str(complete_path),
            "router_sha256": _file_sha(complete_path),
            "router_entrypoint": "release_specs:PinnedBRouter(stored_Linux_T0,local_nuisance)",
            "_partition": partition,
            "_b_status": complete["status"],
            "_parity_artifact_sha256": _file_sha(parity_path),
            "_b_complete_sha256": _file_sha(complete_path)}


def _load_controls(directory: str | Path, *, branch: str, anchor: int,
                   delta: float, center_sha256: str,
                   frozen_cost_pair: Mapping[str, np.ndarray],
                   frozen_cuts: list[dict],
                   partition: refinement.NestedPartition | None,
                   router_spec: Mapping | None) -> list[tuple[str, dict, str]]:
    root = _private_root(directory)
    complete = json.loads((root / "COMPLETE.json").read_text())
    if (complete.get("schema") != "pcrl-adaptive-controls-v1" or
            complete.get("status") != "COMPLETE" or
            complete.get("artifact_sha256") != fit_controls._inventory(root)):
        raise ValueError("fixed-bank control unit artifact inventory differs")
    source = complete.get("source", {})
    valid_branches = ("A",) if branch == "A" else ("B", "B_ALIAS_A")
    if (source.get("branch") not in valid_branches or
            source.get("anchor") != anchor or source.get("delta") != delta or
            source.get("center_sha256") != center_sha256):
        raise ValueError("control source does not match frozen center")
    cost_u = np.asarray(frozen_cost_pair["U"], dtype=np.float64)
    cost_w = np.asarray(frozen_cost_pair["W"], dtype=np.float64)
    expected_cost_sha = {"U": fit_controls._array_sha(cost_u),
                         "W": fit_controls._array_sha(cost_w)}
    expected_bank_sha = fit_controls._bank_sha(.5*(cost_u+cost_w), frozen_cuts)
    if (complete.get("cost_pair_sha256") != expected_cost_sha or
            complete.get("fixed_bank_sha256") != expected_bank_sha):
        raise ValueError("control cost or final frozen bank differs from center")
    manifest = json.loads((root / "CONTROLS.json").read_text())
    rows = manifest["controls"]
    if set(rows) != set(complete["control_ids"]):
        raise ValueError("control ID roster differs from completion receipt")
    out = []
    q_by_id = {}
    for cid in complete["control_ids"]:
        record = rows[cid]
        path = _member(root, record["relative_path"])
        q = _load_q(path, record["Q_file_sha256"])
        if fit_controls._array_sha(q) != record["Q_array_sha256"]:
            raise ValueError("control channel array SHA differs")
        if record.get("alias_of") is not None and not np.array_equal(
                q, q_by_id[record["alias_of"]]):
            raise ValueError("declared control alias differs from frozen channel")
        q_by_id[cid] = q
        if partition is None and q.shape != (32, 17):
            raise ValueError("A control has non-T0 state shape")
        if partition is not None and q.shape != (len(partition.parents), 17):
            raise ValueError("B control state shape differs from B partition")
        spec = {"Q": q, "channel_artifact_path": str(path),
                "channel_artifact_sha256": _file_sha(path)}
        if router_spec is None:
            spec["router"] = "T0"
        else:
            spec.update({key: router_spec[key] for key in
                         ("router", "router_artifact_path", "router_sha256",
                          "router_entrypoint")})
        out.append((f"{branch}_control_{cid}", spec,
                    _channel_identity(q, partition)))
    return out


def build_release_specs(anchor: int, delta: float,
                        a_center_dir: str | Path, *,
                        b_center_dir: str | Path | None = None,
                        b_parity_receipt: str | Path | None = None,
                        a_controls_dir: str | Path | None = None,
                        b_controls_dir: str | Path | None = None,
                        task_only_dir: str | Path | None = None,
                        require_current_linux_for_b: bool = True) -> dict:
    """Return canonical runnable audit specs and every declared name's alias."""
    root_a = _private_root(a_center_dir)
    a = fit_b.load_a_selected(root_a, anchor=anchor, delta=delta)
    a_q = a["Q"]
    a_path = _member(root_a, f"round_r{a['selected_round']:02d}/channel/Q.npz")
    if not np.array_equal(_load_q(a_path), a_q):
        raise ValueError("A selected Q differs from saved channel artifact")
    canonical = {}
    aliases = {}
    identities = {}
    sources = {"A_complete_sha256": a["complete_receipt_sha256"]}

    def add(name: str, spec: dict, identity: str) -> None:
        if name in aliases:
            raise ValueError("duplicate declared release name")
        first = identities.setdefault(identity, name)
        aliases[name] = first
        if first == name:
            canonical[name] = spec

    add("A_selected", {"Q": a_q, "channel_artifact_path": str(a_path),
                       "channel_artifact_sha256": _file_sha(a_path),
                       "router": "T0"}, _channel_identity(a_q, None))
    b_spec = None
    b_partition = None
    b_problem = None
    if b_center_dir is not None:
        if b_parity_receipt is None:
            raise ValueError("B selected release requires verified Linux parity receipt")
        b_spec = load_refined_b_spec(
            b_center_dir, b_parity_receipt, anchor=anchor, delta=delta,
            a_center_dir=root_a, require_current_linux=require_current_linux_for_b)
        b_partition = b_spec.pop("_partition")
        b_status = b_spec.pop("_b_status")
        sources["B_complete_sha256"] = b_spec.pop("_b_complete_sha256")
        sources["B_parity_sha256"] = b_spec.pop("_parity_artifact_sha256")
        if b_status == "COMPLETE":
            b_problem = fit_controls.load_final_problem(
                "B", b_center_dir, anchor=anchor, delta=delta)
        else:
            b_problem = {"cost_pair": {key: a["cost"][key] for key in ("U", "W")},
                         "cuts": a["cuts"]}
        add("B_selected", b_spec, _channel_identity(b_spec["Q"], b_partition))
    if a_controls_dir is not None:
        control_root = _private_root(a_controls_dir)
        sources["A_controls_complete_sha256"] = _file_sha(control_root / "COMPLETE.json")
        for name, spec, identity in _load_controls(
                control_root, branch="A", anchor=anchor, delta=delta,
                center_sha256=a["complete_receipt_sha256"],
                frozen_cost_pair=a["cost"], frozen_cuts=a["cuts"],
                partition=None, router_spec=None):
            add(name, spec, identity)
    if b_controls_dir is not None:
        if b_spec is None or b_partition is None:
            raise ValueError("B controls require verified B selected router")
        control_root = _private_root(b_controls_dir)
        sources["B_controls_complete_sha256"] = _file_sha(control_root / "COMPLETE.json")
        b_root = _private_root(b_center_dir)
        for name, spec, identity in _load_controls(
                control_root, branch="B", anchor=anchor, delta=delta,
                center_sha256=_file_sha(b_root / "COMPLETE.json"),
                frozen_cost_pair=b_problem["cost_pair"],
                frozen_cuts=b_problem["cuts"],
                partition=b_partition, router_spec=b_spec):
            add(name, spec, identity)
    if task_only_dir is not None:
        from .archive_unit import inventory

        root = _private_root(task_only_dir)
        inventory(root)
        _, receipt = task_baselines.load_task_only(root)
        sources["TaskOnly_complete_sha256"] = _file_sha(root / "COMPLETE.json")
        receipt_sha = _file_sha(root / "TASK_ONLY.json")
        model_identity = receipt["model_sha256"]
        for mode, rate, name in [
                ("unmodified", 1., "TaskOnly"),
                *(("constant_replacement", rate, f"TaskOnly_constant_{int(100*rate):03d}")
                  for rate in (.50, .75, .90, 1.)),
                *(("randomized_response", rate, f"TaskOnly_rr_{int(100*rate):03d}")
                  for rate in (.50, .75, .90, 1.))]:
            identity = (f"TASK:{model_identity}:unmodified" if rate == 1.
                        else f"TASK:{model_identity}:{mode}:{rate:.2f}")
            add(name, {"task_only_model_dir": str(root),
                       "task_only_receipt_sha256": receipt_sha,
                       "mode": mode, "publish": rate,
                       "constant_token": 0}, identity)
    return {"releases": canonical, "aliases": aliases,
            "source_receipts": sources,
            "canonical_release_count": len(canonical),
            "declared_name_count": len(aliases),
            "deduplication": "exact structural token-law identity only; named aliases retained",
            "outer_labels_accessed": False}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--delta", type=float, choices=(0., .001, .003), required=True)
    parser.add_argument("--a-center-dir", required=True)
    parser.add_argument("--b-center-dir")
    parser.add_argument("--b-parity-receipt")
    parser.add_argument("--a-controls-dir")
    parser.add_argument("--b-controls-dir")
    parser.add_argument("--task-only-dir")
    parser.add_argument("--output-index", help="private aggregate mapping/receipt JSON")
    args = parser.parse_args(argv)
    bundle = build_release_specs(
        args.anchor, args.delta, args.a_center_dir,
        b_center_dir=args.b_center_dir,
        b_parity_receipt=args.b_parity_receipt,
        a_controls_dir=args.a_controls_dir,
        b_controls_dir=args.b_controls_dir,
        task_only_dir=args.task_only_dir)
    manifest = {key: value for key, value in bundle.items() if key != "releases"}
    manifest.update({"schema": "pcrl-release-audit-spec-index-v1",
                     "anchor": args.anchor, "delta": args.delta,
                     "canonical_ids": list(bundle["releases"]),
                     "source_module_sha256": _file_sha(__file__)})
    if args.output_index is not None:
        target = _private_root(args.output_index)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        encoded = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)+"\n"
        if target.exists() and target.read_text() != encoded:
            raise ValueError("existing audit-spec index differs")
        if not target.exists():
            temporary = target.with_name(target.name+f".tmp.{os.getpid()}")
            temporary.write_text(encoded)
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
    print(json.dumps(manifest, sort_keys=True))
    return bundle


if __name__ == "__main__":
    main()
