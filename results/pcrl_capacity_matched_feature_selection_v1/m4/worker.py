"""Sealed M4 release, RAW_ONLY, complete reader and V2 queue worker.

No role file is opened before M0's exact running-child admission check. The
worker reads only ATTACK_FIT_NEW and INNER_SELECTION role-only exports. It
does not access old assessment or HEAD_VALIDATION records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
EXPECTED_PRIOR = (REPO.parent / "pcrl-method-portfolio-v1" /
                  "results/pcrl_method_portfolio_v1").resolve()
SHARED = REPO / "results/pcrl_method_portfolio_v1/shared"
C_SOURCE = REPO / "results/pcrl_method_portfolio_v1/teams/c"
M2_SOURCE = STUDY / "m2"
for directory in (REPO, SHARED, C_SOURCE, M2_SOURCE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from attack_eval import AttackView
from bank import build_shared_bank
from common_eval import apply_global_temperature, load_calibrator
from prototype import AdultGroupedSchema
from release import release_selected_pair
from release_package import restore_release_package
from s2_bridge import to_s2_attack_views
from safe_data_guard import SafeRoleStore
from results.pcrl_capacity_matched_feature_selection_v1.m0.admission import assert_running_unit
from results.pcrl_capacity_matched_feature_selection_v1.m0.queue import sha256
from results.pcrl_capacity_matched_feature_selection_v1.m1.capacity import capacity_geometry
from attack_runner import fit_complete_view_readers
from raw_union import canonical_raw_union, fit_raw_union_readers
from release_bundle import restore_release_role, save_release_bundle
from v2_runner import run_real_view_controls


STAGES = {"release": "release", "raw_union": "raw_union",
          "attack": "attack_view", "v2": "v2_probe"}
VIEW_TAGS = {"income": "INCOME", "occupation_group": "OCCUPATION",
             "coalition": "COALITION"}
ROLE_INPUTS = {
    "release": {"ATTACK_FIT_NEW:features", "INNER_SELECTION:features"},
    "raw_union": {"ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
                  "INNER_SELECTION:features"},
    "attack": {"ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
               "INNER_SELECTION:features", "INNER_SELECTION:sex"},
    "v2": {"ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
           "INNER_SELECTION:features", "INNER_SELECTION:sex"},
}


def _sha(path: Path) -> str:
    return sha256(path)


def _expected_unit_id(args: argparse.Namespace) -> str:
    prefix = f"M4-"
    if args.mode == "release":
        return f"{prefix}REL-S{args.seed}-{args.pair_id}"
    if args.mode == "raw_union":
        return f"{prefix}RAW-S{args.seed}-{args.union_sha[:16].upper()}"
    tag = VIEW_TAGS[args.view]
    return f"{prefix}{'ATT' if args.mode == 'attack' else 'V2'}-S{args.seed}-{args.pair_id}-{tag}"


def _expected_output(args: argparse.Namespace) -> Path:
    private = STUDY / ".private/m4"
    if args.mode == "release":
        return private / "releases" / f"seed_{args.seed}" / args.pair_id / "RELEASE_RECEIPT.json"
    if args.mode == "raw_union":
        return private / "raw_union" / f"seed_{args.seed}" / args.union_sha / "RAW_UNION_INDEX.json"
    label = "attacks" if args.mode == "attack" else "v2"
    name = "ATTACK_INDEX.json" if args.mode == "attack" else "V2_INDEX.json"
    return private / label / f"seed_{args.seed}" / args.pair_id / args.view / name


def _prior_manifest_path(lock: dict, manifest: dict) -> Path:
    prior = (REPO / lock["prior_package_relative_to_worktree"]).resolve(strict=True)
    path = prior / "DATA_ACCESS_MANIFEST.json"
    if (prior != EXPECTED_PRIOR
            or manifest.get("prior_package_id") != "pcrl_method_portfolio_v1"
            or _sha(path) != manifest["portfolio_manifest_sha256"]
            or _sha(path) != lock["prior_data_manifest_sha256"]):
        raise PermissionError("prior role-only manifest differs from admitted sibling")
    return path


def _pre_role_gate(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """Validate exact running child, seal, argv identity and role allowlist."""
    if (args.mode != "raw_union" and
            re.fullmatch(r"M_DROP_[A-Z_]+__M_DROP_[A-Z_]+", args.pair_id or "") is None):
        raise PermissionError("pair ID is outside the frozen mask namespace")
    if (args.mode == "raw_union" and
            re.fullmatch(r"[0-9a-f]{64}", args.union_sha or "") is None):
        raise PermissionError("raw union SHA is malformed")
    if (args.seed not in (0, 1, 2) or args.unit_id != _expected_unit_id(args)
            or Path(args.output_index).resolve() != _expected_output(args).resolve()
            or Path(args.queue_root).resolve() != (STUDY / ".private/queue").resolve()):
        raise PermissionError("M4 frozen unit/output identity differs")
    spec = assert_running_unit(args.unit_id, args.queue_root,
                               STUDY / "CAMPAIGN_LOCK.json",
                               STAGES[args.mode], args.output_index)
    if set(spec.get("role_inputs", {})) != ROLE_INPUTS[args.mode]:
        raise PermissionError("M4 role input allowlist differs from stage")
    if spec.get("allow_child_processes") is not False:
        raise PermissionError("M4 unit must prohibit child processes")
    lock_path = STUDY / "CAMPAIGN_LOCK.json"
    lock = json.loads(lock_path.read_text())
    manifest_path = STUDY / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if (_sha(manifest_path) != lock.get("source_map_sha256")
            or spec.get("source_map_sha256") != _sha(manifest_path)):
        raise PermissionError("M1 source map differs from effective lock")
    _prior_manifest_path(lock, manifest)
    return spec, lock, manifest


def _dependency_output(spec: dict, unit_id: str, expected_path: Path,
                       queue_root: Path) -> str:
    if unit_id not in spec["deps"]:
        raise PermissionError("required parent queue dependency absent")
    status = json.loads((queue_root / "QUEUE.json").read_text())
    parent = status["units"].get(unit_id)
    if (not parent or parent.get("status") != "DONE"
            or set(parent.get("output_sha256", {})) != {str(expected_path)}):
        raise PermissionError("parent output is not completed and unique")
    expected = parent["output_sha256"][str(expected_path)]
    if _sha(expected_path) != expected:
        raise PermissionError("parent output SHA changed")
    return expected


def _store(lock: dict, manifest: dict) -> SafeRoleStore:
    return SafeRoleStore(_prior_manifest_path(lock, manifest))


def _bank(first_features: dict, manifest: dict):
    names = tuple(first_features["feature_names"])
    bank = build_shared_bank(capacity_geometry(names)["masks"])
    if bank.source_schema_sha256 != manifest["source_schema_sha256"]:
        raise PermissionError("frozen bank source schema differs")
    return bank


def _pair(bank: object, pair_id: str):
    matches = [pair for pair in bank.pairs if pair.pair_id == pair_id]
    if len(matches) != 1:
        raise PermissionError("pair is absent from the frozen 25-pair bank")
    return matches[0]


def _raw_geometry(bank: object, union_sha: str):
    matches = [pair for pair in bank.pairs if pair.union_sha256 == union_sha]
    if not matches or len({(pair.union_atoms, pair.union_columns) for pair in matches}) != 1:
        raise PermissionError("raw union is absent or inconsistent in the frozen bank")
    return matches[0]


def _view_geometry(bank: object, pair: object, view: str):
    if view == "coalition":
        return pair
    mask_id = (pair.income_mask_id if view == "income"
               else pair.occupation_mask_id)
    matches = [item for item in bank.pairs
               if item.income_mask_id == mask_id and
               item.occupation_mask_id == mask_id]
    if len(matches) != 1:
        raise PermissionError("local mask raw-union geometry is absent")
    return matches[0]


def _fit_receipt(spec: dict, seed: int, lock: dict, manifest: dict,
                 queue_root: Path, bank: object | None = None) -> dict:
    path = STUDY / f".private/m2/fit/seed_{seed}/FIT_RECEIPT.json"
    parent_sha = _dependency_output(spec, f"M2-FIT-S{seed}", path, queue_root)
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != "pcrl-focused-fit-receipt-v1"
            or receipt.get("seed") != seed
            or receipt.get("campaign_lock_sha256") != _sha(STUDY / "CAMPAIGN_LOCK.json")
            or receipt.get("data_manifest_sha256") != _sha(STUDY / "m1/DATA_ACCESS_MANIFEST.json")
            or receipt.get("prior_data_manifest_sha256") != lock["prior_data_manifest_sha256"]
            or receipt.get("status") != "FIT_COMPLETE"
            or (bank is not None and receipt.get("bank_sha256") != bank.bank_sha256)
            or _sha(path) != parent_sha):
        raise PermissionError("M2 FIT receipt differs from sealed study")
    return receipt


def _restore_package_record(record: dict):
    root = STUDY / ".private"
    relative = Path(record["relative_package_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("head package path escapes private custody")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or _sha(path) != record["package_sha256"]:
        raise PermissionError("head package SHA differs from FIT receipt")
    receipt_path = root / _sha(STUDY / "CAMPAIGN_LOCK.json") / \
        "head_pool" / record["cache_key"] / "HEAD_POOL_RECEIPT.json"
    if _sha(receipt_path) != record["receipt_sha256"]:
        raise PermissionError("head cache receipt SHA differs from FIT receipt")
    head_receipt = json.loads(receipt_path.read_text())
    return restore_release_package(
        path, expected_sha256=record["package_sha256"], private_root=root,
        expected_campaign_lock_sha256=_sha(STUDY / "CAMPAIGN_LOCK.json"),
        expected_data_manifest_sha256=_sha(STUDY / "m1/DATA_ACCESS_MANIFEST.json"),
        expected_source_role_hashes=head_receipt["cache_identity"]["role_file_sha256"],
        load_calibrator=load_calibrator,
        apply_global_temperature=apply_global_temperature,
    )


def _packages(fit: dict, pair: object, lock: dict, manifest: dict) -> tuple[dict, dict]:
    records = {
        "income": fit["head_packages"]["income"][pair.income_mask_id],
        "occupation_group": fit["head_packages"]["occupation_group"][pair.occupation_mask_id],
    }
    restored, package_shas = {}, {}
    for purpose, record in records.items():
        restored[purpose] = _restore_package_record(record)
        package_shas[purpose] = record["package_sha256"]
    return restored, package_shas


def _source_roles(store: SafeRoleStore, mode: str) -> dict:
    sources = {role: store.read(role, "features") for role in
               ("ATTACK_FIT_NEW", "INNER_SELECTION")}
    if mode in ("raw_union", "attack", "v2"):
        sources["sex_att"] = store.read("ATTACK_FIT_NEW", "sex", "attack")
    if mode in ("attack", "v2"):
        sources["sex_inner"] = store.read("INNER_SELECTION", "sex", "screen")
    return sources


def _release_dependency(spec: dict, seed: int, pair_id: str,
                        queue_root: Path) -> tuple[Path, str]:
    path = STUDY / ".private/m4/releases" / f"seed_{seed}" / pair_id / "RELEASE_RECEIPT.json"
    parent = f"M4-REL-S{seed}-{pair_id}"
    return path, _dependency_output(spec, parent, path, queue_root)


def _restore_views(args: argparse.Namespace, spec: dict, lock: dict,
                   manifest: dict, sources: dict, bank: object,
                   pair: object, packages: dict) -> tuple[dict, dict, str]:
    queue_root = Path(args.queue_root)
    release_path, release_sha = _release_dependency(
        spec, args.seed, args.pair_id, queue_root)
    role_hashes = manifest["role_file_sha256"]
    views, unions = {}, {}
    schema = AdultGroupedSchema()
    for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        views[role], unions[role] = restore_release_role(
            index_path=release_path, expected_index_sha256=release_sha,
            role=role, source_record=sources[role],
            public_releases=packages, pair=pair, bank=bank,
            schema=schema, bridge=to_s2_attack_views,
            attack_view_type=AttackView, private_root=STUDY / ".private",
            lock_sha256=_sha(STUDY / "CAMPAIGN_LOCK.json"),
            manifest_sha256=_sha(STUDY / "m1/DATA_ACCESS_MANIFEST.json"),
            role_feature_sha256=role_hashes[f"{role}:features"],
        )
    return views, unions, release_sha


def _coalition_local_dependencies(args: argparse.Namespace, spec: dict,
                                  views: dict) -> dict[str, tuple]:
    result = {}
    for view, tag in (("income", "INCOME"),
                      ("occupation_group", "OCCUPATION")):
        unit = f"M4-ATT-S{args.seed}-{args.pair_id}-{tag}"
        path = (STUDY / ".private/m4/attacks" / f"seed_{args.seed}" /
                args.pair_id / view / "ATTACK_INDEX.json")
        digest = _dependency_output(spec, unit, path, Path(args.queue_root))
        result[view] = (path, digest,
                        views["ATTACK_FIT_NEW"][view],
                        views["INNER_SELECTION"][view])
    return result


def run(args: argparse.Namespace) -> dict:
    spec, lock, manifest = _pre_role_gate(args)
    queue_root = Path(args.queue_root)
    # M2 publishes a no-go FIT receipt before any protected FIT read when no
    # useful J exists. All M4 stages, including RAW, stop before opening roles.
    _fit_receipt(spec, args.seed, lock, manifest, queue_root)
    role_sources = _source_roles(_store(lock, manifest), args.mode)
    bank = _bank(role_sources["ATTACK_FIT_NEW"], manifest)
    fit = _fit_receipt(spec, args.seed, lock, manifest, queue_root, bank)
    if args.mode == "raw_union":
        geometry = _raw_geometry(bank, args.union_sha)
        columns = geometry.union_columns
        names = tuple(role_sources["ATTACK_FIT_NEW"]["feature_names"][j]
                      for j in columns)
        att = canonical_raw_union(role_sources["ATTACK_FIT_NEW"],
                                  source_columns=columns, feature_names=names)
        inner = canonical_raw_union(role_sources["INNER_SELECTION"],
                                    source_columns=columns, feature_names=names)
        return fit_raw_union_readers(
            att, role_sources["sex_att"], inner, seed=args.seed,
            union_sha256=geometry.union_sha256,
            union_atoms=geometry.union_atoms,
            source_schema_sha256=bank.source_schema_sha256,
            lock_sha256=_sha(STUDY / "CAMPAIGN_LOCK.json"),
            manifest_sha256=_sha(STUDY / "m1/DATA_ACCESS_MANIFEST.json"),
            private_output=Path(args.output_index).parent,
        )
    pair = _pair(bank, args.pair_id)
    packages, package_shas = _packages(fit, pair, lock, manifest)
    if args.mode == "release":
        schema = AdultGroupedSchema()
        results = {}
        for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
            source = role_sources[role]
            results[role] = release_selected_pair(
                source["X"], tuple(source["feature_names"]),
                source["row_id"], source["unit"], role, pair, bank,
                packages["income"], packages["occupation_group"],
            )
        return save_release_bundle(
            role_sources=role_sources, release_results=results,
            public_releases=packages, pair=pair, bank=bank, schema=schema,
            bridge=to_s2_attack_views, attack_view_type=AttackView,
            private_root=STUDY / ".private",
            output_dir=Path(args.output_index).parent, seed=args.seed,
            lock_sha256=_sha(STUDY / "CAMPAIGN_LOCK.json"),
            manifest_sha256=_sha(STUDY / "m1/DATA_ACCESS_MANIFEST.json"),
            role_feature_sha256={role: manifest["role_file_sha256"][f"{role}:features"]
                                 for role in ("ATTACK_FIT_NEW", "INNER_SELECTION")},
            package_sha256_by_purpose=package_shas,
        )
    views, _, release_sha = _restore_views(
        args, spec, lock, manifest, role_sources, bank, pair, packages)
    att_view = views["ATTACK_FIT_NEW"][args.view]
    inner_view = views["INNER_SELECTION"][args.view]
    if args.mode == "v2":
        return run_real_view_controls(
            att_view=att_view, sex_att=role_sources["sex_att"],
            inner_view=inner_view, sex_inner=role_sources["sex_inner"],
            seed=args.seed, pair_id=args.pair_id, view_name=args.view,
            release_index_sha256=release_sha,
            lock_sha256=_sha(STUDY / "CAMPAIGN_LOCK.json"),
            manifest_sha256=_sha(STUDY / "m1/DATA_ACCESS_MANIFEST.json"),
            private_output=Path(args.output_index).parent,
        )
    view_geometry = _view_geometry(bank, pair, args.view)
    raw_path = STUDY / ".private/m4/raw_union" / f"seed_{args.seed}" / \
        view_geometry.union_sha256 / "RAW_UNION_INDEX.json"
    raw_parent = f"M4-RAW-S{args.seed}-{view_geometry.union_sha256[:16].upper()}"
    raw_sha = _dependency_output(spec, raw_parent, raw_path, queue_root)
    columns = (pair.union_columns if args.view == "coalition"
               else bank.mask(pair.income_mask_id if args.view == "income"
                              else pair.occupation_mask_id).column_indices)
    names = tuple(role_sources["ATTACK_FIT_NEW"]["feature_names"][j]
                  for j in columns)
    return fit_complete_view_readers(
        att_view=att_view, sex_att=role_sources["sex_att"],
        inner_view=inner_view, sex_inner=role_sources["sex_inner"],
        att_source=role_sources["ATTACK_FIT_NEW"],
        inner_source=role_sources["INNER_SELECTION"],
        source_columns=columns, feature_names=names,
        raw_index_path=raw_path, expected_raw_index_sha256=raw_sha,
        release_index_sha256=release_sha,
        lock_sha256=_sha(STUDY / "CAMPAIGN_LOCK.json"),
        manifest_sha256=_sha(STUDY / "m1/DATA_ACCESS_MANIFEST.json"),
        seed=args.seed, pair_id=args.pair_id, view_name=args.view,
        private_output=Path(args.output_index).parent,
        coalition_local_dependencies=(
            _coalition_local_dependencies(args, spec, views)
            if args.view == "coalition" else None),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=tuple(STAGES))
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pair-id")
    parser.add_argument("--view", choices=tuple(VIEW_TAGS))
    parser.add_argument("--union-sha")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--output-index", required=True)
    args = parser.parse_args()
    if (args.mode == "raw_union" and
            (args.pair_id is not None or args.view is not None or
             not isinstance(args.union_sha, str) or len(args.union_sha) != 64)):
        parser.error("raw_union needs exactly one full union SHA")
    if (args.mode != "raw_union" and
            (not args.pair_id or args.union_sha is not None or
             (args.mode in ("attack", "v2") and args.view is None) or
             (args.mode == "release" and args.view is not None))):
        parser.error("pair/view arguments differ from the stage contract")
    result = run(args)
    print(json.dumps({"schema": result["schema"], "status": result["status"],
                      "unit_id": args.unit_id,
                      "output_sha256": _sha(Path(args.output_index))},
                     sort_keys=True))


if __name__ == "__main__":
    main()
