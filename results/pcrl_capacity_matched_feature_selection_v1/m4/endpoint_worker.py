"""Sealed M4 aggregate endpoint and frozen-array worker for M3 inference.

Only preselected FIT rules are admitted. All release, reader, RAW and V2
units are checked against the live S3 queue before any INNER role is opened.
The private NPZs contain person arrays; the JSON endpoints contain aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from common_eval import fit_constant_probability
from endpoint_index import REFERENCES, RULES, VIEWS, score_selected_rule
from safe_data_guard import SafeRoleStore, sha_arr
from worker import (
    STUDY, _bank, _fit_receipt, _pair, _packages, _prior_manifest_path,
    _restore_package_record, _sha,
)
from release_bundle import restore_release_role
from attack_eval import AttackView
from prototype import AdultGroupedSchema
from s2_bridge import to_s2_attack_views
from results.pcrl_capacity_matched_feature_selection_v1.m0.admission import assert_running_unit
from results.pcrl_capacity_matched_feature_selection_v1.m2.bank import build_shared_bank
from results.pcrl_capacity_matched_feature_selection_v1.m1.capacity import capacity_geometry


COMMON_ROLE_KEYS = {
    "INNER_SELECTION:features", "INNER_SELECTION:income",
    "INNER_SELECTION:occupation", "INNER_SELECTION:sex",
    "OSF_DEFENSE_FIT:income", "OSF_DEFENSE_FIT:occupation",
}
INDEX_SCHEMA = "pcrl-focused-inference-input-index-v1"
AGGREGATE_SCHEMA = "pcrl-focused-aggregate-endpoints-v1"
PHASE_SEEDS = {"FIRST": (0,), "SCREEN": (0, 1),
               "FINALIST": (0, 1, 2)}


def _write_json_once(path: Path, value: dict) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    return _sha(path)


def _array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _same_people(first: dict, second: dict) -> None:
    for name in ("row_id", "unit"):
        left, right = np.asarray(first[name]), np.asarray(second[name])
        if (left.dtype != np.int64 or right.dtype != np.int64 or
                left.shape != right.shape or not np.array_equal(left, right)):
            raise PermissionError("endpoint role rows or groups differ")


def _sha_private_child(root: Path, filename: str) -> str:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise PermissionError("private custody child path escapes its receipt")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)):
        raise PermissionError("private custody child symlink escaped")
    return _sha(path)


def _pre_role_gate(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    phase = args.phase
    if phase not in PHASE_SEEDS:
        raise PermissionError("endpoint phase is outside frozen study")
    expected_unit = f"M4-ENDPOINTS-{phase}"
    expected_output = (STUDY / ".private/m4/endpoints" / phase /
                       "INFERENCE_INDEX.json")
    required_roles = COMMON_ROLE_KEYS | {f"INNER_SELECTION:references_s{seed}"
                                       for seed in PHASE_SEEDS[phase]}
    if (args.unit_id != expected_unit or
            Path(args.output_index).resolve() != expected_output.resolve()
            or Path(args.queue_root).resolve() != (STUDY / ".private/queue").resolve()):
        raise PermissionError("endpoint unit, queue or output is not frozen")
    spec = assert_running_unit(
        expected_unit, args.queue_root, STUDY / "CAMPAIGN_LOCK.json",
        "endpoint_index", expected_output,
    )
    if (set(spec.get("role_inputs", {})) != required_roles or
            spec.get("allow_child_processes") is not False):
        raise PermissionError("endpoint role allowlist or process contract differs")
    lock = json.loads((STUDY / "CAMPAIGN_LOCK.json").read_text())
    manifest_path = STUDY / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if (_sha(manifest_path) != lock.get("source_map_sha256") or
            spec.get("source_map_sha256") != _sha(manifest_path)):
        raise PermissionError("focused role manifest differs from seal")
    _prior_manifest_path(lock, manifest)
    return spec, lock, manifest


def _done_output(unit_id: str, path: Path, queue_root: Path,
                 queue_state: dict, registry: dict) -> tuple[dict, str]:
    if unit_id not in registry.get("unit_specs", {}):
        raise PermissionError("endpoint child is outside prelock potential registry")
    row = queue_state.get("units", {}).get(unit_id)
    if not isinstance(row, dict) or row.get("status") != "DONE":
        raise PermissionError("selected endpoint child is not DONE")
    spec_path = Path(row["spec_path"]).resolve(strict=True)
    if (not spec_path.is_relative_to(queue_root / "specs") or
            _sha(spec_path) != row.get("spec_sha256")):
        raise PermissionError("child queue spec changed")
    child_spec = json.loads(spec_path.read_text())
    sealed = dict(registry["unit_specs"][unit_id])
    sealed["seal_sha256"] = queue_state["gate"]["seal_sha256"]
    if child_spec != sealed:
        raise PermissionError("completed child differs from sealed potential spec")
    if (child_spec.get("unit_id") != unit_id or child_spec.get("outputs") != [str(path)] or
            row.get("output_sha256") != {str(path): _sha(path)}):
        raise PermissionError("child output is absent or changed")
    return child_spec, row["output_sha256"][str(path)]


def _plan_for_seed(seed: int, parent_spec: dict, lock: dict, manifest: dict,
                   queue_root: Path, public_bank: object) -> tuple[dict, dict, str]:
    fit = _fit_receipt(parent_spec, seed, lock, manifest, queue_root, public_bank)
    relative = Path(fit["control_plan_relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("FIT control plan path escapes private custody")
    path = (STUDY / ".private" / relative).resolve(strict=True)
    if not path.is_relative_to((STUDY / ".private").resolve()) or \
            _sha(path) != fit["control_plan_sha256"]:
        raise PermissionError("FIT control plan hash differs")
    plan = json.loads(path.read_text())
    if (plan.get("schema") != "pcrl-focused-m3-control-plan-v1" or
            set(plan.get("rules", {})) != set(RULES) or
            plan.get("bank_sha256") != public_bank.bank_sha256):
        raise PermissionError("ten FIT rule identities differ")
    declared = plan.get("selection_receipt_sha256")
    content = {key: value for key, value in plan.items()
               if key != "selection_receipt_sha256"}
    actual = hashlib.sha256(json.dumps(
        content, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    if declared != actual or plan["rules"]["J"]["status"] != "SELECTED":
        raise PermissionError("FIT control selection self-hash or J status differs")
    return fit, plan, _sha(path)


def _selected_children(seed: int, plan: dict, rule_ids: tuple[str, ...],
                       bank: object, queue_root: Path,
                       queue_state: dict, registry: dict,
                       lock_sha: str, manifest_sha: str) -> dict:
    """Prove every dynamic selected child before admitted INNER labels are read."""
    selected = {rule: plan["rules"][rule]["pair_id"] for rule in rule_ids
                if plan["rules"][rule]["status"] == "SELECTED"}
    if any(pair_id is None for pair_id in selected.values()):
        raise PermissionError("selected FIT rule lacks a pair")
    result = {"release": {}, "attack": {}, "v2": {}, "raw": {}}
    for pair_id in sorted(set(selected.values())):
        pair = _pair(bank, pair_id)
        release_id = f"M4-REL-S{seed}-{pair_id}"
        release_path = (STUDY / ".private/m4/releases" / f"seed_{seed}" /
                        pair_id / "RELEASE_RECEIPT.json")
        _, release_sha = _done_output(
            release_id, release_path, queue_root, queue_state, registry)
        release = json.loads(release_path.read_text())
        if (release.get("schema") != "pcrl-focused-complete-release-index-v1" or
                release.get("status") != "FROZEN" or
                release.get("seed") != seed or release.get("pair_id") != pair_id or
                release.get("lock_sha256") != lock_sha or
                release.get("manifest_sha256") != manifest_sha):
            raise PermissionError("selected release identity differs")
        result["release"][pair_id] = (release_path, release_sha)
        raw_by_view = {
            "income": bank.mask(pair.income_mask_id).mask_sha256,
            "occupation_group": bank.mask(pair.occupation_mask_id).mask_sha256,
            "coalition": pair.union_sha256,
        }
        # Local RAW units are keyed by the diagonal pair union, not mask SHA.
        for view in ("income", "occupation_group"):
            mask_id = pair.income_mask_id if view == "income" else pair.occupation_mask_id
            diagonal = _pair(bank, f"{mask_id}__{mask_id}")
            raw_by_view[view] = diagonal.union_sha256
        for view in VIEWS:
            raw_sha = raw_by_view[view]
            raw_id = f"M4-RAW-S{seed}-{raw_sha[:16].upper()}"
            raw_path = (STUDY / ".private/m4/raw_union" / f"seed_{seed}" /
                        raw_sha / "RAW_UNION_INDEX.json")
            _, raw_index_sha = _done_output(
                raw_id, raw_path, queue_root, queue_state, registry)
            result["raw"][raw_sha] = (raw_path, raw_index_sha)
            tag = {"income": "INCOME", "occupation_group": "OCCUPATION",
                   "coalition": "COALITION"}[view]
            attack_id = f"M4-ATT-S{seed}-{pair_id}-{tag}"
            attack_path = (STUDY / ".private/m4/attacks" / f"seed_{seed}" /
                           pair_id / view / "ATTACK_INDEX.json")
            _, attack_sha = _done_output(
                attack_id, attack_path, queue_root, queue_state, registry)
            attack = json.loads(attack_path.read_text())
            if (attack.get("schema") != "pcrl-focused-complete-reader-index-v1" or
                    attack.get("status") != "FROZEN" or
                    attack.get("seed") != seed or attack.get("pair_id") != pair_id or
                    attack.get("view_name") != view or
                    attack.get("lock_sha256") != lock_sha or
                    attack.get("manifest_sha256") != manifest_sha or
                    attack.get("release_index_sha256") != release_sha or
                    attack.get("raw_index_sha256") != raw_index_sha):
                raise PermissionError("selected fresh-reader index differs")
            result["attack"][(pair_id, view)] = (attack_path, attack_sha)
            v2_id = f"M4-V2-S{seed}-{pair_id}-{tag}"
            v2_path = (STUDY / ".private/m4/v2" / f"seed_{seed}" /
                       pair_id / view / "V2_INDEX.json")
            _, v2_sha = _done_output(v2_id, v2_path, queue_root, queue_state, registry)
            v2 = json.loads(v2_path.read_text())
            if (v2.get("schema") != "pcrl-focused-real-v2-view-index-v1" or
                    v2.get("status") != "PASS" or
                    v2.get("probe", {}).get("status") != "PASS" or
                    set(v2.get("probe", {}).get("gates", {})) != {
                        "null_exact", "shuffled_independent_inner_at_most_0.60",
                        "planted_at_least_0.99"} or
                    not all(v2["probe"]["gates"].values()) or
                    v2.get("seed") != seed or v2.get("pair_id") != pair_id or
                    v2.get("view_name") != view or
                    v2.get("lock_sha256") != lock_sha or
                    v2.get("manifest_sha256") != manifest_sha or
                    v2.get("release_index_sha256") != release_sha):
                raise PermissionError("real complete-view V2 probe failed or changed")
            result["v2"][(pair_id, view)] = (v2_path, v2_sha)
    return result


def _wire_probability(view: object, field: str, width: int) -> np.ndarray:
    span = [(start, end) for name, start, end in view.component_columns
            if name == field]
    if len(span) != 1 or span[0][1] - span[0][0] != width:
        raise PermissionError("saved complete wire omits exact task probability")
    values = np.ascontiguousarray(view.x[:, span[0][0]:span[0][1]], dtype=np.float64)
    if (values.shape != (len(view.row_id), width) or not np.isfinite(values).all() or
            not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-8)):
        raise PermissionError("actual task probability vector is invalid")
    return values


def _saved_reader_scores(path: Path, index_sha: str, view: object) -> dict[str, np.ndarray]:
    if _sha(path) != index_sha:
        raise PermissionError("saved reader index changed after S3 completion")
    record = json.loads(path.read_text())
    if (record.get("inner_view_content_sha256") != view.content_hashes() or
            record.get("inner_row_unit_sha256") != sha_arr(np.column_stack(
                (view.row_id, view.unit)))):
        raise PermissionError("saved reader score view differs from restored release")
    core_path = path.parent / "core_readers/READERS_RECEIPT.json"
    if _sha(core_path) != record.get("core_readers_receipt_sha256"):
        raise PermissionError("fitted complete-reader receipt changed")
    core = json.loads(core_path.read_text())
    for filename, digest in core.get("files", {}).items():
        if _sha_private_child(core_path.parent, filename) != digest:
            raise PermissionError("fitted complete-reader model or prediction changed")
    npz_path = path.parent / "ALL_READER_PREDICTIONS.npz"
    if _sha(npz_path) != record.get("all_reader_predictions_sha256"):
        raise PermissionError("saved fixed reader score file changed")
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != set(record["reader_order"]):
            raise PermissionError("saved fixed reader slate is incomplete")
        values = {name: np.asarray(archive[name], dtype=np.float64).copy()
                  for name in record["reader_order"]}
    for score in values.values():
        if (score.shape != (len(view.row_id),) or not np.isfinite(score).all()
                or np.any((score < 0) | (score > 1))):
            raise PermissionError("saved fixed reader score shape or range differs")
    return values


def _check_reader_inheritance(predictions: dict[str, dict[str, np.ndarray]]) -> None:
    if (set(predictions) != set(VIEWS) or
            len(predictions["income"]) != 6 or
            len(predictions["occupation_group"]) != 6 or
            len(predictions["coalition"]) != 18 or
            any(not np.array_equal(
                predictions["coalition"].get(f"IGNORE_R2__{reader_id}"), score)
                for reader_id, score in predictions["income"].items()) or
            any(not np.array_equal(
                predictions["coalition"].get(f"IGNORE_R1__{reader_id}"), score)
                for reader_id, score in predictions["occupation_group"].items())):
        raise PermissionError("coalition omits or changes either local reader")


def _reference_records(raw: dict, truth: dict, f_releases: dict,
                       features: dict) -> dict:
    _same_people(truth, raw)
    references = {"income": {}, "occupation_group": {}}
    schema = AdultGroupedSchema()
    for purpose, suffix in (("income", "1"), ("occupation_group", "2")):
        for ref in ("U0", "Ucal"):
            references[purpose][ref] = {
                "row_id": truth["row_id"], "unit": truth["unit"],
                "prob": np.ascontiguousarray(raw[f"{ref}_{suffix}"], dtype=np.float64),
            }
        if tuple(schema.column_indices(f_releases[purpose].selected_atoms)) != tuple(range(83)):
            raise PermissionError("F reference does not disclose the full permitted 83 inputs")
        view = f_releases[purpose].view(
            features["X"], tuple(features["feature_names"]))
        references[purpose]["F"] = {
            "row_id": truth["row_id"], "unit": truth["unit"],
            "prob": np.ascontiguousarray(view["probabilities"], dtype=np.float64),
        }
    return references


def _private_array_file(path: Path, arrays: dict[str, np.ndarray]) -> str:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    return _sha(path)


def run(args: argparse.Namespace) -> dict:
    spec, lock, manifest = _pre_role_gate(args)
    queue_root = Path(args.queue_root)
    queue_state = json.loads((queue_root / "QUEUE.json").read_text())
    registry_path = STUDY / ".private/UNIT_SPECS.json"
    if _sha(registry_path) != lock.get("private_unit_registry_sha256"):
        raise PermissionError("sealed potential unit registry changed")
    registry = json.loads(registry_path.read_text())
    bank = build_shared_bank(capacity_geometry(AdultGroupedSchema().feature_names)["masks"])
    pre = {}
    lock_sha, manifest_sha = _sha(STUDY / "CAMPAIGN_LOCK.json"), _sha(
        STUDY / "m1/DATA_ACCESS_MANIFEST.json")
    seeds = PHASE_SEEDS[args.phase]
    rules = ("J", "T") if args.phase == "FIRST" else RULES
    for seed in seeds:
        fit, plan, plan_sha = _plan_for_seed(
            seed, spec, lock, manifest, queue_root, bank)
        children = _selected_children(
            seed, plan, rules, bank, queue_root, queue_state, registry,
            lock_sha, manifest_sha)
        pre[seed] = (fit, plan, plan_sha, children)
    # The complete selected dependency chain is fixed before opening INNER.
    store = SafeRoleStore(_prior_manifest_path(lock, manifest))
    features = store.read("INNER_SELECTION", "features")
    income = store.read("INNER_SELECTION", "income", "screen")
    occupation = store.read("INNER_SELECTION", "occupation", "screen")
    sex = store.read("INNER_SELECTION", "sex", "screen")
    fit_income = store.read("OSF_DEFENSE_FIT", "income", "defense_fit")
    fit_occupation = store.read("OSF_DEFENSE_FIT", "occupation", "defense_fit")
    for row in (income, occupation, sex):
        _same_people(features, row)
    _same_people(fit_income, fit_occupation)
    truth = {"row_id": features["row_id"], "unit": features["unit"],
             "income": np.asarray(income["income"], dtype=np.int64),
             "occupation": np.asarray(occupation["occupation"], dtype=np.int64),
             "sex": np.asarray(sex["sex"], dtype=np.int64)}
    fit_labels = {"row_id": fit_income["row_id"], "unit": fit_income["unit"],
                  "income": np.asarray(fit_income["income"], dtype=np.int64),
                  "occupation": np.asarray(fit_occupation["occupation"], dtype=np.int64)}
    output_path = (STUDY / ".private/m4/endpoints" / args.phase /
                   "INFERENCE_INDEX.json")
    output = output_path.parent
    if output.exists():
        raise PermissionError("endpoint output is not a fresh private directory")
    output.mkdir(parents=True, mode=0o700)
    seed_index, aggregate_by_seed = {}, {}
    source_receipts = {
        "m1_manifest_sha256": manifest_sha,
        "campaign_lock_sha256": lock_sha,
        "fit_receipt_sha256_by_seed": {},
        "release_index_sha256_by_seed_pair": {},
        "attack_index_sha256_by_seed_pair_view": {},
        "v2_index_sha256_by_seed_pair_view": {},
        "raw_index_sha256_by_seed_union": {},
    }
    schema = AdultGroupedSchema()
    for seed in seeds:
        fit, plan, plan_sha, children = pre[seed]
        if sha_arr(np.column_stack((fit_labels["row_id"], fit_labels["unit"]))) != \
                fit["fit_row_unit_sha256"]:
            raise PermissionError("FIT-only constant source differs from M2 fitting people")
        source_receipts["fit_receipt_sha256_by_seed"][str(seed)] = _sha(
            STUDY / f".private/m2/fit/seed_{seed}/FIT_RECEIPT.json")
        for kind, target in (("release", "release_index_sha256_by_seed_pair"),
                             ("attack", "attack_index_sha256_by_seed_pair_view"),
                             ("v2", "v2_index_sha256_by_seed_pair_view"),
                             ("raw", "raw_index_sha256_by_seed_union")):
            for key, (_, digest) in children[kind].items():
                label = key if isinstance(key, str) else "|".join(key)
                source_receipts[target][f"{seed}|{label}"] = digest
        raw_ref = store.read("INNER_SELECTION", f"references_s{seed}")
        f_releases = {purpose: _restore_package_record(fit["full_f_packages"][purpose])
                      for purpose in ("income", "occupation_group")}
        references = _reference_records(raw_ref, truth, f_releases, features)
        constants = {purpose: fit_constant_probability(fit_labels[label], task=label)
                     for purpose, label in (("income", "income"),
                                            ("occupation_group", "occupation"))}
        const_receipt = {
            "schema": "pcrl-focused-fit-constant-source-v1", "status": "FIT_ONLY",
            "seed": seed, "fit_row_unit_sha256": sha_arr(np.column_stack(
                (fit_labels["row_id"], fit_labels["unit"]))),
            "fit_label_role_sha256": {
                "income": manifest["role_file_sha256"]["OSF_DEFENSE_FIT:income"],
                "occupation": manifest["role_file_sha256"]["OSF_DEFENSE_FIT:occupation"],
            },
            "constant_prob_by_task": {key: value.tolist()
                                      for key, value in constants.items()},
            "constant_prob_array_sha256": {key: _array_digest(value)
                                          for key, value in constants.items()},
        }
        const_path = output / f"FIT_CONSTANT_S{seed}.json"
        const_sha = _write_json_once(const_path, const_receipt)
        source_receipts.setdefault("fit_constant_receipt_sha256_by_seed", {})[str(seed)] = const_sha
        arrays: dict[str, np.ndarray] = {
            f"truth__{key}": np.asarray(value).copy() for key, value in truth.items()
        }
        for purpose in ("income", "occupation_group"):
            for ref in REFERENCES:
                arrays[f"reference__{purpose}__{ref}"] = references[purpose][ref]["prob"]
        rule_index, reader_ids, aggregate = {}, {}, {}
        for rule_id in rules:
            selection = plan["rules"][rule_id]
            if selection["status"] != "SELECTED":
                rule_index[rule_id] = {"status": selection["status"],
                                       "pair_id": selection["pair_id"]}
                reader_ids[rule_id] = {}
                aggregate[rule_id] = rule_index[rule_id]
                continue
            pair_id = selection["pair_id"]
            pair = _pair(bank, pair_id)
            packages, _ = _packages(fit, pair, lock, manifest)
            release_path, release_sha = children["release"][pair_id]
            views, _ = restore_release_role(
                index_path=release_path, expected_index_sha256=release_sha,
                role="INNER_SELECTION", source_record=features,
                public_releases=packages, pair=pair, bank=bank,
                schema=schema, bridge=to_s2_attack_views,
                attack_view_type=AttackView, private_root=STUDY / ".private",
                lock_sha256=lock_sha, manifest_sha256=manifest_sha,
                role_feature_sha256=manifest["role_file_sha256"]["INNER_SELECTION:features"],
            )
            candidate = {
                "income": {"row_id": truth["row_id"], "unit": truth["unit"],
                           "prob": _wire_probability(views["income"], "probabilities", 2)},
                "occupation_group": {"row_id": truth["row_id"],
                                     "unit": truth["unit"],
                                     "prob": _wire_probability(
                                         views["occupation_group"], "probabilities", 6)},
            }
            predictions, statuses = {}, {}
            for view in VIEWS:
                attack_path, attack_sha = children["attack"][(pair_id, view)]
                predictions[view] = _saved_reader_scores(
                    attack_path, attack_sha, views[view])
                v2_path, v2_sha = children["v2"][(pair_id, view)]
                v2 = json.loads(v2_path.read_text())
                if (_sha(v2_path) != v2_sha or
                        v2["inner_view_content_sha256"] != views[view].content_hashes()):
                    raise PermissionError("V2 view differs from actual saved release")
                custody = v2["probe_custody"]
                probe_dir = v2_path.parent / "probe_custody"
                if (_sha(probe_dir / "CUSTODY_RECEIPT.json") !=
                        custody["private_receipt_sha256"] or
                        any(_sha_private_child(probe_dir, filename) != digest
                            for filename, digest in custody["files"].items())):
                    raise PermissionError("real V2 planted/shuffled model custody changed")
                statuses[view] = v2["status"]
            _check_reader_inheritance(predictions)
            aggregate[rule_id] = score_selected_rule(
                seed=seed, rule_id=rule_id, selection=selection,
                truth=truth, fit_labels=fit_labels, candidate=candidate,
                references=references, reader_predictions=predictions,
                v2_status_by_view=statuses,
            )
            rule_index[rule_id] = {"status": "SELECTED", "pair_id": pair_id}
            reader_ids[rule_id] = {view: sorted(predictions[view]) for view in VIEWS}
            arrays[f"rule__{rule_id}__prob_income"] = candidate["income"]["prob"]
            arrays[f"rule__{rule_id}__prob_occupation"] = candidate["occupation_group"]["prob"]
            for view in VIEWS:
                for reader_id, score in predictions[view].items():
                    arrays[f"rule__{rule_id}__reader__{view}__{reader_id}"] = score
        npz_path = output / f"seed_{seed}.npz"
        npz_sha = _private_array_file(npz_path, arrays)
        seed_index[str(seed)] = {
            "npz_file": npz_path.name, "npz_sha256": npz_sha,
            "array_sha256": {key: _array_digest(value) for key, value in arrays.items()},
            "fit_plan_receipt_path": str(
                STUDY / ".private" / fit["control_plan_relative_path"]),
            "fit_plan_receipt_sha256": plan_sha,
            "rules": rule_index, "reader_ids_by_rule_view": reader_ids,
        }
        aggregate_by_seed[str(seed)] = aggregate
    aggregate_path = output / "AGGREGATE_ENDPOINTS.json"
    aggregate_sha = _write_json_once(aggregate_path, {
        "schema": AGGREGATE_SCHEMA, "status": "DEVELOPMENT_POINT_ONLY",
        "phase": args.phase, "seeds": aggregate_by_seed,
    })
    index = {
        "schema": INDEX_SCHEMA, "status": "FROZEN", "phase": args.phase,
        "lock_sha256": lock_sha, "m1_manifest_sha256": manifest_sha,
        "seeds": seed_index,
        "aggregate_endpoints_file": aggregate_path.name,
        "aggregate_endpoints_sha256": aggregate_sha,
        "source_receipts": source_receipts,
    }
    _write_json_once(output_path, index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=tuple(PHASE_SEEDS), required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--output-index", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"schema": result["schema"], "status": result["status"],
                      "unit_id": args.unit_id,
                      "output_sha256": _sha(Path(args.output_index))},
                     sort_keys=True))


if __name__ == "__main__":
    main()
