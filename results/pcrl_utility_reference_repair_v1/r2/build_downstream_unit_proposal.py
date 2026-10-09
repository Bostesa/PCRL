"""Map frozen selected pairs to bounded later units without reading role arrays."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r2.attack_fit import (
    expected_outputs as attack_outputs,
)
from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank
from results.pcrl_utility_reference_repair_v1.r2.raw_union_worker import (
    expected_outputs as raw_outputs, required_union_shas,
)


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
PRIVATE = STUDY / ".private"
BANK_PATH = REPO / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
GATE_PATH = STUDY / "FULL_INPUT_GATE.json"
MANIFEST_PATH = STUDY / "m1/DATA_ACCESS_MANIFEST.json"
OUTPUT = STUDY / "r2/DOWNSTREAM_UNIT_PROPOSAL.json"
VIEWS = ("income", "occupation_group", "coalition")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve(strict=path.exists())
    if not resolved.is_relative_to(REPO):
        raise PermissionError("unit path escapes research worktree")
    return str(resolved.relative_to(REPO))


def existing(inputs: dict, path: Path, expected: str | None = None) -> None:
    digest = sha(path)
    if expected is not None and digest != expected:
        raise PermissionError("frozen unit source bytes differ")
    inputs[relative(path)] = {"sha256": digest, "source": "FROZEN_EXISTING"}


def future(inputs: dict, path: Path, producer: str) -> None:
    inputs[relative(path)] = {"sha256": "FROM_PRODUCER_UNIT",
                              "producer_unit": producer}


def private_record(inputs: dict, record: dict) -> Path:
    value = Path(record["relative_path"])
    if value.is_absolute() or ".." in value.parts:
        raise PermissionError("private component path escapes")
    path = PRIVATE / value
    existing(inputs, path, record["sha256"])
    return path


def common_inputs(gate: dict) -> dict:
    out = {}
    for path in (GATE_PATH, MANIFEST_PATH, BANK_PATH):
        existing(out, path)
    for seed in ("0", "1"):
        gate_dir = PRIVATE / "r1/gate" / f"seed_{seed}"
        existing(out, gate_dir / "GATE_RECEIPT.json",
                 gate["r1_gate_receipt_sha256_by_seed"][seed])
        existing(out, gate_dir / "GATE_PREDICTIONS.npz",
                 gate["r1_gate_prediction_sha256_by_seed"][seed])
        private_record(out, gate["f_receipts"][seed])
        for task in ("income", "occupation_group"):
            path = private_record(out, gate["f_packages"][seed][task])
            package = json.loads(path.read_text())
            for component in ("model", "calibrator"):
                private_record(out, package[component])
    return out


def role_input(inputs: dict, manifest: dict, key: str) -> None:
    value = Path(manifest["role_file_relative_path"][key])
    digest = manifest["role_file_sha256"][key]
    if (value.is_absolute() or ".." in value.parts
            or value.parts[:2] != (".private", "role_export_v1")
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise PermissionError("role identity is incomplete or escapes custody")
    path = STUDY / value
    inputs[relative(path)] = {"sha256": digest,
                              "source": "FROZEN_MANIFEST_ROLE_METADATA"}


def selected_inputs(inputs: dict, seed: int, selection: dict) -> None:
    folder = PRIVATE / "pair_select" / f"seed{seed}"
    existing(inputs, folder / "PAIR_SELECT_RECEIPT.json")
    existing(inputs, folder / "CONTROL_PLAN.json",
             selection["files_sha256"]["CONTROL_PLAN.json"])


def masked_task_inputs(inputs: dict, seed: int, pair: object) -> None:
    table = PRIVATE / "task_table" / f"seed{seed}" / "TASK_TABLE_RECEIPT.json"
    existing(inputs, table)
    for task, mask_id in (("income", pair.income_mask_id),
                          ("occupation_group", pair.occupation_mask_id)):
        existing(inputs, PRIVATE / "mask_gate" / mask_id / f"seed{seed}" /
                 "MASK_GATE_RECEIPT.json")
        fit_path = PRIVATE / "masks" / mask_id / f"seed{seed}" / "MASK_FIT_RECEIPT.json"
        existing(inputs, fit_path)
        fit = json.loads(fit_path.read_text())
        package_path = private_record(inputs, fit["task_packages"][task])
        package = json.loads(package_path.read_text())
        for component in ("model", "calibrator"):
            private_record(inputs, package[component])


def release_files(seed: int, pair_id: str) -> list[Path]:
    folder = PRIVATE / "releases" / f"seed{seed}" / pair_id
    return [folder / name for name in (
        "RELEASE_RECEIPT.json", "ATTACK_FIT_NEW.npz", "INNER_SELECTION.npz")]


def raw_files(seed: int, union_sha: str) -> list[Path]:
    folder = PRIVATE / "raw_union" / f"seed{seed}" / union_sha
    return [Path(name) for name in sorted(raw_outputs(folder / "RAW_UNION_INDEX.json"))]


def attack_files(seed: int, pair_id: str, view: str) -> list[Path]:
    folder = PRIVATE / "attacks" / f"seed{seed}" / pair_id / view
    return [Path(name) for name in sorted(attack_outputs(folder / "ATTACK_FIT_INDEX.json"))]


def unit(uid: str, stage: str, role_keys: list[str], inputs: dict,
         outputs: list[Path], deps: set[str], **identity) -> dict:
    return {"unit_id": uid, "stage": stage, "role_inputs": role_keys,
            "inputs": dict(sorted(inputs.items())),
            "outputs": sorted(relative(path) for path in outputs),
            "deps": sorted(deps), **identity}


def build() -> dict:
    gate = json.loads(GATE_PATH.read_text())
    if gate.get("status") != "PASS_FULL_INPUT_POINT_GATE" or gate.get(
            "selected_recipe") != "A":
        raise PermissionError("F admission source differs")
    bank = load_frozen_bank(BANK_PATH,
                            expected_sha256="a52750191a1e16051367cade59537bc65ed797d69efcb375e0b1cc2e9364722c")
    manifest = json.loads(MANIFEST_PATH.read_text())
    common = common_inputs(gate)
    bank_pairs = {pair.pair_id: pair for pair in bank.pairs}
    units = {}
    selected_by_seed = {}
    unions_by_seed = {}
    selected_source = {}
    for seed in (0, 1):
        folder = PRIVATE / "pair_select" / f"seed{seed}"
        selection_path = folder / "PAIR_SELECT_RECEIPT.json"
        selection = json.loads(selection_path.read_text())
        plan_path = folder / "CONTROL_PLAN.json"
        existing({}, plan_path, selection["files_sha256"]["CONTROL_PLAN.json"])
        plan = json.loads(plan_path.read_text())
        selected = sorted({row["pair_id"] for row in plan["rules"].values()
                           if row["status"] == "SELECTED"})
        unions = sorted(required_union_shas(bank, plan))
        if not selected or not set(selected) <= set(bank_pairs) or len(unions) > 6:
            raise PermissionError("selected pair or union coverage differs")
        selected_by_seed[str(seed)] = selected
        unions_by_seed[str(seed)] = unions
        selected_source[str(seed)] = {
            "pair_receipt_sha256": sha(selection_path),
            "control_plan_sha256": sha(plan_path),
            "selected_rule_count": len([r for r in plan["rules"].values()
                                         if r["status"] == "SELECTED"]),
            "distinct_pair_count": len(selected),
            "canonical_union_count": len(unions),
        }
        release_ids, raw_ids, attack_ids = {}, {}, {}
        for pair_id in selected:
            uid = f"R2_RELEASE_S{seed}_{pair_id}"
            pair = bank_pairs[pair_id]
            inputs = dict(common)
            selected_inputs(inputs, seed, selection)
            masked_task_inputs(inputs, seed, pair)
            for key in ("ATTACK_FIT_NEW:features", "INNER_SELECTION:features"):
                role_input(inputs, manifest, key)
            units[uid] = unit(uid, "r2_release", [
                "ATTACK_FIT_NEW:features", "INNER_SELECTION:features"],
                inputs, release_files(seed, pair_id), set(),
                seed=seed, pair_id=pair_id)
            release_ids[pair_id] = uid
        for union_sha in unions:
            uid = f"R2_RAW_UNION_S{seed}_{union_sha}"
            inputs = dict(common)
            selected_inputs(inputs, seed, selection)
            for key in ("ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
                        "INNER_SELECTION:features"):
                role_input(inputs, manifest, key)
            units[uid] = unit(uid, "r2_raw_union", [
                "ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
                "INNER_SELECTION:features"], inputs,
                raw_files(seed, union_sha), set(),
                seed=seed, union_sha256=union_sha)
            raw_ids[union_sha] = uid
        for pair_id in selected:
            pair = bank_pairs[pair_id]
            diagonal = {item.income_mask_id: item for item in bank.pairs
                        if item.income_mask_id == item.occupation_mask_id}
            for view in VIEWS:
                uid = f"R2_ATT_S{seed}_{pair_id}_{view}"
                union = (pair if view == "coalition" else diagonal[
                    pair.income_mask_id if view == "income" else
                    pair.occupation_mask_id])
                inputs = dict(units[release_ids[pair_id]]["inputs"])
                role_input(inputs, manifest, "ATTACK_FIT_NEW:sex")
                for path in release_files(seed, pair_id):
                    future(inputs, path, release_ids[pair_id])
                for path in raw_files(seed, union.union_sha256):
                    future(inputs, path, raw_ids[union.union_sha256])
                deps = {release_ids[pair_id], raw_ids[union.union_sha256]}
                if view == "coalition":
                    for local in VIEWS[:2]:
                        local_uid = attack_ids[(pair_id, local)]
                        folder = PRIVATE / "attacks" / f"seed{seed}" / pair_id / local
                        for name in ("ATTACK_FIT_INDEX.json",
                                     "ALL_READER_PREDICTIONS.npz",
                                     "ALL_READER_TUNE_PREDICTIONS.npz"):
                            future(inputs, folder / name, local_uid)
                        deps.add(local_uid)
                units[uid] = unit(uid, "r2_attack_fit", [
                    "ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
                    "INNER_SELECTION:features"], inputs,
                    attack_files(seed, pair_id, view), deps,
                    seed=seed, pair_id=pair_id, view=view,
                    canonical_union_sha256=union.union_sha256)
                attack_ids[(pair_id, view)] = uid
        score_uid = f"R2_PRIVACY_SCORE_S{seed}"
        inputs = dict(common)
        selected_inputs(inputs, seed, selection)
        role_input(inputs, manifest, "INNER_SELECTION:sex")
        deps = set()
        for pair_id in selected:
            for path in release_files(seed, pair_id):
                future(inputs, path, release_ids[pair_id])
            deps.add(release_ids[pair_id])
            for view in VIEWS:
                attack_uid = attack_ids[(pair_id, view)]
                for path in attack_files(seed, pair_id, view):
                    future(inputs, path, attack_uid)
                deps.add(attack_uid)
        score_dir = PRIVATE / "privacy_score" / f"seed{seed}"
        units[score_uid] = unit(score_uid, "r2_privacy_score",
            ["INNER_SELECTION:sex"], inputs,
            [score_dir / "PRIVACY_SCORES.json",
             score_dir / "PRIVACY_SCORE_RECEIPT.json"], deps,
            seed=seed, distinct_pairs=len(selected),
            exact_input_count=24 + 51 * len(selected))
        if len(inputs) != 24 + 51 * len(selected):
            raise PermissionError("privacy scorer exact input count differs")
    return {
        "schema": "pcrl-utility-r2-downstream-unit-proposal-v1",
        "status": "PROSPECTIVE_AFTER_FIXED_PAIR_SELECTION_BEFORE_ATT_OR_INNER_SEX",
        "development_only": True,
        "full_input_gate_sha256": sha(GATE_PATH),
        "mask_bank_sha256": sha(BANK_PATH),
        "source_manifest_sha256": sha(MANIFEST_PATH),
        "selected_source_by_seed": selected_source,
        "selected_pairs_by_seed": selected_by_seed,
        "complete_view_unions_by_seed": unions_by_seed,
        "stage_order": [
            ["r2_release", "r2_raw_union"],
            ["r2_attack_fit:income", "r2_attack_fit:occupation_group"],
            ["r2_attack_fit:coalition"], ["r2_privacy_score"],
            ["R3 independent numerical replay", "R0 final aggregate"]],
        "unit_count_by_stage": {
            stage: sum(unit["stage"] == stage for unit in units.values())
            for stage in ("r2_release", "r2_raw_union", "r2_attack_fit",
                          "r2_privacy_score")},
        "unit_map": dict(sorted(units.items())),
        "hash_convention": "Existing paths have SHA256; later outputs name their producer and must be hashed into the next sealed stage lock after completion.",
        "path_convention": "All paths are worktree-relative; private role files and predictions remain uncommitted.",
        "new_att_fit_count": 6 * sum(u["stage"] == "r2_attack_fit" for u in units.values())
        + 2 * sum(u["stage"] == "r2_raw_union" for u in units.values()),
        "separate_fixed_fit_sex_search_fits_already_run": 44,
    }


def main() -> None:
    value = build()
    with OUTPUT.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"sha256": sha(OUTPUT),
                      "unit_count_by_stage": value["unit_count_by_stage"],
                      "new_att_fit_count": value["new_att_fit_count"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
