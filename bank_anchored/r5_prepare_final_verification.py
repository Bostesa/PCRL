"""Build immutable queue specs for the final independent verification units.

This builder checks bytes and public receipts only. It never opens role labels
or private reader arrays; those enter only the sealed queue workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMP = ROOT / "results/pcrl_bank_anchored_release_v1"
R4, R5 = CAMP / "r4", CAMP / "r5"
ARMS = ("J", "P")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise PermissionError("R5 final verification source symlink refused")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise PermissionError("R5 final verification source is not a file")
    return {"path": str(path), "sha256": sha(path)}


def write_once(path: Path, value: dict, mode: int) -> str:
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.read_bytes() != data:
            raise FileExistsError("R5 final verification artifact already differs")
        return sha(path)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    return hashlib.sha256(data).hexdigest()


def source_set(paths: list[Path], inherited: list[dict[str, str]] = ()) -> list[dict[str, str]]:
    result = {item["path"]: item for item in inherited}
    for path in paths:
        item = pin(path)
        if item["path"] in result and item != result[item["path"]]:
            raise PermissionError("R5 final verification conflicting source hash")
        result[item["path"]] = item
    for item in result.values():
        if pin(Path(item["path"])) != item:
            raise PermissionError("R5 final verification inherited source changed")
    return [result[key] for key in sorted(result)]


def build(amendment: Path, python_entry: Path) -> dict:
    plan_path = R5 / "FINAL_VERIFICATION_SOURCE_PLAN.json"
    plan = json.loads(plan_path.read_text())
    if (plan.get("schema") != "BAR1-r5-final-verification-prefit-source-plan-v1" or
            plan.get("status") != "PREFIT_NO_NEW_ROLE_READ" or
            plan.get("unit_count") != 7):
        raise PermissionError("R5 prefit verification plan differs")
    amended = json.loads(amendment.read_text())
    state = json.loads((CAMP / "r0/.private/compute/state.json").read_text())
    latest = [row["sha256"] for row in state["amendments"]
              if "verification" in row["stages"]][-1]
    if (amended.get("schema") != "pcrl-bank-dated-amendment-v1" or
            amended.get("unit_ids") != plan["unit_ids"] or
            latest != sha(amendment) or
            str(python_entry) != "/Users/nathansamson/PCRL/.venv/bin/python" or
            not python_entry.is_file() or not os.access(python_entry, os.X_OK)):
        raise PermissionError("R5 final verification lacks exact registered amendment or venv")
    expected = plan["source_sha256"]
    bound = {
        "public_selection_worker": ROOT / "bank_anchored/r5_replay_selection.py",
        "refit_worker": ROOT / "bank_anchored/r5_refit_reader.py",
        "refit_subset_rule": R5 / "REFIT_SUBSET_RULE.json",
        "public_aggregate": R4 / "ATTACK_AGGREGATE_ALL_H2.json",
        "public_auc_csv": R4 / "ATTACK_AUC_ALL_H2.csv",
        "attack_replay_closeout": R5 / "ATTACK_REPLAY_CLOSEOUT.json",
        "pstar_original_inner_replay": R5 / "PSTAR_INNER_UTILITY_REPLAY.json",
    }
    for key, path in bound.items():
        if sha(path) != expected[key]:
            raise PermissionError(f"R5 prefit source hash changed: {key}")
    shared = [plan_path, amendment, Path(__file__),
              CAMP / "r0/compute_queue.py", python_entry.resolve(strict=True)]
    private = R5 / ".private"
    if private.is_symlink() or private.stat().st_mode & 0o077:
        raise PermissionError("R5 owner-only spec directory changed")
    specs = []
    attack_plan_path = R5 / "ATTACK_REPLAY_PLAN.json"
    attack_plan = json.loads(attack_plan_path.read_text())
    attack_ids = attack_plan["source_attack_unit_ids"]
    if len(attack_ids) != 38 or len(set(attack_ids)) != 38:
        raise PermissionError("R5 effective 38 attack unit family differs")
    gate_paths = [R5 / "PSTAR_INNER_UTILITY_REPLAY.json"]
    for arm in ("B0", "T", "D", "P", "L", "J", "S12", "S21", "W_Q", "P_STAR"):
        gate_paths.append(CAMP / "r3" /
                          (f"{arm}_HISTORICAL_ORIGINAL_INNER_GATE.json" if arm in
                           ("W_Q", "P_STAR") else f"{arm}_TWO_SEED_ATTACK_GATE_V2.json"))
    selected = {"P_STAR": json.loads(gate_paths[-1].read_text())}
    pstar = json.loads((R5 / "PSTAR_INNER_UTILITY_REPLAY.json").read_text())
    if (selected["P_STAR"].get("all_two_seed_task_reference_point_caps_pass") is not True or
            pstar.get("all_24_point_clauses_pass") is not True):
        raise PermissionError("independently admitted P* comparator differs")
    selection_paths = [*shared, attack_plan_path, *bound.values(), *gate_paths]
    for uid in attack_ids:
        selection_paths.extend((R4 / f"{uid}_RECEIPT.json",
                                R5 / f"R5_REPLAY_{uid}_RECEIPT.json"))
    selection_uid = "R5_REPLAY_PUBLIC_SELECTION"
    selection_out = R5 / f"{selection_uid}_RECEIPT.json"
    selection_spec = {
        "unit_id": selection_uid, "stage": "verification",
        "amendment_sha256": latest, "cwd": str(ROOT), "env": {},
        "argv": [str(python_entry), "-m", "bank_anchored.r5_replay_selection",
                 "--plan", str(attack_plan_path), "--plan-sha256", sha(attack_plan_path),
                 "--aggregate", str(bound["public_aggregate"]),
                 "--aggregate-sha256", expected["public_aggregate"],
                 "--csv", str(bound["public_auc_csv"]),
                 "--csv-sha256", expected["public_auc_csv"],
                 "--output", str(selection_out)],
        "deps": [f"R5_REPLAY_{uid}" for uid in attack_ids],
        "max_wall_s": 120, "cpu_reserve_s": 120, "max_rss_gib": 1.0,
        "projected_output_bytes": 200_000, "outputs": [str(selection_out)],
        "sources": source_set(selection_paths),
    }
    specs.append(selection_spec)
    for arm in ARMS:
        for seed in (0, 1):
            source_uid = f"R4_ATTACK_{arm}_S{seed}_R0_H1"
            uid = f"R5_REFIT_{arm}_S{seed}_R0"
            r4spec = R4 / ".private/units" / f"{source_uid}.json"
            r4receipt = R4 / f"{source_uid}_RECEIPT.json"
            source = json.loads(r4spec.read_text())
            if (source["unit_id"] != source_uid or source["arm_id"] != arm or
                    source["source_seed"] != seed or source["realization"] != 0):
                raise PermissionError("fixed J/P draw-zero source differs")
            manifest_path = Path(source["role_manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            if sha(manifest_path) != source["role_manifest_sha256"]:
                raise PermissionError("fixed refit role manifest changed")
            labels = []
            for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
                label_path = manifest_path.parent.parent / manifest["role_file_relative_path"][f"{role}:sex"]
                if (label_path.name != "sex.npz" or role not in label_path.parts or
                        sha(label_path) != manifest["role_file_sha256"][f"{role}:sex"]):
                    raise PermissionError("fixed refit label role differs")
                labels.append(label_path)
            reader_dir = Path(source["private_output_path"]) / "pair/readers"
            reader = reader_dir / "READERS_RECEIPT.json"
            custody = json.loads(reader.read_text())
            private_inputs = [reader, *(reader_dir / name for name in
                               ("view_att.npz", "view_inner.npz", "predictions_inner.npz"))]
            for path in private_inputs[1:]:
                if sha(path) != custody["files"][path.name]:
                    raise PermissionError("fixed refit frozen reader bytes changed")
            replay_spec = R5 / ".private" / f"R5_REPLAY_{source_uid}_QUEUE_SPEC.json"
            replay_sources = json.loads(replay_spec.read_text())["sources"]
            out = R5 / f"{uid}_RECEIPT.json"
            private_out = private / uid
            if out.exists() or private_out.exists():
                raise FileExistsError("fixed refit output already exists")
            paths = [*shared, bound["refit_worker"], bound["refit_subset_rule"],
                     ROOT / "bank_anchored/r5_replay_attack.py",
                     ROOT / "bank_anchored/r5_independent.py", replay_spec,
                     R5 / f"R5_REPLAY_{source_uid}_RECEIPT.json",
                     r4spec, r4receipt, manifest_path, *labels, *private_inputs]
            spec = {
                "unit_id": uid, "stage": "verification",
                "amendment_sha256": latest, "cwd": str(ROOT), "env": {},
                "argv": [str(python_entry), "-m", "bank_anchored.r5_refit_reader",
                         "--r4-spec", str(r4spec), "--r4-spec-sha256", sha(r4spec),
                         "--r4-receipt", str(r4receipt), "--r4-receipt-sha256", sha(r4receipt),
                         "--rule", str(bound["refit_subset_rule"]),
                         "--rule-sha256", expected["refit_subset_rule"],
                         "--private-output-dir", str(private_out), "--output", str(out)],
                "deps": [f"R5_REPLAY_{source_uid}"],
                "max_wall_s": 300, "cpu_reserve_s": 300, "max_rss_gib": 2.0,
                "projected_output_bytes": 10_000_000, "outputs": [str(out)],
                "sources": source_set(paths, replay_sources),
            }
            specs.append(spec)
    record = []
    for spec in specs:
        path = private / f"{spec['unit_id']}_QUEUE_SPEC.json"
        digest = write_once(path, spec, 0o600)
        record.append({"unit_id": spec["unit_id"], "path": str(path),
                       "sha256": digest, "source_count": len(spec["sources"]),
                       "max_wall_s": spec["max_wall_s"],
                       "cpu_reserve_s": spec["cpu_reserve_s"],
                       "max_rss_gib": spec["max_rss_gib"]})
    index = {"schema": "BAR1-r5-final-verification-queue-index-v1",
             "status": "SPECS_READY_NO_NEW_ROLE_READ",
             "amendment_sha256": latest, "source_plan_sha256": sha(plan_path),
             "builder_sha256": sha(Path(__file__)), "units": record}
    write_once(R5 / "FINAL_VERIFICATION_QUEUE_INDEX.json", index, 0o644)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.amendment, args.python), sort_keys=True))


if __name__ == "__main__":
    main()
