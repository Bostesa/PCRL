"""Receipt-only independent replay of the FIT reader nomination."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
from .r5_attack_math import ARMS
from .r5_selection_math import V_ARMS, fit_nominee


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(config_path):
    active = assert_active_from_env(stage="verification")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    source_index = {str(Path(s["path"]).resolve(strict=True)): s["sha256"]
                    for s in state["units"][active["unit_id"]]["spec"]["sources"]}

    def pinned(item):
        raw = Path(item["path"])
        if raw.is_symlink():
            raise PermissionError("nomination source symlink refused")
        path = raw.resolve(strict=True)
        if source_index.get(str(path)) != item["sha256"] or sha(path) != item["sha256"]:
            raise PermissionError("nomination source not byte-pinned by active queue")
        return path

    config = Path(config_path).resolve(strict=True)
    if source_index.get(str(config)) != sha(config):
        raise PermissionError("R5 nomination config unpinned")
    cfg = json.loads(config.read_text())
    if (cfg.get("schema") != "IVR1-r5-nomination-replay-config-v1" or
            cfg.get("unit_id") != active["unit_id"] or
            set(cfg["r2_receipts"]) != {f"{a}:s{s}" for a in V_ARMS for s in (0, 1)} or
            not set(cfg["gates"]).issubset(ARMS) or "P0" not in cfg["gates"]):
        raise PermissionError("fixed pre-attack nomination source family differs")
    if cfg["queue_seal"]["sha256"] != active["seal_sha256"]:
        raise PermissionError("nomination active queue seal differs")
    pinned(cfg["queue_seal"])
    science_sha = cfg["campaign_seal"]["sha256"]
    pinned(cfg["campaign_seal"])
    if science_sha not in {a["sha256"] for a in state["amendments"] if "protection" in a["stages"]}:
        raise PermissionError("nomination science lock is not registered")
    verification = [a for a in state["amendments"] if "verification" in a["stages"]]
    if (not verification or
            state["units"][active["unit_id"]]["spec"].get("amendment_sha256") != verification[-1]["sha256"] or
            json.loads(pinned(cfg["verification_amendment"]).read_text()).get("campaign_lock_sha256") != science_sha):
        raise PermissionError("nomination latest verification amendment differs")
    reported_path = pinned(cfg["r3_nomination"])
    reported = json.loads(reported_path.read_text())
    if (reported.get("schema") != "pcrl-incumbent-r3-internal-nomination-v1" or
            reported.get("final_attack_results_used") is not False):
        raise PermissionError("reported nomination is not FIT-only")
    gates = {}
    for arm, item in cfg["gates"].items():
        record = json.loads(pinned(item).read_text())
        if (record.get("arm_id") != arm or record.get("original_inner", {}).get("required") != 24 or
                reported["source_gate_sha256"].get(arm) != item["sha256"]):
            raise PermissionError("reported original INNER point gate differs")
        gates[arm] = record
    fit_records = {}
    for key, item in cfg["r2_receipts"].items():
        arm, seed_text = key.split(":s")
        seed = int(seed_text)
        record = json.loads(pinned(item).read_text())
        if (record.get("schema") != "IVR1-r2-protection-unit-v1" or
                record.get("arm_id") != arm or record.get("source_seed") != seed or
                reported["source_r2_units"][key]["receipt_sha256"] != item["sha256"]):
            raise PermissionError("R2 selected FIT source differs")
        if record["status"] in ("COMPLETE_SELECTED", "COMPLETE_IDENTITY_ALIAS",
                                "COMPLETE_WHOLE_P0_DEPENDENCY_FALLBACK"):
            nomination = record.get("nomination")
            if (not isinstance(nomination, dict) or
                    nomination.get("crossfit_manifest_sha256") != record.get("crossfit_manifest_sha256")):
                raise PermissionError("FIT OOF nomination record differs")
            fit_records[(arm, seed)] = nomination
        elif record["status"] != "INCOMPLETE":
            raise PermissionError("R2 FIT source status differs")
    independent = fit_nominee(gates, fit_records)
    if independent["nominee"] != reported["nominee"] or independent["fallback"] != reported["fallback"]:
        raise ArithmeticError("FIT-only nominee differs from independent CE rule")
    reported_slate = {x["arm_id"]: x for x in reported["slate"]}
    if set(reported_slate) != {x["arm_id"] for x in independent["slate"]}:
        raise ArithmeticError("FIT eligible reader slate differs")
    max_gap = 0.0
    for row in independent["slate"]:
        control = reported_slate[row["arm_id"]]
        max_gap = max(max_gap, abs(row["equal_seed_mean_gap"] - control["equal_seed_mean_reader_gain"]))
        if row["both_seed_exact_p0_alias"] != control["both_seed_exact_p0_alias"]:
            raise ArithmeticError("FIT exact P0 alias exclusion differs")
    if max_gap > 1e-12:
        raise ArithmeticError("FIT reader nomination gap differs")
    output = Path(cfg["output"]).resolve()
    owner = Path(__file__).resolve().parents[1] / "results/pcrl_incumbent_view_risk_v1/r5"
    if output.parent != owner or output.exists() or output.is_symlink():
        raise PermissionError("new R5 nomination receipt path required")
    result = {"schema": "IVR1-r5-independent-nomination-replay-v1", "status": "PASS",
              "unit_id": active["unit_id"], "nominee": independent["nominee"],
              "fallback": independent["fallback"], "eligible_reader_slate_count": len(independent["slate"]),
              "max_reader_gap_difference": max_gap, "r3_nomination_sha256": sha(reported_path),
              "final_attack_results_used": False, "labels_deserialized": False,
              "queue": active, "not_confirmation": True}
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "w") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    record = run(args.config)
    print(json.dumps({"unit_id": record["unit_id"], "status": record["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
