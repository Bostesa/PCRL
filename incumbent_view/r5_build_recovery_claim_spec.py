"""Receipt-only builder for independent fixed-family recovery and claim replay.

It hashes frozen outputs and public receipts. It never opens a prediction,
person-level view, task label, or SEX archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ARMS = ("P0", "O_J", "V_J", "V_P", "V_D", "V_L", "V_S12", "V_S21")
STOCHASTIC = frozenset(("O_J", "V_J", "V_P", "V_L", "V_S12", "V_S21"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"missing or symlinked source: {path}")
    return {"path": str(path.resolve(strict=True)), "sha256": sha(path)}


def write_new(path: Path, record: dict, mode: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "w") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return sha(path)


def build(repo: Path, amendment: Path, r3_recovery: Path, r3_claims: Path,
          r4_aggregate: Path) -> dict:
    repo = repo.resolve(strict=True)
    base = repo / "results/pcrl_incumbent_view_risk_v1"
    amendment = amendment.resolve(strict=True)
    r3_recovery = r3_recovery.resolve(strict=True)
    r3_claims = r3_claims.resolve(strict=True)
    r4_aggregate = r4_aggregate.resolve(strict=True)
    if (amendment.parent != base / "r0" or r3_recovery.parent != base / "r3" or
            r3_claims.parent != base / "r3" or r4_aggregate.parent != base / "r4"):
        raise PermissionError("inference source namespace differs")
    science = base / "r0/CAMPAIGN_LOCK.json"
    amend_doc = json.loads(amendment.read_text())
    if amend_doc.get("campaign_lock_sha256") != sha(science):
        raise PermissionError("inference amendment does not bind scientific lock")
    recovery = json.loads(r3_recovery.read_text())
    claims = json.loads(r3_claims.read_text())
    attack = json.loads(r4_aggregate.read_text())
    if (recovery.get("schema") != "pcrl-incumbent-r3-recovery-inference-v1" or
            recovery.get("K") != 948 or recovery.get("recovery_slots") != 756 or
            claims.get("schema") != "pcrl-incumbent-r3-fixed-claim-accounting-v1" or
            claims.get("K") != 948 or
            attack.get("schema") != "pcrl-incumbent-r4-fixed-attack-aggregate-v1" or
            attack.get("fixed_slots") != 40 or len(attack.get("units", [])) != 40):
        raise PermissionError("completed R3/R4 fixed-family receipts differ")
    eligible = set(recovery["eligible_arms"])
    if not eligible.issubset(ARMS):
        raise PermissionError("reported arm outside fixed family")
    expected = {(arm, seed, draw) for arm in eligible for seed in (0, 1)
                for draw in (range(3) if arm in STOCHASTIC else (0,))}
    replay = []
    deps = []
    for arm, seed, draw in sorted(expected, key=lambda x: (ARMS.index(x[0]), x[1], x[2])):
        unit = f"R5_FROZEN_AUC_{arm}_S{seed}_R{draw}"
        receipt = base / "r5" / f"{unit}.json"
        vector = base / "r5/.private" / f"{unit}_VECTORS.npz"
        doc = json.loads(receipt.read_text())
        if (doc.get("schema") != "IVR1-r5-frozen-auc-replay-v1" or
                doc.get("status") != "PASS" or doc.get("unit_id") != unit or
                doc.get("private_auc_vectors_sha256") != sha(vector)):
            raise PermissionError("eligible independent release replay is incomplete")
        replay.append({"arm_id": arm, "source_seed": seed, "realization": draw,
                       "receipt": pin(receipt), "vectors": pin(vector)})
        deps.append(unit)
    gates = {arm: base / "r3" / ("O_J_TWO_SEED_INNER_GATE_V2.json" if arm == "O_J"
                                  else f"{arm}_TWO_SEED_INNER_GATE.json") for arm in ARMS}
    gate_docs = {arm: json.loads(path.read_text()) for arm, path in gates.items()}
    if any(gate_docs[arm].get("arm_id") != arm or
           gate_docs[arm].get("original_inner", {}).get("required") != 24 for arm in ARMS):
        raise PermissionError("fixed original INNER gate set differs")
    p0_wire = base / "r1/.private/R1_P0_S0_ADMISSION/INNER_SELECTION/P0_RELEASE.npz"
    p0_receipt = base / "r1/.private/R1_P0_S0_ADMISSION/INNER_SELECTION/P0_ROLE_RECEIPT.json"
    role = json.loads(p0_receipt.read_text())
    if role.get("release_sha256") != sha(p0_wire) or role.get("groups") != 2234:
        raise PermissionError("P0 INNER group source differs")
    unit_id = "R5_RECOVERY_CLAIM_REPLAY_V1"
    out = base / "r5" / f"{unit_id}.json"
    config_path = base / "r5/.private" / f"{unit_id}_CONFIG.json"
    cfg = {"schema": "IVR1-r5-recovery-claim-replay-config-v1", "unit_id": unit_id,
           "admission_lock": pin(base / "r0/ADMISSION_LOCK.json"),
           "campaign_seal": pin(science), "verification_amendment": pin(amendment),
           "r3_recovery": pin(r3_recovery), "r3_utility": pin(base / "r3/R3_UTILITY_INTERVALS_V1.json"),
           "r3_claims": pin(r3_claims),
           "nomination": pin(base / "r3/R3_INTERNAL_READER_NOMINATION_V1.json"),
           "r4_aggregate": pin(r4_aggregate),
           "gates": {arm: pin(path) for arm, path in gates.items()},
           "r5_replays": replay, "p0_inner_wire": pin(p0_wire),
           "p0_inner_receipt": pin(p0_receipt), "output": str(out)}
    config_sha = write_new(config_path, cfg, 0o600)
    venv = repo.parents[1] / ".venv/bin/python"
    if not venv.is_file() or not os.access(venv, os.X_OK):
        raise FileNotFoundError("exact scientific virtualenv entrypoint missing")
    sources = [config_path, base / "r0/ADMISSION_LOCK.json", science, amendment,
               r3_recovery, base / "r3/R3_UTILITY_INTERVALS_V1.json", r3_claims,
               base / "r3/R3_INTERNAL_READER_NOMINATION_V1.json", r4_aggregate,
               p0_wire, p0_receipt, base / "r0/compute_queue.py",
               repo / "incumbent_view/r5_attack_math.py",
               repo / "incumbent_view/r5_selection_math.py",
               repo / "incumbent_view/r5_claims_protocol.py",
               repo / "incumbent_view/r5_recovery_claim_replay.py",
               repo / "incumbent_view/r5_build_recovery_claim_spec.py"]
    sources += list(gates.values())
    sources += [Path(item[k]["path"]) for item in replay for k in ("receipt", "vectors")]
    spec = {"unit_id": unit_id, "stage": "verification", "cwd": str(repo),
            "argv": [str(venv), "-m", "incumbent_view.r5_recovery_claim_replay",
                     "--config", str(config_path)],
            "max_wall_s": 180, "cpu_reserve_s": 180, "max_rss_gib": 2.5,
            "projected_output_bytes": 200000, "deps": deps,
            "outputs": [str(out)], "sources": [pin(path) for path in sources],
            "env": {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                         "VECLIB_MAXIMUM_THREADS")},
            "amendment_sha256": sha(amendment)}
    spec_path = base / "r5/.private" / f"{unit_id}_QUEUE_SPEC.json"
    spec_sha = write_new(spec_path, spec, 0o600)
    public = {"schema": "IVR1-r5-recovery-claim-spec-index-v1", "unit_id": unit_id,
              "private_spec_relative": str(spec_path.relative_to(repo)),
              "config_sha256": config_sha, "spec_sha256": spec_sha,
              "replays": len(replay), "recovery_slots": 756, "K": 948,
              "amendment_sha256": sha(amendment), "label_payload_read": False}
    write_new(base / "r5" / f"{unit_id}_SPEC_INDEX.json", public, 0o644)
    return public


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--amendment", required=True)
    parser.add_argument("--r3-recovery", required=True)
    parser.add_argument("--r3-claims", required=True)
    parser.add_argument("--r4-aggregate", required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(args.repo), Path(args.amendment),
                           Path(args.r3_recovery), Path(args.r3_claims),
                           Path(args.r4_aggregate)), sort_keys=True))


if __name__ == "__main__":
    main()
