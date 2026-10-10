"""Receipt-only builder for the fixed 40-release independent AUC replay.

The builder hashes frozen reader artifacts. It does not deserialize labels,
predictions, views, or policy laws, and it does not select arms or draws.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"frozen source missing or symlinked: {path}")
    return {"path": str(path.resolve(strict=True)), "sha256": sha(path)}


def write_new(path: Path, record: dict, mode: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode), "w") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return sha(path)


def build(repo: Path, amendment: Path, arm: str, seed: int, draw: int) -> dict:
    repo = repo.resolve(strict=True)
    amendment = amendment.resolve(strict=True)
    base = repo / "results/pcrl_incumbent_view_risk_v1"
    fixed = json.loads((base / "r4/ATTACK_PREPARED_INDEX.json").read_text())
    r4id = f"R4_FINAL_{arm}_S{seed}_R{draw}"
    if r4id not in fixed["prepared_units"]:
        raise ValueError("release outside fixed 40-slot family")
    entry = fixed["prepared_units"][r4id]
    if (entry["arm_id"], entry["source_seed"], entry["realization"]) != (arm, seed, draw):
        raise PermissionError("fixed release identity differs")
    r4_receipt = base / "r4" / f"{r4id}_RECEIPT.json"
    r4_spec_path = repo / entry["spec_path"]
    if sha(r4_spec_path) != entry["spec_sha256"]:
        raise PermissionError("prepared R4 worker spec differs")
    r4_spec = json.loads(r4_spec_path.read_text())
    receipt = json.loads(r4_receipt.read_text())
    if (receipt.get("unit_id") != r4id or receipt.get("status") != "COMPLETE_PRIVATE_DEVELOPMENT_ONLY"
            or (receipt.get("arm_id"), receipt.get("source_seed"), receipt.get("realization"))
            != (arm, seed, draw)):
        raise PermissionError("fresh release attack is incomplete")
    gate_path = Path(r4_spec["eligibility"]["path"])
    gate = json.loads(gate_path.read_text())
    if (sha(gate_path) != r4_spec["eligibility"]["sha256"] or
            sha(gate_path) != receipt.get("eligibility_sha256") or
            gate.get("arm_id") != arm or
            gate.get("all_two_seed_task_reference_point_caps_pass") is not True or
            gate.get("original_inner", {}).get("passed") != 24):
        raise PermissionError("original INNER utility gate differs")
    private = base / "r4/.private/attacks" / r4id
    freeze = private / "FROZEN_COMPLETE.json"
    role_manifest = base / "r1/DATA_ACCESS_MANIFEST.json"
    role = json.loads(role_manifest.read_text())["roles"]["INNER_SELECTION"]
    sex = Path(r4_spec["sex_archives"]["INNER_SELECTION"]["path"])
    if (sha(sex) != role["source_files"]["sex"]["sha256"] or
            sha(sex) != r4_spec["sex_archives"]["INNER_SELECTION"]["sha256"]):
        raise PermissionError("INNER SEX source differs from role manifest")
    views = {}
    for view in ("r1", "r2", "pair"):
        folder = private / view / "readers"
        views[view] = {"reader_receipt": pin(folder / "READERS_RECEIPT.json"),
                       "predictions": pin(folder / "predictions_inner.npz"),
                       "view": pin(folder / "view_inner.npz")}
    unit = f"R5_FROZEN_AUC_{arm}_S{seed}_R{draw}"
    out = base / "r5" / f"{unit}.json"
    vector = base / "r5/.private" / f"{unit}_VECTORS.npz"
    config_path = base / "r5/.private" / f"{unit}_CONFIG.json"
    cfg = {"schema": "IVR1-r5-frozen-auc-replay-config-v1", "unit_id": unit,
           "arm_id": arm, "source_seed": seed, "realization": draw,
           "queue_seal": pin(base / "r0/ADMISSION_LOCK.json"),
           "campaign_seal": pin(base / "r0/CAMPAIGN_LOCK.json"),
           "verification_amendment": pin(amendment), "r4_receipt": pin(r4_receipt),
           "frozen": pin(freeze), "role_manifest": pin(role_manifest),
           "inner_sex": pin(sex), "views": views,
           "output_public": str(out), "output_private": str(vector)}
    config_sha = write_new(config_path, cfg, 0o600)
    venv = Path(os.path.abspath(repo.parents[1] / ".venv/bin/python"))
    if not venv.is_file() or not os.access(venv, os.X_OK):
        raise FileNotFoundError("exact scientific virtualenv entrypoint missing")
    sources = [config_path, base / "r0/ADMISSION_LOCK.json", base / "r0/CAMPAIGN_LOCK.json",
               amendment, r4_receipt, r4_spec_path, gate_path, freeze, role_manifest, sex,
               base / "r0/compute_queue.py", repo / "incumbent_view/r5_attack_auc_replay.py",
               repo / "incumbent_view/r5_attack_math.py",
               repo / "incumbent_view/r5_build_frozen_auc_specs.py"]
    for view in ("r1", "r2", "pair"):
        folder = private / view / "readers"
        sources += [folder / "READERS_RECEIPT.json", folder / "predictions_inner.npz",
                    folder / "view_inner.npz"]
    spec = {"unit_id": unit, "stage": "verification", "cwd": str(repo),
            "argv": [str(venv), "-m", "incumbent_view.r5_attack_auc_replay", "--config", str(config_path)],
            "max_wall_s": 300, "cpu_reserve_s": 300, "max_rss_gib": 2.5,
            "projected_output_bytes": 1000000, "deps": [r4id],
            "outputs": [str(out), str(vector)], "sources": [pin(p) for p in sources],
            "env": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"},
            "amendment_sha256": sha(amendment)}
    spec_path = base / "r5/.private" / f"{unit}_QUEUE_SPEC.json"
    spec_sha = write_new(spec_path, spec, 0o600)
    public = {"schema": "IVR1-r5-frozen-auc-spec-index-v1", "unit_id": unit,
              "r4_unit_id": r4id, "arm_id": arm, "source_seed": seed,
              "realization": draw, "private_spec_relative": str(spec_path.relative_to(repo)),
              "config_sha256": config_sha, "spec_sha256": spec_sha,
              "source_count": len(sources), "amendment_sha256": sha(amendment),
              "reader_fits": 0, "role": "INNER_SELECTION"}
    write_new(base / "r5" / f"{unit}_SPEC_INDEX.json", public, 0o644)
    return public


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--amendment", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--draw", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(args.repo), Path(args.amendment), args.arm, args.seed, args.draw), sort_keys=True))


if __name__ == "__main__":
    main()
