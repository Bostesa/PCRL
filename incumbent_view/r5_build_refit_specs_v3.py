"""Receipt-only builder for the predeclared bounded complete-view reader refits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pin(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"refit source missing or symlinked: {path}")
    return {"path": str(path.resolve(strict=True)), "sha256": sha(path)}


def write_new(path: Path, record: dict, mode: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "w") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return sha(path)


def build(repo: Path, amendment: Path, claims_path: Path) -> dict:
    repo = repo.resolve(strict=True)
    amendment = amendment.resolve(strict=True)
    claims_path = claims_path.resolve(strict=True)
    base = repo / "results/pcrl_incumbent_view_risk_v1"
    if amendment.parent != base / "r0" or claims_path.parent != base / "r3":
        raise PermissionError("refit source namespace differs")
    science = base / "r0/CAMPAIGN_LOCK.json"
    if json.loads(amendment.read_text()).get("campaign_lock_sha256") != sha(science):
        raise PermissionError("refit amendment does not bind science")
    claim = json.loads(claims_path.read_text())
    rule_path = base / "r5/BOUNDED_REFIT_SUBSET_V3.json"
    rule = json.loads(rule_path.read_text())
    if (claim.get("schema") != "pcrl-incumbent-r3-fixed-claim-accounting-v1" or
            claim.get("K") != 948 or claim.get("nominee") != "V_L" or
            rule.get("schema") != "IVR1-r5-bounded-refit-subset-v3" or
            rule.get("status") != "RECORDED_AFTER_ALL_ATTACK_JOBS_BEFORE_FINAL_AGGREGATE" or
            rule.get("nominee") != "V_L"):
        raise PermissionError("fixed conditional refit selection differs")
    strongest = claim["strongest_control"].get("selected")
    arms = list(dict.fromkeys(x for x in ("P0", "V_L", strongest) if x is not None))
    if len(arms) > 3 or len(arms) < 2:
        raise PermissionError("bounded comparator inventory differs")
    prepared = json.loads((base / "r4/ATTACK_PREPARED_INDEX.json").read_text())
    output = []
    for arm in arms:
        for seed in (0, 1):
            r4_id = f"R4_FINAL_{arm}_S{seed}_R0"
            source = prepared["prepared_units"][r4_id]
            r4_spec_path = repo / source["spec_path"]
            if sha(r4_spec_path) != source["spec_sha256"]:
                raise PermissionError("prepared R4 source spec differs")
            r4_spec = json.loads(r4_spec_path.read_text())
            r4_receipt = base / "r4" / f"{r4_id}_RECEIPT.json"
            r4_record = json.loads(r4_receipt.read_text())
            if (r4_record.get("status") != "COMPLETE_PRIVATE_DEVELOPMENT_ONLY" or
                    r4_record.get("unit_id") != r4_id or
                    (r4_record.get("arm_id"), r4_record.get("source_seed"),
                     r4_record.get("realization")) != (arm, seed, 0)):
                raise PermissionError("fixed draw-zero R4 reader incomplete")
            root = base / "r4/.private/attacks" / r4_id
            folder = root / "pair/readers"
            unit = f"R5_BOUNDED_REFIT_{arm}_S{seed}_R0"
            out = base / "r5" / f"{unit}.json"
            private = base / "r5/.private" / unit
            config_path = base / "r5/.private" / f"{unit}_CONFIG.json"
            att_sex = Path(r4_spec["sex_archives"]["ATTACK_FIT_NEW"]["path"])
            inner_sex = Path(r4_spec["sex_archives"]["INNER_SELECTION"]["path"])
            if (sha(att_sex) != r4_spec["sex_archives"]["ATTACK_FIT_NEW"]["sha256"] or
                    sha(inner_sex) != r4_spec["sex_archives"]["INNER_SELECTION"]["sha256"]):
                raise PermissionError("role-only SEX byte pins differ")
            cfg = {"schema": "IVR1-r5-bounded-refit-config-v1", "unit_id": unit,
                   "arm_id": arm, "source_seed": seed, "realization": 0, "view": "pair",
                   "queue_seal": pin(base / "r0/ADMISSION_LOCK.json"),
                   "campaign_seal": pin(science), "verification_amendment": pin(amendment),
                   "subset_rule": pin(rule_path), "r3_claims": pin(claims_path),
                   "r4_receipt": pin(r4_receipt), "frozen": pin(root / "FROZEN_COMPLETE.json"),
                   "role_manifest": pin(base / "r1/DATA_ACCESS_MANIFEST.json"),
                   "att_sex": pin(att_sex), "inner_sex": pin(inner_sex),
                   "pair_view": {"reader_receipt": pin(folder / "READERS_RECEIPT.json"),
                                 "predictions": pin(folder / "predictions_inner.npz"),
                                 "att_view": pin(folder / "view_att.npz"),
                                 "inner_view": pin(folder / "view_inner.npz")},
                   "output_public": str(out), "output_private_dir": str(private)}
            config_sha = write_new(config_path, cfg, 0o600)
            venv = repo.parents[1] / ".venv/bin/python"
            if not venv.is_file() or not os.access(venv, os.X_OK):
                raise FileNotFoundError("exact scientific virtualenv entrypoint missing")
            sources = [config_path, base / "r0/ADMISSION_LOCK.json", science, amendment,
                       rule_path, claims_path, r4_receipt, r4_spec_path,
                       root / "FROZEN_COMPLETE.json", base / "r1/DATA_ACCESS_MANIFEST.json",
                       att_sex, inner_sex, folder / "READERS_RECEIPT.json",
                       folder / "predictions_inner.npz", folder / "view_att.npz",
                       folder / "view_inner.npz", base / "r0/compute_queue.py",
                       repo / "incumbent_view/r5_reader_refit_v3.py",
                       repo / "incumbent_view/r5_attack_auc_replay.py",
                       repo / "incumbent_view/r5_attack_math.py",
                       repo / "incumbent_view/r5_build_refit_specs_v3.py"]
            spec = {"unit_id": unit, "stage": "verification", "cwd": str(repo),
                    "argv": [str(venv), "-m", "incumbent_view.r5_reader_refit_v3",
                             "--config", str(config_path)],
                    "max_wall_s": 300, "cpu_reserve_s": 300, "max_rss_gib": 2.5,
                    "projected_output_bytes": 5000000, "deps": [r4_id],
                    "outputs": [str(out), str(private / "common.joblib"),
                                str(private / "stronger.joblib"),
                                str(private / "predictions.npz")],
                    "sources": [pin(path) for path in sources],
                    "env": {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                                 "VECLIB_MAXIMUM_THREADS")},
                    "amendment_sha256": sha(amendment)}
            spec_path = base / "r5/.private" / f"{unit}_QUEUE_SPEC.json"
            spec_sha = write_new(spec_path, spec, 0o600)
            output.append({"unit_id": unit, "arm_id": arm, "source_seed": seed,
                           "r4_unit_id": r4_id, "spec_sha256": spec_sha,
                           "config_sha256": config_sha,
                           "private_spec_relative": str(spec_path.relative_to(repo))})
    if len(output) > rule["maximum_units"]:
        raise PermissionError("bounded refit unit count exceeded")
    index = {"schema": "IVR1-r5-bounded-refit-index-v3", "status": "SPECS_ONLY",
             "subset_rule_sha256": sha(rule_path), "r3_claims_sha256": sha(claims_path),
             "amendment_sha256": sha(amendment), "units": output,
             "planned_fits_per_unit": 2, "result_selection": False}
    write_new(base / "r5/BOUNDED_REFIT_SPEC_INDEX_V3.json", index, 0o644)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--amendment", required=True)
    parser.add_argument("--r3-claims", required=True)
    args = parser.parse_args()
    result = build(Path(args.repo), Path(args.amendment), Path(args.r3_claims))
    print(json.dumps({"status": result["status"], "units": len(result["units"])}, sort_keys=True))


if __name__ == "__main__":
    main()
