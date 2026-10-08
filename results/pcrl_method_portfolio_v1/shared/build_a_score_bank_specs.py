"""Build sealed S3 specs for the ten fixed Team A source-bank preparation units.

This utility reads public manifests and lock metadata only. It never reads role
NPZ values, trains a model, enqueues a job, or runs a source-bank adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from compute_queue import Queue, THREAD_ENV
from safe_data_guard import sha_file


PLAN_SHA256 = "06b8e3ae947d23659f7ba46f3e78d9c83a9ce9966656d96d5f766b5984b07565"
SHARED = Path(__file__).resolve().parent
CAMPAIGN = SHARED.parent
ROOT = CAMPAIGN.parents[1]
ROLES = ("OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION")
RELEASE_KEYS = ("tstar", "pstar", "lra_seq21", "hcal_local_d0", "hcal_seq12_d0")
CACHE_ROOT_RELATIVE = "../../../PCRL_eval_cache_private"
DATA_ACCESS = "A_ROLE_ONLY_REFERENCES_PREPROCESS"
ARM = "A_SCORE_BANK_PREPROCESS"
HEX64 = re.compile(r"[0-9a-f]{64}")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("A source-bank metadata is not an object")
    return value


def _check_sha(value: object) -> None:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError("A source-bank SHA identity is malformed")


def _expected_unit_id(key: str, seed: int) -> str:
    return f"S2-A-SCORE-BANK-{key.upper().replace('_', '-')}-S{seed}"


def build_specs(plan_path: Path, lock_path: Path, expected_lock_sha256: str,
                coordinator_root: Path, queue_root: Path) -> list[dict]:
    """Pure metadata validation and exact S3 spec construction."""
    plan_path, lock_path = Path(plan_path), Path(lock_path)
    root = Path(coordinator_root).resolve()
    queue_root = Path(queue_root).resolve()
    _check_sha(expected_lock_sha256)
    if root != ROOT or plan_path.resolve() != SHARED / "A_SCORE_BANK_PREPROCESS_SPEC_DRAFT.json":
        raise PermissionError("A source-bank plan or coordinator checkout differs")
    if sha_file(plan_path) != PLAN_SHA256 or sha_file(lock_path) != expected_lock_sha256:
        raise PermissionError("A plan or effective campaign lock SHA differs")
    plan, lock = _read_json(plan_path), _read_json(lock_path)
    if (plan.get("schema") != "pcrl-a-score-bank-preprocess-plan-v1" or
            plan.get("status") != "DRAFT_NO_ADULT_ACCESS" or
            plan.get("runner_relative_to_coordinator") !=
            "results/pcrl_method_portfolio_v1/shared/admit_a_score_bank.py" or
            plan.get("cache_root_relative_to_coordinator") != CACHE_ROOT_RELATIVE or
            len(plan.get("units", [])) != 10):
        raise ValueError("A fixed ten-unit public plan differs")
    if (lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE" or
            lock.get("adult_fit_permission") is not True or
            lock.get("fit_admission") != "B_ONLY" or
            lock.get("families", {}).get("A", {}).get("preprocessing_admission") != "ALLOWED" or
            lock.get("families", {}).get("A", {}).get("fit_admission") != "DENIED" or
            lock.get("a_score_bank_preprocess_plan_sha256") != PLAN_SHA256 or
            lock.get("a_score_bank_cache_root") != CACHE_ROOT_RELATIVE or
            lock.get("a_score_bank_runner_sha256") != plan.get("runner_sha256") or
            lock.get("admitted_a_score_bank_unit_ids") !=
            plan.get("required_lock", {}).get("admitted_a_score_bank_unit_ids")):
        raise PermissionError("effective lock does not admit the fixed A preprocessing plan")
    _check_sha(plan["runner_sha256"])
    script = (root / plan["runner_relative_to_coordinator"]).resolve(strict=True)
    if sha_file(script) != plan["runner_sha256"]:
        raise PermissionError("A source-bank adapter differs from the lock")
    python = (root / "../../.venv/bin/python").resolve(strict=True)
    cache_root = (root / CACHE_ROOT_RELATIVE).resolve(strict=True)
    if not cache_root.is_dir():
        raise PermissionError("registered private source cache is unavailable")
    data_path = CAMPAIGN / "DATA_ACCESS_MANIFEST.json"
    if (sha_file(data_path) != plan.get("data_access_manifest_sha256") or
            lock.get("data_admission", {}).get("manifest_sha256") != sha_file(data_path)):
        raise PermissionError("admitted role-only manifest differs")
    data = _read_json(data_path)
    if data.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise PermissionError("role-only files lack independent admission")
    source_paths = {
        "hcal_MODEL_MANIFEST": root / "results/pcrl_heldout_calibration_v1/MODEL_MANIFEST.json",
        "hcal_SOURCE_ADMISSION": root / "results/pcrl_heldout_calibration_v1/SOURCE_ADMISSION.json",
        "lra_MODEL_MANIFEST": root / "results/pcrl_adult_learned_decoder_release_v1/MODEL_MANIFEST.json",
    }
    source_lock_keys = {"hcal_MODEL_MANIFEST": "hcal_model_manifest_sha256",
                        "hcal_SOURCE_ADMISSION": "hcal_source_admission_sha256",
                        "lra_MODEL_MANIFEST": "lra_model_manifest_sha256"}
    source_lock = lock.get("source_model_admission", {})
    for name, path in source_paths.items():
        expected = plan["source_manifests_sha256"][name]
        _check_sha(expected)
        if sha_file(path) != expected or source_lock.get(source_lock_keys[name]) != expected:
            raise PermissionError("A inherited source manifest differs")
    expected_pairs = {(key, seed) for key in RELEASE_KEYS for seed in (0, 1)}
    seen_pairs, ids, specs = set(), [], []
    expected_s3 = {"mode": "real_data", "family": "SHARED", "team": "SHARED",
                   "stage": "a_score_bank_preprocess", "data_access": DATA_ACCESS,
                   "arm": ARM, "arm_id": ARM, "max_wall_s": 300,
                   "cpu_reserve_s": 300, "max_rss_gib": 1.5}
    for unit in plan["units"]:
        key, seed = unit["release_key"], unit["seed"]
        if ((key, seed) not in expected_pairs or (key, seed) in seen_pairs or
                unit.get("unit_id") != _expected_unit_id(key, seed) or
                unit.get("s3") != expected_s3):
            raise ValueError("A unit ID, pair or S3 limits differ")
        seen_pairs.add((key, seed))
        ids.append(unit["unit_id"])
        u_model_path = f"hcal_v1/admitted/rel__s{seed}__U/model.pt"
        if (source_lock.get("U_model_pt_sha256_by_seed", {}).get(f"s{seed}") !=
                unit["source_files_sha256"].get(u_model_path)):
            raise PermissionError("A seed-specific source U checkpoint differs")
        for rel, expected in unit["source_files_sha256"].items():
            _check_sha(expected)
            path = Path(rel)
            if path.is_absolute() or ".." in path.parts or not (cache_root / path).is_file():
                raise PermissionError("A source file is absent or outside the private cache")
        if set(unit["role_inputs"]) != set(ROLES):
            raise ValueError("A score-bank role allowlist differs")
        for role in ROLES:
            actual = data["real_file_specs"][f"{role}:references_s{seed}"]
            expected = unit["role_inputs"][role]
            if expected != {"field": f"references_s{seed}",
                             "role_file_sha256": actual["sha256"],
                             "row_unit_pair_sha256": actual["row_unit_pair_sha256"],
                             "rows": data["roles"][role]["rows"]}:
                raise PermissionError("A role-only reference identity differs")
        output_relative = ("results/pcrl_method_portfolio_v1/shared/.private/" +
                           f"{key}_score_bank_s{seed}_v1/BANK_ADMISSION_RECEIPT.json")
        if unit.get("private_output_receipt_relative_to_coordinator") != output_relative:
            raise ValueError("A private output receipt destination differs")
        output = (root / output_relative).resolve()
        if output.exists() or output.parent.exists():
            raise FileExistsError("A source-bank output is write-once")
        argv = [str(python), str(script), "--cache-root", str(cache_root),
                "--seed", str(seed), "--release", key,
                "--campaign-lock-sha256", expected_lock_sha256,
                "--queue-root", str(queue_root)]
        if unit.get("argv_template") != ["../../.venv/bin/python",
             plan["runner_relative_to_coordinator"], "--cache-root",
             "${RESOLVED_PRIVATE_CACHE_ROOT}", "--seed", str(seed),
             "--release", key, "--campaign-lock-sha256",
             "${EFFECTIVE_CAMPAIGN_LOCK_SHA256}", "--queue-root",
             "${PRIVATE_S3_QUEUE_ROOT}"]:
            raise ValueError("A fixed argv template differs")
        specs.append({"unit_id": unit["unit_id"], "argv": argv,
                      "cwd": str(root), "deps": [], "outputs": [str(output)],
                      "env": {**THREAD_ENV, "PYTHONDONTWRITEBYTECODE": "1",
                              "PYTHONPATH": os.pathsep.join((str(root), str(SHARED)))},
                      "runner_sha256": plan["runner_sha256"],
                      "preprocess_plan_sha256": PLAN_SHA256,
                      "seed": seed, "release_key": key, **expected_s3})
    if (seen_pairs != expected_pairs or ids != lock["admitted_a_score_bank_unit_ids"] or
            plan.get("reserved_cpu_seconds_total") != 3000 or
            plan.get("reserved_cpu_hours_total") != 3000 / 3600):
        raise ValueError("A ten-unit order or CPU reservation differs")
    return specs


def write_specs(specs: list[dict], lock_sha256: str, seal_sha256: str,
                queue_root: Path) -> dict:
    """Write one private immutable spec set only after S1/S4 committed seal."""
    seal_path = CAMPAIGN / "LOCAL_SEAL_RECEIPT.json"
    _check_sha(seal_sha256)
    if (not seal_path.is_file() or sha_file(seal_path) != seal_sha256 or
            not Queue._local_seal_ok({"local_seal_path": str(seal_path),
                                      "local_seal_sha256": seal_sha256})):
        raise PermissionError("committed A preprocessing local seal is absent")
    receipt = _read_json(seal_path)
    if receipt.get("campaign_lock_sha256") != lock_sha256:
        raise PermissionError("local seal and spec campaign lock differ")
    if specs != build_specs(SHARED / "A_SCORE_BANK_PREPROCESS_SPEC_DRAFT.json",
                            CAMPAIGN / "CAMPAIGN_LOCK.json", lock_sha256,
                            ROOT, queue_root):
        raise PermissionError("A queued spec values differ from the exact public plan")
    output_root = SHARED / ".private" / "a_score_bank_specs_v1"
    if output_root.exists():
        raise FileExistsError("A source-bank spec set is write-once")
    output_root.mkdir(parents=True, mode=0o700)
    files = {}
    for spec in specs:
        path = output_root / f"{spec['unit_id']}.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(spec, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        files[path.name] = sha_file(path)
    index = {"schema": "pcrl-a-score-bank-s3-spec-index-v1",
             "status": "SPECS_BUILT_NOT_ENQUEUED", "plan_sha256": PLAN_SHA256,
             "campaign_lock_sha256": lock_sha256, "local_seal_receipt_sha256": seal_sha256,
             "unit_specs_sha256": files}
    index_path = output_root / "SPEC_INDEX.json"
    fd = os.open(index_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(index, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"units": len(specs), "index_sha256": sha_file(index_path)}


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-lock-sha256", required=True)
    parser.add_argument("--local-seal-sha256", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        specs = build_specs(SHARED / "A_SCORE_BANK_PREPROCESS_SPEC_DRAFT.json",
                            CAMPAIGN / "CAMPAIGN_LOCK.json",
                            args.campaign_lock_sha256, ROOT, args.queue_root)
        result = write_specs(specs, args.campaign_lock_sha256,
                             args.local_seal_sha256, args.queue_root)
    except BaseException as exc:
        print("A_SCORE_BANK_SPECS_FAILED:" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True))
