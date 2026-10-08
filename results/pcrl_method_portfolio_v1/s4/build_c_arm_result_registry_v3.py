"""Hash-only post-arm custody index for the 16 corrected S4 C R2 units.

Reads S3 JSON receipts and hashes private S4 report/NPZ bytes without decoding
person arrays. Writes a public 16/16 index required by the aggregate unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
QUEUE_ROOT = CAMPAIGN / "shared/.private/compute"
ADMISSION = CAMPAIGN / "S4_C_R3_REPLAY_ADMISSION.json"
OUT = CAMPAIGN / "S4_C_R3_ARM_RESULT_HASH_REGISTRY.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    item = json.loads(path.read_text())
    if not isinstance(item, dict):
        raise ValueError("S4 C hash registry JSON is not an object")
    return item


def build() -> dict:
    admission = read(ADMISSION)
    if (admission.get("schema") != "pcrl-s4-c-r3-replay-admission-v3" or
            admission.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"):
        raise PermissionError("S4 C effective admission is absent")
    expected = {unit_id: spec for unit_id, spec in admission["unit_specs"].items()
                if spec.get("stage") == "independent_c_r3_arm_replay"}
    if len(expected) != 16:
        raise ValueError("S4 C sixteen arm roster differs")
    queue = read(QUEUE_ROOT / "QUEUE.json")
    units = {}
    for unit_id, admitted in expected.items():
        state = queue.get("units", {}).get(unit_id, {})
        spec = state.get("spec", {})
        receipt_path = QUEUE_ROOT / "units" / unit_id / "RECEIPT.json"
        receipt = read(receipt_path)
        outputs = spec.get("outputs", [])
        if (state.get("status") != "done" or receipt.get("status") != "done" or
                state.get("returncode") != 0 or receipt.get("returncode") != 0 or
                receipt.get("unit_id") != unit_id or
                spec.get("stage") != admitted["stage"] or
                spec.get("mode") != "s4_c_r3_adult_replay" or
                spec.get("family") != spec.get("team") != "S4" or
                spec.get("arm_id") != admitted["arm"] or
                spec.get("arm") != admitted["arm"] or
                spec.get("seed") != admitted["seed"] or
                spec.get("data_access") != admitted["data_access"] or
                spec.get("max_wall_s") != admitted["max_wall_s"] or
                spec.get("cpu_reserve_s") != admitted["cpu_reserve_s"] or
                spec.get("max_rss_gib") != admitted["max_rss_gib"] or
                set(spec.get("deps", [])) != {
                    item["unit_id"] for item in admitted["parents"].values()} or
                not isinstance(outputs, list) or len(outputs) != 1 or
                state.get("output_sha256") != receipt.get("output_sha256")):
            raise ValueError(f"S4 C {unit_id} S3 completion differs")
        if spec.get("runner_sha256") != admission["source_sha256_by_path"][
                "results/pcrl_method_portfolio_v1/s4/run_c_adult_replay_v3.py"]:
            raise ValueError(f"S4 C {unit_id} runner source differs")
        path = Path(outputs[0]).resolve(strict=True)
        report_sha = receipt["output_sha256"].get(outputs[0])
        if (not path.is_relative_to((HERE / ".private").resolve()) or
                path.name != f"{unit_id}.json" or
                not isinstance(report_sha, str) or sha(path) != report_sha):
            raise ValueError(f"S4 C {unit_id} private report hash differs")
        report = read(path)
        artifact = report.get("verified_inner_artifacts", {})
        artifact_path = path.with_suffix(".npz")
        if (report.get("status") != "REPLAYED_C_R3_PASS" or
                report.get("unit_id") != unit_id or
                report.get("arm") != admitted["arm"] or
                report.get("seed") != admitted["seed"] or
                artifact.get("filename") != artifact_path.name or
                artifact.get("sha256") != sha(artifact_path)):
            raise ValueError(f"S4 C {unit_id} replay/artifact identity differs")
        units[unit_id] = {"unit_id": unit_id,
                          "queue_receipt_sha256": sha(receipt_path),
                          "report_sha256": report_sha,
                          "artifact_npz_sha256": artifact["sha256"],
                          "status": "REPLAYED_C_R3_PASS"}
    return {"schema": "pcrl-s4-c-r3-arm-result-hash-registry-v3",
            "status": "S4_HASH_CHECKED_16_OF_16",
            "admission_sha256": sha(ADMISSION),
            "fit_lock_sha256": admission["lock_sha256"],
            "registry_builder_sha256": sha(Path(__file__)),
            "units": units,
            "person_values": "PRIVATE_S4_NPZ_ONLY"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.resolve() != OUT.resolve():
        raise PermissionError("S4 C registry output path differs")
    result = build()
    with OUT.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps({"sha256": sha(OUT), "units": len(result["units"])},
                     sort_keys=True))


if __name__ == "__main__":
    main()
