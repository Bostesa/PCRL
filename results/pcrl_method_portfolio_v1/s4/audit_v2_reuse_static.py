"""Independent metadata-only custody audit for V2 reuse of seed-zero releases.

This script reads JSON receipts and hashes their bytes. It does not import a
candidate, evaluator, or runner, and does not open person-level arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve()
CAMPAIGN = HERE.parents[1]
B_PRIVATE = (
    HERE.parents[4]
    / "pcrl-method-portfolio-b/results/pcrl_method_portfolio_v1/teams/b/.private"
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def main() -> None:
    lock_path = CAMPAIGN / "CAMPAIGN_LOCK.json"
    manifest_path = CAMPAIGN / "DATA_ACCESS_MANIFEST.json"
    queue_path = CAMPAIGN / "shared/.private/compute/QUEUE.json"
    lock, manifest, queue = read(lock_path), read(manifest_path), read(queue_path)
    lock_sha, manifest_sha = digest(lock_path), digest(manifest_path)
    require(lock_sha == "6d82822f5fe54c16f76e6140eb7e780251f4d7cb288af0166855187ec5069dfa", "fit lock changed")
    require(manifest_sha == lock["data_access_manifest_sha256"], "manifest differs from lock")
    att_pair = manifest["real_file_specs"]["ATTACK_FIT_NEW:features"]["row_unit_pair_sha256"]
    inner_pair = manifest["real_file_specs"]["INNER_SELECTION:features"]["row_unit_pair_sha256"]
    b_lock = lock["families"]["B"]
    code = b_lock["code_sha256_by_file"]
    expected_code = {
        "attack_view_builder": code["attack_bridge.py"],
        "release_adapter": code["bottleneck.py"],
        "screen_driver": code["screen_driver.py"],
        "wire_flattener": code["attack_views.py"],
    }
    arms = ("B-TASK", "B-JVIB", "B-DVIB", "B-RAW-TASK", "B-LOCAL")
    findings = []
    for arm in arms:
        suffix = "__R2" if arm == "B-TASK" else ""
        screen_dir = B_PRIVATE / f"screen_v1__{arm}__s0{suffix}"
        probe_dir = B_PRIVATE / f"real_probe_v1__{arm}__s0{suffix}"
        attempt_path = screen_dir / "ATTEMPT_INDEX.json"
        probe_path = probe_dir / "REAL_CONTROL_INDEX.json"
        attempt, probe = read(attempt_path), read(probe_path)
        attempt_sha, probe_sha = digest(attempt_path), digest(probe_path)
        key = f"{arm}:s0"
        screen_id = b_lock["screen_unit_ids"][key]
        probe_id = b_lock["real_control_unit_ids"][key]
        screen_unit, probe_unit = queue["units"][screen_id], queue["units"][probe_id]
        require(screen_unit["status"] == "done" and screen_unit["returncode"] == 0, f"{arm} screen unit")
        require(screen_unit["output_sha256"][str(attempt_path)] == attempt_sha, f"{arm} screen output")
        require(screen_unit["spec"]["arm"] == arm and screen_unit["spec"]["seed"] == 0, f"{arm} screen spec")
        require(screen_unit["spec"]["argv"][-1] == str(screen_dir), f"{arm} attempt root")
        require(probe_unit["status"] in ("done", "failed"), f"{arm} V1 unit status")
        if probe_unit["status"] == "done":
            require(probe_unit["output_sha256"][str(probe_path)] == probe_sha, f"{arm} V1 output")
        else:
            require(probe_unit["returncode"] != 0 and str(probe_path) in probe_unit["spec"]["outputs"], f"{arm} preserved failed V1 output")
        require(attempt["status"] == "FROZEN_FOR_INDEPENDENT_REPLAY" and attempt["arm"] == arm and attempt["seed"] == 0, f"{arm} attempt identity")
        require(attempt["screen_unit_id"] == screen_id, f"{arm} screen ID")
        require(attempt["campaign_lock_sha256"] == lock_sha and attempt["data_manifest_sha256"] == manifest_sha, f"{arm} lock/manifest")
        require(attempt["source_code_sha256"] == expected_code, f"{arm} B source bytes")
        require(attempt["screen_config_sha256"] == b_lock["screen_config_sha256"], f"{arm} config")
        require(probe["attempt_index_sha256"] == attempt_sha and probe["screen_unit_id"] == screen_id, f"{arm} V1 parent")
        require(probe["queue_unit_id"] == probe_id and probe["arm"] == arm and probe["seed"] == 0, f"{arm} V1 identity")
        views = {}
        for view, width in (("income", 36), ("occupation", 44), ("pair", 80)):
            reader_path = screen_dir / f"readers__{view}/READERS_RECEIPT.json"
            reader = read(reader_path)
            require(digest(reader_path) == attempt["reader_receipt_sha256"][view], f"{arm}/{view} reader hash")
            require(reader["status"] == "FROZEN", f"{arm}/{view} reader status")
            require(reader["att_row_unit_pair_sha256"] == att_pair and reader["inner_row_unit_pair_sha256"] == inner_pair, f"{arm}/{view} row order")
            require(len(reader["view_schema"]["wire_components"]) == width, f"{arm}/{view} wire width")
            require(len(reader["reader_order"]) == (12 if view == "pair" else 4), f"{arm}/{view} reader slate")
            require(len(reader["ignore_recipient_aliases"]) == (8 if view == "pair" else 0), f"{arm}/{view} ignore aliases")
            report_path = probe_dir / f"REAL_CONTROL__{view}.json"
            report = read(report_path)
            require(digest(report_path) == probe["view_report_sha256"][view], f"{arm}/{view} V1 report hash")
            require(report["attempt_index_sha256"] == attempt_sha, f"{arm}/{view} V1 index")
            require(report["att_row_unit_pair_sha256"] == att_pair and report["inner_row_unit_pair_sha256"] == inner_pair, f"{arm}/{view} V1 role order")
            require(report["att_view_content_sha256"] == reader["att_view_content_sha256"], f"{arm}/{view} ATT view")
            require(report["inner_view_content_sha256"] == reader["inner_view_content_sha256"], f"{arm}/{view} INNER view")
            views[view] = {"reader_receipt_sha256": digest(reader_path), "v1_report_sha256": digest(report_path), "v1_status_preserved": report["status"], "wire_fields": width}
        findings.append({"arm": arm, "seed": 0, "screen_unit_id": screen_id, "attempt_index_sha256": attempt_sha, "v1_probe_unit_id": probe_id, "v1_index_sha256": probe_sha, "v1_status_preserved": probe["status"], "views": views})

    anchor_root = CAMPAIGN / "shared/.private/feature_ref_raw83_u_s0_v1"
    anchor_path = anchor_root / "ANCHOR_RECEIPT.json"
    anchor, anchor_sha = read(anchor_path), digest(anchor_path)
    anchor_id = "S2-FEATURE-REF-RAW83-U-S0"
    anchor_unit = queue["units"][anchor_id]
    require(anchor_unit["status"] == "done" and anchor_unit["returncode"] == 0, "anchor unit")
    require(anchor_unit["output_sha256"][str(anchor_path)] == anchor_sha, "anchor output")
    require(anchor["campaign_lock_sha256"] == lock_sha and anchor["data_access_manifest_sha256"] == manifest_sha, "anchor lock/manifest")
    require(anchor["runner_sha256"] == lock["shared_anchor_runner_sha256"], "anchor runner")
    for key, file_sha in anchor["role_file_sha256"].items():
        require(manifest["real_file_specs"][key]["sha256"] == file_sha, f"anchor role {key}")
    anchor_views = {}
    for view, width in (("r1", 87), ("r2", 95), ("pair", 182)):
        reader_path = anchor_root / f"readers_{view}/READERS_RECEIPT.json"
        reader = read(reader_path)
        require(digest(reader_path) == anchor["reader_receipt_sha256"][view], f"anchor/{view} reader hash")
        require(reader["att_row_unit_pair_sha256"] == att_pair and reader["inner_row_unit_pair_sha256"] == inner_pair, f"anchor/{view} row order")
        require(reader["view_schema"]["component_columns"][-1][2] == width, f"anchor/{view} numeric width")
        require(len(reader["view_schema"]["wire_components"]) == (14 if view == "pair" else 7), f"anchor/{view} wire fields")
        require(len(reader["reader_order"]) == (18 if view == "pair" else 6) and len(reader["ignore_recipient_aliases"]) == (12 if view == "pair" else 0), f"anchor/{view} reader slate")
        probe_path = anchor_root / f"PROBE_{view}.json"
        probe = read(probe_path)
        require(digest(probe_path) == anchor["real_probe_receipt_sha256"][view] and probe["status"] == "PASS", f"anchor/{view} V1 probe")
        anchor_views[view] = {"reader_receipt_sha256": digest(reader_path), "v1_probe_sha256": digest(probe_path), "wire_fields": width}
    report = {
        "schema": "pcrl-s4-v2-reuse-static-audit-v1",
        "status": "STATIC_IDENTITY_PASS_NUMERICAL_REPLAY_PENDING",
        "scope": "five seed-zero B screens and V1 controls, plus seed-zero feature anchor",
        "fit_lock_sha256": lock_sha,
        "data_manifest_sha256": manifest_sha,
        "queue_metadata_sha256": digest(queue_path),
        "b_arms": findings,
        "feature_anchor": {"unit_id": anchor_id, "receipt_sha256": anchor_sha, "views": anchor_views},
        "limits": [
            "Reads JSON metadata and SHA256 hashes only; no person arrays, fitted models, predictions, or labels were opened.",
            "Preserves V1 PASS/FAIL as historical statuses; no V1 result is reclassified as V2.",
            "Full release, reader, reference, endpoint, and V2 numerical replay remains pending a separate S4 verifier seal.",
        ],
    }
    output = CAMPAIGN / "S4_V2_REUSE_STATIC_AUDIT.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(f"PASS {len(findings)} B screens; anchor; report_sha256={digest(output)}")


if __name__ == "__main__":
    main()
