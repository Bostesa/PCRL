"""Independent identity checks between an evaluated law and its selected R2 release.

This module has no role loader and imports no R2 or R3 endpoint implementation.
Callers hash each source file before parsing it, then pass the resulting
documents and byte hashes here.
"""

from __future__ import annotations

from collections.abc import Mapping


def validate_links(config: Mapping, r3: Mapping, release: Mapping,
                   policy: Mapping, unit: Mapping, *, csv_sha: str,
                   release_sha: str, policy_sha: str) -> dict:
    arm, seed, role = config.get("arm"), config.get("seed"), config.get("role")
    law_sha = config.get("law", {}).get("sha256")
    if (config.get("schema") != "pcrl-bank-r3-utility-job-v1" or
            config.get("stage") != "protection" or
            arm not in ("T", "D", "P", "L", "J", "S12", "S21") or
            seed not in (0, 1) or
            role not in ("OSF_DEFENSE_FIT", "INNER_SELECTION") or
            type(law_sha) is not str or len(law_sha) != 64):
        raise PermissionError("R3 protected law identity differs")
    if (r3.get("schema") != "pcrl-bank-r3-utility-receipt-v1" or
            r3.get("unit_id") != config.get("unit_id") or
            r3.get("arm") != arm or r3.get("seed") != seed or
            r3.get("role") != role or
            r3.get("law_sha256") != law_sha or
            r3.get("output_csv_sha256") != csv_sha):
        raise PermissionError("R3 evaluated law or output bytes differ")
    if (release.get("schema") != "BAR1-role-release-v1" or
            release.get("arm_id") != arm or release.get("source_seed") != seed or
            release.get("role") != role or release.get("realization") != 0 or
            release.get("law_sha256") != law_sha or
            release.get("policy_receipt_sha256") != policy_sha):
        raise PermissionError("R2 selected release law differs from R3 law")
    if (policy.get("schema") != "BAR1-selected-policy-v1" or
            policy.get("arm_id") != arm or policy.get("source_seed") != seed or
            policy.get("source_binding_sha256") !=
            release.get("source_binding_sha256")):
        raise PermissionError("R2 selected policy and release differ")
    if (unit.get("schema") != "BAR1-r2-unit-receipt-v1" or
            unit.get("arm_id") != arm or unit.get("source_seed") != seed or
            unit.get("status") not in
            ("COMPLETE_IDENTITY_ALIAS", "COMPLETE_SELECTED_PROTECTED") or
            unit.get("policy_receipt_sha256") != policy_sha):
        raise PermissionError("R2 unit is not a completed selected policy")
    matches = [item for item in unit.get("role_releases", ())
               if item.get("role") == role and item.get("realization") == 0]
    if (len(matches) != 1 or matches[0].get("receipt_sha256") != release_sha or
            matches[0].get("law_sha256") != law_sha):
        raise PermissionError("R2 unit index does not bind the evaluated release")
    return {"status": "PASS", "arm_id": arm, "source_seed": seed,
            "role": role, "r3_unit_id": config["unit_id"],
            "r2_unit_id": unit["unit_id"], "law_sha256": law_sha,
            "r3_csv_sha256": csv_sha,
            "r2_release_receipt_sha256": release_sha,
            "r2_policy_receipt_sha256": policy_sha}
