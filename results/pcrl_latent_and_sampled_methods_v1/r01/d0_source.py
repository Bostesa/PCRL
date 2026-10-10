"""Fail-closed role-only source for three historical uncalibrated D0 controls.

These releases are contextual score controls with their original smoothed
prototype banks. Their exact role-only wires can be scored on repaired FIT and
INNER, but they are not same-bank aliases of the new sampled A channel.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .pstar_source import PStarSourceStore
from .role_store import AccessRefused, RoleOnlyStore, sha_file


RELEASES = {
    "hcal_local_d0": "U|LOCAL|i8o64|l0.1",
    "hcal_seq12_d0": "U|SEQ-12|i8o64|l0.1",
    "lra_seq21": "U|SEQ-21|i8o64|l0.1",
}
ROLES = frozenset({"OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION"})


class D0SourceStore(PStarSourceStore):
    """One D0 release and source seed, restricted to three admitted roles."""

    def __init__(self, portfolio_package: str | Path,
                 repair_manifest_path: str | Path, audit_path: str | Path,
                 cache_root: str | Path, release_key: str, seed: int):
        if release_key not in RELEASES or seed not in (0, 1):
            raise AccessRefused("D0 release key or paired seed is unregistered")
        self.seed = seed
        self.release_key = release_key
        self.allowed_roles = ROLES
        self.portfolio_package = Path(portfolio_package).resolve(strict=True)
        self.repair_store = RoleOnlyStore(repair_manifest_path)
        self.audit_path = Path(audit_path).resolve(strict=True)
        self.audit = json.loads(self.audit_path.read_text())
        if (self.audit.get("schema") != "pcrl-r01-a-historical-score-source-audit-v1"
                or self.audit.get("status") != "ROLE_ONLY_HISTORICAL_SCORE_SOURCES_VERIFIED"):
            raise AccessRefused("D0 R01 source audit differs")
        self.bound = self.audit["d0"][release_key][f"s{seed}"]
        self.root = (self.portfolio_package / "shared/.private" /
                     f"{release_key}_score_bank_s{seed}_v1").resolve(strict=True)
        receipt_path = self.root / "BANK_ADMISSION_RECEIPT.json"
        self.receipt = json.loads(receipt_path.read_text())
        portfolio_manifest = self.portfolio_package / "DATA_ACCESS_MANIFEST.json"
        if (sha_file(portfolio_manifest) != self.repair_store.manifest["portfolio_manifest_sha256"]
                or self.receipt.get("schema") != "pcrl-portfolio-a-score-bank-admission-v1"
                or self.receipt.get("status") != "ROLE_ONLY_BANK_ADMITTED"
                or self.receipt.get("release_id") != RELEASES[release_key]
                or self.bound["release_id"] != RELEASES[release_key]
                or self.receipt.get("seed") != seed
                or self.receipt.get("preprocessing_only") is not True
                or self.receipt.get("data_access_manifest_sha256") != sha_file(portfolio_manifest)
                or sha_file(receipt_path) != self.bound["receipt_sha256"]
                or self.receipt["source_binding_sha256"] != self.bound["source_binding_sha256"]
                or self.receipt["source_binding"]["feature_names_sha256_newline_join"]
                   != self.repair_store.manifest["source_schema_sha256"]
                or self.receipt["source_json_sha256"]["decoder_file_sha256"] is not None):
            raise AccessRefused("D0 predecessor/repair source identity differs")
        payload = json.dumps(self.receipt["source_binding"], sort_keys=True,
                             separators=(",", ":"), ensure_ascii=True,
                             allow_nan=False).encode()
        if hashlib.sha256(payload).hexdigest() != self.bound["source_binding_sha256"]:
            raise AccessRefused("D0 source binding content differs")
        cache = Path(cache_root).resolve(strict=True)
        expected_keys = {"model.pt": "teacher_model_sha256", "fine.json": "fine_sha256",
                         "policy.json": "policy_sha256"}
        policy_parent = ("lra_v1/run/units" if release_key == "lra_seq21"
                         else "hcal_v1/admitted/units")
        policy_stem = {"hcal_local_d0": "U_LOCAL_i8o64_l0.1",
                       "hcal_seq12_d0": "U_SEQ-12_i8o64_l0.1",
                       "lra_seq21": "U_SEQ-21_i8o64_l0.1"}[release_key]
        exact_relative = {
            "model.pt": f"hcal_v1/admitted/rel__s{seed}__U/model.pt",
            "fine.json": f"hcal_v1/admitted/units/fine__s{seed}/fine.json",
            "policy.json": f"{policy_parent}/pol__s{seed}__{policy_stem}/policy.json",
        }
        for name, record in self.bound["source_paths_sha256"].items():
            if name not in expected_keys or record["path"] != exact_relative[name]:
                raise AccessRefused("D0 source relative identity differs")
            path = (cache / record["path"]).resolve(strict=True)
            if (not path.is_relative_to(cache)
                    or sha_file(path) != record["sha256"]
                    or record["sha256"] !=
                       self.receipt["source_json_sha256"][expected_keys[name]]):
                raise AccessRefused("D0 model, partition or policy bytes differ")
        self._banks: dict[int, dict] = {}
