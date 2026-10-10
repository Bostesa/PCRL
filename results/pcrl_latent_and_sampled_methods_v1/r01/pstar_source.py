"""Fail-closed P* public score wire from admitted ATT/INNER role-only exports.

P* is the historical JOINT .1 H-CLASS-TEMP release. The private qpc fine
cell selects a coarse token; recipients see only that token, its fixed decoded
probability vector and decision. This reader never opens FIT, labels, X,
assessment, mixed source arrays or a historical all-role loader.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .role_store import AccessRefused, RoleOnlyStore, sha_array, sha_file


RELEASE_ID = "U|JOINT|i8o64|l0.1|H-CLASS-TEMP"
ROLES = frozenset({"ATTACK_FIT_NEW", "INNER_SELECTION"})
BANK_FIELDS = frozenset({"probability", "token_class", "token_id",
                         "input_class", "fine_to_hcal_token"})
TOKEN_FIELDS = frozenset({"row_id", "unit", "cell_1", "cell_2", "token_1", "token_2"})


def _archive(path: Path, fields: frozenset[str]) -> dict:
    with ZipFile(path) as zip_file:
        if {entry.filename for entry in zip_file.infolist()} != {name + ".npy" for name in fields}:
            raise AccessRefused("P* archive member schema differs")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: np.array(arrays[name], copy=True) for name in fields}


class PStarSourceStore:
    """Pinned historical P* bank and role-only deterministic public wire."""

    def __init__(self, portfolio_package: str | Path,
                 repair_manifest_path: str | Path, audit_path: str | Path,
                 cache_root: str | Path, seed: int):
        if seed not in (0, 1):
            raise AccessRefused("P* seed must be a sealed paired seed")
        self.seed = seed
        self.allowed_roles = ROLES
        self.portfolio_package = Path(portfolio_package).resolve(strict=True)
        self.repair_store = RoleOnlyStore(repair_manifest_path)
        self.audit_path = Path(audit_path).resolve(strict=True)
        self.audit = json.loads(self.audit_path.read_text())
        if (self.audit.get("schema") != "pcrl-r01-a-historical-score-source-audit-v1"
                or self.audit.get("status") != "ROLE_ONLY_HISTORICAL_SCORE_SOURCES_VERIFIED"
                or self.audit.get("pstar_release_id") != RELEASE_ID):
            raise AccessRefused("P* independent R01 source audit differs")
        self.bound = self.audit["pstar"][f"s{seed}"]
        self.root = (self.portfolio_package / "shared/.private" /
                     f"pstar_score_bank_s{seed}_v1").resolve(strict=True)
        receipt_path = self.root / "BANK_ADMISSION_RECEIPT.json"
        self.receipt = json.loads(receipt_path.read_text())
        portfolio_manifest = self.portfolio_package / "DATA_ACCESS_MANIFEST.json"
        if (sha_file(portfolio_manifest) != self.repair_store.manifest["portfolio_manifest_sha256"]
                or self.receipt.get("schema") != "pcrl-portfolio-a-score-bank-admission-v1"
                or self.receipt.get("status") != "ROLE_ONLY_BANK_ADMITTED"
                or self.receipt.get("release_id") != RELEASE_ID
                or self.receipt.get("seed") != seed
                or self.receipt.get("preprocessing_only") is not True
                or self.receipt.get("data_access_manifest_sha256") != sha_file(portfolio_manifest)
                or sha_file(receipt_path) != self.bound["receipt_sha256"]
                or self.receipt["source_binding_sha256"] != self.bound["source_binding_sha256"]
                or self.receipt["source_binding"]["feature_names_sha256_newline_join"]
                   != self.repair_store.manifest["source_schema_sha256"]):
            raise AccessRefused("P* predecessor/repair receipt identity differs")
        payload = json.dumps(self.receipt["source_binding"], sort_keys=True,
                             separators=(",", ":"), ensure_ascii=True,
                             allow_nan=False).encode()
        if hashlib.sha256(payload).hexdigest() != self.bound["source_binding_sha256"]:
            raise AccessRefused("P* source binding content differs")
        cache = Path(cache_root).resolve(strict=True)
        source_paths = {
            "model.pt": cache / f"hcal_v1/admitted/rel__s{seed}__U/model.pt",
            "fine.json": cache / f"hcal_v1/admitted/units/fine__s{seed}/fine.json",
            "policy.json": cache / f"hcal_v1/admitted/units/pol__s{seed}__U_JOINT_i8o64_l0.1/policy.json",
            "decoder.json": cache / f"hcal_v1/package/decoder__s{seed}__U_JOINT_i8o64_l0.1_H-CLASS-TEMP.json",
        }
        expected_keys = {"model.pt": "teacher_model_sha256", "fine.json": "fine_sha256",
                         "policy.json": "policy_sha256", "decoder.json": "decoder_file_sha256"}
        for name, path in source_paths.items():
            expected = self.bound["source_paths_sha256"][name]
            if (path.resolve(strict=True) != Path(expected["path"]).resolve(strict=True)
                    or sha_file(path) != expected["sha256"]
                    or expected["sha256"] !=
                       self.receipt["source_json_sha256"][expected_keys[name]]):
                raise AccessRefused("P* model/partition/policy/decoder bytes differ")
        self._banks: dict[int, dict] = {}

    @property
    def source_binding_sha256(self) -> str:
        return self.bound["source_binding_sha256"]

    def _verified_path(self, filename: str, expected: str) -> Path:
        if filename not in self.receipt["private_output_sha256"]:
            raise AccessRefused("P* file lacks admitted hash")
        path = (self.root / filename).resolve(strict=True)
        if (path.parent != self.root or path.suffix != ".npz"
                or sha_file(path) != self.receipt["private_output_sha256"][filename]
                or sha_file(path) != expected):
            raise AccessRefused("P* role-only archive path or bytes differ")
        return path

    def read_bank(self, recipient: int) -> dict:
        """Read one globally fixed decoded table; no per-person values."""
        if recipient not in (1, 2):
            raise AccessRefused("P* recipient must be income 1 or occupation 2")
        if recipient in self._banks:
            return {k: np.array(v, copy=True) for k, v in self._banks[recipient].items()}
        source = self.bound["banks"][f"r{recipient}"]
        bank = _archive(self._verified_path(f"bank_r{recipient}.npz", source["sha256"]),
                        BANK_FIELDS)
        p, decision, token_id = bank["probability"], bank["token_class"], bank["token_id"]
        input_class, mapping = bank["input_class"], bank["fine_to_hcal_token"]
        width = 2 if recipient == 1 else 6
        bound = self.receipt["source_binding"]["recipients"][f"r{recipient}"]
        if (p.dtype != np.float64 or p.shape != (source["tokens"], width)
                or any(x.dtype != np.int64 or x.ndim != 1 for x in
                       (decision, token_id, input_class, mapping))
                or decision.shape != token_id.shape or mapping.shape != input_class.shape
                or not np.array_equal(token_id, np.arange(len(token_id)))
                or not np.array_equal(bound["class_order"], list(range(width)))
                or not np.isfinite(p).all() or np.any(p < 0)
                or np.max(np.abs(p.sum(axis=1) - 1), initial=0) > 1e-8
                or np.any(np.sort(p, axis=1)[:, -1] <= np.sort(p, axis=1)[:, -2])
                or not np.array_equal(np.argmax(p, axis=1), decision)
                or np.any((mapping < 0) | (mapping >= len(token_id)))
                or not np.array_equal(decision[mapping], input_class)
                or len({(row.tobytes(), int(d)) for row, d in zip(p, decision)})
                   != len(token_id)
                or sha_array(p) != bound["bank_probability_sha256"]
                or sha_array(decision) != bound["token_class_sha256"]
                or sha_array(input_class) != bound["input_class_sha256"]
                or sha_array(mapping) != bound["fine_to_hcal_token_sha256"]):
            raise AccessRefused("P* calibrated bank, class or token identity differs")
        bank["canonical_action_id"] = token_id.copy()
        self._banks[recipient] = bank
        return {k: np.array(v, copy=True) for k, v in bank.items()}

    def read_role_cells(self, role: str) -> dict:
        """Read only ATT or INNER private fine cells and public coarse tokens."""
        if role not in self.allowed_roles:
            raise AccessRefused("P* person role is not admitted")
        source = self.bound["roles"][role]
        rec = _archive(self._verified_path(f"tokens_{role}.npz", source["sha256"]),
                       TOKEN_FIELDS)
        meta = self.repair_store.manifest["roles"][role]
        reference_key = f"{role}:references_s{self.seed}"
        reference_path = (self.repair_store.package_root /
                          self.repair_store.manifest["role_file_relative_path"][reference_key]
                          ).resolve(strict=True)
        ids, units = rec["row_id"], rec["unit"]
        if (any(rec[k].dtype != np.int64 or rec[k].shape != (meta["rows"],)
                for k in TOKEN_FIELDS)
                or not np.all(np.diff(ids) > 0)
                or len(np.unique(units)) != meta["groups"]
                or sha_array(np.sort(ids)) != meta["row_id_sha256"]
                or sha_array(np.unique(units)) != meta["group_id_sha256"]
                or sha_array(np.column_stack((ids, units))) != meta["row_unit_pair_sha256"]
                or source["row_unit_pair_sha256"] != meta["row_unit_pair_sha256"]
                or source["source_reference_sha256"] !=
                   self.repair_store.manifest["role_file_sha256"][reference_key]
                or sha_file(reference_path) != source["source_reference_sha256"]):
            raise AccessRefused("P* role/reference person identity differs")
        for recipient in (1, 2):
            bank = self.read_bank(recipient)
            cell, token = rec[f"cell_{recipient}"], rec[f"token_{recipient}"]
            bound = self.receipt["role_receipts"][role]
            if (np.any((cell < 0) | (cell >= len(bank["input_class"])))
                    or np.any((token < 0) | (token >= len(bank["token_id"])))
                    or not np.array_equal(bank["fine_to_hcal_token"][cell], token)
                    or sha_array(cell) != bound[f"cell_{recipient}_sha256"]
                    or sha_array(token) != bound[f"token_{recipient}_sha256"]):
                raise AccessRefused("P* private cell/public token mapping differs")
        return rec

    def read_role_wire(self, role: str, recipient: int) -> dict:
        """Return exactly one recipient's deterministic public code, q, decision."""
        if recipient not in (1, 2):
            raise AccessRefused("P* recipient must be income 1 or occupation 2")
        cells = self.read_role_cells(role)
        bank = self.read_bank(recipient)
        code = cells[f"token_{recipient}"]
        return {"row_id": cells["row_id"], "unit": cells["unit"],
                "code": code, "probabilities": bank["probability"][code].copy(),
                "decision": bank["token_class"][code].copy()}
