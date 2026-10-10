"""Fail-closed T* source cells and fixed output bank on admitted roles only.

This adapter never calls the historical all-role loader. Its per-person reads
are limited to the three hash-admitted T* token exports for FIT, ATT and INNER.
There is no task or SEX label access. The `cell_r` values are private channel
inputs; `token_r` is the inherited deterministic coarse output action.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .role_store import AccessRefused, sha_array, sha_file


ROLES = frozenset({"OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION"})
BANK_FIELDS = frozenset({"probability", "token_class", "token_id", "input_class", "fine_to_hcal_token"})
TOKEN_FIELDS = frozenset({"row_id", "unit", "cell_1", "cell_2", "token_1", "token_2"})


def _read_exact_archive(path: Path, fields: frozenset[str]) -> dict:
    with ZipFile(path) as archive:
        if {entry.filename for entry in archive.infolist()} != {name + ".npy" for name in fields}:
            raise AccessRefused("T* archive member set differs")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in fields}


class TStarSourceStore:
    """Hash-bound existing fine cells and canonical fixed actions for one seed.

    `portfolio_package` is the predecessor `results/pcrl_method_portfolio_v1`
    directory. `repair_manifest_path` is its successor repair package's
    `m1/DATA_ACCESS_MANIFEST.json`. `audit_path` is R01 `SOURCE_AUDIT.json`.
    The caller owns the R00 stage/queue gate before any per-person read.
    """

    def __init__(self, portfolio_package: str | Path,
                 repair_manifest_path: str | Path, audit_path: str | Path,
                 seed: int):
        if seed not in (0, 1):
            raise AccessRefused("T* seed must be a sealed paired seed")
        self.seed = seed
        self.portfolio_package = Path(portfolio_package).resolve(strict=True)
        self.repair_manifest_path = Path(repair_manifest_path).resolve(strict=True)
        self.audit_path = Path(audit_path).resolve(strict=True)
        self.repair = json.loads(self.repair_manifest_path.read_text())
        self.audit = json.loads(self.audit_path.read_text())
        if (self.repair.get("schema") != "pcrl-utility-repair-role-admission-v1"
                or self.audit.get("schema") != "pcrl-new-campaign-r01-source-audit-v1"
                or self.audit.get("status") != "READ_ONLY_HASH_AND_MEMBERSHIP_PASS"
                or self.audit["source_schema_sha256"] != self.repair["source_schema_sha256"]):
            raise AccessRefused("source admission manifest differs")
        portfolio_manifest = self.portfolio_package / "DATA_ACCESS_MANIFEST.json"
        if sha_file(portfolio_manifest) != self.repair["portfolio_manifest_sha256"]:
            raise AccessRefused("portfolio/repair role admissions differ")
        self.root = (self.portfolio_package / f"shared/.private/tstar_score_bank_s{seed}_v1").resolve(strict=True)
        receipt_path = self.root / "BANK_ADMISSION_RECEIPT.json"
        self.receipt = json.loads(receipt_path.read_text())
        if (self.receipt.get("schema") != "pcrl-portfolio-a-score-bank-admission-v1"
                or self.receipt.get("status") != "ROLE_ONLY_BANK_ADMITTED"
                or self.receipt.get("seed") != seed
                or self.receipt.get("release_id") != "U|FINE-TASK|i8o64|H-GLOBAL-TEMP"
                or self.receipt.get("preprocessing_only") is not True
                or self.receipt.get("data_access_manifest_sha256") != sha_file(portfolio_manifest)
                or sha_file(receipt_path) != self.audit["tstar_banks"][str(seed)]["receipt_sha256"]
                or self.receipt["source_binding"]["feature_names_sha256_newline_join"]
                   != self.repair["source_schema_sha256"]
                or self.receipt["source_json_sha256"]["teacher_model_sha256"]
                   != self.audit["u0_models"][str(seed)]["components_sha256"]["model.pt"]):
            raise AccessRefused("T* source receipt differs from audited source")
        binding = json.dumps(self.receipt["source_binding"], sort_keys=True,
                             separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
        if (hashlib.sha256(binding).hexdigest() != self.receipt["source_binding_sha256"]
                or self.receipt["source_binding_sha256"]
                   != self.audit["tstar_banks"][str(seed)]["source_binding_sha256"]):
            raise AccessRefused("T* source binding differs")
        self._banks: dict[int, dict] = {}

    @property
    def source_binding_sha256(self) -> str:
        return self.receipt["source_binding_sha256"]

    def _verified_path(self, filename: str) -> Path:
        if filename not in self.receipt["private_output_sha256"]:
            raise AccessRefused("T* file has no admitted byte hash")
        path = (self.root / filename).resolve(strict=True)
        if path.parent != self.root or path.suffix != ".npz":
            raise AccessRefused("T* path escapes admitted role-only root")
        actual = sha_file(path)
        if (actual != self.receipt["private_output_sha256"][filename]
                or actual != self.audit["tstar_banks"][str(self.seed)]["file_sha256"][filename]):
            raise AccessRefused("T* file byte hash differs")
        return path

    def read_bank(self, recipient: int) -> dict:
        """Return fixed canonical action table; no per-person values are read."""
        if recipient not in (1, 2):
            raise AccessRefused("recipient must be income 1 or occupation 2")
        if recipient in self._banks:
            return {name: np.array(value, copy=True) for name, value in self._banks[recipient].items()}
        values = _read_exact_archive(self._verified_path(f"bank_r{recipient}.npz"), BANK_FIELDS)
        probability = values["probability"]
        token_class = values["token_class"]
        token_id = values["token_id"]
        input_class = values["input_class"]
        mapping = values["fine_to_hcal_token"]
        width = 2 if recipient == 1 else 6
        expected = self.receipt["source_binding"]["recipients"][f"r{recipient}"]
        if (probability.dtype != np.float64 or probability.shape != (len(token_id), width)
                or any(array.dtype != np.int64 or array.ndim != 1
                       for array in (token_class, token_id, input_class, mapping))
                or token_class.shape != token_id.shape
                or mapping.shape != input_class.shape
                or not np.array_equal(token_id, np.arange(len(token_id)))
                or not np.array_equal(expected["class_order"], list(range(width)))
                or not np.isfinite(probability).all() or np.any(probability < 0)
                or np.max(np.abs(probability.sum(axis=1) - 1), initial=0) > 1e-8
                or not np.array_equal(np.argmax(probability, axis=1), token_class)
                or np.any((mapping < 0) | (mapping >= len(token_id)))
                or not np.array_equal(input_class, token_class[mapping])):
            raise AccessRefused("T* bank/action/class contract differs")
        for field, hash_key in (("probability", "bank_probability_sha256"),
                                ("token_class", "token_class_sha256"),
                                ("input_class", "input_class_sha256"),
                                ("fine_to_hcal_token", "fine_to_hcal_token_sha256")):
            if sha_array(values[field]) != expected[hash_key]:
                raise AccessRefused("T* bound bank array hash differs")
        # Distinct original IDs are exactly the distinct public output symbols.
        keys = {(row.tobytes(), int(token_class[index]))
                for index, row in enumerate(probability)}
        if len(keys) != len(token_id):
            raise AccessRefused("T* exact output dedup result differs from R01 audit")
        if len(token_id) != (16 if recipient == 1 else 321):
            raise AccessRefused("T* audited action count differs")
        values["canonical_action_id"] = token_id.copy()
        self._banks[recipient] = values
        return {name: np.array(value, copy=True) for name, value in values.items()}

    def read_role_cells(self, role: str) -> dict:
        """Return only admitted role's private fine cells and source coarse tokens."""
        if role not in ROLES:
            raise AccessRefused("T* per-person role is not admitted")
        values = _read_exact_archive(self._verified_path(f"tokens_{role}.npz"), TOKEN_FIELDS)
        meta = self.repair["roles"][role]
        source = self.receipt["role_receipts"][role]
        ids, groups = values["row_id"], values["unit"]
        if (ids.dtype != np.int64 or groups.dtype != np.int64
                or ids.shape != groups.shape or ids.shape != (meta["rows"],)
                or source["rows"] != len(ids)
                or not np.all(np.diff(ids) > 0)
                or len(np.unique(groups)) != meta["groups"]
                or sha_array(np.sort(ids)) != meta["row_id_sha256"]
                or sha_array(np.unique(groups)) != meta["group_id_sha256"]
                or sha_array(np.column_stack((ids, groups))) != meta["row_unit_pair_sha256"]
                or source["row_unit_pair_sha256"] != meta["row_unit_pair_sha256"]):
            raise AccessRefused("T* role row/group identity differs")
        reference_key = f"{role}:references_s{self.seed}"
        reference_path = (self.repair_manifest_path.parent.parent /
                          self.repair["role_file_relative_path"][reference_key]).resolve(strict=True)
        if (sha_file(reference_path) != self.repair["role_file_sha256"][reference_key]
                or source["source_reference_sha256"] != self.repair["role_file_sha256"][reference_key]):
            raise AccessRefused("T* cells are not bound to same admitted reference")
        for recipient in (1, 2):
            bank = self.read_bank(recipient)
            cell = values[f"cell_{recipient}"]
            token = values[f"token_{recipient}"]
            if (cell.dtype != np.int64 or token.dtype != np.int64
                    or cell.shape != ids.shape or token.shape != ids.shape
                    or np.any((cell < 0) | (cell >= len(bank["input_class"])))
                    or np.any((token < 0) | (token >= len(bank["token_id"])))
                    or not np.array_equal(bank["fine_to_hcal_token"][cell], token)
                    or sha_array(cell) != source[f"cell_{recipient}_sha256"]
                    or sha_array(token) != source[f"token_{recipient}_sha256"]):
                raise AccessRefused("T* role cell/token array differs")
        return values
