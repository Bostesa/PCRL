"""Hash-bound admitted A role inputs for independent numerical replay.

This reader opens only the named FIT, ATT or INNER role. It never opens SEX.
The earlier R04 source receipt separately verified U0 fine-cell routing and
the T* public decoder; this module binds those admitted bytes to later runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .independent_a_source import (BANK_FIELDS, REFERENCE_FIELDS, TOKEN_FIELDS,
                                   bound_json, bound_npz, role_reference_path,
                                   sha_array, sha_file)


ADMITTED_ROLES = {"OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION"}


@dataclass(frozen=True)
class AdmittedARole:
    seed: int
    role: str
    row_id: np.ndarray
    unit: np.ndarray
    score1: np.ndarray
    score2: np.ndarray
    cell1: np.ndarray
    cell2: np.ndarray
    token1: np.ndarray
    token2: np.ndarray
    bank1: dict[str, np.ndarray]
    bank2: dict[str, np.ndarray]
    references: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    source_identity: dict[str, str]
    source_binding_sha256: str
    row_unit_pair_sha256: str


def _role_npz(manifest_path: Path, manifest: dict, role: str,
              kind: str, fields: set[str]) -> dict[str, np.ndarray]:
    key = f"{role}:{kind}"
    source = manifest_path.parent.parent / manifest["role_file_relative_path"][key]
    return bound_npz(source, manifest["role_file_sha256"][key], fields)


def read_admitted_a_role(*, seed: int, role: str, portfolio: Path,
                         manifest_path: Path, manifest_sha256: str,
                         source_audit_path: Path, source_audit_sha256: str,
                         labels: bool = False) -> AdmittedARole:
    if seed not in (0, 1) or role not in ADMITTED_ROLES:
        raise PermissionError("A role or seed outside admitted verification scope")
    if labels and role == "ATTACK_FIT_NEW":
        raise PermissionError("A attack-fit source loader cannot open task truth")
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = bound_json(manifest_path, manifest_sha256)
    audit = bound_json(Path(source_audit_path).resolve(strict=True), source_audit_sha256)
    if (manifest.get("schema") != "pcrl-utility-repair-role-admission-v1"
            or audit.get("status") != "READ_ONLY_HASH_AND_MEMBERSHIP_PASS"
            or audit.get("source_schema_sha256") != manifest.get("source_schema_sha256")):
        raise ValueError("A original role manifest/source audit differs")
    portfolio = Path(portfolio).resolve(strict=True)
    root = portfolio / f"shared/.private/tstar_score_bank_s{seed}_v1"
    recorded = audit["tstar_banks"][str(seed)]
    admission = bound_json(root / "BANK_ADMISSION_RECEIPT.json", recorded["receipt_sha256"])
    if (admission.get("status") != "ROLE_ONLY_BANK_ADMITTED"
            or admission.get("seed") != seed
            or admission.get("source_binding_sha256") != recorded["source_binding_sha256"]):
        raise ValueError("A source bank admission differs")
    banks = []
    for recipient in (1, 2):
        name = f"bank_r{recipient}.npz"
        expected = recorded["file_sha256"][name]
        if expected != admission["private_output_sha256"][name]:
            raise ValueError("A source bank file not bound to admission")
        banks.append(bound_npz(root / name, expected, BANK_FIELDS))
    token_name = f"tokens_{role}.npz"
    expected_token = recorded["file_sha256"][token_name]
    if expected_token != admission["private_output_sha256"][token_name]:
        raise ValueError("A role token file not bound to admission")
    tokens = bound_npz(root / token_name, expected_token, TOKEN_FIELDS)
    reference_path = role_reference_path(manifest_path, manifest, role, seed)
    reference_sha = manifest["role_file_sha256"][f"{role}:references_s{seed}"]
    if (reference_sha != admission["role_receipts"][role]["source_reference_sha256"]
            or sha_file(reference_path) != reference_sha):
        raise ValueError("A role U0/Ucal references differ")
    references = bound_npz(reference_path, reference_sha, REFERENCE_FIELDS)
    row, unit = tokens["row_id"], tokens["unit"]
    pair_sha = sha_array(np.column_stack((row, unit)))
    if (row.dtype != np.int64 or unit.dtype != np.int64
            or not np.all(np.diff(row) > 0)
            or pair_sha != admission["role_receipts"][role]["row_unit_pair_sha256"]
            or pair_sha != audit["roles"][role]["row_unit_pair_sha256"]
            or not np.array_equal(row, references["row_id"])
            or not np.array_equal(unit, references["unit"])):
        raise ValueError("A source role row and group order differs")
    for recipient, bank in enumerate(banks, 1):
        cell = tokens[f"cell_{recipient}"]
        old = tokens[f"token_{recipient}"]
        if (cell.dtype != np.int64 or old.dtype != np.int64
                or cell.shape != row.shape or old.shape != row.shape
                or np.any((cell < 0) | (cell >= len(bank["input_class"])))
                or not np.array_equal(old, bank["fine_to_hcal_token"][cell])
                or not np.array_equal(bank["token_class"][old],
                                       references[f"d_{recipient}"])):
            raise ValueError("A role fine cell, token or decision differs")
    task_labels = {}
    if labels:
        for task in ("income", "occupation"):
            record = _role_npz(manifest_path, manifest, role, task,
                               {"row_id", "unit", task})
            if (not np.array_equal(record["row_id"], row)
                    or not np.array_equal(record["unit"], unit)
                    or record[task].dtype != np.int64):
                raise ValueError("A task truth differs from source people")
            task_labels[task] = record[task]
    identity = {
        "tstar_bank_admission_receipt_sha256": recorded["receipt_sha256"],
        "tstar_role_tokens_file_sha256": expected_token,
        "reference_file_sha256": reference_sha,
    }
    return AdmittedARole(seed, role, row, unit, references["U0_1"],
                         references["U0_2"], tokens["cell_1"], tokens["cell_2"],
                         tokens["token_1"], tokens["token_2"], banks[0], banks[1],
                         references, task_labels, identity,
                         admission["source_binding_sha256"], pair_sha)


def verify_policy_source(policy, source: AdmittedARole) -> dict:
    """Verify that global saved code IDs reproduce the admitted T* bank."""
    if (policy.receipt["source_seed"] != source.seed
            or policy.receipt["source_binding_sha256"] != source.source_binding_sha256
            or (source.role == "OSF_DEFENSE_FIT" and
                policy.receipt["row_unit_pair_sha256"] != source.row_unit_pair_sha256)):
        raise ValueError("A selected policy differs from admitted source identity")
    for r, source_bank in ((1, source.bank1), (2, source.bank2)):
        old = policy.arrays[f"bank{r}_old_to_global"]
        global_probability = policy.arrays[f"bank{r}_probability"]
        if (len(old) != len(source_bank["probability"])
                or not np.array_equal(global_probability[old], source_bank["probability"])
                or not np.array_equal(policy.arrays[f"bank{r}_decision"][old],
                                       source_bank["token_class"])):
            raise ValueError("A saved global IDs do not decode source bank exactly")
        actions = policy.actions1 if r == 1 else policy.actions2
        if (len(actions) != len(source_bank["input_class"])
                or any(actions[cell][0] != int(old[int(token)])
                       for cell, token in enumerate(source_bank["fine_to_hcal_token"]))):
            raise ValueError("A action identity differs from source fine-to-token map")
    return {"source_bank_global_mapping_exact": True,
            "fine_cell_identity_action_exact": True}
