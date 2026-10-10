"""Read and encode actual deterministic score-reference releases.

The private row/group IDs only align people. Every public field in the R12
continuous or finite wire is represented in the recipient and pair views.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_array
from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import (
    Field, WireBatch, pair_batch, prepare_views,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.reference_materialize import (
    reconstruct_reference_wires,
)
from .wire import canonical_bank_sha256


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
RECEIPT_KEYS = frozenset({
    "schema", "status", "arm_id", "source_seed", "role", "release_id",
    "rows", "groups", "row_unit_pair_sha256", "source_binding_sha256",
    "source_identity", "interface_kind", "bank_sha256_by_recipient",
    "codes_file_sha256", "wire_sha256_by_recipient",
})
SOURCE_KEYS = frozenset({"role_source_file_sha256", "reference_file_sha256",
                         "bank_receipt_sha256", "source_audit_sha256"})
ARRAYS = ("row_id", "unit", "prob_r1", "prob_r2", "decision_r1",
          "decision_r2", "code_r1", "code_r2")
FIXED_MAP_ARRAYS = ("bank_r1", "bank_r2", "fine_to_code_r1", "fine_to_code_r2")
FIXED_MAP_RECEIPT_KEYS = frozenset({
    "schema", "status", "arm_id", "seed", "release_id", "source_kind",
    "source_binding_sha256", "source_audit_sha256",
    "original_bank_receipt_sha256", "original_bank_file_sha256_by_recipient",
    "array_sha256", "array_shape", "npz_sha256",
})
WIRE_SCHEMA = "SCV1-reference-score-release-v1"
METADATA_KEYS = frozenset({"model_id", "source_binding_sha256",
                           "bank_sha256", "code_length_bits"})


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _wire_sha(wires: tuple[dict, ...]) -> str:
    digest = hashlib.sha256()
    for wire in wires:
        digest.update(json.dumps(wire, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode() + b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class ReferenceRoleRelease:
    receipt: dict
    receipt_sha256: str
    row_id: np.ndarray
    unit: np.ndarray
    wires1: tuple[dict, ...]
    wires2: tuple[dict, ...]


def load_public_fixed_map(receipt_path: str | Path, archive_path: str | Path, *,
                          expected_receipt_sha256: str,
                          expected_archive_sha256: str,
                          arm_id: str, seed: int) -> tuple[dict, dict]:
    """Load only R01's label-free full bank and fine-to-code public map."""
    supplied_receipt, supplied_archive = Path(receipt_path), Path(archive_path)
    if supplied_receipt.is_symlink() or supplied_archive.is_symlink():
        raise PermissionError("public fixed-map source cannot be a symlink")
    receipt_path = supplied_receipt.resolve(strict=True)
    archive_path = supplied_archive.resolve(strict=True)
    if (type(expected_receipt_sha256) is not str or
            type(expected_archive_sha256) is not str or
            HEX64.fullmatch(expected_receipt_sha256) is None or
            HEX64.fullmatch(expected_archive_sha256) is None or
            _sha(receipt_path) != expected_receipt_sha256 or
            _sha(archive_path) != expected_archive_sha256):
        raise ValueError("public fixed-map source bytes differ")
    receipt = json.loads(receipt_path.read_text())
    if (type(receipt) is not dict or receipt.keys() != FIXED_MAP_RECEIPT_KEYS or
            receipt.get("schema") != "pcrl-r01-public-fixed-fine-map-v1" or
            receipt.get("status") != "LABEL_FREE_FIXED_PUBLIC_SOURCE" or
            receipt.get("arm_id") != arm_id or receipt.get("seed") != seed or
            receipt.get("source_kind") != "historical_finite_reference" or
            receipt.get("npz_sha256") != expected_archive_sha256 or
            type(receipt.get("release_id")) is not str or
            not receipt["release_id"] or
            any(type(receipt.get(key)) is not str or
                HEX64.fullmatch(receipt[key]) is None for key in (
                    "source_binding_sha256", "source_audit_sha256",
                    "original_bank_receipt_sha256")) or
            type(receipt.get("original_bank_file_sha256_by_recipient")) is not dict or
            receipt["original_bank_file_sha256_by_recipient"].keys() != {"r1", "r2"} or
            any(type(value) is not str or HEX64.fullmatch(value) is None
                for value in receipt["original_bank_file_sha256_by_recipient"].values()) or
            type(receipt.get("array_sha256")) is not dict or
            type(receipt.get("array_shape")) is not dict or
            set(receipt["array_sha256"]) != set(FIXED_MAP_ARRAYS) or
            set(receipt["array_shape"]) != set(FIXED_MAP_ARRAYS)):
        raise ValueError("public fixed-map receipt identity differs")
    with ZipFile(archive_path) as archive:
        if (len(archive.infolist()) != len(FIXED_MAP_ARRAYS) or
                {item.filename for item in archive.infolist()} !=
                {f"{name}.npy" for name in FIXED_MAP_ARRAYS}):
            raise ValueError("public fixed-map NPZ member set differs")
    with np.load(archive_path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True)
                  for name in FIXED_MAP_ARRAYS}
    for name, value in values.items():
        if (sha_array(value) != receipt["array_sha256"][name] or
                list(value.shape) != receipt["array_shape"][name]):
            raise ValueError("public fixed-map array content differs")
    for recipient in (1, 2):
        bank = values[f"bank_r{recipient}"]
        mapping = values[f"fine_to_code_r{recipient}"]
        if (bank.dtype != np.float64 or bank.ndim != 2 or
                bank.shape[1] != (2 if recipient == 1 else 6) or
                not np.isfinite(bank).all() or np.any(bank < 0) or
                not np.allclose(bank.sum(axis=1), 1, rtol=0, atol=1e-8) or
                mapping.dtype != np.int64 or
                mapping.shape != (64 if recipient == 1 else 641,) or
                np.any((mapping < 0) | (mapping >= len(bank))) or
                np.any(np.sort(bank, axis=1)[:, -1] <=
                       np.sort(bank, axis=1)[:, -2])):
            raise ValueError("public fixed-map bank/code semantics differ")
    return values, receipt


def load_reference_role_package(directory: str | Path, *, receipt_sha256: str,
                                full_bank1: np.ndarray | None = None,
                                full_bank2: np.ndarray | None = None) -> ReferenceRoleRelease:
    """Restore R12's exact scored wires from a write-once role package."""
    supplied = Path(directory)
    if supplied.is_symlink():
        raise PermissionError("reference directory cannot be a symlink")
    root = supplied.resolve(strict=True)
    if (".private" not in root.parts or not root.is_dir() or
            stat.S_IMODE(root.stat().st_mode) & 0o077):
        raise PermissionError("reference package must be owner-only")
    receipt_path = root / "REFERENCE_RELEASE_RECEIPT.json"
    data_path = root / "reference_release.npz"
    for path in (receipt_path, data_path):
        if (path.is_symlink() or not path.is_file() or
                stat.S_IMODE(path.stat().st_mode) & 0o077):
            raise PermissionError("reference artifact is not a private regular file")
    if (type(receipt_sha256) is not str or HEX64.fullmatch(receipt_sha256) is None or
            _sha(receipt_path) != receipt_sha256):
        raise ValueError("reference receipt SHA differs")
    receipt = json.loads(receipt_path.read_text())
    if (type(receipt) is not dict or receipt.keys() != RECEIPT_KEYS or
            receipt["schema"] != "SCV1-private-reference-release-v1" or
            receipt["status"] != "FROZEN_DETERMINISTIC" or
            receipt["role"] not in ("ATTACK_FIT_NEW", "INNER_SELECTION") or
            type(receipt["source_seed"]) is not int or receipt["source_seed"] not in (0, 1) or
            receipt["interface_kind"] not in ("continuous", "finite") or
            type(receipt["rows"]) is not int or receipt["rows"] < 1 or
            type(receipt["groups"]) is not int or receipt["groups"] < 1 or
            type(receipt["source_identity"]) is not dict or
            receipt["source_identity"].keys() != SOURCE_KEYS or
            type(receipt["bank_sha256_by_recipient"]) is not dict or
            receipt["bank_sha256_by_recipient"].keys() != {"r1", "r2"}):
        raise ValueError("reference receipt schema or identity differs")
    for key in ("row_unit_pair_sha256", "source_binding_sha256",
                "codes_file_sha256"):
        if type(receipt[key]) is not str or HEX64.fullmatch(receipt[key]) is None:
            raise ValueError("reference receipt SHA field differs")
    if (any(value is not None and (type(value) is not str or
            HEX64.fullmatch(value) is None)
            for value in (*receipt["source_identity"].values(),
                          *receipt["bank_sha256_by_recipient"].values())) or
            _sha(data_path) != receipt["codes_file_sha256"]):
        raise ValueError("reference source or NPZ SHA differs")
    with ZipFile(data_path) as archive:
        if (len(archive.infolist()) != len(ARRAYS) or
                {item.filename for item in archive.infolist()} !=
                {f"{name}.npy" for name in ARRAYS}):
            raise ValueError("reference NPZ member set differs")
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in ARRAYS}
    row, unit = arrays["row_id"], arrays["unit"]
    n = receipt["rows"]
    if (row.dtype != np.int64 or unit.dtype != np.int64 or
            row.shape != (n,) or unit.shape != (n,) or
            np.any(np.diff(row) <= 0) or len(np.unique(unit)) != receipt["groups"] or
            sha_array(np.column_stack((row, unit))) != receipt["row_unit_pair_sha256"]):
        raise ValueError("reference ordered row/group identity differs")
    finite = receipt["interface_kind"] == "finite"
    banks = (full_bank1, full_bank2)
    if finite != all(bank is not None for bank in banks):
        raise ValueError("reference finite bank availability differs")
    wires = {}
    for recipient, bank in ((1, full_bank1), (2, full_bank2)):
        label = f"r{recipient}"
        probability = arrays[f"prob_{label}"]
        decision = arrays[f"decision_{label}"]
        code = arrays[f"code_{label}"]
        if (probability.dtype != np.float64 or
                probability.shape != (n, 2 if recipient == 1 else 6) or
                decision.dtype != np.int64 or decision.shape != (n,) or
                code.dtype != np.int64 or code.shape != (n,)):
            raise ValueError("reference public vector/code array differs")
        if finite:
            bank = np.asarray(bank)
            if (bank.dtype != np.float64 or bank.ndim != 2 or
                    bank.shape[1] != probability.shape[1] or
                    sha_array(bank) != receipt["bank_sha256_by_recipient"][label] or
                    np.any((code < 0) | (code >= len(bank))) or
                    bank[code].tobytes() != probability.tobytes()):
                raise ValueError("reference emitted code differs from full bank")
        elif receipt["bank_sha256_by_recipient"][label] is not None:
            raise ValueError("continuous reference claims a finite bank")
        wires[label] = reconstruct_reference_wires(
            probability, decision, code, recipient=recipient,
            interface_kind=receipt["interface_kind"],
            release_id=receipt["release_id"],
            source_binding_sha256=receipt["source_binding_sha256"],
            bank_sha256=receipt["bank_sha256_by_recipient"][label],
            bank_size=None if bank is None else len(bank))
    if receipt["wire_sha256_by_recipient"] != {
            label: _wire_sha(wires[label]) for label in ("r1", "r2")}:
        raise ValueError("reference reconstructed wire SHA differs")
    return ReferenceRoleRelease(receipt, receipt_sha256, row, unit,
                                wires["r1"], wires["r2"])


def reference_recipient_batch(package: ReferenceRoleRelease, recipient: int) -> WireBatch:
    """Encode all actually transmitted deterministic fields, including metadata."""
    if recipient not in (1, 2):
        raise ValueError("reference recipient differs")
    wires = package.wires1 if recipient == 1 else package.wires2
    kind = package.receipt["interface_kind"]
    bank_sha = package.receipt["bank_sha256_by_recipient"][f"r{recipient}"]
    bits = wires[0]["public_metadata"]["code_length_bits"]
    if (any(set(wire) != ({"schema", "recipient", "probability", "decision",
                            "release_id", "public_metadata"} |
                           ({"code_id"} if kind == "finite" else set())) or
            set(wire["public_metadata"]) != METADATA_KEYS or
            wire["schema"] != WIRE_SCHEMA or type(wire["recipient"]) is not int or
            wire["recipient"] != recipient or
            wire["release_id"] != package.receipt["release_id"] or
            wire["public_metadata"] != wires[0]["public_metadata"]
            for wire in wires)):
        raise ValueError("reference wire fields differ from exact public contract")
    metadata = wires[0]["public_metadata"]
    if (metadata != {"model_id": package.receipt["release_id"],
                     "source_binding_sha256": package.receipt["source_binding_sha256"],
                     "bank_sha256": bank_sha, "code_length_bits": bits} or
            (kind == "finite" and type(bits) is not int) or
            (kind == "continuous" and bits is not None)):
        raise ValueError("reference public metadata differs from frozen receipt")
    n = len(wires)
    probability = np.asarray([wire["probability"] for wire in wires], np.float64)
    decision = np.asarray([wire["decision"] for wire in wires], np.int64)
    fields = {
        "schema": Field("category", np.asarray([WIRE_SCHEMA] * n)),
        "recipient": Field("numeric", np.full(n, recipient, np.float64)),
        "probability": Field("probability", probability),
        "decision": Field("decision", decision),
        "release_id": Field("category", np.asarray([package.receipt["release_id"]] * n)),
    }
    if kind == "finite":
        code = np.asarray([wire["code_id"] for wire in wires], np.int64)
        fields["code_id"] = Field("category", code)
        fields["code_bits"] = Field("bits", np.asarray([
            [(int(value) >> j) & 1 for j in range(max(1, bits))]
            for value in code], np.float64))
    fields.update({
        "public_metadata.model_id": Field("metadata_category",
                                          np.asarray([metadata["model_id"]] * n)),
        "public_metadata.source_binding_sha256": Field(
            "metadata_category", np.asarray([metadata["source_binding_sha256"]] * n)),
        "public_metadata.bank_sha256": Field(
            "metadata_category", np.asarray([bank_sha or "<NULL>"] * n)),
        "public_metadata.code_length_bits": Field(
            "metadata" if bits is not None else "metadata_category",
            np.asarray([bits if bits is not None else "<NULL>"] * n)),
        "public_metadata.canonical_json": Field(
            "metadata_category", np.asarray([
                json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                           allow_nan=False)] * n)),
    })
    return WireBatch(package.row_id, package.unit, fields,
                     ("code_id",) if kind == "finite" else ())


def prepare_reference_three_views(att: ReferenceRoleRelease,
                                  inner: ReferenceRoleRelease) -> dict:
    """Prepare local and aligned pair views, never using a private source cell."""
    if (att.receipt["role"] != "ATTACK_FIT_NEW" or
            inner.receipt["role"] != "INNER_SELECTION" or
            att.receipt["arm_id"] != inner.receipt["arm_id"] or
            att.receipt["source_seed"] != inner.receipt["source_seed"] or
            att.receipt["release_id"] != inner.receipt["release_id"] or
            att.receipt["source_binding_sha256"] !=
            inner.receipt["source_binding_sha256"] or
            att.receipt["interface_kind"] != inner.receipt["interface_kind"] or
            att.receipt["bank_sha256_by_recipient"] !=
            inner.receipt["bank_sha256_by_recipient"]):
        raise ValueError("reference ATT/INNER release identity differs")
    prepared = {}
    for recipient, name in ((1, "r1"), (2, "r2")):
        prepared[name] = prepare_views(reference_recipient_batch(att, recipient),
                                       reference_recipient_batch(inner, recipient))
    prepared["pair"] = prepare_views(
        pair_batch(reference_recipient_batch(att, 1),
                   reference_recipient_batch(att, 2)),
        pair_batch(reference_recipient_batch(inner, 1),
                   reference_recipient_batch(inner, 2)))
    return prepared


def validate_q_reference_banks(table, *, bank1: np.ndarray, bank2: np.ndarray,
                               att: ReferenceRoleRelease) -> None:
    """Prove a fixed-map Q table uses the full banks behind actual code wires."""
    from .q_aware import table_content_sha256
    if (table_content_sha256(table) != table.table_sha256 or
            att.receipt["interface_kind"] != "finite" or
            table.identity["release_id"] != att.receipt["release_id"] or
            table.identity["source_binding_sha256"] !=
            att.receipt["source_binding_sha256"] or
            table.identity["bank1_sha256"] != canonical_bank_sha256(bank1) or
            table.identity["bank2_sha256"] != canonical_bank_sha256(bank2) or
            att.receipt["bank_sha256_by_recipient"] != {
                "r1": sha_array(bank1), "r2": sha_array(bank2)}):
        raise ValueError("fixed-map Q table differs from actual full reference bank")


def with_reference_q_features(batch: WireBatch, table, *, view: str,
                              full_bank1: np.ndarray, full_bank2: np.ndarray,
                              role_release: ReferenceRoleRelease,
                              expected_mechanism_sha256: str) -> WireBatch:
    """Attach fixed-map Q features without adding untransmitted metadata fields."""
    from .q_aware import with_q_features
    validate_q_reference_banks(table, bank1=full_bank1, bank2=full_bank2,
                               att=role_release)
    if (type(expected_mechanism_sha256) is not str or
            HEX64.fullmatch(expected_mechanism_sha256) is None or
            table.identity["mechanism_sha256"] != expected_mechanism_sha256):
        raise ValueError("fixed-map Q mechanism differs from sealed public map")
    prefixes = ("r1.", "r2.") if view == "pair" else ("",)
    recipients = (1, 2) if view == "pair" else ((1,) if view == "r1" else (2,))
    if view not in ("r1", "r2", "pair"):
        raise ValueError("fixed-map Q view differs")
    enriched = dict(batch.fields)
    for prefix, recipient in zip(prefixes, recipients, strict=True):
        source_bank_sha = role_release.receipt["bank_sha256_by_recipient"][
            f"r{recipient}"]
        name = prefix + "public_metadata.bank_sha256"
        if (name not in enriched or
                not np.all(np.asarray(enriched[name].values) == source_bank_sha)):
            raise ValueError("fixed-map Q full-bank wire identity differs")
        n = len(batch.row_id)
        enriched[prefix + "public_metadata.canonical_bank_sha256"] = Field(
            "metadata_category", np.asarray([
                table.identity[f"bank{recipient}_sha256"]] * n))
        enriched[prefix + "public_metadata.mechanism_sha256"] = Field(
            "metadata_category", np.asarray([expected_mechanism_sha256] * n))
    temporary = WireBatch(batch.row_id, batch.unit, enriched,
                          batch.code_fields, batch.public_derived)
    result = with_q_features(temporary, table, view=view)
    return WireBatch(batch.row_id, batch.unit, batch.fields,
                     batch.code_fields, result.public_derived)
