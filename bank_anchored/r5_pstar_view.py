"""Independent complete-view reconstruction for the original P* finite wire.

This module reads the public fixed map and the two allowed role-only release
packages. It does not import the inherited reference or view encoder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


WIRE_SCHEMA = "SCV1-reference-score-release-v1"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode() + str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def archive(path: Path, keys: set[str]) -> dict[str, np.ndarray]:
    with ZipFile(path) as source:
        names = [item.filename for item in source.infolist()]
        if len(names) != len(keys) or set(names) != {key + ".npy" for key in keys}:
            raise PermissionError("P* source archive member inventory differs")
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in keys}


def role_fields(source: dict[str, np.ndarray], receipt: dict, recipient: int,
                bank: np.ndarray) -> dict[str, tuple[str, np.ndarray]]:
    name = f"r{recipient}"
    probability = source[f"prob_{name}"]
    decision = source[f"decision_{name}"]
    code = source[f"code_{name}"]
    n = len(code)
    if (code.dtype != np.int64 or code.shape != (n,) or
            probability.dtype != np.float64 or
            probability.shape != (n, (2 if recipient == 1 else 6)) or
            decision.dtype != np.int64 or decision.shape != (n,) or
            np.any((code < 0) | (code >= len(bank))) or
            not np.array_equal(bank[code], probability) or
            not np.array_equal(np.argmax(probability, axis=1), decision) or
            array_sha(bank) != receipt["bank_sha256_by_recipient"][name]):
        raise ValueError("P* code, probability, decision or public bank differs")
    bits = (len(bank) - 1).bit_length()
    meta = {"model_id": receipt["release_id"],
            "source_binding_sha256": receipt["source_binding_sha256"],
            "bank_sha256": receipt["bank_sha256_by_recipient"][name],
            "code_length_bits": bits}
    wire_digest = hashlib.sha256()
    for i in range(n):
        wire = {"schema": WIRE_SCHEMA, "recipient": recipient,
                "probability": probability[i].tolist(),
                "decision": int(decision[i]), "release_id": receipt["release_id"],
                "public_metadata": meta, "code_id": int(code[i])}
        wire_digest.update(json.dumps(wire, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode() + b"\n")
    if wire_digest.hexdigest() != receipt["wire_sha256_by_recipient"][name]:
        raise ValueError("P* public wire SHA differs from role receipt")
    def category(value: str) -> np.ndarray:
        return np.asarray([value] * n, dtype=np.str_)
    return {
        "schema": ("category", category(WIRE_SCHEMA)),
        "recipient": ("numeric", np.full(n, recipient, dtype=np.float64)),
        "probability": ("probability", probability),
        "decision": ("decision", decision),
        "release_id": ("category", category(receipt["release_id"])),
        "code_id": ("category", code),
        "code_bits": ("bits", ((code[:, None].astype(np.uint64) >>
                                  np.arange(max(1, bits), dtype=np.uint64)[None, :]) &
                                 1).astype(np.float64)),
        "public_metadata.model_id": ("metadata_category", category(meta["model_id"])),
        "public_metadata.source_binding_sha256":
            ("metadata_category", category(meta["source_binding_sha256"])),
        "public_metadata.bank_sha256":
            ("metadata_category", category(meta["bank_sha256"])),
        "public_metadata.code_length_bits":
            ("metadata", np.full(n, bits, dtype=np.float64)),
        "public_metadata.canonical_json":
            ("metadata_category", category(json.dumps(meta, sort_keys=True,
                                                      separators=(",", ":"),
                                                      allow_nan=False))),
    }


def encode(att: dict[str, tuple[str, np.ndarray]],
           inner: dict[str, tuple[str, np.ndarray]],
           *, code_fields: tuple[str, ...], view: str) -> dict[str, dict[str, np.ndarray]]:
    if tuple(att) != tuple(inner):
        raise ValueError("P* ATT/INNER field order differs")
    vocab = {}
    for name, (kind, values) in att.items():
        if kind in ("category", "metadata_category"):
            vocab[name] = tuple(sorted(set(map(str, np.asarray(values).tolist()))))
    out = {}
    for role, fields in (("att", att), ("inner", inner)):
        n = len(next(iter(fields.values()))[1])
        blocks, metadata, bits = [], [], []
        for name, (kind, values) in fields.items():
            if kind in ("category", "metadata_category"):
                choice = vocab[name]
                lookup = {value: i for i, value in enumerate(choice)}
                block = np.zeros((n, len(choice) + 1), dtype=np.float64)
                block[np.arange(n), [lookup.get(str(v), len(choice))
                                     for v in np.asarray(values).tolist()]] = 1
            else:
                block = np.asarray(values, dtype=np.float64)
                if block.ndim == 1:
                    block = block[:, None]
            blocks.append(block)
            if kind in ("metadata", "metadata_category"):
                metadata.append(block)
            if kind == "bits":
                bits.append(block)
        cell_values = [np.asarray(fields[name][1]) for name in code_fields]
        cell_key = np.asarray([json.dumps([str(int(v)) for v in values],
                                         separators=(",", ":"))
                               for values in zip(*cell_values)], dtype=np.str_)
        x = np.concatenate(blocks, axis=1)
        if view == "pair":
            left = np.asarray(fields["r1.code_bits"][1], dtype=np.float64)
            right = np.asarray(fields["r2.code_bits"][1], dtype=np.float64)
            derived = [np.log(np.maximum(fields["r1.probability"][1], 1e-12)),
                       np.log(np.maximum(fields["r2.probability"][1], 1e-12)),
                       (left[:, :, None] * right[:, None, :]).reshape(n, -1)]
        else:
            derived = [np.log(np.maximum(fields["probability"][1], 1e-12))]
        out[role] = {"x": x,
                     "x_defense_aware": np.concatenate([x, *derived], axis=1),
                     "x_metadata": np.concatenate(metadata, axis=1),
                     "x_code_bits": np.concatenate(bits, axis=1),
                     "cell_key": cell_key}
    return out


def verify(spec: dict, private: Path) -> dict:
    if spec.get("arm_id") != "P_STAR":
        raise ValueError("P* view verifier received another arm")
    fixed_receipt_path = Path(spec["fixed_map_receipt_path"])
    fixed_path = Path(spec["fixed_map_archive_path"])
    if (sha(fixed_receipt_path) != spec["fixed_map_receipt_sha256"] or
            sha(fixed_path) != spec["fixed_map_archive_sha256"]):
        raise PermissionError("P* fixed public map source changed")
    fixed_receipt = json.loads(fixed_receipt_path.read_text())
    fixed = archive(fixed_path, {"bank_r1", "bank_r2", "fine_to_code_r1",
                                 "fine_to_code_r2"})
    if (fixed_receipt["source_binding_sha256"] is None or
            fixed_receipt["npz_sha256"] != spec["fixed_map_archive_sha256"] or
            any(array_sha(value) != fixed_receipt["array_sha256"][name]
                for name, value in fixed.items())):
        raise ValueError("P* fixed map arrays differ from public receipt")
    releases = {}
    for role, directory_key, digest_key in (
            ("att", "att_package_dir", "att_receipt_sha256"),
            ("inner", "inner_package_dir", "inner_receipt_sha256")):
        root = Path(spec[directory_key])
        receipt_path = root / "REFERENCE_RELEASE_RECEIPT.json"
        if sha(receipt_path) != spec[digest_key]:
            raise PermissionError("P* role release receipt changed")
        receipt = json.loads(receipt_path.read_text())
        path = root / "reference_release.npz"
        if (sha(path) != receipt["codes_file_sha256"] or
                receipt["source_binding_sha256"] !=
                fixed_receipt["source_binding_sha256"] or
                receipt["interface_kind"] != "finite" or
                receipt["arm_id"] != "P_STAR" or
                receipt["source_seed"] != spec["source_seed"]):
            raise PermissionError("P* role source binding differs")
        values = archive(path, {"row_id", "unit", "prob_r1", "prob_r2",
                                "decision_r1", "decision_r2", "code_r1", "code_r2"})
        if (values["row_id"].dtype != np.int64 or
                values["unit"].dtype != np.int64 or
                values["row_id"].shape != (receipt["rows"],) or
                values["unit"].shape != values["row_id"].shape or
                len(np.unique(values["unit"])) != receipt["groups"] or
                not np.all(np.diff(values["row_id"]) > 0) or
                array_sha(np.column_stack((values["row_id"], values["unit"]))) !=
                receipt["row_unit_pair_sha256"] or
                receipt["role"] != ("ATTACK_FIT_NEW" if role == "att" else
                                    "INNER_SELECTION")):
            raise PermissionError("P* role ordered people/group binding differs")
        fields = tuple(role_fields(values, receipt, i, fixed[f"bank_r{i}"])
                       for i in (1, 2))
        releases[role] = (values, fields)
    checked = {}
    for view in ("r1", "r2", "pair"):
        def named(fields: tuple[dict, dict]) -> dict:
            selected = (0,) if view == "r1" else (1,) if view == "r2" else (0, 1)
            return {((f"r{i+1}." if view == "pair" else "") + name): value
                    for i in selected for name, value in fields[i].items()}
        att = named(releases["att"][1])
        inner = named(releases["inner"][1])
        codes = ("r1.code_id", "r2.code_id") if view == "pair" else ("code_id",)
        expected = encode(att, inner, code_fields=codes, view=view)
        for role in ("att", "inner"):
            actual = archive(private / view / "readers" / f"view_{role}.npz",
                             {"row_id", "unit", "x", "x_defense_aware", "x_metadata",
                              "x_code_bits", "cell_key"})
            source = releases[role][0]
            if (not np.array_equal(actual["row_id"], source["row_id"]) or
                    not np.array_equal(actual["unit"], source["unit"]) or
                    any(not np.array_equal(actual[name], expected[role][name])
                        for name in ("x", "x_defense_aware", "x_metadata",
                                     "x_code_bits", "cell_key"))):
                raise ValueError("P* frozen view omits or changes a public field")
        checked[view] = {"att_columns": int(expected["att"]["x"].shape[1]),
                         "defense_aware_columns":
                             int(expected["att"]["x_defense_aware"].shape[1]),
                         "complete": True}
    return {"status": "PASS", "views": checked,
            "scope": "P* separate finite full-bank wire, code/probability/decision and every fixed metadata field"}
