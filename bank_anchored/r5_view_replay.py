"""Independent reconstruction of complete bank attack views from cached wires.

This module does not import the release or attack view implementations. It
checks the matrices frozen with the readers against the public code bits,
fixed decoder tables, decisions and deterministic public transforms.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive(path: Path, names: set[str]) -> dict[str, np.ndarray]:
    with ZipFile(path) as source:
        members = [entry.filename for entry in source.infolist()]
        if len(members) != len(names) or set(members) != {x + ".npy" for x in names}:
            raise PermissionError("release/view archive member inventory differs")
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in names}


def public_wire(codes: np.ndarray, table: np.ndarray) -> dict[str, np.ndarray]:
    code = np.asarray(codes)
    prob_bank = np.asarray(table)
    if (code.dtype != np.int64 or code.ndim != 1 or prob_bank.dtype != np.float64 or
            prob_bank.ndim != 2 or len(prob_bank) > 1024 or
            np.any((code < 0) | (code >= len(prob_bank))) or
            not np.isfinite(prob_bank).all() or np.any(prob_bank < 0) or
            not np.allclose(prob_bank.sum(axis=1), 1.0, atol=1e-10, rtol=0)):
        raise ValueError("bank code/table is outside emitted wire contract")
    bits = ((code[:, None].astype(np.uint64) >>
             np.arange(10, dtype=np.uint64)[None, :]) & 1).astype(np.uint8)
    prob = prob_bank[code]
    return {"code": bits, "prob": prob,
            "decision": np.asarray(prob.argmax(axis=1), dtype=np.int64)}


def wire_sha(wire: dict[str, np.ndarray]) -> str:
    memory = io.BytesIO()
    np.savez_compressed(memory, code=wire["code"], prob=wire["prob"],
                        decision=wire["decision"])
    return hashlib.sha256(memory.getvalue()).hexdigest()


def _block(wire: dict[str, np.ndarray], vocab: tuple[str, ...]) -> np.ndarray:
    bits = np.asarray(wire["code"], dtype=np.float64)
    ids = np.sum(bits.astype(np.int64) *
                 (1 << np.arange(10, dtype=np.int64))[None, :], axis=1)
    lookup = {value: j for j, value in enumerate(vocab)}
    categorical = np.zeros((len(ids), len(vocab) + 1), dtype=np.float64)
    categorical[np.arange(len(ids)),
                [lookup.get(str(int(value)), len(vocab)) for value in ids]] = 1.0
    return np.concatenate((bits, categorical, wire["prob"],
                           wire["decision"][:, None].astype(np.float64)), axis=1)


def matrices(codes_att: tuple[np.ndarray, np.ndarray],
             codes_inner: tuple[np.ndarray, np.ndarray],
             tables: tuple[np.ndarray, np.ndarray], view: str) -> dict[str, dict]:
    if view not in ("r1", "r2", "pair"):
        raise ValueError("unrecognized public view")
    emitted = {role: tuple(public_wire(codes[i], tables[i]) for i in (0, 1))
               for role, codes in (("att", codes_att), ("inner", codes_inner))}
    vocab = tuple(tuple(sorted(set(map(str, np.asarray(codes_att[i]).tolist()))))
                  for i in (0, 1))
    selected = (0,) if view == "r1" else (1,) if view == "r2" else (0, 1)
    out = {}
    for role in ("att", "inner"):
        wires = emitted[role]
        x = np.concatenate([_block(wires[i], vocab[i]) for i in selected], axis=1)
        derived = [np.log(np.maximum(wires[i]["prob"], 1e-12)) for i in selected]
        if view == "pair":
            left = wires[0]["code"].astype(np.float64)
            right = wires[1]["code"].astype(np.float64)
            derived.append((left[:, :, None] * right[:, None, :]).reshape(len(left), 100))
        ids = [np.asarray(codes_att if role == "att" else codes_inner)[i]
               for i in selected]
        keys = np.asarray([json.dumps([str(int(v)) for v in values],
                                      separators=(",", ":")) for values in zip(*ids)],
                          dtype=np.str_)
        out[role] = {
            "x": x,
            "x_defense_aware": np.concatenate([x, *derived], axis=1),
            "x_code_bits": np.concatenate([wires[i]["code"].astype(np.float64)
                                            for i in selected], axis=1),
            "cell_key": keys,
            "wire_shas": tuple(wire_sha(wires[i]) for i in (0, 1)),
            "vocab": vocab,
        }
    return out


def verify(spec: dict, public: dict, private: Path) -> dict:
    table_path = Path(spec["public_table_path"])
    if sha(table_path) != spec["public_table_sha256"]:
        raise PermissionError("public decoder table changed")
    family = json.loads(table_path.read_text())["families"]["W_TSTAR"]
    tables = tuple(np.asarray(family[f"table{i}"], dtype=np.float64) for i in (0, 1))
    role_codes = {}
    for role, label in (("ATTACK_FIT_NEW", "att"),
                        ("INNER_SELECTION", "inner")):
        root = Path(spec["release_directory_by_role"][role])
        receipt_path = root / "RELEASE_RECEIPT.json"
        if sha(receipt_path) != spec["release_receipt_sha256_by_role"][role]:
            raise PermissionError("source cached release receipt changed")
        receipt = json.loads(receipt_path.read_text())
        source = root / "release_codes.npz"
        if (sha(source) != receipt["release_codes_sha256"] or
                receipt["role"] != role or
                receipt["arm_id"] != spec["arm_id"] or
                receipt["source_seed"] != spec["source_seed"] or
                receipt["realization"] != spec["realization"]):
            raise PermissionError("cached release source identity differs")
        values = archive(source, {"row_id", "unit", "code_r1", "code_r2"})
        role_codes[label] = values
    checked = {}
    for view in ("r1", "r2", "pair"):
        expected = matrices(
            (role_codes["att"]["code_r1"], role_codes["att"]["code_r2"]),
            (role_codes["inner"]["code_r1"], role_codes["inner"]["code_r2"]),
            tables, view)
        reader_root = private / view / "readers"
        for role, source_role in (("att", "ATTACK_FIT_NEW"),
                                  ("inner", "INNER_SELECTION")):
            actual = archive(reader_root / f"view_{role}.npz",
                             {"row_id", "unit", "x", "x_defense_aware",
                              "x_code_bits", "cell_key"})
            for key in ("row_id", "unit"):
                if not np.array_equal(actual[key], role_codes[role][key]):
                    raise ValueError("attack view people differ from cached release")
            for key in ("x", "x_defense_aware", "x_code_bits", "cell_key"):
                if not np.array_equal(actual[key], expected[role][key]):
                    raise ValueError(f"{view}/{role} {key} omits or changes public wire")
            side = public["public_side_information_by_role"][source_role]
            for purpose in (0, 1):
                if side[str(purpose)]["release_sha256"] != expected[role]["wire_shas"][purpose]:
                    raise ValueError("public wire byte hash differs from attack side information")
        checked[view] = {"att_columns": int(expected["att"]["x"].shape[1]),
                         "defense_aware_columns":
                             int(expected["att"]["x_defense_aware"].shape[1]),
                         "complete": True}
    return {"status": "PASS", "views": checked,
            "scope": "cached actual draw, all emitted bit/probability/decision fields, public derived columns and fixed side information"}
