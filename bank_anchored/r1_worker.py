"""Queue-only real-input admission of the frozen W_TSTAR deployment path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from zipfile import ZipFile

import numpy as np

from code_witness.bank import bits_to_tokens
from code_witness.bank_deploy import load_bank_service
from dpc.deploy import schema_sha256
from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws import (
    IdentityPolicy, wire_fixed_bank_and_actions,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import (
    GateWeights, JointPolicy,
)

from .r1_runtime import release_pair_from_raw


ROLE_KEYS = frozenset({"row_id", "unit", "X", "feature_names"})
WIRE_KEYS = frozenset({"row_id", "unit", "code0", "prob0", "decision0",
                       "code1", "prob1", "decision1"})


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _checked(path_text: str, registered: dict[Path, str]) -> Path:
    original = Path(path_text)
    if original.is_symlink():
        raise PermissionError("source symlink refused")
    path = original.resolve(strict=True)
    if registered.get(path) != _sha(path):
        raise PermissionError("source file is not byte-pinned by active queue unit")
    return path


def _write_npz(path: Path, arrays: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def run(config_path: Path) -> dict:
    active = assert_active_from_env(stage="witness")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    sources = state["units"][active["unit_id"]]["spec"]["sources"]
    registered = {Path(item["path"]).resolve(strict=True): item["sha256"]
                  for item in sources}
    config_path = _checked(str(config_path), registered)
    if ".private" not in config_path.parts or stat.S_IMODE(config_path.stat().st_mode) & 0o077:
        raise PermissionError("owner-only private worker config required")
    cfg = json.loads(config_path.read_text())
    if (cfg.get("schema") != "pcrl-bank-r1-raw-admission-config-v1" or
            cfg.get("unit_id") != active["unit_id"] or
            cfg.get("role") != "OSF_DEFENSE_FIT" or
            cfg.get("seed") not in (0, 1)):
        raise PermissionError("R1 source role or unit identity differs")
    pins = {name: _checked(path, registered)
            for name, path in cfg["pins"].items()}
    seed = int(cfg["seed"])
    source = Path(cfg["source_root"])
    qpc = Path(cfg["qpc_root"])
    expected = {
        "model": source / "admitted" / f"rel__s{seed}__U/model.pt",
        "complete": source / "admitted" / f"rel__s{seed}__U/COMPLETE.json",
        "head0": source / "admitted" / f"rel__s{seed}__U/head_0.joblib",
        "head1": source / "admitted" / f"rel__s{seed}__U/head_1.joblib",
        "record": source / "admitted" / f"rel__s{seed}__U/record.json",
        "policy": qpc / "run" / "units" / f"pol__s{seed}__U_FINE-TASK_i8o64/policy.json",
        "decoder": source / "package" /
                   f"decoder__s{seed}__U_FINE-TASK_i8o64_H-GLOBAL-TEMP.json",
    }
    if any(pins.get(name) != path.resolve(strict=True) for name, path in expected.items()):
        raise PermissionError("frozen source paths differ from queue-pinned model/policy")
    public = json.loads(pins["public_table"].read_text())
    if (public.get("schema") != "code-utility-public-bank-tables-v1" or
            public.get("seed") != seed or public.get("bit_count") != 10 or
            public.get("source_model_sha256") != _sha(pins["model"]) or
            public.get("feature_schema_sha256") != cfg["schema_sha256"] or
            public["families"]["W_TSTAR"]["policy_sha256"] != _sha(pins["policy"]) or
            public["families"]["W_TSTAR"]["decoder_sha256"]
               != json.loads(pins["decoder"].read_text())["decoder_sha256"]):
        raise PermissionError("public W_TSTAR/source binding differs")
    with ZipFile(pins["role_features"]) as archive:
        if {item.filename for item in archive.infolist()} != {f"{key}.npy" for key in ROLE_KEYS}:
            raise ValueError("role-only feature archive has unexpected member")
    with np.load(pins["role_features"], allow_pickle=False) as archive:
        feature = {key: np.array(archive[key], copy=True) for key in ROLE_KEYS}
    names = tuple(str(value) for value in feature["feature_names"])
    rows = len(feature["row_id"])
    if (rows != 15434 or feature["X"].shape != (rows, 83) or
            feature["unit"].shape != (rows,) or len(names) != 83 or
            schema_sha256(names) != cfg["schema_sha256"] or
            not np.all(np.isfinite(feature["X"]))):
        raise ValueError("role-only FIT feature schema or population differs")
    indices = np.asarray(cfg["indices"], dtype=np.int64)
    if (indices.shape != (32,) or not np.array_equal(
            indices, np.linspace(0, rows - 1, 32, dtype=np.int64))):
        raise ValueError("prefixed representative FIT index plan differs")
    with ZipFile(pins["saved_b0"]) as archive:
        if {item.filename for item in archive.infolist()} != {f"{key}.npy" for key in WIRE_KEYS}:
            raise ValueError("historical B0 archive has unexpected member")
    with np.load(pins["saved_b0"], allow_pickle=False) as archive:
        if (not np.array_equal(archive["row_id"], feature["row_id"]) or
                not np.array_equal(archive["unit"], feature["unit"])):
            raise ValueError("historical B0 and role-only FIT people differ")
        saved = {key: np.array(archive[key][indices], copy=True)
                 for key in WIRE_KEYS - {"row_id", "unit"}}
    output = Path(cfg["private_output_dir"])
    if ".private" not in output.parts or output.exists() or not output.parent.is_dir():
        raise PermissionError("new owner-only private output directory required")
    output.mkdir(mode=0o700)
    sample = output / "sample_input.npz"
    _write_npz(sample, {"X": feature["X"][indices],
                        "feature_names": feature["feature_names"]})
    schema_file = output / "schema.json"
    schema_file.write_text(json.dumps(list(names)) + "\n")
    os.chmod(schema_file, 0o600)
    service = load_bank_service(
        source_root=source, qpc_root=qpc, table_path=pins["public_table"],
        expected_table_sha256=_sha(pins["public_table"]),
        expected_source_sha256=_sha(pins["model"]),
        expected_schema_sha256=cfg["schema_sha256"],
        ordered_names=names, bank="W_TSTAR", seed=seed)
    bank_and_actions = []
    for purpose, part in enumerate((service.policy_pair.p1,
                                    service.policy_pair.p2)):
        bank_and_actions.append(wire_fixed_bank_and_actions(
            np.asarray(public["families"]["W_TSTAR"][f"table{purpose}"],
                       dtype=np.float64),
            np.asarray(public["families"]["W_TSTAR"][f"token_class{purpose}"],
                       dtype=np.int64),
            np.asarray(part.cell_token, dtype=np.int64),
            np.asarray(part.fine.cell_class, dtype=np.int64)))
    base = JointPolicy(bank_and_actions[0][0], bank_and_actions[1][0],
                       bank_and_actions[0][1], bank_and_actions[1][1],
                       GateWeights.zeros(input_width=12))
    identity = IdentityPolicy(base)
    new_wires = release_pair_from_raw(
        service, identity, feature["X"][indices], names, None)
    for purpose, wire in enumerate(new_wires):
        if any(not np.array_equal(wire[key], saved[f"{key}{purpose}"])
               for key in ("code", "prob", "decision")):
            raise ValueError("new IdentityPolicy raw route differs from original B0")
        _write_npz(output / f"new_identity_recipient{purpose}.npz", wire)
    maximum_gap = 0.0
    for purpose in (0, 1):
        assert_active_from_env(unit_id=active["unit_id"], stage="witness")
        wire_path = output / f"recipient{purpose}.npz"
        command = [cfg["python"], "-m", "code_witness.bank_deploy",
                   "--bank", "W_TSTAR", "--purpose", str(purpose),
                   "--seed", str(seed), "--source-root", str(source),
                   "--source-sha256", _sha(pins["model"]),
                   "--qpc-root", str(qpc), "--public-table", str(pins["public_table"]),
                   "--public-table-sha256", _sha(pins["public_table"]),
                   "--schema", str(schema_file), "--schema-sha256", cfg["schema_sha256"],
                   "--input", str(sample), "--output", str(wire_path)]
        subprocess.run(command, check=True, cwd=cfg["cwd"],
                       capture_output=True, text=True)
        with np.load(wire_path, allow_pickle=False) as archive:
            if set(archive.files) != {"code", "prob", "decision"}:
                raise ValueError("actual recipient wire has hidden field")
            actual = {name: np.array(archive[name], copy=True)
                      for name in ("code", "prob", "decision")}
        table = np.asarray(public["families"]["W_TSTAR"][f"table{purpose}"],
                           dtype=np.float64)
        token = bits_to_tokens(actual["code"])
        if (not np.array_equal(actual["code"], saved[f"code{purpose}"]) or
                not np.array_equal(actual["prob"], saved[f"prob{purpose}"]) or
                not np.array_equal(actual["decision"], saved[f"decision{purpose}"]) or
                any(not np.array_equal(actual[key], new_wires[purpose][key])
                    for key in ("code", "prob", "decision")) or
                np.any(token >= len(table)) or
                not np.array_equal(actual["prob"], table[token]) or
                not np.array_equal(actual["decision"], table[token].argmax(axis=1))):
            raise ValueError("raw bank deployment differs from saved B0/table")
        maximum_gap = max(maximum_gap, float(np.max(np.abs(
            actual["prob"] - saved[f"prob{purpose}"]))))
    result = {
        "schema": "pcrl-bank-r1-raw-admission-receipt-v1", "status": "PASS",
        "unit_id": active["unit_id"], "seed": seed, "role": cfg["role"],
        "rows": len(indices), "source_role_rows": rows,
        "index_rule": "linspace(0,15433,32,dtype=int64)",
        "exact_code_probability_decision_parity": True,
        "new_identity_runtime_exact_parity": True,
        "maximum_probability_gap": maximum_gap,
        "public_table_sha256": _sha(pins["public_table"]),
        "source_model_sha256": _sha(pins["model"]),
        "policy_sha256": _sha(pins["policy"]),
        "sample_input_sha256": _sha(sample),
        "private_recipient_sha256": {str(i): _sha(output / f"recipient{i}.npz")
                                     for i in (0, 1)},
        "private_new_identity_sha256": {
            str(i): _sha(output / f"new_identity_recipient{i}.npz")
            for i in (0, 1)},
        "scope": "Representative role-only FIT rows and actual raw CLI; no task or SEX label read",
    }
    receipt = output / "RECEIPT.json"
    receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    os.chmod(receipt, 0o600)
    public_receipt = Path(cfg["public_receipt"])
    expected_public = (Path(cfg["cwd"]) / "results" /
                       "pcrl_bank_anchored_release_v1" / "r1" /
                       f"R1_BANK_S{seed}_FIT_RAW_RECEIPT.json")
    if public_receipt.resolve(strict=False) != expected_public.resolve(strict=False):
        raise PermissionError("public aggregate receipt must stay in R1 namespace")
    fd = os.open(public_receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), sort_keys=True))


if __name__ == "__main__":
    main()
