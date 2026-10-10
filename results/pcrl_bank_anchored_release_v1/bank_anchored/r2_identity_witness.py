"""Queue-only full-FIT original-code B0 identity witness, without labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from zipfile import ZipFile

import numpy as np

from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_array,
)
from results.pcrl_latent_and_sampled_methods_v1.r01.tstar_source import TStarSourceStore
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import (
    GateWeights, JointPolicy,
)
from .r2_laws import IdentityPolicy, wire_fixed_bank_and_actions


FIELDS = frozenset({"row_id", "unit", "code0", "code1", "prob0", "prob1",
                    "decision0", "decision1"})
PINS = frozenset({"role_manifest", "source_audit", "portfolio_manifest",
                  "saved_b0", "r2_laws_source", "r2_witness_source"})


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pinned(record, registered):
    if set(record) != {"path", "sha256"}:
        raise PermissionError("witness source pin schema differs")
    path = Path(record["path"])
    if path.is_symlink():
        raise PermissionError("source symlink refused")
    path = path.resolve(strict=True)
    if registered.get(path) != record["sha256"] or _sha(path) != record["sha256"]:
        raise PermissionError("witness source differs from queue pin")
    return path


def run(config_path):
    active = assert_active_from_env(stage="witness")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    registered = {Path(item["path"]).resolve(strict=True): item["sha256"]
                  for item in state["units"][active["unit_id"]]["spec"]["sources"]}
    config_path = Path(config_path).resolve(strict=True)
    if registered.get(config_path) != _sha(config_path):
        raise PermissionError("witness config is not byte-pinned")
    cfg = json.loads(config_path.read_text())
    if (set(cfg) != {"schema", "unit_id", "seed", "source_lock_sha256",
                     "pins", "transitive_pins", "output"} or
            cfg["schema"] != "BAR1-r2-full-fit-identity-config-v1" or
            cfg["unit_id"] != active["unit_id"] or
            cfg["seed"] not in (0, 1) or
            cfg["source_lock_sha256"] != active["seal_sha256"] or
            set(cfg["pins"]) != PINS or
            not isinstance(cfg["transitive_pins"], list)):
        raise PermissionError("full-FIT witness config or seal differs")
    pins = {key: _pinned(value, registered) for key, value in cfg["pins"].items()}
    transitive = {_pinned(value, registered) for value in cfg["transitive_pins"]}
    if len(transitive) != len(cfg["transitive_pins"]):
        raise PermissionError("duplicate transitive source pin")
    if pins["r2_laws_source"].resolve() != Path(__file__).with_name("r2_laws.py").resolve() or \
            pins["r2_witness_source"].resolve() != Path(__file__).resolve():
        raise PermissionError("witness implementation pin differs")
    portfolio_package = pins["portfolio_manifest"].parent
    role_export = (pins["role_manifest"].parent.parent /
                   ".private/role_export_v1")
    role = "OSF_DEFENSE_FIT"
    bank_root = portfolio_package / f"shared/.private/tstar_score_bank_s{cfg['seed']}_v1"
    expected_transitive = {
        role_export / role / f"references_s{cfg['seed']}.npz",
        bank_root / "BANK_ADMISSION_RECEIPT.json",
        bank_root / "bank_r1.npz", bank_root / "bank_r2.npz",
        bank_root / f"tokens_{role}.npz",
        Path(__file__).resolve().parents[2] / "pcrl_latent_and_sampled_methods_v1" /
        "r01/role_store.py",
        Path(__file__).resolve().parents[2] / "pcrl_latent_and_sampled_methods_v1" /
        "r01/tstar_source.py",
    }
    if transitive != {path.resolve(strict=True) for path in expected_transitive}:
        raise PermissionError("transitive role/source pin inventory differs")
    roles = RoleOnlyStore(pins["role_manifest"])
    source = TStarSourceStore(portfolio_package, pins["role_manifest"],
                              pins["source_audit"], cfg["seed"])
    reference = roles.read(role, f"references_s{cfg['seed']}")
    cells = source.read_role_cells(role)
    assert_same_people(reference, cells)
    bank_action = []
    for recipient in (1, 2):
        bank = source.read_bank(recipient)
        bank_action.append(wire_fixed_bank_and_actions(
            bank["probability"], bank["token_class"],
            bank["fine_to_hcal_token"], bank["input_class"]))
    base = JointPolicy(bank_action[0][0], bank_action[1][0],
                       bank_action[0][1], bank_action[1][1],
                       GateWeights.zeros(input_width=12))
    identity = IdentityPolicy(base)
    wire_path = pins["saved_b0"]
    with ZipFile(wire_path) as archive:
        if {item.filename for item in archive.infolist()} != \
                {name + ".npy" for name in FIELDS}:
            raise PermissionError("saved W_TSTAR FIT wire member set differs")
    with np.load(wire_path, allow_pickle=False) as archive:
        saved = {name: np.array(archive[name], copy=True) for name in FIELDS}
    assert_same_people(reference, saved)
    n = len(cells["row_id"])
    if n != 15434:
        raise PermissionError("full FIT people count differs")
    bits = np.empty((2, n, 10), dtype=np.uint8)
    code = np.empty((2, n), dtype=np.int64)
    for i in range(n):
        law = identity.sparse_law(reference["U0_1"][i], reference["U0_2"][i],
                                  int(cells["cell_1"][i]), int(cells["cell_2"][i]))
        if len(law) != 1 or law[0][2] != 1.0:
            raise RuntimeError("identity has a stochastic or secondary action")
        code[0, i], code[1, i] = law[0][:2]
    shifts = np.arange(10, dtype=np.int64)
    for purpose in (0, 1):
        recipient = purpose + 1
        table = bank_action[purpose][0].probability
        bits[purpose] = ((code[purpose, :, None] >> shifts) & 1).astype(np.uint8)
        if (not np.array_equal(code[purpose], cells[f"token_{recipient}"]) or
                not np.array_equal(bits[purpose], saved[f"code{purpose}"]) or
                not np.array_equal(table[code[purpose]], saved[f"prob{purpose}"]) or
                not np.array_equal(bank_action[purpose][0].decision[code[purpose]],
                                   saved[f"decision{purpose}"])):
            raise RuntimeError("full FIT identity differs from original B0 wire")
    output = Path(cfg["output"])
    expected = Path(__file__).resolve().parents[1] / "r2" / output.name
    if (output != expected or output.exists() or output.is_symlink() or
            output.parent.is_symlink() or not output.parent.is_dir()):
        raise PermissionError("witness receipt output is outside R2 namespace")
    receipt = {"schema": "BAR1-r2-full-fit-identity-v1", "status": "PASS",
               "source_lock_sha256": active["seal_sha256"],
               "unit_id": active["unit_id"], "source_seed": cfg["seed"],
               "role": role, "rows": n, "groups": int(len(np.unique(cells["unit"]))),
               "row_unit_sha256": sha_array(np.column_stack((cells["row_id"],
                                                               cells["unit"]))),
               "source_binding_sha256": source.source_binding_sha256,
               "saved_b0_wire_sha256": _sha(wire_path),
               "r1_code_sha256": sha_array(code[0]),
               "r2_code_sha256": sha_array(code[1]),
               "r1_table_sha256": sha_array(bank_action[0][0].probability),
               "r2_table_sha256": sha_array(bank_action[1][0].probability),
               "source_laws_sha256": _sha(pins["r2_laws_source"]),
               "full_row_code_probability_decision_equal": True}
    payload = (json.dumps(receipt, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644),
                   "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().config), sort_keys=True))
