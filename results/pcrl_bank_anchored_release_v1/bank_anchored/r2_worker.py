"""Queue-only protected FIT and selected public-code law materialization.

Only role-specific admitted archives are opened. T and B0 never read SEX.
Checkpoint selection completes on FIT before ATT/INNER scores are opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_array,
)
from results.pcrl_latent_and_sampled_methods_v1.r01.tstar_source import TStarSourceStore
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import (
    GateWeights, JointPolicy,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.fit import (
    TaskOnlyData, ProtectedData, prepare_law,
)
from .r2_laws import IdentityPolicy, wire_fixed_bank_and_actions
from .r2_dispatch import ARMS, fit_arm
from .r2_materialize import materialize_arrays, write_materialization
from .r2_package import save_policy


CONFIG_KEYS = frozenset({"schema", "unit_id", "arm_id", "source_seed",
                         "recipe_seed", "cpu_seconds", "source_lock_sha256",
                         "protection_amendment_sha256", "pins",
                         "private_output_dir", "fit_receipt", "unit_receipt"})
PIN_KEYS = frozenset({"campaign_lock", "protection_amendment", "role_manifest",
                      "portfolio_manifest", "source_audit", "stream_roots",
                      "stream_manifest",
                      "identity_receipt", "worker_source"})
ALL_ARMS = ("B0", *ARMS)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source(record, registered):
    if set(record) != {"path", "sha256"}:
        raise PermissionError("FIT source pin inventory differs")
    path = Path(record["path"])
    if path.is_symlink():
        raise PermissionError("FIT source symlink refused")
    path = path.resolve(strict=True)
    if registered.get(path) != record["sha256"] or _sha(path) != record["sha256"]:
        raise PermissionError("FIT source differs from queue pin")
    return path


def _new_json(path, payload, *, private):
    path = Path(path)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise PermissionError("FIT receipt is not a new regular file")
    if private and stat.S_IMODE(path.parent.stat().st_mode) & 0o077:
        raise PermissionError("private FIT receipt parent is not owner-only")
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                           0o600 if private else 0o644), "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def _checkpoint_archive(root, record):
    """Privately retain every reachable neural checkpoint array and public hashes."""
    arrays = {}

    def normalize(value):
        if isinstance(value, np.generic):
            return normalize(value.item())
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            if not np.isfinite(value):
                raise ValueError("checkpoint scalar is nonfinite")
            return value
        if isinstance(value, np.ndarray):
            name = f"checkpoint_array_{len(arrays):06d}"
            arrays[name] = np.ascontiguousarray(value)
            return {"private_array_member": name,
                    "array_sha256": sha_array(arrays[name])}
        if isinstance(value, JointPolicy):
            return {"private_checkpoint_policy_type": type(value).__name__,
                    "mechanism_arrays": [normalize(v) for v in
                                         value.mechanism_arrays()]}
        if isinstance(value, GateWeights) or \
                type(value).__module__ == \
                "results.pcrl_bank_anchored_release_v1.bank_anchored.r2_source_deterministic" \
                and type(value).__name__ == "GateWeights" or \
                type(value).__name__ == "FactorGate":
            names = ("w1", "b1", "w2", "b2")
            output = {name: normalize(getattr(value, name)) for name in names}
            if getattr(value, "embedding", None) is not None:
                output["embedding"] = normalize(value.embedding)
            return {"private_checkpoint_weights_type": type(value).__name__,
                    "arrays": output}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        if type(value) is dict and all(type(k) is str for k in value):
            return {k: normalize(v) for k, v in value.items()}
        raise TypeError(f"unsupported checkpoint receipt value {type(value).__name__}")

    public = normalize(record)
    path = root / "CHECKPOINTS.npz"
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600),
                   "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    return public, {"checkpoint_archive_sha256": _sha(path),
                    "checkpoint_array_count": len(arrays)}


def _utility(law, policy, labels):
    mass = np.asarray(law["mass"])
    n = len(mass)
    if mass.shape != (n, 16):
        raise ValueError("FIT law shape differs")
    out = {}
    for r, table, y in ((1, policy.bank1.probability, labels[0]),
                        (2, policy.bank2.probability, labels[1])):
        ids = law[f"code_r{r}"]
        b0ids = law[f"b0_code_r{r}"]
        p = table[ids]
        truth = np.eye(table.shape[1], dtype=np.float64)[y]
        ptrue = np.take_along_axis(p, y[:, None, None], axis=2)[:, :, 0]
        nll = np.sum(mass * (-np.log(np.maximum(ptrue, 1e-12))), axis=1)
        brier = np.sum(mass * np.sum((p-truth[:, None, :])**2, axis=2), axis=1)
        accuracy = np.sum(mass * (p.argmax(axis=2) == y[:, None]), axis=1)
        b0p = table[b0ids]
        b0_nll = -np.log(np.maximum(b0p[np.arange(n), y], 1e-12))
        b0_brier = np.sum((b0p-truth)**2, axis=1)
        b0_accuracy = (b0p.argmax(axis=1) == y).astype(np.float64)
        for name, candidate, reference, cap, sign in (
            ("accuracy_loss", accuracy, b0_accuracy, .002, -1),
            ("nll_excess", nll, b0_nll, .001, 1),
            ("brier_excess", brier, b0_brier, .0005, 1)):
            value = float(np.mean(sign*(candidate-reference)))
            out[f"r{r}_{name}"] = value
            out[f"r{r}_{name}_cap"] = cap
            if value > cap + 1e-12:
                raise RuntimeError("selected public law violates bank-relative FIT proxy")
    return out


def _admitted_source(cfg, registered):
    pins = {key: _source(value, registered) for key, value in cfg["pins"].items()}
    lock = json.loads(pins["campaign_lock"].read_text())
    if (lock.get("schema") != "pcrl-bank-anchored-campaign-lock-v1" or
            _sha(pins["campaign_lock"]) != cfg["source_lock_sha256"] or
            lock.get("selected_bank") != "W_TSTAR" or
            _sha(pins["protection_amendment"]) != cfg["protection_amendment_sha256"]):
        raise PermissionError("campaign/protection source seal differs")
    identity = json.loads(pins["identity_receipt"].read_text())
    if (identity.get("schema") != "BAR1-r2-full-fit-identity-v1" or
            identity.get("status") != "PASS" or
            identity.get("source_seed") != cfg["source_seed"] or
            identity.get("source_lock_sha256") != cfg["source_lock_sha256"]):
        raise PermissionError("full-FIT channel identity gate absent")
    roots = json.loads(pins["stream_roots"].read_text())
    stream_manifest = json.loads(pins["stream_manifest"].read_text())
    if (stream_manifest.get("schema") != "scv1-public-stream-root-hashes-v1" or
            stream_manifest.get("private_file_sha256") != _sha(pins["stream_roots"]) or
            stream_manifest.get("release_realization_root_sha256") !=
            {str(k): v for k, v in enumerate(lock["sampling_root_hashes"]["joint_SCv1"]) }):
        raise PermissionError("public HMAC root manifest differs from sealed roots")
    if (roots.get("schema") != "scv1-private-stream-roots-v1" or
            _sha(pins["stream_roots"]) !=
            lock["sampling_root_hashes"]["private_roots_source_sha256"] or
            set(roots.get("release_realization_roots_hex", {})) != {"0", "1", "2"}):
        raise PermissionError("frozen HMAC roots differ from source lock")
    release_roots = {}
    for index, hexvalue in roots["release_realization_roots_hex"].items():
        raw = bytes.fromhex(hexvalue)
        if (len(raw) != 32 or hashlib.sha256(raw).hexdigest() !=
                lock["sampling_root_hashes"]["joint_SCv1"][int(index)]):
            raise PermissionError("frozen release root hash differs")
        release_roots[int(index)] = raw
    seq_root = bytes.fromhex(roots["seq_fit_root_hex"])
    if (len(seq_root) != 32 or hashlib.sha256(seq_root).hexdigest() !=
            stream_manifest["seq_fit_root_sha256"]):
        raise PermissionError("sequential FIT root differs from public commitment")
    return pins, identity, release_roots, seq_root


def _require_role_file_pins(roles, source, registered, *, protected):
    """Demand queue byte pins before any role-specific per-person archive read."""
    requests = [("OSF_DEFENSE_FIT", f"references_s{source.seed}"),
                ("OSF_DEFENSE_FIT", "income"),
                ("OSF_DEFENSE_FIT", "occupation"),
                ("ATTACK_FIT_NEW", f"references_s{source.seed}"),
                ("INNER_SELECTION", f"references_s{source.seed}")]
    if protected:
        requests.append(("OSF_DEFENSE_FIT", "sex"))
    expected = {}
    for role, field in requests:
        key = f"{role}:{field}"
        relative = roles.manifest["role_file_relative_path"][key]
        expected[(roles.package_root / relative).resolve(strict=True)] = \
            roles.manifest["role_file_sha256"][key]
    for name in ("bank_r1.npz", "bank_r2.npz",
                 "tokens_OSF_DEFENSE_FIT.npz", "tokens_ATTACK_FIT_NEW.npz",
                 "tokens_INNER_SELECTION.npz"):
        expected[(source.root / name).resolve(strict=True)] = \
            source.receipt["private_output_sha256"][name]
    expected[(source.root / "BANK_ADMISSION_RECEIPT.json").resolve(strict=True)] = \
        source.audit["tstar_banks"][str(source.seed)]["receipt_sha256"]
    if any(registered.get(path) != digest for path, digest in expected.items()):
        raise PermissionError("role-only source archive absent from queue byte pins")


def run(config_path):
    active = assert_active_from_env(stage="protection")
    queue = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    unit = queue["units"][active["unit_id"]]
    registered = {Path(item["path"]).resolve(strict=True): item["sha256"]
                  for item in unit["spec"]["sources"]}
    config_path = Path(config_path).resolve(strict=True)
    if registered.get(config_path) != _sha(config_path):
        raise PermissionError("FIT config is not byte-pinned by queue")
    cfg = json.loads(config_path.read_text())
    if (set(cfg) != CONFIG_KEYS or set(cfg["pins"]) != PIN_KEYS or
            cfg.get("schema") != "BAR1-r2-protection-job-v1" or
            cfg["unit_id"] != active["unit_id"] or
            cfg["arm_id"] not in ALL_ARMS or cfg["source_seed"] not in (0, 1) or
            type(cfg["recipe_seed"]) is not int or cfg["recipe_seed"] < 0 or
            type(cfg["cpu_seconds"]) not in (int, float) or
            not 0 < cfg["cpu_seconds"] <= 300 or
            cfg["source_lock_sha256"] != active["seal_sha256"] or
            unit["spec"].get("amendment_sha256") !=
            cfg["protection_amendment_sha256"]):
        raise PermissionError("protection arm/seed/config differs from sealed queue")
    if cfg["arm_id"] == "B0" and cfg["cpu_seconds"] != 300:
        raise PermissionError("B0 keeps the common arm ceiling in the registry")
    pins, identity_receipt, release_roots, seq_root = _admitted_source(cfg, registered)
    if pins["worker_source"] != Path(__file__).resolve():
        raise PermissionError("worker source path differs")
    # A separate R0 amendment and milestone must precede all FIT label access.
    if "channel_identity_pass" not in queue["milestones"] or \
            "witness_pass" not in queue["milestones"]:
        raise PermissionError("full identity and bank witness milestones absent")
    output = Path(cfg["private_output_dir"])
    expected_root = Path(__file__).resolve().parents[1] / "r2/.private/units"
    if (output.parent != expected_root or output.name != cfg["unit_id"] or
            output.exists() or output.is_symlink() or
            stat.S_IMODE(expected_root.stat().st_mode) & 0o077):
        raise PermissionError("FIT output must be a new R2 owner-only unit directory")
    public_root = Path(__file__).resolve().parents[1] / "r2"
    for key, suffix in (("fit_receipt", "FIT_RECEIPT.json"),
                        ("unit_receipt", "UNIT_RECEIPT.json")):
        dest = Path(cfg[key])
        if dest.parent != public_root or dest.name != f"{cfg['unit_id']}_{suffix}" or \
                dest.exists() or dest.is_symlink():
            raise PermissionError("public FIT receipt path differs from R2 namespace")
    roles = RoleOnlyStore(pins["role_manifest"])
    source = TStarSourceStore(pins["portfolio_manifest"].parent,
                              pins["role_manifest"], pins["source_audit"],
                              cfg["source_seed"])
    _require_role_file_pins(roles, source, registered,
                            protected=cfg["arm_id"] not in ("B0", "T"))
    bank_action = []
    for r in (1, 2):
        bank = source.read_bank(r)
        bank_action.append(wire_fixed_bank_and_actions(
            bank["probability"], bank["token_class"],
            bank["fine_to_hcal_token"], bank["input_class"]))
    base = JointPolicy(bank_action[0][0], bank_action[1][0],
                       bank_action[0][1], bank_action[1][1],
                       GateWeights.zeros(input_width=12))
    if source.source_binding_sha256 != identity_receipt["source_binding_sha256"]:
        raise PermissionError("full FIT source binding differs from identity gate")
    role = "OSF_DEFENSE_FIT"
    cells = source.read_role_cells(role)
    references = roles.read(role, f"references_s{cfg['source_seed']}")
    income = roles.read(role, "income", "task_fit")
    occupation = roles.read(role, "occupation", "task_fit")
    assert_same_people(cells, references, income, occupation)
    task = TaskOnlyData(references["U0_1"], references["U0_2"],
                        cells["cell_1"], cells["cell_2"],
                        income["income"], occupation["occupation"])
    prepare_law(task, base)
    for r in (1, 2):
        if not np.array_equal(cells[f"token_{r}"],
                              np.array([bank_action[r-1][1][int(c)][0]
                                        for c in cells[f"cell_{r}"]], dtype=np.int64)):
            raise PermissionError("FIT source identity token differs")
    capability = task
    if cfg["arm_id"] not in ("B0", "T"):
        sex = roles.read(role, "sex", "defense_fit")
        assert_same_people(cells, sex)
        capability = ProtectedData(task, sex["sex"])
    output.mkdir(mode=0o700)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(mode=0o700)
    assert_active_from_env(unit_id=cfg["unit_id"], stage="protection")
    if cfg["arm_id"] == "B0":
        selected = IdentityPolicy(base)
        summary = {"arm_id": "B0", "status": "COMPLETE_IDENTITY_CONTROL",
                   "selection": {"start": "exact_identity", "update": 0,
                                 "identity_alias": True},
                   "source": {"complete": True, "starts": [], "updates": 0}}
    else:
        selected, summary = fit_arm(
            cfg["arm_id"], capability, base, source_seed=cfg["source_seed"],
            recipe_seed=cfg["recipe_seed"], cpu_seconds=cfg["cpu_seconds"],
            unit_ids=list(map(int, cells["unit"])),
            source_binding_sha256=source.source_binding_sha256,
            seq_private_root=seq_root,
            seq_private_cache_path=output / "SEQUENTIAL_FIRST_FIT_CODES.json")
    public_summary, checkpoint = _checkpoint_archive(checkpoint_dir, summary)
    if selected is None:
        fit = {"schema": "BAR1-r2-fit-receipt-v1", "unit_id": cfg["unit_id"],
               "arm_id": cfg["arm_id"], "source_seed": cfg["source_seed"],
               "status": "INCOMPLETE_SEARCH", "source_lock_sha256": active["seal_sha256"],
               "source_binding_sha256": source.source_binding_sha256,
               "fit_rows": len(task.y1), "fit_groups": len(np.unique(cells["unit"])),
               "checkpoint": checkpoint, "fit_summary": public_summary}
        fit_sha = _new_json(cfg["fit_receipt"], fit, private=False)
        unit_receipt = {"schema": "BAR1-r2-unit-receipt-v1", "unit_id": cfg["unit_id"],
                        "status": "INCOMPLETE_SEARCH", "fit_receipt_sha256": fit_sha,
                        "policy_receipt_sha256": None, "role_releases": []}
        _new_json(cfg["unit_receipt"], unit_receipt, private=False)
        return unit_receipt
    fit_role = output / "OSF_DEFENSE_FIT_R0"
    fit_role.mkdir(mode=0o700)
    fit_law, fit_codes = materialize_arrays(
        selected, row_id=cells["row_id"], unit=cells["unit"],
        score1=references["U0_1"], score2=references["U0_2"],
        cell1=cells["cell_1"], cell2=cells["cell_2"],
        source_seed=cfg["source_seed"], role=role, realization=0,
        private_root=release_roots[0])
    fit_utility = _utility(fit_law, selected, (task.y1, task.y2))
    fit = {"schema": "BAR1-r2-fit-receipt-v1", "unit_id": cfg["unit_id"],
           "arm_id": cfg["arm_id"], "source_seed": cfg["source_seed"],
           "status": summary["status"], "source_lock_sha256": active["seal_sha256"],
           "source_binding_sha256": source.source_binding_sha256,
           "fit_rows": len(task.y1), "fit_groups": len(np.unique(cells["unit"])),
           "checkpoint": checkpoint, "fit_utility": fit_utility,
           "fit_summary": public_summary,
           "fit_row_unit_sha256": sha_array(np.column_stack((cells["row_id"],
                                                                cells["unit"])))}
    fit_sha = _new_json(cfg["fit_receipt"], fit, private=False)
    policy_dir = output / "policy"
    policy_dir.mkdir(mode=0o700)
    choice = summary.get("selection") or {}
    policy_receipt = save_policy(
        policy_dir, selected, arm_id=cfg["arm_id"], source_seed=cfg["source_seed"],
        source_binding_sha256=source.source_binding_sha256,
        fit_receipt_sha256=fit_sha,
        selected_start=str(choice.get("start", "source_selected")),
        selected_update=int(choice.get("update", 0)))
    outputs = []
    fit_release = write_materialization(
        fit_role, fit_law, fit_codes, arm_id=cfg["arm_id"],
        source_seed=cfg["source_seed"], role=role, realization=0,
        source_binding_sha256=source.source_binding_sha256,
        policy_receipt_sha256=policy_receipt["receipt_sha256"])
    outputs.append({"role": role, "realization": 0,
                    "law_sha256": fit_release["law_sha256"],
                    "release_codes_sha256": fit_release["release_codes_sha256"],
                    "receipt_sha256": fit_release["receipt_sha256"],
                    "private_relative_dir": str(fit_role.relative_to(output))})
    # Selection is already frozen; ATT/INNER labels are never opened here.
    for release_role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        assert_active_from_env(unit_id=cfg["unit_id"], stage="protection")
        role_cells = source.read_role_cells(release_role)
        role_refs = roles.read(release_role, f"references_s{cfg['source_seed']}")
        assert_same_people(role_cells, role_refs)
        for realization in (0, 1, 2):
            dest = output / f"{release_role}_R{realization}"
            dest.mkdir(mode=0o700)
            law, release = materialize_arrays(
                selected, row_id=role_cells["row_id"], unit=role_cells["unit"],
                score1=role_refs["U0_1"], score2=role_refs["U0_2"],
                cell1=role_cells["cell_1"], cell2=role_cells["cell_2"],
                source_seed=cfg["source_seed"], role=release_role,
                realization=realization, private_root=release_roots[realization])
            receipt = write_materialization(
                dest, law, release, arm_id=cfg["arm_id"],
                source_seed=cfg["source_seed"], role=release_role,
                realization=realization,
                source_binding_sha256=source.source_binding_sha256,
                policy_receipt_sha256=policy_receipt["receipt_sha256"])
            outputs.append({"role": release_role, "realization": realization,
                            "law_sha256": receipt["law_sha256"],
                            "release_codes_sha256": receipt["release_codes_sha256"],
                            "receipt_sha256": receipt["receipt_sha256"],
                            "private_relative_dir": str(dest.relative_to(output))})
    unit_receipt = {"schema": "BAR1-r2-unit-receipt-v1", "unit_id": cfg["unit_id"],
                    "arm_id": cfg["arm_id"], "source_seed": cfg["source_seed"],
                    "status": summary["status"],
                    "source_lock_sha256": active["seal_sha256"],
                    "fit_receipt_sha256": fit_sha,
                    "policy_receipt_sha256": policy_receipt["receipt_sha256"],
                    "source_binding_sha256": source.source_binding_sha256,
                    "sampling_contract": ("deterministic_no_random_coin" if
                                          cfg["arm_id"] in ("B0", "D") else
                                          "two_independent_recipient_coins" if
                                          cfg["arm_id"] in ("P", "L") else
                                          "one_pair_coin_per_person"),
                    "role_releases": outputs}
    _new_json(cfg["unit_receipt"], unit_receipt, private=False)
    return unit_receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().config), sort_keys=True))
