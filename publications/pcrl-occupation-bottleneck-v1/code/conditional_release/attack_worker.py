"""Queue-only reader fitting and separately authorized reporting workers.

Reader fitting never loads reporting SEX. The reporting worker requires the
coordinator's global all-reader freeze and exact single-role label archive.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .attacks import (fit_complete_readers, score_frozen_readers, save_private_npz,
                      write_private_json, sha_file, law_arrays)
from .scheduler import assert_active_from_env
from .source import (load_manifest, load_allowed_role, load_bank, load_historical_policy,
                     read_exact, write_once)

LAW_FIELDS = {"row_id", "unit", "code_r1", "code_r2", "mass"}


def _queue_config(config, stage):
    active = assert_active_from_env(stage=stage)
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    registered = {str(Path(item["path"]).resolve()):item["sha256"]
                  for item in state["units"][active["unit_id"]]["spec"]["sources"]}
    def verify(path, expected):
        p = Path(path)
        if p.is_symlink() or registered.get(str(p.resolve())) != expected or sha_file(p) != expected:
            raise PermissionError("artifact differs from active queue's exact sources")
        return p
    path = Path(config)
    verify(path, registered.get(str(path.resolve())))
    value = json.loads(path.read_text())
    if value["unit_id"] != active["unit_id"]:
        raise PermissionError("reader/report unit differs from active queue")
    return active, value, verify


def _policy(config, manifest):
    kind = config["type"]
    if kind == "P0":
        return None
    if kind == "historical":
        return load_historical_policy(manifest, config["seed"], config["source_arm"])
    if kind == "allocation":
        from .fit import load_allocation_policy
        return load_allocation_policy(config["directory"], manifest, expected_receipt_sha256=config["receipt_sha256"])
    if kind == "control":
        receipt=Path(config["directory"])/"POLICY_RECEIPT.json"
        if sha_file(receipt)!=config["receipt_sha256"]:
            raise PermissionError("control receipt differs before schema dispatch")
        schema=json.loads(receipt.read_text())["schema"]
        if schema=="CBR1-control-policy-v1":
            from .controls import load_control_policy
            return load_control_policy(config["directory"], manifest, expected_receipt_sha256=config["receipt_sha256"])
        if schema=="CBR1-wide-control-policy-v1":
            from .control_wide_policy import load_wide_control_policy
            return load_wide_control_policy(config["directory"], manifest, expected_receipt_sha256=config["receipt_sha256"])
        raise PermissionError("unregistered control package schema")
    if kind == "template":
        from .secondary import load_template_policy
        return load_template_policy(config["directory"], manifest, expected_receipt_sha256=config["receipt_sha256"])
    raise PermissionError("unregistered public kernel type")


def public_uniform_prior(bank, policy):
    """Exact public computation aid; no role array or private empirical prior."""
    b1, b2 = (np.asarray(bank["recipients"][str(r)]["probability"]) for r in (1, 2))
    c1 = np.repeat(np.arange(len(b1), dtype=np.int64), len(b2))
    c2 = np.tile(np.arange(len(b2), dtype=np.int64), len(b1))
    if policy is None:
        e1, e2, mass = c1[:, None], c2[:, None], np.ones((len(c1), 1))
    elif hasattr(policy, "law_codes"):
        e1, e2, mass = policy.law_codes(c1, c2)
    else:
        prepared = policy.space.prepare(c1, c2)
        e1, e2 = prepared["code_r1"], prepared["code_r2"]
        mass = policy.law_prepared(prepared)
    dummy = {"row_id":np.arange(len(c1), dtype=np.int64), "unit":np.arange(len(c1), dtype=np.int64),
             "code_r1":e1, "code_r2":e2, "mass":mass}
    law_arrays(dummy, bank)
    joint = np.zeros((len(b1), len(b2)), dtype=np.float64)
    np.add.at(joint, (e1.ravel(), e2.ravel()), mass.ravel() / len(c1))
    return joint


def fit_worker(config):
    active, cfg, verify = _queue_config(config, "attacks")
    science = verify(cfg["science_lock"], cfg["science_lock_sha256"])
    attack = verify(cfg["attack_lock"], cfg["attack_lock_sha256"])
    lock = json.loads(attack.read_text())
    expected = {"arm_id":cfg["arm_id"], "source_seed":cfg["seed"], "point_eligible":True}
    if (lock.get("schema") != "CBR1-final-reader-stage-lock-v1" or
            lock.get("status") != "RELEASES_AND_ENDPOINTS_FROZEN" or lock.get("K") != 552 or
            lock.get("science_lock_sha256") != cfg["science_lock_sha256"] or
            lock.get("source_manifest_sha256") != cfg["source_manifest_sha256"] or
            lock.get("units", {}).get(active["unit_id"]) != expected):
        raise PermissionError("release/endpoint stage freeze does not admit reader unit")
    manifest_path = verify(cfg["source_manifest"], cfg["source_manifest_sha256"])
    manifest = load_manifest(manifest_path, cfg["source_manifest_sha256"])
    bank = load_bank(manifest, cfg["seed"])
    for item in cfg.get("policy_pins", []):
        verify(item["path"], item["sha256"])
    verify(cfg["law"]["path"], cfg["law"]["sha256"])
    law = read_exact(cfg["law"], LAW_FIELDS)
    source = load_allowed_role(manifest, cfg["seed"], "ATTACK_FIT_NEW", fields=("sex",))
    for name in ("row_id", "unit"):
        if not np.array_equal(law[name], source["wire"][name]):
            raise PermissionError("attacked law and admitted ATT people differ")
    policy = _policy(cfg["policy"], manifest)
    public_prior = public_uniform_prior(bank, policy)
    if policy is None:
        e1, e2 = source["wire"]["code_r1"][:, None], source["wire"]["code_r2"][:, None]
        mass = np.ones_like(e1, dtype=float)
    elif hasattr(policy, "law_codes"):
        e1, e2, mass = policy.law_codes(source["wire"]["code_r1"], source["wire"]["code_r2"])
    else:
        p = policy.space.prepare(source["wire"]["code_r1"], source["wire"]["code_r2"])
        e1, e2, mass = p["code_r1"], p["code_r2"], policy.law_prepared(p)
    # Exact shape/action ordering is part of the private finite-law export.
    if not np.array_equal(e1, law["code_r1"]) or not np.array_equal(e2, law["code_r2"]) or not np.allclose(mass, law["mass"], atol=1e-13, rtol=0):
        raise PermissionError("attacked law differs from deployed selected kernel")
    def guard():
        assert_active_from_env(unit_id=active["unit_id"], stage="attacks")
        verify(science, cfg["science_lock_sha256"]); verify(attack, cfg["attack_lock_sha256"])
    frozen = fit_complete_readers(bank, law, source["sex"], uniform_public_pair_mass=public_prior,
        seed=cfg["seed"], private_root=cfg["private_output"], queue_guard=guard,
        source_bindings={"law_sha256":cfg["law"]["sha256"], "source_manifest_sha256":cfg["source_manifest_sha256"],
                         "science_lock_sha256":cfg["science_lock_sha256"], "attack_lock_sha256":cfg["attack_lock_sha256"]})
    result = {**frozen, "unit_id":active["unit_id"], "arm_id":cfg["arm_id"],
              "frozen_complete_sha256":sha_file(Path(cfg["private_output"]) / "FROZEN_COMPLETE.json"),
              "inner_sex_access":False, "old_assessment_access":False}
    write_once(cfg["public_output"], result, False)
    print(json.dumps({"status":"FROZEN_COMPLETE", "unit_id":active["unit_id"], "fits":30}))


def reporting_sex(pin, *, verify, source_wire):
    path = verify(pin["path"], pin["sha256"])
    if ".private" not in path.parts or path.stat().st_mode & 0o077:
        raise PermissionError("reporting SEX requires admitted owner-only one-role archive")
    with ZipFile(path) as archive:
        if {entry.filename for entry in archive.infolist()} != {"row_id.npy", "unit.npy", "sex.npy"} or len(archive.infolist()) != 3:
            raise PermissionError("reporting label archive includes unadmitted fields")
    with np.load(path, allow_pickle=False) as archive:
        values = {key:archive[key].copy() for key in ("row_id", "unit", "sex")}
    if any(values[key].dtype != np.int64 for key in values) or values["sex"].ndim != 1 or not np.isin(values["sex"], (0, 1)).all():
        raise PermissionError("reporting archive integer/binary schema differs")
    for name in ("row_id", "unit"):
        if not np.array_equal(source_wire[name], values[name]):
            raise PermissionError("reporting SEX people differ from source INNER")
    if len(values["sex"]) != 2235 or len(np.unique(values["unit"])) != 2234:
        raise PermissionError("reporting role count differs")
    return values


def report_worker(config):
    active, cfg, verify = _queue_config(config, "attacks")
    lock_path = verify(cfg["reporting_lock"], cfg["reporting_lock_sha256"])
    lock = json.loads(lock_path.read_text())
    if (lock.get("schema") != "CBR1-final-reporting-stage-lock-v1" or lock.get("status") != "ALL_READERS_FROZEN" or
            lock.get("K") != 552 or lock.get("source_manifest_sha256") != cfg["source_manifest_sha256"] or
            lock.get("inner_sex_sha256") != cfg["inner_sex"]["sha256"]):
        raise PermissionError("global final-reader/endpoint freeze is absent")
    # Verify EVERY frozen reader receipt before the first INNER SEX byte read.
    for item in lock["reader_freezes"]:
        verify(item["path"], item["sha256"])
        if json.loads(Path(item["path"]).read_text()).get("status") != "FROZEN_COMPLETE":
            raise PermissionError("a required final reader bank is not frozen")
    manifest_path = verify(cfg["source_manifest"], cfg["source_manifest_sha256"])
    manifest = load_manifest(manifest_path, cfg["source_manifest_sha256"])
    source = load_allowed_role(manifest, cfg["seed"], "INNER_SELECTION")
    bank = load_bank(manifest, cfg["seed"])
    for pin in (cfg["law"], cfg["frozen_receipt"], cfg["score_tables"]):
        verify(pin["path"], pin["sha256"])
    frozen = json.loads(Path(cfg["frozen_receipt"]["path"]).read_text())
    if frozen["score_table_sha256"] != cfg["score_tables"]["sha256"] or frozen["source_seed"] != cfg["seed"]:
        raise PermissionError("frozen reader bank/table binding differs")
    law = read_exact(cfg["law"], LAW_FIELDS)
    with np.load(cfg["score_tables"]["path"], allow_pickle=False) as archive:
        tables = {key:archive[key].copy() for key in archive.files}
    # This is the sole label-opening boundary and follows all global freezes.
    labels = reporting_sex(cfg["inner_sex"], verify=verify, source_wire=source["wire"])
    scores, per_person_laws = score_frozen_readers(bank, law, labels, tables)
    root = Path(cfg["private_output"])
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    arrays = {"row_id":law["row_id"], "unit":law["unit"], "mass":law["mass"], "sex":labels["sex"]}
    for view, readers in per_person_laws.items():
        arrays.update({view + "__" + name:value for name, value in readers.items()})
    save_private_npz(root / "EXPECTED_SCORE_LAWS.npz", **arrays)
    public = {"schema":"CBR1-final-integrated-reader-report-v1", "status":"COMPLETE",
        "unit_id":active["unit_id"], "arm_id":cfg["arm_id"], "source_seed":cfg["seed"],
        "reporting_lock_sha256":cfg["reporting_lock_sha256"], "frozen_receipt_sha256":cfg["frozen_receipt"]["sha256"],
        "law_sha256":cfg["law"]["sha256"], "expected_score_laws_sha256":sha_file(root / "EXPECTED_SCORE_LAWS.npz"),
        "reader_fits":0, "exact_single_query_integration":True, **scores}
    write_once(cfg["public_output"], public, False)
    print(json.dumps({"status":"COMPLETE","unit_id":active["unit_id"],"all_controls_pass":scores["all_controls_pass"]}))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--phase", choices=("fit", "report"), required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    (fit_worker if args.phase == "fit" else report_worker)(args.config)


if __name__ == "__main__":
    main()
