"""Queue-only prospective K552 primary endpoint integration and inference.

The coordinator supplies a presealed explicit endpoint list. NA dependencies
retain their slots and the same K. This worker does not choose a candidate,
reader, orientation, control pool or comparison after seeing reporting SEX.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .attack_worker import _queue_config
from .attacks import save_private_npz
from .inference import (paired_group_counts, envelope_distribution, mean_distribution,
                        simultaneous_interval, upper_cap_status, lower_target_status, lower_strict_status)
from .scheduler import assert_active_from_env
from .source import write_once


def run(config):
    active, cfg, verify = _queue_config(config, "attacks")
    frozen = verify(cfg["endpoint_family"]["path"], cfg["endpoint_family"]["sha256"])
    family = json.loads(frozen.read_text())
    if family.get("K") != 552 or len(family.get("endpoints", [])) != 552 or len({item["endpoint_id"] for item in family["endpoints"]}) != 552:
        raise PermissionError("exact prospective primary endpoint family differs")
    reporting_lock = verify(cfg["reporting_lock"], cfg["reporting_lock_sha256"])
    reporting = json.loads(reporting_lock.read_text())
    if reporting.get("status") != "ALL_READERS_FROZEN" or reporting.get("K") != 552:
        raise PermissionError("global reader/reporting freeze missing")
    # Inputs are postfreeze score-law exports; every label byte was previously
    # admitted at the separately recorded report worker boundary.
    records = {}
    for item in cfg["recovery_inputs"]:
        pin = item["pin"];verify(pin["path"], pin["sha256"])
        with np.load(pin["path"], allow_pickle=False) as archive:
            # Retain only small alignment records across arms. A WIDE law may
            # have hundreds of action columns; all score arrays are streamed
            # one arm/seed/view at a time under the declared RSS ceiling.
            values = {key:archive[key].copy() for key in ("row_id", "unit", "sex")}
            values["score_law_archive"] = pin["path"]
        key = (item["arm_id"], item["seed"])
        if key in records:
            raise PermissionError("duplicate exact recovery identity")
        records[key] = values
    utilities = {}
    for item in cfg["utility_inputs"]:
        pin = item["pin"];verify(pin["path"], pin["sha256"])
        with np.load(pin["path"], allow_pickle=False) as archive:
            values = {key:archive[key].copy() for key in archive.files}
        utilities[(item["arm_id"], item["seed"])] = values
    if not records:
        raise PermissionError("at least fresh P0 reporting law is needed for primary inference")
    anchor = next(iter(records.values()))
    groups = anchor["unit"]
    if len(groups) != 2235 or len(np.unique(groups)) != 2234:
        raise PermissionError("reporting row/group count differs")
    for values in [*records.values(), *utilities.values()]:
        if not np.array_equal(values["unit"], groups) or not np.array_equal(values["row_id"], anchor["row_id"]):
            raise PermissionError("primary arms/seeds/utility people not paired")
        if "sex" in values and not np.array_equal(values["sex"], anchor["sex"]):
            raise PermissionError("source seeds do not share exact reporting labels")
    group_values, counts = paired_group_counts(groups)
    root = Path(cfg["private_output"])
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    save_private_npz(root / "PAIRED_BOOTSTRAP_COUNTS.npz", group_values=group_values, counts=counts)
    distributions = {}
    for (arm, seed), values in records.items():
        with np.load(values["score_law_archive"], allow_pickle=False) as archive:
            mass = archive["mass"].copy()
            for view in ("r1", "r2", "pair"):
                assert_active_from_env(unit_id=active["unit_id"], stage="attacks")
                prefix = view + "__"
                point, distribution, reader_points = -np.inf, np.full(len(counts), -np.inf), {}
                for name in archive.files:
                    if not name.startswith(prefix):
                        continue
                    reader = name[len(prefix):]
                    scores = archive[name]
                    partial = envelope_distribution({reader:scores}, mass, values["sex"], groups,
                        counts=counts, group_values=group_values, chunk=32)
                    point = max(point, partial["point"])
                    distribution = np.maximum(distribution, partial["bootstrap"])
                    reader_points.update(partial["reader_points"])
                    del scores
                if not reader_points:
                    raise PermissionError("frozen view has no declared reader scores")
                distributions[(arm, seed, view)] = {"point":point, "bootstrap":distribution, "reader_points":reader_points}
    means = {}
    def get_recovery(arm, seed, view):
        if seed == "mean":
            first, second = (distributions.get((arm, value, view)) for value in (0, 1))
            if first is None or second is None:
                return None
            return {"point":(first["point"] + second["point"]) / 2,
                    "bootstrap":(first["bootstrap"] + second["bootstrap"]) / 2}
        return distributions.get((arm, seed, view))
    rows, private_bootstrap = [], {}
    for item in family["endpoints"]:
        endpoint = item["endpoint_id"]
        row = dict(item)
        if item["kind"] == "utility":
            identity = (item["arm_id"], item["seed"])
            values = utilities.get(identity)
            key = (*identity, item["array_key"])
            if values is None or item["array_key"] not in values:
                row.update({"status":"UNTESTED_DEPENDENCY", "dependency":"admitted expected utility rows unavailable"})
                rows.append(row);continue
            if key not in means:
                means[key] = mean_distribution(values[item["array_key"]], groups, counts=counts, group_values=group_values)
            d = means[key]
        elif item["kind"] == "recovery":
            first = get_recovery(item["minuend"], item["seed"], item["view"])
            second = get_recovery(item["subtrahend"], item["seed"], item["view"])
            if first is None or second is None:
                row.update({"status":"UNTESTED_DEPENDENCY", "dependency":"a predeclared recovery comparator is unavailable"})
                rows.append(row);continue
            d = {"point":first["point"] - second["point"], "bootstrap":first["bootstrap"] - second["bootstrap"]}
        else:
            raise PermissionError("unknown frozen endpoint kind")
        interval = simultaneous_interval(d["point"], d["bootstrap"], K=552, reporting_groups=2234)
        direction = item["direction"]
        if direction == "upper_cap":
            status = upper_cap_status(interval, item["threshold"])
        elif direction == "lower_target":
            status = lower_target_status(interval, item["threshold"])
        elif direction == "lower_strict":
            status = lower_strict_status(interval, item["threshold"])
        else:
            raise PermissionError("unknown frozen endpoint boundary convention")
        row.update({**interval, "status":status})
        rows.append(row)
        private_bootstrap[endpoint] = d["bootstrap"]
    save_private_npz(root / "PRIMARY_BOOTSTRAP_ENDPOINTS.npz", **private_bootstrap)
    # Every original slot survives, including untriggered ORIGINAL/secondary
    # mechanisms or a missing required comparator. No scientific aliases are
    # guessed by prediction similarity.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    csv_path = Path(cfg["public_csv"])
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    receipt = {"schema":"CBR1-primary-exact-law-inference-v1", "status":"COMPLETE", "unit_id":active["unit_id"],
        "K":552, "primary_slots":len(rows), "evaluated_slots":len(private_bootstrap),
        "missing_slots":len(rows) - len(private_bootstrap), "bootstrap_replicates":2048,
        "bootstrap_seed":2026101105, "reporting_rows":2235, "reporting_groups":2234, "df":2233,
        "scope":"approximate conditional development; reader/kernel fixed; no confirmation or exact coverage claim",
        "primary_endpoint_sha256":__import__("hashlib").sha256(csv_path.read_bytes()).hexdigest(),
        "endpoint_family_sha256":cfg["endpoint_family"]["sha256"], "all_people_paired":True,
        "seed_summary":"max frozen-oriented reader slate per seed, then equal paired-seed mean",
        "reader_refits":0, "release_draw_monte_carlo_error":False,
        "status_counts":{status:sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})}}
    write_once(cfg["public_receipt"], receipt, False)
    print(json.dumps({"status":"COMPLETE", "K":552, "evaluated":len(private_bootstrap),"missing":receipt["missing_slots"]}))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
