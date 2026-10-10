"""Queue-only independent expected-cost replay for a selected public code law.

This module deliberately does not import R2 or R3 policy, scoring, or selection
code. It consumes their frozen artifacts as evidence and computes endpoints
again from the transmitted code table and role-specific task labels.
"""
import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
import json
from math import fsum, log
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env

LAW = {"row_id", "unit", "code_r1", "code_r2", "mass", "valid", "b0_code_r1", "b0_code_r2"}
WIRE = {"row_id", "unit", "code_r1", "code_r2", "prob_r1", "prob_r2", "decision_r1", "decision_r2"}
REF = {"row_id", "unit", "U0_1", "U0_2", "Ucal_1", "Ucal_2", "d_1", "d_2"}
DRAW = {"row_id", "unit", "code_r1", "code_r2"}


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def exact_npz(path, keys):
    with ZipFile(path) as z:
        if len(z.infolist()) != len(keys) or {v.filename for v in z.infolist()} != {k+".npy" for k in keys}:
            raise PermissionError("NPZ member allowlist differs before loading")
    with np.load(path, allow_pickle=False) as src:
        return {k: np.array(src[k], copy=True) for k in keys}


def score(p, label, metric, high=False):
    label = int(label)
    if metric == "accuracy":
        value = int(np.argmax(p) == label)
        return Decimal(value) if high else float(value)
    if high:
        v = [Decimal.from_float(float(x)) for x in p]
        if metric == "nll":
            return -max(v[label], Decimal.from_float(1e-12)).ln()
        return sum(((x-Decimal(int(j == label)))**2 for j, x in enumerate(v)), Decimal(0))
    if metric == "nll":
        return -log(max(float(p[label]), 1e-12))
    return fsum((float(x)-int(j == label))**2 for j, x in enumerate(p))


def expected_margin(law, bank, labels, reference, recipient, metric, *, high=False):
    """Paired mean candidate-reference, from the actual finite law per person."""
    code = law[f"code_r{recipient}"]
    mass = law["mass"]
    cache = {}
    delta = []
    for i in range(len(labels)):
        y = int(labels[i])
        def cost(global_code):
            key = (int(global_code), y)
            if key not in cache:
                cache[key] = score(bank[key[0]], y, metric, high)
            return cache[key]
        if high:
            candidate = sum((Decimal.from_float(float(mass[i, a]))*cost(code[i, a])
                             for a in np.flatnonzero(mass[i])), Decimal(0))
            comparator = score(reference[i], y, metric, True)
        else:
            candidate = fsum(float(mass[i, a])*cost(code[i, a]) for a in np.flatnonzero(mass[i]))
            comparator = score(reference[i], y, metric, False)
        delta.append(candidate-comparator)
    if high:
        return sum(delta, Decimal(0))/Decimal(len(delta))
    return fsum(delta)/len(delta)


def validate_law(law, wire, bank_doc, tables):
    n = len(wire["row_id"])
    if (law["mass"].dtype != np.float64 or law["valid"].dtype != np.bool_
            or law["mass"].shape != (n, 16) or law["valid"].shape != (n, 16)
            or not np.isfinite(law["mass"]).all() or np.any(law["mass"] < 0)
            or np.any(law["mass"][~law["valid"]] != 0)
            or not np.allclose(law["mass"].sum(1), 1, rtol=0, atol=1e-12)):
        raise PermissionError("selected 16-slot law is invalid")
    for key in ("row_id", "unit"):
        if not np.array_equal(law[key], wire[key]):
            raise PermissionError("law person order differs")
    for recipient in (1, 2):
        ids = law[f"code_r{recipient}"]
        base = law[f"b0_code_r{recipient}"]
        table = tables[recipient-1]
        if (ids.dtype != np.int64 or ids.shape != (n, 16) or np.any(ids < 0)
                or np.any(ids >= len(table)) or not np.array_equal(base, wire[f"code_r{recipient}"])
                or not np.array_equal(table[base], wire[f"prob_r{recipient}"])
                or not np.array_equal(table[base].argmax(-1), wire[f"decision_r{recipient}"])):
            raise PermissionError("law global ID, P0 reference or decoder differs")
        if np.any(law["valid"] & (table[ids].argmax(-1) != table[base].argmax(-1)[:, None])):
            raise PermissionError("selected action changes frozen decision")
        support = bank_doc["recipients"][str(recipient)]["support"]
        if len(support) != len(table):
            raise PermissionError("P0 action support inventory differs")
        permitted = np.array([[int(c) in support[int(source)] for c in ids[i]]
                              for i, source in enumerate(base)], dtype=np.bool_)
        if np.any(law["valid"] & ~permitted):
            raise PermissionError("selected law escapes frozen P0 same-class support")
    identity = (law["code_r1"] == law["b0_code_r1"][:, None]) & (law["code_r2"] == law["b0_code_r2"][:, None])
    if not np.all((law["valid"] & identity).any(1)):
        raise PermissionError("P0 identity absent from action support")
    if not np.all(law["valid"][:, 0] & identity[:, 0]):
        raise PermissionError("first action is not exact P0 identity")


def run(config_path):
    active = assert_active_from_env(stage="verification")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"])/"state.json").read_text())
    sources = {str(Path(x["path"]).resolve()): x["sha256"] for x in state["units"][active["unit_id"]]["spec"]["sources"]}
    def pin(path):
        path = Path(path)
        if path.is_symlink() or sources.get(str(path.resolve(strict=True))) != digest(path):
            raise PermissionError("artifact absent from exact verification source pins")
        return path
    cfg = json.loads(pin(config_path).read_text())
    if cfg.get("schema") != "IVR1-r5-selected-utility-replay-v1" or cfg["unit_id"] != active["unit_id"]:
        raise PermissionError("verifier unit differs")
    r3cfg = json.loads(pin(cfg["r3_config"]).read_text())
    role, arm, seed = r3cfg["role"], r3cfg["arm_id"], r3cfg["source_seed"]
    if (r3cfg.get("stage") != "witness" or role not in ("OSF_DEFENSE_FIT", "INNER_SELECTION")
            or arm not in ("O_J", "V_J", "V_P", "V_D", "V_L", "V_S12", "V_S21") or seed not in (0, 1)):
        raise PermissionError("selected protected scoring identity differs")
    for x in r3cfg["pins"].values():
        if digest(pin(x["path"])) != x["sha256"]:
            raise PermissionError("R3 original source bytes differ")
    pins = r3cfg["pins"]
    role_manifest = json.loads(pin(pins["role_manifest"]["path"]).read_text())
    if role_manifest.get("schema") != "pcrl-incumbent-r1-private-source-paths-v1":
        raise PermissionError("role-only source manifest differs")
    for task in ("income", "occupation"):
        allowed = role_manifest["roles"][role][task]
        if allowed["path"] != pins[task]["path"] or allowed["sha256"] != pins[task]["sha256"]:
            raise PermissionError("scored task labels outside admitted role-only export")
    allowed_ref = role_manifest["roles"][role][f"references_s{seed}"]
    if allowed_ref["path"] != pins["references"]["path"] or allowed_ref["sha256"] != pins["references"]["sha256"]:
        raise PermissionError("utility reference outside admitted role-only export")
    science_sha = pins["seal"]["sha256"]
    if not any(a["sha256"] == science_sha and "protection" in a["stages"]
               for a in state.get("amendments", [])):
        raise PermissionError("selected science lock was not registered for protected fits")
    policy_doc = json.loads(pin(pins["policy_receipt"]["path"]).read_text())
    if policy_doc.get("source_lock_sha256") != science_sha:
        raise PermissionError("selected policy and science lock differ")
    bank_doc = json.loads(pin(pins["public_bank"]["path"]).read_text())
    tables = [np.asarray(bank_doc["recipients"][str(r)]["probability"], dtype=np.float64) for r in (1, 2)]
    law = exact_npz(pin(pins["law"]["path"]), LAW)
    wire = exact_npz(pin(pins["p0_wire"]["path"]), WIRE)
    validate_law(law, wire, bank_doc, tables)
    refs = exact_npz(pin(pins["references"]["path"]), REF)
    labels = {task: exact_npz(pin(pins[task]["path"]), {"row_id", "unit", task}) for task in ("income", "occupation")}
    for obj in (refs, *labels.values()):
        if not np.array_equal(obj["row_id"], law["row_id"]) or not np.array_equal(obj["unit"], law["unit"]):
            raise PermissionError("task/reference rows differ from selected law")
    csv_path = pin(cfg["r3_csv"])
    receipt = json.loads(pin(cfg["r3_receipt"]).read_text())
    if (receipt["csv_sha256"] != digest(csv_path) or receipt["unit_id"] != r3cfg["unit_id"]
            or receipt["source_unit_id"] != r3cfg["source_unit_id"]
            or receipt["arm_id"] != arm or receipt["source_seed"] != seed or receipt["role"] != role
            or receipt["law_file_sha256"] != pins["law"]["sha256"]
            or receipt["policy_receipt_sha256"] != pins["policy_receipt"]["sha256"]):
        raise PermissionError("R3 reported utility source differs")
    with csv_path.open(newline="") as f:
        reported = list(csv.DictReader(f))
    primary = [x for x in reported if x["primary_gate"] == "True" and x["estimand"] == "exact_expected"]
    expected_n = 6 if role == "OSF_DEFENSE_FIT" else 12
    if len(primary) != expected_n:
        raise PermissionError("R3 primary clause inventory differs")
    result = {"schema": "IVR1-r5-independent-selected-utility-v1", "status": "PASS", "unit_id": active["unit_id"],
              "r3_unit_id": r3cfg["unit_id"], "arm_id": arm, "source_seed": seed, "role": role,
              "r3_csv_sha256": digest(csv_path), "r3_receipt_sha256": digest(cfg["r3_receipt"]),
              "source_law_sha256": digest(pins["law"]["path"]), "clauses": [], "max_float_gap": 0.,
              "max_decimal_error": 0., "label_scope": [role+":income", role+":occupation"],
              "sensitive_labels_read": False, "head_or_old_assessment_read": False}
    with localcontext() as context:
        context.prec = 80
        for row in primary:
            task, metric, reference = row["task"], row["metric"], row["reference"]
            recipient = 1 if task == "income" else 2
            if reference == "P0":
                comparator = tables[recipient-1][wire[f"code_r{recipient}"]]
            elif reference in ("U0", "Ucal") and role == "INNER_SELECTION":
                comparator = refs[f"{reference}_{recipient}"]
            else:
                raise PermissionError("unexpected primary utility reference")
            y = labels[task][task]
            got = expected_margin(law, tables[recipient-1], y, comparator, recipient, metric)
            high = expected_margin(law, tables[recipient-1], y, comparator, recipient, metric, high=True)
            prior = float(row["margin"])
            gap = abs(got-prior)
            numerical_error = float(abs(Decimal.from_float(got)-high))
            result["max_float_gap"] = max(result["max_float_gap"], gap)
            result["max_decimal_error"] = max(result["max_decimal_error"], numerical_error)
            cap = float(row["cap"])
            pass_ours = got >= cap if metric == "accuracy" else got <= cap
            pass_source = row["point_status"] == "PASS"
            source_gap = abs(prior-cap)
            boundary = pass_ours != pass_source and max(gap, source_gap) <= 1e-10
            if gap > 1e-10 or numerical_error > 1e-10 or (pass_ours != pass_source and not boundary):
                raise ArithmeticError("independent selected-law utility disagrees beyond prefit numerical envelope")
            result["clauses"].append({"task":task,"metric":metric,"reference":reference,
                 "independent_margin":got,"decimal80_margin":str(high),"source_margin":prior,
                 "cap":cap,"source_point_status":row["point_status"],"independent_point_status":"PASS" if pass_ours else "FAIL",
                 "boundary_status":"NUMERIC_BOUNDARY_UNRESOLVED" if boundary else "AGREE","gap":gap,"decimal_error":numerical_error})
    result["boundary_disagreements"] = sum(x["boundary_status"] != "AGREE" for x in result["clauses"])
    # The recipient sees one actual cached draw. Check every reported realized
    # cost against that exact code, without substituting a mean probability.
    draws = []
    for d in range(3):
        item = exact_npz(pin(pins[f"draw{d}"]["path"]), DRAW)
        if (not np.array_equal(item["row_id"], law["row_id"]) or
                not np.array_equal(item["unit"], law["unit"])):
            raise PermissionError("cached sampled release person order differs")
        matching = ((law["code_r1"] == item["code_r1"][:, None]) &
                    (law["code_r2"] == item["code_r2"][:, None]) & (law["mass"] > 0))
        if not matching.any(1).all():
            raise PermissionError("realized public release outside positive law")
        draws.append(item)
    actual_rows = [x for x in reported if x["estimand"] == "actual_single_draw"]
    if len(actual_rows) != 54:
        raise PermissionError("three cached draws and all realized task/reference clauses required")
    result["actual_draw_rows"] = len(actual_rows)
    result["actual_draw_max_gap"] = 0.
    for row in actual_rows:
        d = int(row["realization"])
        task, metric, reference = row["task"], row["metric"], row["reference"]
        if d not in (0, 1, 2) or task not in labels or metric not in ("accuracy", "nll", "brier"):
            raise PermissionError("realized endpoint identity differs")
        recipient = 1 if task == "income" else 2
        if reference == "P0":
            comparator = tables[recipient-1][wire[f"code_r{recipient}"]]
        elif reference in ("U0", "Ucal"):
            comparator = refs[f"{reference}_{recipient}"]
        else:
            raise PermissionError("realized reference identity differs")
        y = labels[task][task]
        code = draws[d][f"code_r{recipient}"]
        delta = (score(tables[recipient-1][int(c)], z, metric)-score(p, z, metric)
                 for c, p, z in zip(code, comparator, y))
        got = fsum(delta)/len(y)
        gap = abs(got-float(row["margin"]))
        result["actual_draw_max_gap"] = max(result["actual_draw_max_gap"], gap)
        if gap > 1e-10:
            raise ArithmeticError("actual sampled-output cost differs from R3 report")
    out = Path(cfg["output"])
    with os.fdopen(os.open(out, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o644), "w") as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False);f.write("\n")
    return result


def main():
    p = argparse.ArgumentParser();p.add_argument("--config", required=True);a = p.parse_args()
    print(json.dumps({k:v for k,v in run(a.config).items() if k in ("status","unit_id","max_float_gap","max_decimal_error","boundary_disagreements")}))


if __name__ == "__main__":
    main()
