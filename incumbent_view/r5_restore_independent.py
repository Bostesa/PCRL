"""Queue-only independent snapshot replay of P0 routing, policy draws, and readers.

This reads only the versioned restore and contains its own endpoint and sampling
arithmetic. It imports the frozen teacher architecture, not R1/R2 deployment,
endpoint, attack, or selection implementations. No task or SEX label is read.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import torch
from jcv.train import Model

from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env


ARMS = ("P0", "V_L", "V_J", "V_S21")
N = 15434
SAMPLE = np.linspace(0, N - 1, 32, dtype=np.int64)
WIRE_SCHEMA = "SCV1-reference-score-release-v1"
P0_KEYS = {"row_id", "unit", "code_r1", "code_r2", "prob_r1", "prob_r2",
           "decision_r1", "decision_r2"}
CODE_KEYS = {"row_id", "unit", "code_r1", "code_r2"}


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def exact_npz(path: Path, names: set[str]) -> dict[str, np.ndarray]:
    with ZipFile(path) as archive:
        actual = [x.filename for x in archive.infolist()]
        if len(actual) != len(names) or set(actual) != {x + ".npy" for x in names}:
            raise PermissionError("restored role archive has forbidden or missing member")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in names}


def entropy_margin(p: np.ndarray) -> np.ndarray:
    ordered = np.sort(p, axis=1)
    entropy = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
    return np.column_stack((entropy, ordered[:, -1] - ordered[:, -2]))


def features(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return np.column_stack((p1, p2, entropy_margin(p1), entropy_margin(p2)))


def gate(x: np.ndarray, arrays: dict[str, np.ndarray], i: int,
         valid: np.ndarray) -> np.ndarray:
    w = [arrays[f"gate{i}_{j}"] for j in range(4)]
    logits = np.tanh(x @ w[0] + w[1]) @ w[2] + w[3]
    masked = np.where(valid, logits, -np.inf)
    mass = np.exp(masked - masked.max(axis=1, keepdims=True))
    return mass / mass.sum(axis=1, keepdims=True)


def mix(q: np.ndarray, eta: float) -> np.ndarray:
    out = q * eta
    out[:, 0] += 1.0 - eta
    return out


def cdf_index(mass: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    index = (np.cumsum(mass, axis=1) <= uniform[:, None]).sum(axis=1).clip(0, mass.shape[1] - 1)
    bad = mass[np.arange(len(index)), index] <= 0
    if bad.any():
        index[bad] = np.asarray([np.flatnonzero(row > 0)[-1] for row in mass[bad]])
    return index.astype(np.int64)


def independent_law(arm: str, arrays: dict[str, np.ndarray],
                    c1: np.ndarray, c2: np.ndarray, eta1: float, eta2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p1, p2 = arrays["probability1"][c1], arrays["probability2"][c2]
    a1, a2 = arrays["actions1"][c1], arrays["actions2"][c2]
    v1, v2 = arrays["valid1"][c1], arrays["valid2"][c2]
    codes1 = np.repeat(a1, 4, axis=1)
    codes2 = np.tile(a2, (1, 4))
    valid = np.repeat(v1, 4, axis=1) & np.tile(v2, (1, 4))
    if arm == "V_J":
        base = gate(features(p1, p2), arrays, 0, valid)
        q = mix(base, eta1)
    elif arm == "V_L":
        left = mix(gate(np.column_stack((p1, entropy_margin(p1))), arrays, 0, v1), eta1)
        right = mix(gate(np.column_stack((p2, entropy_margin(p2))), arrays, 1, v2), eta2)
        q = (left[:, :, None] * right[:, None, :]).reshape(-1, 16)
    elif arm == "V_S21":
        first = mix(gate(features(p1, p2), arrays, 0, v2), eta1)
        pair = np.zeros((len(c1), 4, 4), dtype=np.float64)
        for i in range(4):
            emitted = arrays["probability2"][a2[:, i]]
            later = mix(gate(features(p1, emitted), arrays, 1, v1), eta2)
            pair[:, :, i] = later * first[:, i, None]
        q = pair.reshape(-1, 16)
    else:
        raise ValueError("unregistered snapshot-selected arm")
    if (not np.isfinite(q).all() or np.any(q < 0) or np.any(q[~valid] != 0) or
            not np.allclose(q.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ArithmeticError("independent selected law violates actual support")
    return q, codes1, codes2


def uniform(root: bytes, seed: int, draw: int, unit: int, recipient: int | None) -> float:
    if recipient is None:
        message = ["SCV1-uniform-v1", seed, "OSF_DEFENSE_FIT", draw, unit]
    else:
        message = ["BAR1-factor-uniform-v1", seed, "OSF_DEFENSE_FIT", draw, unit, recipient]
    wire = json.dumps(message, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    integer = int.from_bytes(hmac.new(root, wire, hashlib.sha256).digest()[:8], "big")
    return min((integer + 0.5) / (1 << 64), np.nextafter(1.0, 0.0))


def sampled_codes(arm: str, arrays: dict[str, np.ndarray], c1: np.ndarray, c2: np.ndarray,
                  eta1: float, eta2: float, root: bytes, seed: int, draw: int,
                  unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, codes1, codes2 = independent_law(arm, arrays, c1, c2, eta1, eta2)
    if arm == "V_J":
        coins = np.asarray([uniform(root, seed, draw, int(u), None) for u in unit])
        slot = cdf_index(q, coins)
    else:
        left_coin = np.asarray([uniform(root, seed, draw, int(u), 1) for u in unit])
        right_coin = np.asarray([uniform(root, seed, draw, int(u), 2) for u in unit])
        pair = q.reshape(-1, 4, 4)
        if arm == "V_L":
            left = cdf_index(pair.sum(axis=2), left_coin)
            right = cdf_index(pair.sum(axis=1), right_coin)
        else:
            right_mass = pair.sum(axis=1)
            right = cdf_index(right_mass, right_coin)
            conditional = pair[np.arange(len(unit)), :, right] / right_mass[np.arange(len(unit)), right, None]
            left = cdf_index(conditional, left_coin)
        slot = 4 * left + right
    return codes1[np.arange(len(unit)), slot], codes2[np.arange(len(unit)), slot]


def route_p0(raw: np.ndarray, seed: int, teacher: dict[str, Path], policy: dict) -> tuple[np.ndarray, np.ndarray]:
    model = Model(83, [2, 6], seed)
    model.load_state_dict(torch.load(teacher["model"], map_location="cpu"))
    model.eval()
    heads = [joblib.load(teacher[f"head{i}"]) for i in (0, 1)]
    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(raw, dtype=np.float32))
        probabilities = [np.asarray(heads[i].predict_proba(model.encode(i, tensor).double().numpy()),
                                    dtype=np.float64) for i in (0, 1)]
    answer = []
    for r, prob in enumerate(probabilities, 1):
        pol = policy[f"p{r}"]
        classes = np.asarray(pol["fine"]["cell_class"], dtype=np.int64)
        centroid = np.asarray(pol["fine"]["centroid"], dtype=np.float64)
        cell_token = np.asarray(pol["cell_token"], dtype=np.int64)
        decision = prob.argmax(axis=1)
        code = np.full(len(prob), -1, dtype=np.int64)
        for c in range(prob.shape[1]):
            rows = np.flatnonzero(decision == c)
            if not len(rows):
                continue
            cells = np.flatnonzero(classes == c)
            if len(cells) == 1:
                code[rows] = cell_token[cells[0]]
                continue
            matrix = np.zeros((len(rows), len(cells)), dtype=np.float64)
            for k in range(prob.shape[1]):
                p = prob[rows, k:k + 1]
                logp = np.log(np.where(p > 0, p, 1.0))
                matrix = matrix + np.where(p > 0, p * (logp - np.log(centroid[cells, k])[None, :]), 0.0)
            code[rows] = cell_token[cells[matrix.argmin(axis=1)]]
        if np.any(code < 0):
            raise ArithmeticError("independent P0 router failed to assign a code")
        answer.append(code)
    return answer[0], answer[1]


def wire(bank: dict, recipient: int, code: int) -> dict:
    item = bank["recipients"][str(recipient)]
    table = np.asarray(item["probability"], dtype=np.float64)
    probability = table[code]
    return {"schema": WIRE_SCHEMA, "recipient": recipient,
            "probability": probability.tolist(), "decision": int(probability.argmax()),
            "release_id": bank["release_id"], "code_id": int(code),
            "public_metadata": {"model_id": bank["release_id"],
                                "source_binding_sha256": bank["source_binding_sha256"],
                                "bank_sha256": item["bank_sha256"],
                                "code_length_bits": (len(table) - 1).bit_length()}}


def run(config_path: Path) -> dict:
    active = assert_active_from_env(stage="verification")
    queue = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    spec = queue["units"][active["unit_id"]]["spec"]
    pinned = {str(Path(x["path"]).absolute()): x["sha256"] for x in spec["sources"]}
    if pinned.get(str(config_path.absolute())) != sha(config_path):
        raise PermissionError("independent restore config is not queue-pinned")
    cfg = json.loads(config_path.read_text())
    if cfg.get("schema") != "IVR1-r5-independent-restore-config-v1" or cfg.get("unit_id") != active["unit_id"]:
        raise PermissionError("independent restore identity differs")
    root = Path(cfg["restore_root"]).resolve(strict=True)
    files = root / "files"
    code_root = files / "code"
    if (not str(Path(__file__).resolve()).startswith(str(code_root) + os.sep) or
            sha(root / "RESTORE_RECEIPT.json") != cfg["restore_receipt_sha256"] or
            sha(root / "metadata/CUSTODY_PLAN.json") != cfg["plan_sha256"]):
        raise PermissionError("checker code or restore receipt/plan differs")
    for module_name in ("jcv.train", "results.pcrl_incumbent_view_risk_v1.r0.compute_queue"):
        module_path = Path(importlib.import_module(module_name).__file__).resolve(strict=True)
        if not str(module_path).startswith(str(code_root) + os.sep):
            raise PermissionError("scientific model or queue implementation escaped restored code")
    receipt = json.loads((root / "RESTORE_RECEIPT.json").read_text())
    if receipt.get("status") != "RESTORED_FROM_SNAPSHOT_ONLY" or receipt.get("plan_sha256") != cfg["plan_sha256"]:
        raise PermissionError("versioned restore is not snapshot-only")
    plan = json.loads((root / "metadata/CUSTODY_PLAN.json").read_text())
    index = {x["alias"]: x for x in plan["entries"]}
    if len(index) != len(plan["entries"]):
        raise PermissionError("restored alias repeated")

    def source(alias: str) -> Path:
        if alias not in index or alias.startswith("/") or ".." in Path(alias).parts:
            raise PermissionError("source outside snapshot plan")
        path = files / alias
        if path.is_symlink() or not path.is_file() or sha(path) != index[alias]["sha256"]:
            raise PermissionError("snapshot source byte pin differs")
        if pinned.get(str(path.absolute())) != index[alias]["sha256"]:
            raise PermissionError("snapshot source absent from active queue spec")
        return path

    if (sha(source("locks/campaign.json")) != cfg["campaign_lock_sha256"] or
            sha(source("locks/strongest_control.json")) != cfg["control_lock_sha256"] or
            active["seal_sha256"] != cfg["admission_lock_sha256"] or
            spec.get("amendment_sha256") != cfg["amendment_sha256"]):
        raise PermissionError("independent replay lock bridge differs")
    latest = [a for a in queue["amendments"] if "verification" in a["stages"]][-1]
    if (latest["sha256"] != cfg["amendment_sha256"] or
            sha(Path(latest["path"])) != cfg["amendment_sha256"] or
            plan["nominee"] != "V_L" or plan["joint_diagnostic"] != "V_J" or
            plan["strongest_control"] != "V_S21" or
            plan["strongest_control_lock_sha256"] != cfg["control_lock_sha256"]):
        raise PermissionError("latest verification amendment or frozen selected arms differ")
    output = Path(cfg["output"]).absolute()
    if output.exists() or output.parent != Path(cfg["repo"]).absolute() / "results/pcrl_incumbent_view_risk_v1/r5":
        raise PermissionError("write-once R5 restore result path differs")
    roots = json.loads(source("private/stream_roots.json").read_text())
    if roots["schema"] != "scv1-private-stream-roots-v1":
        raise PermissionError("sampling root schema differs")
    checks = {"P0_raw83_code_rows": 0, "selected_sampled_code_rows": 0,
              "full_wire_rows": 0, "frozen_reader_predictions": 0}
    reader_hashes = {}
    for seed in (0, 1):
        bank = json.loads(source(f"banks/s{seed}/P0_PUBLIC_BANK.json").read_text())
        raw_archive = exact_npz(source(f"role/s{seed}/raw32.npz"), {"X", "feature_names"})
        names = [str(v) for v in raw_archive["feature_names"].tolist()]
        raw = raw_archive["X"]
        if (raw.shape != (32, 83) or not np.isfinite(raw).all() or
                names != bank["feature_names"] or
                hashlib.sha256("\n".join(names).encode()).hexdigest() != bank["feature_schema_sha256"]):
            raise PermissionError("restored raw83 sample/schema differs")
        teacher = {"model": source(f"frozen/s{seed}/teacher/model.pt"),
                   "head0": source(f"frozen/s{seed}/teacher/head_0.joblib"),
                   "head1": source(f"frozen/s{seed}/teacher/head_1.joblib")}
        complete = json.loads(source(f"frozen/s{seed}/teacher/COMPLETE.json").read_text())
        for filename, field in (("model.pt", "model"), ("head_0.joblib", "head0"),
                                ("head_1.joblib", "head1")):
            if complete["files"][filename] != sha(teacher[field]):
                raise PermissionError("restored teacher source COMPLETE differs")
        policy = json.loads(source(f"frozen/s{seed}/policy/policy.json").read_text())
        own_c1, own_c2 = route_p0(raw, seed, teacher, policy)
        p0 = exact_npz(source(f"role/s{seed}/FIT_P0_RELEASE.npz"), P0_KEYS)
        if (len(p0["row_id"]) != N or not np.array_equal(own_c1, p0["code_r1"][SAMPLE]) or
                not np.array_equal(own_c2, p0["code_r2"][SAMPLE])):
            raise ArithmeticError("independently routed P0 raw83 codes differ from frozen FIT")
        for recipient, codes in ((1, own_c1), (2, own_c2)):
            table = np.asarray(bank["recipients"][str(recipient)]["probability"], dtype=np.float64)
            if (not np.array_equal(table[codes], p0[f"prob_r{recipient}"][SAMPLE]) or
                    not np.array_equal(table[codes].argmax(axis=1),
                                       p0[f"decision_r{recipient}"][SAMPLE])):
                raise ArithmeticError("independently routed P0 probability or decision differs")
        checks["P0_raw83_code_rows"] += 32
        for arm in ARMS:
            draws = (0,) if arm == "P0" else (0, 1, 2)
            if arm != "P0":
                base = f"policies/{arm}/s{seed}/"
                policy_receipt = json.loads(source(base + "POLICY_RECEIPT.json").read_text())
                package = source(base + "policy.npz")
                if (policy_receipt["arm_id"] != arm or
                        policy_receipt["source_seed"] != seed or
                        policy_receipt["source_binding_sha256"] != bank["source_binding_sha256"] or
                        policy_receipt["policy_package_sha256"] != sha(package) or
                        policy_receipt["public_bank_sha256"] != sha(source(f"banks/s{seed}/P0_PUBLIC_BANK.json")) or
                        policy_receipt["source_lock_sha256"] != cfg["campaign_lock_sha256"]):
                    raise PermissionError("restored selected policy binding differs")
                arrays = exact_npz(package, set(policy_receipt["array_sha256"]))
                for r in (1, 2):
                    table = np.asarray(bank["recipients"][str(r)]["probability"], dtype=np.float64)
                    if not np.array_equal(arrays[f"probability{r}"], table):
                        raise ArithmeticError("independent selected table differs from original-ID bank")
                eta1, eta2 = float(policy_receipt["eta1"]), float(policy_receipt["eta2"])
            for draw in draws:
                if arm == "P0":
                    sampled1, sampled2 = own_c1, own_c2
                    unit_id = f"R1_DEPLOY_P0_S{seed}_M"
                else:
                    role_codes = exact_npz(source(base + f"draw{draw}/release_codes.npz"), CODE_KEYS)
                    if (not np.array_equal(role_codes["row_id"], p0["row_id"]) or
                            not np.array_equal(role_codes["unit"], p0["unit"])):
                        raise PermissionError("cached selected release rows/groups differ")
                    root_bytes = bytes.fromhex(roots["release_realization_roots_hex"][str(draw)])
                    sampled1, sampled2 = sampled_codes(arm, arrays, own_c1, own_c2, eta1, eta2,
                                                       root_bytes, seed, draw, p0["unit"][SAMPLE])
                    if (not np.array_equal(sampled1, role_codes["code_r1"][SAMPLE]) or
                            not np.array_equal(sampled2, role_codes["code_r2"][SAMPLE])):
                        raise ArithmeticError("independently sampled selected public codes differ")
                    checks["selected_sampled_code_rows"] += 32
                    suffix = "O" if arm == "V_S21" else "M"
                    unit_id = f"R1_DEPLOY_{arm}_S{seed}_R{draw}_{suffix}"
                for recipient, codes in ((1, sampled1), (2, sampled2)):
                    alias = f"campaign/r1/replay/{unit_id}/recipient{recipient}.jsonl"
                    with source(alias).open() as stream:
                        actual = [json.loads(line) for line in stream]
                    expected = [wire(bank, recipient, int(c)) for c in codes]
                    if actual != expected:
                        raise ArithmeticError("independently reconstructed full recipient wire differs")
                    checks["full_wire_rows"] += 32
            # Independently replay one fixed public-probability reader from
            # the restored INNER code bank, without opening INNER SEX.
            unit_id = f"R4_FINAL_{arm}_S{seed}_R0"
            reader_base = f"campaign/r4/attacks/{unit_id}/pair/readers/"
            reader_record = json.loads(source(reader_base + "READERS_RECEIPT.json").read_text())
            model_path = source(reader_base + "model__PROB_HGB_15x200.bin")
            predictions_path = source(reader_base + "predictions_inner.npz")
            view_path = source(reader_base + "view_inner.npz")
            for filename, path in (("model__PROB_HGB_15x200.bin", model_path),
                                   ("predictions_inner.npz", predictions_path),
                                   ("view_inner.npz", view_path)):
                if reader_record["files"][filename] != sha(path):
                    raise PermissionError("restored reader custody file differs")
            model = joblib.load(model_path)
            saved = exact_npz(predictions_path, set(reader_record["reader_order"]))
            if arm == "P0":
                inner = exact_npz(source(f"campaign/r1/R1_P0_S{seed}_ADMISSION/INNER_SELECTION/P0_RELEASE.npz"), P0_KEYS)
            else:
                inner = exact_npz(source(f"campaign/r2/units/R2_{arm}_S{seed}_FIT/INNER_SELECTION_R0/release_codes.npz"), CODE_KEYS)
            c1, c2 = inner["code_r1"][:32], inner["code_r2"][:32]
            p1 = np.asarray(bank["recipients"]["1"]["probability"], dtype=np.float64)[c1]
            p2 = np.asarray(bank["recipients"]["2"]["probability"], dtype=np.float64)[c2]
            x = np.column_stack((p1, p2))
            spans = {name: slice(int(start), int(stop)) for name, start, stop in
                     reader_record["view_schema"]["component_columns"]}
            with np.load(view_path, allow_pickle=False) as view:
                if (not np.array_equal(view["row_id"][:32], inner["row_id"][:32]) or
                        not np.array_equal(view["unit"][:32], inner["unit"][:32]) or
                        not np.array_equal(view["x"][:32, spans["r1.probability"]], p1) or
                        not np.array_equal(view["x"][:32, spans["r2.probability"]], p2)):
                    raise PermissionError("restored frozen reader row order differs")
            prediction = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
            if not np.array_equal(prediction, saved["PROB_HGB_15x200"][:32]):
                raise ArithmeticError("independently replayed restored reader predictions differ")
            reader_hashes[f"{arm}:s{seed}"] = hashlib.sha256(prediction.tobytes()).hexdigest()
            checks["frozen_reader_predictions"] += 32
    expected_counts = {"P0_raw83_code_rows": 64, "selected_sampled_code_rows": 576,
                       "full_wire_rows": 1280, "frozen_reader_predictions": 256}
    if checks != expected_counts or len(reader_hashes) != 8:
        raise ArithmeticError("independent snapshot coverage differs from fixed plan")
    result = {"schema": "IVR1-r5-independent-snapshot-replay-v1", "unit_id": active["unit_id"],
              "status": "PASS", "checks": checks, "reader_prediction_hashes": reader_hashes,
              "source_restore_receipt_sha256": cfg["restore_receipt_sha256"],
              "source_plan_sha256": cfg["plan_sha256"], "amendment_sha256": cfg["amendment_sha256"],
              "source_code_sha256": sha(Path(__file__)), "queue": active,
              "label_fields_read": [], "not_confirmation": True}
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    result = run(Path(parser.parse_args().config))
    print(json.dumps({"unit_id": result["unit_id"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
