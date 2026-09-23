"""Replay paired 2018 inner-pilot uncertainty for the frozen exchange round.

This reads only hash-pinned private contribution archives from completed inner
pilots. It never loads the 2018 outer assessment pool or fits a predictor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm


ROLES = ("utility:A/same_residence", "attack:A/SEX", "attack:A/RAC1P",
         "attack:AB/SEX", "attack:AB/RAC1P")
WEIGHTINGS = ("U", "PWGTP")
N_BOOT = 10000
SEED = 20260923


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_panel(root: Path, directory: Path, *, unit_id: str, release_id: str,
               queue_sha256: str) -> dict:
    receipt_path = directory / "COMPLETE.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("unit_id") != unit_id or receipt.get("queue_sha256") != queue_sha256:
        raise ValueError(f"completion receipt disagrees with registration: {unit_id}")
    files = ("INNER_PANEL.json", "INNER_PILOT_CONTRIBUTIONS.npz")
    for name in files:
        if receipt.get("artifacts", {}).get(name) != sha(directory / name):
            raise ValueError(f"completed artifact hash differs: {unit_id}/{name}")
    panel = json.loads((directory / "INNER_PANEL.json").read_text())
    if (panel.get("release_id") != release_id or panel.get("anchor") != 0 or
            panel.get("score_pool") != "downstream_validation:inner_pilot" or
            panel.get("outer_pool_opened") is not False or
            panel.get("slate") != "standard" or
            set(panel.get("roles", {})) != set(ROLES)):
        raise ValueError(f"inner-pilot scope differs: {unit_id}")
    arrays = {}
    with np.load(directory / "INNER_PILOT_CONTRIBUTIONS.npz", allow_pickle=False) as stored:
        for role in ROLES:
            slug = role.replace(":", "_").replace("/", "_")
            arrays[role] = {name: np.asarray(stored[f"{slug}_{name}"]).copy()
                            for name in ("ids", "households", "weights",
                                         "candidate_loss", "H_loss")}
    return {"panel": panel, "arrays": arrays,
            "receipt_sha256": sha(receipt_path),
            "panel_sha256": sha(directory / "INNER_PANEL.json"),
            "contributions_sha256": sha(directory / "INNER_PILOT_CONTRIBUTIONS.npz")}


def build(root: Path, *, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[dict, dict]:
    if n_boot < 2 or seed < 0:
        raise ValueError("invalid bootstrap count or seed")
    base = root / "results/pcrl_task_aligned_cuts_v1"
    protocol_path = base / "PROTOCOL_LOCK.json"
    protocol = json.loads(protocol_path.read_text())
    sidecar_names = ("EXCHANGE_AUDIT_SIDECAR.json", "EXCHANGE_MILP_AUDIT_SIDECAR.json")
    sidecars = {}
    for name in sidecar_names:
        path = base / "agents/release_math" / name
        sidecar = json.loads(path.read_text())
        if sidecar.get("outer_pool_opened") is not False or sidecar.get("slate") != "standard":
            raise ValueError(f"sidecar scope differs: {name}")
        for pin in sidecar.get("input_files", {}).values():
            file = root / pin["relative_path"]
            if sha(file) != pin["sha256"]:
                raise ValueError(f"sidecar input hash differs: {file}")
        sidecars[name] = (sidecar, sha(path))
    exchange_sidecar, exchange_sidecar_sha = sidecars[sidecar_names[0]]
    milp_sidecar, milp_sidecar_sha = sidecars[sidecar_names[1]]
    if exchange_sidecar["bank_sha256"] != milp_sidecar["bank_sha256"]:
        raise ValueError("exchange and MILP use different frozen banks")
    registered = {
        "exchanged_stochastic": (base / "private/exchange_r01_audit",
                                exchange_sidecar["unit_id"], exchange_sidecar["release_id"],
                                exchange_sidecar_sha),
        "same_bank_MILP": (base / "private/exchange_r01_deterministic_audit",
                           milp_sidecar["unit_id"], milp_sidecar["release_id"],
                           milp_sidecar_sha),
        "round_zero_center": (base / "private/run/a0_u1p1_z000_audit",
                              "a0_u1p1_z000_audit", "a0_u1p1_z000",
                              protocol["run_queue_sha256"]),
    }
    panels = {name: load_panel(root, path, unit_id=unit_id, release_id=release_id,
                               queue_sha256=queue_sha)
              for name, (path, unit_id, release_id, queue_sha) in registered.items()}
    source = panels["exchanged_stochastic"]
    point_path = base / "EXCHANGE_PILOT_SCREEN.json"
    points = json.loads(point_path.read_text())
    if points.get("outer_pool_opened") is not False:
        raise ValueError("point-screen scope differs")
    comparison_keys = {"same_bank_MILP": "exchange_deterministic_MILP",
                       "round_zero_center": "round_zero_center"}
    endpoints = []
    contributions = []
    member_households = set()
    for name, screen_key in comparison_keys.items():
        comparator = panels[name]
        for role in ROLES:
            left = source["arrays"][role]
            right = comparator["arrays"][role]
            for field in ("ids", "households", "weights", "H_loss"):
                if not np.array_equal(left[field], right[field]):
                    raise ValueError(f"person-level pairing differs: {name}/{role}/{field}")
            if len(np.unique(left["ids"])) != len(left["ids"]):
                raise ValueError("duplicate original-person IDs")
            if source["panel"]["roles"][role]["H_selected_model_sha256"] != \
                    comparator["panel"]["roles"][role]["H_selected_model_sha256"]:
                raise ValueError(f"H baseline model differs: {name}/{role}")
            member_households.update(map(str, left["households"]))
            for weighting in WEIGHTINGS:
                weight = (np.ones(len(left["ids"]), dtype=np.float64) if weighting == "U"
                          else np.asarray(left["weights"], dtype=np.float64))
                if not np.isfinite(weight).all() or np.min(weight) < 0 or weight.sum() <= 0:
                    raise ValueError("invalid original-person weights")
                orient = 1.0 if role == ROLES[0] else -1.0
                difference = orient * (left["candidate_loss"] - right["candidate_loss"])
                estimate = float(np.dot(weight, difference) / weight.sum())
                for panel, arr in ((source, left), (comparator, right)):
                    replayed_loss = float(np.dot(weight, arr["candidate_loss"]) / weight.sum())
                    stored_loss = panel["panel"]["roles"][role]["candidate"][weighting]
                    if abs(replayed_loss - stored_loss) > 1e-10:
                        raise ValueError(f"report and contributions disagree: {name}/{role}/{weighting}")
                screen = points["exchange_minus_comparator"][screen_key]
                screen_loss = (screen["task_loss_exchange_minus_comparator_nats"][weighting]
                               if role == ROLES[0] else
                               screen["protected_recovery_exchange_minus_comparator_nats"][role][weighting])
                if abs(estimate - screen_loss) > 1e-10:
                    raise ValueError(f"independent point replay differs: {name}/{role}/{weighting}")
                endpoints.append({"id": f"exchanged_stochastic|{name}|{role}|{weighting}",
                                  "comparator": name, "role": role, "weighting": weighting,
                                  "orientation": ("CE_exchange-CE_comparator" if orient == 1 else
                                                  "CE_comparator-CE_exchange = recovery_exchange-recovery_comparator"),
                                  "estimate": estimate, "original_people": len(left["ids"]),
                                  "households": len(np.unique(left["households"])),
                                  "identical_pairs": bool(np.all(difference == 0))})
                contributions.append((np.asarray(left["households"]).astype(str),
                                      weight, difference))
    if len(endpoints) != 20:
        raise AssertionError("two comparisons × five roles × two weightings required")
    households = np.asarray(sorted(member_households), dtype=str)
    household_index = {value: index for index, value in enumerate(households)}
    count_households, n_endpoints = len(households), len(endpoints)
    numerator = np.zeros((count_households, n_endpoints), dtype=np.float64)
    denominator = np.zeros((count_households, n_endpoints), dtype=np.float64)
    for index, (household_ids, weight, difference) in enumerate(contributions):
        position = np.fromiter((household_index[value] for value in household_ids), dtype=np.int64)
        numerator[:, index] = np.bincount(position, weights=weight * difference,
                                          minlength=count_households)
        denominator[:, index] = np.bincount(position, weights=weight,
                                            minlength=count_households)
    rng = np.random.default_rng(seed)
    accepted = 0
    attempted = 0
    rejected = 0
    running_mean = np.zeros(n_endpoints)
    m2 = np.zeros(n_endpoints)
    while accepted < n_boot:
        if attempted >= max(10000, 100 * n_boot):
            raise RuntimeError("too many invalid household draws")
        batch = min(128, n_boot - accepted)
        draw = rng.multinomial(count_households,
                               np.full(count_households, 1 / count_households), size=batch)
        attempted += batch
        den = draw @ denominator
        valid = np.all(den > 0, axis=1)
        rejected += int(np.sum(~valid))
        if not np.any(valid):
            continue
        values = (draw[valid] @ numerator) / den[valid]
        size = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = np.sum((values - batch_mean) ** 2, axis=0)
        new_accepted = accepted + size
        delta = batch_mean - running_mean
        m2 += batch_m2 + delta * delta * accepted * size / new_accepted
        running_mean += delta * size / new_accepted
        accepted = new_accepted
    se = np.sqrt(np.maximum(0, m2) / (n_boot - 1))
    z_point = float(norm.isf(.025))
    z_family = float(norm.isf(.05 / (2 * n_endpoints)))
    for index, endpoint in enumerate(endpoints):
        endpoint.update({"bootstrap_se": float(se[index]),
                         "pointwise_95_lower": endpoint["estimate"] - z_point * float(se[index]),
                         "pointwise_95_upper": endpoint["estimate"] + z_point * float(se[index]),
                         "descriptive_family20_95_lower": endpoint["estimate"] - z_family * float(se[index]),
                         "descriptive_family20_95_upper": endpoint["estimate"] + z_family * float(se[index])})
    public = {"schema": "pcrl-post-exchange-inner-pilot-uncertainty-v1",
              "scope": "2018 historically reused anchor-0 inner_pilot, post-exchange descriptive only",
              "outer_pool_opened": False,
              "limitations": ["These intervals condition on fitted predictors, selected channels and the post-pilot exchange decision.",
                              "No outer assessment was opened; the 2018 pool was historically reused.",
                              "Only anchor 0 was resampled; model-selection adaptivity and full survey-design uncertainty are not covered."],
              "sampling_unit": "household; one shared multinomial draw for all 20 endpoints",
              "source_release_id": source["panel"]["release_id"],
              "comparators": list(comparison_keys), "n_endpoints": n_endpoints,
              "households_union": count_households,
              "bootstrap": {"seed": seed, "requested": n_boot, "accepted": accepted,
                            "attempted": attempted, "rejected_zero_denominator": rejected,
                            "pointwise_95_z": z_point, "descriptive_family20_95_z": z_family},
              "endpoints": endpoints,
              "public_input_sha256": {"PROTOCOL_LOCK.json": sha(protocol_path),
                                      "EXCHANGE_AUDIT_SIDECAR.json": exchange_sidecar_sha,
                                      "EXCHANGE_MILP_AUDIT_SIDECAR.json": milp_sidecar_sha,
                                      "EXCHANGE_PILOT_SCREEN.json": sha(point_path)}}
    private = {"schema": "pcrl-post-exchange-inner-pilot-uncertainty-replay-v1",
               "script_sha256": sha(Path(__file__)), "bootstrap_seed": seed,
               "bootstrap_replicates": n_boot,
               "private_inputs": {name: {key: panel[key] for key in
                                        ("receipt_sha256", "panel_sha256", "contributions_sha256")}
                                  for name, panel in panels.items()},
               "public_input_sha256": public["public_input_sha256"],
               "person_or_household_rows_copied": False}
    return public, private


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    public, private = build(args.root.resolve())
    serialized = json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.verify_only:
        if args.public.read_text() != serialized:
            raise ValueError("public aggregate differs from independent replay")
        receipt = json.loads(args.private.read_text())
        if receipt != {**private, "public_output_sha256": sha(args.public)}:
            raise ValueError("private replay receipt differs")
        print(json.dumps({"status": "verified", "public_sha256": sha(args.public)}))
        return
    if args.public.exists() or args.private.exists():
        raise FileExistsError("output already exists; use --verify-only")
    args.private.parent.mkdir(parents=True, exist_ok=True)
    args.public.write_text(serialized)
    args.private.write_text(json.dumps({**private, "public_output_sha256": sha(args.public)},
                                       indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "endpoints": public["n_endpoints"],
                      "public_sha256": sha(args.public)}))


if __name__ == "__main__":
    main()
