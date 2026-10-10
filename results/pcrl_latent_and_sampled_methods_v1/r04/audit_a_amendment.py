"""Independent prefit closure audit of the prospective A stage amendment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .independent_a_source import sha_file


ARMS_STOCH = ("A_JOINT_STOCH", "A_TASK_STOCH", "A_CENTRAL_PRODUCT",
              "A_LOCAL", "A_SEQ12", "A_SEQ21")
ARMS_DET = ("A_TASK_DET", "A_JOINT_DET")
DIAGNOSTICS = ("A_IDENTITY", "A_CLASS_ONLY")
REFERENCES = ("T_STAR", "P_STAR", "U0", "Ucal", "D0_LOCAL", "D0_SEQ12", "D0_SEQ21")
FINITE_REFERENCES = tuple(ref for ref in REFERENCES if ref not in ("U0", "Ucal"))
ROLE = {"ATT": "ATTACK_FIT_NEW", "INNER": "INNER_SELECTION"}
EXPECTED_KIND_COUNTS = {"fit": 16, "q_preprocess": 30, "release": 80,
                        "attack": 40, "diagnostic_release": 8,
                        "diagnostic_attack": 4, "reference_release": 28,
                        "reference_attack": 14}


def _expected() -> dict[str, dict]:
    plan = {}

    def add(unit_id: str, kind: str, *, deps=(), **fields) -> None:
        if unit_id in plan:
            raise AssertionError("independent A unit ID collision")
        plan[unit_id] = {"kind": kind, "deps": list(deps), **fields}

    for arm in (*ARMS_STOCH, *ARMS_DET):
        for seed in (0, 1):
            fit = f"A-FIT-{arm}-S{seed}"
            q = f"A-Q-PREPROCESS-{arm}-S{seed}"
            add(fit, "fit", arm_id=arm, seed=seed)
            add(q, "q_preprocess", deps=(fit,), arm_id=arm, seed=seed,
                fit_unit_id=fit, source_kind="selected_fit")
            draws = (0, 1, 2) if arm in ARMS_STOCH else (0,)
            for draw in draws:
                releases = {}
                for short, role in ROLE.items():
                    release = f"A-RELEASE-{arm}-S{seed}-{short}-R{draw}"
                    add(release, "release", deps=(fit,), arm_id=arm, seed=seed,
                        realization=draw, role=role, fit_unit_id=fit)
                    releases[short] = release
                attack = f"A-ATTACK-{arm}-S{seed}-R{draw}"
                add(attack, "attack", deps=(fit, releases["ATT"], releases["INNER"], q),
                    arm_id=arm, seed=seed, realization=draw, fit_unit_id=fit,
                    q_unit_id=q, release_unit_ids=releases,
                    trained_reader_and_probe_fits=33)
    for ref in REFERENCES:
        for seed in (0, 1):
            q = f"A-Q-PREPROCESS-{ref}-S{seed}" if ref in FINITE_REFERENCES else None
            if q:
                add(q, "q_preprocess", reference_id=ref, seed=seed,
                    source_kind="historical_finite_reference")
            releases = {}
            for short, role in ROLE.items():
                release = f"A-REFERENCE-RELEASE-{ref}-S{seed}-{short}"
                add(release, "reference_release", reference_id=ref, seed=seed,
                    realization=0, role=role)
                releases[short] = release
            attack = f"A-REFERENCE-ATTACK-{ref}-S{seed}"
            deps = (releases["ATT"], releases["INNER"])
            add(attack, "reference_attack", deps=(*deps, q) if q else deps,
                reference_id=ref, seed=seed, realization=0, q_unit_id=q,
                release_unit_ids=releases,
                trained_reader_and_probe_fits=33 if q else 24)
    for arm in DIAGNOSTICS:
        for seed in (0, 1):
            q = f"A-Q-PREPROCESS-{arm}-S{seed}"
            add(q, "q_preprocess", arm_id=arm, seed=seed,
                source_kind="same_bank_diagnostic")
            releases = {}
            for short, role in ROLE.items():
                release = f"A-DIAG-RELEASE-{arm}-S{seed}-{short}"
                add(release, "diagnostic_release", arm_id=arm, seed=seed,
                    realization=0, role=role)
                releases[short] = release
            attack = f"A-DIAG-ATTACK-{arm}-S{seed}"
            add(attack, "diagnostic_attack",
                deps=(releases["ATT"], releases["INNER"], q),
                arm_id=arm, seed=seed, realization=0, q_unit_id=q,
                trained_reader_and_probe_fits=33)
    if len(plan) != 220 or Counter(v["kind"] for v in plan.values()) != EXPECTED_KIND_COUNTS:
        raise AssertionError("independent 220-unit graph does not close")
    return plan


def _check_graph(actual: dict) -> dict:
    expected = _expected()
    if set(actual) != set(expected):
        raise ValueError("A 220-unit IDs differ from independent Cartesian plan")
    for unit_id, want in expected.items():
        got = actual[unit_id]
        if got.get("output_id") != unit_id:
            raise ValueError(f"A output ID differs: {unit_id}")
        for key, value in want.items():
            # fit/diagnostic/reference release units have no declared deps.
            if key == "deps" and not value and key not in got:
                continue
            if got.get(key) != value:
                raise ValueError(f"A unit {unit_id} has wrong {key}")
        for bound in ("cpu_seconds", "wall_seconds"):
            value = got.get(bound)
            if (not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value <= 0):
                raise ValueError(f"A unit {unit_id} has no positive {bound} cap")
        if ("role" in got and got["role"] not in ROLE.values()) or "ASSESSMENT" in str(got):
            raise ValueError(f"A unit {unit_id} names a barred person role")
    # All edges stay within the registered prospective graph. A topological
    # pass detects cycles without trusting the builder's order.
    remaining = set(actual)
    visited = set()
    while remaining:
        ready = {u for u in remaining if set(actual[u].get("deps", ())) <= visited}
        if not ready:
            raise ValueError("A unit dependency graph has a cycle or missing source")
        visited |= ready
        remaining -= ready
    return {"units": len(actual), "kind_counts": dict(Counter(v["kind"] for v in actual.values())),
            "cpu_reserve_seconds": sum(v["cpu_seconds"] for v in actual.values()),
            "wall_reserve_seconds": sum(v["wall_seconds"] for v in actual.values())}


def run(args: argparse.Namespace) -> dict:
    amendment_path = Path(args.amendment).resolve(strict=True)
    amendment_sha = sha_file(amendment_path)
    amendment = json.loads(amendment_path.read_text())
    campaign = Path(args.campaign).resolve(strict=True)
    seal = campaign / "CAMPAIGN_LOCK.json"
    family = campaign / "r02/A_FAMILY_ORDER_S1.json"
    family_receipt = campaign / "r04/A_S1_FAMILY_INDEPENDENT_AUDIT.json"
    repair_manifest = Path(args.role_manifest).resolve(strict=True)
    queue_state_path = Path(args.queue_state).resolve(strict=True)
    queue = json.loads(queue_state_path.read_text())
    if (amendment.get("schema") != "pcrl-latent-sampled-A-stage-amendment-v1"
            or amendment.get("status") not in ("PREFIT_COMPLETE_FOR_SEAL", "LOCALLY_SEALED_A_STAGE")
            or amendment.get("original_lock_sha256") != sha_file(seal)
            or amendment["interval_family"]["manifest_sha256"] != sha_file(family)
            or amendment["interval_family"]["ordered_clause_count"] != 6885
            or amendment["role_bindings"]["repair_manifest_sha256"] != sha_file(repair_manifest)
            or queue.get("closed") or queue["seal"]["sha256"] != sha_file(seal)
            or queue.get("cpu_limit_s") != 72000):
        raise ValueError("A final candidate, original lock, source roles or family differs")
    family_audit = json.loads(family_receipt.read_text())
    if (family_audit.get("status") != "PROSPECTIVE_6885_FAMILY_ORDER_PASS"
            or family_audit.get("manifest_sha256") != sha_file(family)
            or amendment["interval_family"]["references"] != list(REFERENCES)
            or amendment["attack_plan"]["reference_ids"] != list(REFERENCES)
            or amendment["utility_caps"] != {
                "nll_excess": 0.010, "brier_excess": 0.005,
                "reference_rule": "min_U0_Ucal_on_FIT"}):
        raise ValueError("A seven-reference family or FIT utility caps differ")
    for field in ("source_sha256", "implementation_source_sha256"):
        for relative, expected_hash in amendment[field].items():
            path = campaign / relative
            if sha_file(path) != expected_hash:
                raise ValueError(f"A {field} bytes differ: {relative}")
    if amendment["source_sha256"].get("r04/A_S1_FAMILY_INDEPENDENT_AUDIT.json") != sha_file(family_receipt):
        raise ValueError("A independent family receipt not bound into amendment")
    graph = _check_graph(amendment["unit_plan"])
    used_a = float(queue["used_by_stage_s"].get("a", 0))
    used_global = float(queue["used_cpu_s"])
    if (amendment["cpu_budget"]["A_stage_total_ceiling_seconds"] != 18000
            or used_a + graph["cpu_reserve_seconds"] > 18000
            or used_global + graph["cpu_reserve_seconds"] > 72000):
        raise ValueError("A stage or global CPU reserve does not close")
    return {"schema": "pcrl-r04-a-prefit-amendment-audit-v1",
            "status": "A_220_UNIT_PREFIT_ADMISSION_PASS", "amendment_sha256": amendment_sha,
            "original_lock_sha256": sha_file(seal), "family_sha256": sha_file(family),
            "family_independent_receipt_sha256": sha_file(family_receipt),
            "queue_state_sha256_at_audit": sha_file(queue_state_path),
            "source_hashes_checked": {field: len(amendment[field]) for field in
                                      ("source_sha256", "implementation_source_sha256")},
            "role_boundary": "FIT, ATTACK_FIT_NEW and INNER_SELECTION; no assessment release unit",
            "graph": graph, "used_a_stage_cpu_seconds_at_audit": used_a,
            "used_global_cpu_seconds_at_audit": used_global,
            "a_stage_reserved_ceiling_seconds": 18000,
            "global_reserved_ceiling_seconds": 72000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("amendment", "campaign", "role-manifest", "queue-state", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run(args)
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("A independent prefit amendment receipt already exists")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
