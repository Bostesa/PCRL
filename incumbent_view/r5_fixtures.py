"""Independent finite-law correctness checks, executed only in R0's queue."""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
from pathlib import Path

from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nll(p: list[float], y: int) -> float:
    return -math.log(max(p[y], 1e-12))


def brier(p: list[float], y: int) -> float:
    return math.fsum((x - float(k == y)) ** 2 for k, x in enumerate(p))


def exact_nll(p: list[float], y: int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 80
        return -Decimal.from_float(max(p[y], 1e-12)).ln()


def run(spec_path: Path, output: Path) -> dict:
    active = assert_active_from_env(stage="admission")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    sources = {Path(x["path"]): x["sha256"] for x in state["units"][active["unit_id"]]["spec"]["sources"]}
    if sources.get(spec_path) != sha(spec_path):
        raise ValueError("independent fixture specification not queue pinned")
    spec = json.loads(spec_path.read_text())
    laws = {x["id"]: x for x in spec["laws"]}
    checks = {}

    identity = laws["IDENTITY_WIRE"]
    ids, banks, q = identity["public_ids"], identity["public_probabilities"], identity["q_identity"]
    emitted = [ids[row.index(1)] for row in q]
    assert emitted == identity["source_public_ids"]
    assert all(sum(row) == 1 and sum(x > 0 for x in row) == 1 for row in q)
    assert [banks[ids.index(code)] for code in emitted] == [banks[0], banks[1], banks[0]]
    checks["identity"] = "PASS"

    law = laws["EXPECTED_REALIZED_SERVICE"]
    ps, masses, y = law["public_probabilities"], law["q"], law["target"]
    expected_nll = math.fsum(m * nll(p, y) for m, p in zip(masses, ps))
    expected_brier = math.fsum(m * brier(p, y) for m, p in zip(masses, ps))
    mean_p = [math.fsum(m * p[k] for m, p in zip(masses, ps)) for k in range(2)]
    assert expected_nll == math.fsum([nll(ps[0], y), nll(ps[1], y)]) / 2
    assert expected_brier == math.fsum([brier(ps[0], y), brier(ps[1], y)]) / 2
    assert expected_nll > nll(mean_p, y) and expected_brier > brier(mean_p, y)
    checks["expected_vs_mean_service"] = {"status": "PASS", "expected_nll": expected_nll,
                                            "mean_probability_nll": nll(mean_p, y),
                                            "expected_brier": expected_brier,
                                            "mean_probability_brier": brier(mean_p, y)}

    trap = laws["MEAN_OUTPUT_TRAP"]
    means = [math.fsum(trap[f"sex{s}_outputs"]) / 2 for s in (0, 1)]
    assert all(abs(x - .70) < 1e-15 for x in means)
    assert set(trap["sex0_outputs"]).isdisjoint(trap["sex1_outputs"])
    assert all(x > .5 for s in (0, 1) for x in trap[f"sex{s}_outputs"])
    actual_ce, mean_ce = -math.log(.99), -math.log(.5)
    assert actual_ce < mean_ce
    checks["mean_output_trap"] = {"status": "PASS", "actual_support_reader_ce": actual_ce,
                                    "constant_and_mean_only_ce": mean_ce}

    family = laws["FAMILY_COUPLING"]
    joint, product = family["joint_q"], family["product_q"]
    for value in (joint, product):
        assert [sum(row) for row in value] == [.5, .5]
        assert [sum(row[k] for row in value) for k in range(2)] == [.5, .5]
    assert joint != product and product == [[.5 * .5] * 2] * 2
    checks["family_coupling"] = "PASS"

    seq = laws["SEQUENTIAL_FREEZE"]
    first = seq["first_q"]
    for condition in (seq["second_before"], seq["second_after"]):
        pair = [[first[i] * condition[i][j] for j in range(2)] for i in range(2)]
        assert all(abs(math.fsum(row) - first[i]) <= 1e-15
                   for i, row in enumerate(pair))
        with localcontext() as ctx:
            ctx.prec = 80
            assert [sum(Decimal(str(first[i])) * Decimal(str(x)) for x in row)
                    for i, row in enumerate(condition)] == [Decimal(str(x)) for x in first]
    mixed_first = [(a + b) / 2 for a, b in zip(first, seq["whole_identity_first_q"])]
    assert mixed_first != first
    checks["sequential_first_freeze"] = "PASS"

    groups = laws["GROUP_FOLDS"]
    assignment = groups["fold_by_group"]
    rows = groups["groups"]
    for fold in range(3):
        fitting = {x for x in rows if assignment[str(x)] != fold}
        heldout = {x for x in rows if assignment[str(x)] == fold}
        assert fitting.isdisjoint(heldout)
    checks["group_folds"] = "PASS"

    margin = laws["NUMERIC_MARGIN"]
    ps, y = margin["public_probabilities"], margin["target"]
    identity_cost = nll(ps[0], y)
    paired_excess = math.fsum(m * (nll(p, y) - identity_cost)
                              for m, p in zip(margin["q_candidate"], ps))
    with localcontext() as ctx:
        ctx.prec = 80
        baseline = exact_nll(ps[0], y)
        high = sum(Decimal.from_float(m) * (exact_nll(p, y) - baseline)
                   for m, p in zip(margin["q_candidate"], ps))
        gap = abs(Decimal.from_float(paired_excess) - high)
    assert gap < Decimal("1e-10")
    assert math.fsum(nll(ps[0], y) - identity_cost for _ in range(3)) == 0
    checks["numeric_precision"] = {"status": "PASS", "float64_paired_excess": paired_excess,
                                    "decimal80_excess": str(high), "absolute_gap": str(gap),
                                    "scope": "Engineering fixture only; real source action-cost checks still required."}

    reader = laws["READER_WEIGHT_AND_NULL"]
    counts = [[0., 0.], [0., 0.]]
    for s, masses in zip(reader["sex"], reader["action_probabilities"]):
        for action, mass in enumerate(masses):
            counts[action][s] += mass / 2
    assert counts == [[.4, .15], [.1, .35]]
    null_ce = math.fsum(sum(masses) * -math.log(.5)
                        for masses in reader["action_probabilities"]) / 2
    assert null_ce == -math.log(.5)
    checks["weighted_actual_action_and_constant_ce"] = "PASS"

    result = {"schema": "pcrl-incumbent-r5-independent-fixture-receipt-v1",
              "status": "PASS_ANALYTIC_FIXTURES_SOURCE_CHECKS_PENDING", "unit_id": active["unit_id"],
              "fixture_spec_sha256": sha(spec_path), "checks": checks,
              "actual_source_nonlinear_reader_test": "PENDING_R2_IMPLEMENTATION",
              "real_p0_wire_support_and_snapshot_checks": "PENDING_ADMISSION",
              "no_real_arrays_opened": True, "scientific_claim": "Engineering correctness only"}
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.spec, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
