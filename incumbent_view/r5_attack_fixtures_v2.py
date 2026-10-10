"""Data-free R5 arithmetic fixtures; execute only as a queued unit."""
import json
import numpy as np

from .r5_attack_math import (ARMS, GROUPS, SUMMARIES, VIEWS,
                             auc_by_group_weights, fixed_recovery_family,
                             oriented_bounds, paired_plan, reader_envelope,
                             summarize_release_vectors)
from .r5_selection_math import (fit_nominee, maximum_local_worsening,
                                minimum_benefit, strongest_control)


def run() -> dict:
    passed = []
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    unit = np.array([10, 11, 12, 13], dtype=np.int64)
    inverse, weights, draws = paired_plan(unit, replicates=7, expected_groups=4)
    assert weights.shape == (8, 4) and draws.shape == (7, 2, 3)
    assert np.array_equal(weights[0], np.ones(4, dtype=np.int64))
    passed.append("fixed paired group/draw plan")
    score = np.array([0.1, 0.8, 0.5, 0.5], dtype=np.float64)
    auc = auc_by_group_weights(y, score, inverse, weights[:1])
    assert abs(float(auc[0]) - 0.875) < 1e-15
    passed.append("half-credit exact score tie")
    envelope = reader_envelope(y, {"reverse": 1 - score}, ("reverse",), inverse, weights[:1])
    assert abs(float(envelope[0]) - 0.875) < 1e-15
    passed.append("frozen reader reverse orientation")
    vectors = {}
    for arm, level in (("P0", 0.8), ("O_J", 0.7)):
        for seed in (0, 1):
            for draw in range(3 if arm == "O_J" else 1):
                for view in VIEWS:
                    vectors[(arm, seed, draw, view)] = np.full(8, level + draw * 0.01)
    summaries = summarize_release_vectors(vectors, {"P0", "O_J"}, draws, replicates=7)
    assert len(summaries) == 2 * 9 * 3
    assert abs(float(summaries[("mean", "P0", "pair")][0]) - 0.8) < 1e-15
    slots = fixed_recovery_family(summaries, {"P0", "O_J"}, replicates=7, groups=4)
    assert len(slots) == 756 and sum(v["point"] is None for v in slots.values()) == 729
    assert set(SUMMARIES) == {key.split(":", 1)[0] for key in slots}
    point, lower, upper = oriented_bounds(slots, "O_J", "P0", "pair")
    assert abs(point - 0.09) < 1e-12 and lower <= point <= upper
    passed.append("756-slot family, conditional nulls, signed benefit")
    testbenefit = minimum_benefit(slots, "O_J", "P0", 0.02)
    assert testbenefit["status"] == "PASS"
    local = maximum_local_worsening(slots, "O_J", "P0", view="r1")
    assert local["status"] == "PASS"
    passed.append("coalition benefit and local cap orientation")
    gates = {a: {"all_two_seed_task_reference_point_caps_pass": a in ("P0", "V_J", "V_P")}
             for a in ARMS}
    prior = {("V_J", s): {"constant_prior_ce": 0.7, "pair_reader_losses": {"a": 0.65},
                           "pair_min_ce": 0.65, "reader_recovery_gap": 0.05,
                           "crossfit_manifest_sha256": "a" * 64, "exact_p0_alias": False}
             for s in (0, 1)}
    prior.update({("V_P", s): {"constant_prior_ce": 0.7, "pair_reader_losses": {"a": 0.60},
                                 "pair_min_ce": 0.60, "reader_recovery_gap": 0.10,
                                 "crossfit_manifest_sha256": "b" * 64, "exact_p0_alias": False}
                  for s in (0, 1)})
    assert fit_nominee(gates, prior)["nominee"] == "V_J"
    assert strongest_control("V_J", gates, {"P0", "V_J", "V_P"},
                             {"P0": 0.8, "V_P": 0.75})["selected"] == "V_P"
    passed.append("FIT CE nomination and minimum-AUC control")
    return {"schema": "IVR1-r5-data-free-final-math-fixture-v2", "status": "PASS",
            "checks_passed": len(passed), "checks": passed,
            "real_people_read": 0, "sensitive_labels_read": 0}


def main() -> None:
    from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
    assert_active_from_env(stage="verification")
    print(json.dumps(run(), sort_keys=True))


if __name__ == "__main__":
    main()
