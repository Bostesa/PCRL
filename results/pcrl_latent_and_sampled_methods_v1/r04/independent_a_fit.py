"""Independent expected-law utility and public recovery arithmetic for A FIT."""

from __future__ import annotations

import math

import numpy as np

from .independent_a_inputs import AdmittedARole
from .independent_a_law import IndependentAPolicy


TASK_KEYS = ("r1_nll", "r1_brier", "r2_nll", "r2_brier")


def _reference_caps(source: AdmittedARole) -> dict[str, float]:
    if set(source.labels) != {"income", "occupation"}:
        raise ValueError("A FIT utility needs both task truths")
    out = {}
    for r, task in ((1, "income"), (2, "occupation")):
        y = source.labels[task]
        for metric, allowance in (("nll", .010), ("brier", .005)):
            alternatives = []
            for name in ("U0", "Ucal"):
                score = source.references[f"{name}_{r}"]
                if metric == "nll":
                    value = float(np.mean(-np.log(np.maximum(score[np.arange(len(y)), y],
                                                       1e-12))))
                else:
                    value = float(np.mean(np.sum(score * score, axis=1) + 1 -
                                          2 * score[np.arange(len(y)), y]))
                alternatives.append(value)
            out[f"r{r}_{metric}"] = min(alternatives) + allowance
    return out


def expected_fit_metrics(policy: IndependentAPolicy, source: AdmittedARole,
                         sex: np.ndarray | None = None) -> dict[str, float]:
    """Score the actual finite emitted law, not a probability-averaged output."""
    if source.role != "OSF_DEFENSE_FIT" or set(source.labels) != {"income", "occupation"}:
        raise ValueError("A expected-law FIT replay requires admitted FIT task labels")
    n = len(source.row_id)
    if sex is not None and (sex.dtype != np.int64 or sex.shape != (n,) or
                            not np.isin(sex, (0, 1)).all()):
        raise ValueError("A FIT protected labels differ")
    sums = {name: 0.0 for name in TASK_KEYS}
    marginal: dict[tuple[int, int], float] = {}
    conditional_entropy = 0.0
    protected = {view: [{}, {}] for view in ("r1", "r2", "pair")}
    residual = 0.0
    y1, y2 = source.labels["income"], source.labels["occupation"]
    for row in range(n):
        law = policy.law(source.score1[row], source.score2[row],
                         int(source.cell1[row]), int(source.cell2[row]))
        residual = max(residual, abs(math.fsum(mass for _, _, mass in law) - 1.0))
        loss = policy.expected_losses(law, int(y1[row]), int(y2[row]))
        for recipient in (1, 2):
            for name in ("nll", "brier"):
                sums[f"r{recipient}_{name}"] += loss[f"r{recipient}"][name] / n
        for i, j, mass in law:
            marginal[i, j] = marginal.get((i, j), 0.0) + mass / n
            if mass > 0:
                conditional_entropy -= mass * math.log(mass) / n
            if sex is not None:
                s = int(sex[row])
                for view, key in (("r1", i), ("r2", j), ("pair", (i, j))):
                    bucket = protected[view][s]
                    bucket[key] = bucket.get(key, 0.0) + mass / n
    marginal_entropy = -math.fsum(p * math.log(p) for p in marginal.values() if p > 0)
    sums.update({"task_rate": marginal_entropy - conditional_entropy,
                 "max_law_residual": residual})
    if sex is not None:
        for view in ("r1", "r2", "pair"):
            a, b = protected[view]
            sums[f"{view}_bayes"] = math.fsum(max(a.get(k, 0.0), b.get(k, 0.0))
                                                 for k in a.keys() | b.keys())
        sums["protected_objective"] = (sums["pair_bayes"] +
                                       .5 * (sums["r1_bayes"] + sums["r2_bayes"]))
    return sums


def compare_fit_metrics(policy: IndependentAPolicy, source: AdmittedARole,
                        fit_receipt: dict, sex: np.ndarray | None = None) -> dict:
    if (fit_receipt.get("status") != "SELECTED_FIT_POLICY"
            or fit_receipt["arm_id"] != policy.receipt["arm_id"]
            or fit_receipt["source_seed"] != source.seed
            or fit_receipt["source_binding_sha256"] != source.source_binding_sha256
            or fit_receipt["row_unit_pair_sha256"] != source.row_unit_pair_sha256):
        raise ValueError("A FIT selected receipt and source identity differ")
    caps = _reference_caps(source)
    cap_error = max(abs(caps[key] - fit_receipt["caps"][key]) for key in TASK_KEYS)
    if cap_error > 1e-10:
        raise ValueError("A FIT U0/Ucal minimum utility caps differ")
    actual = expected_fit_metrics(policy, source, sex)
    if "fit" in fit_receipt:
        claimed = fit_receipt["fit"]["metrics"]
    else:
        selected = fit_receipt["selected"]
        name = "identity_biased" if selected["start"] == 0 else "uniform_logit"
        matches = [record["metrics"] for record in fit_receipt["starts"][name]["checkpoints"]
                   if record["update"] == selected["update"]]
        if len(matches) != 1:
            raise ValueError("A selected checkpoint metrics missing or duplicated")
        claimed = matches[0]
    if not isinstance(claimed, dict):
        raise ValueError("A FIT selected metrics absent")
    discrepancy = {key: abs(actual[key] - value) for key, value in claimed.items()
                   if key in actual and isinstance(value, (int, float))}
    if not set(TASK_KEYS) <= discrepancy.keys() or max(discrepancy.values()) > 2e-8:
        raise ValueError("A selected FIT expected-law metric differs from independent replay")
    feasible = all(actual[key] <= caps[key] + 2e-8 for key in TASK_KEYS)
    return {"status": "SELECTED_FIT_EXPECTED_LAW_REPLAY_PASS",
            "arm_id": policy.receipt["arm_id"], "source_seed": source.seed,
            "rows": len(source.row_id), "groups": len(np.unique(source.unit)),
            "expected_law_metrics": actual, "fit_caps": caps,
            "max_claimed_metric_difference": max(discrepancy.values()),
            "max_cap_difference": cap_error, "task_cap_feasible": feasible}
