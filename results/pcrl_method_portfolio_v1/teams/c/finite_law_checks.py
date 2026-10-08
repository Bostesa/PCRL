"""Exact synthetic premise checks against S1's frozen finite-law registry.

Usage: python finite_law_checks.py PATH/TO/SYNTHETIC_LAWS.json
No Adult data or learned attack is accessed. Fractions enumerate each law.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from prototype import AdultGroupedSchema, IdentityCalibrator, PurposeRelease, release_pair


FROZEN_LAWS_SHA256 = "b2e68f5dec7dd690afa9389c44c0c2f98a78b30cde25c20a48751665f35fdc53"


def exact_auc(records):
    """Bayes-score AUC for a finite law of (mass, SEX, disclosed view)."""
    grouped = {}
    for mass, sex, view in records:
        if mass < 0:
            raise ValueError("negative law mass")
        if view not in grouped:
            grouped[view] = [Fraction(0), Fraction(0)]
        grouped[view][sex] += mass
    positives = sum(v[1] for v in grouped.values())
    negatives = sum(v[0] for v in grouped.values())
    if positives <= 0 or negatives <= 0:
        raise ValueError("both SEX classes are needed")
    auc = Fraction(0)
    for positive in grouped.values():
        score_pos = positive[1] / sum(positive)
        for negative in grouped.values():
            score_neg = negative[1] / sum(negative)
            relation = 1 if score_pos > score_neg else Fraction(1, 2) if score_pos == score_neg else 0
            auc += positive[1] * negative[0] * relation
    return auc / (positives * negatives)


def record(mass, sex, *view):
    return (mass, sex, tuple(view))


class IncomeClueHead:
    def predict_proba(self, selected):
        bit, clue = selected[:, 0], selected[:, 1]
        p1 = np.where(bit == 1, 0.80 + 0.10 * clue, 0.20 - 0.10 * clue)
        return np.column_stack((1 - p1, p1))


class OccupationClueHead:
    def predict_proba(self, selected):
        clue, bit = selected[:, 0], selected[:, 1]
        p1 = np.where(bit == 1, 0.80 + 0.10 * clue, 0.20 - 0.10 * clue)
        return np.column_stack((1 - p1, p1, np.zeros((len(p1), 4))))


def main(law_path: Path):
    raw = law_path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != FROZEN_LAWS_SHA256:
        raise SystemExit(f"frozen law SHA mismatch: {actual_sha}")
    laws = json.loads(raw)
    if [law["id"] for law in laws["laws"]] != [
        "L1_FEASIBLE_NULL", "L2_REMOVABLE_DISTINCTION",
        "L3_COMPLEMENTARY_XOR", "L4_NO_COMPLEMENTARITY",
        "L5_CONFIDENCE_VS_CLASS",
    ]:
        raise SystemExit("frozen law identities changed")

    halves = list(itertools.product((0, 1), repeat=3))
    null_local1 = [record(Fraction(1, 8), s, b1) for s, b1, b2 in halves]
    null_local2 = [record(Fraction(1, 8), s, b2) for s, b1, b2 in halves]
    null_pair = [record(Fraction(1, 8), s, b1, b2) for s, b1, b2 in halves]
    assert sum(x[0] for x in null_pair) == 1
    assert (exact_auc(null_local1), exact_auc(null_local2), exact_auc(null_pair)) == (
        Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)
    )

    removable_before = [record(Fraction(1, 8), s, b1, s, b2, s) for s, b1, b2 in halves]
    removable_after = [record(Fraction(1, 8), s, b1, b2) for s, b1, b2 in halves]
    assert exact_auc(removable_before) == 1
    assert exact_auc(removable_after) == Fraction(1, 2)
    assert all((row[2][0], row[2][2]) == (b1, b2) for row, (_, b1, b2) in zip(removable_before, halves))

    xor_states = list(itertools.product((0, 1), repeat=4))
    xor_first = [record(Fraction(1, 16), s, b1, u) for s, b1, b2, u in xor_states]
    xor_second = [record(Fraction(1, 16), s, b2, u ^ s) for s, b1, b2, u in xor_states]
    xor_pair = [record(Fraction(1, 16), s, b1, u, b2, u ^ s) for s, b1, b2, u in xor_states]
    assert sum(x[0] for x in xor_pair) == 1
    assert (exact_auc(xor_first), exact_auc(xor_second), exact_auc(xor_pair)) == (
        Fraction(1, 2), Fraction(1, 2), Fraction(1)
    )

    # Exercise the actual Team C adapter on L3. The two heads implement the
    # law's stated score1/score2 using only their own selected raw values.
    schema = AdultGroupedSchema()
    X = np.zeros((len(xor_states), 83))
    for i, (s, b1, b2, u) in enumerate(xor_states):
        X[i, 0] = b1
        X[i, 2] = u
        X[i, 3] = u ^ s
        X[i, 4] = b2
    first = PurposeRelease(
        schema, ("age", "capital-gain"), IncomeClueHead(), "income", "L3-head-1",
        "L3-identity-cal", IdentityCalibrator(),
    )
    second = PurposeRelease(
        schema, ("capital-loss", "hours-per-week"), OccupationClueHead(),
        "occupation_group", "L3-head-2", "L3-identity-cal", IdentityCalibrator(),
    )
    views = release_pair(X, schema.feature_names, first, second)
    def view_records(name):
        view = views[name]
        records = []
        for i, (s, _, _, _) in enumerate(xor_states):
            values = tuple(view["selected_values"][i].tolist())
            if name == "coalition":
                probabilities = tuple(view["income_probabilities"][i]) + tuple(
                    view["occupation_group_probabilities"][i]
                )
                metadata = (view["income_mask_id"], view["occupation_group_mask_id"])
            else:
                probabilities = tuple(view["probabilities"][i])
                metadata = (view["mask_id"], view["head_id"], view["calibrator_id"])
            records.append(record(Fraction(1, 16), s, values, probabilities, metadata))
        return records
    assert (exact_auc(view_records("income")), exact_auc(view_records("occupation_group")),
            exact_auc(view_records("coalition"))) == (Fraction(1, 2), Fraction(1, 2), 1)

    redundant_first, redundant_second, redundant_pair = [], [], []
    for s, b1, b2, e in itertools.product((0, 1), repeat=4):
        mass = Fraction(1, 8) * (Fraction(4, 5) if e == 0 else Fraction(1, 5))
        clue = s ^ e
        redundant_first.append(record(mass, s, b1, clue))
        redundant_second.append(record(mass, s, b2, clue))
        redundant_pair.append(record(mass, s, b1, clue, b2, clue))
    assert sum(x[0] for x in redundant_pair) == 1
    assert (exact_auc(redundant_first), exact_auc(redundant_second),
            exact_auc(redundant_pair)) == (Fraction(4, 5),) * 3

    def binary_nll(true_p, released_p):
        return -true_p * math.log(released_p) - (1 - true_p) * math.log(1 - released_p)

    def multiclass_brier(true_p, released_p):
        return 2 * (true_p * (1 - released_p) ** 2 + (1 - true_p) * released_p ** 2)

    confidence_nll = (binary_nll(0.60, 0.60) + binary_nll(0.95, 0.95)) / 2
    class_only_nll = (binary_nll(0.60, 0.775) + binary_nll(0.95, 0.775)) / 2
    confidence_brier = (multiclass_brier(0.60, 0.60) + multiclass_brier(0.95, 0.95)) / 2
    class_only_brier = (multiclass_brier(0.60, 0.775) + multiclass_brier(0.95, 0.775)) / 2
    assert class_only_nll - confidence_nll > 0.010
    assert class_only_brier - confidence_brier > 0.005

    report = {
        "status": "synthetic_checks_passed",
        "law_registry_sha256": actual_sha,
        "adult_data_or_fit_used": False,
        "L1_auc_local1_local2_coalition": [0.5, 0.5, 0.5],
        "L2_auc_before_after": [1.0, 0.5],
        "L3_auc_local1_local2_coalition": [0.5, 0.5, 1.0],
        "L3_actual_adapter_complete_view_auc": [0.5, 0.5, 1.0],
        "L4_auc_local1_local2_coalition": [0.8, 0.8, 0.8],
        "L5_class_only_nll_excess": class_only_nll - confidence_nll,
        "L5_class_only_multiclass_brier_excess": class_only_brier - confidence_brier,
        "interpretation": "Finite-law capability and interface checks, not Adult prevalence or privacy guarantees.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python finite_law_checks.py SYNTHETIC_LAWS.json")
    main(Path(sys.argv[1]))
