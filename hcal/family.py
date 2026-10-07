"""The registered primary family (23 slots), clause classifier and label truth table of the held-out calibration study
(hcal; prompt section 10; frozen in SCIENCE_LOCK).

Slots (nominee / reference roles are resolved from the EVALUATION_LOCK):
  P01        coalition  AUC_pair(T*) - AUC_pair(P*)                          lower bound > 0.02
  P02, P03   local      AUC_i(P*) - AUC_i(T*), i = income (v1), occupation (v2)   upper bound < 0.01
  P04, P05   acc        acc(P*) - acc(U0)                                    lower bound > -0.01
  P06, P07   logloss    LL(P*) - LL(U0)                                      upper bound < 0.01
  P08, P09   brier      Brier(P*) - Brier(U0)                                upper bound < 0.005
  P10, P11   retention  acc(P*) - 0.8 acc(U0) - 0.2 acc(const)               lower bound > 0
  P12, P13   logloss    LL(P*) - LL(Ucal*)                                   upper bound < 0.01
  P14, P15   brier      Brier(P*) - Brier(Ucal*)                             upper bound < 0.005
  D01-D04    fitting role, fixed legacy JOINT lambda 0.1: T-TOKEN32 minus H-TOKEN32 on income LL, occupation LL, income
             Brier, occupation Brier (positive = held-out fitting helped); signed, no target
  D05-D08    parameter sharing, same partition: H-TOKEN32 minus H-GLOBAL-TEMP, same order (positive = sharing helped)
z = NormalDist().inv_cdf(1 - 0.05 / (2 * 23)) = 3.0653831516447343; B = 1999; bootstrap seed 20261011.
"""
from __future__ import annotations

import math
from statistics import NormalDist

ALPHA = 0.05
PRIMARY_SIZE = 23
Z_PRIMARY = NormalDist().inv_cdf(1 - ALPHA / (2 * PRIMARY_SIZE))
assert abs(Z_PRIMARY - 3.0653831516447343) < 1e-15
Z_SUPPLEMENTARY = NormalDist().inv_cdf(0.975)
B, BOOT_SEED = 1999, 20261011
TASKS = ("income", "occupation")
DIAG_PARTITION = "U|JOINT|i8o64|l0.1"

PRIMARY = []
PRIMARY.append({"id": "P01", "group": "original", "kind": "coalition", "nominee": "P*", "ref": "T*",
                "stat": "AUC_pair(T*) - AUC_pair(P*)", "target": 0.02, "side": "lower>"})
for j, (i, v) in enumerate((("P02", "v1"), ("P03", "v2"))):
    PRIMARY.append({"id": i, "group": "original", "kind": "local", "view": v, "task": j, "nominee": "P*", "ref": "T*",
                    "stat": f"AUC_{v}(P*) - AUC_{v}(T*)", "target": 0.01, "side": "upper<"})
for base, kind, tgt, side, stat in ((4, "acc", -0.01, "lower>", "acc(P*) - acc(U0)"),
                                    (6, "logloss", 0.01, "upper<", "LL(P*) - LL(U0)"),
                                    (8, "brier", 0.005, "upper<", "Brier(P*) - Brier(U0)"),
                                    (10, "retention", 0.0, "lower>", "acc(P*) - 0.8 acc(U0) - 0.2 acc(const)")):
    for j in (0, 1):
        PRIMARY.append({"id": f"P{base + j:02d}", "group": "original", "kind": kind, "task": j, "nominee": "P*",
                        "ref": "U0", "stat": f"{stat} [{TASKS[j]}]", "target": tgt, "side": side})
for base, kind, tgt, stat in ((12, "logloss", 0.01, "LL(P*) - LL(Ucal*)"),
                              (14, "brier", 0.005, "Brier(P*) - Brier(Ucal*)")):
    for j in (0, 1):
        PRIMARY.append({"id": f"P{base + j:02d}", "group": "calibrated", "kind": kind, "task": j, "nominee": "P*",
                        "ref": "Ucal*", "stat": f"{stat} [{TASKS[j]}]", "target": tgt, "side": "upper<"})
DIAG_ORDER = ((0, "ll"), (1, "ll"), (0, "br"), (1, "br"))
for n, (j, kind) in enumerate(DIAG_ORDER):
    PRIMARY.append({"id": f"D{n + 1:02d}", "group": "fitting_role", "kind": "diag", "task": j, "loss": kind,
                    "partition": DIAG_PARTITION, "minuend": "T-TOKEN32", "subtrahend": "H-TOKEN32",
                    "stat": f"{'LL' if kind == 'll' else 'Brier'}(T-TOKEN32) - {'LL' if kind == 'll' else 'Brier'}"
                            f"(H-TOKEN32) [{TASKS[j]}], {DIAG_PARTITION}", "target": None, "side": "signed"})
for n, (j, kind) in enumerate(DIAG_ORDER):
    PRIMARY.append({"id": f"D{n + 5:02d}", "group": "parameter_sharing", "kind": "diag", "task": j, "loss": kind,
                    "partition": DIAG_PARTITION, "minuend": "H-TOKEN32", "subtrahend": "H-GLOBAL-TEMP",
                    "stat": f"{'LL' if kind == 'll' else 'Brier'}(H-TOKEN32) - {'LL' if kind == 'll' else 'Brier'}"
                            f"(H-GLOBAL-TEMP) [{TASKS[j]}], {DIAG_PARTITION}", "target": None, "side": "signed"})
assert len(PRIMARY) == PRIMARY_SIZE and len({e["id"] for e in PRIMARY}) == PRIMARY_SIZE
ORIGINAL_IDS = [e["id"] for e in PRIMARY if e["group"] == "original"]
CALIBRATED_IDS = [e["id"] for e in PRIMARY if e["group"] == "calibrated"]
FITTING_IDS = [e["id"] for e in PRIMARY if e["group"] == "fitting_role"]
SHARING_IDS = [e["id"] for e in PRIMARY if e["group"] == "parameter_sharing"]
assert len(ORIGINAL_IDS) == 11 and len(CALIBRATED_IDS) == 4 and len(FITTING_IDS) == 4 and len(SHARING_IDS) == 4

# supplementary fixed diagnostics (nominal 95%, descriptive; counted separately, never primary)
SUPPLEMENTARY_PARTITIONS = ("U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "U|CLASS|i1o1", DIAG_PARTITION)
SUPPLEMENTARY_CONTRASTS = (("fitting_role", "T-TOKEN32", "H-TOKEN32"), ("parameter_sharing", "H-TOKEN32", "H-GLOBAL-TEMP"),
                           ("class_vs_global", "H-GLOBAL-TEMP", "H-CLASS-TEMP"),
                           ("class_vs_token", "H-TOKEN32", "H-CLASS-TEMP"),
                           ("original_d0_vs_global", "D0", "H-GLOBAL-TEMP"), ("original_d1_vs_token", "D1", "H-TOKEN32"))


def _ok(x):
    return x is not None and math.isfinite(float(x))


def clause_outcome(side, target, point, lower, upper):
    """PASS: the registered bound clears the target. NOT_ESTABLISHED_PRECISION: the point satisfies the target but the
    bound does not. MEASURED_VIOLATION: the other bound excludes the target on the failing side. NOT_ESTABLISHED_POINT:
    the point fails but no violation is established. INVALID: any nonfinite input. (lra.family.clause_outcome.)"""
    if not all(_ok(x) for x in (point, lower, upper)):
        return "INVALID"
    if side == "lower>":
        if lower > target:
            return "PASS"
        if upper < target:
            return "MEASURED_VIOLATION"
        return "NOT_ESTABLISHED_PRECISION" if point > target else "NOT_ESTABLISHED_POINT"
    if upper < target:
        return "PASS"
    if lower > target:
        return "MEASURED_VIOLATION"
    return "NOT_ESTABLISHED_PRECISION" if point < target else "NOT_ESTABLISHED_POINT"


def diag_outcome(point, lower, upper):
    """Signed diagnostic: SUPPORTS_POSITIVE if lower > 0, SUPPORTS_NEGATIVE if upper < 0, else UNRESOLVED."""
    if not all(_ok(x) for x in (point, lower, upper)):
        return "INVALID"
    if lower > 0:
        return "SUPPORTS_POSITIVE"
    if upper < 0:
        return "SUPPORTS_NEGATIVE"
    return "UNRESOLVED"


def diag_summary(outcomes):
    """Four contrasts of one diagnostic group -> ALL_SUPPORT_POSITIVE / ALL_SUPPORT_NEGATIVE / MIXED / ... (never a
    favourable majority count)."""
    vals = list(outcomes)
    if any(v == "INVALID" for v in vals):
        return "INVALID"
    if all(v == "SUPPORTS_POSITIVE" for v in vals):
        return "ALL_FOUR_SUPPORT_POSITIVE"
    if all(v == "SUPPORTS_NEGATIVE" for v in vals):
        return "ALL_FOUR_SUPPORT_NEGATIVE"
    if all(v == "UNRESOLVED" for v in vals):
        return "ALL_FOUR_UNRESOLVED"
    return "MIXED"


# ------------------------------------------------------------------ truth table (prompt section 10)
LABELS = ("ENGINEERING_BLOCKED_NOT_RUN", "INPUTS_UNAVAILABLE_NOT_RUN", "INCOMPLETE_OR_INVALID",
          "NO_ELIGIBLE_COMPETITIVE_NOMINEE", "CALIBRATED_PRIVATE_RELEASE_DEVELOPMENT_CRITERION_ESTABLISHED",
          "ORIGINAL_REQUIREMENTS_MET_CALIBRATED_REFERENCE_NOT_ESTABLISHED", "NO_COMPETITIVE_RELEASE_CRITERION_ESTABLISHED")


def criteria(outcomes, nominee_valid, technical_valid=True, comparator_valid=True):
    """(OriginalCriterion, CalibrationMatchedCriterion) from the slot decisions; never PASS beside a technical defect
    or a missing comparator (MATH_REVIEW finding 3)."""
    if not technical_valid:
        return "INVALID_TECHNICAL", "INVALID_TECHNICAL"
    if not comparator_valid:
        return "INVALID_NO_COMPARATOR", "INVALID_NO_COMPARATOR"
    if not nominee_valid:
        return "NOT_TESTED_NO_NOMINEE", "NOT_TESTED_NO_NOMINEE"
    orig = "PASS" if all(outcomes.get(i) == "PASS" for i in ORIGINAL_IDS) else "NOT_ESTABLISHED"
    cal = "PASS" if orig == "PASS" and all(outcomes.get(i) == "PASS" for i in CALIBRATED_IDS) else "NOT_ESTABLISHED"
    return orig, cal


def overall_label(engineering_ready, inputs_available, technical_valid, nominee_state, comparator_state, outcomes):
    """Prompt section 10 precedence. nominee_state / comparator_state in {ELIGIBLE, NO_ELIGIBLE, TECHNICAL_FAILURE};
    outcomes: {slot id: clause decision} (P slots; D slots do not enter the label)."""
    if not engineering_ready:
        return "ENGINEERING_BLOCKED_NOT_RUN"
    if not inputs_available:
        return "INPUTS_UNAVAILABLE_NOT_RUN"
    if not technical_valid or nominee_state == "TECHNICAL_FAILURE" or comparator_state == "TECHNICAL_FAILURE":
        return "INCOMPLETE_OR_INVALID"
    if comparator_state != "ELIGIBLE":
        # no eligible task-only comparator is comparator coverage failure, not a method win (prompt section 9); a
        # missing required comparator makes the dependent claim INCOMPLETE_OR_INVALID
        return "INCOMPLETE_OR_INVALID"
    if nominee_state != "ELIGIBLE":
        return "NO_ELIGIBLE_COMPETITIVE_NOMINEE"
    if any(outcomes.get(i) == "INVALID" for i in ORIGINAL_IDS + CALIBRATED_IDS):
        return "INCOMPLETE_OR_INVALID"
    orig, cal = criteria(outcomes, True)
    if cal == "PASS":
        return "CALIBRATED_PRIVATE_RELEASE_DEVELOPMENT_CRITERION_ESTABLISHED"
    if orig == "PASS":
        return "ORIGINAL_REQUIREMENTS_MET_CALIBRATED_REFERENCE_NOT_ESTABLISHED"
    return "NO_COMPETITIVE_RELEASE_CRITERION_ESTABLISHED"


def truth_table():
    """Every registered case of overall_label (LABEL_TRUTH_TABLE.json)."""
    rows = []
    P = {i: "PASS" for i in ORIGINAL_IDS + CALIBRATED_IDS}
    cases = [
        ("engineering blocked", (False, True, True, "ELIGIBLE", "ELIGIBLE", P)),
        ("inputs unavailable", (True, False, True, "ELIGIBLE", "ELIGIBLE", P)),
        ("technical defect", (True, True, False, "ELIGIBLE", "ELIGIBLE", P)),
        ("nominee technical failure", (True, True, True, "TECHNICAL_FAILURE", "ELIGIBLE", P)),
        ("comparator technical failure", (True, True, True, "ELIGIBLE", "TECHNICAL_FAILURE", P)),
        ("no eligible comparator, nominee eligible", (True, True, True, "ELIGIBLE", "NO_ELIGIBLE", P)),
        ("no eligible comparator, no nominee", (True, True, True, "NO_ELIGIBLE", "NO_ELIGIBLE", P)),
        ("no eligible nominee", (True, True, True, "NO_ELIGIBLE", "ELIGIBLE", {})),
        ("all fifteen pass", (True, True, True, "ELIGIBLE", "ELIGIBLE", P)),
        ("eleven pass, P12 precision", (True, True, True, "ELIGIBLE", "ELIGIBLE",
                                        {**P, "P12": "NOT_ESTABLISHED_PRECISION"})),
        ("eleven pass, P15 violation", (True, True, True, "ELIGIBLE", "ELIGIBLE", {**P, "P15": "MEASURED_VIOLATION"})),
        ("P07 precision failure", (True, True, True, "ELIGIBLE", "ELIGIBLE", {**P, "P07": "NOT_ESTABLISHED_PRECISION"})),
        ("P01 point failure", (True, True, True, "ELIGIBLE", "ELIGIBLE", {**P, "P01": "NOT_ESTABLISHED_POINT"})),
        ("invalid slot", (True, True, True, "ELIGIBLE", "ELIGIBLE", {**P, "P03": "INVALID"})),
    ]
    for name, args in cases:
        tech = args[2] and args[3] != "TECHNICAL_FAILURE" and args[4] != "TECHNICAL_FAILURE"
        rows.append({"case": name, "label": overall_label(*args),
                     "criteria": criteria(args[5], args[3] == "ELIGIBLE", tech, args[4] == "ELIGIBLE")})
    return rows
