"""[lra port of lcr/fixtures.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md]
Known-law fixtures and the mechanism gate of the lra study (role B; prompt sec. 9). SYNTHETIC ONLY: no real data.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.fixtures laws [--write]   # build / verify the laws
    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.fixtures run [--out DIR]  # the fixture stage
    (the registered stage runs through lra.run --lock FIXTURE_LOCK --stage fixture -> stage_fixture(None, shard))

LAWS (PKG/FIXTURE_LAWS.json; written and sent to the lead BEFORE any algorithm ran on them)
  Four fixed families, N = 4096 exact expected counts (atom weights = count / 4096):
    F1_CALIBRATED_NULL     teacher = true conditional per fine cell; labels = exact expectation
    F2_MISCALIBRATED       over/under-confident teacher, K2 = 3, labels favouring a non-predicted class in some cells
    F3_COMPLEMENTARY_XOR   SEX = XOR of one clue bit per recipient (w.p. 3/4): nothing locally, 0.13 nats jointly
    F4_REDUNDANT           the SAME clue bit visible to both recipients: local = pair information
  Each family: per recipient K, fine cells (= the allowed observations: one distinct dyadic teacher vector per cell,
  predicted class = its strict argmax), per-cell label laws, explicit cell-pair counts n(f1, f2), explicit SEX laws
  P(S = 1 | f1, f2), the label-joint rule and the complete atom list (f1, f2, s, y1, y2, count). Labels are
  conditionally independent of SEX given (f1, f2) and each recipient's labels are split proportionally inside every
  (f1, f2, s) group (exact integers by construction); the joint (y1, y2) inside a group is the north-west-corner
  table of the two integer margins (classes ascending). No gate quantity depends on that y1-y2 coupling (decoders and
  budgets are per recipient; MI uses tokens and SEX only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_adult_learned_decoder_release_v1"
N_FIX = 4096
LAWS_SCHEMA = "lra-fixture-laws-v1"
GATE_SCHEMA = "lra-fixture-gate-rule-v1"
BUDGET = {"ll": 0.005, "brier": 0.003}
CAPS = (2, 2)                       # tokens per predicted class (recipient 1, recipient 2); 4 fine cells per class
LAMS = (0.01, 0.025, 0.04, 0.06, 0.08, 0.1)
MAX_PAIRS = 100_000


# ----------------------------------------------------------------------------------------------- law construction
def _cells_binary(correct_teacher_128, correct_labels, label_den):
    """Two classes x 4 cells; class 1 mirrors class 0. correct_* are the probabilities of the PREDICTED class."""
    cells = []
    for c in (0, 1):
        for j, (pt, pl) in enumerate(zip(correct_teacher_128, correct_labels)):
            t = [128 - pt, pt] if c == 1 else [pt, 128 - pt]
            lab = [label_den - pl, pl] if c == 1 else [pl, label_den - pl]
            cells.append({"class": c, "j": j, "teacher": t, "labels": lab, "count": 512})
    return cells


def _spec_F1():
    r1 = _cells_binary([112, 104, 88, 80], [14, 13, 11, 10], 16)          # teacher = labels (16ths = 128ths / 8)
    r2 = _cells_binary([120, 96, 88, 72], [15, 12, 11, 9], 16)
    hi = lambda c: 1 if c["j"] in (0, 1) else 0                           # noqa: E731  high-confidence half
    pc = [[64 for _ in r2] for _ in r1]
    sx = [[1 if hi(a) and hi(b) else (3 if not hi(a) and not hi(b) else 2) for b in r2] for a in r1]
    return {"id": "F1_CALIBRATED_NULL", "label_den": [16, 16], "sex_den": 4, "r1": r1, "r2": r2,
            "pair_counts": pc, "sex_num": sx,
            "purpose": "Calibrated-teacher null: every fine cell's teacher vector equals its true conditional label "
                       "law and the fitting label counts equal their exact expectation, so for EVERY token (any union "
                       "of same-class cells) y_t = s_t and the mean decoder is the population Bayes rule. D1 must not "
                       "improve population log loss or Brier beyond numerical tolerance (prompt sec. 2 null). SEX "
                       "depends additively on the confidence half of both recipients' cells, so privacy arms have a "
                       "real but ordinary trade-off.",
            "design": "R1/R2: 2 classes x 4 cells, 512 rows per cell, cells independent across recipients (64 rows "
                      "per cell pair). P(S=1|f1,f2) = 1/4 if both cells are high-confidence (j in {0,1}), 3/4 if both "
                      "are low-confidence, 1/2 otherwise."}


def _spec_F2():
    r1 = _cells_binary([120, 112, 104, 96], [6, 6, 5, 5], 8)              # teacher overconfident vs 8ths labels
    t0 = [[96, 16, 16], [80, 32, 16], [80, 16, 32], [64, 32, 32]]
    l0 = [[5, 2, 1], [4, 3, 1], [4, 1, 3], [3, 3, 2]]
    t1 = [[32, 80, 16], [16, 96, 16], [48, 64, 16], [32, 64, 32]]
    l1 = [[3, 4, 1], [2, 5, 1], [4, 3, 1], [3, 3, 2]]
    t2 = [[16, 16, 96], [32, 16, 80], [16, 32, 80], [32, 32, 64]]
    l2 = [[1, 2, 5], [2, 1, 5], [1, 3, 4], [3, 2, 3]]
    r2 = []
    for c, (T_, L_, cnt) in enumerate(((t0, l0, 512), (t1, l1, 256), (t2, l2, 256))):
        for j in range(4):
            r2.append({"class": c, "j": j, "teacher": T_[j], "labels": L_[j], "count": cnt})
    pc = [[a["count"] * b["count"] // N_FIX for b in r2] for a in r1]
    sx = [[3 if b["j"] in (0, 3) else 1 for b in r2] for a in r1]
    return {"id": "F2_MISCALIBRATED", "label_den": [8, 8], "sex_den": 4, "r1": r1, "r2": r2, "pair_counts": pc,
            "sex_num": sx,
            "purpose": "Miscalibrated teacher: the binary head is overconfident (teacher 0.94-0.75 vs true 0.75 / "
                       "0.625) and the 3-class head is miscalibrated, including cells whose labels favour a "
                       "NON-predicted class (class-1 cell j=2: teacher (0.375, 0.5, 0.125), labels (0.5, 0.375, "
                       "0.125)) so the class-dominance constraint binds. D1 can improve confidence on a FIXED code; "
                       "the full-token information of that code is unchanged.",
            "design": "R1: 2 classes x 4 cells x 512 rows; R2: K=3, class 0 4 cells x 512, classes 1 and 2 4 cells x "
                      "256; cells independent across recipients. SEX depends on recipient 2 only: P(S=1|f2) = 3/4 for "
                      "within-class cells j in {0, 3}, 1/4 for j in {1, 2} (recipient 1 carries no SEX information)."}


def _spec_xor_cells():
    # within-class index j = 2a + b: a = near-irrelevant teacher shift (labels equal across a), b = clue bit
    r1 = _cells_binary([87, 95, 89, 97], [11, 12, 11, 12], 16)
    r2 = _cells_binary([95, 103, 97, 105], [12, 13, 12, 13], 16)
    for r in (r1, r2):
        for c in r:
            c["a"], c["b"] = c["j"] // 2, c["j"] % 2
    return r1, r2


def _spec_F3():
    r1, r2 = _spec_xor_cells()
    pc = [[64 for _ in r2] for _ in r1]
    sx = [[3 if a["b"] != b["b"] else 1 for b in r2] for a in r1]
    return {"id": "F3_COMPLEMENTARY_XOR", "label_den": [16, 16], "sex_den": 4, "r1": r1, "r2": r2,
            "pair_counts": pc, "sex_num": sx,
            "purpose": "Complementary clues: SEX = b1 XOR b2 with probability 3/4 (b_i a clue bit of recipient i's "
                       "fine cell, independent uniform). Each recipient alone carries exactly zero SEX information; "
                       "the pair carries log 2 - H(3/4) = 0.1308 nats while both bits are visible. The clue bit is "
                       "mildly task-relevant (correct rate 11/16 vs 12/16 for recipient 1, 12/16 vs 13/16 for "
                       "recipient 2) and the other bit a is task-irrelevant (equal label rates, teacher shifted by "
                       "1/64), so task-only compression to 2 tokens per class merges along a and keeps the clue. "
                       "Nonconstant useful tasks; every release is class-preserving.",
            "design": "Both recipients: 2 classes x 4 cells (a, b) x 512 rows; cells independent across recipients "
                      "(64 rows per cell pair). P(S=1|f1,f2) = 3/4 if b1 != b2 else 1/4."}


def _spec_F4():
    r1, r2 = _spec_xor_cells()
    pc = [[128 if a["b"] == b["b"] else 0 for b in r2] for a in r1]
    sx = [[(3 if a["b"] == 1 else 1) if a["b"] == b["b"] else 0 for b in r2] for a in r1]
    return {"id": "F4_REDUNDANT", "label_den": [16, 16], "sex_den": 4, "r1": r1, "r2": r2, "pair_counts": pc,
            "sex_num": sx,
            "purpose": "Redundant / no-coordination case: one latent clue bit b is visible to BOTH recipients "
                       "(b1 = b2 = b) and P(S=1|b) = 3/4 or 1/4, so each recipient's local information equals the pair "
                       "information (0.1308 nats while b is visible anywhere). Removing it requires each recipient to "
                       "hide b on its own; no coordination is needed, so sequential and joint search should not "
                       "differ meaningfully.",
            "design": "Same cells as F3. n(f1, f2) = 128 if b(f1) = b(f2) else 0 (the other coordinates independent "
                      "uniform given b); P(S=1|b) = 3/4 if b = 1 else 1/4."}


SPECS = (_spec_F1, _spec_F2, _spec_F3, _spec_F4)


def _nw_corner(m1, m2):
    """North-west-corner integer table with margins m1 (rows) and m2 (columns), classes ascending."""
    m1, m2 = list(m1), list(m2)
    out = np.zeros((len(m1), len(m2)), dtype=np.int64)
    i = j = 0
    while i < len(m1) and j < len(m2):
        x = min(m1[i], m2[j])
        out[i, j] += x
        m1[i] -= x
        m2[j] -= x
        if m1[i] == 0:
            i += 1
        if j < len(m2) and m2[j] == 0:
            j += 1
    if sum(m1) or sum(m2):
        raise ValueError("margins do not match")
    return out


def atoms_from_spec(sp):
    """Exact integer atoms [f1, f2, s, y1, y2, count] (nonzero, lexicographic) from an explicit family spec."""
    r1, r2 = sp["r1"], sp["r2"]
    den1, den2 = sp["label_den"]
    sden = sp["sex_den"]
    atoms = []
    for f1, a in enumerate(r1):
        for f2, b in enumerate(r2):
            n = int(sp["pair_counts"][f1][f2])
            if n == 0:
                continue
            ns1 = Fraction(n * sp["sex_num"][f1][f2], sden)
            if ns1.denominator != 1:
                raise ValueError(f"{sp['id']}: SEX count not integral at ({f1}, {f2})")
            for s, g in ((0, n - int(ns1)), (1, int(ns1))):
                if g == 0:
                    continue
                m1 = [Fraction(g * x, den1) for x in a["labels"]]
                m2 = [Fraction(g * x, den2) for x in b["labels"]]
                if any(v.denominator != 1 for v in m1 + m2):
                    raise ValueError(f"{sp['id']}: label counts not integral in group ({f1}, {f2}, {s}) of {g}")
                tab = _nw_corner([int(v) for v in m1], [int(v) for v in m2])
                for y1 in range(tab.shape[0]):
                    for y2 in range(tab.shape[1]):
                        if tab[y1, y2]:
                            atoms.append([f1, f2, s, y1, y2, int(tab[y1, y2])])
    return atoms


def _bell_cap(F, m):
    """Number of set partitions of F labelled cells into at most m blocks (sum of Stirling numbers S(F, j), j <= m)."""
    S = [[0] * (F + 1) for _ in range(F + 1)]
    S[0][0] = 1
    for n in range(1, F + 1):
        for k in range(1, n + 1):
            S[n][k] = k * S[n - 1][k] + S[n - 1][k - 1]
    return sum(S[F][j] for j in range(1, min(m, F) + 1))


def law_properties(law):
    """Static law properties (no algorithm): counts, teacher argmax, constants, accuracy gains, enumeration size."""
    A = np.asarray(law["atoms"], dtype=np.int64)
    out = {"rows": int(A[:, 5].sum())}
    for r, key in ((1, "r1"), (2, "r2")):
        cells = law["recipients"][str(r)]["cells"]
        K = law["recipients"][str(r)]["K"]
        fcol, ycol = (0, 3) if r == 1 else (1, 4)
        cnt = np.bincount(A[:, fcol], weights=A[:, 5], minlength=len(cells))
        ycnt = np.bincount(A[:, ycol], weights=A[:, 5], minlength=K)
        correct = sum(int(A[i, 5]) for i in range(A.shape[0]) if A[i, ycol] == cells[A[i, fcol]]["class"])
        const = int(np.argmax(ycnt))
        acc, cacc = correct / N_FIX, float(ycnt[const] / N_FIX)
        per_class = [sum(1 for c in cells if c["class"] == k) for k in range(K)]
        out[f"r{r}"] = {"cell_counts": [int(x) for x in cnt],
                        "cell_counts_match_declared": bool(all(int(cnt[i]) == c["count"] for i, c in enumerate(cells))),
                        "label_counts": [int(x) for x in ycnt], "constant_class": const, "constant_acc": cacc,
                        "teacher_decision_acc": acc, "accuracy_gain": acc - cacc,
                        "cells_per_class": per_class,
                        "canonical_partitions": int(np.prod([_bell_cap(F, law["caps"][r - 1]) for F in per_class]))}
    out["mapping_pairs"] = out["r1"]["canonical_partitions"] * out["r2"]["canonical_partitions"]
    sx = np.bincount(A[:, 2], weights=A[:, 5], minlength=2)
    out["sex_counts"] = [int(x) for x in sx]
    return out


def _check_cell(c, K, tden):
    t = c["teacher"]
    if len(t) != K or sum(t) != tden or min(t) < 0:
        raise ValueError("teacher vector must be K nonnegative numerators summing to the denominator")
    if sorted(t)[-1] == sorted(t)[-2] or int(np.argmax(t)) != c["class"]:
        raise ValueError("teacher vector argmax must be strictly the cell's class")


def build_law(sp):
    r1, r2 = sp["r1"], sp["r2"]
    K1 = len(r1[0]["teacher"])
    K2 = len(r2[0]["teacher"])
    for c in r1:
        _check_cell(c, K1, 128)
    for c in r2:
        _check_cell(c, K2, 128)
    atoms = atoms_from_spec(sp)
    rec = lambda cells, K, den: {"K": K, "teacher_den": 128, "label_den": den, "cells": [  # noqa: E731
        {k: v for k, v in c.items()} for c in cells]}
    law = {"id": sp["id"], "N": N_FIX, "purpose": sp["purpose"], "design": sp["design"],
           "recipients": {"1": rec(r1, K1, sp["label_den"][0]), "2": rec(r2, K2, sp["label_den"][1])},
           "caps": list(CAPS), "budget": dict(BUDGET), "kappa": 32.0, "eps": 1e-12, "lam_grid": list(LAMS),
           "pair_counts": sp["pair_counts"], "sex_den": sp["sex_den"], "sex_num": sp["sex_num"],
           "label_joint_rule": "inside each (f1, f2, s) group of g rows: recipient-i label counts = g * labels_i(f_i) "
                               "/ label_den_i (exact integers); joint (y1, y2) = north-west-corner table of the two "
                               "margins, classes ascending",
           "atoms": atoms}
    law["properties"] = law_properties(law)
    if law["properties"]["rows"] != N_FIX:
        raise ValueError(f"{sp['id']}: atoms sum to {law['properties']['rows']}, not {N_FIX}")
    if law["properties"]["mapping_pairs"] > MAX_PAIRS:
        raise ValueError(f"{sp['id']}: {law['properties']['mapping_pairs']} mapping pairs exceed {MAX_PAIRS}")
    return law


def build_laws():
    fams = [build_law(f()) for f in SPECS]
    body = {"schema": LAWS_SCHEMA, "N": N_FIX, "weights": "atom probability = count / 4096 (integer multiples of "
            "1/4096); the fitting rows are the atoms replicated by count (exact expected counts, no sampling)",
            "recipient_roles": {"1": "income-like task (binary)", "2": "occupation-like task (binary or 3-class)"},
            "allowed_observations": "recipient i observes ONLY its fine cell, i.e. the teacher probability vector "
                                    "of that cell (teacher[f] / teacher_den); the predicted class is its strict "
                                    "argmax. Fine cells are fixed and label-blind (one distinct teacher vector per "
                                    "cell); a coarse token is a union of same-class fine cells",
            "sensitive": "SEX in {0, 1}; never an input of any decoder or deployment; used only by privacy-trained "
                         "partition search and by the MI accounting",
            "decoders": "D0 = smoothed mean teacher vector of the token (qpc); D1 = lra.decoder (kappa = 32, eps = "
                        "1e-12, class-dominant simplex) from the token's fitting label counts and teacher sums",
            "budgets": {"ll": "L_i <= L_i(U) + 0.005 nats (true-label log loss on the fixture rows, clip 1e-12)",
                        "brier": "B_i <= B_i(U) + 0.003 (source multiclass Brier)",
                        "decisions": "released decision == teacher decision on every row",
                        "caps": "at most caps[i-1] tokens per predicted class",
                        "local_information": "constrained (K-) arms and the gate's privacy-trained candidates: I_i <= "
                                             "I_i(C-TASK) of the same fixture"},
            "U": "the continuous teacher release: each row's own teacher vector (its fine cell's vector)",
            "lam_grid": list(LAMS), "families": fams,
            "construction": "lra/fixtures.py build_laws() (SPECS); stage_fixture re-builds the atoms from the explicit "
                            "pair_counts / sex_num / label tables stored here and refuses any mismatch",
            "bank_note": "fixed finite bank of exactly these four families, designed by reasoning before any fixture "
                         "algorithm ran; only static law properties (counts, integrality, teacher argmax, constants, "
                         "accuracy gains, enumeration size) were computed before the lock"}
    body["laws_sha256"] = laws_hash(body)
    return body


def laws_hash(body):
    z = {k: v for k, v in body.items() if k != "laws_sha256"}
    return hashlib.sha256(json.dumps(z, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_laws(path=None):
    p = Path(path) if path else PKG / "FIXTURE_LAWS.json"
    body = json.loads(p.read_text())
    if body.get("schema") != LAWS_SCHEMA or body.get("laws_sha256") != laws_hash(body):
        raise ValueError("FIXTURE_LAWS.json schema/hash mismatch")
    for fam in body["families"]:
        sp = {"id": fam["id"], "r1": fam["recipients"]["1"]["cells"], "r2": fam["recipients"]["2"]["cells"],
              "label_den": [fam["recipients"]["1"]["label_den"], fam["recipients"]["2"]["label_den"]],
              "sex_den": fam["sex_den"], "pair_counts": fam["pair_counts"], "sex_num": fam["sex_num"]}
        if atoms_from_spec(sp) != fam["atoms"]:
            raise ValueError(f"{fam['id']}: stored atoms differ from the explicit law tables")
        if law_properties(fam) != fam["properties"]:
            raise ValueError(f"{fam['id']}: stored properties differ")
    return body


# ----------------------------------------------------------------------------------------------- gate rule
TOL_BUDGET = 1e-12          # summation-order allowance on the budget comparisons (nothing else)
TOL_MI = 1e-12              # local-information cap allowance (plug-in MI summation order)
TOL_TERMS = 1e-10           # reported vs independently reconstructed loss / MI terms
TOL_NULL_Q = 1e-9           # F1: max |q_D1 - q_D0| per token
TOL_NULL_LOSS = 1e-12       # F1: per-row population loss of D1 may not be below D0 by more than this
TOL_OPT = 1e-10             # arm objective within this of the exhaustive optimum -> EXHAUSTIVE_OPTIMAL
TRIGGER_MI = 0.01
MIN_GAIN = 0.03
TASK_ONLY_D1 = ("U|C-TASK|i8o64|D1", "U|FINE-TASK|i8o64|D1", "U|DIRECT-TASK|i8o64|D1", "U|CLASS|i1o1|D1")


def gate_rule():
    from lra import run as R
    return {
        "schema": GATE_SCHEMA,
        "trigger_verbatim": "At least one fixed nontrivial fixture shows that privacy-trained partitioning with D1 "
                            "satisfies all utility/local budgets and reduces pair MI by at least 0.01 nats beyond the "
                            "strongest feasible task-only D1 compression, while both tasks have accuracy gain at least "
                            "0.03 over their constants.",
        "not_required": ["joint beating sequential", "constrained search beating a weighted control"],
        "rows": "each fixture's fitting rows are its atoms replicated by count (N = 4096, exact expected counts, no "
                "sampling); all rows are fitting rows (tr = all); population law = fitting law",
        "measurements": {
            "independence": "every quantity below is recomputed by lra.fixtures from the release rows (tokens, q, "
                            "decisions) and the fixture labels/SEX with its own code (dpc.utility.per_row conventions "
                            "for losses, its own plug-in MI); mapper-reported values are only cross-checked",
            "L_i": "mean true-label log loss of q_i, natural log, clip 1e-12", "B_i": "mean multiclass Brier of q_i",
            "U": "L_i(U), B_i(U) of each row's own teacher vector",
            "I_i": "plug-in MI(SEX; full token identity of recipient i), natural log, exact counts",
            "I12": "plug-in MI(SEX; exact token tuple (t1, t2)), natural log, exact counts",
            "T": "L_1 + L_2 + 0.5 (B_1 + B_2)", "Phi": "I12 + 0.5 (I_1 + I_2)",
            "accuracy_gain_i": "accuracy of the teacher decision minus accuracy of the fitting-majority constant (ties "
                               "-> lower class); identical for every class-preserving release (static law property)"},
        "feasible": {"rule": "for i = 1, 2: L_i <= L_i(U) + 0.005 + TOL_BUDGET and B_i <= B_i(U) + 0.003 + "
                             "TOL_BUDGET; released decision == teacher decision on every row; q_i normalised within "
                             "1e-12 with strict argmax at the decision; tokens per predicted class <= caps[i-1]",
                     "TOL_BUDGET": TOL_BUDGET},
        "local_budget": {"rule": "I_i <= I_i(C-TASK release of the same fixture) + TOL_MI, i = 1, 2", "TOL_MI": TOL_MI},
        "task_only_D1_candidates": list(TASK_ONLY_D1),
        "task_only_note": "C-TASK = the REFINED map returned by lra.mapper.fit_unit('C-TASK') (its D1-recomputed "
                          "start, FINE-TASK|D1, is a separate candidate); FINE-TASK|D1 / DIRECT-TASK|D1 / CLASS|D1 = "
                          "lra.decoder D1 on the D0 maps of the fixture (qpc.compress FINE-TASK / CLASS, qpc.kmeans "
                          "direct per-class KL k-means at the cap, all three registered starts), assignments unchanged",
        "T_star": "the FEASIBLE task-only D1 candidate with the smallest I12 (ties -> smaller T -> config id); none "
                  "feasible -> the fixture cannot trigger (reason NO_FEASIBLE_TASK_ONLY)",
        "privacy_trained_D1_candidates": {
            "d1_fixed": [R.d1_id(f, l) for l in LAMS for f in R.PRIVACY],
            "weighted": [R.weighted_id(f, l) for l in LAMS for f in R.PRIVACY],
            "constrained": [R.constrained_id(a) for a in R.CONSTRAINED]},
        "privacy_note": "d1_fixed = D1 on the 24 D0 privacy maps fitted on the fixture rows by the UNCHANGED qpc "
                        "old-objective search (qpc.compress.fit_policy_pair: LOCAL, SEQ-12, SEQ-21, JOINT x lambda "
                        "grid, JOINT with its same-lambda FINE-TASK / LOCAL / SEQ witnesses as in cbp), then decoded by "
                        "lra.decoder with the assignments unchanged; weighted / constrained = lra.mapper fit_unit (meta "
                        "fixture = True, refs caps = the fixture caps, refs budget = the registered 0.005 / 0.003, "
                        "starts / witnesses = the full registered sets of mapper.registered_starts). Exhaustive "
                        "partitions are references, NEVER candidates.",
        "qualifies": "feasible(R) and local_budget(R) and I12(T_star) - I12(R) >= 0.01 and, for R itself, "
                     "accuracy_i(R) - constant_accuracy_i >= 0.03 for i = 1 and 2 (accuracy of R's released decisions "
                     "on the fixture rows; the constant is the fitting-majority class, ties -> lower class)",
        "trigger_fixture": "some privacy-trained D1 candidate qualifies (nontrivial = both static accuracy gains >= "
                           "0.03 and I12(T_star) >= 0.01; a trivial fixture cannot trigger)",
        "gate": {"GATE_MET": "every mandatory correctness check passes on all four fixtures AND trigger_fixture holds "
                             "for at least one fixture",
                 "GATE_NOT_MET": "otherwise; reasons listed: CORRECTNESS_FAILURE:<check ids> and/or "
                                 "NO_FIXTURE_TRIGGERED"},
        "route_classification": {
            "applies": "only when GATE_MET; computed per triggered fixture, then combined",
            "per_fixture": {"Q_fm": "qualifying d1_fixed candidates", "Q_new": "qualifying weighted + constrained",
                            "best_fm": "min I12 over Q_fm (+inf if empty)", "best_new": "min I12 over Q_new",
                            "decoder_enabled": "Q_fm nonempty",
                            "assignment_search_helps": "Q_new nonempty and (Q_fm empty or best_new <= best_fm - 0.01)"},
            "ASSIGNMENT_SEARCH_ROUTE": "assignment_search_helps on at least one triggered fixture",
            "DECODER_ENABLED_ROUTE": "otherwise (the trigger is met by an OLD assignment decoded by D1 and new search "
                                     "adds < 0.01 nats there); Adult then includes all calibration controls",
            "descriptive_flags": ["d0_version_already_qualifies (the D0-decoded version of a qualifying fixed map is "
                                  "itself feasible, local-feasible and 0.01 below T_star)",
                                  "constrained_qualifies", "weighted_qualifies", "best constrained vs best weighted I12",
                                  "joint vs sequential I12"],
            "descriptive_definitions": "computed on every fixture (triggered or not) by descriptive(): for each group, min I12 over its privacy-trained D1 arms twice, over all arms and over the eligible ones (feasible() and, when a C-TASK release exists, local_budget()), None when empty. Groups: constrained = the 5 K- arms; weighted = the W- arms; joint vs sequential within each family: d1_fixed JOINT vs SEQ-12/SEQ-21, weighted W-JOINT vs W-SEQ-12/W-SEQ-21, constrained K-JOINT-SINGLE and K-JOINT-PAIR vs K-SEQ-12/K-SEQ-21; differences = best_joint - best_sequential and best_constrained - best_weighted. Descriptive only; never used by the verdict or route."},
        "mandatory_correctness_checks": {
            "C1_FIXED_TOKEN_INFORMATION": "for every D0 map and its D1 version: identical token arrays (exact), "
                                          "identical I_1, I_2, I12 (bitwise), and I(SEX; (token, q)) == I(SEX; token) "
                                          "within 1e-15 for both decoders (q is a function of the token); the last condition is also checked on every release of every arm (D0, D1 and mapper C-TASK / W- / K-)",
            "C2_CALIBRATED_NULL": "F1_CALIBRATED_NULL: for every canonical partition of each recipient and every D1 "
                                  f"release of any arm: max |q_D1 - q_D0| <= {TOL_NULL_Q} per token (oracle subset "
                                  "solves and every arm's released q against the D0 decoding of the same pair) and the "
                                  f"law (population) log loss and Brier of D1 >= those of D0 - {TOL_NULL_LOSS} per row",
            "C3_BUDGET_ENFORCEMENT": "every constrained (K-) mapper release reported FEASIBLE satisfies feasible() and "
                                     "local_budget() under the independent recomputation; every unconstrained "
                                     "mapper release (C-TASK, W-) reports FEASIBLE for its own constraints only "
                                     "(decision preservation and caps; its fitting budgets are not enforced and are "
                                     "evaluated by feasible() at qualification); mapper-reported L_i(U), B_i(U) "
                                     "equal the independent values within 1e-12; only K- arms may report INFEASIBLE",
            "C4_DECISION_PRESERVATION": "every release (D0, D1, mapper): hard_i == teacher decision on every row, q_i "
                                        "strict argmax at the decision and normalised within 1e-12; release keys "
                                        "exactly row_id, tok1, q1, hard1, alpha1, tok2, q2, hard2, alpha2",
            "C5_TERM_RECONSTRUCTION": "mapper final_state_terms and deployed terms (L1, L2, B1, B2, I1, I2, I12, T, "
                                      f"Phi) equal the independent reconstruction within {TOL_TERMS}; decoder.json "
                                      "loads with re-solve verification; the exhaustive table's terms for each arm's "
                                      f"partition equal the row-level reconstruction within {TOL_TERMS}; the winning "
                                      "start's INCREMENTAL search-state terms (mapper start records) equal the "
                                      f"from-scratch final_state_terms within {TOL_TERMS}; decoder.token_losses totals "
                                      "equal the row-level sums within 1e-9 (absolute, on totals)",
            "C6_HEURISTIC_LABELLING": "every arm carries EXHAUSTIVE_OPTIMAL (registered objective within "
                                      f"{TOL_OPT} of the exhaustive optimum of its own registered problem, feasible "
                                      "where constrained) or HEURISTIC with its gap, or NOT_A_SEARCH for releases "
                                      "that are not partition searches (own_problem_objectives); a heuristic gap is "
                                      "reported, not a failure",
            "C7_LAW_INTEGRITY": "laws hash verifies, atoms rebuild from the explicit tables, 4096 rows, static "
                                "properties match, mapping pairs <= 100000, fixture rows deploy to their declared "
                                "fine cells; on the registered run FIXTURE_LAWS.json and FIXTURE_GATE_RULE.json equal "
                                "the latest lock's documents_sha256, the rule binds the same laws_sha256, and "
                                "reloaded shard units carry the same laws_sha256"},
        "check_ids": ["C1_FIXED_TOKEN_INFORMATION", "C2_CALIBRATED_NULL", "C3_BUDGET_ENFORCEMENT",
                      "C4_DECISION_PRESERVATION", "C5_TERM_RECONSTRUCTION", "C6_HEURISTIC_LABELLING",
                      "C7_LAW_INTEGRITY"],
        "check_scope": "C2 is evaluated on F1_CALIBRATED_NULL only; every other check on all four fixtures",
        "own_problem_objectives": {
            "D0 FINE-TASK": "min D_1 + D_2 (teacher KL of the smoothed mean decoder, per N) s.t. caps",
            "D0 LOCAL lam": "min D + lam (I_1 + I_2) / 2 s.t. caps",
            "D0 SEQ-12 / SEQ-21 / JOINT lam": "min D + lam ((I_1 + I_2) / 2 + I12) s.t. caps",
            "C-TASK": "min T (D1) s.t. caps", "W-LOCAL lam": "min T + lam (I_1 + I_2) / 2 (D1) s.t. caps",
            "W-SEQ-12 / W-SEQ-21 / W-JOINT lam": "min T + lam Phi (D1) s.t. caps",
            "K-LOCAL": "per recipient min I_i s.t. its own budgets and local cap (D1)",
            "K-SEQ-12 / K-SEQ-21 / K-JOINT-SINGLE / K-JOINT-PAIR": "min Phi s.t. both budgets and both local caps (D1)",
            "DIRECT-TASK, CLASS": "not partition searches over the fine family (labelled NOT_A_SEARCH)",
            "D1 fixed-map controls (<D0 map>|D1)": "not searches: the D0 assignment is kept and only the decoder "
                                                   "changes (labelled NOT_A_SEARCH)"},
        "tolerances": {"TOL_BUDGET": TOL_BUDGET, "TOL_MI": TOL_MI, "TOL_TERMS": TOL_TERMS, "TOL_NULL_Q": TOL_NULL_Q,
                       "TOL_NULL_LOSS": TOL_NULL_LOSS, "TOL_OPT": TOL_OPT, "TRIGGER_MI": TRIGGER_MI,
                       "MIN_GAIN": MIN_GAIN},
        "verdict_strings": ["GATE_MET", "GATE_NOT_MET"],
        "scope": "a pass licenses the Adult study only through SCIENCE_LOCK; a failure means MECHANISM_GATE_NOT_MET "
                 "and no Adult claim. Fixture MI is the exact law MI of these finite laws, not a population guarantee "
                 "for Adult."}


def write_law_files(pkg=PKG):
    laws = build_laws()
    (Path(pkg) / "FIXTURE_LAWS.json").write_text(json.dumps(laws, indent=1, allow_nan=False) + "\n")
    rule = gate_rule()
    rule["laws_sha256"] = laws["laws_sha256"]
    (Path(pkg) / "FIXTURE_GATE_RULE.json").write_text(json.dumps(rule, indent=1, allow_nan=False) + "\n")
    return laws["laws_sha256"]



# ----------------------------------------------------------------------------------------------- fixture rows
def _hex(tag):
    return hashlib.sha256(f"lra-fixture|{tag}".encode()).hexdigest()


def fixture_meta(fid, cid=None):
    """Dummy 64-hex bindings (fixtures have no real teacher/schema) + the mapper's fixture flag."""
    m = {"teacher": "U", "seed": 0, "fixture": True, "fixture_id": fid,
         "teacher_model_sha256": _hex(f"teacher|{fid}"), "feature_names_sha256": _hex("schema")}
    if cid is not None:
        m["config"] = cid
    return m


def fixture_rows(fam):
    """Exact-count rows of one family: atoms replicated by count (lexicographic atom order), teacher outputs from the
    fine-cell vectors, fine partitions (one cell per declared cell) whose statistics equal deployment on the rows."""
    from qpc import kmeans as KM
    A = np.asarray(fam["atoms"], dtype=np.int64)
    rep = np.repeat(np.arange(A.shape[0]), A[:, 5])
    f1, f2, s, y1, y2 = (A[rep, j] for j in range(5))
    out = {"N": int(rep.size), "f1": f1, "f2": f2, "s": s, "y": {1: y1, 2: y2}, "fine": {}, "P": {}, "d": {}}
    for r, f in ((1, f1), (2, f2)):
        R = fam["recipients"][str(r)]
        V = np.asarray([c["teacher"] for c in R["cells"]], dtype=np.float64) / float(R["teacher_den"])
        cls = np.asarray([c["class"] for c in R["cells"]], dtype=np.int64)
        P = V[f]
        d = P.argmax(1)
        if not np.array_equal(d, cls[f]):
            raise ValueError("teacher argmax differs from the declared cell class")
        n, S, Aen = KM.cell_stats(P, f, V.shape[0])
        fine = KM.FinePartition(K=int(R["K"]), cell_class=cls, centroid=KM.smooth(V, cls), mean=S / n[:, None],
                                n=n.astype(np.int64), S=S, A=Aen, fallback=np.zeros(V.shape[0], dtype=bool),
                                receipt={"fixture": fam["id"], "recipient": r})
        cell = KM._check_deployment(fine, P, d)
        if not np.array_equal(cell, f):
            raise AssertionError("deployment does not route fixture rows to their declared cells")
        out["fine"][r], out["P"][r], out["d"][r] = fine, P, d
    out["T"] = {"row_id": np.arange(out["N"]), "p1": out["P"][1], "p2": out["P"][2], "d1": out["d"][1],
                "d2": out["d"][2]}
    out["tr"] = np.arange(out["N"])
    out["fine_dict"] = {"fine1": out["fine"][1].to_dict(), "fine2": out["fine"][2].to_dict()}
    out["caps"] = tuple(int(x) for x in fam["caps"])
    return out


# ----------------------------------------------------------------------------------------------- independent metrics
def plugin_mi(s, *toks):
    """Plug-in MI(S; joint key of toks) in nats from exact counts (own implementation: sum n/N log(n N / n_s n_t))."""
    s = np.asarray(s, dtype=np.int64)
    key = np.zeros(s.shape[0], dtype=np.int64)
    for t in toks:
        t = np.asarray(t)
        if t.ndim == 2:
            t = np.unique(t, axis=0, return_inverse=True)[1].ravel()
        t = np.unique(t, return_inverse=True)[1].ravel().astype(np.int64)
        key = key * (int(t.max()) + 1) + t
    key = np.unique(key, return_inverse=True)[1].ravel()
    N = s.shape[0]
    C = int(key.max()) + 1
    tab = np.bincount(s * C + key, minlength=2 * C).reshape(2, C).astype(np.float64)
    ns, nt = tab.sum(1), tab.sum(0)
    tot = 0.0
    for a in range(2):
        for c in range(C):
            x = tab[a, c]
            if x > 0:
                tot += x / N * np.log(x * N / (ns[a] * nt[c]))
    return float(tot)


def release_metrics(rel, X, pols=None):
    """Independent row-level measurement of one release on the fixture rows."""
    from dpc.utility import per_row
    m = {}
    for i in (1, 2):
        q, tok, hard = np.asarray(rel[f"q{i}"]), np.asarray(rel[f"tok{i}"]), np.asarray(rel[f"hard{i}"])
        pr = per_row(q, X["y"][i], q.shape[1])
        m[f"L{i}"], m[f"B{i}"] = float(pr["ll"].mean()), float(pr["br"].mean())
        m[f"I{i}"] = plugin_mi(X["s"], tok)
        m[f"I{i}_tok_q"] = plugin_mi(X["s"], np.column_stack([tok, q]))
        rows = np.arange(q.shape[0])
        other = q.copy()
        other[rows, hard] = -np.inf
        m[f"decisions_ok{i}"] = bool(np.array_equal(hard, X["d"][i]))
        m[f"q_ok{i}"] = bool(np.all(q[rows, hard] > other.max(1)) and np.max(np.abs(q.sum(1) - 1)) <= 1e-12)
        if pols is not None:
            pol = pols[i - 1]
            m[f"tokens_per_class{i}"] = [int(x) for x in pol.tokens_per_class()]
        m[f"acc{i}"] = float(np.mean(hard == X["y"][i]))
    m["I12"] = plugin_mi(X["s"], rel["tok1"], rel["tok2"])
    m["T"] = m["L1"] + m["L2"] + 0.5 * (m["B1"] + m["B2"])
    m["Phi"] = m["I12"] + 0.5 * (m["I1"] + m["I2"])
    m["keys_ok"] = list(rel) == ["row_id", "tok1", "q1", "hard1", "alpha1", "tok2", "q2", "hard2", "alpha2"]
    return m


def u_losses(X):
    from dpc.utility import per_row
    out = {}
    for i in (1, 2):
        pr = per_row(X["P"][i], X["y"][i], X["P"][i].shape[1])
        out[i] = (float(pr["ll"].mean()), float(pr["br"].mean()))
    return out


def feasible(m, U, caps, budget=BUDGET):
    ok = {}
    for i in (1, 2):
        ok[f"L{i}"] = m[f"L{i}"] <= U[i][0] + budget["ll"] + TOL_BUDGET
        ok[f"B{i}"] = m[f"B{i}"] <= U[i][1] + budget["brier"] + TOL_BUDGET
        ok[f"dec{i}"] = m[f"decisions_ok{i}"] and m[f"q_ok{i}"]
        ok[f"cap{i}"] = all(t <= caps[i - 1] for t in m.get(f"tokens_per_class{i}", [0]))
    return all(ok.values()), ok


def local_ok(m, mct):
    return all(m[f"I{i}"] <= mct[f"I{i}"] + TOL_MI for i in (1, 2))


# ----------------------------------------------------------------------------------------------- exhaustive oracle
def _rgs(n, m):
    """Restricted growth strings of length n with at most m blocks (canonical set partitions)."""
    out = []

    def rec(prefix, mx):
        if len(prefix) == n:
            out.append(tuple(prefix))
            return
        for b in range(min(mx + 2, m)):
            rec(prefix + [b], max(mx, b))
    rec([], -1)
    return out


def enumerate_partitions(fine, cap):
    """Every canonical same-class partition of the fine cells with at most `cap` tokens per class, as per-cell labels
    (label = lowest member fine index). Independent of the mapper."""
    per_class = []
    for c in range(fine.K):
        idx = list(fine.cells_of(c))
        per_class.append((idx, _rgs(len(idx), cap)))
    out = []

    def rec(c, lab):
        if c == fine.K:
            out.append(lab.copy())
            return
        idx, strings = per_class[c]
        for g in strings:
            first = {}
            for f, b in zip(idx, g):
                first.setdefault(b, f)
                lab[f] = first[b]
            rec(c + 1, lab)
    rec(0, np.zeros(fine.F, dtype=np.int64))
    return out


def recipient_oracle(X, r, cap):
    """Per-partition exact terms of recipient r (row-level per subset; D1 and D0 decoders; teacher KL; local MI)."""
    from dpc.partition import kl_rows
    from dpc.utility import per_row
    from lra import decoder as DC
    from qpc import kmeans as KM
    fine = X["fine"][r]
    f = X["f1"] if r == 1 else X["f2"]
    y = X["y"][r]
    P = X["P"][r]
    parts = enumerate_partitions(fine, cap)
    sub_cache = {}

    def subset(cells):
        key = tuple(cells)
        if key in sub_cache:
            return sub_cache[key]
        rows = np.flatnonzero(np.isin(f, cells))
        c = int(fine.cell_class[cells[0]])
        n = int(sum(int(fine.n[x]) for x in cells))
        S = np.zeros(fine.K)
        for x in cells:
            S = S + fine.S[x]
        Yc = np.bincount(y[rows], minlength=fine.K).astype(np.float64)
        q0 = KM.smooth(S / n, c)
        sol = DC.solve_token(Yc, S, n, c)
        q1 = np.asarray(sol.q)
        p0, p1 = per_row(np.repeat(q0[None], rows.size, 0), y[rows]), per_row(np.repeat(q1[None], rows.size, 0), y[rows])
        Dkl = float(kl_rows(P[rows], np.repeat(q0[None], rows.size, 0)).sum())
        tl, tb = DC.token_losses(Yc[None], q1[None])
        v = {"n": n, "class": c, "q0": q0, "q1": q1, "ll0": float(p0["ll"].sum()), "br0": float(p0["br"].sum()),
             "ll1": float(p1["ll"].sum()), "br1": float(p1["br"].sum()), "D": Dkl,
             "token_loss_helper_diff": max(abs(float(tl[0]) - float(p1["ll"].sum())),
                                           abs(float(tb[0]) - float(p1["br"].sum()))),
             "law_ll0": float(p0["ll"].mean()), "law_ll1": float(p1["ll"].mean()),
             "law_br0": float(p0["br"].mean()), "law_br1": float(p1["br"].mean()),
             "q_diff": float(np.max(np.abs(q1 - q0))), "cert": sol.cert}
        sub_cache[key] = v
        return v

    N = X["N"]
    rowsT = []
    for lab in parts:
        blocks = {}
        for x in range(fine.F):
            blocks.setdefault(int(lab[x]), []).append(x)
        L0 = B0 = L1 = B1 = Dt = 0.0
        for cells in blocks.values():
            v = subset(cells)
            L0 += v["ll0"]
            B0 += v["br0"]
            L1 += v["ll1"]
            B1 += v["br1"]
            Dt += v["D"]
        tokmap = np.unique(lab, return_inverse=True)[1].ravel()
        I = plugin_mi(X["s"], tokmap[f])
        rowsT.append({"labels": [int(v) for v in lab], "L_D1": L1 / N, "B_D1": B1 / N, "L_D0": L0 / N,
                      "B_D0": B0 / N, "D": Dt / N, "I": I,
                      "tokens_per_class": [len({int(lab[x]) for x in fine.cells_of(c)}) for c in range(fine.K)]})
    return parts, rowsT, sub_cache


def pair_mi_matrix(X, parts1, parts2):
    """I12 for every (partition of 1, partition of 2) from the exact fine table (own aggregation)."""
    F1, F2 = X["fine"][1].F, X["fine"][2].F
    Nf = np.zeros((2, F1, F2))
    np.add.at(Nf, (X["s"], X["f1"], X["f2"]), 1.0)
    N = float(X["N"])
    ns = Nf.sum((1, 2))
    out = np.zeros((len(parts1), len(parts2)))
    m2 = [np.unique(l, return_inverse=True)[1].ravel() for l in parts2]
    for a, l1 in enumerate(parts1):
        t1 = np.unique(l1, return_inverse=True)[1].ravel()
        A = np.zeros((2, int(t1.max()) + 1, F2))
        for x in range(F1):
            A[:, t1[x], :] += Nf[:, x, :]
        for b, t2 in enumerate(m2):
            Bt = np.zeros((2, A.shape[1], int(t2.max()) + 1))
            for z in range(F2):
                Bt[:, :, t2[z]] += A[:, :, z]
            nt = Bt.sum(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                term = np.where(Bt > 0, Bt / N * np.log(Bt * N / (ns[:, None, None] * nt[None])), 0.0)
            out[a, b] = float(term.sum())
    return out


# ----------------------------------------------------------------------------------------------- arms
def _relabel(pair, cid):
    """Use the lra run-style configuration ID inside a fixture (the fixture caps are recorded separately)."""
    pair.config["qpc_config"] = pair.config.get("config")
    pair.config["config"] = cid
    return pair


def d0_bank(X, fid):
    """D0 maps of one fixture: qpc.compress FINE-TASK, CLASS and the 24 old-objective privacy maps (lambda grid; JOINT
    with the same-lambda FINE-TASK / LOCAL / SEQ witnesses, as cbp), qpc.kmeans DIRECT-TASK at the caps."""
    from lra import run as R
    from qpc import compress as CP
    from qpc import kmeans as KM
    from qpc import release as RL
    f1, f2 = X["fine"][1], X["fine"][2]
    m1, m2 = X["caps"]
    P1, P2, d1, d2, s = X["P"][1], X["P"][2], X["d"][1], X["d"][2], X["s"]
    meta = fixture_meta(fid)
    bank, recs = {}, {}

    def fit(fam, lam=None, wit=None):
        pp, rec = CP.fit_policy_pair(fam, f1, f2, P1, d1, P2, d2, s, m1, m2, lam, witnesses=wit,
                                     meta={**meta, "config": R.d0_id(fam, lam)})
        return _relabel(pp, R.d0_id(fam, lam)), rec

    bank[R.d0_id("FINE-TASK")], recs[R.d0_id("FINE-TASK")] = fit("FINE-TASK")
    bank[R.d0_id("CLASS")], recs[R.d0_id("CLASS")] = fit("CLASS")
    for lam in LAMS:
        for fam in ("LOCAL", "SEQ-12", "SEQ-21"):
            bank[R.d0_id(fam, lam)], recs[R.d0_id(fam, lam)] = fit(fam, lam)
        wit = {"FINE-TASK": bank[R.d0_id("FINE-TASK")], "LOCAL": bank[R.d0_id("LOCAL", lam)],
               "SEQ-12": bank[R.d0_id("SEQ-12", lam)], "SEQ-21": bank[R.d0_id("SEQ-21", lam)]}
        wit = {k: RL.PolicyPair.from_dict(_restore_qpc_config(v.to_dict())) for k, v in wit.items()}
        bank[R.d0_id("JOINT", lam)], recs[R.d0_id("JOINT", lam)] = fit("JOINT", lam, wit)
    pols = []
    for r, fine, P, d, m in ((1, f1, P1, d1, m1), (2, f2, P2, d2, m2)):
        fit_r = KM.fit_recipient(P, d, fine.K, m)
        direct = fit_r.partition
        V = fine.mean
        lab = KM.assign_fine(V, fine.cell_class, direct)
        pol = RL.make_policy(r, fine, lab, "DIRECT-TASK")
        tok, _, _ = RL.encode(pol, P, d)
        dcell = KM.assign_fine(P, d, direct)
        pairs_ = np.unique(np.column_stack([tok, dcell]), axis=0)
        if not (np.unique(pairs_[:, 0]).size == pairs_.shape[0] == np.unique(pairs_[:, 1]).size):
            raise AssertionError("DIRECT-TASK is not a coarsening of the fixture fine cells")
        pols.append(pol)
    cid = R.d0_id("DIRECT-TASK")
    bank[cid] = RL.make_pair(pols[0], pols[1], "DIRECT-TASK", m1, m2, None, {**meta, "config": cid})
    recs[cid] = {"family": "DIRECT-TASK", "note": "qpc.kmeans.fit_recipient (3 registered starts) at the caps"}
    return bank, recs


def _restore_qpc_config(z):
    """qpc's JOINT witness check compares family/caps/lam only; restore its own config string for safety."""
    z = json.loads(json.dumps(z))
    if "qpc_config" in z.get("config", {}):
        z["config"]["config"] = z["config"]["qpc_config"]
        for p in ("p1", "p2"):
            if "config" in z[p].get("meta", {}):
                pass
    return z


def d1_of(pair, X, cid):
    from lra import decoder as DC
    from qpc import release as RL
    decs = []
    for i, pol in ((1, pair.p1), (2, pair.p2)):
        tok, _, _ = RL.encode(pol, X["P"][i], X["d"][i])
        decs.append(DC.decode_policy(pol, tok, X["P"][i], X["y"][i], config=cid))
    rel = DC.release_arrays_d1(pair, decs[0], decs[1], X["T"]["row_id"], X["T"]["p1"], X["T"]["d1"], X["T"]["p2"],
                               X["T"]["d2"])
    body = DC.decoder_pair_dict(cid, pair, decs[0], decs[1])
    return rel, decs, body


def run_mapper_arms(X, fid, bank):
    """C-TASK, the 24 weighted controls and the 5 constrained arms through lra.mapper.fit_unit (fixture mode)."""
    from lra import mapper as MP
    from lra import run as R
    tr, T = X["tr"], X["T"]
    Y = {1: X["y"][1], 2: X["y"][2]}
    refs0 = {"caps": list(X["caps"]), "budget": dict(BUDGET)}
    src = {cid: pp.to_dict() for cid, pp in bank.items()}
    out = {}

    def call(arm, lam=None, starts=None, witnesses=None, refs=None):
        cid = MP.config_id(arm, lam)
        rec, files = MP.fit_unit(arm, X["fine_dict"], T, tr, Y, X["s"], lam=lam, starts=starts, witnesses=witnesses,
                                 refs={**refs0, **(refs or {})}, meta=fixture_meta(fid, cid))
        out[cid] = (rec, files)
        return files["policy.json"]

    def pick(keys, have):
        return {k: have[k] for k in keys if k in have}

    have = dict(src)
    have[R.ctask_id()] = call("C-TASK", starts=pick(MP.registered_starts("C-TASK"), have))
    for lam in LAMS:
        for fam in ("LOCAL", "SEQ-12", "SEQ-21"):
            arm = f"W-{fam}"
            have[R.weighted_id(fam, lam)] = call(arm, lam, starts=pick(MP.registered_starts(arm, lam), have))
        have[R.weighted_id("JOINT", lam)] = call("W-JOINT", lam,
                                                 witnesses=pick(MP.registered_starts("W-JOINT", lam), have))
    kref = MP.refs_from_ctask(out[R.ctask_id()][0])
    for a in ("LOCAL", "SEQ-12", "SEQ-21"):
        arm = f"K-{a}"
        have[R.constrained_id(a)] = call(arm, starts=pick(MP.registered_starts(arm), have), refs=kref)
    for a in ("JOINT-SINGLE", "JOINT-PAIR"):
        arm = f"K-{a}"
        have[R.constrained_id(a)] = call(arm, witnesses=pick(MP.registered_starts(arm), have), refs=kref)
    return out


# ----------------------------------------------------------------------------------------------- references
def references(X, o1, o2, I12, U, ct_local):
    """Exhaustive optima of every registered own problem (fixture caps; D1 or D0 as registered)."""
    N1, N2 = len(o1), len(o2)
    g = lambda key, o: np.array([r[key] for r in o])  # noqa: E731
    L1, B1, L2, B2 = g("L_D1", o1), g("B_D1", o1), g("L_D1", o2), g("B_D1", o2)
    D1_, D2_ = g("D", o1), g("D", o2)
    I1, I2 = g("I", o1), g("I", o2)
    fe1 = (L1 <= U[1][0] + BUDGET["ll"] + TOL_BUDGET) & (B1 <= U[1][1] + BUDGET["brier"] + TOL_BUDGET)
    fe2 = (L2 <= U[2][0] + BUDGET["ll"] + TOL_BUDGET) & (B2 <= U[2][1] + BUDGET["brier"] + TOL_BUDGET)
    lc1, lc2 = I1 <= ct_local[0] + TOL_MI, I2 <= ct_local[1] + TOL_MI
    T = (L1 + 0.5 * B1)[:, None] + (L2 + 0.5 * B2)[None, :]
    Phi = I12 + 0.5 * (I1[:, None] + I2[None, :])
    D = D1_[:, None] + D2_[None, :]
    Isum = (I1[:, None] + I2[None, :]) / 2
    feas = fe1[:, None] & fe2[None, :]
    feasK = feas & (lc1[:, None] & lc2[None, :])

    def best(V, mask=None):
        W = np.where(mask, V, np.inf) if mask is not None else V
        v = float(W.min())
        if not np.isfinite(v):
            return {"value": None, "argmin": None, "n_optimal": 0}
        idx = np.argwhere(W <= v + TOL_OPT)
        a, b = (int(x) for x in idx[0])
        return {"value": v, "argmin": [a, b], "n_optimal": int(idx.shape[0]),
                "labels1": o1[a]["labels"], "labels2": o2[b]["labels"], "I12": float(I12[a, b]),
                "T": float(T[a, b]), "Phi": float(Phi[a, b])}

    ref = {"mapping_pairs": N1 * N2, "feasible_pairs": int(feas.sum()), "feasible_local_pairs": int(feasK.sum()),
           "C-TASK": best(T), "min_I12_feasible": best(I12, feas), "min_I12_feasible_local": best(I12, feasK),
           "K-SEQ/JOINT": best(Phi, feasK), "D0 FINE-TASK": best(D)}
    kl = {}
    for r, I, fe, lc in ((1, I1, fe1, lc1), (2, I2, fe2, lc2)):
        m = fe & lc
        kl[str(r)] = {"value": float(I[m].min()) if m.any() else None,
                      "n_optimal": int(np.sum(I[m] <= I[m].min() + TOL_OPT)) if m.any() else 0}
    ref["K-LOCAL"] = kl
    for lam in LAMS:
        ref[f"W-LOCAL|l{lam:g}"] = best(T + lam * Isum)
        ref[f"W-SEQ/JOINT|l{lam:g}"] = best(T + lam * Phi)
        ref[f"D0 LOCAL|l{lam:g}"] = best(D + lam * Isum)
        ref[f"D0 SEQ/JOINT|l{lam:g}"] = best(D + lam * (Isum + I12))
    arrays = {"T": T, "Phi": Phi, "D": D, "Isum": Isum, "I12": I12, "feas": feas, "feasK": feasK}
    return ref, arrays


def _partition_index(parts, pol):
    from qpc import compress as QC
    lab = QC.labels_from_policy(pol)
    for k, p in enumerate(parts):
        if np.array_equal(p, lab):
            return k
    raise AssertionError("release partition is not in the exhaustive enumeration (cap or class violation)")


# ----------------------------------------------------------------------------------------------- one fixture
TERM_KEYS = ("L1", "L2", "B1", "B2", "I1", "I2", "I12")


def run_fixture(fam, mapper=True):
    """Every arm on one family + the exhaustive oracle + the mandatory checks + the trigger. Returns a JSON-safe
    result and the oracle tables."""
    import time
    from lra import decoder as DC
    from lra import run as R
    from qpc import release as RL
    t0, c0 = time.perf_counter(), time.process_time()
    fid = fam["id"]
    X = fixture_rows(fam)
    U = u_losses(X)
    props = fam["properties"]
    gains = [props["r1"]["accuracy_gain"], props["r2"]["accuracy_gain"]]
    arms, fails = {}, []

    def add(cid, kind, rel, pair, extra=None):
        m = release_metrics(rel, X, (pair.p1, pair.p2))
        ok, parts = feasible(m, U, X["caps"])
        arms[cid] = {"cid": cid, "kind": kind, "metrics": m, "feasible": bool(ok), "feasible_parts": parts,
                     "pair": pair, "rel": rel, **(extra or {})}
        return m

    # D0 bank and D1 fixed-map controls
    bank, d0recs = d0_bank(X, fid)
    for cid, pair in bank.items():
        rel0 = RL.release_arrays(pair, X["T"]["row_id"], X["T"]["p1"], X["T"]["d1"], X["T"]["p2"], X["T"]["d2"])
        add(cid, "d0", rel0, pair)
        c1 = cid + "|D1"
        rel1, decs, body = d1_of(pair, X, c1)
        add(c1, "d1_fixed" if R.parse_id(c1)["privacy_trained"] else "d1_task", rel1, pair,
            {"decoder_body": body, "d0_of": cid})
    # mapper arms
    mrecs = {}
    if mapper:
        for cid, (rec, files) in run_mapper_arms(X, fid, bank).items():
            pair = RL.PolicyPair.from_dict(files["policy.json"])
            p = R.parse_id(cid)
            add(cid, p["arm"], files["release.npz"], pair, {"decoder_body": files["decoder.json"], "record": rec})
            mrecs[cid] = rec
    # exhaustive oracle
    parts1, o1, sub1 = recipient_oracle(X, 1, X["caps"][0])
    parts2, o2, sub2 = recipient_oracle(X, 2, X["caps"][1])
    I12 = pair_mi_matrix(X, parts1, parts2)
    ct = arms.get(R.ctask_id())
    ct_local = (ct["metrics"]["I1"], ct["metrics"]["I2"]) if ct else (np.inf, np.inf)
    ref, arr = references(X, o1, o2, I12, U, ct_local)
    checks = {}
    # C1 fixed-token information
    c1 = []
    for cid, a in arms.items():
        if a["kind"] in ("d1_fixed", "d1_task"):
            b = arms[a["d0_of"]]
            same = all(np.array_equal(a["rel"][f"tok{i}"], b["rel"][f"tok{i}"]) for i in (1, 2))
            mi_same = all(a["metrics"][k] == b["metrics"][k] for k in ("I1", "I2", "I12"))
            tq = all(abs(x["metrics"][f"I{i}_tok_q"] - x["metrics"][f"I{i}"]) <= 1e-15 for x in (a, b) for i in (1, 2))
            if not (same and mi_same and tq):
                c1.append(cid)
    for cid, a in arms.items():          # every D1 release (mapper too): (token, q) carries exactly the token's MI
        if any(abs(a["metrics"][f"I{i}_tok_q"] - a["metrics"][f"I{i}"]) > 1e-15 for i in (1, 2)):
            c1.append(cid + ":tok_q")
    checks["C1_FIXED_TOKEN_INFORMATION"] = {"pass": not c1, "failures": c1,
                                            "pairs_checked": sum(1 for a in arms.values() if "d0_of" in a)}
    # C2 calibrated null (F1 only)
    if fid == "F1_CALIBRATED_NULL":
        bad, qmax, lgap = [], 0.0, 0.0
        for sub in (sub1, sub2):
            for key, v in sub.items():
                qmax = max(qmax, v["q_diff"])
                g = min(v["law_ll1"] - v["law_ll0"], v["law_br1"] - v["law_br0"])
                lgap = min(lgap, g)
                if v["q_diff"] > TOL_NULL_Q or g < -TOL_NULL_LOSS:
                    bad.append(str(key))
        for cid, a in arms.items():
            if R.parse_id(cid).get("decoder") == "D1":
                q0rel = RL.release_arrays(a["pair"], X["T"]["row_id"], X["T"]["p1"], X["T"]["d1"], X["T"]["p2"],
                                          X["T"]["d2"])
                m0 = release_metrics(q0rel, X)
                for i in (1, 2):
                    dq = float(np.max(np.abs(np.asarray(a["rel"][f"q{i}"]) - np.asarray(q0rel[f"q{i}"]))))
                    qmax = max(qmax, dq)
                    if dq > TOL_NULL_Q:
                        bad.append(f"{cid}:q{i}")
                    gl = a["metrics"][f"L{i}"] - m0[f"L{i}"]
                    gb = a["metrics"][f"B{i}"] - m0[f"B{i}"]
                    lgap = min(lgap, gl, gb)
                    if gl < -TOL_NULL_LOSS or gb < -TOL_NULL_LOSS:
                        bad.append(f"{cid}:r{i}")
        checks["C2_CALIBRATED_NULL"] = {"pass": not bad, "failures": bad, "max_q_diff": qmax,
                                        "min_D1_minus_D0_loss_per_row": lgap,
                                        "subsets_checked": len(sub1) + len(sub2)}
    # C3 budget enforcement
    c3 = []
    for cid, rec in mrecs.items():
        a = arms[cid]
        for i in (1, 2):
            if abs(rec["budgets"]["L_U"][str(i)] - U[i][0]) > 1e-12 or abs(rec["budgets"]["B_U"][str(i)] - U[i][1]) > 1e-12:
                c3.append(f"{cid}:U{i}")
        if rec["status"] == "FEASIBLE":
            if R.parse_id(cid)["arm"] == "constrained":
                if not (a["feasible"] and local_ok(a["metrics"], ct["metrics"])):
                    c3.append(f"{cid}:claimed_feasible")
            elif not all(a["feasible_parts"][f"{k}{i}"] for k in ("dec", "cap") for i in (1, 2)):
                c3.append(f"{cid}:decision_or_cap")
        elif rec["status"] == "INFEASIBLE" and R.parse_id(cid)["arm"] != "constrained":
            c3.append(f"{cid}:infeasible_status_on_unconstrained_arm")
    checks["C3_BUDGET_ENFORCEMENT"] = {"pass": not c3, "failures": c3, "mapper_units": len(mrecs)}
    # C4 decision preservation
    c4 = [cid for cid, a in arms.items() if not (a["metrics"]["keys_ok"] and all(
        a["metrics"][f"decisions_ok{i}"] and a["metrics"][f"q_ok{i}"] for i in (1, 2)))]
    checks["C4_DECISION_PRESERVATION"] = {"pass": not c4, "failures": c4, "releases": len(arms)}
    # C5 term reconstruction
    c5, worst = [], 0.0
    for cid, rec in mrecs.items():
        a = arms[cid]["metrics"]
        for blk in ("final_state_terms", "deployed"):
            t = rec[blk] if blk == "final_state_terms" else rec[blk]["terms"]
            for k in TERM_KEYS:
                if k in t:
                    dlt = abs(float(t[k]) - a[k])
                    worst = max(worst, dlt)
                    if dlt > TOL_TERMS:
                        c5.append(f"{cid}:{blk}:{k}")
            for k, mine in (("T", a["T"]), ("Phi", a["Phi"])):
                if k in t and abs(float(t[k]) - mine) > TOL_TERMS:
                    c5.append(f"{cid}:{blk}:{k}")
    no_state = []
    for cid, rec in mrecs.items():           # incremental search state of the winner vs its from-scratch rebuild
        inc = _incremental_terms(rec)
        if inc is None:
            c5.append(f"{cid}:incremental_terms_missing")
            continue
        if inc == NO_SEARCH_STATE:
            no_state.append(cid)
            continue
        for k, v in inc.items():
            if k in rec["final_state_terms"]:
                dlt = abs(float(v) - float(rec["final_state_terms"][k]))
                worst = max(worst, dlt)
                if dlt > TOL_TERMS:
                    c5.append(f"{cid}:incremental:{k}")
    for cid, a in arms.items():
        if "decoder_body" in a:
            try:
                DC.load_decoder_pair(json.loads(json.dumps(a["decoder_body"])), a["pair"], verify_solve=True)
            except ValueError as e:
                c5.append(f"{cid}:decoder_json:{e}")
        k1, k2 = _partition_index(parts1, a["pair"].p1), _partition_index(parts2, a["pair"].p2)
        a["oracle_index"] = [k1, k2]
        dec_key = "D1" if R.parse_id(cid).get("decoder") == "D1" else "D0"
        tab = {"L1": o1[k1][f"L_{dec_key}"], "B1": o1[k1][f"B_{dec_key}"], "L2": o2[k2][f"L_{dec_key}"],
               "B2": o2[k2][f"B_{dec_key}"], "I1": o1[k1]["I"], "I2": o2[k2]["I"], "I12": float(I12[k1, k2])}
        for k, v in tab.items():
            dlt = abs(v - a["metrics"][k])
            worst = max(worst, dlt)
            if dlt > TOL_TERMS:
                c5.append(f"{cid}:oracle:{k}")
    helper = max(v["token_loss_helper_diff"] for sub in (sub1, sub2) for v in sub.values())
    if helper > 1e-9:
        c5.append(f"token_loss_helper:{helper}")
    checks["C5_TERM_RECONSTRUCTION"] = {"pass": not c5, "failures": c5, "max_abs_diff": worst,
                                        "incremental_not_applicable_no_search_state": no_state,
                                        "token_loss_helper_max_abs_diff_totals": helper}
    # C7 law integrity (the laws were verified by load_laws before this call; rows deploy to declared cells)
    checks["C7_LAW_INTEGRITY"] = {"pass": bool(X["N"] == N_FIX and props["mapping_pairs"] <= MAX_PAIRS and
                                               law_properties(fam) == props),
                                  "rows": X["N"], "mapping_pairs": props["mapping_pairs"]}
    # C6 heuristic labelling
    labels = label_arms(arms, ref, arr, o1, o2, ct)
    checks["C6_HEURISTIC_LABELLING"] = {"pass": all(v["label"] for v in labels.values()), "labels": labels}
    # trigger
    const_acc = [props["r1"]["constant_acc"], props["r2"]["constant_acc"]]
    trig = trigger(arms, ct, gains, const_acc)
    res = {"fixture": fid, "U": {str(i): {"L": U[i][0], "B": U[i][1]} for i in (1, 2)}, "accuracy_gain": gains,
           "caps": list(X["caps"]), "checks": checks, "trigger": trig, "references": ref,
           "arms": {cid: _arm_row(a, ct) for cid, a in arms.items()},
           "d0_bank_records": {cid: {"final": r.get("final"), "summary": r.get("summary")} for cid, r in d0recs.items()},
           "mapper_status": {cid: {"status": r["status"], "winner": r["winner"]} for cid, r in mrecs.items()},
           "wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0}
    tables = {"r1": o1, "r2": o2, "I12": I12, "parts1": parts1, "parts2": parts2}
    return res, tables, arms


NO_SEARCH_STATE = "NO_SEARCH_STATE"


def _incremental_terms(rec):
    """Terms of the winning start's incrementally updated search state (mapper start records); NO_SEARCH_STATE when
    the release is the unchanged descriptive copy of the first start of an INFEASIBLE constrained unit in which no
    start was ever refined (every start infeasible: no incremental state exists); otherwise None (a failure)."""
    name, kind = rec["winner"]["start"], rec["winner"]["kind"].replace("_descriptive", "")
    for s in rec["starts"]:
        nm = s.get("name") or (s.get("start") or {}).get("name")
        if nm != name:
            continue
        blk = s.get("final") if kind == "refined" else (s.get("unchanged") or s.get("start"))
        if blk:
            return blk.get("terms")
        break
    if (rec["status"] == "INFEASIBLE" and rec["winner"]["kind"] == "unchanged_descriptive"
            and not any("final" in s for s in rec["starts"])):
        return NO_SEARCH_STATE
    return None


def _arm_row(a, ct):
    m = a["metrics"]
    row = {"kind": a["kind"], "feasible": a["feasible"],
           "local_ok": bool(local_ok(m, ct["metrics"])) if ct else None,
           **{k: m[k] for k in TERM_KEYS + ("T", "Phi")},
           "tokens_per_class": [m["tokens_per_class1"], m["tokens_per_class2"]],
           "feasible_parts": a["feasible_parts"], "oracle_index": a.get("oracle_index"),
           "pair_fingerprint": a["pair"].fingerprint()}
    if "record" in a:
        row["mapper_status"] = a["record"]["status"]
    return row


def _own_value(cid, m):
    """(registered own objective value, reference key) of an arm, from the independent metrics."""
    from lra import run as R
    p = R.parse_id(cid)
    lam = p.get("lam")
    Isum = (m["I1"] + m["I2"]) / 2
    if p["arm"] == "d0":
        if p["base_family"] in ("DIRECT-TASK", "CLASS"):
            return None, None
        D = m.get("D")
        if p["base_family"] == "FINE-TASK":
            return D, "D0 FINE-TASK"
        if p["base_family"] == "LOCAL":
            return D + lam * Isum, f"D0 LOCAL|l{lam:g}"
        return D + lam * (Isum + m["I12"]), f"D0 SEQ/JOINT|l{lam:g}"
    if p["arm"] == "ctask":
        return m["T"], "C-TASK"
    if p["arm"] == "weighted":
        if p["base_family"] == "LOCAL":
            return m["T"] + lam * Isum, f"W-LOCAL|l{lam:g}"
        return m["T"] + lam * m["Phi"], f"W-SEQ/JOINT|l{lam:g}"
    if p["arm"] == "constrained":
        if p["base_family"] == "LOCAL":
            return None, "K-LOCAL"
        return m["Phi"], "K-SEQ/JOINT"
    return None, None


def label_arms(arms, ref, arr, o1, o2, ct):
    out = {}
    for cid, a in arms.items():
        m = dict(a["metrics"])
        if "oracle_index" in a:
            k1, k2 = a["oracle_index"]
            m["D"] = o1[k1]["D"] + o2[k2]["D"]
        if a["kind"] in ("d1_fixed", "d1_task"):
            out[cid] = {"label": "NOT_A_SEARCH", "note": "fixed D0 map decoded by D1 (assignments unchanged)"}
            continue
        val, key = _own_value(cid, m)
        if key is None:
            out[cid] = {"label": "NOT_A_SEARCH"}
            continue
        if key == "K-LOCAL":
            gaps = {}
            for i in (1, 2):
                rv = ref["K-LOCAL"][str(i)]["value"]
                gaps[str(i)] = None if rv is None else m[f"I{i}"] - rv
            feas = a["feasible"] and local_ok(m, ct["metrics"])
            opt = feas and all(g is not None and g <= TOL_OPT for g in gaps.values())
            out[cid] = {"label": "EXHAUSTIVE_OPTIMAL" if opt else "HEURISTIC", "gap_per_recipient": gaps,
                        "feasible": bool(feas)}
            continue
        rv = ref[key]["value"]
        cons = key == "K-SEQ/JOINT"
        feas = (a["feasible"] and local_ok(m, ct["metrics"])) if cons else True
        gap = None if rv is None else val - rv
        opt = feas and gap is not None and gap <= TOL_OPT
        out[cid] = {"label": "EXHAUSTIVE_OPTIMAL" if opt else "HEURISTIC", "objective": val, "reference": rv,
                    "gap": gap, "feasible_where_constrained": bool(feas), "reference_key": key}
    return out


def descriptive(arms, ct):
    """Registered descriptive flags (FIXTURE_GATE_RULE.json descriptive_definitions); never used by the verdict."""
    from lra import run as R
    groups = {"constrained": [], "weighted": [], "d1_fixed_joint": [], "d1_fixed_seq": [], "weighted_joint": [],
              "weighted_seq": [], "constrained_joint": [], "constrained_seq": []}
    for cid, a in sorted(arms.items()):
        p = R.parse_id(cid)
        if not (p.get("privacy_trained") and p.get("decoder") == "D1"):
            continue
        ok = bool(a["feasible"] and (ct is None or local_ok(a["metrics"], ct["metrics"])))
        item = (a["metrics"]["I12"], ok)
        fam, base = p["arm"], p["base_family"]
        if fam in ("constrained", "weighted"):
            groups[fam].append(item)
        key = {"d1_fixed": "d1_fixed", "weighted": "weighted", "constrained": "constrained"}.get(fam)
        if key is None:
            continue
        if base in ("JOINT", "JOINT-SINGLE", "JOINT-PAIR"):
            groups[f"{key}_joint"].append(item)
        elif base in ("SEQ-12", "SEQ-21"):
            groups[f"{key}_seq"].append(item)
    best = {g: {"all": min((v for v, _ in it), default=None), "eligible": min((v for v, o in it if o), default=None),
                "n": len(it), "n_eligible": sum(o for _, o in it)} for g, it in groups.items()}

    def diff(a, b, k):
        x, y = best[a][k], best[b][k]
        return None if x is None or y is None else x - y
    return {"best_I12": best,
            "constrained_minus_weighted": {k: diff("constrained", "weighted", k) for k in ("all", "eligible")},
            "joint_minus_sequential": {f: {k: diff(f"{f}_joint", f"{f}_seq", k) for k in ("all", "eligible")}
                                       for f in ("d1_fixed", "weighted", "constrained")},
            "local_budget_applied": ct is not None}


def trigger(arms, ct, gains, const_acc):
    """The registered trigger on one fixture (FIXTURE_GATE_RULE.json)."""
    out = _trigger(arms, ct, gains, const_acc)
    out["descriptive"] = descriptive(arms, ct)
    return out


def _trigger(arms, ct, gains, const_acc):
    from lra import run as R
    acc_ok = all(g >= MIN_GAIN for g in gains)
    if ct is None:
        return {"accuracy_condition": acc_ok, "T_star": None, "reason": "NO_C_TASK", "triggered": False,
                "qualifying": [], "nontrivial": False, "route": {"assignment_search_helps": False}}
    task = []
    for cid in TASK_ONLY_D1:
        a = arms.get(cid)
        if a is not None and a["feasible"]:
            task.append((a["metrics"]["I12"], a["metrics"]["T"], cid))
    if not task:
        return {"accuracy_condition": acc_ok, "T_star": None, "reason": "NO_FEASIBLE_TASK_ONLY", "triggered": False,
                "qualifying": [], "nontrivial": False, "route": {"assignment_search_helps": False}}
    task.sort()
    i12_t, _, tstar = task[0]
    qual, near = [], []
    for cid, a in sorted(arms.items()):
        p = R.parse_id(cid)
        if not (p.get("privacy_trained") and p.get("decoder") == "D1" and p["arm"] in ("d1_fixed", "weighted",
                                                                                         "constrained")):
            continue
        m = a["metrics"]
        lok = local_ok(m, ct["metrics"])
        red = i12_t - m["I12"]
        g = [m[f"acc{i}"] - const_acc[i - 1] for i in (1, 2)]
        row = {"cid": cid, "arm": p["arm"], "feasible": a["feasible"], "local_ok": bool(lok), "I12": m["I12"],
               "reduction": red, "accuracy_gain": g}
        if a["feasible"] and lok and red >= TRIGGER_MI and min(g) >= MIN_GAIN:
            qual.append(row)
        else:
            near.append(row)
    nontrivial = acc_ok and i12_t >= TRIGGER_MI
    fm = [q for q in qual if q["arm"] == "d1_fixed"]
    new = [q for q in qual if q["arm"] in ("weighted", "constrained")]
    best_fm = min((q["I12"] for q in fm), default=None)
    best_new = min((q["I12"] for q in new), default=None)
    helps = bool(new) and (not fm or best_new <= best_fm - TRIGGER_MI)
    d0_ok = []
    for q in fm:
        b = arms[q["cid"][:-3]]
        if b["feasible"] and local_ok(b["metrics"], ct["metrics"]) and i12_t - b["metrics"]["I12"] >= TRIGGER_MI:
            d0_ok.append(q["cid"][:-3])
    return {"accuracy_condition": acc_ok, "T_star": tstar, "T_star_I12": i12_t,
            "task_only_feasible": [c for _, _, c in task], "nontrivial": bool(nontrivial),
            "triggered": bool(nontrivial and qual), "qualifying": qual, "not_qualifying": near,
            "route": {"decoder_enabled": bool(fm), "assignment_search_helps": helps, "best_fm_I12": best_fm,
                      "best_new_I12": best_new, "d0_version_already_qualifies": d0_ok,
                      "constrained_qualifies": any(q["arm"] == "constrained" for q in qual),
                      "weighted_qualifies": any(q["arm"] == "weighted" for q in qual)}}


def gate_from_results(results):
    """GATE_MET / GATE_NOT_MET and the route classification from the four per-fixture results."""
    bad = sorted({f"{r['fixture']}:{k}" for r in results for k, v in r["checks"].items() if not v["pass"]})
    trig = [r["fixture"] for r in results if r["trigger"]["triggered"]]
    reasons = []
    if bad:
        reasons.append("CORRECTNESS_FAILURE:" + ",".join(bad))
    if not trig:
        reasons.append("NO_FIXTURE_TRIGGERED")
    law_ok = all(r.get("law_integrity", True) for r in results)
    if not law_ok:
        reasons.append("CORRECTNESS_FAILURE:C7_LAW_INTEGRITY")
    met = not reasons and len(results) == 4
    if len(results) != 4:
        reasons.append("INCOMPLETE_FIXTURE_SET")
        met = False
    route = None
    if met:
        helps = any(r["trigger"]["route"]["assignment_search_helps"] for r in results if r["trigger"]["triggered"])
        route = "ASSIGNMENT_SEARCH_ROUTE" if helps else "DECODER_ENABLED_ROUTE"
    return {"verdict": "GATE_MET" if met else "GATE_NOT_MET", "reasons": reasons, "triggered_fixtures": trig,
            "route": route}


# ----------------------------------------------------------------------------------------------- stage
def _write_tables(out_dir, fid, tables, res):
    import csv
    d = Path(out_dir) / "fixture_oracle"
    d.mkdir(parents=True, exist_ok=True)
    for r in ("r1", "r2"):
        with open(d / f"{fid}_partitions_{r}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "labels", "tokens_per_class", "L_D1", "B_D1", "L_D0", "B_D0", "D_teacher_kl", "I"])
            for k, row in enumerate(tables[r]):
                w.writerow([k, " ".join(map(str, row["labels"])), " ".join(map(str, row["tokens_per_class"])),
                            repr(row["L_D1"]), repr(row["B_D1"]), repr(row["L_D0"]), repr(row["B_D0"]),
                            repr(row["D"]), repr(row["I"])])
    I12 = tables["I12"]
    with open(d / f"{fid}_pair_I12.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index1", "index2", "I12"])
        for a in range(I12.shape[0]):
            for b in range(I12.shape[1]):
                w.writerow([a, b, repr(float(I12[a, b]))])
    with open(d / f"{fid}_arms.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["config", "kind", "feasible", "local_ok"] + list(TERM_KEYS) + ["T", "Phi", "oracle_index", "label"]
        w.writerow(cols)
        lab = res["checks"]["C6_HEURISTIC_LABELLING"]["labels"]
        for cid, a in sorted(res["arms"].items()):
            w.writerow([cid, a["kind"], a["feasible"], a["local_ok"]] + [repr(a[k]) for k in TERM_KEYS] +
                       [repr(a["T"]), repr(a["Phi"]), " ".join(map(str, a["oracle_index"] or [])),
                        lab.get(cid, {}).get("label")])
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob(f"{fid}_*"))}
    return hashes


def stage_fixture(D, shard_spec=None, out=None, laws_path=None, save_units=True, families=None):
    """The registered fixture stage (FIXTURE_LOCK): D must be None (synthetic known laws only). Runs every family
    (or the shard's families), writes PKG/FIXTURE_GATE.json (when all four are available) and the oracle tables."""
    import time
    if D is not None:
        raise ValueError("REFUSED: the fixture stage loads no real data (D must be None)")
    t0, c0 = time.perf_counter(), time.process_time()
    out_dir = Path(out) if out else PKG
    laws = load_laws(laws_path)
    if laws_path is None:                  # the registered run: laws and rule must be exactly the locked documents
        from lra import lock as LK
        docs = LK.latest()["documents_sha256"]
        rule = json.loads((PKG / "FIXTURE_GATE_RULE.json").read_text())
        for d in ("FIXTURE_LAWS.json", "FIXTURE_GATE_RULE.json"):
            if docs.get(d) != _file_sha(PKG / d):
                raise ValueError(f"REFUSED: {d} differs from the locked documents_sha256")
        if rule.get("laws_sha256") != laws["laws_sha256"]:
            raise ValueError("REFUSED: the gate rule binds different laws")
    fams = laws["families"]
    if families is not None:
        fams = [f for f in fams if f["id"] in families]
    if shard_spec:
        i, n = map(int, shard_spec.split("/"))
        fams = [f for k, f in enumerate(fams) if k % n == i]
    results, hashes = [], {}
    for fam in fams:
        res, tables, _ = run_fixture(fam)
        res["law_integrity"] = True
        hashes.update(_write_tables(out_dir, fam["id"], tables, res))
        results.append(res)
        if save_units:
            from lra import run as R
            R.save(f"fix__{fam['id']}", {"result.json": res}, {"fixture": fam["id"], "laws_sha256": laws["laws_sha256"],
                                                               "trigger": res["trigger"]["triggered"]})
    if shard_spec and save_units and len(results) < 4:
        from lra import run as R
        have = [json.loads((R.U(f"fix__{f['id']}") / "result.json").read_text())
                for f in laws["families"] if R.done(f"fix__{f['id']}")
                and json.loads((R.U(f"fix__{f['id']}") / "record.json").read_text()).get("laws_sha256")
                == laws["laws_sha256"]]
        if len(have) == 4:
            results = have
    gate = gate_from_results(results) if results else {"verdict": "GATE_NOT_MET", "reasons": ["NO_RESULTS"]}
    from lra import decoder as DC
    body = {"schema": "lra-fixture-gate-v1", "verdict": gate["verdict"], "reasons": gate["reasons"],
            "route": gate.get("route"), "triggered_fixtures": gate.get("triggered_fixtures"),
            "laws_sha256": laws["laws_sha256"], "gate_rule_sha256": _file_sha(PKG / "FIXTURE_GATE_RULE.json"),
            "decoder_tolerances": DC.TOLERANCES,
            "fixtures": [{k: v for k, v in r.items() if k != "references"} | {"references": r["references"]}
                         for r in results],
            "oracle_table_sha256": hashes, "synthetic_only": True,
            "wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0,
            "scope": "known-law fixtures; fitted MI = exact law MI of these finite laws; heuristic optimisers are "
                     "labelled against exhaustive references; no Adult claim follows from this file"}
    from lra.run import _finite
    if len(results) == 4 or not shard_spec:
        (out_dir / "FIXTURE_GATE.json").write_text(json.dumps(_finite(body), indent=1, allow_nan=False) + "\n")
    return body


def _file_sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lra.fixtures")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("laws")
    s1.add_argument("--write", action="store_true")
    s2 = sub.add_parser("run")
    s2.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "laws":
        if a.write:
            print(write_law_files())
        else:
            b = load_laws()
            print(json.dumps({"ok": True, "laws_sha256": b["laws_sha256"],
                              "families": [f["id"] for f in b["families"]]}))
    else:
        r = stage_fixture(None, None, out=a.out)
        print(json.dumps({"verdict": r["verdict"], "reasons": r.get("reasons")}))


if __name__ == "__main__":
    main()
