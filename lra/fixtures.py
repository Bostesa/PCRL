"""[lra port of lcr/fixtures.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md]
Known-law fixtures and the CORRECTNESS-ONLY launch gate of the lra study (role B; prompt sec. 9). SYNTHETIC ONLY: no
real data is loaded.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.fixtures laws          # verify the pinned laws
    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.fixtures rule [--write] # ENGINEERING_GATE_RULE.json
    (the registered stage runs through lra.run --lock CORRECTNESS_LOCK --stage correctness -> stage_correctness(None))

LAWS (PKG/FIXTURE_LAWS.json): the predecessor's four laws, copied UNCHANGED (file sha256 24f70745..., laws_sha256
5c5e3bda...; schema lcr-fixture-laws-v1 kept). They are KNOWN, ALREADY-OPENED regression cases. build_laws() still
reproduces them byte for byte from SPECS (a static construction, tested); nothing here rewrites the file.
  Four fixed families, N = 4096 exact expected counts (atom weights = count / 4096):
    F1_CALIBRATED_NULL     teacher = true conditional per fine cell; labels = exact expectation
    F2_MISCALIBRATED       over/under-confident teacher, K2 = 3, labels favouring a non-predicted class in some cells
    F3_COMPLEMENTARY_XOR   SEX = XOR of one clue bit per recipient (w.p. 3/4): nothing locally, 0.13 nats jointly
    F4_REDUNDANT           the SAME clue bit visible to both recipients: local = pair information
  Each family: per recipient K, fine cells (= the allowed observations: one distinct dyadic teacher vector per cell,
  predicted class = its strict argmax), per-cell label laws, explicit cell-pair counts n(f1, f2), explicit SEX laws
  P(S = 1 | f1, f2), the label-joint rule and the complete atom list (f1, f2, s, y1, y2, count).

ENGINE (unchanged from lcr): run_fixture = the D0 bank (qpc), D1 decodes of every D0 map, the lra.mapper arms
(C-TASK, 24 W-, 5 K-; fixture mode), the exhaustive oracle (every canonical same-class partition pair) and the
predecessor's checks C1-C7.

GATE (new): engineering_gate_rule() -> ENGINEERING_GATE_RULE.json; stage_correctness computes the twelve mandatory
checks E01..E12 of prompt sec. 9 and engineering_verdict() -> ENGINEERING_READY iff all pass, else
ENGINEERING_BLOCKED. The old trigger / T* / route logic (trigger, descriptive, gate_from_results) is DESCRIPTIVE
output only and never decides the verdict. Outputs: PKG/ENGINEERING_GATE_RESULT.json, PKG/correctness_oracle/*.csv and
the private units cor__<FID> (mapper records and traces, policies, decoders, releases, D0 starts, fine partitions) for
the independent replay.
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
LAWS_SCHEMA = "lcr-fixture-laws-v1"           # the PINNED source laws keep their source schema (unchanged copy)
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
            "decoders": "D0 = smoothed mean teacher vector of the token (qpc); D1 = lcr.decoder (kappa = 32, eps = "
                        "1e-12, class-dominant simplex) from the token's fitting label counts and teacher sums",
            "budgets": {"ll": "L_i <= L_i(U) + 0.005 nats (true-label log loss on the fixture rows, clip 1e-12)",
                        "brier": "B_i <= B_i(U) + 0.003 (source multiclass Brier)",
                        "decisions": "released decision == teacher decision on every row",
                        "caps": "at most caps[i-1] tokens per predicted class",
                        "local_information": "constrained (K-) arms and the gate's privacy-trained candidates: I_i <= "
                                             "I_i(C-TASK) of the same fixture"},
            "U": "the continuous teacher release: each row's own teacher vector (its fine cell's vector)",
            "lam_grid": list(LAMS), "families": fams,
            "construction": "lcr/fixtures.py build_laws() (SPECS); stage_fixture re-builds the atoms from the explicit "
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


# gate_rule() (the predecessor's FIXTURE_GATE_RULE.json generator) and write_law_files() were REMOVED in lra: the laws
# are a pinned unchanged copy (never rewritten) and the old performance gate is replaced by engineering_gate_rule().
# The old trigger / route functions below remain as DESCRIPTIVE output only.


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
        miss = [k for k in keys if k not in have]
        if miss:                                               # the correctness engine passes the FULL registered set
            raise ValueError(f"{fid}: registered starts/witnesses missing from the fixture bank: {miss}")
        return {k: have[k] for k in keys}

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
    mrecs, mfiles = {}, {}
    if mapper:
        for cid, (rec, files) in run_mapper_arms(X, fid, bank).items():
            pair = RL.PolicyPair.from_dict(files["policy.json"])
            p = R.parse_id(cid)
            add(cid, p["arm"], files["release.npz"], pair, {"decoder_body": files["decoder.json"], "record": rec})
            mrecs[cid] = rec
            mfiles[cid] = files
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
    tables = {"r1": o1, "r2": o2, "I12": I12, "parts1": parts1, "parts2": parts2,
              # context for the engineering checks (stage_correctness); not written to the oracle CSVs
              "X": X, "U": U, "sub1": sub1, "sub2": sub2, "mrecs": mrecs, "mfiles": mfiles, "bank": bank,
              "d0_records": d0recs, "ref_arrays": arr}
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
    """The predecessor's descriptive flags (lcr FIXTURE_GATE_RULE.json descriptive_definitions); DESCRIPTIVE only."""
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
    """The predecessor's trigger on one fixture (lcr FIXTURE_GATE_RULE.json); DESCRIPTIVE only in lra."""
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


# ----------------------------------------------------------------------------------------------- engineering gate
# The CORRECTNESS-ONLY launch gate of this study (prompt sec. 9). It REPLACES the predecessor's performance gate: the
# old trigger / route logic above (trigger, descriptive, gate_from_results) is kept as DESCRIPTIVE output only and is
# never read by engineering_verdict(). Verdicts: ENGINEERING_READY iff every mandatory check E01..E12 passes.
ENG_RULE_SCHEMA = "lra-engineering-gate-rule-v1"
ENG_RESULT_SCHEMA = "lra-engineering-gate-result-v1"
ENG_RULE_FILE = "ENGINEERING_GATE_RULE.json"
ENG_RESULT_FILE = "ENGINEERING_GATE_RESULT.json"
VERDICTS = ("ENGINEERING_READY", "ENGINEERING_BLOCKED")
FIXTURE_IDS = ("F1_CALIBRATED_NULL", "F2_MISCALIBRATED", "F3_COMPLEMENTARY_XOR", "F4_REDUNDANT")
PINNED_LAWS_FILE_SHA256 = "24f7074519dc9d24afb75a14e3601be61ee0e17929fe150a2fb8ca0db5ce9d9c"
PINNED_LAWS_SHA256 = "5c5e3bda08c971d50513093179207546059fe2259e8707ca866fd0e162942434"
SOURCE_GATE_FILE = "SOURCE_FIXTURE_GATE.json"
PINNED_SOURCE_GATE_SHA256 = "772f17e35a471d7069a44c707890bcdfe76a3996f1ebab071331655ae5cb325e"
SOURCE_RESULTS = "results/pcrl_learned_decoder_constrained_release_v1"
SOURCE_ORACLE_DIR = WT / SOURCE_RESULTS / "fixture_oracle"
SOURCE_ORACLE_SHA256 = {
    "F1_CALIBRATED_NULL_arms.csv": "16d8bd36a60209bcf147fb06b2ef1803bec5398dc9f6be66695107a71a3b9be0",
    "F1_CALIBRATED_NULL_pair_I12.csv": "a16108a182cb0b2ef9c3125c36d17dc018f1e59b426e81d09e5515952a3ccb16",
    "F1_CALIBRATED_NULL_partitions_r1.csv": "28e5a52ece943326bbd90ebee6082f3546326bffe4ef0c84777958d5687bc38b",
    "F1_CALIBRATED_NULL_partitions_r2.csv": "0b2401786ab9f1c05a91b7024773157e7a1b0f23d5d787e5e4df8c2e1af8b0f1",
    "F2_MISCALIBRATED_arms.csv": "ebbd92ce171072c4cd58257b4f2708e3ed930a519de803232b142938da9a1fe2",
    "F2_MISCALIBRATED_pair_I12.csv": "d0bcc732d0b1ee39558b3d926bb267ba4b881079653073875d8bd8f0e7bb538b",
    "F2_MISCALIBRATED_partitions_r1.csv": "40fe97b22026a1408e7cdeb6efb4f5748caf8c2d99ae7a8574295b75cf98b8d4",
    "F2_MISCALIBRATED_partitions_r2.csv": "22c27fab9dc92759b01d4153fa3088a48029db7da59d82b3a263e1c81b0e2f52",
    "F3_COMPLEMENTARY_XOR_arms.csv": "ec52e4cedf140e50924596fba837860f3839567eda07bea204b509eb9c3a463c",
    "F3_COMPLEMENTARY_XOR_pair_I12.csv": "19114dbbdcd297050ce0414e552e719d8dbc1128c76cf392d6bc5335991fa3b0",
    "F3_COMPLEMENTARY_XOR_partitions_r1.csv": "780eddd0f8e2e9d5c97d740b67885b4cae8710c9f7e2076b58158eab963c8862",
    "F3_COMPLEMENTARY_XOR_partitions_r2.csv": "2deaf44c298217984068a9f9ed4a0f0faa6255a86c581beeebbf95a849adaef4",
    "F4_REDUNDANT_arms.csv": "e9949fcc8cbf87f4fabd838b0db4a793ff919c3da0d0223be14a6f2cc6c72181",
    "F4_REDUNDANT_pair_I12.csv": "4b4adab0d9a468dc2eb95f562b6d5c7b617860f499eaaa6eb73d3bfc6f4f182e",
    "F4_REDUNDANT_partitions_r1.csv": "fb495cd3249eecc602c26263d3f5711e8a9072447c78025843c5221d46238a84",
    "F4_REDUNDANT_partitions_r2.csv": "10bfd2c62b698d9153ee2a9eb52515d781b56090b2a1ffac1f27fd96bf5dfc75"}
ORACLE_DIR = "correctness_oracle"
TOL_FLOOR = 1e-12           # E10: I12(R) >= I12(CLASS) - TOL_FLOOR (plug-in MI summation order only)
TOL_ROW = 1e-12             # E05: own row-level mean loss vs the source per_row convention (summation order)
TOL_FW = 1e-9               # E04: independent Frank-Wolfe gap / largest gradient term on the final released vector
TOL_STATS = 1e-9            # E01: token teacher sums vs the law (x max(n_t, 1)); = decoder.STATS_ROW_TOL
TOL_REPLAY = 1e-12          # E08: trace terms vs from-scratch (mapper SEARCH_RULES "replay"; abs)
TOL_ORACLE = TOL_TERMS      # E09: new enumeration vs the published source oracle tables (abs, per entry)
SEARCH_MARGIN = 1e-10       # E07: the mapper's search feasibility margin (SEARCH_RULES BUDGET_MARGIN); band reported
CONSTRAINED_ARMS = ("K-LOCAL", "K-SEQ-12", "K-SEQ-21", "K-JOINT-SINGLE", "K-JOINT-PAIR")
# pytest node IDs wired into E11 (run in one subprocess by the stage; owners in brackets)
WIRING_TESTS = {
    "science_stages_refuse_without_ENGINEERING_READY [A, lra/run.py]": ["lra/tests/test_run_gate.py"],
    "label_truth_table_matches_lra.family.overall_label [D]": [
        "lra/tests/test_truth_table.py::test_truth_table_document_matches_executable",
        "lra/tests/test_truth_table.py::test_overall_label_whole_truth_table_with_engineering_gate",
        "lra/tests/test_truth_table.py::test_overall_label_required_gate_argument_and_q_never_hides_incomplete_work",
        "lra/tests/test_truth_table.py::test_f08_f12_no_stale_cbp_or_lcr_names_and_every_reason_code_mapped"],
    "eval_lock_refuses_when_technical_validity_false [D]": [
        "lra/tests/test_late.py::test_f13_eval_lock_main_refuses_technical_failure_and_has_no_escape_flag",
        "lra/tests/test_late.py::test_f13_technical_validity_each_failure_alone",
        "lra/tests/test_late.py::test_f10_f13_technical_validity_ok_only_with_ready_gate_controls_locks_admission_and_clean_selection"],
    "assess_and_infer_refuse_without_validity_or_ENGINEERING_READY [D]": [
        "lra/tests/test_late.py::test_f10_f13_assess_verify_validity_refuses",
        "lra/tests/test_late.py::test_f13_assess_open_calls_the_validity_check_with_no_override",
        "lra/tests/test_late.py::test_f10_infer_refuses_without_engineering_ready"],
    "fit_audit_inner_loaders_never_return_assessment_labels [D]": [
        "lra/tests/test_audit.py::test_fit_audit_inner_loaders_never_return_assessment_labels"],
    "synthetic_coverage_of_paired_moves_and_temporary_partner [C, lra/mapper.py]": [
        "lra/tests/test_mapper.py::test_budgets_and_local_caps_enforced_including_paired",
        "lra/tests/test_mapper.py::test_paired_step_refuses_infeasible_update",
        "lra/tests/test_mapper.py::test_sequential_temporary_partner_not_required_feasible"],
    "engineering_verdict_and_rule_wired_to_code [B]": [
        "lra/tests/test_fixtures.py::test_engineering_verdict_truth_table",
        "lra/tests/test_fixtures.py::test_engineering_rule_file_matches_code",
        "lra/tests/test_fixtures.py::test_stage_correctness_refusals"],
}
FINDINGS_FILE = "REVIEW_FINDINGS_DISPOSITION.json"
N_FINDINGS = 14
RESOLVED_DISPOSITIONS = ("RESOLVED", "DUPLICATE_CONSOLIDATED_RESOLVED", "SUPERSEDED_RESOLVED")
NOT_BLOCKING = [
    "CLASS (the decision-only release) is zero-leakage on a fixture (I12(CLASS) = 0)",
    "CLASS is affordable (budget-feasible) on a fixture, so no class-preserving release can reveal less pair "
    "information than it (the decision disclosure floor, E10)",
    "no fixture shows a superior privacy release (no 0.01-nat gain, no equal-leakage utility gain, no favourable "
    "forecast)",
    "a constrained arm ties (or loses to) a weighted control",
    "a joint arm ties (or loses to) a sequential arm",
    "the source mechanism gate reads GATE_NOT_MET (SOURCE_FIXTURE_GATE.json, historical)",
    "a heuristic search arm has a gap to the exhaustive optimum of its own problem (labelled HEURISTIC with the gap)",
    "a constrained arm is INFEASIBLE under this registered decoder and search (reported as such; not a population or "
    "mathematical impossibility)",
    "the old trigger / route classification (descriptive output only)"]


def _file_sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def engineering_checks_text():
    from lra import decoder as DC
    return {
        "E01_LAW_COUNTS_ROUTING_HASHES": (
            "FIXTURE_LAWS.json sha256 == PINNED_LAWS_FILE_SHA256 and laws_sha256 == PINNED_LAWS_SHA256 (rule "
            "binding); load_laws verifies the hash and rebuilds every atom from the explicit tables; every atom count "
            "is a positive integer and they sum to N = 4096; static properties equal law_properties; per (f1, f2) the "
            "row counts equal pair_counts and the SEX counts equal n * sex_num / sex_den exactly; per fine cell the "
            "true-label counts equal count * labels / label_den exactly; fixture rows deploy to their declared fine "
            "cells; every release routes each row to a token of its teacher-predicted class (token class == teacher "
            "decision == released decision, strict argmax at the decision, |sum q - 1| <= 1e-12, release keys exactly "
            "row_id, tok1, q1, hard1, alpha1, tok2, q2, hard2, alpha2); every D1 decoder table's token counts n_t and "
            "label counts y_t equal the exact sums of the law counts of its member cells, and its teacher sums s_t "
            f"agree within {TOL_STATS} * max(n_t, 1)"),
        "E02_FIXED_TOKEN_INFORMATION": (
            "for every D0 map and its D1 version: identical token arrays and decisions (exact); I_1, I_2, I12 "
            "identical (bitwise); and for EVERY release of every arm (D0, D1, mapper): I(SEX; (token_i, q_i)) == "
            "I(SEX; token_i) and I(SEX; (token_1, q_1, token_2, q_2)) == I12 within 1e-15 (the decoder is a function "
            "of the token: the complete interface carries exactly the token information)"),
        "E03_CALIBRATED_NULL": (
            "F1_CALIBRATED_NULL only: for every same-class subset token of both recipients (every token of every "
            f"canonical partition) and every D1 release of every arm: max |q_D1 - q_D0| <= {TOL_NULL_Q} per token, "
            f"and the law (population) log loss and Brier of D1 >= those of D0 - {TOL_NULL_LOSS} per row"),
        "E04_D1_FINAL_VECTOR_CERTIFICATES": (
            "every D1 decoder table of every release (D1 fixed maps and every mapper unit) and every oracle subset "
            "solve: decoder.json reloads with its hash and a BITWISE re-solve of every supervised token; every "
            "supervised token certificate on the FINAL released vector is converged with stationarity_rel <= "
            f"{DC.STAT_TOL}, dual_infeas_rel <= {DC.DUAL_TOL}, projection_magnitude <= {DC.PROJ_TOL}, sum_q_residual "
            f"<= {DC.PROTO_SUM_TOL}, bracket_ulps <= 2, margin > 0, min_u >= 0; independently of the decoder code: u "
            "lies in the class-dominant simplex EXACTLY (u >= 0, u_k <= u_d), q == smooth(u, d) bitwise, |sum u - 1| "
            f"<= {DC.PROTO_SUM_TOL}, strict argmax d on q, and the Frank-Wolfe gap of the prompt objective over the "
            "2^(K-1) vertices of the class-dominant simplex (uniform vectors on subsets containing d), divided by the "
            f"largest gradient term, is <= {TOL_FW}; fallback tokens (n_t = 0) carry the pinned D0 vector bitwise"),
        "E05_LOSS_RECONSTRUCTION": (
            "true-label log loss (clip 1e-12, natural log) and source multiclass Brier of every release and of U "
            "recomputed row by row by this module's own code equal the source per_row convention within "
            f"{TOL_ROW} (means); mapper final_state_terms and deployed terms (L1, L2, B1, B2, I1, I2, I12, T, Phi) "
            f"equal the independent reconstruction within {TOL_TERMS}; the exhaustive table's terms for each arm's "
            f"partition equal the row-level reconstruction within {TOL_TERMS}; decoder.token_losses totals equal the "
            "row-level sums within 1e-9 (absolute, on totals); mapper-reported L_i(U), B_i(U) equal the independent "
            "values within 1e-12"),
        "E06_ACCEPTED_STATE_BUDGETS": (
            "for every constrained (K-) unit: every ACCEPTED deployed state in its persisted trace (each single move "
            "and each atomic paired move, both parts applied before the check), rebuilt by this module from the "
            "persisted stage start labels and the move parts and decoded from scratch with D1, satisfies, for every "
            "recipient enforced at that stage (K-LOCAL / K-SEQ stage r: r; K-JOINT-*: both; paired moves: both), "
            f"L_r <= L_r(U) + 0.005 + {TOL_BUDGET}, B_r <= B_r(U) + 0.003 + {TOL_BUDGET} and I_r <= I_r(C-TASK) + "
            f"{TOL_MI} by row-level recomputation, with tokens per predicted class <= the caps; every release reported "
            "FEASIBLE satisfies both recipients' budgets and local caps; unconstrained units (C-TASK, W-) report "
            "FEASIBLE for decisions and caps only and only K- arms may report INFEASIBLE; mapper.replay_unit reports "
            "no ENFORCED_CONSTRAINT_VIOLATED, PAIR_INFEASIBLE or FINAL_INFEASIBLE"),
        "E07_SEQUENTIAL_TEMPORARY_PARTNER": (
            "for every K-SEQ-12 / K-SEQ-21 start: the stage-1 partner view is exactly the CLASS-ONLY map of the other "
            "recipient and its constraints are not enforced; stage 1 is refined whenever the first recipient's own "
            "start satisfies its own budgets and local cap (recomputed here), whatever the partner's confidence "
            "(the number of starts refined while the CLASS-ONLY partner violates its own budget is reported); the "
            "stage-2 start keeps the frozen first map; every SEQ release reported FEASIBLE meets BOTH recipients' "
            "budgets and local caps by row-level recomputation; mapper.replay_unit reports no "
            "PARTNER_NOT_CLASS_ONLY, PARTNER_REQUIRED_FEASIBLE or FROZEN_MAP_CHANGED"),
        "E08_INCREMENTAL_REPLAY": (
            "for every mapper unit (C-TASK, W-, K-): mapper.replay_unit on the PERSISTED record and trace is ok and "
            f"reports no TERMS_MISMATCH, DELTA_MISMATCH, STATS_HASH_MISMATCH or CACHED_SOLVE_MISMATCH (terms and "
            "deltas <= 1e-12 abs; statistics and cached q bitwise) and its checks check6 / check7 / check8 are not "
            "false; every unit's starts_given equals mapper.registered_starts(arm, lam) exactly (the full registered "
            "start / witness list, no fixture subset); independently, every traced state's terms_after "
            f"equal this module's from-scratch row-level terms within {TOL_TERMS}; the winning start's incremental "
            f"search-state terms equal the from-scratch final_state_terms within {TOL_TERMS}; each stage start equals "
            "the previous stage end; the final labels equal the released policy's labels"),
        "E09_EXHAUSTIVE_ORACLE": (
            "the stage's exhaustive enumeration covers every canonical same-class partition pair (count == "
            "properties.mapping_pairs) and reproduces the published source oracle tables (partitions_r1, "
            f"partitions_r2, pair_I12; sha256-pinned) entry by entry within {TOL_ORACLE} (index, labels and "
            "tokens_per_class exact); every arm's partition is in the enumeration; every arm carries "
            "EXHAUSTIVE_OPTIMAL, HEURISTIC (with its gap) or NOT_A_SEARCH; a labelled HEURISTIC gap is reported, NOT "
            "a failure; the source arms tables are compared descriptively only"),
        "E10_DECISION_DISCLOSURE_FLOOR": (
            "for every release R of every arm and every enumerated partition pair: I12(R) >= I12(CLASS) - "
            f"{TOL_FLOOR} and I_i(R) >= I_i(CLASS) - {TOL_FLOOR} (data processing: every class-preserving release "
            "determines both decisions); I12(CLASS) equals the plug-in MI of SEX with the two teacher decisions; "
            "CLASS|D1 budget feasibility (affordability) is RECORDED, never a failure"),
        "E11_LAUNCH_WIRING": (
            "the registered pytest node IDs of WIRING_TESTS, run in one subprocess by the stage, all collect and pass "
            "(every group non-empty): science stages refuse without a lock-bound pushed ENGINEERING_READY "
            "(lra/run.py); LABEL_TRUTH_TABLE.json matches lra.family.overall_label (no stale cbp / lcr names); "
            "lra.eval_lock refuses when technical validity is false; lra.assess and lra.infer refuse without validity "
            "/ ENGINEERING_READY; fit / audit / inner loaders never return assessment labels; the mapper's synthetic coverage of accepted paired moves and of the "
            "temporary sequential partner (the fixture laws need not exercise a paired acceptance; E06 reports the "
            "count); this module's verdict / rule / refusal paths; and ENGINEERING_GATE_RULE.json on disk equals "
            "engineering_gate_rule() (rule text wired to code)"),
        "E12_REVIEW_FINDINGS": (
            f"{FINDINGS_FILE} lists exactly the {N_FINDINGS} inherited findings (ordinals 1..{N_FINDINGS}), each with a "
            f"disposition in {list(RESOLVED_DISPOSITIONS)} and at least one regression test node ID; every listed "
            "node ID collects and passes in the same subprocess; no finding is unresolved (an unresolved finding "
            "whose affects_required_adult_execution is not false would also be named as an Adult-execution "
            "blocker)"),
    }


def engineering_gate_rule():
    """ENGINEERING_GATE_RULE.json body (prompt sec. 9). Generated by code; the stage refuses if the file differs."""
    from lra import decoder as DC
    checks = engineering_checks_text()
    return {
        "schema": ENG_RULE_SCHEMA,
        "question": "is the implementation correct enough to run the registered Adult development test? (NOT: has a "
                    "synthetic example already proved superiority?)",
        "verdict_strings": list(VERDICTS),
        "verdict_rule": "ENGINEERING_READY iff every mandatory check E01..E12 passes (each of E01, E02 and E04..E10 on "
                        "all four pinned fixtures, E03 on F1_CALIBRATED_NULL, E11 and E12 once); otherwise "
                        "ENGINEERING_BLOCKED with the failing check ids (and any fixture that raised) as reasons. "
                        "Nothing else enters the verdict.",
        "mandatory_checks": checks,
        "check_ids": list(checks),
        "prompt_check_map": {str(i + 1): k for i, k in enumerate(checks)},
        "must_not_block": NOT_BLOCKING,
        "descriptive_only": "the predecessor's trigger, T*, qualifying rule, route classification and descriptive "
                            "flags (lra.fixtures.trigger / descriptive) are computed and written under 'descriptive' "
                            "per fixture; engineering_verdict() never reads them",
        "laws": {"file": "FIXTURE_LAWS.json", "file_sha256": PINNED_LAWS_FILE_SHA256,
                 "laws_sha256": PINNED_LAWS_SHA256, "families": list(FIXTURE_IDS),
                 "status": "the four lcr laws, copied unchanged (N = 4096 exact expected counts, caps 2/2, budgets "
                           "0.005 / 0.003, kappa = 32, eps = 1e-12); KNOWN, ALREADY-OPENED regression cases, not "
                           "fresh mechanism evidence; never edited or replaced to make a check pass"},
        "source_gate": {"file": SOURCE_GATE_FILE, "sha256": PINNED_SOURCE_GATE_SHA256, "verdict": "GATE_NOT_MET",
                        "status": "the predecessor's historical performance-gate result (MECHANISM_GATE_NOT_MET); "
                                  "preserved unchanged; never this study's launch verdict and never a blocker"},
        "source_oracle_tables": {"dir": f"{SOURCE_RESULTS}/fixture_oracle", "sha256": SOURCE_ORACLE_SHA256,
                                 "compared": "partitions_r1, partitions_r2, pair_I12 (mandatory, E09); arms "
                                             "(descriptive)"},
        "wiring_tests": WIRING_TESTS,
        "review_findings": {"file": FINDINGS_FILE, "count": N_FINDINGS},
        "rows": "each fixture's fitting rows are its atoms replicated by count (exact expected counts, no sampling); "
                "all rows are fitting rows; population law = fitting law",
        "arms": "D0 bank (FINE-TASK, CLASS, DIRECT-TASK, 24 old-objective privacy maps), their D1 versions with "
                "assignments unchanged, and lra.mapper C-TASK, 24 W- and 5 K- units (fixture mode: fixture caps and "
                "the registered 0.005 / 0.003 budgets), as in the predecessor's fixture engine",
        "enumeration_vs_certificates": "exhaustive enumeration is exact over PARTITIONS of these finite laws only; "
                                       "within a partition the decoder is a numerical convex solve, certified by E04. "
                                       "A fixed-token certificate is not discrete-search optimality (E09 labels).",
        "tolerances": {"TOL_BUDGET": TOL_BUDGET, "TOL_MI": TOL_MI, "TOL_TERMS": TOL_TERMS, "TOL_NULL_Q": TOL_NULL_Q,
                       "TOL_NULL_LOSS": TOL_NULL_LOSS, "TOL_OPT": TOL_OPT, "TOL_FLOOR": TOL_FLOOR,
                       "TOL_ROW": TOL_ROW, "TOL_FW": TOL_FW, "TOL_STATS": TOL_STATS, "TOL_REPLAY": TOL_REPLAY,
                       "TOL_ORACLE": TOL_ORACLE, "decoder": DC.TOLERANCES},
        "stage": "lra.fixtures.stage_correctness(None, None) through lra.run --stage correctness under the pushed "
                 "CORRECTNESS_LOCK; one process, no shard; writes ENGINEERING_GATE_RESULT.json, "
                 f"{ORACLE_DIR}/*.csv and private units cor__<FID> (records, traces, policies, decoders, releases)",
        "repairs": "if blocked: bounded engineering repairs before real fitting, as dated code-only amendments pushed "
                   "before reruns; prior attempts kept; no law, objective, comparator bank or success criterion is "
                   "changed to fix a failing check",
        "scope": "fixture MI is the exact MI of these finite laws, not a population guarantee for Adult; "
                 "ENGINEERING_READY licenses the locked Adult study through SCIENCE_LOCK and nothing else",
    }


def write_engineering_rule(pkg=PKG):
    rule = engineering_gate_rule()
    (Path(pkg) / ENG_RULE_FILE).write_text(json.dumps(rule, indent=1, allow_nan=False) + "\n")
    return _file_sha(Path(pkg) / ENG_RULE_FILE)


def engineering_verdict(checks, fixtures_expected=FIXTURE_IDS, fixture_errors=None):
    """(verdict, reasons) from {check_id: {"pass": bool, ...}}. The ONLY input is the mandatory checks (and the
    completeness of the fixture set); trigger / route / descriptive output is never read."""
    reasons = []
    ids = list(engineering_checks_text())
    for cid in ids:
        c = checks.get(cid)
        if c is None:
            reasons.append(f"MISSING_CHECK:{cid}")
        elif c.get("pass") is not True:
            reasons.append(f"CHECK_FAILED:{cid}")
    for cid in checks:
        if cid not in ids:
            reasons.append(f"UNREGISTERED_CHECK:{cid}")
    for f, e in sorted((fixture_errors or {}).items()):
        reasons.append(f"FIXTURE_ERROR:{f}:{e}")
    return ("ENGINEERING_READY" if not reasons else "ENGINEERING_BLOCKED"), reasons


# ------------------------------------------------------------------ independent helpers
def _own_losses(q, y):
    """Own row-level (mean log loss with clip 1e-12, mean multiclass Brier); no dpc / decoder code."""
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, K = q.shape
    py = q[np.arange(n), y]
    ll = -np.log(np.minimum(np.maximum(py, 1e-12), 1.0))
    oh = np.zeros((n, K))
    oh[np.arange(n), y] = 1.0
    br = ((q - oh) ** 2).sum(1)
    return float(ll.mean()), float(br.mean())


def _canon(lab):
    """Canonical labels: each cell labelled by the lowest member index of its block."""
    lab = np.asarray(lab, dtype=np.int64)
    out = np.empty_like(lab)
    first = {}
    for f, v in enumerate(lab):
        first.setdefault(int(v), f)
    for f, v in enumerate(lab):
        out[f] = first[int(v)]
    return out


def _state_metrics(X, labels, need=(1, 2)):
    """Own from-scratch D1 terms of a labelled state: per recipient tokens, q (decode_policy), L, B (own row-level
    code), I (own plug-in MI); I12 when both recipients are given. labels: {r: per-cell labels}."""
    from lra import decoder as DC
    from qpc import release as RL
    out, toks, tabs, pols = {}, {}, {}, {}
    for r in need:
        lab = _canon(labels[r])
        pol = RL.make_policy(r, X["fine"][r], lab)
        tok, _, _ = RL.encode(pol, X["P"][r], X["d"][r])
        dec = DC.decode_policy(pol, tok, X["P"][r], X["y"][r])
        q = dec.q[tok]
        out[f"L{r}"], out[f"B{r}"] = _own_losses(q, X["y"][r])
        out[f"I{r}"] = plugin_mi(X["s"], tok)
        out[f"tokens_per_class{r}"] = [int(x) for x in pol.tokens_per_class()]
        out[f"decisions_ok{r}"] = bool(np.array_equal(dec.token_class[tok], X["d"][r]))
        toks[r], tabs[r], pols[r] = tok, dec, pol
    if len(need) == 2:
        out["I12"] = plugin_mi(X["s"], toks[1], toks[2])
    return out, toks, tabs, pols


def _fw_gap(y, s, n, d, u, q):
    """Independent Frank-Wolfe certificate of the prompt objective at the released q over the class-dominant simplex
    (vertices = uniform vectors on subsets containing d). Returns (relative gap, absolute gap, scale)."""
    from itertools import combinations
    from lra.decoder import EPS, KAPPA
    K = len(y)
    Z = 1.0 + (K + 1) * EPS
    pbar = np.asarray(s, dtype=np.float64) / float(n)
    a = np.asarray(y, dtype=np.float64) + KAPPA * pbar
    q = np.asarray(q, dtype=np.float64)
    # d/du_k of sum y_k(-log q_k) + 0.5 sum_rows ||q - e_Y||^2 + kappa KL(pbar||q), q = (u + c)/Z
    g = (-a / q + n * q - np.asarray(y, dtype=np.float64)) / Z
    scale = float(np.max((a / q + n * q + np.asarray(y, dtype=np.float64)) / Z))
    others = [k for k in range(K) if k != d]
    best = float(g[d])
    for m in range(1, K):
        for A in combinations(others, m):
            v = (float(g[d]) + sum(float(g[k]) for k in A)) / (m + 1)
            best = min(best, v)
    gap = float(np.dot(g, np.asarray(u, dtype=np.float64))) - best
    return gap / scale, gap, scale


def _arm_tables(a):
    """(dec1, dec2) DecoderTables of a D1 release (no re-solve here)."""
    from lra import decoder as DC
    _, d1, d2, _ = DC.load_decoder_pair(json.loads(json.dumps(a["decoder_body"])), a["pair"], verify_solve=False)
    return d1, d2


def _start_pair(name, bank, mfiles):
    from qpc import release as RL
    if name in bank:
        return bank[name]
    if name in mfiles:
        return RL.PolicyPair.from_dict(mfiles[name]["policy.json"])
    return None


def _labels_of(pair):
    from qpc import compress as QC
    return {1: _canon(QC.labels_from_policy(pair.p1)), 2: _canon(QC.labels_from_policy(pair.p2))}


def _apply_move(lab, parts):
    """Apply one (single or atomic paired) move to {r: labels}; parts [{"r", "f", "to_canon"}]; canonicalise."""
    new = {r: np.array(v, dtype=np.int64) for r, v in lab.items()}
    for p in parts:
        r, f, to = int(p["r"]), int(p["f"]), int(p["to_canon"])
        if int(new[r][to]) != to:
            raise ValueError(f"move target {to} is not a canonical label of recipient {r}")
        new[r][f] = to
    return {r: _canon(v) for r, v in new.items()}


def _trace_states(trace):
    """Yield every traced state: (start, stage dict, step, kind, labels {r: array}, terms_after or None, parts)."""
    for st in trace.get("starts", []):
        for sg in st.get("stages", []):
            lab = {int(r): _canon(v) for r, v in sg["start_labels"].items()}
            yield st, sg, -1, "start", lab, sg.get("start_terms"), []
            for mv in sg.get("moves", []):
                lab = _apply_move(lab, mv["parts"])
                yield st, sg, int(mv["step"]), mv["type"], lab, mv.get("terms_after"), mv["parts"]


def _enforced(arm, sg):
    if arm in ("K-JOINT-SINGLE", "K-JOINT-PAIR"):
        return (1, 2)
    if arm in ("K-LOCAL", "K-SEQ-12", "K-SEQ-21"):
        return tuple(int(r) for r in sg.get("recipients", sg.get("enforced", [])))
    return ()


# ------------------------------------------------------------------ per-fixture engineering checks
def _guard(fn, *a):
    try:
        return fn(*a)
    except Exception as e:                                     # a crash is a FAILED check, never a silent pass
        return {"pass": False, "failures": [f"EXCEPTION:{e.__class__.__name__}:{e}"]}


def _e01(fam, laws_ok, X, arms, res):
    fails = list(laws_ok)
    A = fam["atoms"]
    if not all(isinstance(a[5], int) and a[5] > 0 for a in A) or sum(a[5] for a in A) != N_FIX:
        fails.append("atom_counts_not_positive_integers_summing_to_4096")
    if law_properties(fam) != fam["properties"]:
        fails.append("static_properties")
    if X["N"] != N_FIX:
        fails.append("rows")
    F1, F2 = X["fine"][1].F, X["fine"][2].F
    pc = np.zeros((F1, F2), dtype=np.int64)
    sc = np.zeros((F1, F2), dtype=np.int64)
    np.add.at(pc, (X["f1"], X["f2"]), 1)
    np.add.at(sc, (X["f1"], X["f2"]), X["s"])
    if pc.tolist() != fam["pair_counts"]:
        fails.append("pair_counts")
    exp_s = [[Fraction(fam["pair_counts"][a][b] * fam["sex_num"][a][b], fam["sex_den"]) for b in range(F2)]
             for a in range(F1)]
    if any(exp_s[a][b] != int(sc[a, b]) for a in range(F1) for b in range(F2)):
        fails.append("sex_counts")
    cellY = {}
    for r in (1, 2):
        R = fam["recipients"][str(r)]
        f = X["f1"] if r == 1 else X["f2"]
        E = np.zeros((len(R["cells"]), R["K"]), dtype=np.int64)
        for c, cell in enumerate(R["cells"]):
            got = np.bincount(X["y"][r][f == c], minlength=R["K"])
            ex = [Fraction(cell["count"] * x, R["label_den"]) for x in cell["labels"]]
            if any(e.denominator != 1 for e in ex) or [int(e) for e in ex] != got.tolist():
                fails.append(f"label_counts:r{r}:cell{c}")
            E[c] = [int(e) for e in ex]
            if int(X["fine"][r].n[c]) != cell["count"] or int(X["fine"][r].cell_class[c]) != cell["class"]:
                fails.append(f"fine_cell_count_or_class:r{r}:cell{c}")
        cellY[r] = E
    keys = ["row_id", "tok1", "q1", "hard1", "alpha1", "tok2", "q2", "hard2", "alpha2"]
    n_tables = 0
    for cid, a in arms.items():
        rel = a["rel"]
        if list(rel) != keys:
            fails.append(f"{cid}:release_keys")
        for r, pol in ((1, a["pair"].p1), (2, a["pair"].p2)):
            tok, q, hard = np.asarray(rel[f"tok{r}"]), np.asarray(rel[f"q{r}"]), np.asarray(rel[f"hard{r}"])
            tc = np.asarray(pol.token_class)[tok]
            rows = np.arange(q.shape[0])
            other = q.copy()
            other[rows, hard] = -np.inf
            if not (np.array_equal(tc, X["d"][r]) and np.array_equal(hard, X["d"][r])
                    and np.all(q[rows, hard] > other.max(1)) and np.max(np.abs(q.sum(1) - 1.0)) <= 1e-12):
                fails.append(f"{cid}:routing_r{r}")
        if "decoder_body" in a:
            d1, d2 = _arm_tables(a)
            for r, pol, tab in ((1, a["pair"].p1, d1), (2, a["pair"].p2, d2)):
                n_tables += 1
                ct = np.asarray(pol.cell_token)
                T = tab.T
                ny = np.zeros((T, tab.K), dtype=np.int64)
                nn = np.zeros(T, dtype=np.int64)
                ss = np.zeros((T, tab.K))
                R = fam["recipients"][str(r)]
                for c in range(len(R["cells"])):
                    ny[ct[c]] += cellY[r][c]
                    nn[ct[c]] += R["cells"][c]["count"]
                    ss[ct[c]] += np.asarray(R["cells"][c]["teacher"], dtype=np.float64) * (
                        R["cells"][c]["count"] / float(R["teacher_den"]))
                if not (np.array_equal(nn, tab.n) and np.array_equal(ny.astype(np.float64), tab.y)):
                    fails.append(f"{cid}:decoder_label_counts_r{r}")
                if np.any(np.abs(ss - tab.s) > TOL_STATS * np.maximum(nn, 1)[:, None]):
                    fails.append(f"{cid}:decoder_teacher_sums_r{r}")
    return {"pass": not fails, "failures": fails, "releases": len(arms), "decoder_tables": n_tables}


def _e02(X, arms, res):
    c1 = res["checks"]["C1_FIXED_TOKEN_INFORMATION"]
    fails = list(c1["failures"])
    for cid, a in arms.items():
        if a["kind"] in ("d1_fixed", "d1_task"):
            b = arms[a["d0_of"]]
            if not all(np.array_equal(a["rel"][f"hard{i}"], b["rel"][f"hard{i}"]) for i in (1, 2)):
                fails.append(f"{cid}:decisions_differ_from_d0")
        rel = a["rel"]
        full = plugin_mi(X["s"], np.column_stack([rel["tok1"], rel["q1"], rel["tok2"], rel["q2"]]))
        if abs(full - a["metrics"]["I12"]) > 1e-15:
            fails.append(f"{cid}:pair_interface")
    return {"pass": not fails, "failures": fails, "pairs_checked": c1["pairs_checked"], "releases": len(arms)}


def _e03(fid, res):
    if fid != "F1_CALIBRATED_NULL":
        return {"pass": True, "applicable": False}
    c2 = res["checks"]["C2_CALIBRATED_NULL"]
    return {"pass": bool(c2["pass"]), "applicable": True, **{k: v for k, v in c2.items() if k != "pass"}}


def _e04(arms, tables):
    from lra import decoder as DC
    from qpc.kmeans import smooth
    fails, worst = [], {"stationarity_rel": 0.0, "dual_infeas_rel": 0.0, "projection_magnitude": 0.0,
                        "sum_q_residual": 0.0, "fw_rel": 0.0, "sum_u_residual": 0.0}
    min_margin, n_tok, n_fb, n_tab = None, 0, 0, 0

    def cert_ok(c, tag):
        nonlocal min_margin
        bad = (not c.get("converged") or c["stationarity_rel"] > DC.STAT_TOL or c["dual_infeas_rel"] > DC.DUAL_TOL
               or c["projection_magnitude"] > DC.PROJ_TOL or c["sum_q_residual"] > DC.PROTO_SUM_TOL
               or c["bracket_ulps"] > 2 or not c["margin"] > 0 or c["min_u"] < 0)
        for k in ("stationarity_rel", "dual_infeas_rel", "projection_magnitude", "sum_q_residual"):
            worst[k] = max(worst[k], float(c[k]))
        min_margin = c["margin"] if min_margin is None else min(min_margin, c["margin"])
        if bad:
            fails.append(f"{tag}:certificate")

    def own(y, s, n, d, u, q, tag):
        u = np.asarray(u, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        ok = bool(np.all(u >= 0) and np.all(u <= u[d]))
        if not np.array_equal(smooth(u[None], np.array([d]), check=False)[0], q):
            ok = False
        su = abs(float(sum(float(x) for x in u)) - 1.0)
        worst["sum_u_residual"] = max(worst["sum_u_residual"], su)
        oth = np.delete(q, d)
        if su > DC.PROTO_SUM_TOL or not q[d] > oth.max():
            ok = False
        rel, _, _ = _fw_gap(y, s, n, d, u, q)
        worst["fw_rel"] = max(worst["fw_rel"], rel)
        if rel > TOL_FW:
            ok = False
        if not ok:
            fails.append(f"{tag}:independent")

    for cid, a in sorted(arms.items()):
        if "decoder_body" not in a:
            continue
        try:
            _, d1, d2, _ = DC.load_decoder_pair(json.loads(json.dumps(a["decoder_body"])), a["pair"], verify_solve=True)
        except (ValueError, DC.DecoderError) as e:
            fails.append(f"{cid}:reload:{e}")
            continue
        for r, pol, tab in ((1, a["pair"].p1, d1), (2, a["pair"].p2, d2)):
            n_tab += 1
            for t in range(tab.T):
                c = tab.certs[t]
                if tab.fallback[t]:
                    n_fb += 1
                    if not np.array_equal(tab.q[t], np.asarray(pol.token_proto)[t]):
                        fails.append(f"{cid}:r{r}:t{t}:fallback")
                    continue
                n_tok += 1
                cert_ok(c, f"{cid}:r{r}:t{t}")
                own(tab.y[t], tab.s[t], int(tab.n[t]), int(tab.token_class[t]), tab.u[t], tab.q[t], f"{cid}:r{r}:t{t}")
    n_sub = 0
    for r, sub in ((1, tables["sub1"]), (2, tables["sub2"])):
        for key, v in sub.items():
            n_sub += 1
            cert_ok(v["cert"], f"oracle:r{r}:{key}")
    return {"pass": not fails, "failures": fails[:200], "n_failures": len(fails), "decoder_tables": n_tab,
            "supervised_tokens": n_tok, "fallback_tokens": n_fb, "oracle_subset_solves": n_sub, "worst": worst,
            "min_margin": min_margin}


def _e05(X, arms, res, mrecs):
    c5 = res["checks"]["C5_TERM_RECONSTRUCTION"]
    fails = [f for f in c5["failures"] if ":incremental" not in f]
    worst = 0.0
    Uown = {}
    from dpc.utility import per_row
    for i in (1, 2):
        Uown[i] = _own_losses(X["P"][i], X["y"][i])
        pr = per_row(X["P"][i], X["y"][i], X["P"][i].shape[1])
        for v, w in zip(Uown[i], (float(pr["ll"].mean()), float(pr["br"].mean()))):
            worst = max(worst, abs(v - w))
            if abs(v - w) > TOL_ROW:
                fails.append(f"U:r{i}")
    for cid, a in arms.items():
        for i in (1, 2):
            L, B = _own_losses(np.asarray(a["rel"][f"q{i}"]), X["y"][i])
            for k, v in ((f"L{i}", L), (f"B{i}", B)):
                dlt = abs(v - a["metrics"][k])
                worst = max(worst, dlt)
                if dlt > TOL_ROW:
                    fails.append(f"{cid}:own_row:{k}")
    for cid, rec in mrecs.items():
        for i in (1, 2):
            if abs(rec["budgets"]["L_U"][str(i)] - Uown[i][0]) > 1e-12 or \
                    abs(rec["budgets"]["B_U"][str(i)] - Uown[i][1]) > 1e-12:
                fails.append(f"{cid}:U{i}_own")
    return {"pass": not fails, "failures": fails, "max_abs_diff_own_vs_source_rows": worst,
            "max_abs_diff_terms": c5["max_abs_diff"],
            "token_loss_helper_max_abs_diff_totals": c5["token_loss_helper_max_abs_diff_totals"]}


def _replay(cid, rec, files, X, refs, starts=None):
    from lra import mapper as MP
    tr = X["tr"]
    Y = {1: X["y"][1], 2: X["y"][2]}
    return MP.replay_unit(rec, X["fine_dict"], X["T"], tr, Y, X["s"], refs, trace=files["trace.json"],
                          files={"policy.json": files["policy.json"], "release.npz": files["release.npz"]},
                          starts=starts)


def _limits(U, ct_metrics):
    return ({r: U[r][0] + BUDGET["ll"] for r in (1, 2)}, {r: U[r][1] + BUDGET["brier"] for r in (1, 2)},
            {r: ct_metrics[f"I{r}"] for r in (1, 2)} if ct_metrics else None)


def _ok_r(m, r, limL, limB, cap, caps):
    v = []
    if m[f"L{r}"] > limL[r] + TOL_BUDGET:
        v.append("L")
    if m[f"B{r}"] > limB[r] + TOL_BUDGET:
        v.append("B")
    if cap is not None and m[f"I{r}"] > cap[r] + TOL_MI:
        v.append("I")
    if any(t > caps[r - 1] for t in m[f"tokens_per_class{r}"]):
        v.append("cap")
    if not m[f"decisions_ok{r}"]:
        v.append("decisions")
    return v


def _unit_states(cid, arm, files, X, limL, limB, cap):
    """Own check of every traced state of one unit: (failures, n_states, n_accepted, n_pair, max term diff, per-stage
    end labels)."""
    trace = files.get("trace.json")
    if not isinstance(trace, dict) or not trace.get("starts"):
        return [f"{cid}:trace_missing"], 0, 0, 0, 0.0, {}
    fails, n_states, n_acc, n_pair, worst = [], 0, 0, 0, 0.0
    ends = {}
    for st, sg, step, kind, lab, terms, parts in _trace_states(trace):
        n_states += 1
        if kind != "start":
            n_acc += 1
            n_pair += kind == "pair"
        m, _, _, _ = _state_metrics(X, lab)
        if terms:
            for k in ("L1", "L2", "B1", "B2", "I1", "I2", "I12"):
                if k in terms and terms[k] is not None:
                    dlt = abs(float(terms[k]) - m[k])
                    worst = max(worst, dlt)
                    if dlt > TOL_TERMS:
                        fails.append(f"{cid}:{st['name']}:{sg['stage']}:{step}:terms:{k}")
        if arm in CONSTRAINED_ARMS and kind != "start":
            enf = (1, 2) if kind == "pair" else _enforced(arm, sg)
            for r in enf:
                bad = _ok_r(m, r, limL, limB, cap, X["caps"])
                if bad:
                    fails.append(f"{cid}:{st['name']}:{sg['stage']}:{step}:{kind}:r{r}:{','.join(bad)}")
        ends[(st["name"], sg["stage"])] = lab
    return fails, n_states, n_acc, n_pair, worst, ends


def _e06_e07_e08(fid, X, U, arms, mrecs, mfiles, bank):
    from lra import run as R
    ct = arms.get(R.ctask_id())
    limL, limB, cap = _limits(U, ct["metrics"] if ct else None)
    e6, e7, e8 = [], [], []
    stats = {"units": len(mrecs), "states": 0, "accepted_states": 0, "paired_accepted": 0, "replay_ok": 0,
             "max_trace_vs_own": 0.0, "max_replay_abs_diff": 0.0, "seq_starts": 0,
             "seq_refined_with_partner_over_budget": 0, "partner_class_only_checked": 0}
    codes6 = {"ENFORCED_CONSTRAINT_VIOLATED", "PAIR_INFEASIBLE", "FINAL_INFEASIBLE"}
    codes7 = {"PARTNER_NOT_CLASS_ONLY", "PARTNER_REQUIRED_FEASIBLE", "FROZEN_MAP_CHANGED"}
    codes8 = {"TERMS_MISMATCH", "DELTA_MISMATCH", "STATS_HASH_MISMATCH", "CACHED_SOLVE_MISMATCH"}
    kref = None
    if ct is not None:
        from lra import mapper as MP
        kref = MP.refs_from_ctask(mrecs[R.ctask_id()])
    for cid, rec in sorted(mrecs.items()):
        files = mfiles[cid]
        arm = rec["arm"]
        # mapper replay (C) on the persisted record + trace
        try:
            sdict = {}
            for nm in rec.get("starts_given", []):
                sp = _start_pair(nm, bank, mfiles)
                if sp is not None:
                    sdict[nm] = sp.to_dict()
            rep = _replay(cid, rec, files, X, {"caps": list(X["caps"]), "budget": dict(BUDGET),
                                                **(kref if arm in CONSTRAINED_ARMS and kref else {})}, sdict)
            stats["max_replay_abs_diff"] = max(stats["max_replay_abs_diff"], float(rep.get("max_abs_diff") or 0.0))
            vc = [v["code"] for v in rep.get("violations", [])]
            if rep.get("ok"):
                stats["replay_ok"] += 1
            else:
                e8.append(f"{cid}:replay_not_ok:{sorted(set(vc))}")
            rc = rep.get("checks") or {}
            if rc.get("check6") is False:
                e6.append(f"{cid}:replay_check6_false")
            if rc.get("check7") is False and "SEQ" in arm:
                e7.append(f"{cid}:replay_check7_false")
            if rc.get("check8") is False:
                e8.append(f"{cid}:replay_check8_false")
            e6 += [f"{cid}:replay:{c}" for c in vc if c in codes6]
            e7 += [f"{cid}:replay:{c}" for c in vc if c in codes7 or (c == "FINAL_INFEASIBLE" and "SEQ" in arm)]
            e8 += [f"{cid}:replay:{c}" for c in vc if c in codes8]
        except Exception as e:
            e8.append(f"{cid}:replay_exception:{e.__class__.__name__}:{e}")
        e8 += res_incremental_failures(arms, {cid: rec})
        try:
            from lra import mapper as MP
            reg = MP.registered_starts(arm, rec.get("lam"))
            if list(rec.get("starts_given", [])) != list(reg):
                e8.append(f"{cid}:starts_given_differ_from_registered_starts")
        except Exception as e:
            e8.append(f"{cid}:registered_starts_exception:{e}")
        # own state-by-state rebuild
        try:
            f6, ns, na, npair, worst, ends = _unit_states(cid, arm, files, X, limL, limB, cap)
        except Exception as e:
            f6, ns, na, npair, worst, ends = [f"{cid}:own_rebuild_exception:{e.__class__.__name__}:{e}"], 0, 0, 0, 0.0, {}
        stats["states"] += ns
        stats["accepted_states"] += na
        stats["paired_accepted"] += npair
        stats["max_trace_vs_own"] = max(stats["max_trace_vs_own"], worst)
        e6 += [f for f in f6 if ":terms:" not in f]
        e8 += [f for f in f6 if ":terms:" in f or f.endswith("trace_missing")]
        # final release: feasible claims (both recipients for K- arms)
        a = arms[cid]
        if rec["status"] == "FEASIBLE" and arm in CONSTRAINED_ARMS:
            m, _, _, _ = _state_metrics(X, _labels_of(a["pair"]))
            for r in (1, 2):
                bad = _ok_r(m, r, limL, limB, cap, X["caps"])
                if bad:
                    (e7 if arm.startswith("K-SEQ") else e6).append(f"{cid}:final:r{r}:{','.join(bad)}")
        # final labels == released policy labels
        tr_ = files.get("trace.json") or {}
        win = (tr_.get("winner") or {}).get("labels")
        if win is not None:
            rl = _labels_of(a["pair"])
            if any(not np.array_equal(_canon(win[str(r)]), rl[r]) for r in (1, 2)):
                e8.append(f"{cid}:winner_labels_differ_from_release")
        # stage chaining and start maps (own): each start's labels equal the registered start / witness map
        for st in tr_.get("starts", []):
            sp = _start_pair(st["name"], bank, mfiles)
            if sp is None:
                e8.append(f"{cid}:{st['name']}:unknown_start")
                continue
            sl = _labels_of(sp)
            if any(not np.array_equal(_canon(st["labels"][str(r)]), sl[r]) for r in (1, 2)):
                e8.append(f"{cid}:{st['name']}:start_labels_differ_from_start_map")
            sgs = st.get("stages", [])
            if arm.startswith("K-SEQ") or arm.startswith("W-SEQ"):
                a_, b_ = (1, 2) if arm.endswith("12") else (2, 1)
                if sgs:
                    s1 = sgs[0]
                    from qpc import compress as QC
                    cls_b = _canon(QC.class_labels(X["fine"][b_]))
                    stats["partner_class_only_checked"] += 1
                    plab = s1["start_labels"].get(str(b_))
                    if plab is None:
                        plab = (s1.get("partner") or {}).get("labels")
                    if plab is None or not np.array_equal(_canon(plab), cls_b):
                        e7.append(f"{cid}:{st['name']}:partner_not_class_only")
                    if (s1.get("partner") or {}).get("constraints_enforced") not in (False,):
                        e7.append(f"{cid}:{st['name']}:partner_constraints_enforced")
                if arm.startswith("K-SEQ") and sgs:
                    stats["seq_starts"] += 1
                    s1 = sgs[0]
                    m1, _, _, _ = _state_metrics(X, {r: _canon(v) for r, v in
                                                     ((int(k), v) for k, v in s1["start_labels"].items())})
                    own_bad = _ok_r(m1, a_, limL, limB, cap, X["caps"])
                    strict_ok = not own_bad and m1[f"L{a_}"] <= limL[a_] - SEARCH_MARGIN and \
                        m1[f"B{a_}"] <= limB[a_] - SEARCH_MARGIN
                    refined = s1.get("status") == "REFINED"
                    if strict_ok and not refined:
                        e7.append(f"{cid}:{st['name']}:stage1_not_refined_with_feasible_own_start")
                    if own_bad and refined:
                        e7.append(f"{cid}:{st['name']}:stage1_refined_from_infeasible_own_start")
                    if not strict_ok and not own_bad:
                        stats["seq_own_start_in_margin_band"] = stats.get("seq_own_start_in_margin_band", 0) + 1
                    if refined and _ok_r(m1, b_, limL, limB, None, X["caps"]):
                        stats["seq_refined_with_partner_over_budget"] += 1
                    if len(sgs) > 1:
                        end1 = ends.get((st["name"], s1["stage"]))
                        if end1 is not None and not np.array_equal(_canon(sgs[1]["start_labels"][str(a_)]),
                                                                   end1[a_]):
                            e7.append(f"{cid}:{st['name']}:frozen_map_changed_at_stage2_start")
    return ({"pass": not e6, "failures": e6, **{k: stats[k] for k in ("units", "states", "accepted_states",
                                                                      "paired_accepted")}},
            {"pass": not e7, "failures": e7, **{k: stats[k] for k in ("seq_starts",
                                                                      "seq_refined_with_partner_over_budget",
                                                                      "partner_class_only_checked")}},
            {"pass": not e8 and stats["replay_ok"] == len(mrecs), "failures": e8,
             **{k: stats[k] for k in ("replay_ok", "max_trace_vs_own", "max_replay_abs_diff")}})


def res_incremental_failures(arms, mrecs):
    """The predecessor's incremental-winner comparison (C5 incremental part, with the A1 NO_SEARCH_STATE scope)."""
    out = []
    for cid, rec in mrecs.items():
        inc = _incremental_terms(rec)
        if inc is None:
            out.append(f"{cid}:incremental_terms_missing")
            continue
        if inc == NO_SEARCH_STATE:
            continue
        for k, v in inc.items():
            if k in rec["final_state_terms"] and abs(float(v) - float(rec["final_state_terms"][k])) > TOL_TERMS:
                out.append(f"{cid}:incremental:{k}")
    return out


def _read_oracle_csv(path):
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def _e09(fid, fam, tables, res, written, oracle_ref):
    import csv  # noqa: F401
    fails, info = [], {}
    n1, n2 = len(tables["parts1"]), len(tables["parts2"])
    info["enumerated_pairs"] = n1 * n2
    if n1 * n2 != fam["properties"]["mapping_pairs"] or tables["I12"].shape != (n1, n2):
        fails.append("enumeration_coverage")
    ref_dir, ref_sha = oracle_ref
    if ref_dir is None:                                        # no reference table: E09 cannot pass vacuously
        return {"pass": False, "failures": ["no_reference_oracle_tables"], **info}
    cmp = {}
    for kind in ("partitions_r1", "partitions_r2", "pair_I12"):
        name = f"{fid}_{kind}.csv"
        src = Path(ref_dir) / name
        if ref_sha is not None and _file_sha(src) != ref_sha.get(name):
            fails.append(f"source_table_hash:{name}")
            continue
        h0, r0 = _read_oracle_csv(src)
        h1, r1 = _read_oracle_csv(written[name])
        if h0 != h1 or len(r0) != len(r1):
            fails.append(f"table_shape:{name}")
            continue
        num = [j for j, c in enumerate(h0) if c not in ("index", "index1", "index2", "labels", "tokens_per_class")]
        worst = 0.0
        for a, b in zip(r0, r1):
            for j, (x, y) in enumerate(zip(a, b)):
                if j in num:
                    worst = max(worst, abs(float(x) - float(y)))
                elif x != y:
                    fails.append(f"table_entry:{name}")
                    break
        if worst > TOL_ORACLE:
            fails.append(f"table_values:{name}:{worst}")
        cmp[name] = {"max_abs_diff": worst, "byte_identical": _file_sha(src) == _file_sha(written[name])}
    info["source_tables"] = cmp
    arms_src = Path(ref_dir) / f"{fid}_arms.csv"
    if arms_src.exists():
        h0, r0 = _read_oracle_csv(arms_src)
        h1, r1 = _read_oracle_csv(written[f"{fid}_arms.csv"])
        d0 = {r[0]: r for r in r0}
        same = sum(1 for r in r1 if d0.get(r[0]) == r)
        info["arms_table_descriptive"] = {"rows_new": len(r1), "rows_source": len(r0), "rows_identical": same}
    labels = res["checks"]["C6_HEURISTIC_LABELLING"]["labels"]
    bad = [c for c, v in labels.items() if v.get("label") not in ("EXHAUSTIVE_OPTIMAL", "HEURISTIC", "NOT_A_SEARCH")]
    if bad:
        fails.append(f"unlabelled:{bad}")
    info["heuristic_gaps"] = {c: v.get("gap", v.get("gap_per_recipient")) for c, v in labels.items()
                              if v.get("label") == "HEURISTIC"}
    info["label_counts"] = {k: sum(1 for v in labels.values() if v.get("label") == k)
                            for k in ("EXHAUSTIVE_OPTIMAL", "HEURISTIC", "NOT_A_SEARCH")}
    info["note"] = "a labelled HEURISTIC gap is reported, not a failure"
    return {"pass": not fails, "failures": fails, **info}


def _e10(X, arms, tables, U):
    from lra import run as R
    fails = []
    cl = arms.get(R.d0_id("CLASS"))
    if cl is None:
        return {"pass": False, "failures": ["CLASS release missing"]}
    i12c = cl["metrics"]["I12"]
    direct = plugin_mi(X["s"], X["d"][1], X["d"][2])
    if abs(direct - i12c) > TOL_FLOOR:
        fails.append("I12(CLASS) != I(SEX; d1, d2)")
    worst = np.inf
    for cid, a in arms.items():
        m = a["metrics"]
        worst = min(worst, m["I12"] - i12c)
        if m["I12"] < i12c - TOL_FLOOR:
            fails.append(f"{cid}:I12_below_CLASS")
        for i in (1, 2):
            if m[f"I{i}"] < cl["metrics"][f"I{i}"] - TOL_FLOOR:
                fails.append(f"{cid}:I{i}_below_CLASS")
    tmin = float(np.min(tables["I12"]))
    if tmin < i12c - TOL_FLOOR:
        fails.append("enumerated_pair_below_CLASS")
    cd1 = arms.get(R.d0_id("CLASS") + "|D1")
    afford = {"CLASS|D1_feasible": bool(cd1["feasible"]) if cd1 else None,
              "CLASS|D1_slack": ({f"L{i}": U[i][0] + BUDGET["ll"] - cd1["metrics"][f"L{i}"] for i in (1, 2)}
                                 | {f"B{i}": U[i][1] + BUDGET["brier"] - cd1["metrics"][f"B{i}"] for i in (1, 2)})
              if cd1 else None,
              "note": "recorded, never a failure: an affordable CLASS comparator can make privacy superiority "
                      "impossible without making the implementation wrong"}
    return {"pass": not fails, "failures": fails, "I12_CLASS": i12c, "I12_decisions_direct": direct,
            "min_release_I12_minus_CLASS": float(worst), "min_enumerated_I12": tmin, "affordability": afford}


def engineering_fixture_checks(fid, fam, res, tables, arms, laws_ok, written, oracle_ref):
    X, U = tables["X"], tables["U"]
    r678 = _guard(_e06_e07_e08, fid, X, U, arms, tables["mrecs"], tables["mfiles"], tables["bank"])
    e6, e7, e8 = (r678, r678, r678) if isinstance(r678, dict) else r678
    out = {"E01_LAW_COUNTS_ROUTING_HASHES": _guard(_e01, fam, laws_ok, X, arms, res),
           "E02_FIXED_TOKEN_INFORMATION": _guard(_e02, X, arms, res),
           "E03_CALIBRATED_NULL": _guard(_e03, fid, res),
           "E04_D1_FINAL_VECTOR_CERTIFICATES": _guard(_e04, arms, tables),
           "E05_LOSS_RECONSTRUCTION": _guard(_e05, X, arms, res, tables["mrecs"]),
           "E06_ACCEPTED_STATE_BUDGETS": e6, "E07_SEQUENTIAL_TEMPORARY_PARTNER": e7,
           "E08_INCREMENTAL_REPLAY": e8,
           "E09_EXHAUSTIVE_ORACLE": _guard(_e09, fid, fam, tables, res, written, oracle_ref),
           "E10_DECISION_DISCLOSURE_FLOOR": _guard(_e10, X, arms, tables, U)}
    # the predecessor's budget-enforcement check C3 is part of E06 (final releases)
    c3 = res["checks"]["C3_BUDGET_ENFORCEMENT"]
    if not c3["pass"]:
        out["E06_ACCEPTED_STATE_BUDGETS"] = {**out["E06_ACCEPTED_STATE_BUDGETS"], "pass": False,
                                             "failures": out["E06_ACCEPTED_STATE_BUDGETS"]["failures"] + c3["failures"]}
    if not res["checks"]["C4_DECISION_PRESERVATION"]["pass"]:
        out["E01_LAW_COUNTS_ROUTING_HASHES"] = {**out["E01_LAW_COUNTS_ROUTING_HASHES"], "pass": False,
                                                "failures": out["E01_LAW_COUNTS_ROUTING_HASHES"]["failures"]
                                                + res["checks"]["C4_DECISION_PRESERVATION"]["failures"]}
    return out


# ------------------------------------------------------------------ E11 / E12: registered pytest node IDs
def _findings(pkg):
    """(findings list normalised, failures) from REVIEW_FINDINGS_DISPOSITION.json (role D)."""
    p = Path(pkg) / FINDINGS_FILE
    if not p.exists():
        return [], [f"{FINDINGS_FILE} missing"]
    z = json.loads(p.read_text())
    items = z.get("findings", z) if isinstance(z, dict) else z
    out, fails = [], []
    for f in items if isinstance(items, list) else []:
        tests = None
        for k in ("regression_tests", "regression_test", "tests", "test", "node_ids"):
            if k in f:
                tests = f[k]
                break
        tests = [tests] if isinstance(tests, str) else list(tests or [])
        tests = [t.split(" ")[0] for t in tests if isinstance(t, str) and t.strip()]
        ordv = f.get("source_ordinal", f.get("ordinal", f.get("id")))
        disp = f.get("disposition", f.get("status", f.get("resolution_status")))
        out.append({"ordinal": ordv, "disposition": disp, "tests": tests,
                    "affects_required_adult_execution": f.get("affects_required_adult_execution")})
    ords = sorted(str(f["ordinal"]) for f in out)
    if len(out) != N_FINDINGS or ords != sorted(str(i) for i in range(1, N_FINDINGS + 1)):
        fails.append(f"expected ordinals 1..{N_FINDINGS}, got {ords}")
    for f in out:
        d = str(f["disposition"] or "")
        resolved = d in RESOLVED_DISPOSITIONS
        if not resolved:
            fails.append(f"finding {f['ordinal']}: disposition {f['disposition']!r} is not one of {RESOLVED_DISPOSITIONS}")
            if f["affects_required_adult_execution"] is not False:   # an unresolved defect that can touch Adult execution
                fails.append(f"finding {f['ordinal']}: unresolved and affects (or may affect) required Adult execution")
        if not f["tests"]:
            fails.append(f"finding {f['ordinal']}: no regression test node id")
    return out, fails


def run_pytest_nodes(nodes, workdir):
    """Run pytest node IDs in ONE subprocess (single thread; the caller holds the semaphore slot). Returns {node:
    {"pass", "cases": [...]}} and the raw summary. A node passes iff it collected >= 1 case and every case passed."""
    import os
    import subprocess
    import xml.etree.ElementTree as ET
    nodes = sorted(set(nodes))
    out = {n: {"pass": False, "cases": []} for n in nodes}
    if not nodes:
        return out, {"returncode": None, "ran": 0}
    Path(workdir).mkdir(parents=True, exist_ok=True)
    jx = Path(workdir) / "wiring_junit.xml"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "PYTHONPATH": str(WT)}
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={jx}", *nodes]
    r = subprocess.run(cmd, cwd=str(WT), env=env, capture_output=True, text=True, timeout=3600)
    cases = []
    if jx.exists():
        for tc in ET.parse(jx).getroot().iter("testcase"):
            path = tc.get("classname", "").replace(".", "/") + ".py"
            name = tc.get("name", "")
            bad = any(ch.tag in ("failure", "error") for ch in tc)
            skip = any(ch.tag == "skipped" for ch in tc)
            cases.append({"file": path, "name": name, "ok": not bad and not skip, "skipped": skip})
    for n in nodes:
        f, _, t = n.partition("::")
        hit = [c for c in cases if c["file"] == f and (not t or c["name"] == t or c["name"].startswith(t + "["))]
        out[n] = {"pass": bool(hit) and all(c["ok"] for c in hit), "cases": [f"{c['file']}::{c['name']}:"
                                                                          f"{'ok' if c['ok'] else 'FAIL'}" for c in hit]}
    tail = (r.stdout or "").strip().splitlines()[-1:] if r.stdout else []
    return out, {"returncode": r.returncode, "ran": len(cases), "summary": tail}


def _e11_e12(pkg, runner, workdir, rule_ok):
    findings, ffails = _findings(pkg)
    nodes = [n for v in WIRING_TESTS.values() for n in v] + [t for f in findings for t in f["tests"]]
    res, summ = runner(nodes, workdir)
    e11f = list(rule_ok)
    groups = {}
    for g, ns in WIRING_TESTS.items():
        groups[g] = {n: res.get(n, {"pass": False}).get("pass") for n in ns}
        if not ns:
            e11f.append(f"no registered node id for: {g}")
        e11f += [f"{g}: {n} did not pass" for n, ok in groups[g].items() if not ok]
    e12f = list(ffails)
    per = []
    for f in findings:
        ok = {t: res.get(t, {"pass": False}).get("pass") for t in f["tests"]}
        per.append({**f, "test_results": ok})
        e12f += [f"finding {f['ordinal']}: {t} did not pass" for t, v in ok.items() if not v]
    return ({"pass": not e11f, "failures": e11f, "groups": groups, "pytest": summ,
             "node_results": {n: res[n] for g in WIRING_TESTS.values() for n in g if n in res}},
            {"pass": not e12f, "failures": e12f, "findings": per, "file_sha256": _file_sha(Path(pkg) / FINDINGS_FILE)})


# ------------------------------------------------------------------ outputs
def _write_tables(out_dir, fid, tables, res, sub=ORACLE_DIR):
    import csv
    d = Path(out_dir) / sub
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
    return {p.name: p for p in sorted(d.glob(f"{fid}_*.csv"))}


def _save_unit(fid, laws_sha, res, eng, tables, arms):
    """Private unit cor__<FID>: everything the independent verifier needs to replay the mapper (records, traces,
    policies, decoders, releases, the D0 starts and the fine partitions)."""
    from lra import run as R
    rel = {}
    for cid, a in arms.items():
        for k in ("tok1", "q1", "hard1", "tok2", "q2", "hard2"):
            rel[f"{R.safe(cid)}__{k}"] = np.asarray(a["rel"][k])
    files = {"result.json": res, "engineering_checks.json": eng,
             "mapper_records.json": tables["mrecs"],
             "mapper_traces.json": {cid: f.get("trace.json") for cid, f in tables["mfiles"].items()},
             "policies.json": {cid: a["pair"].to_dict() for cid, a in arms.items()},
             "decoders.json": {cid: a["decoder_body"] for cid, a in arms.items() if "decoder_body" in a},
             "d0_bank_policies.json": {cid: p.to_dict() for cid, p in tables["bank"].items()},
             "fine.json": tables["X"]["fine_dict"], "releases.npz": rel}
    R.save(f"cor__{fid}", files, {"fixture": fid, "laws_sha256": laws_sha,
                                  "engineering_checks": {k: bool(v.get("pass")) for k, v in eng.items()},
                                  "n_mapper_units": len(tables["mrecs"]), "n_releases": len(arms)})


def stage_correctness(D, shard_spec=None, out=None, laws_path=None, save_units=True, families=None, run_tests=None,
                      expect_laws=None, oracle_ref=None, pkg=None):
    """The registered correctness stage (CORRECTNESS_LOCK; prompt sec. 9). D must be None and there is no shard: one
    process runs the four pinned laws, the twelve mandatory checks and writes ENGINEERING_GATE_RESULT.json,
    correctness_oracle/*.csv and the private units cor__<FID>. Test-only keywords (toy laws): laws_path, expect_laws
    = (file_sha256, laws_sha256), oracle_ref = (dir, sha map or None), run_tests, pkg, out."""
    import time
    if D is not None:
        raise ValueError("REFUSED: the correctness stage loads no real data (D must be None)")
    if shard_spec:
        raise ValueError("REFUSED: the correctness stage runs in one process (no shard)")
    t0, c0 = time.perf_counter(), time.process_time()
    pkg = Path(pkg) if pkg else PKG
    out_dir = Path(out) if out else pkg
    registered = laws_path is None
    lp = Path(laws_path) if laws_path else pkg / "FIXTURE_LAWS.json"
    laws = load_laws(lp)
    rule_path = pkg / ENG_RULE_FILE
    rule_ok = []
    if registered:
        from lra import lock as LK
        docs = LK.latest()["documents_sha256"]
        for d in ("FIXTURE_LAWS.json", ENG_RULE_FILE, SOURCE_GATE_FILE, FINDINGS_FILE):
            if docs.get(d) != _file_sha(pkg / d):
                raise ValueError(f"REFUSED: {d} differs from the latest lock's documents_sha256 (or is missing)")
        expect_laws = (PINNED_LAWS_FILE_SHA256, PINNED_LAWS_SHA256)
        oracle_ref = (SOURCE_ORACLE_DIR, SOURCE_ORACLE_SHA256)
    if not rule_path.exists():
        rule_ok.append(f"{ENG_RULE_FILE} missing")
    else:
        rule = json.loads(rule_path.read_text())
        if rule != json.loads(json.dumps(engineering_gate_rule())):
            rule_ok.append(f"{ENG_RULE_FILE} differs from engineering_gate_rule() (rule text not wired to code)")
    laws_ok = []
    if expect_laws is not None:
        if _file_sha(lp) != expect_laws[0]:
            laws_ok.append("laws_file_sha256")
        if laws["laws_sha256"] != expect_laws[1]:
            laws_ok.append("laws_sha256")
    if registered:
        if _file_sha(pkg / SOURCE_GATE_FILE) != PINNED_SOURCE_GATE_SHA256:
            laws_ok.append("source_gate_sha256")
        if [f["id"] for f in laws["families"]] != list(FIXTURE_IDS):
            laws_ok.append("family_ids")
    fams = laws["families"] if families is None else [f for f in laws["families"] if f["id"] in families]
    expected = [f["id"] for f in laws["families"]] if registered or families is None else list(families)
    per_fix, desc, errors, hashes = {}, {}, {}, {}
    for fam in fams:
        fid = fam["id"]
        try:
            res, tables, arms = run_fixture(fam)
            written = _write_tables(out_dir, fid, tables, res)
            hashes.update({k: _file_sha(v) for k, v in written.items()})
            eng = engineering_fixture_checks(fid, fam, res, tables, arms, laws_ok, written,
                                             oracle_ref or (None, None))
            per_fix[fid] = eng
            desc[fid] = {"descriptive_only": True, "old_trigger": res["trigger"],
                         "predecessor_checks_C1_C7": {k: v["pass"] for k, v in res["checks"].items()},
                         "mapper_status": res["mapper_status"], "U": res["U"]}
            if save_units:
                from lra import run as R
                _save_unit(fid, laws["laws_sha256"], R._finite(res), R._finite(eng), tables, arms)
        except Exception as e:                                  # a crashed fixture blocks; it is never skipped
            errors[fid] = f"{e.__class__.__name__}: {e}"
    for fid in expected:
        if fid not in per_fix and fid not in errors:
            errors[fid] = "not run"
    checks = {}
    ids = list(engineering_checks_text())
    for cid in ids[:10]:
        rows = {fid: per_fix[fid][cid] for fid in per_fix}
        ok = bool(rows) and all(r.get("pass") is True for r in rows.values()) and not errors
        checks[cid] = {"pass": ok, "per_fixture": rows}
    from lra import run as R
    runner = run_tests or run_pytest_nodes
    e11, e12 = _e11_e12(pkg, runner, R.RUN / "correctness", rule_ok)
    checks["E11_LAUNCH_WIRING"], checks["E12_REVIEW_FINDINGS"] = e11, e12
    verdict, reasons = engineering_verdict(checks, expected, errors)
    from lra import decoder as DC
    body = {"schema": ENG_RESULT_SCHEMA, "verdict": verdict, "reasons": reasons,
            "verdict_rule": "ENGINEERING_READY iff every mandatory check E01..E12 passes",
            "checks": {k: checks[k] for k in ids},
            "check_pass": {k: bool(checks[k]["pass"]) for k in ids},
            "gate_rule_sha256": _file_sha(rule_path), "laws_file_sha256": _file_sha(lp),
            "laws_sha256": laws["laws_sha256"], "fixtures": expected, "fixture_errors": errors,
            "source_gate": {"file": SOURCE_GATE_FILE, "sha256": _file_sha(pkg / SOURCE_GATE_FILE),
                            "verdict": "GATE_NOT_MET", "role": "historical; never this study's verdict"},
            "not_blocking": NOT_BLOCKING,
            "descriptive": desc, "oracle_table_sha256": hashes, "decoder_tolerances": DC.TOLERANCES,
            "synthetic_only": True, "registered_run": registered,
            "wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0,
            "scope": "known, already-opened fixture laws; a correctness verdict, not mechanism evidence; fixture MI is "
                     "exact law MI of finite laws, not a population guarantee; no Adult claim follows"}
    (out_dir / ENG_RESULT_FILE).write_text(public_text(json.dumps(R._finite(body), indent=1, allow_nan=False)) + "\n")
    return body


def public_text(txt):
    """Public files carry no private paths or user names: the private store, the worktree and the home directory
    are replaced by placeholders (exception texts can embed absolute paths)."""
    from lra import run as R
    for real, ph in ((str(R.PRIV.parent), "<PRIVATE_CACHE>"), (str(WT), "<WORKTREE>"), (str(Path.home()), "~")):
        txt = txt.replace(real, ph)
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lra.fixtures")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("laws")
    s2 = sub.add_parser("rule")
    s2.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "laws":
        b = load_laws()
        print(json.dumps({"ok": True, "laws_sha256": b["laws_sha256"], "file_sha256": _file_sha(PKG / "FIXTURE_LAWS.json"),
                          "pinned": b["laws_sha256"] == PINNED_LAWS_SHA256 and
                          _file_sha(PKG / "FIXTURE_LAWS.json") == PINNED_LAWS_FILE_SHA256,
                          "families": [f["id"] for f in b["families"]]}))
    else:
        if a.write:
            print(write_engineering_rule())
        else:
            print(json.dumps(engineering_gate_rule(), indent=1))


if __name__ == "__main__":
    main()
