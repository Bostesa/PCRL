# Correction pass: pre-registered exact checks (written before any check code was run)

Written by the coordinator on 2026-10-16 at about 12:45Z, during the correction pass on the mechanism-design campaign. No check code had been written or run when the laws and expected outcomes below were fixed. The hand-computed values were worked out on paper from the definitions in this file.

**Purpose.** These are small correctness checks for the corrected statements in `../CORRECTIONS.md`. They are not a mechanism campaign: no parameter is swept, nothing is selected, no synthetic benchmark is built, and no real-person data is involved. Each law is a finite joint law written in exact fractions.

**Definitions used by every check.**
- A discrete view V has joint law P(V = v, S = s). Its optimal (Bayes) AUC is
  AUC\*(S; V) = Σ_{v,w} P(v, 1) P(w, 0) [1{P(v,1)P(w,0) > P(w,1)P(v,0)} + ½·1{P(v,1)P(w,0) = P(w,1)P(v,0)}] / (P(S=1) P(S=0)).
  This compares likelihood ratios by cross-multiplication, counts ties as ½ (including the pairs inside one value of V), and equals the area under the optimal ROC curve.
- A fixed score h(V) has AUC = Σ_{v,w} P(v,1)P(w,0)[1{h(v) > h(w)} + ½·1{h(v) = h(w)}] / (P(S=1)P(S=0)).
- Task metrics follow `PROBLEM_SPECIFICATION.md` §4 in expectation under the law: accuracy of the first-index argmax, summed Brier Σ_k (p_k − 1[y = k])², and NLL −log p_y (in nats; no floor is needed because every predicted probability here is positive).
- All quantities are exact `Fraction`s except NLL and mutual information, which need logarithms. Equality claims for NLL are established by exact equality of the predicted distributions in every cell.

## A. An injective release preserves the optimal AUC (and only the optimal AUC)

- Cells x ∈ {0, 1, 2, 3}, with P(X = x) = (1/8, 3/8, 1/4, 1/4) and π(x) = P(S = 1 | x) = (1/4, 1/3, 2/3, 3/4).
- A binary task with η(x) = P(Y = 1 | x) = (2/5, 4/5, 1/5, 3/5). The release is R(x) = (1 − η(x), η(x)) with its argmax: four distinct vectors, so R is injective.
- **A1.** AUC\*(S; R) = AUC\*(S; X) exactly. Hand value: 3299/4606 ≈ 0.71624.
- **A2.** A reader that uses the released probability η as its score gets 1763/4606 ≈ 0.38276. Its reversal gets 2843/4606 ≈ 0.61724. Both are strictly below AUC\*. Equality holds for the unrestricted optimal attacker, not for a restricted reader.

## B. A deterministic, utility-preserving release that removes all sensitive information

- T ∈ {0, 1, 2} with P(T) = (1/4, 1/2, 1/4). Z ∈ {0, 1} with P(Z = 1) = 2/5. T and Z are independent. **S = Z.** The input is X = (T, Z): six cells.
- Two tasks depend only on T, as two recipients would have them:
  - Y1 ∈ {0, 1}: P(Y1 = 1 | T) = (1/5, 3/5, 9/10).
  - Y2 ∈ {0, 1, 2}: P(Y2 | T = 0) = (7/10, 1/5, 1/10); P(Y2 | T = 1) = (1/5, 3/5, 1/5); P(Y2 | T = 2) = (1/10, 3/10, 3/5).
  - P(t, z, y1, y2) = P(t) P(z) P(y1 | t) P(y2 | t).
- The release is R_i(t, z) = P(Y_i | T = t) with its argmax, for i = 1, 2. It is a deterministic many-to-one probability release: six inputs, three distinct release pairs. It fits the existing output contract (a deterministic map of the inputs into each task's probability simplex, plus its argmax).
- **B1.** In every cell, R_i(x) equals the full-input Bayes predictor P(Y_i | X = x) exactly, for both tasks. Accuracy, summed Brier and NLL therefore equal the full-input Bayes values. Hand values: income accuracy 29/40, summed Brier 73/200; occupation accuracy 5/8, summed Brier 53/100.
- **B2.** AUC\*(S; X) = 1.
- **B3.** AUC\*(S; R_1) = AUC\*(S; R_2) = AUC\*(S; (R_1, R_2)) = 1/2.
- **B4.** Distinct release pairs: 3 for 6 inputs. I(S; X) = h(2/5) ≈ 0.97095 bits; I(S; (R_1, R_2)) = 0.
- **Stated, not computed.** Because the release is the Bayes predictor under the law, no predictor of X has lower expected summed Brier or NLL, or higher accuracy. So its clause excess against any reference predictor (such as analogues of U0 or Ucal) is ≤ 0.
- **Scope.** This refutes the general claim that a deterministic release under this output contract cannot reduce optimal sensitive recovery below the raw-field level without utility loss. It does not show that the Florida data have this structure, or that MMP or any other candidate succeeds there.

## C. Sample uniqueness cannot certify population injectivity

**C1 (deterministic).**
- Declared support {0, …, 5}, each cell with probability 1/6. π = (1/10, 1/2, 3/10, 7/10, 1/5, 9/10).
- A piecewise-constant release f: f(0) = f(1) = 1/4, f(2) = f(3) = 1/2, f(4) = f(5) = 3/4 (released as the binary vector (1 − f, f)).
- The observed inputs are O = {0, 2, 4}.
- **C1a.** On O, f has 3 distinct outputs for 3 distinct inputs: it passes a sample-uniqueness check. The identity map passes the same check.
- **C1b.** On the declared support, f is 2-to-1: 3 distinct outputs for 6 cells.
- **C1c.** AUC\*(S; f(X)) < AUC\*(S; X) under the population law. Hand values: 1091/1782 ≈ 0.61223 against 1461/1782 ≈ 0.81987.

**C2 (a map that merges every cell, and a random sample).**
- N = 2M equally likely cells, merged in M disjoint pairs by the release. An i.i.d. sample of size n shows no collision exactly when no pair has both members drawn. Its probability is
  P(M, n) = Σ_{k=0}^{M} C(M, k) (−1)^{M−k} 2^k k^n / (2M)^n
  (exponential generating function (2e^x − 1)^M; 0^0 = 1).
- **C2a.** The formula equals brute-force enumeration of all (2M)^n samples exactly at (M, n) = (3, 4) and (2, 5).
- **C2b.** Approximate values follow from exp(−n(n − 1)/(4M)): P(1000, 20) in [0.88, 0.94]; P(1000, 60) in [0.35, 0.47]; P(1000, 200) < 10⁻³.
- **Reading.** A map that is 2-to-1 on its whole support passes a sample-uniqueness check with high probability when the sample is small relative to the support. Collision checks on a sample can rule out frequent merges but not rare ones. They also say nothing about cells outside the sample's distribution.

## D. A non-injective release can keep all the sensitive information

- Cells x ∈ {0, 1, 2, 3}, each with probability 1/4. π = (1/5, 4/5, 1/5, 4/5).
- The release is η(x) = (1/3, 2/3, 1/3, 2/3), which merges {0, 2} and {1, 3}.
- **D1.** The release is 2-to-1.
- **D2.** AUC\*(S; R) = AUC\*(S; X) = 4/5 exactly. Merging cells with the same sensitive posterior loses nothing, so merging inputs is not by itself protection.

## E. The data-processing bound and its equality condition, exhaustively on a small law

- Five cells, with P(X) = (1/10, 1/5, 3/10, 1/4, 3/20) and π = (1/5, 2/5, 2/5, 3/5, 4/5). Cells 1 and 2 share a posterior.
- Every deterministic release of X is equivalent, for AUC\*, to a set partition of the cells (release = block label). All 52 partitions are enumerated.
- **E1.** AUC\*(S; block) ≤ AUC\*(S; X) for all 52.
- **E2.** Equality holds exactly for the partitions whose blocks each have a constant π: the finest partition and {{0}, {1, 2}, {3}, {4}}. That is 2 partitions; the first is injective and the second is not.
- **E3.** The other 50 are strictly below.

## Run plan and independence

- **Coordinator.** `mechanism_design/correction/exact_checks.py` writes `corrections/EXACT_CHECKS.json`, run once as queue unit `CORR_EXACT_CHECKS_V1` through `mechanism_design/correction/cqueue.py`.
- **Independent verifier.** It re-enters the laws from this file by hand, writes its own code under `mechanism_design_independent/correction_v1/`, and runs it as its own queue unit. It does not import the coordinator's correction module or the campaign's `finite_law.py` metric functions.
- **Laws are not changed after either run.** If a check fails, it is reported as failed, with the cause.
