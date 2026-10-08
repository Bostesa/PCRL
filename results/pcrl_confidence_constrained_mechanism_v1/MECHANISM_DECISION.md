# Mechanism decision: NO-GO (roles A, B, D; Stage E; written after the locked Stage D run, before any real-data fit)

**Decision.** The prototype is not built.

**Registered go rule** (PROTOCOL.md §5, frozen at FEASIBILITY_LOCK, pushed at d5337e5 before any real-array run): go iff,
under the pointwise guard G, every seed and both recipients have F3 ≥ 0.95 and F4 ≤ 0.05. It fails in all 6 of 6
cells:
- every coverage failure is intrinsic, so no admissible code at the registered capacity can do better;
- every cell is fallback-dominated.

Under the prompt's own instruction ("If the required constraint makes it essentially an identity release, say so and do
not launch a full experiment"), the answer is:
- **occupation:** yes, it is essentially an identity release;
- **income:** it is fallback-dominated at the registered capacity.

No PILOT_LOCK was written, no attacker was fitted, no SEX or task label was read, and INNER_SELECTION was not scored.

## 1. Selected mechanism and why it would change information

**Selected.** The confidence-constrained score channel of MECHANISM_TABLE.md, column (1). Each recipient receives one
of at most 8 (income) or 64 (occupation) tokens per predicted class. Each token carries a representative probability
vector q that satisfies G for every member, together with a disclosed fallback.

**How it would change information.** Many teacher outputs map to one token, so the distinctions inside a token are
withheld.

**What G allows.** Any admissible bin has total-variation diameter ≤ e^d − 1 ≈ 0.005 (MATH_REVIEW.md R5). The only
distinctions G permits the channel to withhold are therefore score differences of about half a percentage point.

## 2. What confidence is guaranteed or only measured

**Guaranteed** by certification of every member with the canonical predicate (UTILITY_CONTRACT.md §6), pointwise for
every input and every label, relative to Ucal only:
- log-loss excess ≤ 0.005;
- all-label Brier excess ≤ 0.0025;
- strict source decision.

Unseen inputs are covered by the fallback: Ucal itself, or the η-nudge at an exact tie (none occurred: 0 tied tops in
fitting or held-out rows on every seed). In the locked run, n_released_failing_contract was 0 in every cell.

**Not guaranteed:**
- calibration;
- task correctness;
- utility relative to U0;
- any privacy property.

**Not measured:** privacy. No label was read.

## 3. Locked Stage D result (FEASIBILITY_GEOMETRY.json; PRIMARY_ENDPOINTS.csv)

| Recipient | Seed | F3 coverage at capacity | Bound on any admissible code | F4 held-out fallback | Bins for all fitting rows (F1) | Lower bound on bins (F2) |
|---|---|---|---|---|---|---|
| income (K = 2, 8/class) | 0 | 0.3878 | exact F3x = 0.3878 | 0.6085 | 281 | 194 |
| income | 1 | 0.3704 | exact 0.3704 | 0.6300 | 281 | 193 |
| income | 2 | 0.3753 | exact 0.3753 | 0.6160 | 280 | 194 |
| occupation (K = 6, 64/class) | 0 | 0.1745 | ≤ 0.3660 | 0.8510 | 10,372 | 10,105 |
| occupation | 1 | 0.1791 | ≤ 0.3594 | 0.8485 | 10,430 | 10,207 |
| occupation | 2 | 0.1738 | ≤ 0.3573 | 0.8500 | 10,457 | 10,240 |

All counts are out of 15,434 fitting rows; F4 is measured on 2,000 held-out rows.

**What these numbers mean:**
- **Occupation.** At least 10,105 of 15,434 people need a bin of their own. With 384 tokens no admissible code can
  cover more than 36.6% of fitting rows, and 85% of new inputs receive their raw calibrated scores. This is an
  identity release in all but name.
- **Income.** The best possible code at the registered 16 tokens covers 37–39% of fitting rows, which is an exact
  optimum. 61–63% of new inputs receive their raw calibrated scores.
- **Decision floor.**
  - The decision-only release is confidence-ineligible: it has no vector.
  - CLASS (one token per class) is G-infeasible for every class on every seed: Σ_k max p_k over the class is about
    1.50 for income and 1.96–2.69 for occupation, against the bound e^d = 1.005. Occupation has five predicted
    classes in the fitting rows.

## 4. What prior work already covers

PRIOR_WORK_AND_NOVELTY.md:
- G is Liao et al.'s hard distortion ball.
- log F2 ≤ (maximal leakage about p) ≤ log F1.
- Two colluding recipients is Taylor et al. with m = 2.
- Greedy privacy-funnel merging, LP mechanisms and argmax-preserving confidence release all exist.

The near-identity outcome agrees with the hard-distortion literature and is not novel. What this sprint adds is a
verified, label-free measurement of how restrictive the guard is on these frozen teachers.

## 5. Strongest matched control

Under the registered contract every eligible arm shares the same admissible support and the same fallback. The
strongest SEX-blind control is task-only minimal cover at the same capacity. It inherits the same 61–63% (income) and
85% (occupation) fallback.

**Smooth-law bound (MATH_REVIEW.md R6).** If P(SEX | p) is L-Lipschitz in total variation, every admissible arm,
deterministic or randomised, is within L·0.005 Bayes accuracy of releasing p itself.
- A 0.02 change in Bayes accuracy would need L ≥ 4, and then only on the covered rows; most rows fall back.
- The prospective criterion (pair AUC benefit ≥ 0.02 with the 0.01 local guard) is an AUC criterion. This bound is
  stated in Bayes accuracy and is not converted to AUC here.

## 6. Remaining uncertainty

- **Privacy on Adult** was never measured. The smooth-law bound is an expectation, not a measurement, and a
  non-smooth law can lose information inside one cell (R6). In-sample plug-in "protection" from tiny bins would not
  be population evidence.
- **The teacher-expected guard G_exp** (secondary diagnostic, never a pass) leaves room for income but not for
  occupation:
  - income: F3 0.954–0.961 with 16 tokens, F4 0.037–0.049;
  - occupation: F3 0.567–0.595 (constructive), bound ≤ 0.96, F4 0.44–0.48.

  G_exp is weaker and calibration-dependent. Whether privacy-aware binning beats plain compression under it is
  unknown.
- **Capacity and contract constants were not searched**, as the protocol requires. The result is specific to G(0.005,
  0.0025) at i8o64 on these teachers.

## 7. Reason to stop

The registered feasibility premise failed intrinsically in every cell. The finite-law oracles show that the code can
find privacy-aware partitions when a law permits them: on POSITIVE, local 0.0642 against task-only 0.1285 nats; on
JOINT_HEADROOM, joint 0.1853 against sequential 0.3230. But the Adult geometry gives the mechanism almost nothing to
operate on.

**Exact architecture of what would have been built.** None was built. The specification is frozen in PROTOCOL.md
§3–§6 and ccm/guard.py, ccm/geometry.py:
- certified greedy cover (K = 6) or exact interval DP (K = 2);
- first-certifying-representative serving rule;
- disclosed Ucal fallback.
