# Screen review (statistical reviewer; question C)

**Status.** Written 2026-10-18, about 19:55-20:05Z.
- It comes after the reader investigator's screen study (`screen/SCREEN_STUDY.json`; screen_study.py sha 4cfa4a10…, cts/rules.py
  f2ceae96…, RULES.json 5393b143…) and **before any new method outcome was scored by readers**.
- This is a read-only review. Nothing was rerun, and **no frozen rule is changed**.
- All rates hold under the stated laws only, on adaptively used MODEL_SELECT data with the registered reader slate.
- A screen pass is permission for further development, not evidence of protection.

## 1. Do the simulations match the registered definitions?

**Yes.** Both screens are imported unchanged from `cts.rules`.
- The φ_prior values recomputed from the frozen grid equal the registered gross values for all three views (income 5.6365,
  occupation 0.7782, pair 1.0867).
- The L1 components (net sd_A, sd_k, rms cell SE) recomputed equal the values used.
- The old gate reproduces the source's recipient failures for all 18 two-seed configurations.

**Labelled investigator additions** (not part of the registered laws):
- extra reader-seed slots and P2A_PAIR_B80 in L0b;
- the `h3_actual` multiplicity structure;
- the A1_PO_16 base in L2;
- a sign search in the η calibration.

**Simplifications of L1 to keep in mind.**
- Income and occupation views are drawn independently, although they share household splits.
- Cell SEs are fixed rather than estimated.
- φ_prior comes from the same grid components that define L1. AGG1's boundary behaviour under L1 is therefore close to α partly by
  construction. L0b is the empirical check.

## 2. What the two screens do

| Law (what is true) | Old per-cell gate | AGG1 |
|---|---|---|
| L0a exact aliases (380 income / 28 occupation ordered pairs; every cell exactly 0) | passes 100% | passes 100% (`EXACT_ALIAS`) |
| L0b same release, separately fitted readers (true 0; reader-fitting and winner noise only) | income 18/18; **occupation 21/24; pair 21/30** (single cells up to +0.013 / +0.023) | income 18/18; occupation 24/24; pair 29/30 |
| L1, single comparison at the allowance boundary, mu = +0.010 (12 settings) | **0.002-0.137**: above 0.05 when seeds share splits (income 0.137, occupation 0.070 at rho 0.76, n_R = 2) | **0.004-0.033** in every setting |
| L1, truly equal, mu = 0, income view, n_R = 2 (rho 0 / 0.76) | 0.19 / 0.31 | 0.08 / 0.12 |
| L1, truly safer, mu = -0.010, income view: n_R = 2 / n_R = 3 (rho 0.76) | 0.55 / 0.40 (more cells hurt) | 0.31 / 0.43 (a third repetition helps) |
| Multiplicity, 12 comparisons, everything truly 0.010 safer (rho 0.76, n_R = 2; actual H3 structure) | 0.108; 0.150 | 0.032; 0.045 |
| Multiplicity IUT error: one comparison +0.010, the rest -0.010 (+0.020 case) | 0.049 (0.019) | 0.008 (0.002) |
| L2 semi-synthetic shift (12 rows, one realization each) | all FAIL | all FAIL |
| Frozen source configurations (descriptive what-if) | 18 of 18 fail | 18 of 18 fail; NO_QUALIFYING_CONFIGURATION unchanged |

**Reading the table.**
- *Old gate at the boundary.* It is not a conservative screen when the two seeds share household splits: it admits a candidate that is
  worse by exactly the allowance up to 14% of the time.
- *Old gate under reader noise.* It also rejects identical releases through winner-selection noise in single cells (L0b, occupation
  and pair views).
- *AGG1.* Its boundary admission stayed at or below about 0.033, and it does not reject reader-only noise.
- *Resolution limit.* With two repetitions **neither screen can resolve ±0.010 for the income view**. The source's split noise for
  income (sd_A 0.021) is twice the allowance.
- *Multiplicity.* At the configuration level, even a candidate truly 0.010 safer than every comparator in both views passes rarely
  (old 11-15%, AGG1 3-5%).
- *Third repetition.* It moves AGG1 in the useful direction and the old gate in the wrong one.

**On the frozen source wires** (descriptive; the truth is unknown), the screens disagree on 12 of 72 oriented income pairs and 12 of
420 occupation pairs, in both directions.

**L2 is uninformative near the boundary, and this is a flaw in the law I registered, not in its execution.**
- The calibration itself worked. η met the LOG_LR target within 2e-4 on R1, and the fixed recipe over 10 extra splits gives +0.003 to
  +0.022, close to the targets.
- The full slate realized far larger shifts: +0.015 to +0.035 (H_LJ income), +0.020 to +0.082 (JOINT occupation), and +0.15 to +0.20
  (A1_PO_16 occupation).
- The likely reason: a SEX-dependent shift of a tree learner's discrete-valued outputs creates a fingerprint that lookup and tree
  readers exploit far beyond a linear reader.
- L2 therefore shows only the correct rejection of large realized shifts.

## 3. Caveats for interpreting new configurations

1. **Decided before any fit: every H3-anchored release fails both configuration-level screens through H3's frozen income view.**
   - Old gate: against H_LJ (+0.031) and TWJ (+0.015) in S0R1.
   - AGG1: against H_LJ (U = +0.0227) and RETIRED_B3CTL (U = +0.0159).
   - Every Track F candidate keeps H3 income, so only the **occupation-view** comparisons carry new information. Report them per
     comparator. Never report a configuration-level screen "pass" or "failure" for a Track F candidate as a finding.
2. **Only 2 distinct occupation comparators.** The occupation-view comparator set has 2 distinct wires: one JOINT wire shared by five
   comparators, and NEURAL_RAW. Six occupation comparisons are effectively two.
3. **Neither screen is strong evidence with two repetitions.**
   - A failure of either screen is weak evidence of worsening.
   - An old-gate pass is not evidence of non-worsening.
   - An AGG1 pass is development permission only.
4. **Near-identical candidates behave like L0b.** Identity or s = 0.1 is likely (see the note below).
   - A release bitwise equal to the witness is an exact alias against it: a pass with no information.
   - A near-identical release can fail the old gate through winner noise alone, especially in the pair view.
5. **A tree-based candidate's pass says nothing about fingerprint-type leakage.** Its discrete outputs can carry SEX in ways that
   flexible readers exploit (L2, A1_PO_16). A pass under either screen does not address that.

**Note for the records (what the frozen rules imply; not a rule change).** The refitted D witnesses' men's-guard reserve slack is
+0.0005 (D0) and -0.0008 (D1).
- On D1, SIMPLE and UNCERTAINTY accept s = 0.1 only if the correction *lowers* men's Brier by at least 0.0008 (UNCERTAINTY: plus
  z·τ).
- OLD's budget is 0.8 × (-0.0008) < 0, so its projection returns 0, because s = 0 already counts as infeasible.
- Identity on D1 is therefore likely for every rule. Recipe gate G1 (a nonzero strength on both D partitions) would then give
  `NO_RECIPE`, a registered outcome.
