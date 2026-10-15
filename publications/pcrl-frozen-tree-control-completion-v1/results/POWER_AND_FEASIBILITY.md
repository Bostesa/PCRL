# Confirmation power and feasibility (frozen L_TREE release)

**Bottom line.** (Table values are Monte Carlo quantile estimates with 95% MC intervals in the CSV; round up for planning, e.g. ≈317,000 with MC 95% 312k–321k.) Under the observed effects, a fresh confirmation of the frozen release would need about **317,000 independent evaluation groups** for 80% joint power. That is for the narrower claim B, with the IUT design; it is about 641,000 with the original simultaneous bounds. The constraint is the occupation utility caps, not privacy: the privacy clauses alone need about **4,100** groups.

The ORIGINAL criterion (claim A) is **not achievable at any sample size** unless the true benefit over the historical score codes exceeds 0.020. Its observed value is 0.014–0.018.

No compatible unexposed Adult population exists, and 317,000 exceeds the whole Adult dataset (48,842 records). See `POPULATION_AUDIT.md`.

## Method

- **Inputs.**
  - Private paired loss vectors and frozen reader predictions from the already-used INNER role (2,234 independent duplicate groups; 2,235 rows).
  - They are reduced to **per-group influence functions** for 84 base statistics: the 24 utility clause statistics, and coalition plus recipient AUCs for the candidate and 9 comparators, both seeds.
  - AUC influence uses the two-class (Hajek / placement) form.
  - Groups are SEX-pure, so the covariance splits by class.
  - Dependence is preserved: both seeds are on the same people, and comparators, recipients, tasks and clauses all share the same groups.
- **Validation.** Influence-function SEs reproduce the original 2,048-draw bootstrap: 0.00735 vs 0.00726 for the L_TREE − RAW_PROJ coalition benefit, and 0.00329 vs 0.00324 for the binding Brier clause. Only aggregate moments are published (`power/FEASIBILITY_INPUTS.json`, `power/FEASIBILITY_SUPPLEMENT.json`).
- **Model.**
  - For G future groups: T̂ ~ N(μ, AΣAᵀ/G).
  - The sample size is G, the number of **independent groups**. Bootstrap draws and model seeds do not count: adding seeds on the same people adds no evaluation records.
  - Joint power uses 20,000 Monte Carlo draws, with the exact G* construction (`tree_completion/power.py`). Monte Carlo SE is ≤ 0.0035, and 95% Monte Carlo intervals for N80 are in the CSV.
- **Resampling check** (`power/RESAMPLING_CHECK.json`).
  - It draws G groups with replacement from INNER and applies the rule with estimated SEs.
  - It agrees with the normal model within Monte Carlo error at the N80 points: 0.798 vs 0.800 (B, IUT) and 0.806 vs 0.800 (A, Bonferroni).
  - One tail cell differs by about 3.4 MC SE: B, Bonferroni, G = 10⁶, 0.979 vs 0.967. There the normal model is slightly conservative.
  - A bootstrap of old people is a planning device, not new evidence.
- **Designs** (`STATISTICAL_DESIGN_IUT.md`).
  - **IUT:** one fixed global claim; every clause one-sided at α = 0.025. False-confirmation probability ≤ 0.025 for that one conjunction, under any dependence.
  - **SIM:** the original two-sided Bonferroni simultaneous bounds over K = number of clauses (30 for B, 42–45 for A), and the original K = 78.

## Table A: ORIGINAL criterion

The criterion is 24 utility clauses, plus benefit > 0.020 against RAW_PROJ, N_EQ, T_TREE, H_JOINT_CODE, H_SEQ21_CODE and H_DPC_JOINT, plus no recipient excess (42 clauses). Cells show groups for 80% / 90% joint power.

| true benefit vs the three codes | IUT α = 0.025 | SIM (K = 42) | SIM (K = 78) |
|---|---|---|---|
| 0.010 | not achievable | not achievable | not achievable |
| 0.015 | not achievable | not achievable | not achievable |
| 0.020 | not achievable (boundary: joint power ≤ 0.009 at 10⁶) | not achievable | not achievable |
| 0.025 | 319,087 / 426,253 | 676,016 / 828,505 | 734,187 / 892,774 |
| 0.030 | 319,087 / 426,253 | 676,016 / 828,505 | 734,187 / 892,774 |
| 0.040 | 319,087 / 426,253 | 676,016 / 828,505 | 734,187 / 892,774 |
| observed (0.014–0.018) | not achievable | not achievable | not achievable |

- **Adding the completion's required RAW_J_NEQ (45 clauses)** at benefit 0.025: 321,057 (IUT) and 685,098 (SIM).
- **Privacy clauses alone** need 46,550 (IUT) at benefit 0.025, 11,640 at 0.030 and 4,105 at 0.040. They are not achievable at or below 0.020.
- **A true benefit at or below the margin cannot be confirmed.** If the true benefit over the codes is below 0.020, no sample size gives high power to establish that it exceeds 0.020: power stays at or below α at the boundary and goes to 0 below it. The observed development effects (0.014–0.018) sit in that region.

## Table B: NEW narrower claim (not passage of A)

The claim is 24 utility clauses, plus benefit > 0.020 vs RAW_PROJ, plus superiority > 0 vs H_SEQ21_CODE, plus 4 no-excess clauses (30 clauses). Cells show groups for 80% / 90% joint power.

| true superiority vs H_SEQ21_CODE | IUT α = 0.025 | SIM (K = 30) | SIM (K = 78) |
|---|---|---|---|
| 0.010 – 0.040, and observed (0.015) | 316,601 / 422,593 | 640,719 / 788,244 | 730,363 / 887,353 |

- **Utility binds everywhere.** The N80 is identical across the whole privacy grid because utility binds.
- **Privacy clauses alone** (IUT): 9,225 at 0.010, 4,101 at 0.015, 2,346 at 0.020, and 4,079 at the observed values.
- **Joint power at the observed effects (IUT):**

  | groups | joint power |
  |---|---|
  | 2,234 | 0.002 |
  | 20,000 | 0.052 |
  | 100,000 | 0.333 |
  | 200,000 | 0.602 |
  | 500,000 | 0.941 |

## Utility sensitivity (occupation summed Brier dominates)

The binding clause is seed-1 occupation summed Brier excess vs U0. It is 0.004225 against the 0.005 cap, a slack of 0.000775 with per-group SD 0.156. That is only about 0.24 INNER standard errors of slack, so the required size is extremely sensitive to the true value. **Do not size the study from privacy alone.**

The shift is applied to all four occupation Brier clauses. Cells show 80% / 90% groups.

| shift | IUT α = 0.025 | SIM (K = 30) |
|---|---|---|
| −0.002 | 61,340 / 80,872 | 121,935 / 150,165 |
| −0.001 | 76,731 / 96,794 | 144,414 / 171,867 |
| 0 (observed) | 316,601 / 422,593 | 640,719 / 788,244 |
| +0.0005 | 2,511,574 / 3,352,308 | 5,083,644 / 6,254,150 |
| +0.001 (beyond the cap; illustrative) | not achievable | not achievable |

An **underpowered acceptable effect** has a true excess below the cap: a large enough study eventually confirms it. An **unacceptable effect** has a true excess above the cap: no study confirms it, and the IUT correctly refuses.

## Variance and prevalence sensitivity (claim B, 80% groups)

| scenario | IUT | SIM (K = 30) |
|---|---|---|
| variance × 0.8 | 253,280 | 512,575 |
| variance × 1.25 | 395,751 | 800,899 |
| variance × 1.5 | 474,901 | 961,079 |
| SEX minority share 0.25 (observed 0.323) | 110,818 | 223,370 |
| SEX minority share 0.45 | **not achievable** | not achievable |

- **Threshold.** Under the label-shift model, claim B's binding clause exceeds its cap once the minority share passes about 0.362.
- **Why prevalence matters so much.** The occupation Brier excess is concentrated in the minority SEX class: class-conditional mean about +0.0177, against about −0.0022 in the majority.
- **How the shift is modelled.** A label shift with unchanged class-conditional distributions moves the utility means, not just the variances (a statistics-review correction). At a minority share of 0.45, the implied true excess (about 0.0068) exceeds the cap.
- **A descriptive concern.** L_TREE's occupation calibration cost falls disproportionately on the minority SEX class.

## Limitations

1. **The estimates are selected and optimistic.** All μ and Σ come from the already-used INNER role and from a candidate selected (on FIT2) among many. A published flat-prior assurance calculation (`power/ASSURANCE_CLAIM_B.json`, `power/ASSURANCE_CLAIM_A.json`; `tree_completion.power.assurance`) gives the following, and selection makes even these optimistic:
   - claim B: the probability that every clause truly passes is about 0.38; average joint power is about 0.31 at 316,601 groups, 0.34 at 1 million and 0.36 at 5 million;
   - claim A: the probability is about 0.05, and average power stays below 0.05.
2. **A different population means assumptions, not estimates.** Any population large enough would not be Adult. Its μ, Σ, prevalence and reader AUCs would be assumptions.
3. **Simplifications.** Known-variance normal approximation, and z instead of t (immaterial at G ≥ 4,000; pre-register t(G−1) for the analysis). Readers are frozen, so their fitting variability on new attacker data is not included.
4. **Fixed comparator set.** The comparator set is the frozen one; including FARE would require it to be utility-feasible, which it is not.

## Feasibility verdict

The original criterion is **not practically feasible**:
- it is impossible at the observed code margins;
- even if those margins were 0.025, it needs about 320,000–735,000 groups;
- no compatible unexposed population is available.

Claim B is a scientifically narrower, prospective claim: materially better than the incumbent and better than the best code control by a positive margin. Its comparator choice used exposed INNER outcomes. It also needs about 317,000 groups under the observed utility slack, more than any compatible population, and is infeasible if the evaluation population's SEX composition is more balanced. Neither table authorizes opening new labels.
