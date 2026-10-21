# Research decision: two independently designed release families (campaign `pcrl_dual_design_release_v1`)

Florida ACS 2019 development cohort (admitted public-use PUMS records), 2026-10-21, start 16:37:33Z.

**The service.** It predicts an income class and an occupation group from six fields (AGEP, WKHP, COW, SCHL, MAR, POBP). Each
recipient receives only its own probability vector and argmax decision.

**The privacy evaluation.** It measures how well classifiers ("readers") infer the dataset's recorded SEX attribute from what the
recipients receive: from each output alone and from the linked pair.

**Scope.** Everything here is exploratory development evidence on previously and adaptively used MODEL_SELECT records (8,001 people,
5,306 households), for the declared readers only. Lower recovery for these readers is not protection against every reader.

## Headline (the statistical reviewer's required wording, `roles/statistical_reviewer/CLOSING_STATEMENT.md` section 2)

> This is development evidence on previously and adaptively used MODEL_SELECT records of the admitted ACS 2019 Florida cohort (8,001
> people), for the declared readers only: it is not confirmation, and lower recovery for these readers is not protection against
> every reader. Two independently designed release families, which belong to one idea family (SEX-standardized posteriors, STD, and
> propensity-tilt cancellation, PTC, both subtract the propensity-proportional part of the released task posteriors, so their joint
> negative closes one idea, not two), met the original utility limits at the point, but neither reduced the readers' recovery of the
> recorded SEX attribute beyond its matched controls: D2 was met in 0 of 2 registered family readings and in 0 of 2 non-registered
> (ADAPTIVE_R1) readings, 4 D2 looks in total; one AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates under
> six moderate NRM1 known laws and about 8% when one version or split dominates, and with 4 looks the chance that at least one passes
> by chance is at most 4 times the per-look rate. Registered readings (linked-pair benefit = comparator AUC - candidate AUC, mean
> (se_agg) over the 8 registered cells): DA_STD is PARTIAL_SBC_FRONTIER_ONLY, with +0.0175 (0.0090) over the equally trained twin and
> +0.0137 (0.0174) over its SEX-blind spending (SPEND) bracket, both unresolved; its one passed comparison, +0.0273 (0.0104) over its
> SEX-blind-target (SBC) bracket, is against controls that are themselves more recoverable than the twin at the point. DB_PTC is
> NOT_SEPARATED: +0.0164 (0.0107), +0.0173 (0.0171) and +0.0170 (0.0094) over the twin, SBC and SPEND brackets. It is not resolved below its
> release-only control DB_ZTILT (+0.0170 (0.0121)). Both are TASK_ONLY_FEASIBLE_NO_PROTECTION. Neither registered release is robust to
> classifier refitting (D2R ROBUSTNESS_NOT_CONFIRMED; for DB_PTC an extra, descriptive reading). Beside them, never as family results:
> DA's revision (IPW-STD, run outside its declared trigger, deviation 7) is ADAPTIVE_R1__NOT_SEPARATED (+0.0072 (0.0087) over the
> twin). DB's revision DB_PTC_R1 (recipient-split strengths 0.35 / 0.65) was afterwards selected, on the same registered cells, as DB's
> strongest feasible release, so it carries the winner's curse. It is ADAPTIVE_R1__PARTIAL_TWIN+SPEND_FRONTIER_ONLY: resolved over
> the twin (+0.0239 (0.0101)), over the SPEND bracket by a 2-SE margin of 0.0001 (+0.0245 (0.0122)), and not separated from its SBC bracket
> (+0.0265 (0.0139)). On refit cells (new household splits and reader seeds of the same people; descriptive, D2 unchanged) its benefits
> over the twin, SBC and SPEND brackets stay resolved (+0.0219 (0.0103), +0.0231 (0.0091), +0.0189 (0.0091); ROBUSTNESS_CONFIRMED).
> By the reading declared in card DB_ZTILT_R1 before that control's fit, it is not separated from the release-only control DB_ZTILT_R1, which tilts by
> P(SEX | the twin's released vectors): +0.0128 (0.0081) on the registered cells and +0.0066 (0.0095) on the refit cells. So its gain
> is not separated from release-only information; that control overspent (0.0169 against 0.0099 nats of MODEL_SELECT total NLL).
> DB_PTC_R1 is point-feasible but not reserve-qualified. No reading is competitive with the nine inherited references (D3
> NOT_SUPERIOR in all four), and none meets the 0.020 material criterion (D4 NOT_MET; the largest values, DB_PTC_R1's, are +0.0110
> (0.0135) over NEURAL_RAW and -0.0058 (0.0151) over H_LJ). The +0.007 to +0.024 differences against the twin are a smaller
> result against a weaker baseline, not the criterion. No release is a method, candidate or component; joint design (D5) was not
> assessed and novelty (D6) is not claimed.

## Decision

**No method is established.**

Both families test one idea family. STD and PTC each subtract the propensity-proportional part of the released task posterior, and
the revisions (IPW-STD and recipient-split PTC) stay inside that family. So this joint negative closes one idea, not two
(PROTOCOL_AMENDMENT_1, finding 1).

D2, reducing recovery beyond matched controls, was met in 0 of 2 registered family readings and 0 of 2 non-registered (ADAPTIVE_R1)
readings: 4 D2 looks in total (`claims/CAMPAIGN_SUMMARY.json`). One AGG_POSITIVE comparison passes a true null in about 0.07-2% of
replicates under six moderate NRM1 known laws, and about 8% when one version or split dominates. With 4 D2 looks, the chance that at
least one passes by chance is at most 4 times the per-look rate.

Every release met all 28 original utility conditions at the point (G1: 12 clauses and 2 guards in each of its two
version-specific configurations; all 48 configurations pass). Point feasibility is not confidence preservation. No reading is
competitive with the inherited references (D3), and none meets the 0.020 material criterion (D4).

**Registered readings, the family results.** DA_STD is PARTIAL_SBC_FRONTIER_ONLY and DB_PTC is NOT_SEPARATED. Both are
TASK_ONLY_FEASIBLE_NO_PROTECTION.

**Beside them, never as a family result:** DB's adaptive revision DB_PTC_R1 (ADAPTIVE_R1; recipient-split strengths 0.35 / 0.65). The
registered rule selected it afterwards, on the same registered cells, as DB's strongest feasible nonzero release (winner's curse;
descriptive):
- **Registered cells.** For the declared readers on the registered cells:
  - its linked-pair AUC is below the twin's by 0.0239 (se_agg 0.0101; lower 2-SE bound +0.0037);
  - it is below the SPEND bracket DB_SPD_HI's by 0.0245 (se_agg 0.0122), which passes the SPEND frontier comparison with a lower
    2-SE bound of only +0.0001;
  - it is not separated from its SBC bracket (+0.0265, se_agg 0.0139; lower bound -0.0013).
- **Refit cells.** On new household splits R3 / R4 and reader seeds k2 / k3 of the same 8,001 MODEL_SELECT people, its benefits stay
  resolved (D2R ROBUSTNESS_CONFIRMED, descriptive):
  - over the twin, +0.0219 (se 0.0103);
  - over the SBC bracket, +0.0231 (se 0.0091);
  - over the SPEND bracket, +0.0189 (se 0.0091).

  Its benefit over the release-only control DB_ZTILT_R1 roughly halves and is not confirmed (+0.0066, se 0.0095). The refit reading
  never changes D2 and is not a replication. It inherits the winner's curse: the refit estimates are below the registered ones
  against all three comparators.
- **Declared reading against the release-only control.** By the reading declared in card DB_ZTILT_R1 before its fit, DB_PTC_R1 is
  NOT_RESOLVED_BELOW_ZTILT. Its pair benefit over DB_ZTILT_R1, a tilt whose amplitude is estimated from the twin's released vectors
  alone, is +0.0128 (se_agg 0.0081; lower 2-SE bound -0.0035; version means +0.0149 / +0.0108) on the registered cells. So its gain
  is not separated from release-only information (PRC's failure repeated, as declared). The control overspent, which makes the
  comparison a bracket from above; that is disclosed and does not change the declared reading.
- **References and utility.** It stays 0.005-0.027 AUC above four strong references at the point, resolved only against
  RETIRED_B3CTL (-0.0267, se 0.0111). It is point-feasible but not reserve-qualified in version 1.

| Reading | Status | D1 task feasibility | D2 beyond matched controls | D3 vs nine references | D4 0.020 | Outcome class |
|---|---|---|---|---|---|---|
| DA_REG (STD, card DA_STD_v1) | registered | `TASK_FEASIBLE` | `PARTIAL_SBC_FRONTIER_ONLY` | `NOT_SUPERIOR` | `NOT_MET` | `TASK_ONLY_FEASIBLE_NO_PROTECTION` |
| DB_REG (PTC, card DB_PTC_v1) | registered | `TASK_FEASIBLE` | `NOT_SEPARATED` | `NOT_SUPERIOR` | `NOT_MET` | `TASK_ONLY_FEASIBLE_NO_PROTECTION` |
| DB_R1 (recipient-split PTC, card DB_PTC_R1) | `ADAPTIVE_R1` | `ADAPTIVE_R1__TASK_FEASIBLE` | `ADAPTIVE_R1__PARTIAL_TWIN+SPEND_FRONTIER_ONLY` | `ADAPTIVE_R1__NOT_SUPERIOR` | `ADAPTIVE_R1__NOT_MET` | `ADAPTIVE_R1__TASK_ONLY_FEASIBLE_NO_PROTECTION` |
| DA_R1 (IPW-STD, card DA_STD_R1) | `ADAPTIVE_R1` | `ADAPTIVE_R1__TASK_FEASIBLE` | `ADAPTIVE_R1__NOT_SEPARATED` | `ADAPTIVE_R1__NOT_SUPERIOR` | `ADAPTIVE_R1__NOT_MET` | `ADAPTIVE_R1__TASK_ONLY_FEASIBLE_NO_PROTECTION` |

D5 (joint design) is `NOT_ASSESSED_MISSING_CONTROLS` and D6 (novelty) is `NOT_CLAIMED` in all four readings, with the same prefix on
the adaptive ones. The adaptive readings are reported beside their family's registered reading and never as its result. No partial
D2 outcome is a claim. No release is a method, candidate or component.

## The two families and their matched controls

- **DA, STD (Designer A).** Release the SEX-standardized task posteriors: a mixture of SEX-aware task models evaluated at both SEX
  values with fixed, guard-balanced weights. It is blended with the twin at strength lambda (MODEL_FIT-selected: 1.0 in both
  versions). The SEX input is used at training time only.
- **DB, PTC (Designer B).** Tilt each recipient's log-probabilities against a cross-fitted SEX propensity, with closed-form weights
  that cancel the propensity's covariance with the released scores. Strength alpha 0.35 in both versions. The release writes the
  runtime propensity into the vector and is injective on its support (disclosed in the card before any outcome). Revision R1
  splits the strength by recipient: (alpha_I, alpha_O) = (0.35, 0.65) in both versions.
- **DA revision R1, IPW-STD.** SEX-blind task models trained with inverse-propensity weights; lambda* 1.0 / 0.75, so
  `CAND_STRENGTH_DIFFERS_BY_VERSION` applies (deviation 12(b)).
- **Matched controls (per family, PROTOCOL 8):**
  - the equally trained task-only twin DA_TWIN, shared by both families;
  - an SBC path: the same operation driven by a SEX-independent target. That is permuted SEX, or the PSEUDO2 label-derived surrogate
    when permutation was degenerate or could not bracket the release's spend. The construction differs by version: DA used PSEUDO2
    gamma 2 / 1; DB, registered and R1, used permuted SEX in version 0 and PSEUDO2 gamma 1 in version 1;
  - a SPEND path: SEX-blind subtractive-dithered quantization of the twin;
  - two MODEL_FIT bracket points per path. A revision refits its SBC path and reuses the SPEND path.
- **Simplest alternative explanation, DB:**
  - ZTILT, the same tilt with the amplitude P(SEX | the twin's own release);
  - SHUF, a keyed donor propensity;
  - for DB_PTC_R1, the release-only control DB_ZTILT_R1. It is the registered ZTILT tilt along DB_PTC_R1's split law and was
    intended to match its spend, but the registered scale grid made it spend more: 0.0169 against 0.0099 nats of MODEL_SELECT total
    NLL (card DB_ZTILT_R1; deviation 13).
- **Component ablations:**
  - DA_COMP: SEX weights from the person's propensity instead of fixed weights;
  - DA_STD_OCC and DB_PTC_OCC: the occupation component only;
  - DB_PTC_R1_OCC: the occupation component of DB_PTC_R1. It and DB_ZTILT_R1 were added after card DB_PTC_R1 had said they
    would not be refitted (deviation 9).
- **References:** the nine inherited point-eligible references (NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100),
  with U0 / Ucal as diagnostics.

## Task feasibility

Every release met all 28 original utility conditions at the point (G1): 12 clauses and 2 guards in each of its two version-specific
configurations. All 48 configurations of the 48-configuration budget pass. The limits are:
- accuracy loss <= 0.010, NLL excess <= 0.010 nats and summed Brier excess <= 0.005, against U0 and Ucal, for both tasks and both
  versions;
- men's and women's occupation Brier excess <= 0.005.

Point feasibility is not confidence preservation. The twin passes with a wide margin (worst value -0.42 / -0.10 of its cap). The
two-SE reserve heuristic is reported separately and never gates. In the table, the worst value is taken over clauses and guards;
for DA_STD and DB_PTC_R1 the binding condition is the men's occupation Brier guard.

| Release | Worst clause or guard, value / cap (V0 / V1) | Total NLL vs U0 (nats) | Two-SE reserve ratio (>= 2 qualifies) |
|---|---|---|---|
| DA_TWIN | -0.425 / -0.100 | -0.0247 / -0.0209 | 4.88 / 3.82 |
| DA_STD | -0.089 / +0.099 | -0.0167 / -0.0128 | 3.00 / 2.58 |
| DB_PTC | -0.362 / +0.012 | -0.0194 / -0.0156 | 3.90 / 3.22 |
| DB_PTC_R1 | +0.011 / +0.325 | -0.0149 / -0.0110 | 2.66 / **1.87** |
| DA_IPW | -0.025 / -0.012 | -0.0147 / -0.0154 | 2.85 / 3.50 |

Not reserve-qualified in at least one version:
- the candidate DB_PTC_R1 (`RESERVE_ONE_SEED_ONLY`) and its ablation DB_PTC_R1_OCC;
- the controls DB_ZTILT_R1 (0.86 / 0.09; its version-1 men's occupation Brier guard is at 0.96 of the cap), DB_SBC_R1_HI (1.37 / 1.59),
  DA_SBC_HI (0.61 / 1.84), DA_SPD_HI (2.95 / 0.93) and DB_SPD_HI (2.90 / 1.94).

DB_PTC_R1 is near the limits and carries the flag `POINT_FEASIBLE_NOT_RESERVE_QUALIFIED`. Under RULES.md 6 it is therefore never
called robust. The D2R label `ROBUSTNESS_CONFIRMED` below concerns its recovery comparisons under classifier refitting, not the
release.

The recipient gate passed in every reading against the twin. No recipient's own output is more than 0.010 AUC more recoverable than
under the twin, in both versions (`RECIPIENT_AGG_RESOLVED`: within this allowance at 2 SE; `_EDGE`: within it at the point). At the
point, every candidate's single-view AUCs are below the twin's.

## Improvement beyond matched controls (D2)

Pair benefit = comparator pair AUC - candidate pair AUC (positive favours the candidate). The aggregate is over the 8 registered
cells, with se_agg and the version means in brackets. "Resolved" = mean - 2 se_agg > 0 and both version means > 0.

| Reading | vs TWIN | vs SBC bracket Q* | vs SPEND bracket Q* |
|---|---|---|---|
| DA_REG (registered) | +0.0175 +- 0.0090 [0.0160 / 0.0190], point only | DA_SBC_HI: +0.0273 +- 0.0104, passes the frontier comparison | DA_SPD_HI: +0.0137 +- 0.0174, not separated |
| DB_REG (registered) | +0.0164 +- 0.0107 [0.0236 / 0.0091], point only | DB_SBC_HI: +0.0173 +- 0.0171, not separated | DB_SPD_HI: +0.0170 +- 0.0094, not separated |
| DB_R1 (ADAPTIVE_R1) | +0.0239 +- 0.0101 [0.0254 / 0.0224], resolved (lower bound +0.0037) | DB_SBC_R1_HI: +0.0265 +- 0.0139, not separated (lower bound -0.0013) | DB_SPD_HI: +0.0245 +- 0.0122, passes the frontier comparison (lower bound +0.0001) |
| DA_R1 (ADAPTIVE_R1) | +0.0072 +- 0.0087, point only | DA_IPW_SBC_HI: +0.0095 +- 0.0164, not separated; targeting fails in version 1 | DA_SPD_HI: +0.0034 +- 0.0178, not separated; targeting fails in version 1 |

Caveats that change the reading:
- **DA_REG's SBC pass is weak evidence.** Both DA SBC points are more recoverable than the twin at the point (pair AUC 0.6604 and
  0.6603 against 0.6506), so beating them partly reflects a control path that adds recoverable signal (verifier observation 2).
- **DB_R1's SBC miss is a non-separation.** Its 2-SE lower bound is -0.0013: the interval includes zero, so the comparison is not
  resolved. The SBC bracket costs only 0.0008 nats more than the release, so the bracket is already close in spend. The registered
  rule reads this "not separated", never "equal" and never "a power problem".
- **Power was far below plan.** The protocol planned for se_agg about 0.005, with 0.015 resolved at about 80% power. Realized D2
  se_agg values were 0.0087-0.0178, so the 2-SE threshold was 0.017-0.036, and "not separated" is weak evidence of absence.

Descriptive beside D2 (DB's simplest alternative explanation):
- **DB_PTC vs DB_ZTILT:** +0.0170 +- 0.0121, not resolved. The card's declared reading is that PRC's failure repeats: the registered
  gain is not separated from release-only information. DB_ZTILT's pair AUC is not distinguishable from the twin's (point -0.0006,
  se 0.0079).
- **DB_PTC_R1 vs the registered DB_ZTILT:** +0.0245 +- 0.0075, resolved. That control uses a different split law and was not matched
  to the revision's spend: its MODEL_SELECT cost against the twin is 0.0085 nats against the release's 0.0099. The same-law control
  DB_ZTILT_R1 is not separated (below).
- **DB_PTC_R1 vs DB_ZTILT_R1** (card DB_ZTILT_R1; `claims/ZTILT_R1_READING.json`):
  - registered cells: +0.0128 +- 0.0081 [0.0149 / 0.0108], point only (lower 2-SE bound -0.0035);
  - declared reading: `NOT_RESOLVED_BELOW_ZTILT`;
  - refit cells: +0.0066 +- 0.0095, not confirmed.

  The control overspent. By its registered first-reaching rule it spent 2.08x / 1.73x the release's MODEL_FIT spend, because the
  registered grid jumps from k = 1 (0.90x / 0.75x) to k = 1.5. That makes the comparison a bracket from above. The overshoot is
  disclosed and does not set the declared reading aside. Equally, the reading is not "the gain is fully explained by release-only
  information": the pair comparison is unresolved, with a positive point estimate in both versions. On each single view the control
  matches DB_PTC_R1 (income AUC 0.5329 vs 0.5333, occupation 0.6177 vs 0.6166); the difference is on the linked pair (0.6395 vs
  0.6267).
- **DB_PTC_R1 vs SHUF:** +0.0231 +- 0.0125, point only. SHUF spends 0.0098 / 0.0123.

Data-rich readers (descriptive point values with no uncertainty estimate). LOG_LR, HGB15 and MLP64 were trained on 20,000 MODEL_FIT
releases and scored on the same SCORE parts:
- For DB_PTC_R1, every version mean is positive against the twin (+0.025 to +0.040), both bracket controls (+0.013 to +0.040) and the
  registered ZTILT (+0.017 to +0.025).
- No data-rich readers were trained for DB_ZTILT_R1, the release-only control at which the grid comparison is unresolved.
- On the income view, the data-rich LOG_LR finds DB_PTC_R1's output (bitwise DB_PTC's) more recoverable than DB_ZTILT's
  (-0.0040 / -0.0042).
- For DA_STD vs DA_SPD_HI, HGB15 sees only +0.002 / +0.005.
- DA_IPW's benefit over DA_SPD_HI is negative in version 1 (HGB15 -0.0053). The amendment's rule would have labelled that
  READER_BUDGET_DEPENDENT had its D2 passed.
- DA_IPW vs DA_STD is mostly negative.
- Agreement in sign is not evidence of information removal.

## Robustness to classifier refitting (D2R)

Under RULES.md 8, each family's strongest feasible nonzero release gets its readers refitted. That release has the largest
registered-cell pair benefit against the twin over all readings: DA_STD for DA and DB_PTC_R1 for DB (`claims/STRONGEST.json`).
- **New cells.** The refits use new household splits R3 / R4 and new reader seeds k2 / k3 of the same MODEL_SELECT people. Those
  cells are added to the registered ones.
- **Effectiveness.** The refits were genuinely effective in every case: distinct split hashes, and reader seeds effective on both
  seed pairs.
- **The family D2Rs** are DA_REG_D2R and DB_R1_D2R. DB_REG_D2R, of the registered DB_PTC, is an extra, descriptive reading
  (deviation 12(d)).

| Release | D2R | vs TWIN (refit cells) | vs SBC bracket | vs SPEND bracket |
|---|---|---|---|---|
| DA_STD (DA's strongest; DA_IPW's +0.0072 is lower) | `ROBUSTNESS_NOT_CONFIRMED` | +0.0176 +- 0.0110 | DA_SBC_HI: +0.0219 +- 0.0099, confirmed | DA_SPD_HI: +0.0103 +- 0.0125 |
| DB_PTC (registered; extra, descriptive) | `ROBUSTNESS_NOT_CONFIRMED` | +0.0126 +- 0.0067 | DB_SBC_HI: +0.0119 +- 0.0122 | DB_SPD_HI: +0.0096 +- 0.0098 |
| DB_PTC_R1 (DB's strongest; ADAPTIVE_R1) | `ROBUSTNESS_CONFIRMED` | +0.0219 +- 0.0103 (lower bound +0.0013) | DB_SBC_R1_HI: +0.0231 +- 0.0091 | DB_SPD_HI: +0.0189 +- 0.0091 (lower bound +0.0007) |
| DB_PTC_R1 vs DB_ZTILT_R1 (release-only control; beside D2, outside the decisive set) | `ROBUSTNESS_NOT_CONFIRMED` | +0.0066 +- 0.0095 | | |

Against the references, DB_PTC_R1 is not confirmed: NEURAL_RAW +0.0018 +- 0.0107 and H_LJ -0.0094 +- 0.0135 on the refit cells.

The robustness reading is descriptive and never changes D2:
- **D2 stays partial.** On the registered cells, DB_PTC_R1 is not separated from its SBC bracket.
- **The new cells are not new people.** For k2 / k3 the R1 / R2 splits are the registered ones, and the versions share every
  training row. This is not a replication.
- **Selected on the same cells.** DB_PTC_R1 was chosen by the registered rule (RULES.md 8) as DB's strongest feasible nonzero release
  on the registered cells, from a pool that included the adaptive revision. Deviation 8 maps its strength pair to alpha_I + alpha_O,
  which can only break exact ties, and there was none. The winner's curse applies: its refit estimates are below its registered
  ones against the twin (0.0219 vs 0.0239), the SBC bracket (0.0231 vs 0.0265) and the SPEND bracket (0.0189 vs 0.0245).
- **Not stable in every block.** In version 0, split R1, the benefit over the twin is +0.034 with seeds k0 / k1 and +0.003 with
  k2 / k3. The other blocks lie between +0.017 and +0.033.

Component ablations. The sign convention is the same: benefit = comparator AUC - ablation AUC.

| Ablation | vs its release | vs TWIN | Reading (descriptive; no ablation difference is resolved against the twin) |
|---|---|---|---|
| DA_COMP | -0.0304 +- 0.0096 | -0.0129 +- 0.0070 | With the person's own propensity as the SEX weight, the release is resolved more recoverable than DA_STD, and at the point only also more recoverable than the twin (not resolved). DA_STD's point benefit depends on the fixed weights. |
| DA_STD_OCC | -0.0068 +- 0.0083 | +0.0107 +- 0.0075 | Keeps about 60% of DA_STD's point benefit over the twin |
| DB_PTC_OCC | -0.0101 +- 0.0127 | +0.0063 +- 0.0083 | Keeps about 40% of DB_PTC's |
| DB_PTC_R1_OCC | -0.0146 +- 0.0116 (registered); -0.0107 on the refit cells | +0.0093 (registered, point only); +0.0112 on the refit cells, not confirmed | Keeps about 40% of DB_PTC_R1's |

These shares are ratios of unresolved point estimates. Pair AUC differences are not additive over recipients.

## Competitiveness against strong references (D3)

`NOT_SUPERIOR` in every reading. D3 needs a resolved benefit over all nine point-eligible references; every reading fails eight or
all nine. Four references have a registered-cell mean pair AUC at or below 0.6217:

| Reference | RETIRED_B3CTL | SUCC | H_LJ | B2TWIN |
|---|---|---|---|---|
| Pair AUC | 0.6000 | 0.6145 | 0.6209 | 0.6217 |

For comparison, the twin is at 0.6506, DA_STD 0.6331, DB_PTC 0.6342 and DB_PTC_R1 0.6267. DB_PTC_R1's pair benefit over each
reference:

| Reference | Pair benefit | Status |
|---|---|---|
| RETIRED_B3CTL | -0.0267 +- 0.0111 | resolved worse |
| SUCC | -0.0122 +- 0.0126 | not resolved |
| H_LJ | -0.0058 +- 0.0151 | not resolved |
| B2TWIN | -0.0050 +- 0.0135 | not resolved |
| NEURAL_RAW | +0.0110 +- 0.0135 | point only |
| TT100 | +0.0278 +- 0.0111 | resolved |

The four strongest references share an occupation output whose occupation-view AUC is 0.5600. Their utility sits near the limits
(worst clause or guard at 0.79-0.88 of its cap). The candidate releases here have occupation-view AUC 0.617-0.632 and use less of the utility
budget. That is not a utility-matched comparison, and it changes no label.

For the registered releases DA_STD and DB_PTC, the verifier's own LR and HGB readers change the sign against NEURAL_RAW and H_LJ, so
those differences are reader-dependent. No independent-reader check was run for DB_PTC_R1. D3 fails under every reader used.

## The material criterion (D4)

`NOT_MET` in every reading. D4 needs a resolved 0.020 pair benefit over both NEURAL_RAW and H_LJ. The largest values, DB_PTC_R1's,
are +0.0110 (se 0.0135) over NEURAL_RAW and -0.0058 (se 0.0151) over H_LJ. No smaller gain is presented as the criterion.

At the point, each of the four candidate releases (DA_STD, DB_PTC, DB_PTC_R1, DA_IPW) had lower linked-pair recovery than the twin.
Those differences are a separate, smaller result against a
weaker baseline:
- registered: DA +0.0175 (se 0.0090) and DB +0.0164 (se 0.0107), neither resolved;
- adaptive revisions: DB +0.0239 (se 0.0101), resolved; DA +0.0072 (se 0.0087).

## Joint-design evidence (D5)

`NOT_ASSESSED_MISSING_CONTROLS`. The registered rule needs separately designed recipients and both sequential orders as controls.
They were not built, and no joint-design advantage is claimed. They were not planned: the protocol registered D5 as not assessed
without them, and the 48-configuration budget (PROTOCOL 8) allocated none.

At the point, no ablation difference is resolved, and pair AUC differences are not additive over recipients. The occupation-only
ablations keep about 60% of DA_STD's, 40% of DB_PTC's and 40% of DB_PTC_R1's registered point benefit over the twin.

DB's revision differs from DB_PTC only in its occupation strength (its income output is bitwise DB_PTC's). Its pair difference from
DB_PTC is +0.0075 (se 0.0120), not resolved. At the point both recipients' outputs contribute; no contribution is resolved, and no
comparison isolates a joint design.

## Novelty (D6)

`NOT_CLAIMED`. No separate prior-art review was run, both designers named close published work, and novelty is never inferred from
performance:
- **STD:** attribute averaging and standardization (Pope and Sydnor 2011; Zliobaite and Custers 2016; Pearl's adjustment formula),
  and this repository's earlier MMP proposal.
- **PTC:** AttriGuard (Jia and Gong 2018), LEACE (Belrose et al. 2023), multiaccuracy-style proxies (Kim, Ghorbani and Zou 2019) and
  exponential-tilt posterior adjustment (Saerens et al. 2002).
- **IPW-STD:** inverse-propensity reweighing (for example Kamiran and Calders 2012).

## What the evidence does and does not support

- **No release supports an information-removal claim.**
  - None is a post-processing of the twin's release, and all are deterministic functions of the six fields.
  - PTC writes the propensity into the released vector and is injective on its support (a design property stated in card
    DB_PTC_v1; not tested here).
  - Neither DB release is separated from its release-only control.
  - DA's lambda = 1 release kept about 2/3 (income) and 2/5 (occupation) of the twin's linear propensity dependence on MODEL_FIT
    (`da/R1_TRIGGER_B2.json`).
- **Supported:** the declared and data-rich readers recover SEX less well from these re-encodings, as descriptive point values.
- **Not supported:** removal, robustness of DA, or a guarantee against other readers.
- **Scope of the negative.** The two registered releases cannot be told apart at the point (DA_STD vs DB_PTC +0.0011, se 0.0123), as
  one idea family would predict. The joint negative says nothing about families outside this one, for example the finite or coarsened
  occupation outputs of the strongest references.

## Favourable and adverse evidence

Favourable:
- DB_PTC_R1's resolved 0.0239 reduction against the twin (registered cells, declared readers).
- Its pass of the SEX-blind spending frontier (lower 2-SE bound +0.0001).
- Its refit confirmation against the twin and both bracketing controls (descriptive).
- Its resolved benefit over the registered ZTILT. That control uses a different split law; the same-law control DB_ZTILT_R1 is not
  separated.
- Data-rich readers: for DB_PTC_R1 they agree in sign against every comparator they were trained for (point values).
- The verifier's own readers: for the registered releases they agree in sign against the twin and the controls.
- DA_STD's point benefit over the twin is similar on registered and refit cells (+0.0175, se 0.0090; +0.0176, se 0.0110) and
  resolved on neither (D2R not confirmed against the twin).

Adverse:
- No reading met D2.
- The only DA pass was against an SBC path that leaks more than the twin.
- DB_PTC_R1 is not separated from its SBC path on the registered cells.
- DB_PTC_R1 is not separated from its release-only control DB_ZTILT_R1. That control matches it on each single view, and the
  benefit falls to +0.0066 (se 0.0095) on the refit cells.
- Two of DB_PTC_R1's three refit confirmations clear the 2-SE threshold by about 0.001. The benefit over the twin in block S0R1 is
  +0.034 with seeds k0 / k1 and +0.003 with k2 / k3.
- The registered predictions for ZTILT_R1 failed in the direction adverse to DB_PTC_R1. ZTILT_R1's pair AUC was predicted within
  +-0.008 of the twin's (p 0.7) and was observed 0.0111 lower. DB_PTC_R1 was predicted resolved above it (p 0.55) and was not.
- DB_PTC_R1 is not reserve-qualified.
- The registered DB gain was not separated from release-only information (ZTILT).
- DA's revision was not better than the registered STD (point -0.0103, se 0.0107, not resolved). Its descriptive MODEL_FIT reader
  check (pair LOG_LR benefit over the twin +0.022 / +0.016 by version; DA_STD's was +0.023 / +0.024) did not carry over to
  MODEL_SELECT (+0.0072, point only).
- Every candidate stays above the four strongest references at the point.
- The effects are at the grid's resolution limit. All four readings use the same MODEL_SELECT people.

## Revisions, triggers and what was not run

- **DB R1** (recipient-split strengths) was triggered by DB_PTC_v1's registered MODEL_FIT condition. Its card was committed after the
  first MODEL_SELECT utility read and before any reader-grid output, hence ADAPTIVE_R1.
- **DB R2** (joint-decision strata) was not triggered on the registered or the R1 reading (`claims/DB_R2_TRIGGER_*.json`). The family
  stop rule did not fire either.
- **DA R1** (IPW-STD) was run outside its card's declared trigger (deviation 7), after the registered DA privacy outcome (DA_REG's
  D2): trigger (b) names `NOT_SEPARATED`, and DA_REG is `PARTIAL_SBC_FRONTIER_ONLY`. It addressed a recorded failure: D2 not met,
  and on MODEL_FIT the income release kept about two thirds of the twin's linear propensity dependence (R^2 ratio 0.666 / 0.661,
  threshold 0.5). It counts as one of DA's two revisions and as one D2 look.
- **DA R2** (decision-preserving STD) was not triggered. Its trigger needs lambda* < 1 with an accuracy clause as the only failure at
  the next grid value. The registered selection has lambda* = 1.0 in both versions. In R1, version 1 stopped at lambda* = 0.75, but
  the condition failing at 1.0 was the women's occupation Brier budget clause (MODEL_FIT), not an accuracy clause. No
  configuration budget was left either.
- **The replacement family was not used.** The 48-configuration budget was spent:
  - 32 registered;
  - 6 for each of the two revisions;
  - 2 for DB_PTC_R1's ablation and 2 for DB_ZTILT_R1, both added after card DB_PTC_R1 had said they would not be refitted
    (deviation 9).

## Untested questions

- **A confirmation population.** It would carry a registered reading of DB_PTC_R1 against its SBC bracket and against a release-only
  ZTILT point that is spend-matched along its split law. That needs a finer ZTILT scale grid: the registered one jumped from
  0.90x / 0.75x to 2.08x / 1.73x of the release's MODEL_FIT spend. A finer SBC grid would not help, because the SBC bracket is
  already within 0.0008 nats of the release. Any new reading on these people would be a fifth adaptive look. Confirmation was not
  authorized here.
- **Joint-design controls:** separately designed recipients and both sequential orders.
- **A different family.** Coarser occupation outputs, as the strongest references use, belong to a different family. So does any
  operation that does not subtract the propensity-proportional part.
- **Readers beyond the declared bank and the three data-rich recipes.** Other readers could recover more.

## Limits of this evidence

- Development evidence on previously and adaptively used records: MODEL_SELECT was read for every decision. Not confirmation.
- The sealed assessment, the reserved populations and the final-assessment roles stayed closed. ATTACK_FIT, ATTACK_VALIDATION and
  DEV_REPORT labels were never read; DEV_REPORT features were used label-free for deployment parity, as registered.
- The two versions share every training row and the MODEL_SELECT people. Repetitions and reader seeds are procedure repeats, not
  replications.
- Four D2 looks, all at one idea family, with no multiplicity adjustment across looks; the count and the per-look false-pass rate are
  disclosed instead.

## Resources, verification, custody and publication

**Resources** (`RESOURCE_ACCOUNTING.json`, `EXPERIMENT_LEDGER.csv`, `CONFIG_LEDGER.csv`; queue closed at 20:13:28Z):
- **Elapsed:** 3.60 h of the 8 research hours (start 16:37:33Z).
- **CPU:** 3.077 CPU-h of 12 charged.
  - Queue units: 2.809 CPU-h (10,113 CPU-s) in 84 units, 83 done and 1 failed. The failure is the reader-custody attempt on a
    composite recipe (deviation 10). By stage: reader grids and readers 9,272, fits 527, verification 180, deployment checks 62,
    custody 37, evaluation 18, claims 17.
  - Out-of-queue charges: 0.268 CPU-h (965 CPU-s) in 14 charges, all upper estimates. These cover the designers (288), the
    coordinator (510), the verifier (95), the challenger (16) and the reviewers (56).
- **Paid compute:** $0. At most two heavy jobs at a time, one thread each.
- **Configurations:** 48 of 48 (DA 22, DB 26; candidates 8, controls 30, ablations 8, twin 2).
- **Agents:** at most six live at once, including the coordinator. Over the campaign they were the coordinator, two designers, a
  challenger, the statistical reviewer (rules, then the closing review), the verifier (two rounds) and the engineering and claims
  reviewer, which used three forked helpers.

**Access** (`ACCESS_RECEIPT.json`):
- **63 label reads, all after the verified registration receipt (17:42:42Z) and each inside its queue unit.**
  - 13 fitting reads on MODEL_FIT, each under a card committed before the read.
  - 50 reads on MODEL_SELECT: 25 utility, 22 reader, 3 verification.
- **Not read:** CALIBRATION, the ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels, the sealed assessment and the capped-out
  records. DEV_REPORT features were used label-free for deployment parity, as registered.
- **Lineage:** the DEV_REPORT lineage log is unchanged.

**Publication.** Sanitized code and aggregate records are on `publication/pcrl-dual-design-release-v1`, as a child of the
registration receipt c5a84c19d. The commit SHA and its remote verification are recorded in `PUBLICATION_RECEIPT.json`, which is
written in the research copy after publishing. The research branch stays local.

**Verification** (`VERIFICATION.md`). An independent verifier replayed every decision, robustness and strongest-release record and
every grid bank exactly, with its own code. Every check passed except one procedural FAIL: DB_ZTILT's late utility record,
deviation 5.

**Custody** (`custody/BACKUP_VERIFICATION.json`). The same-device backup was verified from the backup alone at 19:45:49Z:
- 48 models and 48 releases re-hash;
- the 10 packaged releases re-run bitwise;
- the 6 custody readers reproduce their scores bitwise;
- 1,816 reader-cache entries check out;
- the 10 deployment packages redeploy bitwise with every refusal.

Off-device custody is pending: the owner's drive (YOTUO) was not attached.
