# Finalist nomination rule (frozen before the first new fit)

**Scope.** Statistical machine-learning research on public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for
Florida, development roles only. The rule uses MODEL_SELECT development evaluations and the development record only; it never
uses DEV_REPORT, ATTACK_FIT or ATTACK_VALIDATION. **At most one finalist.** If no configuration qualifies, no finalist is
frozen, DEV_REPORT labels are not read, and the campaign delivers the measured frontier and the best diagnostic packages.

**Units.** A *configuration* is one mechanism specification fitted once per source seed (0 and 1), with only the seed changed.
Every new configuration of the campaign (`counts_toward_36` = 1, including control-designated ones) is subject to the rule.
Historical references (`counts_toward_36` = 0) are comparators, never nominees; RETIRED_B3CTL is a diagnostic baseline only.
The first complete evaluation of each (configuration, seed) counts; a re-evaluation of the same release must reproduce it
exactly, and a changed release is a new configuration with a new id.

**Inputs.** For each seed: `EVAL_MS.json` from `ihs.devrun.evaluate` (frozen evaluator `ihs.evaluate`, grid version
`IHS1-dev-proxy-v1`), the reserve screen it contains (`ihs.reserve`, registered `RESERVE_RULE.json`), and the cached reference
evaluations of units IHS_REFERENCES_S0_V1 / S1_V1 (same grid cells, same cache).

## (a) Original point eligibility: `POINT_ELIGIBLE`

On MODEL_SELECT, for both seeds: all 12 utility clauses (`utility.point_pass_12`) and both within-SEX guards
(`utility.guard_pass`) hold at the point, inclusive caps. That is the original 24 clauses and 4 guards; nothing is relaxed.
Failures are listed as `s{s}|task|reference|metric` and `s{s}|guard|sex{g}`.

## (b) Prospective utility-reserve screen: `RESERVE_MET`

`ihs.reserve.two_seed_status([screen_s0, screen_s1])['status'] == 'RESERVE_MET'` (both seeds; `UTILITY_RESERVE_RULE.md`).
Always reported with it: `min_income_ratio`, the per-seed failed conditions, conditions met only with own SEs, the occupation
side (`INHERITED`, `CHANGED_*`) and the inherited-constraint report.

## (c) Reader-relative conditions on the development proxy grid

The grid is fixed for every configuration and reference. It has 2 household-split repetitions of MODEL_SELECT (A, B) x 2 reader
seeds (0, 1) = 4 cells per source seed. In each cell, each reader is fitted on one half and scored on the other in both
directions, and the cell value is the mean AUC. The pair value is the maximum over pair readers **and** single-view readers.
The four cells share the same 8,001 people. They measure reader-fitting variability (split and initialization), not person
sampling. So every-cell positivity is a robustness requirement, not a significance test.

- **c1 `PAIR_SUPERIOR`** (gate). For each eligible reference r in {NEURAL_RAW, H_LJ} and each seed: benefit = pair(r) -
  pair(configuration) **> 0 in every cell** (16 cell-level differences, strict). A positive mean with a non-positive cell
  does not qualify.
- **c2 `RECIPIENT_SCREEN`** (gate). For each r, seed and single view (income, occupation): view max(configuration) - view
  max(r) **<= 0.010 in every cell** (16 cell-level differences).
- **c3 `GAIN_OVER_B2TWIN`** (gate). pair(B2TWIN) - pair(configuration) **> 0 in every cell on both seeds** (8 differences).
  Reason: on this grid, the existing historical 6-level LOCAL code B2TWIN already satisfies (a), (b), c1 and c2. Its cell
  minima are +0.0079 / +0.0149 against NEURAL_RAW and +0.0028 / +0.0051 against H_LJ, and its two-seed mean pair proxy is
  0.6178782. Its enhanced-reader DEV_REPORT evidence already exists in the validation bank. A finalist that does not read
  lower than B2TWIN adds nothing the lineage does not have, and would spend two more DEV_REPORT reads on it. Without c3, the
  label is `NO_GAIN_OVER_B2TWIN`.

**Always reported, never gates:**
- per (r, seed): the 4 cell benefits, their mean, min, max and range; per-reader AUCs per cell and view; the winning reader in
  each cell for the configuration and for r;
- `PAIR_NEAR_TIE` if any (r, seed) mean benefit is < 0.005. That is about the largest reference cell range on this grid
  (RETIRED_B3CTL seed 0: 0.0096) halved, and the lower end of the validation's reader-refit SD (0.003-0.011);
- `READER_UNSTABLE` if any (r, seed) cell range exceeds its mean benefit, or the winning pair reader differs between cells;
- `RECIPIENT_EDGE` with the maximum cell difference, if any recipient cell difference is > 0. The retired candidate's
  validation failed exactly here: seed-1 income vs H_LJ, point +0.0076, clause bound 0.0235 against 0.010, after a near-tie
  change of H_LJ's income winner;
- the same cell differences against the configuration's own matched task-only control and any matched component control.

## (d) Simultaneous uncertainty (reported; not a gate)

- **Utility.** For each of the 28 conditions (24 clauses and 4 guards), the one-sided upper bound value + 3.1237346 x own SE.
  The critical value is Bonferroni over 28 at one-sided 0.025 (Phi^-1(1 - 0.025/28)). The status is
  `UTILITY_CONFIDENCE_RESOLVED` when no bound exceeds its cap, otherwise `UTILITY_CONFIDENCE_UNRESOLVED (n of 28)`. These
  bounds are conditional on the fitted configuration, do not cover the adaptive search, and use an adaptively used role.
- **Readers.** The grid has no sampling SE. Only cell ranges (reader variability) are reported. Inference on recovery happens
  only in the final evaluation: household bootstrap on DEV_REPORT, with enhanced-bank readers fitted on ATTACK_FIT and
  selected on ATTACK_VALIDATION.

## Qualification and labels

`QUALIFYING` = admissible ∧ (a) ∧ (b) ∧ c1 ∧ c2 ∧ c3. Admissible means: both seeds evaluated with the frozen evaluator and
grid; the release is computed from the six permitted raw fields; it is not an alias of a reference (equal wire hashes for both
tasks on both seeds); and it is not RETIRED_B3CTL.

Each configuration gets the first applicable label:
1. `NOT_CONSIDERED_ONE_SEED_ONLY`;
2. `ALIAS_OF_<REFERENCE>`;
3. `POINT_INELIGIBLE`;
4. `RESERVE_NOT_MET`, with the two-seed reserve status. If c1 also holds, it is qualified `EDGE_PAIR_SUPERIOR_WITHOUT_RESERVE`
   (the retired candidate's profile);
5. `UTILITY_HEADROOM_ONLY`: (a) and (b) hold but c1 fails. The configuration creates task headroom without lowering pair
   recovery against both eligible references;
6. `RECIPIENT_SCREEN_FAILED`;
7. `NO_GAIN_OVER_B2TWIN`;
8. `QUALIFYING`, then `NOMINATED_FINALIST` or `QUALIFYING_NOT_CHOSEN`.

Every label carries the full condition vector, so no failure hides another.

## Choice among qualifying configurations (deterministic; at most one)

1. **Score.** The two-seed mean pair proxy: the mean of the 8 cell values, 2 seeds x 4 cells, of `max_incl_single`. Lower is
   better.
2. **Near-tie set.** Every qualifying configuration with score <= best score + 0.005.
3. **Robust reserve before point AUC.** Within the near-tie set, choose the largest `min_income_ratio` (the minimum over 12
   income conditions of margin / max(own SE, H_LJ SE)). Next, the fewest model fits (`n_model_fits`, summed over both
   seeds). Last, the earliest ledger entry of the seed-0 evaluation.
4. **Controls are not silently excluded.** A control-designated configuration competes on equal terms. If chosen, it is
   nominated with the qualifier `CONTROL_DESIGNATED_FINALIST`. Its mechanism attribution is then made against its own matched
   task-only control (`CLAIMS_DRAFT.md`, claim M). It is never dropped in favour of a lower-ranked non-control.
5. **No qualifying configuration.** No finalist. The campaign reports every configuration's label and condition vector, and
   the measured frontier: the two-seed mean pair proxy against `min_income_ratio`, with the eligible references, B2TWIN and
   RETIRED_B3CTL marked. The best diagnostic package is the lowest-score configuration that meets (a) and (b), if any.

The nominated configuration's labels, near-tie, reader-instability and recipient-edge flags, utility-confidence status and
inherited-constraint report are copied into `FINALIST_LOCK.json`.

## Calibration on the historical references (descriptive; references are never nominees)

| Release | (a) | (b) | c1 | c2 | c3 | Label it would get |
|---|---|---|---|---|---|---|
| B2TWIN | yes | RESERVE_MET | yes (minima +0.0079 / +0.0149 vs NEURAL_RAW, +0.0028 / +0.0051 vs H_LJ) | yes | n/a | would qualify; it sets the c3 bar |
| RETIRED_B3CTL | yes | RESERVE_NOT_MET | yes | yes (income vs H_LJ max -0.0123 / -0.0121) | yes | `RESERVE_NOT_MET` + `EDGE_PAIR_SUPERIOR_WITHOUT_RESERVE` |
| PUB_IBSI | no (seed-1 men's guard 0.0050142) | ONE_SEED_ONLY | yes | no (occupation vs H_LJ +0.0491 / +0.0352) | - | `POINT_INELIGIBLE` |
| H_JJ | no (income) | NOT_MET | yes | yes | yes | `POINT_INELIGIBLE` |
| H_LL, T_TREE, TASK_REFIT, B3_TASKONLY | yes | RESERVE_MET | no (they read higher) | - | - | `UTILITY_HEADROOM_ONLY` |

So the retired candidate would now stop at (b), and the bar for a new finalist is to read lower than B2TWIN in every cell
while keeping the reserve.
