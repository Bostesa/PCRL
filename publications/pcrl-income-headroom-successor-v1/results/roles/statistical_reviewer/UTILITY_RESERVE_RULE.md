# Utility-reserve rule (baseline and statistical reviewer)

**Status.** Registered before any new release fit in this campaign. Machine-readable form: `RESERVE_RULE.json` (schema
`IHS1-reserve-rule-v1`), generated from the committed clause-level table `REFERENCE_MS_TABLE.csv` by
`code/make_reserve_rule.py` (no label, no person-level array, no private file read). Implementation: `ihs/reserve.py`
(`screen(m, seed)`, `two_seed_status(screens)`); invented-data tests: `ihs/test_reserve.py` (13 of 13 pass).

**What it is.** A development selection heuristic, applied to MODEL_SELECT, a role that this campaign and earlier campaigns
have used adaptively. **It is not a confidence guarantee and it does not change any cap.** The acceptance limits remain the
original ones (accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed Brier excess <= 0.005 for both tasks, both references
U0 / Ucal and both seeds; within-SEX occupation summed-Brier excess against U0 <= 0.005). Point eligibility is judged
separately (`NOMINATION_RULE.md`, part a). The screen asks one extra question of a configuration that already passes at the
point: is its income margin wide enough that a separately authorized prospective study could plausibly confirm it?

Data context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, development roles only. The
service predicts income class (2 classes) and occupation group (5 classes) from AGEP, WKHP, COW, SCHL, MAR and POBP. SEX is
the dataset's recorded demographic column; readers estimate it from the released probability vectors.

## 1. The rule

Per seed, on the dict returned by `ihs.evaluate.measure` for MODEL_SELECT (5,306 households):

| Item | Decision |
|---|---|
| Conditions screened | The **12 income clauses** (6 per seed: references U0 and Ucal x accuracy loss, NLL excess, summed Brier excess) |
| Condition | margin = cap - value **>= k x se_used** (inclusive). A non-finite value or SE fails the condition |
| Multiplier | **k = 2.0**, applied to SEs at the MODEL_SELECT scale |
| SE | **se_used = max(own SE, floor SE)**. Own SE: the configuration's household-clustered paired SE on MODEL_SELECT (`ihs.evaluate.cluster_se`). Floor SE: the eligible reference **H_LJ's MODEL_SELECT SE for the same clause and seed** |
| Household scale | No projection: SEs are used at the 5,306-household scale at which they are measured. If an evaluation has G households, the floor is rescaled by sqrt(5306 / G). The 17,510-household scale enters through the multiplier (section 2.3): k = 2 at MODEL_SELECT equals **3.633 SEs at 17,510 households** and 1.727 SEs at DEV_REPORT's 3,955 |
| Occupation clauses and within-SEX guards | **Inherited output** (all 6 occupation clause values and 2 guard values equal H_LJ's to 1e-12): **not screened**; returned as an inherited-constraint report (`INHERITED_CONSTRAINTS_BIND` when any of them is below the reserve). **Changed output:** screened. Each of the 6 clauses and 2 guards must either meet the same reserve (floor: H_LJ's SE) or keep at least the inherited output's margin (non-worsening); otherwise it is a failed condition |
| Status per seed | `RESERVE_MET`: all 6 income conditions met and no occupation condition worsened. `RESERVE_NOT_MET` otherwise; `failed` names each failing condition |
| Status of a configuration (two seeds) | `RESERVE_MET` only if seeds 0 and 1 are both `RESERVE_MET`. `RESERVE_ONE_SEED_ONLY`: both screened, exactly one met. `RESERVE_NOT_MET`: no screened seed met. `RESERVE_PENDING_SECOND_SEED`: one seed screened and met. **Only `RESERVE_MET` meets the reserve** |
| Ranking quantity | `min_income_ratio`: the minimum over the 12 income conditions of margin / se_used (floored SE units) |

Implied absolute minimum income margins (k x floor; a configuration with a larger own SE needs more):

| Income clause | Seed 0 | Seed 1 |
|---|---|---|
| accuracy loss (U0 = Ucal) | 0.0056250 (loss <= 0.0043750) | 0.0057655 (<= 0.0042345) |
| NLL excess vs U0 | 0.0035351 (<= 0.0064649) | 0.0036641 (<= 0.0063359) |
| NLL excess vs Ucal | 0.0031034 (<= 0.0068966) | 0.0033425 (<= 0.0066575) |
| Brier excess vs U0 | 0.0027109 (<= 0.0022891) | 0.0028245 (<= 0.0021755) |
| Brier excess vs Ucal | 0.0024846 (<= 0.0025154) | 0.0026557 (<= 0.0023443) |

For a coarse income code whose own SEs resemble B2TWIN's (Brier vs U0 0.0015861 / 0.0016821), the binding requirement is
income Brier excess vs U0 <= 0.0018278 (seed 0) and <= 0.0016358 (seed 1).

## 2. Why these choices

### 2.1 Which clauses

- **Income (screened).** The retired candidate failed on income: MODEL_SELECT income Brier excess vs Ucal +0.0043657
  (SE 0.0015763) and +0.0044174 (SE 0.0016700), margins 0.40 and 0.35 SE; on DEV_REPORT 0.00492 / 0.00494, 0.03-0.04 SE of
  slack. That is the bottleneck this campaign addresses, and income-side mechanisms can change it.
- **Occupation clauses and guards (reported; screened only when changed).** The inherited H_LJ / JOINT_TREE occupation output
  is fixed for every income-side mechanism. Its men's guard sits at 0.32 / 0.62 MODEL_SELECT SE from the cap (0.58 / 1.13 SE
  even projected to 17,510 households; `INHERITED_OCCUPATION_MARGINS.json`). Screening it would fail every configuration that
  keeps it, whatever its income. So it is reported for every configuration (`inherited_constraint_report`), carried into the
  claims as a separate conclusion (claim O, `CLAIMS_DRAFT.md`), and never hidden. A family that changes the occupation output
  may not make any occupation condition worse than the inherited one unless that condition meets the reserve.

### 2.2 Which SE: a floor combining own and reference SEs

- **Own SE alone is not enough.** A configuration that barely differs from U0 or Ucal has a tiny paired SE, so "2 own SEs" can
  allow a value close to the cap. A difference concentrated on few households also gives an unreliable SE. Own SE is also
  silent about the adaptive use of MODEL_SELECT.
- **A fixed reference SE alone is not enough either.** A configuration whose income output moves further from U0 than any
  reference has a larger own SE, and would be screened too leniently.
- **Floor = H_LJ's SE.** H_LJ is the eligible reference that carries the inherited occupation output. Its income output (a
  tree-based LOCAL income) differs materially from U0. Every recovery-lowering income output in the table has an own SE at
  least as large: B2TWIN, RETIRED_B3CTL, H_JJ and PUB_IBSI have Brier-vs-Ucal SEs of 0.00145-0.00167, against H_LJ's
  0.0012423 / 0.0013278. Across the table, the floor exceeds the own SE by more than 1% only for outputs close to U0 (U0,
  Ucal and NEURAL_RAW, by factors 1.37-3.84), which is where own SEs mislead; elsewhere by at most 0.84% (accuracy clauses of
  TASK_REFIT and B3_TASKONLY on seed 0, TASK_REFIT's seed-1 NLL vs U0).

### 2.3 Which multiplier and scale

The comparison the reserve is meant to protect is a future, separately authorized confirmatory test of the frozen release on
the only same-population Florida 2019 people never used by any model or reader: about 17,510 households (sealed 20%, 9,853,
plus capped-out, 7,657; disclosure D2). Planning arithmetic per clause (`RESERVE_RULE.json` `derivation`; normal
approximation, households iid from the same population, unweighted, flat prior, no selection adjustment):

| Quantity | Value |
|---|---|
| SE at 17,510 households / SE at 5,306 | sqrt(5306 / 17510) = 0.5504787 |
| Plug-in true margin needed for a one-sided 5% bound <= cap with 80% power at 17,510 | (1.6448536 + 0.8416212) x 0.5504787 = **1.3687515 MODEL_SELECT SEs** |
| k = 2 leaves, for development noise and selection on adaptively used data | 0.6312485 MODEL_SELECT SEs |
| Flat-prior probability that the true margin is positive, at margin = 2 SE | 0.9772499 |
| ... that the true margin meets the 17,510 requirement above | 0.7360610 |
| Predictive probability of a point pass in a new same-population sample of DEV_REPORT's size (3,955) | 0.9043930 |
| ... of a one-sided 5% bound <= cap at 17,510 households | 0.8311859 |

The 12 income clauses are strongly correlated (accuracy vs U0 equals accuracy vs Ucal; NLL and Brier move together), so joint
probabilities are lower when several clauses sit at the threshold. Selection on MODEL_SELECT biases the selected estimate
towards larger margins; the 0.63 SE allowance is a heuristic for that, not a correction.

**Rejected alternative: 2 SEs at the 17,510 scale (1.1009575 MODEL_SELECT SEs).** It is too lenient for the purpose. At its
threshold the plug-in power of a one-sided 5% bound at 17,510 is 0.6387600 per clause, the predictive probability of a
DEV-size point pass 0.7640754, and of a 17,510 bound 0.5679927. It would accept a candidate about as fragile as the retired one
became on DEV_REPORT. (It gives the same verdicts on the references below, so the choice does not tilt any comparison.)

## 3. What the screen is not

- Not a confidence statement. MODEL_SELECT is a development role, used adaptively in this campaign and by its predecessors
  (the retired candidate's family was selected on it).
- Not a changed cap. A configuration that fails the screen still passes or fails the original 24 clauses and 4 guards at the
  point exactly as before.
- Not a statement about DEV_REPORT, which has been read 12 times in the lineage and is not fresh.
- Not a substitute for the final reader evaluation or for a prospective study.

## 4. Every historical reference on this screen (MODEL_SELECT, both seeds)

From `RESERVE_RULE.json` `reference_screen` (min ratio = minimum income margin / se_used; "floor" means H_LJ's SE was larger
than the release's own).

| Reference | Seed 0 | Seed 1 | Configuration status | Own SEs only | Occupation side |
|---|---|---|---|---|---|
| U0 | MET, 3.5556 (accuracy, floor) | MET, 3.4689 (accuracy, floor) | RESERVE_MET | RESERVE_MET | changed, reserve met |
| Ucal | MET, 3.5556 (floor) | MET, 3.4689 (floor) | RESERVE_MET | RESERVE_MET | changed, reserve met |
| T_TREE | MET, 3.8733 | MET, 3.5761 | RESERVE_MET | RESERVE_MET | changed, reserve met |
| TASK_REFIT | MET, 4.0000 (floor) | MET, 2.9559 | RESERVE_MET | RESERVE_MET | changed, reserve met |
| **NEURAL_RAW** (eligible) | MET, 2.6545 (Brier vs Ucal, floor) | **NOT MET**, 1.5575: accuracy loss vs U0 and Ucal, Brier vs U0 and Ucal (floor) | **RESERVE_ONE_SEED_ONLY** | RESERVE_MET (min 2.2566) | changed, non-worsening |
| **H_LJ** (eligible) | MET, 5.5553 | MET, 4.1626 | **RESERVE_MET** | RESERVE_MET | inherited (constraints bind: men's guard) |
| H_JJ | NOT MET, -0.4499: accuracy x2, Brier x2, NLL vs Ucal | NOT MET, -0.0750: accuracy x2, Brier x2 | RESERVE_NOT_MET | RESERVE_NOT_MET | inherited |
| H_LL | MET, 5.5553 | MET, 4.1626 | RESERVE_MET | RESERVE_MET | changed, non-worsening |
| **RETIRED_B3CTL** | NOT MET, 0.4024: NLL and Brier vs U0 and Ucal | NOT MET, 0.3489: same 4 | **RESERVE_NOT_MET** | RESERVE_NOT_MET | inherited (constraints bind) |
| B3_TASKONLY | MET, 2.0030 (Brier vs Ucal) | MET, 2.2618 | RESERVE_MET (by 0.003 SE on seed 0) | RESERVE_MET | changed, reserve met |
| **B2TWIN** | MET, 2.7978 (Brier vs Ucal) | MET, 3.1148 (Brier vs U0) | **RESERVE_MET** | RESERVE_MET | inherited (constraints bind) |
| PUB_IBSI | MET, 2.3866 | NOT MET: men's guard 0.0050142 > 0.005 (worse than inherited) | RESERVE_ONE_SEED_ONLY | RESERVE_ONE_SEED_ONLY | changed; worsened on seed 1 |

What this implies:

1. **One eligible reference meets the screen, one does not.** H_LJ meets it on both seeds with wide margins (min 4.16).
   NEURAL_RAW meets it on seed 0 only: its seed-1 income accuracy loss (0.0052493) and Brier excesses (0.0028003 vs U0,
   0.0028496 vs Ucal) are 1.56-1.65 floored SEs from the caps. With its own small SEs it would meet the screen (min 2.26).
   This is the floor doing its job: a successor with NEURAL_RAW's seed-1 income quality would not count as having a robust
   reserve. It does not change NEURAL_RAW's standing as an eligible comparator, which rests on the original caps.
2. **The screen catches the retired candidate on MODEL_SELECT,** on both seeds and 4 conditions each (NLL and Brier against
   both references; min 0.40 / 0.35), before any DEV_REPORT read.
3. **The headroom is real but narrow.** The retired candidate's matched 6-level LOCAL code, B2TWIN, meets the screen (min
   2.80 / 3.11). The retired candidate's MIX50 component (half of the LOCAL_TREE income statistic replaced by the
   replication's pair-aware JOINT_TREE statistic) spent 0.0034169 (seed 0) and 0.0046075 (seed 1) more income Brier
   vs U0 than B2TWIN. A successor with B2TWIN-like SEs may spend only about 0.0013 (seed 0) and 0.0019 (seed 1) beyond B2TWIN's
   income Brier excess vs U0. To earn a pair benefit like the retired candidate's, it must do so at well under half the
   retired candidate's income cost.
4. **B3_TASKONLY passes by 0.003 SE on seed 0.** It is a reminder that the threshold is sharp: near-threshold values carry
   ordinary sampling noise and the near-tie reporting of `NOMINATION_RULE.md` applies.
5. **PUB_IBSI fails on seed 1 because of its own occupation output** (men's guard above the cap), not because of income.
6. **The inherited occupation constraints bind for every configuration that keeps the inherited output** (B2TWIN, H_LJ, H_JJ,
   RETIRED_B3CTL here): the men's guard is below the reserve on both seeds. No income reserve fixes this. It is reported for
   every such configuration and is a separate conclusion in `CLAIMS_DRAFT.md` (claim O).

## 5. Use in this campaign

- `ihs.devrun.evaluate` calls `ihs.reserve.screen(m, seed)` for every development evaluation; the ledger records
  `reserve_status` and `reserve_failed`. The configuration status is `two_seed_status` of the two per-seed screens.
- The registered rule file is `RESERVE_RULE.json` (its SHA-256 is recorded in every screen result as `rule_sha256`). Any
  change after registration is a protocol deviation.
- Value identity is used to recognise the inherited occupation output (no private wire hash was read for this rule). The
  rule supports a recorded inherited wire hash, but none is recorded, so identity is by values to 1e-12.
- Reproduce: `OMP_NUM_THREADS=1 python3 results/pcrl_income_headroom_successor_v1/roles/statistical_reviewer/code/make_reserve_rule.py`
  then `OMP_NUM_THREADS=1 python3 -m ihs.test_reserve`.
