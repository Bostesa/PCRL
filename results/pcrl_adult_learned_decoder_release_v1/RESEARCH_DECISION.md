# Research decision — Adult learned decoder and constrained release (lra)

**Label: CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION**
[A=NOT_ESTABLISHED; B=NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE; C=NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE; Q=PASS]

## Three simple answers

1. **Did learning the probabilities improve Adult confidence on the SAME codes? No: it made held-out confidence
   worse.**
   - On the fitting rows (OSF_DEFENSE_FIT), the learned decoder D1 lowered occupation log loss by about 0.026 nats on
     identical tokens. That is in-sample: the frozen U heads were trained on these rows, and D1 calibrates against them.
   - On held-out rows it raised occupation log loss. Inner rows, mean over all 81 fixed maps × seeds: +0.014 nats. The
     assessment's same-map contrasts, D1 − D0 (nominal 95%):

     | Map | Occupation LL | Income LL | Brier |
     |---|---|---|---|
     | Inner-named best fixed-map control (JOINT λ0.01) | +0.0102 [0.0081, 0.0123] | +0.0042 [0.0033, 0.0052] | both worse |
     | C-TASK | +0.0238 [0.0209, 0.0267] | | |
     | CLASS | +0.0010 [0.0006, 0.0014] | | |

   - All twelve contrasts are positive, with lower bounds above 0.
   - The tokens are identical, so the information in these codes is unchanged; this is a utility result only.
2. **Did any useful private release meet the original full criterion? No.**
   - The strongest candidate, P\*, is the EXISTING label-blind mean-decoder JOINT λ0.1 map; no new construction.
   - Against T\* (FINE-TASK) it lowers pair AUC by 0.0336 [0.0286, 0.0386], and 10 of 11 clauses pass.
   - It fails only occupation log-loss preservation: excess over U 0.0081, upper bound 0.0121 > 0.01. That is
     unresolved preservation (NOT_ESTABLISHED_PRECISION), not an established violation. It reproduces qpc's earlier
     finding for the same map (0.0081, upper bound 0.0121).
3. **Did new constrained or paired joint search add anything beyond the strongest controls? No.**
   - No D1 release (0 of 57) and no constrained arm was ordinarily eligible on inner rows. N\* and J\* have no eligible
     nominee.
   - Descriptively, the constrained fallbacks lower pair AUC a further ~0.02 below C\* (K-SEQ-21 0.7947 vs C\* SEQ-21
     λ0.1 0.8157). But they violate occupation confidence outright: log-loss excess 0.048–0.061, MEASURED_VIOLATION.
   - Paired joint ties single-move joint: K-JOINT-PAIR 0.7922 vs K-JOINT-SINGLE 0.7914.

**Scope.**
- A decoder-only utility change is not information removal.
- A fixture result is not an Adult result.
- An Adult development result is not confirmation.
- A useful calibrated existing code would not be a newly invented algorithm. Here, the best release is an existing,
  uncalibrated map.
- Fewer leaking finite attackers is not a population privacy guarantee.

| Item | Value |
|---|---|
| Branch | `research/pcrl-adult-learned-decoder-release-v1`, from the lcr tip 091afc2 (evidence 418e529) |
| SOURCE_ADMISSION_LOCK | `3090d83e5` (committed 2026-10-07T00:11:31-04:00); the admission ran afterwards, at 04:11:40Z |
| CORRECTNESS_LOCK | `de4495f` (pushed 05:07:48Z); correctness stage 05:07:54–05:08:58Z: ENGINEERING_READY |
| SCIENCE_LOCK | `15bb242` (pushed 05:22:30Z) |
| EVALUATION_LOCK | `186afb6` (pushed 07:47:30Z) |
| Inference evidence | `5b4beff` |

## 1. Stages that ran, with work counts

| Stage | Window (UTC) | Work |
|---|---|---|
| Admission | 04:11Z | 189 custody units by verified copy, matching cbp hashes and the lcr manifest; 6/6 teachers and 81/81 codes bitwise |
| Correctness gate | 05:07–05:09Z | 4 pinned lcr fixture laws × 30 mapper units; 2,043 accepted states replayed from persisted traces; 12/12 checks → ENGINEERING_READY. Role E reproduced it independently. |
| `d1` | 05:22:39–05:23:04Z | 81 fixed-map D1 decoder units: 26 per seed as registered, plus CLASS\|D1 (a registered addition) |
| `ctask` | –05:23:24Z | 3 C-TASK fits, all feasible on the fitting budgets |
| `fit` | 05:23:32–05:41:51Z | 72 weighted + 15 constrained mapping-pair fits, all FEASIBLE. 0 evaluation-ceiling hits; paired moves accepted 12/44/43 by seed. Every fit has a persisted trace. |
| `inner` | 05:42–07:15Z | 261 code and reference inner audits |
| `inner_src` | –07:19Z | 6 composed source audits, over 84 codes per seed |
| `controls` | 07:16–07:37Z | real-data controls (null, confidence, collision, XOR, rotation): all pass |
| `select` | 07:21–07:23Z | inner-only roles; 0 technical failures |
| `d0same` | –07:37Z | 3 same-map D0 C-TASK units |
| Assessment | 07:47:40–08:31:03Z | 63 outer units (21 scored labels × 3 seeds) |
| `infer` | 08:31–08:32Z | 37 primary slots; 1,999 group-bootstrap draws (seed 20261010) |

- **Totals.** 171 new decoder units (81 + 90) and 267 inner audits; 4 correctness fixture units and 3 same-map D0 units.
- **Pruning.** None; the full mandatory bank ran.

## 2. Claims (PRIMARY_ENDPOINTS.csv)

| Claim | Nominee vs comparator | Status | Exact failed clause(s) |
|---|---|---|---|
| A | P\* = D0 JOINT λ0.1 vs T\* = D0 FINE-TASK | NOT_ESTABLISHED | P07, occupation LL − U: 0.0081 [0.0040, 0.0121], upper bound ≥ 0.01 (precision) |
| B | N\* (no eligible nominee; fallback K-SEQ-21 D1) vs C\* = D0 SEQ-21 λ0.1 | NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE | ineligible on inner (ORDINARY_UTILITY_FAILURE). Descriptive: P12 pair gain 0.0209 [0.0167, 0.0252] (precision); P17 income LL 0.0102 (point); P18 occupation LL 0.0478 (violation); P19 income Brier (precision); P20 occupation Brier 0.018 (violation) |
| C | J\* (no eligible nominee; fallback K-JOINT-PAIR D1) vs C_pair\* = D0 SEQ-21 λ0.1 | NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE | ineligible on inner. Descriptive: P23 pair gain 0.0234 [0.0191, 0.0278] (precision); P28 income LL (point); P29 occupation LL 0.0538 (violation); P30 (precision); P31 occupation Brier (violation) |
| Q | D0 DIRECT-TASK i8o64 | PASS | none: LL excess income 0.0008 / occupation 0.0030; Brier excess 0.0005 / 0.0017; all four upper bounds within limits |

Notes:
- Inference uses z = 3.2048452050105634 (37 slots) on 1,999 paired exact-record-group bootstrap replicates.
- Nominal intervals condition on the fitted artifacts and do not correct for the adaptive research history
  (EXPOSURE_LEDGER.md).

## 3. Absolute assessment levels (mean over seeds; ALL_LEVELS.csv)

| Release | Pair AUC | Income AUC | Occupation AUC | Acc income / occupation | LL income / occupation | Brier income / occupation |
|---|---|---|---|---|---|---|
| U (continuous source) | 0.8585 | 0.6972 | 0.8563 | 0.8440 / 0.4754 | 0.3382 / 1.2686 | 0.2141 / 0.6524 |
| Q = D0 DIRECT-TASK | 0.8491 | 0.6957 | 0.8342 | 0.8440 / 0.4754 | 0.3390 / 1.2715 | 0.2146 / 0.6540 |
| T\* = D0 FINE-TASK | 0.8462 | 0.6957 | 0.8310 | same | 0.3393 / 1.2716 | 0.2149 / 0.6538 |
| **P\* = D0 JOINT λ0.1** | **0.8125** | 0.6948 | 0.7823 | same | 0.3396 / 1.2766 | 0.2147 / 0.6553 |
| C\* = D0 SEQ-21 λ0.1 | 0.8157 | 0.6945 | 0.7711 | same | 0.3400 / 1.2770 | 0.2150 / 0.6556 |
| C-TASK (D1) | 0.8327 | 0.6952 | 0.8101 | same | 0.3457 / 1.3074 | 0.2166 / 0.6651 |
| C-TASK, same map, D0 | 0.8337 | 0.6952 | 0.8097 | same | 0.3407 / 1.2836 | 0.2156 / 0.6583 |
| K-SEQ-21 (N\* fallback) | 0.7947 | 0.6916 | 0.7133 | same | 0.3484 / 1.3164 | 0.2185 / 0.6703 |
| K-JOINT-PAIR (J\* fallback) | 0.7922 | 0.6912 | 0.7270 | same | 0.3490 / 1.3224 | 0.2187 / 0.6726 |
| K-JOINT-SINGLE | 0.7914 | 0.6908 | 0.7210 | same | 0.3483 / 1.3230 | 0.2185 / 0.6729 |
| K-LOCAL | 0.8105 | 0.6867 | 0.7296 | same | 0.3507 / 1.3297 | 0.2185 / 0.6745 |
| CLASS (decisions alone, D0) | 0.7391 | 0.5871 | 0.6866 | same | 0.4228 / 1.3787 | 0.2573 / 0.6934 |
| RAW-J β0.3 | 0.7884 | 0.6846 | 0.7743 | 0.8452 / 0.4665 | 0.3347 / 1.2820 | 0.2125 / 0.6589 |
| FARE | 0.7042 | 0.6851 | 0.6356 | 0.8441 / 0.4509 | 0.3515 / 1.3145 | 0.2240 / 0.6757 |

Every class-preserving code keeps U's decisions, which is why the accuracies are identical. The fitting-majority
constant accuracies are 0.7458 (income) and 0.2842 (occupation).

## 4. Attribution

- **Calibrated old map vs new assignment search.**
  - Learning the probabilities on unchanged maps worsened held-out confidence (answer 1). The calibrated old maps were
    therefore never eligible.
  - The label-supervised assignment search also overfit. C-TASK's map decoded with the old mean decoder has
    occupation excess 0.0151, against 0.0031 for the label-blind FINE-TASK map. Adding D1 raises it to 0.0388.
  - The best release is an old, uncalibrated, label-blind map.
- **Constrained vs weighted.**
  - Both fail held-out confidence: every D1 release is ineligible.
  - The best weighted control (W-SEQ-21 λ0.01, D1) has pair AUC 0.8319 and occupation excess 0.040.
  - The constrained arms reach lower pair AUC (0.79–0.81) at larger confidence costs (0.048–0.061).
  - The hard fitting budgets were met in-sample on every seed. They did not transfer to held-out rows.
- **Paired joint vs sequential and single-move.**
  - K-JOINT-PAIR 0.7922 ≈ K-JOINT-SINGLE 0.7914 ≈ K-SEQ-21 0.7947 ≈ K-SEQ-12 0.7963. There is no joint-specific gain.
  - The pair step accepted 99 paired moves (the verifier replayed every one); they changed nothing material.
- **Decision floor (DECISION_FLOOR_AND_FEASIBILITY.csv).**
  - CLASS is far from eligible under either decoder: inner occupation LL excess about +0.09, fitting slack about −0.13.
  - So CLASS did not sit at T\*, and claim A was a genuine comparison against FINE-TASK.

## 5. Strongest favourable and strongest adverse result

- **Favourable.**
  - The existing D0 JOINT λ0.1 map cuts pair AUC by 0.0336 [0.0286, 0.0386] below the strongest eligible task-only
    release.
  - It keeps decisions identical, and income LL/Brier and occupation Brier within bounds.
  - It misses only the occupation LL upper bound (0.0121 vs 0.01).
- **Adverse.**
  - Learning the probabilities, the study's central mechanism, made held-out confidence significantly worse on every
    named same-token contrast. On C-TASK: +0.024 [0.021, 0.027] nats occupation.
  - Every new learned or constrained release was ineligible for that reason.

## 6. Predictions (PREDICTIONS.json, registered 04:15:14Z), scored

| ID | Forecast | Outcome |
|---|---|---|
| LA1 | ENGINEERING_READY (0.85) | hit |
| LA2 | D1 lowers held-out LL on unchanged maps (0.6) | miss: it raised it |
| LA3 | A passes (0.12) | did not happen |
| LA4 | B passes (0.04) | did not happen |
| LA5 | C passes (0.02) | did not happen |
| LA6 | Q passes (0.8) | hit |
| LA7 | winner construction | an existing D0 map (forecast 0.10) |
| LA8 | CLASS eligible (0.1) | did not happen |
| LA9 | overall label | CONFIDENCE_FEASIBILITY (forecast 0.6) |

## 7. What is runnable

- **Deployment** (`python -m lra.deploy`; DEPLOYMENT_RECEIPT.json). On seed 1, five releases deploy bitwise equal to
  their stored releases: Q, P\*, C-TASK D1, K-SEQ-21 D1, and the best fixed-map D1 control. Nine bad inputs are
  refused with exit 2.
- **Commands.** QUICKSTART.md lists only executed commands.
- **Release status.**
  - P\* is a usable development-benchmark release that keeps decisions identical and most confidence.
  - It does NOT meet the full criterion, because occupation log-loss preservation is unresolved.
  - It is an existing map (qpc/cbp lineage), not a new construction.

## 8. Verification, cost, custody

- **Independent verifier** (role E; INDEPENDENT_VERIFICATION.json): phases 0, 1 and 2 PASS; phase 3 is recorded
  there.
- **Tests:** 284 pass before the locks (VALIDATION.md).
- **Compute:** about 6.8 CPU-h at the inference point, with at most 2 concurrent processes; $0 cloud. Final figures
  are in COST_AND_CLOSEOUT.md.
- **Custody:** BACKUP_VERIFICATION.json and RESTORE_INDEX.json, written by role F after the assessment opening.

## 9. Decision

- **Close this recipe.** Per prompt §17, a valid full loss means closing this exact decoder/constrained recipe. No
  further small λ or κ adjustment is recommended on this assessment.
- **Why it lost.** The learned decoder and label-supervised assignments overfit the in-sample fitting rows, because the
  frozen heads were trained there. Held-out confidence got worse, not better.
- **If calibration is revisited:** it would need held-out calibration rows the heads never saw. That is a new design
  question for a new registration, not this one.
- **Confirmation.** No confirmation population is spent: no development claim passed.
