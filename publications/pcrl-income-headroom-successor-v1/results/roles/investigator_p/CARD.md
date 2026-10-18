# Family P mechanism card: pair-aware learned income output with the occupation output frozen (PAIO)

Investigator P, PCRL income-headroom successor development campaign (exploratory development on previously used Florida ACS
2019 records; not confirmation). Written 2026-10-18 ~02:25Z, before any real-data fit of this family. Code: `ihs/fam_p.py`.

Context, exact terms: public-use ACS 2019 1-Year PUMS person records for Florida. The service predicts income class (2) and
occupation group (5) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding `X`). Each recipient
receives only its own float64 probability vector and argmax decision. SEX is the dataset's recorded demographic column; readers
are separate evaluation classifiers that estimate SEX from the released probability vectors (each output alone and the linked
pair).

## Exact operation that changes the release

- **Occupation output: unchanged.** The accepted JOINT_TREE E20 occupation output (T_TREE occupation routing plus the stored
  Stage B JOINT_TREE E20 occupation leaf corrections), identical to `ihs.bases.joint_occupation`. Never trained here; the fit
  asserts the trainer's occupation wire equals it (max abs difference <= 1e-10) and that its corrections never change.
- **Income output: learned.** Released income log-odds = T_TREE income teacher raw score + theta[leaves(x)].sum / T (the
  `acs_replication.leafcorr` parametrization on the frozen T_TREE income routing; theta = 0 is the teacher exactly), optionally
  followed by a 2-parameter Platt map fitted on CALIBRATION income labels (off in P1). Probability vector =
  `trees.probabilities('income', raw)`, decision = argmax. Runtime input: `X` only. The income output does not read the
  occupation output or any hidden occupation quantity at runtime.
- **Training (MODEL_FIT only).** Task loss: income cross-entropy (leafcorr recipe: AdamW lr 0.01, weight decay 0.01, batch 256,
  40 steps per epoch, 30 epochs). Privacy term: the leafcorr online SEX critic bank (linear / MLP32 / MLP64 per view; 5 critic
  rounds per update) on the **income wire** and on the **linked pair wire** = (this income wire, the frozen JOINT_TREE
  occupation wire): P = risk(income) + risk(pair), coefficient lambda = rho x median||grad CE|| / median||grad P|| at the start
  (16 fixed draws, leafcorr calibration rule), rho = 3 in P1. SEX is a training-time label for the critics only.
- **Explicit income-utility budget (label-free, new here).** After every epoch theta is scaled toward the teacher
  (theta <- s x theta, the largest s in [0, 1] by bisection) so that on the MODEL_FIT people
  - BRIER = 2 x mean((p1 - p1_T)^2) <= 0.004: the expected summed-Brier increase over the teacher if the teacher's probabilities
    were calibrated (the cross term vanishes); about 43% of what JOINT_TREE's income output spent on MODEL_SELECT (+0.0094 /
    +0.0095 over the teacher);
  - ACC = mean(decision flip x |2 p1_T - 1|) <= 0.003: expected accuracy loss under the same assumption (set at 0.0025 in
    the first draft; raised to 0.003 at about 02:27Z, before any fit, after the label-free check below showed JOINT_TREE-like
    corrections spend about as much ACC as BRIER, so 0.0025 would have bound well before the Brier budget; the teacher's
    MODEL_SELECT accuracy loss vs U0 is -0.0012 / -0.0006, so 0.003 aims at about +0.0018 / +0.0024 against the 0.010 cap);
  - KL = mean(KL(p_T || p)) <= 0.008: expected NLL increase under the same assumption.

  Design target: the teacher's MODEL_SELECT income Brier excess vs Ucal is -0.00367 / -0.00446, so the budget aims at about
  +0.0003 / -0.0005 (about 3 SE below the 0.005 cap with SE ~0.0015), leaving about 0.0015 for the cross term and for the known
  in-sample under-statement. The budget is measured against the teacher, never against labels, so the in-sample position of
  MODEL_FIT people in the teacher routing does not enter it (no internal MODEL_FIT fold is used or described as unseen data).
  Its relation to MODEL_SELECT costs is checked only through `ihs.devrun score` and the published reference values. Every
  record also logs the same label-free distortions of the stored LOCAL_TREE / JOINT_TREE / TASK_REFIT income corrections (E5,
  E10, E20), so the budget-to-MODEL_SELECT mapping can be checked against H_LJ / H_JJ without a new label read.

## Why it could improve the observed trade-off; bottleneck addressed

- The retired release reduced income SEX evidence by crude 6-level quantization, paying summed Brier everywhere (+0.0044 vs
  Ucal). JOINT_TREE's income part reads 0.595 on the pair but costs +0.0057 / +0.0050 Brier: its fixed privacy coefficient has
  no cost control, and it was trained while the occupation output was still moving.
- Here the critics see the occupation wire that is actually released, from the first step, so the pair term targets the SEX
  evidence the income output adds to the accepted occupation output. The budget fixes how much utility is spent and leaves the
  rest as reserve; inside it, the critics choose where to spend (where SEX evidence is cheap to remove) instead of quantizing
  every person's score.

**Initial version P1, two budget levels (decided about 02:29Z, before any fit, after reading the reviewer's draft reserve rule):**
`P1_PAIR_B40` (BRIER 0.004 / ACC 0.003 / KL 0.008) and `P1_PAIR_B50` (BRIER 0.005 / ACC 0.0035 / KL 0.009). The draft rule
(k = 2 SEs floored at H_LJ's) implies about income Brier excess vs U0 <= +0.0018 (seed 0) / +0.0016 (seed 1) for B2TWIN-like
own SEs, and accuracy loss <= about 0.0043; with the teacher at -0.0041 / -0.0045 (Brier vs U0) and -0.0012 / -0.0006
(accuracy), B50 is the largest level with some slack for the cross term and accuracy noise. First wave, seed 0:
`P1_PAIR_B40`, `P1_PAIR_B50`, `PC_TWIN_B40`, `PC_JSCALE_B40`.

## Simple controls that could explain away a benefit

- **PC_TWIN** (matched task-only twin): the identical loop, people, batches, epochs, critic updates and projection with
  lambda = 0. If P1's pair recovery is no lower than the twin's, the critics did nothing.
- **PC_TEMP** (matched-cost shrinkage, if P1 is markedly less confident): the twin's income log-odds / tau with tau chosen
  label-free so its BRIER distortion against the teacher equals P1's. Separates "less confident" from "less SEX-structured".
- **PC_JSCALE** (ordinary comparator at the same budget, no training): the stored JOINT_TREE E20 income corrections scaled
  toward the teacher to the same label-free budget. If it matches P1, the frozen-occupation pair training adds nothing over
  shrinking the existing JOINT_TREE income.
- Published comparators: H_LJ, H_JJ, B2TWIN (ordinary 6-level compression of LOCAL income: pair 0.6185 / 0.6173 at about
  +0.0046 / +0.0043 Brier over the teacher), RETIRED_B3CTL.

## Revise or abandon if

- Revise if the MODEL_SELECT income cost exceeds the reserve (the label-free budget under-states MODEL_SELECT cost): reset the
  budget from the logged MODEL_SELECT offset of this configuration (adaptive choice, never a fit on MODEL_SELECT), or add the
  Platt map fitted on CALIBRATION.
- Revise if the pair proxy is not below both eligible references in every development cell at a feasible cost: change the
  objective (pair-only weighting, strength) or the starting point (LOCAL_TREE / JOINT_TREE income corrections).
- Abandon if the twin, PC_JSCALE or PC_TEMP explains the reduction at matched cost, or if no budget inside the reserve
  reads below H_LJ on the pair.

## Closest known methods

- JOINT_TREE / LOCAL_TREE in this project (the transported L_TREE recipe with a pair term); family A of the source campaign
  (CCLC: both recipients, holdout utility controller). Differences: income only, occupation frozen at the accepted output,
  label-free distortion budget by projection rather than a label-based internal-holdout controller.
- Adversarial debiasing (Zhang et al. 2018), adversarial gradient tree boosting (FAGTB); constrained optimization by
  projection onto a distortion ball (privacy-utility trade-off with a distortion constraint, e.g. the privacy-funnel /
  generative adversarial privacy line).

## Invented-data check (2026-10-18 02:23Z, unit IHS_P_SELFTEST_V1, no real data)

`python -m ihs.fam_p selftest` through the queue: PASS, 17/17 checks. Zero distortion at the teacher; BRIER monotone on the
scale path; projection meets the budget, is tight, and is inactive when already within budget; the Brier identity
(increase = 2 x mean(dp^2) under calibrated labels) holds within 1.2% on 400,000 invented people; the temperature solver hits
its target; a 3-epoch training run keeps the occupation wire frozen and the budget in every epoch, lambda > 0, the twin has
lambda = 0 and differs, a rerun is bitwise deterministic, critic risks are logged, the probability contract holds and the Platt
map is finite. Wall 6.8 s, 6.0 CPU s. Re-run unchanged after adding the control configurations (IHS_P_SELFTEST_V2, 02:25Z)
and after the budget edit, label-free control loading and a chunked occupation check (IHS_P_SELFTEST_V3, 02:30Z): PASS 17/17
both times. IHS_P_SELFTEST_V4 (02:31Z) added the optional Lagrangian penalty checks (20/20 PASS; the option was never used in
a real fit) and IHS_P_SELFTEST_V5 (02:41Z) the end-projection checks used by P2 / P3 (22/22 PASS).

## Pre-fit label-free check of the budget (2026-10-18 02:26Z, unit IHS_P_DIAG_REFDIST_V1, features only)

`python -m ihs.fam_p refdist` (no label of any role read; aggregates in `REFERENCE_DISTORTIONS.json`). The label-free
BRIER distortion of stored income outputs against the teacher on MODEL_FIT people, next to the published MODEL_SELECT income
Brier increment over the teacher (vs Ucal, `REFERENCE_MS_TABLE.csv`), seed 0 / seed 1:

| Stored income output | label-free BRIER (MODEL_FIT) | published MODEL_SELECT increment | label-free KL | published NLL increment |
|---|---|---|---|---|
| JOINT_TREE E20 (H_JJ) | 0.00945 / 0.00953 | +0.00938 / +0.00945 | 0.01144 / 0.01161 | +0.01182 / +0.01191 |
| RETIRED_B3CTL | 0.00878 / 0.00890 | +0.00804 / +0.00888 | 0.01308 / 0.01339 | +0.01216 / +0.01317 |
| B2TWIN | 0.00551 / 0.00584 | +0.00462 / +0.00427 | 0.00901 / 0.00937 | +0.00790 / +0.00742 |
| LOCAL_TREE E20 (H_LJ) | 0.00137 / 0.00153 | +0.00061 / +0.00052 | 0.00167 / 0.00189 | +0.00084 / +0.00069 |

So the label-free budget predicts MODEL_SELECT Brier and NLL increments closely and, for these outputs, never under-states
them by more than 0.0007 (Brier); it does not have the in-sample under-statement of a label-based MODEL_FIT holdout. Expected
accuracy loss is noisier (JOINT_TREE E20 label-free 0.0095 / 0.0099 vs published +0.0070 / +0.0109). With BRIER <= 0.004 the
expected MODEL_SELECT income Brier excess vs Ucal is about +0.0003 / -0.0005. This check used only existing outputs and the
published table; it was run before the registration commit and changed no budget value in the card above.

## Revisions

(appended below with time, reason and result)

### First wave, seed 0 (fits 02:35-02:39Z, scores 02:37-02:40Z; ledger entries 26, 28, 30, 31)

| Config (seed 0) | label-free BRIER / ACC (MODEL_FIT) | MS income Brier vs Ucal (SE) | MS Brier increment over teacher | MS acc loss vs U0 | pair mean (cells) | income view | reserve (min ratio) |
|---|---|---|---|---|---|---|---|
| P1_PAIR_B40 | 0.00299 / 0.00299 (ACC binds) | +0.00043 (0.00127) | 0.0041 (1.37 x label-free) | +0.0021 | 0.6180 (0.6177-0.6183) | 0.5423 | MET (2.88) |
| P1_PAIR_B50 | 0.00354 / 0.00349 (ACC binds) | +0.00105 (0.00131) | 0.0047 | +0.0001 | 0.6158 (0.6150-0.6171) | 0.5399 | MET (3.01) |
| PC_TWIN_B40 | 0.00033 / 0.00022 | -0.00408 (0.00125) | -0.0004 | -0.0006 | 0.6355 (0.6345-0.6366) | 0.5581 | MET |
| PC_JSCALE_B40 (s = 0.5204) | 0.00255 / 0.00300 (ACC binds) | -0.00108 (0.00125) | 0.0026 (1.0 x) | -0.0018 | 0.6150 (0.6146-0.6155) | 0.5385 | MET (4.18) |

References (seed 0): H_LJ 0.6241, NEURAL_RAW 0.6295, B2TWIN 0.6185 (cells 0.6205 / 0.6198 / 0.6168 / 0.6168), H_JJ 0.5952.

- The critics did real work (P1 vs its twin: pair -0.0175, income view -0.0158), but P1 is **dominated by the no-training
  comparator** PC_JSCALE_B40 at the same label-free budget: lower pair (0.6150 vs 0.6180) at less MODEL_SELECT cost (Brier
  increment 0.0026 vs 0.0041). P1's MODEL_SELECT cost is 1.37 x its label-free distortion; JSCALE's (like H_JJ's) is 1.0 x.
- The ACC budget bound in all three releases, before the Brier budget.
- On seed 0, P1_PAIR_B50 and PC_JSCALE_B40 both read below H_LJ, NEURAL_RAW and B2TWIN in every cell with RESERVE_MET.

### Revision 1 -> P2_JR_B40 (about 02:42Z; ledger decision entry 33)

Reason: the card's abandon/revise condition was met for P1 as designed (a comparator explains, and beats, its reduction at a
matched budget). Hypothesis: P1's inefficiency comes from its training dynamics (rho = 3, projection after every epoch, so
the critics keep pushing against a shrinking output), not from learning against the frozen occupation. P2 uses JOINT_TREE's
own strength (rho = 1) against the frozen accepted occupation output, no per-epoch projection, one projection at the end to the
same budget (BRIER 0.004 / ACC 0.003 / KL 0.008). Same loop, people, batches and epochs as P1, so its exact matched twin is
PC_TWIN_B40 (the twin's projection never binds). Matched twins recorded: P1_* and P2_* -> PC_TWIN_B40; PC_JSCALE_B40 ->
PC_TSCALE_B40 (TASK_REFIT E20 income corrections, Stage B's identical loop with lambda = 0, under the same scaling rule).

Result (seed 0 02:43Z, seed 1 02:45Z; ledger entries 38, 39, 41, 42):

| Config | seed | label-free BRIER / ACC; end scale s | MS income Brier vs Ucal (SE) | MS Brier increment (x label-free) | MS acc loss vs U0 | pair mean (cells) | income view | reserve (min ratio) |
|---|---|---|---|---|---|---|---|---|
| P2_JR_B40 | 0 | 0.00268 / 0.00300; 0.5485 | -0.00147 (0.00126) | 0.0022 (0.82 x) | -0.0021 | 0.6133 (0.6124-0.6139) | 0.5378 | MET (4.15) |
| P2_JR_B40 | 1 | 0.00265 / 0.00298; 0.5920 | -0.00262 (0.00136) | 0.0018 (0.69 x) | -0.0014 | 0.6157 (0.6145-0.6170) | 0.5392 | MET (3.82) |
| PC_TWIN_B40 | 1 | 0.00045 / 0.00022 | -0.00520 (0.00135) | -0.0007 | +0.0009 | 0.6363 (0.6360-0.6365) | 0.5587 | MET |
| PC_JSCALE_B40 | 1 | 0.00288 / 0.00299; 0.52 | -0.00157 (0.00135) | 0.0029 | +0.0016 | 0.6169 (0.6147-0.6196) | 0.5388 | MET (2.80) |

- P2 beats the ordinary comparator PC_JSCALE_B40 on both seeds (pair -0.0017 / -0.0012 at lower MODEL_SELECT Brier) and
  P1 (seed 0: -0.0047 at 0.0019 less Brier increment). Rho = 1 with one end projection removed P1's excess out-of-sample
  cost (0.82 / 0.69 x instead of 1.37 x).
- P2 vs its matched twin: pair -0.0222 / -0.0206, income view -0.0203 / -0.0195.
- By my reading of NOMINATION_RULE.md, P2_JR_B40 meets (a), (b) RESERVE_MET on both seeds, c1 (min cell benefit vs H_LJ
  +0.0102 / +0.0069, vs NEURAL_RAW +0.0150 / +0.0170), c2 (income recipient max +0.0019 vs NEURAL_RAW, -0.0010 vs H_LJ;
  occupation identical to H_LJ) and c3 (B2TWIN minus P2 per cell: s0 +0.0074 / +0.0074 / +0.0029 / +0.0029; s1 +0.0023 /
  +0.0018 / +0.0011 / +0.0011). The seed-1 c3 margin is thin (reader variability on this grid is of that order).
- PC_JSCALE_B40 fails c3 on seed 1 (cell A1: 0.6196 vs B2TWIN 0.6188), so its matched task-only control PC_TSCALE_B40 was not
  run (no recommendation for it).

### Revision 2 -> P3_JR_B60 (02:46Z; ledger decision entry 46, whose text says "about 02:57Z" in error)

Reason: P2 qualifies with wide utility margins and a thin seed-1 c3 margin; its MODEL_SELECT cost is below its label-free
distortion and its accuracy is better than the teacher on both seeds. P3 keeps the identical deterministic training (same
seed streams; projection only at the end) and raises only the end-projection budget to BRIER 0.006 / ACC 0.0045 / KL 0.010:
a pure budget effect along the learned direction. Matched twin PC_TWIN_B40. Under the nomination rule P3 would replace P2
only if its two-seed mean pair proxy is more than 0.005 lower or its reserve ratio is larger.

Result (02:48-02:49Z; ledger 50, 51): pre-projection trajectory bitwise identical to P2 (end scale 0.6812 / 0.7385, ACC
binds). Pair 0.607159 / 0.611296 (cells 0.6069-0.6075 / 0.6091-0.6134), income view 0.532973 / 0.534940, income Brier vs
Ucal -0.000142 / -0.001356, accuracy loss -0.001250 / +0.001750, RESERVE_MET (min income ratio 3.7246 / 2.6768, binding:
seed-1 accuracy). Every cell below H_LJ (min +0.015771 / +0.011026), NEURAL_RAW (+0.020389 / +0.020616) and B2TWIN
(+0.009912 / +0.005716). Two-seed mean 0.609228 vs P2 0.614483 (difference 0.005255, outside the 0.005 near-tie window).

### Controls after revision 2 (no new revision; allotment raised 12 -> 14 by the coordinator)

- PC_TEMP_P2 (seed 0, ledger 52; tau 1.2984): pair 0.636172 (twin 0.635526). PC_TEMP_P3 (seed 0, ledger 60; tau 1.3756):
  pair 0.636261, income Brier vs Ucal +0.001145. Matched-cost temperature shrinkage does not lower pair recovery, so P2 / P3
  are not merely "less confident".
- PC_JSCALE_B60 (stored JOINT_TREE E20 income corrections scaled to P3's budget; seed 1 ledger 55, seed 0 ledger 59): pair
  0.608279 / 0.611959, income Brier vs Ucal +0.000453 / -0.000068, accuracy loss -0.001750 / +0.003250, RESERVE_MET (min
  ratio 3.4025 / 2.2122), every cell below B2TWIN. P3 minus PC_JSCALE_B60: pair -0.001120 / -0.000663 (6 of 8 cells lower),
  Brier -0.000595 / -0.001288. So most of the benefit comes from a pair-trained income direction plus the explicit
  label-free budget; training against the frozen accepted occupation adds a small, mostly consistent gain.
- Family P stops here: initial version + 2 revisions, 14 of 14 configurations. REPORT.md content was delivered in the
  hand-back message because the file-writing tool refused a report file (see the hand-back).
