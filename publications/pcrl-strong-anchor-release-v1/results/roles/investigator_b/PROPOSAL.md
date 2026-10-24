# Investigator B proposal: pair Fisher-discriminant training of the two U0 networks (FB_PFD)

Written 2026-10-24 from 04:22Z to about 04:55Z (`date -u`). Label-free: no label file of any role was opened and nothing new was
computed from MODEL_SELECT. The evidence used is committed records (`results/`), the replication's stored training logs and
calibration record (`NEURAL_RAW_<v>/NEURAL_RAW__rho*__LOGS.json`, `NEURAL_RAW_SEED.json`) and frozen release arrays on MODEL_FIT and
CALIBRATION (`check_release_geometry.py` / `.json` in this directory). Code: `sar/fb/pfd.py`, deployment exports in `sar/fb/__init__.py`.

**Addendum, 2026-10-24 05:04Z (coordinator adjudication: PROCEED, with required changes).** The control structure of section 5 is
superseded by `CARD_DRAFT.md` item 7, as implemented in `sar/fb/pfd.py`:
- CAND FB_PFD and CONT FB_PFD_TASK (gating) are unchanged.
- CTRL FB_PFD_DQ_LO / _HI is the required untargeted control: SEX-blind dithered quantization of NEURAL_RAW, with its basis on the
  cross-fitted RAWJ-E2 outputs.
- ALT FB_PFD_RAWJ_LO / _HI is NEURAL_RAW's own recipe continued beyond E2 (grid E3..E10). It replaces the single matched point.
- Both paths have origin START = NEURAL_RAW and use the bracket rule on MODEL_FIT out-of-fold total-NLL spends over START.
- At most 12 specifications.

The SPEND path is now built (CTRL). SBC and RO stay not built: both are constructible (PFD with the PSEUDO2 attribute, or with a
release-only SEX surrogate, in place of SEX) and are left out for the specification budget. This corrects the reason given in section
5. The registered predictions (section 9) are unchanged. The added predictions for the new controls are in the card's dated note.

## 1. Starting release and the condition addressed

**NEURAL_RAW** (U0 networks d-64-64-16-K per task, continued from U0 with online SEX critics, joint objective risk(income) +
risk(occupation) + risk(pair), lambda_J calibrated per recipient, rho 1, checkpoint E2). IPR1_MAIN: pair 0.6253 = occupation view 0.6253,
income view 0.5336 (per version 0.5334 / 0.5338); U0 pair 0.6383. Point margins: max 0.34 / 0.57 of cap.

**What stops NEURAL_RAW's own path is income utility, not the coalition's recovery.** From the replication's own records:

| Fact | Record |
|---|---|
| Every stronger point fails an INCOME clause first: rho 1 E5 (proxy pair 0.6065) fails only version-1 income Brier vs U0 and Ucal (max 1.059 of cap); rho 3 E2 and rho 10 E2 fail income Brier / accuracy first. Their occupation costs are moderate (rho 1 E5: occupation NLL +0.0048 / +0.0045, Brier +0.0018 / +0.0013). | `results/pcrl_acs_protected_tree_replication_v1/SELECTION_DETAIL.json` |
| The income-view critics read almost nothing (logged risk(income) 0.0000-0.0045, most steps exactly 0, i.e. no critic beats the constant), yet the income network's weighted privacy gradient is 1.1-2.1x its task gradient (v0 step 60: 0.557 vs 0.262). | `NEURAL_RAW__rho1.0__LOGS.json` (both versions) |
| Per-recipient calibration gives income about twice the occupation multiplier: lambda_J 1.57 / 0.88 (v0), 1.82 / 0.77 (v1). | `NEURAL_RAW_SEED.json` |
| U0's income logit difference is a strong proxy of the occupation coordinates: R^2 of income log-odds on the 4 occupation log-ratios 0.68-0.70 (linear), 0.73-0.74 (quadratic), MODEL_FIT and CALIBRATION, both versions. | `check_release_geometry.json` (label-free) |
| Along NEURAL_RAW's path the income output moves as much as or more than the occupation output (RMS change in U0-SD units; v1 E2 0.113 vs 0.071, E10 0.243 vs 0.142; v0 E2 0.092 vs 0.081). | `check_release_geometry.json` |
| As occupation protection deepens the pair critic finds more than the occupation critic (logged risk at step 780, rho 1: pair 0.0275 vs occupation 0.0055 in v0; 0.0169 vs 0.0115 in v1): the coalition becomes super-additive, so income cannot simply be left alone. | logs |

Reading (a hypothesis, not established): the pair critics are early-stopped SGD critics on diagonally normalized wires; to first order
they weight each coordinate by its marginal mean difference, so a coordinate that merely proxies another recipient's signal (the
income log-odds, 70% predictable from the occupation output) receives critic weight and its network is pushed, while the per-recipient
calibration doubles that push. The protection that matters for the coalition (the occupation view) is therefore bought with the income
utility budget, which binds first. **Condition addressed: the coalition's recovery under the income-utility limit of NEURAL_RAW's path**:
spend the unused occupation margin (occupation NLL at 0.04 / 0.13 of cap at E2) on the linked pair, give the income recipient only its
incremental share of the pair plus what its own view needs (the recipient condition against NEURAL_RAW requires income view <= 0.5434 /
0.5438; U0's is 0.5476 / 0.5526, so some income protection is required).

## 2. The operation (one)

Both task networks start from U0_S<v> and are trained (NEURAL_RAW's optimizer and task rows: Adam 1e-4, clip 5 per network, 40 steps of
256 rows per epoch from the THP1task stream) with

    loss = CE_income + CE_occupation + lambda * P,    P = sum_{v in income view, occupation view, pair} [Phi(D_v / sqrt 2) - 1/2]
    D_v^2 = Delta_v' (S_v + eps_v I)^-1 Delta_v,

phi_v = degree-2 polynomial features of the view's logit differences (income z1 - z0: 2 features; occupation z_k - z0: 14; pair: 20),
Delta_v = women-minus-men mean of phi_v and S_v the pooled within-SEX covariance on a 2,048-row MODEL_FIT privacy batch, eps_v = 1e-3
tr(S_v)/p_v. Phi(D/sqrt 2) is the AUC of the Fisher discriminant under Gaussian class conditionals: P is the summed AUC of the exact
in-batch best quadratic-discriminant reader of each view and of the coalition. ONE coefficient lambda = rho x median ||grad of both task
CEs|| / median ||grad P|| (16 fixed draws at U0), rho = 1. Strength path: checkpoints E1 ... E10. Deployment: a float64 numpy forward
pass of the saved weights (python3; SEX never a runtime input; deterministic; one release per person).

**Why it targets the pair and keeps useful probabilities.** (i) D_pair bounds both views' D (a sub-vector's Mahalanobis distance cannot
exceed the full vector's), so the coalition term carries the target and the view terms only add the recipient condition. (ii) The
whitened weight S^-1 Delta is a partial (incremental) weight: a coordinate that only proxies another recipient's signal gets ~0 weight
(invented-data check: whitened income-gradient share 0.040 against a marginal-weight share 0.460), so the income network is pushed
only for what it adds to the coalition, and the spend moves to the occupation side, where NEURAL_RAW has margin. (iii) D is affine-
invariant in the logit differences, so lowering P requires changing who looks like whom, not a temperature or bias move (closed
rescues cannot lower it). (iv) The task CE stays the full-vector NLL of each recipient, and the strength rule budgets all 8 MODEL_FIT
components including both within-SEX occupation guards.

## 3. Closest existing implementations and the operational differences

| | NEURAL_RAW objective (`acs_replication/neural_raw.py`, `leafcorr.Critics`) | JOINT_TREE pair critic (`acs_replication/leafcorr.py`) | FB_PFD |
|---|---|---|---|
| Trainable function | all U0 weights | leaf-score corrections on frozen T_TREE routing | all U0 weights (same as NEURAL_RAW) |
| Reader in the objective | 9 online SGD critics (linear, MLP32, MLP64 per view), 5 rounds per update on 256 rows, lagging | same critics | the exact best response, re-solved on every 2,048-row batch (no adversarial game) |
| Normalization / attribution | diagonal (DiagonalReference): marginal weights for correlated coordinates | same | full within-SEX covariance: partial (incremental) weights |
| Risk measure | 1 - min(CE, H)/H, clamped at 0 (zero gradient when critics lose to the constant) | same | summed Gaussian AUC of the best quadratic discriminant; never clamped |
| Strength | lambda_J per recipient (income ~2x occupation) x rho | per recipient | one coalition coefficient for both recipients |
| Recipients | simultaneous | simultaneous (S12 / S21 variants exist) | simultaneous |

Not a closed recipe: no leaf-score correction (no routing), no post-processing of a frozen vector (no tilt, LEACE projection, decoder,
calibrator, temperature or bias: the release is a retrained function of the six fields, and P is invariant to those moves), no noise,
no merging, flattening or code, no input generalization, no SEX standardization or reweighting (the task loss and rows are U0's),
no controller (lambda and the path are fixed before the fit; the rule only picks a checkpoint). The nearest published relatives are
fairness penalties on mean or kernel discrepancies (MMD, HSIC) and closed-form linear erasure (LEACE); the differences that matter here
are the whitened, coalition-level best response used as a training objective and the single coalition strength. Novelty is not claimed
(no literature search).

## 4. Evidence that would contradict the rationale

1. **Attribution premise (MODEL_FIT, recorded before any MODEL_SELECT read):** at matched total-NLL spend the candidate's income share of
   the NLL spend is not below the matched control's (`selection.income_share_of_nll_spend`).
2. **The functional does not matter at equal spend:** on the reader grid the candidate's pair is not below its matched control's
   (benefit within about 1 se_agg of 0), i.e. "spend explains the drop" holds for this anchor too.
3. **Best-response evasion:** the candidate's pair exceeds its occupation view by > 0.010, or the MODEL_FIT HGB15 pair reader falls by
   less than half of the LOG_LR pair reader's drop (higher-order structure left for nonlinear readers).
4. **The occupation side cannot absorb the reallocation:** the MF rule stops at k* <= 2 on the men's occupation guard while occupation NLL
   uses < 50% of its budget (the within-SEX asymmetry, not the objective, is the limit; NEURAL_RAW's own men's guard is +0.0015 / +0.0020
   at an overall occupation Brier of +0.0003).

## 5. Controls (as first proposed: 6 specifications; superseded by the 05:04Z addendum above and CARD_DRAFT.md item 7)

- **FB_PFD_RAWJ (matched control, the simplest informative one).** NEURAL_RAW's own recipe (online critics, per-recipient lambda_J,
  rho 1) from the same start, task rows, optimizer, folds and SEX exposure, at the FIRST checkpoint E1..E10 whose MODEL_FIT out-of-fold
  paired total NLL spend over the fold start reaches the candidate's (`famkit.match_spend`; the lower neighbour is recorded; flag
  CONTROL_BELOW_CANDIDATE_SPEND with E10 if none). Cost is matched prospectively on MODEL_FIT out-of-fold paired NLL; both are
  evaluated on MODEL_SELECT by the coordinator. On all MODEL_FIT people this loop reproduces NEURAL_RAW step for step: its E2 weights
  are checked bitwise against `NEURAL_RAW__rho1.0__E2.pt` (recorded check), so the control is the anchor's own path continued to the
  candidate's cost. It answers: at the same task cost, does the whitened coalition objective lower recovery more than the anchor's
  critic objective?
- **FB_PFD_TASK (equally exposed zero-protection continuation).** The identical PFD loop with lambda = 0 (no SEX) at E_k*: separates the
  protection from the effect of continuing U0 for k* epochs at lr 1e-4.
- Not built: SBC / SPEND / RO paths of the earlier families (the operation is a retraining, not a function of a twin's release; a
  release-only version of a whitened discriminant of the twin's own outputs would be a closed tilt). Stated so the reading carries it.

## 6. Bounded choices, rule and budgets (all fixed before the fit)

- rho = 1 (one value; NEURAL_RAW's selected nominal ratio); path E1..E10 (400 steps at most; NEURAL_RAW's grid went to E20); privacy
  batch 2,048; degree-2 features; ridge 1e-3 tr/p; Adam 1e-4, clip 5 per network; task batch 256 x 40 steps per epoch.
- Cross-fitting: famrun folds `SAR_xf_FB_S<v>` (5 household folds); each fold restarts from the U0 recipe refitted on its training rows
  (Recipe A with U0's admitted epoch counts 7 / 14 (v0), 9 / 13 (v1) and Stage A's seed; checked bitwise against U0 on all MODEL_FIT).
- **Rule (MODEL_FIT only):** k* = the largest E_k such that E_1..E_k all keep estimate + 2 household-clustered SE <= BUDGET on every
  component of the out-of-fold paired spend over the fold start (`famkit.select_strength`). No feasible E1: nothing is released.
- **Budgets** (out-of-fold spend over the start; caps are 0.010 accuracy / 0.010 NLL / 0.005 Brier and guards): occupation NLL 0.0060,
  income NLL 0.0050, occupation Brier 0.0025, income Brier 0.0025, men's and women's occupation Brier 0.0028 each, occupation and income
  accuracy 0.0060. About 0.5-0.6 of each cap at +2 SE, leaving 0.4 for MODEL_FIT -> MODEL_SELECT transfer (earlier campaigns saw up to
  +0.003 on the men's guard) and for the Ucal clauses (about 0.001 tighter in NLL). These allowances exceed NEURAL_RAW's own MS spend on
  every component (it uses at most 0.34 / 0.57 of cap) except version-1 income Brier (0.0028), which the hypothesis says the candidate
  needs less of; they are not near-identity allowances (NEURAL_RAW rho 1 E5's occupation spend fits inside them).

## 7. Measurements

Coordinator (MODEL_SELECT): the 28 utility conditions and the registered reader grid (pair, income view, occupation view) for the six
specifications; paired benefits against NEURAL_RAW, H_LJ, U0, the matched control and the continuation; recipient conditions. Fit
records (MODEL_FIT, `results/<campaign>/fb/FIT_PFD_V<v>.json`): every component of the out-of-fold spend along all three paths E0..E10,
k*, k_control and its bracket, the income share of the NLL spend, lambda and the logged D_v per 40 steps, the descriptive MODEL_FIT
readers (LOG_LR per view, HGB15 pair) of the start, candidate, control and continuation, the two bitwise checks, CPU.

## 8. How a failure would be read

- D2 not met with the candidate about equal to its control at matched spend: the dependence functional and its attribution do not
  matter for this anchor; the gain is spend, and the caps leave no room for more. This closes "a better objective for the same
  networks" here.
- Contradiction 1 holds (income share not lower): the income push in NEURAL_RAW was not proxy waste (e.g. the coalition's incremental
  income signal is real); the attribution hypothesis is withdrawn.
- Contradiction 3 holds: the quadratic best response is evaded; R1 (higher-order features) is the only revision on that branch.
- Contradiction 4 holds: the within-SEX asymmetry of the occupation cost is the binding limit, the same limit investigator A faces; a
  within-SEX construction would be a different budget, not a revision.
- Utility met on MODEL_FIT but missed on MODEL_SELECT: transfer error, not a mechanism result.

## 9. Registered predictions (probabilities; written before any fit)

| # | Prediction | P |
|---|---|---|
| 1 | Both bitwise checks hold (U0 recipe = U0; control E2 = NEURAL_RAW rho 1 E2) | 0.85 |
| 2 | k* in E2..E6 in both versions | 0.60 (k* = E1 or none: 0.10; k* = E10: 0.10) |
| 3 | The MF binding component is an occupation-side one (men's guard 0.45, occupation NLL 0.20) rather than income (0.20) | 0.65 |
| 4 | MF: candidate's income share of the NLL spend < matched control's (contradiction 1 fails) | 0.70 |
| 5 | Candidate meets all 28 conditions at the MS point | 0.75 |
| 6 | Candidate pair AUC (registered grid, mean over cells): median 0.610, 80% interval [0.592, 0.628] | - |
| 7 | Pair benefit vs NEURAL_RAW > 0 at the point / resolved (both versions > 0, mean - 2 se_agg > 0) | 0.75 / 0.40 |
| 8 | Pair benefit vs the matched control > 0 at the point / resolved (D2) | 0.65 / 0.15 |
| 9 | Pair benefit vs H_LJ > 0 at the point | 0.55 |
| 10 | Material target (resolved 0.020 vs NEURAL_RAW and H_LJ, D4) | 0.04 |
| 11 | Candidate's income view <= NEURAL_RAW's + 0.010 in both versions (recipient condition) | 0.70 |
| 12 | Candidate's pair - occupation view <= 0.010 | 0.75 |
| 13 | Continuation FB_PFD_TASK: pair within 0.005 of U0's in both versions | 0.80 |
| 14 | Most likely outcome: candidate feasible, pair 0.600-0.618, point benefit +0.007 to +0.025 over NEURAL_RAW, not separated from its matched control | 0.45 |

## 10. Specification count and CPU

6 specifications (FB_PFD, FB_PFD_RAWJ, FB_PFD_TASK x versions 0, 1). Measured on invented data at the real input width: 0.0041 CPU-s
per PFD step with the 2,048-row privacy batch; the RAWJ loop about 0.02 CPU-s per step. Per version: 5 folds x (U0 recipe ~1 s + PFD
~2 s + TASK ~1 s + RAWJ ~10 s) + the deployed fit (~15 s) + releases, spends and the MODEL_FIT readers (~1 min): about 3-5 CPU-min,
upper estimate 10 CPU-min; both versions <= 20 CPU-min, one thread, torch venv. Design work so far: about 25 CPU-s (selftests 20 s,
label-free checks 5 s), far inside the 600 CPU-s cap.
