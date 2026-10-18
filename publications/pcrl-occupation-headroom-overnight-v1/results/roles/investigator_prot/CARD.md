# Protection-stage mechanism card (privacy stage; protection investigator)

PCRL overnight occupation-headroom campaign. EXPLORATORY DEVELOPMENT on previously used records; not confirmation.
Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64, civilian
employed), development roles only. The service predicts an income class (2) and an occupation group (5 official broad civilian
groups) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding `X`). Each recipient receives only its
own float64 probability vector and argmax decision. SEX is the dataset's recorded demographic column (sex0 = men, sex1 = women);
readers are separate evaluation classifiers that estimate SEX from the released vectors (each output alone and the linked pair).
SEX is a training-time label on MODEL_FIT only; never a runtime input or output.

Written 2026-10-18 05:15-05:45Z, after the registration receipt and BEFORE any real-data protection fit and before any task-only
witness has been selected (the coordinator names the witness at about 07:30-08:00Z). Code: `ohn/protect.py`. Unit ids
`OHN_PROT_<config>_S<s>_V<n>` (fits), `..._SCORE_V<n>`, `OHN_PROT_SELFTEST_V<n>`, `OHN_PROT_DRYRUN_V<n>` (real features, invented
labels), grid units `OHN_PROT_GRID_<bank>_S<s>_V<n>`. Revisions are appended as dated sections BEFORE their first fit; nothing above
them is rewritten.

## 0. Facts this design starts from (measured today with the registered evaluator; no new label read here)

From `runs/REF/S<s>/EVAL_MS_REFERENCES.json` (H3 income + the named occupation output), MODEL_SELECT, seed 0 / seed 1:

| occupation output | men's guard (excess vs U0) | women's guard | occupation-side minimum ratio |
|---|---|---|---|
| T_TREE (task-only teacher) | -0.00103 / +0.00059 | -0.00931 / -0.00788 | 3.33 / 2.60 (men's guard binds) |
| TASK_REFIT (task-only leaf corrections) | -0.00119 / -0.00041 | -0.01020 / -0.00815 | 3.41 / 3.19 |
| JOINT_TREE (SUCC, H_LJ; critic-trained) | +0.00442 / +0.00394 | -0.00494 / -0.00304 | 0.32 / 0.62 (men's guard binds) |
| O1 (JOINT_TREE + SEX-balancing bias) | +0.00166 / +0.00230 | -0.00219 / -0.00276 | 1.83 / 1.59 |

- The protection cost of JOINT_TREE over its teacher is about the SAME for both SEX groups (men +0.0055 / +0.0034, women +0.0044 /
  +0.0048). The men's guard binds because the task-only BASE is asymmetric: the teacher's gain over U0 is concentrated among women
  (men -0.0010 / +0.0006 against women -0.0093 / -0.0079; investigator A's card reports the same). Under the frozen reserve (margin
  >= 2 x se_used, se_used >= H_LJ's floor 0.00181 / 0.00170) a teacher-level base leaves about 0.0024 / 0.0010 of men's Brier,
  but about 0.0109 / 0.0095 of women's.
- O1 shows that a per-class bias fitted with SEX on MODEL_FIT can move about 0.0016-0.0028 of men's guard excess to women. A
  per-class bias is a bijection of the released vector, so it cannot change what an unconstrained reader can recover.
- Recipient gate (frozen rule): every grid cell's occupation-view AUC must be <= the comparator's + 0.010 for every required
  comparator; five of the six required comparators release the JOINT_TREE occupation output. A task-only witness whose occupation
  view reads clearly above JOINT_TREE's fails that gate whatever its pair benefit; the occupation view is therefore a target of
  this stage, not only the pair.

## 1. Arms, exact operation and transmitted output

W denotes the top `SELECTED_WITNESS` (family F, configuration c) named by the coordinator; its deployed occupation output is
`p_W(x) = ohn.fam_<f>.predict(seed, c, X, names)` on every role; its household-cross-fitted MODEL_FIT output is
`CROSSFIT.npz['MODEL_FIT__oof']` (each household predicted by the witness pipeline refitted without its OHN1_crossfit fold).
Income is H3's frozen output in every arm (`ohn.devrun.anchor_income('H3', s, role)`); only the occupation output changes.

**Common correction model (CONT, LOC, PAIR).** `z_W(x) = log p_W(x)` (centred). Correction on the frozen T_TREE occupation routing
of the same seed (the lineage's validated leaf-score parametrization, `acs_replication.leafcorr`):

    Delta_theta(x) = sum over trees t of theta[t, leaf_t(x), :] / T          (theta: T x 16 x 5; theta = 0 gives W exactly)
    released occupation vector  p'(x) = softmax(z_W(x) + s * Delta_theta(x)),   decision = argmax p'(x)

with `s in [0, 1]` the end-projection scale (section 4). Transmitted: the float64 vector p' and its argmax, nothing else. Runtime
inputs: `X` only (the witness's own `predict` plus the routing); no SEX, label, identifier, household id or income quantity enters
the runtime output (the H3 income wire is used only by the PAIR training critics).

| arm | training objective for theta (all else identical) | class |
|---|---|---|
| `W` | none (the witness unchanged; no new configuration) | - |
| `CONT` (matched task-only continuation twin) | CE + guard term (section 3); critics updated exactly as in LOC/PAIR but unused (lambda = 0) | CTRL12 |
| `LOC` (local occupation recovery penalty) | CE + guard term + lambda * risk(occupation wire) | PRIV24 |
| `PAIR` (coalition-aware occupation correction, H3 income frozen) | CE + guard term + lambda * [risk(occupation wire) + risk(pair wire)], pair wire = (H3 income wire, occupation wire) | PRIV24 |
| `COMP` (compression control at matched cost) | none: label-free k-means code (below) | CTRL12 |

- `risk(view) = 1 - min(critic CEs, H(SEX)) / H(SEX)` over the leafcorr online critic bank of that view (linear, MLP32, MLP64).
  Wires: occupation = (p'_0..p'_4, argmax) (width 6), income = (H3 p_0, p_1, argmax) (3), pair = both (9). The income-view critics
  are trained (as in leafcorr) but have no gradient path to theta. PAIR is a coalition-aware occupation correction, not a
  two-output joint design (claim J stays NOT_CLAIMABLE).
- `COMP`: K-means (scikit-learn KMeans, k-means++, one initialization, fixed seed, Lloyd) on the witness's cross-fitted MODEL_FIT
  probability vectors (label-free); released vector = the centroid of the nearest centroid (Euclidean in the simplex) of `p_W(x)`;
  decision = its argmax. Under calibration of W within cells, the Brier increase of a cell-mean code equals the within-cell
  variance, which K-means minimizes, so it is the Brier-efficient way to remove information without SEX. K is the smallest value
  of the ladder (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768,
  1024, 1536, 2048) whose budget utilization (section 4) is <= that of the PAIR configuration it is matched to (`--match`).
  A temperature or shrinkage toward the prior is NOT used as COMP: both are bijections of the released vector (they cannot lower
  an unconstrained reader's recovery) and the lineage measured matched-cost temperature controls with no pair effect (source
  family P, PC_TEMP_P2 / P3).

**What changes the transmitted probabilities.** Only the additive log-score `s * Delta_theta(x)`, a function of `X` through the
teacher's routing cells (CONT, LOC, PAIR), or the replacement of p_W by its cell centroid (COMP). In CONT it follows the task
gradient alone; in LOC / PAIR it also follows the critics' gradient, which points along the output directions (per routing cell)
in which SEX is currently recoverable from the released vector (LOC) or from the vector next to the actual H3 income vector (PAIR).

**Why it might lower SEX recovery beyond ordinary continuation and compression.** Continuation follows the task gradient only and
has no reason to remove SEX structure (in the lineage the task-only twins read ABOVE the critic-trained outputs: source family P
twin vs P2, pair -0.022 / -0.021). Compression removes information uniformly along the Brier-cheapest directions without knowing
where SEX evidence lives. The critic term spends the same budget where SEX evidence is cheap to remove per unit of task cost, and
PAIR spends it on the occupation evidence that ADDS to the frozen H3 income vector actually released next to it. Evidence for
protection = LOC / PAIR below BOTH the CONT twin and COMP at matched utilization on the same 8 grid cells, at no larger
MODEL_SELECT cost.

## 2. Training base: cross-fitted witness output (decided before fitting)

- The correction is trained on MODEL_FIT people with `z_W` = the witness's household-CROSS-FITTED output (`CROSSFIT.npz`), not
  its deployed MODEL_FIT output. Reason: the deployed MODEL_FIT output is in-sample for the witness (its fit saw those labels), so
  the CE and guard terms would see residuals that are smaller and more confident than out of sample, the critics would see a
  different SEX structure, and the in-sample cost would understate the MODEL_SELECT cost (the lineage's P1 ran at 1.37 x its
  MODEL_FIT distortion). The cross-fitted output is honest for the correction's labels.
- Deployment uses the witness's deployed output (`predict`) plus the same `s * Delta_theta(x)` on every role. Transfer is
  checked label-free (the distribution of the correction's label-free distortion on the cross-fitted MODEL_FIT base vs the deployed
  bases of MODEL_FIT, CALIBRATION and MODEL_SELECT) and by the scorer (MODEL_SELECT costs against the out-of-fold prediction).
- Exposure caveats recorded with every release: (i) for a family-C witness the frozen teacher layer is in-sample even in the
  witness's cross-fitted output; (ii) the witness's cross-fitted base of a training fold was produced by witness pipelines that saw
  the held-out fold of the correction (non-nested; standard stacking approximation); (iii) H3's income on MODEL_FIT is in-sample
  for H3 (fitted on MODEL_FIT and CALIBRATION; its transplanted corrections were trained against SEX critics on MODEL_FIT), and no
  cross-fitted H3 output exists, so the PAIR critics see an income wire that may carry LESS SEX evidence than out of sample; the
  in-sample income-view critic risk is logged next to the grid's income view.

## 3. Guard-aware element (the men's guard binds; stated before fitting)

YES: the per-SEX Brier cost is constrained DURING TRAINING with SEX as a MODEL_FIT training label, and again in the end projection.

- **Training (all trained arms, identical form).** Task loss on a minibatch = CE (all people, SEX-blind)
  + mu_men * mean_{men in batch}(summed Brier(p')) + mu_women * mean_{women in batch}(summed Brier(p')).
  The multipliers start at 0 and follow dual ascent once per epoch on the IN-SAMPLE per-SEX summed-Brier increase over the
  cross-fitted witness base on the training people: `mu_g <- min(20, max(0, mu_g + 1.0 * (Inc_g / tau_g - 1)))`, target
  `tau_g = 0.8 x slack_g` (section 4; floor 1e-4). This is a Lagrangian on each guard: it only acts when a group's cost exceeds its
  target, and because the women's slack is several times the men's, it lets the correction pay for protection with women's Brier
  (shifting the shared vector toward the men's conditional distribution, which includes the O1-type bias direction) instead of
  men's. In CONT the in-sample increase is negative, so its multipliers stay 0 (pure continuation): the twin differs only by
  lambda = 0.
- **End projection (all trained arms and COMP).** Per-SEX constraints on cross-fitted (out-of-fold) MODEL_FIT costs: the men's and
  the women's summed-Brier increase over the witness base must each stay within its budget (section 4). In-sample Lagrangian costs
  are optimistic; the projection is the honest control.
- Not used: SEX at runtime, a SEX-specific output, a group-specific threshold.

## 4. Budgets, cross-fitted projection and configurations

- **Cross-fitted correction.** For each arm the entire training pipeline (warm critics, lambda calibration, 20 epochs, dual
  updates) is run 6 times: on MODEL_FIT minus each OHN1_crossfit fold k (k = 0..4; predicts fold k) and on all MODEL_FIT (the
  deployed correction). Out-of-fold release on MODEL_FIT: `softmax(z_W^oof + s * Delta_{theta_k}(x))` for fold-k people.
- **Slack.** For the witness's own MODEL_SELECT evaluation (`EVAL_MS_H3.json`, seed s, frozen reserve rule `ohn.rules.reserve`):
  slack_c = margin_c - 2 x se_used_c for each occupation-side condition c (6 clauses, 2 guards). This is an adaptive design input
  read from the logged evaluation (as the source family P set its budgets from published MODEL_SELECT values); no parameter is
  fitted on MODEL_SELECT.
- **Budget at level f** (two predefined levels, f in {0.5, 0.8}, configuration suffixes `_B50`, `_B80`), on the 20,000 out-of-fold
  MODEL_FIT people (labels and SEX of MODEL_FIT; household-clustered SEs recorded):

  | statistic (protected minus witness, cross-fitted) | budget |
  |---|---|
  | men's mean summed-Brier increase | f x slack(guard men) |
  | women's mean summed-Brier increase | f x slack(guard women) |
  | summed-Brier increase | f x min(slack(occ U0 Brier), slack(occ Ucal Brier)) |
  | NLL increase | f x min(slack(occ U0 NLL), slack(occ Ucal NLL)) |
  | accuracy loss | f x slack(occ accuracy) |

  Utilization u = max over the five of statistic / budget. Projection: s = the largest value such that every point of the grid
  0, 0.01, ..., up to s is feasible, refined by 30 bisection steps toward the first infeasible grid point (s = 1 when the
  correction is already within budget; s = 0 means no protection fits the budget and no release is saved). A slack <= 0 gives a
  zero budget.
- **Training effort (identical in every trained arm, recorded):** theta = 0 start; AdamW lr 0.01, betas (0.9, 0.999), eps 1e-8,
  decoupled weight decay 0.01; batch 256 (task batch: epoch permutation; privacy batch: sampled from the same people with SEX);
  40 steps per epoch, 20 epochs (800 updates); critic bank of `acs_replication.leafcorr.Critics` (Adam lr 0.003, 80 warm rounds,
  5 critic rounds per update, DiagonalReference refresh at every epoch start on a fixed 256-row reference sample);
  lambda = rho x median||grad CE|| / median||grad P|| at the exact start with warm critics (16 fixed draws), **rho = 1** (the
  inherited JOINT_TREE strength), each arm calibrated on its own objective; one strength, no sweep, no controller, no
  temperature. Seeds: streams from SHA-256 of ('OHN-PROT', seed, fold, purpose), shared by all arms (CONT is the exact twin).
- **Logged:** lambda and its calibration draws; per epoch the mean training-critic risk and critic CE per view, the effective ratio
  lambda||g_P|| / ||g_task||, in-sample CE, in-sample per-SEX Brier increases, mu_men / mu_women; the cross-fitted cost curve
  over s; the chosen s; out-of-fold and in-sample task costs; label-free transfer statistics; an internal MODEL_FIT recovery
  diagnostic (logistic and HGB SEX readers fitted on out-of-fold outputs of folds 0-2, scored on folds 3-4, for W, CONT and the
  arm; diagnostic only, never a selection criterion).
- **Configurations (seed-specific; classes announced):** `P1_CONT` (CTRL12; one configuration when the projection does not bind at
  either level, else `_B50` / `_B80`), `P1_LOC_B50`, `P1_LOC_B80`, `P1_PAIR_B50`, `P1_PAIR_B80` (PRIV24), `P1_COMP_B50`,
  `P1_COMP_B80` (CTRL12; matched to `P1_PAIR_B50` / `_B80`). Revisions `P2_*`, `P3_*`. Individual fits: per trained arm and seed
  6 corrections + 6 x 9 critics (deployed: 1 + 9; cross-fitting: 50) and 1 projection scale (calibrator); COMP: K-means fits of
  the ladder search (calibrator-type, label-free) recorded.

## 5. Plan (seed 0 first)

1. Selftest on invented data (`OHN_PROT_SELFTEST_V1`); label-free dry run on real features with INVENTED labels and SEX and the
   stored T_TREE output standing in for a witness (`OHN_PROT_DRYRUN_V1`: timing, routing of every role, H3 income loading;
   nothing saved as a release).
2. When the coordinator names the witness: check `predict` of the witness under the protection interpreter on all six roles
   against its saved arrays (label-free), then seed 0: `P1_CONT`, `P1_LOC`, `P1_PAIR` (one training pipeline each, both budget
   levels), then `P1_COMP_B50` / `_B80` matched to the PAIR levels. Score each (`ohn.devrun score --kind privacy|control
   --budget-class PRIV24|CTRL12`). Reader grid seed 0 (bank `PROT_P1`): members `PROT/<id>@H3`, the witness `<F>/<c>@H3`;
   comparators = the six required references + the witness (matched component) + `PROT/P1_CONT@H3` and the matched COMP;
   fractions [1.0].
3. Diagnose; seed 1 for the arms that are promising (definition below) with their CONT twin and matched COMP; the grid seed 1;
   `ohn.grid_summary` for the 8-cell labels.
4. At most two evidence-driven revisions (announced to the coordinator and appended here before their first fit).

"Promising" (seed 0): RESERVE_PENDING_SECOND_SEED (all 14 seed-0 conditions met) AND pair benefit > 0 in all 4 seed-0 cells
against the witness and against CONT AND occupation-view recipient difference <= 0.010 in all 4 cells against every required
comparator.

## 6. Revise / abandon triggers (decided now)

- **Transfer revision:** the MODEL_SELECT men's-guard increase over the witness exceeds the cross-fitted projected increase by more
  than one floor SE (0.0018 / 0.0017) -> the out-of-fold projection does not transfer; revise the projection (margin or base).
- **Guard revision:** at B80 the projection keeps s <= 0.25 because a guard binds, or the occupation view still fails the
  recipient gate -> revise the guard-aware element (candidate: a bijective SEX-balancing per-class bias re-optimized with the scale,
  i.e. an (s, b) projection), which cannot add reader-recoverable information.
- **Abandon the critic mechanism (report the control):** PAIR is not below COMP at matched utilization (seed-0 mean pair benefit
  versus COMP <= 0.003 or negative in 2 or more of 4 cells), or not below CONT (mean benefit <= 0.005): compression or continuation
  explains it.
- **Abandon the stage:** no level keeps the 14 seed-0 occupation-side and income conditions at the reserve, or the occupation view
  stays above JOINT_TREE's + 0.010 at every feasible level -> report the frontier and stop.
- Never: relax a cap, the reserve, the comparator set or the grid; select on ATTACK / DEV_REPORT; sweep strength, temperature or
  controllers.

## 7. Closest established methods

- Adversarial debiasing / adversarial removal (Zhang, Lemoine and Mitchell 2018; Edwards and Storkey 2016; Elazar and Goldberg
  2018, who show that residual leakage survives adversarial training and must be measured by fresh readers - here the grid's
  refitted readers); multi-view adversaries; fair adversarial gradient tree boosting (Grari et al. 2019).
- In this lineage: JOINT_TREE / LOCAL_TREE (leaf-score corrections against online critics on the teacher routing; the same
  critic recipe and strength), source family P (pair-aware correction of one output against the other output frozen, with an
  explicit budget enforced by an end projection), source family O (SEX-balancing per-class bias on MODEL_FIT).
- Group-constrained learning with Lagrangian multipliers (Cotter et al. 2019; Agarwal et al. 2018 reductions); group DRO
  (Sagawa et al. 2020) for the per-group cost term. Compression control: Lloyd / k-means vector quantization; the privacy-funnel
  framing (Makhdoumi et al. 2014).
- Algorithmic novelty: NOT_CLAIMED.

## 8. What is new relative to the old controller / temperature / strength lines

(a) The base is a stronger task-only witness selected for occupation headroom, not the teacher; (b) the binding constraint was
identified as the men's guard and traced to the base's SEX asymmetry, not to an asymmetric protection cost, so the new element is a
PER-SEX (guard) constraint that can trade cost between SEX groups during training and in the projection; none of the old lines
constrained a SEX-group cost (they controlled aggregate cost, strength or confidence); (c) the correction is trained on the
witness's cross-fitted output and projected on out-of-fold costs rather than in-sample or label-free teacher distortion; (d) the
PAIR critics see the actually released frozen H3 income vector from the first step. The strength is fixed at the inherited value
(rho = 1); no strength, temperature or controller sweep is run.

## Revisions and results

(appended below with time, reason and result)

### Addendum A (2026-10-18 05:50-06:00Z, still BEFORE any real-data protection fit): targets from the reference grid and a frontier level

New information since the card was written (coordinator messages; `runs/GRID/REF_BANK/S0, S1`; no label read by me):

- Development reader grid, mean SCORE AUC over the 4 cells of each seed (seed 0 / seed 1): H3 + T_TREE occupation: occupation view
  0.6325 / 0.6299, pair 0.6363 / 0.6360; SUCC (H3 + JOINT_TREE): occupation 0.5611 / 0.5657, pair 0.5997 / 0.6020; H3 + O1: pair
  0.6002 / 0.5986; RETIRED_B3CTL pair 0.5964 / 0.5914; H3's income view alone 0.5262 / 0.5165 (the floor of any occupation-only step).
- Recipient gate: five of the six required comparators release the JOINT_TREE occupation output (occupation view about 0.561 /
  0.566 per cell), so a candidate's occupation view must come down by about 0.06 from a teacher-level witness (to <= the comparator
  + 0.010 in EVERY cell) before any pair benefit counts.
- Stability bar (coordinator, from REF_BANK SUMMARY): sd_total 0.009-0.015 per comparator, so STABLE_POSITIVE needs a mean pair
  benefit of roughly 0.02-0.03 over every required comparator, i.e. a pair AUC of about 0.575 or lower on seed 0, with the whole
  reserve kept. SUCC itself is POSITIVE_NOT_STABLE vs H_LJ / NEURAL_RAW and NOT_POSITIVE vs RETIRED_B3CTL.
- Where JOINT_TREE pays (registered scorer, per class / per SEX, seed 0): its occupation NLL increase over H3 + T_TREE sits on class
  3 (natural resources / construction / maintenance; +0.0144 vs -0.0012) and on men (men's NLL excess +0.0170 vs -0.0034; women's
  -0.0122 vs -0.0211). In summed Brier the per-SEX costs are closer (+0.0055 / +0.0044), but in NLL the protection cost is
  concentrated on men through the male-typed class. The per-SEX Brier Lagrangian of section 3 acts on the guard metric; the CE term
  and the out-of-fold NLL budget act on NLL.

Arithmetic stated before fitting (teacher-level base, seed 0, frozen reserve): men's slack 0.0024, women's 0.0109, aggregate Brier
0.0076, NLL 0.016; JOINT_TREE-level protection cost 0.0055 / 0.0044 / 0.0049 / 0.0147. Even with perfect rebalancing between SEX
groups, a JOINT-level occupation correction only just fits the reserve (NLL binds at f about 0.9), and the 0.575 pair bar would need
roughly twice JOINT_TREE's reduction. Expected outcome on a teacher-level witness: the reserve-compatible levels will NOT reach the
bar; the experiment is still informative as a matched frontier (protection vs continuation vs compression at equal cost), and a
stronger witness may shift it.

Design additions (no change to sections 1-4 otherwise):
- A third, frontier-only level for the PAIR arm: `M90` = 0.9 x the witness's POINT margin (cap - value) of every occupation-side
  condition, without the 2-SE reserve (configuration `P1_PAIR_M90`, PRIV24; its COMP match `P1_COMP_M90`, CTRL12). It answers "how
  far does this mechanism get inside the original caps" and is expected to fail the reserve by construction; it is never presented
  as a reserve-meeting candidate. LOC keeps the two reserve levels.
- `fit --twin <CONT id>` records the matched twin in each LOC / PAIR record.
- Strength stays at rho = 1 (no sweep): the end projection makes the spent budget, not the strength, the binding quantity; the
  frontier is traced by the budget levels B50, B80 and M90.
- Self-test `OHN_PROT_SELFTEST_V3`: PASS, 29 of 29 invented-data checks (V1 failed on a dictionary key, V2 on two over-strict
  check designs - the guard activation check used labels the correction could improve in-sample, the COMP check an unreachable
  target; both checks were redesigned, no mechanism code changed).

### Addendum B (2026-10-18 06:00-06:10Z, still BEFORE any real-data protection fit): dry run, parity, guard rule amended

- Label-free dry run `OHN_PROT_DRYRUN_V1` (real MODEL_FIT features, INVENTED occupation labels and SEX, the frozen T_TREE output as
  stand-in witness, PAIR arm, 5 fold pipelines; nothing saved; 131.7 CPU-s, peak RSS 1.1 GB): about 25 CPU-s per training pipeline,
  so a real arm (6 pipelines + release + verification + diagnostics) costs about 3 CPU-minutes; the projection binds tightly
  (s = 0.644 at the men's constraint, utilization 1.000); the release path on all six roles gives equal label-free distortion on
  every role (summed squared change 0.0019-0.0021); on the invented SEX the fresh held-out readers fall (occupation view 0.694 ->
  0.643). Two mechanics facts: (i) the online-critic outcome differs across fold pipelines (lambda 0.24-0.38; final in-sample CE
  change -0.0002 to +0.0050), so the out-of-fold projection averages heterogeneous corrections and its transfer to the deployed
  full-data correction must be checked (label-free distortion on MODEL_SELECT vs out of fold, then the scorer); (ii) the
  absolute in-training guard target (0.8 x slack) never activated, because in-sample costs stay below it - the intended
  rebalancing would therefore not happen during training.
- Witness parity `OHN_PROT_PARITY_A1_PO_16_S0_V1` (label-free): `ohn.fam_a.predict` of A1_PO_16 seed 0 under the protection
  interpreter (.venv Python 3.13) reproduces the witness's saved release bitwise on all six roles.
- **Guard rule amended (section 3), before any real-data fit, because of fact (ii):** equal-utilization Lagrangian. Per epoch,
  with in-sample per-SEX summed-Brier increases Inc_g over the cross-fitted base and the witness's reserve slacks S_g:
  u_g = Inc_g / S_g; target tau_men = max(0.0005, S_men x max(u_women, 0)), tau_women = max(0.0005, S_women x max(u_men, 0));
  mu_g <- min(5, max(0, mu_g + 1.0 x (Inc_g / tau_g - 1))). The multiplier of a group rises whenever that group uses its slack
  faster than the other group (or by more than 0.0005 while the other group gains), so the correction pays with the women's
  larger slack instead of the men's; the 0.0005 dead zone keeps pure continuation (CONT) inactive; mu <= 5 limits how far the men's
  term can dominate the task loss. The amount spent is still set only by the out-of-fold projection. Identical in every trained
  arm.

### Initial attempt P1 on witness B/B2_SINGLE, seed 0 (fits 06:29-06:44Z; scores 06:44-06:45Z; ledger rows of OHN_PROT_*_S0_SCORE_V1)

Configurations (seed 0): `P1_CONT` (CTRL12; scale 1.0), `P1_LOC_B80` (PRIV24; 1.0), `P1_PAIR_B50` (PRIV24; 0.609),
`P1_PAIR_B80` (PRIV24; 0.861), `P1_PAIR_M90` (PRIV24; 1.0). COMP for P1 was not fitted (see revision 1). Two stopped attempts
(`OHN_PROT_P1_PAIR_S0_V1`, `OHN_PROT_P1_CONT_S0_V1`, rc -15, nothing saved) were restarted as `_V2` only to record the witness id
as a string. MODEL_SELECT (H3 income; value vs U0, ratio margin / se_used): witness men's guard -0.00407 (5.00); CONT -0.00488
(5.45); LOC_B80 -0.00012 (2.80); PAIR_B50 -0.00280 (4.30); PAIR_B80 -0.00194 (3.83); PAIR_M90 -0.00139 (3.51); every P1
configuration meets the seed-0 reserve (14 conditions). Occupation NLL vs U0: witness -0.01469, CONT -0.01592, LOC -0.00940,
PAIR -0.01185 / -0.00937 / -0.00766.

Mechanism-level findings in the training records (no grid outcome used):
- The equal-utilization guard Lagrangian OSCILLATES: mu_men and mu_women jump between 0 and the cap 5 from epoch to epoch in PAIR,
  LOC and CONT; the in-sample per-SEX increases flip sign (PAIR full pipeline epoch 3: men -0.011 / women +0.018; epoch 15: men
  +0.0013 / women -0.0020); fold pipelines end in opposite states (fold 1: men +0.012 / women -0.013; fold 4: men -0.015 / women
  +0.025). It also activated in CONT, so the card's statement (section 3, addendum B) that pure continuation keeps it inactive is
  FALSE on the real data (CONT is still the matched twin: same rule, lambda = 0).
- Consequently the out-of-fold projection averages heterogeneous fold endpoints and does not transfer: at the B80 scale the
  fold-average correction moves outputs about twice as much as the deployed full-data correction (label-free summed squared change
  0.0029 out of fold vs 0.0015 on MODEL_SELECT, CALIBRATION and deployed MODEL_FIT), and the per-SEX split differs (M90: out of fold
  men +0.0011 / women +0.0106; MODEL_SELECT men +0.0027 / women +0.0019 over the witness). CONT: out-of-fold NLL +0.0041 over the
  witness, MODEL_SELECT NLL -0.0012.

### Revision 1 -> P2 (decided 06:47Z, before any P2 fit; announced to the coordinator before its first fit)

- **Reason:** the two mechanism-level findings above (an unstable guard term, and a projection that cannot transfer while the fold
  endpoints are that heterogeneous). They are defects of the guard element, not outcomes of the reader grid (the P1 grid was still
  running when this was decided).
- **Single change:** the in-training per-SEX Lagrangian is removed from every trained arm (mu = 0 throughout: task loss = CE on all
  people, SEX-blind). The guard-aware element is now only the per-SEX constraints of the out-of-fold end projection (section 4,
  unchanged). Everything else is identical to P1: base, routing, critics, rho = 1, 20 epochs, cross-fitting, levels, COMP rule.
  Code: `ohn.protect fit ... --guard none` (P1 = `--guard equal`, the default).
- **Configurations:** `P2_CONT` (CTRL12; the exact twin: lambda = 0 and no guard term = pure continuation), `P2_LOC_B80`,
  `P2_PAIR_B50`, `P2_PAIR_B80`, `P2_PAIR_M90` (PRIV24), `P2_COMP_B80` (CTRL12; and `P2_COMP_M90` on B2_SINGLE seed 0 only), on
  B2_SINGLE seed 0 first, then A/A1_PO_16 seed 0, then seed 1 of both witnesses for the informative arms. Names for A1_PO_16:
  `P2A_*`. CTRL12 plan: P1_CONT 1 + 4 x (CONT + COMP_B80) + 1 COMP_M90 = 10 of 12.
- **Registered predictions (before any P2 fit):** (i) fold endpoints become homogeneous enough to transfer: at each saved PAIR level
  the ratio of out-of-fold to MODEL_SELECT label-free squared change is within 0.67-1.5 (P1: 1.9); (ii) the MODEL_SELECT men's-guard
  increase over the witness is within one floor SE (0.0018) of its out-of-fold projected value at every saved level; (iii) on the
  seed-0 grid P2_PAIR_B80's mean pair benefit is > +0.010 against P2_CONT and > +0.003 against P2_COMP_B80; (iv) without the guard
  term the men's guard binds the projection at B50 / B80 (the protection cost falls on men as for JOINT_TREE). Each is reported as
  confirmed or refuted.

### P1 seed-0 reader grid (bank PROT_P1, unit OHN_PROT_GRID_PROT_P1_S0_V1, 230.8 CPU-s; read at 06:51Z, after revision 1 was decided)

Mean SCORE AUC over the 4 seed-0 cells (occupation view / pair) and mean pair benefit (comparator minus candidate; min cell):
witness B2_SINGLE 0.6354 / 0.6426; P1_CONT 0.6323 / 0.6327 (vs witness +0.0099, min -0.0015); P1_LOC_B80 0.5978 / 0.6240 (vs
witness +0.0186, vs CONT +0.0087); P1_PAIR_B50 0.6095 / 0.6233 (+0.0193, +0.0093); P1_PAIR_B80 0.5936 / 0.6190 (+0.0237, vs CONT
+0.0137, min +0.0080); P1_PAIR_M90 0.5871 / 0.6170 (+0.0256 min +0.0184; vs CONT +0.0157 min +0.0124; vs SUCC -0.0173; vs H_LJ
+0.0047; vs NEURAL_RAW +0.0021; vs RETIRED_B3CTL -0.0206). Occupation-view recipient vs the JOINT-occupation comparators: +0.029
(M90) to +0.055 (B50), all above the 0.010 gate. Reading: the P1 penalty arms lower recovery beyond their continuation twin in
every cell at B80 / M90, but remain far above SUCC (0.5997) and RETIRED_B3CTL (0.5964) while spending about half of JOINT_TREE's
MODEL_SELECT cost (M90: men +0.0027, NLL +0.0070 over the witness); at M90 the projection did not bind (s = 1, out-of-fold
utilization 0.83), i.e. at rho = 1 the correction does not use the available point-margin budget.

### P2 seed 0 results (fits 06:51-07:00Z on B/B2_SINGLE and A/A1_PO_16; scores 06:59-07:05Z; grid PROT_P2 seed 0 at 07:05Z)

Witness B2_SINGLE (MODEL_SELECT value vs U0, ratio; grid mean SCORE AUC occupation / pair over the 4 seed-0 cells):

| config | scale / K | men's guard | women's | occ NLL | seed-0 reserve | occ view / pair | pair benefit vs CONT; vs COMP (matched); vs SUCC |
|---|---|---|---|---|---|---|---|
| witness | - | -0.00407 (5.00) | -0.00908 | -0.01469 | MET | 0.6354 / 0.6426 | - |
| P2_CONT | 1.0 | -0.00507 (5.55) | -0.00876 | -0.01576 | MET | 0.6289 / 0.6351 | -; -; -0.0354 |
| P2_LOC_B80 | 0.953 | +0.00075 (2.20) | -0.00525 | -0.00260 | MET | 0.5773 / 0.6142 | +0.0209; +0.0104 (vs COMP_B80); -0.0146 |
| P2_PAIR_B50 | 0.638 | +0.00018 (2.62) | -0.00831 | -0.00743 | MET | 0.5933 / 0.6076 | +0.0275; +0.0170; -0.0079 |
| P2_PAIR_B80 | 0.856 | +0.00230 (1.39) | -0.00705 | -0.00251 | NOT MET | 0.5812 / 0.5972 | +0.0378 (min +0.032); +0.0274 (min +0.022); +0.0024 (min +0.001) |
| P2_PAIR_M90 | 1.0 | +0.00385 (0.57) | -0.00592 | +0.00142 | NOT MET | 0.5732 / 0.5919 | +0.0431; +0.0314 (vs COMP_M90, min +0.021); +0.0077 |
| P2_COMP_B80 | K 32 | +0.00077 (2.30) | -0.00875 | -0.00443 | MET | 0.6025 / 0.6246 | +0.0105 (min -0.001); -; -0.0249 |
| P2_COMP_M90 | K 20 | +0.00327 (0.90) | -0.00597 | +0.00313 | NOT MET | 0.5871 / 0.6233 | +0.0117; -; -0.0237 |

A1_PO_16 (scores only; grid PROT_P2A running): P2A_PAIR_B50 men +0.00092 (2.15, MET), B80 +0.00329 (0.85), M90 +0.00421
(0.38); P2A_COMP_B80 (K 24) +0.00053 (2.34); P2A_LOC_B80 scale 0.788; P2A_CONT -0.00434.

Registered predictions of revision 1: (i) CONFIRMED (out-of-fold / MODEL_SELECT label-free change 0.94-0.96 at every PAIR level,
P1 1.9); (ii) REFUTED: the MODEL_SELECT men's-guard increase exceeds its out-of-fold projection by 0.0016 / 0.0020 / 0.0023
(B2_SINGLE B50 / B80 / M90) and 0.0022 / 0.0029 (A1_PO_16 B50 / B80), more than one floor SE at B80 and beyond on both witnesses;
the label-free COMP codes show the same excess (+0.0013 to +0.0018), so it is a property of the men's subgroup transfer between
the out-of-fold MODEL_FIT costs and MODEL_SELECT, not of the critics (the aggregate NLL transfers: B80 +0.0122 MODEL_SELECT vs
+0.0126 out of fold); (iii) CONFIRMED on B2_SINGLE seed 0 (+0.0378 vs P2_CONT, +0.0274 vs P2_COMP_B80); (iv) CONFIRMED (the
men's constraint binds the projection at B50 and B80 without the guard term).

Reading: without the in-training guard the pair critic is much stronger. P2_PAIR_B80 reaches SUCC's recovery (0.5972 vs 0.5997)
with better occupation utility than SUCC (NLL vs U0 -0.0025 vs +0.0025; men's guard +0.0023 vs +0.0044), but outside the 2-SE
reserve; the reserve-keeping level B50 stays 0.008 above SUCC. The binding constraint is again the men's guard on MODEL_SELECT,
while the women's guard keeps >= 0.0085 of unused reserve slack at B80.

### Revision 2 -> P3 (proposed 07:15Z, before any P3 fit; WAITING for the coordinator's commit before the first P3 fit)

- **Reason:** P2's binding condition is the men's guard on MODEL_SELECT (and its out-of-fold projection under-states the men's
  cost by about 0.002), while the women's guard keeps most of its slack; P1 showed that shifting task pressure toward men lowers
  the men's cost (P1_PAIR_B80 men +0.0021 vs P2 +0.0064 over the witness) but its multiplier oscillated and weakened the critic.
- **Single change relative to P2:** a STATIC per-SEX weighting of the task cross-entropy in every trained arm: person i gets weight
  w_g(i) = (1 / S_g) / mean over the training people of (1 / S_g(j)), with S_g the witness's MODEL_SELECT reserve slack of guard g
  (the same per-seed quantities that set the budgets; B2_SINGLE seed 0: S_men 0.00544, S_women 0.0107, so men weigh about 2x
  women). No multiplier, no update, no per-SEX term in the critics or the output; SEX remains a MODEL_FIT training label only.
  Everything else identical to P2 (critics, rho 1, 20 epochs, cross-fitting, projection, levels, COMP rule). Code:
  `ohn.protect fit ... --guard static`.
- **Configurations (proposed):** B2_SINGLE seed 0: `P3_PAIR_B50`, `P3_PAIR_B80`, `P3_PAIR_M90` (PRIV24) and `P3_CONT` (CTRL12; the
  weighted-CE continuation twin); seed 1 of `P3_PAIR_B80` and `P3_CONT` only if the seed-0 predictions (i)-(ii) hold.
- **Registered predictions (before any P3 fit):** (i) P3_PAIR_B80's MODEL_SELECT men's-guard increase over the witness is lower than
  P2_PAIR_B80's (+0.0064) by at least 0.0015 and P3_PAIR_B80 meets the seed-0 reserve (men's ratio >= 2); (ii) its seed-0 grid pair
  recovery is within 0.005 of P2_PAIR_B80's (<= 0.6022); (iii) the women's guard stays at reserve at every saved level; (iv)
  P3_PAIR_B80 beats P3_CONT by > 0.020 on the seed-0 pair. Each is reported as confirmed or refuted.
- **Amendment (07:20Z, still before any P3 fit and before the coordinator's commit), after the P2A seed-0 grid:** on A1_PO_16
  P2A_PAIR_B80 reads pair 0.5813 (below all six required comparators in every seed-0 cell; occupation-view recipient max +0.0074)
  but misses the seed-0 men's reserve (ratio 0.85), so the same single change is tested on BOTH selected witnesses.
  Configurations: `P3_PAIR_B50`, `P3_PAIR_B80` (B2_SINGLE) and `P3A_PAIR_B50`, `P3A_PAIR_B80` (A1_PO_16), seed 0 (PRIV24 4); M90 is
  dropped; NO new CONT twin is fitted (CTRL12 is nearly exhausted and kept for the coordinator's stability controls): the
  reference twin is the unweighted P2_CONT / P2A_CONT, so the weighting's own effect on a continuation is NOT isolated (disclosed;
  P3 minus P2 at the same level isolates the weighting in the penalty arm). Seed 1 of `P3(A)_PAIR_B80` only if (i)-(ii) hold for
  that witness. Prediction (iv) now reads: P3_PAIR_B80 beats P2_CONT (and P3A_PAIR_B80 beats P2A_CONT) by > 0.020 on the seed-0
  pair; predictions (i)-(iii) apply to each witness (for A1_PO_16 the P2A_PAIR_B80 reference is men +0.0071 over the witness, pair
  0.5813, tolerance <= 0.5863).

### P2 seed-1 replication and the 8-cell grid summaries (fits 07:08-07:17Z; scores 07:17Z; grids 07:17-07:21Z; summaries 07:21Z)

Seed-1 configurations (not a revision): `P2_PAIR_B50/B80/M90` (scales 0.529 / 0.717 / 1.0), `P2_CONT`, `P2_COMP_B80` (K 32) on
B2_SINGLE; `P2A_PAIR_B50/B80` (0.546 / 0.719), `P2A_CONT`, `P2A_COMP_B80` (K 32) on A1_PO_16. Seed-1 MODEL_SELECT men's guard (ratio):
P2_PAIR B50 -0.00123 (3.67), B80 -0.00039 (3.10), M90 +0.00121 (2.03); P2_COMP_B80 +0.00158 (1.92); P2A_PAIR B50 -0.00094 (3.43),
B80 -0.00008 (2.82); P2A_COMP_B80 +0.00172 (1.80). On seed 1 the out-of-fold projection OVER-states the men's cost (B2 B50 MS +0.0013
vs projected +0.0021; B80 +0.0021 vs +0.0033), the reverse of seed 0: the seed-0 excess was not systematic. Reserve over both seeds:
P2_PAIR_B50 and P2A_PAIR_B50 RESERVE_MET; P2_PAIR_B80, P2_PAIR_M90, P2A_PAIR_B80 met on seed 1 only; the COMP_B80 codes met on seed
0 only.

8-cell labels (frozen rule, `ohn.grid_summary`; mean pair benefit; seed-0 / seed-1 means):

| candidate | vs matched CONT | vs matched COMP_B80 | vs witness | vs SUCC | vs RETIRED_B3CTL |
|---|---|---|---|---|---|
| P2_PAIR_B50 | SIGN_CHANGES +0.0205 (+0.027 / +0.013) | SIGN_CHANGES +0.0065 | POSITIVE_NOT_STABLE | NOT_POSITIVE -0.0154 | NOT_POSITIVE |
| P2_PAIR_B80 | POSITIVE_NOT_STABLE +0.0319 (+0.038 / +0.026) | POSITIVE_NOT_STABLE +0.0179 (+0.027 / +0.008) | POSITIVE_NOT_STABLE | NOT_POSITIVE -0.0040 (+0.002 / -0.010) | NOT_POSITIVE |
| P2_PAIR_M90 | STABLE_POSITIVE +0.0395 | POSITIVE_NOT_STABLE +0.0255 | STABLE_POSITIVE | SIGN_CHANGES +0.0036 (+0.008 / -0.000) | NOT_POSITIVE |
| P2A_PAIR_B50 | POSITIVE_NOT_STABLE +0.0245 | SIGN_CHANGES +0.0109 | POSITIVE_NOT_STABLE | NOT_POSITIVE -0.0110 | NOT_POSITIVE |
| P2A_PAIR_B80 | STABLE_POSITIVE +0.0398 (+0.058 / +0.021) | POSITIVE_NOT_STABLE +0.0262 (+0.043 / +0.010) | STABLE_POSITIVE | SIGN_CHANGES +0.0042 (+0.018 / -0.010) | NOT_POSITIVE -0.0027 |
| P2_COMP_B80 | SIGN_CHANGES +0.0140 | - | POSITIVE_NOT_STABLE +0.0182 | NOT_POSITIVE | NOT_POSITIVE |
| P2A_COMP_B80 | POSITIVE_NOT_STABLE +0.0136 | - | POSITIVE_NOT_STABLE +0.0149 | NOT_POSITIVE | NOT_POSITIVE |

Reading: the protection effect beyond the matched continuation twin and beyond the matched compression code replicates on both
witnesses and both release seeds (every one of the 8 cells positive at B80), and is STABLE_POSITIVE against continuation at the
largest saved scale on each witness; superiority over SUCC does NOT replicate (seed-1 projections chose smaller scales because the
seed-1 men's reserve slack of the witnesses is smaller, 0.0041 vs 0.0054 on B2_SINGLE), and no configuration is positive against
RETIRED_B3CTL on both seeds. H3's income view fails the recipient gate for every H3-anchored release (+0.0309 vs H_LJ in some cells),
so no protection-stage configuration can be nominated under the frozen rule.

Provenance notes (07:25Z): (1) Protocol deviation 14 (coordinator's record): `ohn/protect.py` was edited on disk from 07:08:47Z (the
P3 `--guard static` code) while OHN_PROT_P2A_PAIR_S1_V1, OHN_PROT_P2A_CONT_S1_V1, OHN_PROT_P2_COMP_S1_V1 and OHN_PROT_P2A_COMP_S1_V1
ran, so those four units ran an uncommitted file; its `none` path and the COMP path compute exactly what the committed version
computes (the coordinator read the diff); the records gain `sex_weights: null`. From 07:21Z every fit uses committed code only.
(2) Disclosure: the static P3 weights 1 / S_g use the witness's logged MODEL_SELECT reserve slacks (the same values that set the
budgets, recorded in every P3 OCC.json as `guard_targets_tau` = `witness_slack` guard entries, with `budget_provenance`): a design
input from logged development selection quantities, not fitting or calibration on MODEL_SELECT; no MODEL_SELECT label enters any fit.

### P3 results, seed 0 (committed 06b0ca47a at 07:21:30Z; fits 07:22-07:30Z; scores 07:30Z; grids PROT_P3 / PROT_P3A 07:33Z)

| config | scale | MS men's guard (ratio) | MS women's (ratio) | occ NLL vs U0 | seed-0 grid occ / pair | vs P2 same level | vs P2 CONT | vs P2 COMP_B80 | vs SUCC |
|---|---|---|---|---|---|---|---|---|---|
| P3_PAIR_B50 (B2_SINGLE) | 0.318 | -0.00587 (5.99) | -0.00523 (5.97) | -0.01213 | 0.6153 / 0.6304 | -0.0228 | +0.0047 | -0.0058 | -0.0308 |
| P3_PAIR_B80 (B2_SINGLE) | 0.469 | -0.00642 (6.30) | -0.00278 (4.47) | -0.00971 | 0.6108 / 0.6156 | -0.0184 | +0.0195 | +0.0090 | -0.0159 |
| P3A_PAIR_B50 (A1_PO_16) | 0.289 | -0.00575 (5.93) | -0.00529 (5.81) | -0.01313 | 0.6178 / 0.6274 | -0.0231 | +0.0121 | -0.0034 | -0.0277 |
| P3A_PAIR_B80 (A1_PO_16) | 0.430 | -0.00635 (6.25) | -0.00244 (4.10) | -0.01035 | 0.6087 / 0.6180 | -0.0367 | +0.0215 | +0.0060 | -0.0184 |

Out of fold the static weights shift the whole cost to women (B80: men -0.0035 / -0.0042, women +0.0085 / +0.0090), so the
women's constraint binds the projection at about half of P2's scale. Registered predictions: (i) CONFIRMED on both witnesses (men's
MODEL_SELECT increase over the witness -0.0024 / -0.0025, i.e. an improvement, vs P2's +0.0064 / +0.0071; seed-0 reserve met);
(ii) REFUTED on both (pair +0.018 / +0.037 above P2_PAIR_B80 / P2A_PAIR_B80); (iii) CONFIRMED (women's guard at reserve at every
level); (iv) REFUTED on B2_SINGLE (+0.0195 vs P2_CONT, below 0.020) and CONFIRMED on A1_PO_16 (+0.0215 vs P2A_CONT). As planned, no
P3 seed 1 and no cost-matched COMP twin (the best P3 level does not clear the references on the seed-0 grid). Lesson: the per-SEX
balance of the protection cost is controllable, but with a fixed 2:1 weight the cost flips to the women's guard; the reserve-feasible
optimum lies between P2 (men binding) and P3 (women binding), and no third revision is allowed to locate it.

### Last step: LOC arm on seed 1 (predeclared arm, committed code 06b0ca47a, launched 07:35Z; not a revision)

`P2_LOC_B80` and `P2A_LOC_B80` seed 1 (PRIV24 -> 23), so the coalition-aware (PAIR) vs local (LOC) comparison has 8 cells; banks
PROT_P2L / PROT_P2AL with P2(A)_PAIR_B80, P2(A)_CONT and P2(A)_COMP_B80 as comparators (cached readers).

Result (fits 07:35-07:37Z, scores 07:37Z, grids 07:37-07:39Z): seed-1 scales 0.809 (B2_SINGLE) / 0.659 (A1_PO_16); MODEL_SELECT
men's guard +0.00012 (2.76) / +0.00104 (2.26). P2_LOC_B80 therefore meets the whole-release reserve on BOTH seeds (min ratio 2.20,
seed-0 men's guard); P2A_LOC_B80 one seed only (seed 0: 1.41). 8-cell labels for P2_LOC_B80: vs P2_CONT POSITIVE_NOT_STABLE +0.0258,
vs P2_COMP_B80 POSITIVE_NOT_STABLE +0.0118 (every cell positive), vs SUCC NOT_POSITIVE -0.0101. PAIR_B80 minus LOC_B80 (pair AUC):
A1_PO_16 PAIR lower in all 8 cells (-0.008 to -0.036, mean -0.0206); B2_SINGLE PAIR lower in 6 of 8 (seed 0 all four, about -0.017;
seed 1 mixed, where LOC's projected scale 0.809 exceeded PAIR's 0.717).

### Closing summary of the protection stage (07:45Z)

Configurations: PRIV24 23 (P1: 4; P2 on B2_SINGLE: 8; P2A on A1_PO_16: 7; P3 / P3A: 4), CTRL12 10 (P1_CONT; P2_CONT x2; P2_COMP_B80
x2; P2_COMP_M90; P2A_CONT x2; P2A_COMP_B80 x2); 2 CTRL12 slots left unused. No configuration passes the frozen nomination gates (H3's
income view fails the recipient gate for every H3-anchored release). Component answers: (1) a critic-trained occupation correction
lowers linked-pair recovery beyond the matched task-only continuation twin and beyond a label-free compression code at matched
out-of-fold budget utilization, on both witnesses and both release seeds (every cell positive at B80; STABLE_POSITIVE vs the
continuation twin at the largest saved scale on each witness); (2) the coalition-aware pair term adds to the local term on A1_PO_16 in
all 8 cells and on B2_SINGLE on seed 0; (3) the protected outputs reach the inherited JOINT_TREE output's recovery (SUCC 0.600) on
seed 0 with BETTER occupation utility (P2_PAIR_B80 0.597, P2A_PAIR_B80 0.581, men's guard +0.0023 / +0.0033 vs SUCC +0.0044), but
not on seed 1 and not inside the 2-SE reserve on seed 0; the reserve-meeting levels (P2_PAIR_B50, P2A_PAIR_B50, P2_LOC_B80) read
0.604-0.625 and are NOT_POSITIVE against SUCC; (4) the per-SEX balance of the protection cost is controllable (P1 oscillating
multiplier, P3 static weights shift it to women) but only at a loss of protection in this bounded search. Record artifact: COMP
records carry `guard: equal` from the command-line default; COMP involves no training, so the field is meaningless there.
