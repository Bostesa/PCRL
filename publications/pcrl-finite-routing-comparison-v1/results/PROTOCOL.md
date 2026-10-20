# Protocol: finite-routing comparison (campaign `pcrl_finite_routing_comparison_v1`, code `frc/`)

Coordinator. Writing started 2026-10-20T23:06Z (`date -u`); the completion time is in section 15. EXPLORATORY DEVELOPMENT on
previously and adaptively used public-use ACS 2019 1-Year PUMS Florida person records (the admitted development cohort). Nothing here
is confirmation.

## 0. Status and disclosure (before any label read)

- Campaign start: 2026-10-20T22:49:23Z (first `date -u`; the coordinator's memory notes were read in the minute before). Research
  window ends 06:49:23Z on 2026-10-21; closeout ends 07:19:23Z.
- Source: the previous campaign `pcrl_new_release_mechanisms_v1` (NRM1), publication 682068d52 (pinned and verified on the GitHub
  remote by `frc.admit`), local research tip fd933a824 = the base of this branch `research/pcrl-finite-routing-comparison-v1`.
- **Known before writing** (published or label-free): every NRM1 published record, including PRC_TWIN's MODEL_SELECT utility (point
  feasible; worst value / cap 0.100 in version 0 and 0.510 in version 1, the binding condition being the version-1 men's guard
  +0.00255; whole-release reserve ratio 2.311 / 1.289), its registered-grid pair AUC 0.6509, the failed capacity gates and Designer
  A's label-free capacity curve. PRC_TWIN_C128 was never evaluated on MODEL_SELECT.
- **Pre-registration label-free check** (coordinator, 23:00Z; saved models and cached out-of-fold teacher predictions only; no label;
  about 5 CPU-s, charged when the queue opens):
  - the saved models reproduce their saved MODEL_SELECT releases bitwise through `nrm.fa.prc.predict`;
  - the stored routing prototypes are not the actual out-of-fold teacher means of the final routed cells: maximum absolute difference
    0.00023-0.00033 (income) and 0.0019-0.0062 (occupation), because Lloyd's last codeword is the mean of the previous partition;
  - the label-refit decoders differ from the prototypes by up to 0.09-0.14 occupation probability;
  - deployed routing of MODEL_FIT people with the full-fit teachers agrees with their out-of-fold routing for only 56-76% of people
    (full-fit predictions on training rows are in-sample). Full-fit cell means are therefore not used for B.
- No outcome of this campaign exists. No label has been read by this campaign.
- **Reviews before registration.** The challenger's review (`roles/challenger/REVIEW_PREREGISTRATION.md`) and the statistical
  reviewer's report (`roles/statistical_reviewer/REPORT_PREREGISTRATION.md`) were adjudicated in
  `roles/coordinator/ADJUDICATION_PREREG.md`; every adopted change is in this text, the card and the code.
- **Known from published records, so not a test outcome:** the source construction A at (16, 64) meets Q1 and FAILS the
  registered Q2 (section 5; version-1 men's guard 0.00255 + 0.0030 = 0.00555 > 0.005); it would meet an S1-slack form, which is
  not this rule. A's MODEL_SELECT utility at (24, 128) is unknown.
- Disclosed adaptive use: QUAL reads MODEL_SELECT SEX through the within-SEX guards (utility only; common to PRC and its
  controls).

## 1. The question and the hypothesis

**Question Q-PE.** Did estimating each cell's probabilities separately from MODEL_FIT labels (the label-refit decoders of FA_PRC_v1)
cause avoidable held-out task error in the saved task-only finite releases?

**Hypothesis H-PE (from Designer A's note; a hypothesis, not a finding).** The decoders add an estimation term (about 0.0055 nats at
64 occupation cells, 0.0092 at 128) that teacher-mean probabilities avoid. Fitted quantization error (the label-free capacity curve) is
not held-out task performance; only held-out losses of the actual emitted vectors and decisions count here.

If Q-PE's repair keeps useful task margin (section 5), the family proceeds to the propensity-aware routing comparison that NRM1 never
fitted (sections 7-10). If no task-only finite release qualifies, the family stops and protected routing stays untested.

## 2. Data, roles and the service (unchanged definitions)

- Admitted Florida ACS 2019 cohort only. Roles: MODEL_FIT (20,000 people, 13,337 households; all fitting), CALIBRATION (authorized;
  not planned), MODEL_SELECT (8,001 people, 5,306 households; evaluation and development readers only, never a fitting input).
  ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels are closed; DEV_REPORT features are used label-free for deployment parity. The
  sealed assessment and capped-out records are closed. No people or populations are downloaded.
- Runtime fields: AGEP, WKHP, COW, SCHL, MAR, POBP (admitted 145-column encoding; the A1_PO_16 compact coding `ohn.fam_a.compact_po`).
- Tasks: the exact income class (2) and five-class occupation group definitions, household groups, reference models and cohort
  manifests of the admitted chain (`frc.admit` pins them by hash).
- Service: each recipient receives its own task probability vector (float64) and the corresponding argmax decision. No routing index,
  cell identity or additional model feature is released. Internal cells emitting identical vectors are the same observable value.
- SEX (the dataset's recorded sex column; 0 = men, 1 = women) is a training label only in the protected stage (the SEX propensity and
  the SEX-conditional occupation statistics of FA_PRC_v1), an evaluation label of the MODEL_FIT cross-fitted check, and the target of
  the development readers on MODEL_SELECT. Runtime prediction receives only the six fields.

## 3. Stages and units (every numerical unit through `frc.jobqueue`; one thread; at most two at once)

| Stage | Units | Labels read | What |
|---|---|---|---|
| T0 reproduction | `FRC_REPRO_V<v>` | none | saved PRC_TWIN / PRC_TWIN_C128 releases recomputed from the saved models on all six roles (bitwise); the saved routing books refitted from the cached out-of-fold teachers (bitwise) |
| T1 fixed-routing comparison | `FRC_NEST_C64_V<v>`, `FRC_NEST_C128_V<v>` | MODEL_FIT income, occupation (`fit`) | nested estimator of the NRM1 gate (5 outer household folds `NRM_FA_ncv_S<v>`, whole pipeline refitted on each outer-training part at the full-pool T*) emitting CONT, A, B0 and B1 on identical partitions; reproduces the GATE records for A and the saved decoders from labels |
| T2 deployable task-only B releases | `FRC_TWB_C64_V<v>`, `FRC_TWB_C128_V<v>` | none | B1 tables from the cached full-pool out-of-fold teachers inside the saved books' out-of-fold partition; saved teachers and books unchanged; partition identity with the saved A release checked on all six roles |
| T3 MODEL_SELECT utility | `FRC_EVAL_*` | MODEL_SELECT income, occupation, SEX (`benchmark`) | 24 clauses + 4 guards, reserve, paired costs B1 - A (household-clustered), against the A twins and W |
| T4 qualification | (rule, no unit) | none | section 5 |
| T5 prospective third capacity (conditional) | `FRC_NEST_C256_V<v>`, `FRC_TWB_C256_V<v>` | MODEL_FIT income, occupation | (32, 256), chosen now from the recorded label-free curve (section 4.3); only if no existing capacity qualifies |
| T6 one structural repair (conditional) | new card first | per its card | section 11 |
| P1 protected paths | `FRC_PATHS_V<v>` | MODEL_FIT income, occupation, SEX (`fit`) | SEX-side teachers; nested estimator over the PRC / HQ / SHUF strength chains and the ORD ladder with the selected construction; predeclared budget selection; bracketing points (section 8); nested held-out pair LOOKUP AUC of every entry with a household-bootstrap SE against the twin, and F4 (section 10) |
| P2 configurations | `FRC_CFG_V<v>` | MODEL_FIT income, occupation (A only); none for B1 | the selected configurations and the bracketing controls from the full-pool statistics |
| P3 MODEL_FIT check | `FRC_CHECK_V<v>` | MODEL_FIT SEX (evaluation) | household cross-fitted LOOKUP of the OUT-OF-FOLD-routed fit cells of every saved finite release; labelled "routing optimized on these people (optimistic)"; never a credit input |
| E1 utility | `FRC_EVAL_*` | MODEL_SELECT (`benchmark`) | as T3 for every configuration |
| E2 reader grid | `FRC_GRID_*` | MODEL_SELECT SEX (`readers`) | fresh FRC1 grid, 8 registered cells (section 9) |
| E3 classifier refit | `FRC_GRID_ROB_*` | MODEL_SELECT SEX | robustness repetitions R3 / R4 and reader seeds k2 / k3 (section 10.4) |
| E4 ablation | `FRC_ABL_V<v>` | MODEL_FIT income, occupation, SEX | section 10.5 |
| V verification | `FRC_VER_*` | `verify` | independent replay |
| D, C, publication | closeout stages | none | deployment packages, custody, sanitized publication |

## 4. The probability constructions (fixed routing)

### 4.1 Shared parts (identical for every construction)

For one version v and one capacity (KI income cells, KO occupation cells), exactly as FA_PRC_v1 / `nrm.fa.prc`:
- task teachers qI (income) and qO (occupation): sklearn HGB with the A1_PO_16 learner settings on the compact coding; T* = first
  argmin of the pooled inner household-CV NLL (`OHN_A_cv_S<v>`); deployed teachers = full-pool refits at T*; out-of-fold predictions
  over 5 household folds `NRM_FA_xf_S<v>` (fold models at the same T*, non-nested, as recorded);
- routing book = two independent KL k-means quantizers of the out-of-fold posteriors (KL k-means++ seed SeedSequence([v, KI, KO]),
  Lloyd steps); routing rule: nearest codeword in cross-entropy, from the deployed teachers' posteriors of x;
- the out-of-fold partition (k0, l0) = the routing rule applied to the out-of-fold posteriors (rule-consistent: it is the partition
  the fitted book reproduces);
- shared by every construction and arm: the non-nested full-pool T* (the held-out outer folds' labels helped choose it, as in the
  NRM1 gate) and the shift between out-of-fold and deployed routing (56-76% agreement on MODEL_FIT in-sample); neither is
  differential between A and B.

### 4.2 The constructions (only the emitted table of each cell differs)

| Code | Emitted vector of income cell k (occupation cell l analogous) | Labels used beyond the teachers |
|---|---|---|
| **A** (source) | label-refit decoder: (MODEL_FIT label counts of the out-of-fold-routed people in the cell + 20 x codeword) / (n + 20) | MODEL_FIT income / occupation |
| **B1** (primary B) | mean of the OUT-OF-FOLD teacher predictions of the people whose out-of-fold route is the cell (the same people A counts), clipped at 1e-6 and renormalized; an unoccupied cell emits its stored prototype (count recorded) | none |
| **B0** (diagnostic) | the stored clustering prototype (routing codeword) | none |

- Means use out-of-fold predictions, never full-fit predictions of training rows (section 0).
- Decisions are the argmax of the emitted vector (first index on ties), for every construction.
- The partition is a function of the routing book and the teachers only. Changing the emitted table never changes it: T1 asserts
  identical held-out cells across A, B0 and B1; T2 asserts that A and B1 releases induce the same partition of every role's people;
  distinct emitted vectors = occupied cells (no two cells emit the same vector, checked).
- Training exposure is identical: the same 20,000 MODEL_FIT people and labels reach the same teachers; B uses no further label.

### 4.3 Capacities

Existing capacities first: (16, 64) and (24, 128). One additional capacity, chosen now from the recorded label-free curve (Designer
A's CAPACITY_INFO_LABEL_FREE.json; fitted occupation KL 0.0098 / 0.0067 / 0.0046 / 0.0031 at 64 / 128 / 256 / 512 cells, income 0.0009 /
0.0004 / 0.0002 at 16 / 24 / 32): **(32, 256)**, the next recorded rung on both curves. It is used only under the trigger of section 6
(no existing capacity qualifies). There is no other capacity, no calibration sweep and no shrinkage sweep.

## 5. Task feasibility and the qualification rule (registered before outcomes)

Original requirements, evaluated directly on MODEL_SELECT by `frc.evaluate` (= `nrm.evaluate` definitions; U0_S<v> / Ucal_S<v>):
accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed Brier excess <= 0.005 (2 tasks x 2 references x 3 metrics x 2 versions =
24 clauses), and the men's and women's occupation summed-Brier excess vs U0 <= 0.005 (4 guards). G1 = all 28 at the point.

Reported beside G1, never part of it:
- the two-SE reserve (`ohn.rules.reserve` unchanged; k = 2; H_LJ SE floor; whole release and occupation side);
- the NRM1 internal capacity gate, recomputed in its original definition from T1 (nested held-out spend over CONT: estimate + 2 SE <=
  0.004 nats NLL and <= 0.003 accuracy loss per task) and labelled `ORIGINAL_CAPACITY_GATE_PASS` / `_FAIL`. It is not this study's
  eligibility rule.

**QUAL (qualification for a small protection comparison).** A task-only construction X at a capacity qualifies iff, in BOTH versions:
- Q1: G1 holds (all 12 clauses and 2 guards of the version at the point); and
- Q2 (positive utility slack at the assessment budget S2, where D2 and its HQ / SHUF controls sit): value + d <= cap for every
  condition, with d = the S2 budget of FA_PRC_v1: occupation NLL 0.008, income NLL 0.004, occupation accuracy loss 0.006, income
  accuracy loss 0.004, men's and women's occupation Brier guards 0.0030; and, for the two summed-Brier clauses that the budget does
  not name, occupation 0.0030 (the guard budget) and income 0.0020 (half the income NLL budget, in proportion to the caps
  0.005 / 0.010).

Known before registration (section 0): A at (16, 64) meets Q1 and fails Q2. QUAL is a gate for starting the comparison, not a claim.
Passing point limits with slack does not establish confidence preservation: the reserve is reported separately and no qualified
release is called confidently feasible. (Adjudication: challenger 1 adopted; the draft's S1-slack form was replaced before any label
read.)

## 6. Construction selection, capacity order and triggers (MODEL_FIT evidence chooses; MODEL_SELECT only qualifies)

1. **Q-PE labels per capacity** (T1, MODEL_FIT nested; d = paired per-person held-out occupation NLL difference B1 - A per version;
   SE_used = max(household-clustered SE, SD of the 5 outer-fold means / sqrt(5)), which carries fold-to-fold training variability):
   - `PE_ERROR_SUPPORTED`: d + 2 SE_used < 0 in both versions, and the MODEL_SELECT paired occupation NLL difference B1 - A (T3) < 0
     at the point in both versions;
   - `PE_ERROR_POINT_ONLY`: d < 0 in both versions otherwise;
   - `PE_ERROR_NOT_SUPPORTED`: d >= 0 in some version.
   Reported beside: the same for income, total NLL, summed Brier and accuracy (nested and MODEL_SELECT); the men's / women's
   occupation Brier from MODEL_SELECT only (T1 reads no SEX); the share of A's nested held-out occupation spend over CONT that B1
   removes; B0 - B1; the cells and people whose A and B1 decisions differ, with accuracy and Brier differences restricted to them.
2. **Construction choice, per capacity** (MODEL_FIT nested point estimates only): B1 iff the nested paired occupation NLL difference
   and the nested paired total (income + occupation) NLL difference B1 - A are both < 0 in both versions; otherwise A (ties go to
   A). The construction chosen at a capacity is final for that capacity and is used consistently afterwards (QUAL, TWIN, PRC, HQ,
   SHUF, ORD, ablation). If it fails QUAL, the next capacity is tried even when the other construction would qualify; whether it
   would have qualified is recorded, never used.
3. **Capacity order:** (16, 64), then (24, 128). The protection capacity is the first whose chosen construction meets QUAL.
4. **Trigger T5:** if neither existing capacity qualifies, the prospective (32, 256) is run (nested A / B0 / B1, deployable release of
   the construction chosen at (32, 256), MODEL_SELECT utility, QUAL).
5. **Trigger T6:** if (32, 256) does not qualify either, one documented structural repair may be proposed (section 11).
6. **Stop:** if no task-only finite release qualifies, family FA stops, and the record states that protected routing remains untested.

Reproduction checks gate everything after T0 / T1: T0 bitwise; T1's A must reproduce the GATE records' nested means and spends
(maximum absolute difference <= 1e-12) and the saved decoders bitwise. A failed reproduction stops the stage for diagnosis.

## 7. The protection operation (only after QUAL; no further approval needed)

The propensity-aware routing operation of FA_PRC_v1 section 1, unchanged except for the emitted tables:
- statistics e(x) = (qI, qO, pi, qO0, qO1): the task teachers above; pi = P(SEX = 1 | x) and the SEX-conditional occupation posteriors
  qO0 / qO1 (HGB on [Z, s], evaluated at s = 0, 1), fitted on MODEL_FIT with SEX as a training label (inner CV T*, full refit,
  out-of-fold predictions over `NRM_FA_xf_S<v>`);
- rule: (k, l) = argmin CE(qI, cI_k) + CE(qO, cO_l) + t [D(pi; rP_kl) + D(pi; rI_k) + D(pi; rO_l)] + t kappa [per-SEX guard], kappa = 1,
  pair and local weights 1, shrinkage 20; protected books warm-started at the twin and continued up the grid lambda in {0.25, 0.5, 1,
  2, 4, 8} with damped, rule-consistent Lloyd steps (tilts lambda / 2^j, j = 0..6);
- **emitted tables: the selected construction, recomputed on the protected partition by the same rule** (B1: the out-of-fold teacher
  means inside the final protected out-of-fold cells; A: label-refit decoders of those cells).

**Budget path (predeclared; identity plus useful nonzero settings):** identity = the twin (lambda = 0); S1, S2, S3 = the FA_PRC_v1
spend budgets over the twin (nested MODEL_FIT held-out estimate + 2 household-clustered SE <= budget on every metric):

| Budget | occ NLL | inc NLL | men's / women's occ Brier | occ acc loss | inc acc loss |
|---|---|---|---|---|---|
| S1 | 0.004 | 0.002 | 0.0015 | 0.003 | 0.002 |
| S2 | 0.008 | 0.004 | 0.0030 | 0.006 | 0.004 |
| S3 | 0.012 | 0.006 | 0.0045 | 0.008 | 0.006 |

- Selection (`nrm.fa.prc.select_path`, MODEL_FIT only): the largest grid lambda such that it and every smaller lambda meet the budget;
  a budget no lambda meets is NOT_FEASIBLE and is not fitted; a budget whose selection equals an earlier budget's is an alias, never
  saved twice. Release parameters are never fitted on MODEL_SELECT labels.
- Recorded per strength and configuration: actual task costs (MODEL_SELECT paired costs), cell counts and occupancy per role, the
  fitted objective against the twin's, the people re-routed relative to the twin (MODEL_FIT out-of-fold and MODEL_SELECT) and the
  changed transmitted vectors (a nonzero setting that changes nothing is recorded as an exact alias), the nested held-out LOOKUP AUC,
  the fitted pi-weighted plug-in pair MI, and their difference from the measured grid recovery. Estimated propensities and plug-in
  objectives are approximations, never guarantees about other classifiers.
- **F4 (FA_PRC_v1 falsifier 4, MODEL_FIT):** for the PRC configuration selected at S2, the nested held-out pair LOOKUP AUC is below
  the twin's by more than 2 household-bootstrap SE (B = 400 MODEL_FIT household draws, scores fixed) in both versions. It is
  computed in P1 and committed before any MODEL_SELECT read of a protected release. It is a required conjunct of D2, not a stop of the
  evaluation (adjudication of challenger 3: the brief asks for the numerical comparison and allows diagnostic readings); a failed F4
  may justify revision R2 (section 11).

## 8. Controls (aligned decoder rule, training exposure, runtime disclosure and search effort)

| Control | What | Configurations |
|---|---|---|
| TWIN | the equally trained task-only finite release (selected construction, same capacity) | 2 (or the source releases if A) |
| HQ_S2 | task-score-only grouping: the same objective, optimizer, grid, tilts and budget rule, with pi, qO0, qO1 replaced by their MODEL_FIT means within the twin's joint task cells (routing is a function of the task posteriors); stores the task teachers only | 2 |
| SHUF_S2 | shuffled-signal control: the same optimizer, grid, tilts and budget rule with pi replaced by a keyed-hash donor copy (`NRM_FA_shuf_S<v>`; deployable; no person-level SEX information) and the guard's qO0 = qO1 = the person's own qO (a SEX-blind task-fidelity guard; card change from FA_PRC_v1 before any SHUF fit, adjudication of challenger 6); stores the task teachers and the donor bank only | 2 |
| ORD_S1-S3 | ordinary compression path: the twin at coarser codebooks, the most coarsened rung meeting each budget; SEX-blind mapping, SEX-informed operating point. Ladders, finest to coarsest: (16, 64): (12, 48), (8, 32), (6, 24), (4, 16), (3, 12), (2, 8); (24, 128): (16, 64) then the same; (32, 256): (24, 128), (16, 64) then the same | up to 6 |
| HQ_BR, SHUF_BR, ORD_BR | bracketing points (adjudication of challenger 5): the first strength in path order (grid lambda; ladder rung) whose nested MODEL_FIT total-NLL spend over the twin (point) is >= that of the PRC configuration selected at S2; an alias of an already saved point is not saved again; none reaching it gives `CONTROL_DID_NOT_REACH_SPEND` | up to 6 |
| References | NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100 (all point-eligible; the original benchmark), plus U0 / Ucal as utility diagnostics | 0 |

Control paths for the frontier comparisons (`nrm.rules.frontier_comparison`, MODEL_SELECT paired total-NLL costs; the bracket is
the cheapest path point spending at least as much as the candidate): HQ path [HQ_S2, HQ_BR]; SHUF path [SHUF_S2, SHUF_BR]; ORD path
[TWIN, distinct ORD_S1-S3, ORD_BR]. Realized spending of every control is reported beside every comparison.

Every control uses the selected construction recomputed on its own partition, the same MODEL_FIT exposure, the same disclosure (vectors
and decisions only) and, for HQ and SHUF, exactly the PRC search (grid, tilts, selection rule). Utility costs are reported as measured,
never assumed equal from nominal capacity. Point-eligible references stay in the original benchmark with or without the reserve. A
simpler control may be the best result.

## 9. Readers (what recipients actually receive)

- Fresh development grid FRC1 inside MODEL_SELECT: repetitions R1, R2 (registered) and R3, R4 (robustness) with new tags
  `FRC1_devgrid_R<r>`: households ordered by SHA-256(tag | household) and dealt into 10 parts, FIT = parts 0-4, VALIDATION = 5-6, SCORE
  = 7-9 (household-disjoint); reader seeds MODEL_SEED_BASE + 700 + 10 v + k (k0, k1 registered; k2, k3 robustness; never an earlier
  campaign's offset 300 / 400 / 500 / 600); household bootstrap of SCORE (B = 400, seed 2026102100 + 100 v + repetition index).
  Household overlaps with the earlier grids are recorded in READER_SPLITS.json.
- The admitted expanded bank (16 recipes from the locked recipe code: OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB,
  LOOKUP, ET5, ET25, HGB15, HGB31, MLP64, MLP6432, CATI01, CATI1): LOOKUP is the frequency-table model on the exact transmitted values
  (pseudo-count 2 toward the FIT base rate; a value never seen in FIT gets the FIT base rate); the others pool across cells. Fitted on
  FIT, oriented and selected on VALIDATION (maximum oriented VALIDATION AUC), scored on SCORE; the pair winner is chosen among pair
  readers AND the ignore-occupation and ignore-income alternatives. Same recipes and effort for every release.
- Readers receive only the transmitted vectors (and therefore decisions); never cell ids, routing indices, raw fields or other labels.
- Grid extras of every cell (descriptive; `frc/readerextra.py`, a campaign copy of `nrm.readerextra` so that its splits are the FRC1
  splits): the non-table winner (`nolookup`: the registered selection with LOOKUP excluded), TABLE_CV (a frequency-table reader with
  cross-validated smoothing; in the pair view an additive logit on one-hot(income value) + one-hot(occupation value) with
  cross-validated L2, then the joint table shrunk towards it) and the ENHANCED winner (the 16 recipes plus TABLE_CV).
- Reported per release (label-free, `frc.readerbudget`): reader FIT / VALIDATION / SCORE sizes, distinct values per view, FIT people
  per value, the share of SCORE people whose value is unseen in FIT (they get the base rate) and the mean LOOKUP data weight. For
  every D2 and D3 comparison the pair-view budget contrast (`frc.readerbudget.contrast`, thresholds as committed) is recorded; the
  FIT-fraction curve (0.25 / 0.5 / 1.0) is reported for the pair view of TWIN, the PRC configuration at S2 and the bracketing
  controls (descriptive).
- Uncertainty and dependence: the NRM1 aggregate screen (block means over version x repetition; se_agg = sqrt(se_hh^2 + se_rs^2 +
  se_mc^2) with the 0.001 floor; both version means must agree), which bounds the correlation of overlapping households, shared
  readers and versions. Versions, repetitions and reader seeds are not independent replications; an ineffective reader seed is
  counted once.

## 10. Decisions (kept separate; formal definitions in roles/statistical_reviewer/RULES.md)

1. **D1 task-only repair:** `REPAIRED_TASK_ONLY_RELEASE` iff B1 is the chosen construction, its Q-PE label is PE_ERROR_SUPPORTED at
   the protection capacity and it meets QUAL; `QUALIFIED_B1_REPAIR_NOT_RESOLVED` if B1 is chosen and qualifies with Q-PE only
   PE_ERROR_POINT_ONLY (not a repair claim); `QUALIFIED_WITHOUT_PROBABILITY_REPAIR` if A is chosen and meets QUAL (possible only at
   (24, 128) or (32, 256)); `POST_HOC_REPAIR_QUALIFIED` for a T6 repair; `NO_QUALIFYING_TASK_ONLY_RELEASE` otherwise.
2. **D2 protection component beyond matched controls**, assessed at the PRC configuration selected at S2 (an alias of an earlier
   budget's selection is that configuration, with its brackets). Conjuncts: (a) G1; (b) against TWIN: pair benefit `AGG_POSITIVE`
   and the recipient gate (`nrm.rules.comparison`); (c) against the HQ, SHUF and ORD paths (section 8): `DOMINATES_CONTROL_FRONTIER`
   each (spend-matched; an empty HQ or SHUF path is compared at TWIN, flag `<name>_AT_IDENTITY`); (d) F4 passes (else `NOT_MET_F4`);
   (e) reader targeting: against TWIN and every bracketing point, the point benefit is > 0 in both version means for the non-table
   winner and for the nested MODEL_FIT LOOKUP (else `READER_TARGETED`). Partial outcomes keep their names (`PARTIAL_<passed>_ONLY`);
   none is a component claim. `NOT_ASSESSED_NO_FEASIBLE_STRENGTH` if no PRC strength meets S2 (no fallback budget);
   `NOT_ASSESSED_EXACT_ALIAS` (and the family stops) if every nonzero selection transmits exactly the twin's vectors. Default
   reading: the minimum detectable pair benefit is about 0.015 (2 se_agg); a smaller effect reads "not separated".
3. **D3 improvement over the strong original references:** NRM1 G4 against all nine (intersection-union, recipient gates), at the
   same configuration as D2. A comparison flagged `READER_BUDGET_ASYMMETRIC` counts as beaten only if the two point conditions of
   2(e) also hold against that reference.
4. **D4 established 0.020 material benefit:** NRM1 G5 against NEURAL_RAW and H_LJ, at the same configuration as D2.
5. **D5 joint-design contribution:** not assessed (`NOT_ASSESSED_MISSING_CONTROLS`) unless the separate design and both sequential
   orders exist; ABL_LOC alone is not that control set, so no joint-design advantage is claimed.
6. **D6 novelty:** `NOT_CLAIMED` (deterministic privacy funnel / IBSI family; FARE is the published finite-cell input-routing method).
7. **Outcome class:** `CANDIDATE` (D2 and D3), `COMPONENT` (D2 without D3), `ENGINEERING_REPAIR` (D1 = REPAIRED_TASK_ONLY_RELEASE and
   D2 not met), or `FINITE_NEGATIVE_RESULT`; flag `D3_WITHOUT_D2` (references beaten without separation from the matched controls:
   not a candidate). A near-limit point estimate is never promoted as robust.

### 10.4 Classifier refit of the strongest feasible nonzero release

The strongest feasible nonzero release = the G1-feasible PRC configuration (lambda > 0) with the largest registered-cell aggregate pair
benefit against TWIN (ties to the smaller lambda). It and its matched controls (TWIN, the HQ, SHUF and ORD bracketing points,
NEURAL_RAW, H_LJ) get the robustness cells (R3, R4 with k0-k3; R1, R2 with k2, k3) whenever such a release exists, even if the
registered interval crosses zero (adjudication of challenger 19: not conditional). The refit counts as genuinely effective only if
the split-assignment hashes differ, the SCORE household overlaps are recorded and lie in [0, 1), and grid validity holds on all
eight reader seeds (recorded checks); otherwise the cells are counted once.

### 10.5 Bounded component ablation of the strongest feasible nonzero release

At the same lambda and capacity: ABL_LOC (pair weight 0: income and occupation dependence terms only, i.e. separately designed
recipients) and ABL_PAIR (local weights 0: the pair term only). 4 configurations. Read on the registered grid against the strongest
release and TWIN. Descriptive attribution, not a joint-design claim.

## 11. Revisions and stop rules

- Up to two justified revisions of the protected stage. Before each revision's fit, a committed card records the failure, the
  proposed change, its control and its stopping rule. The FA_PRC_v1 revision plan is the default menu (R2 optimizer: exact
  single-person moves under the same objective; R3 guard: kappa = 4 once).
- At most one structural repair of the task-only stage (T6), with the same record before its fit. It is labelled POST_HOC, its card
  is committed before any MODEL_SELECT read of its outputs, it is judged by the same construction choice and QUAL (its card cannot
  change them), and it can give only `POST_HOC_REPAIR_QUALIFIED`, never `REPAIRED_TASK_ONLY_RELEASE`.
- A failed F4 (section 7) is the registered trigger of R2; R2 is fitted only under its own committed card, and F4 is re-read on the
  revised path.
- If the routing approach fails, its result is preserved. A different successor may be fitted only after a new mechanism card (prefix
  FS_) and its controls are reviewed by the challenger and statistical reviewer and committed, within the same access and resource
  limits.

## 12. Configuration plan (48 at most, controls included)

| Block | Configurations |
|---|---|
| T2 TWB_C64, TWB_C128 (B1) | 4 |
| T5 TWB_C256 or TWA_C256 (conditional) | 2 |
| T6 repair (conditional, POST_HOC) | at most 2 |
| P2 PRC_S1-S3, ORD_S1-S3, HQ_S2, SHUF_S2 | at most 16 |
| P2 brackets HQ_BR, SHUF_BR, ORD_BR | at most 6 |
| E4 ablation | 4 |
| Revisions (two at most) | at most 14 (the remainder) |

Sum of the planned blocks without revisions: at most 34; the 48 limit is enforced by `frc.release.save`.

## 13. Registered predictions (written before any label read)

1. Q-PE at (16, 64): nested B1 - A occupation NLL in [-0.005, -0.001] nats in both versions; P(PE_ERROR_SUPPORTED) about 0.5.
2. Q-PE at (24, 128): B1 - A in [-0.008, -0.003]; |difference| larger than at (16, 64) in both versions; P(SUPPORTED) about 0.65.
3. B0 - B1: |difference| < 0.0005 nats (the prototypes are close to the cell means).
4. B1's nested held-out occupation spend over CONT stays above 0.004 at both capacities (about 0.010-0.013 at 64 cells, 0.007-0.010 at
   128): the original capacity gate still fails (not this study's rule).
5. B1 meets G1 at both capacities. Under the S2-slack QUAL: P(B1 qualifies at (16, 64)) about 0.45 (it must recover at least 0.00055
   of the version-1 men's guard); P(the protection capacity is (16, 64) or (24, 128)) about 0.85. QUAL(A, (16, 64)) is known from
   published records (fails) and is not a prediction.
6. Protected stage: PRC_S2 pair AUC below TWIN by 0.005-0.020; P(F4 passes) about 0.5; P(AGG_POSITIVE vs TWIN) about 0.45;
   P(beyond the SHUF path) about 0.25; P(beyond the HQ path) about 0.2; P(DOMINATES the ORD path) about 0.15; P(D2) about 0.08;
   P(D3) about 0.03; P(D4) about 0.01. The distance to NEURAL_RAW and H_LJ (0.015-0.018 pair AUC) was measured on the A twin; the B1
   twin's recovery may differ, because smooth readers see different coordinates on the same partition.

## 14. What is not claimable

Protection against arbitrary readers; confirmation; independent replication from versions, repetitions or reader seeds; confidence
preservation from point feasibility; superiority from a reserve-frontier or narrower comparison; a joint-design advantage without its
controls; novelty from performance; anything about the closed roles.

## 15. Completion

Completed 2026-10-20T23:47:14Z (`date -u`, read before this line was written), after the adjudication of both pre-registration
reviews. No label had been read by this campaign. The engineering reviewer's code review was still running at this time; any
change it causes in code that governs a fit is committed before that fit and recorded in PROTOCOL_DEVIATIONS.md if it departs from
this text.
