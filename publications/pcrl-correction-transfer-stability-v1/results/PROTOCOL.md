# PCRL correction transfer, genuine refits and a stable method comparison: registration

**Status.** Registered before any new fit or real-label computation of this campaign. The registration commit adds this file,
`PARTITIONS.json`, `EXPOSURE_MAP.md`, `ACCESS_POLICY.json`, the statistical reviewer's `roles/statistical_reviewer/RULES.md` and
`RULES.json`, the label-free `SOURCE_ADMISSION.json` and the campaign code (`cts/`). Label reads are refused by the single access
layer `cts.common.load` until these files are committed AND their sanitized public receipt is verified on the GitHub remote with
`git ls-remote` (`cts.receipt verify`). Only label-free admission and checks of already published numbers precede it. This is
exploratory development on previously used records, not confirmation. Later decisions are dated amendments recorded before the stage
they affect; locked files are never edited to insert retrospective explanations.

**Starting point (unchanged).** The occupation-headroom campaign (publication 1435c395d, research tip ca8725e72) ended
NO_QUALIFYING_CONFIGURATION; nothing here reinterprets it or applies a new selection rule to it. What stands from it: the occupation
witnesses give development margin, including the men's Brier guard; their advantage over the equally exposed TASK_REFIT is modest;
their task-model seeds are identical or near aliases; the B80 coalition-aware correction lowers declared-reader recovery beyond
continuation and compression but spends too much subgroup reserve on one seed; safer corrections do not surpass the strongest
inherited releases; the subgroup-cost prediction error is comparable to the slack being protected; no final ATTACK or DEV_REPORT
label was opened. The source protection card disclosed non-nested upstream cross-fitting, frozen routing and an exposed H3 income
anchor: possible explanations of the transfer error, tested here, not assumed.

## 1. Questions and success criteria

- **A. Cost estimation.** Can incremental occupation and subgroup costs be predicted on excluded households accurately enough to
  choose a useful correction strength? A successful diagnosis is not a competitive method.
- **B. Method stability.** Does an improved correction rule keep actual utility while delivering a repeatable recovery benefit
  beyond its strongest matched controls? A useful method result must improve the measured utility-recovery frontier, not merely
  choose a smaller correction or change a label.
- **C. Comparison design.** How much do the old recipient screen and unequal reserve requirements affect interpretation? A revised
  screen creates no algorithmic gain.

Objective: purpose-specific probabilities useful for recipient thresholds, ranking or deferral (illustrative benchmark uses), with
lower unintended demographic recovery. The recipient-access adversary (explicitly granted linked outputs, labelled auxiliary
examples, one release per person) is standard; it is not claimed to be new.

## 2. Data, service and limits (unchanged definitions)

Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, housing-unit residents aged 25-64, civilian employed;
the replication's cohort, household grouping and role assignment (`SOURCE_ADMISSION.json`). Tasks: income class (2) and occupation
group (5 official broad civilian groups). Runtime fields AGEP, WKHP, COW, SCHL, MAR, POBP only. Each recipient receives only its own
float64 probability vector and argmax decision. SEX is the recorded demographic column: a training-time label (critics, subgroup
costs) on authorized training households and an evaluation label for development readers; never a runtime input. U0 and Ucal are
the frozen utility references. Limits (never relaxed): accuracy loss <= 0.010, NLL excess <= 0.010 nats (inherited 1e-12
probability floor, kept), summed Brier excess <= 0.005; 24 utility clauses = 2 tasks x 2 references x 3 metrics x 2 source seeds;
the men's and women's occupation Brier guards (<= 0.005) are separate; original recipient non-worsening allowance 0.010; the 0.020
material-benefit target is separate from positive superiority. The two-SE reserve is a development robustness objective, not a
replacement for these limits or a guarantee. New training repetitions never silently change the 24-clause indexing.

## 3. Boundaries and resources

| Item | Limit |
|---|---|
| Elapsed | research window 8 h from 2026-10-18T18:32:40Z (first action) to 2026-10-19T02:32:40Z; closeout allowance 30 min to 03:02:40Z; enforced by `cts.jobqueue` |
| CPU | 14 CPU-hours in total, including agents, failed attempts, refits and reader fits (`cts.jobqueue charge` for out-of-queue work) |
| Concurrency | one shared queue; at most two heavy numerical jobs at once; one numerical thread each; code and dependency hashes recorded at job START (`CODE_AT_START.json`, re-hashed at launch) |
| Fitted release specifications | at most 72 across training repetitions (witness fits, correction directions per arm / routing / variant, compression codes, redesign fits, Track R whole-service refits). Ladder strengths selected from a fitted direction are recorded selections, not new fits. Routing, critic, calibration, cross-fitting and reader fits are counted separately; model seeds, training repetitions, reader seeds, configurations, cache hits and people are counted separately |
| Data | no new people, population, microdata download, sealed assessment or capped-out population; ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels closed for the whole campaign (their features only label-free, for deployment parity); no paid or cloud compute |
| Software | CatBoost 1.2.8 (the replication's pinned version) installed from PyPI into a campaign-private virtual environment, used only to fit routing teachers; inference stays numpy-only |
| Agents | waves, at most six live including the coordinator; agents never add heavy processes beyond the queue |

Branch `research/pcrl-correction-transfer-stability-v1` (own worktree, based on the source research tip ca8725e72; local, its
history holds private paths). Main, manuscripts, source branches, earlier publications and locks are not modified. Person-level
arrays, fitted models and reader artifacts stay under `.private/`. Remote: `origin` = https://github.com/Bostesa/PCRL.git (the
GitHub remote, not a local mirror; checked at registration). Interpreters: `python3` (3.14.3, numpy 2.4.4, scikit-learn 1.8.0) for
tree learners and evaluation; the project `.venv` (3.13.7, torch 2.10.0) for correction training; the CatBoost venv (3.13.7, numpy
2.4.2, catboost 1.2.8) for routing fits.

## 4. Roles, partitions and access

Admitted roles: MODEL_FIT 20,000 people / 13,337 households; CALIBRATION 2,982 / 1,974 (no SEX); MODEL_SELECT 8,001 / 5,306
(adaptively used by every earlier campaign; development evidence only). `PARTITIONS.json` freezes the label-free household
partitions of MODEL_FIT:
- repair CHECK panels `C0` (2,981 people / 2,001 households) and `C1` (2,997 / 2,001), reserved now and excluded from every new
  diagnostic and repair fit, calibration and preprocessing fit; their labels are refused until the committed REPAIR_FREEZE;
- pool (14,022 / 9,335) for everything else;
- D0, D1 (diagnosis; seeds 0, 1): FIT 60% / GUARD 20% / CHECK 20% of the pool (about 8,400 / 2,800 / 2,800 people);
- R0, R1 (repair; seeds 0, 1): FIT 75% / GUARD 25% of the pool (about 10,500 / 3,500 people); CHECK = C0 for R0, C1 for R1.
FIT is used for task learning, the preprocessing dictionary, routing, correction directions, training critics and cross-fitting;
GUARD only for choosing a strength from the fixed ladder; CHECK only for realized cost transfer after that choice. CALIBRATION (for
permitted calibration) is disjoint from every CHECK. The D0 / D1 FIT sets share 3,329 households and the R0 / R1 FIT sets 5,268: these
are different training partitions of the same people, not independent population replications. These are internal outcome-reserved
checks on previously used people, not fresh confirmation and not people untouched by historical models (every MODEL_FIT person was
seen by U0, T_TREE, the Stage B corrections and H3). A registered support check (class x SEX counts, before any fit) may move a
D or R partition to its next tag suffix; the panels are fixed.

Access is defined by role x field x operation x purpose (`ACCESS_POLICY.json`, enforced by `cts.common.load`): source replay,
support count, fit, calibrate, select, check_diag, check_repair, benchmark, readers, screen_study, verify. Each operation has
authorized entry points; diagnostic and independent-verifier entry points are authorized there before use, and later additions are
dated, committed amendments (`ACCESS_AMENDMENTS.json`) made before their first use. MODEL_FIT labels are returned only for the rows of
the requested partition role. Every label read appends one line to the access log (entry point, operation, purpose, rows hash).

## 5. Preflight (before real fits)

- Representative source occupation, income and correction outputs recomputed from admitted features (witnesses A1_PO_16 and
  B2_SINGLE, H3 income, P2_PAIR_B80 / P2A_PAIR_B80; both seeds) against the stored releases (label-free; `preflight/SOURCE_OUTPUTS.json`).
- Source replay after the receipt: the P2_PAIR / P2A_PAIR cost curves and B80 / B50 projection decisions (both seeds) recomputed with
  `ohn.protect.project` from the stored out-of-fold arrays and compared with the recorded scales, plus the source's realized
  MODEL_SELECT incremental costs of the selected releases (its transfer error); `preflight/SOURCE_REPLAY.json`.
- Routing reproduction: the replication's T_TREE occupation recipe refitted on all MODEL_FIT with the installed CatBoost and compared
  with the admitted teacher (exact or not; never used in a release).
- Exposure: `EXPOSURE_MAP.md` (from source code) states which fitted component saw which role.
- Infrastructure repaired with data-free tests before the lock: the receipt helper no longer writes its write-once record on a failed
  verification (`cts.test_receipt`); job-start code hashes in the queue.

## 6. Track F: fixed H3 income, newly fitted occupation pipeline

Per partition P (associated seed s), with the occupation pipeline refitted on FIT only: the FIT-only dictionary (an admitted level
with fewer than 10 FIT people moves to its block's OTHER column; because the admitted dictionary kept levels with >= 10 MODEL_FIT
people, the composite equals a dictionary built from FIT raw codes alone; AGEP / WKHP enter through the admitted increasing affine
map, to which the tree learners are invariant); the witness (one fixed specification, the A1_PO_16 recipe: HistGradientBoosting,
compact_po coding, 16 leaves, min leaf 100, lr 0.025, l2 3, T* <= 1000 by household 5-fold inner CV on FIT, refit on FIT; household
cross-fitting on FIT with nested inner CV gives the out-of-fold base and 5 fold pipelines); REFIT routing (the replication's T_TREE
occupation recipe on FIT) and the FROZEN routing (admitted T_TREE of seed s, fitted on all MODEL_FIT); correction directions with the
source P2 recipe (`ohn.protect.crossfit_train`, guard none): PAIR (critics on the occupation wire and the linked pair with H3's
frozen income wire of seed s), LOC (occupation wire) and CONT (critic weight zero, matched effort; it still reads FIT SEX for
inactive critics, so it is not called SEX-blind), 5 fold pipelines on the witness's outer folds plus the full-FIT pipeline. Releases on
the fixed ladder S = {0, 0.1, ..., 1.0}: deployed r_s = softmax(log p_W + s Delta_full); deployed-ensemble e_s = mean_k softmax(log
p_Wk + s Delta_k). The sensitivity witness (B2_SINGLE recipe) is a bounded check on one partition. No broad task-learner search.
Model identity is audited: different household FIT sets must reach every refitted layer; deterministic aliases count once.

Track F evidence is occupation-only: H3's income saw every MODEL_FIT household, so the complete pair is never wholly
excluded-household; occupation CHECK losses are excluded-household for the newly fitted occupation layers (REFIT routing) and not
for the FROZEN routing.

## 7. Diagnosis of cost transfer (D0, D1)

Within each D partition: directions are frozen before any GUARD or CHECK outcome is computed. For routing (FROZEN, REFIT) x arm (PAIR,
LOC, CONT) x deployment (full, ensemble), all ladder points, the incremental costs against the uncorrected release on the same people
(men's / women's / all summed-Brier increase, NLL increase, accuracy loss; household-clustered SEs) are computed by the source's
out-of-fold estimator (OOF, within FIT), on GUARD, on CHECK and on MODEL_SELECT; with signed and absolute errors, errors relative to
the available slack, constraint crossings, identity selections and correction magnitude (mean absolute logit change, flip rate).
Baseline slack (the witness's MODEL_SELECT margins with their SEs) and incremental cost are reported separately; baseline uncertainty
is not free budget. Actual probability curves are used; no metric is assumed monotone in strength. Explanations tested: frozen
routing (FROZEN vs REFIT), train/deploy mismatch (OOF fold pipelines vs deployed full refit vs deployed ensemble), exposed H3 anchor
(PAIR vs LOC; Track R), non-nested upstream cross-fitting (a nested variant on at least one partition: every upstream layer refitted
inside each outer fold, compared at the same strengths), correction magnitude (error by strength) and subgroup sample variation
(error relative to its SE, across partitions). Diagnostic CHECK is never used to choose a strength; it may motivate the repair (design
use, disclosed).

## 8. Strength rules and the repair comparison

The statistical reviewer's `RULES.md` / `RULES.json` (frozen here, implemented in `cts/rules.py`) specify OLD (the source projection
reproduced), SIMPLE (conservative scalar shrinkage on GUARD over the fixed ladder) and UNCERTAINTY (household-aware uncertainty in
GUARD costs and baseline slack, explicit subgroup uncertainty, finite ladder selection), the transfer metrics, the claim ladder, the
development-recipe selection, the repair comparison plan, the screen comparison and the frontier definitions. Identity is always a
valid outcome and earns no protection-component claim. A bootstrap or normal upper bound is not a distribution-free certificate.

At most one direction redesign, only if the diagnosis shows that the direction rather than its scale estimate drives unequal group
cost: its card (with a falsification condition and the simplest alternative explanation, and its unweighted control) is committed
before its first fit; the coordinator may reallocate unused configurations to it after written statistical review.

**REPAIR_FREEZE.** Before either repair CHECK panel is opened, the committed `REPAIR_FREEZE.json` records: the frozen OLD, SIMPLE and
UNCERTAINTY rules and any redesign; the one selected development recipe (direction x rule), chosen by the predeclared criteria on
GUARD and diagnostic-stage evidence; the comparison plan; and the R-partition fits it governs. All frozen rules are then evaluated on
R0/C0 and R1/C1, but the selected recipe is not switched to the winner. After a repair CHECK outcome is seen, revisions using it are
adaptive and descriptive and cannot replace or relabel the primary repair result. A public receipt of the freeze is published if the
network allows; otherwise its local commit time is the record.

## 9. Track R: whole-service refits

On R0 and R1, if the source recipe can be rebuilt within budget: T_TREE income and occupation (CatBoost recipe on FIT), Stage B
LOCAL and JOINT corrections (replication recipe, SEX critics on FIT), H3's income learner (HGB Step 1 on FIT), its CALIBRATION
stack, beta budget and Platt map, then the occupation pipeline with the refitted income wire. No stored SEX-trained correction that
saw CHECK households is reused. A recipe test on different training subsets, not confirmation of the frozen H3 package; historical
controls keep their actual exposure. If it cannot be completed, Track F finishes and whole-service refit stability is marked untested.

## 10. Readers, controls and the screen study

Every serious candidate is compared with the unchanged witness, the matched zero-critic continuation, the local correction, the
coalition-aware correction, an ordinary compression code at matched cost, and the OLD / SIMPLE / UNCERTAINTY selections, with
matched task access, output contract, training effort and reader effort. Development readers use MODEL_SELECT with the source grid
(household-disjoint FIT 50% / VALIDATION 20% / SCORE 30%; repetitions R1, R2; reader seeds k0, k1; the inherited 16-recipe slate with
ignore-recipient alternatives; recipes chosen on VALIDATION, recovery measured on SCORE; household-bootstrap SEs; winner near ties
recorded); frozen-reference results are reused as counted cache hits; a third repetition and nested learning curves for finalists
(never more fitting people than exist). Paired candidate-control differences are reported; training-partition, model-fitting and
reader-fitting variation are separated; many fits and overlapping repetitions are not population replications. Readers see only
released probability vectors and decisions.

The screen study (question C) characterizes the old per-cell recipient gate against the statistical reviewer's predeclared
aggregate paired rule on frozen source releases, exact aliases, identical releases under separately fitted readers and known-law
synthetic shifts, before any new method outcome is seen by readers. Results are simulated rates under those laws only. The old gate
is reported beside the new screen for every new configuration; the old campaign's label is never rewritten; a screen admitting a
candidate is permission for further development, not evidence of protection.

## 11. Comparisons, claims and the fallback programme

Two comparisons, labelled separately: (A) the original benchmark, keeping every original point-eligible strong reference (NEURAL_RAW,
H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, including weak-reserve references and the retired release), with the original conditions and
the 0.020 target; (B) the reserve-qualified frontier, applying the same reserve to candidates and controls and showing which
historical references fail it. A positive B is not superiority over every A reference. Actual accuracy, NLL, Brier, subgroup burdens,
recovery and uncertainty are shown together; no exact cost match is enforced, no frontier is interpolated into a victory, and if no
reserve-qualified protected control exists that is said. Track F optimizes occupation only: a coalition-aware occupation
correction, never a joint two-output design. The source's income-accuracy weakness stays visible.

At most one development recipe is selected before repair CHECK; final ATTACK roles and DEV_REPORT stay closed; at most a frozen
development candidate and a reviewable proposed final-evaluation protocol are delivered. If the method fails, the fallback programme
runs within the remaining budget: replicate the diagnosis across partitions; decide whether transfer error is systematic or too
uncertain to resolve; compare nested and non-nested pipelines at the same magnitude; test whether a simple rule explains an
uncertainty-aware rule's benefit; identify what drives the frontier (men's or women's cost, task confidence, income accuracy,
recovery); complete the screen study; verify model identities and the effect of training-household changes; package the strongest
honest task witness. Exhausted lines stop honestly; no impossibility claim follows from this finite experiment.

## 12. Verification, deployment, custody and publication

An independent verifier with separate metric and selection code checks source pins and role membership, actual exclusion of GUARD and
CHECK from each claimed upstream fit, training-set and model-identity differences, cost estimates, realized costs and errors, rule
selections and zero-correction accounting, recovery comparisons and uncertainty, original and prospective gates (separately), frontier
labels and scope, and restored deployment. Its label reads go through the same access layer under the 'verify' operation. It never
widens a failing criterion; a genuine check bug gets a separately recorded correction (original result, changed rule, reason,
independent adjudication) in an append-only check history; numerical replay and procedural compliance get separate statuses.
Packages: an unchanged anchor, a task witness, the selected correction if any and the decisive control, deployed from the six raw
fields, bitwise against stored wires, refusing missing, extra, reordered, target and SEX fields at the documented and underlying
entry points, with no hidden outputs, identifiers or subgroup labels at runtime. Backup and restore of representative models,
corrections and readers from the backup alone; off-device custody only on an already attached writable drive (never forced; a warm
read is not called cold). Sanitized protocol and reporting records go to the dedicated publication branch
`publication/pcrl-correction-transfer-stability-v1` on the GitHub remote (git plumbing; no history rewritten; earlier branches
untouched); remote SHAs and public entry points are checked; nothing is backdated; no messages are sent to people (repository handoff
files only).
