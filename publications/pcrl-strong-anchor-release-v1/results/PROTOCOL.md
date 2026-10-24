# Protocol: strong-anchor release development (campaign `pcrl_strong_anchor_release_v1`, code `sar/`)

Coordinator. Writing started 2026-10-24T04:30Z (`date -u`); the completion time is in section 16. EXPLORATORY DEVELOPMENT on
previously and adaptively used public-use ACS 2019 1-Year PUMS Florida person records (the admitted development cohort). Nothing
here is confirmation.

## 0. Purpose, status and disclosure (before any label read)

- **Purpose.** A statistical machine-learning benchmark. A service predicts an income class (2) and an occupation group (5) from
  six raw fields. The privacy evaluation measures whether classifiers can infer the dataset's recorded SEX column (0 = men,
  1 = women) from the probability vectors released for those tasks: from each output alone and from the linked pair. This campaign
  starts from the strongest usable frozen releases (not from a weak task-only service), identifies their binding conditions, and
  develops substantive operations that address them, with fitted matched comparisons. A successful method is not assumed.
- Campaign start 2026-10-24T04:02:04Z (first `date -u`). Research window ends 12:02:04Z; closeout ends 12:32:04Z (the queue refuses
  research stages after 8 h and every stage after 8.5 h).
- **Sources** (pinned and verified label-free by `sar.admit`, `SOURCE_ADMISSION.json`, `ok`): the input-and-pair release campaign
  (publication 9c2aa4d99, child of its registration receipt e909fb8c0; research tip 9c9f95385 = the base of this branch
  `research/pcrl-strong-anchor-release-v1`), the dual-design release (e30cf1c6b), finite-solver adequacy (ddc231a97), finite-routing
  comparison (04ac9161f), new release mechanisms (682068d52) and the chain they admitted. Their conclusions are preserved unchanged
  (`FAILURE_MAP.md`).
- **Known before writing** (published or label-free): every published record of those campaigns, including the IPR1_MAIN grid
  AUCs and MODEL_SELECT utility of every reference (`anchors/ANCHOR_TABLE.md`, built label-free from saved records). The anchors were
  chosen from these records (`anchors/ANCHOR_SELECTION.md`, committed 9dc59946b before any fit): an informed, adaptive choice.
- No outcome of this campaign exists. No label has been read by this campaign.

## 1. Data, roles and the service (unchanged definitions)

- Admitted Florida ACS 2019 cohort only. MODEL_FIT (20,000 people, 13,337 households): all fitting; SEX a training label only where a
  committed card says so (a protection objective or a within-SEX utility constraint). CALIBRATION (2,982 people): income / occupation
  labels only if a card says so (never SEX); calibration maps are otherwise fitted on separated MODEL_FIT household folds or frozen
  admitted calibrators are reused. MODEL_SELECT (8,001 people, 5,306 households): logged development utility evaluation and reader
  fitting / selection on its predeclared subroles; never a fitting input of any task learner, calibration map, release parameter,
  strength, control point or selection rule. ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels, the sealed assessment and the
  capped-out records stay closed (DEV_REPORT features label-free for deployment parity only). No population is added and no records
  are downloaded. Closed labels are never computed and discarded.
- Runtime fields: AGEP, WKHP, COW, SCHL, MAR, POBP (admitted 145-column encoding X). Exact source task definitions, class counts,
  occupation grouping, preprocessing, household groups and cohort manifests of the admitted chain. Reference training exposure (the
  20,000 MODEL_FIT people) unchanged.
- Service: each recipient receives only its own probability vector (float64) and its argmax decision. The six fields are model
  inputs, not disclosed fields. The evaluated recipient can fit a SEX reader from the declared labelled auxiliary data; the linked-pair
  recipient is explicitly granted person-level linkage. One released instance per person (a randomized release emits one draw per
  person in every role; further stored draws are Monte Carlo repetitions, never pooled per person). No repeated-query, model-access or
  insider claim. The probabilities serve recipient thresholds, ranking or deferral in the illustrative service: motivation, not
  evidence of an observed deployment.
- Access: `sar.common.load` is the only label path (ROLE x FIELD x OPERATION x PURPOSE; `ACCESS_POLICY.json`); every read is logged;
  no label read before this registration is committed and its sanitized receipt is verified on the GitHub remote (`origin` =
  https://github.com/Bostesa/PCRL.git, checked); fitting reads need a committed, unchanged card; MODEL_SELECT labels are never read
  inside a fitting entry point (`sar.fa`, `sar.fb`, `sar.ctl`, `sar.fs`); label reads run only inside queue units.

## 2. Storage and workspace admission

- **Worktree.** New isolated research worktree `.claude/worktrees/pcrl-strong-anchor-release-v1` (created 04:11:05Z with
  `git worktree add` from the input-and-pair tip; the session switched into it). Main, manuscripts, prior study branches and other
  projects are not changed; the source worktree `pcrl-correction-transfer-stability-v1` is read-only for this campaign.
- **Relocation index read** (`~/storage-relocation-20261022/`: manifest.json, RESTORE.md, LEDGER.csv, verify_restored.py, the
  dot-archive-first extraction order for `._*` side archives). 84 items are external-only on YOTUO; 72 archived and kept.
- **Drive.** YOTUO was absent at 04:02:04Z and present at 04:25:08Z (not mounted by this session): `/dev/disk6s1`, ExFAT (FSKit), USB,
  volume UUID 3D74932B-33DD-3896-8CC6-0D2FDDAD6B8D (equal to the manifest), 1.0 TB, 582,460,375,040 B free, writable. The external
  index (the drive's `NathanSamson-Mac-relocated-2026-10-22/manifest.json`) is byte-identical to the local copy (SHA-256
  f904c871...4efc). Nothing on the drive is written before closeout; existing archives and relocation records are never modified.
- **Dependencies.** The anchors (SUCC, NEURAL_RAW), every reference and every source artefact this campaign reads are local: the
  protected-tree replication tree (`.worktrees/pcrl-acs-protected-tree-replication-v1/.private/acs_replication`, archived and kept),
  and the inherited private trees, cloned (APFS `cp -cRp`, 04:11:26-31Z) from the source worktree into this worktree's `.private/` as a
  separate working cache and verified file by file (sizes and SHA-256; `SOURCE_ADMISSION.json` `working_cache`). The relocated items
  the chain's code mentions (the occupation-headroom, income-headroom and autonomous-method-development worktree `.private` trees,
  the frozen-hybrids `.private`) are read through their admitted mirrors in the working cache, hash-checked against their admission
  records. **No archive is restored** (none is needed); if a later need appears, the item is restored from YOTUO into
  `.private/pcrl_strong_anchor_release_v1/restored/` with RESTORE.md's commands, archive and restored-file hashes verified, and
  recorded.
- **Space.** Free internal space 126.1 GiB at 04:26Z; floor 20 GiB; peak temporary storage recorded in `RESOURCE_ACCOUNTING.json`.
  Disposable caches (reader cache, restore-test folders) are checkpointed and reduced before approaching the floor.

## 3. Anchors (frozen: `anchors/ANCHOR_SELECTION.md`)

- **A: SUCC** (H3_STRONG_LJDIR_PB income + JOINT_TREE E20 occupation). Binding: the men's within-SEX occupation Brier guard
  (+0.0044 / +0.0039 against 0.005; reserve 0.32 / 0.62); the pair sits 0.034 above its occupation view.
- **B: NEURAL_RAW** (rho 1, E2). Useful with point margin on every condition (max 0.34 / 0.57 of cap); its protection is mild and
  sits in the occupation view (pair = occupation view = 0.6253).
- A retired release (RETIRED_B3CTL) remains a reference without its retired claim.

## 4. Families, investigators, cards and the order of work

Two investigators work independently: **FA** (investigator A; can SUCC recover usefulness margin while retaining its recovery
profile, and turn it into lower recovery?) and **FB** (investigator B; a different construction of the two task outputs that targets
recovery directly while retaining useful probabilities). Families: `FA`, `FB`, `CTL` (shared controls), `FS` (at most one replacement
family). Each may redesign both task models, learn a new mapping from the permitted inputs, or use admitted teachers as training
signals; deployment uses the six fields only (never SEX, a true task label, undisclosed recipient information or unavailable
person-level state). Closed recipes (`FAILURE_MAP.md`) are not continued; a related idea must show its structural change in its card
before its fit.

For each family, before its first fit, a short prospective mechanism card `results/<campaign>/cards/<FAMILY>_<name>_v<k>.md` is
committed with the governing code (template `cards/README.md`). It fixes: the starting release and the binding condition; the exact
operation; the closest existing implementation and the operational difference; what would contradict its rationale; the bounded
choices (strength grid, the selection rule on the declared selection data (MODEL_FIT out-of-fold or nested held-out losses, or CALIBRATION task-label paired costs when START is in-sample on MODEL_FIT; guard parts MODEL_FIT out-of-fold; never MODEL_SELECT), component budgets chosen from the actual anchor margins); the matched controls
(section 6); the measurements; the failure interpretation; registered predictions; the specification count. A card change after its
first fit is a new card (revision), never an edit.

Order: registration -> cards and code (selftests on invented data) -> MODEL_FIT fits, strength selection and control points on the declared selection data ->
MODEL_SELECT utility of every saved release -> SAR1 reader grid (candidates, controls and references on the same cells) -> decisions
-> diagnosis and revisions (each with a recorded diagnosis, mechanism change and predictions before its fit) -> robustness cells for
promising comparisons -> verification, packaging, custody, publication. No synthetic or preliminary privacy result is required before
the first fitted comparison; an initial fitted comparison for both families is the aim for the first half of the campaign.

## 5. Task feasibility (original limits; unchanged)

`sar.evaluate` (= `nrm.evaluate` definitions, U0_S<v> / Ucal_S<v>): accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed Brier
excess <= 0.005 (2 tasks x 2 references x 3 metrics = 12 clauses per version) and the men's and women's occupation summed-Brier excess
vs U0 <= 0.005 (2 guards per version): 28 conditions over the two versions (24 clauses + 4 guards), verified against the authoritative
`nrm.rules` / `sar.evaluate` definitions. **G1 = all 28 at the point.** Point eligibility, the confidence conjunction and the inherited
two-SE reserve heuristic (reported beside, never gating, never redefining eligibility or removing an otherwise eligible reference) are
reported separately. The confidence conjunction (reported only): for CAND, START and CONT, each condition from its own
household-clustered SE (with se_mc in quadrature for a randomized release; no floor), z = 2 one-sided per condition: PASS iff
value + 2 se <= cap, VIOLATION iff value - 2 se > cap, UNRESOLVED otherwise (neither a pass nor a violation); 28 one-sided bounds,
not a simultaneous statement, no multiplicity correction (RULES.md 2). Estimated MODEL_FIT fitting cost is not a bound on held-out cost: the reported usefulness result uses the unchanged
caps on MODEL_SELECT. Versions v = 0, 1 are the two procedure repeats of the admitted chain (shared training rows and MODEL_SELECT
people; repeats, not replications).

## 6. Comparisons and controls

Every serious release is compared with:

| Comparator | What |
|---|---|
| START | its unchanged starting release (FA: SUCC; FB: NEURAL_RAW), frozen, both versions |
| CONT | an equally exposed task-only or zero-protection continuation (same learners, data, folds, seeds, iterations and selection effort as CAND with the protection term off), where attribution needs it (declared in the card) |
| CTRL | the simplest untargeted operation matched to the mechanism, at comparable task cost: a path on its own grid, two points saved before any MODEL_SELECT read (the largest grid point whose spend over START on the card's declared selection data (MODEL_FIT out-of-fold, or CALIBRATION task-label paired costs when START is in-sample on MODEL_FIT; never MODEL_SELECT) is at most CAND's, and the smallest whose spend is at least CAND's plus 2 household-clustered SE from the same data; flagged CONTROL_PATH_MAY_NOT_BRACKET if none reaches it), read on MODEL_SELECT with the inherited spend-bracketing frontier rule; a missing bracket reads NOT_SEPARATED with that reason (a limit of the control, never evidence of equality); the residual cost mismatch is reported, and the mechanism's contribution is never called isolated when matching fails |
| ALT (optional) | the starting release's own protection operation continued to matched spend; card-declared (`alt_declared`); origin START; the same bracket rule (spend on the card's declared selection data, with the SE from the same data) and frontier conjuncts as CTRL; gating when declared (FB: NEURAL_RAW's RAW-J critic recipe, whose E2 is NEURAL_RAW) |
| References | NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100 (frozen, point-eligible), re-read on this campaign's grid; U0 / Ucal utility diagnostics; DDR_PTC_R1 beside |


**Declarations and records (before each family's first fit; RULES.md 1, 4, 10).** Each card declares `cont_declared` (boolean),
`ctrl_origin` (START, or CONT when the untargeted operation starts there), `alt_declared` and the selection data of every path. Only a
declared CONT or ALT gates; a CONT built without a declaration is reported beside; a declared control not run for budget gives
NOT_ASSESSED_CONTROLS_NOT_RUN, never a pass. The SE in "CAND's spend + 2 SE" is the household-clustered SE of that grid point's paired
difference against CAND on the declared selection data: MODEL_FIT out-of-fold paired costs, or CALIBRATION paired costs (task labels
only) when START is in-sample on MODEL_FIT; never MODEL_SELECT. The full grid (spend and SE per point) is recorded so that
`grid_fineness` can check that the saved points are the largest / smallest qualifying ones. Each reading's plan file (RULES.md 10 and the
reviewer's report section 7) is committed before that reading's first MODEL_SELECT utility or reader computation, and its hash is
recorded in the reading's decision record.

Equal code size or equal training iterations alone do not establish equal usefulness cost. An invertible transform (temperature,
affine re-mixing) removes no information and is never a CTRL point. Every saved configuration gets the same utility evaluation and the
same registered reader cells; an identical released wire is evaluated and counted once. Joint design is a separate claim: only a card
that claims it builds the local twin and both sequential orders (section 8, D5).

## 7. Readers (what recipients actually receive)

- Fresh development grid SAR1 inside MODEL_SELECT (`READER_SPLITS.json`, written label-free 04:28Z): repetitions R1, R2 (registered)
  and R3, R4 (robustness; genuinely different household assignments of the same people); households ordered by
  SHA-256('SAR1_devgrid_<R>|<household>') and dealt into 10 parts: FIT 0-4, VALIDATION 5-6, SCORE 7-9 (household-disjoint), frozen
  before any new outcome. Reader seeds MODEL_SEED_BASE + 1000 + 10 v + k (k0, k1 registered; k2, k3 robustness); the seeds of every
  realized earlier grid (offsets 300-900) are forbidden. Household bootstrap of SCORE (B = 400, seed 2026102400 + 100 v + repetition
  index). Overlaps with the earlier grids (OHN1, TSZ1, NRM1, FRC1, DDR1, IPR1) are recorded. These remain reused development people,
  not a fresh confirmation set; new splits do not reverse their exposure.
- The strongest admitted bank: the 16 recipes (OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB, LOOKUP, ET5, ET25, HGB15,
  HGB31, MLP64, MLP6432, CATI01, CATI1) on the transmitted vectors and decisions only, fitted on FIT (the declared auxiliary sample),
  oriented and selected on VALIDATION, scored on SCORE; the pair winner chosen among pair readers and the ignore-one-recipient
  readers; grid extras (non-table winner, TABLE_CV, enhanced winner) recorded and reported beside. Candidates, controls and
  references get the same fitting and selection opportunity on the same cells. At most one bounded mechanism-specific reader per
  family (declared in its card or before its grid), reported beside, never gating. A few-valued output may make reader seeds
  ineffective: exact aliases are recorded and the actual diversity (n_eff) is counted; the comparison is not declared invalid for it.
  Operationally (RULES.md 9): a seed-dependent equality on a view with at most 320 distinct released inputs (8001 // 25) is explained,
  and counted once, iff the same recipe differs on another wire or view in the same version, repetition and seed pair; any other
  unexplained equality on the minimum set still invalidates the grid. The grid records distinct-input counts per wire, version and view
  for every minimum-set wire (a missing count is never few-valued).
  Code locked in `DEVGRID_CODE_LOCK.json` (written 04:28Z with the torch interpreter, committed before any grid run).
- Income, occupation and linked-pair recovery are all reported. The coalition can ignore either view; a fitted pair reader that does
  worse than a single view is an engineering diagnostic, not a population property.
- Uncertainty: the inherited aggregate screen (block means over version x repetition; se_agg = sqrt(se_hh^2 + se_rs^2 + se_mc^2),
  0.001 floor; both version means must agree). Versions, repetitions and reader seeds are not independent replications. Intervals do
  not correct the adaptive search and do not establish confirmation.

## 8. Decisions (kept separate; formal definitions in `roles/statistical_reviewer/RULES.md`, schema SAR1-rules-v1)

Per family reading (FA, FB at their registered cards; every revision or replacement read under the same rules with the ADAPTIVE_ /
REVISED_ prefix, reported beside, never as a registered result):
1. **D1 task feasibility:** G1 at CAND gates; the confidence conjunction and the reserve are reported separately (section 5).
2. **UM usefulness margin against START** (preparatory evidence only; RULES.md 3): START's binding condition per version is the condition
   with the smallest two-SE reserve ratio of START's MODEL_SELECT record (equal to the frozen anchor table: SUCC the men's guard in both
   versions; NEURAL_RAW the men's guard in v0 and income Brier vs U0 in v1; declared in the plan); a resolved improvement = the pooled
   paired CAND - START difference on that condition has its upper bound (+ 2 SE) below 0 and both version differences are below 0;
   MARGIN_RECOVERED also needs G1, the recovery profile retained (pair and both single views not more recoverable than START's by more
   than 0.010 per version at the aggregate point, and no resolved pair worsening) and CAND's whole-release point margin (min over the 14
   conditions of (cap - value) / cap) not below START's in either version (otherwise MARGIN_GAIN_SHIFTED_ELSEWHERE).
3. **D2 component gain** beyond START, CONT (when declared), the CTRL frontier and the ALT frontier (when declared): each with the pair
   benefit resolved (aggregate mean - 2 se_agg > 0 and both version means > 0) and the recipient gates; for each frontier, the
   bracketing point Q* by the inherited rule, no cheaper path point resolved less recoverable than CAND; and (d) reader targeting: the
   non-table winner's point pair benefit positive in both version means against START, CONT (when gating) and every Q* (else
   READER_TARGETED). Met: COMPONENT_GAIN_BEYOND_START_AND_CONTROLS; partial outcomes PARTIAL_<passed>_ONLY in the order START, CONT,
   CTRL_FRONTIER, ALT_FRONTIER, with a statement of what was not separated; NOT_SEPARATED is the default reading, never "equal".
4. **D2R robustness** to classifier refitting, read at CAND with the registered decisive set (START, CONT when gating, the registered
   CTRL / ALT brackets), never re-chosen on robustness cells; at least one of R3 / R4 plus k2 / k3 (both splits preferred;
   ONE_ROBUSTNESS_SPLIT_ONLY flagged); a sign change against any comparator, decisive or reference, goes into the headline sentence of
   that reading.
5. **D3** against every strong reference (per reference, resolved pair benefit with the recipient gates; the references passed and
   failed are listed; SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES only if all pass). D3's START entry is D2's START comparison, not a second
   piece of evidence.
6. **D4 the material target:** a resolved 0.020 pair benefit against NEURAL_RAW and against H_LJ with the recipient gates; never weakened;
   smaller gains keep their level.
7. **D5** joint design (only if claimed with its controls); **D6** novelty (NOT_CLAIMED unless separately reviewed).
8. **Outcome classes** (RULES.md): METHOD_CANDIDATE (D2 and D3), COMPONENT_GAIN_NOT_COMPETITIVE, REFERENCE_ADVANTAGE_NOT_ATTRIBUTED (D3
   without D2), USEFULNESS_MARGIN_ONLY, FEASIBLE_NO_RELIABLE_GAIN, NO_OUTPUT_CHANGE, plus the inherited infeasible and incomplete
   classes. No class implies D4.

Three outcomes are always reported separately: (1) any reliable component gain; (2) performance against the strong references under
the original conditions; (3) whether the original material target is established. A utility-margin improvement alone is preparatory
evidence, not a competitive privacy method; a recovery advantage against a weak twin alone is not competitiveness. Partial results keep
their partial names. Version-dominated block means carry the NRM1 caveat (about 8% false pass per comparison when one version or split
dominates); the block means of every passed comparison are reported, and the single-view and pair differences are reported with their
SEs beside every retention or recipient label (point rules: a true 0.010 worsening passes about a third of the time).

## 9. Revisions, replacement and stop rules

- At most two substantive revisions per family and one replacement family (`FS_`) within the global budget. Each revision records its
  diagnosis (from recorded development evidence), its mechanism change and its predictions before its fit, as a new card; the earlier
  card is never modified and an initial control is never relabelled as a nominated method. A revision never re-reads MODEL_SELECT to
  choose a parameter; a revision triggered by a MODEL_SELECT outcome is adaptive as a whole.
- Component budgets come from the actual anchor margins; a design that never changes the transmitted output is established as such
  directly and stops being treated as a protection experiment.
- The previously declined cross-recipient swap enters only with a new, substantive rationale.
- Stop a family when its revisions are used, when it cannot change the output, or when the budget forbids its controls (then the
  comparison is reported as not run).

## 10. Budget and accounting

At most 36 new seed-specific fitted release specifications (each candidate recipe, operating setting and version evaluated as a
release counts, matched controls and revisions included; `sar.release.save` refuses the 37th), of which at least 12 are reserved for
matched controls and follow-through. Internal optimizer iterations, learner checkpoints and cross-validation fits are recorded
separately unless they become separately compared release candidates. Replaying an unchanged frozen artefact is a replay (no
specification) but its resources are charged and its job logged. Eight hours elapsed (a ceiling, not a target), 12 CPU-hours
including failed attempts and out-of-queue work, zero paid compute, at most two heavy processes at once with one numerical thread
each, one shared queue (`sar.jobqueue`), at most six live agents including the coordinator (no recursive spawning).

**Planned allocation** (adopting the reviewer's finding 3): the initial comparisons use more than the design brief's indicative 6 because
every CTRL / ALT path saves two points in each version. FA: CAND 2 + CTRL (lambda = 0 interpolation, two points) 4 + CONT 2 = 8. FB: CAND 2
+ CONT 2 + CTRL (DQ, two points) 4 + ALT (RAW-J, two points) 4 = 12. Total 20 of 36, of which 16 are matched-control specifications; 16
remain for revisions (each with its own controls), a replacement family and follow-through. If a later reading's budget forbids a
path's lower point, only the upper point is saved (the origin serves as the lower point; fineness reads COARSE_BELOW; (c3) checks only
the origin), recorded in its plan.

## 11. Verification, packaging and custody

An independent verifier recomputes the decisive metrics, selections and outcomes from frozen outputs with separate code and refits a
bounded sample of winning readers; a failed check is preserved with any adjudication, never silently relaxed. Raw-input deployment
packages for surviving releases and informative controls bind the source, model and schema versions; checks: probability validity,
input refusals, absence of forbidden inference inputs, equivalence to the evaluated output (bitwise, or the stated tolerance). Stale
packages are rebuilt after code changes before the final backup; restore from the backup alone and rerun representative deployment and
reader predictions. New artefacts are copied to YOTUO (identity re-checked; existing archives preserved; no forced unmount; no cold-read
claim while the drive stays mounted). The sanitized publication (code and aggregate records; no private arrays, fitted private
artefacts, personal paths or secrets) goes to `publication/pcrl-strong-anchor-release-v1`; inherited research history is not pushed
unchecked.

## 12. Registered expectations (coordinator, before any label read)

- P(a family's CAND is G1-feasible) about 0.75; P(UM MARGIN_RECOVERED for FA) about 0.4; P(D2 met for at least one family) about 0.12;
  P(D3 for either) about 0.03; P(D4) about 0.02. Each card adds its own numerical predictions.
- The most likely outcome: FA recovers part of the men's guard margin at a small pair cost without a resolved recovery gain over
  SUCC; FB lowers NEURAL_RAW's occupation view by 0.005-0.02 at the point without separating from its untargeted control.

## 13. What is not claimable

Protection against arbitrary readers; confirmation; independent replication from versions, repetitions or reader seeds; confidence
preservation from point feasibility; superiority from a narrower comparison; competitiveness from a margin gain or from a gain against a
weak twin; a joint-design advantage without its controls; novelty from performance; anything about the closed roles; a non-registered
reading presented as registered; a campaign-level statement without the number of D2 looks and their registration status.

## 14. Known stale or reused items (disclosed)

`sar/` was derived from `ipr/` by a mechanical rename at about 04:20Z (campaign names, paths, reader-seed offset 1000, bootstrap base
2026102400, split tags SAR1); some docstrings still describe their sources' history. The family modules (`sar/fa`, `sar/fb`), any shared
controls (`sar/ctl`) and the rules (`sar/rules.py`, rewritten by the statistical reviewer) are new; draft family modules present at
registration are superseded by the versions committed with their cards.

The rules `sar/rules.py` and `sar/test_rules.py` were rewritten (not renamed) by the statistical reviewer (the mechanical copy's tests
asserted the IPR1 reader seeds and could not pass); their hashes are in RULES.md's completion record and the reviewer's hand-back.
Disclosed: SAR1's v0 bootstrap seeds 2026102400-2026102403 equal IPR1's v1 seeds; with the same SCORE size the bootstrap weight
matrices coincide but attach to different household sets, so no SE estimate is shared (not a validity condition).

## 15. Adjudication of the pre-registration review and of the designs

**Statistical reviewer** (`roles/statistical_reviewer/REPORT_PREREGISTRATION.md` and `HANDBACK_PREREGISTRATION.md`, saved verbatim by the
coordinator because the harness refused the agent's own write of its report). REQUIRED findings 1-9 are adopted in sections 5, 6, 7, 8
and 10 above in the reviewer's wording (the coordinator's answers: the UM whole-release point-margin clause adopted; the targeting
conjunct (d) kept as a gate; the 8 / 12 initial allocation; the few-valued bound 320). Advisories 10-15 are adopted in sections 8 and
14. The reviewer's revision 2 (05:07Z; `HANDBACK_PREREGISTRATION_2.md`, addendum
`REPORT_PREREGISTRATION_ADDENDUM.md`) implements the ALT path and the declared selection data; its REQUIRED findings 16-18 are adopted
in sections 4, 6 and 8 (and in every plan file); advisories 19-22 are adopted (an ALT point aliased to START only after a MODEL_SELECT
wire-hash check; the FB power statement; CALIBRATION-bracket conservatism; about two revisions in the remaining 16 specifications).
Every plan file states `alt_declared` explicitly. Two rule changes were then requested of the reviewer before the freeze (both before any outcome): an optional gating ALT path
(the starting release's own protection operation continued to matched spend), and declared selection data (CALIBRATION paired costs
when START is in-sample on MODEL_FIT). The frozen rules are RULES.md / RULES.json / `sar/rules.py` (schema SAR1-rules-v1) with the hashes
in RULES.md's completion record.

**Investigators' designs** (both label-free; hand-backs saved verbatim in `roles/investigator_a/HANDBACK.md` and
`roles/investigator_b/HANDBACK.md`). Both PROCEED with required changes, recorded in their cards before the first fit:
- **FA_TFR** (investigator A; START SUCC): a zero-initialized residual network over the compact six fields, SUCC's logits and the admitted
  HGB task teachers' logits, trained by distillation toward the teachers' out-of-fold outputs with the lineage's joint SEX critics
  (income, occupation, pair); strength on CALIBRATION paired spends over SUCC plus a MODEL_FIT out-of-fold guard difference against the
  matched lambda = 0 point. Not a scalar leaf-score correction, decoder or calibrator: a learned function of the permitted inputs that
  starts at a protected release and moves toward a stronger teacher where the critics accept it; both outputs jointly. Required: the
  card corrects the statement that SUCC is out of sample on CALIBRATION (H3 and Ucal are in-sample there; the income-side CALIBRATION
  spends are biased in SUCC's favour, conservative for CAND); CTRL = the lambda = 0 interpolation path, origin START, two CALIBRATION
  bracket points; CONT = the lambda = 0 point at full scale, declared.
- **FB_PFD** (investigator B; START NEURAL_RAW): both U0 networks retrained from U0 with an exact in-batch best-response penalty, the
  summed Gaussian AUC of the whitened (pooled within-SEX covariance) quadratic discriminant of each view and of the linked pair, one
  coalition strength; replaces NEURAL_RAW's lagging, diagonally normalized online critics and per-recipient multipliers. Required: the
  brief's untargeted control CTRL = SEX-blind dithered quantization of NEURAL_RAW (two MODEL_FIT bracket points); ALT = NEURAL_RAW's own
  RAW-J recipe continued to matched spend (two points); CONT = the lambda = 0 continuation, declared.

## 16. Completion

Completed 2026-10-24T05:10:24Z (UTC clock read in the same command that wrote this line), after the adoption of the statistical reviewer's
pre-registration review (both revisions) and the adjudication of both designs. The mechanism cards FA_TFR_v1 and FB_PFD_v1 are
committed with this registration. No label had been read by this campaign.
