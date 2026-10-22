# Protocol: an input-side family and a joint pair-release family (campaign `pcrl_input_and_pair_release_v1`, code `ipr/`)

Coordinator. Writing started 2026-10-22T01:38Z (`date -u`); the completion time is in section 14. EXPLORATORY DEVELOPMENT on
previously and adaptively used public-use ACS 2019 1-Year PUMS Florida person records (the admitted development cohort). Nothing
here is confirmation.

## 0. Purpose, status and disclosure (before any label read)

- **Purpose.** A statistical machine-learning benchmark. A service predicts an income class (2) and an occupation group (5) from
  six raw fields. The privacy evaluation measures whether classifiers can infer the dataset's recorded SEX column (0 = men,
  1 = women) from the probability vectors released for those tasks. This campaign develops and tests materially different release
  methods that keep useful predictions while reducing that inference from the individual outputs and the linked pair. A successful
  method is not assumed.
- Campaign start 2026-10-22T01:19:06Z (first `date -u`). Research window ends 09:19:06Z; closeout ends 09:49:06Z.
- Sources (pinned and verified label-free by `ipr.admit`, `SOURCE_ADMISSION.json`, `ok`): the dual-design release campaign
  (publication e30cf1c6b, child of its registration receipt c5a84c19d; research tip c577424c0 = the base of this branch
  `research/pcrl-input-and-pair-release-v1`), finite-solver adequacy (ddc231a97), finite-routing comparison (04ac9161f), new release
  mechanisms (682068d52) and the chain they admitted. Their conclusions are preserved unchanged (`FAILURE_MAP.md`, rows 1-14).
- **Known before writing** (published or label-free): every published record of those campaigns, including the registered-grid AUCs
  of the references, of the dual-design twin and of its strongest release (`roles/coordinator/DESIGN_BRIEF.md`). The designers'
  proposals are computed from public records, cached MODEL_FIT out-of-fold predictions and label-free features only (no label file,
  nothing of MODEL_SELECT); their feasibility numbers are disclosed and are not outcomes.
- No outcome of this campaign exists. No label has been read by this campaign.

## 1. Data, roles and the service (unchanged definitions)

- Admitted Florida ACS 2019 cohort only. MODEL_FIT (20,000 people, 13,337 households): all fitting, SEX as a training label only
  where a committed mechanism card says so. CALIBRATION (2,982 people): authorized for income / occupation labels (never SEX).
  MODEL_SELECT (8,001 people, 5,306 households): utility evaluation and development readers only, never a fitting input of any
  release parameter, strength, control point or selection rule. ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels, the sealed
  assessment and the capped-out records stay closed; DEV_REPORT features are used label-free for deployment parity. No records are
  downloaded.
- Runtime fields: AGEP, WKHP, COW, SCHL, MAR, POBP (admitted 145-column encoding X). Exact source task definitions, household groups
  and cohort manifests of the admitted chain.
- Service: each recipient receives only its own probability vector (float64) and its argmax decision. Readers may link the two
  outputs of a person (pair view). A randomized release emits one draw per person in every role; further stored draws are Monte Carlo
  repetitions, never pooled per person.
- Access: `ipr.common.load` is the only label path (ROLE x FIELD x OPERATION x PURPOSE; `ACCESS_POLICY.json`); every read is logged;
  no label read before this registration is committed and its sanitized receipt is verified on the GitHub remote; fitting reads need
  a committed, unchanged card; MODEL_SELECT labels are never read inside a fitting entry point (`ipr.ia`, `ipr.ib`, `ipr.ctl`,
  `ipr.fs`).

## 2. Families, cards and the order of work

Two designers work independently in disjoint spaces (Designer A, family `IA`: before the release, that is what the task models see
and learn from; Designer B, family `IB`: after the task models, the joint release map designed for the linked pair). A challenger
reviews both for equivalence with each other and with the closed recipes before any card is committed (`roles/challenger/`); if the
two are equivalent, one designer is redirected. Families: `IA`, `IB`, `CTL` (shared controls and the shared twin), `FS` (at most one
replacement family). Closed recipes (propensity-routed cells and their solvers, Gaussian and coupled logit channels, learned decoders
and calibrators, controllers, leaf corrections, and the dual-design idea family of SEX standardization, inverse-propensity
reweighting, linear propensity tilts and release-only tilts as mechanisms) are not continued.

For each family, before its first fit, a mechanism card `results/<campaign>/cards/<FAMILY>_<name>.md` is committed with the governing
code (template `cards/README.md`). It fixes the exact operation; closest previous implementation and what differs; the measured
failure it addresses; the twin; the strength grid and the MODEL_FIT-only selection rule; the matched control paths (section 4); the
ablation; for a jointly designed family, the joint-design controls or the statement that they are not built (section 6, D5); at most
two revisions with their triggers; registered predictions; the configuration count. A card change after its first fit is a new card
(revision), never an edit.

Order: registration -> cards and code (selftests on invented data) -> MODEL_FIT fits, selection and control points -> MODEL_SELECT
utility of every saved configuration -> reader grid -> decisions -> diagnosis and revisions -> ablation and classifier refits of the
strongest feasible release -> verification, packaging, custody, publication. **No synthetic or preliminary privacy result is required
before the first development comparison**: MODEL_FIT reader checks named in a card are descriptive and never stop the comparison. The
strongest feasible nonzero release (for the ablation and the refits) is chosen on registered MODEL_SELECT cells among the family's own
operation configurations (winner's curse; descriptive); its ablation comparisons never upgrade D2.

## 3. Task feasibility (original limits; unchanged)

`ipr.evaluate` (= `nrm.evaluate` definitions, U0_S<v> / Ucal_S<v>): accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed Brier
excess <= 0.005 (2 tasks x 2 references x 3 metrics x 2 versions = 24 clauses) and the men's and women's occupation summed-Brier
excess vs U0 <= 0.005 (4 guards). **G1 = all 28 at the point.** Reported beside, never gating: the two-SE reserve (`ohn.rules.reserve`
unchanged; whole release and occupation side) and paired costs against every comparator. A release that fails G1 is reported as
infeasible, never as protection. Point feasibility is not confidence preservation.

Versions: v = 0, 1 are the two procedure repeats of the admitted chain (their own internal folds and seeds, U0_S<v> / Ucal_S<v>); they
share every training row and the MODEL_SELECT people (repeats, not replications).

## 4. Comparisons and controls (equal training exposure, tuning effort and evaluation effort)

Every family supplies, on the same MODEL_FIT people and labels:

| Role | What | Purpose |
|---|---|---|
| TWIN | the equally trained task-only model of the family (same learners, data, folds, seeds and selection effort; protection off); where the operation has a natural zero, TWIN is the operation at strength 0. Both families share one twin when their task-only models coincide (identical learners, settings, folds, seeds and selection effort: the cached HGB task teachers); an input-side family that re-selects hyperparameters on modified inputs needs a TWIN given the same re-selection on the raw inputs | the reference for the protection effect |
| CAND | the family's operation at the strength chosen by its card's MODEL_FIT-only rule | the candidate |
| SBC path | the same operation with every SEX-dependent fitted input replaced by a declared SEX-independent statistic (permuted SEX, or the declared fallback PSEUDO2 of `ipr.ctlkit`) | targeting vs the operation's generic effect |
| SPEND path | DQ: SEX-blind subtractive-dithered quantization of TWIN's released vectors (`ipr.ctlkit`) | the ordinary-spending explanation |
| RO path | the release-only explanation, declared in the card before CAND's first fit as `ro.kind`. SUBSTITUTED (default): CAND's operation with the MODEL_FIT SEX column replaced, in every SEX-dependent fitted input, by the release-only SEX surrogate P(SEX \| TWIN's released log-probabilities of both recipients), fitted on MODEL_FIT and cross-fitted on the household folds (used as a soft label where the fit accepts weights, otherwise one fixed-seed draw; the card says which), on its own strength grid. GENERIC_ZTILT: the generic release-only tilt ZTILT of TWIN (`ipr.ctlkit`) on a fine grid; required when the operation is itself release-only (every SEX-dependent fitted input already a function of TWIN's released vectors, so that the substitution would return the operation itself), and otherwise allowed only as the card's declared fallback. The card also records `ro.operation_release_only` | whether the gain needs anything beyond what a SEX-targeted function of the twin's release achieves |
| ABL | one or two component ablations of the strongest feasible nonzero release (card) | which part carries any effect (descriptive) |
| D5 controls | for a family whose card declares joint design: the separately designed recipients (each recipient's map designed for its own view only) and both sequential orders (income first, occupation first) at matched MODEL_FIT spend, each selected on MODEL_FIT as the smallest grid strength whose spend is at least CAND's (one point each) (or the declared collapse of a sequential order; the challenger and the verifier check the declaration, by an exact-alias wire check where both comparators exist) | joint-design evidence |
| References | NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100 (frozen, point-eligible); U0 / Ucal as utility diagnostics; beside (descriptive, not D3 members): the dual-design campaign's DA_TWIN, DB_PTC_R1 and DB_ZTILT_R1 | the original benchmark and the previous campaign's strongest release |

**Strength selection (MODEL_FIT only).** Default rule: the largest grid strength such that it and every smaller strength meet the
card's budget over the twin on MODEL_FIT out-of-fold or nested held-out data (estimate + 2 household-clustered SE). A card chooses one
of the three predeclared budgets before its first fit, with its reason: S1 / S2 / S3 = `nrm.fa.prc.BUDGETS` plus the men's and
women's occupation Brier components (S2: occupation NLL 0.008, income NLL 0.004, men's and women's occupation Brier 0.0030, occupation
accuracy loss 0.006, income accuracy loss 0.004; S3: 0.012 / 0.006 / 0.0045 / 0.0045 / 0.008 / 0.006). MODEL_FIT SEX labels enter the
selection of CAND and of the SEX-blind controls through the men's and women's guards (a constraint, not an input of the operation).
The rule is applied per version; CAND is the pair of per-version selections, recorded, flagged CAND_STRENGTH_DIFFERS_BY_VERSION if they
differ. A strength is recorded as a number, a component vector (for example recipient-split strengths) or a per-version dict of either;
it is nonzero when every component is >= 0 and one is > 0; the tie-break scalar is the sum of all components over both versions. The
record carries CAND's per-version strength and each control path's construction per version; the rules flag
CAND_STRENGTH_DIFFERS_BY_VERSION and SBC_CONSTRUCTION_DIFFERS_BY_VERSION from them (reported). If either version has no qualifying nonzero strength, the family has no CAND (D1 NO_FEASIBLE_STRENGTH, class
TASK_INFEASIBLE; no fallback strength or budget). A card may instead fix one strength for both versions (the smaller of the two
selections), stated before the fit.

**What a matched control must satisfy.** A matched control point (SBC, SPEND or RO) must (i) be built on the family's own TWIN with
the same MODEL_FIT people, folds, learners, seeds, number of fitted components and, for a randomized release, the same draw structure
as CAND; (ii) for the SBC path, be CAND's operation with every SEX-dependent fitted input replaced by a declared SEX-independent
statistic of the same type and dimension (default: a fixed-seed permutation of the MODEL_FIT SEX column across people), never a
constant, a zero or the removal of the component; if the permuted path cannot reach CAND's MODEL_FIT spend at the largest grid
strength, the declared fallback PSEUDO2 (`ipr.ctlkit`; the dual-design campaign's adopted construction) is used by the MODEL_FIT-only
rule of the card; (iii) for the SPEND path, be DQ on TWIN's released vectors; (iv) for the RO path, be the declared SUBSTITUTED or
GENERIC_ZTILT construction, with the surrogate (or the tilt's amplitude) cross-fitted on the household folds, so that no MODEL_FIT
person's own SEX label enters its own surrogate value; (v) be named with its construction, grid and saved-point rule in the card before CAND's
first fit and be fitted before any MODEL_SELECT read; (vi) start at TWIN (strength 0 of every control path). (i) applies in full to SBC and to a SUBSTITUTED RO path; DQ and ZTILT
share TWIN, the MODEL_FIT people, folds and seeds, and a randomized control gets at least two evaluated draws on the registered cells,
as a randomized CAND does. An invertible transform (temperature, affine re-mixing) removes no information and is never a SPEND
control. A card states whether its operation has any SEX-dependent fitted input (`operation_sex_blind`; SEX in the MODEL_FIT
strength-selection guards does not count). For a SEX-blind operation the SBC and RO comparisons are degenerate by construction (each
would be the operation itself): no SBC path and no substituted RO path is built (2 x 2 x 2 = 8 configurations saved), and D2 gates on
TWIN and the SPEND path (targeting vs TWIN and the SPEND bracket). A generic ZTILT path, if built, is reported beside and never gates.
When both families share TWIN, a DQ or ZTILT point at the same strength is the same released wire: it is saved once, counted once and
listed in both readings' paths. Every control path's grid is fine enough that, on
MODEL_FIT and in each version, its saved lower point spends at least 0.75 s and its saved upper point at most s + 2 SE + 0.25 s, where s
is CAND's out-of-fold paired total-NLL spend over TWIN and SE the household-clustered SE of the upper point's paired difference
(`ipr.rules.grid_fineness`; the dual-design release-only grid, 0.90x / 2.08x, reads COARSE_ABOVE). The check runs on MODEL_FIT before
any MODEL_SELECT read; a coarse path gets added grid points on MODEL_FIT (CAND's own path is not re-tuned). When s <= 0, TWIN is the
bracket of every path and the check does not apply. A remaining coarse path is flagged CONTROL_GRID_COARSE_<path> beside D2: reported,
never gating, never a reason to set a reading aside. The card states the grid.

**Control points, bracketing and equal effort.** Each control path saves exactly two points, chosen on MODEL_FIT (the card's
out-of-fold or nested held-out paired total-NLL spend over TWIN) before any MODEL_SELECT read: the largest grid strength whose spend is
at most CAND's, and the smallest grid strength whose spend is at least CAND's plus 2 household-clustered SE of the paired difference
(if none reaches it, the largest grid strength, flagged CONTROL_PATH_MAY_NOT_BRACKET before MODEL_SELECT). Duplicates and exact aliases
are saved once. The bracket is then read on MODEL_SELECT paired total NLL by the inherited rule. A missing MODEL_SELECT bracket reads
NOT_SEPARATED with that reason; this is a limit of the control path, never evidence of equality, and no control point is added after
MODEL_SELECT is read. Equal tuning effort: every control path is evaluated on MODEL_FIT on a grid at least as fine as CAND's, with the
same learners, folds, seeds, fitted components and budgets. Equal evaluation effort: every saved configuration gets the same utility
evaluation and the same registered reader cells; robustness cells go to the D2R set only (descriptive). An identical released wire
(exact alias) is evaluated once and counted once.

## 5. Readers (what recipients actually receive)

- Fresh development grid IPR1 inside MODEL_SELECT (`READER_SPLITS.json`): repetitions R1, R2 (registered) and R3, R4 (robustness);
  households ordered by SHA-256('IPR1_devgrid_<R>|<household>') and dealt into 10 parts: FIT 0-4, VALIDATION 5-6, SCORE 7-9
  (household-disjoint). Reader seeds MODEL_SEED_BASE + 900 + 10 v + k (k0, k1 registered; k2, k3 robustness); the seeds of every
  realized earlier grid (offsets 300-800) are forbidden. Household bootstrap of SCORE (B = 400, seed 2026102300 + 100 v + repetition
  index). Overlaps with the earlier grids (OHN1, TSZ1, NRM1, FRC1, DDR1) are recorded.
- The admitted 16-recipe bank (OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB, LOOKUP, ET5, ET25, HGB15, HGB31, MLP64,
  MLP6432, CATI01, CATI1) on the transmitted vectors only (and their decisions): fitted on FIT (the declared auxiliary sample),
  oriented and selected on VALIDATION, scored on SCORE (one target release per person). Pair winner chosen among pair readers and the
  ignore-one-recipient readers. Grid extras (non-table winner, TABLE_CV, enhanced winner) recorded; the enhanced winner's benefits are
  reported beside D2 as a reader-sensitivity check. The data-rich readers (`ipr.richreaders`: trained on MODEL_FIT out-of-fold
  releases with MODEL_FIT SEX inside the family fit units, scored on all MODEL_SELECT people) are reported beside, descriptive. Code
  locked in `DEVGRID_CODE_LOCK.json` (committed before any grid run).
- **Grid validity** (READER_SEEDS_EFFECTIVE, the rules' section 9): only with the registered reader and bootstrap seeds and when every
  seed-dependent recipe was checked and every seed-pair equality is explained (the prospective ER explanation restricted to
    extra-trees readers on bounded-support views). Otherwise every D2-D5 label is NOT_COMPUTED_GRID_INVALID, and any later repair is
  labelled POST HOC. The minimum set is a reading's CAND, TWIN and the saved points of its gating control paths; references,
  ablations, D5 comparators and a reported-beside ZTILT path are off it. The claims builder maps the READER_SPLITS keys V<v>k<k> to the
  rules' S<v>k<k> and passes the bootstrap seeds S<v><R> (required inputs: without them the grid is invalid by rule).
- Uncertainty: the inherited aggregate screen (block means over version x repetition; se_agg = sqrt(se_hh^2 + se_rs^2 + se_mc^2),
  0.001 floor; both version means must agree). Versions, repetitions and reader seeds are not independent replications. **Planned
  resolution**: a comparison resolves when its aggregate mean - 2 se_agg > 0 with both version means > 0. The dual-design grid
  realized se_agg 0.009-0.011 against its twin, 0.009-0.017 against its SBC and SPEND brackets and 0.008-0.012 against its
  release-only tilts, so the per-comparison threshold is about 0.016-0.036 and a true benefit of about 2.84 se_agg (0.023-0.051) is
  resolved with roughly 80% probability (normal approximation; the version-mean condition lowers it). D2 needs every gating comparison
  resolved: with equal true benefits against TWIN and every bracket and correlation 0.5 between comparisons, its power is about 0.24
  at 0.025, 0.56 at 0.035 and 0.90 at 0.050; when the release-only control reproduces part of the gain (0.035 against the others,
  0.015 against RO) about 0.29 (reviewer's invented-data calculation, RULES.md 3.2). Smaller effects read 'not separated', never
  'equal'. For a randomized configuration entering D2, at least two evaluated draws on the registered cells.

## 6. Decisions (kept separate; formal definitions in `roles/statistical_reviewer/RULES.md`)

Per family, at CAND of its registered card (revision 0, committed before any MODEL_SELECT outcome of this campaign). A family
reading is IA, IB or FS_<name>; the shared controls (CTL) are never a reading; only revision 0 of IA or IB can be a registered reading. Every other
reading (a revision, the replacement family or a late card) is read under the same rules, carries the tag `ADAPTIVE_<...>` (card
committed after a MODEL_SELECT outcome) or `REVISED_<...>` (before) as a prefix of every label, is reported beside its family's
registered reading and is never presented as a registered result.

1. **D1 task feasibility:** G1 for TWIN and CAND; reserve reported beside.
2. **D2 improvement beyond matched controls:** (a) G1 at CAND; if CAND fails G1, D2 is NOT_ASSESSED_TASK_INFEASIBLE (not run; never
   negative, never protection) and D3-D5 carry the same label; (b) against TWIN, pair benefit `AGG_POSITIVE` and the recipient gate;
   (c) against the SBC path, the SPEND path and the RO path (each = TWIN followed by its saved points; aliases once), each separately:
   `DOMINATES_CONTROL_FRONTIER` (spend-bracketing rule, `nrm.rules.frontier_comparison`), the recipient gate against the bracketing
   point Q* (both views, both versions), and no cheaper path point with a resolved lower pair AUC than CAND; (d) reader targeting: the
      non-table winner's point benefit > 0 in both version means against TWIN and every bracketing control. Partial outcomes keep their
   names; none is a component claim. D2 met is labelled `IMPROVEMENT_BEYOND_MATCHED_CONTROLS` when the RO path is SUBSTITUTED (or the
   declared GENERIC_ZTILT fallback), and `RELEASE_ONLY_MAP_BEYOND_MATCHED_CONTROLS` when the operation is itself release-only. Every
   sentence quoting a D2 label or the outcome class repeats the reading's scope sentence (`ipr.rules.RO_SCOPE`); a release-only map that
   passes is never described as using SEX information beyond the twin's release. Partial outcomes are `PARTIAL_<passed>_ONLY`, joined by
   + in the order TWIN, SBC_FRONTIER, SPEND_FRONTIER, RO_FRONTIER. For a SEX-blind operation D2 met is labelled
   `SEX_BLIND_IMPROVEMENT_BEYOND_TWIN_AND_SPEND` and is quoted with: no SEX-targeting claim and no claim about SEX information beyond
   the twin's release is possible.
3. **D2R robustness to classifier refitting:** the strongest feasible nonzero release of each family, its TWIN and its bracketing
   controls (of the SBC, SPEND and RO paths, recomputed by `ipr.rules.brackets`; a path whose bracket is not recorded makes D2R
   INCOMPLETE, so a decisive comparator is never dropped silently) (and NEURAL_RAW, H_LJ) get the robustness cells (R3, R4 with k0-k3; R1, R2 with k2, k3) even if the registered interval
   crosses zero; ROBUSTNESS_CONFIRMED only when `nrm.rules.robustness` confirms on the merged cells AND the refit is genuinely
   effective (distinct assignments, SCORE overlaps recorded, grid validity on the eight seeds). R3 / R4 are new splits of the same
   people: robustness to classifier refitting, never replication. Descriptive; never changes D2.
4. **D3 competitiveness against strong references:** NRM1 G4 against all nine references at the same configuration as D2, with the
   reader-budget credit rule. The dual-design releases are reported beside.
5. **D4 the material criterion:** NRM1 G5 (0.020 pair-AUC benefit over NEURAL_RAW and H_LJ, resolved). Smaller gains are reported as
   smaller gains, never as the material criterion.
6. **D5 joint design:** NRM1 G6 unchanged for a family whose card declares joint design and builds the separately designed recipients
   and both sequential orders. A sequential order that the card declares, before its first fit and with its construction argument, to
   equal the separate design by construction is replaced by the separate comparator (reduced test, NR's label, flag
   D5_REDUCED_<ORDER>_EQUALS_SEPARATE, stated with the label); an order declared equal to the joint design itself gives
   NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL (no joint-design advantage over sequential design can exist). A declared comparator whose grid
   entry or paired cost is missing reads INCOMPLETE. Otherwise NOT_ASSESSED_MISSING_CONTROLS, or NOT_APPLICABLE_NOT_JOINT /
   NOT_APPLICABLE_ONE_OUTPUT_CHANGED (NR). D5 carries D2's NOT_ASSESSED_* / NOT_COMPUTED_* label as D3 and D4 do and never changes the
   outcome class.
7. **D6 novelty:** `NOT_CLAIMED` unless a separate review finds supporting evidence; uncertain novelty never stops a comparison.
8. **Outcome class per family reading** (first applicable): `TASK_INFEASIBLE`; `SELECTED_STRENGTH_INFEASIBLE`; `INCOMPLETE`;
   `PROTECTION_NOT_ASSESSED` (a matched control path not run for budget); `CANDIDATE` (D2 and D3); `COMPONENT` (D2 without D3);
   `TASK_ONLY_FEASIBLE_NO_PROTECTION` (CAND passes G1, D2 not met). Flag `D3_WITHOUT_D2`.

## 7. Revisions, replacement and stop rules

- At most two substantive revisions per family. Each revision addresses an observed failure (named, with its record), is written as a
  new card with its control and stopping rule, and is committed before its fit. A revision never re-reads MODEL_SELECT to choose a
  parameter; it may use the diagnosis of a MODEL_SELECT outcome (disclosed as adaptive). A revision triggered by a MODEL_SELECT outcome
  is adaptive as a whole.
- If task feasibility fails, the first revision repairs the architecture (or the family switches), and the protection comparison is
  reported as not run, never as negative.
- At most one replacement family (`FS_`) within budget, after the challenger and the statistical reviewer have read its card.
- Stop a family when: its two revisions are used; or its candidate is an exact alias of its twin at every allowed strength; or the
  budget forbids its controls (then the comparison is reported as not run).

## 8. Budget and accounting

At most 48 new seed- / version-specific release configurations (controls included; `ipr.release.save` refuses the 49th), 12
CPU-hours including failed attempts and agents' numerical work, two heavy jobs at once with one thread each, zero paid compute, at
most six live agents including the coordinator. Every numerical unit runs through `ipr.jobqueue`; out-of-queue agent work is charged.
One complete D2 reading costs CAND, two saved points per control path (three paths) and the ablation: 8 releases = 16 configurations
per family, plus the shared twin (2). Planned allocation: shared twin 2; IA up to 16; IB up to 16 (plus 6 for the D5 controls if its
card declares joint design); revisions and the replacement family the remainder. A revision reuses the family's SPEND path
whenever its TWIN is unchanged and its points still bracket the revision's MODEL_FIT spend, a GENERIC_ZTILT RO path under the same
condition, and the SBC path and a SUBSTITUTED RO path only if its operation and TWIN are unchanged. Otherwise its D2 is
NOT_ASSESSED_CONTROLS_NOT_RUN (class PROTECTION_NOT_ASSESSED) unless new control points fit the budget (a revision with new SBC and
SUBSTITUTED RO paths needs 10 configurations; 8 remain after the planned allocation).
Inherited releases of earlier campaigns used as references cost no configuration.

## 9. Verification, packaging and custody

An independent verifier recomputes the decisive utility values, grid comparisons and decisions with separate code. Any useful release
is packaged with its twin and decisive control (deployment from raw fields, refusal checks). A same-device backup is verified from the
backup alone; the off-device drive is used only if attached (never mounted or forced). The sanitized publication (code and aggregate
records, no person-level arrays) goes to `publication/pcrl-input-and-pair-release-v1`.

## 10. Registered expectations (coordinator, before any label read)

- P(a family's CAND is G1-feasible) about 0.8; P(D2 met for at least one family) about 0.08 (three gating control paths now);
  P(D3 for either) about 0.03; P(D4) about 0.01. Each card adds its own numerical predictions, per D2 label that can occur (the met
  label that applies to its declarations, NOT_SEPARATED, PARTIAL_..._ONLY with RO failing, NO_BRACKET_<path>, PROTECTION_NOT_ASSESSED)
  and the predicted pair benefit against TWIN and each Q*, with an interval; by the planned resolution, a prediction of D2 met above
  about 0.3 needs an expected benefit of at least about 0.035 against every comparator.
- The most likely outcome is a protection effect against the twin of 0.005-0.02 pair AUC that is not separated from at least one
  matched control (most likely the release-only path or the SEX-blind spend path).

## 11. What is not claimable

Protection against arbitrary readers; confirmation; independent replication from versions, repetitions or reader seeds; confidence
preservation from point feasibility; superiority from a narrower comparison; a joint-design advantage without its controls; novelty
from performance; anything about the closed roles; a non-registered reading presented as registered; a campaign-level statement
without the number of D2 looks and their registration status.

## 12. Known stale or reused items (disclosed)

- `ipr/` was derived from `ddr/` by a mechanical rename (campaign names, paths, seeds, split tags) at about 01:33Z; some docstrings of
  derived modules still describe their source's history (for example `ipr/jobqueue.py` names its frc / nrm ancestry; no derived
  module's constants were left at the dual-design values: the reader-seed offset is 900 in `ipr/devreaders.py` and `ipr/rules.py`, the
  bootstrap base 2026102300). The family modules (`ipr/ia`, `ipr/ib`), the shared twin (`ipr/ctl`) and the shared control paths
  (`ipr/paths.py`) are new; draft family modules committed with this registration are superseded by the versions committed with their
  cards.
- The designers' proposals (`roles/designer_a/`, `roles/designer_b/`) were written before this registration from label-free inputs; the
  challenger's review was still running at registration. Its required changes go into the mechanism cards (and, where they change
  this protocol, into a dated amendment file), which are committed before the fits they govern.

## 13. Adjudication of the pre-registration review

The statistical reviewer's report (`roles/statistical_reviewer/REPORT_PREREGISTRATION.md`, saved verbatim by the coordinator because the
harness refused the agent's own write; hand-back `HANDBACK_PREREGISTRATION.md`) was adopted in full: REQUIRED findings 1-10 are in
sections 4, 5, 6 and 8 above in the reviewer's wording; ADVISORY findings 12, 13, 14, 15, 16 and 17 are adopted in the same sections;
advisory 11 (a SUBSTITUTED RO path that cannot reach CAND's spend: extend its grid, else the declared fallback) is applied in the
cards; advisory 18 (stale derived docstrings) is applied to the modules the families call (`ipr/devreaders.py`, `ipr/common.py`,
`ipr/ctlkit.py`, `ipr/admit.py`) and disclosed in section 12 for the rest. The rules are frozen as `RULES.md` / `RULES.json` /
`ipr/rules.py` (schema IPR1-rules-v1; `python3 -m ipr.test_rules` 28 of 28 pass; the reviewer completed RULES.md at 02:08:15Z).

## 14. Completion

Completed 2026-10-22T02:13:01Z (`date -u` equivalent, read in the same command that wrote this line), after the adoption of the statistical reviewer's
pre-registration review. No label had been read by this campaign.
