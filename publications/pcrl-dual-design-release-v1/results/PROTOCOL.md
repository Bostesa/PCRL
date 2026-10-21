# Protocol: two independently designed release families (campaign `pcrl_dual_design_release_v1`, code `ddr/`)

Coordinator. Writing started 2026-10-21T16:52Z (`date -u`); revised 17:28-17:41Z to adopt the statistical reviewer's
pre-registration findings (section 13); the completion time is in section 14. EXPLORATORY DEVELOPMENT on previously and adaptively used public-use ACS
2019 1-Year PUMS Florida person records (the admitted development cohort). Nothing here is confirmation.

## 0. Purpose, status and disclosure (before any label read)

- **Purpose.** A statistical machine-learning benchmark. A service predicts an income class (2) and an occupation group (5) from
  six raw fields. The privacy evaluation measures whether classifiers can infer the dataset's recorded SEX column (0 = men,
  1 = women) from the probability vectors released for those tasks. This campaign develops and tests materially different release
  methods that keep useful predictions while reducing that inference from the individual outputs and the linked pair. A successful
  method is not assumed.
- Campaign start 2026-10-21T16:37:33Z (first `date -u`). Research window ends 00:37:33Z on 2026-10-22; closeout ends 01:07:33Z.
- Sources (pinned and verified label-free by `ddr.admit`, `SOURCE_ADMISSION.json`): finite-solver adequacy (publication ddc231a97,
  research tip f0edb8ad1 = the base of this branch `research/pcrl-dual-design-release-v1`), finite-routing comparison (04ac9161f),
  new release mechanisms (682068d52) and the chain they admitted. Their conclusions are preserved unchanged (`FAILURE_MAP.md`).
- **Known before writing** (published or label-free): every published record of those campaigns, including the registered-grid pair
  AUCs of the references and of the finite twin (`roles/coordinator/DESIGN_BRIEF.md`). The designers' proposals and the
  coordinator's label-free diagnostics (`roles/coordinator/labelfree/`) were computed from public records and cached MODEL_FIT
  out-of-fold predictions only (no label file, nothing of MODEL_SELECT); their feasibility numbers are disclosed and are not outcomes.
- No outcome of this campaign exists. No label has been read by this campaign.

## 1. Data, roles and the service (unchanged definitions)

- Admitted Florida ACS 2019 cohort only. MODEL_FIT (20,000 people, 13,337 households): all fitting, SEX as a training label only
  where a committed mechanism card says so. CALIBRATION (2,982 people): authorized for income / occupation labels (never SEX).
  MODEL_SELECT (8,001 people, 5,306 households): utility evaluation and development readers only, never a fitting input of any
  release parameter, strength or selection rule. ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels, the sealed assessment and the
  capped-out records stay closed; DEV_REPORT features are used label-free for deployment parity. No records are downloaded.
- Runtime fields: AGEP, WKHP, COW, SCHL, MAR, POBP (admitted 145-column encoding X). Exact source task definitions, household
  groups and cohort manifests of the admitted chain.
- Service: each recipient receives only its own probability vector (float64) and its argmax decision. Readers may link the two
  outputs of a person (pair view). A randomized release emits one draw per person in every role; further stored draws are Monte
  Carlo repetitions, never pooled per person.
- Access: `ddr.common.load` is the only label path (ROLE x FIELD x OPERATION x PURPOSE; `ACCESS_POLICY.json`); every read is
  logged; no label read before this registration is committed and its sanitized receipt is verified on the GitHub remote; fitting
  reads need a committed, unchanged card; MODEL_SELECT labels are never read inside a fitting entry point.

## 2. Families, cards and the order of work

Two designers worked independently (Designer A: what the task models are trained on or how; Designer B: how released vectors are
formed from fitted statistics). A challenger reviews both for equivalence with each other and with the closed recipes before any
card is committed (`roles/challenger/`). Families: `DA` (designer A), `DB` (designer B), `CTL` (shared controls, each built on the
twin of the family it serves), `FS` (at most one replacement family). Closed recipes (propensity-routed cells and their solvers,
Gaussian logit channels, learned decoders, controllers, leaf corrections) are not continued.

For each family, before its first fit, a mechanism card `results/<campaign>/cards/<FAMILY>_<name>.md` is committed with the
governing code (template `cards/README.md`). It fixes the exact operation; closest previous implementation and what differs; the
measured failure it addresses; the twin; the strength grid and the MODEL_FIT-only selection rule; the matched control paths
(section 4); the ablation; at most two revisions with their triggers; registered predictions; the configuration count. A card
change after its first fit is a new card (revision), never an edit.

Order: registration -> cards and code (selftests on invented data) -> MODEL_FIT fits, selection and control points -> MODEL_SELECT
utility of every saved configuration -> reader grid -> decisions -> diagnosis and revisions -> ablation and classifier refits of the
strongest feasible release -> verification, packaging, custody, publication. **No synthetic or preliminary privacy result is required
before the first development comparison**: MODEL_FIT reader checks named in a card are descriptive and never stop the comparison.
The strongest feasible nonzero release (for the ablation and the refits) is chosen on registered MODEL_SELECT cells among the
family's own operation configurations (winner's curse; descriptive); its ablation comparisons never upgrade D2.

## 3. Task feasibility (original limits; unchanged)

`ddr.evaluate` (= `nrm.evaluate` definitions, U0_S<v> / Ucal_S<v>): accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed
Brier excess <= 0.005 (2 tasks x 2 references x 3 metrics x 2 versions = 24 clauses) and the men's and women's occupation
summed-Brier excess vs U0 <= 0.005 (4 guards). **G1 = all 28 at the point.** Reported beside, never gating: the two-SE reserve
(`ohn.rules.reserve` unchanged; whole release and occupation side) and paired costs against every comparator. A release that fails
G1 is reported as infeasible, never as protection. Point feasibility is not confidence preservation.

Versions: v = 0, 1 are the two procedure repeats of the admitted chain (their own internal folds and seeds, U0_S<v> / Ucal_S<v>);
they share every training row and the MODEL_SELECT people (repeats, not replications).

## 4. Comparisons and controls (equal training exposure, tuning effort and evaluation effort)

Every family supplies, on the same MODEL_FIT people and labels:

| Role | What | Purpose |
|---|---|---|
| TWIN | the equally trained task-only model of the family (same learners, data, folds, seeds and selection effort; protection off); where the operation has a natural zero, TWIN is the operation at strength 0 | the reference for the protection effect |
| CAND | the family's operation at the strength chosen by its card's MODEL_FIT-only rule | the candidate |
| SBC path | the same operation with every SEX-dependent fitted input replaced by a declared SEX-independent statistic (below) | targeting vs spending |
| SPEND path | a SEX-blind spend-only transformation of TWIN's released vectors | the ordinary-spending explanation |
| ABL | one or two component ablations of the strongest feasible nonzero release (card) | which part carries any effect (descriptive) |
| References | NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100 (frozen, point-eligible); U0 / Ucal as utility diagnostics | the original benchmark |

**Strength selection (MODEL_FIT only).** Default rule: the largest grid strength such that it and every smaller strength meet the
S2 spend budget over the twin on MODEL_FIT out-of-fold or nested held-out data (estimate + 2 household-clustered SE): occupation
NLL 0.008, income NLL 0.004, men's and women's occupation Brier 0.0030, occupation accuracy loss 0.006, income accuracy loss 0.004.
A card may choose one of the three predeclared FA_PRC_v1 budgets (S1, S2, S3) before its first fit, with its reason; its control
paths use the same budget for their own grid. MODEL_FIT SEX labels enter the selection of CAND and of the SEX-blind controls through
the men's and women's guards (a constraint, not an input of the operation). The rule is applied per version (as
`nrm.fa.prc.select_path`); CAND is the pair of per-version selections, recorded, flagged CAND_STRENGTH_DIFFERS_BY_VERSION if they
differ. If either version has no qualifying nonzero strength, the family has no CAND (D1 NO_FEASIBLE_STRENGTH, class
TASK_INFEASIBLE; no fallback strength or budget). A card may instead fix one strength for both versions (the smaller of the two
selections), stated before the fit.

**What a matched control must satisfy.** A matched control point (SBC or SPEND) must (i) be built on the family's own TWIN with the
same MODEL_FIT people, folds, learners, seeds, number of fitted components and, for a randomized release, the same draw structure as
CAND; (ii) for the SBC path, be CAND's operation with every SEX-dependent fitted input replaced by a declared SEX-independent
statistic of the same type and dimension (default: a fixed-seed permutation of the MODEL_FIT SEX column across people), never a
constant, a zero or the removal of the component; if the permuted path cannot reach CAND's MODEL_FIT spend at the largest grid
strength, the card's declared fallback SEX-blind target (named before CAND's first fit, for example a feature- or task-derived
binary of comparable task entanglement) is used, by that MODEL_FIT-only rule; (iii) for the SPEND path, be a SEX-blind spend-only
transformation of the TWIN's released vectors; (iv) be named with its construction, grid and saved-point rule in the card before
CAND's first fit and be fitted before any MODEL_SELECT read; (v) start at TWIN (strength 0 of every control path). A shared CTL path
counts as a family's SPEND path only if it is built on that family's TWIN. An invertible transform (temperature, affine
re-mixing) removes no information and is never a SPEND control.

**Control points, bracketing and equal effort.** Each control path saves exactly two points, chosen on MODEL_FIT (the card's
out-of-fold or nested held-out paired total-NLL spend over the TWIN) before any MODEL_SELECT read: the largest grid strength whose
spend is at most CAND's, and the smallest grid strength whose spend is at least CAND's plus 2 household-clustered SE of the paired
difference (if none reaches it, the largest grid strength, flagged CONTROL_PATH_MAY_NOT_BRACKET before MODEL_SELECT). Duplicates
and exact aliases are saved once. The bracket is then read on MODEL_SELECT paired total NLL by the inherited rule. A missing
MODEL_SELECT bracket reads NOT_SEPARATED with that reason. This is a limit of the control path, never evidence of equality, and no
control point is added after MODEL_SELECT is read. Equal tuning effort: every control path is evaluated on MODEL_FIT on a grid of
the same size as CAND's, with the same learners, folds, seeds, fitted components and budgets. The saved points differ by design
(CAND: the selected strength; each control path: its two bracket points). Equal evaluation effort: every saved configuration gets
the same utility evaluation and the same registered reader cells; robustness cells go to the D2R set only (descriptive). An identical
released wire (exact alias) is evaluated once and counted once.

## 5. Readers (what recipients actually receive)

- Fresh development grid DDR1 inside MODEL_SELECT (`READER_SPLITS.json`): repetitions R1, R2 (registered) and R3, R4 (robustness);
  households ordered by SHA-256('DDR1_devgrid_<R>|<household>') and dealt into 10 parts: FIT 0-4, VALIDATION 5-6, SCORE 7-9
  (household-disjoint). Reader seeds MODEL_SEED_BASE + 800 + 10 v + k (k0, k1 registered; k2, k3 robustness). This base and the
  bootstrap base below were registered by FSA1 and never used (FSA1 built no grid). The seeds of every realized earlier grid
  (MODEL_SEED_BASE + 300, 400, 500, 600, 700 + 10 v + k) are forbidden, and the grid is valid only with exactly these reader and
  bootstrap seeds (RULES.md 9). Household bootstrap of SCORE (B = 400, seed 2026102200 + 100 v + repetition index). Overlaps with
  earlier grids are recorded.
- The admitted 16-recipe bank (OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB, LOOKUP, ET5, ET25, HGB15, HGB31, MLP64,
  MLP6432, CATI01, CATI1) on the transmitted vectors only (and their decisions): fitted on FIT (the declared auxiliary sample),
  oriented and selected on VALIDATION, scored on SCORE (one target release per person). Pair winner chosen among pair readers and
  the ignore-one-recipient readers. Grid extras (non-table winner, TABLE_CV, enhanced winner) recorded; the enhanced winner's
  benefits are reported beside D2 as a reader-sensitivity check. Code locked in `DEVGRID_CODE_LOCK.json` (committed before any grid
  run).
- **Grid validity.** The grid is valid (READER_SEEDS_EFFECTIVE, RULES.md 9) only with the registered reader and bootstrap seeds and
  when every seed-dependent recipe was checked and every seed-pair equality is explained; on the minimum set (a reading's CAND, TWIN
  and saved control points) an equality is explained only for an extra-trees reader (ET5, ET25) on a view with at most
  8001 // min_leaf distinct released vectors that differs on another view while another seed-dependent reader differs on the same
  view (FRC1's ER1, adopted prospectively and restricted to its mechanism). Otherwise every D2-D4 label is NOT_COMPUTED_GRID_INVALID,
  and any later repair is labelled POST HOC.
- Uncertainty: the inherited aggregate screen (block means over version x repetition; se_agg = sqrt(se_hh^2 + se_rs^2 + se_mc^2),
  0.001 floor; both version means must agree). Versions, repetitions and reader seeds are not independent replications. With two
  versions and two registered splits, the aggregate screen's false-pass rate is about 8% when one version or split dominates; such
  a reading carries that caveat. For a randomized configuration entering D2, at least two evaluated draws on the registered cells.

## 6. Decisions (kept separate; formal definitions in `roles/statistical_reviewer/RULES.md`)

Per family, at CAND of its registered card (revision 0, committed before any MODEL_SELECT outcome of this campaign). Every other
reading (a revision, the replacement family or a late card) is read under the same rules, carries the tag `ADAPTIVE_<...>` (card
committed after a MODEL_SELECT outcome) or `REVISED_<...>` (before) as a prefix of every label (RULES.md 7), is reported beside its
family's registered reading and is never presented as a registered result.

1. **D1 task feasibility:** G1 for TWIN and CAND; reserve reported beside.
2. **D2 improvement beyond matched controls:** (a) G1 at CAND; if CAND fails G1, D2 is NOT_ASSESSED_TASK_INFEASIBLE (not run; never
   negative, never protection) and D3 / D4 carry the same label; (b) against TWIN, pair benefit `AGG_POSITIVE` and the recipient
   gate; (c) against the SBC path and the SPEND path (each = TWIN followed by its saved points; aliases once), each separately:
   `DOMINATES_CONTROL_FRONTIER` (spend-bracketing rule, `nrm.rules.frontier_comparison`), the recipient gate against the bracketing
   point Q* (both views, both versions), and no cheaper path point with a resolved lower pair AUC than CAND (pair benefit upper bound
   < 0 and both version means < 0); (d) reader targeting: the non-table winner's point benefit > 0 in both version means against
   TWIN and every bracketing control. Partial outcomes keep their names; none is a component claim. A pair benefit of about 0.015 is
   resolved with roughly 80% power per comparison (threshold 2 se_agg, about 0.010); D2 needs three such comparisons and has less
   power; a smaller effect reads "not separated", never "equal".
3. **D2R robustness to classifier refitting:** the strongest feasible nonzero release of each family, its TWIN and its bracketing
   controls (and NEURAL_RAW, H_LJ) get the robustness cells (R3, R4 with k0-k3; R1, R2 with k2, k3) even if the registered interval
   crosses zero. Per comparator (TWIN and the release's own bracketing controls, recomputed by the same bracket rule from its
   MODEL_SELECT paired costs; NEURAL_RAW and H_LJ reported beside), ROBUSTNESS_CONFIRMED only when `nrm.rules.robustness` confirms
   on the merged registered + robustness cells AND the refit is REFIT_GENUINELY_EFFECTIVE (R3 / R4 assignment hashes distinct from
   each other and from R1 / R2, SCORE overlaps recorded in [0, 1), and grid validity on the eight seeds k0-k3 with both seed pairs
   judged separately, on the minimum set {release, TWIN, bracketing controls}); otherwise ROBUSTNESS_NOT_CONFIRMED with reasons. R3 /
   R4 are new splits of the same people: robustness to classifier refitting, never replication. Descriptive; never changes D2.
4. **D3 competitiveness against strong references:** NRM1 G4 against all nine references at the same configuration as D2, with the
   reader-budget credit rule.
5. **D4 the material criterion:** NRM1 G5 (0.020 pair-AUC benefit over NEURAL_RAW and H_LJ, resolved). Smaller gains are reported
   as smaller gains, never as the material criterion.
6. **D5 joint design:** `NOT_ASSESSED_MISSING_CONTROLS` unless the separately designed recipients and both sequential orders exist.
7. **D6 novelty:** `NOT_CLAIMED` unless a separate review finds supporting evidence; uncertain novelty never stops a comparison.
8. **Outcome class per family reading** (first applicable): `TASK_INFEASIBLE` (no grid strength met the MODEL_FIT budget, or CAND and
   every other saved nonzero strength fail G1); `SELECTED_STRENGTH_INFEASIBLE` (CAND fails G1, another saved nonzero strength
   passes; reported, never promoted to CAND); `INCOMPLETE` (the class cannot be determined from the records: D1 or D2 incomplete or
   not computed for grid validity, or D2 met with D3 undecided); `PROTECTION_NOT_ASSESSED` (a matched control path not run for
   budget); `CANDIDATE` (D2 and D3); `COMPONENT` (D2 without D3); `TASK_ONLY_FEASIBLE_NO_PROTECTION` (CAND passes G1, D2 not met; a
   TWIN failing G1 is the flag `TWIN_INFEASIBLE`). Flag `D3_WITHOUT_D2` (references beaten without separation from matched
   controls: not a candidate).

## 7. Revisions, replacement and stop rules

- At most two substantive revisions per family. Each revision addresses an observed failure (named, with its record), is written as
  a new card with its control and stopping rule, and is committed before its fit. A revision never re-reads MODEL_SELECT to choose a
  parameter; it may use the diagnosis of a MODEL_SELECT outcome (disclosed as adaptive). A revision triggered by a MODEL_SELECT
  outcome is adaptive as a whole, even though its parameters are chosen on MODEL_FIT; its G1 on the same MODEL_SELECT people is
  selection-dependent.
- If task feasibility fails, the first revision repairs the architecture (or the family switches), and the protection comparison is
  reported as not run, never as negative.
- At most one replacement family (`FS_`) within budget, after the challenger and the statistical reviewer have read its card.
- Stop a family when: its two revisions are used; or its candidate is an exact alias of its twin at every allowed strength; or the
  budget forbids its controls (then the comparison is reported as not run).

## 8. Budget and accounting

At most 48 new seed- / version-specific release configurations (controls included; `ddr.release.save` refuses the 49th), 12
CPU-hours including failed attempts and agents' numerical work, two heavy jobs at once with one thread each, zero paid compute, at
most six live agents including the coordinator. Every numerical unit runs through `ddr.jobqueue`; out-of-queue agent work is
charged. One complete D2 reading costs TWIN, CAND, two saved points per control path and the ablation: 7-8 releases = 14-16
configurations; each card states its count. Planned allocation: DA up to 16, DB up to 16, revisions and the replacement family the
remaining 16. A revision reuses the family's SPEND path whenever its TWIN is unchanged, and the SBC path only if its operation and
TWIN are unchanged (the bracket is read as usual). Otherwise its D2 is NOT_ASSESSED_CONTROLS_NOT_RUN (class PROTECTION_NOT_ASSESSED)
unless new control points fit the budget. Inherited releases of earlier campaigns used as references cost no configuration.

## 9. Verification, packaging and custody

An independent verifier recomputes the decisive utility values, grid comparisons and decisions with separate code. Any useful
release is packaged with its twin and decisive control (deployment from raw fields, refusal checks). A same-device backup is
verified from the backup alone; the off-device drive is used only if attached (never mounted or forced). The sanitized publication
(code and aggregate records, no person-level arrays) goes to `publication/pcrl-dual-design-release-v1`.

## 10. Registered expectations (coordinator, before any label read)

- P(a family's CAND is G1-feasible) about 0.8; P(D2 met for at least one family) about 0.10; P(D3 for either) about 0.04; P(D4)
  about 0.02. Each card adds its own numerical predictions (pair benefit against TWIN and each bracketing control with an interval;
  probabilities of IMPROVEMENT_BEYOND_MATCHED_CONTROLS, NOT_SEPARATED, a missing bracket and PROTECTION_NOT_ASSESSED).
- The most likely outcome is a protection effect against the twin of 0.005-0.015 pair AUC that is not separated from a SEX-blind
  control at matched spend.

## 11. What is not claimable

Protection against arbitrary readers; confirmation; independent replication from versions, repetitions or reader seeds; confidence
preservation from point feasibility; superiority from a narrower comparison; a joint-design advantage without its controls;
novelty from performance; anything about the closed roles; a non-registered reading presented as registered; a campaign-level
statement without the number of D2 looks and their registration status (`ddr.rules.campaign_summary`).

## 12. Known stale or reused items (disclosed)

- `ddr/devreaders.py`'s docstring carried the finite-routing campaign's seed constants until about 17:31Z (the code constants were
  correct throughout); it was corrected and the grid code re-locked (`DEVGRID_CODE_LOCK.json` rewritten) before any grid run.

## 13. Adjudication of the pre-registration review

The statistical reviewer's report (`roles/statistical_reviewer/REPORT_PREREGISTRATION.md`, saved verbatim) was adopted in full:
REQUIRED findings 1-10 are in sections 2, 4, 5, 6, 7, 8 and 11 above in the reviewer's wording; ADVISORY findings 11, 12 (first
part), 13, 14, 17, 18, 19 and 20 are adopted. Advisory 12 (second part, a fold-level SE floor) is left to each card. Advisory 15 is
applied in the cards. Advisory 16 applies to any randomized configuration.

## 14. Completion

Completed 2026-10-21T17:41:46Z (`date -u`, read in the same command that wrote this line), after the adjudication of the statistical
reviewer's pre-registration review. No label had been read by this campaign. The challenger's review of the two proposals was still
running; its required changes go into the mechanism cards, which are committed before the fits they govern.
