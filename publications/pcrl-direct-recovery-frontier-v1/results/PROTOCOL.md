# Protocol: direct-recovery finite release (campaign `pcrl_direct_recovery_frontier_v1`, code `drf/`, tag DRF1)

Coordinator. Writing started 2026-10-25T02:55Z (`date -u`); the completion time is in section 16. EXPLORATORY DEVELOPMENT on
previously and adaptively used public-use ACS 2019 1-Year PUMS Florida person records (the admitted development cohort). Nothing
here is confirmation.

## 0. Purpose, status and disclosure (before any label read)

- **Purpose.** A statistical machine-learning benchmark. A service predicts an income class (2) and an occupation group (5) from six
  raw fields. The privacy evaluation measures how well classifiers can infer the dataset's recorded SEX column (0 = men, 1 = women)
  from the probability vectors released for those tasks: from each output alone and from the linked pair. This campaign asks one
  question: can a realizable many-to-one (finite) probability release keep the original task and subgroup usefulness limits while
  reaching a linked-pair recovery that the strongest existing releases have not reached reliably? A successful method is not assumed.
- Campaign start 2026-10-25T02:17:51Z (first `date -u`). The queue refuses research stages after 7.5 h (09:47:51Z) and every stage
  after 8 h (10:17:51Z).
- **Starting evidence (read, frozen, preserved unchanged):** the strong-anchor release campaign (SAR1; publication c8aaa2df4, child of
  its receipt ede0755fb via bca6e5b28; research tip 73931682e = the base of this branch): no competitive method (D2, D3, D4 met in 0 of
  6 readings); FA improved a usefulness guard without keeping its protection profile across versions; FB was not separated from its
  start's own recipe continued. Its residuals, Fisher penalties, guard estimators and strength sweeps are not resumed. The finite-routing
  line (NRM1 FA_PRC_v1, FRC1 04ac9161f, FSA1 ddc231a97) is the closest prior work: a Lagrangian of task cross-entropy and pi-weighted
  plug-in MI with a Lloyd / tilt optimizer and a post hoc utility screen; its protected path saturated and was not separated from
  controls (`FAILURE_MAP.md`).
- **Known before writing** (published or label-free): every published record of those campaigns, including SAR1's grid AUCs and the
  MODEL_SELECT utility of every reference and of the twin TWB_C64 (FRC1 `dev/UTILITY_LEDGER.csv`). The constraint budgets of this
  campaign are set from the twin's published MODEL_SELECT slack (section 5): an informed, adaptive choice, disclosed.
- The roughly 0.58 pair AUC mentioned in SAR1's last report was a planning consequence of its estimated variability. It is not a
  threshold here; every criterion stays comparator-based.
- No outcome of this campaign exists. No label has been read by this campaign.

## 1. Data, roles and the service (unchanged definitions)

- Admitted Florida ACS 2019 cohort only. MODEL_FIT (20,000 people, 13,337 households): all fitting (release parameters, output support,
  mapping rules, tables; internal validation on household-separated MODEL_FIT folds); SEX a training label for the recovery objective
  and the within-SEX guard constraints, as the committed card says. MODEL_SELECT (8,001 people, 5,306 households): logged development
  utility evaluation and reader fitting / selection / scoring on prospectively separated household subroles; never a fitting input of
  any task learner, calibration map, release parameter, output support or mapping rule. It remains heavily reused development data.
- **CALIBRATION is not read** (no CALIBRATION label is requested; the frozen admitted teachers are reused). ATTACK_FIT,
  ATTACK_VALIDATION and DEV_REPORT labels, the sealed assessment and the capped-out records stay closed; their features are used only
  label-free to compute the release for deployment parity, as in earlier campaigns. No population is added; no record is downloaded;
  closed labels are never computed, in memory or otherwise.
- Runtime fields: AGEP, WKHP, COW, SCHL, MAR, POBP (admitted 145-column encoding X). Exact source task definitions (income 2 classes,
  occupation 5 groups), household groups and cohort manifests of the admitted chain.
- Service: each recipient receives only its own float64 probability vector and argmax decision; the linked-pair recipient is
  explicitly granted person-level linkage; the declared recipient's labelled-data budget is the reader FIT split. One released
  instance per person. No cell identifier, routing key, original field, donor record or additional model feature is disclosed. No
  repeated-query, insider or model-access claim.
- At inference the mapping uses only the six fields and its frozen model (frozen teachers, codebook, pi thresholds, lookup tables).
  SEX and true task labels inform training only; no person identifier is used to memorize an assignment.
- Access: `drf.common.load` is the only label path (ROLE x FIELD x OPERATION x PURPOSE; `ACCESS_POLICY.json`); every read is logged;
  no label read before this registration is committed and its sanitized receipt is verified on the GitHub remote (`origin` =
  https://github.com/Bostesa/PCRL.git); fitting reads need a committed, unchanged card; MODEL_SELECT labels are never read inside the
  fitting entry point `drf.dr`; label reads run only inside queue units.

## 2. Storage and workspace admission

- **Worktree.** New isolated research worktree `.claude/worktrees/pcrl-direct-recovery-frontier-v1`, branch
  `research/pcrl-direct-recovery-frontier-v1`, created 02:30:39Z with `git worktree add` from the SAR1 tip 73931682e; the session
  switched into it. Main, manuscripts, prior study branches, prior campaigns' records and public history are not changed.
- **Relocation index read** (`~/storage-relocation-20261022/`: manifest.json, RESTORE.md, LEDGER.csv): 84 items are external-only on
  YOTUO and some archives are the only remaining copies; they are preserved. **No archive is restored**: every dependency is local.
- **Dependencies (working cache).** The nine inherited private trees the reused code reads were cloned (APFS `cp -cRp`,
  02:32:32-02:32:37Z) from the lineage worktree `pcrl-correction-transfer-stability-v1/.private/` into this worktree's `.private/` and
  verified file by file (sizes and SHA-256; `SOURCE_ADMISSION.json` `working_cache`, ok). The protected-tree replication tree
  (`.worktrees/pcrl-acs-protected-tree-replication-v1`, archived and kept) is read in place. The frozen inputs of the family (NRM1 task
  cache, FRC1 SEX-side cache, FRC1 twin model, FRC1 utility ledger) are pinned by SHA-256 (`drf/dr.py` SOURCE_PINS; admission
  `dr_frozen_inputs`, ok).
- **Drive.** YOTUO was not attached at 02:21:25Z (`/Volumes` lists no YOTUO); nothing is mounted or forced. The off-device copy is made
  at closeout only if the drive is attached and its identity verifies (volume UUID 3D74932B-33DD-3896-8CC6-0D2FDDAD6B8D); otherwise the
  missing custody is disclosed.
- **Space.** Free internal space 121 GiB at 02:17:51Z, 120 GiB after the clone (02:32:37Z); floor 20 GiB; peak temporary storage is
  recorded in `RESOURCE_ACCOUNTING.json`. Unrelated projects are not cleaned.

## 3. The investigation, the changed ingredient and its comparisons

- **Family DR** (card `cards/DR_DRF_v1.md`, design `roles/coordinator/DESIGN_BRIEF.md`): a deterministic lookup table from legal fine
  cells (admitted twin codeword x quartile of the admitted propensity, both functions of the six fields through frozen models) to at
  most 16 income and 64 occupation atoms, each atom emitting its B1 table (the mean out-of-fold teacher posterior of its MODEL_FIT
  people). The identity is the admitted task-only twin TWB_C64 and is always the start.
- **Changed ingredient:** the exact optimal-reader AUC of the emitted finite law (pair + income + occupation) as the objective, with the
  usefulness conditions as explicit constraints inside a feasible local search, replacing a dependence proxy (pi-weighted plug-in MI)
  under a Lagrangian path followed by a utility screen. This overlaps privacy funnels and the earlier SCM / FRC finite routing; no
  novelty is presumed.
- **Isolating comparison:** DR_MI, the PRC pi-weighted plug-in MI in the same family with identical fine cells, moves, constraints,
  data exposure, seed and optimizer budget. A family change (finite vs continuous SUCC) or a capacity / exposure change is reported
  separately and never credited to the objective.
- **Preliminary phase (capped at 90 minutes, from 02:40Z):** the exact optimal-reader AUC is derived and verified by enumeration on small
  laws with ties, unequal class frequencies, merged outputs and linked pairs (investigator A, `drf/finite_law.py`); tiny exact
  instances compare the bounded optimizer with the exhaustive constrained optimum, to tell an optimizer defect from a failed
  objective (`drf/tiny.py`). These calculations are properties of the specified finite laws only, not population privacy guarantees.
  No preliminary significance, optimistic forecast or novelty gate may prevent the fitted comparisons; an unresolved correctness
  problem must be repaired before results are trusted.
- **Optimistic reference vs deployable release:** OPT_L (legal cells, true-SEX empirical law, in-sample) and OPT_S (SEX-split cells:
  routing uses the true SEX, not deployable) are MODEL_FIT-only diagnostics with nested held-out values; DR_AUC is the legal mapping,
  fitted on the estimated law and evaluated on separated MODEL_SELECT households. One support size / capacity is frozen: (16, 64) with
  4 pi bins.

## 4. Card and order of work

Order: registration (this protocol, rules, card, access policy, splits; receipt verified on the remote) -> MODEL_FIT fits (DR_AUC,
DR_MI, coarsening path, nested estimator, CTRL brackets saved before any MODEL_SELECT read; OPT_L, OPT_S) -> reading plan committed ->
MODEL_SELECT utility of every saved release -> DRF1 reader grid (candidate, controls, twin and references on the same cells) ->
decisions -> evidence-driven revisions (each with a committed prospective card naming the observed failure and the changed operation)
-> robustness cells for a promising comparison -> verification, packaging, custody, publication. A card change after its first fit is
a new card, never an edit.

## 5. Task feasibility (original limits; unchanged)

`drf.evaluate` (the authoritative `nrm.evaluate` / `sar.evaluate` definitions, U0_S<v> / Ucal_S<v>): accuracy loss <= 0.010, NLL excess
<= 0.010 nats, summed Brier excess <= 0.005 (2 tasks x 2 references x 3 metrics = 12 clauses per version) and the men's and women's
occupation summed-Brier excess vs U0 <= 0.005 (2 guards per version): **14 conditions per version, 28 over the two versions** (24 clauses
+ 4 guards), not 28 per version. G1 = all 28 at the point on MODEL_SELECT. The confidence conjunction and the two-SE reserve are reported
separately (RULES.md). Versions v = 0, 1 are procedure repeats of the admitted chain (shared training rows and MODEL_SELECT people;
repeats, not replications). The optimizer's MODEL_FIT budgets (0.6 x the twin's published MODEL_SELECT slack per metric) are an internal
device; they never replace the original conditions. An original usefulness failure may stop the competitive reader stage for that
release: the fitted release is preserved and the failure is classified honestly.

## 6. Comparisons and controls

| Comparator | What |
|---|---|
| START | SUCC, the unchanged strong starting release (frozen, both versions; not a member of the finite family) |
| CTRL | DR_CTRL_LO / DR_CTRL_HI: the SEX-blind task-only coarsening path of the twin in the same family (greedy least-total-NLL atom merges); two bracket points saved before any MODEL_SELECT read on the nested MODEL_FIT total-NLL spend (largest <= DR_AUC's; smallest >= DR_AUC's + 2 household-clustered SE of their paired difference); read on MODEL_SELECT with the inherited spend-bracketing frontier rule; a missing bracket reads NOT_SEPARATED with that reason |
| OBJ_MI | DR_MI, the previous dependence objective in the same family (section 3); spend mismatch reported |
| TWIN | TWB_C64, the family origin (FRC1 frozen release; a weak task-only twin: reported, never sufficient alone) |
| References | NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100 (frozen, point-eligible), re-read on this campaign's grid with equal reader fitting and selection opportunity; U0 / Ucal utility diagnostics |

Every saved configuration gets the same utility evaluation and the same registered reader cells; an identical released wire is
evaluated and counted once. An unmatched comparison cannot establish the mechanism's contribution.

## 7. Readers (what recipients actually receive)

- Fresh development grid DRF1 inside MODEL_SELECT (`READER_SPLITS.json`, written label-free 02:51:46Z): repetitions R1, R2 (registered)
  and R3, R4 (robustness; genuinely different household assignments); households ordered by SHA-256('DRF1_devgrid_<R>|<household>')
  and dealt into 10 parts: FIT 0-4, VALIDATION 5-6, SCORE 7-9 (household-disjoint), frozen before any outcome. Reader seeds
  MODEL_SEED_BASE + 1100 + 10 v + k (k0, k1 registered; k2, k3 robustness); the offsets of every realized earlier grid (300-1000) are
  forbidden. Household bootstrap of SCORE (B = 400, seed 2026102500 + 100 v + repetition index). Overlaps with earlier grids (OHN1,
  TSZ1, NRM1, FRC1, DDR1, IPR1, SAR1) are recorded; these remain reused development people.
- The strongest admitted bank: the 16 recipes (OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB, LOOKUP, ET5, ET25, HGB15,
  HGB31, MLP64, MLP6432, CATI01, CATI1) on the transmitted vectors and decisions only, fitted on FIT, selected on VALIDATION, scored on
  SCORE; pair winner among pair and ignore-one-recipient readers; extras (non-table winner, TABLE_CV, enhanced winner) reported beside.
  Candidates, controls, twin and references get the same opportunity on the same cells. A few-valued output may make reader seeds
  redundant: exact aliases are recorded and effective distinct fits (n_eff) counted (RULES.md). Code locked in
  `DEVGRID_CODE_LOCK.json` (torch interpreter) before any grid run.
- Income, occupation and linked-pair recovery are all reported; the strongest recipient can be used alone by a coalition; a finite
  pair reader that does worse than a single view is an engineering diagnostic, not a population property.
- For a promising mapping, a different household reader split (R3 / R4 with k2 / k3) is run without retuning; sign reversals and
  failed recipient conditions are carried into the main conclusions.
- Uncertainty: the inherited aggregate screen (block means over version x repetition; se_agg; both version means must agree). The
  statistical reviewer's estimator section (RULES.md) distinguishes household uncertainty, fitting variation and numerical randomness.
  All intervals are conditional development summaries that do not correct the accumulated adaptive search.

## 8. Decisions

Formal definitions in `roles/statistical_reviewer/RULES.md` (schema DRF1-rules-v1; `drf/rules.py`, tests `python3 -m drf.test_rules`):
D1 task feasibility (G1 gates; confidence conjunction and reserve reported); **D2 component gain beyond the two family controls**, the
CTRL frontier and OBJ_MI, with the recipient gates and reader targeting (met label `COMPONENT_GAIN_BEYOND_FAMILY_CONTROLS`); the
**family change FC** (DR against START = SUCC) reported separately and never part of D2 (SUCC is also one of the D3 references); TWIN
reported beside, never sufficient alone; D2R robustness; D3 against every strong reference; D4 the material target (a resolved 0.020
pair benefit against NEURAL_RAW and against H_LJ with the recipient gates; never weakened); D6 novelty NOT_CLAIMED; the three outcomes
(component gain, strong references, material target) reported separately; and the failure stage (section 8 of the brief, RULES.md 5):
no useful optimistic point; optimistic success but legal failure; utility kept but tied with controls; controls beaten but strong
references missed; original competitive conditions pass (exploratory evidence requiring fresh confirmation). Every G1 statement about
DR carries the disclosure that its MODEL_FIT constraints were tuned on the twin's published MODEL_SELECT aggregates, so a MODEL_SELECT
G1 pass is not an independent test of constraint transfer.

## 9. Revisions and stop rules

At most two evidence-driven revisions within this question, each with a prospective card (observed failure, the actual changed
operation, its controls and predictions) committed before its fit; a revision triggered by a MODEL_SELECT outcome is ADAPTIVE as a
whole and never re-reads MODEL_SELECT to choose a parameter. Work continues while the question remains and resources permit; unused
budget is not spent on unrelated sweeps. The family stops when its revisions are used or the remaining budget forbids its controls.

## 10. Budget and accounting

At most 32 new version-specific release specifications (each compared operating point counts, optimistic diagnostics, matched controls
and revisions included; `drf.release.save` / `register` refuse the 33rd), at least 16 reserved for legal mappings, controls and
follow-through. Internal optimizer iterations and cross-validation fits are recorded separately. Eight elapsed hours, 12 CPU-hours
including failed attempts and out-of-queue work, zero paid compute, at most two heavy jobs at once with one numerical thread each, one
shared queue (`drf.jobqueue`), at most six live agents including the coordinator (no recursive spawning). Initial plan: 12 of 32.

## 11. Verification, packaging, custody and publication

- An independent verifier (separate code) recomputes the finite-law objective, the decisive task metrics, the reader comparisons and
  the outcome from frozen outputs and refits a bounded sample of winning readers; failed checks and corrections are preserved; no check
  is relaxed after its failure.
- Raw-input deployment packages for useful fitted mappings and informative controls (six fields only): probability validity,
  decisions, schema refusals, forbidden inputs, equivalence to the attacked output across request sizes (bitwise or the stated
  tolerance).
- Backup of final code, packages, readers and records; restore and rerun from the backup alone; copy to YOTUO if attached and verified,
  without touching existing archives; warm read-back is distinguished from a cold check; missing custody is disclosed.
- **Publication and the owner-identifier gap.** SAR1's public receipt ede0755fb carries identifying strings; it is not force-pushed or
  rewritten. Before this campaign's first receipt: relative paths are exported, the proposed tree and its history are scanned privately
  for the owner's names, username and host names (derived at run time, never written into code or records), and the matched strings are
  never published in a scan report. Public account and commit metadata may still identify the owner; no anonymity is claimed. Only the
  sanitized publication branch `publication/pcrl-direct-recovery-frontier-v1` is pushed; the research branch is not.

## 12. Registered expectations (coordinator, before any label read)

P(DR_AUC G1-feasible in both versions) about 0.6; P(DR_AUC's in-sample exact objective below DR_MI's in both versions) about 0.8;
P(D2 met) about 0.10; P(D3) about 0.03; P(D4) about 0.02. Most likely: the men's occupation guard binds first and DR_AUC lowers the
twin's grid pair AUC by 0.01-0.03, mostly not separated from the coarsening control or not reaching SUCC.

## 13. What is not claimable

Protection against arbitrary readers; a population privacy guarantee from exact finite-law calculations; confirmation; independent
replication from versions, repetitions or reader seeds; superiority from a narrower comparison; a mechanism contribution from an
unmatched comparison; competitiveness from a gain against a weak twin; novelty; anything about the closed roles; anonymity.

## 14. Known stale or reused items (disclosed)

`drf/` was derived from `sar/` by a mechanical rename at 02:33:05Z (campaign names, paths, reader-seed offset 1100, bootstrap base
2026102500, split tags DRF1, family DR, 32 specifications, CALIBRATION removed); some docstrings still describe their sources'
history. New modules: `drf/solver.py`, `drf/dr.py` (coordinator), `drf/finite_law.py`, `drf/tiny.py` (investigator A), `drf/rules.py`
and `drf/test_rules.py` (statistical reviewer). Out-of-queue work before registration is charged (`drf.jobqueue charge`), including a
reader-grid selftest run without the one-thread environment (deviation 1).

## 15. Adjudication of the pre-registration review and of the preliminary phase

**Statistical reviewer** (`roles/statistical_reviewer/REPORT_PREREGISTRATION.md`, saved verbatim because the harness refused the
reviewer's write; hand-back `HANDBACK_PREREGISTRATION.md`). Every REQUIRED finding is adopted:
- R1: the 16 budgets (8 metrics x 2 versions), the 0.6 factor and the source hash are frozen in `dr/BUDGETS.json` (written
  03:12:27Z, committed with this registration); `drf.dr.frozen_budgets` refuses a fit whose file is uncommitted or whose recomputation
  differs; the same numbers serve DR_AUC, DR_MI, OPT_L and OPT_S; the G1 disclosure is in section 8.
- R2: each fit record holds, for DR_AUC and DR_MI side by side, the fine cells, capacity, move set, seeds, sweep cap, restarts, law,
  budgets and teacher pins, the binding constraint (largest spend / budget ratio) and the nested spends; DR_MI is fitted in the same
  unit as DR_AUC, before any MODEL_SELECT read.
- R3: the CTRL brackets come from the same nested estimator as DR_AUC's spend; the full path (spend and SE of every point) is recorded
  and its fineness read before any MODEL_SELECT read; if it reads CONTROL_PATH_MAY_NOT_BRACKET the path is extended first.
- R4, R7, R8: the reading plan records distinct-input counts of all three views of every minimum-set wire on MODEL_SELECT, the
  exact-alias relations computed on the wires, the first MODEL_SELECT read time and each card commit.
- R5, R6: a usefulness failure is recorded with `reader_stage` and the release kept; the coarsening path is evaluated in-sample under
  both laws and in the nested mode (held-out exact and LOOKUP) so the optimistic diagnostic has its comparator on the same law and
  mode. No diagnostic gates any fit: DR_AUC, DR_MI and the CTRL points are fitted and read whatever OPT_L / OPT_S show.
- R9: the optimizer is deterministic given its seeds (recorded); a restart replaces the incumbent only if it lowers the MODEL_FIT
  objective under the construction's own law by more than 1e-12; no Monte Carlo term exists.
- R10, R11: D2R grid entries carry all three views; FC and TWIN never enter D2 in any sentence.
- R12: section 8 now says D2 is beyond the two family controls, with FC against SUCC reported separately.
Advisories: A2 kept as frozen (single-point resolution rule, the strict point rule beside); A3 adopted (in-sample values under the
estimated and the empirical law, nested values on the empirical law); A5 adopted (LOOKUP against the winner per view); A1, A6, A7, A8
noted and disclosed. **A4 (tiny-instance criterion)** is applied as investigator A ran it before this registration, on invented data
only: an *implementation defect* is an infeasible, misreported or non-local-optimal answer; an *optimization shortfall* is a feasible
local optimum above the exhaustive constrained optimum of the same objective.

**Preliminary phase (02:38-03:12Z, inside its 90-minute cap; invented data only).**
- `drf/finite_law.py` (investigator A): the exact optimal-reader AUC (likelihood-ratio order; closed form 1/2 + 1/4 sum |P1(a) P0(b) -
  P1(b) P0(a)|; identical wires merged) agrees with enumeration over all weak orders in exact arithmetic (14 of 14 checks; ties,
  unequal class frequencies, pure and merged wires, linked pairs, data processing). A two-wire law with lower MI, covariance and critic
  loss but higher AUC* shows the proxies are not substitutes (`roles/investigator_a/DERIVATION.md`).
- Tiny exact instances (`drf/tiny.py`; `TINY_EXACT.json`, `TINY_EXACT_V2.json`): no implementation defect in `drf/solver.py`; the
  single-move search alone fell short of the exhaustive optimum in 35 / 36 of 348 improvable cases (AUC / MI). After the prospective
  changes recorded in the card (atom-merge polish, 2 seeds x 2 restarts, 30 sweeps; identical for every objective), 4 / 3 of 348
  remain, which a joint two-cell move would close; this residual optimization limitation is the same for both objectives and is
  disclosed. On tiny laws the MI-optimal assignment had a higher AUC* than the AUC-optimal one in 12 of 360 cases (misalignment
  witnesses).
- A label-free plumbing rehearsal on the frozen statistics (no label read): the identity mapping with the twin's tables reproduces
  FRC1's stored TWB_C64 MODEL_SELECT wires bitwise; the identity tables recomputed by `drf.solver` differ from the twin's by at most
  6.7e-16 (summation order).

## 16. Completion

See the completion line appended at registration.

Completed 2026-10-25T03:14:24Z (UTC clock read in the same command that wrote this line), after the adoption of the statistical reviewer's
pre-registration review (section 15). The mechanism card `cards/DR_DRF_v1.md` and the frozen budgets `dr/BUDGETS.json` are committed
with this registration. No label had been read by this campaign.
