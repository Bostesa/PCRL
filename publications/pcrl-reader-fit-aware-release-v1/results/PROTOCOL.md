# Protocol: reader-fit-aware release (campaign `pcrl_reader_fit_aware_release_v1`, RFA1, code `rfa/`)

Coordinator. Writing started 2026-10-25T21:21Z (`date -u`); the completion time is in section 16. EXPLORATORY DEVELOPMENT on
previously and adaptively used public-use ACS 2019 1-Year PUMS Florida person records (the admitted development cohort). Nothing here
is confirmation.

## 0. Purpose, status and disclosure (before any label read)

- **Purpose.** Statistical privacy research on an existing census benchmark. A service predicts an income class (2) and an occupation
  group (5) from six raw fields. The privacy evaluation measures whether classifiers can infer the dataset's recorded SEX column
  (0 = men, 1 = women) from the probability vectors released for those tasks, from each output alone and from the linked pair. This
  campaign runs one concrete fitted investigation toward a competitive release method: does training the release against how an
  auxiliary-data reader **learns and generalizes** (the derivative of a held-out reader objective through the reader's fit) transfer
  better to newly fitted recipients than training against a critic's current fitted responses? A successful method is not assumed,
  and no novelty is claimed.
- Campaign start 2026-10-25T20:58:41Z (first `date -u`). Ceiling 6 h elapsed (02:58:41Z), 8 CPU-h including failures and work outside
  the queue, 24 new version-specific release specifications, zero paid compute, at most two heavy jobs with one numerical thread each,
  at most five live agents including the coordinator (no recursive spawning). The ceiling is not a target.
- **Sources** (pinned and verified label-free by `rfa.admit`, `SOURCE_ADMISSION.json`, `ok`): the direct-recovery frontier campaign
  DRF1 (publication c4896479e, child of its receipt dae5914f4; research tip 913eb5c91 = the base of this branch
  `research/pcrl-reader-fit-aware-release-v1`), the estimator diagnostic EDG1 (publication d987e3685), the strong-anchor campaign SAR1
  (publication c8aaa2df4, an ancestor of this branch) and the chain they admitted. Their conclusions are preserved unchanged
  (`FAILURE_MAP.md`).
- **Known before writing** (published or label-free): every published record of those campaigns, including SAR1's anchor table
  (NEURAL_RAW pair 0.6253 = its occupation view; income view 0.5336) and SAR1 FB's MODEL_FIT records (FB_PFD_v1's income absorption,
  FB_PFO_R1's S50). No outcome of this campaign exists. No label has been read by this campaign.

## 1. Data, roles and the service (unchanged definitions)

- Admitted Florida ACS 2019 cohort only. MODEL_FIT (20,000 people, 13,337 households): all fitting; SEX is a training label only where a
  committed card says so (a protection objective or a within-SEX utility constraint). MODEL_SELECT (8,001 people, 5,306 households):
  logged development utility evaluation and reader fitting / selection on its predeclared FIT / VALIDATION / SCORE subroles; never a
  fitting input of any task learner, release gradient, calibration map, inner reader, strength or operating choice. CALIBRATION labels,
  ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels, the sealed assessment and the capped-out records stay closed (admitted features
  label-free for deployment parity only). Closed labels are never computed and discarded. No population is added; no record is
  downloaded.
- Runtime inputs: the six fields AGEP, WKHP, COW, SCHL, MAR, POBP (admitted 145-column encoding). Exact source task definitions, class
  counts, occupation grouping, preprocessing, household groups and cohort manifests of the admitted chain. One release per person.
- Service: each recipient receives only its own float64 probability vector and argmax decision. The evaluated recipient can fit a SEX
  reader from the declared labelled auxiliary data; the linked-pair recipient is granted person-level linkage. These grants are
  assumptions of the benchmark, not claims about a deployment. No SEX, true task label, person identifier, extra feature, hidden
  routing key or reader state is required or disclosed at deployment.
- Access: `rfa.common.load` is the only label path (ROLE x FIELD x OPERATION x PURPOSE; `ACCESS_POLICY.json`); every read is logged; no
  label read before this registration is committed and its sanitized receipt is verified on the GitHub remote; fitting reads need a
  committed, unchanged card; MODEL_SELECT labels are never read inside a fitting entry point (`rfa.fam`); label reads run only inside
  queue units (`rfa.jobqueue`).

## 2. Storage and workspace admission

- **Worktree.** New isolated research worktree `.claude/worktrees/pcrl-reader-fit-aware-release-v1` (created 2026-10-25 about 21:05Z
  with `git worktree add` from the DRF1 research tip; the session switched into it). Main, manuscripts, prior study branches and
  public history are not changed.
- **Working cache.** The nine inherited private trees were cloned (APFS `cp -cRp`, 21:09:58-21:10:03Z) from the lineage worktree into
  this worktree's `.private/` and verified file by file (sizes and SHA-256; `SOURCE_ADMISSION.json` `working_cache`, ok).
- **Relocation index read** (`~/storage-relocation-20261022/`: manifest.json, RESTORE.md). The only dependency outside the working
  cache, the protected-tree replication worktree (U0 and NEURAL_RAW files), is `archived-kept`: its local copy is used read-only and
  every file this campaign reads equals the relocation inventory's SHA-256 (`STORAGE_ADMISSION.json`, ok). **No archive is restored.**
- **Space.** 117 GiB free at 21:10Z; floor 20 GiB.
- **Drive.** YOTUO was attached at 21:11Z (volume UUID 3D74932B-33DD-3896-8CC6-0D2FDDAD6B8D; not mounted by this session). The owed
  EDG1 off-device copy was made at 21:12Z and its restore check passed from the drive copy (warm; the first attempt used an
  interpreter without matplotlib and failed at its analysis step; both records are kept in `roles/coordinator/`). Nothing else on the
  drive is written before closeout; existing archives are never modified.

## 3. Measurement contract

`MEASUREMENT_CONTRACT.md` (registered with this protocol) fixes the primary target (recovery by the declared finite-data reader
procedure, per recipient and for the linked pair), equal opportunities, the meaning of a lower reader AUC, the separation of empirical
optimal AUC as a diagnostic, the qualification of EDG1's opposing-bias recommendation, the untested status of DRF1's
propensity-estimated objective, the training-data status of inner episodes, and the uncertainty components.

## 4. The mechanism and its card

One family (FAM; cards `cards/RF_*.md`; code `rfa/fam/`): START = NEURAL_RAW. Per release step, an episode draws household-separated
auxiliary-training rows A and auxiliary-evaluation rows E from the training rows (MODEL_FIT, or the fold's training rows); bounded
readers (a simple and a bounded nonlinear family on fixed features of the released logits) are fitted exactly on A; a symmetric
recovery surrogate of their held-out scores on E is minimized together with the occupation task cross-entropy, with the derivative
taken through both the E outputs and the reader fit (implicit-function derivative of the regularized solve). The card fixes the
computational graph, the reader families, sizes, solver, surrogate, guards, lambda rule, schedule, budgets, controls, predictions and
falsifiers, the operational difference from the closed recipes (online critics, the closed-form Fisher penalty, the continuation of
NEURAL_RAW) and the prior-work comparison. It is committed with the code before the fit it governs; a change after its first fit is a
new card (revision), never an edit.

## 5. Task feasibility (original limits; unchanged)

`rfa.evaluate` (= `nrm.evaluate` definitions, U0_S<v> / Ucal_S<v>): accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed Brier
excess <= 0.005 (2 tasks x 2 references x 3 metrics = 12 clauses per version) and the men's and women's occupation summed-Brier excess vs
U0 <= 0.005 (2 guards per version): 28 conditions over the two versions. **G1 = all 28 at the point.** The confidence conjunction and the
inherited two-SE reserve heuristic are reported beside, never gate and never remove an otherwise eligible reference. Versions v = 0, 1
are the two procedure repeats of the admitted chain (shared training rows and MODEL_SELECT people; repeats, not replications).

## 6. Comparisons and controls

| Comparator | What | Role |
|---|---|---|
| START | NEURAL_RAW, frozen, both versions | gating |
| CONT | the identical occupation-only loop with lambda = 0 at CAND's checkpoint (same task rows, optimizer, steps) | gating |
| STOPFIT | identical episodes, readers, loss, lambda and schedule, with the fitted reader parameters detached from their fitting path (direct gradient through the E outputs kept); LO and HI bracket points on its own path at matched MODEL_FIT out-of-fold total-NLL spend over START | gating; the mechanism-isolating control |
| ALTCRIT | an ordinary alternating persistent critic of the same family and loss (lagging; direct path), HI bracket point | gating unless shown to be an exact alias of STOPFIT (then its artefact is reused) |
| ALT | NEURAL_RAW's own RAW-J recipe continued beyond E2 (both networks), HI bracket point | gating (the starting service's own protection recipe at comparable task cost) |
| References | NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100 (frozen, point-eligible), re-read on this campaign's grid; U0 / Ucal diagnostics | D3; D4 = 0.020 vs NEURAL_RAW and H_LJ |

The bracket rule (SAR1 `sar.fb.pfd.bracket`): on MODEL_FIT out-of-fold total NLL over START, LO = the largest path point whose spend is
at most CAND's, HI = the smallest whose spend is at least CAND's + 2 household-clustered SE of the paired difference; if none reaches it,
the largest-spend point with CONTROL_PATH_MAY_NOT_BRACKET. MODEL_SELECT then applies the inherited frontier rule to the saved points.
Equal epochs do not establish equal effort or equal task cost: each fit record holds the inner-reader computation, exposure, update
counts and the residual cost differences at the identical checkpoint. No joint-design claim is made (no local twin is built).

## 7. Readers (what recipients actually receive)

- Fresh development grid RFA1 inside MODEL_SELECT (`READER_SPLITS.json`, written label-free 21:16Z): repetitions R1, R2 (registered)
  and R3, R4 (the second split: genuinely different household assignments of the same people); households ordered by
  SHA-256('RFA1_devgrid_<R>|<household>') and dealt into 10 parts: FIT 0-4, VALIDATION 5-6, SCORE 7-9 (household-disjoint), frozen
  before any new outcome. Reader seeds MODEL_SEED_BASE + 1200 + 10 v + k (k0, k1 registered; k2, k3 with R3 / R4); every earlier grid's
  offsets (300-1100) are excluded. Household bootstrap of SCORE (B = 400, seed 2026102520 + 100 v + repetition index). Overlaps with the
  earlier grids are recorded. These are reused development people, not a confirmation set.
- The admitted 16-recipe bank on the transmitted vectors and decisions only, fitted on FIT, oriented and selected on VALIDATION, scored
  on SCORE; the pair winner chosen among pair readers and ignore-one-recipient readers. No inner training reader is reused. Candidates,
  controls and references get the same opportunities on the same cells. Code locked in `DEVGRID_CODE_LOCK.json` before any grid run.
- Budget sensitivity: the learning-curve reading uses FIT fractions 1.0 and 0.5 (the inherited fixed `RFA1_devgrid_LC` half), on R1 / R2.

## 8. Decisions

Formal definitions: `roles/statistical_reviewer/RULES.md` / `RULES.json` / `rfa/rules.py` (schema RFA1-rules-v1), frozen before any new
evaluation outcome. Reported separately: usefulness (D1); gain beyond the mechanism-specific controls (D2: START, CONT, STOPFIT, ALTCRIT,
ALT); D2R on the second split; the learning-curve reading; performance against the strong references (D3); the 0.020 material target
(D4); joint design (not claimed) and novelty (not claimed). Outcome classes distinguish a mechanism component, a competitive candidate
and a bounded negative experiment, with partial names.

## 9. Revision and stop rules

- One substantive revision of the selected mechanism (R1) within the 24-specification budget. Before its fit, a new card records the
  observed failure, the changed operation, the prediction and the control that isolates the change. No open-ended capacity, strength or
  temperature sweep. A revision triggered by a MODEL_SELECT outcome is adaptive as a whole (ADAPTIVE_ / REVISED_ labels).
- A design that never changes the transmitted output is established as such directly and stops being treated as a protection
  experiment. Stop when the revision is used, when the operation cannot change the output, or when the budget forbids its controls (then
  the comparison is reported as not run).

## 10. Budget and accounting

At most 24 new version-specific release specifications (`rfa.release.save` refuses the 25th), of which at least 12 are reserved for
the mechanism's controls, the revision and follow-through. Planned: CAND 2 + CONT 2 + STOPFIT 4 + ALTCRIT 2 + ALT 2 = 12 initial (10
controls); the revision with its isolating control about 6; the rest for follow-through. Internal optimizer steps, inner reader fits,
cross-fitting fits and checkpoints are recorded separately. A replayed frozen package is not a specification, but its resources are
charged. One shared queue (`rfa.jobqueue`); out-of-queue work is charged (`rfa.jobqueue charge`).

## 11. Verification, packaging, custody and publication

An independent verifier checks the computational graph and the new derivative, recomputes the decisive usefulness and reader
comparisons with separate code, refits a bounded sample of winning external readers, and verifies model selection, data boundaries and
the final classification; a failed check is preserved, never silently relaxed. Raw-input deployment packages for useful candidates and
informative controls: probability validity, schema refusals, absence of forbidden inputs, equivalence to the evaluated outputs at
several request sizes with stated tolerances. Final backup, restore from the backup alone, off-device copy to YOTUO if attached (warm
read-back distinguished from a cold check). The sanitized publication (code and aggregate records only) goes to
`publication/pcrl-reader-fit-aware-release-v1` after a private scan for names, usernames, host names and absolute paths; earlier public
history is not rewritten and anonymity is not claimed.

## 12. Registered expectations (coordinator, before any label read)

- P(CAND G1-feasible at the point) about 0.7. P(the fit path changes the release measurably: CAND and STOPFIT outputs differ beyond
  rounding) about 0.9. P(D2 against STOPFIT resolved) about 0.10. P(D2 against all gating controls) about 0.06. P(D3) about 0.02. P(D4)
  about 0.01.
- Most likely outcome: CAND lowers NEURAL_RAW's occupation view by 0.005-0.015 at the point, close to STOPFIT and ALT at matched spend;
  a bounded negative experiment.

## 13. What is not claimable

Protection against arbitrary readers; information removal from a lower fitted-reader AUC; confirmation; independent replication from
versions, repetitions or reader seeds; held-out evidence from inner episodes; a mechanism contribution from an unmatched comparison;
competitiveness from a gain against a weak twin or a narrower comparison; a joint-design advantage; novelty; anything about the closed
roles; anonymity.

## 14. Known stale or reused items (disclosed)

`rfa/` was derived from `drf/` by a mechanical rename at 21:08:39Z (`scripts/rfa1_rename.py`; log `roles/coordinator/RENAME_LOG.json`:
campaign names, paths, reader-seed offset 1200, bootstrap base 2026102520, split tags RFA1, family FAM with card prefix RF_, 24
specifications), plus `famkit`, `famrun` and `ctlkit` from `sar/`; some docstrings still describe their sources' history. The DR modules
were not copied. New modules: `rfa/fam/fit.py`, `rfa/fam/net.py` (coordinator), `rfa/fam/rfit.py` (mechanism investigator),
`rfa/rules.py` and `rfa/test_rules.py` (statistical reviewer). `rfa/custody.py` and `rfa/restore_check.py` still name DRF1's family
inputs in places; they are adapted before custody and the adaptation is recorded.

## 15. Adjudication of the reviews and of the design

- **Statistical reviewer, wave 1** (`roles/statistical_reviewer/HANDBACK_1.md`, saved verbatim). The contract review found 7 REQUIRED
  and 9 ADVISORY items; all are adopted in MEASUREMENT_CONTRACT.md section 9 (se_rs and se_hh scope, the definitions chain, the
  strict material-target wording, the learning-curve reading with the fold-rule HALF and paired SEs, the second-split trigger, the scope
  sentence, release-side selection on MODEL_FIT only). The rules RFA1-rules-v1 (`RULES.md`, `RULES.json`, `rfa/rules.py`,
  `rfa/test_rules.py`) were drafted from SAR1's and finalized for the card below; their completion record is in RULES.md.
- **Mechanism investigator** (`roles/investigator/HANDBACK_1.md`). The operation is implemented in `rfa/fam/rfit.py` (FULL, STOPFIT,
  ALTCRIT; 19 of 19 selftests; complete-derivative finite-difference checks to about 1e-10; the envelope identity; FB_PFD's gradient
  shown to be a stopped fit). The recipe values were set label-free; n_A about 4,096 and n_E about 2,400 follow the declared recipient
  budget. Adopted.
- **Independent challenger** (`roles/challenger/HANDBACK_1.md`, `HANDBACK_MESSAGE.md`). Verdict PROCEED: the operation is not
  equivalent to a closed recipe; its only genuine difference is the fit path. Required changes C1-C7 are adopted: the F0 mechanism gate
  and the registered falsifiers F1-F6 (card section 6), parameter-space mechanism logging per epoch (`rfa.fam.fit.mechanism_stats`), the
  ALTCRIT lag diagnostics and NEAR_ALIAS rule, the stopped A-moment path stated, guard readouts (descriptive per-recipe SCORE AUCs added
  to the grid output, the held-out degree-2 Fisher monitor and the A-row SD logged in the loop), matched-spend claims only, and
  label-free recipe values.
- **The mechanism selected:** card `cards/RF_RFIT_v1.md`, because the challenger found a genuine operational difference (the fit path)
  with a mechanism-specific control (STOPFIT) and falsifiers; no substitute operation was proposed.
- **Grid code changes before registration:** descriptive per-recipe SCORE AUCs in each grid entry, the docstring bootstrap seed and
  the selftest's forbidden-offset list (DRF1's 1100 added); the reader-grid selftest passed and DEVGRID_CODE_LOCK.json was rewritten
  afterwards.

## 16. Completion

Completed 2026-10-25T21:51:39Z (UTC clock read immediately before this line was written), after the statistical reviewer's wave-1b
hand-back (`roles/statistical_reviewer/HANDBACK_1b.md`: rules final at 21:49:38Z; `rfa/rules.py` 76baa04a..., `rfa/test_rules.py`
b00482c4..., RULES.json 4e03270c..., RULES.md 1548b6de...; 24 of 24 tests pass). The reviewer's reading that a NEAR_ALIAS ALTCRIT still
gates (never counted as independent) is adopted. The mechanism card RF_RFIT_v1 is committed with this registration. No label had been
read by this campaign.
