# Coordinator log (UTC, real clock; append-only)

- **14:29:43Z** Campaign start (`date -u`, the first action).
  - AGENTS.md was inspected first: it does not exist at the worktree root or in the main checkout.
  - Branch `research/pcrl-new-release-mechanisms-v1` was created from 0f27e5937, the controlled training-size research tip and the base of
    publication ed7bc1ccd.
- **14:30-14:40Z** Read the latest source (START_HERE, RESEARCH_DECISION, EXPOSURE_MAP, conditional/EQUAL_RESERVE.csv) and the reusable
  infrastructure (tsz.common, tsz.jobqueue, tsz.devreaders, tsz.evaluate, ohn.evaluate, ohn.devrun).
- **~14:38Z** Wrote `roles/coordinator/AGENT_CONTEXT.md`, the shared brief for every agent. Launched wave 1, four assignments, five live agents
  including the coordinator:
  - Designer A: learned finite release;
  - Designer B: joint task / randomized channel;
  - challenger: failure map, controls catalogue, generic SEX-blind control code;
  - statistical reviewer: generic claims, selection rules, screen checks.
- **14:38-14:47Z** Built and selftested `nrm.common`, `nrm.jobqueue`, `nrm.lane`, `nrm.release`, `nrm.evaluate`, `nrm.screen`, `nrm.devreaders`,
  `nrm.admit`, `nrm.receipt` and `nrm.publish_build`. Deviation 1: the grid selftest's CPU was charged out of the queue.
- **14:45Z** Source admission `ok`:
  - pin ed7bc1ccd equals the remote branch tip;
  - 222 published results files and 28 code files are identical to the local copies (some after the source sanitiser);
  - all 2,618 private source copies are bitwise identical to the source worktree.
- **14:46Z** READER_SPLITS.json written (label-free): NRM1_devgrid_R1..R4 and the NRM1_screen halves. ACCESS_POLICY.json written.
- **14:48Z** Commit febfba43a: infrastructure and registration drafts. Waiting for the failure map and rules before the receipt.
- **14:49-14:52Z** Checked the earlier MMP proposal (mechanism-design campaign, data-free):
  - it was declined because an injective deterministic release leaves P(S | X) learnable;
  - so a SEX-standardization family would add only a learnability effect;
  - it is not proposed as a third family. Many-to-one cells or randomization are the information-removing routes.
- **14:51-14:52Z** Grid timing (one thread): about 16 CPU-s per release per cell. Deviation 2: the first timing unit failed on an odd
  invented size. Reference evaluation and screening lane written (runs after the receipt).
- **14:53Z** Publication dry run: sanitizer and leak scan report 0 problems. Agents' uncommitted files were skipped.
- **14:54Z** `nrm.deploy`: a generic raw-field packager over the family interface `predict` / `model_paths`.
- **~14:56Z** Launched the independent verifier, an early wave-2 assignment, to build separate numerical code. That makes six live
  agents including the coordinator, the limit.
- **14:57Z** Challenger returned. FAILURE_MAP.md (10 families), the controls catalogue and nrm/ctl (selftests 29/29 and 17/17) are
  committed. The closeout tools (custody, restore check, ledgers, receipts) are adapted; the lineage DEV_REPORT log snapshot is
  c7829e99 at 14:55:55Z.
- **15:00-15:02Z** An extraction sub-agent of the challenger reported directly to the coordinator. ListAgents showed one more unknown
  agent: the statistical reviewer's code-review sub-agent. So sub-agents had pushed the live count above six (deviation 3).
  About 15:02Z every running agent was told not to spawn sub-agents; the rule was added to AGENT_CONTEXT.md.
- **15:00-15:05Z** Designer B returned FSC-CA (channel-aware joint training with a SEX-targeted, Fisher-shaped, coupled Gaussian logit
  channel; code ef0bc1ddf) and a second hand-back with its sub-agent disclosure. Designer A returned FA-PRC (propensity-routed finite
  cells; code 47916952c), disclosing 3 sub-agents and a multi-threaded selftest (deviation 4). All saved verbatim.
- **15:10Z** Decisions, provisional pending the challenger's equivalence review and the receipt:
  - FA-PRC: 18 configurations (+2 contingent for R0). No joint-design claim unless PRC beats LOC clearly; sequential orders would
    then be the decisive phase.
  - FSC-CA: 16 configurations (S1 dropped from both paths). No coupling claim, since the SEX and PCA paths are both coupled.
  - Configuration plan: A 18 + B 16 = 34. About 8-12 go to generic controls, about 8 to ablations and perturbations, and the rest are
    kept for revisions.
  - Both designers are wiring access, release and deployment; the deploy contract now carries out-of-band draw keys for randomized
    packages.
  - The challenger is reviewing both proposals, due about 15:30Z.
- **15:12Z** The statistical reviewer returned. Rules NRM1-rules-v1 are frozen at 6dc7375f0, with 18/18 invented-data tests and the
  known-law screen behaviour in SCREEN_CHECKS.json; its report is saved verbatim. Its open point 5 is done: `total|nll` paired cost
  added to nrm.evaluate.
- **15:14Z** Receipt build: 59 committed files, sanitizer and leak scan 0 problems, the 8 gated files unchanged. Commit 301b9459e
  (orphan) on `publication/pcrl-new-release-mechanisms-v1`; pushed at 15:14:39Z; `nrm.receipt verify` VERIFIED at 15:14:43Z
  (`ls-remote` tip = 301b9459e, gated bytes identical to research commit 67b70af6d). `nrm.common.registered()` is True: label reads
  are open under the policy from here. No real-label computation or fit happened before this moment.
- **15:15Z** Started the reference lane REFS_V1 through the queue: utility of W, the six references, U0, Ucal, TASK_REFIT and T_TREE,
  plus screening readers on W, NEURAL_RAW, H_LJ, RETIRED_B3CTL and U0.
- **~15:16Z** Launched the engineering/claims reviewer (wave 2 assignment) for a pre-outcome code review, due about 16:00Z. Live
  agents: coordinator, designers A and B, challenger, verifier, engineering reviewer (six).
- **15:15:30Z** Reference lane done (6 units, about 23 CPU-s).
  - All references pass the point limits.
  - Screening pair AUC (adaptive, screening halves): W 0.641/0.641, U0 0.640/0.643, NEURAL_RAW 0.622/0.625, H_LJ 0.628/0.626 (its
    occupation view 0.568/0.575), RETIRED_B3CTL 0.600/0.600.
- **15:16-15:17Z** The grid supports randomized releases (one draw per person per cell; instance = repetition index; Monte Carlo
  draw offsets). Designer B reported its wiring done: 16 configurations, keyed deployment interface, card drafted but uncommitted.
- **15:19Z** Challenger review saved.
  - A: closed cell line plus one real operation (routing on π); SHUF control needed; sparse pair reading favours PRC over ORD.
  - B: a new operation, but the PCA SEX-blind control is a strawman.
  - Coordinator card decisions v1 committed (CARD_DECISIONS_V1.md):
    - A: LOC replaced by SHUF, 18 configurations;
    - B: PCA replaced by MINF S2/S3 plus ISO S2, 18 configurations;
    - CTL_REM on FB_TWIN.
  - Deviation 5: the decisions header time was ahead of the clock; corrected.
- **15:19-15:20Z** GO sent to both designers: commit the cards, then fit. A fits its twin first and waits for the coordinator's
  MODEL_SELECT utility check before the protected paths.
- **About 15:20Z** Launched the reader-robustness investigator (seventh assignment). This entry first said "~15:21Z", and the commit
  message of 7c06dc0ce says "to 15:21Z"; both are ahead of the clock, which read 15:20:24Z at that commit (deviation 5). Phase 1:
  reader-budget diagnostics, robustness plan, stability-summary code. Live: coordinator, A, B, verifier, engineering reviewer,
  reader-robustness (six).
- **15:24-15:28Z** Family B fitted 18 releases; card FB_FSC_v1 was committed at 15:24:05Z, before the fit.
  - MODEL_SELECT utility: FB_TWIN v1 FAILS the men's guard (0.0085); every FB configuration fails in at least one version (P1 false).
  - Screening: SEX targeting beats MINF/ISO at matched second-order cost (about 0.005-0.007 pair AUC); POSTHOC ≈ SEX (F3); ID ≈ twin.
- **15:28-15:37Z** Family A.
  - Card FA_PRC_v1 committed at f19614b05 (before its fit). PRC_TWIN passes the original limits on MODEL_SELECT, but quantization
    uses about 0.012 nats of A1's occupation margin.
  - The registered internal capacity gate FAILED in both versions, and so did the registered R0 (24 x 128) gate.
  - Family A closed at the capacity stage (FINITE_CELL_CAPACITY_NOT_ESTABLISHED); 4 configurations used. Deviation 7: the R0 twins
    followed the coordinator's own 15:31Z instruction.
- **15:31-15:38Z** Family B revision.
  - Coordinator proposed FB2 (frozen SEX-targeted channel post hoc on HC_STRONG income + A1_PO_16 occupation).
  - The statistical reviewer's card check (D1-D6) was adopted verbatim after FB outcomes had been read (deviation 6).
  - The FB2 ruling is APPROVE WITH CHANGES, final version at 15:36:04Z (C1-C7; deviation 8).
  - GO and a correction were sent to Designer B.
- **15:34-15:44Z** Engineering review wave 1: 15 findings, no blocker.
  - The coordinator applied ENG1-01..06 and 10..12 to its own modules (adjudication A1).
  - Reader-robustness extras integrated (A2).
  - Grid code locked at 15:43:59Z (1128afd67) after the full grid selftest passed (32 CPU-s, one thread).
  - No replacement family (A3).
- **15:44-15:48Z** Challenger tasked with CTL_REM on FB2_BASE (CTL v2 card). nrm.claims written. Reference wires evaluated
  (EVAL_MS__WIRES).
- **15:45-15:49Z** Designer B fitted FB2.
  - Card FB_FSC_v2 committed at 15:45:14Z, before the fit; 10 configurations. The base gate is bitwise inside the fit; deployment
    parity bitwise.
- **15:49Z** C3 gate on FB2_BASE: PASS (24 + 4 both versions; maximum value/cap −0.287 / 0.050; reserve met 4.58 / 3.30).
- **15:50Z** FB2 evaluation and screening.
  - SEX_S3 and SEX_S5 pass the original limits in both versions. SEX_S5's reserve holds in one seed only (2.38 / 1.53).
  - MINF_S5 v1 fails occupation NLL vs Ucal.
  - Screening pair AUC: BASE 0.649, SEX_S3 0.636, SEX_S5 0.623, MINF_S3 0.645, MINF_S5 0.639.
- **About 15:51Z** Spend-only decisions: no MINF_S6 (both SEX points are bracketed on the MINF path); CTL_REM targets 0.0074 / 0.0226
  sent to the challenger.
- **15:51:50Z** Bank NRM1_MAIN (28 members) and PLAN_MAIN committed before the grid ran. Grid units NRM_GRID_MAIN_S0/S1_V1 launched.
- **About 15:53Z** Designer B asked to prepare the decisive-phase code (MEAN per C5; a genuine training perturbation of the channel
  parameters for SEX and MINF at one strength). Nothing is to be fitted before the grid and a card v3.
- **About 15:59Z** Designer B prepared the decisive-phase code: MEAN, and the FB2P perturbation (2fa4e64ac).
- **About 16:00Z** The statistical reviewer ruled on card v3, before any NRM1_MAIN output was read: APPROVE WITH CHANGES
  (RULING_FB_V3.md).
  - MEAN per C5, with an exact F2 definition.
  - FB2P runs only at a strength that is TASK_FEASIBLE in both versions and AGG_POSITIVE against its same-strength MINF.
  - The perturbation reading is defined exactly and is descriptive only.
  - Designer B told to commit v3 and not to fit.
- **16:02Z** nrm.claims reads wires from the newest record that carries them. The four descriptive members are re-evaluated with
  tag WIRES (queued behind the grid).
- **15:20:24Z** Queue: 16 units done, 3 failed. The failures are NRM_GRID_TIMING_V1 (deviation 2) and the verifier's two selftest units
  launched with malformed unit ids (`VER_ACCESS_SELFTEST_V1 v1_access.py`, `VER_UTILITY_SELFTEST_V1 v2_utility.py`; rc 1, 0.015 CPU-s
  each; rerun as V2). Charged CPU so far: 0.622 h.
- **16:03Z** Designer B committed card v3 (c4ded0dd4, 16:02:46Z). MEAN and FB2P wait for the rules applied to the grid.
- **16:07-16:10Z** Challenger: CTL_REM runner and CTL_CARD_v2 committed (bcd9df4fd, 485a329ca, ac488d353); plan units queued. Its
  deploy question is already covered by REPO:: keys. Reports saved verbatim (131a3ec76).
- **16:13Z** nrm.decisive written and committed (1add70187) before any NRM1_MAIN output was read: the frozen decisive-phase rules
  (FB2P 2(a), F1 C6, MEAN C5, F2, perturbation section 3) applied label-free from committed claims and saved grids.
- **Before 16:15:46Z** The statistical reviewer confirmed delta and se_delta and replaced the applicability check (registered Monte
  Carlo B0; flag B0_SINGLE_INSTANCE_NOT_RESOLVED). Saved verbatim; nrm.decisive updated (9a77e0852, header time corrected 59b4661ac).
- **16:17:12Z / 16:17:17Z** Grid units NRM_GRID_MAIN_S0_V1 / S1_V1 done (exit 0; 4,800 reader fits and 1,469.5 / 1,474.7 reader
  CPU-s each). The 12 locked grid code files are unchanged from the lock (1128afd67). REEVAL_WIRES_V1: 4 units done.
- **16:18Z** Claims assembly NRM_CLAIMS_MAIN_V1 queued on the committed PLAN_MAIN (label-free).
- **Before 16:19:39Z** Scheduling message to the reader-robustness investigator (headed "16:20Z", ahead of the clock; deviation 11):
  at most one robustness unit in the queue at a time. RR_ROB_REPS_S1_V1 had already started at 16:18:38Z, so withdrawing it ended a
  running unit (failed at 16:20:10Z, 91.2 CPU-s); it was rerun as RR_ROB_REPS_S1_V2 (done 16:37:45Z).
- **16:20:13Z** First claims run (NRM_CLAIMS_MAIN_V1): every grid READER_SEEDS_NOT_EFFECTIVE. Cause found within about a minute: an
  input dropped in nrm.claims.validity (deviation 10). Fixed (3d0ef8c15); rerun NRM_CLAIMS_MAIN_V2 at 16:21Z: FB2 grids valid, G2
  NOT_MET, G3 NOT_SEPARATED (MINF path), G4 NOT_SUPERIOR, G5 NOT_MET; selection NO_DEVELOPMENT_CANDIDATE. The verifier replayed MAIN_V2
  with 0 mismatches (interim report 16:24Z).
- **16:22Z** nrm.decisive: FB2P rule 2(a) NOT_RUN (SEX vs same-strength MINF +0.0019 se 0.0089 at S5, +0.0027 se 0.0089 at S3); F1
  HOLDS. Family B closes by its card's stop rule.
- **16:26-16:31Z** CTL path: evaluations (lane EVAL_CTL_V1), bank NRM1_CTL and PLAN_FINAL committed (390b16ca8) before the CTL grid;
  grid units 96 s (V1) and about 2 min (V0); final claims: NOT_SEPARATED on both paths at both strengths; MEAN NOT_FITTED (C5); F1
  HOLDS. CTL_REM_1p00 lowered pair recovery by 0.0148 (resolved) but fails the occupation clauses at 2.7-3.0 times the caps.
- **16:32Z** A4: no replacement family (ADJUDICATIONS.md); deviation 12 (A3 repeated the withdrawn injectivity premise; erratum).
- **16:30-16:35Z** Closing reports saved verbatim: Designer B, the statistical reviewer (no departure; required headline wording; no
  objection to A4 with a wording condition, applied), the challenger (untested routes listed; no objection to A4).
- **16:34Z** Reader custody: 6 representative grid readers refit bitwise to the grid cache and pickled (entry point authorized in
  ACCESS_AMENDMENTS.json before use).
- **16:36Z** Deployment: 6 packages, all bitwise on DEV_REPORT raw inputs at both entry points, all 19 refusals each.
- **16:36-16:41Z** Frontier figure, TABLES.md (incl. the reserve-qualified frontier: FB2_SEX_S3 NO_RESERVE_FRONTIER_IMPROVEMENT).
- **16:40Z** Engineering / claims reviewer resumed for wave 2 (code). Documents drafted (4083605e4).
- **16:44:57Z** Last robustness grid unit (RR_ROB_SEEDS_S1_V1) ended; queue empty.
- **16:45:38-16:46:18Z** Same-device backup B1 (unit NRM_CUSTODY_B1_V1): restore check from the backup alone all_ok (6 packaged models
  rerun bitwise, 6 representative readers re-scored bitwise under both interpreters, 1,000 cache entries, 6 packages redeployed with all
  refusals). Off-device: YOTUO not attached; `nrm.custody offdevice` returned UNAVAILABLE at 16:46:48Z (custody/BACKUP_VERIFICATION.json).
- **16:47Z** Verifier resumed for its final pass. Engineering review wave 2 (code) returned: ENG2-01..11, none blocking.
- **About 16:48Z** Reader-robustness Phase 2 report: every comparison ROBUSTNESS_NOT_CONFIRMED; reader seeds partly effective. Saved
  verbatim (510293ef5).
- **16:50-16:54Z** ENG2 fixes (735cde202); claims re-run as FINAL_V2 (one selection-record field differs; summaries identical); F1 and
  MEAN re-applied; tables (Monte Carlo pooled pair AUC) and figure regenerated; documents updated; A5 recorded (b2f0dccce).
- **16:54Z** Verifier told to replay FINAL_V2; engineering reviewer asked for part 2 (documents).
- **16:48-16:57Z** Verifier final pass: all decision-relevant records agree (claims FINAL and FINAL_V2 0 mismatches, six decisive records,
  reserve frontier, CTL and robustness grids exact, deployment from the backup alone bitwise with all 21 of its probes, access audit of
  97 reads clean). Saved verbatim; VERIFICATION.md written (7914dbf2d).
- **About 17:03Z** Engineering review wave 2, documents: DOC2-01..15, none blocking; all applied (5d02d3ee7, A6). The message of commit
  b2f0dccce wrongly says the figure uses pooled values; it keeps the base instance (labelled).
- **17:09:06-17:09:53Z** Final same-device backup at research tip 9974d9efe (unit NRM_CUSTODY_B1_FINAL_V1): all_ok. Off-device still
  pending (drive absent at 17:10:09Z).
- **17:10:41Z** Final ledgers and receipts (845a03bb7): 2.68 elapsed h, 2.36 CPU-h of 14, 171 units done and 6 failed, 36 of 60
  configurations, 97 label reads, none before the receipt, no campaign process left. Next: publication, then the queue is closed.
