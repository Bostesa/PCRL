# Coordinator log (campaign `pcrl_dual_design_release_v1`)

Times are UTC, each copied from a `date -u` read at or before the event.

- **16:37:33Z** Campaign start (first `date -u`); memory notes and the finite-solver-adequacy / finite-routing records read.
- **16:43:54Z** Branch `research/pcrl-dual-design-release-v1` created from f0edb8ad1 (finite-solver-adequacy research tip).
- **16:45:22Z** Design brief written (`DESIGN_BRIEF.md`); Designer A (training-side space) and Designer B (release-side space) launched
  independently.
- **16:48-16:50Z** `ddr/` derived from `frc/` (constants, paths, reader seeds +800, split tags DDR1, bootstrap 2026102200);
  `ddr.admit run` (nrm 682068d52, frc 04ac9161f, fsa ddc231a97 pinned; `ok` after the full-SHA fix); access policy; queue opened
  16:49:33Z; release / evaluate / reader-grid selftests pass; split manifest and grid code lock written.
- **16:52Z** Protocol draft; statistical reviewer launched (rules and pre-registration review).
- **16:55-17:00Z** Infrastructure committed (3dbd8c50e). Label-free coordinator diagnostics on cached MODEL_FIT out-of-fold predictions
  (`labelfree/PROPENSITY_STRUCTURE.json`, `MIXTURE_PROXY.json`): model-based expected AUCs of the propensity carried by the task
  posteriors (income 0.556, occupation 0.644, pair 0.654 vs the propensity's own 0.681) and of shrunk-propensity SEX mixtures.
- **17:19Z** Designer A's proposal (STD, SEX-standardized task posteriors) received; DA integration begun (`ddr/da/std.py`).
- **17:23Z** The statistical reviewer's RULES.md stated a completion time ahead of the clock (17:30:53Z at 17:23:55Z); asked to
  correct it (corrected to 17:26:03Z, disclosed in RULES.md).
- **17:30Z** Reviewer's hand-back: rules frozen (22 / 22 tests); REPORT_PREREGISTRATION.md saved verbatim (the harness refused the
  agent's own write). All REQUIRED findings 1-10 adopted into the protocol; advisory 11-14, 17-20 adopted.
- **17:31Z** Stale seed constants in `ddr/devreaders.py`'s docstring fixed; grid code re-locked before any grid run.
- **17:33Z** Designer B asked to finalize by about 17:45Z.
- **17:36Z** Designer B's proposal (PTC, propensity-tilt cancellation) received; challenger launched on both proposals.
- **17:36-17:41Z** DA module rewritten to the adopted control rules (SBC: permuted SEX, declared label-derived fallback; SPEND:
  coarsening of DA's twin; two MODEL_FIT bracket points each); DB module written (`ddr/db/ptc.py`); both pass invented-data
  selftests end to end.
- **17:41:46Z** Protocol completed. **17:42:05Z** Registration committed (8f736cbe3).
- **17:42:37Z** Receipt `c5a84c19d` pushed to `publication/pcrl-dual-design-release-v1`; **17:42:42Z** verified on the remote
  (`ddr.receipt`); `ddr.common.registered()` true. No label read before this point.
- **17:43Z** Decision: no MODEL_SELECT read (not even reference utilities or reference readers) until both mechanism cards are
  committed, so that the registered readings stay registered under RULES.md 7. Draft card DB_PTC_v1 written.
- **17:57Z** Challenger's hand-back (both designs PROCEED_WITH_REQUIRED_CHANGES; not equivalent; no redirect); review saved verbatim
  (`roles/challenger/REVIEW_DESIGNS.md`).
- **17:58-18:06Z** Adjudication (`PROTOCOL_AMENDMENT_1.md`): shared twin; continuous SEX-blind SPEND path DQ replaces the finite
  coarsening ladder (both families); PSEUDO2 SBC fallback with the SEX-independence rule and recorded profiles (both families); ZTILT
  (required) and SHUF for DB; data-rich reader check (`ddr.richreaders`, access amendment for the scorer); DB covariance threshold
  1e-4; one DB candidate (S3). Both cards rewritten from the code; all selftests pass. Still no label read.
- **18:06:14Z** Cards DA_STD_v1 and DB_PTC_v1 committed with their code (5f2fd02a9). **18:06:27-18:08:16Z** registered fits (four
  queue units): 32 configurations; DA lam* = 1.0 / 1.0, DB alpha* = 0.35 / 0.35; fit records `da/FIT_V*.json`, `db/FIT_V*.json`.
- **18:09:16Z** First MODEL_SELECT read (DA utility evaluation); utilities of DA, DB and the references; registered grid DDR1_MAIN
  started for both versions (18:09:16Z / 18:09:27Z).
- **18:10:43Z** Robustness bank DDR1_ROB written from the MODEL_SELECT bracket costs, before any grid output (8cb3ec9f9).
- **18:13:37Z** Restore check adapted; claims lane; verifier brief (53e53a213); verifier launched.
- **18:15:30Z** DB's registered MODEL_FIT trigger (card DB_PTC_v1 section 6) held; card DB_PTC_R1 committed with its code before its
  fit and before any reader-grid output (ae967496a). Its tag is set by the rules (card after the first MODEL_SELECT read: ADAPTIVE).
  Waiting robustness units withdrawn so that the short R1 fits run first (deviation 3).
- **18:17:46Z** Table generator, revision-plan writer and closing-review brief committed (51def3497).
- **18:31:57Z / 18:33:14Z** Registered grid done (V0 1,355 CPU-s, V1 1,420 CPU-s). **18:32Z** DB R1 fits: (alpha_I, alpha_O) =
  (0.35, 0.65) in both versions. **18:33:17Z** its bank DDR1_R1 and plan PLAN_DB_R1 written before any R1 MODEL_SELECT read
  (24f929bf9); deviation 4 (ROB2_V1 withdrawn before start, re-queued as ROB3_V1).
- **18:33-18:34Z** Verifier's label, integrity and access units; data-rich scores of the registered bank. **18:34:27Z** registered DA
  claims: D1 TASK_FEASIBLE, D2 PARTIAL_SBC_FRONTIER_ONLY (SPEND frontier not separated: +0.0137, se 0.0174; TWIN point only:
  +0.0175, se 0.0090), D3 NOT_SUPERIOR, D4 NOT_MET; class TASK_ONLY_FEASIBLE_NO_PROTECTION.
- **18:38-18:41Z** DA's declared R1 trigger (b) needs its second, MODEL_FIT condition: `trigger_r1` written (261089a74) and queued.
  R1 code (IPW-STD) written and tested on invented labels (38d5fa6ba); its card is drafted and is committed only if the trigger
  holds. Deviation 5 (ZTILT's utility evaluation missing; queued); first tables (ecc260651).
- **18:48:28Z** Before any revision grid output: DB R1 beside bank (vs ZTILT, SHUF and DB_PTC) and its data-rich spec; revision reader
  files; DA R1 trains its data-rich readers; DA strength key in revision plans (1c7064bd6). D2R plans and lanes queued behind the
  robustness grids and DB's registered claims; DA R1 lanes pre-written (started only if its trigger holds).
- **18:52-18:56Z** Found that DA_STD_v1's trigger (b) names `NOT_SEPARATED`, while DA_REG is `PARTIAL_SBC_FRONTIER_ONLY`; the draft
  card DA_STD_R1 was rewritten to say that, strictly, (b)'s first condition fails (deviation 7). Deviation 6 (code changed during
  grid runs, outside the lock; lock re-hashed unchanged).
- **19:10:20Z** Registered DB claims: D2 NOT_SEPARATED; DB's R2 trigger computed on the registered reading (version 1 only: not met).
- **19:13Z** Robustness grids done; D2R of DA_STD and DB_PTC: both ROBUSTNESS_NOT_CONFIRMED. DA R1 trigger: the MODEL_FIT condition
  held in both versions (income R^2 ratio 0.666 / 0.661). **19:13:36Z** card DA_STD_R1 committed (00eef8eb7), deviation 7; DA R1 fits
  queued.
- **19:16:36Z** DB R1 claims: ADAPTIVE_R1 PARTIAL_TWIN+SPEND_FRONTIER_ONLY (TWIN resolved +0.0239; SBC +0.0265 se 0.0139 not
  separated). DB R2 not triggered on R1. **19:17-19:20Z** Strongest-release rule: DB_PTC_R1 (split strength passed as alpha_I + alpha_O,
  deviation 8). The verifier's first report arrived and was saved verbatim (all checks reproduce; V2.5 = deviation 5).
- **19:18-19:24Z** DA R1 reading: ADAPTIVE_R1 NOT_SEPARATED (+0.0072 vs the twin); DA's strongest stays DA_STD.
- **19:21:36Z / 19:26:25Z** Cards DB_PTC_R1_ABL and DB_ZTILT_R1 committed before their fits (deviation 9; 48 of 48 configurations).
  Ablation fitted 19:22Z; refit bank DDR1_ROB_DB_R1 and its D2R plan committed (8d2d1b560) before its grid; ZTILT_R1 fitted 19:27Z.
- **19:29-19:31Z** Campaign summary (0 of 4 D2 looks met); deployment of 10 packages checked (all bitwise, all probes refused).
- **19:33-19:34Z** Reader custody (one failed attempt on a composite winner recipe, deviation 10; re-run in another cell).
- **19:36:24Z** D2R of DB_PTC_R1: ROBUSTNESS_CONFIRMED against TWIN, its SBC and SPEND brackets on the refit cells; ablation: the income
  component carries about 60%. Statistical reviewer launched for the closing review (19:37Z); verifier resumed with an addendum
  (revisions, refits, new configurations, deployment) at 19:33Z.
- **19:42:53Z** DB_PTC_R1 vs its release-only control ZTILT_R1 (intended to match its spend; it overspent): +0.0128 (se 0.0081), not resolved (declared reading: not separated from
  release-only information); tables regenerated over every bank (21f588e2b, 19:43:44Z). Document drafts committed (0678392a3).
- **19:44:59Z** Same-device custody backup started (queue unit DDR_CUSTODY_V1). **19:45:49Z** verified from the backup alone (all_ok).
  The verifier's addendum arrived (A1-A8 all pass) and was saved verbatim; its observations were disclosed as deviation 12.
  Claims reviewer launched at about 19:46Z.
- **20:00Z** The statistical reviewer's closing statement arrived (completed 19:59:56Z by its own `date -u`): every label follows
  from the frozen rules; required headline; corrections C1-C29. Five factual premises were checked against the records before
  adoption (one idea family in PROTOCOL_AMENDMENT_1; 8,001 MODEL_SELECT people; the S0R1 seed-pair blocks; the 0.010 recipient
  allowance; injectivity disclosed in card DB_PTC_v1). Documents rewritten with the headline verbatim (ae6626c44, 20:04:58Z);
  deviations 12(e) and 13.
- **20:08-20:13Z** The claims review arrived (based on ae6626c44); its open corrections applied (fdc9d6d32, 20:13:14Z). They
  included one coordinator error copied from the verifier's discrepancy text: DB_ZTILT_R1's near-limit guard is the men's (`sex0`),
  not the women's. Out-of-queue CPU of the designers, the challenger, the reviewers and the coordinator charged (upper estimates).
- **20:13:28Z** Queue closed (no unit running or waiting); ledgers, RESOURCE_ACCOUNTING.json and ACCESS_RECEIPT.json written;
  resource sections of the documents filled.
