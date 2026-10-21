# Coordinator log (campaign `pcrl_finite_routing_comparison_v1`; times from `date -u` outputs)

- **22:49:23Z** Campaign start (first `date -u`; the coordinator's memory notes were read in the minute before). AGENTS.md inspected
  first (22:52Z): it does not exist at the worktree root or in the main checkout.
- **22:49-22:58Z** Read the pinned source package (publication 682068d52): START_HERE, RESEARCH_DECISION, cards/FA_PRC_v1,
  Designer A's closing report, FA_CAPACITY_DECISION, TABLES, VERIFICATION and the verifier's FINAL_REPORT; then the source code
  (nrm.fa.prc, nrm.common, nrm.evaluate, nrm.release, nrm.jobqueue, nrm.devreaders, nrm.rules, receipts, admission, publication,
  deployment, custody) and NRM1 RULES.md.
- **22:58-23:00Z** Pre-registration label-free check on the saved twin models and cached out-of-fold teachers (no label; 4.5
  CPU-s; charged at 23:21:42Z): stored prototypes are not the out-of-fold cell means (max 0.0019-0.0062 on occupation), decoders
  differ from prototypes by up to 0.09-0.14, deployed and out-of-fold routing of MODEL_FIT people agree for 56-76%; saved releases
  reproduce bitwise.
- **23:01:38Z** Branch `research/pcrl-finite-routing-comparison-v1` created in this worktree from the source research tip
  fd933a824 (the same worktree carried the two previous campaigns; their private caches are read-only inputs here).
- **23:01-23:21Z** Package `frc/` derived from `nrm/` (access layer, queue with 12 CPU-h, release storage with 48 configurations,
  evaluator with source comparators, FRC1 grid, banks, reader budget, receipt, admission, publication, closeout, ledger, custody,
  deployment). Family module `frc/fa/route.py`; selftest 34/34 at 23:15Z. Draft PROTOCOL.md (from 23:06Z), AGENT_CONTEXT.md
  (23:05Z), EXPOSURE_MAP.md and card FA_FRC_v1.md (from 23:20Z). Source admission ok (23:19Z); reader-split manifest written.
- **23:07Z** Statistical reviewer and challenger launched (three live agents with the coordinator).
- **23:21:15Z** Commit 4b6059d28 (infrastructure and drafts; no label read).
- **23:21:42Z** Queue opened by the first charge (pre-registration work, 150 CPU-s upper estimate).
- **23:21:53Z** T0 units FRC_REPRO_V0_V1 / FRC_REPRO_V1_V1 (label-free): saved twins reproduced bitwise on all six roles, books
  refitted bitwise, partitions rule-consistent (4.9 CPU-s each).
- **23:24-23:41Z** Both reviews returned (challenger 4 BLOCKER / 9 SHOULD FIX / 8 NOTE; statistical reviewer rules 15/15 with
  C1-C11). Adjudication written (ADJUDICATION_PREREG.md, from 23:30Z): QUAL slack moved to the S2 budget; P3 on out-of-fold cells;
  F4 and reader targeting added to D2 (F4 a conjunct, not a stop); SHUF guard on the person's own qO; HQ / SHUF / ORD bracketing
  points; T6 labelled POST_HOC. The coordinator also found that `nrm.readerextra` imports `nrm.devreaders` relatively (the grid
  extras would have used NRM1's splits); fixed by the campaign copy `frc/readerextra.py`. Statistical reviewer second and third
  passes (17/17 tests). Family selftest extended to 45/45 (23:37Z).
- **23:41Z** Engineering reviewer launched (pre-registration code review). **23:46Z** Independent verifier launched (writes its
  separate replay code; no label until the receipt). Live agents: coordinator, engineering reviewer, verifier.
- **23:44Z** Reader custody module `frc/readercustody.py` added and authorized in `frc.common.ENTRY_POINTS['readers']` before
  registration (no amendment needed). Publication dry build: 43 files, 0 leak-scan problems, gated files unchanged by the
  sanitizer.
- **23:47:33Z** Registration commit 98ede167e. **23:47:58Z** Receipt 7fed2ca44 (root commit of
  publication/pcrl-finite-routing-comparison-v1, 46 files, 0 leak-scan problems) pushed to GitHub. **23:48:04Z** Receipt verified
  on the remote (frc.receipt; git ls-remote equals the receipt commit; the eight gated files byte-identical). Label reads open.
- **23:48:50-23:49:41Z** T1 (4 units, 33 CPU-s each): GATE records reproduced exactly (max difference 0.0) and decoders bitwise;
  B1 - A nested occupation NLL -0.0039 / -0.0049 (16 x 64), -0.0085 / -0.0090 (24 x 128), all resolved; B0 = B1 within 1e-5.
- **23:50:08-23:50:30Z** T2 (4 units, label-free): TWB_C64, TWB_C128 saved (configurations 1-4); partition identity with the saved A
  releases on all six roles. **23:50:31-23:50:52Z** T3 (5 evaluation units, MODEL_SELECT). **23:51:18Z** `frc.decide qual`:
  QUALIFIED at 16 x 64 with B1 (D1 REPAIRED_TASK_ONLY_RELEASE); committed e74248a18.
- **23:52Z** Engineering review returned (13 findings); fixes applied, card addendum A1, committed b538fa7a4 (23:56:09Z);
  source pins re-checked (all equal). **23:56:41Z** Grid code lock a2c751cb7. restore_check adapted (4db84f699).
- **23:56:20-23:59:20Z** P1 (2 units, about 171 CPU-s each). F4 passes in both versions. SHUF never reaches PRC_S2's nested spend;
  HQ reaches it in version 1 only. Committed 90d7b9a31 (00:01Z).
- **00:01:30-00:02:02Z** P2 (2 units): 12 configurations (16 in total). PRC_S1 V1 found to be an exact alias of PRC_S2 V1 (deviation 3).
- **00:03-00:06:27Z** Revision R1 decided on MODEL_FIT evidence only (the registered untargeted control cannot reach the matched
  spend), implemented (selftest 50/50), card FA_FRC_R1 committed 01a86ce20 before its fit. **00:06:40-00:07:17Z** R1 units:
  UNT_BR saved in both versions (f_O 0.10 / 0.30; 18 configurations).
- **00:07:30-00:08:03Z** E1: 8 evaluation units (MODEL_SELECT utility and paired costs; every configuration point-feasible).
- **00:08:15-00:10:28Z** Bank FRC1_MAIN (21 members), reader budget (label-free), claims plans PLAN_MAIN / PLAN_R1; committed
  70e84adf9 before any grid output existed. **00:10:40Z** Grid units FRC_GRID_MAIN_S0_V1 / S1_V1 launched; E4 ablation fits queued.
- **00:11-00:13Z** E4 ablation fits ABL_LOC / ABL_PAIR at PRC_S2's strength (both versions); robustness and learning-curve bank
  specs FRC1_ROB / FRC1_LC written (00:13:19Z) before any grid output.
- **00:25:07 / 00:25:34Z** FRC1_MAIN grid done (863 / 890 CPU-s). **00:25:48-00:25:50Z** First claims runs (registered and R1
  plans): `NOT_COMPUTED_GRID_INVALID` (85 minimum-set equalities, all ET5 / ET25 on few-valued finite views). Committed ab0506490.
  **00:27Z** Engineering repair requested from the statistical reviewer before any D2-D4 value was read (deviation 6).
  **00:27:43Z** FRC1_ROB grid units launched.
- **00:30Z** Verifier first hand-back (92 checks, no numerical FAIL); **00:31Z** ER1 returned (18/18 tests). Dispositions:
  deviations 7 and 8; `refit_effectiveness` keeps the strict rule. Committed 6d6cd0829.
- **00:32:21-00:32:23Z** Claims under ER1, both readings: D1 REPAIRED_TASK_ONLY_RELEASE, D2 NO_REDUCTION_BEYOND_ANY_CONTROL, D3
  NOT_SUPERIOR, D4 NOT_MET, outcome ENGINEERING_REPAIR. Committed c9458cd6c (00:34:12Z) with the label-free probe module.
- **00:34Z** Observation (MODEL_FIT records only): S2 and S3 both selected PRC@8.0, the top of the registered grid, in both
  versions, and the S2 budget did not bind (V0 occupation accuracy spend 0.00225 + 2 x 0.00095 against 0.006; V1 income accuracy
  0.00115 + 2 x 0.00052 against 0.004). The registered strength path therefore realized one distinct nonzero point per version.
  Before deciding whether this justifies the second allowed revision, a label-free probe (`frc.fa.probe_strength`; cached MODEL_FIT
  out-of-fold statistics, no label file, no MODEL_SELECT) continues the same chain to strengths 16, 32, 64, 128.
- **00:44Z, registered before the probe ran (no probe output existed; committed c7b80cb1c at 00:44:34Z, whose log line first said 00:46Z in error):** decision rule for the probe.
  - `SATURATED` if, in a version, the partition at every extended strength equals the partition at 8. If either version is
    saturated, no revision R2 is made: version-specific points are not pooled path points (deviation 4), so a one-version extension
    could not enter the registered pooled comparisons. The stronger budget point is then recorded as unreachable by the registered
    optimizer (its smallest rule tilt is strength / 64, so the candidate moves grow with the strength), an untested question about
    the optimizer, not a result about the objective.
  - `MOVES` if, in both versions, some extended strength changes the partition and lowers the fitted pi-weighted pair MI below its
    value at 8. Only then is R2 considered: card FA_FRC_R2 (failure, change, controls bracketed at the new spend, stopping rule)
    committed before any fit, its reading labelled post hoc (made after the registered MODEL_SELECT reading of PRC_S2), and the
    registered and R1 labels never overwritten.
  - Prediction (coordinator, before the run): `SATURATED`, probability about 0.6, because at 4 and 8 no tilt from strength / 1 to
    strength / 64 lowered the exact objective and the smallest tilt grows with the strength.
- **00:49:21-00:49:45Z** Probe units FRC_PROBE_STRENGTH_V0_V1 / V1_V1 (8 CPU-s each; no label file, no MODEL_SELECT). Version 0:
  one partition at all ten strengths 0.25-128 (15.6% re-routed, fitted pi pair MI 0.03322 against the twin's 0.03568), so
  `SATURATED`. Version 1: one partition from 0.25 to 8 (13.8%, MI 0.03273), and a new one from 16 to 128 (39.3% re-routed, of which
  income 18.0% and occupation 29.5%, MI 0.03224), so `MOVES`. **By the rule registered at 00:44Z, no revision R2**: version 0 is
  saturated and a one-version point cannot enter the pooled comparisons. The prediction (`SATURATED`) held for version 0, not for
  version 1. Descriptive: version 1's extra move re-routes 2.8 times as many people for a further fitted-MI reduction of 0.0005
  nats. The S3 budget point stays untested; with the registered optimizer it is unreachable in version 0.
- **00:49:20-00:49:39Z** Four label-reading units refused by the access gate (deviation 9): ER1 had edited two registration files.
  **00:52:31Z** ER1 text moved verbatim to RULES_ER1.md / .json; registration files restored to the receipt bytes; frc.test_rules
  18/18.
- **00:53-01:07Z** Reruns under new unit ids: reader custody S0/S1 (`_V2`, bitwise; 10 readers), FIT-fraction grid FRC1_LC (`_V2`).
  Classifier stability (FRC1_ROB): every comparison `ROBUSTNESS_NOT_CONFIRMED`; refit `REFIT_NOT_GENUINELY_EFFECTIVE` (strict rule
  kept); descriptive breakdown frc.refitcheck. Deployment lane DEPLOY_V1: 6 packages bitwise, 19/19 refusals each. Verifier's
  second hand-back (00:58:37Z, no numerical FAIL) saved verbatim. Documents drafted.
- **01:03-01:12:51Z** Statistical reviewer's closing review (new agent; no label read). Required headline adopted verbatim. All
  corrections C1-C11 applied to RESEARCH_DECISION.md, ADVISOR_BRIEF.md, HANDOFF.md, START_HERE.md, TABLES.md (TWIN comparison
  line; per-version columns, already filled at 01:07Z) and deviation 9 (rules hash). The coordinator's ablation descriptions had
  been reversed in the drafts (ABL_LOC = pair weight 0, per-recipient terms only; ABL_PAIR = local weights 0, pair term only).
  C12 (the reviewer's own erratum to the frozen ER1 text) is carried in HANDOFF.md. Wording rulings followed: the refit label is
  stated without softening, no "efficiency edge", the interpolations are always labelled "never gating", and "explained the lost
  margin" is replaced by the point-ratio wording.
- **01:08:46-01:09:19Z** Same-device backup B1 at tip 628937bb7 (FRC_CUSTODY_B1_V1), verified from the backup alone (both
  interpreter passes all_ok). YOTUO absent at 00:59:11Z and 01:20:41Z; off-device pending; nothing mounted or forced.
  **01:12Z** Verifier resumed for V8 (independent restore and deployment replay from the backup).
