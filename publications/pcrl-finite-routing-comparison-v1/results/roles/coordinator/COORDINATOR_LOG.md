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
