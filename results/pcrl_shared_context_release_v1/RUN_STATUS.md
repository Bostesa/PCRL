# Shared-context release v1: run status

- Execution start: 2026-09-24T17:26:27Z (first discovery command).
- Binding ceiling: 20 h elapsed -> 2026-09-25T13:26Z (earlier than the 2026-09-27T03:59Z calendar cutoff). Budget ceiling US$50.
- Branch `research/pcrl-shared-context-release-v1` created at predecessor tip `2ce5d171f079f759eb755a21ab415765ea234479`; sparse worktree `.worktrees/pcrl-shared-context-release-v1` (full 7.8 GB tracked tree does not fit in 7.1 GiB free disk; first full checkout attempt failed with ENOSPC and git removed the partial tree).
- Host: macOS, 24 GiB RAM under pressure, 14 cores, ~7.1 GiB free disk. Unrelated live AWS instances `s1-aws-s1` (g7e.2xlarge) and `p0-pilot` (g5.2xlarge) are outside scope and untouched.
- Outcome-access status: no 2018 outer-assessment label has been read by this study.

## Log
- 17:26Z start; discovery; sparse worktree; symlink to predecessor sanitized inputs (gitignored).
- 17:30–17:46Z parallel agents: support census (label-blind, reproduces DATA_ROLE_COUNTS exactly; T0 codes stored), code audit (A1–A3, A5 confirmed; A4 partly), fit-pipeline map, math review (3 design-changing findings -> Amendment M1), confirmation plan (running).
- ~17:44Z implementers launched: method (channel/policies/contexts/fit_nm/release), baselines (rd/adv/poscontrol + BASELINE_MATCHING/AUDIT_CONTRACT), infra (laws/runner/audit_panel/cloud/stage/archive). Time box 21:15Z; method implementer finished 18:07Z.
- 17:48:02Z pre-fit registration committed and pushed: ba391494379ece6b21b4dc8952495a5ec35a5a4d (remote verified). Outcome access: none.
- Early manuscript-owner notice: `.git/pcrl_week_finish_v2/evaluation/TO_MANUSCRIPT_SHARED_CONTEXT_START.json`.

## Agent board
| Role | Owner | Deliverable | Status |
|---|---|---|---|
| Coordinator | main session | protocol, manifest, lock, decision | active |
| Data/support | agent | SUPPORT_CENSUS, DATA_CONTRACT | census done; DATA_CONTRACT pending (coordinator) |
| Math reviewer | agent | MATH_REVIEW | done |
| Method implementer | agent | NM code, METHOD | running |
| Baseline/audit owner | agent | RD/ADV/poscontrol, BASELINE_MATCHING, AUDIT_CONTRACT | running |
| Infrastructure | agent | runner/cloud/audit integration | running |
| Independent verifier | agent (later, fresh context) | INDEPENDENT_VERIFICATION, REVIEW | not started |
- Timestamp correction (18:08Z): earlier entries in RUN_STATUS/DESIGN_SPEC/RESPONSE carried estimated times that ran ahead of the clock; corrected to git commit times (registration 17:48:02Z, confirmation plan 17:59:21Z). No content change.
- 18:53:41Z launched i-0732026d054f6d3ee (c7i.8xlarge, $1.428/h, SG sg-01e1c3010494c470c zero ingress, tags Project=pcrl Study=pcrl_shared_context_release_v1) at launch commit 8ac75775e0c35a1746c6870a91037d18aff48e25; watchdog 2026-09-25T09:00Z, hard stop 09:30Z.
- 18:56Z runner started (16 workers, 58 units). 19:06Z: banks x3, decide_k, POS x6, J_inner x3 complete, no failures.
- K decision: nm4_K = 4 (no anchor failed the registered support rule; K=2 fallback not triggered).
- Positive controls (revealing channel, detection = H-minus-leak >= .01 nats in both weightings on inner_check): AB/SEX and AB/RAC1P detected on all three anchors (6/6). The predecessor ran only a0 AB/RAC1P.
- 19:45Z: all 55 fit/control units complete with zero technical failures; a1_inner_audit complete; a0/a2 inner audits running. Structural capacity report (coefficient/inner_selection only) built on host: /opt/pcrl/reports_out/ENCODER_CAPACITY.json.
  - U-form NM (NM1_U, NM4_U) never produced within-T32 variation at any round (LP declined the richer columns); final: a0 selected a round (T32 function), a1/a2 WITNESS_FALLBACK (every round failed attackers refit on itself).
  - P-form NM used within-T32 variation on 5/6 unit-anchors; selected non-D17 rounds except a1_NM1_P (WITNESS_SELECTED).
  - RD_PRIV: D17 by its own task rule on all anchors; ADV task-selected units: D17 by rule; ADV privacy-selected units: WITNESS_FALLBACK (no epoch within the D17+.001 inner task cap) at least on a1/a2 (a0 per report).
  - RD_TASK: non-D17 only on a2.
- 19:54Z: inner selection + SELECTION_LOCK written on host from the three inner panels; both slots DIAGNOSTIC_ONLY (U: NM1_U, P: NM4_P); primary 40 endpoints (comparators that are exact D17 on all anchors merged), z=3.2272; secondary 120; capability 4; POS 6/6 pinned detected. Lock sha256 d995484103cfbc9ca9f1411a2f89d6f5d9ee1f46e1bbdce1c8d0d82d20cadbfc. Outer not yet opened.
- 19:56:26Z OUTER_UNLOCK verified (lock commit e5d555a on origin). 19:58Z restore-outer + run-outer (frozen routes, no refit/selection on outer) + assess. Outer opened once. Results: U 8/20 primary clauses (DIAGNOSTIC_ONLY NM1_U), P 0/20 (DIAGNOSTIC_ONLY NM4_P); capability 4/4; no route passed.
