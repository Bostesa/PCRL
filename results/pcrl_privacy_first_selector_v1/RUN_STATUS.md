# Privacy-first selector v1: run status

- Execution start: 2026-09-24, about 23:10Z. It was not logged to the second. Worktree creation was the first write.
- Branch `research/pcrl-privacy-first-selector-v1`, created at `537e74a44bef44ea86882a19769f3c4eb53d410c`.
- Worktree: sparse, at `.worktrees/pcrl-privacy-first-selector-v1`. The Mac has about 4.6 GiB free disk. The sanitized inputs are a gitignored symlink, as in the predecessor.
- Outcome access at registration: none. This study has read no 2018 outer-role label, and no release has been solved.
- Ceilings: US$50 or 20 instance-hours, and 2026-09-25T19:30Z.

## Log
- 23:34:05Z: registration committed and pushed as `5f91117`, remote-verified, before any solve. Outcome access: none.
- 23:35Z: early manuscript-owner notice written: `.git/pcrl_privacy_first_selector_v1/TO_MANUSCRIPT_PRIVACY_FIRST_SELECTOR_START.json`.
- 23:40–23:43Z: code written and pushed as launch commit `9bb77ac`. It covers solve, host, lockgate, outer and cloud, and 11 synthetic tests pass. AWS preflight passed.
- 23:44:19Z: launched `i-07d9bc2619c9f5b11` (c7i.8xlarge, $1.428/h), with security group `sg-07748c7024ec577d4` (zero ingress) and tags `Project=pcrl` and `Study=pcrl_privacy_first_selector_v1`. Watchdog 2026-09-25T09:00Z; hard stop 09:30Z.
- 23:45:41Z: host READY. Code pinned at 9bb77ac; 18 inputs restored and SHA-verified. The host test suite passed 137/137.
- 23:46:22Z: the 9 predecessor archives (a{0,1,2}_{bank,NM4_U,DET_SEL4}) were restored and verified against the MODEL_MANIFEST pins.
- 23:49:33Z: `BASIS_MANIFEST.json` written before any solve, and committed as 0ded779.
- ~23:50Z: first fixed-bank solve on coefficient_split.
- ~23:51Z: amendment A1 committed as bf040be (a lexicographic task tie rule for the LP arms), then the re-solve. The first solve is kept as `SOLVE_REPORTS_SUPERSEDED_A1.json`.
- 23:53Z: decision-variation step started, along with the three inner audits, in parallel.
- 23:58Z: amendment A2 committed as 5576bea (Texas planning values). No audit output had been read.
- 23:54–23:59Z: variation and the three audits complete. Independent verification Part A (separate code) passed 21/21 (d2c1ac5).
- ~00:00Z: inner report and pre-lock route notes committed as 3b40815. For D4, R4 and DET_SEL4 the AB/SEX route is the H-only one.
- 00:02:15Z: `SELECTION_LOCK.json` (SHA-256 d89cfe1b…) committed and pushed as 7fcfaf1, and verified on the remote.
- 00:02:45Z: `OUTER_UNLOCK.json` written from the separate /opt/pcrl/lockcheck checkout at 7fcfaf1 after remote verification.
- 00:03:09Z: outer originals restored behind the gate and SHA-verified. Outer scoring started, once, from the lockcheck checkout.
- ~00:04–00:06Z: outer scoring completed for anchors 0–2, followed by the outer decision-variation report.
- ~00:07Z: the locked assessment ran once, with bootstrap seed 20260926 and 10,000 draws. **Status: NOT_ESTABLISHED.** No registered label was earned: D passed 4/10 clauses, R 3/10 and R_vs_D 4/10, and LEAD_REPRODUCED is false.
- 00:07Z: private archives uploaded and read back (see COST_AND_CLOSEOUT.md).
- ~00:12Z: independent verification CONFIRMED with separate code: Part A 21/21; Part B reproduced all 134 estimates, with one seed-sensitive guard clause and no label change; Part C custody passed.
- 00:14:35Z: instance terminated and study SG deleted; 0 study resources remain. Cost ≈ $0.72.
- **Study CLOSED. Status NOT_ESTABLISHED.** See RESEARCH_DECISION.md and PAPER_ADDENDUM.md.
