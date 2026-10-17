# Decision log (UTC)

Campaign: PCRL structural-mechanism campaign, branch `research/pcrl-structural-mechanism-pilot-v1`. It started from `da5f5fe89`, the corrected mechanism-design record that contains the ACS replication and frozen-hybrid code. Times come from the queue ledger, git commit times and agent completion notices.

| Time | Event |
|---|---|
| 2026-10-16 23:38:57 | Campaign start: brief, memory and AGENTS.md search. The repository has no `AGENTS.md` and no `CLAUDE.md`. |
| 23:4x–00:5x | Read the corrected mechanism-design package (`c9709a2` content, `CORRECTIONS.md`, `PROBLEM_SPECIFICATION.md`, `EVIDENCE_PACK.md`, `CLOSED_LANES.md`, `CLOSEST_PRIOR_WORK.md`, the prepared pilot) and the ACS replication and hybrid code and manifests. |
| ~00:50 | Isolated worktree and branch created from `research/pcrl-mechanism-design-correction-v1` (`da5f5fe89`). Main, the manuscript branch and every earlier branch are untouched. |
| ~00:51 | Draft v0 of `SERVICE_AND_THREAT_MODEL.md` and the designer brief written. |
| ~00:52 | Four agents launched in parallel: designer 1, designer 2, the recipient-use and adversary reviewer, and the closest-work reviewer (phase 1). The designers received identical briefs. |
| 00:53:52 | Queue opened: 12 CPU-hours, 8 hours elapsed, 2 slots, single-threaded, counted from 23:38:57. |
| 00:55:33 | `SMP_RESTORE_V1`: 34 source files copied with hash checks. `SMP_RESTORE_V2` added the published FARE and FAGTB outputs, for 38 files. |
| ~01:00 | `SMP_SUPPORT_FACTS_V1`, label-free: about 90% of people have a unique encoded input in their role, and about 17% of held-out inputs occur in MODEL_FIT. |
| 00:59:54 | Commit `2d96c471e`: selection criteria fixed. Both designer folders were empty at this commit; checked at 00:58:58 and 01:00:51. |
| ~01:11 | Adversary review of the service draft: 6 must-fix, 5 should-fix and 4 notes, all applied. Service model v1. |
| ~01:13 | Designer 1 delivered SCM (`6bc0617b…`). The closest-work reviewer delivered the phase-1 field matrix. |
| 01:14:52 | Commit `16d41c0c9`: service model v1 and designer 1's proposal as received. |
| 01:1x–01:24 | The session was interrupted twice: a safety classifier withheld parts of two coordinator responses. The user's addendum then confirmed the scope (statistical privacy research on public Census records, accurate variable names) and asked to resume in place. No work was lost, and nothing was disguised or renamed. |
| 01:24 | State verified on resumption: git status, queue ledger, agents and processes. |
| 01:26:42 | Commit `8131067b3`: the composition statement made precise as the user asked (exact linear guardedness composes; nonlinear and approximate guarantees do not), and the lookup reader frozen and implemented for every arm. |
| ~01:31 | Designer 2 delivered SCAR (`74b42067…`). Commit `cd67c17cc`. |
| 01:32 | Both proposals sent to the closest-work reviewer (phase 2) and the adversary reviewer (threat check). |
| 01:36:04 | Commit `0a2d4f9b0`: shared merge engine, data-free; brute-force identity checks pass (maximum error 6.4e-17). |
| ~01:38 | Adversary threat check returned. Neither design has a hidden output. SERIOUS for both: pre-fit proxies biased by MODEL_FIT SEX (X1), and a noisy permuted-SEX control (X2). SERIOUS for SCAR: cells too small for readers (S1) and an ungated within-SEX guard (S2). SCM judged safer. |
| ~01:39 | Closest-work phase 2 returned: `ENGINEERING_COMBINATION_NO_NEW_INGREDIENT` for both. Both overlap the closed frozen-bank-channel lane in substance; an attribution arm is required if the family is piloted; SCM narrowly more defensible. |
| 01:41:54 | Commit `91f203db4`: `SELECTION_DECISION.md` = `NO_CANDIDATE_SURVIVED`. Both proposals fail S3, and the record first stated an S2 failure "in substance" as well. The verifier later found that overstated and the wording was corrected: the outcome rests on S3 alone, and S2 is a post-reading ruling. No pilot, no fits, no readers, no MODEL_SELECT labels, no DEV_REPORT truth. |
| ~01:42 | Independent verifier launched on the decision and the artifacts. |

**Disclosures.**
- The designer brief was committed about eight minutes after the designers were launched; its content was written before launch.
- The coordinator built the shared merge engine and teacher utilities while the reviews were running. This was data-free preparation in case a variant was selected. They are delivered as tested, unused code.
- The adversary reviewer was given the lookup-reader idea in its own review (S4) and the coordinator adopted it. The designers did not see the frozen lookup specification before proposing; SCAR's proposal cites the v1 service model, which by then contained it.
