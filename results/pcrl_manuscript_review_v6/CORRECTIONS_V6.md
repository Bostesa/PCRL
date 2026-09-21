# CORRECTIONS_V6 — additive to CORRECTIONS_V5.md and the historical study records

Nothing here edits a historical study file. Study 6 evidence: `research/pcrl-utility-extension-aws-v1`,
evidence commit `ba531ab424c593fe8573cd320d2bcef379907cf1`, read at handoff `0517c06a7`. Every number is
checked in `STUDY6_VERIFICATION.json` (118 checks, all agreeing; the pilot's pass/fail legs were re-derived
from the stored per-anchor increments, not read from the study's flags).

| # | Said | Where | Correction |
|---|---|---|---|
| D1 | "A sixth, utility-first programme has not produced an outcome and none is reported" | manuscript v5 title block, §9, abstract | Stale. Tiers 0–2 ran on 2026-09-20; the pilot closed negative and is integrated in v6 §9. Tiers 3–4 were never triggered. |
| D2 | "The best protection achieved was +0.0016 nats (worst endpoint, `X_r2_L2_b030`)" | Study 6 `RESEARCH_DECISION.md` §1, `PILOT_GATE.md` | That is the **unweighted** worst-endpoint mean. Person-weighted the same configuration reaches **+0.0021**. Both miss the .001 screen, so the decision is unchanged, but the headline figure is one weighting. Both are now printed. |
| D3 | "the closest are all A0-channel releases" | Study 6 `RESEARCH_DECISION.md` §2 | Correct as stated, and worth spelling out: three of the eight closest are Track N `N_A0_*_none` continuations (bitwise A0), not `E_A0_*` projections. Verified by recount. |
| D4 | Cross-platform audit reproduction | Study 6 `VALIDATION.md` | Reported by Terminal 1 as the one check that did not pass (max 0.0031 against a 0.003 tolerance). Carried into the manuscript limitations rather than absorbed: Study 6 numbers must not be differenced against earlier studies' published audit values at a finer precision than ~0.003 nats. |
| D5 | Tier counting | Study 6 `TIER_STATUS.md` | Confirmed cumulative, not additive-on-top: the 42 pilot units are the first 42 of the 135-slot expanded plan, and the 93 unrun Tier-3 slots are the remainder. No count in the manuscript treats the pilot and the expansion as disjoint totals. |
| D6 | "T3/T4 NOT STARTED" | Study 6 gate files | Rendered in the manuscript as **not triggered**, with the reason (gate rule after a failed pilot) and the explicit statement that this is absent evidence for an unrun design, not missing evidence for a claimed one. No claim about `r ∈ {4,8}`, a nominee comparison or a 2017 transport of this mechanism appears anywhere. |

Unchanged Study 6 conclusions (not re-litigated): mechanism works as specified; capability improved on the
proxy; disclosure screen failed at every configuration; no coalition-specific benefit; added stress strength
not established; candidate for confirmation not reached.

## Carried unchanged from v5
C1–C10 of `CORRECTIONS_V5.md`, including the family-X attribution of the A0 k=2 race effect (C3), the
unweighted-only LEACE-on-J proximity (C4), and the Study 3 repair table figure of 8 of 48 (C6).

## Added 2026-09-20 (after the v6 publication commit)

| # | Said | Where | Correction |
|---|---|---|---|
| D7 | The Study 6 inseparability "appeared on a proxy that never saw a residential label, which removes the usual 'the utility target leaked' explanation" | v6 `CONTRIBUTION_ASSESSMENT.md`; the v6 Conclusion, in weaker form | **Withdrawn — the inference is invalid.** Excluding residence labels keeps the *reserved task* out of fitting; it does not make the objective innocuous. The target is the residual of A0, the unprotected channel with the largest sensitive recovery in this work (+0.032 A/SEX, +0.051 A/RAC1P over H), so improving on the proxy can be achieved by carrying sensitive-correlated directions. "The objective itself rewarded sensitive information" remains a live explanation and the study cannot exclude it. Conclusion and §9 rewritten; this is a correction, not a new finding. |
| D8 | Study 6 METHOD §1: untouched-J predictor candidates "are part of every comparison" | `pcrl_utility_extension_v1/METHOD.md` §1, reported in v6 §9 | Confirmed against source by the successor study: the slate routed H-only ancestors and audited `ref_J` as a separate condition; no in-slate predictor read `[H_A, Z_J]` while ignoring R. Disclosed in v6 §9 and §10. The fix can only raise measured increments, so no historical result becomes a win under it and nothing is re-derived. |
| D9 | Study 6's training penalty measures incremental disclosure | `pcrl_utility_extension_v1/METHOD.md` §3, implicit in v6 §9 | It mixes frozen-baseline approximation slack with incremental disclosure: a constant extension registers a positive gain, and the statistic is one-sided. Affects the interpretation of β, not the audited leakage. Magnitude in the study is **unmeasured**, not estimated. Disclosed in v6 §9 and §10. |
| D10 | "A .001-nat confirmation is not demonstrable at this sample size **for any mechanism**" | stochastic study `gates/G0.json`, `REPORT.md` §5, and its handoff instruction for any write-up | Not adopted as written. The arithmetic is correct, but the half-width is `z × SE` and the paired SE depends on the dispersion of per-household loss *differences*, which is mechanism-dependent. Counterexample in committed evidence: 60 of 320 sensitive rows in Study 5's family P have SE exactly 0 (releases bitwise identical to J). Defensible restatement and the full review are in `G0_REVIEW.md`. Recorded as a correction to an interpretation, **not** as a scientific result. |
