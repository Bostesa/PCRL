# Factual addendum for the manuscript owner: privacy-first per-context selector (2018 development)

**Status:** `NOT_ESTABLISHED`. No registered criterion was met.

Required wording, verbatim, wherever any number below is used:

> "These results may meet the new predeclared numerical criteria, but remain exploratory development evidence because the assessment data informed this study's design. They cannot provide independent confirmation."

In this study no new criterion was met. The study was motivated by a post-hoc look at an outer role that had already been opened. It changes neither the completed 2016 prospective result nor the shared-context study's locked result.

## What was tested

- **Question.** Selecting existing policies per context *for privacy* (arm D: one policy per context), and randomizing over the same policies (arm R: probabilities per context).
- **Objective.** Both arms used one registered objective: maximize the worst bank-measured AB/SEX improvement over D17 across both weightings. Constraints: task within +.001 nats of D17 in both weightings, and every other protected comparison within +.001.
- **Solving.** D was solved exactly by enumerating all 625 assignments (gap 0). R was solved as an LP.
- **Reused objects.** The policy bank, contexts, frozen decoder, attacker bank, roles and D17 witness were all reused from the shared-context study (537e74a) and pinned by hash.
- **Scoring.** Fresh attackers were fitted per release by the inherited audit slate. The outer role was scored once, under a pushed and remote-verified lock.
- **Uncertainty.** 10,000 common household draws, with simultaneous Bonferroni bounds: z = 3.144 over 30 primary endpoints.

## Strongest favourable comparison

**D (and R) versus D17 on coalition SEX recovery:**

| Weighting | Contrast | 95% simultaneous interval |
|---|---|---|
| Unweighted | −.00363 | [−.0083, +.0011] |
| PWGTP | −.00340 | [−.0088, +.0020] |

This is a favourable point estimate only; the intervals allow no improvement.

Where to see it: it is identical to five decimals to the task-selected control DET_SEL4. Both releases reach the audit's H-only route, meaning the fresh coalition attacker using the token did no better than the attacker ignoring it. The number therefore measures D17's own coalition-SEX leakage over H. It is not a reduction the selector produced.

Second favourable comparison, randomization over selection (R versus D):

- task −.0017 unweighted / −.0013 PWGTP;
- coalition RAC1P recovery −.0019 / −.0018.

Both are unresolved.

## Strongest adverse comparison

**Privacy-first selection (D) versus task-first selection (DET_SEL4), on the same policies and basis:**

- Coalition SEX recovery is **exactly equal** (a zero-width interval), so the −.002 target is demonstrably not reached.
- Task is **+.0028 unweighted / +.0024 PWGTP** worse. The intervals are [−.0010, +.0067] and [−.0019, +.0066] (secondary family, z = 3.481).

Against D17, D's task cost is +.0026 / +.0028, above the +.001 cap in point estimate; the primary intervals are [−.0010, +.0063] and [−.0012, +.0067].

**Randomization added nothing on the target endpoint:** R versus D on coalition SEX recovery is an exact zero on every anchor.

The full nested mixture was worse again. Against D, its coalition SEX recovery was +.0045 / +.0026 (unresolved), and its task was +.0028 / +.0024 against D17.

## Decision variation (not only token changes)

- **Anchors 0 and 2.** D changes about 19% of tokens relative to D17 (TV, both weightings), affecting 859 and 891 coefficient households. Its decisions vary *within* 16 and 18 of the 32 T32 states: within-state V ≈ .28 on coefficient rows and .27–.30 on outer rows.
- **Anchor 1.** Every selector, and R, equals D17 exactly.
- **R** mixes in all 32 states on anchors 0 and 2.

## Why the lead did not survive

- The earlier post-hoc lead (DET_SEL4, coalition SEX −.0036 / −.0034) is reproduced exactly.
- It comes from a route switch in the audit, not from graded privacy. Any release whose token-using attacker loses to H scores identically.
- A privacy-first choice therefore cannot beat it under this audit. Here it only added task cost.
- The frozen-bank advantage of randomization (t: +.0018 on anchor 0) did not appear against refit attackers.

## Wording guardrails

**Say:**

- "a privacy-first per-context selector over existing policies did not improve measured coalition-SEX recovery beyond the task-selected control and cost task in point estimate";
- "randomization over the same policies added nothing on the target endpoint";
- "fitted-attacker recovery";
- "development evidence on repeatedly used 2018 households".

**Do not say:**

- "privacy guarantee";
- "reduces sex inference";
- "stochastic release is needed";
- "the selector outperforms D17";
- "confirmed";
- "label-free".

**Also:** do not cite DET_SEL4's or D's −.0036 as a privacy improvement without the H-route explanation.

**The Texas 2018 confirmation power table was not triggered.** It required LEAD_REPRODUCED, which failed on task. No Texas data were read.

## Pins

- **Branch:** `research/pcrl-privacy-first-selector-v1`.
- **Registration:** `5f91117`.
- **Lock:** commit `7fcfaf1a8a70b6fa93b47d8f826ea504c1fe08ed`; `SELECTION_LOCK.json` SHA-256 `d89cfe1bfb3c625ca279bfe256b884f482371ad510aa0892d53026538fc0bc7a`.
- **Results:** `INFERENCE.json`, `ENDPOINT_TABLE.json`, `FULL_RESULTS.csv`, `RESEARCH_DECISION.md`, `DECISION_VARIATION_{INNER,OUTER}.json`, `PRELOCK_NOTES.md`.
- **Independent verification:** `INDEPENDENT_VERIFICATION.json`. Separate code was used, and the result is CONFIRMED:
  - all point estimates are reproduced;
  - all labels agree;
  - custody is verified.
  - One guard clause (R vs D, A/SEX PWGTP) sits within 1.5e-5 of its threshold and flips with the bootstrap seed; no label depends on it.
