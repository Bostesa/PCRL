# Privacy-first per-context selector: 2018 development decision

**Status: `NOT_ESTABLISHED`.** No registered decision label was earned:

| Label | Result | Primary clauses passed |
|---|---|---|
| ARM_D_MEETS_CRITERIA | false | 4/10 |
| ARM_R_MEETS_CRITERIA | false | 3/10 |
| RANDOMIZATION_ADDS | false | 4/10 |
| LEAD_REPRODUCED | false | — |

Registered statement, which applies to every number here: "These results may meet the new predeclared numerical criteria, but remain exploratory development evidence because the assessment data informed this study's design. They cannot provide independent confirmation." In this study, no new numerical criterion was met.

**Pins.**

- Lock: `SELECTION_LOCK.json`, SHA-256 `d89cfe1bfb3c625ca279bfe256b884f482371ad510aa0892d53026538fc0bc7a`, commit `7fcfaf1a8a70b6fa93b47d8f826ea504c1fe08ed`. It was pushed and remote-verified at 00:02:15Z.
- Outer unlock: 00:02:45Z. The outer role was scored once, for the whole panel, and the controls were re-scored within this study.
- Bootstrap: 10,000 common household draws with seed 20260926.
- Family critical values (Bonferroni): primary 30 endpoints, z = 3.144; secondary 100, z = 3.481; capability 4, z = 2.498.

## Answer to the question

1. **Can per-context selection for privacy reduce coalition SEX recovery while keeping task near D17? No new evidence that it can.**
   - D4 reached the same measured AB/SEX recovery as the task-selected control DET_SEL4: −.00363 unweighted and −.00340 PWGTP versus D17, identical to five decimals.
   - That value is not a graded reduction. It is the H-only floor: on both anchors where D4 differs from D17, the fresh AB/SEX attacker chose the ignore-channel H route (see `PRELOCK_NOTES.md`, written before the lock).
   - The intervals allow no improvement: [−.0083, +.0011] unweighted and [−.0088, +.0020] PWGTP.
   - D4 also paid task **+.00264 unweighted / +.00275 PWGTP** in point estimate, above the +.001 cap; the intervals are [−.0010, +.0063] and [−.0012, +.0067].
   - By comparison, DET_SEL4's task was −.00017 / +.00041. Privacy-first selection bought no measured privacy over task-first selection and gave up task.
2. **Does randomization over the same policies add anything? Not on the target endpoint.**
   - On AB/SEX, R4 versus D4 is an **exact zero on every anchor**. The two either share the H-only route (anchors 0 and 2) or are both D17 (anchor 1).
   - The registered −.002 target is therefore unreachable, and its clause fails with a zero-width interval.
   - In point estimates, R4 has lower task than D4 (−.0017 / −.0013) and lower AB/RAC1P recovery (−.0019 / −.0018). Both are unresolved.
   - Relative to D17, R4's task is +.0009 / +.0015, so the PWGTP point estimate is above the cap.
3. **The lead.**
   - The post-hoc DET_SEL4 comparison that motivated this study is reproduced exactly by re-scoring: AB/SEX −.00363 / −.00340 and task −.00017 / +.00041.
   - The study now shows **why** it looked favourable. DET_SEL4 was the release where the token-using coalition attacker lost to the H-only attacker on inner_selection. The −.0036 is D17's measured gain over H, not a reduction produced by the selector.
   - Any release reaching the same route switch scores identically. A privacy-first choice cannot improve on it under this audit.

## Fixed-bank versus fresh attackers

| Anchor | t(D4) | t(R4) | Fresh AB/SEX route (D4, R4) |
|---|---|---|---|
| 0 | .00185 | .00365 | H-only for both |
| 1 | 0 (D4 = D17) | 0 (R4 = D17) | D17's own route |
| 2 | .00005 | .00025 | H-only (B view) for both |

- **The frozen bank rewards randomization; the fresh audit does not.** R's fixed-bank gain over D (+.0018 on anchor 0) did not appear against attackers refit on each release. This agrees with the math review's warning that frozen attackers overstate what mixing buys.
- **The full nested mixture (NM4PF, secondary) was worse.**
  - η = 0 on every anchor: it is a stochastic T32 kernel, not a policy mixture.
  - AB/SEX versus D17: +.0008 / −.0008. Against D4: +.0045 / +.0026 (intervals [−.0007, +.0096] and [−.0025, +.0078]). On anchor 1 its fresh AB/SEX recovery rose (+.0055 U).
  - Task versus D17: +.0028 / +.0024.
  - Its fixed-bank task gain (about −.007 on the fitted decoder) did not survive the fresh utility probe.

## Within-32-state decision variation (not just token differences)

The table uses `DECISION_VARIATION_INNER.json` (coefficient_split) and `DECISION_VARIATION_OUTER.json` (outer rows, label-free). Each cell gives unweighted / PWGTP values.

| Release | Anchor | TV to D17 (coef) | Households with a changed token (coef) | Within-T32 V (coef) | States varying | Within-T32 V (outer) | Outer households in varying states |
|---|---|---|---|---|---|---|---|
| D4 | 0 | .189 / .200 | 859 | .280 / .291 | 16 | .273 / .297 | 1,309 |
| D4 | 1 | 0 (= D17) | 0 | 0 | 0 | 0 | 0 |
| D4 | 2 | .193 / .182 | 891 | .288 / .272 | 18 | .302 / .279 | 1,496 |
| R4 | 0 | .455 / .476 | 2,595 | .374 / .368 | 32 | .355 / .350 | 2,409 |
| R4 | 2 | .173 / .166 | 1,699 | .274 / .262 | 32 | .277 / .264 | 2,371 |
| DET_SEL4 | 0 | .401 / .426 | 1,639 | .300 / .298 | 16 | .278 / .278 | 1,309 |
| DET_SEL4 | 2 | .151 / .143 | 731 | .245 / .231 | 18 | .249 / .234 | 1,496 |
| NM4PF | 0–2 | .67–.82 | 2,746–3,109 | 0 (T32 kernel) | 0 | 0 | 0 |

Reading the table:

- D4's decisions do differ within T32 states on anchors 0 and 2.
- About 19% of people (by TV) receive a different token from D17 there.
- The within-state spread V is about .28, which is comparable to DET_SEL4's.
- R4 additionally mixes in every state.
- Anchor 1 is D17 for every selector and for R4.

## Registered predictions (PROTOCOL §9)

| # | Prediction | P | Outcome |
|---|---|---|---|
| 1 | D beats D17 on AB/SEX by ≥ .002 within the cap (point estimates) | .40 | **False.** AB/SEX −.0036 / −.0034 clears .002, but task +.0026 / +.0028 is above the +.001 cap. |
| 1b | ARM_D_MEETS_CRITERIA | .04 | False (4/10) |
| 2 | R beats D (R-vs-D AB/SEX ≤ −.001 in both weightings; R task within the cap) | .15 | **False** (exact 0) |
| 2b | RANDOMIZATION_ADDS | .01 | False |
| 3 | D4 varies within T32 states on at least one anchor | .90 | **True** (anchors 0 and 2) |
| 3b | … on all three anchors | .55 | False (anchor 1 = D17) |
| 4 | D4 ≠ DET_SEL4 on at least one anchor | .85 | True (anchors 0 and 2) |
| 5 | R4 one-hot on every anchor | .15 | False |
| 6 | Fixed-bank t(R4) − t(D4) > 1e-9 on at least one anchor | .85 | True (.0018 on anchor 0; .0002 on anchor 2) |
| 7 | LEAD_REPRODUCED | .30 | False |
| 8 | NM4PF AB/SEX beats D4 by ≥ .001 in both weightings | .15 | False (it is worse in point) |

## Texas 2018 power table

**Not triggered.** PROTOCOL §10 required LEAD_REPRODUCED, and it is false because the task cost exceeds the cap in both weightings.

- The planning values registered in amendment A2 stay unused.
- `CONFIRMATION_PLAN.md` was not completed.
- No Texas file was downloaded or opened.

## Deviations and caveats

- **Amendment A1** (before any audit): the LP arms received the same lexicographic task tie rule as D. It changed only anchor 1, where R4 and NM4PF had returned costly t = 0 vertices. The first solves are kept in `SOLVE_REPORTS_SUPERSEDED_A1.json`.
- **Amendment A2** (before any audit result): Texas planning values.
- **The audit's AB/SEX measurement saturates at the H-only route.** This is a property of the inherited slate: the route that is best on inner_selection is scored, and the ignore-channel ancestor is always a candidate. The resolution of this study's target endpoint is bounded by it.
- **Fixed basis.** Every arm used the NM4_U closing bank, which was never refit on the arm's own law. The audit's attackers were refit on each release, and they are the only privacy evidence.
- **Survey design.** The intervals condition on the fitted, selected objects. They do not cover the adaptive research history or the ACS survey design.
- **Development data.** The 2018 outer role was opened for the second time, having been opened by the predecessor study. This is development evidence on repeatedly used data.

## Independent verification

**Status: CONFIRMED.** The verifier worked from `agents/verifier/VERIFIER_BRIEF.md` with fresh context, using separate code in `verify_independent.py` that imports nothing from the study or SC/TAC inference. Details are in `INDEPENDENT_VERIFICATION.json`.

- **A. Solves.** 21/21 checks pass.
  - D4, D1, TASK_SEL4 and DET_SEL4 were re-derived by the verifier's own enumeration and match bit-exact.
  - R4, R1 and NM4PF were re-solved by interior point and by an explicit dual. t matches to 1.1e-15 and task to 7.8e-16.
- **B. Outer.** All 134 point estimates were reproduced, with a maximum difference of 8.7e-19. Bootstrap SEs, computed with an independent seed, agree within 2.6%. All four labels agree.
  - One clause flips with the seed: primary R_vs_D A/SEX PWGTP guard. Its upper bound is within 1.5e-5 of +.001; it fails at the locked seed and passes at the verifier's.
  - R_vs_D therefore passes 4/10 clauses at the locked seed and 5/10 at the verifier's. RANDOMIZATION_ADDS is false either way.
- **C. Custody.** The lock commit is on origin with the pinned bytes. The order is lock (00:02:15) < unlock (00:02:45) < original restore (00:03:09) < every outer completion. The inner panel and unit pins match.
