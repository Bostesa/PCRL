# Advisor brief: is the finite-routing optimizer adequate, and does fixing it help?

Campaign `pcrl_finite_solver_adequacy_v1`, 2026-10-21 (start 02:22:58Z). Florida ACS 2019 development cohort (admitted public-use
PUMS records).
- **Data used.** MODEL_FIT only, through its cached out-of-fold teacher statistics (derived from training labels) and nested refits.
  MODEL_SELECT was not opened.
- **Release.** Six input fields; an income recipient and an occupation recipient, each receiving only its own probability vector and
  decision.
- **Recovery.** The AUC of classifiers ("readers") that estimate the recorded SEX attribute from what the recipients receive.

**Result in one line:** the rule outcome is `OPTIMIZER_REPAIR_CLOSED`, and no pilot outcome class was computed.
- A revised solver finds better solutions of the stated objective.
- Their held-out recovery gain did not pass the pre-registered test, so the conditional pilot did not run.
- No fitted method is claimed.

The required headline is in `RESEARCH_DECISION.md`.

## The seven questions

**1. Did the seven-tilt proposal search miss demonstrably better deployable solutions?** Yes, of its own objective. The original Lloyd
step tries only tilts λ/2^j, so improving windows that contain no such tilt are invisible:
- on registered invented laws (law D: a feasible partition with exact pair AUC 0.5803 against the stalled 0.6882);
- on the real cached statistics (version 1, strength 8: an improving partition in a tilt window 4.8e-6 wide near 0.1875, between the
  tried 0.125 and 0.25).

The revised solver SV1 searches the exact tilt lines of the refreshed and the current routing book with best improvement.
- It lowers PRC's stated objective at every registered strength in both versions. For HQ it does so in 11 of 12 cells.
- Every state it reaches is reproduced by the deployed six-field rule.
- It is not optimal:
  - at version 1, strength 8 its chain ends at 2.5797, above the 2.5732 partition the audit found;
  - on the exact laws it stays above the best same-class reference in 10 of 30 distinct cells (largest gap 0.0102).

**2. Did the objective / probability inconsistencies matter?** Under the rules they are documented approximations, not defects. Route
D classifies them; it does not measure whether they change decisions. The seven-tilt search does change the solutions (questions 1
and 3).
- **Shrunk routing score.** Along the strength-8 tilt line, the routing's shrunk, linearized dependence score is anti-ranked against
  the exact objective's dependence term (Spearman about -0.995). This is consistent with large moves being rejected. The exact
  acceptance test keeps every step valid.
- **Codeword refresh.** The cell-mean codeword refresh never raised the objective on the real chains.
- **Emitted tables.** They differ from the search prototypes by at most 0.0095 (PRC). Their training cross-entropy is lower or
  equal.

**3. Did a revised solver improve actual usefulness or recovery?** Not by the registered standard.
- The comparison point is the "bracket": the cheapest original point whose MODEL_FIT nested held-out spend is at least SV1's spend
  minus 0.0005.
- Against it, SV1's selected point has a held-out pair LOOKUP AUC 0.0018 / 0.0038 lower. Its spend is lower in version 0 and slightly
  higher in version 1 (-0.0006 / +0.0003).
- Pooled, the difference is -0.0028 with se 0.0019 (2-SE interval -0.0065 to +0.0009). The per-version point conditions held; the
  pre-registered pooled 2-SE condition failed.
- On the exact laws, better optimization helped only in the witness law built to admit a gain. In two other laws it traded task
  accuracy for recovery, and it went beyond the utility limits (infeasible) at strength 2 and above: the objective has no utility
  constraint.

**4. Did HQ improve equally?**
- SV1 lowered the task-score-only control's objective in 11 of 12 cells.
- In nested held-out data, revised HQ reaches lower recovery than original HQ (0.6225 / 0.6251 against 0.6262 / 0.6276), but at
  higher spend.
- Revised PRC sits 0.0016 / 0.0048 below revised HQ at lower spend.
  - Under the original solver this gap was larger (0.0035 / 0.0070), and in the previous campaign it did not carry to the
    MODEL_SELECT readers (benefit +0.0003, se 0.0051).
  - It is descriptive only, without SEs, on the objective's own reader type and not on MODEL_SELECT.
- Task scores identify the six-field support states for about 99% of people. Any PRC-HQ difference is therefore about search and
  parametrization, not about information that propensity adds. Question C is unanswered.

**5. Did the conditional pilot run?** No.
- Route S (source evidence) ended `S_PROXY_ONLY`: a better objective, with no recovery or task rationale that passes the rule.
- Route E's advance on invented laws rests on the witness law, and route S supersedes it.
- No implementation defect was found (route D).

The coordinator predicted this outcome, committing the prediction before any nested result existed. The nested units had already
started, and the source results were known.

**6. Method candidate, component, engineering result or bounded negative result?** None of the rules' pilot classes was computed,
because no pilot ran. The rule outcome is `OPTIMIZER_REPAIR_CLOSED`.
- Descriptively, the optimizer-repair hypothesis closes on MODEL_FIT, with the bound above. It was measured with the objective's own
  LOOKUP reader only.
- Verified engineering findings (descriptive, not the rules' `ENGINEERING_RESULT` class):
  - the previous campaign's P1 numbers reproduced bitwise from a rerun of its nested estimator, and were independently recomputed;
  - a tested, rule-consistent better solver of the stated objective;
  - exact references.

There is no method candidate and no component.

**7. What remains unknown?**
- Nothing is known about the MODEL_SELECT readers. No reader grid was built and MODEL_SELECT was not read, by rule.
- Whether propensity adds anything beyond the equally optimized HQ (question C).
- Whether a utility-constrained selection rule could turn better optimization into usefulness.
- Other solver variants. None was built after SV1, which is path-dependent and not optimal.
- Larger exact references.
- Whether any finite-cell routing can close the 0.018-0.023 pair-AUC gap to the target references that the previous campaign
  measured.

## Strongest evidence on each side

- **Favourable:**
  - The stall is real and diagnosed: the original search misses deployable improvements of its own objective.
  - SV1 finds deployable improvements, though on the source they fall short of the best partition the audit exhibited (2.5797
    against 2.5732 at version 1, strength 8). It is never worse on the exact laws.
  - Against the bracket, SV1's selected point has lower held-out pair recovery in point estimate in both versions (-0.0018 /
    -0.0038), and the per-version point conditions held.
  - The P1 replay is exact.
- **Adverse:**
  - The better objective is mostly not better recovery. The fitted linked-pair information falls by only 0.0002 nats, while SV1's
    state re-routes 21-24% of people relative to the twin (the original re-routes 14-16%).
  - The held-out gain is not resolved. Against the original S2 point it is only -0.0018 / -0.0003.
  - On the exact laws, stronger optimization can trade utility for the surrogate.
  - The equal-effort-per-step control (the finer candidate set with first improvement) ran into its 30-step cap while still
    improving. Under that cap it ended worse than the original on the source.
  - Route E rests on one constructed witness law.

## Not run

- The development-data pilot: revised PRC and HQ configurations, MODEL_SELECT evaluation, reader grid, matched controls, ablation
  and refits. It was not authorized by the frozen advancement rule.
- No deployment package was built, because no new release exists.

## Recommendation

- Keep the original finite-routing conclusions.
- Use SV1 if this objective is revisited, but put the utility limits into its selection rule before testing it again. A better
  optimum of the unconstrained objective is not a better release.
- The repaired task-only twin remains the finite baseline.

Resources, verification, custody and publication: see the end of `START_HERE.md`.
