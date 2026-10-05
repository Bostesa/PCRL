# Paper addendum: matched update strength and active local feedback (outcome-specific integration text)

**Scope.**
- Integration text for the combined paper; the manuscript branch is not edited.
- No method advantage goes in the abstract.

## Suggested paragraph

**Question.** The refreshed-critic study left a confound: the online-critic joint arm that protected best also applied about three times the protection push. We separated the two.

**Design.**
- Each encoder's protection push was set to a fixed fraction ρ ∈ {0.25, 0.75, 1.5} of its own task push.
- Three critic schedules were compared at equal strength: online, refreshed, and online with refit-matched effort.
- An AUC-driven controller redistributed a fixed protection budget between recipients.
- Joint and local arms, with and without feedback, were compared against a faithful raw-penalty arm and official LEACE and FARE baselines, on a new 3,796-row development partition.

**Findings.**
- **Schedule.** At equal strength the schedules were indistinguishable (assessment differences ≤ 0.003).
- **Feedback.** The controller engaged in 16 of 18 runs but behaved like a switch, with no measurable benefit.
- **Candidate.** Joint feedback gained 0.006 (0.001–0.012) on the pair over its local twin, with no recipient shift and task-only accuracy, but never qualified as a nominee.
- **Strongest arm.** The raw-penalty joint arm, whose measured push was 2.6–3.2× the task push, beyond the normalized grid, reached coalition SEX AUC 0.810 against 0.882 for task-only training, at equal accuracy.

**Conclusion.** Update strength, not critic freshness or this feedback rule, was the measured lever within the tested range.

## Table rows (keep LEACE and FARE)

| Method | Income / occupation acc. (U 0.852 / 0.480) | Coalition SEX AUC | Coalition linear R² |
|---|---|---|---|
| LEACE on U (E) | 0.799 / 0.434 | 0.871 | 0.00 |
| FARE (F) | 0.847 / 0.462 | 0.721 | 0.13 |
| Raw-penalty joint, online critics (RAW-J, β 0.3) | 0.853 / 0.474 | 0.810 | 0.19 |
| Norm-controlled joint + AUC feedback (J-F; descriptive) | 0.851 / 0.481 | 0.864 | 0.22 |
| Norm-controlled local + AUC feedback (L-F) | 0.851 / 0.476 | 0.870 | 0.26 |

## Favourable and adverse evidence

- **Favourable.** At task-only accuracy, J-F leaks 0.026 less than L-F on the income view, with a small resolved pair gain.
- **Adverse.**
  - The faithful raw penalty beats J-F by 0.054 on the pair and on both recipients.
  - The schedule and feedback components show no effect.
  - The refreshed schedule stalls at high strength.
  - All neural arms still leak (pair AUC 0.81–0.88).
  - FARE leaks far less but fails the occupation gate.
  - LEACE removes linear signal at a large accuracy cost.

**Footnote.** Development evidence on reused rows. Intervals are conditional on the fitted models, and attacks are against a declared slate. Released as EXPERIMENTAL_NO_ADVANTAGE, with RAW-J as the best development baseline.
