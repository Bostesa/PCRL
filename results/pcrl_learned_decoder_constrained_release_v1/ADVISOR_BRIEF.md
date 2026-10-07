# Advisor brief — lcr (learned decoders and confidence-constrained releases)

**Bottom line.** The pre-registered mechanism gate was not met, so the Adult study was not run. The result is
MECHANISM_GATE_NOT_MET. It is not an Adult result in either direction.

## What we tried to test

- **The idea.** Two parts:
  - learn the released probability vectors (a certified decoder, "D1", convex for each fixed token) instead of using mean teacher vectors;
  - search the code assignments under hard confidence budgets.
- **Hoped-for outcome.** Together, a release that keeps both tasks' confidence and leaks less SEX information than
  task-only compression.
- **Before touching Adult.** The method had to pass a gate on four small known-law fixtures. Some privacy-trained
  partition with D1 had to stay within the budgets and cut pair mutual information by at least 0.01 nats below the
  strongest budget-feasible task-only compression.

## What happened

- **The gate failed for a structural reason, and the reason was written down before the gate ran.**
  - Every release in this design reveals both predicted classes, so no release can leak less than the decision-only
    release.
  - In all four fixtures the decisions carry zero SEX information, and the decision-only release fits the budgets
    (except in the calibrated null, where no task-only release fits).
  - So "0.01 nats below the strongest task-only compression" was impossible on this fixture bank.
- **We did not redesign the fixtures.**
  - The prompt forbids an outcome-informed hunt for favourable fixture laws. An independent oracle had already
    enumerated them before the lock, a disclosed deviation from the registered order.
  - Our original forecast (85% that the gate would pass) is recorded as a miss. A pre-stage update to 1%, informed by
    the oracle's counts, is recorded separately and is not an independent forecast.
- **Engineering checks.** Every mandatory correctness check passed on every fixture:
  - decoder changes never change information;
  - no gain in the calibrated null;
  - budgets enforced;
  - decisions preserved;
  - incremental terms match a from-scratch rebuild;
  - heuristics labelled against exhaustive references.
- **One spurious failure, repaired by amendment.** A correctness check added before the lock misfired on the first
  attempt. After that attempt, a pushed code-only amendment (A1) scoped it. The verdict and oracle tables are identical
  across attempts, and both attempts are kept.

## What the fixtures still teach (descriptive)

- **Miscalibrated teacher.**
  - On the miscalibrated fixture F2, the learned decoder lowers log loss by 0.033–0.083 nats on identical tokens.
  - It turns all 27 of that fixture's fixed mean-decoder maps (task-only and privacy-trained) from over-budget into
    within-budget.
  - Information is unchanged. These are fixture facts, not Adult evidence.
- **Unregistered, post-hoc observation.**
  - At the same (zero) pair leakage as the decision-only release, decoded privacy maps have a lower task objective
    T = L1 + L2 + ½(B1 + B2): by 0.042 on F2 and 0.004 on F3. T is log loss in nats plus half the Brier score.
  - On F2, almost all of that is available from OLD fixture maps once decoded; the new constrained search adds about
    0.0006 in T.
  - On these fixtures this suggests that any value lies in the decoder, not the new search. It is not a registered
    result and says nothing about Adult.

## What it does not mean

- Not "the method fails on Adult", and not "learned decoders or constrained search do not work".
- On Adult, decisions do carry SEX signal (cbp measured a decision-only pair AUC of 0.739). Whether the decision-only
  release meets these confidence budgets on Adult was never computed.

## Recommendation

- **Close this gate as registered.** No Adult claim and no weight adjustment.
- **If the line continues,** it needs a NEW registered study with a new fixture bank written before any algorithm runs.
  - At least one fixture must make the decision-only release over-budget, or give decisions SEX content. Otherwise the
    trigger is unattainable by construction.
  - That study must also say up front whether "same leakage, more utility" counts as success. This gate counted only
    leakage reduction.

- No novelty is claimed. Hard utility constraints, privacy-funnel objectives, proper-loss calibration, greedy partition
  search and sequential collusion accounting are established (Privacy Funnel, arXiv:1402.1774; Taylor, Vippathalla and
  Coon, arXiv:2601.21859 v2; PRIOR_ART_AND_CLAIM_SCOPE.md).

## Cost and custody

- About 1.7 CPU-h of the 20 allowed, $0 cloud.
- Verified same-device copy with a restore; off-device backup pending (no drive). The predecessor custody (cbp, and
  through it qpc, dpc, osf and smf) also stays PENDING, because it would open the shared assessment rows.
- This study never opened the assessment rows (earlier studies did; EXPOSURE_LEDGER.md).
