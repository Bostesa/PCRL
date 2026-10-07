# Advisor brief — lcr (learned decoders and confidence-constrained releases)

**Bottom line.** The pre-registered mechanism gate was not met, so the Adult study was not run. The result is
MECHANISM_GATE_NOT_MET. It is not an Adult result in either direction.

## What we tried to test

- **The idea.** Two parts:
  - learn the released probability vectors (a certified convex decoder, "D1") instead of using mean teacher vectors;
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
  - The protocol forbids changing fixture laws after algorithms have seen them, and an independent oracle had already
    enumerated them.
  - Our original forecast (85% that the gate would pass) is recorded as a miss.
- **Engineering checks.** Every mandatory correctness check passed on every fixture:
  - decoder changes never change information;
  - no gain in the calibrated null;
  - budgets enforced;
  - decisions preserved;
  - incremental terms match a from-scratch rebuild;
  - heuristics labelled against exhaustive references.
- **One spurious failure, repaired.** A correctness check I added before the lock misfired on the first attempt. It was
  repaired by a pushed, code-only amendment. Both attempts are kept.

## What the fixtures still teach (descriptive)

- **Miscalibrated teacher.**
  - The learned decoder lowers log loss by 0.033–0.083 nats on identical tokens.
  - It turns 27 of 27 old private maps from over-budget into within-budget.
  - Information is unchanged.
- **Unregistered, post-hoc observation.**
  - At the same (zero) pair leakage as the decision-only release, decoded privacy maps keep 0.042 nats more task
    utility on F2, and 0.004 on F3.
  - Almost all of that is available from OLD maps once decoded. The new constrained search adds about 0.0006 nats.
  - This suggests that any value lies in the decoder, not the new search. It is not a registered result.

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

## Cost and custody

- About 1.5 CPU-h of the 20 allowed, $0 cloud.
- Verified same-device copy with a restore; off-device backup pending (no drive).
- The assessment rows were never opened.
