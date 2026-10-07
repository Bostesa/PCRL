# Advisor brief — lra (Adult learned decoder and constrained release)

**Bottom line.** The real Adult test ran end to end and gave a clean negative for the new method.
- **Decoder.** Learning the released probabilities made held-out confidence WORSE, not better.
- **Search.** The new constrained search produced no eligible release.
- **Best release.** The best useful release is still the old label-blind JOINT λ0.1 map. It cuts pair attribute
  recovery by 0.034 AUC below the strongest task-only compression. It misses the full criterion only because its
  occupation log-loss preservation is unresolved: upper bound 0.0121 against the 0.01 limit.
- **Label.** CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION.

## What was tested

- **The question.** Can a learned decoder (certified, class-preserving, convex for each fixed token), plus assignment
  search under explicit confidence and information budgets, give a more private release that keeps both tasks'
  predictions and confidence?
- **The comparison bank.** It was complete:
  - old maps;
  - the same maps with the learned decoder;
  - a supervised task-only compression;
  - 72 weighted controls;
  - 15 constrained fits (local, both sequential orders, joint single-move, joint paired-move);
  - the usual references.
- **The launch gate.** It was correctness-only: twelve implementation checks on the four known-law fixtures, all
  passed and independently reproduced. It did not demand a synthetic win, which the predecessor showed was
  unattainable by construction.

## What happened

1. **The learned decoder overfit.**
   - On the fitting rows it cut occupation log loss by about 0.026 nats on identical tokens.
   - Those are the rows the frozen teacher heads were trained on.
   - On held-out rows it ADDED log loss: +0.010 nats on the best fixed map and +0.024 on the supervised task map. Every
     contrast's interval excludes zero.
   - Information is unchanged, because the tokens are identical; this is purely a confidence loss.
2. **The supervised assignment search overfit too.**
   - The new task-only map is worse held out than the old label-blind one, even with the old decoder: occupation excess
     0.015 against 0.003.
   - The constrained arms met their hard fitting budgets in-sample on every seed. Held out, they violate confidence by
     0.048–0.061 nats.
3. **No method component helped.**
   - Constrained search did not beat the weighted controls; both were ineligible.
   - Paired joint search tied single-move and sequential search (pair AUC 0.792 vs 0.791–0.796).
4. **The engineering held up.**
   - Independent replay reproduced every certified decoder vector and 185,057 search states, including 99 paired
     moves.
   - It also reproduced every selection value and the evaluation lock.
   - All real-data controls passed.

## What it does not mean

- Not a population privacy guarantee. Not confirmation: these are reused development rows.
- Not "calibration can never help". This decoder was fitted on rows the teacher had already seen.
- Not a new algorithm. The best release is an existing map from the qpc/cbp lineage.

## Recommendation

- **Close this recipe.** No further λ or κ tweak on this assessment.
- **If calibration is revisited:** it needs held-out calibration data the heads never saw. That is a new, separately
  registered design.
- **No confirmation population is spent.**

## Cost and custody

- About 7 CPU-h of 20, at most two processes, $0 cloud.
- Custody: BACKUP_VERIFICATION.json; off-device status recorded there.
- No novelty is claimed. Privacy Funnel (arXiv:1402.1774), Taylor–Vippathalla–Coon (arXiv:2601.21859v2), proper-loss
  calibration and greedy partition search are established ideas (PRIOR_ART_AND_CLAIM_SCOPE.md).
