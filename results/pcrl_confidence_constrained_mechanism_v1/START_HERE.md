# Start here: confidence-constrained release sprint (ccm)

**Label:** PREMISE_NOT_SUPPORTED. The registered feasibility premise failed, so no prototype was built.

## The question

Could we protect sex better than plain compression by releasing coarsened teacher scores whose confidence is guaranteed
to stay close to the calibrated teacher's? Here "close" means a guarantee for every person and every possible label:
- log loss within 0.005;
- multiclass Brier within 0.0025;
- the same predicted class.

Each recipient gets at most the usual capacity: 8 income tokens or 64 occupation tokens per predicted class. Anyone the
tokens cannot serve gets the calibrated scores themselves, with a flag.

## What we learned (label-free; no sex or task label was read)

**The guarantee leaves almost nothing to coarsen.** One token may stand for several people only if their calibrated
score vectors differ by at most about half a percentage point in total.

**Occupation.**
- At least 10,105 of the 15,434 fitting people need a token of their own.
- With the allowed 384 tokens, no possible code can serve more than 36.6% of them; the code we built serves 17–18%.
- 85% of new people would receive their raw calibrated scores.
- This is an identity release in all but name.

**Income.**
- The best possible code with the allowed 16 tokens serves 37–39% of fitting people. This is exact: it is an optimum,
  not just a construction.
- 61–63% of new people would receive their raw calibrated scores.

**Consistency across seeds.** All three teacher seeds agree, and every failure is intrinsic to the guarantee rather
than to our construction. The locked go rule therefore said stop, and we stopped.

**Small worlds.** In five small hand-built worlds the code finds sex-protecting partitions when they exist:
- it halves the information in the POSITIVE world;
- joint design beats both sequential orders in the JOINT_HEADROOM world, but only ties the sex-blind fewest-token
  code there.

These worlds show the machinery works. They say nothing about Adult.

**A weaker guarantee.** As a registered side check, we tried a guarantee that holds only on average if the teacher is
calibrated. Under it, income compresses well: 16 tokens serve 95–96% of people and 4–5% fall back. Occupation still
falls back for 44–48% of new people. This is a hint for a possible next contract, not a result.

## What this does not say

- It says nothing about privacy on Adult: no attacker was fitted.
- It does not say that average-loss confidence targets are impossible; those are looser.
- It does not say that other capacities fail; none was tried, by design.

## Where to look next

- **MECHANISM_DECISION.md:** the go/no-go and why.
- **FEASIBILITY_RESULTS.md:** all numbers.
- **UTILITY_CONTRACT.md and MATH_REVIEW.md:** what is guaranteed and proved.
- **INDEPENDENT_VERIFICATION.json and VALIDATION.md:** the independent replay.
- **QUICKSTART.md:** the commands that produced every number.
- **RESEARCH_DECISION.md:** the recommendation.
