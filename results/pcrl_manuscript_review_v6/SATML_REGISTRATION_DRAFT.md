# SaTML registration draft — title and abstract

**Status: DRAFT, not submitted. Nothing has been sent to anyone, and no venue account has been touched.**
Prepared 2026-09-20 from the v6 manuscript and the committed evidence behind it.

Every sentence below describes work that is **already committed and verified**. The stochastic
(role-constrained stochastic release) study has **no ACS result**: no pool read, no release fitted, no
compute spent. It is therefore mentioned only as registered, ongoing work with no outcome, and nothing in
this abstract promises that it will succeed. If it closes negative, or never runs, **no sentence here needs
to change**.

---

## Title

**Adding a reusable feature channel beside a prediction service you cannot change: an evaluated release
problem, one confirmed effect, and six negative method results**

Alternative, if a shorter title is required:

**A release problem where the published output is immutable: one confirmed coalition effect and six negative
mechanism results**

## Abstract (235 words)

An organisation already publishes a prediction service — fixed probability vectors that downstream recipients
consume and cannot be asked to re-accept. We formalise and evaluate what happens when it adds a second,
reusable feature channel for one recipient without altering a published number, while trying to limit what
that recipient, and that recipient colluding with another, can newly infer about sex and race. Our
contribution is an implemented release interface with a measurement contract — per-recipient and coalition
views, independently fitted attackers, incremental disclosure always reported beside absolute disclosure,
negative increments retained unclipped — and six studies measured through it on American Community Survey
data. On a locked, previously unused survey year, a penalty built from the coalition's joint view beats an
equal-strength and an equal-total-mass local control in 14 of 16 sensitive cells, none worse; its utility cost
is bounded by a point rule, not by an interval. Every subsequent attempt to make the mechanism competitive
failed, under both definitions of competitiveness we registered: a nonlinear penalty, an exact
rotation-invariant repair that disclosed significantly more, adversarial refinement, coalition-conditioned
projection of the strongest channel, and an extension asked to buy capability at bounded added disclosure,
whose pilot found capability and disclosure inseparable at every adversarial weight tried. Adapted published
erasers beat our own mechanism. We report the release problem and the evaluation design as the contribution,
and list every claim this revision withdraws or narrows.

---

## What this abstract deliberately does not say

* It does not mention a stochastic or randomized mechanism as a result, a contribution or an expectation.
  That study is registered, its machinery is validated on synthetic fixtures only, and it has produced no ACS
  number.
* It does not promise a positive result in any ongoing work.
* It claims confirmation only for the locked-year coalition effect; every other study is labelled development.
* It does not claim any bound on mutual information, any certificate, or survival against a stronger attack
  (three searches for a stronger attacker each failed to find one).
* It does not describe the residence-label-free proxy as evidence that the utility target was innocuous
  (corrections D7).
* It does not assert that a .001-nat confirmation is impossible for any mechanism (correction D10,
  `G0_REVIEW.md`).

## If the stochastic study later produces ACS results

Integrate under the existing rules: results only from a pinned evidence commit, recomputed independently,
with tiers that did not run marked *not triggered*. The abstract gains at most one sentence, and only if the
outcome is measured. The sentence would be a template with the outcome left blank until measured, e.g.
*"A seventh study asks whether a randomized finite release reaches an operating point the deterministic
extension could not; on the fitted finite model it <MEASURED OUTCOME>, and on the audited continuous release
it <MEASURED OUTCOME>."* No filled-in version of that sentence exists yet, and none may be written before the
audit runs.

## Practical notes (not part of the submission text)

* Word count above is for the abstract only; SaTML's limit should be checked against the call before use, and
  the "10–11 page" figure used elsewhere in this project is an internal editorial target, **not** a verified
  venue page limit.
* The submission would be anonymous; the repository is public during review by the author's explicit decision.
* Nothing here is submitted, registered or sent. The next action is the author's, not this terminal's.
