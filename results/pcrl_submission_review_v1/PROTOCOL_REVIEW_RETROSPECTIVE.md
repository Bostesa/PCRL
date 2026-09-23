# Retrospective review of the ACS 2016 protocol

**Timing, stated plainly.** This review was written **after** the study's outcomes were published, on
2026-09-23, against protocol commit `3f9aaee5ce6317660d987f81caf542dd59b4efbb` and evidence commit
`5e154e5c4fdaeb23d327a0ebefe838525f1a19cb`. I was asked for a pre-outcome review in an earlier round and
did not perform one: when I last checked, `protocol_commit` was `null`, and by the time it was registered
I had moved on to the audit integration. **It is therefore retrospective, it is not backdated, and it
carries none of the force of a pre-registration review.** A reviewer should treat it as a reading of a
locked protocol whose results the reader already knows, which is a weaker thing.

What partially compensates is that the protocol's own lock is verifiable independently of me: the
evaluation lock (SHA-256 `e17b8167…`) binds 15,268 files including all 156 fitted units and validation
selections, and it was committed and pushed at `0b65a1cd0` **before** the final labels were read. That
ordering is checkable from the commit graph, and it does not depend on when I read anything.

## Checked against the protocol text and the evidence

| item | finding |
|---|---|
| Frozen hypotheses and unchanged release objects | **Correct.** Q and D17 are named as the two primary hypotheses with their exact configuration ids; no encoder, teacher, eraser, code, dictionary, budget or quantizer is fitted or tuned on 2016. The only new 2016 fits are the independent attackers and utility probes. |
| One shared partition across anchors | **Correct.** All three frozen anchors use the same final pool of 23,684 people in 15,928 households, and the reported estimand averages three anchor-specific ratios. |
| Fresh-year fitting restricted to auditors and probes; no release reselection | **Correct**, and the panel completed with no candidate replacement and no outcome-driven stopping: 156/156 fit and 156/156 score units. |
| Two conjunctions, each with two task and eight privacy clauses | **Correct**, and both are reported in full rather than summarised to a verdict. |
| Component bounds are pointwise, not simultaneous | **Correctly handled.** The primary clauses use one-sided 97.5% bounds per clause; the intersection–union logic makes the conjunction valid without requiring simultaneity across clauses. The protocol does not claim a global 95% property across primary *and* secondary families, and neither does the paper. |
| Secondary family kept separate with its own correction | **Correct.** 70 endpoints, simultaneous two-sided 95% Bonferroni at `z = 3.384`, reported apart from the primary decision. |
| Loss-difference signs and H-baseline cancellation | **Correct.** Each row carries an explicit `orientation` string, and the two families use opposite-signed conventions that I re-derived independently rather than trusting the labels. |
| Exact expected one-token loss; no loss-of-average-probability substitution | **Correct.** Scores are exact expectations over the released token; a persistent-token fixture agreed with the exact mean within 1.03 Monte Carlo standard errors and is explicitly secondary to the exact estimator. |
| Continuous H in attacks; no token or anchor pseudoreplication | **Correct.** Anchors are averaged rather than treated as independent samples, and the bootstrap resamples the shared household population. |
| A validity process that does not stop on an unfavourable result | **Correct**, and demonstrated: the study ran to completion and published a failing conjunction. |

## The one question I would have asked, and what the record shows

I expected to find that the 0.003-nat task margin had been fixed without checking whether it was
attainable. **That expectation was wrong, and I checked before writing it down.**
`PRECISION_PLANNING.json`, created 2026-09-22T21:47Z from archived 2018 per-person losses and before final
scoring, projects the 2016 standard error at 0.00094 (independent) to 0.00164 (correlated) — brackets that
contain the realised 0.0009–0.0011 — and tabulates pass probabilities under scenarios that scale the 2018
effect size. At the full 2018 effect the projected pass probability is ~1.00; at half of it, ~0.88.

So the margin was analysed in advance and was attainable **if the development effect size transported**.
It did not: the 2018 task improvement over J was −0.0163 nats, and the 2016 estimate is −0.0033, about a
fifth of it. The conjunction failed because the effect shrank on a fresh year, not because the study was
underpowered for the effect it expected. That is a more informative outcome than a power failure, and it
is the single most useful thing this prospective test produced.

The residual caveat is about the planning input rather than the plan: the projection was built from
already-used 2018 pools, so its effect-size prior inherited the optimism of development data. Nothing
here changes a reported number, and none of it is a defect in the executed study.
