# D fixed-family independent interpretation

R04's `D_567_INDEPENDENT_INTERVALS.json` (SHA-256
`3c8ad07fbedcc62044c538451f2ffd4292f0b95088120fc9c26206c02c7d5efa`)
was computed in R03 unit `R04-D-INTERVALS-001`. Its status is
`FIXED_567_INTERVALS_COMPUTED`. It includes all 567 predeclared raw signed
contrasts, both original seeds and their equal mean, 2,048 paired resamples of
2,234 INNER groups, and one two-sided maximum-|t| 95% critical value
`3.7939044372402653`. R04 independently reconstructed all nine release
inputs, with four U0/Ucal anchor attacks; the point vector agrees *exactly*
with its separate task and reader replay receipts in each seed. The earlier
data-free tied/duplicated AUC arithmetic fixture passed.

The raw task contrast is **candidate minus U0/Ucal**. Accuracy preservation
passes only when its *lower* bound is at least `-0.010`; NLL and summed Brier
pass only when their *upper* bounds are at most `0.010` and `0.005`.
Conversely, an NLL/Brier cost is established beyond cap when its lower bound
exceeds that cap. Cap equality passes. The raw `D_T:D_K` AUC contrast is
**T minus D-K**, so a positive value is a coalition recovery reduction for
D-K relative to T.

| Mean clause | Point | Simultaneous 95% interval | Sealed target | Result |
| --- | ---: | ---: | ---: | --- |
| D-K occupation NLL minus U0 | 0.032253 | [0.010368, 0.054137] | ≤0.010 | Violation |
| D-K occupation Brier minus U0 | 0.015353 | [0.005914, 0.024793] | ≤0.005 | Violation |
| D-K occupation NLL minus Ucal | 0.033705 | [0.015890, 0.051520] | ≤0.010 | Violation |
| D-K occupation Brier minus Ucal | 0.014797 | [0.006994, 0.022600] | ≤0.005 | Violation |
| T minus D-K coalition AUC | 0.034429 | [0.013719, 0.055138] | ≥0.020 | Unresolved |

D-K's T-relative AUC point reduction exceeds 0.020 in both seeds, but the
simultaneous lower bound is `-0.001877` for seed 0, `0.013132` for seed 1,
and `0.013719` for their mean. None reaches the registered `0.020` target.
D-K's occupation utility failure is established on the mean by four clauses
above; the point utility gate also fails in both seeds. T and every matched
hard-code control L/S12/S21/C fail the both-seed utility point gate. Hence
there is no eligible strongest matched hard-code comparator and no D ten-bit
competitive method claim. D-K remains the specified ineligible diagnostic
fallback. No statistical benefit should be inferred from C's lower recovery
because C is utility-ineligible.

F is the sole seven-arm release that passes the both-seed task **point** gate,
but it transmits 16 continuous latent values and is a capacity diagnostic.
Its mean task family has 10 passed and 2 unresolved clauses, so simultaneous
task preservation is not fully established. F's mean coalition AUC exceeds
D-K's by `0.065924` with simultaneous interval `[0.043268, 0.088580]`, and
exceeds T's by `0.031495` with interval `[0.012683, 0.050307]` (intervals
obtained by reversing the registered K-minus-F and T-minus-F raw contrasts).
Thus the observed continuous utility advantage comes with established higher
coalition recovery than either ten-bit first-pair arm. This is a descriptive
cross-capacity comparison.

Relative to the fixed U0/Ucal *score-wire* anchors, D-K's mean coalition AUC
reduction is `0.056371` and `0.056925`; the corresponding simultaneous
benefit intervals are `[0.033618, 0.079123]` and `[0.034748, 0.079102]`.
Those anchor contrasts clear 0.020, while D-K's task failure and lack of an
eligible matched comparator still rule out the competitive method claim.
These conclusions use only development INNER data; the old assessment role
was not read.
