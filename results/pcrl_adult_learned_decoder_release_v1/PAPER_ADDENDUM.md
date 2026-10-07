# Paper addendum — lra (for manuscript owners)

**Status:** CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION
[A=NOT_ESTABLISHED; B=NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE; C=NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE; Q=PASS].

This is reused development evidence, not confirmation. Nominal intervals condition on fitted artifacts
(EXPOSURE_LEDGER.md).

## Allowed summary (drop-in)

> We tested, on the Adult development benchmark, whether learning class-preserving released probabilities (a certified
> per-token convex decoder with a fixed teacher prior) and searching code assignments under explicit fitting budgets
> could yield a more private release that preserves both tasks' predictions and confidence. A correctness-only launch
> gate passed and was independently reproduced. On held-out rows the learned decoder worsened confidence on identical
> tokens (occupation log loss +0.010 [0.008, 0.012] nats for the inner-named fixed map, +0.024 [0.021, 0.027] for the
> supervised task map), while improving it in-sample on the fitting rows on which the frozen teacher heads were also
> trained. No learned-decoder or constrained release was eligible under the original inner utility rules. The strongest
> useful release remained an existing label-blind map, which reduced pair attribute recovery by 0.034 [0.029, 0.039]
> AUC below the strongest task-only compression with identical decisions, but whose occupation log-loss preservation was
> unresolved (excess 0.0081, upper bound 0.0121 against 0.01). Q (task-only confidence feasibility) passed.

## Supporting facts

| Fact | Value | Source |
|---|---|---|
| Launch gate | ENGINEERING_READY, 12/12 checks, independently reproduced | ENGINEERING_GATE_RESULT.json; INDEPENDENT_VERIFICATION.json |
| Claim A | pair AUC gain 0.0336 [0.0286, 0.0386]; 10/11 clauses pass; P07 occupation LL 0.0081 [0.0040, 0.0121] fails the upper bound | PRIMARY_ENDPOINTS.csv |
| Claims B and C | no eligible nominee: every D1 release fails inner confidence. Descriptive constrained pair gains 0.021 / 0.023 with occupation LL violations of 0.048 / 0.054 | PRIMARY_ENDPOINTS.csv |
| Q | all four upper bounds within limits | PRIMARY_ENDPOINTS.csv |
| Same-token D1 − D0 | fitting rows −0.026 (occupation LL, mean of 81 maps × seeds); inner +0.014; assessment +0.010 / +0.024 / +0.001 for the three named pairs | DECODER_UTILITY_ABLATION.csv |

## May be said only with its qualifier

- "Learning the probabilities improved fitting-row (in-sample) loss." Always pair this with the held-out reversal.
- "Constrained arms met their hard fitting budgets on every seed." Fitting rows only; these budgets did not transfer.

## Do not write

- "the method works" or "a useful privacy release meeting the criterion";
- "learned decoders improve confidence";
- "constrained or joint search helps";
- any novelty claim;
- "beats Taylor / Privacy Funnel";
- "the joint program is convex";
- "fitted MI is a population guarantee";
- "a fixture result shows …" about Adult;
- "information removal" for a decoder-only change;
- "confirmed".

## Prior art

- Privacy Funnel (arXiv:1402.1774); Taylor, Vippathalla and Coon (arXiv:2601.21859v2); proper-loss calibration;
  greedy partition search. These are established ideas.
- The sequential arms are matched adaptations, not the official Taylor solver.
- The decision-disclosure floor is a standard data-processing argument.
