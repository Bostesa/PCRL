# Paper addendum — lra (for manuscript owners)

**Status:** CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION
[A=NOT_ESTABLISHED; B=NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE; C=NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE; Q=PASS].

This is reused development evidence, not confirmation. Nominal intervals condition on fitted artifacts
(EXPOSURE_LEDGER.md).

## Allowed summary (drop-in)

> We tested, on the Adult development benchmark, whether learning class-preserving released probabilities (a certified
> per-token convex decoder with a fixed teacher prior) and searching code assignments under explicit fitting budgets
> could yield a more private release that preserves both tasks' predictions and confidence. A correctness-only launch
> gate passed and was independently reproduced. On held-out rows of this development assessment, this learned decoder
> (κ = 32 teacher pseudo-observations, fitted on the rows on which the frozen teacher heads were also trained) worsened
> confidence on identical tokens (occupation log loss +0.010 [0.008, 0.012] nats for the inner-named fixed map and
> +0.024 [0.021, 0.027] for the supervised task map; nominal 95% intervals), while improving it in-sample on those
> fitting rows. No learned-decoder or constrained release was eligible under the original inner utility rules. The
> inner-nominated privacy release was an existing privacy-trained map with its original mean decoder, nominated by a
> selection rule chosen after an earlier study (adaptive). On the registered attacker slate it reduced pair attribute
> recovery by 0.034 [0.029, 0.039] AUC (family-wise interval) below the strongest task-only compression, with identical
> decisions. Its occupation log-loss preservation was unresolved (excess 0.0081, upper bound 0.0121 against 0.01), as in
> an earlier study of the same map on the same reused rows. The fixed task-only code Q kept both tasks' log loss and
> Brier within the original allowances on all four bounds, repeating earlier results on the same reused rows; Q uses the
> original mean decoder and is not evidence for learned decoders.

**Scope (prompt §17).**
- A decoder-only utility change is not information removal.
- A fixture result is not an Adult result.
- An Adult development result is not confirmation.
- A useful calibrated existing code is not a newly invented algorithm.
- Fewer leaking finite attackers is not a population privacy guarantee.

## Supporting facts

| Fact | Value | Source |
|---|---|---|
| Launch gate | ENGINEERING_READY, 12/12 checks, independently reproduced | ENGINEERING_GATE_RESULT.json; INDEPENDENT_VERIFICATION.json |
| Claim A | pair AUC gain 0.0336 [0.0286, 0.0386]; 10/11 clauses pass; P07 occupation LL 0.0081 [0.0040, 0.0121] fails the upper bound | PRIMARY_ENDPOINTS.csv |
| Claims B and C | no eligible nominee: every D1 release fails inner confidence. Descriptive pair gains 0.021 / 0.023 with occupation LL violations of 0.048 / 0.054 (fallbacks K-SEQ-21 / K-JOINT-PAIR vs C\* = C_pair\* = D0 SEQ-21 λ0.1; ineligible on inner) | PRIMARY_ENDPOINTS.csv |
| Q | all four upper bounds within limits | PRIMARY_ENDPOINTS.csv |
| Same-token D1 − D0 | fitting rows −0.026 (occupation LL, mean over 27 fixed maps × 3 seeds); inner +0.014 (the same 81 map-seed units); assessment +0.010 / +0.024 / +0.001 for the three named pairs (nominal 95% lower bounds > 0) | DECODER_UTILITY_ABLATION.csv |
| Independent reproduction of the assessment endpoints | PASS (PHASE_3_FINAL): all 37 endpoints and the label reproduced exactly | INDEPENDENT_VERIFICATION.json |

## May be said only with its qualifier

- "Learning the probabilities improved fitting-row (in-sample) loss." Always pair this with the held-out reversal.
- "Constrained arms met their hard fitting budgets on every seed." Fitting rows only; these budgets did not hold on
  held-out rows.

## Do not write

- "the method works" or "a useful privacy release meeting the criterion";
- "learned decoders improve confidence";
- "learned decoders make confidence worse" without its scope: this decoder (κ = 32, fitted on teacher-trained rows),
  these maps, this development assessment;
- "calibration cannot help";
- "label-blind" for P\*, whose partition is privacy-trained with SEX;
- "the best/strongest useful release" for P\*, since claim A was not established;
- "constrained or joint search helps";
- any novelty claim;
- "beats Taylor / Privacy Funnel";
- "the joint program is convex";
- "fitted MI is a population guarantee";
- "a fixture result shows …" about Adult;
- "information removal" for a decoder-only change;
- "confirmed";
- "independently verified endpoints" before verifier phase 3 completes.

## Prior art

- Privacy Funnel (arXiv:1402.1774); Taylor, Vippathalla and Coon (arXiv:2601.21859v2); proper-loss calibration;
  greedy partition search. These are established ideas.
- The sequential arms are matched adaptations, not the official Taylor solver.
- The decision-disclosure floor is a standard data-processing argument.
