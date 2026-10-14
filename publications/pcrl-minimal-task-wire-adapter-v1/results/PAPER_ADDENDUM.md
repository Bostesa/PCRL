# Factual addendum for the manuscript owners (not an edit to any manuscript)

These are exploratory development results on repeatedly used Adult people. Intervals are conditional on the fitted models and readers, and the two model seeds reuse the same people.

1. A 325-parameter residual in the occupation head's row space was trained over a frozen RAW_JOINT trunk with a teacher-function KL anchor of 0.001 and absolute fitting budgets.
   - It passed all 24 original point utility clauses on FIT2 and INNER in both seeds (epoch 10).
   - Under simultaneous intervals: 5 PASS, 19 UNRESOLVED (12 of them frozen income), 0 violations.
2. Its coalition (income + occupation) recovery under 18 fresh readers was 0.0049 AUC below RAW_PROJ, with interval [−0.0017, +0.0115]. That is an established shortfall against the 0.020 criterion. The same holds against all four matched trainable controls.
3. Its occupation-only recovery was 0.0100 [0.0026, 0.0173] below RAW_PROJ and 0.0082 [0.0008, 0.0155] below the matched task-only control.
4. The trust constraint improved held-out utility descriptively but did not change recovery. Coalition coupling (joint vs local objective) changed nothing measurable.
5. The earlier separate and global guards (epoch 5) lowered fresh coalition recovery relative to RAW_HYBRID: by 0.0151 [0.0044, 0.0257] and 0.0116 [0.0025, 0.0207] respectively. Both fail the original utility caps, so this is a diagnosis, not a method result.
6. Moving from RAW_HYBRID to RAW_PROJ (known compression) changed fresh coalition recovery by 0.0019 [−0.0060, 0.0097], which is unresolved.

Suggested wording, if used at all: "a constrained occupation adapter preserved point utility but did not establish coalition protection." Do not describe it as a privacy mechanism, a guarantee, novel, or joint design of both recipients.
