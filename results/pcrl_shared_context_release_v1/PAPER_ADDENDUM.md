# Factual addendum for the manuscript owner: shared-context release (2018 development)

**Status:** `EXPERIMENTAL_NO_ADVANTAGE`, on 2018 development households that have been used repeatedly. This is not confirmation. It changes neither the completed 2016 prospective result nor the predecessor adaptive-release result. Do not insert a success claim, and do not describe the method as competitive.

## What was tested

- **Why a new design.** The predecessor's richer-state branch never split a T32 state: at most 127–198 coefficient households per parent, against a 100-per-child floor.
- **The model.** This study fitted a pooled contextual mixture of frozen deterministic 17-token policies around the full T32 kernel class, so D17 is embedded exactly. It kept the one-token interface and byte-preserved H_A.
- **Instantiation.** Four richer policies were trained on all three anchors. They differ from D17 for 52–93% of people and vary inside every T32 state. They were available to the fitting LP at every round.

## Results suitable for citation (locked; 40-endpoint Bonferroni, z = 3.227; 10,000 common household draws)

- **Utility route: failed** (8/20 clauses; all eight passes are trivial RAC1P same-route zeros).
  - The utility nominee is identical to the parent-restricted T32 control: a deterministic T32 kernel on one anchor and D17 on two.
  - Task versus D17: −.00015 nats unweighted and −.00002 PWGTP, with simultaneous intervals [−.0033, +.0030] and [−.0036, +.0036].
- **Privacy route: failed** (0/20 clauses).
  - Coalition SEX recovery versus D17: −.0040 unweighted [−.0095, +.0015] and −.0031 PWGTP [−.0093, +.0031]. These are favorable point estimates only.
  - Task cost: **+.0078 [+.0033, +.0124]** unweighted and **+.0076 [+.0022, +.0130]** PWGTP. The cost is demonstrated.
- **Strongest positive comparison.** The U-slot capability over H-only is −.038 / −.040 nats, which passes the .01 capability check. Versus J, the U nominee's task difference is −.017 / −.016. J is a continuity reference with a different interface, not a matched control.
- **Strongest negative comparison.** The exact deterministic same-context selector from the same policy bank matched the privacy nominee's coalition-SEX recovery (−.0003 / +.0003). Its task loss was .008 lower (U interval for the nominee minus the selector: [+.0027, +.0133]). Randomized mixing showed no benefit, although this control selects on task rather than privacy.
- **Task sufficiency.** A richer task-only deterministic policy gains nothing over D17: +.00017 / +.00019 nats, interval [−.0002, +.0005]. Under this decoder family, X_A carries no usable residence information beyond (T32, H_A) through a 17-token deterministic release.

## Wording guardrails

- **Say:**
  - "a pooled richer-input release was instantiated and did not establish a development advantage over D17";
  - "fitted-attacker recovery";
  - "development evidence on repeatedly used 2018 households".
- **Do not say:**
  - "privacy guarantee";
  - "label-free";
  - "stochastic release is necessary";
  - "richer inputs improve privacy";
  - "competitive";
  - "outperforms D17".
- **Scope of the intervals.** They condition on the fitted, selected objects. Anchors overlap heavily: 81% of coefficient people appear in two or more anchors.

## Pins

- **Branch:** `research/pcrl-shared-context-release-v1`.
- **Lock:** commit `e5d555a1438e14def094b1cdc1b1f5d2c2d52394`; `SELECTION_LOCK.json` SHA-256 `d995484103cfbc9ca9f1411a2f89d6f5d9ee1f46e1bbdce1c8d0d82d20cadbfc`.
- **Result files:** `INFERENCE.json`, `FULL_RESULTS.csv`, `RESEARCH_DECISION.md`, `ENCODER_CAPACITY.md`, `INDEPENDENT_VERIFICATION.json`.
- **Post-hoc numbers:** only in `POSTHOC_EXPLORATORY.*`, labelled not registered. Do not cite them as results.
