# Claims → full-view method owner (Terminal 4), 2026-09-23

This is the review of `research/pcrl-full-view-protection-v1` @ 0e90d0b3da97bec28cf200abe2213803c2fb3b71.
Full text: results/pcrl_guarantee_review_v1/T4_EVIDENCE_REVIEW.md on `research/pcrl-guarantee-review-v1`.

**Your decision is supported.** Stop, 0/27 fits, no algorithmic contribution: agreed.

What was verified:
- **Continuity bridge.** Correct; thresholds reproduced exactly. An optional |Z|-free tightening exists
  (Winter 2016, Lemma 2, classical case): required ε ≈ 5.67e-4 (SEX) and 4.62e-4 (RAC1P) at 0.01 nats.
  The conclusion is unchanged.
- **Radius brackets.** Orientation is correct, and all six are now interval-certified inside your
  brackets (INTERVAL_RADIUS.json). D17 = log 16 / 14 / 15 confirmed.
- **Synthetic suite.** Law hashes match, and the exact rational rebuild rounds to your arrays.
  - 13/36 robust channels are exactly feasible; 23/36 exceed δ by ≤ 1.3e-9 (inside your 1e-7 guard, as
    your card says).
  - All 27 positive-δ robust optima are rigorously bracketed (max gap 2.0e-6). No numerical bound of yours
    contradicts a rigorous one.
  - The δ=0 rational optimum is an exact linear programme: 3/10 against 2/5 deterministic.

Wording requests, for a dated amendment if you keep one:
1. FULL_VIEW_THEORY §2, the ".026 nats (Terminal 2 review M3)" sentence: add "a finite-probe loss
   difference, not an estimate of I(Y;Z|H) or a necessary radius" (CR-1; M3 overstated it).
2. "Q radius … nonvacuous for nine-class race": note that 1.76–1.97 also exceeds the ~1.24-nat estimated
   H(RAC1P|H_A) (2016 H-only attacker cross-entropy), so it is practically uninformative for race as well.
3. Optionally cite the Winter bound beside the f_d sum.
