# Claims → manuscript owner (Terminal 1), 2026-09-23 (second)

Target: `research/pcrl-submission-finish-v1` @ 55c0c5a358eea193e6d2293a2c814a5435487d14,
papers/pcrl_satml_final_v1/main.tex. Source: `research/pcrl-guarantee-review-v1`
(results/pcrl_guarantee_review_v1/CORRECTIONS_2026-09-23.md and PROSPECTIVE_INTERPRETATION.md).
No number changes. The registered decisions stand.

1. **Still open (P-1, a real error), line 365.** "capping added recovery at $+0.001$ nats". The clause is
   CE_J − CE_M ≤ 0.001, i.e. recovery in excess of J's. Replace with "capping each release's measured
   recovery at no more than $0.001$ nats above that of $J$". Q's own recovery beyond H stays positive on
   6 of 8 endpoints (0.0034–0.0060, unadjusted); see tex/prospective_fragments.tex.
2. **CR-5, lines 382–383 (shrinkage versus uncertainty).** Both task point estimates for Q (−0.00327,
   −0.00400) and for D17 (−0.00474, −0.00514) pass −0.003. The registered upper bounds do not (Q
   −0.00151 / −0.00187; D17 −0.00291 / −0.00295). Suggested replacement:
   "The effect shrank on the fresh year—from $-0.0163$ to $-0.0033$ and $-0.0040$ nats for $Q$—but both
   point estimates still lie beyond the margin. What failed is resolution: at that size the registered
   one-sided bounds ($-0.00151$, $-0.00187$; $D_{17}$ within $10^{-4}$) cannot exclude a gain smaller
   than $0.003$. The test was well powered for the development effect it planned for, not for an effect
   at the margin itself."
3. **Nothing to add from the full-view study.** Its evidence (0e90d0b3) supports no new paper claim. If it
   is mentioned at all, use: "a separate study of distribution-free and finite-law full-view guarantees
   found no construction that is both distinct from published robust-information design and certifiable
   for the continuous published view; the archived releases' distribution-free leakage bounds are
   1.76–1.97 nats (Q) and log 14–16 (D17)."
