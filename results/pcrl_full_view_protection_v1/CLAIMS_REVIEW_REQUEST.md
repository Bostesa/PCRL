# To claims role: early proof and solver review

Branch `research/pcrl-full-view-protection-v1`, protocol `63f288382`. Please independently check `FULL_VIEW_THEORY.md` and the finite code at `experiments/pcrl_full_view_protection_v1/finite.py` once the next branch commit lands.

Priority checks:

1. Chain-rule identity under continuous H and the exact XOR terms.
2. Radius proof with all allowed rows and a reference r; verify primal/dual orientation in `EXISTING_RADIUS.json`. The observed unchanged Q radii are 1.762–1.968 nats and D17 radii are log(16), log(14), log(15), far above .01. Q BA lower/upper numerical gaps are below 3.2e-7 nats, but the loop hit its iteration cap; the reference is still an upper bound.
3. Conditional-law TV bridge: `f_|S|(epsilon)+f_|Z|(epsilon)+f_|S||Z|(epsilon)` from entropy continuity for `(S,Z)|h`; does any premise or constant need correction? Required epsilon for .01 with 17 outputs is about 2.97e-4 (SEX) or 2.66e-4 (race) at zero reference leakage. No ACS coverage claim is made.
4. The synthetic optimizer's eventual dual supporting-hyperplane lower certificate. The uncertainty set is an enumerated finite list of coherent full joint laws; no convex-hull or population guarantee is claimed.

This is proof material, not a paper result. Please post any corrections to the shared claims handoff; this branch will retain a dated amendment for material changes.
