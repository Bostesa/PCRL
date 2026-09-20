# MATHEMATICAL_REVIEW — v6 addendum

The v4 review and the v5 addendum (`results/pcrl_manuscript_review_v5/MATHEMATICAL_REVIEW.md`, part A on the
Study 5 projection) are carried unchanged. This addendum records how each item of the v5 **pre-outcome**
review of the utility-first extension (part B, and R1–R10 in the v5 handoff) was answered by the executed
study, checked against its committed code and evidence.

| v5 review item | How the executed study answered it | Evidence |
|---|---|---|
| R1 Monotonicity: appending R cannot lower optimal population sensitive recovery; never sell it as removal | Stated as the study's own "inclusion fact" with the one-line proof, and used to frame the target as bounded **empirical** disclosure. Manuscript §9 states it and rejects any "appending R protected us" reading. | `METHOD.md` §1 |
| R2 Finite probes need explicit baseline candidates; bitwise J preservation ≠ equal probe scores | `ref_J` is re-audited **on the same host under the identical slate** and every increment is taken against it; the study states finite probes can move either way. | `METHOD.md` §1, §4; `VALIDATION.md` |
| R3 Residual reconstruction is a proxy; no residence/commute in fitting, selection or the gate | Proxy declared as such; the gate legs are reconstruction, source allowance and validation sensitive increments only. The test-split residence table is explicitly post-hoc and entered no gate. | `PILOT_GATE.md`; `RESEARCH_DECISION.md` §3 |
| R4 Deployed map must not touch out-of-fold targets, row ids, H_B or labels | Out-of-fold residuals are training-only; every non-training use takes the single final decoder; R is a function of A-side inputs through J's frozen standardiser. The PCA-residual control is likewise a deployable map, and the audited function is exactly that map. | `METHOD.md` §2, §4; parity assertions per unit |
| R5 Show increments over H and over J | Pilot increments are over J (the relevant baseline for an extension of J); the reference rows give A0 and LEACE-on-A0 over H in the same slate, so both readings are available. Manuscript prints the J-relative screen and names it as such. | `PILOT_SCREEN.json`; `PILOT_TABLES.json` |
| R6 Keep local/coalition/equal-opportunity roles; don't let a larger input buy apparent protection | Baseline views include J for local (H_A, Z_J) and coalition (H_A, H_B, Z_J); differentiable attackers are corrections to a frozen baseline, so the incremental gain is **zero at initialisation by construction** — an increase cannot come from the new input making the attacker fit worse at the starting point. | `METHOD.md` §3 |
| R7 Finite adversarial objective is not an MI bound | Stated in the study's non-claims and in the manuscript limitations; the baseline subtraction is noted to add no gradient and no benefit is claimed from it. | `METHOD.md` §3, §5 |
| R8 Machine-readable gate; cumulative counts; untriggered tiers labelled | Gate files per tier with inputs, counts, elapsed time and next action; the 42 pilot units are the first 42 of the 135-slot plan with 93 unrun; T3/T4 recorded as not started by the gate rule. | `gates/T*.json`; `TIER_STATUS.md` |
| R9 Utility-first criterion needs bounded upper bounds, not nonsignificance | The Tier-1 reanalysis applies exactly this form (residence improvement with an adjusted interval below zero **and** one-sided adjusted upper bounds ≤ .001 per endpoint, both weightings, m = 2480). The pilot never reached a nominee, so no interval was computed there, and the manuscript says so rather than implying screening thresholds are inference. | `TIER1_REANALYSIS.json`; `PILOT_GATE.md` |
| R10 Declare one correction level before fitting | Declared for all primary comparisons before fitting; no primary comparison was reached, so no level was applied selectively. The reanalysis explicitly does **not** reuse the old m = 3900 adjustment. | `PROTOCOL.md` §7; `TIER1_REANALYSIS.json` |

## Items that remain open (no artefact exists, because the tier never ran)
* LEACE-on-R with measured projection rank and per-class support: defined for Tier 3, never fitted.
* Any nominee-level comparison, its adjusted intervals, and a 2017 transport of the extension.
* An ordinary (non-coalition) extension counterexample test at nominee level; the pilot's C1-versus-L2 row is
  the only coalition-specificity evidence, and it is negative.

## One new mathematical caveat introduced by the executed study
Re-auditing an identical channel on different hardware refits the attacker slate, and that optimisation is
platform dependent: the deterministic parts reproduce to 5e-9 or exactly, but the refit moved a reference
audit by up to 0.0031 nats against a declared 0.003 tolerance. Consequence enforced in the manuscript: all
Study 6 comparisons use same-host references, and its numbers are not differenced against earlier studies'
published audit values at a finer precision.
