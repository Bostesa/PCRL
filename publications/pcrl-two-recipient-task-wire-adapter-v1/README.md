# PCRL two-recipient task-wire adapter (v1): public package

This is the public code, aggregates, locks and claims for a closed negative study. Person-level data, checkpoints, per-person predictions and deployment packages are private.

- **Start with** `results/START_HERE.md`, then `results/RESEARCH_DECISION.md`.
- **Result:** `POINT_FEASIBLE_NO_COMPETITIVE_CRITERION`. Adapting both recipients' predictions (65 income + 325 occupation parameters) keeps the original point utility. Coalition recovery falls only 0.0036 [−0.0025, 0.0097] AUC against RAW_PROJ, an established shortfall against 0.020. Joint training does not beat local or sequential training, and changing income does not help.
- **Prefit receipt:** `prefit/` was published in the first commit of this branch, before any bank fit.
- **Code:** `code/two_recipient_adapter/` is the campaign, `code/two_recipient_adapter_independent/` is the verifier, and `code/inherited/` holds the reader slate, models and reporting helpers. The numerical workers need the campaign queue and the private inputs, so they will not run from this package alone.
- **Exposure:** exploratory development on repeatedly used Adult people; the design came after the previous occupation-only results. This is not confirmation evidence, and there is no population privacy guarantee.
