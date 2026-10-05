# Strength-matched feedback study (development)

**Result: EXPERIMENTAL_NO_ADVANTAGE.**
- At matched update strength the critic schedule made no measurable difference.
- The AUC-driven local controller engaged but gave no measurable benefit.
- The candidate J-F never qualified as a nominee.
- The strongest arm was the faithful raw-penalty joint arm (RAW-J, β 0.3), packaged as the best development baseline: coalition SEX AUC 0.810 against 0.882 for task-only training, at equal accuracy.

**Start here.**
- `RESEARCH_DECISION.md`: the five answers, tables and next decision.
- `ADVISOR_BRIEF.md`
- `PROTOCOL.md`: rules and registration, including §11.
- `METHOD_CARD.md`
- `METHOD_DELTA.md`
- `MATH_REVIEW.md`
- `QUICKSTART.md`: tested commands.
- `VALIDATION.md` and `INDEPENDENT_VERIFICATION.json`
- `COST_AND_CLOSEOUT.md`

**Data.** Already-exposed Adult rows. NEW_DEVELOPMENT_ASSESSMENT is 20% of the previous fitting groups, sealed at load until EVALUATION_LOCK.json was pushed. It is not fresh confirmation (`EXPOSURE_LEDGER.md`).

**Code.** `smf/`, with pinned imports from `rgj`/`jcv`. Private units, checkpoints and per-row predictions are in `<PRIVATE_CACHE>/smf_v1/`, with a drive copy listed in `RESTORE_INDEX.json`.
