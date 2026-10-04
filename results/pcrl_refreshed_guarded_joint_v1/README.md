# Refreshed guarded joint protection (development study)

**What this is.** A focused attempt to keep the accuracy recovered by removing mandatory erasure while repairing two
measured problems of the predecessor (research/pcrl-penalty-no-erasure-v1 @ b628ff5): the training critics fell behind
the encoder, and joint protection improved the pair partly by making one recipient less protected. The candidate J-G
periodically freezes the features so strong critics can relearn them, trains against those refreshed critics, raises a
recipient's protection weight when its own protection slips past a local budget, and is nominated only if it keeps useful
accuracy on both tasks and is locally no worse than the comparators.

**Result: EXPERIMENTAL_NO_ADVANTAGE.**
- Both registered claims are NOT_ESTABLISHED, and J-G had no feasible nominee on any seed.
- The strongest control was the joint arm with the inherited online critics (J-O).
- Refreshed critics kept pace with fresh bounded critics but did not lower independently audited recovery.
- The local multipliers never engaged.
- J-G kept task-only accuracy and lowered coalition recovery relative to the refreshed local control by 0.012 [0.006, 0.018], below the 0.02 target. See `RESEARCH_DECISION.md`.

**Data status.** Already-exposed Adult rows. DEVELOPMENT_ASSESSMENT is a new development partition (70% of the old
head-validation pool), sealed until `EVALUATION_LOCK.json` was pushed; it is not fresh confirmation.

**Start here.** `RESEARCH_DECISION.md` (verdict) · `ADVISOR_BRIEF.md` (one page) · `PROTOCOL.md` (registered rules) ·
`METHOD_CARD.md` · `METHOD_DELTA.md` · `QUICKSTART.md` (tested commands) · `VALIDATION.md` and
`INDEPENDENT_VERIFICATION.json` (checks) · `COST_AND_CLOSEOUT.md`.

**Code.** `rgj/` (new package; predecessor `jcv`/`pnx` code is imported pinned, never edited). Private units, predictions,
critics and checkpoints live outside the repository (`~/PCRL_eval_cache_private/rgj_v1/`, drive copy listed in
`MODEL_MANIFEST.json` / `RESTORE_INDEX.json`).
