# Mechanism cards (template; this file is not a card)

A card is committed, with the code it governs, before the first fit it governs (`ddr.common.card_info` refuses an uncommitted or
changed card; the card's path and SHA-256 are logged with every fitting read and stored in every release record). File names carry
the family prefix: `DA_` (designer A), `DB_` (designer B), `CTL_` (shared controls), `FS_` (a replacement family). A change after a
fit is a new card (a revision `_R1`, `_R2`), never an edit.

Every card states:

1. **Operation**: inputs (the six permitted fields only at runtime), fitted statistics (learner, settings, role and labels of each),
   the formulas that produce the released probability vectors and decisions, and the code entry point.
2. **Closest previous implementation** (path, campaign) and **what is substantively different**.
3. **Measured failure addressed** (with the record).
4. **Why usefulness may be preserved** (with any label-free feasibility number, disclosed as such).
5. **Twin**: the equally trained task-only release (same learners, data, folds and selection effort; protection off).
6. **Strength grid and selection rule** (MODEL_FIT out-of-fold or nested data only; CALIBRATION task labels only if stated).
7. **Matched controls**: the same-form SEX-blind control path (SBC) and the SEX-blind spend path (SPEND), both bracketing the
   candidate's spend, with equal tuning effort; the ablation of the strongest feasible release.
8. **Revisions**: at most two, each with the observable failure that triggers it and its stop rule.
9. **Registered predictions** (numbers), configuration count and CPU estimate.
10. **Novelty status** (closest published method; claims only with evidence).
