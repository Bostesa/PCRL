# Mechanism cards (template; this file is not a card)

A card is committed, with the code it governs, before the first fit it governs (`ipr.common.card_info` refuses an uncommitted or
changed card; the card's path and SHA-256 are logged with every fitting read and stored in every release record). File names carry
the family prefix: `IA_` (designer A: before the release), `IB_` (designer B: the joint release map), `CTL_` (shared controls),
`FS_` (a replacement family). A change after a fit is a new card (a revision `_R1`, `_R2`), never an edit.

Every card states:

1. **Operation**: inputs (the six permitted fields only at runtime), fitted statistics (learner, settings, role and labels of each),
   the formulas that produce the released probability vectors and decisions, and the code entry point.
2. **Closest previous implementation** (path, campaign) and **what is substantively different**.
3. **Measured failure addressed** (with the record).
4. **Why usefulness may be preserved** (with any label-free feasibility number, disclosed as such).
5. **Twin**: the equally trained task-only release (same learners, data, folds and selection effort; protection off).
6. **Strength grid and selection rule** (MODEL_FIT out-of-fold or nested data only; CALIBRATION task labels only if stated).
7. **Matched controls** (PROTOCOL section 4), each bracketing the candidate's MODEL_FIT spend with equal tuning effort: the same-form
   SEX-independent-attribute path (SBC), the SEX-blind spend path (SPEND, dithered quantization of the twin) and the release-only
   path (RO: the operation with every person-level SEX statistic replaced by its counterpart estimated from the twin's released
   vectors, or the generic release-only tilt when the operation is release-only already); the ablation of the strongest feasible
   release; for a family declared jointly designed, the separately designed recipients and both sequential orders (D5), or the
   statement that they are not built.
8. **Revisions**: at most two, each with the observable failure that triggers it and its stop rule.
9. **Registered predictions** (numbers), configuration count and CPU estimate.
10. **Novelty status** (closest published method; claims only with evidence).
