# Mechanism cards (template; this file is not a card)

A card is committed, with the code it governs, before the first fit it governs (`sar.common.card_info` refuses an uncommitted or
changed card; the card's path and SHA-256 are logged with every fitting read and stored in every release record). File names carry
the family prefix: `FA_` (investigator A, starting release SUCC), `FB_` (investigator B, starting release NEURAL_RAW), `CTL_` (shared
controls), `FS_` (a replacement family). A change after a fit is a new card (a revision `_R1`, `_R2`), never an edit. Cards are short
and prospective: they fix what will be done and how a failure will be read, not an optimistic forecast.

Every card states:

1. **Starting release and binding condition** (with the record: `anchors/ANCHOR_TABLE.md` and the source records).
2. **Operation**: inputs (the six permitted fields only at runtime), fitted statistics (learner, settings, role and labels of each;
   which MODEL_FIT folds train teachers and which check them), the formulas that produce the released probability vectors and
   decisions, the deployment function and the code entry point.
3. **Closest existing implementation** (path, campaign) and **the operational difference** (the structural change if the idea is
   related to closed work).
4. **What evidence would contradict its rationale.**
5. **Bounded choices**: strength grid and the MODEL_FIT-only selection rule (out-of-fold or nested held-out losses; CALIBRATION task
   labels only if stated, then for the controls too); component budgets chosen from the starting release's actual margins.
6. **Matched controls** (PROTOCOL section 6): the unchanged START; the equally exposed continuation CONT if attribution needs it; the
   simplest untargeted operation CTRL on its own grid, with the two MODEL_FIT bracket points and how cost is matched; any declared
   mechanism-specific reader (at most one, reported beside); for a family declaring joint design, the local twin and both sequential
   orders, or the statement that no joint claim is made.
7. **Measurements** and **failure interpretation** (what each outcome would mean, including an output that never changes).
8. **Revisions**: at most two, each with its trigger; the diagnosis, mechanism change and predictions of a revision are recorded in its
   own card before its fit.
9. **Registered predictions** (numbers with probabilities), specification count and CPU estimate.
10. **Novelty status** (closest published method; no claim without separate review).
