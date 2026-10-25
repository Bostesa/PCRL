# Mechanism cards (template; this file is not a card)

A card is committed, with the code it governs, before the first fit it governs (`rfa.common.card_info` refuses an uncommitted or
changed card; the card's path and SHA-256 are logged with every fitting read and stored in every release record). File names carry
the family prefix `RF_`. A change after a fit is a new card (a revision `_R1`), never an edit; a revision card names the observed
failure (from recorded development evidence), the changed operation, the prediction and the control that isolates the change.

Every card states: the starting release and why; the permitted inputs and the emitted outputs; the exact operation and its
computational graph (which derivative passes through the reader fit); the closest existing implementations and published methods and
the operational difference; the frozen recipe (episodes, reader families, solver, surrogate, guards, lambda rule, schedule, budgets);
the matched controls and what each isolates; the cost-matching rule and the recorded residual differences; the measurements; registered
predictions and mechanism-specific falsifiers; the failure interpretation; the specification count; the novelty status (no claim).
