# Mechanism cards (template; this file is not a card)

A card is committed, with the code it governs, before the first fit it governs (`drf.common.card_info` refuses an uncommitted or
changed card; the card's path and SHA-256 are logged with every fitting read and stored in every release record). File names carry
the family prefix `DR_`. A change after a fit is a new card (a revision `_R1`, `_R2`), never an edit; a revision card names the
observed failure (from recorded development evidence) and the actual changed operation, its controls and its predictions.

Every card states: the finite input representation and permitted assignment information; the emitted probability vectors and
decisions; the recovery objective and the usefulness constraints; the legal mapping class; the changed ingredient and the comparison
that isolates it; the bounded optimizer, support choices, revisions and specification accounting; registered predictions; the failure
interpretation; the novelty status (no claim without separate review).
