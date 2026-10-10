# A source admission finding

R03 verification unit `R04-A-PREFIT-SOURCE-001` completed with a technical
pass. Its aggregate receipt is `A_PREFIT_INDEPENDENT_REPLAY.json` (SHA-256
`2f5fed77088d665aeac2be0147b73cbe9bba4f09e4b15807e5c4e78c85e19b8d`).
R04 independently reconstructed both seeds' qpc fine-cell routing from
admitted U0 scores on FIT, ATT and INNER; the fixed cell-to-token policy;
and H-GLOBAL-TEMP output bank from frozen prototypes and decoder alpha.
Every routed cell and token matches the role-only export, both bank tables
equal decoder `q` bitwise, temperature replay has maximum difference zero,
and each bank action has a unique public decision/probability identity.
No A channel was fitted, no assessment or HEAD role was read, and this
receipt does not assess A privacy or held-out utility.

The deterministic T* identity map passes the INNER point utility caps
against U0 and Ucal on both seeds. It fails the original-reference U0 FIT
occupation NLL cap in both seeds: excess `0.014312010568` for seed 0 and
`0.015420555695` for seed 1 against the `0.010` cap. Seed 1 FIT occupation
Brier excess is `0.005348076600` against the `0.005` cap. The FIT identity
passes Ucal and all income comparisons. These are data-dependent findings
about the fixed identity map, not a revised cap or an A method outcome.

**Admission conclusion:** source, role, bank and decoder provenance is
cleared for an A-stage seal. The identity channel is not a feasible FIT
starting point under all original utility caps. A fitted channel must meet
those unchanged caps. R04 computed separate optimistic, label-aware loss
minima over same-class bank actions; they lie below the caps, so the bank
does not give a simple impossibility certificate. Those minima use task
truth and may select different actions for NLL and Brier, so they do not
establish that an implementable single channel is feasible. Channel
feasibility and A scientific success remain unproven before fitting.
