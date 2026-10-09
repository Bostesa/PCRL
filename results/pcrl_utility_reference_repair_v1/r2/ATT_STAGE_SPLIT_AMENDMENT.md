# ATT execution-stage split — 2026-10-09

The frozen downstream proposal has 12 fresh complete-view ATT units: eight
recipient views and four coalition views. Every coalition unit reads the
corresponding two recipient reader indices and prediction arrays so it can
include exact ignore-one-recipient aliases. Those files do not exist when the
recipient units are registered.

The durable queue requires every input file and SHA-256 to exist at enqueue,
and its sealed registry is immutable. Therefore the 12 proposed scientific
units execute under two consecutive seals:

1. q18 registers the eight local recipient ATT units after q16 complete
   release/raw custody and q17 independent release replay pass. It binds each
   q16 producer file to its exact SHA-256. q18 reads ATTACK_FIT_NEW features
   and SEX plus INNER_SELECTION features; it never reads INNER_SELECTION SEX.
2. q19 registers the four coalition ATT units after all eight q18 units are
   DONE. It binds local reader outputs to their actual exact SHAs and retains
   the frozen coalition ignore-one-reader contract.
3. q20 registers two seed-level INNER_SELECTION SEX score units after all
   twelve ATT units and controls are frozen.

This is an execution and custody amendment only. It does not add or remove a
pair, view, reader, control, fit, score, selection rule, data role or endpoint.
The proposal's 86 new ATT/control fits remain 14 canonical RAW_ONLY fits in
q16 and 72 complete-view fits in q18/q19. The separate 44 FIT SEX search fits
were already charged in q13. No scientific outcome informed this split.

The coalition worker must validate each ignore-one-recipient reader against
the archived q18 **local** science lock, while writing its own index under the
current q19 coalition lock. A required `local_attack_source_lock_sha256`
field makes those two identities explicit. The prediction files are still
verified by exact SHA-256 and complete-view/person identity before aliasing;
the separate lock field does not permit a new reader fit or a weaker match.
