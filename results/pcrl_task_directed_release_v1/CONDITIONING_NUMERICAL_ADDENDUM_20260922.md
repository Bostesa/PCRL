# Finer-conditioning numerical incident — 2026-09-22

At 05:29:51 UTC the registered optional finer-conditioning unit `Trisk_L_0.002_a17_fineC`, anchor 2, failed the original independent acceptance check. CLARABEL reported a solver error; SCS reported optimal but its raw row-sum error was 1.988526323604134e-7, above the unchanged 1e-7 tolerance. No map was accepted and no audit fit began for that unit. This is neither a disclosure finding nor evidence of mathematical infeasibility.

The V4 controller stopped and cancelled three C sibling attempts, preserving their files. The A scheduler continued unchanged. Two sibling C maps had already been accepted; they are retained, and interrupted audits will restart deterministically with prior partial artifacts preserved. A serial prefit worker now completes only unattempted C solves and does not rerun a saved failed solve or any accepted map.

The original protocol's bounded Branch D permits one corrected equivalent formulation. A separate hash-pinned retry receipt will authorize exactly one normalized-entropy call for each failed C map, using the existing tested identity m·rel_entr(a/m,b/m)=rel_entr(a,b). Every original and finer constraint, reporting weight, cost, row support rule, solver setting and acceptance tolerance remains fixed. No accepted C map is re-optimized for utility. Any rejected bounded retry remains incomplete.

The specific retry source, registration, preserved original hashes, acceptance and replay receipts will be published separately. At this addendum's writing no such retry has executed. Evaluation remains unopened. See [CONDITIONING_NUMERICAL_INCIDENT_20260922.json](CONDITIONING_NUMERICAL_INCIDENT_20260922.json) for exact observed diagnostics.
