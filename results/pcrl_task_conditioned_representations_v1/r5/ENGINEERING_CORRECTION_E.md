# Nomination admission and literal utility gates correction E

Dated 2026-10-12 UTC. This corrects the implementation before any new INNER task result is opened and preserves the original scientific requirements. R00 owns the shared amendment and final nomination lock.

The utility worker previously accepted a requested checkpoint hash when it appeared anywhere in serialized nomination JSON. That was insufficient to establish its nominated family and source seed. It now requires exactly one canonical `instances` association for the requested arm ID and integer source seed, then checks that association's exact checkpoint hash. Unrelated hashes, wrong arm IDs, wrong seeds and duplicate associations fail. R3 confirmed the final nomination builder uses `instances:[{arm_id,seed,checkpoint,checkpoint_sha256,...}]`; preselected fallback IDs remain distinct from nominees.

Original point utility caps now use literal `value <= cap`. The previous 1e-12 tolerance was not part of the frozen protocol. This correction removes that allowance without changing any cap, role, endpoint, selection rule or model. A fixture checks that the cap itself passes and its next representable larger value fails.

Root ran the combined D/E RED and GREEN suite through the shared queue. The corrected suite passed all 18 tests, including exact association and literal-boundary fixtures. These are admission and numerical correctness checks. They are not Adult utility or privacy findings.
