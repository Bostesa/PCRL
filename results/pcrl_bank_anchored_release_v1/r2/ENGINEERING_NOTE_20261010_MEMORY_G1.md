# R2 protection queue memory correction, 2026-10-10

The F1 bank identity seed-zero unit completed and its public receipt is
preserved. The remaining 15 F1 units were queued but did not start because
the shared queue's available-memory gate required more headroom than the
declared 4 GiB per unit allowed at the observed occupancy. This is a resource
admission issue, not an Adult method outcome.

The G1 successor specs use a 3 GiB maximum RSS and new `_G1` unit IDs for
the 15 remaining arm/seed fits. F1 B0 seed zero remains the matched identity
unit. The frozen method source, seed recipe, allowed data, utility proxy,
controls, 300 CPU-second optimizer ceiling, and 600-second queue reservation
are unchanged. Existing F1 configs/specs and queue records are preserved.
R0 must seal this dated resource change before G1 specs or fits run. A unit
that actually exceeds the 3 GiB bound fails and is reported as incomplete;
the bound must not be silently raised after seeing fit outcomes.
