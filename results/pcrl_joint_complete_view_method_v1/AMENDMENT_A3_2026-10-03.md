# Amendment A3 (2026-10-03, descriptive diagnostic only; no fit, no locked code changed)

**What happened.** The outer audit called FARE's native certificate on the cert role. The official wrapper refused for every selected tree (`NATIVE_VS_AUDIT.csv`: UNAVAILABLE): "476 certificate rows are identical to fit rows". Different people share identical permitted-input vectors.

**Diagnostic.** `report/fare_certificates_a3.py` (not a locked file) recomputes the certificates under the record-identity guard used by the earlier study's amendment A1.
- It asserts that cert and fitting records are disjoint by role construction.
- It then clears the feature-hash guard.
- Output: `FARE_CERTIFICATES_A3.json`.

**Result.**

| Tree | Result |
|---|---|
| F income tree | OK but **vacuous**: bound 1.728 > 1, the maximum possible DP distance |
| F occupation tree; both F0 trees | UNAVAILABLE |

**Effect.** No registered decision uses certificates. `NATIVE_VS_AUDIT.csv` keeps the original wrapper status (UNAVAILABLE), and this file records the descriptive recomputation. `LOCK.json` is not changed, because no locked code was modified.
