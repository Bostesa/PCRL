# Post-run clarifications to PROTOCOL.md (2026-10-17, from the independent verifier; no number or decision changes)

`PROTOCOL.md` is unchanged and matches ANALYSIS_LOCK `protocol_sha256` (`d19d364d…`). These clarifications are recorded here so the frozen protocol keeps its locked bytes.

1. **Claim B's DEV utility gate (verifier D2).** Section 9 named the DEV 24-point utility gate only under claim A. The frozen `report.py`, whose hash is in the pre-fit ANALYSIS_LOCK, also applies it to claim B, together with MODEL_SELECT eligibility. That follows the inherited source convention (`acs_hybrids/report.py` lines 283–284, claim B "fails_gate"). The decision document reports the label from the frozen code and, separately, what the competitive clauses alone show.
2. **Numerical conventions in `engine.py` (verifier D6).**
   - Admissibility allows a slack of 1e-15.
   - The binary KL clips rates to [1e-15, 1 − 1e-15]; the clip was inactive in the verifier's tests.
   - The constant `CAP` in `engine.py` is unused, and its accuracy value of 0.010 is not the 0.005 budget, which `budgets()` sets correctly.
   - Absolute plug-in costs use global teacher constants rather than per-cell sums. This is equivalent, because the constants cancel in every change.
3. **Table labels (verifier D3 and D4).** The per-seed DEV feasibility flag covers that seed's 12 clauses. Secondary endpoints are descriptive differences, so only their intervals and difference statuses are meaningful; the clause fields carried in the JSON for them are not.
