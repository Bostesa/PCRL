# Implementation ↔ claim alignment

Each row states a claim, what the code does, whether they align, and the repair. The code was read at the
pinned commits; the method-family detail is in METHOD_LINEAGE.md.

| # | Claim (where) | Code (commit) | Aligned? | Repair |
|---|---|---|---|---|
| A1 | "the corresponding API now refuses to run" (main.tex:149) | research branches from 5f162ab3: raises ✔. **Public origin/main 55e4cb1 and local main 0eee48f:** `certified_accuracy_bound` returns the invalid bound and states it as a theorem; `generate_report` calls `NonlinearComplianceCertificate.check`; `print_compliance_table` prints the bound | **No** on the default branch | `patches/0001-retire-accuracy-guarantee-on-main.patch` (minimal; separable; its tests fail before and pass after). Compatibility: callers in `experiments/{deployment_case_study,run_distribution_shift,run_extended_baselines}.py` on main will now raise, which is fail-closed by design; `ComplianceReport.nonlinear_bound` stays in the schema and is `None`. Applying it to main is the owner's decision; Terminal 3 did not push to main |
| A2 | "label-free code" (main.tex:191, :280) | T0 = quantile cells of logit p − logit b, with both teachers fitted on same_residence (encoding.py `fit_encoder`, `Codebook.fit` @ f4bdf4cd) | **No** | patch 0002 hunks `construct` and `baselines` |
| A3 | the release limits the recipient and the coalition (abstract) | the selected T0_L imposes A constraints only; AB CMI fitted 0.012–0.018 (MATH_REPLAY_FINAL) | partial | hunk `convex`; optional abstract edit |
| A4 | the constraint is on a coarsened partition (main.tex:211) | `ServicePartitions.fit(n_local=2)`: 2 KMeans cells of H_A[:,[1,3]]; AB = ×(H_B[:,1] > median) → 4 | yes, but under-specified | hunk `cells` |
| A5 | expected-loss scoring (main.tex:105-107) | `audits.py:167-180`, `finite.py:85-109`: Σ_z Q·loss | yes | — |
| A6 | attacks see continuous H; H ancestors; no J in views without J (main.tex:99-101) | `evaluation.py:269-270, 316-333` | yes | — |
| A7 | one sampled token (main.tex:103-105) | one token per person per anchor, reused (RELEASE_CONTRACT.md; lineage notes) | yes | — |
| A8 | a convex programme (main.tex:195-200) | CVXPY with CLARABEL, then SCS; HiGHS for zero or unconstrained budgets; statuses recorded (76 optimal, 49 optimal_inaccurate, 1 feasible_witness over 126 maps); no dual bound | yes (convexity); **optimality not certified** | hunk `convex` |
| A9 | DA = "worst single direction" (main.tex:129) | `compute_dominant_axis_r2` = max_k OvR in-sample ridge R² | **No** (a lower bound on the worst direction) | hunk `da_def` |
| A10 | the identity is validated (main.tex:132-133) | at 135e440e the aggregate ran in float32 and the per-class path in float64; the current code casts both to float64 (test `test_repository_identity_holds_for_float32_input_after_cast`) | the historical numbers carry a precision artefact; the current code is fine | hunk `da_val` |
| A11 | absent classes (audit) | the current code returns NaN and `coverage_complete=False` for missing declared classes (tested) | yes | — |
| A12 | the encoder line's "shared frozen backbone" (main.tex:340; NeurIPS §4.1 "pretrained") | `run_v2_dataset.py run_seed` @ dbe0fdc and 0eee48f: seeded random `StandardEncoder`, frozen by `PerPurposeLoRAEncoder` (lora.py:182); BN buffers left at defaults | **No** for the NeurIPS wording; main.tex is silent | hunk `lineage` plus a corrections item |
| A13 | P4 rank floor applies to the adapters (NeurIPS §5.2) | LoRA hooks every `nn.Linear` in the backbone (lora.py @ dbe0fdc) | **No** | corrections item (hunk `corr4`); fixture F12 |
| A14 | multiple recipients get optimised channels | no implementation; B always receives H_B | n/a; do not claim it | CORRECTED_CONTRIBUTIONS |
| A15 | the encoder line and the release line are one algorithm | no code path composes them | **No**; main.tex App. A already says "not an algorithm" ✔ | keep |

## Tests added (Terminal 3's owned paths)

`tests/pcrl_claims_foundation_v1/test_claims_foundation.py` (10 tests):
- the exact fixtures;
- the repository's own `_empirical_ridge_fit` and `compute_dominant_axis_r2`, loaded from source by AST,
  satisfy the identity to 1e-12 on matched rows (1e-9 with float32 input);
- both fail closed on a missing declared class.

These tests check the implementation against an independently derived identity rather than restating it.
The main-branch patch carries its own 4 tests, 3 of which fail on unpatched main.
