# Final mathematical replay — 2026-09-22

All 126 scheduled maps passed independent mathematical replay across three anchors, including evaluation encodings. The actual fitted cost/joint-table aggregation and deployment parent identities reproduce the coarse-to-refined embedding.

| Check | Maximum error/count |
|---|---:|
| maximum_cmi_excess | 1.64213223269e-13 |
| simplex_error | 6.66133814775e-16 |
| negative_mass_error | 0 |
| objective_error | 0 |
| decoder_error | 2.22044604925e-16 |
| refinement_aggregation_max_error | 6.93889390391e-17 |
| refinement_deployment_parent_mismatches | 0 |
| local_coalition_ordering_warnings | 0 |
| out_of_support_evaluation_count | 0 |

Solver status counts: {"feasible_witness": 1, "optimal": 76, "optimal_inaccurate": 49}. These are retained solver reports. Feasibility replay does not establish an independent optimum certificate. Approximate/feasible candidates receive qualified ordering diagnostics; no population impossibility follows.

The full source records, including every CMI value and residual, support counts/ESS and H parity checks, are `MATH_REPLAY_FINAL_anchor0.json`, `MATH_REPLAY_FINAL_anchor1.json`, and `MATH_REPLAY_FINAL_anchor2.json`. All recorded warnings and exact source hashes are in `MATH_VALIDATION_FINAL_20260922.json`. Exact zero-budget algebra remains separately documented in `ZERO_PRIVACY_CERTIFICATES.json`; toy separation remains a distinct result.
