# Independent mathematical review

Checked 2026-09-21 against the current `PROTOCOL.md` and `CONFIGS.json`. No ACS outcomes were accessed; no reviewed implementation files were changed.

## Defect requiring correction

**SPLINCE numerical rank collapse is mislabeled as mathematical infeasibility.** In `baselines.py`, `_fit_one` computes an intersection dimension from truncated SVD ranks and immediately returns `status='infeasible'` (reviewed lines 232–241). The existing near-intersection construction, changed to `x[:,1] = noise + 1e-11*(a*b)`, has nonzero determinant `4.0816329907770244e-13` for the two-coordinate sensitive/task covariance vectors. Thus those vectors are linearly independent and simultaneous erasure/preservation is algebraically feasible. Nevertheless, the routine returns `infeasible`, intersection dimension 1. With coefficient `1e-9`, it returns the appropriate `numerically_unresolved` status.

Fix: a numerical rank refusal should remain `numerically_unresolved` unless an independent algebraic contradiction is certified. An exactly duplicated protected/task label supplies such a certificate, but an SVD threshold alone does not. This was sent to the parent and baseline author; they own the correction.

## Material audit limitation, already partly registered

The HistGB correction in `audits.py` scales a nominal minimum of 5/20 by the maximum positive token support. For a dense 17-token channel, thresholds become 85/340 expanded rows. The distinct-person lower bound is valid, but in a token-specific leaf each person contributes only one row, so these settings require 85/340 distinct people rather than 5/20. Tiny positive channel probabilities can trigger the same large multiplier.

A controlled 60-person, constant-H, 17-token example with each person's label-indicating token having probability .99 confirmed this consequence. Both corrected thresholds produced only one-node trees and expected CE `0.6931471805599451`. Threshold 5 produced five-node trees and expected CE `0.013676641182314582`. This is a synthetic sensitivity check, not a proposed replacement analysis or survey result.

The installed sklearn source also shows that HistGB's bin mapper receives `X` without `sample_weight`; histogram thresholds and its optional expanded-row subsampling are therefore influenced by token support counts even for tiny-probability rows. The loss itself still uses the correct weights. Report this limitation and avoid interpreting a common nominal candidate list as equal effective tree capacity across deterministic and stochastic channels. A prospectively approved supplementary candidate could fit token-specific HistGB models on original-person rows with weights `w_i Q_i,z` and evaluate their exact token expectation. No unregistered candidate was added during review.

## Checks with no defect found

- Exact expansion conserves each original person's weight. Expected log loss is averaged over token-conditional predictions, not computed from an averaged prediction.
- The balanced person weights have mean one. The installed sklearn LBFGS implementation bases penalty strength on the sum of sample weights; expansion preserves that sum, so it does not silently weaken logistic regularization.
- MLP minibatches divide weighted token losses by original-person batch size; optimizer steps and epoch exposures count original people. Token expansion does not multiply the Adam weight-decay schedule.
- Checkpoints use balanced validation risk, preserve the earliest epoch on ties, and continue the actual live trajectory for the 360-epoch extension. The short selection is retained in the longer family.
- Candidate selection applies the same balanced criterion and uses the same selected predictions for both report weightings. Full-schema class padding is followed by a declared uniform probability floor, normalization and validity checks.
- The support-space LEACE and SPLINCE formulas, map orientation, task covariance constraints, and disclosed identity extension outside empirical covariance support agree with the intended finite-moment implementation. Numerical tolerance remains a qualification on guardedness.
- Caller-owned split disjointness and accessible-ancestor candidates are required by the protocol but deliberately outside these modules; their absence here is not reported as a defect.

The current audit/baseline test suite passed: `40 passed in 2.22s`. Those tests do not cover the newly reported `1e-11` rank-collapse case.

## Exact supplied fixture replay

`INDEPENDENT_FIXTURE_REPLAY.json` records a fresh execution using the supplied standard-library fixture. Both file hashes match its manifest. The generated JSON is exactly equal both structurally and byte for byte to the supplied result.

- Manifest SHA-256: `1f8d93502db30b95843b72943988c7ba4ad27fae22f652ecf362e8f2922ade32`
- Reproduced and supplied result SHA-256: `4316fa41eb499d49518074c34e2869927b7c66090e01cd7cb1fb52be72b80c8e`

The fixture supports only its exact constructed separation; it makes no empirical ACS or priority claim.
