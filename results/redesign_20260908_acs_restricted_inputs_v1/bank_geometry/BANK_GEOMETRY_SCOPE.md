# Bank geometry coverage supplement — 2026-09-08

This completes the requested affine diagnostics for **all final released interfaces**, including the six source-only banks. The original study's source banks publish only three source-task probabilities. Their completed training, utility, audits, native predictions, composition witnesses and immutable unit artifacts are unchanged.

The supplement fits exactly **18 affine decoders**: E-bank and S-bank, seeds 0/1/2, each predicting the original-standardized raw PCA16, retained E, and removed qE=raw−E targets. Every input is the existing published three-probability bank. No hidden 16-coordinate bank mapper output is read. There is no direct coordinate error between three bank coordinates and 16 teacher coordinates, and no new experimental condition or outcome-driven choice.

Fit float64 least squares with a fitting-only intercept on the original representation-fitting rows, with fixed `rcond=1e-12`. Use the existing immutable E map and original PCA16 fitting scales. Targets contain no residential/commute or other outcome labels. Preserve the mature fitting-prior reference, target variances, ranks/spectra, per-coordinate errors, and undefined ratio policy: fitting/evaluation target variance and evaluation fitting-prior MSE must each exceed `1e-12`. No decoder, rank tolerance, target, checkpoint or weight is selected using validation/development performance.

Execution order is E-bank seeds 0/1/2, then S-bank seeds 0/1/2, with targets raw/E/qE in that order. Freeze this source, configuration and every required input hash before fitting. Fit all 18 decoders and freeze every coefficient before extracting held-out arrays for source-validation and the existing **DEVELOPMENT EVALUATION**. Save fitted coefficients locally, with compact hashes and scores for publication. An independent verifier will replay the rectangular solves and scores.

Run locally with one numerical thread only in the parent's assigned scientific slot. Charge its full process wall time to the same 60-minute scientific ceiling and original four-hour work ceiling. This bounded coverage completion does not change the original protocol or invalidate/rerun any completed representation, head or attack.

Low affine error supports recoverability from the actual bank interface; high error does not exclude nonlinear recovery. qE is not pure sensitive information. These diagnostics do not replace task utility, attribute audits, composition witnesses or the exact coordination stream.

Preparation and execution:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/acs_restricted_bank_geometry.py --prepare
/usr/bin/time -p env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/acs_restricted_bank_geometry.py --run
```

Both stages create fresh artifacts and reject overwriting completed evidence.
