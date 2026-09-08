# Focused validation

No broad historical suite or research rerun was performed.

- `.venv/bin/python -m pytest -q tests/test_acs_pca16.py tests/test_acs_pca16_comparisons.py`:
  **10 passed in2.62s**, two existing torch.jit deprecation warnings.
  [Recorded outcome](FOCUSED_TESTS.json). The miniature fixture actually fits
  all20 candidates across seven suites on artificial data, with fixture-only
  shortened training. Checks cover exact original-coordinate slicing, immutable
  copies, no PCA fit, fitting-only standardizers, fixed9 race schema, test-data
  rejection/selection gates, unchanged historical flags and original-parent
  paired criteria. Full scientific schedules were not shortened.
- New-only [score replay](SCORE_REPLAY.json): **60 candidates,240 score
  dictionaries,18,339 numeric comparisons**, no errors; maximum difference
  **2.22e−16**. All120 saved-model probability sets reproduce bitwise. Verified
  21 exact slices, three component orders,60 standardizers,21 primary and48
  family selections,27 MLP curve/schedule records and57 referenced file hashes.
  Replay took3.285s and did not fit models or replay historical model scores.
- The scientific runner itself replays the original PCA transformation against
  each cached pool, verifies original artifact hashes and matching split/fitting
  rows, and checks maps/outputs/source artifacts/selection immutability after
  evaluation. Evaluation starts after all seven suite choices are saved.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/verify_acs_pca16.py \
  --out results/redesign_20260908_acs_pca16_v1
```

The verifier requires local artifacts. It prints a fresh report by default;
`--report` accepts a new file and refuses to overwrite existing evidence.
Original maps, metrics, selections and source hashes are retained unchanged.
