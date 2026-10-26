# Verifier interim message: R1 derivative check (verbatim; delivered to the coordinator at about 22:27Z)

Verifier, interim result: the R1 objective (rfa/fam/rfit_r1.py, SHA-256 24c2bd99..., unchanged during the check) PASSES all 11 derivative checks, so the R1 fits are clear to start from the derivative side.

Setup: my own numpy softbin implementation (centres from scipy norm.ppf, not statistics.NormalDist), a tiny float64 MLP release (101 parameters) with constant income logits, invented SEX, household-separated A (402 rows) / E (400 rows), the registered R1 recipe (occupation + pair; simple + rff + softbin). Moments and decision scales were frozen at theta0, as the module detaches them.

- Complete FULL gradient vs central FD of the re-solved bilevel J: rel error 6.3e-10 at h = 1e-5 (6.2e-8 at h = 1e-4, h^2 scaling); max abs error 4.0e-11.
- softbin terms alone vs FD: rel 1.4e-9.
- STOPFIT vs FD at fixed w*: rel 3.3e-10. STOPFIT vs the re-solved J: rel 0.62 (as designed). cos(FULL, STOPFIT) = 0.817; fit path / direct = 0.59.
- Sign: steps of 1e-4 and 1e-3 along -grad both lower J; actual/predicted 1.000004 and 1.00003.
- Forward value vs mine: 3.3e-16. Feature maps vs mine: 6.1e-16. Decision scales vs mine: < 1e-13. Bin centres vs scipy: < 1e-14. cond(H) for softbin about 530-560.
- R1 with families ['simple', 'rff'] reproduces v1 (rfit) BITWISE for FULL, STOPFIT and ALTCRIT over two consecutive calls (max abs diff 0.0, S values identical).

Disclosure: my first run failed the feature-map check (9.7e-9). Cause: my test harness passed torch.tensor(python float), which is float32, as the decision scales; the module's own path is float64. I fixed only my test input; the check was not loosened. The failed run is kept as out/V2_R1_CHECK_run1_HARNESS_FLOAT32_SCALE_FAILED.json beside the passing out/V2_R1_CHECK.json. CPU about 4.6 s outside the queue. The rest of wave 2 continues; my verify units RFA_VER_GRID_v1 / RFA_VER_REFIT_v1 are queued behind the RFA1_ROB grids.
