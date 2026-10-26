# Verifier interim message 2 (verbatim; delivered to the coordinator at about 22:29Z)

Verifier: both queued units are small, well under 10 CPU-minutes each, and neither will hold back the R1 fits.

Expected run times:
- RFA_VER_GRID_v1 (about 1-3 min CPU): reads the saved RFA1_MAIN outputs (GRID.json, BOOT.npz, cache entries) for both versions; no reader fits.
- RFA_VER_REFIT_v1 (about 1-2 min CPU): 12 single-recipe refits = pair winners of RF_FULL, RF_STOP_HI and NEURAL_RAW in S0R1k0, S0R2k1, S1R1k0 and S1R2k1 (MLP64 / ET25 / CATI1-type recipes on about 4,000 FIT rows).
- After them comes one light unit, RFA_VER_DECISIONS_v1 (under 1 min): label-free D1-D4 comparison plus a frozen-rules replay.

R1 plumbing check (label-free, invented data, about 1.7 s CPU outside the queue): rfa/fam/fit.py d1c9efb5... with rf_module = rfit_r1 (24c2bd99...) passes every core wiring check.
- Every objective call (calibration, gate, FULL / STOPFIT / ALTCRIT) is the R1 class with families simple, rff, softbin.
- Lambda is shared by the three episode arms and reproduced exactly; TASK has lambda 0, no episode and no SEX.
- The pre-clip gradient equals grad CE + lambda x grad S_R1 bitwise in all 4 arms (steps 0-2).
- Episodes are identical across arms and household-separated; only the occupation network is trained; income is bitwise START.
- The v1 path with rf_module None is unchanged by this diff. The v1 fits ran on fit.py 9c5577b8 (HEAD 43830ae19), the version I verified.

Notes, not blockers:
(1) D1 (ALTCRIT inner-fit undercount) applies equally to R1.
(2) A new OccTrainer built with an R1 recipe but without rf_module fails loudly (rfit refuses the softbin family); stage_fit sets rf_module correctly.
(3) On invented data with 16 episodes in 4 groups, the R1 F0 gate read MECHANISM_INERT with cos_tot_dir = 1.016 > 1 and orthogonal share clipped to 0.0. The unbiased cross-group estimates can exceed their logical bounds, so the INERT label can come from estimator noise. The registered 64 / 8 gate is less noisy; this is information only, and I am not proposing a rule change.
