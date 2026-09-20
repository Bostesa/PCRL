# REPRODUCE — `pcrl_utility_extension_v1`

Environment: Python 3.13.7, numpy 2.4.2, torch 2.10.0 (CPU), scikit-learn 1.8.0, scipy 1.17.1,
pandas 3.0.1, joblib 1.5.3, threadpoolctl 3.6.0, concept-erasure 0.2.4; one BLAS/OpenMP thread per
worker. The cloud run used Ubuntu 24.04 x86-64 on one m7i.2xlarge with 6 worker processes.

1. **Layout.** The audit code resolves inputs through absolute paths, so the host reproduces the
   original worktree layout at fixed commits (see `infra/pcrl_utility_extension_v1/bootstrap.sh`):
   this branch, plus the main checkout, the invariant study `73903b7f2`, and the residual-spectral
   worktree `349efa454`.
2. **Inputs.** Restore the execution bundle (`exec_main`, `exec_rs`, `exec_inv`) with
   `infra/pcrl_utility_extension_v1/restore.py`; it verifies every file's SHA-256 against the manifest.
3. **Run.** `python -m experiments.pcrl_utility_extension_v1.scheduler` with `PCRL_UX_RUNCFG` and
   `PCRL_UX_OUT` set (the systemd unit in `bootstrap.sh` shows the exact environment). Gates are
   written to `$PCRL_UX_OUT/_scheduler/gates/`. Completed units are reused by marker, so a rerun
   resumes rather than refits.
4. **Single unit, no scheduler.** `pg.fit_job(seed, 'X_r2_C1_b030', ext.Config())` then
   `pg.audit_job(seed, 'X_r2_C1_b030')` (`program.py`), after `pg.references(seed, ext.Config())`.
5. **Tables.** `PILOT_SCREEN.json` is produced by `program.screen`; the descriptive test-split table
   by `infra/.../pilot_tables.py`; the reanalysis by
   `python -m experiments.pcrl_utility_extension_v1.tier1_reanalysis <INTERVALS_X.csv> <out.json> <out.md>`.
6. **Determinism.** Seeds are fixed formulas of (anchor, width, policy) and are listed in `PROTOCOL.md`
   section 3. Results reproduce bitwise on the same platform; across CPU architectures expect the
   deviations quantified in `VALIDATION.md` (log loss ~5e-9 for fixed models, ~0.003 nats for refits).
