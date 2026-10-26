# Independent verifier, wave-1 hand-back (verbatim; delivered to the coordinator at about 21:58Z; harness indentation removed)

# RFA1 verifier hand-back 1 (wave 1, label-free, invented data only)

**Verdict.** The `rfa/fam/rfit.py` derivative is correct. On the re-solved bilevel objective, the FULL gradient matches central finite differences to 2.2e-10 relative error, and the verifier's own analytic hypergradient agrees with the module to 1.4e-15. STOPFIT is exactly the fixed-w direct path. ALTCRIT is a direct path through a lagging Adam critic, not an alias of STOPFIT. The `rfa/fam/fit.py` wiring is correct: the pre-clip gradient reaching the optimizer equals ∇CE + λ∇S bitwise in all four arms. The prepared later-wave code (a)-(c) reproduces the campaign evaluator and reader grid exactly on invented data.

- **Checks:** 26/26 derivative checks pass, 19/19 wiring checks pass, and the (a)-(c) tests pass.
- **Defects:** three minor ones (D1-D3) plus three interpretation notes (N1-N3).
- **Hard rules:** no label, feature file, model or private array was opened. No campaign code or git state was changed.

## Files checked (SHA-256)

| File | SHA-256 | Note |
|---|---|---|
| rfa/fam/rfit.py | `04b22c664caa8c55c5341bfd02e3b8604a6a17fe82c619f84e40a235f70336a7` | All derivative checks ran on this version; unchanged during the runs. |
| rfa/fam/fit.py | `9c5577b8d0080035530274c8c38051ef0d93ce5ad9cb9f4dd3653be18eb6ec44` | Final probe ran on this version (current, 17:49); unchanged during the run. |
| rfa/fam/net.py | `1f08e9c71f91e63a5d10b8744b19e372c089ee47e336d4b5ab0221b2aa93ebe6` | |
| rfa/devreaders.py | `4046bf1d0b7ffcab9f05f705d5815d047d8997c390b22014fb421f22b723782e` | Equals its entry in DEVGRID_CODE_LOCK.json (lock rewritten 21:43:47Z). |

- **rfit.py history:** I first read `a2cf78a5…`. It was superseded at 17:30 by the current version, which changed views to occupation + pair and set n_A 4096 / n_E 2400.
- **fit.py history:** I read `7efbab03…`, then `4c03432d…`, then `b7112a64…`. The first probe run used `b7112a64…`.
- **devreaders.py history:** I first read `7935c3e1…`. Its docstring had the bootstrap seed as 2026102400; this was corrected before the re-lock.
- **Governing definitions read** (prefixes; full hashes in out/V1_PREPARE_TEST.json): rfa/evaluate.py `20ee7103`, ohn/evaluate.py `8e72de6a`, ihs/evaluate.py `03f449e2`, nrm/evaluate.py `e9180a84`, scm_bench/selection.py `87de258a`, scm_bench/common.py `d8e1c8a7`, scm_bench/readers.py `3d9838e7`, ffv/readers.py `fcad58f2`, ffv/features.py `2e858562`, task_conditioned_representation/attacks.py `bdcabb20`, inference.py `b0342990`, rfa/readerextra.py `8fabe7d1`.

## 1. Derivative check of rfit.py (verifier's own code)

**Setup.**
- The objective is the verifier's own float64 numpy implementation: view inputs, A-row moments, c·tanh soft bound, SIMPLE and RFF features, binormal AUC, R and S.
- The RFF frequencies were regenerated from the recipe text and are bitwise equal to `rff_params`.
- The inner fit uses the verifier's own damped Newton solver, polished to the floating-point floor (max|grad| 4e-17).
- Tiny release: an MLP 6→8 (tanh)→5 for the occupation logits (101 parameters). Income logits are constants.
- Invented SEX and households. The episode is household-separated: A has 402 rows (159 households), E has 400 rows (168 households), with zero household overlap.
- Recipe: the registered `RECIPE_RF` (occupation + pair, simple + rff, ridge 1e-2, warm start on).
- J(θ) re-solves w* at every perturbed θ, with the A moments frozen at θ0 (the module detaches them).
- The module was only called: `EpisodeObjective.objective`, plus `fit_reader`, `rff_params` and the basis helpers, each compared against my own versions.

| Check | Result | Tolerance | Pass |
|---|---|---|---|
| Forward value: module P vs verifier J0 = 0.1487024 | diff 2.8e-17; per-term R 5.6e-17; module w* vs verifier w* 8e-15; helpers 5.6e-15 | 1e-12 | yes |
| FULL vs central FD of re-solved J, all 101 coordinates | rel L2 err 2.2e-10 (h 1e-5), 2.0e-8 (h 1e-4, h² scaling); max abs 2.0e-11 | 1e-6 | yes |
| Same, all three views | 2.2e-10 | 1e-6 | yes |
| Verifier's own analytic IFT gradient (one Newton step from detached w*) vs module FULL | rel 1.4e-15 | 1e-10 | yes |
| STOPFIT vs FD of R(w fixed at w*(θ0), E(θ)) | rel 1.8e-10 | 1e-6 | yes |
| STOPFIT vs re-solved J (should not match) | rel 0.65 | > 1e-2 | yes |
| U0-size float64 nets (14,629 params, n_A 513 / n_E 512), 3 random directions | FULL rel err 2.1e-10, 4.5e-10, 7.1e-10; STOPFIT 1.4e-9, 1.2e-10, 2.6e-10 | 1e-5 | yes |
| Optimization sign: steps of length 1e-4, 1e-3, 1e-2 along −g/‖g‖ | J decreases each time; actual/predicted 1.00001, 1.0001, 1.0010 | ratio within 1% at 1e-4 | yes |
| FULL vs STOPFIT differ | see list below | fit/direct > 0.3 in designed case | yes |
| Income view, SIMPLE, θ-dependent income: fit path is exactly zero | ‖fit‖/‖direct‖ 4.7e-17 | 1e-8 | yes |
| Gradient w.r.t. A-row outputs | FULL 0.0176; STOPFIT and ALTCRIT exactly 0 | | yes |
| ALTCRIT first call vs STOPFIT | bitwise equal | | yes |
| ALTCRIT second call (new episode, 10 Adam rounds), direct path vs FD with critic frozen | rel 1.4e-10 | 1e-6 | yes |
| Verifier's own Adam replication vs module critic | max abs 3.3e-16 | 1e-12 | yes |
| Module `critic_rel_lag` vs verifier | diff 3.3e-16 (lags 0.23-0.54) | 1e-9 | yes |
| ALTCRIT vs STOPFIT on the same episode (lag, not alias) | rel 0.29 (cos 0.963) | > 1e-3 | yes |
| Alias claim: ALTCRIT with `altcrit_inner='newton'` = STOPFIT | warm start off: bitwise over 3 calls; registered warm start on: rel 2.2e-14 / 2.8e-15, not bitwise | rel 1e-8 | yes, qualified |

FULL vs STOPFIT, by case:
- **Base case:** cos 0.802, ‖fit‖/‖direct‖ 0.613.
- **Designed to differ** (n_A 98, RFF, pair view): cos 0.493, ratio 1.088. FULL still matches FD (5.1e-10); STOPFIT is 0.95 off.
- **U0-size nets:** cos 0.783, ratio 0.699.
- **Campaign-size invented episode** (n_A 4097 / n_E 2402): cos 0.863, ratio 0.612.

**Moment path (informational; detached by design, RECIPE_RF `moments_convention`).** If the A moments are recomputed at each θ, the FD differs from FULL by rel 0.075 (cos 0.997). The omitted moment path is 11.6% of the fit-path norm and 7.1% of the direct-path norm.

## 2. Wiring of fit.py

I read the file and ran a dynamic probe of `arm_paths` for 3 steps per arm on invented data: 3,104 rows, 1,250 households, random U0-architecture float32 networks, the registered recipe except n_A 600 / n_E 400. `torch.optim.Adam`, `clip_grad_norm_`, the episode draw, `objective_parts` and `fit_reader` were wrapped to record what happened, and everything was then recomputed independently. All core checks pass:

- **Only the occupation network trains.** The optimizer holds exactly its 8 tensors in every arm. Income layers are bitwise START at every checkpoint, and the occupation layers move in every arm.
- **Income logits are constants.** `z_inc` has no graph. Every objective call gets income logits without grad and occupation logits with grad. SEX and income rows are bitwise aligned with the episode rows.
- **Same episodes.** FULL, STOPFIT and ALTCRIT get identical, household-separated A/E sequences. My own implementation of the documented rule and the `RFA1-episodes` seed reproduces them, and the calibration episodes too.
- **Same λ.** One calibration on FULL with 16 draws gives λ = 0.68866, shared by FULL, STOPFIT and ALTCRIT; TASK has λ = 0. My independent recomputation of λ gives a relative difference of 0.0.
- **TASK uses no episode and no SEX.** Its trainer has `sex=None`, it draws no episode and makes no objective call.
- **Derivative passed to the optimizer.** For every arm and steps 0-2, the pre-clip gradient equals ∇CE(task rows) + λ∇S bitwise (rel err 0.0). The task rows came from my own replication of the `THP1task` stream, and ∇S from a fresh objective sequence. The clip is total norm 5.0; it was inactive in the probe because the norms stayed below 5.
- **FULL's separate parts.** g_dir + g_fit equals ∇S (covered by the bitwise check above).
- **Mechanism statistics.** `mechanism_stats` matches my own cross-group estimator (diff < 1e-10), and `mechanism_gate` leaves the weights unchanged.

**Defects** (current fit.py `9c5577b8`):

- **D1 (minor, accounting).** fit.py:320 records `inner_reader_fits = n_episodes × views × families`. This is exact for FULL and STOPFIT (12 = 12 in the probe) but undercounts ALTCRIT.
  - In the probe, ALTCRIT made 16 exact `fit_reader` calls against 12 recorded: its first call does both the critic initialization (rfit.py:247) and the lag diagnostic (rfit.py:263).
  - ALTCRIT's 80 Adam rounds appear only in the text note, not in the integer.
  - The calibration's 64 exact fits and the gate fits are not counted in any arm record.
- **D2 (minor, robustness and docstring).** fit.py has no `ReaderRefused` handling anywhere: not in `calibrate` (fit.py:182), `mechanism_gate` (fit.py:241) or `run_arm` (fit.py:279/288).
  - A refusal propagates and aborts the whole unit without writing a FIT record.
  - rfit.py:25-27 and :91 say "the training loop records it and stops".
  - ALTCRIT can now also be stopped by a refusal of its diagnostic-only exact solve (rfit.py:263), which never enters its gradient.
- **D3 (cosmetic).** fit.py:420 stores `FULL_logs_every_10` as `logs[::4]` of 10-step logs, which is every 40 steps.

**Fixed during the wave** (seen on `b7112a64`, resolved in `9c5577b8`):
- `fit_path_share` read a key rfit never writes. On the first probe it returned None while the logged share was 0.3619; the fix is confirmed (fit.py:350).
- `objective_counters` was always None; it is now replaced by `inner_reader_fits`.
- The short-E pool edge case: a guard was added at fit.py:146-147.

The first probe JSON was overwritten by the corrected rerun; the failed values are the ones recorded above.

**Interpretation notes** (not defects):

- **N1.** FULL differentiates through the fit of the logistic weights only. The A-row standardization moments are also fitted on A but are detached (rfit.py:64). That omitted path is about 12% of the fit path on invented data. Wording should be "derivative through the reader's weight fit".
- **N2.** The alias statement at rfit.py:24 is exact mathematically but bitwise only with `newton_warm_start` off. The registered ALTCRIT uses Adam and is not an alias, so it stays a gating control and its artefact cannot be reused.
- **N3.** On invented data, the F0 gate (16 episodes, 4 groups) labelled the mechanism INERT (orthogonal share 0.046, cos 0.9994). Yet the single-episode FULL/STOPFIT cosines were 0.78-0.86, so the per-episode fit path can be largely non-systematic. This is invented data only; no inference is drawn.

## 3. Prepared later-wave code (tested on invented data only)

- **(a) The 28 usefulness conditions.** Compared against `rfa.evaluate.utility_of` (→ `ohn.evaluate.complete_eval` → `ihs.evaluate.utility`).
  - Test data: invented labels, two versions, four release types: deterministic all-pass, 3-draw randomized, argmax ties with zero true-class probability, and known failures.
  - Values and SEs agree with max abs diff 0.0; pass flags, failure lists and se_mc are identical.
  - G1 agrees and is True on the all-pass release (28 conditions).
  - Values are a difference of means, as in `scm_bench.selection.clauses`.
- **(b) Reader grid recomputation.** The campaign's `run_bank` was run on invented releases: cell S0R1k0, 2 members, 96 recipe fits.
  - My code reproduces household folds (R1, R2 and LC tags) bitwise, plus reader roles, reader seeds and cache keys.
  - Winners from the stored VALIDATION AUCs match 6/6, including near-tie and gap.
  - Winner and per-recipe SCORE AUCs, the B = 400 bootstrap draws, and the paired differences and SEs all match with max abs diff 0.0.
- **(c) Bounded refit with sklearn.** My own feature tables (including the CATI design), scaler and the 16 recipe definitions.
  - 21 distinct refits reproduce the cached oriented SCORE scores bitwise (21/21), and the VALIDATION and SCORE AUCs are identical.
  - Library versions sklearn 1.8.0, numpy 2.4.2 and scipy 1.17.1 equal the grid lock.
- **Later-wave entry point.** `v2_verify.py` (`usefulness`, `grid` and `refit` commands) is syntax-checked only and not run.
  - It refuses to start outside a queue unit, and reads labels only through `rfa.common.load(..., 'verify')` from its own `__main__`.
  - It checks release and frozen-bank hashes itself, and checks member wires against the grid provenance.

## Data boundaries and CPU

- **Guard.** An audit hook (`vguard.py`) refused any open under `.private` or `.worktrees`, and of model or grid file names outside `out/invented_grid`. It refused 0 opens apart from its own self-check probe.
- **Writes** went only to `roles/verifier/`.
- **CPU:** about 33 CPU-seconds, one thread, all outside the queue, so it needs charging. This includes one early graph run that crashed with KeyError 'income/simple' after the recipe's views changed. That was an adaptation of my code to the new recipe, not a failed check, and an all-three-views variant was kept.

## Files

All in <worktree>/results/pcrl_reader_fit_aware_release_v1/roles/verifier/:
- out/V1_GRAPH_CHECK.json
- out/V1_FIT_WIRING.json
- out/V1_PREPARE_TEST.json
- out/invented_grid/ (invented arrays only, 332 KB)
- code/: vguard.py, v1_graph_check.py, v1_fit_wiring_probe.py, v1_prepare_test.py, vlib.py, vcore.py, v2_verify.py
