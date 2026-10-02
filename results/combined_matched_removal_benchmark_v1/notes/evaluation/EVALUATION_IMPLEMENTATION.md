# Evaluation implementation (role 3), 2026-10-02

This note records how the evaluator implements `BENCH_DESIGN.md`. The code is the authority:

- `stored_model_eval/bench_effective.py`: the frozen `BENCH_EFFECTIVE` protocol, the panel, the unit lists and the primary family.
- `stored_model_eval/bench.py`: the runner.
- `stored_model_eval/bench_infer.py`: inference.
- `stored_model_eval/bench_report.py`: report tables and calibration.
- `stored_model_eval/bench_lock.py`: `LOCK.json`.
- `scripts/run_benchmark.sh` and `scripts/plots.py`.

Before the lock, no real-data eraser, attacker or probe was fitted. On the real inputs only `--plan` and `--dry-run` ran
(no fits).

## Interface as consumed

- **Inputs.** The runner reads role 1's `INPUTS_INDEX.json` (schema `bench_v1.inputs_index/v1`). See
  `INPUTS_CONTRACT_CONSUMED.md`.
- **Defenses.** It calls role 2's `defenses.py`:
  - `concat_marginal_onehots` and `fit_leace(…, fit_row_ids, concept_spec, auth, synthetic)`;
  - `LeaceMap.transform/save/load/native_check`, and `alias_test`;
  - `noise_release` and `scale_report`.

  The runner refuses if the settings the map was fitted with differ from the protocol's official defaults (`svd_tol` 0.01,
  shrinkage on).

## Decisions taken in code (coordinator decisions marked [C])

1. **Support** [C].
   - Evaluation support is per role: 100 / 30 / 100 rows per class on attacker_fit / attacker_val / assessment. It is
     identical for every arm of a cell and is frozen in `<private>/support/SUPPORT_FROZEN.json` at lock build, from labels and
     roles only.
   - An eraser concept (B: target; C: policy set) is fitted on the full declared one-hot only if every class of every concept
     attribute has at least 100 defense_fit rows. Otherwise that unit is written as NE, with no fit.
   - Real data: HMDA race scores classes {0, 1, 2} (3 pairs). Classes 3 and 4 are NE, and their coverage is reported.
2. **B/C native check** [C].
   - The primary status is the official implementation bound `implementation_bound_holds`: whitened residual
     ≤ `svd_tol`. A failure is C1.
   - Exact-zero cross-covariance (`crosscov_max_abs_rel_erased`, tolerance 1e-6) is a descriptive column. Its status
     `OUTSIDE_TOLERANCE_SVD_TOL_TRUNCATION` is not C1.
   - An independent evaluator-side recomputation is stored next to it.
   - So is the label-free out-of-support component on attacker_fit and assessment rows: the transfer limitation of a
     rank-deficient defense_fit covariance.
3. **U1** [C]. The frozen head is Linear-ReLU-Linear, so U1 is labelled outside LEACE's scope. U1 refuses if the head
   applied to the clean representation disagrees with the stored logits by more than 1e-4.
4. **Arm A native check.**
   - N0 is the historical one-hot ridge R² on the whole test split (`split == "test"`), in mixed precision and in float64.
   - It is compared per seed against `dominant_axis_audit.json` `r2_onehot`, read from pinned git blobs (Adult
     `76f485a…`, HMDA `e56c885…`). REPRODUCED means an absolute difference ≤ 1e-4.
5. **Alias.** C aliases B iff `defenses.alias_test` finds max|P_B − P_C| ≤ 1e-10 and max|mean diff| ≤ 1e-10.
   - The C unit is then written as an alias record: no fits, and `alias_of` set.
   - Inference scores it once and labels its rows "alias". The family stays at 24.
6. **Shared work, fitted once and recorded as an alias in every consumer.**
   - LEACE maps per (dataset, seed, purpose, concept), cached under `defenses/`. Each is appended to `MAP_PINS.json` and
     verified on every reuse and by the lock verifier. An unpinned map directory is refused.
   - `outputs_only` per (dataset, seed, purpose, attribute).
   - U2 per (dataset, seed, purpose, release).
7. **C_rep_plus_clean_out slate.**
   - NL = {GBT, MLP on [rep, clean logits], ignore_rep, ignore_out}.
   - L = {L on the concatenation, ignore_rep (outputs_only L), ignore_out (C_rep L)}.
   - All candidates are scored on the same attacker_val rows and chosen by log loss.
   - A selected ignore candidate aliases the source model's predictions and attacker seeds.
8. **Attacker seeds.**
   - The selected recipe is refit at random_state 1 and 2. Seed 0 is the grid fit (random_state 0).
   - L / lbfgs and the LRTs are deterministic and are aliased (recorded).
   - Saved keys are `P__<surface>__<recipe>__as<k>`.
9. **Noise.** The draw is `defenses.noise_release` over the scored-role rows (attacker_fit ∪ attacker_val ∪ assessment)
   in ascending row_id order, with `default_rng(release seed)`. It is persistent, and identical across attributes of a
   purpose. The sha256 of the release is recorded.
10. **σ\*** (`--sigma-star`).
    - Reads only `val_preds.npz` (hash-checked) of the Tier-1 noise units.
    - Statistic: mean over encoder seeds × release seeds of the attacker_val C_rep NL (attacker seed 0) macro AUC over
      the frozen supported classes. σ\* is the smallest σ with a value ≤ 0.55; if none qualifies, σ\* = 8, flagged.
11. **Inference.**
    - One cluster bootstrap stream per dataset over `assess_unit`. Each cell regenerates the identical draws, so pairing
      across arms holds.
    - Every endpoint is the mean over encoder × release × attacker seeds, computed within each replicate.
    - Worst class / pair: simultaneous per-component bounds at α/K, and LCB/UCB(max) = max of the component bounds.
    - Primary: 24 endpoints, one-sided bounds at 0.05/24, B = 20,000, seed 20261004.
    - Exploratory: 90% intervals, B = 2,000, seed 20261003.
12. **Tier-2 trigger and stop rule.**
    - Trigger (technical validity only): every admitted Tier-1 unit is COMPLETE and hash-verified; A is REPRODUCED;
      B/C are PASS; U1 is consistent; `SIGMA_STAR.json` is present.
    - Stop rule, checked before each Tier-2 unit in the fixed order E1 → E2 → E3: stop if
      spent CPU + projected(unit) + inference reserve (30 CPU-s per completed-or-started unit) > 8 CPU-h.
    - The projection is the measured mean CPU of completed units of the same dataset and arm. Until one exists, the
      calibration estimate is used.
    - The run never skips ahead to a cheaper unit.
    - Interpretation: "projected remaining cost" is read as the cost of the next unit plus the inference it implies. It
      is not the cost of finishing all of Tier 2; that reading would stop before E1 whenever E3 is large.
13. **Shuffled-label sanity** (`--shuffled-label-sanity --units …`).
    - Labels are permuted within attacker_fit and within attacker_val (seed 20261005).
    - The C_rep slate is fitted and selected through the same code.
    - It is reported on attacker_val only and is flagged above 0.55. Assessment rows never enter.
14. **Writes.** All writes go outside git (`require_outside_git`). Units are checkpointed with `COMPLETE.json`.
    `--resume` skips hash-verified units and moves partial ones aside; it never overwrites them.

## Pilot compatibility

Pilot modules are not modified, apart from the additive `bench` subcommand in `cli.py`. The 52 pilot tests pass.

Adding files to `stored_model_eval/` makes the pilot's `PILOT_LOCK_v2.json` verifier report "code file added". Pilot
replay must therefore use the pilot commit `661db8d` (unchanged).

## Calibration (synthetic, real role sizes, full frozen grids, OMP_NUM_THREADS=1; not science)

Details are in `calibration.json`. These are estimates, not measurements of the real run.

**Run cost:** 389 CPU-s user (444 s wall), peak RSS 1.24 GB.

**Per-unit CPU (s)**

| Dataset | A (incl. shared outputs_only) | B | C | D (mean) |
|---|---|---|---|---|
| Adult | 31 | 16 | 17 | 21 |
| HMDA | 53 | 20 | 21 | 32 |

Inference costs about 17.5 CPU-s per unit, at exploratory B = 2,000 plus primary B = 20,000. The protocol reserves
30 CPU-s per unit.

**Projected CPU-hours**

| Stage | Projected CPU-h | Cumulative CPU-h |
|---|---|---|
| Tier 1 (126 units; 0.93 fits + 0.61 inference + 0.02 sanity) | 1.57 | 1.57 |
| E1 | 0.96 | 2.53 |
| E2 | 0.37 | 2.90 |
| E3 (648 units) | 8.07 | 10.97 |

**What fits the 8 CPU-h budget**

- Tier 1 fits comfortably, so no reduced schedule is needed.
- E1 and E2 are projected to fit.
- E3 does not fit. The frozen stop rule ends it at a unit boundary, in the fixed order dataset → pair → seed → σ →
  release seed. The expected outcome is that Adult E3 mostly completes and HMDA E3 is largely not run. The real stop
  point depends on the measured per-arm costs.
