# Role 2 implementation notes: CELL-A stored-model pilot runner

Written on 2026-10-02, before any real-data attacker or probe fit. Authority: `FROZEN_DESIGN.md` and Addendum D1.
No real-data fit was run. No assessment outcome was computed or opened.

## What exists now

| Item | Location |
|---|---|
| Effective protocol: frozen settings, unit list, primary family | `stored_model_eval/effective.py` (`EFFECTIVE_PROTOCOL`; the lock build writes it out as JSON) |
| Support freeze (100 / 30 / 100, NE vs UNRESOLVED) | `stored_model_eval/support.py` |
| Recipes: L, GBT, MLP, NL selection, LRT_A2 / LRT_A4, LO, N0 / N1 / G1 / G2 / ρ₁² | `stored_model_eval/recipes.py` |
| Unit runner: plan, dry-run, execute, resume, reuse, U1 / U2 | `stored_model_eval/pilot.py` |
| Inference and report tables | `stored_model_eval/pilot_infer.py` |
| Input builder: `task_labels_v1.npz` and 26 v2 manifests | `stored_model_eval/pilot_inputs.py`; real driver `scripts/build_inputs_v1.py` |
| Lock: `PILOT_LOCK_v2.json` build and verify | `stored_model_eval/lock.py` |
| CLI | `pilot`, `lock build\|verify`, `infer --units-dir`, `report --pilot-infer` |
| Shell runner | `scripts/run_pilot.sh` (`STAGE=lock\|run\|infer\|report\|all`; `EXECUTE=1`, `RESUME=1`, `UNITS=`) |
| Tests | 31 original tests, unchanged and passing, plus 21 new tests (`test_16_pilot_e2e.py`, `test_17_pilot_units.py`). 52 pass in about 45 s. |

## Small repairs to pinned code

- **`access.py` `ReleaseContract`.** Gained `from_manifest`, `release_count`, `seed` and a `persistent` property.
  - A `gaussian_noise` release with `release_count: "one"` maps to `persistent_token` (D1 #3).
  - The legacy `pipeline.fit_attackers` now takes the contract from the manifest. Before, it applied the global
    `noise="none"` (tested).
- **`admission.py`.** Applies per-role thresholds when a manifest declares a `support_rule` (D1 #11).
  `load_admitted` now carries the release block.
- **`forward.py`.** New `frozen_head_logits` (U1). It loads with `weights_only=True` and checks sha256.
- Legacy `attackers.py` GBT and MLP classes are unchanged. The pilot does not use them.

## Real inputs already built (non-fitting step)

Output: `~/PCRL_eval_cache_private/pilot_adult_s0/run_v1/inputs/`, holding `task_labels_v1.npz`, 26 v2 manifests
and `INPUTS_INDEX.json`.

- **`task_labels_v1.npz`.** sha256 `1e17ec9c…`.
- **`labels.npz` is untouched.** Its sha256 is still `367972fb…`.
- **Alignment check.** The task labels and 6 reference columns were taken from the b96c412 dataset object and joined
  to `labels.npz` by `row_id`. The reference columns were record key, sex, race, age_group, marital_status and
  income. All 6 had 0 disagreements over 15,060 rows.
- **Re-verification on every load.** Each v2 manifest declares a reference check, so admission repeats this join
  every time.

Checks already run (non-fitting):
- `pilot --plan` against the original directory selects exactly the 26 registered IDs from 152 manifests.
- The dry-run admits all 26 units.
- The frozen-head forward pass reproduces the stored logits for all 3 heads with max |Δ| = 0.
- Task-class support: occupation class 5 (3 / 1 / 1 rows by role) is excluded with a reason code.

## Interpretations to confirm (all recorded in `EFFECTIVE_PROTOCOL`)

1. **LRT_A2 eigenvalue floor.** The floor is 1e-6 · tr(T)/d, where T is the *released* class covariance. I did not
   use the trace of (T − σ²I), because that trace goes negative at large σ.
2. **G2.** The ρ grid is {0, 1e-4, 1e-2}, selected on attacker_val held-out R² (methodology R02). FROZEN_DESIGN is
   silent on this.
3. **LO smoothing.** Laplace α = 1.
4. **Normalised lift.** The 0.03 rule is applied to the clean-lift **point**. The methodology text says LCB; the
   D1 wording is ambiguous.
5. **Early-stopping settings.** GBT and MLP both use validation_fraction 0.1 and n_iter_no_change 10. MLP uses
   max_iter 200 and adam. The methodology text does not specify these values.
6. **NL on every surface.** NL uses the same GBT + MLP slate on all surfaces, a superset of R06N's GBT-only list.
   If GBT and MLP tie, GBT is selected.
7. **Macro OvR AUC.** Rows from unsupported classes stay in the negative set. Brier and log-loss use all rows and
   all classes.
8. **Decision boundaries.** P1 uses "≤ τ" for the below decision; P2 uses "< bar". An interval counts as NE when
   the share of non-finite replicates exceeds the tail probability.
9. **Key names.** `preds.npz` uses `P__<surface>__{L,GBT,MLP,NL,LRT_A2,LRT_A4}`, with `rep+outputs` stored as
   `repPLUSoutputs`. Role 1's coverage table calls the NL key `NLsel`. There are also extra keys: `RHO_u`, `RHO_v`,
   `s_prior_fit` and `t_prior_fit`.
10. **Reused outputs records.** Reused outputs-surface records in noise units carry the arm's contract. The source
    unit's contract is kept as `source_contract`, and the record is marked REUSED and "not new evidence".

## Calibration (`calibration.json`)

The estimate is synthetic, at pilot sizes: 15,060 rows, d = 64, output dimensions 2 / 6 / 4, and K = 2, 4 or 5.
It ran single-threaded through `run_unit` with the real `EFFECTIVE_PROTOCOL`.

Per-unit CPU time:

| Unit type | CPU seconds |
|---|---|
| Untreated, K = 2 | 25–33 |
| Untreated, K = 4 | 36 |
| Untreated, K = 5 | 27 |
| Noise unit | 19–22 |

Totals:
- Fits for all 26 units: about 642 s.
- Inference: about 54 s for 6 units (724 exploratory statistics plus the B = 20,000 primary run), scaled to about
  283 s for 26 units.
- **Total: about 0.26 CPU-hours against the 2 CPU-hour allowance.**

The full slate fits within the allowance, so no staged schedule is proposed. One caveat: synthetic signal can
shorten early stopping compared with real data. Even a 4× slowdown in fitting would stay under 1 CPU-hour.

## Lock note

The lock pins every `stored_model_eval/*.py` file (tests excluded) and every file in
`results/combined_stored_model_pilot_v1/scripts/`. Build it **after** the final code commit: any later edit to
either set, or to a manifest or data file, blocks `--execute-scientific-fits`.
