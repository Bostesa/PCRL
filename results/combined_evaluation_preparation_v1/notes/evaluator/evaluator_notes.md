# Evaluator notes (2026-10-02)

Package: `stored_model_eval/` at the worktree root (pure Python; numpy/scipy/sklearn; torch only inside
`forward.py`, imported lazily). No real-data attacker or probe was fitted. No network was used; the CLI
installs a socket guard. Row-level arrays and model outputs were written only to
`~/PCRL_eval_cache_private/` (outside git) and session scratch.

## Module map

| module | role |
|---|---|
| `metrics.py` | NOT_ESTIMABLE sentinel; PCRL one-hot ridge R2 (fit=score default, held-out option, float32 replay option); OLS R2; R02 relative-ridge held-out R2 (methodology); per-class (dominant-axis) R2; rho1^2 (in-sample and held-out); macro/per-class/worst-class/pairwise AUC with support and coverage; label-only reference; Brier skill; log-loss reduction; health (flagged scale-dependent) |
| `inference.py` | union-find units (unit id OR record key); paired cluster bootstrap with seeds as refit replicates; bootstrap-of-max statistics; permutation null re-selecting the max; decision rule (no "pass") |
| `admission.py` | manifest: sha256, explicit ID arrays, reorder/mismatch/duplicate rejection, reference join check, roles, units crossing roles, per-role class support, PENDING drive members |
| `surfaces.py`, `access.py` | rep / outputs / rep+outputs; access tags A1-A5; release contracts; surface x attacker x metric decomposition (sequential + Shapley + order range; refuses cross-scale subtraction) |
| `attackers.py` | linear, GBT, MLP (val log-loss selection); noise LRT on population clean vectors with known Sigma (A4); adaptive attacker on defense-simulated releases (A2); repeated-release averaging attacker (A3(N), valid only for fresh noise) |
| `forward.py` | independent re-implementation of the PCRL v2 tabular encoder + LoRA + task heads from tensor shapes and embedded config; sha256 check before load; weights_only first; strict keys; refuses erase-layer checkpoints; cache must be outside git |
| `recount.py` | probability-array recount (stored held-out probabilities); PCRL strict counts from committed JSON via `git show` |
| `plan.py`, `cli.py`, `config.py`, `guards.py`, `fixtures.py`, `pipeline.py` | CLI, cost model, protocol loader (own schema or the methodology `protocol_config.json`, translated), guards, synthetic fixtures, fit/score/infer/report, C1-C5 classifier |

## Defects found in reused / historical code (reported, not fixed in place)

1. `combined_evidence_reconciliation_v1/.../F_common.py::worst_pair` returns `0.0` when no pair is supported
   (a numeric value that reads as "no leakage"); uses `orientation_free=True` by default (max(a, 1-a), biased
   upward under the null); support is `>= 10` rows on the pair, not per class.
2. `F_common.py::macro_ovr` silently averages present classes when a class is absent (no coverage, no flag).
3. `F_common.py` hardcodes `OUTDIR` into the other worktree (`combined-evidence-reconciliation-v1`) and `INV`
   into `/Volumes/YOTUO` (not mounted): rerunning it from elsewhere writes into a foreign worktree.
4. `D_common.py::per_dim_std_and_eff_rank` computes effective rank from `s/sum(s)`; PCRL
   (`run_v2_dataset.py@b158abc54:153`) uses `s^2/sum(s^2)`. Values are not the PCRL quantity.
5. durable-guarantees `utils/pcrl_io.py` loads the backbone with `strict=False` and `weights_only=False`; a
   missing key would be silently ignored. (Our forward pass is strict; all keys matched.)
6. PCRL `pcrl/data/adult.py@b96c412::_load_data`: with `download=False` and missing files it silently
   generates synthetic fallback data; our scripts refuse if `adult.data`/`adult.test` are absent. Race codes are
   assigned per split (`sorted(unique)` within the split) - harmless on Adult today (all 5 races in both
   splits) but a latent misalignment for subsets.
7. PCRL `LinearComplianceCertificate` computes in the input dtype (float32 for torch representations), not
   float64; our `r2_onehot_ridge(dtype=np.float32)` replays it, default is float64.
8. The historical probability arrays (`analysis/tpr59_scores`, `tpr_ext_scores`) carry no row IDs and no
   units: they are fine for recounts (y is co-stored per draw) but cannot pass the new admission rules for new
   analyses (no ID alignment, no cluster bootstrap by person).

## Recount findings

* AAAI audit: 64 / 59 / 51 failing at 0.52 / 0.55 / 0.60 from the stored probabilities (max |dAUC| vs stored
  1.7e-5; 64/64 files match the drive inventory sha256). The 67 configurations contain only **64 distinct
  measurements**: E2 sigma=1.0/2.0/8.0 and E4S1 sigma=1.0/2.0/8.0 are the same files with identical AUCs. Counted
  once, the failures are **62 / 57 / 51 of 64**. The single multiclass row (HMDA race sigma=8) sits on the 0.55
  bar on macro AUC (XGB 0.5494, MLP 0.5511) but its supported worst-class AUC is 0.64-0.67 (all 5 classes
  >= 126 test rows).
* PCRL NeurIPS: 56/60 (final.pt) and 54/60 (best.pt), same under `<` and `<=`; Adult flips are s0
  employment_analysis/age_group and /marital_status. Input sha256 equal to VERIFICATION_REPORT A1.
* Forward smoke: the cached checkpoint is **Round 4** (state epoch 204, best_epoch -1), the encoder
  durable-guarantees audited; it is NOT a NeurIPS Round-5 headline checkpoint (those are drive-only, PENDING).

## Deviations from / questions for the methodology protocol (coordinator to reconcile)

* Role name `assessment` and shares 0.50/0.15/0.35 are honoured via the config translator; the pilot role
  assignment is a hash of the record key, **not** stratified by (s, y) as `roles.stratify` asks.
* Macro AUC: methodology says "mean over supported classes, report coverage"; this package's default is
  NOT_ESTIMABLE with the supported mean as a labelled partial. The translator switches to the methodology rule.
* Interval: methodology alpha 0.05 one-sided = two-sided 90 percent; translated to `bootstrap.alpha = 0.10`.
* R02 held-out R2: SS_tot around attacker_fit means (methodology) vs around assessment means gives different
  numbers (synthetic n=400: 0.068 vs 0.058, i.e. ~0.01 at tau 0.05). Both implemented; must be fixed in the lock.
* Linear floor: methodology drops eigen-directions < 1e-6 x largest; rho1^2 here uses eps = 1e-6 tr/d as
  specified to the evaluator. R02 uses the methodology floor.
* Pair support: methodology requires support in attacker_fit AND assessment; this package checks the scored rows
  only (attacker_fit support is visible in the admission role summary).
* GBT grid: translator maps lr x max_leaf_nodes (9 configs, max_iter 500, no early stopping); the methodology's
  20 random configs incl. min_samples_leaf / l2 and early stopping are not implemented. MLP learning_rate_init not
  mapped.
* Not implemented: Holm adjustment, R05 multinomial-LR label-only (a frequency-table reference is), R10, R13,
  R14 coalition attacker, utility probes U1/U2, the F0-F6 factorial's "rows" axis (the decomposition is generic
  over 3 axes; adding rows in-sample vs held-out as a 4th axis is a small change).
* Adult race in the pilot assessment role has classes of 61 and 40 rows (< 100): race per-class / pair
  statistics will be NOT_ESTIMABLE (coverage 3/5 classes, 3/10 pairs) unless the lock pools categories or
  changes n_min.
