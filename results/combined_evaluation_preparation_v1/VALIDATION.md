# stored_model_eval: validation (2026-10-02)

Machine-readable version: `validation_results.json` (written by `notes/evaluator/build_validation.py`).
Package: `stored_model_eval/` (worktree root). Suite: **26/26 pass, 2.1 s wall** (incl. interpreter start),
deterministic, CPU, `OMP_NUM_THREADS=2`. Every number below comes from synthetic fixtures or from stored
artifacts. No attacker or probe was fitted on real data.

## Required regression tests

| # | test | result | key numbers | what it guards |
|---|---|---|---|---|
| 1 | positive control (S planted in H) | pass | macro AUC 0.999 [0.998, 1.000]; ESTABLISHED_ABOVE at 0.52/0.55/0.60 | recovery pipeline end to end |
| 2 | null control | pass | linear 0.502 [0.435, 0.572], GBT 0.539 [0.477, 0.599]; UNRESOLVED at 0.52 and 0.55, ESTABLISHED_BELOW at 0.60; vocabulary has no "pass" | a non-significant result is never "pass" |
| 3 | XOR signal | pass | max per-column correlation z = 1.5; in-sample ridge R2 0.014, held-out -0.033; rho1^2 < 0.01; linear AUC 0.480, GBT 0.999 [0.998, 1.0] -> C3 under linear-scope guarantees (also when population=True); the same recovery in scope -> C4 | recovery outside linear scope is C3, not C4 |
| 4 | minority-class contrast (K=30) | pass | pooled R2 0.023, mean per-class R2 0.025 (both pass tau 0.05), dominant axis 0.385, **rho1^2 0.709**; macro AUC 0.602 vs max pair AUC 1.000 (pair 0,1); canonical-contrast R2 0.709 reported separately from pair AUC; convex-combination identity holds to 1e-9 | averaging hides contrast leakage; rho1^2 reveals it |
| 5 | absent / singleton / below-support classes | pass | per-class NE with n_pos 0 / 1 / 5; macro NE with coverage 2/5 (partial labelled); pairs 1/10; `bool()`, `<`, `float()` raise; decision NOT_ESTIMABLE; outcome C5 | NE is never 0/1/pass/fail |
| 6 | decomposition | pass | sequential and Shapley sums equal total to 1e-12; interaction exposed as order range (surface 0.12-0.15); R2-to-AUC change refused; fitted output-leak fixture: surface step +0.507, attacker 0, metric 0 | surface, attacker and metric changes reported separately |
| 7 | rescaling H by 0.01 | pass | health label healthy -> collapsed (std 0.940 -> 0.0094); fixed-penalty ridge R2 changes by 8.8e-11; OLS R2 change 0.0; rho1^2, held-out rho1^2, R02 (rho 0/1e-4/1e-2) unchanged to 1e-10; held-out AUC 0.6494 = 0.6494 | health labels and fixed ridge are scale-dependent; invariants are not |
| 8 | admission | pass | rejects: equal-length reordered arrays, ID set mismatch, duplicated IDs, sha256 mismatch, missing ID array, unit in two roles; values shuffled under intact IDs pass without a reference and are rejected with one | no implicit positional alignment |
| 9 | duplicated records | pass | 2,500 rows (500 records x 5, fresh unit ids) -> 500 units; cluster SD 0.0181 = undup SD 0.0181; naive row SD 0.0086; seeds reported as refit replicates (n_units stays 400) | duplication cannot inflate units |
| 10 | repeated release | pass | fresh noise: AUC 0.608 (N=1) -> 0.762 (N=16), record valid A3(16); persistent token: probabilities bitwise equal to single release (AUC 0.620 both), record invalid, effective N = 1 | A3 only where the contract issues fresh noise |

Auxiliary tests (pass): permutation null re-selects the max (null worst-pair mean > 0.5) and detects a planted
signal; Brier skill / log-loss reduction are 0 at the fit prior; fits refused on non-synthetic data without
`--execute-scientific-fits` (API and CLI); `--dry-run` fits nothing; sockets blocked; weighted rank AUC equals
`sklearn.roc_auc_score` with ties; protocol defaults/override/invalid; the methodology `protocol_config.json` loads.

Mutation check (scratch copies): six deliberate bugs, each killed by at least one test: record keys ignored
(T9); reordered arrays accepted (T8); UNRESOLVED read as below (T2); release contract ignored (T10); support rule
disabled (T5); ridge penalty made relative (T7).

## Historical recounts (independent code, stored arrays only)

| id | quantity | previous | recomputed | status | scope |
|---|---|---|---|---|---|
| a | AAAI audit failures at 0.52 / 0.55 / 0.60 | 64 / 59 / 51 of 67 | **64 / 59 / 51 of 67** | match | macro OvR AUC per stored draw, mean over draws, max(XGB, MLP); 64 npz files, all sha256 = drive inventory; max abs dAUC vs stored 1.7e-5; + LoRA where stored: same counts |
| a' | same, one count per distinct measurement | n/a | **62 / 57 / 51 of 64** | new | E2 and E4S1 sigma = 1, 2, 8 are the same files and AUCs |
| b | PCRL NeurIPS strict R2 <= 0.05 | 56/60 final, 54/60 best | **56/60, 54/60** (Adult 23/21, HMDA 16/16, Diabetes 17/17; `<` identical) | match | committed JSON at origin/main 55e4cb1d1; input sha256 equal to VERIFICATION_REPORT A1 |
| c | frozen forward smoke, Round-4 Adult s0 `final.pt` | n/a | shapes rep 3 x (256, 64), logits (256, 2/6/4); bitwise-deterministic repeat; batch 32 = batch 512 exactly; vs archived PCRL b96c412 classes max rel diff 9.4e-7 (both float32 implementations about 1e-7 from a float64 reference); bad hash and in-git cache refused; `weights_only=True` load | pass | 256 rows of the local Adult test split; architecture rebuilt from tensor shapes + embedded config (rank 8, alpha 16); code exported read-only with `git archive b96c412 pcrl` |

## Scope limits

* Synthetic tests validate the machinery, not any real-data conclusion.
* The cached checkpoint is Round 4 (the encoder durable-guarantees audited), not the NeurIPS Round-5 headline
  checkpoints (drive-only, PENDING).
* Historical probability arrays carry no row IDs or units: they can be recounted, not re-admitted.
* Not implemented: Holm adjustment, methodology recipes R05 (LR form), R10, R13, R14, utility probes, the
  stratified role split. See `notes/evaluator/evaluator_notes.md`.

## Coordinator checks (2026-10-02)

Run after integrating all three roles. Machine-readable copy: `validation_results.json` →
`coordinator_checks_2026_10_02`.

| Check | Result |
|---|---|
| Full suite including owner tests 14–15 | **31 passed** (about 2 s) |
| Fixed-penalty ridge R² vs OLS under rescaling, signal at the penalty's eigenvalue scale | Ridge moves 0.065 → 0.194 for scale 0.5 → 2; OLS stays at 0.208. test_07 checked only the benign case; test_14 adds this one. |
| Noise releases | Untreated release is the identity. Historical `default_rng(seed)` convention reproduced. Fresh per-query draws average down; a persistent token does not. |
| R02 primary endpoint | Fit-mean denominator; never clamped; gated behind `--execute-scientific-fits`. The score-mean sensitivity is reported separately. |
| Synthetic end-to-end CLI, methodology config | Exit code 0. On the contrast fixture, held-out R02 is 0.020 while held-out ρ₁² is 0.962: the contrast leak is visible only to the contrast diagnostic. |
| Held-out LR recount (HL-04) rerun | Byte-identical to the methodology output. |
| Real pilot inputs (no fitting) | Frozen forward pass of the hash-verified Round-4 Adult checkpoint. 152/152 manifests admitted. |
| Backup read-back | 102/102 match using uncached reads. A cold-remount read is still pending. |
