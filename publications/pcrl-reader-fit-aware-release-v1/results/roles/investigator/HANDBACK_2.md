# HANDBACK 2: mechanism investigator, revision R1 reader family `softbin`

Campaign `pcrl_reader_fit_aware_release_v1`. Worked 2026-10-25, 22:17Z to 22:22Z (`date -u`), in reply to the coordinator's 22:16Z
request. Label-free: invented data only, one thread. No real data, label, model file, `.private/` path or git command was touched.
CPU used for R1: about 5 CPU-seconds (selftest 1.9 s, bench 2.3 s, summaries). The session total is about 30 CPU-seconds.

Files written:
- `rfa/fam/rfit_r1.py`: new, 469 lines, sha256 prefix **24c2bd996dabe396**.
- `results/pcrl_reader_fit_aware_release_v1/roles/investigator/RFIT_R1_SELFTEST.json`: `all_pass: true`, 12 checks, 0.76 CPU-s.
- `results/pcrl_reader_fit_aware_release_v1/roles/investigator/RFIT_R1_BENCH.json`.
- This file.

`rfa/fam/rfit.py` is **unchanged**: sha256 prefix still 04b22c664caa8c55.

## 1. What R1 is

`rfa.fam.rfit_r1` imports `rfa.fam.rfit` and reuses it unchanged. These all come from rfit:
- `fit_reader` (the Newton solve and guards);
- `binormal_auc` and the surrogate;
- the implicit-function fit path;
- `_critic_update` (ALTCRIT, including the lag diagnostics);
- `standardize` and `moments_of`;
- `exact_auc`;
- `monitor_fisher2`, re-exported unchanged.

The subclass `EpisodeObjective(RF.EpisodeObjective)` overrides only `objective_parts`. It runs v1's loop with a per-family feature
hook and a per-family ridge. `objective`, the modes (FULL / STOPFIT / ALTCRIT), the signatures, the info keys and
`monitor_fisher2` are as in v1. `info` gains `revision` and `decision_scales`. As a drop-in for `fit.py`:
`from . import rfit_r1 as RF` gives `RF.EpisodeObjective`, `RF.RECIPE_RF` (= `RECIPE_RF_R1`) and `RF.monitor_fisher2`.

`RECIPE_RF_R1` is v1's `RECIPE_RF` with everything unchanged, including:
- n_A 4096 / n_E 2400;
- views occupation, pair;
- soft bound 6;
- detached moments;
- Newton settings and guards;
- the surrogate;
- the ALTCRIT settings.

Only these change:
- `families` = ['simple', 'rff', 'softbin'];
- `family_features` gains 'softbin';
- `ridge_by_family` = {simple 1e-2, rff 1e-2, softbin 1e-3};
- the softbin constants are added.

### The softbin family, per view

It is built from the same detached-moment, soft-bounded standardized coordinates s that v1 uses:

1. **Soft one-hot bins** of each coordinate.
   - B = 8 Gaussian kernels at the standard-normal quantile centres Phi^-1((b - 1/2)/8): +-0.157, +-0.489, +-0.887, +-1.534.
   - Common width h = 0.3, normalized per coordinate: m_b = softmax_b(-(s - c_b)^2 / (2h^2)), so each coordinate's memberships sum
     to 1. The end bins are open.
2. **Soft decision** d_r = softmax(z_r / (tau * scale_r)), with tau = 0.25.
   - scale_r is the **detached** A-row RMS SD of recipient r's logit differences: income SD(z1 - z0); occupation
     sqrt(mean_k Var(z_k - z0)).
   - One small deviation from the literal `softmax(z/tau)`: tau is in standardized units. A global temperature change of the release
     therefore cannot blur the soft decision across steps, while the hard argmax on the wire stays unchanged. This is the same
     convention as the bins.
   - Income view: d_inc (2). Occupation view: d_occ (5).
3. **Pair view products:** the 10 products d_inc (x) d_occ, a soft CATI-like decision interaction. They replace the two marginals,
   which are their sums.
4. **Intercept.**

| View | Width |
|---|---|
| Income | 8 + 2 + 1 = 11 |
| Occupation | 32 + 5 + 1 = 38 |
| Pair | 40 + 10 + 1 = 51 |

The fit is ridge-logistic with v1's Newton / IFT machinery and guards. Measured cond(H) is about 620 to 660.

### Frozen a priori

All values were written into the module before its first run. No data, no outcome, no grid. The invented-data tests changed nothing.

| Value | Reason |
|---|---|
| B = 8 | Equal-mass bins for standardized, roughly Gaussian coordinates; a coarse table resolution |
| h = 0.3 | About the centre spacing in the bulk (0.31 to 0.40) |
| tau = 0.25 | Standardized units: a gap of a quarter of an SD gives a factor e on the soft decision |
| softbin ridge 1e-3 | Memberships have variance about 1/B. A bin holding 1/8 of the rows has data curvature about 0.03 against 1e-3 (about 3% shrinkage), comparable to v1's 1e-2 on unit-variance features (0.25 against 1e-2, about 4%) |

### Softbin alone, or all three families?

I see **no technical reason** to use softbin alone. Three families work in the same machinery. Conditioning is fine. Cost rises about
1.5x. R1 with ['simple', 'rff'] reproduces v1 exactly.

The one consideration is strategic, not technical. S takes the mean over families, so softbin carries one third of S. Since lambda is
recalibrated on the gradient norm, keeping SIMPLE and RFF spends part of the privacy push on the linear and smooth directions, where
v1 already moved LOG_LR most. Softbin is not a superset of RFF: its only cross-coordinate terms are the decision products. Keeping all
three is the cleaner single-change revision, and I agree with it.

## 2. Selftest (invented data, float64, deterministic; `RFIT_R1_SELFTEST.json`)

| Check | Result |
|---|---|
| `r1_reproduces_v1_families` | With families ['simple', 'rff'], R1's FULL and STOPFIT gradients equal rfit's **bitwise** (max abs diff 0.0; P 0.2109036 identical). The copied loop is faithful |
| `fd_softbin_complete_derivative` | Softbin alone, 3 views, tiny linear release, all 49 parameters, n_A = n_E = 256, eps 1e-5, moments and decision scales frozen at theta0. FULL vs central finite differences: **relative error 1.85e-9** (max abs 3.1e-10). STOPFIT vs FD: 0.68. Ratio ||fit|| / ||direct|| 0.73, cos(FULL, STOPFIT) 0.75, cos(fit, direct) -0.26. Newton at most 7 iterations, cond at most 634, fit scalar value 1e-17 |
| `fd_registered_r1_income_constant` | Registered R1 recipe (3 families, views occupation + pair), income logits constant, FD over the 35 occupation parameters: **relative error 8.8e-10**. Exactly the 6 expected terms; zero gradient on the income parameters. Ratio 0.52, cosine 0.86 |
| `optimization_sign_softbin` | eta 1e-3: P falls by 8.854e-5 against a first-order prediction of 8.849e-5. eta 1e-2: falls by 8.89e-4 |
| `softbin_fit_path_nonzero` | n_A 4096 / n_E 2400, invented mean plus variance signal, softbin per view (see the table below) |
| `softbin_structure` | Each coordinate's memberships, the decisions and the 10 pair products sum to 1 (max error 4.4e-16); all features in [0, 1]; pair width 51 |
| `altcrit_newton_alias_of_stopfit_r1` | Bitwise alias (0.0), relative lag 0 |
| `altcrit_adam_lagging_r1` | Relative lag median 0.72, cos(ALTCRIT, STOPFIT) 0.57: the Adam critic lags and is not an alias |
| `objective_parts_consistent_r1` | grad(S_dir) - grad(STOPFIT) = 0.0; grad(S_fit) - (FULL - STOPFIT) = 6.9e-18; S_fit value -2e-17 |
| `one_class_A_refused_r1` | PASS |
| `interface_r1` | info, critic info and monitor are plain floats; `RECIPE_RF` n_A 4096, n_E 2400, views and families as registered; a subclass of rfit's class |

Softbin per view at n_A 4096 / n_E 2400 (invented data):

| Term | ratio | cos(FULL, STOPFIT) | cos(fit, direct) | held-out exact AUC |
|---|---|---|---|---|
| income/softbin | 0.30 | 0.99 | +0.73 | 0.548 |
| occupation/softbin | 0.30 | 0.96 | -0.44 | 0.774 |
| pair/softbin | 0.48 | 0.88 | -0.33 | 0.776 |
| registered R1 (all 6 terms) | 0.43 | 0.91 | -0.44 | |

For the registered R1 run, the output-space fit share is 0.15.

## 3. Cost per step: one thread, n_A 4096 / n_E 2400, the loop's pattern (`RFIT_R1_BENCH.json`)

Setup: the occupation network trained, START income logits constant, diagnostics on, a 256-row task batch, Adam, clip 5, random
weights, identical invented rows for both revisions.

| Arm | R1 (3 families) | v1 (2 families, same rows) |
|---|---|---|
| FULL | 0.051 s | 0.034 s |
| STOPFIT | 0.032 s | 0.021 s |
| ALTCRIT (+ lag-diagnostic solve) | 0.046 s | 0.028 s |
| TASK | 0.0005 s | |

Campaign estimate for 2 versions x 6 units x 400 steps:

| Arm | Estimate |
|---|---|
| FULL | about 250 s |
| STOPFIT | about 155 s |
| ALTCRIT | about 220 s |
| Reader arms together | about 10 CPU-minutes |
| START gates and calibration | about 1 minute more |

## 4. Caveats

- Every number above is from invented data. On real data, v1's fit path was small: F0 active, orthogonal share 0.10 / 0.11. Nothing
  here shows that softbin will make it larger or will lower recovery by CATI1 / ET25 / MLP64. That is the empirical question for R1.
- Softbin approximates table and decision readers with soft memberships and soft decisions. The real winning readers use hard splits
  and the hard decision, so the gradient is a surrogate of their response, not their response.
- The detached decision scale is a convention choice (section 1). It changes the literal spec only by rescaling tau, and it closes a
  temperature-blur exploit.
