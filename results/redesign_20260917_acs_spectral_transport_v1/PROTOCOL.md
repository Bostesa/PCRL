# Prospective protocol: locked temporal transport of the fixed 14-interface ACS slate

Written 2026-09-17 before any California 2017 model fitting, prediction, selection or outcome score. Starting research commit: `ad2c08872815185e4b63ae1d160e44f3aea1d5f4` (branch `ablations-facct-2026-07-24`, worktree branch `residual-spectral-20260910`). This protocol's SHA256 at first commit is recorded in `PROTOCOL_FREEZE.json`; later changes are appended as dated amendments below and never rewrite this text.

## 0. What already exists and is reused, not rerun

The residual spectral development study (`results/redesign_20260910_acs_residual_spectral_v1`, commit `ad2c088`) is complete: 24 new spectral maps (8 arms x seeds 0/1/2) frozen before reserved-label evaluation, 18 historical systems (H, E, A0, L025, L20, J) recovered without retraining, 42/42 evaluation units, independent replay passed. Its conclusion was: residual preservation adds residence capability beyond H in every seed and weighting; no spectral recipe met the prospective nomination rule; C1 did not meet the fixed coordination criterion against both L1 and equal-mass L2. That is **development evidence on the previously exhausted 2018 cohort**. This protocol does not reinterpret it as confirmatory, does not refit any map, and does not expand the grid. The six development arms' mathematics, fixtures and moment diagnostics are documented in that study's `MATHEMATICS.md`; this study's `METHOD_AND_SCOPE.md` restates them with the transport-specific scope.

The California ACS 2017 one-year person file was admitted at commit `ad2c088` for file-level temporal transport on provenance and schema grounds only (`TRANSPORT_ADMISSION.md`, `DATA_ADMISSION.json` in the development study). No 2017 prediction or outcome has been computed. The 2016 fallback was never accessed and will not be.

## 1. Partitions (reuse of the existing locked household hash partition)

The admission already created a deterministic, label-blind household partition of all 79,869 eligible 2017 people (53,504 published SERIALNO groups): SHA256 of `PCRL-residual-spectral-transport-v1-20260910|2017|06|SERIALNO` orders groups; rounded cumulative boundaries allocate 25% attacker fit, 15% attacker validation, 25% task fit, 15% task validation, 20% final evaluation (19,988 / 11,894 / 20,017 / 12,046 / 15,924 people; 13,376 / 8,026 / 13,376 / 8,025 / 10,701 groups). Array hashes are in the development `DATA_ADMISSION.json`.

**This protocol uses that partition instead of the default 50/20/30 split.** Reasons, fixed before any outcome: (i) it already exists, is committed and hash-bound, and creating a second split of the same file would be exactly the re-splitting this study must avoid; (ii) its separate attacker and task pools reproduce the development design, where utility probes and attackers were fitted on disjoint household pools (fitting 50% = attacker fit + task fit; validation 30% = attacker validation + task validation); (iii) it reads no label, feature or weight. Cost: the final partition is 20% (15,924 people) rather than 30%. Admission inspected only aggregate class support of the final partition (documented in `TRANSPORT_ADMISSION.md`); no predictions exist. The final partition contains one Alaska Native (RAC1P=4) person and attacker validation contains none; this is a fixed support limitation, not a reason to re-split.

The same partition applies to every interface, method, mode and source-model seed. Year-qualified identifiers (`2017|SERIALNO|SPORDER`) are used for every per-person key. Public identifiers cannot establish that underlying individuals differ from 2018 respondents.

## 2. Frozen interface objects (42 interface instances; 24 new maps plus 18 historical systems)

Per seed s in {0,1,2} the following 2018-fitted objects are applied without refitting, recalibration or re-selection:

1. `CovariatePreprocessor` from `redesign_20260907_acs_transfer_v1/seed_s/preprocessing.json`. Its existing inference path maps a valid but unseen category to that feature's `unseen` column and a missing/invalid value to its `missing` column. No category is merged or remapped.
2. PCA32 teacher from that directory's `release_maps.joblib`, output cast to float32 exactly as in the original release code.
3. Source service heads (income, employment, coverage) recorded in `redesign_20260909_acs_fixed_predictions_v1/ANCHOR_PARITY.json`, applied to the float32 PCA32 coordinates. Both probability columns are released exactly as float64 values.
4. Historical auxiliary channels: E = saved selective teacher on PCA coordinates 1-16; A0, L025, L20, J = saved `final.pt` Auxiliary mappers with their public auxiliary heads (the derived view); H = no channel.
5. The eight spectral maps from the development study's `seed_s/maps.joblib` (`spectral_S0, M025, M1, L025, L1, C025, C1, L2`). Inference uses only T and H_A.

Wires: A = [income0, income1, employment0, employment1, auxiliary], B = [coverage0, coverage1], AB = A then B. B-only values are identical across all 14 interfaces. **Pipeline identity check before any 2017 use:** the same code applied to the 2018 cohort must reproduce bitwise the saved 2018 PCA arrays, anchors, historical releases (wire and derived) and spectral releases for every pool; failure blocks transport.

## 3. Evaluation modes

### Mode A: frozen operational transfer

All 2018 objects stay frozen: service heads, channels, the 2018 validation-selected utility probes of each interface, the complete 2018 attacker candidate sets and their 2018 validation selections in the six development scopes (`standard_independent`, `expanded_independent`, `expanded_catchup` and their `kernel_` variants) at budgets 120 and 360, the 2018 fitting priors, and the 2018 PCA32 / rich-bank / tree-bank reference probes from `redesign_20260908_acs_protection_v1`. They are scored once on the 2017 final partition. Nothing is fitted or selected on 2017 data. Mode A primary scope: `kernel_expanded_catchup`, budget 360 (the development decision scope). A weaker frozen attack after shift is not evidence that fresh attackers fail.

### Mode B: fresh adaptation to the fixed interfaces

Interfaces stay frozen. On 2017 data only:

* **Utility probes:** for each authorized view/task (A: income, employment, residence; B: coverage, commute), subset `subset_indices(task_fit labels, 2048, 1230000+100s+j)` with j the TASKS index order (`same_residence, commute_over20, income_binary, civilian_at_work, public_coverage`); fit logistic (C=1) and one MLP (40 epochs) with seed `1250000+100s+TASKS.index(task)` via `fit_candidates`; select minimum unweighted task-validation log loss, then candidate ID. B-view probes are fitted once per seed on the shared B wire and aliased to every interface.
* **Reference probes:** the same recipe on the frozen 2018 PCA32 (all five tasks), frozen rich-bank and tree-bank releases (residence), giving the Mode B parent for the legacy source allowance and half-headroom criteria.
* **Priors:** `fit_prior` (pseudocount 1) on the attacker-fit subset labels of each target.
* **Attackers:** per target subset `subset_indices(attacker_fit labels, 4096, 1240000+100s+j)` in the TARGETS order (`SEX, RAC1P, income_binary, civilian_at_work, public_coverage, same_residence, commute_over20`). For every interface and each A/AB forbidden role (and, once per seed via H, each B role): the fixed five-candidate family (logistic C=1; two restarted MLPs with nested 120/360-epoch checkpoints; HistGB with min leaf 20 and 5) with `role_seed`, plus the fixed Gaussian random-feature kernel ridge family (256 features, training-only median bandwidth on <=512 rows, alphas 1e-4, 1e-2, 1, probability conversion by clipping to [1e-12,1] and row normalization) with the development seed formula. For interfaces with public auxiliary heads (A0, L025, L20, J) the same families are fitted on the derived view. For the six historical interfaces that have saved 2018 observers, one saved-start catch-up trajectory per A/AB observer role (and B roles once via H) is fitted from the frozen observer on 2017 attacker-fit rows with the original catch-up recipe (360 epochs, nested 120). Spectral arms have no trained observer; this unequal exposure is disclosed, not equalized by deletion.
* **Legal projections:** every H candidate (fresh, kernel, catch-up and frozen, see below) is routed to the A/AB anchor coordinates of every other interface (`anchor__` candidates); every A and B sensitive candidate is inherited by AB on its coordinate block (`inherited_` candidates). B-view candidates are H's.
* **Frozen 2018 candidates:** every non-diagnostic, non-anchor, non-inherited 2018 candidate of the same interface and budget from the development audit record (fresh, kernel, derived, catch-up) is added with ID prefix `frozen2018__`, then routed/inherited like the others. Saved-observer diagnostics (`saved_adversary`) remain diagnostic only.
* **Selection:** every candidate is scored on the full valid 2017 attacker-validation rows; minimum unweighted log loss, then candidate ID, within each scope. Selection records are written before the final partition can be read.

Mode B scopes (each at budgets 120 and 360):

| Scope | Candidate origins |
|---|---|
| `common_fresh` | 2017-fitted wire candidates (five-family + kernel), H anchors of that origin, inherited singletons of that origin. Identical family structure for all 14 interfaces. |
| `fresh_expanded` | plus 2017-fitted derived-view candidates (A0/L025/L20/J only). |
| `fresh_catchup` | plus 2017 saved-start catch-up candidates (historical interfaces only). |
| `transport_all` | plus all frozen 2018 candidates. **Primary Mode B scope.** |

Primary budget: 360. All scopes and both budgets are reported. Where a race class occurs in the 2017 attacker fit subset but was absent from 2018 attacker fitting, fresh candidates learn it; complete nine-class schema log loss is always reported. A final-partition class absent from a candidate's fitting data is scored with the fixed 1e-12 probability floor and renormalization; class-specific recall/AUROC are reported only where defined, and classes with fewer than 10 final-partition people are flagged as insufficient for class-level statements. No category is dropped from final scoring. Comparable-category diagnostics (restricting to classes supported in all fitting pools) are reported separately and labeled.

Nuisance models are not refitted. The declared residual-moment diagnostic applies the frozen 2018 nuisance ensemble to 2017 partitions.

## 4. Lock and sealed final scoring

Before final scoring, `TRANSPORT_LOCK.json` records SHA256 hashes of: this protocol and `COMPARISONS.json`; all evaluation code; all 42 interface objects and their inputs; the 2017 fitting/validation release arrays; every Mode B fitted object and selection record (utility, reference probes, priors, attackers); the Mode A source records; the final partition file and its manifest. The lock is committed and pushed to the research branch before final scoring. Final-partition arrays can only be opened through a loader that verifies the lock and every hashed input; a missing or mismatched hash aborts. The lock is not an external preregistration.

After final scoring no representation, hyperparameter, threshold, primary contrast, scope, budget or selection may change. A genuine implementation error requires a labeled corrected replay with the original output preserved.

## 5. Scoring

All log losses use natural logs with the original 1e-12 floor and renormalization. Selection is unweighted; unweighted and PWGTP-weighted scores are computed on identical selected predictions. Absolute recovery = prior loss minus attack loss (same mode's prior). Additional (incremental) recovery = interface absolute recovery minus H absolute recovery in the same seed, mode, scope, budget, view, target and weighting. Negative values are retained, never clipped. Every candidate, including every fixed ancestor, has its own final score row. Counts of negative selected increments are reported. Service quality = log loss of the exact released anchor probabilities on the final partition, both modes.

**Withholding controls:** for E, A0, L025, L20, J and spectral_S0 and p in {0, .25, .5, .75, 1}, H is always released and an independent visible Bernoulli(p) branch fixed per person decides whether the auxiliary channel is also released. The routed family uses H's selected predictor on H-branch persons and the interface's selected predictor on augmented persons, so expected per-person loss is exactly (1-p) loss_H + p loss_aug, mixed before weighted averaging, from aligned per-person arrays in the same mode/scope/budget. A realized routing uses uniforms from SHA256 of `spectral-transport-withholding-v1|2017|s|condition|SERIALNO|SPORDER`; one realized example is checked against the expectation. No p is selected. These are expected outcomes of a specified stochastic mechanism and routed family, not new optimal attackers.

## 6. Estimands, uncertainty and fixed comparisons

Estimands average equally over the three fixed fitted seed systems after computing each seed's unweighted or PWGTP-weighted final-partition loss. They concern these fixed systems, not future retraining. Seed-level values are always shown.

Uncertainty: paired household-cluster bootstrap, 2,000 replicates, `numpy.random.default_rng(20260917)`, sampling the 10,701 final SERIALNO groups with replacement and retaining all their rows. The same draws apply to every method, seed, task, weighting, mode and view. Weighted means are recomputed as ratio estimators in each replicate. This is household-resampling uncertainty for fixed fitted systems; it is not Census replicate-weight variance, not retraining variability, and it treats the three seed predictions for one person as one person. Effective counts (households, valid rows, Kish PWGTP effective n) are reported.

Paired difference directions: every difference is candidate minus comparator. Residence loss: lower is better. Recovery (absolute and incremental; identical within a paired Mode B comparison because prior and H selection are shared, and both are reported): lower is better.

Adjusted families (single-step studentized max-|t|): for family endpoints k, se_k is the bootstrap SD of the replicate estimate; T_b = max over k with se_k > 1e-15 of |theta*_bk - theta_hat_k| / se_k; c = 95th percentile of T_b (`numpy.quantile`, method `higher`); interval theta_hat_k +/- c se_k. Endpoints with se_k <= 1e-15 are reported with a degenerate interval and flagged; they do not enter the maximum. Unadjusted 95% percentile intervals are descriptive only.

| Family | Mode, scope, budget | Contrasts | Endpoints (x unweighted, PWGTP) | Count |
|---|---|---|---|---|
| F1 primary | B, `transport_all`, 360 | spectral_C1 - spectral_L1; spectral_C1 - spectral_L2 | residence loss; A/SEX, AB/SEX, A/RAC1P, AB/RAC1P recovery | 20 |
| F2 neural replication | B, `transport_all`, 360 | J - L025; J - L20 | same five | 20 |
| F3 secondary method | B, `transport_all`, 360 | spectral_C1 - J | same five | 10 |
| F4 frozen-transfer view | A, `kernel_expanded_catchup`, 360 | the four contrasts of F1-F2 | same five | 40 |

C025 - L025 and all other arms are mechanism context with unadjusted intervals.

## 7. Decision rules (fixed)

* **Advantage of X over comparator Y (per weighting)** in a family: at least one of the four sensitive endpoints has adjusted upper bound < 0, and no sensitive endpoint has adjusted lower bound > 0.
* **Primary coordination result (F1):** C1 has an advantage over both L1 and L2 under both weightings, and the seed-mean residence loss difference is <= .001 (the original directional utility delta) for both comparators and weightings. The residence interval is always shown. Local race (A/RAC1P) significantly worse (adjusted lower bound > 0) is a reported failure regardless of other endpoints.
* **Neural replication (F2):** the same rule for J against L025 and L20.
* **Residence capability** (each interface, each mode, each weighting): seed-mean gain over H >= .01 and every seed's gain >= .01; the unadjusted bootstrap interval for the seed-mean gain is reported. The original half-headroom criterion (residence probe loss <= (PCA32 loss + min(rich-bank, tree-bank loss))/2, same seed/weighting/mode) is counted separately per seed.
* **Source:** exact service-output parity is structural and verified on every 2017 release. Service quality after shift is reported. The legacy allowance (each selected source-probe loss <= PCA32 reference probe loss + .01, per seed, weighting and mode) is counted separately and never averaged.
* **Costs:** for each F1/F2 contrast, all eleven forbidden roles and five utility tasks get unadjusted paired intervals; any sensitive or policy role with unadjusted lower bound > 0 is listed as a candidate cost.
* **Simpler controls:** for every fixed p and withholding source, the development directional rule (all five utility losses within +.001, all six A/B/AB SEX/race recoveries no worse within 1e-12, one strict improvement) is evaluated per seed and weighting against each spectral arm in both directions; a mechanism dominating C1 in all seeds and both weightings is reported as a simpler equal-or-better option.
* **Development-rule replay:** the development fixed-vector gate (per seed and weighting, all five utility tasks within .001 and six sensitive recoveries within .0005 plus one shared strict improvement > .001) is recomputed on transport points for C1 against L1 and L2 and for every spectral arm against J.
* **Consistency:** each result is reported per weighting, per seed, per Mode B scope and in Mode A. Development versus transport: "consistent" if the seed-mean transport difference has the development sign; "contradicted" if the adjusted transport interval excludes zero with the opposite sign; otherwise "unresolved". Non-significance is never called equivalence.

No combined utility/privacy score, leakage budget or cross-target nat exchange rate is used.

## 8. Artifacts and publication

Fitted objects, 2017 arrays, per-person predictions and losses stay local under ignored paths. Compact aggregate tables, locks, manifests, code, tests, figures and reports are published to the research branch with normal commits; main is untouched; no force push; no paid compute.

## Amendments

(None at protocol commit.)
