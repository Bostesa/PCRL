# Matched removal benchmark: coordinator design (2026-10-02, before any benchmark fit)

## Setup

**Branch:** `research/combined-matched-removal-benchmark-v1`.
- Base: pilot tip 661db8dcbba3abd29bd42d6674c2493c77eb47d8.
- Ancestors: reconciliation 07a9ca3ffaf27133b6cf955b3b8d526682961e8d; preparation 031860fbfd910b9aaed19b7600d288d15858dbc5.
- durable-guarantees: 956f5c883f515646aa457db55ecbd74b913768b2.
- Worktree: `/Users/nathansamson/PCRL/.worktrees/combined-matched-removal-benchmark-v1`. Package: `results/combined_matched_removal_benchmark_v1/`.

**Private root (never in git):** `~/PCRL_eval_cache_private/bench_v1/`.
- `inputs/`, `units/`, `infer/`, `logs/`, `defenses/`.
- Reuse `~/PCRL_eval_cache_private/checkpoints/` and the pilot cache read-only.

**Drive:** `/Volumes/YOTUO/NathanSamson-Mac-relocated-2026-09-30/`, exFAT. It can transiently disappear, so retry reads and never write inside `archives/`.

**Official LEACE:** concept-erasure 0.2.4, as installed in `/Users/nathansamson/PCRL/.venv`.
- Upstream tag v0.2.4 = 9b18b3d5c73f552798212c51d6533d649fa434cd.
- Installed `.py` tree sha256 = fffac29d5914f334396f4af1f6b65ba09fcbf658fa8a013972bb50cbc1568597.
- Use `LeaceFitter` / `LeaceEraser` only. Never `OracleLeaceEraser`, never the in-house `pcrl.models.baselines.LEACEEraser`.

**Budget:** 8 CPU-h for science (fits + inference), laptop only, OMP_NUM_THREADS = 1, one scheduler. Calibration, repairs and
synthetic validation are recorded separately. 8 h is a ceiling, not a target.

## Tier 1: mandatory cells

Names are frozen from PCRL b96c412 `pcrl/data/{adult,hmda}.py`.

| Dataset | Purpose (index) | Task | Target attribute (B) | Policy set (C) = disallowed_attrs of the purpose |
|---|---|---|---|---|
| Adult | `income_prediction` | `income` (2) | `sex` (2) | {`race` (5), `sex` (2)} |
| HMDA | `underwriting` | `loan_decision` (2) | `race` (5) | {`race` (5), `ethnicity` (2)} |

Freeze the purpose index from the checkpoint/dataset object (Adult p0 per durable-guarantees).

**Encoders:** PCRL **Round-4** `final.pt`, seeds {0, 1, 2}, per dataset (`checkpoints/v2_{adult,hmda}_s{0,1,2}/final.pt` on the
drive). Each is admitted only if its Round-4 lineage verifies: epoch, and saved λ/state against `results/v2_*_ROUND4/per_seed_results.json`.
- A seed that fails is reported missing with the exact file and reason.
- No other round is ever substituted.

**Methods per (dataset, encoder seed):**

| Arm | Definition |
|---|---|
| A | Untreated frozen representation h_p of the purpose |
| B | Official LEACE fitted on defense_fit h_p with the one-hot of the target attribute |
| C | Official LEACE fitted on defense_fit h_p with the **concatenated marginal one-hots** of the policy set. Target scope ≠ policy scope; no claim about intersections of attributes. If C's fitted map equals B's within tolerance, record an alias and score once. |
| D | Gaussian noise on h_p (untreated): σ_abs ∈ {0.25, 0.5, 1, 2, 4, 8} × release seeds {0, 1, 2}. One persistent draw per person (`default_rng(seed)` over the full matrix, pilot convention). Report ‖h‖ and σ/σ_train-scale; equal σ is not a matched operating point. |

**Controls (references, not methods):**
- label-only (true task label → s);
- constant prediction (attacker_fit prior);
- clean outputs (original head logits);
- the untreated release (arm A).

## Roles (fixed before fitting)

- **Attacker roles, Adult:** the pilot roles on the PCRL Adult test split (`pilot-roles-v1`; attacker_fit 7,571 /
  attacker_val 2,239 / assessment 5,250). Unit = de-duplicated record.
- **Attacker roles, HMDA:** the PCRL HMDA test split. Role = sha256('bench-roles-hmda-v1|' + record_key), shares
  0.50 / 0.15 / 0.35. Same record-key definition as the pilot. Duplicate records share a unit and a role.
- **defense_fit (LEACE), preferred:** the historical encoder-training rows (the PCRL train split, regenerated
  deterministically by the b96c412 loader). Exclude every record whose record key also occurs in any test-split role.
  - These rows already trained the encoder: disclose that exposure.
- **Fallback, only if train rows cannot be regenerated and verified:** a deterministic 30 % of attacker_fit units,
  sha256('bench-defense-fallback-v1|' + key) < 0.30, become defense_fit; the remaining 70 % stay attacker_fit. All
  methods and references then use these roles.
  - The choice is made from provenance evidence alone, recorded before any fit, and never from an outcome.
- **Same role IDs across methods and encoder seeds within a dataset.** Never fit an eraser on all rows.

**Support:** 100 rows per class in defense_fit (where an eraser is fit), attacker_fit and assessment; 30 in
attacker_val.
- Pairs need both classes supported.
- Unsupported → NE with reason and coverage. No pooling.
- Supported sets are frozen in `supported.json` before any fit.

## Contracts and surfaces

| Contract or surface | Recipient receives |
|---|---|
| **C_rep** | The transformed representation only |
| **C_rep_plus_clean_out** | [transformed representation, clean head full logit vector]. The clean outputs are identical across methods within (dataset, seed). |
| **outputs_only** | Clean logits only. Computed once per (dataset, encoder seed) and aliased across methods. |
| **label_only** | True task label → s. Frequency table, Laplace α = 1 (pilot convention). |
| **constant** | The attacker_fit prior |

**Utility:**
- **U1:** the frozen head applied to the transformed representation. A compatibility diagnostic only.
- **U2:** a common LR probe predicting the task from the transformed representation: C ∈ {0.01, 0.1, 1, 10, 100}, fit
  on attacker_fit, selected on attacker_val, scored on assessment. The capability retained.
- **C_rep_plus_clean_out utility:** the clean head's task accuracy and log loss. Unchanged across methods; say so.
- U2 outputs are a different object from the clean outputs and are never swapped into C_rep_plus_clean_out.

## Attackers

These are the pilot's effective recipes.

**Families:**
- **L:** C ∈ {0.01, 0.1, 1, 10, 100}.
- **GBT:** 20 frozen configs.
- **MLP:** 18 frozen configs.

**Selection:** family and hyper-parameters are chosen on attacker_val log-loss (NL-selected), with L also reported
separately.

**For C_rep_plus_clean_out:** the selection slate additionally contains the fitted **ignore-rep** candidate (the
outputs_only model) and the **ignore-outputs** candidate (the C_rep model). Selection over all of them is on
attacker_val.

**Training variation:**
- The selected recipe (family + hyper-parameters) is retrained at attacker seeds {0, 1, 2}. The grid is not rerun.
- Deterministic recipes (L / lbfgs) are aliased across seeds and recorded as such.
- Each seed's predictions and model are saved.

**Noise arms** add LRT-A2 (σ-informed, released rows only) and LRT-A4 (clean-population stress test), kept separate.

**LEACE arms** use the ordinary slate trained on erased attacker-population features. "Simulation" is an alias of that
(recorded); no stronger attack is claimed.

## Native, held-out and recovery statistics

Every method gets all of these, on identical assessment unit IDs. Prediction outputs are audited as their own surface:
a nonlinear head after erasure can re-create linear signal.

**Native check under each method's own scope (from GUARANTEE_CARDS):**

| Arm | Native check |
|---|---|
| A | PCRL historical one-hot ridge R² in-sample on the test split ≤ 0.05. Reproduce against `dominant_axis_audit.json` per seed. |
| B / C | LEACE condition on defense_fit: cross-covariance of the erased h with the concept one-hot is 0 within a stated numerical tolerance; plus fit-row affine OLS R² ≈ 0 for the concept. |
| D | No native certificate (unclipped Gaussian noise; Prop 3 covers only the clipped channel). NA. |

**Held-out (all arms; separately and unclamped):**
- G1, fixed ridge 1e-6 (the pilot quantity);
- G2, scale-invariant;
- ρ₁²;
- B/C additionally: the held-out cross-covariance norm.

**Recovery:**
- macro OvR AUC over supported classes;
- supported worst-class and worst-pair (maximum);
- Brier skill;
- LLR_nats and LL_skill (prob clip 1e-12);
- output recovery vs the label-only and clean-output references.

## Inference: frozen before scoring

**Sampling unit:** the assessment record unit. Cluster bootstrap with fitted predictors held fixed; the bootstrap does
not refit.

**Replicate summaries (frozen):** the endpoint statistic is the mean over encoder seeds, release seeds and attacker
seeds, computed within each bootstrap replicate on the same resampled people. Encoder seeds, release seeds and
attacker seeds are **not** extra units.
- Per-seed tables are reported.
- Attacker-training variation = SD of point estimates across attacker seeds.
- Encoder-seed variation = SD across encoder seeds.
- Both are reported separately from the sampling intervals.

**Exploratory intervals:** 90 % two-sided percentile, B = 2,000, seed 20261003, numpy `linear` quantiles.

**Worst-class and worst-pair bounds:** derived from simultaneous per-class/pair bounds,
LCB(max) = max_k LCB_k(α/K) and UCB(max) = max_k UCB_k(α/K), with K = number of supported classes/pairs. This is
conservative.

**Diagnostic noise operating point σ\* per dataset:** the smallest σ in the grid whose **attacker_val** C_rep NL macro
AUC (mean over encoder seeds × release seeds, attacker seed 0) is ≤ 0.55. If none qualifies, σ\* = 8, flagged.
Selected from validation only; assessment is reported once. The full σ curves are development descriptions.

**Primary family (Tier 1):** 24 endpoints. Bonferroni simultaneous one-sided bounds, α = 0.05/24, B = 20,000,
seed 20261004, tail count 41.7.

For each dataset d ∈ {Adult, HMDA}:

| Endpoint | Arms | Test |
|---|---|---|
| G1[d,A] | A | Held-out fixed-ridge R² vs τ = 0.05. "Below" if UCB < τ; "above" if LCB > τ. |
| Rrep[d,m] | A, B, C, D(σ\*) | C_rep NL-selected macro AUC vs 0.55 |
| Rplus[d,m] | A, B, C, D(σ\*) | C_rep_plus_clean_out NL-selected macro AUC vs 0.55 |
| U2NI[d,m] | B, C, D(σ\*) | U2 accuracy(m) − U2 accuracy(A), paired. NONINFERIOR if LCB ≥ −0.01; INFERIOR if UCB < −0.01; else UNRESOLVED. |

That is 1 + 4 + 4 + 3 = 12 per dataset. If C is an exact alias of B, its endpoints are reported as alias rows and the
family size stays 24 (aliases are counted, not dropped).

**Exploratory only:** everything else (bars 0.52 / 0.60, L attackers, LRTs, worst-class/pair, Brier/LL, U1, label-only
contrasts, per-seed values, Tier 2).

## Tier 2 (registered now; triggered only by technical validity and remaining budget)

Trigger: Tier 1 is complete with every control valid and no unresolved technical failure, and the projected cost
fits in the remaining budget. Run in this fixed order, stopping at a unit boundary when the projected remaining cost
exceeds the budget.

1. **E1, untreated + LEACE-B** on the remaining disallowed (purpose, attribute) pairs, same encoders and roles:
   - Adult: income_prediction/race; employment_analysis/{race, age_group, marital_status}; education_assessment/{sex, race, income}.
   - HMDA: underwriting/ethnicity; pricing_analysis/{race, sex}; fair_lending_audit/{race, sex}.
2. **E2, LEACE-C** (policy sets) for those purposes.
3. **E3, noise arms** (same σ grid and seeds) on the E1 pairs.

**Round-5/7 NeurIPS headline checkpoints:** inventory and verify only. A separate, labelled extension; never mixed into
Round-4 method rows.

## Interface contracts (for parallel roles)

**Inputs index:** `~/PCRL_eval_cache_private/bench_v1/inputs/INPUTS_INDEX.json`. Per dataset:
- the record-key array path;
- role arrays for defense_fit, attacker_fit, attacker_val and assessment (per row; unit IDs);
- labels npz (sensitive + task, by row_id);
- per encoder seed: checkpoint path + sha256 + lineage status, and the forward cache npz holding `row_id`,
  `rep_p<i>` (float64), `logits_<purpose>` (float64) for **all** rows (train + test);
- the purpose → index map.

Everything is hash-pinned.

**Defense API:** `stored_model_eval/defenses.py`.
- `fit_leace(H_fit, Z_fit_onehot, *, dtype=float64) -> LeaceMap`, which wraps `concept_erasure.LeaceFitter`.
- `LeaceMap.transform(H)` (fixed map; original fitting mean).
- `LeaceMap.save(dir)` / `LeaceMap.load(dir)`: proj_left, proj_right, bias/mean, rank, dtype, tolerances,
  fit_row_hash, concept_spec.
- `LeaceMap.native_check(H_fit, Z_fit)` returns the cross-covariance max-abs and fit-row OLS R².
- `noise_release(H, sigma, seed)` delegates to `releases.gaussian_release`.
- `scale_report(H_train)` returns the per-dim std, ‖h‖ quantiles and σ/scale.

**Unit outputs:** `~/PCRL_eval_cache_private/bench_v1/units/<unit_id>/` holding `preds.npz`, `models/`,
`fit_records.json`, `supported.json` and `COMPLETE.json`.
- `unit_id` = `<dataset>__s<encoder seed>__<purpose>__<attribute>__<arm>`, where arm ∈ {A, B, C, D_sigma<σ>_rs<k>}.
- Saved prediction keys follow the pilot, plus attacker-seed suffixes `__as<k>` and the `ignore_rep` /
  `ignore_out` candidates.
