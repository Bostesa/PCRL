# Scope review: FROZEN_DESIGN against the prompt and the preparation lock

Role 1. Written 2026-10-02. I read source, documents, manifests and label marginals only. I fitted nothing
and opened no assessment results.

**Abbreviations:**
- FD = FROZEN_DESIGN.md
- P = `PCRL_Run_Repaired_Pilot_Prompt.txt`
- Lock = preparation `PILOT_LOCK.json`
- PP = `PILOT_PROTOCOL.md`
- CFG = `protocol_config.json`
- AT = `ATTACKER_ACCESS_TABLE.csv`

## 1. Fixed items: consistent

| Item | FD | Lock / PP / P | Status |
|---|---|---|---|
| Checkpoint | Round-4 Adult s0, 1cfc2fef… | lock `encoder_sha256` 1cfc2fef…c061; the `forward_manifest.json` pin matches | consistent |
| Rows | test split, 15,060 rows / 15,055 units | lock pool; `labels.npz` `row_id` = 0..15059 = post-dropna test position (checked: `sex` and `income` re-derived from `data/adult/adult.test`, sha a2a9044b…, agree on all rows) | consistent |
| Roles | 'pilot-roles-v1'; 7,571 / 2,239 / 5,250 | lock `role_assignment` + counts | consistent; CFG `roles.stratify` / `split_seed` are stale (amendment A23) |
| Panel | 8 untreated + 18 income/sex noise = 26 | lock "arms", `pilot_units: 26`; P "Keep the scope" | consistent. The manifest stems use `%g` (`sigma1`, not `sigma1.0`). The runner must take its expected-ID list from FD names as written in the manifests. |
| Noise grid | σ_abs {0.25, 0.5, 1, 2, 4, 8} × seeds {0, 1, 2}, income/sex only | lock, PP §18, P | consistent; P forbids expanding to 152 |
| Bars | 0.52 / 0.55 / 0.60 (0.55 historical) | CFG, PP §11 | consistent |
| Support | 100 / 30 / 100 | PP §10 and §18 | consistent; CFG `pair_rule` is phrased on fit and assessment only, and FD is equal or stricter |
| Exposure | development data | CFG `exposure`, P | consistent |

## 2. Conflicts to resolve before the lock

1. **L C-grid.** FD uses {0.01, 0.1, 1, 10}. CFG, AT and the translated pin use {…, 100}, and the hashed CFG
   is locked. Either execute 5 values, or amend with a reason. Do not let the translation silently run 5
   while FD says 4.
2. **Log-loss reduction.** FD uses `1 − LL/LL₀`. CFG and `metrics.py:519` use `LL₀ − LL` (nats).
3. **ReleaseContract API.** FD's `noise="gaussian", persistent=True` is not valid for the pinned
   `access.py:26-37`, which accepts `none | fresh_per_query | persistent_token`. Choose a mapping and freeze
   it.
4. **A4 LRT form.** FD and AT R09 specify class Gaussians from clean reps plus σ²I. The pinned
   `NoiseLRTAttacker` is an exact point-mass mixture over clean vectors. FD calls the Gaussian form "the
   historical Tier-2 form"; I did not verify dg's Tier-2 source, so verify it, or name the executed form
   explicitly.
5. **Noise-unit N0 wording.** FD says "N0 is not defined historically". GUARANTEE_CARDS card 5 records an
   empirical dg approval (in-sample R² averaged over 5 draws) on dg rows. Correct reading: a historical check
   exists, but not on this pool. Mark it NA or unverified; never infer it from N1.
6. **N0 versus P item 5.** P calls the historical check "the defense-fitting-row check". For PCRL v2 the
   reported native check is in-sample on the **test** split (card 2; `eval_round4_dominant_axis.py:158-172`),
   which is what FD reproduces. Name it "historical native check (test split, in-sample)", not "fitting-sample
   compliance". A train-split forward-pass N0 is feasible fit-free if the coordinator wants the literal
   reading.
7. **The float32 replica.** FD says "the historical code used a float32 Gram; replicate both".
   `r2_onehot_ridge(dtype=float32)` casts everything to float32. The historical arithmetic is mixed:
   float32 centring and Gram, float64 ridge, one-hot and solve.
8. **Primary family against PP §13 and CFG multiplicity.** The original has ≤ 3 endpoints including P3
   (output leakage), on native-pass (cell, arm) combinations including noise, with Holm and bootstrap
   p-values. FD has 16 endpoints, untreated only, not conditioned on native pass, P3 demoted, Bonferroni
   simultaneous bounds. P allows "simultaneous bounds under an explicitly dated pre-fit amendment", so this is
   permitted, but it must be dated in AMENDMENT_1.
   - **Interpretation note:** income_prediction/race already has a known historical N0 above τ (s0 final 0.058,
     `final_vs_best.md`). Its P1 decision is then not a C2 statement. Say so before fitting.
9. **Decomposition.** FD's F0–F6 differs from CFG/PP F0–F7: F2 becomes the pure metric, the Brier-skill
   step goes, F7 adaptive goes. This is an amendment. PP §7 also requires reporting cross-scale steps (F1→F2)
   as **verdict changes**, not numeric differences; FD is silent on that.
10. **R06N/R07N.** FD's NL-selected (GBT vs MLP) on outputs and rep+outputs replaces CFG's GBT-only R06N (10
    configs) and the nested R07N slate. Freeze the effective grids per surface.
11. **R05.** FD uses a frequency table only, with smoothing unspecified (the pinned helper uses Laplace
    α = 1). CFG/AT: LR on one-hot y plus a frequency-table check.
12. **U2.** FD uses LR only. CFG/PP: LR + R04-style MLP. FD omits the macro-F1 utility metric, the
    normalised-lift rule and the frontier rule.
13. **Seed and refit variance.** CFG/PP require 3 attacker seeds per selected configuration (refit SD) and
    a 200-permutation null; FD has neither. The noise seed aggregation is consistent with PP §12.
14. **interval.unit for Adult.** CFG says "row"; FD says "record unit (duplicates collapse)". The code
    already merges by record key; amend the CFG text.

## 3. Required by P but omitted or underspecified in FD

- **Controls** (P "PREFIT VALIDATION"):
  - synthetic direct-signal positive control and independent or permuted null, through the same shell/CLI
    path and effective config;
  - P also says the real release need not beat LO for the evaluator to count as working;
  - PP §16's per-cell controls (untreated > R05; permuted-s null, where the cell stops on failure) need an
    explicit keep or drop. The first is the circular one P warns against.
- **Budget:** synthetic recalibration of the executed slate. If the slate exceeds 2 CPU-hours, a registered
  staged schedule; completed core kept distinct from untriggered extensions.
  - Pinned-slate reference (QUICKSTART): 38.5 s per binary unit and 147 s per K = 5 unit for 23 configs × 3
    surfaces at n = 15,060.
  - My rough extrapolation for 26 units with the *pinned* slate is about 0.35 CPU-hours. The FD slate (plus
    LRTs, U2 and B = 20,000 primary bootstraps) needs a real measurement.
- **R11 deferral:** must be named as deferred or unavailable. FD is silent.
- **Assessment-selection prohibition test:** P requires one; FD states the rule but not the test.
- **Code- and manifest-change blocks execution:** P item 8 asks for this; FD lists no lock contents (see
  `amendment_items.md` A20/A21).
- **Inputs the FD quantities need but no manifest declares:**
  - A4 needs the clean attacker_fit reps (`fwd:rep_p0`) declared in the noise manifests.
  - U1 for noise units needs the checkpoint, sha-pinned (`forward.py:119` head). Noise manifests do not
    reference it.
  - Task labels need a new array or file (`amendment_items.md` A12).
- **Saved-prediction gaps:**
  - held-out ρ₁² needs per-row `u`/`v` arrays to get an interval; they are not in FD's `preds.npz` key list;
  - candidate (GBT/MLP) assessment predictions: say whether they are stored and confirm they are replay-only.
- **Quantile interpolation for the primary bounds:** tail 62.5 of 20,000 is non-integer. Freeze the
  `np.quantile` method.
- **Task-label support rule:** none for U1, U2 or LO. occupation_group class 5 (Armed-Forces) has 3 / 1 / 1
  rows in fit / val / assessment. Macro AUC and LO cells for it are not estimable. Define the utility support
  rule (for example, reuse 100/30/100 over task classes with coverage).
- **Reuse beyond the outputs surface.** P: identical objects reuse fitted references and are not counted as
  independent evidence. FD marks noise outputs as reused. The same applies to:
  - LO and the LO contrast for all 18 noise units (identical to untreated income_prediction/sex);
  - U1 and U2 across untreated pairs of the same purpose: 3 distinct objects, not 8;
  - R08 (= untreated outputs NL-selected).
- **Near-ceiling utility.** The employment_analysis and education_assessment task labels are deterministic
  recodings of one-hot input columns (occupation and education). Historical frozen-head accuracy was
  0.9995 / 0.9999 (`per_seed_results.json`). U1 and U2 for those 6 units will be near ceiling, and LO for
  those pairs is P(s | recoded input column). Report this so the utility contrasts are not over-read.
- **Projection track:** P says not to substitute the old recount. FD is silent; list it as unavailable in
  coverage.
- **Private archive index and hash pins:** required by P's output list. FD covers only `run_v1/`.
- **τ sensitivity grid {0.01, 0.02, 0.10}:** not stated.
- **Versioning:** if a consumed scientific component changes after scoring, version it. FD has no rule.

## 4. Task labels (PCRL b96c412, `pcrl/data/adult.py`)

**Purposes** (`get_adult_purposes`, `:447-484`):

| Purpose | Allowed task | Classes | Disallowed attributes |
|---|---|---|---|
| income_prediction | income | 2 | race, sex |
| employment_analysis | occupation_group | 6 | race, age_group, marital_status |
| education_assessment | education_level | 4 | sex, race, income |

The stored head logits widths agree: 2, 6 and 4 (`cache/adult_s0_test.npz`).

**Label definitions** (`_preprocess`, `:345-369`; after `dropna`, `:311`):

- **income:** `{"<=50K":0, "<=50K.":0, ">50K":1, ">50K.":1}`, fillna 0.
- **employment_analysis → `occupation_group`:** `df["occupation"].map(OCCUPATION_GROUPS).fillna(2)` (`:20-35`).

  | Code | Group | Occupations |
  |---|---|---|
  | 0 | Professional | Prof-specialty, Tech-support |
  | 1 | Executive | Exec-managerial, Adm-clerical |
  | 2 | Service | Other-service, Priv-house-serv, Protective-serv, Handlers-cleaners |
  | 3 | Sales | Sales |
  | 4 | Manual | Machine-op-inspct, Transport-moving, Craft-repair, Farming-fishing |
  | 5 | Military | Armed-Forces |

- **education_assessment → `education_level`:** `df["education"].map(EDUCATION_LEVELS).fillna(0)` (`:37-54`).

  | Code | Level | Education values |
  |---|---|---|
  | 0 | Less than HS | Preschool, 1st-4th, 5th-6th, 7th-8th, 9th, 10th, 11th, 12th |
  | 1 | HS | HS-grad |
  | 2 | Some college | Some-college, Assoc-voc, Assoc-acdm |
  | 3 | College+ | Bachelors, Masters, Doctorate, Prof-school |

No test-split value is unmapped, so `fillna` never fires. Both labels are deterministic functions of model
input columns (the one-hot occupation and education features).

**Class counts.** Re-derived from `data/adult/adult.test` with the b96c412 rules and joined on `row_id` to
the existing `labels.npz` roles. These are label marginals only; there is no fit and no attribute×task cross
tab.

| Label | All 15,060 | attacker_fit 7,571 | attacker_val 2,239 | assessment 5,250 |
|---|---|---|---|---|
| income [0, 1] | 11,360 / 3,700 | 5,721 / 1,850 | 1,705 / 534 | 3,934 / 1,316 |
| occupation_group [0..5] | 2,478 / 3,811 / 2,713 / 1,824 / 4,229 / **5** | 1,288 / 1,917 / 1,310 / 915 / 2,138 / **3** | 342 / 562 / 427 / 272 / 635 / **1** | 848 / 1,332 / 976 / 637 / 1,456 / **1** |
| education_level [0..3] | 1,920 / 4,943 / 4,372 / 3,825 | 965 / 2,441 / 2,238 / 1,927 | 282 / 741 / 659 / 557 | 673 / 1,761 / 1,475 / 1,341 |

**Implementation notes:**
- **Source and order.** Take these from the b96c412 `AdultDataset(split="test", norm_stats=train.norm_stats).task_labels`. The order equals `row_id`, which I checked through `sex` and `income`.
- **Storage.** Recommendation: a new `task_labels.npz` (`row_id`, `income`, `occupation_group`,
  `education_level`) plus a reference check against the dataset object. Rewriting `labels.npz` would change
  its sha.
- **Reuse.** Untreated U1, U2 and LO need `y_task` per purpose. Noise units use `income`.
