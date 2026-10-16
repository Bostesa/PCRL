# Access facts for the per-experiment threat model (adversary reviewer, Phase 1)

2026-10-16. Fact-finding only: these are suggestions for the manuscript owner, and the manuscript was not edited.

## 0. Anchors, conventions and data handling

- **Manuscript anchor.** "HEAD n" means line n of the committed `papers/combined_empirical_v2/main.tex` at
  `133d2c740`, SHA-256 `cf0b25f2…`, which is the verified adversary-patch-v2 output
  (`git show c9709a2:publications/pcrl-mechanism-design-v1/results/ADVERSARY_AND_MOTIVATION.md`, line 169).
  The working copy changed while this pass was running. "WC n" means line n of the working copy as observed
  (SHA-256 `1cd2bc69…`, 569 lines). That copy contains a `tab:access` placeholder and a TBD "noise attacker
  sentence" at WC 194–195. Section 3.6 supplies the facts for that sentence.
- **How the working copy numbers its experiments.** WC Exp. 1 covers studies 1 and 2 below (the native check and the
  matched benchmark). Exp. 2 is study 3 (output-aware). Exp. 3 is study 4 (output diagnosis, including the recipient
  pair and the useful-task FARE). Exp. 4 is study 5 (useful heads). Sources: WC 278–285 and 343–353.
- **Code citations** are to this worktree. I checked with `git diff --stat` that the code here matches each study's
  commit: `stored_model_eval/` against `70f978ffc`, `odx/` against `dc6a2e961` and `cap/` against `3b5065805` show no
  differences. The one exception is `oar/study.py:63`. That line changed from `<U16` to `<U32` in repair R1, and the
  change leaves every role's set of rows unchanged
  (`results/combined_output_diagnosis_v1/AMENDMENT_R_2026-10-03.md:6-11`).
- **Rows and groups.** "Rows" are records in a role. "Groups" are de-duplicated record units, keyed by a canonical
  record key; duplicate records share a group and a role. Adult and HMDA have **no household or applicant
  identifiers**:
  - `results/combined_output_diagnosis_v1/PROTOCOL.md:130-132`;
  - `results/combined_matched_removal_benchmark_v1/PROTOCOL.md:95`.

  Roles are hash splits of the record key:
  - Adult `pilot-roles-v1`, shares 0.50 / 0.15 / 0.35 (`results/combined_evaluation_preparation_v1/PILOT_LOCK.json:18`);
  - HMDA `bench-roles-hmda-v1` (`results/combined_matched_removal_benchmark_v1/EFFECTIVE_PROTOCOL.json:198-205`).
- **Path abbreviations:** `MRB/` = `results/combined_matched_removal_benchmark_v1/`, `OAR/` =
  `results/combined_output_aware_removal_v1/`, `ODX/` = `results/combined_output_diagnosis_v1/`, `CAP/` =
  `results/combined_analysis_paper_v1/`.
- **Data handling.**
  - No per-record file, array, model file or private cache was opened.
  - Nothing was fitted, run or computed. The only exception is one read-only `python3 -I` one-liner, which printed fields
    from six committed selection JSONs.
  - Every number below is copied from a committed report, aggregate table, lock or protocol.

## Key findings

1. **The matched benchmark ran noise-aware attackers, and the paper does not report them.**
   - Two exploratory Gaussian likelihood-ratio attackers (LRT-A2 and LRT-A4) know the noise scale σ. LRT-A4 is also
     fitted on clean, pre-noise representations of the auxiliary records.
   - At σ\* both recovered more from the noisy features than the reported slate did:
     - Adult: 0.546 and 0.563, against 0.528;
     - HMDA: 0.526 and 0.575, against 0.513.
   - So HEAD 137–139 ("our attackers use no such knowledge beyond this refitting") is true only of the attackers whose
     numbers are reported (§3.6).
2. **The data budgets differ by study, and the paper states none of them.** attacker_fit / attacker_val, in rows:

   | Study | Adult | HMDA |
   |---|---|---|
   | Matched benchmark | 7,571 / 2,239 | 6,794 / 2,089 |
   | The three later studies | 6,065 / 2,235 | 5,388 / 2,082 |

   - The tree and MLP attackers also use an internal 10 % of attacker_fit for early stopping.
   - The encoders' native check uses no attacker roles. It is in-sample on all 15,060 Adult and 13,661 HMDA test rows
     (§1).
3. **Both defense operating points were chosen with this attacker slate on attacker_val.**
   - σ\* is the smallest σ whose attacker_val AUC for the slate is at most 0.55.
   - The FARE nominee is the admissible configuration with the lowest attacker_val AUC for the slate.
   - HEAD 214–215 calls attacker_fit and attacker_val "attacker fitting and selection" only, and HEAD 158 says
     "validation only". Neither mentions that the owner selects on these roles too (§3, §4).
4. **The label-only reference receives the scored record's true task label.** This contradicts:
   - HEAD 123 ("not … its true task label");
   - the WC table caption at WC 201 ("no attacker receives … its task label").
5. **Two phrases carry the later ACS evaluator's features into the Adult/HMDA studies.**
   - "Chosen and oriented" (HEAD 128, still at WC 170): no orientation step exists in these studies.
   - "Household identifiers" (HEAD 139; the WC now says "group").
6. **Outside access: the committed native-check table gives 11 of 42, not 12 of 42.**
   - The source decision file itself lists only 11 combinations.
   - "12 of 42" appears at HEAD 37, 265 and 282, and at WC 87, 343, 362 and 511 (§10).

---

## 1. Attacker data budgets (item c, every study)

| Study | Dataset | attacker_fit, rows / groups | attacker_val, rows / groups | assessment, rows / groups | Other roles, rows / groups | Source |
|---|---|---|---|---|---|---|
| 1. Native check | Adult | no attacker roles: fit and score on all **15,060** test rows (15,055 units), using their protected labels | — | same rows | — | `MRB/NATIVE_AND_HELDOUT_CHECKS.csv` (`N0_rows` = 15060 on Adult `__A` rows, e.g. line 95); `MRB/notes/admission/rows_provenance.json:20,23`; `MRB/EFFECTIVE_PROTOCOL.json:274-300` |
| 1. Native check | HMDA | in-sample on all **13,661** test rows (13,655 units) | — | same rows | — | `NATIVE_AND_HELDOUT_CHECKS.csv` (`N0_rows` = 13661, e.g. line 344); `rows_provenance.json:98,101` |
| 2. Matched benchmark | Adult | **7,571 / 7,567** | **2,239 / 2,238** | 5,250 / 5,250 | defense_fit 24,127 rows; no certification carve-out; no head holdout | `MRB/PROTOCOL.md:84-85`; `rows_provenance.json:29-38`; `MRB/EFFECTIVE_PROTOCOL.json:210-214`; `MRB/notes/evaluation/INPUTS_CONTRACT_CONSUMED.md:62` |
| 2. Matched benchmark | HMDA | **6,794 / 6,790** | **2,089 / 2,089** | 4,778 / 4,776 | defense_fit 63,704 rows | `MRB/PROTOCOL.md:88-89`; `rows_provenance.json:107-116`; `INPUTS_CONTRACT_CONSUMED.md:62` |
| 3. Output-aware | Adult | **6,065 / 6,061** | **2,235 / 2,234** | 5,243 / 5,243 | cert 1,500 / 1,500; defense_fit 24,127 / 24,113, split into head_fit 19,230 / 19,221 and head_val 4,897 / 4,892 | `OAR/ROLES_AND_SUPPORT.json:2-38`; `OAR/PROTOCOL.md:29-35` |
| 3. Output-aware | HMDA | **5,388 / 5,384** | **2,082 / 2,082** | 4,764 / 4,762 | cert 1,385 / 1,385; defense_fit 63,704 / 63,586, split into head_fit 50,895 / 50,798 and head_val 12,809 / 12,788 | `OAR/ROLES_AND_SUPPORT.json:128-164` |
| 4. Output diagnosis | both | same as study 3 (identical role row-set hashes) | same | same | same | `ODX/repaired/ROLES_EXCLUSIONS_REPAIRED.json:2-47,49-94`; `ODX/PROTOCOL.md:24-30` |
| 5. Useful heads | Adult only | **not restated in the CAP package.** Its roles are declared "`oar-roles-v1`, unchanged from the output-aware study" (`CAP/PROTOCOL_USEFUL_HEADS.md:19`), so the committed sizes are those of study 3 (6,065 / 6,061) | not restated; inherited (2,235 / 2,234) | 5,243 records, "one record per group" | inherited | `CAP/PROTOCOL_USEFUL_HEADS.md:5,19`; `CAP/USEFUL_HEAD_COMPARISON.md:9`; `CAP/LOCK.json` records no role sizes |

**Notes for the budget statement:**
- **Early stopping.** The GBT and MLP attackers set `early_stopping` with `validation_fraction` 0.1: an internal random
  10 % of attacker_fit is held out to decide when to stop, and attacker_val is not used for this
  (`MRB/EFFECTIVE_PROTOCOL.json:489-496,522-525`; `stored_model_eval/recipes.py:71-90`). Studies 3–5 reuse this protocol
  unchanged (`OAR/EXECUTION_LOCK.json:165`; `oar/study.py:24,410-411`). The logistic family fits on all of attacker_fit.
- **The certification carve-out affects every arm.** In studies 3–5, 20 % of attacker_fit groups go to FARE
  certification (`OAR/PROTOCOL.md:25-27`; `oar/study.py:41,68-70`). This shrinks the attacker budget for **every** arm,
  not only FARE. That is the main reason the benchmark's attacker_fit is larger.
- **Exposure exclusion.** The 17 Adult and 42 HMDA training-overlapping test records leave all scored roles in studies 3–5:
  Adult 6 / 4 / 7 and HMDA 21 / 7 / 14 by attacker_fit / attacker_val / assessment
  (`ODX/repaired/ROLES_EXCLUSIONS_REPAIRED.json:39-47,86-94`). `OAR/ROLES_AND_SUPPORT.json:39,165` reports 0 because of
  the `<U16` truncation, but the exclusion was always applied (`ODX/AMENDMENT_R_2026-10-03.md:11`). In study 2 these
  records remain in all three roles (`OAR/CORRECTION_ADDENDUM.md:93`).
- **Per-class attacker_fit counts for the primary cells:**

  | Study and cell | attacker_fit by class | Source |
  |---|---|---|
  | Benchmark, Adult sex | 2,447 / 5,124 | `MRB/notes/admission/support_counts.csv:56-57` |
  | Benchmark, HMDA race | 4,411 / 415 / 1,780 / 121 / 67 | `support_counts.csv:123-127` |
  | Studies 3–5, Adult sex | 1,973 / 4,092 | `OAR/ROLES_AND_SUPPORT.json:53-56` |
  | Studies 3–5, HMDA race | 3,488 / 335 / 1,417 / 95 / 53; classes 3–4 not estimable | `OAR/ROLES_AND_SUPPORT.json:185-191` |
  | Study 4 pair, Adult race | 54 / 187 / 568 / 49 / 5,207; supported classes {1, 2, 4} | `ODX/COVERAGE_AND_SUPPORT.csv:6-10`; `ODX/MULTI_RECIPIENT_RESULTS.md:20` |
- **Owner-side roles.** defense_fit is the encoders' own training split in every study (`MRB/PROTOCOL.md:85,89,92`).

---

## 2. Study 1: the encoders' stored native check (the "12 of 42" check)

The six PCRL Round-4 `final.pt` encoders (epoch 204) were admitted with verified lineage (`MRB/PROTOCOL.md:43-44`). The
reported count is the benchmark's reproduction of the historical value under its lock (`MRB/RESEARCH_DECISION.md:46-61`;
`MRB/VALIDATION.md:21`).

### Surface 1.1: representation only, checked by an in-sample linear predictor

- **a. Input.**
  - The purpose's 64-dimensional representation `h_p` of every test-split record.
  - The check is a closed-form, centred one-hot ridge (λ = 1e-6), fitted and scored on the same rows. Its R² is clamped at 0
    (`pcrl/purposes/verification.py:62-109`; `scripts/eval_round4_dominant_axis.py:158,165-172`;
    `MRB/EFFECTIVE_PROTOCOL.json:274-300`).
  - The benchmark summary describes it as "test split, fit = score; one ridge-fitted affine predictor, squared loss;
    covers outputs: no" (`MRB/GUARANTEE_CARDS.md:257`).
- **b. Auxiliary information.**
  - The protected attribute of the very records being scored (in-sample).
  - No task labels and no inputs.
- **c. Labelled sample.** All 15,060 Adult and 13,661 HMDA test-split rows, including the 17 / 42 rows that also occur in
  training. No attacker_fit or attacker_val role exists (§1).
- **d. Fitting, selection and seeds.**
  - Fitting rows equal scoring rows.
  - Nothing is selected: λ and τ = 0.05 are fixed.
  - There are no attacker seeds.
  - The count runs over 14 pairs × 3 encoder seeds = 42 combinations. The categorisation uses the historical
    mixed-precision variant (`MRB/EFFECTIVE_PROTOCOL.json:282`).
- **e. Knowledge of the defense.** Not applicable: this is the method's own check. The training-time constraint is a
  different object: a one-hot ridge R² (λ = 1e-4) computed on each training minibatch inside a proxy-Lagrangian
  (`pcrl/training/v2_trainer.py:1-25,778-796,879-884`; `pcrl/training/losses.py:475-518`).
- **f. Linkage.** Single surface; none.
- **g. Releases.** One representation per record.
- **h. Untested.**
  - Everything beyond in-sample linear prediction from the representation.
  - Its held-out counterpart G1 was run in the benchmark: Adult was established below τ and HMDA unresolved
    (`MRB/RESEARCH_DECISION.md:72-81`). The manuscript does not report G1.
  - Nonlinear recovery, outputs and pairs are covered only by the other studies.
- **Manuscript.** HEAD 37–38, 108–109, 237, 265, 281–283. The manuscript's description ("in-sample one-hot ridge") is
  accurate, but see §10 for the count.

---

## 3. Study 2: the matched removal benchmark (`70f978ffc`; WC Exp. 1)

**Common to every surface**

- **b. Auxiliary information.** The releases of the attacker_fit records under the same arm and release seed, with
  their protected attribute. A second labelled set, attacker_val, is used for selection. Raw inputs never reach the
  evaluator: `features_npz` is "pinned but not read" (`MRB/notes/evaluation/INPUTS_CONTRACT_CONSUMED.md:17`). Attackers
  never get task labels (`stored_model_eval/bench.py:928,946,983-986`).
- **c. Labelled sample.** Adult 7,571 / 2,239 rows and HMDA 6,794 / 2,089 rows (§1).
- **d. Fitting, selection and scoring.**
  - Grids: L with C ∈ {0.01, 0.1, 1, 10, 100}, GBT with 20 configurations and MLP with 18, each fitted on attacker_fit.
  - Family and hyper-parameters are chosen by attacker_val log loss (clip 1e-12). NL is GBT or MLP, whichever has the
    lower attacker_val log loss.
  - Scoring is the macro one-vs-rest AUC on assessment, over supported classes.
  - Sources: `MRB/EFFECTIVE_PROTOCOL.json:351-565`; `bench.py:648-661,944-968`.
- **d. Seeds.**
  - Attacker seeds {0, 1, 2}. Seed 0 is the grid fit; the selected recipe is refit at seeds 1 and 2. L is deterministic
    and aliased (`bench.py:664-694`; `EFFECTIVE_PROTOCOL.json:556-565`).
  - Release seeds {0, 1, 2} for noise.
  - Encoder, release and attacker seeds are averaged inside each bootstrap replicate (`MRB/notes/BENCH_DESIGN.md:151-153`).
- **e. Knowledge of the defense.** Attackers are refitted on the arm's released attacker_fit rows. The benchmark's own
  access table says "implicit only … for B/C this equals simulating the fixed fitted map on its own population (alias,
  no stronger attack claimed); no explicit mechanism model" (`MRB/RELEASE_ACCESS_TABLE.csv:2-5`;
  `MRB/notes/BENCH_DESIGN.md:117-118`; `MRB/EFFECTIVE_PROTOCOL.json:582,696`). No attacker reads the LEACE map, the
  frozen head or σ, except LRT-A2 and LRT-A4 (§3.6). **The defense was selected with the attacker slate:**
  - σ\* is "the smallest σ whose attacker_val C_rep NL macro AUC (mean over encoder seeds x release seeds, attacker seed
    0) is <= 0.55" (`MRB/EFFECTIVE_PROTOCOL.json:645-652`; `MRB/SIGMA_STAR_SUMMARY.json:1-45`).
  - The result was σ\* = 2.0 for Adult and 4.0 for HMDA.
- **f. Linkage.** There is no pair. The permission table states "coalition_membership: none (single-recipient
  benchmark) … not evaluated in this benchmark" (`MRB/PERMISSION_TABLE.csv:1-7`).
- **g. Releases.**
  - **One release per record.** A, B and C are deterministic.
  - **D is "one persistent draw per person: default_rng(release seed) over the matrix of scored-role rows … in ascending
    row_id order"** (`MRB/EFFECTIVE_PROTOCOL.json:255-272`; `bench.py:602-633`;
    `stored_model_eval/releases.py:1-28`).
  - The three release seeds are separate realizations. Each is attacked separately and the results are averaged; they
    are never pooled for one record.
  - Repeated release (access level A3) was staged and not run: "persistent contract: N collapses to 1"
    (`MRB/EFFECTIVE_PROTOCOL.json:694-701`; `stored_model_eval/access.py:3-24`).
  - No input is chosen by the attacker.
- **h. Untested.**
  - Repeated or fresh-noise queries (A3).
  - The insider (A5: `access.py:8-9`).
  - General adaptive attacks beyond refitting, including use of the LEACE map (R11 alias, `EFFECTIVE_PROTOCOL.json:696`).
  - Oracle LEACE (`EFFECTIVE_PROTOCOL.json:697`).
  - Own-head outputs; any output format other than full logits; coalitions.
  - σ-aware attacks on the representation-plus-output surface (`MRB/RELEASE_ACCESS_TABLE.csv:16-17`).
  - Noise on 8 of the 12 Tier-2 pairs. The budget stop cut 447 units (`MRB/RESEARCH_DECISION.md:20-35`).

**Native check of the LEACE arms (owner side).** The implementation bound is a whitened residual at most `svd_tol`,
computed on defense_fit (24,127 / 63,704 rows) with the protected one-hot (`MRB/EFFECTIVE_PROTOCOL.json:301-310`). The
manuscript reports it at HEAD 283.

### 3.1 Representation only (C_rep): arms A, B, C and D

- **a. Input.** 64 columns in every arm (`BENCH_DESIGN.md:43-48`; `bench.py:607-633`; `RELEASE_ACCESS_TABLE.csv:2-5`):
  - A: the untreated `h_p`;
  - B / C: the official LEACE affine map (concept-erasure 0.2.4), fitted on defense_fit;
  - D: `h_p + N(0, σ²I)`.
- **b–h.** As above. There are three attacker seeds, and for D also three release seeds.
- **Manuscript.** Table 1 at HEAD 266–268; HEAD 283–285.

### 3.2 Representation plus historical clean output (C_rep_plus_clean_out)

- **a. Input.** The arm's release (64 columns) concatenated with the purpose's frozen-head clean **full logit vector**
  (K_t columns, 2 in the primary cells). The logits are computed from the untreated `h` and are identical across arms
  (`BENCH_DESIGN.md:82`; `bench.py:1126,1203`; `stored_model_eval/surfaces.py:39-56`; `MRB/GUARANTEE_CARDS.md:184-185,222-223`).
- **d. Selection.** The selection set adds two candidates on attacker_val:
  - **ignore-rep** (the fitted outputs-only model);
  - **ignore-outputs** (the C_rep model).

  The chosen candidate is the one scored (`bench.py:994-1024,1220-1234`; `EFFECTIVE_PROTOCOL.json:540-555`).
- **f. Linkage.** By construction one recipient holds features and output: they are concatenated row by row.
- **Manuscript.** Table 1 at HEAD 268, "plus clean outputs 0.778 / 0.758".

### 3.3 Output only (outputs_only)

- **a. Input.** The frozen head's clean full logits, fitted once per (dataset, encoder seed, purpose, attribute) and
  shared by every arm (`EFFECTIVE_PROTOCOL.json:586-592`; `bench.py:762-818,970-979`). It is only the purpose's own head
  (`OAR/RELEASE_CONTRACTS.md:11-15`).
- **e. Knowledge of the defense.** No defense acts on this surface.
- **h. Untested.** Probabilities and hard decisions; these were added in study 3.
- **Manuscript.** HEAD 289–290, "output-only recovery exceeds the label-only reference by 0.177 / 0.236".

### 3.4 Label-only reference (a comparator, not a recipient)

- **a. Input.** **The scored record's true task label.** A frequency table P(s | y), with Laplace α = 1, maps it to the
  attribute (`bench.py:1045-1051,1086`; `stored_model_eval/recipes.py:192-197`; `MRB/RELEASE_ACCESS_TABLE.csv:26`;
  `MRB/GUARANTEE_CARDS.md:246`).
- **b. Auxiliary information.** Task labels and protected labels of the attacker_fit records.
- **d. Fitting and selection.** Fitted on attacker_fit. Nothing is selected and it is deterministic.
- **Manuscript.** HEAD 289–290.

### 3.5 Constant reference

The constant is the attacker_fit class prior; it reads no input (`EFFECTIVE_PROTOCOL.json:594-602`).

### 3.6 Exploratory noise-aware likelihood-ratio attackers on arm D (not in the manuscript)

- **a. Input.** The same noisy 64-column representation as 3.1. These attackers run only on arm D and only on the
  representation surface (`MRB/RELEASE_ACCESS_TABLE.csv:8-9,16-17`).
- **b, e. What they know.**

  | | Training data | Extra knowledge | How the benchmark describes it |
  |---|---|---|---|
  | **LRT-A2** | released attacker_fit rows with labels | σ, "public mechanism parameter" | "defense-informed, released rows only" |
  | **LRT-A4** | **clean, pre-noise `h` of the attacker_fit rows** with labels | Σ = σ²I | "white-box population stress test; never refutes a guarantee" |

  - Both are class-conditional Gaussian likelihood-ratio tests. A2 estimates each class covariance from the releases and
    subtracts σ²I; A4 estimates it from the clean rows. Both floor the eigenvalues at 1e-6·trace/d and take the class
    prior from attacker_fit.
  - They know the noise law: isotropic Gaussian with known σ. This is the AAAI "Tier 2" kind of noise awareness, not a
    general adaptive or white-box attack.
  - Sources: `bench.py:1026-1043`; `stored_model_eval/recipes.py:133-184`; `MRB/EFFECTIVE_PROTOCOL.json:566-581`;
    `MRB/notes/BENCH_DESIGN.md:115`; `MRB/GUARANTEE_CARDS.md:226-229`.
- **d. Fitting and selection.** Closed form, fitted on attacker_fit. Nothing is selected, the attacker seeds are aliased,
  and they are scored on assessment. The benchmark labels them exploratory (`BENCH_DESIGN.md:184`).
- **g. Releases.** One draw per record per release seed.
- **Committed values at σ\*.** Means over encoder × release seeds, with 90 % exploratory unadjusted intervals:

  | Cell, σ\* | Slate (NL) | LRT-A2 | LRT-A4 | Source |
  |---|---|---|---|---|
  | Adult income / sex, σ = 2 | 0.528 [0.523, 0.533] | 0.546 [0.539, 0.553] | 0.563 [0.558, 0.568] | `MRB/RECOVERY.csv:7057,7101,7105` |
  | HMDA underwriting / race, σ = 4 | 0.513 [0.508, 0.518] | 0.526 [0.519, 0.533] | 0.575 [0.570, 0.580] | `MRB/RECOVERY.csv:10155,10199,10203` |

  There is no committed paired contrast between an LRT and the slate: `MRB/MATCHED_COMPARISONS.csv` has no LRT row.
  RESEARCH_DECISION, ADVISOR_BRIEF and HANDOFF do not mention these results.
- **Why this matters.** Table 1 (HEAD 268) reports only the slate. HEAD 137–139 says the attackers use no defense
  knowledge beyond refitting. The noise row is therefore not the strongest attacker the benchmark ran on that surface.

### 3.7 Held-out linear statistics on the representation (G1, G2, ρ₁²)

- **a. Input.** The arm's 64-column release.
- **b–d.** G1 is a ridge with fixed λ = 1e-6, fitted on attacker_fit and scored on assessment. G2 selects its ridge on
  attacker_val. ρ₁² is the top canonical correlation, fitted on attacker_fit and scored on assessment
  (`MRB/EFFECTIVE_PROTOCOL.json:316-349`; `bench.py:930-942,1067-1074`).
- **e–g.** As 3.1.
- **Manuscript.** HEAD 283–285, "held-out ρ₁² … on one already-used held-out split".

---

## 4. Study 3: the output-aware removal study (`f7425b15c`; WC Exp. 2)

**Common to every surface**

- **b. Auxiliary information.** Labelled attacker_fit releases with the protected attribute, and attacker_val for
  selection. Raw inputs never reach the code: it reads only labels, roles and the cached forward pass
  (`oar/study.py:57-101`). The recipient "never receives sensitive labels, row identifiers or lookup keys"
  (`OAR/RELEASE_CONTRACTS.md:5-8`).
- **c. Labelled sample.** Adult 6,065 / 2,235 rows and HMDA 5,388 / 2,082 rows, after the 20 % certification carve-out
  (§1).
- **d. Fitting, selection and scoring.**
  - The slate is the benchmark's, reused unchanged (`OAR/EXECUTION_LOCK.json:165-167`; `OAR/PROTOCOL.md:95-107`).
  - Surfaces that combine features and output add the ignore-rep / ignore-out candidates, selected on validation. The
    selected component unit is the one scored (`oar/study.py:277-322`; `oar/infer.py:55-70`).
  - R is "averaged over attacker seeds, then release seeds, then encoder seeds" (`OAR/PROTOCOL.md:132-133`).
  - Three attacker seeds, three noise draws and three encoder seeds.
- **e. Knowledge of the defense.**
  - Attackers refit on releases only. No attacker reads the LEACE map, the FARE tree or its thresholds, σ, or the head
    parameters.
  - **The owner's FARE choice used the attacker slate:** "admissible: U2 val acc >= untreated U2 val acc - 0.01; min val
    NL(as0) macro AUC on FARE features" (`OAR/EXECUTION_LOCK.json:291`; `oar/fare_run.py:25-47`;
    `OAR/run_records/selection/{adult,hmda}__s{0,1,2}.json`). The U2 probe is fitted on attacker_fit task labels and
    selected on attacker_val (`oar/study.py:325-344`).
  - σ\* is taken from the benchmark, where it was selected the same way (`OAR/PROTOCOL.md:55`; `oar/study.py:38`).
- **e. The cell-conditional attacker (CC).**
  - It applies to finite releases. Each released vector, rounded to 9 decimals, is hashed into a cell.
  - It tabulates attacker_fit counts of the attribute per cell and smooths them toward the attacker_fit prior with
    Dirichlet α ∈ {0.1, 1, 10, 100}, choosing α by attacker_val log loss. Unseen cells get the prior.
  - It is deterministic and uses **no** FARE tree, thresholds or objective, only the released cell values.
  - Sources: `oar/study.py:179-236`; `OAR/EXECUTION_LOCK.json:168-177`.
  - **Terminology hazard:** this package calls it the "defense-aware attacker" (`OAR/PROTOCOL.md:109-114`;
    `oar/study.py:180`; `OAR/RUN_STATUS.md:13`; `OAR/OUTPUT_SURFACES.md:38`).
  - CC equals NL wherever both were run: F\* features 0.550 / 0.501; O_hard 0.513 / 0.505, a difference of 0.0
    (`OAR/EXPLORATORY.csv:5,27,52-53,57,79,104-105`).
- **f. Linkage.** Features and output are concatenated row by row (`oar/run.py:85,93,102`; `oar/fare_run.py:91,95`).
  There is no pair: "Any coalition of purposes would be a different contract, and it is not evaluated here"
  (`OAR/RELEASE_CONTRACTS.md:11-15`).
- **g. Releases.** "A single, persistent, per-person release. There are no repeated queries and no fresh noise per
  query" (`OAR/RELEASE_CONTRACTS.md:5-7`). The noise draws are identical to the benchmark's
  (`oar/study.py:80-82,118-124`; `OAR/EXECUTION_LOCK.json:186-197`).
- **h. Untested.**
  - Coalitions; repeated or fresh-noise queries.
  - Any σ-aware attacker (this study has no LRT).
  - Attacks that use FARE's tree or objective.
  - The kernelized eraser, which was optional and not run (`OAR/RUN_STATUS.md:60`).
  - Own-head formats other than log-probabilities; these were added in study 5.
  - CC was not fitted on the own-head outputs of F\* and FZ (`oar/fare_run.py:94-97`) or on mixed releases.
  - Inputs, parameters and insiders.

### 4.1 Output only, frozen head: O_full, O_prob, O_hard

- **a. Input.** One of three formats: 2 logits, 2 softmax probabilities, or the argmax as a 2-column one-hot. CC is
  also fitted on O_hard (`oar/run.py:73-77`; `oar/study.py:153-159`; `OAR/RELEASE_CONTRACTS.md:45-58`).
- **Manuscript.** Table 1 at HEAD 269 (0.776 vs 0.513 and 0.768 vs 0.505).

### 4.2 Features only: A, B, C, D\*, F\*, FZ

- **a. Input** (`OAR/RELEASE_CONTRACTS.md:19-27`; `oar/run.py:79-84,97-100`; `oar/fare_run.py:73-74,83-88`):
  - A, B, C and D\* (3 draws): 64 columns;
  - F\* / FZ: a one-hot FARE cell index. The Adult nominee has 10 / 50 / 10 cells. The HMDA nominee has 4 / 1 / 4
    cells, and HMDA seed 1 is a constant release (`ODX/FARE_USEFUL_TASK_RESULTS.md:18-23`).
- **Fitting of FARE.** The FARE tree is fitted on defense_fit with the task label and the protected attribute, for
  fitting only (`OAR/PROTOCOL.md:56`). CC is also fitted on the F\* grid and on FZ.
- **Manuscript.** Table 1 at HEAD 273–274.

### 4.3 Features plus historical clean logits (view 2, the bypass): A, B, C, F\*, FZ

- **a. Input.** The features concatenated with the frozen head's clean 2-column logits (`oar/run.py:85-87`;
  `oar/fare_run.py:90-92`; `OAR/RELEASE_CONTRACTS.md:34`).
- **Manuscript.** Table 1 at HEAD 275 (0.788 / 0.768); HEAD 290–291.

### 4.4 Features plus own-head output (view 3): A, B, F\*, FZ

- **a. Input.** The features concatenated with the release head's 2-column **log-probabilities**.
  - The head is a logistic regression over the U2 grid, fitted on head_fit (19,230 / 50,895 rows) and selected on
    head_val (4,897 / 12,809 rows). Both roles lie inside defense_fit.
  - At runtime the head reads only the features (`oar/study.py:133-149,347-369`; `oar/run.py:89-95`;
    `OAR/RELEASE_CONTRACTS.md:35-43`).
- **Manuscript.** Reused by study 5 (HEAD 310–312, 367–369).

### 4.5 Own-head output only (O_headA, O_headB, O_headF, O_headFZ)

- **a. Input.** 2-column log-probabilities. No CC was fitted (`oar/run.py:92`; `oar/fare_run.py:94`).
- **Manuscript.** Reused by study 5 as its output-only "full" format (HEAD 310–312).

### 4.6 Noise plus output: D\* + O_full, O_prob or O_hard

- **a. Input.** Noisy features concatenated with a clean output in one of the three formats (`oar/run.py:101-104`;
  `OAR/RELEASE_CONTRACTS.md:63-64`).
- **Manuscript.** Not in the HEAD table. `OAR/OUTPUT_SURFACES.md:51-56` reports noise plus hard at 0.527 / 0.510.

### 4.7 Label-only and constant references

Fitted on attacker_fit. The label-only reference receives the scored record's true task label (`oar/study.py:372-383`;
`OAR/PROTOCOL.md:59`).

### 4.8 FARE certificate (native, owner side)

- **Data.** The certification role: 1,500 Adult and 1,385 HMDA rows. δ = 0.05, with a 50 / 50 split.
- **Scope.** It bounds the demographic-parity distance of classifiers that read only the cell. It "is not an
  attack-AUC bound" and does not cover releases that include outputs (`OAR/PROTOCOL.md:176-182`;
  `oar/fare_run.py:98-111`).
- **Manuscript.** HEAD 286–287. The manuscript does not state the certification sizes.

---

## 5. Study 4: the output-leak diagnosis (`dc6a2e961`; WC Exp. 3)

**Common to every surface**

- **b, c. Auxiliary information and budget.** As in study 3, with `oar-roles-v1` repaired (`ODX/PROTOCOL.md:24-30`;
  `odx/run.py:58-65`).
- **d. Fitting and selection.**
  - Each format unit is attacked by the benchmark slate.
  - A **bank** selects one format "once per encoder seed" using "the NL attacker's attacker-seed-0 validation log
    loss". Ties go to the candidate listed first, and L is not a candidate (`ODX/PROTOCOL.md:72-89`;
    `ODX/LOCK.json:595-611`; `odx/run.py:98-119`).
  - The reported recovery always uses the NL recipe (`odx/infer.py:59-68`).
- **e. Knowledge of the defense.** Stages S2–S4 release **untreated** frozen or refitted heads, so no defense is
  involved. CC is fitted on hard surfaces but is not used in any endpoint.
- **g. Releases.** One deterministic output per record.

### 5.1 Output only, frozen head: primary cells and the 14 stored pairs

- **a. Formats** (`odx/surfaces.py:13-45`; `odx/run.py:39-43,173-187`; `ODX/PROTOCOL.md:60-83`):

  | Format | What the recipient receives | Columns |
  |---|---|---|
  | full | the logits | K |
  | dc | (centred, offset) | K+1, or 2 if binary |
  | centred | binary: the margin d; multiclass: the centred K-vector | 1 or K |
  | prob | the probabilities | K |
  | offset | the common offset c | 1 |
  | hard | one-hot argmax | K |
- **Banks.** The full bank is {full, dc, centred, prob}; the offset-free bank is {centred, prob}.
- **f. Linkage.** None.
- **h. Untested.** Any feature defense; formats of defended heads.
- **Manuscript.** HEAD 270–272, 293–302; HMDA fair_lending at HEAD 296–297.

### 5.2 Output only, refitted head HEAD\_\_A (primary cells)

- **a. Input.** The same formats, applied to the output-aware HEAD\_\_A log-probabilities. Here the full bank carries no
  more information than the offset-free bank (`ODX/PROTOCOL.md:39-41,91-94`; `odx/run.py:171-172`).
- **Manuscript.** HEAD 302–304.

### 5.3 Label-only and constant references

As in study 3 (`odx/run.py:164-169`).

### 5.4 Recipient pair (S4): Adult income and employment recipients, attribute race

- **a. Input.** The two recipients' frozen-head outputs, concatenated row by row under one contract
  (`odx/run.py:44-45,193-211`; `ODX/LOCK.json:228-240`):

  | Contract | Income part | Employment part | Columns |
  |---|---|---|---|
  | full | 2 logits | 6 logits | 8 |
  | centred | margin | centred 6-vector | 7 |
  | hard | one-hot of 2 | one-hot of 6 | 8 |

  CC is also fitted on the hard contract.
- **b. Auxiliary information.** Race labels, plus the **linked** pair outputs of the attacker_fit records. The labelled
  sample is itself perfectly linked.
- **c. Labelled sample.** Adult 6,065 / 2,235 rows. Race classes {1, 2, 4} are supported (`ODX/MULTI_RECIPIENT_RESULTS.md:20`).
- **d. Pair bank.** The bank holds the pair attacker plus each single recipient's attacker. For the "full" contract each
  single comparator is that recipient's full bank, for "centred" its offset-free bank, and for "hard" its hard unit.
  Selection is on attacker_val (`odx/run.py:212-214`; `ODX/PROTOCOL.md:168-176`).
  - The pair attacker sees only the contract's format, while the single comparators under "full" and "centred" choose
    among formats. The comparison is therefore asymmetric, and the asymmetry works against the pair.
  - Seed 2 under the full contract selected the income-only attacker (`ODX/S4_ENDPOINTS.csv:2-3`;
    `ODX/MULTI_RECIPIENT_RESULTS.md:29-31`).
- **f. Linkage.** Perfect linkage is granted by construction. Row ids, groups, race labels, roles and exclusions are
  asserted identical across the two purposes (`odx/run.py:199-203`; `ODX/MULTI_RECIPIENT_RESULTS.md:8-9`).
- **h. Untested.**
  - Only one coalition was evaluated: one dataset, two purposes, one attribute, and untreated frozen heads.
  - Not evaluated: coalitions that include features or defended outputs; HMDA; more than two recipients; imperfect
    linkage.
- **Manuscript.** HEAD 45–46, 397–406.

### 5.5 Useful-task FARE (S5): Adult employment_analysis / age_group

- **a. Input.** One of three surfaces, each scored with the ignore-one rule after amendment S1:
  - features only;
  - own-head output only;
  - features plus own head (the "complete contract").

  The arms are A (64 columns), B (the benchmark's target-LEACE map), F (one-hot FARE cells: 5 / 5 / 11) and FZ
  (`odx/fare5.py:88-152`; `ODX/FARE_USEFUL_TASK_RESULTS.md:46-69`; `ODX/AMENDMENT_S1_2026-10-03.md:1-24`).
- **c. Labelled sample.** Adult, as in study 3.
- **d, e. How the cell and the nominee were chosen.**
  - The screen used attacker_val task labels and the frozen head's validation gain (`odx/fare5.py:23-50`).
  - Among feasible configurations, the nominee has the lowest attacker_val NL AUC. A configuration is feasible if its U2
    validation accuracy is within one point of untreated and it keeps at least 80 % of the gain
    (`odx/fare5.py:126-130`; `ODX/PROTOCOL.md:204-208`).
  - The certificate was vacuous or unavailable (`ODX/FARE_USEFUL_TASK_RESULTS.md:75-88`).
- **Manuscript.** HEAD 379–385.

---

## 6. Study 5: the useful-head comparison (`a0de44953` to `3b5065805`; WC Exp. 4)

**Common to every surface**

- **b, c. Auxiliary information and budget.** Adult `oar-roles-v1`, inherited (§1). Labelled releases with sex.
- **d. Fitting and selection.**
  - The benchmark slate, with the ignore-one rule on features-plus-output surfaces and the banks selected per encoder
    seed by validation log loss.
  - Three attacker seeds.
  - Resolution follows the bank, then the plus rule, then the alias (`cap/infer.py:37-52`; `CAP/PROTOCOL_USEFUL_HEADS.md:61-66`).
- **e. Knowledge of the defense.**
  - Attackers refit only. The heads, LEACE maps and FARE trees are reused, and no attacker reads their parameters
    (`CAP/PROTOCOL_USEFUL_HEADS.md:14-25`).
  - The F arm is the output-aware nominee, which was selected on attacker_val with the slate
    (`CAP/PROTOCOL_USEFUL_HEADS.md:22`).
  - CC is fitted on every new F / F0 output surface and on every hard surface, and reported only descriptively (NLDA)
    (`cap/run.py:118,125-132`; `CAP/PROTOCOL_USEFUL_HEADS.md:65`).
- **f. Linkage.** Features plus own output are concatenated row by row (`cap/run.py:128`).
- **g. Releases.** One release per record.
- **h. Untested.**
  - One cell only, and no HMDA.
  - "No repeated-query or multi-recipient attack in this cell" (`CAP/USEFUL_HEAD_COMPARISON.md:141-146`).
  - Only one head family; no noise or C arm.

### Surfaces

| # | Surface | a. What the recipient receives | Notes | Manuscript |
|---|---|---|---|---|
| 6.1 | Output only, own head | the logistic head's output for each arm (A, B, F, F0) in one of five formats: full (2 log-probabilities), dc, centred (margin), prob, hard (one-hot) | full bank {full, dc, centred, prob}; offset-free bank {centred, prob}; "full" is an alias of the output-aware O_head\* units (`cap/run.py:116-134`; `CAP/PROTOCOL_USEFUL_HEADS.md:29-44,52-66`) | the primary endpoints use centred and hard (HEAD 314–315; Table 2 at HEAD 327–347; HEAD 355–365) |
| 6.2 | Features plus own output | the features plus one of full, centred, prob or hard | full bank {full, centred, prob}; "full" is an alias of the output-aware rep+head unit (`cap/run.py:122,127-136`) | HEAD 367–369 |
| 6.3 | Features only | the features | alias of the output-aware rep units (`cap/run.py:89-93,120`) | HEAD 367 |
| 6.4 | Features plus historical clean logits (bypass) | the features plus the 2 untreated logits | alias of the output-aware rep+clean unit (`cap/run.py:123`) | HEAD 368–369 |
| 6.5 | Linear attacker on the margin (post hoc) | the 1-column margin, read by the slate's L family | post hoc (`CAP/USEFUL_HEAD_COMPARISON.md:104-106`) | HEAD 358–360 |

---

## 7. (i) Manuscript sentences that misdescribe or over-describe access

Each item quotes the HEAD sentence, gives the WC line where the issue persists, and proposes a replacement. The most
important come first.

1. **HEAD 137–139** (persists at WC 175–178; the TBD is at WC 194–195):
   > "…only in that limited sense are they defense-aware. A recipient may know a defense's published algorithm and
   > objective, but our attackers use no such knowledge beyond this refitting, and none uses fitted parameters or
   > gradients."

   **Issue.** True for every attacker whose number is reported. Not true for the benchmark as run: LRT-A2 knows σ, and
   LRT-A4 knows σ and holds clean auxiliary representations. Both outperformed the slate at σ\* (§3.6).

   **Replacement:** "…The attackers whose recovery we report use no knowledge of the defense beyond this refitting, and
   none uses fitted parameters or gradients. The matched benchmark also ran two exploratory Gaussian likelihood-ratio
   attackers on the noise arm only. Both know the noise scale σ; one of them is also fitted on clean representations of
   the auxiliary records, which no contract releases. At σ\* they recovered 0.546 and 0.563 (Adult) and 0.526 and 0.575
   (HMDA) from the noisy features, against 0.528 and 0.513 for our slate. These are exploratory values with no paired
   test."

2. **HEAD 127–129** (persists at WC 169–173):
   > "…a second labelled set, attacker_val, on which the attacker is chosen and oriented. Both come from the same
   > population as the scored records, and these two roles (Section 4) fix the attacker's data budget."

   **Issue.**
   - No orientation step exists. attacker_val is used only for log-loss selection: of the family and
     hyper-parameters, the ignore-one candidate, the bank format and the CC α.
   - The pair-AUC orientation is fixed in advance (`ODX/PROTOCOL.md:111-112`). The orientation-free pair AUC is off by
     default and never enabled (`stored_model_eval/metrics.py:356-375`).
   - Section 4 states no sizes.

   **Replacement:** "…attacker_val, on which the attacker's family and hyper-parameters (and, where applicable, the
   ignore-one candidate, release format or smoothing level) are chosen by log loss…, and these roles fix the
   attacker's data budget. In rows, attacker_fit / attacker_val are 7,571 / 2,239 (Adult) and 6,794 / 2,089 (HMDA) in
   the matched benchmark, and 6,065 / 2,235 and 5,388 / 2,082 in the later studies. Tree and MLP attackers stop early on
   an internal 10 % of attacker_fit."

3. **HEAD 157–159 and HEAD 214–218** (persist at WC 138–142 and WC 287–292):
   > "F: official FARE, whose nominee is chosen from a frozen grid on validation only among configurations within one
   > point of untreated validation accuracy"

   > "attacker_fit and attacker_val (attacker fitting and selection) … No sensitive assessment label selects a release,
   > grouping, penalty, attacker or utility probe."

   **Issue.**
   - The FARE nominee is the admissible configuration with the **lowest attacker_val AUC for our nonlinear
     attacker**.
   - σ\* is the smallest σ whose attacker_val AUC for our slate is at most 0.55.
   - The U2 probe that gates FARE admissibility is fitted on attacker_fit task labels.
   - So the owner's defense choices use the attackers' roles and the attacker slate itself. The assessment statement is
     correct.

   **Replacement:** add "The owner also uses these roles. Their task labels fit and select the common probe, and the
   protected labels in attacker_val choose σ\* (the smallest σ whose validation AUC for our slate is at most 0.55) and
   the FARE configuration (the admissible one with the lowest validation AUC for our slate). Both defenses are
   therefore tuned against the evaluated slate on its own selection rows."

4. **HEAD 122–125** (persists at WC 147–151; also in the caption at WC 199–202):
   > "…nothing else about the record: not its inputs, its true task label or its protected attribute… Each record is
   > released once…"

   **Issue.**
   - The label-only reference, reported at HEAD 289–290, maps the scored record's **true task label** to the attribute.
     WC 201 says "no attacker receives … its task label".
   - The noise arm has three independent release realizations.

   **Replacement:** add "The label-only reference is a comparator, not a recipient. It is the one exception: it reads
   the scored record's true task label through a frequency table fitted on attacker_fit." Also add "Noise is drawn once
   per record for each of three release seeds, and the realizations are attacked separately and averaged, never pooled
   for one record."

5. **HEAD 139–140** (already changed at WC 178–179 to "Record and group identifiers"):
   > "Record and household identifiers assign roles and align releases; they are never attacker inputs."

   **Issue.** Adult and HMDA have no household identifiers (`ODX/PROTOCOL.md:130-132`).

   **Replacement (or to keep the WC wording precise):** "Canonical record keys assign roles by hash and group duplicate
   records; row identifiers align releases. Neither is an attacker input, and Adult and HMDA carry no household or
   applicant identifiers."

6. **HEAD 222–224** (persists at WC 296–298):
   > "Every surface is attacked by the same slate, with the access of Section 2…"

   **Issue.** Several statistics fall outside the slate or outside the access of Section 2:
   - the label-only reference (true task label) and the constant reference;
   - the descriptive CC;
   - the benchmark's exploratory LRTs (σ-aware, and in A4's case with clean auxiliary representations);
   - the encoders' native check, which is in-sample on the scored records' own labels.

   **Replacement:** append "The label-only and constant references, the descriptive cell-conditional attacker, the
   benchmark's exploratory noise-aware likelihood-ratio attackers and the encoders' in-sample native check are not part
   of the slate; Table [access] states their access."

7. **HEAD 226–227** (persists at WC 300–301):
   > "averaged over attacker seeds and then encoder seeds."

   **Issue.** For noise, recovery is also averaged over three release seeds (`OAR/PROTOCOL.md:132-133`;
   `MRB/notes/BENCH_DESIGN.md:151-153`).

   **Replacement:** "…averaged over attacker seeds, release realizations (noise only) and then encoder seeds."

8. **HEAD 229–230** (persists at WC 303–304):
   > "For finite releases we additionally fit a cell-conditional attacker…"

   **Issue.** It was not fitted for every finite release. In study 3 it was missing for the F\* / FZ own-head outputs
   and for features-plus-output releases (`oar/fare_run.py:91-97`). It also equalled NL wherever reported.

   **Replacement:** "For the finite releases marked in Table [access] we additionally fit…; where both were fitted it
   matched the nonlinear attacker."

9. **HEAD 112–115** (persists at WC 118–121):
   > "A recipient receives (1) a task output only; (2) features together with the output of a head fitted only on those
   > features; or, as an adverse control, (3) features together with the historical clean task logits"

   **Issue.** The list omits **features only**. That surface is the benchmark's primary contract C_rep, and it supplies
   Table 1 rows at HEAD 266–268 and 273–274.

   **Replacement:** prepend "(0) features only (reported for interpretation)".

10. **HEAD 85–86** (persists at WC 280):
    > "roles and bootstrap differ slightly across studies"

    **Issue.** attacker_fit falls from 7,571 to 6,065 rows (Adult) and from 6,794 to 5,388 (HMDA) between studies, and
    Table 1 mixes both role sets.

    **Replacement:** "roles differ between studies (the matched benchmark's attacker_fit is 7,571 / 6,794 rows, the
    later studies' 6,065 / 5,388), as do the bootstraps". WC Table 1 now has an "Exp." column, which is enough if the
    access table gives the sizes.

11. **HEAD 45–46 and 397–398** (WC 41–42 and 496–497):
    > "When two recipients link their outputs by record, our attackers recover more from the pair than from either
    > output alone"

    **Issue.** One coalition was evaluated: Adult income and employment recipients, race, untreated frozen heads, no
    feature defense, three encoder seeds. The linkage is granted, as a row-aligned concatenation; the recipients do not
    perform it.

    **Replacement for the abstract:** "On one Adult coalition (income and employment recipients, race), granted perfect
    linkage, our attackers recover more from the pair than from either output alone, including from decisions only."

12. **HEAD 39** (WC 38–39 has narrowed it):
    > "Releasing an unchanged task output bypasses any feature defense"

    **Issue.** This was evaluated for A, B, C, D\*, F\* and FZ only.

    **Replacement:** "…bypasses every feature defense we evaluated".

13. **HEAD 108–109** (WC 114–115):
    > "The encoders were trained with a linear (one-hot ridge R²≤0.05) constraint"

    **Issue.** This is accurate in form, but the training constraint is a per-minibatch ridge R² with λ = 1e-4 on
    training rows (`pcrl/training/losses.py:475-518`). The stored check is a different object: an in-sample ridge on the
    test split with λ = 1e-6.

    **Replacement (optional precision):** "…trained with a per-minibatch one-hot ridge-R² constraint (τ = 0.05) …; their
    stored check is the in-sample one-hot ridge R² on the test split."

14. **HEAD 249–250** (WC 324–326):
    > "Each study's protocol, code, inputs, unit list and endpoint families were committed and pushed before its first
    > fit; an independent replay … recomputed every primary result."

    **Issue.** The encoders' check (study 1) is a historical value computed by training-era tooling
    (`scripts/eval_round4_dominant_axis.py`). The benchmark reproduced it under its lock, but its replay checked
    reproduction, not the failure count (`MRB/VALIDATION.md:21`; see §10).

    **Replacement:** "…for each stored-model study; the encoders' stored check is a historical value that the matched
    benchmark reproduced under its lock."

## 8. (ii) Where the manuscript gives an older experiment a capability that only a later evaluator implemented, and the reverse

**Later evaluator's features carried into older experiments:**

- **(a) "Chosen and oriented"** (HEAD 128; WC 170). Orientation on validation belongs to the later ACS reader bank,
  "selected and oriented on ATTACK_VALIDATION"
  (`git show c9709a2:publications/pcrl-mechanism-design-v1/results/ADVERSARY_AND_MOTIVATION.md`, line 90;
  `…/CORRECTIONS.md`, line 169). None of the five studies implements it.
- **(b) "Household identifiers"** (HEAD 139). This belongs to the ACS service ("Household and person identifiers enforce
  separation and linkage", `ADVERSARY_AND_MOTIVATION.md` at `c9709a2`, line 92). Adult and HMDA have none. The WC has
  fixed the word, but the access table should not mention households.
- **(c) The single "Attackers and recovery" paragraph** (HEAD 222–231; WC 296–306) describes, as one common protocol,
  features that each exist only in some studies:

  | Feature | Studies where it exists | Source |
  |---|---|---|
  | Ignore-one candidates | 2, 3, 4-S5 (only after amendment S1), 5 | `ODX/AMENDMENT_S1_2026-10-03.md:3-6` |
  | Pair bank | 4-S4 only | |
  | CC | 3, 4, 5. Never in study 2, which had no finite release | |
  | Format banks | 4 and 5 only | |
  | Paired multinomial bootstrap | 4 and 5 only. HEAD 241–245 already separates this | |

  Within study 2, "features plus output" means features plus historical clean **full logits** only. There were no
  own-head outputs and no other formats. A per-experiment table should not show contract (2) or the four formats for the
  benchmark.
- **(d) The utility paragraph** (HEAD 233–235; WC 308–310) reports accuracy, balanced accuracy, minority recall, log
  loss and gain. Those measures come only from studies 4 (frozen heads) and 5 (useful heads).
  - Study 3 reported view-3 head accuracy and log loss (`oar/study.py:359-364`).
  - Study 2 reported U2 and U1 only (`MRB/EFFECTIVE_PROTOCOL.json:603-635`).
- **(e) Pre-fit lock and independent replay for every study** (HEAD 249–250): see §7 item 14. This applies to study 1.

**The reverse: an older experiment had more attacker capability than Section 2 states.**

- **(f) The benchmark's LRT-A2 and LRT-A4** (noise law; clean auxiliary representations for A4) exceed the access
  stated at HEAD 121–145 (§3.6). The WC TBD at line 194–195 is the place to say so.

## 9. (iii) Compact summary table, one row per study and surface (ready to turn into LaTeX)

Notes for the table:
- **Budgets.** "Fit / sel." is attacker_fit / attacker_val in rows. B87 = 7,571 / 2,239 (Adult) and 6,794 / 2,089
  (HMDA). O = 6,065 / 2,235 (Adult) and 5,388 / 2,082 (HMDA).
- **Pipeline.** "AF→AV→AS" = fit on attacker_fit, select by attacker_val log loss, score on assessment.
- **Seeds.** Wherever the slate is used there are three attacker seeds and three encoder seeds, plus three release seeds
  for noise.
- **Common to every row.** One release per record. No chosen inputs. No raw inputs, model parameters or gradients. No
  identifiers as attacker inputs.

| Study (WC Exp.) | Surface | Recipient receives | Fit / sel. | Fitting and selection | Knowledge of the defense | Linkage | Manuscript (HEAD) |
|---|---|---|---|---|---|---|---|
| 1. Native check (Exp. 1) | Representation | 64-dimensional `h_p` | none; in-sample on all test rows (15,060; 13,661) with their own labels | closed-form ridge, λ = 1e-6; fit = score; no selection or seeds | n/a (own check) | none | 37–38, 237, 265, 281–283 |
| 2. Benchmark (Exp. 1) | Features only, A/B/C/D | 64 columns (D: one Gaussian draw per release seed) | B87 | slate, AF→AV→AS | refit only; σ\* chosen on AV by this slate | none | 266–268, 283–285 |
| 2. | Features + clean logits | 64 + K_t clean logits | B87 | slate plus ignore-one, AV | refit only | same recipient; row concatenation | 268 |
| 2. | Output only | clean full logits (K_t) | B87 | slate | none | none | 289–290 |
| 2. | Label-only reference | **true task label** of the scored record | AF task + protected labels | frequency table; no selection | n/a | n/a | 289–290 |
| 2. | LRT-A2 / A4 (exploratory; not reported) | noisy 64 columns (D only) | AF only (A4 uses clean AF representations) | closed form; no selection | **σ (A2 and A4); clean auxiliary representations (A4)** | none | — (§3.6) |
| 2. | Held-out linear (G1, ρ₁²) | 64 columns | AF (G2 selects on AV) | closed form | none | none | 283–285 |
| 3. Output-aware (Exp. 2) | Output only: full / prob / hard | 2 logits / 2 probabilities / one-hot of 2 | O | slate (+ CC on hard) | none | none | 269 |
| 3. | Features only: A/B/C/D\*/F\*/FZ | 64 columns, or a one-hot FARE cell | O | slate (+ CC on F\*, FZ) | refit only; FARE nominee and σ\* chosen on AV by the slate | none | 273–274 |
| 3. | Features + clean logits (bypass) | features + 2 clean logits | O | slate plus ignore-one | refit only | row concatenation | 275, 290–291 |
| 3. | Features + own-head output | features + 2 log-probabilities | O | slate plus ignore-one | refit only | row concatenation | reused at 310–312, 367–369 |
| 3. | Own-head output only | 2 log-probabilities | O | slate | refit only | none | reused at 310–312 |
| 3. | Noise + output format | noisy 64 + full / prob / hard | O | slate plus ignore-one | refit only | row concatenation | — |
| 3. | FARE certificate (native) | — | cert: 1,500; 1,385 | owner statistic | — | — | 286–287 |
| 4. Diagnosis (Exp. 3) | Output only, frozen head: 6 formats; full vs offset-free bank | full / dc / centred / prob / offset / hard | O | slate; bank chosen per encoder seed on AV | none (untreated) | none | 270–272, 293–302 |
| 4. | Output only, refitted head | log-probability formats | O | as above | none | none | 302–304 |
| 4. | Pair S4 (Adult income + employment, race) | concatenation of both outputs (8 / 7 / 8 columns) | O, Adult | pair slate; bank of pair + both singles | none | **perfect; asserted row alignment** | 45–46, 397–406 |
| 4. | S5 useful task (employment / age_group) | features / own-head output / both | O, Adult | slate plus ignore-one (+ CC on cells) | refit only; FARE nominee on AV by the slate | row concatenation | 379–385 |
| 5. Useful heads (Exp. 4) | Output only, own head: 5 formats + banks | log-probabilities / margin / prob / hard | O, Adult (inherited) | slate; banks (+ CC on F/F0 and hard) | refit only | none | 314–315, 327–347, 355–365 |
| 5. | Features + own output: 4 formats | features + output | O, Adult | slate plus ignore-one; banks | refit only | row concatenation | 367–369 |
| 5. | Features only | features | O, Adult | slate | refit only | none | 367 |
| 5. | Features + clean logits (bypass) | features + 2 clean logits | O, Adult | slate plus ignore-one | refit only | row concatenation | 368–369 |
| 5. | Linear attacker on the margin (post hoc) | 1-column margin | O, Adult | slate's L family | refit only | none | 358–360 |

## 10. A count discrepancy outside access: 11 of 42, not 12 of 42

- **The table.** In `MRB/NATIVE_AND_HELDOUT_CHECKS.csv`, the 42 untreated rows (scope "historical in-sample one-hot
  ridge R2 on the test split (tau 0.05)") carry these per-row decisions:
  - **11** read "above tau (historical check fails)", at lines 74, 122, 188, 287, 332, 371, 377, 380, 404, 407 and 416;
  - 31 read "historical check passes".
- **The decision file.** `MRB/RESEARCH_DECISION.md:55` says "**12 of 42 fail**", but its own list (lines 56–61) names
  3 + 1 + 2 + 2 + 1 + 2 = **11** combinations:
  - Adult income/race on s0, s1 and s2;
  - Adult education/sex on s1;
  - HMDA fair_lending/sex on s0 and s2;
  - HMDA pricing/race on s1 and s2;
  - HMDA underwriting/ethnicity on s1;
  - HMDA underwriting/race on s1 and s2.
- **Other variants agree.** The float64 variant gives the same 11. The largest passing value is 0.0438 (HMDA s0
  underwriting/race), and the smallest failing value is 0.0546.
- **Why it was not caught.** The independent verification checked reproduction (42 of 42), not the failure count
  (`MRB/VALIDATION.md:21`). The CAP manuscript review compared the manuscript with the decision file's text
  (`CAP/MANUSCRIPT_REVIEW.md:319,337,352`).
- **Where "12" appears.**
  - Manuscript: HEAD 37, 265, 282; WC 87, 343, 362, 511.
  - Repository: `MRB/ADVISOR_BRIEF.md:44`; `MRB/HANDOFF.json:29`; `OAR/RESEARCH_DECISION.md:48`;
    `CAP/CLAIM_LEDGER.csv:2`.
- **Suggestion.** Use "11 of 42" after the owner confirms the count from the committed table. The statement "including
  HMDA underwriting/race on two of three seeds" stays correct.
