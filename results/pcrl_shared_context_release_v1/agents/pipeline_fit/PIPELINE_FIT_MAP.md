# PIPELINE_FIT_MAP — fitting-side implementation map for the shared-context release

Role: pipeline cartographer (fitting side). Read-only survey of the predecessor code at
worktree HEAD `2ce5d171f` (sparse checkout). No fitting run, no cloud resource touched,
no outer/2018 assessment label opened. `prepared.joblib` was inspected **structurally only**
(keys/shapes/dtypes) on the *sanitized* copies; no label value, class count or person row
was printed.

Abbreviations: `AR` = `experiments/pcrl_adaptive_release_v1/`, `TAC` =
`experiments/pcrl_task_aligned_cuts_v1/`, `TDR` = `experiments/pcrl_task_directed_release_v1/`,
`RES` = `results/pcrl_adaptive_release_v1/`. `file:N` / `file:N-M` are line numbers.

---

## 1. Input loading

### 1.1 Pins and loaders
- Index: `results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json` (`TAC/data.py:19`).
  `data.index()` (`TAC/data.py:35-44`) hard-checks `source_commit=f4bdf4cd…` and
  `completed_prospective_evidence_commit=5e154e5c…`. Per anchor it pins `encoder`,
  `prepared`, `historical_fineC_tables`, and maps `Q`,`D17`,`D33` × {`Q.npz`,`solution.joblib`,`tables.npz`}
  (`TAC/data.py:198-207`), each with local `path`, `sha256`, and S3 archive member
  (bucket `pcrl-ux-archive-ed9d21fd`, key `pcrl_final_prospective_v1/archive/part-0001.tar.zst`).
  Local paths point into `…/.worktrees/pcrl-final-prospective-v1/results/pcrl_final_prospective_v1/private/restore/…`.
- Maps (`TAC/data.py:23-24`): `Q = T0_L_0.01_a17` (historical stochastic 32×17),
  `D17 = T0_U_unconstrained_a17` (deterministic 32×17), `D33` (32×33). `load_map`
  (`TAC/data.py:316-330`) validates shape (32,17|33) and simplex.
- `load_prepared` (`TAC/data.py:258-295`): prefers the **sanitized** label-stripped copy
  found at `<worktree root>/results/pcrl_task_aligned_cuts_v1/private/sanitized_2018_v2/`
  (`TAC/data.py:20,52-82`), else falls back to the raw pinned pickle and deletes
  `attacker_validation.labels` in memory (`:292-294`). Validates pool order, T0 range [0,32),
  `ha`(n,4), `hb`(n,2), and that `prepared['roles']` partitions `representation_fit`.
- **Gotcha for the new worktree:** `_sanitized_root()` is relative to the *worktree root*.
  This worktree has **no** `results/pcrl_task_aligned_cuts_v1/private/`. The adaptive worktree
  used a symlink `…/pcrl-adaptive-release-v1/results/pcrl_task_aligned_cuts_v1/private/sanitized_2018_v2 ->
  …/pcrl-task-aligned-cuts-v1/results/pcrl_task_aligned_cuts_v1/private/sanitized_2018_v2`
  (receipt `SANITIZATION.json`, schema `pcrl-sanitized-prepared-v1`, anchor_0 sha `d69e763e…`).
  `fit_a.run_center` (`AR/fit_a.py:708-711`), `evaluate.audit_panel` (`AR/evaluate.py:212-214`)
  and `task_baselines.run_task_fit` (`AR/task_baselines.py:753-756`) **refuse** to run without it.
  Recreate the same symlink (or re-run `data.py sanitize`) before any fit.

### 1.2 Structure of `prepared` (sanitized anchor_0; anchors 1/2 identical schema)
```
prepared: dict[6]
  ctx: {anchor:int, input_contract:str, portability:{A0|J:{pool:float}},
        pools: {representation_fit, downstream_fit, downstream_validation,
                attacker_fit, attacker_validation}}
    each pool: x (n,32) float32 | ha (n,4) f64 | hb (n,2) f64 | J (n,16) f64 | A0 (n,16) f64
               ids (n,) object[str] | households (n,) object[str] | raw_rows (n,) int64
               weights (n,) f64 (PWGTP)
               labels: {SEX, RAC1P, public_coverage, same_residence, civilian_at_work,
                        income_binary, commute_over20}  each (n,) int64
               (attacker_validation: labels key ABSENT in sanitized copy)
  encoder: TDR.encoding.Encoder  (attrs baseline, code[Codebook], dictionaries, metadata,
           partitions, risk, task)
  encoded: {pool: {p (n,) f64 task posterior, b (n,) f64 H-only baseline, r (n,) f64 residual
                   logit(p)-logit(b), risk (n,11) f64 [2 SEX + 9 RAC1P historical risk],
                   codes:{T0 (n,) int64 [0,32), Ttask (n,) int64, Trisk (n,) int64},
                   actions:{'17' (n,17), '33' (n,33)} f64, global_offsets (n,49) f64}}
  tables: {T0: cost_U/cost_W/cost (32,17), state_mass (32,) int64, roles{A|AB/SEX|RAC1P/U|W}
           (classes, |view H bins|, 32), support{count,sumw,sumw2,ess}, population{...},
           aggregation{...}; Ttask, Trisk: same with 64 states}   <- historical, unused by AR
  roles: {teacher_fit (≈5.1k,), teacher_internal_validation (≈1.3k,), mechanism (≈4.1k,)}
         int64 indices into representation_fit (historical encoder roles)
  erasers: {leace_supervised, splince_supervised}: TDR.baselines.AffineEraser
```
Pool sizes (rows): a0 10513/4539/2984/2993/2985; a1 10428/4521/3041/3021/2964;
a2 10551/4464/3024/2941/3062 (rep_fit/ds_fit/ds_val/att_fit/att_val). IDs and
household IDs are `str`. Missing labels use a negative code (loaders accept `labels >= -1`,
`TAC/audit.py:289`; `AR/nuisance.py:105`).

### 1.3 Where each array comes from
| Quantity | Source | Loader |
|---|---|---|
| X_A (32) | `pools[p]['x']` float32 (PCA32) | `AR/roles.py:50,64` → `rows['x']` |
| H_A (4) / H_B (2) | `pools[p]['ha']` / `['hb']` f64 | `AR/roles.py:50` |
| household / person IDs | `pools[p]['households']` / `['ids']` | `AR/roles.py:50` |
| PWGTP | `pools[p]['weights']` | `AR/roles.py:50` |
| labels | `pools[p]['labels'][same_residence|SEX|RAC1P]` | `AR/roles.py:73-74` |
| T32 code | `encoded[p]['codes']['T0']` **stored** | `AR/roles.py:67` → `rows['token_codes']` |
| teacher p, residual r, risk(11) | `encoded[p]['p'|'r'|'risk']` stored | `AR/roles.py:68-70` → `teacher_p, residual, risk` |

- **T32 is stored, not recomputed.** Every fitting/audit path reads
  `encoded[pool]['codes']['T0']` (`TAC/data.py:5,271`; `AR/roles.py:67`). Runtime recomputation
  exists only for deployment/parity: `Encoder.encode()` (`TDR/encoding.py:198-208`:
  torch/sklearn teacher `p`, baseline `b`, `r=logit(p)-logit(b)`, `risk`, then
  `code.assign(r, x_a, risk)`); `release.OneReleaseSession` (`TAC/release.py:103-109,153`);
  `fit_b.runtime_parity_receipt` (`AR/fit_b.py:71-114`) compares archived vs live T0 and only
  sets `linux_x86_parity_verified` on Linux x86 (Mac float32 re-encoding is not a substitute —
  `RES/QUICKSTART.md` "Do not use Mac float32 re-encodings"). Any policy `d_m` that uses
  `teacher_p`/`residual`/`T0` must use the stored values on 2018 data and needs a Linux parity
  receipt for a deployment claim. `x`/`ha`-only policies are parity-insensitive except float32.
- Legacy dependence: historical teacher/risk/codebook were fitted on `representation_fit`
  `teacher_fit` rows (`TDR/encoding.py:211-230`); 1,289 of those households fall in the
  adaptive `outer_assessment` role (`RES/DATA_ROLE_COUNTS.json`).

### 1.4 Household roles (`AR/roles.py`)
- `role_of(h)` (`:19-26`): `u = int(SHA256("pcrl_adaptive_release_v1|"+household)[:8]) / 2^64`;
  bounds `(.20,.40,.65,.75,.85,1.0)` → `nuisance_train, audit_fit, coefficient_split,
  inner_selection, inner_check, outer_assessment` (`:10-15`). Depends only on household ID, so
  roles are **global across anchors** (overlapping households across anchors get one role).
- `pooled_role(prepared, role)` (`:33-89`) concatenates the role's rows from **four** pools
  `representation_fit, downstream_fit, downstream_validation, attacker_fit` (`:55-56`);
  `attacker_validation` (historical outer pool) is never pooled pre-lock; `outer_assessment`
  raises `PermissionError` (`:48-49`). Returned keys: `x, ha, hb, weights, ids, households,
  token_codes, teacher_p, residual, risk, labels{same_residence,SEX,RAC1P}`. Asserts unique
  IDs, ha 4 / hb 2 / x 32 / risk 11 columns, T0 in [0,32).
- Applied per anchor by `{name: roles.pooled_role(prepared, name) for name in SCIENCE_ROLES}`
  (`AR/fit_a.py:25-26,716`). Realised counts (people/households), `RES/DATA_ROLE_COUNTS.json`:

| role | a0 | a1 | a2 | global HH |
|---|---|---|---|---|
| nuisance_train | 4138/2795 | 4113/2777 | 4124/2809 | 3884 |
| audit_fit | 4158/2765 | 4176/2807 | 4144/2776 | 3894 |
| coefficient_split | 5257/3506 | 5317/3556 | 5334/3569 | 4907 |
| inner_selection | 2215/1468 | 2187/1453 | 2144/1420 | 2003 |
| inner_check | 2131/1454 | 2142/1442 | 2119/1447 | 2018 |
| outer_assessment* | 3555/2409 | 3535/2384 | 3549/2371 | 2968 |
\*outer counts include the attacker_validation pool (census only, `AR/roles.py:106-108`).

---

## 2. Decoder family (task decoder `d(y | H_A, z)`)

- Entry: `fit_a.fit_frozen_decoder(role_dict, current_q, q_ref, output_dir, seed)`
  (`AR/fit_a.py:193-255`) → `alternate.fit_decoder_round` (`AR/alternate.py:31-52`) →
  `TDR/audits.fit_slate(..., n_classes=2, slate="standard")` (`TDR/audits.py:502-599`).
- Rows: **fit on `nuisance_train`**, candidate selection on **`inner_selection`**
  (`AR/fit_a.py:201-209`), task-valid rows only (`_valid_mask`, `:82-87`).
- Training law (expected-token law): `(current_Q[t] + D17[t] + 1/17) / 3` per person
  (`AR/alternate.py:17-28`; recorded as `training_law`, `AR/fit_a.py:217`). The same law is used
  on the selection rows.
- Weighted token expansion: `expand_person_tokens` (`TDR/audits.py:155-164`) emits one row per
  (person, token with p>0) with weight `w_person * p(z)`; conservation asserted. Person weight
  `0.5 + 0.5*PWGTP/mean(PWGTP)` (`balanced_person_weights`, `:148-152`) — **one predictor
  serves both U and PWGTP**.
- Features: `H` standardized by fitting-row mean/std (`:531-532`); **token enters as an
  unscaled one-hot** (`_encoded`, `:209-217`); logistic additionally gets `H ⊗ onehot`
  interactions (`interactions = family=='logistic'`, `:354`). Width: logistic 4+17+68=89,
  HGB/MLP 21.
- Standard slate = 6 candidates (`:540-579`, config `:42-49`): `logistic` (C=1, lbfgs, 2000 it),
  `hist_gb_20`, `hist_gb_5` (150 iters, 15 leaves, lr .1, l2 1; `min_samples_leaf =
  leaf × max tokens/person`, `:365-372`), `sampled_hist_gb_20/5` (one fixed sampled token per
  person, seed `20260921+seed`; aliased to exact trees when law is deterministic, `:560-572`),
  `mlp_120` (64-32 ReLU, Adam 1e-3, wd 1e-4, batch 256 persons, loss divided by original
  persons `:201-206`, checkpoint chosen on validation every 5 epochs, `:405-499`).
  `catchup` slate adds a 360-epoch MLP (used by audits, not the bank).
- Predictions floored: `(1 - K·1e-9)·p + 1e-9` (`:256-258`); losses use `-log max(p,1e-9)`
  (`TAC/method.py:11,54-67`).
- Selection rule within slate: min `0.5·U + 0.5·PWGTP` exact expected-token log loss, lexical tie
  (`TDR/audits.py:41,586-589`).
- Seeds: round seed `20260924 + 10000·anchor + 100·round` (`AR/fit_a.py:608`).
  Threads forced to 1 (`torch.set_num_threads(1)`, `threadpool_limits(1)`, `:536-539`).
- Across rounds, all previous decoders are retained and re-selected on `inner_selection`
  under the **current** channel (`alternate.select_frozen_decoder`, `AR/alternate.py:55-76`,
  called at `AR/fit_a.py:623-627`).
- Artifacts: `decoder_rNN/FIT_DECODER.json` + `slate/<cid>/{candidate.joblib, model.pt,
  metadata.json}`, hash-pinned via `model_directory_hash` (`TAC/audit.py:66-76`); reload with
  `TDR/audits.load_candidate` (`:298-310`). API: `decoder.predict_token_proba(ha, 17)` →
  (n,17,2).

## 3. Attack bank

- Entry: `fit_a.build_attack_bank(role_dict, source_channels, out, seed, target_roles,
  n_states=32)` (`AR/fit_a.py:304-426`) → `TAC/audit.fit_role_slate` (`TAC/audit.py:319-373`)
  → `TDR/audits.fit_slate(slate="standard")` — **same 6-family slate as the decoder**, with
  `n_classes` 2 (SEX) or 9 (RAC1P).
- Rows: fit on **`audit_fit`**, slate validation on **`inner_selection`**, coefficients on
  **`coefficient_split`** (`AR/fit_a.py:331-335`); households asserted disjoint.
- Views/features (`TAC/audit.py:36-63`, `role_arrays :282-316`): view A → H_A(4); AB →
  [H_A,H_B](6); B → H_B(2) H-only. Wire `H` (no token; token_probs = ones(n,1)) or `release`
  (+17-token one-hot).
- Slates per round-0 target (`AR/fit_a.py:337-361`): view A × sources {H,D17,Q,coverage},
  AB × same 4 sources, B × {H} → 9 slates/target, 18 slates → 108 fitted models.
  Cut roles (`:363-405`): `A/S` uses source view A only; `AB/S` uses source views A, AB, B
  (legal coalition ancestors). **Every candidate in every slate becomes a cut**, not only the
  slate winner (`:372`). Round 0 total: 156 attack specs → 312 cuts (U and W per spec),
  confirmed by `RES/EVENT_LOG.jsonl` ("156 attacks, 312 cuts"). Later rounds use the single
  source `candidate_r{r-1}` = previous Q (`:614-615`): +4 slates, +36 specs, +72 cuts per round
  (≈744 cuts after 6 rounds).
- Source law for a fitted attack: `q[token_codes]` per person (`:349-353`) — a T32 row lookup.
- Seeds: `(round_seed + 50000) + 1000·counter` (`:358,617-618`).
- Class floors/support: full class schema kept even if a class is absent in `audit_fit`;
  absent classes get the 1e-9 floor, recorded in `fit_missing_classes` (`TAC/audit.py:357`).
  Label-missing rows dropped per target (`labels >= 0`, `:306`).
- Unweighted vs PWGTP: one fitted predictor (balanced training weight, §2); two cuts per model:
  U coefficient normalized `1/n_valid`, W normalized `PWGTP/sum(PWGTP)` over that target's
  valid coefficient rows (`AR/fit_a.py:90-106,394-404`). `_population_hash` pins the same
  eligible people for U and W (`:156-171`).
- Per-person replay: `person_loss_from_attack(spec, rows)` (`:266-301`) returns
  `(valid_mask, losses (n_valid,17))`; H-only losses repeated across tokens and asserted
  token-invariant. `fit_b._attack_loss_rows` (`AR/fit_b.py:611-636`) replays a whole frozen
  bank on any rows with caching — directly reusable for new coefficient blocks.
- Persistence: `bank_rNN/ATTACK_BANK.json`, `coefficients.npz` (32×17 per cut),
  `slates/<view>_<target>/<source>/models/<cid>/`; `load_frozen_bank` (`:429-466`).

## 4. Coefficients, reference calibration, LP, privacy-first

### 4.1 Coefficients
- Task: `task_person_losses` (`AR/fit_a.py:121-135`) → per-person CE for each of 17 tokens
  (from `decoder.predict_token_proba(ha,17)`), then `aggregate_loss_coefficients(codes, losses,
  weights, n_states)` (`:90-106`): `U[t,z] = Σ_{i:T=t} ℓ_i(z)/n`, `W[t,z] = Σ ℓ_i(z)·w_i/Σw`.
  **State mass is inside the coefficient; objective is `Σ coeff ⊙ Q`** (never multiply mass
  again, `TAC/method.py:1-5`). Task objective `cost = 0.5·U + 0.5·W` (`AR/alternate.py:79-91`).
- Attack cut coefficients: same aggregation of `person_loss_from_attack` losses.

### 4.2 Calibrated reference (`AR/reference.py`)
- `calibrate_reference(q_ref, bank, delta)` (`:47-126`): group cuts by `role|weighting`
  (8 groups: {A,AB}×{SEX,RAC1P}×{U,W}); require identical `coefficient_pool_sha256`,
  `class_order`, `weight_normalization` within a role (`:84-99`); compute
  `L_a(D17)=Σ coeff_a ⊙ D17`; `rho_g = min_a L_a(D17)` (`:104-110`); every cut floor
  `= rho_g − delta` (`:107,115-118`). Replays D17 witness and asserts zero violation (`:120-122`).
  Called on the **entire retained bank every round** (rebasing; `AR/alternate.py:87`).
- `solve_calibrated(cost, q_ref, bank, delta)` (`:129-148`): calibrate → `solver.solve_p1`;
  phase-I infeasibility despite witness is a hard error (`:141-145`).
- Allowed delta values hard-coded `{0, .001, .003}` (`AR/fit_a.py:560-561`); focused run .001.

### 4.3 LP solver (`TAC/solver.py`)
- Library: **SciPy `linprog(method="highs")`**, tolerances primal/dual 1e-9 (`:22-23`);
  replay tolerances SIMPLEX 1e-7, NONNEG 1e-8, PRIMAL 1e-7 (`:18-21`).
- `solve_p1(cost, cuts)` (`:149-207`): constant-map check (`:86-98`) → phase I
  (`phase_one_p1 :121-146`: minimize common slack `s ≥ 0` with `<A_a,Q> + s ≥ floor_a`) →
  main LP: `min Σ cost⊙Q` s.t. `-<A_a,Q> ≤ -floor_a`, `Σ_z Q[t,z]=1` (`_simplex_matrix`
  = kron(eye(32), ones(1,17)), `:101-104`), `Q ≥ 0`. Returns replayed Q (`_clean`
  renormalizes, `:107-118`), reconstructed dual lower bound and `fixed_bank_gap` (`:184-207`).
  Size: 544 variables, ≤~750 inequality rows — solves in well under a second.
- `replay_p1` (`:73-83`) and `bank_sha256` (`:60-70`) require `coeff.shape == cost.shape`.

### 4.4 Privacy-first (`AR/privacy_first.py`, `AR/privacy_first_fit.py`)
- Inputs: final frozen A-center bank (U and W task costs + calibrated cuts) and the D17 control
  file from the completed controls unit (`load_frozen_inputs`, `privacy_first.py:217-251`); no
  refit of anything.
- LP (`_matrices :104-130`, `solve_privacy_first :172-214`): variables `Q` (32×17, bounds [0,1])
  and `τ ≥ 0`; **maximize τ** subject to
  AB/SEX cuts `L_a(Q) ≥ rho_a + τ`; all other cuts `L_a(Q) ≥ rho_a − 0.001`;
  task caps `C_U(Q) ≤ C_U(D17)+0.001` and `C_W(Q) ≤ C_W(D17)+0.001` (caps relative to **D17**,
  separately U and W); row simplex. Witness `(D17, τ=0)` replayed first (`:178-180`); phase-I
  over all clauses (`:133-152`); HiGHS dual reconstructed as a τ upper bound (`:155-169`).
  Time limit 60 s. Realised τ: a0 .00893, a1 0, a2 .00025 nats (`RES/EVENT_LOG.jsonl`).
- Unit runner `run_fit_unit` (`privacy_first_fit.py:101-170`): A only, delta .001;
  writes `INPUTS.json`, `FIT.json`, `channel/Q.npz`, `RELEASE_SPEC.json` (router T0), `COMPLETE.json`.

## 5. Alternation loop (`AR/fit_a.py:546-703`)

Per round `r = 0..max_rounds` (`MAX_ROUNDS=6`, `:27`; registered run used 6):
1. fit new decoder slate under law `(current_Q + D17 + U17)/3` (`:609-613`);
2. fit fresh attack slates on sources — round 0: `{H,D17,Q,coverage}`; round r>0:
   `{candidate_r{r-1}: current_Q}` (`:614-618`); append to retained bank (`:619-620`);
3. choose among **all retained decoders** by inner_selection loss under `current_Q` (`:623-627`);
4. task costs on `coefficient_split` (`:628`); calibrate whole bank and solve LP (`:629-634`);
5. save round artifacts; record fixed-decoder task score on inner_selection and inner_check
   (inner_check recorded only) (`:660-682`); `current_Q = Q` (`:687`).
- No convergence-based stop; fixed round count. `current_Q` starts at historical Q (`:597`).
- Final choice `select_final_bank_feasible` (`:510-543`): among round channels feasible on
  the **last rebased union bank**, minimum inner_selection balanced task CE (the round's own
  fixed decoder), earliest round tie. Realised: a0 round 1, a1/a2 round 6.
- Unit layout (`root` must contain a `private` path part, `:572-574`):
  `INPUTS.json`; `decoder_rNN/{FIT_DECODER.json, slate/…}`; `bank_rNN/{ATTACK_BANK.json,
  coefficients.npz, slates/…}`; `round_rNN/{COSTS.npz (cost_U,cost_W,cost), CALIBRATED_COEFFICIENTS.npz,
  CALIBRATED_BANK.json (rho, cuts), channel/{Q.npz,…} (release.ChannelArtifact), ROUND.json}`;
  `FINAL_BANK_SELECTION.json`; `SELECTED.json`; `COMPLETE.json` (full artifact SHA inventory,
  `_inventory :487-490`). Resume = `_load_complete` inventory check (`:493-507`); every JSON/NPZ is
  write-once-or-verify (`:56-79`).
- Consumers of a finished center: `fit_b.load_a_selected` (`AR/fit_b.py:215-304`) returns
  selected Q, final-round cost pair, final calibrated cuts, all attack specs, final decoder.
- Archiving (`AR/archive_unit.py`): verifies `COMPLETE.json` inventory (`:28-46`), tars the unit,
  `put-object` AES256 to `s3://pcrl-ux-archive-ed9d21fd/pcrl_adaptive_release_v1/units/<unit>.tar.gz`
  (versioned), reads back, SHA-checks, extracts, re-inventories, writes
  `private/archive_manifests/<unit>.json` (`:55-118`). Never deletes the original.
- Unit names used (SSM log): `a{k}_A_checkpoint_r0_d001`, `a{k}_A_center_d001`,
  `a{k}_A_controls_d001`, `a{k}_A_CERTIFICATE.json`, `a{k}_TaskOnly`, `a{k}_PartitionControls`,
  `a{k}_B_center_d001`, `a{k}_privacy_first_d001`, `a{k}_inner_panel_d001`, `a0_ABRACE_POSITIVE`.
  Checkpoint dirs 2.1k files per 6-round center (a0 archive 2,119 files; round-0 unit 811 files, 44 MB).

## 6. Controls

| Module / entry | Control | Procedure | Inputs / role |
|---|---|---|---|
| `AR/fit_controls.py:181-331` `fit_controls_from_frozen` (CLI `:334-377`) | 27 fixed-bank 32×17 controls: `D17`, `historical_Q`, `D_task` = row-wise argmin of the final frozen cost (`TAC/controls.py:77-83`); constant-replacement (token 0) and 17-token randomized response of D17 and D_task at publish {.5,.75,.9,1} (`:20,277-288`); `MILP`; `deterministic_search`; 6 `GRADIENT_*` | No data refit: consumes the selected center's **final** cost pair + calibrated cuts via `load_final_problem` (`:74-178`). All controls replayed on the same bank. | coefficient_split-derived coefficients only |
| MILP `TAC/controls.py:114-168` | same-partition deterministic P1 | `scipy.optimize.milp` (HiGHS), binary Q (32×17), row one-hot, same cuts; 120 s, rel gap 1e-6; incumbent re-rounded and replayed; valid dual bound reported | same bank |
| heuristic `TAC/controls.py:171-220` | bounded coordinate search | starts: rowwise min, 17 constants, D17, D_task; random restarts; 60 s, seed 20260924 | same bank |
| gradient `TAC/controls.py:223-322` | "adversarial categorical" comparator (fixed bank) | softmax-logit Adam (lr .05), augmented Lagrangian (dual lr 1.0, penalties 10/100/1000), 1000 steps, init D17 / D_task smoothed .01; best by (violation, objective) | same bank; **not** a learned x→z policy, no attacker refresh (`:322` note) |
| `AR/certify_controls.py:23-182` | LP vs MILP certificate | re-solves final-bank LP, checks MILP bound/incumbent; status `NO_CERTIFIED_FIXED_BANK_GAP` / `NUMERICAL_FIXED_BANK_INTEGRALITY_GAP` | frozen center + controls unit |
| `AR/task_baselines.py:102-150` `fit_task_only` (CLI `fit-task`) | **richer deterministic task-only token policy** from X_A,H_A | standardized logistic C=1 on 36 features `[X_A,H_A]` (`TDR/data.py:24-36`), PWGTP fit weights rescaled to mean 1; 16 PWGTP-weighted quantile cut points of fitted p on the same rows (`:60-69,133`) → token = `searchsorted(cuts, p)` ∈ 0..16 (`:91-95`). Varies **within** T32 cells. Law variants (`:224-249`): unmodified, constant-replacement, RR. Keyed HMAC persistent release session (`:289-426`). | **nuisance_train** only (`:116-117`); seed 20260924 |
| `AR/task_baselines.py:472-698` partition controls | T32→64 nested partitions: task_only / joint_risk / random_eligible | support floor 100 unique & effective households; check-gain ratio 0.25 (`AR/fit_b.py:26`) | coefficient_split + inner_selection; realised as T32 aliases (support-limited) |
| `AR/nuisance.py:77-138` | frozen local SEX/RAC1P risk predictors (not a control but reusable) | standardized logistic C=1 on `[X_A,H_A]`, PWGTP weights, 1e-8 class floor | **nuisance_train** |

Notes: the "strong adversarial categorical policy" row of the new panel has **no predecessor
implementation as an x-dependent policy**; the gradient control only optimizes a 32×17 table.
Task-only token IDs are PWGTP quantile ranks of p — **not** D17's token semantics.

## 7. Cloud execution

- Host: one task-owned **c7i.8xlarge** (us-east-1, $1.428/h), 120 GiB encrypted gp3,
  SSM-only (security group with 0 ingress), `InstanceInitiatedShutdownBehavior=terminate`,
  tag `Study=pcrl_adaptive_release_v1` (checked by `AR/cloud.py:28-42`). Launched
  02:54:24Z, termination requested 05:53:42Z (≤3.01 h, ≤$4.29 compute; total <$4.50)
  (`RES/RESOURCE_LEDGER.json`, `RES/COST_AND_CLOSEOUT.md`). Local CLI profile `vein`
  (`AR/cloud.py:18`). **AMI ID and instance profile are not recorded in this study's
  checkout** (`private/AWS_RESOURCES.json` in the adaptive worktree lists type/SG/volume only).
  Predecessor lineage used instance profile `pcrl-ux-ec2` (pre-existing; `…/pcrl_final_prospective_v1/private/AWS_RESOURCES.json`)
  and `ami-025d99823a4caad37` on c7i.4xlarge (`…/pcrl_task_directed_release_v1/private/run_instances.json`).
  Host Python is a pre-provisioned `/opt/pcrl/venv`; pip freeze archived at
  `s3://…/pcrl_adaptive_release_v1/provenance/HOST_ENVIRONMENT.json`.
- User data (`AR/cloud_user_data.sh`): installs a systemd **watchdog timer at a fixed absolute
  UTC** (`:30-40`, 21:33:31Z) that `aws s3 sync`s `/opt/pcrl/work/results/pcrl_adaptive_release_v1`
  to `s3://pcrl-ux-archive-ed9d21fd/pcrl_adaptive_release_v1/emergency/results` (AES256) then
  `shutdown -h now` (`:4-21`).
- Source staging (`AR/cloud.py:82-113`): `git archive` of an exact pushed HEAD → S3
  `pcrl_adaptive_release_v1/source/source-<sha>.tar.gz` → SSM `get-object --version-id`,
  sha256sum, extract to `/opt/pcrl/work`, write `SOURCE_COMMIT.txt`. Each job asserts
  `SOURCE_COMMIT.txt` before running. Commands logged to `private/SSM_COMMANDS.jsonl`.
- Input staging (`AR/stage_inputs.py:34-74`): 17 objects under
  `pcrl_adaptive_release_v1/inputs/NN-<name>` (index, SANITIZATION.json, 3 sanitized anchors
  ≈26.4 MB each, 3 × {encoder.joblib, Q/D17/D33 Q.npz}), 79.5 MB total, AES256, versioned;
  host destinations = the **absolute pinned Mac paths** from the index for encoder/maps and
  `/opt/pcrl/work/results/pcrl_task_aligned_cuts_v1/…` for index/sanitized copies. Private
  manifest `private/INPUT_STAGE.json`; restore commands `private/SSM_RESTORE_INPUTS.json`.
- Parallelism: **no worker pool**. One SSM `AWS-RunShellScript` per unit (`AR/cloud.py:45-61`),
  each a single-threaded process (`OMP/OPENBLAS/MKL_NUM_THREADS=1`; fit_slate forces torch 1
  thread). Concurrency = number of simultaneous SSM jobs: 3–4 (e.g. 03:31Z: a0 B, a0 controls,
  a1 A, a2 A). Some jobs wrapped in `flock -n /opt/pcrl/locks/<unit>.lock`. Queue/state are
  coordinator JSON (`RES/QUEUE.json` 98 registered units, `RES/RUN_STATE.json`,
  `RES/UNIT_LEDGER.json` 49 completed ids); no per-unit elapsed-time field exists there
  (only `a0_benchmark.wall_time_minutes_upper_bound = 8`).

## 8. Runtime (upper bounds from SSM start times → next check/archive; 1 thread/job)

| Unit | Content | Wall time |
|---|---|---|
| a0 A round-0 checkpoint | 1 decoder slate + 18 attack slates (108 models) + LP | ≤7.5 min (RUN_STATE: ≤8) |
| a0 A center, 6 rounds | 7 decoder slates + 42 attack slates (~252 models) + 7 LPs | ≈11–12 min (03:15:11→≤03:26:07) |
| a1, a2 A centers (concurrent, 4 jobs on host) | same | ≈11 min (03:31:56→≤03:42:50) |
| A controls (MILP 120 s cap + 60 s search + 6×1000-step gradient) | 27 channels | ≈1.6–2 min |
| task-only fit ×3 (concurrent) | logistic + quantizer | <40 s |
| partition controls ×3 | | ≈1 min |
| LP/MILP certificate ×3 | | minutes at most (archived ≤5 min after start) |
| privacy-first LP ×3 | 60 s cap | ≈1 min total |
| common inner audit panel (catchup slate, ~24 canonical releases) | fit+select+score | a0 ≈31 min; a1/a2 ≈39–40 min |
| single-release inner audit (privacy-first) | | ≈1.7 min |

LP solves are negligible; the cost is predictor slates (each slate trains 5 fits + MLP;
per-candidate `fit_runtime_seconds` is inside archived `slate.json`/`metadata.json`, not local).
Whole study: ≈3 h host time, <$4.50.

## 9. Recommendation: what to reuse, what to copy, where the new channel plugs in

### 9.1 Import unchanged (verified generic)
- Data/roles: `TAC/data.index, load_prepared, load_map, member_record, verified_member,
  sha256_file`; `AR/roles.role_of, pooled_role, summarize_roles` (keep SALT to preserve roles).
- Predictor slates: `TDR/audits.fit_slate, load_candidate, expand_person_tokens,
  expected_token_loss`; `TAC/audit.fit_role_slate, role_arrays, view_features,
  model_directory_hash, assert_household_disjoint`. **These already accept arbitrary per-person
  `token_probs (n,17)`** — the new q(z|x) law plugs in directly.
- Per-person losses: `AR/fit_a.task_person_losses` (`:121`), `person_loss_from_attack`
  (`:266`), `_valid_mask`, `_population_hash`; `AR/fit_b._attack_loss_rows` (`:611`).
- Normalization: `AR/fit_a.aggregate_loss_coefficients` for the B-block (grouping by T32).
- Cost-sensitive policy targets: `AR/fit_b.fixed_bank_dual_prices` (`:307-365`, λ from a
  fixed LP) + `priced_original_person_rows` (`:368-414`, per-person 17-action Lagrangian
  cost `task − Σλ·attack` with correct U/W normalization) — exactly the "per-action cost
  vectors from frozen decoders and attackers" needed for bank members 3–4.
- Legal context features: `AR/nuisance.fit_frozen_nuisance` (nuisance_train) +
  `AR/fit_b.stored_deployable_features` (`:472-501`) → `residual, task_posterior, h_a_0..3,
  sex_risk_0..1, race_risk_0..8` (`AR/refinement.py:17-22`); runtime twin
  `refinement.build_deployable_features` (`:368`).
- Task-only policy: `AR/task_baselines.fit_task_only / load_task_only / TaskOnlyPredictor`
  and the keyed session pattern `TaskOnlyReleaseSession` for per-person-law emission.
- Controls kernels (for any finite table problem): `TAC/controls.solve_deterministic_p1,
  deterministic_coordinate_search, gradient_control_grid, rowwise_minimizer`
  (shape-agnostic in rows; require 17 columns).
- Infrastructure: `AR/archive_unit.py`, `AR/cloud.py`, `AR/stage_inputs.py`,
  `AR/cloud_user_data.sh` — copy with the study constants changed (STUDY/PREFIX/ROOT/tag/
  watchdog time are module constants: `cloud.py:12-18`, `stage_inputs.py:15-19`,
  `archive_unit.py:18-20`, `cloud_user_data.sh:7,34`).

### 9.2 Copy and generalize (hard-coded "one row per T32 state")
| Location | Assumption | Needed change |
|---|---|---|
| `TAC/solver._simplex_matrix/_cuts/replay_p1/solve_p1/phase_one_p1` (`:33-207`) | variable = Q (S×17), each row sums to 1, `coeff.shape==cost.shape` | new LP over `x=[vec B (32×17), vec A (K×M), η]`: `Σ_z B[t,z]+η=1 ∀t`; `Σ_m A[k,m]−η=0 ∀k`; `0≤η≤1`; cuts `<G_B,B>+<G_A,A> ≥ ρ−δ`. Reuse the phase-I → main → replay → dual-reconstruction pattern and tolerances verbatim. |
| `AR/reference.calibrate_reference` (`:47-126`) | `validate_channel(q_ref)`; `coeff.shape == q_ref.shape`; witness via `replay_p1` | same grouping/provenance logic, but ρ = `<G_B^a, D17>` at witness `(B=D17, A=0, η=0)` (A-block irrelevant at witness); replay the generalized witness. |
| `AR/alternate.coverage_law / fit_decoder_round` (`:17-52`) | law = `current_Q[t]` | per-person `(q_cur(x_i) + D17[t_i] + 1/17)/3`, then call `fit_slate` directly. |
| `AR/alternate.select_frozen_decoder` + `method.score_expected` (`:55-76`) and `fit_a.task_scores` (`:145-153`) | `q[t]` lookup | score with per-person law (`TDR/audits.expected_token_loss`). |
| `AR/fit_a.build_attack_bank` (`:349-353`) | `fit_law = q[token_codes]`; source must be 32×17 (`:329`) | accept per-role per-person law arrays (or a callable over rows). Cut aggregation (`:392`) becomes the two-block builder below. |
| `AR/fit_a.run_center_from_roles` (`:568-569,597,629-659`) | `validate_channel(n_states=32)`, `ChannelArtifact(Q)` save, `current_Q` 32×17 | new unit runner storing `B.npz, A.npz, eta, policy-bank pins, context pins`; keep the loop/record/receipt structure. |
| `AR/fit_a.select_final_bank_feasible` (`:510-543`) | `solver.replay_p1(q,…)` | generalized replay on parameter vector. |
| `AR/privacy_first._matrices` (`:104-130`) | 32×17 equality rows, D17 shape | re-express with the two-block variable vector; caps/τ logic unchanged. |
| `AR/fit_controls.load_final_problem` / `fit_controls_from_frozen` (`:74-331`) | final cost/cuts are 32×17 | MILP/gradient controls stay on the T32 table (they are the "parent-restricted T32" comparators); feed them the B-block cost/cuts only. |
| `AR/evaluate.token_law_for_release` (`:78-124`) and `AR/audit.person_token_law` (`:30-45`) | release = T0/leaf row lookup, or `task_only_model_dir` | add a new spec kind (like the task-only branch `:80-103`) that recomputes the per-person law from pinned B/A/η, frozen policies and φ on `x, ha, token_codes, teacher_p, residual` only. |
| `TAC/release.ChannelArtifact / OneReleaseSession` (`:46-209`) | T0 route into Q rows | new session: compute law privately, HMAC keyed draw (copy `TaskOnlyReleaseSession`). |
| `AR/fit_a.PROTECTED_ROLES`, delta whitelist (`:23,560`) | fine | keep. |

### 9.3 Where the new parameterization plugs in (coefficient builder)
For one frozen predictor with per-person token losses `ℓ_i ∈ R^17` on its valid rows and
normalized weights `w_i` (1/n or PWGTP/ΣPWGTP, computed on the **same** valid rows):
- B-block: `G_B[t,z] = Σ_{i:T_i=t} w_i ℓ_i(z)` — existing `aggregate_loss_coefficients`.
- A-block (new): `G_A[k,m] = Σ_i w_i φ_k(x_i) ℓ_i(d_m(x_i))` — needs `phi (n,K)` and policy
  actions `D (n,M)` int in 0..16 (column `m=0` = D17 action `argmax D17[T_i]`).
- Expected loss is exactly `<G_B,B> + <G_A,A>`; per-person law
  `q_i = B[T_i] + Σ_k φ_ik Σ_m A[k,m] e_{d_m(x_i)}` must reproduce it (build a replay test).
Build task cost pair (U,W) and each cut (U,W) this way from `task_person_losses` and
`person_loss_from_attack` / `_attack_loss_rows`; carry the existing provenance keys
(`coefficient_pool_sha256`, `class_order`, `weight_normalization`) so calibration checks still fire.
LP size stays tiny (544 + K·M + 1 variables).

### 9.4 Things that will break or mislead if not handled
1. Missing sanitized symlink in this worktree (§1.1) — fits refuse to start.
2. Codebook: task-only tokens are p-quantile ranks, D17 tokens are a different 17-action map; a
   common decoder trained only on the old coverage law never sees task-only token semantics.
   Decoder training law must include the policy-bank actions (e.g. mix in `e_{d_m(x)}`), and
   any alignment must be fitted on construction rows only (spec §8).
3. Bank attackers (`audit_fit`/`inner_selection`) and the independent audit
   (`evaluate.audit_panel` also fits on `audit_fit`, selects on `inner_selection`, scores
   `inner_check`) share rows; decoder selection, attack-slate validation and round selection all
   use `inner_selection`. Decide explicitly whether policy/context fitting may use
   `nuisance_train` only (predecessor precedent: nuisance and task-only on `nuisance_train`).
4. Support: B-branch found max 198 unique coefficient households per T32 parent (a0); any
   per-T32 × context estimate will be thin — the pooled A-block is the intended fix.
5. `select_frozen_decoder` scores under the *previous* channel; keep or change deliberately.
6. A 32×17 `validate_channel` tolerance 1e-8 vs `controls._channel` 1e-10 vs evaluate 1e-10:
   renormalize LP output (`solver._clean`) before persisting.
7. Stored T0/teacher_p/residual are Linux-x86 artifacts; any new policy/context using them needs
   a Linux parity receipt analogous to `fit_b.runtime_parity_receipt` before deployment claims.
