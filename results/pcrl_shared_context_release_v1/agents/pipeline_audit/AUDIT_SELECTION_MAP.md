# Audit / selection / inference implementation map (predecessor commit 2ce5d171)

Prefixes: `E/` = `experiments/pcrl_adaptive_release_v1/`, `CUT/` = `experiments/pcrl_task_aligned_cuts_v1/`,
`TD/` = `experiments/pcrl_task_directed_release_v1/`. The code was read only; the 181/181 predecessor tests pass locally.

## 1. Data roles

- `E/roles.py:10-26`: each household is assigned one global role by `SHA256("pcrl_adaptive_release_v1|"+hh)`, with cut points
  nuisance_train .20 / audit_fit .40 / coefficient_split .65 / inner_selection .75 / inner_check .85 / outer_assessment 1.0.
  The role is independent of the anchor, so the three anchors share outer households. That is why the bootstrap uses a common resample (§5).
- `roles.pooled_role` refuses `outer_assessment` (`E/roles.py:33-50`).
- Who uses which role:

  | Role | Used by |
  |---|---|
  | `nuisance_train` | task decoder fit (`E/fit_a.py:201`); B nuisance model |
  | `coefficient_split` | coefficients, dual prices, split fitting |
  | `audit_fit` | fitting both the optimization attack bank (`E/fit_a.py:331`) **and** the independent audit slate (`E/evaluate.py:256-282`) |
  | `inner_selection` | selecting routes for the bank and the audit; selecting candidates and family representatives; checking B splits (AMENDMENT_01) |
  | `inner_check` | the independent inner score only |

  Disclosure: the "fresh" audit attackers are separate fits on the *same* audit_fit/inner_selection households as the optimization bank. They are not a disjoint resource.

## 2. Independent audit slate

- **Roles audited** (`CUT/audit.py:18-22`): `utility:A/same_residence`, `attack:A/SEX`, `attack:A/RAC1P`, `attack:AB/SEX`, `attack:AB/RAC1P`.
  `attack:B/*` exists only as an H-only coalition ancestor (`E/evaluate.py:24`).
- **Views** (`CUT/audit.py:36-63`): A = H_A (4 columns, continuous and unrounded); B = H_B (2); AB = [H_A, H_B].
  The wire is `H` (no token) or `release` (token). B never receives the token (`CUT/audit.py:295-297, 413-418`).
- **Estimand**: the exact expected-token cross-entropy `E_{Z~q_i}[−log p(y_i|H_i,Z)]` per original person, floored at 1e-9 (`CUT/audit.py:153-169`).
  Score = U mean and PWGTP-weighted mean; selection uses the balanced value 0.5U + 0.5W (`CUT/audit.py:172-190`).
- **Families** (`TD/audits.py:42-49, 502-599`). The standard slate, which the lock requires (`E/selection_lock.py:82`), contains:
  - `logistic` (C=1, with H×token one-hot interactions);
  - `hist_gb_20` and `hist_gb_5` (exact person-token expansion; `min_samples_leaf = leaf × max supported tokens per person`, `TD/audits.py:366-368`);
  - `sampled_hist_gb_20` and `sampled_hist_gb_5` (one fixed token draw per person, seed 20260921+seed);
  - one 120-epoch MLP [64,32], weight decay 1e-4.

  The `catchup` slate adds `mlp_360`. Fits use balanced person weights `.5+.5·w/mean(w)` (`TD/audits.py:148-152`).
  In `E/audit.py`, all fits and selection wrap `CUT/audit.py`; `E/audit.py:67-72` rejects any auxiliary continuous wire.
- **Fitting roles**: `fit_role_slate` fits on audit_fit and validates on inner_selection, and asserts that they are household-disjoint (`CUT/audit.py:319-373`).
  Seeds are `26000+1000·anchor+role_index`, identical across releases, which gives common random numbers (`E/evaluate.py:260, 281`).
- **Ignore-channel ancestors**: `build_role_route_bank` (`CUT/audit.py:449-492`), called through `E/evaluate.py:127-133`.
  - Utility and A roles: own-release slate plus the same-role H-only slate (12 routes).
  - AB roles: own AB slate, AB H-only slate, A same-release token slate, and B H-only slate (24 routes).

  `_legal_route` (`CUT/audit.py:403-429`) forbids cross-release token semantics and B/AB inputs to A. The fixed decoder route is *not* included (`include_fixed_decoder` defaults to False).
- **Validation-based selection**: `select_frozen_routes` takes the argmin of the balanced validation loss on inner_selection, with a lexical tie-break (`CUT/audit.py:495-515`, `CUT/audit.py:183-190`).
  `score_frozen_route` scores inner_check without refitting (`CUT/audit.py:518-536`).
- **Same-host H reference**:
  - Within a panel, one H-only slate per role is fitted once and shared by every release (`E/evaluate.py:252-271`). Each role reports `H_minus_candidate` (`E/evaluate.py:299`).
  - PrivacyFirst copies the H slates byte-for-byte rather than refitting (`E/run_privacy_inner.py:1-7, 190-201`). The merge verifies that H models, validation scores and person rows are byte-identical (`E/merge_inner_panels.py:217-248`).
  - Selection refuses a panel whose H validation score differs across releases (`E/selection.py:52-53`). Inference refuses H person rows that differ across releases (`E/inference_run.py:144-145`).
- **Private contributions**: per-person and per-household U/W numerators for candidate and H (`E/evaluate.py:136-162`). These are the only inputs to inference.
- **External continuous comparators** (`E/external_audit.py:157-325`): J and five 16-coordinate releases are appended as `aux` beside H.
  They use the same roles, slate, seeds and ancestor routes, and their own H slate is refitted inside that panel on the same host (`:232-247`).
  The AB feature order is permuted consistently (`:307`). J's outer score is `score_locked_j_outer` (`:361-480`), which runs behind the same unlock gate.
  They are labeled contextual, not matched.
- **Outer replay** (`E/outer_audit.py`):
  - `verify_locked_inner` pins the lock, the inner COMPLETE and INNER_AUDIT hashes, and the release descriptors (`:82-119`);
  - `verify_scoring_provenance` checks the index and scorer hashes (`:34-45`);
  - `reconstruct_selected_routes` reuses the saved route IDs and model hashes and never reselects (`:147-181`);
  - `score_locked_outer` opens outer rows only through `outer_access` (`:193-269`).

## 3. Inner selection rules (`E/selection.py`)

- **Inputs**: `inner_validation_from_report` reads the *validation* score of each role's selected route and of the H route. It never reads inner_check (`:21-56`).
  Caveat: these are the minimum validation losses, re-used as the selection statistic, so they carry winner's-curse optimism for every release. The comparison is symmetric but not held out.
- **Aliases**: `expand_named_scores` (`:59-83`) and `global_exact_aliases` (`E/selection_lock.py:31-50`) collapse laws only when they are identical on all three anchors.
- **Contrasts**: the equal three-anchor mean (`:93-113`).
  - task = CE(cand) − CE(D17);
  - sensitive = CE_S(D17) − CE_S(cand), where negative favors the candidate;
  - 4 sensitive roles × 2 weightings = 8 sensitive numbers.
- **Point screens** (`:116-127`, margins `:15-18`, `E/inference.py:17-21`):
  - **U**: max_w task ≤ **−0.003** and all 8 sensitive ≤ **+0.001**.
  - **P**: max_w task ≤ **+0.001**, all 8 sensitive ≤ +0.001, and AB/SEX (`P_TARGET`) ≤ **−0.002** in both weightings.
  - Both screens also require an **H capability** check: H−candidate residence benefit ≥ **0.01** in both weightings (`:198-205`).
- **Lexicographic ranking** (`:130-139`):
  - U: (max task, mean task, max sensitive, id);
  - P: (max AB/SEX, mean AB/SEX, max task, id).
- **Candidate choice and the diagnostic label** (`:190-215`): the code picks the rank-min among point-eligible candidates.
  If none are eligible, it picks the rank-min over **all** candidates and labels it `DIAGNOSTIC_ONLY`.
  - The label is written only into the lock slot (`E/selection_lock.py:159`).
  - Neither `outer_access`, `inference_run` nor `evaluate_family` reads it or gates on it, so a diagnostic slot receives the same endpoints and correction.
  - Both 2018 slots were `DIAGNOSTIC_ONLY`: U→A_selected, P→PrivacyFirst_selected.
- **Family representatives** (`:142-187`): simple, task_only, gradient and deterministic families, with prefixes at `E/selection_lock.py:16-23`, plus a D17 fallback.
  - The eligibility screen is weaker than the success screen: U requires max sensitive ≤ .001; P also requires task ≤ .001. There is no −.003, −.002 or .01 check.
  - Ranking uses the same lexicographic key.
- **Lock** (`E/selection_lock.py:111-262`):
  - candidate names are hard-coded: `A_selected`, `B_selected` (dropped if it is an exact A alias, `:144-145`) and `PrivacyFirst_selected`;
  - there is one slot per route, with comparators D17, the four family representatives and `A_control_historical_Q`, de-duplicated (`:152-160`);
  - the lock requires `slate=="standard"` and `delta==.001` (`:81-82`);
  - the J context is optional (`:226-251`).
  - `run_inner.py` builds the canonical (exact-law de-duplicated) release specs and audits each distinct law once, keeping aliases (`:64-109`).
  - `run_privacy_inner.py` and `merge_inner_panels.py` add PrivacyFirst without refitting H (§2).

## 4. Endpoint generation and correction

- `E/inference.enumerate_endpoints` (`:24-68`) iterates slots × canonical comparator groups × 5 roles × 2 weightings. Each endpoint carries its clause, threshold, sign and ID.
  - Clauses: `task` for utility; `target` for AB/SEX when the arm is P; `guard` otherwise.
  - Thresholds: U task −.003; P task +.001; target −.002; guard +.001.
- `family_manifest` (`:71-80`) is embedded in `SELECTION_LOCK.json`. Its hash is the lock SHA `4f894f98…`, which is committed and remote-verified before unlock.
  `inference_run.validate_locked_resolution` re-derives the manifest from the slots and refuses any mismatch (`E/inference_run.py:39-67`).
  `outer_access.verify_lock_structure` repeats the check (`E/outer_access.py:85-183`). The count comes from the list itself: `family_size = len(rows)`.
- **Primary family**: U has 3 comparator groups (D17, TaskOnly, historical_Q; constant_replace_100 is aliased to D17), giving 30 endpoints. P has 4 groups (plus GRADIENT_001), giving 40.
  The total is **70**, with two-sided Bonferroni z = norm.isf(.05/140) = **3.384**.
- **Separate H-capability family**: 2 slots × 2 weightings = 4, z = 2.498, threshold −.01 (`CUT/inference.py:74-82`, `E/inference_run.py:25-31, 214-234`).
- **Bootstrap**: `TD/uncertainty.paired_household_bounds` (`:112-245`).
  - Per anchor, the ratio Σ_hh(w·Δ)/Σ_hh w; the estimate is the equal mean of the three anchors (`:40-73, 153`).
  - One multinomial resample over the union of household IDs across all anchors and endpoints (`:145-192`). Overlapping anchors therefore share each draw, and endpoints are paired.
  - 10,000 draws, seed 20260923.
  - Bounds are `est ± z·SE_boot`: a normal approximation, not percentiles (`:205-222`).
  - An endpoint that is exactly identical on every anchor is forced to est = SE = 0. This happens, for example, when both releases choose the same H-only route (TaskOnly A/RAC1P). Such an endpoint trivially "passes" a +.001 guard, and this accounts for some of the 2/30 and 5/40 passes.
  - `passed_upper_bound = upper ≤ threshold` for every clause (`CUT/inference.py:121-151`).

## 5. Outer access control

- `E/outer_access.py`:
  - The lock path, the unlock receipt path, the census `DATA_ROLE_COUNTS.json` and the branch name are **module constants** tied to this study (`:16-27`).
  - `verify_outer_unlock` checks the lock SHA, the lock structure and a private `OUTER_UNLOCK.json` with `remote_verified: true` and a 40-hex commit (`:185-204`).
  - The scorer-code hash must match the lock, or the one registered post-lock scorer correction (`:40-83`).
  - `load_locked_outer_role` (`:207-227`).
- `E/outer_pool.py`: the SHA-pinned original is unpickled only after the gate. The code checks fifth-pool byte equality between the sanitized and original objects (`:83-109`) and replays the census counts: people, households and weight sum (`:34-80`).

## 6. Positive controls (`E/positive_control.py`)

- `revealing_token_law` (`:22-32`) is a nondeployable token equal to S.
- `run_inner_diagnostic` (`:96-214`) runs the same slate, ancestors and roles. It flags detection at H−leak ≥ .01 nats in both weightings.
- Exact XOR fixtures: `:35-59`.
- **Ran**: only `a0_ABRACE_positive_control` (AB/RAC1P, anchor 0). It improved 1.2106 (U) and 1.2065 (PWGTP) nats and was detected (`RUN_STATE.json`).
- **Not run**: the a1/a2 slots, a registered deviation (`UNIT_LEDGER.json:55`). No SEX-target and no weak or conditional-leak control ran.

## 7. Reuse for a new candidate family

**Unchanged interface: a released object that is one 17-token integer per person, audited as (H_A,Z) and (H_A,H_B,Z).**

The audit already consumes a **per-person token law** of size n×17. `E/audit.person_token_law` is only `Q[leaf]` (`E/audit.py:30-45`), and the positive control feeds per-person rows directly.

Reusable unchanged:
- `CUT/audit.py`: `role_arrays`, `fit_role_slate`, `build_role_route_bank`, `_legal_route`, `select_frozen_routes`, `score_frozen_route`, `expected_token_loss`, `score_weightings`, `assert_household_disjoint`;
- `TD/audits.fit_slate` (standard slate);
- `E/audit.py` wrappers;
- `E/evaluate._private_contributions`, `_inventory`, `_seal_private_permissions`;
- `E/selection.py`: all functions, if the margins are kept;
- `E/inference.py`, `CUT/inference.py`, `TD/uncertainty.py`;
- `E/outer_audit.reconstruct_selected_routes` and `score_locked_outer`;
- `E/outer_pool.pooled_locked_outer`;
- `E/positive_control.py`.

Must be re-parameterized, not edited in place:
- `outer_access` path, branch and census constants;
- `selection_lock.FAMILIES`, the hard-coded candidate names, the panel directory names, `delta==.001` and the J option;
- `roles.SALT` and role fractions, if the new study re-allocates households;
- the seeds, which should be new and registered.

Also run the missing SEX positive control and the a1/a2 controls.

**Candidate represented per person, as one q(z|x) row each, instead of a 32×17 matrix indexed by T32.** The audit maths is unchanged. What must change is the release plumbing:

1. `E/evaluate.token_law_for_release` (`:78-124`) accepts only a (Q, T0 or callable router) pair or the hard-coded task-only model.
   Add a new pinned law kind, modeled on the task-only branch at `:80-104`. It should load a SHA-pinned frozen model and compute the n×17 law from the legal-field whitelist `x, ha, token_codes, teacher_p, residual` (`:119-120`).
2. Release descriptors and provenance must know the new kind:
   - `E/evaluate.py:220-247`;
   - `E/outer_audit.release_descriptor` (`:48-79`) and `current_inner_code_hashes`/`verify_scoring_provenance` (`:21-45`), which hash `task_baselines.py` only for the task-only kind;
   - `E/evaluate.py:321-328`;
   - `E/run_inner._verify_complete` (`:48-60`);
   - `E/merge_inner_panels` source checks and the T0-alias check (`:340-353`).
3. Exact-law identity and de-duplication (`E/release_specs.py:56-92`) hash a 32×17 array.
   A per-person law needs an identity such as the model SHA plus mode, or the hash of the law on a pinned canonical row set.
   Otherwise a candidate that collapses to D17 will not be aliased, and the endpoints will double-count one physical release.
4. The fitting-side reference and constraints (spec §11) currently aggregate to state×17 costs: `E/fit_a.aggregate_loss_coefficients` (`:90`) and `E/fit_b.fixed_bank_dual_prices` (`:307-365`).
   For per-person q, compute ρ_r and L_a(q) = Σ_i w_i Σ_z q_i(z)·loss_ia(z) on the same rows from `fit_a.person_loss_from_attack` (`:266`). These are still linear in q.
   The 32-row LP must be replaced by a parametric or person-level fit.
5. Audit-capacity side effect: the exact `hist_gb` families set `min_samples_leaf = leaf × max_z-support per person` (`TD/audits.py:366-368`).
   Diffuse per-person rows, with tiny mass on many tokens, raise this bound up to 17×, weakening those two attackers, and inflate the expanded rows up to 17×.
   Keep the `sampled_hist_gb_*` variants and report the per-row support, or impose a registered mass floor.
6. Selection, inference, outer access and positive controls need no change beyond the new release names and family prefixes.
