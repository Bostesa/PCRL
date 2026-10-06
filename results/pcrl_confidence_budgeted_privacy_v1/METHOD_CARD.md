# Method card: confidence-budgeted privacy bank (`cbp/fit.py`, `cbp/deploy.py`)

Owner: role B (optimizer engineer). Tests: `cbp/tests/test_fit.py`, run on synthetic data only. Every rule below is
fixed in code before any new real-data fit. Role B code reads no Adult row, task label or SEX column. The lead runs the
real-data stages (`cbp.run --stage fit`) after FIT_LOCK and AUDIT_AND_SELECTION_LOCK are pushed.

**Scope.** This study does not change the optimiser (prompt §7). All mathematics is the qpc confidence-capacity
study's (`results/pcrl_confidence_capacity_v1/METHOD_CARD.md`, sections 1–3 and 6). It is imported by module, not
copied:

- `qpc/compress.py`, `qpc/release.py`, `qpc/partition.py`, `qpc/kmeans.py`, `qpc/stagea.py`;
- through them, `dpc/compress.py`, `dpc/release.py`, `dpc/partition.py`;
- `qpc/deploy.py` and `dpc/deploy.py` for deployment.

None of these files is edited. Their sha256 pins come from the qpc `STAGE_B_LOCK.json` (written
2026-10-06T04:43:38Z), and the Stage A code (`qpc/stagea.py`, `qpc/kmeans.py`, `qpc/release.py`) from
`STAGE_A_LOCK.json`. `cbp.fit` hard-codes the pins and checks them against both the lock files and the working tree
(`code_version`). At import it also asserts that the qpc constants it relies on are unchanged:

| Constant | Value |
|---|---|
| `TOL`, `TIE_TOL` | 1e-12 |
| `SWEEPS` | 5 |
| `EPS` | 1e-12 |
| `WITNESSES` | FINE-TASK, LOCAL, SEQ-12, SEQ-21 |
| `JOINT_START_ORDER` | JOINT-GREEDY, FINE-TASK, LOCAL, SEQ-12, SEQ-21 |
| `PERM_SEED` | 20261006 |
| `N_PERM` | 100 |
| `OLD_RULE_DIAGNOSTIC` | True |

A changed qpc refuses to load `cbp.fit`.

The study adds **none** of the following: convergence rescue, occupation-only objective, randomisation, feedback,
coordinated-move solver, extra JOINT start, continuation path or a privileged JOINT start. A valuable new optimiser
idea would need a separate protocol.

Recipients: 1 = income (K = 2) and 2 = occupation_group (K = 6). Arithmetic is float64 throughout, with natural logs.

## 1. The fixed bank (`cbp.fit.bank`, `reference_units`, `new_fit_order`)

| Item | Rule |
|---|---|
| Rate | ONE rate, (m1, m2) = (8, 64) states per teacher-predicted class. It is never changed in response to results |
| Fine partitions | the admitted qpc `fine__s{k}`: at most 32 income and 128 occupation fine cells per predicted class. Never refit |
| λ grid | exactly (0.01, 0.025, 0.04, 0.06, 0.08, 0.1), as the tuple `LAMS` |
| Families | LOCAL, SEQ-12, SEQ-21, JOINT |
| Seeds | 0, 1, 2 (frozen U teachers `tea__s{k}__U`) |
| Logical privacy units | 6 × 4 × 3 = **72** |
| REUSED (admitted after parity) | λ ∈ {0.01, 0.1}: **24** qpc units |
| NEW fits | λ ∈ {0.025, 0.04, 0.06, 0.08}: **48** units |
| Reused task-only references | `U|DIRECT-TASK|i8o64`, `U|FINE-TASK|i8o64`, `U|CLASS|i1o1` × 3 seeds = **9** units |
| Config IDs | `U|{FAM}|i8o64|l{lam:g}` (`l0.01 l0.025 l0.04 l0.06 l0.08 l0.1`), `U|FINE-TASK|i8o64`, `U|DIRECT-TASK|i8o64`, `U|CLASS|i1o1` (the qpc scheme, `qpc.compress.config_id`) |
| Unit names | `pol__s{k}__<config with "|" replaced by "_">`, for example `pol__s2__U_SEQ-21_i8o64_l0.025` |
| JOINT witnesses (same seed and λ) | `U|FINE-TASK|i8o64` (λ-free, admitted) and `U|{LOCAL,SEQ-12,SEQ-21}|i8o64|l{lam}`. At λ 0.01 and 0.1 these are the admitted qpc units |
| Fit order (`new_fit_order`) | per seed and new λ: LOCAL, SEQ-12, SEQ-21, then JOINT |
| Exact aliases | counted after fitting with `cbp.fit.aliases`, which is qpc `find_aliases` on pair and recipient fingerprints. An alias is one fitted or reused unit, never an extra model fit |

Reuse, alias and new-fit counts are kept separate. The 72 logical units are not 72 model fits.

**Reuse rule.** A REUSED unit is used only if `endpoint_parity` passes (section 6). A valid unit is never refit to
manufacture a receipt. `cbp.fit.fit_unit` refuses λ 0.01 or 0.1 unless an explicit `refit_reason` documents why the
admitted unit could not be reused or restored. Such a unit is recorded as `bank_status = "ENDPOINT_REFIT"`, so it can
never pass as a restoration.

## 2. Objectives (qpc, unchanged)

On the N OSF_DEFENSE_FIT rows, with C_i the complete released token identity of recipient i:

- `D_i = (1/N) sum_t [A_t - S_t . log q_t]`, with `q_t = smooth(S_t / n_t, class_t)`. This is the mean fitting
  KL(p_i ‖ decoded).
- `I_i = I(S; C_i)` and `I_12 = I(S; C_1, C_2)` are plug-in MI of the exact fitting count tables, with no smoothing
  and 0 log 0 = 0.
- `F_task = D1 + D2`.
- `F_local = D1 + D2 + λ (I1 + I2)/2`.
- `F_joint = D1 + D2 + λ [(I1 + I2)/2 + I12]`.

Prototypes and partitions use only OSF_DEFENSE_FIT teacher probabilities and SEX. True task labels never enter a fit;
they enter only the registered inner utility selection.

## 3. Search engine (qpc, unchanged; qpc METHOD_CARD 6.3)

**Canonicalisation and sufficient statistics.** Each coarse cell is labelled by a member fine index and never crosses
predicted classes. Per-label `n`, `S` (accumulated over members in increasing fine index), `h = -S . log q` and the
exact int64 count tables are kept, and every coarse table is an exact sum of the fine table n(s, f1, f2). Exported
groupings are canonical, with each cell labelled by its lowest fine index. Token IDs are canonical, numbered by first
appearance in increasing fine-cell order.

**Greedy to the caps.** Each step evaluates every same-class merge in classes above the cap and applies the smallest
exact increment. Candidates within `TIE_TOL = 1e-12` of the minimum are tied, and the first in lexicographic order
(recipient, class, label a, label b) wins. Positive increments are allowed and recorded.

**Objective-improving extra merges ("at most the cap").** These are applied only while the increment is `< -TOL`, so
actual alphabets may be smaller than the cap. They are reported.

**Refinement.** There are at most **5** sweeps (`SWEEPS = 5`). A sweep is one exact single-fine-cell exchange pass:
recipients 1 then 2; fine cells in increasing index; a move may never empty a cell; the best target wins, with ties
going to the lowest coarse label; a move is accepted only if `ΔF < -TOL`. The sweep then runs the improving-merge pass.
Converged means a sweep with no move and no merge. Hitting the sweep cap is recorded as `converged = False`. It is not
rescued.

**Checks.** Every reported objective is recomputed from scratch from the final state and from the released tokens and
decoded vectors of the fitting rows. qpc asserts that these agree within 1e-9 absolute; in qpc they agreed to about
1e-15. The search is a greedy local search, and no global optimum is claimed.

## 4. Families

- **LOCAL.** Each recipient is optimised alone on `D_i + λ I_i / 2`: greedy, extra merges, then refinement.
- **SEQ-12 / SEQ-21 (the corrected sequential design, qpc).**
  - **Stage 1.** The first recipient starts from its fine cells and is optimised under the actual `F_joint`, with the
    other recipient held at its CLASS-ONLY release (one token per predicted class, because the decision is always
    disclosed).
  - **Stage 2.** The first map is frozen. The second recipient restarts from its fine cells and is optimised under
    `F_joint` given the frozen map. The first recipient is never revised (asserted).
  - The other recipient is never replaced by a fictitious constant or no-release view.
  - dpc's old stage-one surrogate, `D_a + 1.5 λ I_a`, is fitted only as an unselected diagnostic
    (`baseline_correction.old_rule_stage1`, `OLD_RULE_DIAGNOSTIC = True` in qpc). It is never released.
- **JOINT.**
  - **Starts.** There are five fixed starts:
    - JOINT-GREEDY: `F_joint` greedy over both recipients, extra merges, then joint refinement;
    - joint refinement (at most 5 sweeps) of each of the four witnesses: FINE-TASK, LOCAL, SEQ-12 and SEQ-21 at the
      same caps and λ.
  - **Witness validation.** The witnesses are passed in as the bank's own `policy.json` dicts. `cbp.fit.fit_unit`
    requires exactly these four. qpc validates their fine-partition fingerprints, family, caps and λ.
  - **Candidates.** There are nine: the five refined starts and the four UNCHANGED witnesses.
  - **Selection.** The lowest from-scratch `F_joint` wins by exact comparison. Ties go in the order JOINT-GREEDY,
    FINE-TASK, LOCAL, SEQ-12, SEQ-21, with refined before unchanged.
  - **Dominance.** The assertion is final `F_joint` ≤ the `F_joint` of every unchanged witness, **on the fitting
    objective only**. It does not imply lower held-out recovery, global optimality, or containment of DIRECT-TASK
    (which lies outside the fine-state family).
  - **Unresolved local optima.** These are reported: refined starts that end at a different map with a strictly
    higher `F_joint`. Starts that did not converge are also reported.
  - **No privileged start.** No continuation path or privileged start is available to JOINT that the source design
    did not have.
- **References (reused, never refit).**
  - FINE-TASK is the family-matched task-only comparator, on `D_i` only.
  - CLASS-ONLY maps every fine cell of a class to one token.
  - DIRECT-TASK is the Stage A direct k-means code at (8, 64), on its own partition. It is a stronger external
    compression control and need not lie in the fine-state search family.

### 4.1 Computational asymmetry (recorded, not equalised)

Equal λ access, data, state caps and per-stage refinement limits do **not** imply equal realised compute.

**Per-unit receipt (`cbp.computational_asymmetry`).**

| Family | Search paths | Candidates compared | Witnesses consumed | Unselected diagnostic fits |
|---|---|---|---|---|
| JOINT | 5 | 9 | 4 (each a complete fitted unit of another family) | 0 |
| SEQ | 1 (two frozen stages) | 1 | 0 | 1 (old-rule stage 1) |
| LOCAL | 1 (two independent per-recipient stages) | 1 | 0 | 0 |

Each receipt also records the unit's CPU and wall seconds, its total sweeps, accepted exchanges and logged merges, and
qpc's `work` counters. The counters cover different scopes:

- JOINT: summed over its five starts, excluding its witnesses' own fits;
- SEQ: the stage-2 state only, because qpc's counters omit stage 1 and the diagnostic. Use the logged merges and
  moves, and the CPU seconds, instead;
- LOCAL: the single state.

**Bank-level table (`cbp.fit.asymmetry_table`).** For every (seed, λ) it gives JOINT's own CPU, JOINT's CPU including
its four witness fits, each sequential arm's CPU and their ratios.

**Interpretation.** The JOINT-versus-sequential comparison is not an exact compute match. The sequential adaptations
are qpc's corrected design, not the official Taylor solver.

## 5. Release interface and exact properties (qpc/dpc, unchanged)

**What each recipient receives.** Its categorical token identity, the decoded probability vector of that token, and the
unchanged teacher decision. Two tokens with identical decoded vectors are distinct disclosures. Both recipients'
complete interfaces are aligned for the pair view: `pair_index = t1 · T2 + t2`, with full alphabets that include
fallback tokens.

**Deployment of a row (p, d).**

- Inputs are only the recipient's own frozen teacher probabilities, with `d = argmax p` (numpy first-index ties).
  Any other label array is refused.
- The row's predicted class d selects that class's assignment cells. The nearest cell by KL to the stored smoothed
  routing centroid wins, with ties going to the lowest index. A class with only a fallback cell maps to that cell.
- The row then receives the cell's token, the token's decoded vector and the token's class. Nothing is updated at
  deployment.
- Deployment never reads SEX, true labels, the partner's output, person or row-role IDs, fine-cell debug IDs or
  assessment status.

**Smoothing.** `smooth(mean, c) = (mean + ε·1 + ε·e_c) / (1 + (K+1) ε)` with ε = 1e-12, float64.

- Decoded prototype of token t: `smooth(S_t / n_t, class_t)`, with S_t and n_t accumulated over member cells in
  increasing index.
- A fallback token (a class with no fitting row; in the real data, occupation class 5) decodes to
  `smooth(uniform, class)`.
- `encode` re-checks every emitted vector bitwise against this formula.

**Decision containment (construction; the formal statement is role E's).** Decisions are preserved under three
explicit assumptions, each enforced at construction:

- (i) every class has at least one cell, because an absent class gets its reserved fallback cell;
- (ii) a token never mixes predicted classes (`token_tables` refuses);
- (iii) every prototype's strict argmax is its class (`check_prototypes` refuses otherwise).

The argument has two steps. Under (i) and (ii), a row with teacher decision d is routed only to cells of class d, and
its released decision, the token class, equals d. For (iii): each member p has `argmax = c` under first-index ties, so
`p_c ≥ p_k` for every k. IEEE summation and division are monotone, so the token mean keeps `mean_c ≥ mean_k`. The +ε·e_c
term then makes the argmax strict, and any violation is refused rather than released.

Exact decision preservation is still checked on every row of every encode (`check_release_rows`), and on ALL rows of
every fitted or admitted unit (`qpc.stagea.encode_all` raises on a mismatch). Class-preserving compression cannot
improve a teacher with zero recall for an absent class, and no such claim is made.

**Log loss.** Clip 1e-12, natural log (`qpc.utility`; lead and selection). This is not used in fitting.

**Never exported.** Raw teacher scores, continuous residuals, fine IDs, distances, hidden identifiers and debug
outputs.

**Scope of guarantees.** The structural guarantees are decision preservation and the standard data-processing facts.
They imply no SEX AUC bound, no population MI bound and no training-data privacy.

## 6. What `cbp.fit` adds

### 6.1 `fit_unit(family, fine_dict, T, tr, S_fit, lam, meta, witnesses=None, *, refit_reason=None)`

**The qpc call.** `fit_unit` calls `qpc.compress.fit_unit(family, fine_dict, T, tr, S_fit, 8, 64, lam, meta,
witnesses=witnesses)` unchanged. qpc's `(record, files)` are returned untouched; `files` are `policy.json` and
`release.npz` over ALL rows. The record gains one key, `"cbp"`. A test asserts that the record without `"cbp"`, and
both files, are identical to a direct qpc call.

**Refusals.**

- a family outside LOCAL, SEQ-12, SEQ-21 and JOINT;
- a λ off the grid;
- a REUSED λ without `refit_reason`;
- JOINT without exactly the four witness dicts;
- witnesses passed to a non-JOINT family;
- a `meta.config` that differs from the configuration ID;
- a teacher other than U.

**`cbp.section13`.** These are the prompt §13 diagnostics. Every statistic comes from the DEPLOYED release on the
fitting rows.

- Alphabets:
  - full alphabets of r1, r2 and the pair (T1, T2 and T1·T2, including fallbacks);
  - occupied alphabets on the fitting rows;
  - tokens and occupied states per predicted class;
  - total occupied states.
- Code statistics:
  - emitted entropy, singleton cells, cells with n < 5, and unseen fractions (local and pair);
  - per-class support: tokens, occupied, rows, min and median count, singletons, fallback;
  - fallback tokens.
- Objective terms:
  - `objective_terms_deployed`: D1, D2, I1, I2, I12 and the F values;
  - `deployed_vs_record_final`: per-term absolute and relative differences against the record's `final`, with the
    1e-12 relative flag;
  - `deployed_vs_qpc_row_level_check`: expected bitwise, because it is the same arithmetic as qpc's own receipt;
  - the λ-weighted privacy terms versus distortion.
- Fitted MI vs permutation null: fitted I1, I2, I12 against qpc's 100-permutation null (mean, q95, max, excess).
- `reconstruction`: token counts against the stored `n_t` (exact); per-token mean teacher probability against
  `S_t/n_t` (≤ 1e-12 absolute; the summation order differs); decoded vectors against the policy prototypes (bitwise);
  decisions against the teacher and the token class (exact); decoded argmax against the decision.
- `search`: per stage or start, merges to the cap, greedy extra merges, logged merges, positive-increment merges,
  accepted exchanges and refinement merges; sweeps against the cap of 5; convergence; JOINT's unresolved local optima,
  defined only across JOINT starts and `None` for a single-path family; the winner; and qpc's summary.

True-label log loss and Brier versus teacher KL are not fitting statistics. They come from the inner utility stage.
Fitted plug-in MI is a training criterion. It is not a privacy bound or a population MI.

**`cbp.computational_asymmetry`.** See 4.1.

**Failure handling.** A structural reconstruction failure (counts, prototypes, decisions) raises. The 1e-12 term
comparison for NEW fits is recorded, not raised. qpc's own 1e-9 absolute assertion remains the hard check.

### 6.2 Registered tolerances (fixed before any fit)

| Quantity | Rule |
|---|---|
| D1, D2, I1, I2, I12 and the F values, release-derived against the record (parity gate) | `|a − b| ≤ 1e-12 · max(|a|, |b|)`; equality passes, including at 0 |
| Per-token mean probability, release rows against stored statistics | `≤ 1e-12` absolute |
| Token IDs, decoded vectors, decisions, release arrays, fingerprints, fine partitions, code hashes | exact |

On the admitted qpc receipts, the bound `row_level_max_abs_diff / min term` over all 42 Stage B units is
≤ 5.3e-13 relative. The synthetic units in this study show about 4e-15. A real unit that failed only the term gate by
a summation-order margin would be reported to the lead. The tolerance is not relaxed after data are seen.

## 7. Endpoint parity for REUSED units (`endpoint_parity`)

`endpoint_parity(unit_dir, T, tr, S_fit, fine_dict, *, expected_config=None, meta=None, witness_records=None)`

It applies to the 8 reusable privacy configurations (λ 0.01 and 0.1 × four families) and to DIRECT-TASK, FINE-TASK and
CLASS (11 per seed, 33 in total). It **never refits and writes nothing**. It returns `ok`, `status`
(`PARITY_PASS` or `PARITY_FAIL`), `failed` and per-check detail, plus the full section 13 block for the reused unit, so
§13 coverage is uniform over all 72 + 9 units. All gates must hold:

| Gate | Check |
|---|---|
| `complete_json_hashes` | every file matches the unit's `COMPLETE.json` (`jcv.finalize.unit_complete`) |
| `policy_integrity` | `policy.json` loads as `qpc.PolicyPair`: prototypes recomputed from the stored sums, centroid and fingerprint checks; dpc kinds refused |
| `config_registered_reusable` | the record's config is one of the 11 reusable configurations |
| `config_consistent` | record config = policy config = `config_id(family, λ)` (= `expected_config`); caps (8, 64), or (1, 1) for CLASS |
| `fingerprint_matches_record` | policy fingerprint = the record's pair fingerprint (and `pair_record.fingerprint`) |
| `row_id_equal` | stored `row_id` = teacher `row_id` |
| `release_bitwise` | `qpc.stagea.encode_all(pair, T)` (re-encode with `qpc.release` from `policy.json` and the admitted teacher, ALL rows) equals `release.npz` key by key: same key set, dtype, shape and bytes |
| `class_preservation_all_rows` | stored and re-encoded decisions = teacher decisions on ALL rows; stored decoded argmax = decision |
| `fitting_statistics_exact` | the section 6.1 reconstruction from the stored release on the fitting rows |
| `fitting_terms_within_rtol` | D1, D2, I1, I2, I12 (+ F) from the stored release against the record's `final` (DIRECT-TASK: D1, D2 against its `fit_distortion`), rtol 1e-12 |
| `fine_partition` | Stage B units: both assignment partitions' fingerprints = the admitted `fine__s{k}` and = the record's `fine_fingerprints`. DIRECT-TASK: not applicable (own Stage A partition); its partitions must match its record's receipts |
| `code_version` | current `qpc/compress.py` and `qpc/release.py`, and the dpc/qpc closure, = the qpc STAGE_B_LOCK pins; DIRECT-TASK also = the STAGE_A_LOCK pins of `qpc/stagea.py`, `qpc/kmeans.py`, `qpc/release.py` |
| `binding` (with `meta`) | the policy's `teacher_model_sha256` and `feature_names_sha256` = the admitted teacher and schema |
| `joint_witness_dominance` (JOINT) | every `final_minus_witness ≤ 0`; all four witnesses `passed_in`; with `witness_records`, each unchanged-witness F_joint = that witness unit's final F_joint within 1e-12 relative (FINE-TASK's F_joint is computed from its terms at the JOINT's λ) |

**Tamper tests.** Each of the following is detected, with the exact expected failed-gate set:

- a token flip, with and without rehashing `COMPLETE.json`;
- a 1-ulp change to a decoded value;
- a decision flip;
- an extra release key;
- a final term changed by 1e-10 relative;
- the wrong fine partition;
- a changed prototype in `policy.json`;
- a code pin mismatch;
- a binding mismatch;
- a new-λ unit passed as reusable;
- a perturbed witness record;
- the wrong expected config;
- a changed record fingerprint;
- an unreadable release.

## 8. Deployment (`python -m cbp.deploy`)

```
OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m cbp.deploy --unit <teacher unit dir> \
    --policy <policy.json> --X <input.npz> --schema <pinned schema npz|json> --out <release.npz> \
    [--schema-sha256 <hex>] [--seed k]
```

`cbp.deploy` is a thin wrapper around `qpc.deploy` (d0c8a45), with the following parts imported unchanged:

| Part | Behaviour |
|---|---|
| Flag allow-list | `--unit --policy --X --schema --schema-sha256 --out --seed --help`. Every other flag is refused, with an explicit message for export or bypass attempts (fine/cell IDs, raw/teacher scores, logits, features, latent, distances, debug, export, include/extra, SEX or labels, allow/unbound, continuous) |
| Input | exactly `X` (n, 83) finite float and `feature_names`, equal to the pinned schema in order. Missing, extra, renamed or reordered columns and any extra array are refused |
| Teacher | the hash-complete unit's `jcv.train.Model(83, [2, 6], seed)` with its deployed heads |
| Bindings | mandatory: teacher `model.pt` sha256 and feature-schema sha256. There is no unbound override |
| Policy records | only `qpc.PolicyPair` records, with integrity checks |
| Output writer | writes only `tokens_1, probs_1, decision_1, tokens_2, probs_2, decision_2` |

**The one cbp addition.** The policy's configuration must be a registered cbp configuration (`registered_ids()`: the 24
grid configurations plus DIRECT-TASK, FINE-TASK and CLASS). It must also be consistent with the policy's own family,
caps and λ. Anything else is refused: a λ 1 policy, a different rate, or a relabelled config. Every refusal exits
with code 2.

**Tests.** The tests use a synthetic torch teacher. Deployment equals the stored release on every row, and also from
a restored copy of the policy.

## 9. Synthetic timing of the full registered bank (TIMING.json, key `fitting`)

**Data.** `python -m cbp.fit timing` ran once, at 17:04–17:07Z under `cbp.sema` (label `B:fit-timing`), with one
thread. It used SYNTHETIC real-shaped data (`cbp.fit.synthetic_teacher`):

- 39,170 rows, equal to the five OSF role sizes, with a fixed random 15,434 as the fitting rows;
- income K = 2, with 195 exact 0/1 rows and class imbalance;
- occupation K = 6, with 5 predicted classes and class 5 never predicted;
- binary S correlated with both scores;
- fine partitions of 32 income / 128 occupation cells per class (F = 64 / 641).

**Setup, not cbp work.** For three seeds, 74.7 CPU-s went to fitting synthetic analogues of the ADMITTED artifacts with
plain qpc. These are the fine partitions (5.0–5.5 CPU-s per seed), DIRECT-TASK, FINE-TASK, CLASS and the λ 0.01/0.1
privacy units. They exist only so that the new JOINT units have witnesses and the parity step has units to check. In
the real study these artifacts are admitted, not refit.

**Mandatory cbp fitting work, 3 seeds.**

| Item | Units | Process CPU (s) |
|---|---|---|
| NEW LOCAL (λ 0.025/0.04/0.06/0.08) | 12 | 9.7 total; mean 0.81, max 0.88 |
| NEW SEQ-12 | 12 | 16.3 total; mean 1.36, max 1.46 |
| NEW SEQ-21 | 12 | 15.9 total; mean 1.33, max 1.43 |
| NEW JOINT (four witnesses passed in) | 12 | 53.9 total; mean 4.49, max 4.81 |
| **All 48 new fits** (qpc fit + section 13 + unit write) | 48 | **95.9** |
| Endpoint parity of 24 reused privacy units + 9 references | 33 | **4.7** (0.10–0.19 per unit); **33/33 PARITY_PASS** |
| **Mandatory total** | | **100.6 CPU-s** |

**Measured overheads and limits.**

- The section 13 augmentation costs at most 0.026 CPU-s per unit.
- Worker peak RSS was 0.66 GB.
- Units average 1.04 MB on disk, at most 1.28 MB, so the 48 new units take about 50 MB of private disk.

**Numerical results.** Deployed-release terms matched the record's `final` within 7.0e-14 relative on new units and
3.1e-13 on reused units. Both are inside the 1e-12 gate.

**Synthetic search behaviour.** This is descriptive only, not a real-data result.

- Every JOINT unit had 4 unresolved local optima.
- The JOINT winner was always a refined start: refined witnesses in 11 of 12 units, JOINT-GREEDY in 1.
- JOINT starts that did not converge within 5 sweeps, and one SEQ-12 stage-2 that did not converge, are recorded,
  not rescued.
- No exact pair aliases occurred.

**Projection.** qpc's real/synthetic ratio for the comparable bank was about 1.0: synthetic 83.8 CPU-s against real
partition + fit of 85 CPU-s. On that basis the real mandatory fitting work is projected at about 100 CPU-s, or at most
about 0.06 CPU-h with a ×2 margin. Against the 20 CPU-h budget this is negligible.

**Recommendation.** Run the FULL registered bank, with no reduction. No scientific choice depends on the timing:
there is no λ pruning.

**Scheduling.** Use one or two shards. Parity of the 33 reused units comes first. Then, per seed and new λ, fit
LOCAL, SEQ-12 and SEQ-21 before JOINT.

## 10. Not claimed

- No global optimum is claimed: the search is greedy local search with a 5-sweep cap.
- No JOINT superiority is claimed beyond fitting-objective dominance over its unchanged witnesses.
- No exact compute match between JOINT and the sequential arms is claimed.
- No official Taylor solver is claimed.
- No population or SEX-AUC bound is claimed.
- Not general-purpose representation learning: the code releases fixed-task probabilities and decisions only.
- Synthetic timings are not real-data results.
