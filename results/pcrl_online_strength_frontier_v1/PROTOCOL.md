# Protocol: online strength frontier and fixed recipient allocation

Branch `research/pcrl-online-strength-frontier-v1`, created from the pinned commit `a9951ed2fed9943d445a208a8a7e456a56f39114` (`research/pcrl-strength-matched-feedback-v1`). Package `osf/`.

This is **exploratory development** on 39,205 historically exposed UCI Adult rows. No result here is fresh or prospective confirmation.

## 1. Question

The predecessor study (smf) closed EXPERIMENTAL_NO_ADVANTAGE. Its best development model was ordinary online raw-penalty joint training, RAW-J (β 0.3, 40 epochs):

| Model | Pair AUC | Income AUC | Occupation AUC | Income accuracy | Occupation accuracy |
|---|---:|---:|---:|---:|---:|
| U | 0.882 | 0.860 | 0.880 | 0.852 | 0.480 |
| RAW-J | 0.810 | 0.768 | 0.796 | 0.853 | 0.474 |

Its normalized grid stopped at ρ 1.5. RAW-J's combined penalty/task ratio was about 2.6–3.2.

This study tests the remaining strength and frontier hypothesis on a fixed, small bank:
1. Does online normalized training reach RAW-J's actual strength and performance?
2. Does a fixed income/occupation allocation help, without the dynamic controller?
3. Does either normalized or raw joint training beat strong local controls at useful task performance?

Equal average norm does not equalize the temporal profile, directions, task interference, clipping or trajectory. This is a **strength-frontier comparison**, not a causal isolation of magnitude.

## 2. Data roles (`osf/data.py`; `ROLE_MANIFEST.json`)

The input is `adult_jcv.npz` (sha256 `e0d9e54a…2f12`).

**Fitting and selection roles are the smf roles, unchanged:**

| Role | Rows | Use |
|---|---:|---|
| OSF_DEFENSE_FIT | 15,434 (15,428 groups) | Encoders, training heads, critics, LEACE maps, FARE trees |
| HEAD_VALIDATION | 1,500 | Deployed-head C selection |
| AUDIT_FIT | 6,065 | Attacker fitting |
| INNER_SELECTION | 2,235 | Candidate evaluation, nomination, attacker selection |

The fitting tensors are byte-identical to smf `NEW_DEFENSE_FIT`. Critic subroles inside OSF_DEFENSE_FIT are CRITIC_FIT 10,764, CRITIC_VAL 2,343 and DIAGNOSTIC_CALIB 2,327 (the smf CONTROLLER_CALIB; no controller is trained). Encoders train on all OSF_DEFENSE_FIT rows.

**OSF_DEVELOPMENT_ASSESSMENT** is the fixed union of four previously used pools, minus every exact-record group that overlaps a fitting role or the inherited exclusions (excluded_exposure, excluded_dup):

| Pool | Nominal rows | Kept rows (groups) |
|---|---:|---:|
| Original assessment | 5,243 | 5,243 (5,243) |
| rgj DEVELOPMENT_ASSESSMENT | 3,397 | 3,397 (3,393) |
| smf NEW_DEVELOPMENT_ASSESSMENT | 3,796 | 3,796 (3,793) |
| Original certification pool | 1,500 | 1,500 (1,500), subject to the custody determination in `ROLE_MANIFEST.json` |
| **Total** | 13,936 | **13,936 rows, 13,929 groups** |

All eligible groups of the named pools are used, not a favourable subset. If custody cannot prove that no eligible model ever fitted or selected on the certification pool, the whole pool is excluded and the missing coverage documented.

**Preprocessing.** The numeric columns are inverted exactly from the admitted normalisation and re-standardised on OSF_DEFENSE_FIT only (`smf.data.refit_numeric`), with the same transformation applied everywhere. The 78 one-hot columns use the loader's fixed category sets; an unseen category is an all-zero block. The 83 permitted columns exclude SEX, RAC1P, income, occupation, fnlwgt and identifiers. Hard proxies, including Husband/Wife, are kept.

**Guards.**
- Assessment labels are masked (−1) at load. `osf.data.labels_for` is a role allowlist that refuses assessment labels to the training, heads, inner-audit and selection procedures.
- Only `osf.assess` unseals, and only after verifying that EVALUATION_LOCK is committed and present on origin.
- Earlier committed outcomes are planning evidence. No consolidated-cohort performance summary is produced before the lock.

**Statement of reuse.** These pools were scored or trained on before (`EXPOSURE_LEDGER.md`), and earlier results shaped this direction. The locked consolidated assessment is an exploratory benchmark check. Its adjusted intervals do not correct for that history.

## 3. Admitted predecessor checkpoints (`ADMISSION.json`, `osf/admit.py`)

The following are admitted only when receipts prove all four conditions: unchanged OSF_DEFENSE_FIT tensors and preprocessing; an unchanged task/critic configuration; unchanged head-selection roles; and full exclusion of OSF_DEVELOPMENT_ASSESSMENT from their fitting.
- smf warm starts `warm__s{0,1,2}`;
- U (`tl__s{k}__e40`, task-only, salt 0);
- RAW β 0.1 and 0.3 for J and L (`raw__s{k}__RAW-{J,L}__b{β}__e{20,40}`, pinned `rgj.train` J-O/L-O, stage B = salt 0);
- the smf LEACE maps and FARE trees (audit owner; `osf/baselines.py`).

Older checkpoints trained on the original 19,230 defense rows include smf's assessment rows and are ineligible.

Admitted releases are rebuilt on the osf rows with the identical head procedure. They must equal the smf releases bitwise on every smf row.

## 4. Common training path (`osf/train.py`)

**Architecture and optimiser.**
- Two encoders 83-64-64-16 (ReLU) with affine task heads; source task losses.
- Batch 256, SGD 0.05, global update-norm clip 5.
- Three paired seeds (0, 1, 2), all starting from the admitted per-seed warm state.
- Each run makes 40 protection/task epochs directly (61 steps per epoch, 2,440 steps). No selected local midpoint initialises any arm.

**Checkpoints.** Nomination uses epoch 40 only. Epoch 20 and the snapshots are diagnostic.

**Critics (ONLINE only, every protected arm).**
- Before every encoder step, the floored ZCA transform is recomputed on the fixed 4,096-row CRITIC_FIT reference subset.
- Five Adam(3e-3) steps follow per critic on CRITIC_FIT minibatches.
- Critic kinds are A and B per view (v1, v2, pair), with deterministic init (`rgj.train.new_critic`, tag "init").
- Critics read [features, centred logits of the fixed warm-start head]; they never read a moving head.
- Local arms train the pair bank as a shadow bank with zero encoder weight.

**RNG (explicit numeric salt 0 for every arm).** This is the raw stage-B order of `rgj.train`, which equals smf stage A, the order U and the admitted RAW runs used.
- Task minibatches use `rng([seed, 0, ep])`; critic minibatches use `rng([seed, 0, ep, 7])`.
- Nothing else draws randomness. Receipts, snapshots and the equivalence check consume none.

**Surrogate and proxy.**
- Surrogate: R_v = (CE_const − min(CE_const, CE_A, CE_B)) / H on the encoder minibatch. The constant is included. This is not mutual information or a secrecy certificate.
- Joint: P = (R_income + R_occupation + R_pair)/3. Local: P = (R_income + R_occupation)/2, with no pair gradient.
- t_i and p_i are the task and proxy gradients on encoder i. Heads receive task gradients only.

**Update modes.**
- **RAW β:** q_i = β p_i, computed with the pinned `rgj.train` operations (coefficients β/3 joint, β/2 local; bitwise-identical updates).
- **NORM ρ, a:** q_i = stop_grad(ρ s_i ‖t_i‖/‖p_i‖) p_i.
  - Norms are float64, the scalar is capped at 100, and the zero threshold is 1e-12.
  - A zero task or proxy direction gives q_i = 0 and a logged reason; the nominal budget is never forced.
  - Allocation: s_income = a/√((a²+1)/2) and s_occupation = 1/√((a²+1)/2). Before caps and zero events, the RMS of ‖q_i‖/‖t_i‖ equals ρ.
  - The combined ratio ‖q_enc‖/‖t_enc‖ generally differs. Both are archived.
- **Common update:** u = [−t_enc − q, −t_heads], clipped to ‖u‖ ≤ 5, then θ += 0.05 u.

No dynamic feedback, refits, critic restarts, learned budgets or projections exist anywhere. The fixed allocation changes per-recipient update magnitude, not proxy coefficients.

**Release.** Recipient i receives [features_i, centred logits of its deployed head]. The deployed head (StandardScaler + LogisticRegression) is fitted on OSF_DEFENSE_FIT, with C in {0.01, 0.1, 1, 10, 100} chosen by HEAD_VALIDATION log loss. The procedure is identical for every arm. Probabilities and hard decisions are secondary formats.

## 5. Fixed bank (`osf.train.bank`)

| Family | Parameters | Treatment | Runs |
|---|---|---|---:|
| U | no privacy term | task only | 3 (admitted) |
| RAW | β 0.1, 0.3, 0.6 | joint and local | 18 (12 admitted, 6 new) |
| NORM symmetric | ρ 1.5, 3, 5; a = 1 | joint and local | 18 new |
| NORM fixed allocation | ρ 3, 5; a = 0.5 or 2 | joint and local | 24 new |

That is 63 logical continuations: 15 admitted and 48 new.

**Full or reduced bank.** The choice is made from timing alone (§6) before any bank fit, and is recorded in TRAINING_PROTOCOL_LOCK. The reduced bank drops ρ 5 and β 0.6 symmetrically. Weak-looking configurations are never removed afterwards.

**Incumbent.** The fixed RAW-J β 0.3, epoch 40, stays visible whatever wins.

**Nonfinite rule.** Nonfinite training allows one logged half-learning-rate retry. The original receipts are kept, and the rescued run is labelled. Poor utility, poor recovery and zero directions are outcomes, not retry conditions.

## 6. Engineering preflight under DATA_AND_ENGINEERING_LOCK

| Check | Units | Requirement |
|---|---|---|
| Parity | `parity__s{k}__*` | ρ = 0 and β = 0 (J and L, critics running) equal TASK bitwise from the same warm state, 4 epochs. |
| Raw fidelity | `fid__s{k}__RAW-{J,L}_b0.3` | 2-epoch osf RAW equals pinned `rgj.train` J-O/L-O bitwise (model and critics). Receipts reproduce rgj's logged norms. Synthetic fixtures run first. |
| Frozen-minibatch equivalence | in `fid__*` | On an identical real minibatch, r_i = ‖β p_i‖/‖t_i‖ in the norm expression reconstructs β p_i within max error ≤ 1e-4 ‖β p_i‖. Registered failures: a common ρ when r₁ ≠ r₂, a cap, t_i = 0. An algebra check, not an arm. |
| Instrumented replay | `run__s{k}__{U, RAW-*}` | The 12 admitted RAW runs and 3 U runs are replayed for 40 epochs with receipts. Epoch-20 and epoch-40 models (and RAW epoch-40 critics) must equal the admitted checkpoints bitwise. These are replays, not fits. |
| Timing | `timing__s0__NORM-J_r3_a1` | One logged 2-epoch calibration fit. |

At most two engineering amendments are allowed before a dependent nonzero stage, and only for a demonstrated numeric or implementation bug, a raw fidelity mismatch or a documented precision fix. Each amendment and its new lock are pushed **before** the affected work. `osf.lock.verify_lock` refuses to run a stage unless the lock and amendments are byte-identical on origin.

## 7. Instrumentation (STRENGTH_PROFILES.csv)

**Per-step receipts (`steps.npz`), for every raw and normalized run and both encoders:**
- t, p and q norms;
- realized q/t ratios (a zero step counts as 0), the combined and RMS versions, and conditional ratios (nonzero steps only, labelled as such);
- cos(t_i, p_i), the scale, cap and zero reason;
- pre- and post-clip total, encoder and head update norms, and the clip factor;
- R per active view and cumulative critic updates;
- epoch/step, and the minibatch and transform fingerprints;
- fixed-head, initial-state and critic-init hashes.

**Summaries:** mean, RMS, median, 10/90 percentiles, epoch profiles, and zero and clipping fractions.

**Aligned frozen snapshots** at progress fractions 0, 0.25, 0.5, 0.75, theta_{T−1} and theta_T:
- for the incumbent RAW-J β 0.3, symmetric NORM-J ρ 3, and their local twins;
- critic-gap diagnostics (`rgj.critic_track.compare`) on the same views and rows, comparing aligned critics at theta_{T−1} with the final theta_T;
- these never change a model or the assessment attack bank.

## 8. Inner-only selection (`osf/select.py`)

Selection uses OSF_DEFENSE_FIT, HEAD_VALIDATION and INNER_SELECTION only, and picks one global configuration per family across the three seeds.

**Task gates (point gates on every seed, each task, deployed heads on INNER_SELECTION; const = OSF_DEFENSE_FIT majority class):**
1. Acc ≥ Acc(U) − 0.01;
2. Acc − const ≥ 0.8 (Acc(U) − const);
3. Acc ≥ const + 0.03.

If U fails nontrivial utility, or a task lacks support, the status is INVALID.

**Controls (frozen first):**
1. **L\***: the lowest mean inner coalition AUC among task-feasible RAW-L and NORM-L configurations. Ties go to lower measured compute, then the lexicographic config ID.
2. **C\***: the lowest mean inner coalition AUC among task-feasible controls: all RAW-J and RAW-L, NORM-L, U, LEACE (E), FARE (F) and the compression tree (F0). It may be RAW-J, including β 0.6. No control is excluded for its local recovery or optimiser.

**Nominees (frozen next):**
- **N\*** (normalized joint): among NORM-J configurations that pass the gates and, on every seed, have local AUC ≤ L\*'s + 0.005 and ≤ C\*'s + 0.005 for both recipients, the lowest mean coalition AUC. Ties: lower ρ, a closest to 1, then config ID.
- **R\*** (ordinary online joint): among RAW-J configurations that pass the gates and have local AUC ≤ L\*'s + 0.005 on every seed for both recipients, the lowest mean coalition AUC. Ties: lower β. β 0.3 may be selected again. R\* is a predeclared benchmark nominee.

**No valid configuration.** The status is NO_FEASIBLE_NOMINEE. A descriptive fallback minimises the summed positive nomination shortfalls over seeds (native accuracy/AUC units), then mean coalition AUC, then the tie rules. Fallback scores never pass a claim. Missing comparators make the affected claims invalid.

**Deployable model.** The deployable best development model is fixed on inner selection before scoring. It is the model with the lowest mean inner coalition AUC among the valid N\*, R\* and C\* (ties: lexicographic config ID), whatever its family. No control is demoted for winning. EVALUATION_LOCK freezes tables, model IDs, heads, priors, attack slates, scalar grids, hashes and statuses before its push.

## 9. Primary family: 27 slots, three nine-clause conjunctions (`PRIMARY_FAMILY.json`)

| Claim | Nominee vs comparator | Question |
|---|---|---|
| A | N\* vs L\* | Does normalized joint design improve on the best local design? |
| B | N\* vs C\* | Does the normalized recipe improve on the strongest eligible existing control? |
| C | R\* vs L\* | Does ordinary online joint training improve on strong local design? |

**Clauses (each needs a valid nominee on every seed):**
1. Comparator pair AUC − nominee pair AUC: lower bound > 0.02.
2–3. Nominee local AUC − comparator local AUC, each recipient: upper bound < 0.01.
4–5. Nominee accuracy − U accuracy, each task: lower bound > −0.01.
6–7. Acc(nominee) − 0.8 Acc(U) − 0.2 Acc(const), each task: lower bound > 0.
8–9. Acc(nominee) − Acc(const), each task: lower bound > 0.03.

**Inference.**
- Paired per-seed statistics are averaged over seeds 0–2.
- Exact-record-group bootstrap: B = 1,999, seed 20261006, the same group draw across arms and seeds.
- z = Φ⁻¹(1 − 0.05/54) = 3.113017. Thresholds are strict; aliases are retained and the family never shrinks.

**Interpretation.**
- Intervals condition on the fitted models and the declared attacker slate. They do not cover refit variability or the adaptive history of these rows.
- A numerical pass means the fixed development criteria were met. It is not confirmation.

**Status labels.**
- NORM_DEVELOPMENT_CRITERION_MET: A and B both complete.
- RAW_JOINT_DEVELOPMENT_CRITERION_MET: C complete.
- Otherwise EXPERIMENTAL_NO_ADVANTAGE, or INCOMPLETE_OR_INVALID if required validity or coverage is missing.

Clause counts are not partial success.

## 10. Secondary ledger (`SECONDARY_FAMILY.json`, frozen before the assessment)

**Mandatory contrasts:**
- the incumbent vs U, and vs RAW-L β 0.3;
- symmetric NORM-J ρ 3 vs the incumbent;
- NORM-J ρ 1.5 vs 3 vs 5;
- NORM-J vs NORM-L at ρ 3, a = 1;
- allocations 0.5 and 2 vs a = 1 at ρ 3, joint and local;
- each nominee vs the incumbent.

Each contrast covers task and local/pair recovery, plus the attacker statistics: absolute recovery, proper loss, linear R², task utility, and pair minus the better local.

Its multiplicity count is frozen. Adjusted intervals support individual declared contrasts and are never an alternate route to primary success.

**Descriptive frontier.** Every locked final grid point is scored once, with the nominees and official references, across all seeds and both tasks. Infeasible points stay visible. It is never a post-assessment selection bank.

## 11. Locks and execution

| Lock | Governs |
|---|---|
| DATA_AND_ENGINEERING_LOCK | admit, parity, fidelity, replay, timing |
| TRAINING_PROTOCOL_LOCK | bank, references; carries the bank choice, the timing plan, `PREDICTIONS.json` and every pre-fit repair |
| SELECTION_AND_AUDIT_LOCK | inner, select, tracking; carries the attackers, families, inference and assessment code |
| EVALUATION_LOCK | the single assessment |

After the assessment, repairs are confined to identifiable code, numeric or reporting defects. The originals are preserved and the defect independently diagnosed. No new defense, grid, allocation, endpoint, attacker bank, candidate or claim is derived from the assessment.

**Ceilings:**
- 10 h elapsed (start 03:11:22Z; watchdog at 13:11Z stops only `osf` processes);
- 20 CPU-h, two heavy workers, 8 GiB, ≥ 5 GiB free, $0 cloud;
- the final two hours are reserved for the assessment, verification, packaging, backup and closeout.

## 12. Attribution and limits

Adversarial representation learning is established (Madras et al., ICML 2018, https://proceedings.mlr.press/v80/madras18a.html). A fixed coefficient/strength bank and norm rescaling are not algorithmic novelty. A winning comparison would establish behaviour on this benchmark, not novelty, a population privacy guarantee or confirmation. No confirmation population is spent in this run.
