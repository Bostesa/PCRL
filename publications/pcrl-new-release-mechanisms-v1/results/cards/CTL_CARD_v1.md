# CTL_CARD_v1: generic challenger controls (code `nrm/ctl`, commit 5cfe2fb5d)

Author: challenger, 2026-10-20 (about 15:20Z).

- **Status.** This is a generic card. It covers every control fit made with `nrm.ctl` on a named family's task-only twin.
- **Family-specific revision.** A v2 card (`CTL_CARD_v2_<FAMILY>.md`) must be committed before the first control fit for a
  family. It names:
  - the twin release (family, config_id, both versions);
  - the controls;
  - the levels to save;
  - the configuration ids.
- **Margin m.** It comes from the statistical reviewer's `nrm/rules.py` once that file is committed.

## 1. Operation

Every control maps the named twin's emitted vectors to new vectors:

- inputs: income (n, 2) and occupation (n, 5) for all six roles in the admitted row order, computed from the six permitted
  fields only;
- outputs: income (n, 2) and occupation (n, 5); decisions are the argmax of the emitted vectors;
- nothing else is transmitted.

Strengths are the module constants of `nrm/ctl/post.py` and are predeclared.

**C2 BLIND: SEX-blind coarsening** (`Coarsener('blind')` per task; `JointCoarsener('blind')` for candidates that couple both
outputs).
- K0 = 256 label-free initial cells: quantile bins of income p1; KL k-means for occupation; joint KL k-means for the joint
  variant.
- Greedy merges by the smallest plug-in task-information loss.
- Snapshots at these levels:
  - income: 64, 32, 16, 8, 6, 4, 2;
  - occupation: 128, 64, 32, 16, 8, 5;
  - joint: 128, 64, 32, 16, 10.
- The released vector is the cell posterior.
- Variants coarsen only the recipient(s) the candidate changes.

**C3 FLAT: flattening.**
- Temperature T ∈ {1.25, 1.5, 2, 3}, or label smoothing ε ∈ {0.05, 0.1, 0.2, 0.3}. Both are data-free.
- Calibration-matched variant, for families that change calibration (decoders, Platt maps, transplants, mean-vector
  constructions): one temperature per task, fitted on CALIBRATION by NLL.

**C4 AWARE / SHUF / RMERGE: post-hoc grouping.** All three use the same cells and levels as C2.
- AWARE: merge score = task-information loss − λ·(SEX information removed), λ ∈ {0.5, 1, 2}, using MODEL_FIT SEX only.
- SHUF: AWARE with SEX permuted (seed + 7919).
- RMERGE: random pair merges (seeded).

**C5 RAND: random replacement without task adaptation** (`replace_random`).
- With probability q ∈ {0.05, 0.1, 0.2, 0.4}, a person's vector is replaced by the twin's MODEL_FIT vector of a uniformly
  random MODEL_FIT person.
- `same_class=True` draws only among bank vectors with the same argmax, so every decision is kept.
- Channel law: as stated.
- Noise coupling: the two recipients draw independently. A shared-coin variant is used only if the candidate couples its
  recipients, and is recorded.
- Seed handling: numpy `default_rng` with seed = SHA-256 of (config_id, version, role, recipient), truncated to 64 bits and
  recorded.
- R = 8 one-draw-per-person repetitions, stored as (8, n, k).

**C9 REM: decision- and top-probability-preserving remainder flattening** (`flatten_remainder`).
- The non-argmax coordinates move toward SEX-blind class shares R[argmax] with weight min(s, cap), s ∈ {0.25, 0.5, 0.75, 1}.
- The shares come from `fit_remainder_shares`: MODEL_FIT out-of-fold twin vectors with labels, or CALIBRATION, or label-free.
- Occupation only; it is the identity for income.
- It is deterministic and many-to-one in the low-probability log-odds.

## 2. Closest prior PCRL attempt, and what is different

Closest attempts:
- `pcrl_scm_utility_matched_v1` ORD path. It used the TWIN rule (β = 1 from 512 cells, on MODEL_FIT out-of-fold statistics),
  but read SEX through the within-SEX guard and never reached SCM's DEV spend.
- `pcrl_heldout_calibration_v1` temperature calibrators.
- The frozen-bank channels of `pcrl_latent_and_sampled_methods_v1`.

What is different:
- Strictly SEX-blind paths: no SEX in fitting, guards, posterior estimation or level choice.
- Predeclared ladders.
- Saved levels bracket the candidate's measured spend.
- Comparisons use frontier verdicts (`nrm.ctl.frontier`), not one cost-matched point.

Bottlenecks addressed:
- benefit not separated from task spending (SCM);
- reader artifacts from relabelings (decoders, temperature);
- randomness credited without a SEX-blind or deterministic counterpart.

## 3. Why

These are controls, not candidates. A control explains a candidate's benefit when a realized control point reaches the
candidate's recovery at no more task spending. No protection claim is made for any control.

## 4. Task-only twin

The controls use the named family's TWIN: same architecture, capacity, data, training size, folds, seeds, epochs, tuning
and calibration, with every SEX term set to zero. Controls never refit the twin; they only post-process its emitted vectors.

## 5. Simplest controls and published objectives

- SEX-blind: C2, C3, C5, C9.
- SEX-aware post-hoc: C4 AWARE. Its objective is the agglomerative information bottleneck with a privacy-funnel / IBSI-type
  Lagrangian.

## 6. Falsifying outcome

**For each control's role:**
- (i) A BLIND path whose measured spend is not non-decreasing in coarseness is diagnosed before use. It is never repaired by
  changing knobs.
- (ii) FLAT must leave LOOKUP and decision readers unchanged, because they are invariant by construction (up to float
  rounding). If not, the reader pipeline is investigated.

**For the comparison:** if the candidate is CANDIDATE_AHEAD in both versions, on the primary axis and on every secondary
axis, against every applicable control, then the controls do not explain it. This is development evidence only.

## 7. Fitting plan

**Roles and labels:**
- Partition: the twin's deployed MODEL_FIT vectors, label-free.
- Merges: MODEL_FIT task labels (income, occupation), using the twin's MODEL_FIT out-of-fold vectors when the family
  cross-fits, otherwise its in-sample vectors (recorded).
- SEX (MODEL_FIT): only for C4 AWARE and SHUF.
- Cell posteriors, temperatures and remainder shares: CALIBRATION income and occupation labels (calibration parameters only;
  levels ≤ 32 when posteriors come from CALIBRATION), or MODEL_FIT out-of-fold labels.
- Never MODEL_SELECT labels. MODEL_SELECT enters only through `nrm.evaluate`, `nrm.screen` and `nrm.devreaders` outputs.

**Folds and sizes:**
- Full pool, with no internal folds beyond the twin's own cross-fitting.
- Training sizes are the twin's; controls add only post-processing parameters.

**Calibration role:** CALIBRATION.

**Seeds per version:** control seed = 1000·v + control index, recorded.

**Level and strength selection, by spend only, never by recovery:**
- Predict each level's spend from the MODEL_FIT plug-in information loss (`record()['plugin_task_mi_nats']`) or from
  CALIBRATION utility.
- Save the two levels that bracket the candidate's MODEL_SELECT-measured spend (`nrm.evaluate`), plus at most one stronger
  level.
- If `nrm.ctl.frontier.compare` returns EXTEND_CONTROL or AHEAD_VS_SPARSE_PATH, save the next level as a logged adaptive
  step.

**Entry point:** `nrm/ctl/run.py`, committed with the v2 card.
- It loads the twin release from runs/.
- It reads labels only through `nrm.common.load(..., operation='fit' or 'calibrate', card=<this card or v2>)`.
- It saves through `nrm.release.save` with kind 'control' and records the `Coarsener.record()` fields and roles.

## 8. Readers

- The coordinator's formal grid (16 recipes, including LOOKUP and DEC_LR) and the screening split, with the same splits,
  cells and seeds as the candidate.
- Finite-cell controls (C2, C4): report the reader budget (median FIT people per released vector and per joint value) beside
  every AUC.
- C5 RAND: one draw per person, from one repetition per cell.

## 9. Budget

- Each saved (control, level, version) is one configuration of the 60 (coordinator ruling on Q1). This generic card
  requests none.
- Proposed v2 requests:
  - family A: +2 (SHUF_S2, both versions), +2 contingent ORD bracketing;
  - family B: +4 (REM and BLIND-occupation on FB_TWIN, one bracketing level each, both versions).
- CPU: under 60 CPU-s of fitting per family (8 coarsener fits at n = 20,000 took 2.4 CPU-s on invented data), plus grid
  readers.

## 10. Revision plan and verdict rules

**Revisions:**
- Each family's v2 card names the twin, controls, levels and configuration ids before its first control fit.
- A control fit that fails a check (non-finite values, invalid vectors, monotonicity) is diagnosed and refit under a new
  configuration id. No knob values are added.

**Verdict rules:**
- Use `nrm.ctl.frontier.compare` and `compare_axes`, on the primary axis (maximum normalized clause usage over the version's
  12 clauses and 2 guards) and the secondary axes (per-task NLL, men's occupation Brier).
- Tolerance tol = 0.10 usage units on the primary axis, 0.001 nats on the NLL axes and 0.0005 on the Brier axes.
- Margin m comes from `nrm/rules.py`.
- A candidate is credited only with CANDIDATE_AHEAD in both versions; CONTROL_EXPLAINS on any axis in either version means
  the control explains it.
- The 0.020 material target against NEURAL_RAW and H_LJ is reported separately.
