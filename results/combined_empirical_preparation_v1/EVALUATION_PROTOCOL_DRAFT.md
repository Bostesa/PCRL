# Evaluation protocol: reviewable draft

**What this is.** A reviewable draft of the protocol, written 2026-10-01 for advisor review.

**What it is not:**
- It is **not** a preregistration.
- Nothing in it has been run.
- It does not certify historical outcomes as untouched.

**How to read it:**
- Items marked **[DECIDE]** need a decision from Dr. Yus or the authors.
- Items marked **[agreed direction]** come from the meeting's analysis-first instruction.
- Everything else is a proposed implementation choice.

## 0. Empirical question and relation to prior analyses

**Working question.** When do attribute-removal methods' *own stated protection tests* agree with
sensitive-attribute recovery by recipients who know the defense? Three conditions apply:
- task utility is measured under one held-out contract;
- recovery from task outputs is included;
- recovery from combining permitted releases is included.

This is a candidate organising question, not a novelty claim.

**What is already established.** See ANALYSIS_POSITIONING.md for details.
- Linear certificates do not bound nonlinear recovery. Sources: Ravfogel et al. ICML 2022 §5.1 and
  App. B.1, and the EMNLP 2022 kernel paper.
- Training-time adversaries are fooled while fresh attackers succeed (Elazar & Goldberg 2018; RLACE).
- Overlearned representations reveal attributes the task never needed (Song & Shmatikov 2020).
- Universal least privilege is incompatible with utility, and label-only "fundamental leakage" is
  measurable before training (Stadler et al. 2024 Thm 2 and §4.1).
- Noise on an isolated subspace is undone by averaging repeated draws (FairNVT v2, Table 3).
- Certification needs smoothing (Gitiaux & Rangwala 2021, Thm 2.1).
- Coalition budgets are formalised for finite alphabets (Taylor, Vippathalla & Coon 2026).

**What this protocol would add, if it adds anything:**
- A *controlled agreement measurement* between each method's own test and defense-aware recovery.
- One access contract and one held-out utility contract for every method.
- The output surface measured against label-only and clean-output references.
- A combination analysis of learned releases under a declared finite sensitive set.

The positioning document asks whether this is enough to matter.

## 1. Datasets and cohorts, with exposure status

From DATA_EXPOSURE_LEDGER.csv: all candidate data is **development** data unless stated otherwise.

| Role | Candidate | Exposure | Note |
|---|---|---|---|
| Development and reproduction | Adult; HMDA CA 2023; Diabetes 130 (deduplicated by patient); Folktables CA 2018 | **Spent** (tuning, selection and narrative depended on outcomes) | Use these for reproduction under historical definitions and for pilots of the repaired metrics. They are not confirmation data. |
| Development and replication | LSAC and Dutch census (durable-guarantees 20% holdouts) | Holdout rows apparently unread [UV]. Cells were selected from the same files. | Development replication only. |
| Vision | CelebA, with a new identity-disjoint split (`celeba_iddisjoint_v1`, PE-F30) | Official test rows 5,000+ may be untouched [UV]. Identities overlap across official partitions. | Requires retraining for the protected heads. |
| Language (optional) | Bias in Bios | Dev spent; test untouched by stored code [UV] | |
| Confirmation population | **[DECIDE]** ACS 2018 TX or NY (never opened by either repository) | Unused | Choose by an outcome-blind rule written before download: state, year, cohort and tasks. Do **not** acquire it in this phase. |

**Shared-cohort requirement for ACS.** If the tasks are income and commute time on the same
workers, use a single worker cohort where both labels have support (e.g. ACSIncome ∩ ACSTravelTime
eligibility). Report the counts at every filter. Do not use the low-income ACSPublicCoverage cohort for
income above $50,000.

**Linkage units.**
- ACS: household (SERIALNO).
- Diabetes: patient_nbr.
- CelebA: identity.
- Adult and HMDA: row (no linkage key available; state this).

All role splits are grouped by the linkage unit.

## 2. Declared sensitive set and purpose policies

[agreed direction: a finite declared set, deny-by-default] The specific table below is a proposal.

S (declared) = {sex, race, age band}. Any other attribute is out of scope; there is no promise about
undeclared attributes.

| Recipient | Prediction task | Permitted inputs | Allowed from S | Disallowed from S | Released | Coalition membership | Coalition policy |
|---|---|---|---|---|---|---|---|
| R1 (e.g. credit underwriting) | task A (e.g. income > 50k / loan decision) | X without S [DECIDE: also run the S-in-X regime as secondary] | ∅ | sex, race, age | representation Z₁ and/or task output ŷ₁ (object declared: label or score) | {R1, R2} | Coalition tests target only {sex, race, age} \ (allowed(R1) ∪ allowed(R2)) |
| R2 (e.g. marketing / commute service) | task B (e.g. commute time) | X without S | age band | sex, race | Z₂ and/or ŷ₂ | {R1, R2} | as above → target {sex, race} |
| R3 (e.g. fair-lending auditor) | audit task | X | race | sex, age | Z₃ | not in coalition with R1 [DECIDE: enforceable no-sharing assumption] | If R1 and R3 can share, race is authorised to the coalition, and measuring its "leak" is incoherent. |

**Rules:**
- Permissions are frozen before any recovery measurement.
- When two recipients have conflicting permissions, identical representations cannot give them
  different restrictions. This is the *specific* limitation of a shared representation.

## 3. Access models (executable)

Every reported number carries exactly one tag.

| Tag | Inputs to the recipient's attacker at **training** time | Inputs at **evaluation** time | Interface condition |
|---|---|---|---|
| A1 release | (release, S) pairs from the attacker-train role | the target's release (one draw) | always |
| A2 defense-aware | A1 + mechanism code + public parameters; may simulate the mechanism on its own clean-free data only if it has such data **under the same access** | the target's release (one draw) | always |
| A3(N) repeated query | as A2 | N releases of the target | only where fresh randomness is issued per query; persistent per-person tokens ⇒ N = 1 |
| A4 white-box population | A2 + clean (pre-protection) representations of the attacker-train role | the target's release (one draw) | stress test |
| A5 insider | — | the target's pre-protection representation | stress test; never used to refute a guarantee that excludes it |

Each method's state is enumerated in METHOD_CATALOG.csv (`exposes` column):
- public state;
- outputs and tokens;
- metadata;
- query interface.

## 4. Common data, supervision and utility contract

**Per seed, disjoint roles by linkage unit:**

| Role | Share |
|---|---|
| fit (representation, eraser, head) | 40% |
| attacker-train | 25% |
| validation (selection and calibration) | 15% |
| evaluation | 20% |

[DECIDE] the exact shares.

**Supervision recorded per arm:**
- task labels used;
- protected labels used at training;
- whether S is in X at deployment;
- external pretrained features.

No arm may use more labels or more evaluation feedback than its comparators.

**Utility reporting:**
- absolute task log-loss, accuracy/AUC, and calibration (ECE);
- normalised lift (clean − constant), with a bootstrap CI;
- the clean reference uses the same architecture;
- a minimum clean-lift rule [DECIDE, e.g. lift ≥ 0.03 and CI half-width < 0.25·lift]; cells below it are
  reported without ratios.

**Fixed-head versus refit-head:**
- **Fixed-head:** the deployed head, evaluated on the evaluation role.
- **Refit-head:** a common decoder (logistic regression, plus a small MLP) fit on the attacker-train
  role.

The two are reported separately.

**Head exploiting residual S:** report the head's accuracy with S-information removed (eraser applied to
the head input) versus not. A label-free proxy is not assumed safe.

**Selection budget:** an equal number of hyperparameter configurations per arm [DECIDE: 25], selected on
validation only.

## 5. Methods (first comparison)

See METHOD_CATALOG.csv and `notes/methods/PRIORITY_LIST.md`.

**Core arms (CPU):**
- **Anchors:**
  - a clean task-only release;
  - withhold/constant;
  - a label-only predictor of S.
- **LEACE** (`concept-erasure==0.2.4`), both per recipient and on the union of disallowed attributes.
- **SPLINCE** @fced0d3.
- **INLP** @e1edcc1.
- **Isotropic noise channel** (dg@956f5c8).
- **FARE** @89cb1b6.

**Extended arms:**
- DANN-style adversary, plus an official-LAFTR fidelity check on binary cells;
- VFAE (reimplementation);
- Obliviator @0f2233f;
- the PCRL erase layer, plus a no-LEACE arm.

**Method rules:**
- Binary-only methods: N/A on multiclass cells.
- Each method's own stated test is recorded alongside the common metrics.

## 6. Removal-test reproduction and repaired diagnostics

**Track H (historical).** Recompute the original statistics *as originally defined* from stored
artifacts:
- one-hot ridge R² (float32 as shipped);
- dominant axis;
- macro OvR AUC at 0.55;
- worst-pair.

**Track R (repaired), as METHODOLOGY_IMPROVEMENTS §4–5 specify:**
- float64 aggregate R²;
- per-class R² with support;
- held-out ρ₁² (top canonical correlation);
- nonlinear categorical recovery;
- not-estimable handling;
- the registered threshold grid;
- permutation nulls.

The two tracks are never mixed in a table.

## 7. Task outputs and multiple-recipient views

For each attribute target, recoverability is reported from:
- each recipient's own permitted view;
- the other recipient's view;
- the union of views;
- the task output alone;
- the representation plus the task output;
- a reference with the same allowed information (the label-only and clean-output references).

**Estimands:**
- Absolute recovery: the log-loss reduction over prior, and AUC.
- Combination increment: Δ = loss(best single view) − loss(union).
- Output increment: relative to the clean-output reference.

A negative fitted Δ is reported unclipped and never read as negative information.

**Combination effect, predefined.** Primary: Δ on jointly-disallowed attributes, paired over evaluation
units, with refit replicates.

## 8. Adaptive recipient construction, controls and budgets

**Attacker slate per data type:**
- Tabular:
  - multinomial logistic regression;
  - an MLP with 2 hidden layers and early stopping on validation;
  - gradient-boosted trees;
  - a method-aware attacker per access tag (A2 likelihood-ratio test for Gaussian channels, using the
    known σ; de-censoring à la Song & Shmatikov Alg. 1 for adversarially trained encoders [DECIDE]).
- Images: a linear probe and an MLP on frozen features. Attackers are refit for each defended
  representation.

**Budgets:**
- Equal tuning budget per attacker per arm [DECIDE: 20 configurations].
- Candidate selection on validation only.
- The locked slate is scored on evaluation.
- Identical releases receive identical candidates.

**Stochastic mechanisms:**
- Report the expected loss over draws (Monte Carlo, ≥ 32 draws) at N = 1.
- Report A3 curves for N ∈ {1, 4, 16, 64, 256} where the interface allows them.
- Token probabilities are never averaged into a "sampled token".

**Controls.** These must pass before any method arm is scored (fixture F7 shows the pattern):
- a directly included S (recovery ≈ 1);
- an XOR interaction (linear fails, trees and MLP succeed);
- a planted contrast missed by max-per-class (ρ₁² detects it);
- a null (S independent of release: AUC within the null CI);
- a constant extension (Δ CI covers 0);
- independent replay with a new seed.

If a "stronger" recipe recovers less on every control, it is not called stronger.

## 9. Selection and evaluation locks; inference

**Locks.**
- The policy table, threshold grid, slate, budgets and estimands are committed before any rescoring
  (lock commit).
- Outcomes on spent data remain development evidence.

**Units and pairing.**
- The sampling unit is the linkage unit (household, patient or identity), using cluster bootstrap.
- Seeds are refit replicates over shared evaluation people: variance combines between-refit and
  cluster components.
- Losses are paired by person.

**Estimands.**
- Unweighted and person-weighted (PWGTP) estimands, both reported for ACS.
- Primary: combination increment Δ and agreement rate.

**Hypothesis families.**
- [DECIDE] Primary hypotheses (≤ 3), for example:
  - H1: each method's own test passes while A2 recovery exceeds the margin;
  - H2: output-surface increment over the clean-output reference;
  - H3: Δ > margin on jointly-disallowed attributes.
- Secondary families use Holm correction.
- Simultaneous intervals cover only the reported family.

**Decisions and margins.**
- Intersection-union decisions are reported as such. "k of m clauses passed" is not partial
  confirmation.
- Margins: [DECIDE] AUC margin 0.02; log-loss margin 0.003 nats (historical ACS value).

**Support and missing cases.**
- Minimum class support n_min = 100 per role [DECIDE].
- Unsupported, infeasible and not-applicable cases are counted and listed, never imputed.

## 10. Compute, storage and artifacts

**Compute.**
- Core arms on tabular cells: CPU-only, roughly < 30 CPU-hours total [estimate, not measured].
- Extended arms with the PCRL erase layer: about 2–3 GPU-hours per dataset (methods role estimate).
- CelebA: a GPU is required for feature extraction once per backbone.

**Storage.**
- Per-person score arrays, compressed.
- Manifests with SHA-256 for every array.
- Archive to the non-expiring bucket, never to the 7-day-lifecycle bucket.

**Artifacts.** Configuration, split manifests, slate manifest, lock commit, per-seed raw metrics,
recount scripts.

## 11. Smallest pilot that would reveal a methodological failure

1. Run controls F7, extended with the A2/A3/A4 tags, plus the repaired diagnostics on synthetic data.
   **Done in part this phase.**
2. On **one** spent development cell (Adult sex/income), with S excluded from X, run:
   - anchors;
   - per-recipient LEACE;
   - the noise channel.

   Under the full split contract, check:
   - (a) held-out utility versus in-sample;
   - (b) A2 versus A4 likelihood-ratio test;
   - (c) the output surface against the label-only reference;
   - (d) the nested-slate assertion.

   About 1 CPU-hour. **Requires approval; not run.**
3. Stop if any control fails, or if (a)–(d) shows the contract cannot be implemented as written.

## 12. Proposed order after advisor review

1. Reproduce existing artifacts under their original definitions (Track H; mostly done as recounts,
   see INDEPENDENT_CHECKS.json).
2. Repair measurement and access (METHODOLOGY_IMPROVEMENTS §1–7).
3. Run the cheap controls and the one-cell pilot.
4. Run the controlled baseline benchmark, core arms, on the development cells.
5. Run the multi-recipient analysis, only where the policy table creates jointly-disallowed targets.
6. Choose and lock the confirmation population, then evaluate once.
7. Use the gap found to specify method work. The joint-purpose algorithm prompt is kept as a future
   proposal; it is not on this machine.

## Unresolved design decisions (summary)

- **[DECIDE]** S-in-X regime as primary or secondary.
- **[DECIDE]** Role split shares, selection budgets, minimum lift, n_min and margins.
- **[DECIDE]** Confirmation population (TX vs NY ACS 2018 or another year) and its outcome-blind
  selection rule.
- **[DECIDE]** Whether R3 (auditor) is modelled as non-colluding with an enforceable assumption.
- **[DECIDE]** Whether de-censoring (Song & Shmatikov) is part of A2.
- **[DECIDE]** One paper versus two. Deferred until the outline and the evidence map are reviewed.
- Official review text is unavailable. The threshold range, worst-class wording and missing-method list
  should be checked against it.
