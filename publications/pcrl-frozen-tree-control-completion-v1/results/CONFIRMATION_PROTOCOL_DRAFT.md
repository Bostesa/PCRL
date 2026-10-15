# Confirmation protocol draft (NOT approved; fresh execution is disabled)

**Status:** `DRAFT_NOT_APPROVED_FOR_FRESH_EXECUTION`. This draft prepares a test; it does not authorize opening any new labels.
- **Machine-readable bindings:** `confirmation/CONFIRMATION_PROTOCOL_PUBLIC.json` (public copy; private paths scrubbed).
- **Runner:** `tree_completion/confirmation_runner.py`.

## 1. Two estimands; only one is implemented

| estimand | what is tested | what stays frozen | what is refitted | runner |
|---|---|---|---|---|
| **FROZEN-RELEASE CONFIRMATION** | the exact L_TREE seed-0/1 packages vs frozen comparators on new, compatible (Adult raw83) people | L_TREE packages (teachers, leaf corrections, routing, class order, float64 inference); RAW_PROJ (checkpoint, geometry, adapter binding); H_SEQ21_CODE (U teacher + registered policy, mean decoder); U0/Ucal reference models; reader procedure; clauses, caps, margins and α | only the attacker readers, on new attacker-fit groups, with winners chosen on new attacker-validation groups | implemented; development and synthetic rehearsals pass |
| **RECIPE REPLICATION** | the learning recipe refit on another population with separate training, selection, attacker and assessment roles | registries, selection rules, recipe hyperparameters and decision rule | teachers, N_EQ, leaf corrections, comparators, readers | **not implemented.** If the schema, labels or baselines change (e.g. ACS/Folktables), it becomes a new transported benchmark, not confirmation of the Adult releases or of their 24 numerical comparisons |

## 2. Claims (choose before any execution; B is not passage of A)

- **A: ORIGINAL criterion.**
  - 24 utility clauses: accuracy loss ≤ 0.010, NLL excess ≤ 0.010, summed Brier excess ≤ 0.005, vs U0 and Ucal, both tasks and seeds.
  - Coalition benefit > 0.020 vs every required comparator: RAW_PROJ, N_EQ, T_TREE, H_JOINT_CODE, H_SEQ21_CODE, H_DPC_JOINT, plus RAW_J_NEQ under the completion rule.
  - Recipient excess ≤ 0.010 for each.

  Meaning: the release is competitively protective against the strongest available references, including the historical score codes.
- **B: NEW, narrower prospective claim.**
  - The same 24 utility clauses.
  - Coalition benefit > 0.020 vs the incumbent RAW_PROJ.
  - Superiority > 0 vs the strongest utility-feasible code control (H_SEQ21_CODE).
  - Recipient excess ≤ 0.010 vs both.

  Meaning: a useful, materially better release than the incumbent neural service, and better than the best code control by some positive margin. It does **not** establish the 0.020 competitive margin over the codes. **Disclosure:** the choice of H_SEQ21_CODE as the strongest utility-feasible code (and H_DPC_JOINT's exclusion from claim B) used exposed INNER outcomes.
- **Relationship:** A implies B. If both are ever tested, use the fixed sequence B, then A, each at the full α (nested conjunctions; no α split is needed). Do not switch the project's primary claim to B because it is cheaper.

## 3. Decision rule (prospective; historical results unchanged)

- **Primary: one intersection-union test (IUT) of the single pre-specified global claim.** Each clause is tested one-sided at α = 0.025 with q = t(0.975, G−1) and group-level influence-function SEs. The claim is CONFIRMED iff every one-sided bound lies on the passing side: utility/excess upper ≤ cap, benefit lower > 0.020, superiority lower > 0.
  - The false-confirmation probability is ≤ 0.025 under any dependence among clauses (IUT theorem; FDA 2022 multiple-endpoints guidance on co-primary endpoints). See `STATISTICAL_DESIGN_IUT.md`.
  - The per-clause bounds are marginal. They are not simultaneous intervals; their joint coverage at 30 clauses is about 0.975^30 ≈ 0.47 under independence.
- **Sensitivity: the original two-sided Student-t Bonferroni simultaneous bounds** over the clause family (K = 30 for B, 42–45 for A, and the original K = 78).
- **What the rule does not license:**
  - Selecting among candidates, seeds, populations or claims on confirmation data invalidates the IUT size.
  - Adding model seeds on the same people adds no evidence.
  - The independent sample size is the number of duplicate groups.
- **Choice of α:** chosen from convention (one-sided 0.025 corresponds to the two-sided 0.05 standard). It was not tuned to the sample size; α = 0.05 would cut the required size by only about 20%.

## 4. Roles and order of operations (enforced by the runner)

1. **Locks:** protocol hash; package, comparator and reference hashes; the 83-name schema lock.
2. **Role separation:** attacker groups (fit + validation, frozen salted split) and assessment groups must be disjoint in rows and duplicate groups, and disjoint from every previously used role (verified against a usage manifest).
3. **Releases:** produced from frozen services only (probabilities + decision).
4. **Readers:** fitted on attacker-fit groups; winners chosen on attacker validation; `READER_FREEZE.json` is written **before** any assessment label is read.
5. **Assessment:** utility clause statistics vs U0/Ucal; coalition and recipient AUCs of the frozen winners.
6. **Inference and report:** IUT plus simultaneous bounds; a write-once report directory with a SHA-256 manifest.

## 5. Preconditions before any fresh execution (all currently unmet)

- **(a) A compatible, verified-unexposed population of the required size** (`POPULATION_COMPATIBILITY_AND_USAGE.csv`).
- **(b) Runnable, hash-bound U0/Ucal reference packages for new people.** The utility-reference-repair study restored U0/Ucal inference bitwise; that loader must be bound into the protocol.
- **(c) An owner authorization file** whose sha256 is recorded in an approved protocol (status `APPROVED_FOR_FRESH_EXECUTION`).
- **(d) A new lock receipt** pushed to the external GitHub remote before any label opens.
- **(e) Removal of the hard-disable** in a reviewed commit.

## 6. Required size (from `POWER_AND_FEASIBILITY.md`)

Under the observed effects, claim B needs about **317,000 independent evaluation groups** for 80% joint power under the IUT (Monte Carlo 95% 312k–321k; about 423,000 for 90%), and about 641,000 under the Bonferroni design. Planning should round these up. The binding clause is seed-1 occupation summed Brier vs U0 (observed excess 0.00422 against the 0.005 cap). The privacy clauses alone need only about 4,000 groups.

Claim A is not achievable at any size unless the true benefit over the historical codes exceeds 0.020. At 0.025 or more, it needs the same roughly 319,000 because utility binds.

Sensitivity:
- If the true occupation Brier excess were 0.001 lower, about 77,000 groups (IUT).
- If 0.0005 higher, about 2.5 million.
- If above the cap, not achievable.

The binding Brier excess is concentrated in the minority SEX class. Under the label-shift model, once the evaluation population's SEX minority share exceeds about 0.36 the implied true excess exceeds the cap, and claim B becomes not achievable (at 0.45 the implied excess is about 0.0068). At a minority share of 0.25, about 111,000 groups would be needed.

## 7. Populations (from `POPULATION_AUDIT.md`)

- **Frozen-release confirmation: no compatible, unexposed population identified.** All 48,842 Adult records are accounted for:
  - 39,170 used row space;
  - 6,017 PCRL validation split, spent on earlier checkpoint selection;
  - 3,620 missing-value rows outside the raw83 contract;
  - 35 duplicates.

  No Adult rows are formally reserved.
- **Recipe replication: only ACS, as a transported benchmark.**
  - Schema crosswalks are needed. ACS lacks capital gain/loss, its spouse code is sex-neutral, occupation must be mapped to the 6 groups, and the income threshold needs a decision.
  - California 2016–2018 is spent.
  - TX 2018 is reserved as another programme's primary confirmation cohort; NY 2018 is only that plan's conditional fallback, not a reservation.
  - Other 2018 states and 2019 or 2021–2023 show no use in the searched records, so their status is otherwise UNKNOWN.
  - No counts are known without a registered, label-blind admission.
- **Any ACS run tests a different estimand.** It is a recipe replication with transported definitions; it cannot reproduce the frozen Adult releases or their 24 numerical comparisons.
