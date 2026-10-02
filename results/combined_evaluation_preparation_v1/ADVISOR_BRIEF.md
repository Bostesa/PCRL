# Advisor brief: the repaired evaluation, ready to run

Written 2026-10-02. This is development work on data that has already been used. Nothing scientific was fitted
in this preparation.

## Empirical question

When a release passes a method's own protection check, what can recipients still recover under explicitly stated
access conditions, and how much task performance does the release keep?

The evaluation reports five things separately:

1. **Fitting-sample compliance.** Does the method pass its own check on its own rows?
2. **Held-out generalisation of that same quantity.**
3. **Recovery outside the guarantee's scope.** For example, nonlinear attackers against a linear guarantee.
4. **Leakage through prediction outputs**, measured against a label-only reference.
5. **Recovery from combined releases.**

Each outcome is labelled with one category:

| Category | Meaning |
|---|---|
| C1 | Fails its own check |
| C2 | Passes the check, but the result does not hold on held-out rows |
| C3 | Recovered by an attacker or release surface outside the guarantee's scope |
| C4 | A stated population guarantee is contradicted under its own assumptions |
| C5 | Not estimable |

A nonlinear attack is never counted as a broken linear theorem.

## Protection scopes

GUARANTEE_CARDS.md has 16 cards.

- **LEACE and PCRL's linear check.** Both bound linear (least-squares) prediction on the fitting rows, within a
  numerical tolerance.
- **The Gaussian-noise guarantee.** It covers only the clipped channel, one release, and a frozen map.
- **FARE.** It bounds demographic parity, not attack AUC.

Historical failures re-classified so far: **4 C1, 10 C3, 1 C4, 3 C5**. The single C4 is the refuted R²→accuracy
bound.

## Pilot panel

Selected by a stated completeness rule, not by historical effect size.

**CELL-A: Adult on the PCRL Round-4 seed-0 encoder** (sha256 1cfc2fef…). This is the exact encoder the AAAI audit
attacked.

| Item | Detail |
|---|---|
| Untreated releases | All 8 disallowed (purpose, attribute) pairs |
| Noise releases | σ ∈ {0.25, 0.5, 1, 2, 4, 8} × 3 seeds on income/sex (the AAAI "hard" cell) |
| Projection arms | Recount only. The projection matrices were never saved, and refitting them is not authorized. |
| Rows | PCRL test split (the encoder never trained on it): 7,571 attacker-fit, 2,239 attacker-validation, 5,250 assessment |

**CELL-B: HMDA race/loan decision.** Correction (2026-10-02): it fails eligibility rule 4 (no documented projection arm on this encoder) and is not part of the CELL-A run. Preparing it would be extension E1.

## Access table

ATTACKER_ACCESS_TABLE.csv has 16 recipes. Each states:

- whether the attacker knows the defense procedure and its fitted parameters;
- whether it has a separate labelled population, paired clean/released examples, or the target's own clean input;
- whether it sees outputs and/or representations;
- whether it sees gradients;
- whether it gets one release, repeated fresh releases, or a persistent token.

The noise-specific likelihood-ratio attacker is kept separate from the general adaptive attacker.

## Thresholds, support and inference

- **AUC bars.** 0.52, 0.55 and 0.60. The 0.55 bar is the historical one.
- **Decision rule.** The verdict comes from a one-sided 95 % bound (a two-sided 90 % cluster bootstrap, B = 2,000):
  - upper bound below the bar: established below;
  - lower bound above the bar: established above;
  - otherwise unresolved. Failing to establish recovery is not proof of its absence.
- **Support.** A class needs at least 100 rows in attacker-fit and in assessment. Otherwise it is "not estimable".
  For Adult race, 2 of 5 classes are not estimable, and no classes are pooled after the fact.
- **Metrics.**
  - Native check (kept separate).
  - Held-out linear R² (R02, scale-invariant).
  - Linear and nonlinear AUC on each surface.
  - Macro AUC over supported classes.
  - Supported worst class and worst pair, each with its coverage.
  - Held-out top canonical correlation as the contrast diagnostic.
- **Seeds.** Seed variation is reported apart from sampling uncertainty.

## Utility available now

A **frozen-head** comparison is possible on the common held-out rows. It only needs the forward pass that is
already done.

A **newly fitted utility probe** on a frozen representation is prepared, but not run, because fitting a probe needs
the execution flag.

## What the pilot can establish

**It can establish:**
- The surface / metric / attacker decomposition for one real encoder.
- Whether the native linear check generalises to held-out rows (C2 vs C1).
- How much is recovered outside the linear scope (C3).
- The output leakage relative to the label-only reference.
- How noise level trades recovery against utility on one cell.

**It cannot establish:**
- The full 59/67 breakdown across all audit cells.
- Any relationship across heterogeneous tasks, such as label coupling against cost.
- The NeurIPS headline models, which are Round 5/7.

Those need the extension matrix: 27 groups, 376 units, of which 183 are missing interfaces.

## Completed earlier vs measured now

**Earlier results (unchanged):**
- The AAAI audit counts are 64/59/51 of 67 failing at the three bars. The 67 rows include repeats, so counted per
  distinct measurement it is **62/57/51 of 64**.
- NeurIPS: 56/60 at the final checkpoint and 54/60 at the best checkpoint.

**Measured in this preparation:**
- Held-out logistic regression on the NeurIPS models: 4/60 pairs exceed majority by more than 1 pp. This
  overturns the earlier statement that no held-out linear probe had ever been run.
- The native ridge R² depends on representation scale.
- The frozen forward pass, and admission of all 152 pilot manifests.
- The backup of 102 laptop-only files (including 18 single-copy checkpoints), verified on the drive.

## Execution decision requested

Run CELL-A: `EXECUTE=1 sh results/combined_evaluation_preparation_v1/notes/evaluator/run_pilot_cell_a.sh`.

- About 1 CPU-hour; at most 2 hours with margin.
- Laptop only: no cloud, and no new data.

Before it runs, the coordinator decisions in PILOT_PROTOCOL.md §18 are written to the lock file, so the fixed
choices precede any scientific fit.
