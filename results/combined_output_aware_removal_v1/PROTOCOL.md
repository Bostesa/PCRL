# Protocol: output-aware removal on frozen PCRL encoders

**Frozen 2026-10-03, before any new defense, head or attacker fit.** The machine-readable version is
`EXECUTION_LOCK.json`, pushed before execution.

**Data status.** All rows are previously used development data. A newly frozen assessment procedure does not make an
already-opened population fresh. Texas, New York, other cohorts and unused years remain unopened.

## Cells, encoders, roles

**Mandatory cells.** Each uses the frozen PCRL Round-4 encoders, seeds {0, 1, 2}, and that purpose's
representation `rep_p0`:

| Dataset | Purpose | Task | Protected attribute | Supported classes |
|---|---|---|---|---|
| Adult | income_prediction | income | sex | {0, 1} |
| HMDA | underwriting | loan_decision | race | {0, 1, 2}; classes 3 and 4 are below 100 assessment rows and not estimable |

The encoders are never retrained.

**Roles (`oar-roles-v1`, see `ROLES_AND_SUPPORT.json`).** These are the benchmark roles with two changes, both made
from keys and roles only:
1. Every test-split record that also occurs in the encoder-training split is removed from all scored roles. That is
   17 Adult and 42 HMDA rows.
2. 20 % of the remaining attacker_fit groups, chosen by hash `oar-cert-v1|<ds>|canon_key`, become the FARE
   **certification** set. The certification set never tunes a defense, never chooses a certificate, and never trains
   an attacker.

| Role | Adult rows | HMDA rows |
|---|---|---|
| defense_fit | 24,127 | 63,704 |
| cert | 1,500 | 1,385 |
| attacker_fit | 6,065 | 5,388 |
| attacker_val | 2,235 | 2,082 |
| assessment | 5,243 | 4,764 |

The assessment membership is one and the same for every method, seed and view in a cell.

**Differences from the old benchmark:**
- exposure groups removed;
- the certification carve-out (smaller attacker_fit);
- release heads use a hash holdout (`oar-head-v1`) inside defense_fit.

**Support.** 100 / 30 / 100 per class on attacker_fit / attacker_val / assessment. Sparse classes are never pooled.

## Methods and access views

These are named exactly in `RELEASE_CONTRACTS.md`.

| Arm | Defense | Notes |
|---|---|---|
| A | Untreated | |
| B | Official target-only LEACE | The benchmark's pinned maps, fitted on defense_fit; not refitted |
| C | Official policy-set LEACE | Descriptive |
| D\* | Gaussian noise at σ\* | Adult 2.0, HMDA 4.0, frozen from benchmark validation. 3 persistent draws, identical to the benchmark draws. The missing noise grid is not expanded. |
| F\* | Official FARE (eth-sri/fare @89cb1b66ed268c16659cbf7428c43e60da2df641) | Applied to the stored encoder features and fitted on defense_fit (task label and protected attribute used for fitting only). Released features = one-hot cell index. |
| FZ | The same tree/output budget as the nominee, with fairness weight γ = 0 | Compression and task-tree control |

**References.** Constant prior; the true task label (label-only attacker), as a diagnostic only.

**FARE grid.** Frozen, shared by both datasets: k̄ / n_min / γ =
- 100 / 100 / 0.3;
- 50 / 100 / 0.7;
- 20 / 100 / 0.85;
- 10 / 1000 / 0.9;
- 5 / 1000 / 0.95;
- 3 / 1000 / 0.95.

The criterion is `fair_gini_dp`, and γ is capped at 0.95 because of the HMDA γ ceiling (`notes/fare`). A configuration
whose encoder yields cells identical to an earlier configuration's is an **alias**, counted and not refitted.

**FARE nominee, chosen on validation only:**
- **Admissible:** U2 attacker_val accuracy ≥ untreated U2 attacker_val accuracy − 0.01.
- **Among admissible:** the lowest attacker_val macro AUC of the base NL (attacker seed 0) on the FARE features.
- **Ties:** the lower grid id.
- **None admissible:** there is no admissible nominee. The highest-validation-accuracy configuration is used as a
  descriptive fallback, and that cell **cannot** pass the competitive conjunction.

**Views:**
1. features alone;
2. features plus the historical clean logits;
3. features plus the logits of a release head fitted only on those features.

The view-3 head is an LR grid with the U2 budget, fitted on defense_fit minus the `oar-head-v1` holdout and selected
on that holdout. The same head family is used for A, B, F\* and FZ.

**Output surfaces:**
- `O_full`: the historical full logit vector;
- `O_prob`: softmax;
- `O_hard`: the frozen head's argmax, one-hot;
- references: label-only and constant.

These are evaluated alone, and with the noisy features at σ\*.

## Attackers

**Base slate.** This is the benchmark slate, unchanged:
- L, 5 C values;
- GBT, 20 configurations;
- MLP, 18 configurations.

Selection works as follows:
- NL = GBT vs MLP, chosen on attacker_val log loss;
- the selected recipe is refitted at attacker seeds {0, 1, 2};
- plus-surfaces add ignore-rep and ignore-outputs candidates, through the same validation selection.

Every fit uses attacker_fit, every selection uses attacker_val, and assessment is only scored.

**Defense-aware attacker.**
- For finite surfaces (FARE cells, hard outputs): a cell-conditional P(s | cell) predictor, with Dirichlet smoothing
  α ∈ {0.1, 1, 10, 100} toward the attacker_fit prior, selected on attacker_val.
- Mixed releases: the base slate on the joint input.

It is reported beside the base slate. It is called "stronger" only if its measured assessment AUC is higher.

**Real-data controls.** Run on attacker_fit / attacker_val rows, seed 0, once per new interface type:
- **Shuffled-label null:** flagged if validation AUC > 0.55.
- **Planted-leak positive control:** a noisy copy of s is appended; it must be detected above 0.75.

**Utility:**
- U2 = the benchmark LR probe on the released features (attacker_fit / attacker_val), scored on assessment accuracy.
- The view-3 head's assessment accuracy and log loss are reported as release utility.
- The FARE tree's own task accuracy is a compatibility diagnostic only.
- **Exact check:** the frozen head's full output and its argmax have identical accuracy (the same decision). This
  says nothing about log loss, calibration or ranking.

## Primary family

The family is the literal table in `oar/family.py` (PRIMARY_FAMILY.csv mirrors it). It has **12 rows**, enforced by
an assertion.

**Recovery quantity R.** The supported-class macro one-vs-rest AUC of the base NL, averaged over attacker seeds, then
release seeds, then encoder seeds.

| # | Question | Δ | Pass condition |
|---|---|---|---|
| P1 | Hard predictions lower outputs-only recovery | R(O_full) − R(O_hard) | > 0.02 |
| P2 | FARE features lower recovery vs target LEACE | R(B, rep) − R(F\*, rep) | > 0.02 |
| P3 | FARE accuracy non-inferior to untreated | Acc(F\*) − Acc(A) | > −0.01 |
| P4 | FARE accuracy non-inferior to target LEACE | Acc(F\*) − Acc(B) | > −0.01 |
| P5 | Own-head outputs lower recovery vs clean outputs, FARE features | R(F\*, rep+clean) − R(F\*, rep+head) | > 0.02 |
| P6 | Same contract: FARE complete release vs LEACE complete release | R(B, rep+head) − R(F\*, rep+head) | > 0.02 |

Each of P1–P6 is applied to both cells, giving 12 rows.

**Decision rule.**
- **PASS** iff the simultaneous one-sided lower bound of Δ is strictly above the bar.
- The bound uses Bonferroni at α = 0.05/12, a group bootstrap over assessment groups with fitted predictors fixed,
  B = 20,000, seed 20261011, and the numpy "linear" quantile.
- Otherwise **NOT_ESTABLISHED**.
- Missing rows are **UNRESOLVED** and still count toward the 12.

**Competitive method.** A cell has a competitive method only if P2 ∧ P3 ∧ P4 all PASS and every encoder seed has an
admissible nominee.

**Secondary (exploratory).**
- 90 % intervals: B = 2,000, seed 20261013.
- Supported worst-class and worst-pair AUC, from simultaneous per-class/pair bounds.
- The defense-aware attacker.
- The FZ control.
- The policy-set LEACE (C).
- Every output surface, alone and with noise.
- The full FARE frontier: assessment results for every grid configuration. These are **descriptive** and never used
  for selection.

## Native tests

These are reported separately from recovery.

| Method | Native test |
|---|---|
| PCRL | Historical in-sample ridge R² (benchmark) |
| LEACE | Implementation bound on defense_fit (benchmark maps) |
| FARE | The official certificate (paper §5, App. D.1) on the certification rows |

**FARE certificate.**
- δ = 0.05, with a 50/50 split into D_val and D_test.
- Primary: all fit groups, which for HMDA race means 10 pairs at δ/10 each.
- **Secondary, declared now:** HMDA race groups {0, 1, 2}, 3 pairs.
- UNAVAILABLE whenever its premises fail. It is never patched.
- It bounds demographic-parity distance for classifiers that see only the FARE cell. It is not an attack-AUC bound,
  and it does not cover releases with outputs.

## Runtime

**Ceilings:**
- 12 aggregate CPU-h for fitting, inference and verification. Runner budget: 12 h minus a 3.5 h reserve for the
  stages already spent plus inference and verification.
- 6 GiB peak memory.
- 2 workers (one per dataset), 1 BLAS thread each.

**Fixed order.** Seeds 0, 1, 2. Within a seed: references → output surfaces → A/B/C (views, U2, heads) → noise (3
draws) → FARE grid → nominee → views → FZ → certificate. Then the controls, on seed 0.

**Mandatory vs optional.** All units are mandatory. The kernelized adversarial concept erasure is optional, and is not
planned within this budget.

**Budget stop.** The runner stops before any unit once its ledger exceeds the runner budget. Units are never skipped
ahead. Unrun units are reported as budget-unrun.
