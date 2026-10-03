# Method admission

**Methods admitted:**
- FARE (official, newly admitted for this study);
- LEACE (reused from the benchmark);
- noise (reused).

**Reading.** The details are in `notes/fare/FARE_METHOD_NOTES.md`, `notes/fare/FARE_ENV_RECIPE.md`,
`notes/fare/FARE_ADMISSION_CHECKS.json` and `notes/fare/FARE_TEST_RESULTS.txt`.

## FARE

**Source.** Jovanović, Balunović, Dimitrov and Vechev, "FARE: Provably Fair Representation Learning with Practical
Certificates", ICML 2023, PMLR 202. Appendix D was read, including the multi-group extension D.1.

**Code pin.**
- github.com/eth-sri/fare @ **89cb1b66ed268c16659cbf7428c43e60da2df641**. This is HEAD on 2026-10-03 and the same
  commit pinned in the earlier durable-guarantees study. That study's installation knowledge was reused; its results
  were not.
- Tree sha256 over the 109 tracked `*.py` files: `56a447007fb963090d5f7ea65de3d8b69b9bcddf1669e86234ebab4a52d89cbd`.
- The repository has no licence, so it is used and cited only.

**Environment.** An isolated environment at `~/…/oar_v1/env_fare`:
- CPython 3.9.12;
- the official scikit-learn base `fd60379f` plus the official `sktree` overlay, built with `SKLEARN_NO_OPENMP=1`.

Nothing was installed into the shared project environment.

**Reproduction gate.** The official entry point on the public ACSIncome-CA-2014, with k = 50, n_i = 100, α = 0.9,
gives dp_ub = 0.15712399439471292. This is **bit-exact** to the shipped value, before and after the compiled fix.

### Adaptations

None changes the published objective or the certificate formulas.

1. **Compiled buffer fix.** The sensitive-count buffers were sized by the number of task classes. With 5 race groups
   and a binary task, writes ran out of bounds, giving nondeterministic trees and allocation errors. The buffers are
   now sized by max(groups, classes). This is a no-op when groups ≤ classes. The 5-group root split then matches a
   brute-force search of the published FairGini.
2. **Pickling.** The official tree pickling is broken. The tree is saved and rebuilt through the official state
   methods.
3. **Multi-group certificate budget.** The official script keeps 0.005 / 0.005 per pair. For 4 or more groups that
   makes the per-cell budget negative; the result is NaN, and `max(0, NaN)` silently reports **0**. This means the
   earlier HMDA "dp_ub = 0.000" values were never certificates. The wrapper keeps the paper's 10 / 80 / 10 proportions
   within δ / #pairs and refuses NaN.
4. **Certificate inputs.** Cell ids are used instead of medians, which is bit-identical on the gate.
5. **Base rates.** Lemma 5.1 uses the fit rows' aggregate counts, exactly as the official code does.
6. **Group codes.** These are recoded to 0..G−1.
7. **Seed.** random_state = 43 + seed, so seed 0 equals the official hard-coded 43.
8. **Post-run amendment A1.** The wrapper's duplicate guard compared feature hashes and refused distinct records that
   share a feature vector. It was replaced by a row-identity guard (disjoint roles are asserted). The original
   UNAVAILABLE results are kept. See `NATIVE_TEST_VS_RECOVERY.md`.

### Application in this study

**Scope.** FARE is applied to the **stored frozen PCRL representation** (64 columns, `rep_p0`), not to raw inputs. The
original encoder is not retrained.
- Fit rows: defense_fit, the same admitted rows as LEACE.
- Labels: the task label and the full declared protected attribute (Adult sex with 2 groups; HMDA race with 5 groups,
  never binarised or pooled) are used for fitting only.
- Deployment encoding reads the features only, which was asserted structurally in tests.
- Released features: the one-hot cell index.

**Grid.** Frozen in the lock before any real fit; shared by both datasets; criterion `fair_gini_dp`.

| id | k̄ | n_min | γ | Character | Source |
|---|---|---|---|---|---|
| 1 | 100 | 100 | 0.3 | task-first | official sweep; paper "Accurate" |
| 2 | 50 | 100 | 0.7 | task-first | official sweep; gate configuration |
| 3 | 20 | 100 | 0.85 | balanced | paper "Balanced" |
| 4 | 10 | 1000 | 0.9 | protection-first | official sweep 2 |
| 5 | 5 | 1000 | 0.95 | protection-first | official sweep 2 |
| 6 | 3 | 1000 | 0.95 | protection-first | paper "Fair" (γ capped from 0.999) |

- **γ cap.** HMDA's γ ceiling is 0.957: above it the root becomes a single leaf, because the official criterion keeps
  the binary constant 0.5 while Gini_s for 5 groups exceeds 0.5. The paper's 0.999 is therefore capped at 0.95.
- **Zero-fairness control.** The nominee's k̄ and n_min with γ = 0, which is identical to the official `gini` tree.
- **Aliases.** Configurations whose cells were identical to an earlier configuration's were aliased: HMDA s0 6 → 5;
  HMDA s1 5 → 4 and 6 → 4.

### Native metric

**What it bounds.** The FARE certificate bounds the population demographic-parity distance of any classifier that
receives only the FARE cell, which equals the total variation between the groups' cell distributions. It holds with
probability ≥ 1 − δ, given:
- i.i.d. rows;
- a tree that is independent of the certification rows;
- every cell present in every certificate split, for every group pair.

**More than two groups.** These are handled per pair at δ / C(G, 2); the bound is the maximum over pairs.

**What it is not.** It is not mutual information and not an attack-AUC bound. It does not cover releases with other
channels such as clean outputs, and it does not cover joint 5-class race inference.

## LEACE and noise

These are the benchmark's official concept-erasure 0.2.4 maps (upstream v0.2.4, tree sha256 fffac29d…), reused by
hash and not refitted. Noise σ\* (Adult 2.0, HMDA 4.0) was selected on validation in the benchmark, and its per-person
draws are reproduced identically.

## Not admitted

Kernelized adversarial concept erasure (Ravfogel et al., EMNLP 2022) is optional and was not run. Its released
feature and preimage definition and its controls were not frozen, and the mandatory work was prioritised.
