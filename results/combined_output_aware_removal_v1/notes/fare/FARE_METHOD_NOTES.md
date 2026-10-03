# FARE: method notes for admission (official code and paper)

Owner: method admission, 2026-10-03.

- **Paper:** Jovanović, Balunović, Dimitrov and Vechev, "FARE: Provably Fair Representation Learning with Practical Certificates", ICML 2023, PMLR 202. The PDF read is proceedings.mlr.press/v202/jovanovic23a/jovanovic23a.pdf, sha256 `bb3108f5…853b2`.
- **Code:** github.com/eth-sri/fare @ `89cb1b66ed268c16659cbf7428c43e60da2df641`. This is HEAD of the default branch on 2026-10-03, and it is the same commit the earlier durable-guarantees study pinned. The tree hash over the 109 tracked `*.py` files is `56a447007fb963090d5f7ea65de3d8b69b9bcddf1669e86234ebab4a52d89cbd`. The installation and every adaptation are in `FARE_ENV_RECIPE.md`.

Line references below are to that commit. `main.py` means `code/src/tree/main.py`, `ab.py` means `code/src/tree/alphabeta_adversary.py`, and `sktree/*` is the Cython patch of scikit-learn (base `fd60379f`).

## 1. Objective and fairness criterion

**What is bounded (paper §3).** The quantity is the demographic-parity (DP) distance of a downstream classifier g:

  Δ(g) = |E_{z~Z0}[g(z)] − E_{z~Z1}[g(z)]|,

where Z_s is the distribution of the representation z given s.

Let h\* be the Bayes-optimal adversary that predicts s from z under *balanced* accuracy BA. Then for **any** g:

  Δ(g) ≤ 2·BA(h\*) − 1 = d_TV(Z0, Z1)   (Eq. 2; App. B)

The goal is a *practical certificate* T\* (Def. 4.1, §4). It must satisfy sup_{g∈G} Δ(g) ≤ T\* with probability ≥ 1 − ε over the data sample, where G is the set of all downstream classifiers. The certificate must be:

- high-probability (R1);
- finite-sample (R2);
- distribution-free (R3);
- model-free (R4);
- empirically non-vacuous (R5).

So yes: the bound covers the DP distance of **any** downstream classifier, provided that classifier sees only the representation.

**Restricted encoder (§5).** f maps x to one of k cells {z_1 … z_k}. For such a finite-support z:

  BA(h\*) = Σ_i p(z_i) · max(α0 q_i(0), α1 q_i(1)),  with α_s = 1 / (2 q(s))

Here q_i(s) = P(s | z_i) and q(s) = P(s).

**The fair tree (§6).** This is a classification tree on (x, y) whose split criterion is

  FairGini(D) = (1 − γ)·Gini_y(D) + γ·(0.5 − Gini_s(D)),

where Gini = 1 − Σ_c p_c². In the binary case this is 2p(1 − p), and FairGini ∈ [0, 0.5].

- Increasing γ pushes the sensitive mix in each leaf towards uniform.
- Every row in leaf i is encoded as z_i: the per-feature *median* of the training rows in that leaf for continuous features, or their *mode* for categorical features.
- Categorical features use an ordinal encoding with a generalised Breiman ordering (q ∈ {1, 2, 4}). This is irrelevant here: our 64 features are continuous, so `cat_pos = []`.

**In the code:**

- `sktree/_criterion.pyx:831-851` defines `FairGiniDP`: `(1.0 - alpha) * g_label + alpha * (0.5 - g_sens)` (L839). The children impurity is computed the same way (L850-851).
- `sktree/_criterion.pyx:671-829` defines `FairGini._pvt_*`; Gini is summed over all `n_sens` groups.
- Best-first growth with `max_leaf_nodes` is in `sktree/_tree.pyx:350-470`.
- Leaf medians are computed in `main.py:110-126`, and encoding (`T.apply` then median lookup) in `main.py:129-134`.
- The tree is built by `main.py:20-33`: `DecisionTreeClassifier(criterion='fair_gini_dp', max_leaf_nodes=k, min_samples_leaf=ni, random_state=43)` followed by `fit(x, y, s, cat_pos=…, alpha=γ)`.

## 2. Hyperparameters

The paper's names (App. E) map to the official CLI (`main.py:234-251`) and to the wrapper as follows:

| paper | official CLI / API | wrapper (`FareConfig`) | paper range (App. E) |
|---|---|---|---|
| γ, the fairness weight | `--alpha` → `fit(alpha=)` | `gamma` | [0, 1] |
| k̄, the upper bound on the number of cells (leaves) | `--max-k` → `max_leaf_nodes` | `max_leaf_nodes` | [2, 200] |
| n_i, the minimum training rows per leaf | `--min-ni` → `min_samples_leaf` | `min_samples_leaf` | [50, 1000] |
| v, the fraction of training data held out as D_val | `--val-split` (`main.py:184-197`) | replaced by separate certificate rows | {0.1, 0.2, 0.3, 0.5} |
| ε, the certificate error | `err_budget = 0.05` (`main.py:355`) | `cert_cfg.delta` | 0.05 |
| ε_b, ε_s | `eps_ab = eps_glob = 0.005` (`main.py:390`) | 10 % each of the pair budget | 0.005 each (ε_c = 0.04) |
| criterion | `--gini-metric dp` | `fair_gini_dp` only | dp / eopp / eo |
| random_state | hard-coded 43 (`main.py:33`) | 43 + seed | |

The official sweeps are in `code/shell/*/run_tree.sh`. For ACSIncome-CA:

- k̄ ∈ {2, 4, 6, 8, 10, 20, 50, 100, 200}, n_i = 100, γ ∈ {0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999}, v = 0.3;
- k̄ ∈ {2, …, 10}, n_i = 1000, γ ∈ {0.8, …, 0.999}, v = 0.5.

The paper's named settings (App. H.2) are:

- **Fair:** γ = 0.999, n_i = 1000, v = 0.5.
- **Balanced:** γ = 0.85, n_i = 100, v = 0.3.
- **Accurate:** γ = 0.3, n_i = 10, v = 0.1.

Each was run with k̄ ∈ {3, 5, 8, 20, 50}.

From App. E: a larger n_i and a larger v tighten the bound, and a larger k̄ raises accuracy but loosens the bound.

## 3. The certificate

### What it bounds

With probability ≥ 1 − ε over the sampling of the rows used in the procedure, and **for the fixed, already-trained encoder f**, T\* bounds sup_g Δ(g): the population DP distance between the two groups of any binary classifier g whose input is z alone. By Eq. 2, T\* also upper-bounds:

- 2·BA(h\*) − 1 for the optimal hard adversary;
- d_TV between the cell distributions of the two groups (2·BA\* − 1 = d_TV for finite z).

### The procedure and its data split (§5 "Applying the lemmas")

The split is D = D_train ∪ D_val ∪ D_test, mutually disjoint, with f trained on D_train only.

1. **Lemma 5.1, base rates, on D_train (error ε_b).**
   - A Clopper–Pearson interval on q(0) gives ᾱ0 and ᾱ1.
   - Using D_train is sound because q(s) does not depend on f.
   - Code: `ab.py:117-134`, called on `z_train` (`ab.py:108`).
2. **Lemma 5.2, per-cell bounds, on D_val (error ε_c).**
   - D_val is "held out … and not used in training of f in any capacity".
   - Each cell gets a two-sided Clopper–Pearson interval at ε_c / k, and t_i = max(ᾱ0·p_ub, ᾱ1·q_ub).
   - Code: `ab.py:137-197`, with per-cell budget `budget/k` (L155) and t_i at L162. t_i is clipped at max(ᾱ0, ᾱ1) (L163-167); the clip is sound.
3. **Lemma 5.3, Hoeffding sum, on D_test (error ε_s).**
   - S\* = (1/n) Σ_j t_{idx(z_j)} + (b − a)·√(−log ε_s / (2n)), where a = min t_i and b = max t_i.
   - Code: `ab.py:200-226` (L218).
4. **Combination.**
   - T\* = 2S\* − 1, and by the union bound ε = ε_b + ε_c + ε_s.
   - Code: `ab.py:72-74` and `ab.py:101-114`.

In the official script:

- D_train is the training rows after the v hold-out;
- D_val is the v hold-out (`main.py:184-197`);
- D_test is the dataset's test split.

### Premises

**Independence.**

- Rows are independent draws from the distribution the bound refers to (R3: no other distributional assumption).
- f is fixed before D_val and D_test are drawn, and is independent of them.
- D_val and D_test are disjoint.
- The bound is conditional on f. It is not a statement about the training procedure or about other seeds.

**Distribution shift.** No guarantee is given under shift. App. H.4 shows the results degrade.

**Duplicated data.** Data duplication does not tighten a valid bound (§7, Fig. 5 caveat).

**Confidence parameter.** ε = 0.05 (95 %), split as ε_b = ε_s = 0.005 and ε_c = 0.04 (App. E). §5 calls this decomposition a heuristic choice.

**Every cell must appear in every split.** The code asserts that each of the k cells occurs in D_train, D_val and D_test (`ab.py:121, 140, 209, 233`; also `main.py:103, 106`). If a cell is missing, no certificate is produced. The wrapper reports this as `UNAVAILABLE` and never patches the bound.

**Sample size.** No minimum is stated; tightness is governed by:

- the number of D_val rows per cell, because the Clopper–Pearson width at ε_c/k grows as cells shrink and as k grows;
- the size of D_test, through the Hoeffding term ∝ (b − a)/√n;
- the minority-group share, because ᾱ_s = 1/(2 q_lb) inflates t_i.

Fig. 5 shows the bound tightening as n grows.

**Implications for this study.**

- The certificate rows are about 20 % of the attacker_fit groups: roughly 1.5k rows for Adult and 1.36k for HMDA, split 50/50 into D_val and D_test.
- With k̄ = 100 this leaves about 7 D_val rows per cell, so the bound will be loose and may be vacuous.
- On HMDA, the race-4 share in attacker_fit is about 1.0 % and race-3 about 1.8 %. These shares are aggregate counts read for planning only, with no features and no fits. Pairs that involve race groups 3 or 4 will have about 5–15 D_val rows, and a cell missing from a split is likely. In that case the all-pairs HMDA certificate is `UNAVAILABLE` by the official premise.

## 4. More than two sensitive groups (App. D.1)

**Metric.** "maximal absolute difference in prediction rates of any class y, w.r.t. any two sensitive groups i and j". The certificate holds for any cardinality of y, because the most unfair classifier is binary.

**Certificate (class-pair accounting).** The binary procedure is run on each of the C(|s|, 2) pairs, using only that pair's rows and coding group i as 0 and group j as 1. ε is reduced C(|s|, 2)-fold per pair, so the overall confidence stays 1 − ε. The reported bound is the max over pairs.

- Official code: `main.py:403-430`, with pair recoding at L414-422, `err_budget /= C(G,2)` at L408, and max over pairs at L427.
- For HMDA race (5 groups) this means 10 pairs at δ/10 each.

**Training.** The paper says the Gini definition "directly supports multiple values of y/s". The code computes Gini_s over all G groups but keeps the **binary constant 0.5**. With G > 2, Gini_s can exceed 0.5 (its maximum is 1 − 1/G), so the fairness term 0.5 − Gini_s becomes negative.

The builder turns any node whose impurity is ≤ EPSILON into a leaf (`sktree/_tree.pyx:507`). Consequences for HMDA defense_fit (Gini_s = 0.5087, Gini_y = 0.1912):

- the root becomes a leaf for γ ≥ 0.957, so the encoder is constant;
- nodes stop splitting early. At seed 0 the realised cells were F2 22/50, F4 4/10 and F5 = F6 = 2 (`FARE_CALIBRATION.json`).

This is the published code's objective, and we run it unchanged.

**Two defects in the official code (we do not change the objective):**

1. **Compiled buffer overflow when #groups > #task classes.**
   - The sensitive-count buffers are sized by the number of *task* classes (`sktree/_criterion.pyx:256-265`, `# TODO add max_n_sens`).
   - With binary y and 5 race groups, writes run past the buffer. The unfixed build gave non-deterministic trees and random `ValueError: array is too big` (logged privately).
   - The paper's own multi-group run (ACSIncomeMulti: 4 classes, 3 groups) never hit this, because #groups ≤ #classes there.
   - Fix: size these buffers with max(#classes, #groups). This changes no arithmetic when #groups ≤ #classes; the reproduction gate stays bit-exact (§6).
2. **Negative Lemma 5.2 budget in the multi-group script.**
   - `main.py:425` keeps `eps_glob = eps_ab = 0.005` while dividing the total by the number of pairs. For ≥ 4 groups, ε/C(G,2) < 0.01, so the Lemma 5.2 budget is negative.
   - Clopper–Pearson then returns NaN, and `max(total_dpub, nan)` silently keeps the previous value, which is 0.
   - Demonstration on synthetic data (`FARE_ADMISSION_CHECKS.json`): with 4 or 5 groups the official script would report 0 where the correct per-pair bound is 0.375 or 0.383.
   - **The earlier durable-guarantees study's HMDA "dp_ub = 0.000" values come from this artefact. They are not certificates.**
   - The wrapper keeps the paper's proportions within each pair (ε_b = ε_s = 10 % and ε_c = 80 % of δ/#pairs). Any decomposition with positive parts is valid by the union bound (§5). The wrapper never folds a NaN into the max.

## 5. What the certificate does not say

**Mutual information.** It makes no statement about I(z; s) or any other information measure. It bounds a TV distance and a DP distance only.

**Attacker AUC.** It does not bound an attacker's AUC. Two groups can have d_TV = 0.5 while a scorer reaches AUC 0.875 > (1 + 0.5)/2. Example: P0 = U{1, 2}, P1 = U{2, 3}, score = z. What it does bound is the population *balanced accuracy* of any hard pairwise decision rule on z: at most (1 + T\*)/2.

**Joint multi-group inference.** It only bounds pairwise quantities, maximised over pairs. It does not bound accuracy at recovering the full 5-class race.

**Other channels.** It does not cover a classifier that sees anything besides z, such as clean model outputs, other features, or a head trained on other inputs. "FARE + clean outputs" is outside its scope.

**Finite assessment rows.** It is a population statement. An empirical DP measured on a finite assessment set can exceed T\* through sampling noise.

**Other trees.** It does not cover a different tree: other seeds, other configurations, or refits. It is conditional on the fitted f.

**Contaminated rows.** It does not hold if the rows in D_val or D_test overlap or depend on the fit rows (for example the same person or unit), or under distribution shift.

**Unavailable or vacuous results.** If `status = UNAVAILABLE` or T\* ≥ 1, there is no usable guarantee. A small T\* on a certificate set of about 1.5k rows is possible only for small k.

## 6. Verification performed

**Reproduction gate.**

- Run: official entry point, unmodified, on public ACSIncome-CA-2014 with k̄ = 50, n_i = 100, γ = 0.9, v = 0.3.
- Result: dp_ub = **0.15712399439471292**, identical to the shipped `result/_eval/ACSIncome-CA-2014/tree.npy` value. This holds with both the official build and the fixed build, and z_test is byte-identical between the two builds. The tree's own test accuracy is 79.17 %; the shipped value is a mean 1-NN accuracy of 0.7959 and was not re-run.

**Wrapper core.** The wrapper core (`official_pair_bound`) re-derives the same bound bit-exactly from the gate's saved embeddings. It does so with both median vectors and integer cell ids, because the adversary uses only `np.unique(z)`. All of this is in `FARE_ADMISSION_CHECKS.json`.

**Synthetic tests.** The 12 synthetic tests in `oar/tests/test_fare_synthetic.py` pass (`FARE_TEST_RESULTS.txt`). They include:

- the root split for 5 groups matches a brute-force search of the published FairGini objective;
- γ = 0 gives exactly the official `gini` tree.
