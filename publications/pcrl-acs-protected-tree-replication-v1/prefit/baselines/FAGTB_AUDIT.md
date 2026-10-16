# FAGTB audit: Fair Adversarial Gradient Tree Boosting (published baseline)

- **Campaign:** `pcrl_acs_protected_tree_replication_v1`
- **Branch:** `research/pcrl-acs-protected-tree-replication-v1`
- **Role:** PUBLISHED-BASELINE
- **Date:** 2026-10-15
- **Data policy:** no ACS or Adult value was read. Every run below uses synthetic fixtures.
- **Deliverables:**
  - `acs_replication/fagtb.py`, with two classes:
    - `FAGTBBinary`: a faithful port of the official code;
    - `FAGTBMulticlassAdaptation`, called **FAGTB_MC_ADAPTATION**: not official.
  - `acs_replication/test_fagtb.py`: 17 tests, all passing.
  - This audit.

## 0. Verdict in one paragraph

The official code implements one variant only: **FAGTB-1-Unit, demographic parity**. That variant is:
- a gradient-boosted regression-tree classifier;
- whose residual subtracts λ times the input gradient of a single logistic unit;
- where that unit predicts SEX from p = σ(F) and takes one gradient-descent step (lr 0.01, on the **summed** cross-entropy) per boosting iteration.

`FAGTBBinary` reproduces it with the TensorFlow-1 graph replaced by closed-form numpy. Against the **unmodified official class** run in an isolated TensorFlow environment:
- **λ = 0:** the port is bitwise identical.
- **λ > 0:** every official step is reproduced at float32 precision.
- **λ > 0, free-running:** the two trajectories drift apart. The drift exactly matches the algorithm's own sensitivity: the port drifts the same way against itself when the adversary init is moved by 1e-12. So no implementation, the official one included, can reproduce λ > 0 runs bitwise.

**Recommended grid:**
- λ ∈ {0.03, 0.06, 0.15};
- 300 trees, learning rate 0.01, depth 9, `max_features` 20 (notebook settings);
- adversary init seed = campaign seed.

**Cost at 20,000 × 150, one thread:** about 12–13 s per binary fit, and about K × 12 s per K-class adaptation fit.

**Main caveats:**
- **n-dependence:** the summed-loss adversary takes an effective step of 0.01·n, so its strength grows with the number of fitting rows.
- **Logit cap:** at λ = 0, F cannot move more than 3 logits from F₀. This matters for NLL-based model selection.
- **Seed sensitivity:** results vary materially across adversary seeds.
- **Degenerate cells:** a high-λ cell can end up worse than a constant predictor.

## 1. Provenance (pinned)

| Item | Value |
|---|---|
| Repository | https://github.com/vincent-grari/FAGTB |
| Commit | `0cd0bb100d9258790e8fe47b3a47cc6170eac652` ("Créé avec Colaboratory", 2020-07-03 19:33 +0200). The code was released about 8 months after arXiv v1 (13 Nov 2019). |
| Local copy | `.private/acs_replication/vendor/FAGTB` (gitignored). `git status` is clean; test (e) asserts it stays clean. |
| `functions.py` | sha256 `30bd91eccb6e90f263614b95b7f06e7620e57784e8845ee1aa87bbc6116f0a5d` (class `FAGTB`, lines 139–275) |
| `FAGTB.ipynb` | sha256 `b6b430de7ce871418252ceadd7e14e182c709ca94456c284370e3429fd22e0a2` |
| `ProcessedData.py` | sha256 `ca02b5a97327ccad1a812188ca1e8442b261cdd004da1fc7de08b8c2c0023d35` (80/20 split, `TRAINING_PERCENT = 0.8`) |
| `README.md` | sha256 `4dfd1c0c424177a1298136cca251a684a692e12a2727d893687d3c6f93a42e2d` |
| `FAGTB.png` | sha256 `5f7345854cf9e40773c91acaf35b117228908b553c093c63d0b451ebcb8535b6` |
| Paper | Grari, Ruf, Lamprier, Detyniecki, *Fair Adversarial Gradient Tree Boosting*, IEEE ICDM 2019, arXiv:1911.05369v1. The 6-page PDF has sha256 `344f1e8b85f516e54c41adee037207a802157507bccdbda9dd109995f607a055`. |

## 2. What the official code computes (`functions.py`, `FAGTB.fit`)

```
F0 = log(sum y / sum(1-y))                                          # L184
graph: logit = W1*p + b1; loss = reduce_SUM(sigmoid_CE(logit, s));  # L203-212, float32
       W1, b1 = tf.random_normal(seed=7); GradientDescent(0.01)     # L197-212
for i in range(n_estimators):
    p = 1/(1+exp(-F))                                               # L229
    one GD step on (W1, b1) at the current p                        # L232
    g = d loss / d p     (with the UPDATED W1, b1)                  # L219, L237
    t = -(g * p * (1-p));  u = y - p - LAMBDA*t - proj (proj = 0)   # L242-246
    tree_i = DecisionTreeRegressor(criterion='friedman_mse', max_depth=9,
             max_features=max_features, min_samples_split=2, min_samples_leaf=1,
             min_impurity_decrease=0, min_impurity_split=None, random_state=0).fit(X, u)  # L158-163, L247
    F += learning_rate * tree_i(X)                                  # L248-250
predict(X) = sigmoid(F0 + sum_i learning_rate * tree_i(X))          # L267-275
```

- **Why u is the right residual.** Since the adversary loss is a sum over rows, u_i = −∂/∂F_i [L_F,i − λ L_A,i]. This is the paper's u = r − λt (Algorithm 1, steps a–c) for the min-max objective of eq. (6), with the demographic-parity adversary input V = σ(F(X)) (Sec. III-A).
- **Update order.** The adversary is updated first (on F_i). Its input gradient is then taken with the updated parameters, and only then is the tree fit. This is Algorithm 1 shifted by one step, plus one extra adversary step on the constant F₀.

### 2.1 Defects and discrepancies (each verified)

| # | Finding | Where | How verified | Consequence | Port |
|---|---|---|---|---|---|
| 1 | `max_depth=9` is hard-coded and the constructor's `max_depth` is ignored. The notebook passes 10. | L158 | Read; the parity harness records the official tree kwargs (`max_depth: 9`). | Depth 9 is what actually ran. | Default 9, exposed as a parameter; `NOTEBOOK_MAX_DEPTH_PASSED = 10` recorded. |
| 2 | Constructor arguments `min_samples_split`, `min_impurity` and `regression` are ignored; `min_samples_split=2` is hard-coded. | L141–163 | Read. | Notebook values 1.0 / False / 1 have no effect. | Hard-coded like the official code. |
| 3 | `min_impurity_split=None` raises `TypeError` on scikit-learn ≥ 1.0. | L160 | Reproduced on sklearn 1.8.0. | The official class cannot run on modern sklearn. | Argument dropped (§5). |
| 3a | What `None` meant depends on the sklearn version: 0.22.x means **leaf if impurity ≤ 1e-7**; 0.23.x means 0; ≥ 1.0 means machine epsilon. | sklearn 0.22.2.post1 `tree/_classes.py` L296–304 and `_tree.pyx` L233/457; 0.23.2 `_classes.py` L304–316 | Read from the official sdists. | Not reproducible with sklearn ≥ 0.23. | Recorded; impact discussed in §6.3. |
| 3b | The authors' environment was sklearn 0.22 or 0.23. | The notebook's Standard-GTB cell passes `presort='auto'`, accepted by GBC 0.22/0.23 (`ensemble/_gb.py`) and removed in 0.24; the Colab path is Python 3.6. | Read. | | |
| 4 | `fit(X, y, sensitive, LAMBDA, Xtest, yt, sensitivet)` takes **test labels and test sensitive values**. They feed only printed monitoring (test accuracy and test p-rule each iteration). | L178, L256–264 | Read. Parity shows training is unaffected. | Assessment labels sit in the fitting interface (forbidden by the campaign). | Removed. `fit(X, y, s, lam)` only. |
| 5 | `predict` calls `np.full(n, self.Init, self.Init)`, passing a value as the dtype. **This is not a live bug:** it works on numpy 2.4.2 and 2.4.6 because `Init` is an `np.float64`, whose `.dtype` attribute numpy accepts. It raises `TypeError` only if `Init` is a plain Python float. With `n_estimators=0` it also raises `UnboundLocalError`. | L268 | Run directly, and the official `predict` ran in parity. | A latent fragility. The caller's "bug" claim is corrected. | Rewritten; outputs are equal (§6). |
| 6 | The adversary is a TF1 **float32** graph. W1 and b1 use the same op seed (7) with the graph seed unset, so **both start at the same value, −0.046945683658123016, in every official run.** | L196–219 | Measured in TF 2.20.0. | Every official run starts its adversary from one fixed pair. | numpy init (repair). The constant is exposed as `OFFICIAL_TF_SEED7_ADVERSARY_INIT`. |
| 7 | The adversary loss is `reduce_sum`, not a mean, with a fixed learning rate of 0.01. The effective step on the mean loss is therefore 0.01·n. | L210–212 | Analysis and synthetic probe (§8). | Adversary strength depends on n; above n ≈ 600 gradient descent is in a large-step, oscillating regime. | Preserved (this is the official behaviour) and documented. |
| 8 | The paper says "a Xavier initialization is used in our experiments" (Sec. III-B), but the code uses `tf.random_normal` (std 1). For a 1-in/1-out unit, Glorot-normal std is 1, but biases are normally initialised at 0. | Paper vs L206–207 | Read. | Paper and code disagree. | The port follows the code. |
| 9 | Dead code: the numpy `LogisticRegression` class (instances built into `self.clfs`, never used; its `_initialize_parameters` uses `math`, which is never imported), `fit2`, duplicate `Sigmoid` and `p_rule`. | L165–171, L278–320 | Read. | None. | Dropped. |
| 10 | Heavy imports: `fire`, `fairness.*` and `torch` are unused by the class; `pandas` is used only by `DATA_TRAIN_TEST`. | L9–51 | Parity ran with these stubbed. | Install burden. | Not required. |
| 11 | `fit` returns `{'y_pred2','S_ADV'}` (state at the start of the last iteration), not the model. `lossglobal = losstraining − LAMBDA*t` mixes a loss with a gradient and is only printed. | L254, L265 | Read. | None. | `fit` returns `self`; task and adversary losses are recorded separately. |
| 12 | The release contains only FAGTB-1-Unit for demographic parity. There is no FAGTB-NN (several adversary steps per iteration, biased-GTB warm start, Sec. IV), no equalized-odds variant (adversary input (σ(F), y), Sec. III-A), no CV grid search, no line search and no early stopping. | Whole repository | Read. | Only FAGTB-1-Unit for demographic parity can be replicated from code. | n/a |

## 3. Hyperparameters: paper vs code vs what the notebook ran

| Item | Paper (arXiv:1911.05369v1) | Official code | Notebook `FAGTB.ipynb` (0-based cells) |
|---|---|---|---|
| Variant | FAGTB-1-Unit and FAGTB-NN (Sec. IV); demographic parity and equalized odds (Sec. III-A) | 1-Unit, demographic parity | 1-Unit, demographic parity |
| Objective / residual | eq. (6); u = r − λt (Alg. 1 a–c) | L242–246 | — |
| Line search γ_m | Replaced by "a specific fixed learning rate" (Sec. IV) | Fixed `learning_rate` | 0.01 |
| Trees M | Not reported. "Best hyperparameters by grid search in 5-fold cross validation (specific to each of them)" (Sec. IV). The only tree count given is for Standard GTB on Bank: depth 3 with 800 trees. | Argument | **300** (cell 13) |
| Depth | Not reported | **9** (hard-coded, L158) | Passes 10 (ignored) |
| `max_features` | Not reported | Argument | **20**, an integer count (cell 13) |
| `min_samples_split` / `min_samples_leaf` | — | 2 / 1 (hard-coded) | Passes 1.0 (ignored) |
| Adversary | Logistic unit; "the two parameters … are chosen randomly"; one step per boosting iteration (Sec. IV); learning rate α given in Alg. 1 but no value reported | GD lr 0.01 on the **summed** CE; one step per iteration | (code) |
| Adversary init | Xavier (Sec. III-B) | N(0,1) with seed 7; W1 = b1 = −0.0469457 | (code) |
| λ | Fig. 2 (FAGTB-NN, Adult, 10 runs each): {0, 0.03, 0.05, 0.07, 0.08, 0.09, 0.12, 0.15, 0.16, 0.17}. Fig. 3: 0 (unfair), **0.06** (p-rule 51.9%), **0.15** (p-rule 96.2%). | `LAMBDA` argument | **0.15** (cell 13) |
| Early stopping | None mentioned | None | None |
| Protocol | 10 random 80/20 splits; test-set means reported (Sec. IV). Demographic-parity target is a 90% p-rule. | `ProcessedData.py` 80/20 | 10 splits from the global numpy RNG (FAGTB loop never reseeds; Standard-GTB loop uses `np.random.seed()`) |
| Adult fitting rows | The paper uses the fairness-comparison Adult file: 32,561 − 2,399 = **30,162 rows** (Friedler et al., arXiv:1802.04422v1, dataset section), so about **24,129** fitting rows. Taken from the literature; not checked against data. | | |

**Notebook outputs vs Table I.**
- **FAGTB (λ = 0.15):** the summary cell (cell 15, execution count 34) shows accuracy **84.26%**, p-rule **90.36%**, DispFPR 0.034, DispFNR 0.271. Table I's FAGTB-1-Unit Adult row is **84.4% / 90.4%**.
- **Standard GTB:** the notebook's GBC uses 500 trees, lr 0.05, depth 3, `max_features` 90, `min_samples_leaf` 2. Cell 10 gives 86.75% / 31.89% / 0.063 / 0.071, against Table I (86.8% / 32.6%) and Table II (0.06 / 0.07).
- **Caveats on that match:**
  - The FAGTB training cell (13) has no saved execution count or output, so the link from cell 13's saved code to cell 15's numbers is inferred from the agreement with Table I.
  - Cell 16's plot title reads "Unfair model (FAGTB with   =0.015)". The λ glyph is lost, and 0.015 is inconsistent with `LAMBDA=0.15`.

## 4. Repairs, adaptations, unchanged

**Repairs (engineering, behaviour-preserving):**

| Repair | Evidence it preserves behaviour |
|---|---|
| TF1 float32 graph replaced by closed-form numpy float64 (same equations, same order) | Teacher-forced parity: the port's adversary step matches an exact `math.fsum` reference to ≤ 3.6e-15 relative. TF's own float32 step is off by up to 2.3e-5. The input-gradient difference is ≤ 3.1e-7 relative. |
| Adversary init: `np.random.default_rng(seed).standard_normal(2)` instead of TF seed 7 | Both are fixed N(0,1) draws, and the trajectory is chaotic in the init anyway (§6). The exact official start is available as `adversary_init=OFFICIAL_TF_SEED7_ADVERSARY_INIT`. |
| `min_impurity_split=None` dropped | At λ = 0, official and port are bitwise identical with the same sklearn. The remaining difference from the authors' sklearn 0.22 (1e-7 leaf threshold) is in §6.3. |
| Test-label arguments, monitoring prints and the gradient-check print removed | Training never reads them. λ = 0 parity is bitwise. |
| `predict` rewritten; `predict_proba` returns (n, 2); `decision_function` added | Column 1 equals the official `predict()` (0.0 difference at λ = 0). |
| Unused imports and dead code removed; `fit` returns `self` | None of them affect training. |
| Input validation: binary y and s, both classes present, finite X, λ ≥ 0 | The official code yields inf/nan instead of an error. |
| `max_depth` exposed (default 9 = official effective value) | Default reproduces the official tree (kwargs compared in parity). |

**Scientific adaptations:**

| Adaptation | Status |
|---|---|
| The whole of `FAGTBMulticlassAdaptation` (FAGTB_MC_ADAPTATION, §9) | Not official, and not in the paper. |
| (binary) none | Non-default `adversary_lr` or `tree_random_state` would be deviations. Defaults are official. |

**Unchanged (official semantics kept):**
- F₀ = log-odds;
- σ written as `1/(1+exp(−F))`;
- 1-input logistic adversary on p, trained by gradient descent with lr 0.01 on the summed CE, one step per iteration;
- the order: adversary step, then gradient with the updated adversary, then tree;
- the residual u = y − p − λt with t = −g·p(1−p);
- the tree: `friedman_mse`, depth 9, `max_features` passed through, `min_samples_split` 2, `min_samples_leaf` 1, `random_state` 0 for every tree;
- `F += lr·h(X)`;
- no line search, no early stopping, no sample weights;
- the protected attribute is not a model input (the notebook drops `sex` from X; the campaign must do the same).

## 5. Port interface

- **Binary:** `FAGTBBinary(n_estimators=300, learning_rate=0.01, max_depth=9, max_features=20, adversary_lr=0.01, seed=0, adversary_init=None, tree_random_state=0).fit(X, y, s, lam)`.
- **Multiclass:** `FAGTBMulticlassAdaptation(..., n_classes=None).fit(X, y, s, lam)`.
- **Inputs:** fitting arrays only. `inspect.signature` is pinned by a test, and keyword arguments such as `Xtest` raise `TypeError`.
- **Outputs:** `predict_proba(X)` and `decision_function(X)`.
- **Diagnostics:** `history_` holds per-iteration values computed on the **fitting data only**: task loss, training accuracy, adversary summed and mean CE, adversary accuracy, adversary parameters, mean |y − p|, and mean |λt|.
- **Determinism:** fits are deterministic given `seed`, which is test-pinned.

## 6. Official-environment parity (achieved)

**Environment** (`.private/acs_replication/vendor/parity_venv`, gitignored):
- Python 3.11.15, TensorFlow 2.20.0, scikit-learn 1.8.0 (same as the campaign venv), numpy 2.4.6, pandas 3.0.6, on macOS 15.6 arm64.
- Built with: `uv venv --python 3.11.15 …/parity_venv && uv pip install --python …/parity_venv/bin/python tensorflow==2.20.0 scikit-learn==1.8.0 pandas`.
- The sklearn versions that still accept `min_impurity_split` (0.22–0.24) have **no macOS-arm64 wheels** (checked on PyPI), so they could not be installed here.

**Harness:** `.private/acs_replication/vendor/parity/official_parity.py`. It:
- loads the sha256-checked, **unmodified** `functions.py` by file path;
- stubs only `fire`, `fairness.*` and `torch.*`, which class FAGTB never uses;
- replaces the module-global `DecisionTreeRegressor` with a factory that drops `min_impurity_split=None` (asserted to be None) and records each residual;
- proxies `tf.Session` (single-threaded) to record (W1, b1) before and after each step, plus every fetched value (graph unchanged);
- calls the notebook-style constructor and `fit(..., Xtest, yt, sensitivet)` on synthetic data;
- runs the port with `adversary_init` set to TF's initial values;
- checks free-running agreement, teacher-forced agreement, and a self-sensitivity control (the port against itself with the init moved by 1e-12).

Rerun with: `parity_venv/bin/python -I parity/official_parity.py --worktree <wt> --out <json>`, or via test (e). The test writes `parity/parity_result_test.json`.

**Results.** All scenarios use 25 features, `max_features` 20, and the notebook's tree settings.

| Scenario | n | λ | M, lr | F₀ equal | Teacher-forced: TF step vs exact / port step vs exact / input-grad rel / residual abs | Free-running: first iteration with residual Δ > 1e-6 / final-prediction mean \|Δ\| | Control (port vs port, init + 1e-12): first iteration / mean \|Δ\| |
|---|---|---|---|---|---|---|---|
| A | 300 | 0.15 | 300, 0.01 | yes | 1.2e-7 / 1.9e-16 / 1.8e-7 / 6.1e-9 | 1 / 4.2e-3 | 1 / 4.4e-3 |
| B | 300 | **0** | 300, 0.01 | yes | 1.3e-7 / 2.2e-16 / 2.3e-7 / **0** | **never / 0.0**, bitwise on train and holdout | never / 0.0 |
| C | 4000 | 0.15 | 100, 0.01 | yes | 2.3e-5 / 3.6e-15 / 3.1e-7 / 2.3e-7 | 1 / 3.4e-3 | 1 / 3.5e-3 |
| D | 300 | 0.5 | 100, 0.1 | yes | 1.2e-7 / 2.2e-16 / 1.7e-7 / 1.5e-8 | 5 / 3.6e-3 | 5 / 3.8e-3 |

Additional checks:
- TF's initial (W1, b1) = (−0.046945683658123016, −0.046945683658123016) in every scenario.
- The official tree kwargs equal the port's tree parameters, apart from `min_impurity_split`.
- The official `predict()` ran without error.
- Metric-level results (train log-loss, train and holdout accuracy, holdout positive-rate gap by S) agree within the control's spread. For example, scenario C train log-loss is 0.5752 (official), 0.5748 (port) and 0.5746 (control).

### 6.1 Interpretation

- **λ = 0 is bitwise identical.** This validates the init, the boosting loop, the tree configuration and `predict`.
- **λ > 0, teacher-forced.** Given the official state at every iteration, the port's equations reproduce TF's adversary step, input gradient and residual at float32 precision. In fact the port is *more* exact than TF: about 1e-15 vs up to 2.3e-5.
- **λ > 0, free-running.** The two runs drift apart after 1–5 iterations, and this is explained by two mechanisms:
  - **CART tie sensitivity.** At iteration 0, F₀ is constant, so the residual takes only four values (y × s) and many candidate splits are *exact* ties in real arithmetic. Floating-point rounding decides between them. A 1e-15 relative perturbation of a 2-valued residual changes a depth-9 tree's predictions by up to 0.91 (test `test_d_tree_learner_sign_symmetry_and_tie_sensitivity`).
  - **The large-step adversary** (C: |W| reaches 21.5 with effective step 40).

  The port drifts against *itself* at the same iteration and by the same amount when the init moves by 1e-12. So the gap is the algorithm's own sensitivity, not a porting error.
- **Consequence.** Bitwise replication of any λ > 0 official run (including the authors' numbers) is impossible in principle. FAGTB results must be reported as distributions over seeds.

### 6.2 Exact-start option

`adversary_init=fagtb.OFFICIAL_TF_SEED7_ADVERSARY_INIT` starts the port exactly where the official code starts. This was measured in TF 2.20; it is not verified for the authors' 2020 TF build.

### 6.3 Residual environment difference (unrepairable)

- **What differs.** In sklearn 0.22, a node becomes a leaf when its impurity is ≤ 1e-7. In sklearn ≥ 1.0 the threshold is ≤ 2.2e-16. This affects only nodes whose residual variance is between those values. Splitting such a node also consumes random draws, which changes the `max_features` sampling for later nodes in the same tree.
- **What it means.** Trees are an equally valid but *different* random realisation compared with the authors' sklearn. This is no worse than the tie sensitivity above.

## 7. Test record (`python -m pytest acs_replication/test_fagtb.py -q`, or `python acs_replication/test_fagtb.py`)

Both runners report **17 passed, 0 failed, 0 skipped**. The pytest run takes about 21 s, mostly test (e).

| Test | Checks | Result |
|---|---|---|
| (a) `test_a_binary_residual_matches_finite_difference` | u = −d/dF[Σ L_F − λ Σ L_A] by central differences, computed from an independent objective (tanh-σ, logaddexp). λ ∈ {0, 0.15, 3}; \|w\| up to 40. Also single-coordinate checks on the total objective. | Max \|u − FD\| = **1.1e-9** |
| (b) `test_b_adversary_input_and_param_gradients_match_finite_difference` | Input gradient and (w, b) gradient of the summed CE | Input **1.6e-9**; parameter relative **8.3e-9** |
| (b) `test_b_fit_uses_official_update_order` | Iteration 0 rebuilt by hand: GD step on the summed CE (lr 0.01), then gradient with the updated adversary, then tree. A negative control uses the pre-update adversary. | Equal to 1e-12; the control differs by > 1e-4 |
| (c) `test_c_lambda_zero_is_plain_gradient_boosting_with_same_trees` | λ = 0 against a hand-written log-loss gradient-boosting loop | **Bitwise** equal per tree and in F; adversary seed has no influence |
| (c) `test_c_predict_proba_matches_cumulative_sum_definition` | σ(F₀ + Σ lr·h); rows sum to 1; fit-time log-loss equals predict-time log-loss | Bitwise |
| (d) `test_d_multiclass_residual_matches_finite_difference` | U = −dJ/dF through the softmax Jacobian; adversary input and parameter gradients | Max \|U − FD\| = **7.9e-9** |
| (d) `test_d_multiclass_probabilities_sum_to_one_and_cumulative_definition` | Rows sum to 1; F₀ = log-priors | 1e-12 |
| (d) `test_d_multiclass_K2_lambda0_equals_binary_with_doubled_learning_rate` | K=2, λ=0: margin equals binary FAGTB with 2·lr (tie-free configuration, 3 seeds, depth 1 and 3, 30 iterations); class-0 tree = −class-1 tree; per-step identities | ≤ 1e-9 |
| (d) `test_d_multiclass_K2_residual_is_binary_residual_under_reparameterisation` | U₁ = binary residual with W = w₁ − w₀ and B = w₀ + b; U₀ = −U₁ | 1e-12 |
| (d) `test_d_multiclass_K2_lambda_positive_is_not_the_binary_algorithm` | Documents that K=2 with λ > 0 differs from binary | Holds |
| (d) `test_d_multiclass_requires_every_class` | Missing class raises `ValueError` | Holds |
| (d) `test_d_tree_learner_sign_symmetry_and_tie_sensitivity` | tree(−u) = −tree(u) bitwise; tie-sensitivity characterisation | Holds |
| `test_fit_interfaces_accept_only_fitting_arrays` | Signature is exactly (X, y, s, lam); test-label keyword arguments raise `TypeError` | Holds |
| `test_input_validation` / `test_defaults_record_official_values` / `test_determinism_given_seed` | Validation; official defaults and tree kwargs; seed determinism | Hold |
| (e) `test_e_official_environment_parity` | Runs the §6 harness and asserts its criteria, including that the vendor clone stays unmodified. Skips if the venv is absent. | Holds |

**Note on the K=2 test.** A first version compared K=2 against binary end-to-end with depth-9 trees. It failed after 1–3 iterations, because of the CART tie sensitivity of §6.1 (U₀ = −U₁ only to 1 ulp). The test was restructured into per-step identities plus a tie-free end-to-end check. That restructuring is documented in the test, and is not a loosened tolerance.

## 8. Behavioural findings relevant to the campaign (synthetic; probe in `parity/compute_and_dynamics_probe.py`)

**Fixture:** 20 standardised continuous columns plus 130 standardised one-hot columns from 10 categorical variables, n = 20,000, 150 columns. s shifts both the features and the label: P(y=1|s=1) = 0.52 vs P(y=1|s=0) = 0.30. Base rate 0.414; constant-predictor log-loss 0.678.

1. **The summed-loss adversary runs in a large-step regime.**
   - Its effective step on the mean loss is 0.01·n. Stable gradient descent needs roughly ≲ 6, so for n ≳ 600 the adversary oscillates instead of converging.
   - At n = 20,000 and 300 iterations, |W| reaches 48–83 for λ > 0 (379 at λ = 0, where the adversary does not affect the model) and |b| reaches about 100–200.
   - Adversary accuracy is pinned at 0.50 over the last 50 iterations.
2. **Adversary force grows roughly in proportion to n** (λ = 0.15, 100 iterations; median over iterations):

   | n | effective step | median \|W\| | median mean\|λt\| | median mean\|y−p\| |
   |---|---|---|---|---|
   | 2,000 | 20 | 2.5 | 0.045 | 0.47 |
   | 5,000 | 50 | 6.2 | 0.115 | 0.48 |
   | 20,000 | 200 | 20.7 | 0.378 | 0.48 |
   | 36,000 | 360 | 50.4 | 0.924 | 0.49 |

   So **λ does not transfer across n**. The notebook's Adult runs used about 24,129 fitting rows against the campaign's 20,000, a 1.21× factor.
3. **Logit-drift cap at λ = 0.** Each leaf is a mean of residuals in (−1, 1), so |F − F₀| ≤ lr·M = 0.01 × 300 = **3**, and probabilities stay within σ(F₀ ± 3). This bounds the achievable NLL and confidence. With λ > 0, leaves can exceed 1. FAGTB at λ = 0 is first-order gradient boosting with mean-residual leaves; it is *not* sklearn's `GradientBoostingClassifier`, which uses Newton leaves.
4. **Seed sensitivity is real algorithmic variance.** At λ = 0.06, the fitting-data positive-rate gap across adversary seeds 0/1/2 is 0.126 / 0.069 / 0.108, against 0.153 at λ = 0. Training accuracy is 0.795 / 0.784 / 0.796.
5. **High-λ cells can be degenerate.** At λ = 0.15 and n = 20,000 on this fixture, all three seeds collapse to training accuracy 0.421 (about "always positive") with training log-loss 0.739–0.743, worse than the constant predictor (0.678). The median |λt| / |y − p| is ≈ 1.0. The fixture has a strong s-label dependence, so this may not transfer to ACS. The Adult notebook point at λ = 0.15 was not degenerate.

## 9. Recommended bounded grid and compute

| Setting | Value | Justification |
|---|---|---|
| **λ** | **{0.03, 0.06, 0.15}** | All three are values the paper reports: Fig. 2 legend (0.03), Fig. 3 (0.06, 0.15), and notebook cell 13 plus Table I's 1-Unit row (0.15). They span weak to strong on the published Adult trade-off (p-rule 51.9% at 0.06; 90–96% at 0.15). The n correction (§8.2) is ×1.21, so the campaign's 0.15 corresponds to roughly the notebook's 0.124, inside the published range. The n-matched alternative {0.036, 0.072, 0.181} is recorded but not recommended, because it depends on an inferred row count. |
| Trees / learning rate | 300 / 0.01 | Notebook cell 13. The paper reports no FAGTB values, so these are the only concrete published 1-Unit settings, and their saved output reproduces Table I's 1-Unit row (84.26 / 90.36 vs 84.4 / 90.4). |
| Depth | 9 | Official effective value (L158). The notebook passed 10, which was ignored. |
| `max_features` | 20 (count) | Notebook cell 13. That is about 13% of 150 columns. |
| Tree | `friedman_mse`, `min_samples_split` 2, `min_samples_leaf` 1, `random_state` 0 | functions.py L158–163 |
| Adversary | 1 input (p); GD lr 0.01 on the summed CE; one step per iteration; update before gradient | L196–246 |
| Seeds | FAGTB `seed` = campaign seed (adversary init only; trees stay at `random_state` 0) | Captures the algorithmic variance of §8.4 without departing from the official tree configuration. |
| Early stopping / line search | None | Paper Sec. IV and code |
| Pre-registration note | A cell whose fitted model is no better than the constant predictor on MODEL_SELECT NLL should be reported as degenerate, not re-tuned. | §8.5 |

**Compute** (20,000 × 150 synthetic, one thread, Apple-silicon arm64; measured):

| Fit | Per iteration | 300 iterations | Peak RSS |
|---|---|---|---|
| Binary | 0.040 s | 12.1 s extrapolated; full runs measured at 11.4–13.0 s | 282 MiB (whole process) |
| FAGTB_MC_ADAPTATION, K=4 | 0.161 s | ≈ 48 s | — |
| FAGTB_MC_ADAPTATION, K=8 | 0.330 s | ≈ 99 s extrapolated; full run measured at **95.9 s** | 348 MiB (whole process) |
| Binary with `max_features=None` (not recommended) | 0.251 s | ≈ 75 s | — |

**Whole grid** (3 λ × 2 seeds × binary + MC):
- binary: 6 × ~13 s ≈ 1.3 min;
- MC: 6 × K × ~12 s, i.e. about 10 min for K = 8 (6 × 96 s).

That is well under 1 CPU-hour of the campaign's 20 CPU-hour budget, with peak RSS under 0.4 GiB per job (the protocol allows 6 GiB in total). Cost scales roughly linearly in n, K and M.

## 10. FAGTB_MC_ADAPTATION: specification and what blocks a valid adaptation

**Specification** (`FAGTBMulticlassAdaptation`):
- **Init:** F₀,k = log(n_k / n).
- **Each iteration:**
  1. Compute P = softmax(F).
  2. Take one gradient-descent step (lr 0.01, **summed** CE) on a logistic regression of binary SEX on the K-vector P (w ∈ ℝᴷ, scalar b).
  3. Compute the input gradient G = (σ(P·w + b) − s)·w using the updated adversary.
  4. Push it through the softmax Jacobian: ∂L_A,i/∂F_ik = P_ik (G_ik − Σⱼ G_ij P_ij). The residual is U = (Y − P) + λ·∂L_A/∂F = −∂J/∂F, with J = Σ CE_task − λ Σ CE_adv.
  5. Fit K official-configuration trees, one per class column, at the same F.
  6. Update F[:, k] += lr·h_k(X).
- **Leaves:** plain gradient leaves, with no multinomial Newton step.

**Properties** (tested):
- **K=2, λ=0:** equals binary FAGTB with 2·lr, exact per step.
- **K=2, λ>0:** not the binary algorithm. The adversary becomes a 3-parameter overparameterisation (logit = (w₁ − w₀)p + (w₀ + b)), which follows a different gradient-descent path.

**What blocks a valid adaptation.** None of these block running it; items 1–3 block *calling it a replication*:
1. **No official or published multiclass formulation, and no λ calibration.** Results must be labelled FAGTB_MC_ADAPTATION.
2. **λ is not comparable to binary λ.** The softmax-Jacobian factor P_ik(w_k − w·P_i) and the K trees per iteration change the scale. Using the binary grid for MC is a registered *convention*, not a transfer.
3. **The adversary sees the whole K-probability vector.** That pushes for independence of the entire predicted distribution from S (a demographic-parity analogue on scores), which is stronger than parity of argmax decisions. The campaign's fairness and recovery metrics should be read with that in mind.
4. **Every class must occur in the fitting labels.** F₀ uses log-priors; a missing class raises an error rather than producing −∞.
5. **It inherits the binary caveats.** These are the summed-loss n-dependence and oscillation, the per-class logit-drift cap (|F_k − F₀,k| ≤ 3 at λ = 0), and the seed sensitivity.
6. **Compute grows with K** (K trees per iteration).

## 11. Limitations of this audit

- **Unverified external facts.** No real data was read. The Adult fitting-row count comes from Friedler et al. The link between the notebook and Table I is inferred from matching numbers, because the FAGTB training cell has no saved output.
- **Environment differences.** Parity used TF 2.20 and sklearn 1.8, not the authors' 2020 TF and sklearn 0.22. The 0.22 leaf threshold of 1e-7 cannot be reproduced, and TF seed-7 values may differ across TF builds.
- **Parity scope.** Parity covered n ≤ 4000 and M ≤ 300. Per-step (teacher-forced) agreement is scale-free. Free-running exactness ends at the first tie flip for any implementation.
- **Timing.** Timings come from one Apple-silicon machine on one thread.
- **Fixture dependence.** The synthetic dynamics in §8 depend on the fixture.

## 12. Files

- `acs_replication/fagtb.py`: port and adaptation.
- `acs_replication/test_fagtb.py`: tests.
- `results/pcrl_acs_protected_tree_replication_v1/baselines/FAGTB_AUDIT.md`: this file.
- Local only, not committed (`.private/` is gitignored), under `.private/acs_replication/vendor/`:
  - `FAGTB/`: pinned official clone;
  - `parity_venv/`: the parity environment;
  - `parity/official_parity.py`: parity harness;
  - `parity/compute_and_dynamics_probe.py`: synthetic compute and dynamics probe;
  - `parity/parity_result*.json`, `parity/compute_probe.json`: results.
- **Hazard noticed (not this role's files):** `acs_replication/queue.py` and `acs_replication/select.py` shadow standard-library modules whenever a script inside `acs_replication/` is run directly, because Python puts the script's directory first on `sys.path`. This breaks `import subprocess`. `test_fagtb.py` guards against it in script mode.
