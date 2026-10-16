"""Fair Adversarial Gradient Tree Boosting (FAGTB): faithful repaired binary port + multiclass adaptation.

Official source audited (pinned, unmodified copy kept under .private/acs_replication/vendor/FAGTB):
    github.com/vincent-grari/FAGTB @ 0cd0bb100d9258790e8fe47b3a47cc6170eac652
    functions.py sha256 30bd91eccb6e90f263614b95b7f06e7620e57784e8845ee1aa87bbc6116f0a5d (class FAGTB)
    FAGTB.ipynb  sha256 b6b430de7ce871418252ceadd7e14e182c709ca94456c284370e3429fd22e0a2 (author runs)
Paper: V. Grari, B. Ruf, S. Lamprier, M. Detyniecki, "Fair Adversarial Gradient Tree Boosting",
IEEE ICDM 2019, arXiv:1911.05369 (Sec. III, Algorithm 1; Sec. IV).
Full audit, repair/adaptation table and test record:
    results/pcrl_acs_protected_tree_replication_v1/baselines/FAGTB_AUDIT.md

What the official binary algorithm does (functions.py, FAGTB.fit, lines 178-265), reproduced here:
    F_0 = log(sum y / sum (1 - y))                                    (line 184)
    for m = 0 .. n_estimators - 1:
        p   = 1 / (1 + exp(-F))                                       (line 229)
        one gradient-descent step, learning rate 0.01, on the SUM over rows of
        sigmoid-cross-entropy(logit = W * p + b, label = s)           (lines 200-212, 232)
        g   = d[sum CE]/d p  evaluated with the UPDATED (W, b)        (lines 219, 237)
        t   = -(g * p * (1 - p))                                      (lines 242-244)
        u   = y - p - lam * t                                         (line 246, proj == 0)
        h_m = DecisionTreeRegressor(friedman_mse, max_depth=9, random_state=0, ...).fit(X, u)
        F  += learning_rate * h_m(X)                                  (lines 247-250)
    predict(X) = sigmoid(F_0 + sum_m learning_rate * h_m(X))          (lines 267-275)

This is the demographic-parity variant of the paper: the adversary's only input is
p = P_F(Y=1|X) (Sec. III-A), a single logistic unit ("FAGTB-1-Unit", Sec. IV), one adversary
update per boosting iteration, fixed boosting learning rate instead of the line search of
Algorithm 1 step (e) (Sec. IV). Since sum_i CE_i is separable, u_i = -d/dF_i [L_F - lam * L_A]
exactly, i.e. the residual of the paper's min-max objective (eq. 6) with u = r - lam * t.

Engineering repairs (behaviour-preserving; see the audit for the full table):
  * TensorFlow-1 graph adversary replaced by closed-form numpy (float64 instead of float32).
    The adversary's initial (W, b) are two numpy N(0, 1) draws from a fixed seed. The official
    TF ``random_normal(seed=7)`` gives one fixed N(0, 1) value used for both W and b
    (``OFFICIAL_TF_SEED7_ADVERSARY_INIT``, injectable via ``adversary_init``). This is a repair, not
    a scientific change. (Paper Sec. III-B says "Xavier initialization"; the code uses std-1 normal,
    and the port follows the code.)
  * ``min_impurity_split=None`` dropped from the tree constructor (removed in scikit-learn 1.0).
  * ``fit`` takes only fitting arrays: the official ``Xtest, yt, sensitivet`` arguments (used only
    for monitoring prints) are removed. Per-iteration diagnostics use fitting data only.
  * ``predict`` rewritten (the official ``np.full(n, Init, Init)`` passes a value as the dtype;
    it works only because ``Init`` is an ``np.float64``).
  * Unused imports (fairness, fire, torch, tensorflow, pandas) are not required.
  * ``max_depth`` is exposed. The default (9) is the value the official code actually uses: it
    hard-codes ``max_depth=9`` and ignores its own ``max_depth`` argument (the notebook passes 10).

FAGTB_MC_ADAPTATION (class ``FAGTBMulticlassAdaptation``) is NOT an official implementation. The
paper and code are binary-only. It is a documented softmax extension written for this campaign.
"""
from __future__ import annotations

import numbers

import numpy as np
from sklearn.tree import DecisionTreeRegressor

__all__ = [
    "FAGTBBinary",
    "FAGTBMulticlassAdaptation",
    "official_tree",
    "binary_adversary_param_grad",
    "binary_adversary_input_grad",
    "binary_pseudo_residual",
    "multiclass_adversary_param_grad",
    "multiclass_adversary_input_grad",
    "multiclass_pseudo_residual",
    "softmax",
]

OFFICIAL_REPO = "https://github.com/vincent-grari/FAGTB"
OFFICIAL_COMMIT = "0cd0bb100d9258790e8fe47b3a47cc6170eac652"
OFFICIAL_FUNCTIONS_SHA256 = "30bd91eccb6e90f263614b95b7f06e7620e57784e8845ee1aa87bbc6116f0a5d"
OFFICIAL_NOTEBOOK_SHA256 = "b6b430de7ce871418252ceadd7e14e182c709ca94456c284370e3429fd22e0a2"

OFFICIAL_MAX_DEPTH_USED = 9  # functions.py line 158: DecisionTreeRegressor(..., max_depth=9, ...)
NOTEBOOK_MAX_DEPTH_PASSED = 10  # FAGTB.ipynb cell 13: FAGTB(max_depth=10) -- ignored by the class
OFFICIAL_ADVERSARY_LR = 0.01  # functions.py line 200: learning_rate2 = 0.01
OFFICIAL_TREE_RANDOM_STATE = 0  # functions.py line 163: random_state=0 for every tree
NOTEBOOK_N_ESTIMATORS = 300  # FAGTB.ipynb cell 13
NOTEBOOK_LEARNING_RATE = 0.01  # FAGTB.ipynb cell 13
NOTEBOOK_MAX_FEATURES = 20  # FAGTB.ipynb cell 13
NOTEBOOK_LAMBDA = 0.15  # FAGTB.ipynb cell 13
# Initial (W1, b1) of the official adversary: tf.random_normal(shape, seed=7) for BOTH variables with the
# graph-level seed unset (functions.py lines 197, 206-207). Both ops get the same (graph, op) seed pair, so
# they draw the SAME value, and every official run starts from this one fixed pair. Measured under
# TensorFlow 2.20.0 CPU (macOS arm64) by the parity harness. Not verified on the authors' 2020 TF build.
# Pass as ``adversary_init`` to start the port exactly where the official code starts.
OFFICIAL_TF_SEED7_ADVERSARY_INIT = (-0.046945683658123016, -0.046945683658123016)

_CLIP = 1e-15  # functions.py lossgr / FAGTB.gradient clip


def official_tree(max_depth=OFFICIAL_MAX_DEPTH_USED, max_features=NOTEBOOK_MAX_FEATURES,
                  random_state=OFFICIAL_TREE_RANDOM_STATE):
    """Regression tree configured as in functions.py lines 158-163.

    The only change is that ``min_impurity_split=None`` is dropped, because the argument was removed
    in scikit-learn 1.0. With None, sklearn 0.22 used an internal leaf-impurity threshold of 1e-7;
    0.23 used 0; scikit-learn >= 1.0 uses machine epsilon. That threshold cannot be set in modern
    sklearn (see the audit).
    """
    return DecisionTreeRegressor(
        criterion="friedman_mse", max_depth=max_depth, max_features=max_features,
        max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=random_state)


def _sigmoid(z):
    """1 / (1 + exp(-z)), the expression the official code uses for p (overflow -> exact 0)."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_ce_with_logits(logits, labels):
    """Elementwise TF ``sigmoid_cross_entropy_with_logits``: max(a,0) - a*z + log(1 + exp(-|a|))."""
    return np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))


def softmax(F):
    """Row-wise softmax with max-subtraction."""
    Z = F - F.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------------
# Binary (official) adversary and residual
# --------------------------------------------------------------------------------------------

def binary_adversary_param_grad(p, s, w, b):
    """Gradient of sum_i CE(sigmoid(w * p_i + b), s_i) w.r.t. (w, b).

    This is what TF's GradientDescentOptimizer(0.01).minimize(reduce_sum(...)) applies
    (functions.py lines 209-212). It is a SUM over rows, so the effective step on the per-row mean
    loss is 0.01 * n.
    """
    r = _sigmoid(w * p + b) - s
    return float(np.sum(r * p)), float(np.sum(r))


def binary_adversary_input_grad(p, s, w, b):
    """d/dp_i of sum_j CE(sigmoid(w * p_j + b), s_j) = (sigmoid(w * p_i + b) - s_i) * w.

    This is TF ``tf.gradients(loss, X_input)`` (functions.py line 219).
    """
    return (_sigmoid(w * p + b) - s) * w


def _binary_residual_terms(p, y, s, w, b, lam):
    """(u, t) from p = sigmoid(F): same operations and order as functions.py lines 242-246."""
    g = binary_adversary_input_grad(p, s, w, b)
    lfadv = g * p * (1.0 - p)
    t = -lfadv
    return y - p - lam * t, t


def binary_pseudo_residual(F, y, s, w, b, lam):
    """Official FAGTB residual u = y - p - lam * t, with t = -(dL_A/dp) * p * (1 - p).

    Mirrors functions.py lines 229, 242-246 (proj == 0). Equals -d/dF_i [L_F,i - lam * L_A,i]
    where L_F is the log-loss and L_A the adversary's cross-entropy. (w, b) must be the adversary
    parameters AFTER this iteration's update (the official order). ``FAGTBBinary.fit`` uses the
    same code path.
    """
    return _binary_residual_terms(_sigmoid(F), y, s, w, b, lam)[0]


# --------------------------------------------------------------------------------------------
# Multiclass adaptation (NOT official) adversary and residual
# --------------------------------------------------------------------------------------------

def multiclass_adversary_param_grad(P, s, w, b):
    """FAGTB_MC_ADAPTATION: gradient of sum_i CE(sigmoid(P_i . w + b), s_i) w.r.t. (w, b)."""
    r = _sigmoid(P @ w + b) - s
    return P.T @ r, float(np.sum(r))


def multiclass_adversary_input_grad(P, s, w, b):
    """FAGTB_MC_ADAPTATION: d/dP_ik of sum_j CE(sigmoid(P_j . w + b), s_j) = (q_i - s_i) * w_k."""
    return (_sigmoid(P @ w + b) - s)[:, None] * w[None, :]


def multiclass_pseudo_residual(F, Y, s, w, b, lam):
    """FAGTB_MC_ADAPTATION residual U = (Y - P) - lam * T, with T = -dL_A/dF via the softmax Jacobian.

    dL_A,i/dF_ik = P_ik * (G_ik - sum_j G_ij P_ij), with G = dL_A/dP. This makes
    U = -dJ/dF for J = sum_i [CE_task,i - lam * CE_adv,i]. Y is one-hot (n, K), and (w, b) are the
    adversary parameters after this iteration's update.
    """
    return _multiclass_residual_terms(softmax(F), Y, s, w, b, lam)[0]


def _multiclass_residual_terms(P, Y, s, w, b, lam):
    """(U, T) from P = softmax(F); shared by ``multiclass_pseudo_residual`` and the class's fit."""
    G = multiclass_adversary_input_grad(P, s, w, b)
    dLA_dF = P * (G - np.sum(G * P, axis=1, keepdims=True))
    T = -dLA_dF
    return Y - P - lam * T, T


# --------------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------------

def _check_X(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"X must be a non-empty 2-D array, got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    return X


def _check_binary(v, name, n):
    v = np.asarray(v)
    if v.ndim != 1 or v.shape[0] != n:
        raise ValueError(f"{name} must be 1-D with {n} entries, got shape {v.shape}")
    vf = v.astype(np.float64)
    if not np.all((vf == 0.0) | (vf == 1.0)):
        raise ValueError(f"{name} must be binary 0/1")
    return vf


def _check_lam(lam):
    if isinstance(lam, bool) or not isinstance(lam, numbers.Real) or not np.isfinite(lam) or lam < 0:
        raise ValueError(f"lam must be a finite real >= 0, got {lam!r}")
    return float(lam)


def _check_hparams(n_estimators, learning_rate, adversary_lr):
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, numbers.Integral) or n_estimators < 1:
        raise ValueError(f"n_estimators must be a positive integer, got {n_estimators!r}")
    for name, v in (("learning_rate", learning_rate), ("adversary_lr", adversary_lr)):
        if not isinstance(v, numbers.Real) or not np.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be a finite real > 0, got {v!r}")


class _Unfitted:
    def _check_fitted(self):
        if not hasattr(self, "estimators_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit(X, y, s, lam) first")


# --------------------------------------------------------------------------------------------
# Binary official algorithm
# --------------------------------------------------------------------------------------------

class FAGTBBinary(_Unfitted):
    """Faithful, repaired port of the official binary FAGTB (functions.py class FAGTB).

    Parameters
    ----------
    n_estimators, learning_rate : boosting iterations and fixed shrinkage. Notebook: 300 and 0.01.
    max_depth : tree depth. Default 9 is the value the official code actually uses (hard-coded at
        functions.py line 158). The notebook passes 10, which the class ignores.
    max_features : per-split feature subsample of the regression tree, passed through as in the
        official code. Notebook: the integer 20.
    adversary_lr : adversary gradient-descent learning rate on the SUMMED cross-entropy. Official: 0.01.
        The sum makes the effective step on the mean loss 0.01 * n, so the adversary's strength
        depends on the number of fitting rows.
    seed : seeds the numpy draw of the adversary's initial (w, b) ~ N(0, 1). This replaces TF
        ``random_normal(seed=7)`` (repair). It does not affect the trees.
    adversary_init : optional explicit (w0, b0) that overrides ``seed``, e.g.
        ``OFFICIAL_TF_SEED7_ADVERSARY_INIT`` to start exactly where the official code starts.
    tree_random_state : random_state of every tree. Official: 0 for all trees.
    Non-default ``adversary_lr`` or ``tree_random_state`` values are deviations from the official code.

    fit(X, y, s, lam) accepts only fitting arrays. ``history_`` holds per-iteration diagnostics on the
    fitting data only: task_logloss and train_accuracy (after the tree update); adversary_loss_sum,
    adversary_loss_mean, adversary_accuracy, adversary_w and adversary_b (after the adversary step);
    task_residual_abs_mean = mean|y - p|; adversary_term_abs_mean = mean|lam * t|.
    """

    def __init__(self, n_estimators=NOTEBOOK_N_ESTIMATORS, learning_rate=NOTEBOOK_LEARNING_RATE,
                 max_depth=OFFICIAL_MAX_DEPTH_USED, max_features=NOTEBOOK_MAX_FEATURES,
                 adversary_lr=OFFICIAL_ADVERSARY_LR, seed=0, adversary_init=None,
                 tree_random_state=OFFICIAL_TREE_RANDOM_STATE):
        _check_hparams(n_estimators, learning_rate, adversary_lr)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.max_features = max_features
        self.adversary_lr = float(adversary_lr)
        self.seed = seed
        self.adversary_init = adversary_init
        self.tree_random_state = tree_random_state

    def _initial_adversary(self):
        if self.adversary_init is not None:
            w0, b0 = self.adversary_init
            return float(w0), float(b0)
        rng = np.random.default_rng(self.seed)
        w0, b0 = rng.standard_normal(2)
        return float(w0), float(b0)

    def fit(self, X, y, s, lam):
        """Fit on fitting arrays only. X: (n, d) features; y: binary task label; s: binary sensitive."""
        X = _check_X(X)
        n = X.shape[0]
        y = _check_binary(y, "y", n)
        s = _check_binary(s, "s", n)
        lam = _check_lam(lam)
        n_pos, n_neg = float(np.sum(y)), float(np.sum(1.0 - y))
        if n_pos == 0.0 or n_neg == 0.0:
            raise ValueError("y must contain both classes (the official log-odds init is undefined otherwise)")

        self.init_ = float(np.log(n_pos / n_neg))  # functions.py line 184
        F = np.full(n, self.init_)
        w, b = self._initial_adversary()
        self.adversary_init_ = (w, b)
        lr, alr = self.learning_rate, self.adversary_lr
        hist = {k: [] for k in ("task_logloss", "train_accuracy", "adversary_loss_sum",
                                "adversary_loss_mean", "adversary_accuracy", "adversary_w",
                                "adversary_b", "task_residual_abs_mean", "adversary_term_abs_mean")}
        trees = []
        for _ in range(self.n_estimators):
            p = _sigmoid(F)
            # (1) one adversary GD step on the summed CE at the current predictions (lines 229-232)
            gw, gb = binary_adversary_param_grad(p, s, w, b)
            w, b = w - alr * gw, b - alr * gb
            a = w * p + b  # adversary diagnostics after the step (official cur_loss / train_acc)
            adv_loss_sum = float(np.sum(_sigmoid_ce_with_logits(a, s)))
            hist["adversary_loss_sum"].append(adv_loss_sum)
            hist["adversary_loss_mean"].append(adv_loss_sum / n)
            hist["adversary_accuracy"].append(float(np.mean((_sigmoid(a) >= 0.5) == (s == 1.0))))
            hist["adversary_w"].append(w)
            hist["adversary_b"].append(b)
            # (2) input gradient with the UPDATED adversary, combined residual (lines 237-246)
            u, t = _binary_residual_terms(p, y, s, w, b, lam)
            hist["task_residual_abs_mean"].append(float(np.mean(np.abs(y - p))))
            hist["adversary_term_abs_mean"].append(float(np.mean(np.abs(lam * t))))
            # (3) regression tree on the residual, fixed-shrinkage update (lines 247-250)
            tree = official_tree(self.max_depth, self.max_features, self.tree_random_state)
            tree.fit(X, u)
            F = F + lr * tree.predict(X)
            trees.append(tree)
            pf = np.clip(_sigmoid(F), _CLIP, 1.0 - _CLIP)  # official lossgr (line 253)
            hist["task_logloss"].append(float(np.mean(-y * np.log(pf) - (1.0 - y) * np.log(1.0 - pf))))
            hist["train_accuracy"].append(float(np.mean((_sigmoid(F) > 0.5) == (y == 1.0))))
        self.estimators_ = trees
        self.adversary_params_ = (w, b)
        self.history_ = {k: np.asarray(v, dtype=np.float64) for k, v in hist.items()}
        self.n_features_in_ = X.shape[1]
        self.lam_ = lam
        return self

    def decision_function(self, X):
        """Raw score F(X) = F_0 + sum_m learning_rate * h_m(X), accumulated in the official order."""
        self._check_fitted()
        X = _check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, fitted with {self.n_features_in_}")
        F = np.full(X.shape[0], self.init_)
        for tree in self.estimators_:
            F = F + self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        """(n, 2) array [P(Y=0), P(Y=1)]. Column 1 equals the official ``FAGTB.predict(X)``."""
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])


# --------------------------------------------------------------------------------------------
# FAGTB_MC_ADAPTATION -- documented softmax extension, NOT an official implementation
# --------------------------------------------------------------------------------------------

class FAGTBMulticlassAdaptation(_Unfitted):
    """FAGTB_MC_ADAPTATION: softmax extension of FAGTB. NOT an official implementation.

    The paper (arXiv:1911.05369) and the official code (functions.py) are binary-only. This class
    is a campaign adaptation that keeps every official design choice that still applies:

      * F_0,k = log(n_k / n) (class log-priors; for K=2 the margin F_0,1 - F_0,0 equals the official
        log-odds init). Every class must occur in the fitting labels.
      * Each iteration: P = softmax(F). One gradient-descent step (adversary_lr, default 0.01) on the
        SUM over rows of CE(sigmoid(P_i . w + b), s_i). The adversary is a logistic regression of the
        binary sensitive attribute on the K-probability vector (w in R^K, scalar b; the input is
        collinear with the bias since sum_k P_ik = 1, which is harmless).
      * Then the adversary input gradient with the UPDATED (w, b) is pushed through the softmax
        Jacobian. Residual U = (Y - P) - lam * T with T = -dL_A/dF, i.e. U = -dJ/dF for
        J = sum_i [CE_task,i - lam * CE_adv,i].
      * K regression trees per iteration (one per class column, same configuration as the official
        tree, all with tree_random_state), all fit at the same F, and F[:, k] += learning_rate * h_k(X).
        Plain gradient leaves (no multinomial Newton step), as in the official binary code.

    Known behaviour: for K=2 and lam=0 the margin F_1 - F_0 equals FAGTBBinary with learning_rate
    doubled. Per step this is exact: U_0 = -U_1 to 1 ulp, U_1 is the binary residual, and the tree
    learner is sign-symmetric. Over many iterations with deep trees it holds only up to CART tie-breaking
    (exactly tied splits are resolved by floating-point rounding). For lam > 0 the K=2 adversary is a
    3-parameter overparameterisation of the binary one (logit = (w_1 - w_0) p + (w_0 + b)). Its
    gradient-descent path therefore differs, so K=2 here is NOT the official binary algorithm.
    """

    def __init__(self, n_estimators=NOTEBOOK_N_ESTIMATORS, learning_rate=NOTEBOOK_LEARNING_RATE,
                 max_depth=OFFICIAL_MAX_DEPTH_USED, max_features=NOTEBOOK_MAX_FEATURES,
                 adversary_lr=OFFICIAL_ADVERSARY_LR, seed=0, adversary_init=None,
                 tree_random_state=OFFICIAL_TREE_RANDOM_STATE, n_classes=None):
        _check_hparams(n_estimators, learning_rate, adversary_lr)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.max_features = max_features
        self.adversary_lr = float(adversary_lr)
        self.seed = seed
        self.adversary_init = adversary_init
        self.tree_random_state = tree_random_state
        self.n_classes = n_classes

    def _initial_adversary(self, K):
        if self.adversary_init is not None:
            w0, b0 = self.adversary_init
            w0 = np.asarray(w0, dtype=np.float64).reshape(-1)
            if w0.shape != (K,):
                raise ValueError(f"adversary_init weight must have shape ({K},), got {w0.shape}")
            return w0.copy(), float(b0)
        rng = np.random.default_rng(self.seed)
        w0 = rng.standard_normal(K)
        b0 = float(rng.standard_normal())
        return w0, b0

    def fit(self, X, y, s, lam):
        """Fit on fitting arrays only. y: integer labels in {0..K-1}; s: binary sensitive."""
        X = _check_X(X)
        n = X.shape[0]
        y = np.asarray(y)
        if y.ndim != 1 or y.shape[0] != n:
            raise ValueError(f"y must be 1-D with {n} entries, got shape {y.shape}")
        if not np.issubdtype(y.dtype, np.integer):
            yf = y.astype(np.float64)
            if not np.all(yf == np.round(yf)):
                raise ValueError("y must hold integer class labels")
            y = yf.astype(np.int64)
        if y.min() < 0:
            raise ValueError("y labels must be >= 0")
        K = int(self.n_classes) if self.n_classes is not None else int(y.max()) + 1
        if K < 2 or y.max() >= K:
            raise ValueError(f"need K >= 2 and labels < K; got K={K}, max label {int(y.max())}")
        counts = np.bincount(y, minlength=K).astype(np.float64)
        if np.any(counts == 0):
            raise ValueError(f"every class must occur in the fitting labels; counts={counts.astype(int).tolist()}")
        s = _check_binary(s, "s", n)
        lam = _check_lam(lam)

        Y = np.zeros((n, K))
        Y[np.arange(n), y] = 1.0
        self.n_classes_ = K
        self.init_ = np.log(counts / n)
        F = np.tile(self.init_, (n, 1))
        w, b = self._initial_adversary(K)
        self.adversary_init_ = (w.copy(), b)
        lr, alr = self.learning_rate, self.adversary_lr
        hist = {k: [] for k in ("task_ce", "train_accuracy", "adversary_loss_sum", "adversary_loss_mean",
                                "adversary_accuracy", "adversary_b", "task_residual_abs_mean",
                                "adversary_term_abs_mean")}
        hist_w = []
        rounds = []
        for _ in range(self.n_estimators):
            P = softmax(F)
            gw, gb = multiclass_adversary_param_grad(P, s, w, b)
            w, b = w - alr * gw, b - alr * gb
            a = P @ w + b
            adv_loss_sum = float(np.sum(_sigmoid_ce_with_logits(a, s)))
            hist["adversary_loss_sum"].append(adv_loss_sum)
            hist["adversary_loss_mean"].append(adv_loss_sum / n)
            hist["adversary_accuracy"].append(float(np.mean((_sigmoid(a) >= 0.5) == (s == 1.0))))
            hist["adversary_b"].append(b)
            hist_w.append(w.copy())
            U, T = _multiclass_residual_terms(P, Y, s, w, b, lam)
            hist["task_residual_abs_mean"].append(float(np.mean(np.abs(Y - P))))
            hist["adversary_term_abs_mean"].append(float(np.mean(np.abs(lam * T))))
            round_trees = []
            for k in range(K):  # all K residual columns were computed at the same F
                tree = official_tree(self.max_depth, self.max_features, self.tree_random_state)
                tree.fit(X, U[:, k])
                F[:, k] = F[:, k] + lr * tree.predict(X)
                round_trees.append(tree)
            rounds.append(round_trees)
            Pn = np.clip(softmax(F), _CLIP, 1.0)
            hist["task_ce"].append(float(np.mean(-np.log(Pn[np.arange(n), y]))))
            hist["train_accuracy"].append(float(np.mean(np.argmax(F, axis=1) == y)))
        self.estimators_ = rounds
        self.adversary_params_ = (w.copy(), b)
        self.history_ = {k: np.asarray(v, dtype=np.float64) for k, v in hist.items()}
        self.history_["adversary_w"] = np.asarray(hist_w, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self.lam_ = lam
        return self

    def decision_function(self, X):
        """Raw scores F(X), shape (n, K), accumulated in the same order as during fit."""
        self._check_fitted()
        X = _check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, fitted with {self.n_features_in_}")
        F = np.tile(self.init_, (X.shape[0], 1))
        for round_trees in self.estimators_:
            for k, tree in enumerate(round_trees):
                F[:, k] = F[:, k] + self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        """(n, K) class probabilities softmax(F(X)); rows sum to 1."""
        return softmax(self.decision_function(X))
