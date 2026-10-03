"""Thin wrapper around the OFFICIAL FARE implementation (Jovanovic et al., ICML 2023; github.com/eth-sri/fare).

Nothing here re-implements FARE. The fair tree is the official ``sktree`` patch of scikit-learn (criterion
``fair_gini_dp``), built in an isolated Python 3.9 environment; the certificate is the official
``code/src/tree/alphabeta_adversary.py::AlphaBetaAdversary``, loaded from the pinned clone and hash-checked.
See results/combined_output_aware_removal_v1/notes/fare/FARE_METHOD_NOTES.md for the method, the certificate
premises and every adaptation.

Execution model
  The project venv (Python 3.13) cannot import the patched scikit-learn. Every call that touches official code
  therefore runs in the FARE environment, either in-process (when this module is imported there) or as a
  subprocess ``<fare-python> -m oar.fare_official --worker <job_dir>``. Inputs travel as .npz files with a
  per-operation whitelist of array names (``encode`` accepts only ``X``). The worker routes fd 1 to /dev/null
  because the compiled splitter printf()s every split.

Locations (overridable by environment variables)
  OAR_FARE_PYTHON       default ~/PCRL_eval_cache_private/oar_v1/env_fare/bin/python
  OAR_FARE_ROOT         default ~/PCRL_eval_cache_private/oar_v1/vendor/fare            (pinned clone)
  OAR_FARE_SKLEARN_SRC  default ~/PCRL_eval_cache_private/oar_v1/vendor/scikit-learn    (patched build source)

Public API
  FareConfig(max_leaf_nodes, min_samples_leaf, gamma, criterion="fair_gini_dp", name="")
  FareConfig.zero_fairness() -> FareConfig      (same k-bar / n_i budget, gamma = 0, same official criterion)
  fit(X_fit, y_task, s, config, seed) -> FareModel
  fit_zero_fairness_control(X_fit, y_task, s, config, seed) -> FareModel
  encode(model, X) -> int32 cell ids in [0, n_cells)      (features only; official DecisionTreeClassifier.apply)
  embed(model, X) -> float64 (n, d) per-cell fit-row medians (the official FARE representation z_i)
  encode_portable(model, X) -> same cell ids from the stored tree arrays in pure numpy (cross-check only)
  cell_table(model) -> list of per-cell dicts (fit counts, decision rule)
  certificate(model, X_cert, s_cert, delta=0.05, groups=None, ...) -> dict(bound, pairs, premises, ...)
  FareModel.save(dir) / FareModel.load(dir) / FareModel.fingerprint
  verify_official_fare() -> dict
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import io
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------------------------------------------
# Pins (notes/fare/FARE_ENV_RECIPE.md)
# --------------------------------------------------------------------------------------------------------------
FARE_REPO_URL = "https://github.com/eth-sri/fare"
FARE_COMMIT = "89cb1b66ed268c16659cbf7428c43e60da2df641"  # == HEAD of main on 2026-10-03 == durable-guarantees pin
FARE_GIT_TREE = "2e06b0fdd00a6c59ebfe89527436c1b1af21c281"
# sha256 of the manifest "<sha256(file)>  <path>\n" over `git ls-files '*.py'`, paths sorted with LC_ALL=C (109 files)
FARE_PY_TREE_SHA256 = "56a447007fb963090d5f7ea65de3d8b69b9bcddf1669e86234ebab4a52d89cbd"
# same construction over `git ls-files 'sktree/*'` (14 files: .py/.pyx/.pxd/build.sh)
FARE_SKTREE_TREE_SHA256 = "2b0c24d89e416fc0b3ff5bef5e7773978485259f3b29de68e0a658aa437c139e"
FARE_ALPHABETA_SHA256 = "85138f6db01b3dd01edc0a7f347f230b7f0fc62f18874facfe98572ee7cacbcd"
FARE_SKTREE_CLASSES_SHA256 = "80756cee2132c77df9281cb87e4dbd4c7273f0bcc9f72e25232d541fe70eb94b"
SKLEARN_BASE_COMMIT = "fd60379f95f5c0d3791b2f54c4d070c0aa2ac576"  # fare/install.sh
SKLEARN_ZIP_SHA256 = "0020f3075c4a30833574544689c0edf88540126c2f13bed74673137ad94f829d"
SKLEARN_EXPECTED_VERSION = "1.2.dev0"
CRITERION_PYX_OFFICIAL_SHA256 = "ec12afa8f7f2f54ad9b1bf61768d1aca337fb4c9cbb7c2f05d4ea6f9ddd6acd4"
# The single adaptation of compiled code: sensitive-count buffers sized max(#y classes, #s groups) (FARE_ENV_RECIPE.md)
CRITERION_PYX_FIXED_SHA256 = "4fa73adee14f3f2e515725d9abceca0a158371bdf4a55eeafd891efce0127f47"

ENV_PYTHON, ENV_ROOT, ENV_SKSRC = "OAR_FARE_PYTHON", "OAR_FARE_ROOT", "OAR_FARE_SKLEARN_SRC"
DEFAULT_FARE_PYTHON = "~/PCRL_eval_cache_private/oar_v1/env_fare/bin/python"
DEFAULT_FARE_ROOT = "~/PCRL_eval_cache_private/oar_v1/vendor/fare"
DEFAULT_SKLEARN_SRC = "~/PCRL_eval_cache_private/oar_v1/vendor/scikit-learn"

# Official src/tree/main.py:33 hard-codes random_state=43. seed s -> random_state 43 + s (seed 0 == official).
RANDOM_STATE_OFFSET = 43
# Paper Sec. 5 / App. E and main.py:355,390: eps = 0.05 = eps_b (0.005) + eps_c (0.04) + eps_s (0.005).
PAPER_DELTA = 0.05
EPS_B_FRACTION = 0.1
EPS_S_FRACTION = 0.1
CERT_METHOD = "cp"  # Clopper-Pearson, as main.py:390/425

SCHEMA = "oar.fare_official.model/v1"
# grid entries may carry these descriptive keys; they never reach the official code
CONFIG_ANNOTATION_KEYS = {"id", "range", "rationale", "source", "label"}
_ALLOWED_INPUTS = {"fit": {"X", "y", "s"}, "encode": {"X"}, "embed": {"X"}, "certificate": {"X", "s"},
                   "verify": set()}
_REPO_ROOT = Path(__file__).resolve().parent.parent


class CertificateRefused(RuntimeError):
    """The certificate premises are violated by the supplied rows (e.g. they include fit rows)."""


# --------------------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FareConfig:
    """Official FARE tree hyperparameters (paper App. E names / official CLI names).

    max_leaf_nodes   k-bar, upper bound on the number of cells          (--max-k  -> max_leaf_nodes)
    min_samples_leaf n_i, lower bound on fit rows per cell              (--min-ni -> min_samples_leaf)
    gamma            fairness weight in FairGini = (1-g) Gini_y + g (0.5 - Gini_s)   (--alpha -> fit(alpha=))
    criterion        official criterion; only 'fair_gini_dp' (demographic parity) is admitted here
    """

    max_leaf_nodes: int
    min_samples_leaf: int
    gamma: float
    criterion: str = "fair_gini_dp"
    name: str = ""

    def validate(self) -> "FareConfig":
        if self.criterion != "fair_gini_dp":
            raise ValueError("only the official 'fair_gini_dp' criterion is admitted")
        if int(self.max_leaf_nodes) != self.max_leaf_nodes or self.max_leaf_nodes < 2:
            raise ValueError("max_leaf_nodes must be an integer >= 2")
        if int(self.min_samples_leaf) != self.min_samples_leaf or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")
        if not (0.0 <= float(self.gamma) <= 1.0):
            raise ValueError("gamma must lie in [0, 1]")
        return self

    def zero_fairness(self) -> "FareConfig":
        """Same k-bar / n_i budget and the same official criterion with the fairness weight set to 0.

        Official behaviour at gamma = 0 (sktree/_criterion.pyx FairGiniDP.node_impurity / children_impurity):
        impurity = (1 - 0) * Gini_y + 0 * (0.5 - Gini_s) = Gini_y exactly, i.e. the plain Gini tree; s is still
        counted but carries zero weight. Verified equal to criterion='gini' in tests (test_b2).
        """
        base = self.name or "cfg"
        return replace(self, gamma=0.0, name=f"{base}__gamma0")

    @classmethod
    def from_any(cls, cfg: Any) -> "FareConfig":
        if isinstance(cfg, FareConfig):
            return cfg.validate()
        if isinstance(cfg, dict):
            keys = {"max_leaf_nodes", "min_samples_leaf", "gamma", "criterion", "name"}
            extra = set(cfg) - keys - CONFIG_ANNOTATION_KEYS
            if extra:
                raise ValueError(f"unknown FARE config keys: {sorted(extra)}")
            return cls(**{k: v for k, v in cfg.items() if k in keys}).validate()
        raise TypeError("config must be a FareConfig or a dict")


# --------------------------------------------------------------------------------------------------------------
# Model container (portable across the two environments; no per-row labels are stored)
# --------------------------------------------------------------------------------------------------------------
class FareModel:
    """A fitted official FARE tree.

    meta            JSON-able: config, seed, pins, tree arrays, leaf->cell map, per-cell fit medians and per-cell
                    AGGREGATE fit counts by task class and by group (used by cell_table and the Lemma 5.1 base rate).
    tree_pickle     pickle of the official sklearn estimator (only loadable in the FARE environment).
    fit_row_hashes  sorted unique 64-bit hashes of the fit feature rows; read ONLY by certificate() to refuse
                    fit rows. encode() never reads it.
    runtime         timings / peak memory of the fit call (not saved, not part of the fingerprint).
    """

    def __init__(self, meta: Dict[str, Any], tree_pickle: bytes, fit_row_hashes: np.ndarray,
                 runtime: Optional[Dict[str, Any]] = None):
        self.meta = meta
        self.tree_pickle = tree_pickle
        self.fit_row_hashes = np.asarray(fit_row_hashes, dtype=np.uint64)
        self.runtime = runtime or {}

    # ---- convenience -------------------------------------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return int(len(self.meta["leaf_node_ids"]))

    @property
    def n_features(self) -> int:
        return int(self.meta["n_features"])

    @property
    def config(self) -> FareConfig:
        return FareConfig(**self.meta["config"])

    @property
    def fingerprint(self) -> str:
        blob = json.dumps(self.meta, sort_keys=True, separators=(",", ":")).encode()
        h = hashlib.sha256(blob)
        h.update(self.fit_row_hashes.tobytes())
        return h.hexdigest()

    # ---- persistence -------------------------------------------------------------------------------------------
    def save(self, directory) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "tree.pkl").write_bytes(self.tree_pickle)
        np.save(d / "fit_row_hashes.npy", self.fit_row_hashes, allow_pickle=False)
        manifest = {"schema": SCHEMA, "fingerprint": self.fingerprint,
                    "tree_pkl_sha256": hashlib.sha256(self.tree_pickle).hexdigest()}
        (d / "model.json").write_text(json.dumps(self.meta, sort_keys=True, indent=1))
        (d / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1))
        return d

    @classmethod
    def load(cls, directory) -> "FareModel":
        d = Path(directory)
        meta = json.loads((d / "model.json").read_text())
        manifest = json.loads((d / "manifest.json").read_text())
        if manifest.get("schema") != SCHEMA or meta.get("schema") != SCHEMA:
            raise ValueError("not an oar.fare_official model directory")
        tree_pickle = (d / "tree.pkl").read_bytes()
        if hashlib.sha256(tree_pickle).hexdigest() != manifest["tree_pkl_sha256"]:
            raise ValueError("tree.pkl does not match its manifest hash")
        model = cls(meta, tree_pickle, np.load(d / "fit_row_hashes.npy", allow_pickle=False))
        if model.fingerprint != manifest["fingerprint"]:
            raise ValueError("model fingerprint mismatch after load")
        return model


# --------------------------------------------------------------------------------------------------------------
# Helpers usable in both environments
# --------------------------------------------------------------------------------------------------------------
def _path_from_env(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default)).expanduser()


def fare_python() -> Path:
    return _path_from_env(ENV_PYTHON, DEFAULT_FARE_PYTHON)


def fare_root() -> Path:
    return _path_from_env(ENV_ROOT, DEFAULT_FARE_ROOT)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _check_features(X, n_features: Optional[int] = None, name: str = "X") -> np.ndarray:
    """Features only: a plain 2-D finite float array. Rejects dicts, record arrays, objects, labels-in-columns."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray of features (got {type(X).__name__})")
    if X.dtype.names is not None or X.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be a plain numeric feature matrix (no structured/object arrays)")
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"{name} has {X.shape[1]} columns; the model was fit on {n_features} features")
    X = np.ascontiguousarray(X, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains non-finite values")
    return X


def _check_codes(v, n: int, name: str) -> np.ndarray:
    v = np.asarray(v)
    if v.ndim != 1 or v.shape[0] != n:
        raise ValueError(f"{name} must be 1-D with {n} entries")
    if v.dtype.kind not in "iub":
        if v.dtype.kind == "f" and np.all(v == np.round(v)):
            v = v.astype(np.int64)
        else:
            raise TypeError(f"{name} must hold integer codes")
    return v.astype(np.int64)


def row_hashes(X: np.ndarray) -> np.ndarray:
    """Sorted unique 64-bit blake2b hashes of float64 feature rows (exact-byte identity)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    out = np.empty(X.shape[0], dtype=np.uint64)
    for i in range(X.shape[0]):
        out[i] = int.from_bytes(hashlib.blake2b(X[i].tobytes(), digest_size=8).digest(), "little")
    return np.unique(out)


def encode_portable(model: FareModel, X: np.ndarray) -> np.ndarray:
    """Pure-numpy traversal of the stored official tree arrays (cross-check of ``encode``; features only).

    Mirrors sktree/_tree.pyx::_apply_dense for continuous features: X is cast to float32 (sklearn DTYPE) and a
    row goes left iff x[feature] <= threshold.
    """
    X = _check_features(X, model.n_features)
    Xf = X.astype(np.float32).astype(np.float64)
    m = model.meta
    left, right = np.asarray(m["children_left"]), np.asarray(m["children_right"])
    feat, thr = np.asarray(m["feature"]), np.asarray(m["threshold"], dtype=np.float64)
    node = np.zeros(X.shape[0], dtype=np.int64)
    active = left[node] != -1
    while np.any(active):
        idx = np.nonzero(active)[0]
        nd = node[idx]
        go_left = Xf[idx, feat[nd]] <= thr[nd]
        node[idx] = np.where(go_left, left[nd], right[nd])
        active = left[node] != -1
    leaf_ids = np.asarray(m["leaf_node_ids"])
    pos = np.searchsorted(leaf_ids, node)
    if np.any(pos >= len(leaf_ids)) or np.any(leaf_ids[np.minimum(pos, len(leaf_ids) - 1)] != node):
        raise RuntimeError("row reached a leaf that is not a FARE cell")
    return pos.astype(np.int32)


def cell_table(model: FareModel, include_medians: bool = False) -> List[Dict[str, Any]]:
    """Per-cell table from AGGREGATE fit statistics (no per-row data)."""
    m = model.meta
    left, right = m["children_left"], m["children_right"]
    feat, thr = m["feature"], m["threshold"]
    parent: Dict[int, tuple] = {}
    for nd in range(len(left)):
        if left[nd] != -1:
            parent[left[nd]] = (nd, "<=")
            parent[right[nd]] = (nd, ">")
    rows = []
    for c, leaf in enumerate(m["leaf_node_ids"]):
        rule = []
        nd = leaf
        while nd in parent:
            p, side = parent[nd]
            rule.append({"feature": int(feat[p]), "op": side, "threshold": float(thr[p])})
            nd = p
        rule.reverse()
        row = {"cell": c, "leaf_node": int(leaf),
               "n_fit": int(sum(m["cell_task_counts"][c])),
               "fit_task_counts": dict(zip([int(t) for t in m["task_classes"]], m["cell_task_counts"][c])),
               "fit_group_counts": dict(zip([int(g) for g in m["group_codes"]], m["cell_group_counts"][c])),
               "rule": rule}
        if include_medians:
            row["median"] = list(m["medians"][c])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------------------------
# Implementations that call official code (FARE environment only)
# --------------------------------------------------------------------------------------------------------------
def in_fare_env() -> bool:
    try:
        from sklearn.tree import DecisionTreeClassifier  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return False
    params = inspect.signature(DecisionTreeClassifier.fit).parameters
    return "s" in params and "cat_pos" in params and "alpha" in params


def _c_fflush() -> None:
    try:
        import ctypes
        ctypes.CDLL(None).fflush(None)
    except Exception:  # noqa: BLE001
        pass


@contextlib.contextmanager
def _quiet_fd1():
    """Route fd 1 to /dev/null (the compiled splitter printf()s every split); Python stdout too."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        sys.stdout.flush()
        _c_fflush()  # C stdio buffers the printf output; flush it into /dev/null before restoring fd 1
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def _load_alphabeta():
    path = fare_root() / "code" / "src" / "tree" / "alphabeta_adversary.py"
    if _sha256_file(path) != FARE_ALPHABETA_SHA256:
        raise RuntimeError("official alphabeta_adversary.py does not match the pinned hash")
    spec = importlib.util.spec_from_file_location("fare_official_alphabeta_adversary", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.AlphaBetaAdversary


def _serialize_tree(T) -> bytes:
    """Pickle the official estimator WITHOUT relying on the patched Tree.__reduce__, which is broken upstream:
    sktree/_tree.pyx:708-712 passes 4 constructor arguments while Tree.__cinit__ (:668) requires 6 (cat_pos,
    cat_maxval), so pickle.loads fails. We store the estimator shell, the 6 constructor arguments and the
    official Tree.__getstate__() dict, and rebuild with Tree(*args).__setstate__(state) (the same two official
    methods a working __reduce__ would call). Only categorical-free trees are produced here (cat_pos = [])."""
    import copy
    import pickle

    from sklearn.tree import _tree

    tr = T.tree_
    shell = copy.copy(T)
    del shell.tree_
    n_feat = int(tr.n_features)
    ctor = (n_feat, np.asarray(tr.n_classes, dtype=np.intp), np.asarray(tr.n_sens, dtype=np.intp),
            int(tr.n_outputs), np.zeros(n_feat, dtype=_tree.BOOL), np.zeros(n_feat, dtype=_tree.INT))
    return pickle.dumps({"shell": shell, "ctor": ctor, "state": tr.__getstate__()}, protocol=4)


def _tree_from_pickle(model: FareModel):
    import pickle

    from sklearn.tree._tree import Tree

    d = pickle.loads(model.tree_pickle)
    tree = Tree(*d["ctor"])
    tree.__setstate__(d["state"])
    est = d["shell"]
    est.tree_ = tree
    return est


def _fit_impl(X, y, s, cfg: FareConfig, seed: int) -> FareModel:
    import pickle

    import sklearn
    from sklearn.tree import DecisionTreeClassifier

    t_all = time.perf_counter()
    task_classes, y_enc = np.unique(y, return_inverse=True)
    group_codes, s_enc = np.unique(s, return_inverse=True)
    if len(task_classes) < 2 or len(group_codes) < 2:
        raise ValueError("need at least two task classes and two groups in the fit rows")
    # s is passed as contiguous codes 0..G-1: the official criterion indexes count buffers by the raw s value and
    # sizes them by the number of unique values (_classes.py: n_sens_ = len(np.unique(s))).
    T = DecisionTreeClassifier(criterion=cfg.criterion, max_leaf_nodes=int(cfg.max_leaf_nodes),
                               min_samples_leaf=int(cfg.min_samples_leaf),
                               random_state=RANDOM_STATE_OFFSET + int(seed))
    t0 = time.perf_counter()
    with _quiet_fd1():
        T.fit(X, y_enc.reshape(-1, 1), s_enc.reshape(-1, 1).astype(np.float64),
              cat_pos=np.asarray([], dtype=np.int32), alpha=float(cfg.gamma))
    t_fit = time.perf_counter() - t0
    leaves = T.apply(X.astype(np.float32))
    leaf_ids = np.unique(leaves)
    nb_cells = int((T.tree_.children_left == -1).sum())
    # official eval() (main.py:98-103) asserts every leaf is reached by the training rows
    assert len(leaf_ids) == nb_cells, "official invariant: every leaf is reached by fit rows"
    cell = np.searchsorted(leaf_ids, leaves)
    k, G, C = len(leaf_ids), len(group_codes), len(task_classes)
    medians = np.zeros((k, X.shape[1]))
    for c in range(k):
        medians[c] = np.median(X[cell == c], axis=0)  # main.py:116-126 (continuous features -> median)
    cell_group = np.zeros((k, G), dtype=np.int64)
    np.add.at(cell_group, (cell, s_enc), 1)
    cell_task = np.zeros((k, C), dtype=np.int64)
    np.add.at(cell_task, (cell, y_enc), 1)
    tr = T.tree_
    meta = {
        "schema": SCHEMA,
        "config": asdict(cfg),
        "seed": int(seed),
        "random_state": RANDOM_STATE_OFFSET + int(seed),
        "n_features": int(X.shape[1]),
        "n_fit": int(X.shape[0]),
        "task_classes": [int(v) for v in task_classes],
        "group_codes": [int(v) for v in group_codes],
        "leaf_node_ids": [int(v) for v in leaf_ids],
        "medians": medians.tolist(),
        "cell_group_counts": cell_group.tolist(),
        "cell_task_counts": cell_task.tolist(),
        "children_left": [int(v) for v in tr.children_left],
        "children_right": [int(v) for v in tr.children_right],
        "feature": [int(v) for v in tr.feature],
        "threshold": [float(v) for v in tr.threshold],
        "n_node_samples": [int(v) for v in tr.n_node_samples],
        "impurity": [float(v) for v in tr.impurity],
        "max_depth": int(tr.max_depth),
        "pins": {"fare_commit": FARE_COMMIT, "fare_py_tree_sha256": FARE_PY_TREE_SHA256,
                 "sklearn_base_commit": SKLEARN_BASE_COMMIT, "sklearn_version": sklearn.__version__,
                 "criterion_pyx_sha256": CRITERION_PYX_FIXED_SHA256, "numpy_version": np.__version__,
                 "python": sys.version.split()[0]},
    }
    model = FareModel(meta, _serialize_tree(T), row_hashes(X))
    model.runtime = {"tree_fit_seconds": t_fit, "fit_total_seconds": time.perf_counter() - t_all}
    return model


def _encode_impl(model: FareModel, X: np.ndarray) -> np.ndarray:
    T = _tree_from_pickle(model)
    leaves = T.apply(X.astype(np.float32))  # official encode(): T.apply (main.py:129-134)
    leaf_ids = np.asarray(model.meta["leaf_node_ids"])
    pos = np.searchsorted(leaf_ids, leaves)
    if np.any(pos >= len(leaf_ids)) or np.any(leaf_ids[np.minimum(pos, len(leaf_ids) - 1)] != leaves):
        raise RuntimeError("row reached a leaf that is not a FARE cell")
    return pos.astype(np.int32)


def official_pair_bound(AB, k: int, z_tr, c_tr, z_va, c_va, z_te, c_te, eps_pair: float, eps_b: float,
                        eps_s: float) -> Dict[str, Any]:
    """One call of the official AlphaBetaAdversary.ub_demographic_parity on a binary-coded group pair.

    Arguments are passed exactly as main.py builds them (c_* as (n,1); 0 = first group of the pair, 1 = second).
    Returns the bound 2*T-1 and the official empirical 2*BA-1 of the optimal cell adversary on the D_test rows.
    """
    emb = {"z_train": np.asarray(z_tr, dtype=np.float64), "c_train": np.asarray(c_tr).reshape(-1, 1),
           "z_val": np.asarray(z_va, dtype=np.float64), "c_val": np.asarray(c_va).reshape(-1, 1),
           "z_test": np.asarray(z_te, dtype=np.float64), "c_test": np.asarray(c_te).reshape(-1, 1)}
    adv = AB(k, eps_pair, eps_glob=eps_s, eps_ab=eps_b, method=CERT_METHOD, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        ret = adv.ub_demographic_parity(emb)
        emp = adv.empirical(emb["z_test"], emb["c_test"])
    return {"ub": float(ret[0]), "empirical_test": float(emp)}


def _certificate_impl(model: FareModel, X: np.ndarray, s: np.ndarray, delta: float, groups: Sequence[int],
                      split_seed: int, val_fraction: float, eps_b_fraction: float,
                      eps_s_fraction: float) -> Dict[str, Any]:
    AB = _load_alphabeta()
    t0 = time.perf_counter()
    overlap = int(np.isin(row_hashes(X), model.fit_row_hashes).sum())
    if overlap:
        raise CertificateRefused(f"{overlap} certificate rows are identical to fit rows; the certificate needs rows "
                                 "never seen by the tree (paper Sec. 5, D_val/D_test)")
    k = model.n_cells
    cells = _encode_impl(model, X).astype(np.float64).reshape(-1, 1)
    n = X.shape[0]
    perm = np.random.RandomState(split_seed).permutation(n)
    n_val = int(round(val_fraction * n))
    va, te = perm[:n_val], perm[n_val:]
    # Lemma 5.1 base rates from the fit rows, as main.py (z_train = tree training rows); rebuilt from the stored
    # aggregate per-cell group counts (the official code only uses those counts).
    gc = np.asarray(model.meta["cell_group_counts"])
    gcodes = list(model.meta["group_codes"])
    pairs = list(itertools.combinations(sorted(int(g) for g in groups), 2))
    n_pairs = len(pairs)
    eps_pair = delta / n_pairs
    eps_b = eps_pair / (1.0 / eps_b_fraction)
    eps_s = eps_pair / (1.0 / eps_s_fraction)
    eps_c = eps_pair - eps_b - eps_s
    if not (eps_b > 0 and eps_s > 0 and eps_c > 0):
        raise ValueError("budget decomposition must leave eps_b, eps_c, eps_s > 0")
    out_pairs = []
    for gi, gj in pairs:
        rec: Dict[str, Any] = {"groups": [gi, gj], "eps_pair": eps_pair, "eps_b": eps_b, "eps_c": eps_c,
                               "eps_s": eps_s}
        if gi not in gcodes or gj not in gcodes:
            rec.update(status="unavailable", reason="group absent from fit rows", ub=None)
            out_pairs.append(rec)
            continue
        ci, cj = gcodes.index(gi), gcodes.index(gj)
        z_tr = np.concatenate([np.repeat(np.arange(k), gc[:, ci]), np.repeat(np.arange(k), gc[:, cj])])
        c_tr = np.concatenate([np.zeros(gc[:, ci].sum(), int), np.ones(gc[:, cj].sum(), int)])
        split = {}
        for nm, idx in (("val", va), ("test", te)):
            m = (s[idx] == gi) | (s[idx] == gj)
            split[nm] = (cells[idx][m], (s[idx][m] == gj).astype(int))
        rec.update(n_base=int(len(c_tr)), n_val=int(len(split["val"][1])), n_test=int(len(split["test"][1])))
        missing = {nm: int(k - len(np.unique(zz))) for nm, (zz, _) in
                   (("base", (z_tr, None)), ("val", split["val"]), ("test", split["test"]))}
        rec["cells_missing"] = missing
        rec["min_cell_n_val"] = int(np.bincount(split["val"][0].ravel().astype(int), minlength=k).min()) \
            if rec["n_val"] else 0
        try:
            r = official_pair_bound(AB, k, z_tr.reshape(-1, 1), c_tr, split["val"][0], split["val"][1],
                                    split["test"][0], split["test"][1], eps_pair, eps_b, eps_s)
        except AssertionError:
            rec.update(status="unavailable", ub=None,
                       reason="official assertion: every cell must appear in base, D_val and D_test rows of the "
                              "pair (alphabeta_adversary.py:121,140,209,233)")
            out_pairs.append(rec)
            continue
        if not np.isfinite(r["ub"]):
            rec.update(status="unavailable", ub=None, reason="official bound is not finite")
        else:
            rec.update(status="ok", **r)
        out_pairs.append(rec)
    ok = all(p["status"] == "ok" for p in out_pairs)
    bound = max(p["ub"] for p in out_pairs) if ok else None
    return {
        "status": "ok" if ok else "unavailable",
        "bound": bound,
        "bound_is_vacuous": (bound is not None and bound >= 1.0),
        "metric": "demographic-parity distance of ANY binary classifier of the FARE cell (max over group pairs)",
        "delta": delta, "confidence": 1.0 - delta, "method": "official AlphaBetaAdversary(method='cp')",
        "n_cells": k, "pairs": out_pairs,
        "premises": {
            "fit_rows_excluded_by_feature_hash": True, "n_cert_rows": int(n), "n_overlap_with_fit_rows": overlap,
            "split": {"seed": int(split_seed), "val_fraction": float(val_fraction), "n_val": int(len(va)),
                      "n_test": int(len(te)), "lemma_5_2_rows": "D_val", "lemma_5_3_rows": "D_test"},
            "base_rate_rows": "fit rows (official: Lemma 5.1 on D_train; q(s) does not depend on the encoder)",
            "budget": {"delta": delta, "pairs": n_pairs, "per_pair": eps_pair, "eps_b": eps_b, "eps_c": eps_c,
                       "eps_s": eps_s, "union_bound": "delta = sum over pairs; per pair eps_b+eps_c+eps_s"},
            "groups": [int(g) for g in groups],
            "stated_not_checked": [
                "base, D_val and D_test rows are independent draws from the distribution the bound refers to",
                "the encoder was fixed before D_val/D_test were drawn and never saw them (enforced only for "
                "exact feature-row duplicates; rows sharing a unit/person with fit rows must be removed by "
                "the caller)",
                "the bound covers classifiers that see only the FARE representation (cell id / median), not "
                "classifiers that also see other channels (e.g. clean model outputs)"],
        },
        "seconds": time.perf_counter() - t0,
    }


def _verify_impl() -> Dict[str, Any]:
    import sklearn
    from sklearn.tree import DecisionTreeClassifier
    import sklearn.tree._classes as skc

    root = fare_root()
    sk_src = _path_from_env(ENV_SKSRC, DEFAULT_SKLEARN_SRC)
    res: Dict[str, Any] = {"python": sys.version.split()[0], "numpy": np.__version__,
                           "sklearn": sklearn.__version__}
    res["sklearn_version_ok"] = sklearn.__version__ == SKLEARN_EXPECTED_VERSION
    res["fit_signature_patched"] = in_fare_env()
    res["installed_classes_is_official"] = _sha256_file(Path(skc.__file__)) == FARE_SKTREE_CLASSES_SHA256
    res["alphabeta_hash_ok"] = _sha256_file(root / "code/src/tree/alphabeta_adversary.py") == FARE_ALPHABETA_SHA256
    crit = sk_src / "sklearn" / "tree" / "_criterion.pyx"
    res["criterion_pyx_is_fixed_build_source"] = crit.exists() and _sha256_file(crit) == CRITERION_PYX_FIXED_SHA256
    official_crit = root / "sktree" / "_criterion.pyx"
    res["official_criterion_pyx_hash_ok"] = _sha256_file(official_crit) == CRITERION_PYX_OFFICIAL_SHA256
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        head = None
    res["fare_head"] = head
    res["fare_commit_ok"] = head == FARE_COMMIT
    # runtime probe: 5 groups > 2 task classes; three sequential fits must be identical (unfixed build is not)
    rng = np.random.RandomState(0)
    X = rng.randn(3000, 5)
    s = rng.randint(0, 5, 3000)
    y = (X[:, 0] + 0.4 * s + rng.randn(3000) > 1).astype(int)
    outs = set()
    for _ in range(3):
        T = DecisionTreeClassifier(criterion="fair_gini_dp", max_leaf_nodes=8, min_samples_leaf=50, random_state=43)
        with _quiet_fd1():
            T.fit(X, y.reshape(-1, 1), s.reshape(-1, 1).astype(float), cat_pos=np.asarray([], np.int32), alpha=0.5)
        outs.add(hashlib.sha256(T.apply(X.astype(np.float32)).tobytes()).hexdigest())
    res["multigroup_probe_deterministic"] = len(outs) == 1
    res["ok"] = all(res[k] for k in ("sklearn_version_ok", "fit_signature_patched", "installed_classes_is_official",
                                     "alphabeta_hash_ok", "criterion_pyx_is_fixed_build_source",
                                     "official_criterion_pyx_hash_ok", "fare_commit_ok",
                                     "multigroup_probe_deterministic"))
    return res


def _dispatch(op: str, arrays: Dict[str, np.ndarray], params: Dict[str, Any], model: Optional[FareModel]):
    if set(arrays) - _ALLOWED_INPUTS[op]:
        raise PermissionError(f"operation {op!r} may only receive {sorted(_ALLOWED_INPUTS[op])}")
    if op == "fit":
        cfg = FareConfig(**params["config"]).validate()
        return {}, _fit_impl(arrays["X"], arrays["y"], arrays["s"], cfg, int(params["seed"]))
    if op == "encode":
        t0 = time.perf_counter()
        cells = _encode_impl(model, arrays["X"])
        return {"cells": cells, "_encode_seconds": time.perf_counter() - t0}, None
    if op == "embed":
        cells = _encode_impl(model, arrays["X"])
        return {"z": np.asarray(model.meta["medians"], dtype=np.float64)[cells]}, None
    if op == "certificate":
        return {"result": _certificate_impl(model, arrays["X"], arrays["s"], **params)}, None
    if op == "verify":
        return {"result": _verify_impl()}, None
    raise ValueError(op)


def _peak_rss_bytes() -> int:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def _worker_main(job_dir: str) -> int:
    jd = Path(job_dir)
    job = json.loads((jd / "job.json").read_text())
    arrays = {}
    if (jd / "in.npz").exists():
        with np.load(jd / "in.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    model = FareModel.load(jd / "model_in") if (jd / "model_in").exists() else None
    t0, c0 = time.perf_counter(), time.process_time()
    with _quiet_fd1():
        outputs, new_model = _dispatch(job["op"], arrays, job.get("params", {}), model)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    result = {"op": job["op"], "worker_seconds": wall, "worker_cpu_seconds": cpu, "peak_rss_bytes": _peak_rss_bytes()}
    arr_out = {}
    for k, v in outputs.items():
        if isinstance(v, np.ndarray):
            arr_out[k] = v
        else:
            result[k] = v
    if arr_out:
        np.savez(jd / "out.npz", **arr_out)
    if new_model is not None:
        new_model.save(jd / "model_out")
        result["runtime"] = new_model.runtime
    (jd / "out.json").write_text(json.dumps(result, default=_json_default))
    return 0


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# --------------------------------------------------------------------------------------------------------------
# Router: in-process inside the FARE env, otherwise a subprocess in the FARE env
# --------------------------------------------------------------------------------------------------------------
LAST_CALL: Dict[str, Any] = {}


def _run(op: str, arrays: Dict[str, np.ndarray], params: Dict[str, Any], model: Optional[FareModel] = None,
         workdir: Optional[str] = None):
    if set(arrays) - _ALLOWED_INPUTS[op]:
        raise PermissionError(f"operation {op!r} may only receive {sorted(_ALLOWED_INPUTS[op])}")
    LAST_CALL.clear()
    LAST_CALL.update(op=op, input_keys=sorted(arrays))
    if in_fare_env() and os.environ.get("OAR_FARE_FORCE_SUBPROCESS") != "1":
        t0, c0 = time.perf_counter(), time.process_time()
        outputs, new_model = _dispatch(op, arrays, params, model)
        LAST_CALL.update(mode="inprocess", seconds=time.perf_counter() - t0,
                         worker_cpu_seconds=time.process_time() - c0)
        if new_model is not None:
            LAST_CALL["runtime"] = new_model.runtime
        return outputs, new_model
    py = fare_python()
    if not py.exists():
        raise FileNotFoundError(f"FARE environment python not found at {py} (set {ENV_PYTHON})")
    jd = Path(tempfile.mkdtemp(prefix="oar_fare_", dir=workdir))
    try:
        if arrays:
            np.savez(jd / "in.npz", **arrays)
        (jd / "job.json").write_text(json.dumps({"op": op, "params": params}, default=_json_default))
        if model is not None:
            model.save(jd / "model_in")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env.setdefault(v, "1")
        t0 = time.perf_counter()
        proc = subprocess.run([str(py), "-m", "oar.fare_official", "--worker", str(jd)], cwd=str(_REPO_ROOT),
                              env=env, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-3000:]
            if "CertificateRefused" in tail:
                raise CertificateRefused(tail.strip().splitlines()[-1])
            raise RuntimeError(f"FARE worker failed (rc={proc.returncode}):\n{tail}")
        result = json.loads((jd / "out.json").read_text())
        outputs = dict(result)
        if (jd / "out.npz").exists():
            with np.load(jd / "out.npz", allow_pickle=False) as z:
                outputs.update({k: z[k] for k in z.files})
        new_model = None
        if (jd / "model_out").exists():
            new_model = FareModel.load(jd / "model_out")
            new_model.runtime = dict(result.get("runtime", {}))
            new_model.runtime.update(worker_seconds=result["worker_seconds"],
                                     worker_cpu_seconds=result["worker_cpu_seconds"],
                                     peak_rss_bytes=result["peak_rss_bytes"], subprocess_wall_seconds=wall)
        LAST_CALL.update(mode="subprocess", seconds=wall, worker_seconds=result["worker_seconds"],
                         worker_cpu_seconds=result["worker_cpu_seconds"], peak_rss_bytes=result["peak_rss_bytes"])
        return outputs, new_model
    finally:
        shutil.rmtree(jd, ignore_errors=True)


# --------------------------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------------------------
def fit(X_fit: np.ndarray, y_task: np.ndarray, s: np.ndarray, config, seed: int, *,
        workdir: Optional[str] = None) -> FareModel:
    """Fit the official FARE fair tree on (X_fit, y_task, s). Every fit row is used to grow the tree.

    The certificate needs rows the tree never saw: hold them out before calling fit (paper: a fraction v of the
    training data, App. E; official main.py:184-197).
    """
    cfg = FareConfig.from_any(config)
    X = _check_features(X_fit, name="X_fit")
    y = _check_codes(y_task, X.shape[0], "y_task")
    g = _check_codes(s, X.shape[0], "s")
    _, model = _run("fit", {"X": X, "y": y, "s": g}, {"config": asdict(cfg), "seed": int(seed)},
                    workdir=workdir)
    return model


def fit_zero_fairness_control(X_fit, y_task, s, config, seed: int, **kw) -> FareModel:
    """Same official criterion and k-bar / n_i budget with gamma = 0 (see FareConfig.zero_fairness)."""
    return fit(X_fit, y_task, s, FareConfig.from_any(config).zero_fairness(), seed, **kw)


def encode(model: FareModel, X: np.ndarray) -> np.ndarray:
    """Cell ids from FEATURES ONLY (official DecisionTreeClassifier.apply on the pickled official tree).

    Structural label-freedom: the signature is (model, X); X must be a plain 2-D numeric matrix with exactly the
    fit width; the worker whitelist for 'encode' is {'X'}; the model carries no per-row labels, and the only
    per-row state it holds (fit-row hashes) is not read here.
    """
    X = _check_features(X, model.n_features)
    out, _ = _run("encode", {"X": X}, {}, model)
    return np.asarray(out["cells"], dtype=np.int32)


def embed(model: FareModel, X: np.ndarray) -> np.ndarray:
    """The official FARE representation z = per-cell fit-row median (main.py:110-134). Features only."""
    X = _check_features(X, model.n_features)
    out, _ = _run("embed", {"X": X}, {}, model)
    return np.asarray(out["z"], dtype=np.float64)


def certificate(model: FareModel, X_cert: np.ndarray, s_cert: np.ndarray, delta: float = PAPER_DELTA, *,
                groups: Optional[Sequence[int]] = None, split_seed: int = 0, val_fraction: float = 0.5,
                eps_b_fraction: float = EPS_B_FRACTION, eps_s_fraction: float = EPS_S_FRACTION,
                workdir: Optional[str] = None) -> Dict[str, Any]:
    """Official FARE DP-distance certificate on held-out rows (paper Sec. 5, App. D.1; main.py:354-430).

    X_cert / s_cert must be rows the tree never saw. They are split at random (split_seed) into D_val (Lemma 5.2,
    per-cell bounds) and D_test (Lemma 5.3, Hoeffding sum); Lemma 5.1 (base rates) uses the fit rows' aggregate
    group counts, as the official code does. For >2 groups the official procedure is run on every pair with
    delta / #pairs per pair and the bound is the max over pairs; the per-pair decomposition keeps the paper's
    proportions (eps_b = eps_s = 10%, eps_c = 80%). Refuses (CertificateRefused) any row identical to a fit row.
    """
    X = _check_features(X_cert, model.n_features, name="X_cert")
    sg = _check_codes(s_cert, X.shape[0], "s_cert")
    overlap = int(np.isin(row_hashes(X), model.fit_row_hashes).sum())
    if overlap:
        raise CertificateRefused(f"{overlap} certificate rows are identical to fit rows; refusing")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1)")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must lie in (0, 1)")
    grp = sorted(int(g) for g in (groups if groups is not None else model.meta["group_codes"]))
    if len(grp) < 2:
        raise ValueError("need at least two groups")
    keep = np.isin(sg, grp)
    params = {"delta": float(delta), "groups": grp, "split_seed": int(split_seed),
              "val_fraction": float(val_fraction), "eps_b_fraction": float(eps_b_fraction),
              "eps_s_fraction": float(eps_s_fraction)}
    out, _ = _run("certificate", {"X": X[keep], "s": sg[keep]}, params, model, workdir=workdir)
    res = out["result"]
    res["premises"]["n_cert_rows_supplied"] = int(X.shape[0])
    res["premises"]["n_cert_rows_outside_groups_dropped"] = int((~keep).sum())
    return res


def verify_official_fare() -> Dict[str, Any]:
    out, _ = _run("verify", {}, {})
    return out["result"]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        sys.exit(_worker_main(sys.argv[2]))
    print(json.dumps(verify_official_fare(), indent=1))
