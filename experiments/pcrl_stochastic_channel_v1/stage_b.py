"""Stage B — the unconstrained within-family capacity diagnostic (gate G2).

Question: before any privacy optimization, does a small prespecified A-side code `T` retain a
residence advantage over the **same-host** untouched-J release? If it does not, the representation
family is exhausted at the tested resolution and the study stops (REGISTRATION G2).

Two measurements, deliberately different in kind.

* **B1, fitted-model ceiling.** The largest residence log-loss reduction *any* function of `T` could
  deliver alongside a binned baseline view, estimated by household-cross-fitted tables. This is a
  ceiling for the fitted finite model and this input code — not for arbitrary predictors, and not
  for the ACS population.
* **B2, deployable probe.** The house utility probe family (`logistic` + `mlp`, selection by
  validation log loss) on `[H_A, Z_J, onehot(T)]` against `[H_A, Z_J]`, on the historical subset
  indices so the comparison is same-host. This is what a real release would actually deliver.

The quantizer sees **no labels of any kind** — not residence, not protected. It is fitted on the
standardised A-side inference inputs (`PCA_32` through J's frozen standardiser) over
`representation_fit` rows. Residence labels enter only the designated probes, which is what makes
this a development diagnostic and never a reserved-task claim.
"""
from __future__ import annotations

import time

import numpy as np

from experiments.pcrl_direct_adversarial_v1 import inputs as dax

N_CODES = 64                      # declared resolution
N_CODES_FALLBACK = 256            # the one predeclared higher-resolution fallback
RESIDENCE = 'same_residence'
QUANTIZER_SALT = 20269020         # fixed before any outcome


# ------------------------------------------------------------------ the code
def fit_code(x_fit: np.ndarray, n_codes: int, seed: int):
    """Prespecified A-side quantizer: k-means on the permitted inference inputs. No labels.

    Returned object exposes `assign(x)` only, so nothing downstream can reach inside it.
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_codes, n_init=10, random_state=QUANTIZER_SALT + seed).fit(x_fit)

    class Code:
        n = n_codes
        centers = km.cluster_centers_
        inertia = float(km.inertia_)

        @staticmethod
        def assign(x):
            return km.predict(np.asarray(x, np.float64)).astype(np.int64)

    return Code


def onehot(t, n):
    out = np.zeros((len(t), n), dtype=np.float64)
    out[np.arange(len(t)), t] = 1.0
    return out


# ------------------------------------------------------------------ B1: fitted-model ceiling
def _bin_view(v: np.ndarray, edges: int, ref: np.ndarray) -> np.ndarray:
    """Coarsen a continuous view into a product of per-column quantile bins, then relabel densely.

    The partition is declared, not tuned: `edges` equal-count bins per column of the baseline view,
    computed on the reference rows only.
    """
    codes = np.zeros(len(v), dtype=np.int64)
    for j in range(v.shape[1]):
        q = np.quantile(ref[:, j], np.linspace(0, 1, edges + 1)[1:-1])
        codes = codes * edges + np.digitize(v[:, j], q)
    _, dense = np.unique(codes, return_inverse=True)
    return dense.astype(np.int64)


def ceiling_from_code(y, base_cells, t, fold, n_folds=5, pseudocount=0.5) -> dict:
    """Household-cross-fitted table estimate of the residence log-loss reduction from adding `T`.

    Both sides use the identical smoothing and the identical folds, so the comparison is matched.
    A positive value is an upper bound on what any function of `T` can add **in this fitted model**;
    it is not a population quantity and it is not achievable by a deployable probe in general.
    """
    y = np.asarray(y, np.int64)
    n_classes = int(y.max()) + 1
    loss_base, loss_joint = np.zeros(len(y)), np.zeros(len(y))
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if not te.any() or not tr.any():
            continue
        for cells, out in ((base_cells, loss_base), (base_cells * (t.max() + 1) + t, loss_joint)):
            _, key = np.unique(cells, return_inverse=True)
            table = np.full((key.max() + 1, n_classes), pseudocount)
            np.add.at(table, (key[tr], y[tr]), 1.0)
            table /= table.sum(1, keepdims=True)
            prior = (np.bincount(y[tr], minlength=n_classes) + pseudocount)
            prior = prior / prior.sum()
            seen = np.zeros(key.max() + 1, dtype=bool)
            seen[np.unique(key[tr])] = True
            p = np.where(seen[key[te]][:, None], table[key[te]], prior[None, :])
            out[te] = -np.log(np.maximum(p[np.arange(te.sum()), y[te]], 1e-12))
    return {'baseline_log_loss': float(loss_base.mean()),
            'joint_log_loss': float(loss_joint.mean()),
            'ceiling_reduction_nats': float(loss_base.mean() - loss_joint.mean()),
            'n_rows': int(len(y)), 'n_base_cells': int(len(np.unique(base_cells))),
            'n_joint_cells': int(len(np.unique(base_cells * (t.max() + 1) + t))),
            'scope': 'fitted finite model and this input code only; not a population bound'}


# ------------------------------------------------------------------ B2: deployable probe
def probe_residence(x_fit, y_fit, x_val, y_val, seed, weights_val=None) -> dict:
    """The house utility probe family, selected on validation log loss, as in `reusable_utilities`."""
    from experiments.acs_transfer_heads import fit_candidates, metrics
    fitted = fit_candidates(x_fit, y_fit, x_val, y_val, 2, seed,
                            families=('logistic', 'mlp'), budget={'mlp': {'epochs': 40}})
    cs = fitted['candidates']
    pick = min(cs, key=lambda k: (cs[k].metadata['validation_scores']['log_loss'], k))
    out = {'selected': pick,
           'per_family': {k: cs[k].metadata['validation_scores']['log_loss'] for k in cs},
           'log_loss': float(cs[pick].metadata['validation_scores']['log_loss'])}
    if weights_val is not None:
        out['log_loss_person_weighted'] = float(
            metrics(y_val, cs[pick].predict_proba(x_val), 2, weights=weights_val)['log_loss'])
    out['per_row_loss'] = _per_row(cs[pick], x_val, y_val)
    return out


def _per_row(candidate, x_val, y_val):
    """Per-row log loss, kept so a paired-household interval can be computed later."""
    p = np.asarray(candidate.predict_proba(x_val), np.float64)
    y = np.asarray(y_val, np.int64)
    return -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))


# ------------------------------------------------------------------ driver
def run_seed(seed: int, n_codes: int = N_CODES, edges: int = 3) -> dict:
    """One anchor of Stage B. Reads 2018 development pools. Fits nothing that sees a label."""
    from experiments.run_acs_residual_spectral import load_labels
    from experiments.run_acs_transfer import subset_indices
    from experiments.pcrl_nonlinear_rank_v1.inputs import HIST_ROOT
    from experiments.pcrl_utility_extension_v1 import extension as ext

    tick = time.perf_counter()
    registry = dax.Registry.new()
    state, j, _ = ext.portable_state(seed, registry)
    jm = j['model']
    x_all = {p: dax.standardize(state['pca'][p], jm['input_mean'], jm['input_scale'])
             for p in dax.POOLS}
    ha = {p: np.asarray(state['anchors'][f'{p}/A'], np.float64) for p in dax.POOLS}
    zj = {p: np.asarray(j['channel'][p], np.float64) for p in dax.POOLS}
    base = {p: np.column_stack([ha[p], zj[p]]) for p in dax.POOLS}   # the same-host J wire/A

    # The code: fitted on representation_fit inference inputs, label-free.
    code = fit_code(x_all['representation_fit'], n_codes, seed)
    t = {p: code.assign(x_all[p]) for p in dax.POOLS}

    # Residence labels enter only here, for the designated probe.
    frame, pools, labels, weights = load_labels(HIST_ROOT, seed)
    ti = subset_indices(labels['downstream_fit'][RESIDENCE], 2048, 1230000 + 100 * seed + 0)
    valid = labels['downstream_validation'][RESIDENCE] >= 0

    y_fit = labels['downstream_fit'][RESIDENCE][ti]
    y_val = labels['downstream_validation'][RESIDENCE][valid]
    w_val = np.asarray(weights['downstream_validation'])[valid] if weights else None

    j_fit, j_val = base['downstream_fit'][ti], base['downstream_validation'][valid]
    code_fit = np.column_stack([j_fit, onehot(t['downstream_fit'][ti], n_codes)])
    code_val = np.column_stack([j_val, onehot(t['downstream_validation'][valid], n_codes)])

    pseed = 1250000 + 100 * seed + 0
    ref_j = probe_residence(j_fit, y_fit, j_val, y_val, pseed, w_val)
    unconstrained = probe_residence(code_fit, y_fit, code_val, y_val, pseed, w_val)

    # B1 on the validation rows, household-cross-fitted, under two declared partitions.
    # The per-column product partition is reported first as declared, and is ALSO reported as
    # sparse: with 20 baseline columns it yields ~1 cell per 2 rows, which is the "condition on
    # near-unique raw H and celebrate zero conditional entropy" failure the review warned about.
    # The probability-decile partition is a low-dimensional alternative on the same rows. The
    # sparsity is a property of the partition and the data, visible without reading any outcome.
    serial_val = np.asarray(frame.iloc[pools['downstream_validation']]['SERIALNO'])[valid]
    fold = _household_fold(serial_val, 5)
    tv = t['downstream_validation'][valid]
    product_cells = _bin_view(j_val, edges, j_val)
    decile_cells = np.digitize(_baseline_score(ref_j, j_val),
                              np.quantile(_baseline_score(ref_j, j_val), np.linspace(0, 1, 11)[1:-1]))
    ceiling = {'product_partition': ceiling_from_code(y_val, product_cells, tv, fold),
               'probability_decile_partition': ceiling_from_code(y_val, decile_cells, tv, fold)}
    ceiling['product_partition']['rows_per_cell'] = float(
        len(y_val) / max(1, ceiling['product_partition']['n_base_cells']))
    ceiling['product_partition']['sparse_warning'] = bool(
        ceiling['product_partition']['rows_per_cell'] < 10)

    # Utility advantage, two accountings.
    #  * `code_only`: the probe sees [J, onehot(T)] and nothing else. With |T| = 64 appended to a
    #    20-column baseline on 2048 fitting rows, a finite learner can be HANDICAPPED by the extra
    #    width -- the same effect that motivates in-slate J predictors on the protection side.
    #  * `inclusion_respecting`: the release CONTAINS J, so its utility slate also contains the
    #    J-only predictor. This is the accounting consistent with METHOD's inclusion fact, and it is
    #    the one G2 is judged on.
    raw_adv = ref_j['log_loss'] - unconstrained['log_loss']
    best_code = min(unconstrained['log_loss'], ref_j['log_loss'])
    adv = ref_j['log_loss'] - best_code
    inclusion_row_loss = (unconstrained['per_row_loss']
                          if unconstrained['log_loss'] <= ref_j['log_loss'] else ref_j['per_row_loss'])
    out = {'seed': seed, 'n_codes': n_codes, 'bin_edges_per_column': edges,
           'quantizer': {'inertia': code.inertia, 'salt': QUANTIZER_SALT,
                         'labels_used_to_fit': 'none'},
           'code_occupancy': np.bincount(t['downstream_validation'][valid],
                                         minlength=n_codes).tolist(),
           'B2_deployable': {'ref_J_log_loss': ref_j['log_loss'],
                             'unconstrained_code_log_loss': unconstrained['log_loss'],
                             'residence_advantage_nats_code_only': float(raw_adv),
                             'residence_advantage_nats': float(adv),
                             'accounting': 'inclusion_respecting (the release contains J)',
                             'ref_J_selected': ref_j['selected'],
                             'code_selected': unconstrained['selected'],
                             'ref_J_per_family': ref_j['per_family'],
                             'code_per_family': unconstrained['per_family'],
                             'code_probe_input_dim': int(code_fit.shape[1]),
                             'ref_J_probe_input_dim': int(j_fit.shape[1])},
           'B2_interval_code_only': paired_household_interval(
               serial_val, ref_j['per_row_loss'], unconstrained['per_row_loss']),
           'B2_interval_inclusion_respecting': paired_household_interval(
               serial_val, ref_j['per_row_loss'], inclusion_row_loss),
           'B1_fitted_model_ceiling': ceiling,
           'paired': {'serial': serial_val, 'ref_J_row_loss': ref_j['per_row_loss'],
                      'code_row_loss': unconstrained['per_row_loss'],
                      'weights': w_val},
           'validation_rows': int(valid.sum()), 'fit_rows': int(len(ti)),
           'seconds': time.perf_counter() - tick,
           'split': 'validation only; no test split read'}
    if 'log_loss_person_weighted' in ref_j:
        out['B2_deployable']['residence_advantage_nats_person_weighted'] = float(
            ref_j['log_loss_person_weighted'] - unconstrained['log_loss_person_weighted'])
    return out


def paired_household_interval(serial, loss_left, loss_right, *, weights=None, n_boot=4000,
                              seed=7, z=1.6448536269514722) -> dict:
    """Candidate-specific paired-household bootstrap of `mean(left) - mean(right)`.

    The resampling unit is the **household**, because rows within a household are not independent.
    More release draws, more optimizer seeds or more fitted channels do not add units here, and a
    Monte Carlo replication count is never reported as a sample size.

    Reported whenever prescribed. It is **not** withheld because historical precision was poor
    (AMENDMENT_1.md §4): this interval belongs to this candidate and this pair.
    """
    serial = np.asarray(serial)
    d = np.asarray(loss_left, np.float64) - np.asarray(loss_right, np.float64)
    w = np.ones(len(d)) if weights is None else np.asarray(weights, np.float64)
    uniq, inv = np.unique(serial, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    num = np.array([(d[g] * w[g]).sum() for g in groups])
    den = np.array([w[g].sum() for g in groups])
    estimate = num.sum() / den.sum()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    boot = num[draws].sum(1) / den[draws].sum(1)
    se = float(boot.std(ddof=1))
    return {'estimate': float(estimate), 'bootstrap_se': se,
            'households': int(len(uniq)), 'rows': int(len(d)), 'n_boot': int(n_boot),
            'critical_value': float(z),
            'adjusted_note': 'unadjusted one-sided 95%; a family adjustment is applied at report time',
            'upper_one_sided': float(estimate + z * se),
            'lower_one_sided': float(estimate - z * se),
            'unit': 'household'}


def _baseline_score(probe: dict, x) -> np.ndarray:
    """The baseline probe's per-row loss, used only as a 1-d partition coordinate for B1."""
    return np.asarray(probe['per_row_loss'], np.float64)


def _household_fold(serials, n_folds: int) -> np.ndarray:
    """Deterministic household-level fold, so no household spans a fold boundary."""
    import hashlib
    uniq = np.unique(serials)
    assign = {s: int(hashlib.sha256(f'{QUANTIZER_SALT}/{s}'.encode()).hexdigest(), 16) % n_folds
              for s in uniq}
    return np.array([assign[s] for s in serials], dtype=np.int64)
