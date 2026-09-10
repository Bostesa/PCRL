"""Bounded, train-only residual spectral baseline; no privacy guarantee.

All numerical choices are prospective constants. This module never reads data,
reserved labels, files, or environment settings. Inference needs only T and hA.
"""
from dataclasses import dataclass
from itertools import combinations_with_replacement
import warnings

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import pdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

SCHEMA = {'SEX': 2, 'RAC1P': 9, 'public_coverage': 2}
TOL = 1e-12
LS_RCOND = 1e-10
RFF_WIDTH = 96
TARGET_RANK = 16
ARMS = {'spectral_S0': ('none', 0.), 'spectral_M025': ('marginal', .25),
        'spectral_M1': ('marginal', 1.), 'spectral_L025': ('local', .25),
        'spectral_L1': ('local', 1.), 'spectral_C025': ('combined', .25),
        'spectral_C1': ('combined', 1.), 'spectral_L2': ('local', 2.)}


def _matrix(x, name, width=None):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or (width is not None and x.shape[1] != width) or not np.isfinite(x).all():
        raise ValueError(f'{name} must be finite two-dimensional with width {width}')
    return x


def _validate_labels(labels, n):
    if set(labels) != set(SCHEMA):
        raise ValueError(f'labels must contain exactly {list(SCHEMA)}')
    out = {}
    for name, classes in SCHEMA.items():
        y = np.asarray(labels[name])
        if y.shape != (n,) or not np.isin(y, np.arange(-1, classes)).all():
            raise ValueError(f'invalid fixed-schema labels for {name}')
        out[name] = y.astype(np.int64)
    return out


def _canonical(W):
    W = W.copy()
    if W.size:
        signs = np.sign(W[np.argmax(abs(W), axis=0), np.arange(W.shape[1])])
        W *= np.where(signs == 0, 1, signs)
    return W


def _trace_normalize(P):
    trace = float(np.trace(P))
    return (P / trace if trace > TOL else np.zeros_like(P)), trace


@dataclass
class PolynomialBasis:
    terms: list
    means: np.ndarray
    scales: np.ndarray
    retained: list
    input_width: int

    @staticmethod
    def _raw(x, terms):
        return np.column_stack([np.prod(x[:, term], axis=1) for term in terms])

    @classmethod
    def fit(cls, x):
        x = _matrix(x, 'basis input')
        terms = [()] + [(i,) for i in range(x.shape[1])] + list(combinations_with_replacement(range(x.shape[1]), 2))
        raw = cls._raw(x, terms)
        means = raw.mean(0)
        scales = raw.std(0)
        means[0], scales[0] = 0., 1.
        standardized = (raw-means)/np.maximum(scales, TOL)
        retained = [0]
        for j in range(1, len(terms)):
            if scales[j] <= TOL:
                continue
            if any(np.max(abs(standardized[:, j]-standardized[:, k])) <= TOL for k in retained):
                continue
            retained.append(j)
        return cls(terms, means, np.maximum(scales, TOL), retained, x.shape[1])

    def transform(self, x):
        x = _matrix(x, 'basis input', self.input_width)
        return ((self._raw(x, self.terms)-self.means)/self.scales)[:, self.retained]

    def metadata(self):
        return {'terms': [list(t) for t in self.terms], 'retained_indices': self.retained,
                'retained_terms': [list(self.terms[i]) for i in self.retained],
                'means': self.means.tolist(), 'scales': self.scales.tolist()}


@dataclass
class Nuisance:
    estimator: object
    constant: np.ndarray
    classes: int

    def predict(self, X):
        if self.estimator is None:
            return np.tile(self.constant, (len(X), 1))
        out = np.zeros((len(X), self.classes))
        out[:, self.estimator.classes_.astype(int)] = self.estimator.predict_proba(X)
        return out


def _oof(X, y, classes, folds):
    predictions = np.zeros((len(y), classes))
    records, fitted = [], []
    for f in range(3):
        train = (folds != f) & (y >= 0)
        hold = folds == f
        support = np.bincount(y[train], minlength=classes)
        hold_support = np.bincount(y[hold & (y >= 0)], minlength=classes)
        seen = np.flatnonzero(support)
        record = {'fold': f, 'train_support': support.tolist(), 'holdout_support': hold_support.tolist(),
                  'train_rows': int(train.sum()), 'holdout_rows': int(hold.sum()),
                  'unsupported_classes': np.flatnonzero(support == 0).tolist()}
        if len(seen) < 2:
            prior = support / max(int(support.sum()), 1)
            predictor = Nuisance(None, prior, classes)
            record.update(predictor='constant_one_class' if len(seen) else 'zero_no_training_support',
                          iterations=[], converged=True, convergence_warnings=[])
        else:
            estimator = LogisticRegression(C=1., solver='lbfgs', max_iter=10000, tol=1e-8,
                                           fit_intercept=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                estimator.fit(X[train], y[train])
            messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            predictor = Nuisance(estimator, None, classes)
            record.update(predictor='logistic_regression', iterations=estimator.n_iter_.tolist(),
                          converged=not messages and bool(np.all(estimator.n_iter_ < 10000)),
                          convergence_warnings=messages)
            if not record['converged']:
                raise ArithmeticError(f'nuisance fold {f} failed convergence: {record}')
        predictions[hold] = predictor.predict(X[hold])
        records.append(record)
        fitted.append(predictor)
    support = np.bincount(y[y >= 0], minlength=classes)
    return predictions, fitted, {'folds': records, 'full_support': support.tolist(),
                                'unsupported_classes': np.flatnonzero(support == 0).tolist()}


def moment_penalty(V, basis, y, predicted, classes, maps=None):
    """Full-K, unweighted masked moment Gram matrix and class-level diagnostics."""
    valid = y >= 0
    n = int(valid.sum())
    support = np.bincount(y[valid], minlength=classes)
    raw = np.zeros((V.shape[1], V.shape[1]))
    details = []
    for c in range(classes):
        residual = (y[valid] == c).astype(float) - predicted[valid, c]
        G = V[valid].T @ (basis[valid] * residual[:, None]) / max(n, 1)
        raw += G @ G.T
        item = {'class': c, 'support': int(support[c]), 'supported': bool(support[c]),
                'squared_norm': float(np.sum(G*G)), 'moment_matrix': G.tolist()}
        if maps is not None:
            item['output_squared_norms'] = {arm: float(np.sum((W.T @ G)**2)) for arm, W in maps.items()}
        details.append(item)
    normalized, trace = _trace_normalize(raw)
    return normalized, {'valid_rows': n, 'excluded_rows': int(len(y)-n), 'raw_trace': trace,
                        'zero_trace': trace <= TOL, 'schema_complete': bool(np.all(support > 0)),
                        'unsupported_classes': np.flatnonzero(support == 0).tolist(), 'classes': details}


@dataclass
class SpectralModel:
    t_mean: np.ndarray
    t_scale: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    qA: PolynomialBasis
    qAB: PolynomialBasis
    residual_coefficients: np.ndarray
    residual_mean: np.ndarray
    whitening: np.ndarray
    maps: dict
    nuisance_models: dict
    priors: dict

    @property
    def rank(self):
        return self.whitening.shape[1]

    @property
    def output_rank(self):
        return self.maps['spectral_S0'].shape[1]

    def features(self, T, hA):
        T, hA = _matrix(T, 'T', 32), _matrix(hA, 'hA', 4)
        if len(T) != len(hA):
            raise ValueError('row counts differ')
        standardized = (T-self.t_mean)/self.t_scale
        phi = np.column_stack((standardized, np.sqrt(2/RFF_WIDTH)*np.cos(standardized @ self.omega+self.phase)))
        residual = phi-self.qA.transform(hA[:, [1, 3]]) @ self.residual_coefficients-self.residual_mean
        return residual @ self.whitening

    def transform(self, T, hA, arm):
        return self.features(T, hA) @ self.maps[arm]

    def moment_diagnostics(self, T, hA, hB, labels):
        V = self.features(T, hA)
        labels = _validate_labels(labels, len(V))
        hB = _matrix(hB, 'hB', 2)
        if len(hB) != len(V):
            raise ValueError('row counts differ')
        bases = {'local': self.qA.transform(np.asarray(hA)[:, [1, 3]]),
                 'coalition': self.qAB.transform(np.column_stack((np.asarray(hA)[:, [1, 3]], hB[:, 1]))),
                 'marginal': np.ones((len(V), 1))}
        report = {'prediction_source': 'equal_ensemble_of_three_training_fold_models'}
        for role, basis in bases.items():
            attrs = SCHEMA if role != 'coalition' else {'SEX': 2, 'RAC1P': 9}
            report[role] = {}
            for name, classes in attrs.items():
                if role == 'marginal':
                    pred = np.tile(self.priors[name], (len(V), 1))
                else:
                    pred = sum(p.predict(basis) for p in self.nuisance_models[role][name])/3
                _, report[role][name] = moment_penalty(V, basis, labels[name], pred, classes, self.maps)
        return report


def fit_spectral(T, hA, hB, labels, households, seed):
    T, hA, hB = _matrix(T, 'T', 32), _matrix(hA, 'hA', 4), _matrix(hB, 'hB', 2)
    n = len(T)
    if n < 3 or len(hA) != n or len(hB) != n:
        raise ValueError('at least three rows and matching row counts required')
    labels = _validate_labels(labels, n)
    households = np.asarray(households)
    if households.shape != (n,):
        raise ValueError('household IDs must have one value per row')
    unique, inverse = np.unique(households, return_inverse=True)
    if len(unique) < 3:
        raise ValueError('three distinct households required for grouped three-fold OOF')
    order = np.random.default_rng(20262910+100*seed).permutation(len(unique))
    group_fold = np.empty(len(unique), dtype=int)
    group_fold[order] = np.arange(len(unique)) % 3
    folds = group_fold[inverse]
    mean = T.mean(0)
    standard_deviation = T.std(0)
    scale = np.where(standard_deviation > TOL, standard_deviation, 1.)
    standardized = (T-mean)/scale
    subset = np.random.default_rng(20260910+100*seed).choice(n, min(1024, n), replace=False)
    distances = pdist(standardized[subset])
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError('bandwidth diagnostic failure: no positive pairwise distances')
    sigma = float(np.median(positive))
    rng = np.random.default_rng(20261910+100*seed)
    omega = rng.normal(size=(32, RFF_WIDTH))/sigma
    phase = rng.uniform(0., 2*np.pi, RFF_WIDTH)
    phi = np.column_stack((standardized, np.sqrt(2/RFF_WIDTH)*np.cos(standardized @ omega+phase)))
    a = hA[:, [1, 3]]
    ab = np.column_stack((a, hB[:, 1]))
    qa, qab = PolynomialBasis.fit(a), PolynomialBasis.fit(ab)
    Q, QAB = qa.transform(a), qab.transform(ab)
    coefficients = np.linalg.lstsq(Q, phi, rcond=LS_RCOND)[0]
    V0 = phi-Q @ coefficients
    residual_mean = V0.mean(0)
    V0 -= residual_mean
    cov = V0.T @ V0/n
    eigenvalues, eigenvectors = np.linalg.eigh((cov+cov.T)/2)
    threshold = max(TOL, 1e-10*float(eigenvalues[-1]))
    retained = eigenvalues > threshold
    whitening = _canonical(eigenvectors[:, retained])/np.sqrt(eigenvalues[retained])
    V = V0 @ whitening
    rank, r = V.shape[1], min(TARGET_RANK, V.shape[1])
    teacher_coeff = np.linalg.lstsq(Q, T, rcond=LS_RCOND)[0]
    R = T-Q @ teacher_coeff
    R -= R.mean(0)
    cross = V.T @ R/n
    U, utility_trace = _trace_normalize(cross @ cross.T)
    if utility_trace <= TOL:
        raise ValueError(f'utility trace diagnostic failure: {utility_trace} <= {TOL}')
    priors, models, nuisance_records, penalties, penalty_records = {}, {}, {}, {}, {}
    bases = {'marginal': np.ones((n, 1)), 'local': Q, 'coalition': QAB}
    for role, basis in bases.items():
        attrs = SCHEMA if role != 'coalition' else {'SEX': 2, 'RAC1P': 9}
        P = np.zeros((rank, rank))
        records = {}
        models[role], nuisance_records[role] = {}, {}
        for name, classes in attrs.items():
            y = labels[name]
            if role == 'marginal':
                support = np.bincount(y[y >= 0], minlength=classes)
                priors[name] = support/max(int(support.sum()), 1)
                pred = np.tile(priors[name], (n, 1))
                nuisance_records[role][name] = {'prior': priors[name].tolist(), 'full_support': support.tolist(),
                                               'unsupported_classes': np.flatnonzero(support == 0).tolist()}
            else:
                pred, models[role][name], nuisance_records[role][name] = _oof(basis, y, classes, folds)
            normalized, records[name] = moment_penalty(V, basis, y, pred, classes)
            P += normalized/len(attrs)
        penalties[role] = P
        penalty_records[role] = {'denominator': len(attrs), 'attributes': records, 'trace': float(np.trace(P))}
    penalties['none'] = np.zeros_like(U)
    penalties['combined'] = penalties['local']+penalties['coalition']
    maps, arm_records = {}, {}
    for arm, (role, coefficient) in ARMS.items():
        A = U-coefficient*penalties[role]
        values, vectors = np.linalg.eigh((A+A.T)/2)
        W = _canonical(vectors[:, -r:][:, ::-1]) if r else np.zeros((rank, 0))
        maps[arm] = W
        Z = V @ W
        direct_coeff = np.linalg.lstsq(Z, R, rcond=LS_RCOND)[0]
        direct_error = float(np.sum((R-Z @ direct_coeff)**2)/n)
        explained = float(np.sum((W.T @ cross)**2))
        identity_error = float(np.sum(R*R)/n-explained)
        objective = float(np.trace(W.T @ A @ W))
        arm_records[arm] = {'penalty_role': role, 'lambda': coefficient, 'output_rank': r,
                            'objective': objective, 'top_eigenvalue_sum': float(values[-r:].sum()) if r else 0.,
                            'objective_gap': abs(objective-(float(values[-r:].sum()) if r else 0.)),
                            'utility_raw_explained': explained, 'utility_normalized': float(np.trace(W.T @ U @ W)),
                            'penalty_normalized': float(np.trace(W.T @ penalties[role] @ W)),
                            'reconstruction_direct_ls_mse': direct_error, 'reconstruction_identity_mse': identity_error,
                            'reconstruction_identity_abs_error': abs(direct_error-identity_error),
                            'orthogonality_max_abs': float(np.max(abs(W.T @ W-np.eye(r)))) if r else 0.,
                            'eigenvalues': values[::-1].tolist()}
    model = SpectralModel(mean, scale, omega, phase, qa, qab, coefficients, residual_mean, whitening, maps, models, priors)
    for role in bases:
        for details in penalty_records[role]['attributes'].values():
            for item in details['classes']:
                G = np.asarray(item['moment_matrix']).reshape(rank, bases[role].shape[1])
                item['output_squared_norms'] = {arm: float(np.sum((W.T @ G)**2)) for arm, W in maps.items()}
    diag = {'n_rows': n, 'seed': int(seed), 'sigma': sigma, 'sigma_fallback_no_positive_distances': not len(positive),
            'bandwidth_subset_indices': subset.tolist(), 'feature_width': 128, 'whitened_rank': rank, 'output_rank': r,
            'rank_mismatch_with_16d_baselines': r != TARGET_RANK, 'covariance_eigenvalues': eigenvalues.tolist(),
            'covariance_rank_threshold': threshold, 'qA': qa.metadata(), 'qAB': qab.metadata(),
            'whitening_covariance_max_abs': float(np.max(abs(V.T @ V/n-np.eye(rank)))) if rank else 0.,
            'whitening_mean_max_abs': float(np.max(abs(V.mean(0)))) if rank else 0.,
            'residual_basis_cross_max_abs': float(np.max(abs(Q.T @ V/n))) if rank else 0.,
            'utility_raw_trace': utility_trace, 'utility_zero_trace': utility_trace <= TOL,
            'fold_assignments': folds.tolist(), 'household_count': len(unique), 'nuisances': nuisance_records,
            'penalties': penalty_records, 'arms': arm_records,
            'inference_replay_max_abs': float(np.max(abs(model.features(T, hA)-V))) if rank else 0.,
            'nuisance_fit_budget': 15, 'extra_full_train_nuisance_fits': 0,
            'claim_scope': 'Finite sample matrix objective only; no unrestricted conditional or marginal privacy guarantee.'}
    if diag['whitening_covariance_max_abs'] > 1e-6 or diag['whitening_mean_max_abs'] > 1e-6:
        raise ArithmeticError('whitening failed numerical validation')
    return model, diag


def finite_source_privacy_lp(probabilities, useful, sensitive, side):
    """Finite channel Q(z|source), binary z and fixed decoder useful_hat=z.

    Privacy rows enforce P(z|s,h)=P(z|h) for supported (s,h).
    This diagnoses supplied distributions; it is not the fitted spectral problem.
    """
    p = np.asarray(probabilities, dtype=float)
    useful, sensitive, side = map(np.asarray, (useful, sensitive, side))
    n = len(p)
    if p.ndim != 1 or np.any(p < 0) or not np.isclose(p.sum(), 1) or not np.isin(useful, [0, 1]).all():
        raise ValueError('normalized finite probabilities and binary useful labels required')
    eq, rhs = [], []
    for i in range(n):
        row = np.zeros((n, 2)); row[i] = 1
        eq.append(row.ravel()); rhs.append(1.)
    for h in np.unique(side):
        mh = side == h
        if p[mh].sum() == 0:
            continue
        for s in np.unique(sensitive[mh]):
            mhs = mh & (sensitive == s)
            if p[mhs].sum() == 0:
                continue
            contrast = p*mhs/p[mhs].sum()-p*mh/p[mh].sum()
            for z in range(2):
                row = np.zeros((n, 2)); row[:, z] = contrast
                eq.append(row.ravel()); rhs.append(0.)
    cost = p[:, None]*(useful[:, None] != np.arange(2))
    A, b = np.asarray(eq), np.asarray(rhs)
    result = linprog(cost.ravel(), A_eq=A, b_eq=b, bounds=(0, None), method='highs')
    residual = max(float(np.max(abs(A @ result.x-b))), float(max(0., -np.min(result.x)))) if result.success else None
    return {'success': bool(result.success), 'status': int(result.status), 'message': result.message,
            'error': float(result.fun) if result.success else None, 'primal_residual_max': residual,
            'channel': result.x.reshape(n, 2).tolist() if result.success else None}


def numerical_fixtures():
    independent = finite_source_privacy_lp([.25]*4, [0, 0, 1, 1], [0, 1, 0, 1], [0]*4)
    identical = finite_source_privacy_lp([.5, .5], [0, 1], [0, 1], [0, 0])
    S = np.array([-1., -1., 1., 1.]); H = np.array([-1., 1., -1., 1.]); Z = S*H
    nonlinear_z = np.array([-1., 1., -2., 2.]); nonlinear_s = np.array([0., 0., 1., 1.])
    return {'independent_useful_sensitive_lp': independent, 'useful_equals_sensitive_lp': identical,
            'xor': {'marginal_moment': float(np.mean(Z*S)), 'interaction_moment': float(np.mean(Z*S*H)),
                    'marginal_independence': True, 'conditional_exact_recovery': True},
            'nonlinear_first_moment_failure': {'first_moment': float(np.mean(nonlinear_z*(nonlinear_s-.5))),
                                              'recovery_accuracy': float(np.mean((abs(nonlinear_z) > 1.5) == nonlinear_s))}}
