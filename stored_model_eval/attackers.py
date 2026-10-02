"""Attacker families. Every fit goes through a FitAuthorization (guards.py) and every attacker declares an
AccessRecord (access.py). Hyper-parameters are selected on attacker-validation rows only (log-loss);
evaluation rows are never seen by fit or selection.

  linear            multinomial logistic regression on standardised inputs           A1
  gbt               histogram gradient-boosted trees                                  A1
  mlp               sklearn MLP with early stopping                                   A1
  noise_lrt         exact likelihood-ratio test for a Gaussian noise channel using the
                    attacker population's CLEAN vectors and the known Sigma            A4
  adaptive          base attacker trained on defense-simulated releases of the attacker
                    population (mechanism code + public params)                       A2
  repeated_release  averages N releases of the target; valid only if the contract issues
                    fresh noise per query (persistent token => N collapses to 1)       A3(N)
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import log_loss

from .access import AccessRecord, ReleaseContract
from .guards import FitAuthorization


def _full_proba(clf, X, K):
    p = clf.predict_proba(X)
    out = np.zeros((len(X), K))
    out[:, np.asarray(clf.classes_).astype(int)] = p
    return out


class _SklearnAttacker:
    name = "base"
    tag = "A1"

    def __init__(self, cfg: dict | None = None, random_state: int = 0):
        self.cfg = cfg or {}
        self.random_state = random_state
        self.model = None
        self.selected = None
        self.selection_table = []

    def grid(self):
        raise NotImplementedError

    def make(self, **hp):
        raise NotImplementedError

    def fit(self, X_fit, y_fit, X_val, y_val, *, auth: FitAuthorization, synthetic: bool):
        auth.check(f"attacker {self.name}", synthetic)
        y_fit, y_val = np.asarray(y_fit).astype(int), np.asarray(y_val).astype(int)
        self.K = int(max(y_fit.max(), y_val.max())) + 1
        best = None
        for hp in self.grid():
            m = self.make(**hp)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X_fit, y_fit)
            p = np.clip(_full_proba(m, X_val, self.K), 1e-12, 1)
            ll = float(log_loss(y_val, p / p.sum(1, keepdims=True), labels=list(range(self.K))))
            self.selection_table.append({"hp": hp, "val_log_loss": ll})
            if best is None or ll < best[0]:
                best = (ll, hp, m)
        self.selected, self.model = best[1], best[2]
        return self

    def predict_proba(self, X):
        return _full_proba(self.model, X, self.K)

    def access_record(self, surface: str, contract: ReleaseContract | None = None) -> AccessRecord:
        return AccessRecord(tag=self.tag, attacker=self.name, surface=surface,
                            fit_inputs=["release(attacker_fit)", "S(attacker_fit)"],
                            eval_inputs=["release(target), one draw"],
                            contract=None if contract is None else contract.to_json())


class LinearAttacker(_SklearnAttacker):
    name = "linear"

    def grid(self):
        return [{"C": c} for c in self.cfg.get("C", [0.01, 0.1, 1.0, 10.0])]

    def make(self, C):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


class GBTAttacker(_SklearnAttacker):
    name = "gbt"

    def grid(self):
        return [{"learning_rate": lr, "max_leaf_nodes": ml} for lr, ml in itertools.product(
            self.cfg.get("learning_rate", [0.05, 0.1]), self.cfg.get("max_leaf_nodes", [15, 31]))]

    def make(self, learning_rate, max_leaf_nodes):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                                              max_iter=int(self.cfg.get("max_iter", 200)),
                                              early_stopping=False, random_state=self.random_state)


class MLPAttacker(_SklearnAttacker):
    name = "mlp"

    def grid(self):
        return [{"hidden": tuple(h), "alpha": a} for h, a in itertools.product(
            self.cfg.get("hidden", [[64], [128, 64]]), self.cfg.get("alpha", [1e-4, 1e-3]))]

    def make(self, hidden, alpha):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=hidden, alpha=alpha, max_iter=int(self.cfg.get("max_iter", 200)),
            early_stopping=True, random_state=self.random_state))


FAMILIES = {"linear": LinearAttacker, "gbt": GBTAttacker, "mlp": MLPAttacker}


# --------------------------------------------------------------------------------------------------
# release channel (for noise mechanisms and repeated queries)
# --------------------------------------------------------------------------------------------------


class ReleaseChannel:
    """Gaussian release r = h + e. fresh_per_query: e depends on (person token, query index);
    persistent_token: e depends on the person token only (every query returns the same release)."""

    def __init__(self, contract: ReleaseContract, seed: int = 0):
        if contract.noise == "none":
            raise ValueError("ReleaseChannel needs a noise contract")
        self.persistent = contract.persistent
        self.contract, self.seed = contract, int(seed)
        self.sigma = float(contract.sigma)

    def release(self, H: np.ndarray, tokens: np.ndarray, query: int = 0) -> np.ndarray:
        H = np.asarray(H, dtype=np.float64)
        E = np.empty_like(H)
        for r, t in enumerate(np.asarray(tokens).tolist()):
            key = [self.seed, int(t)] if self.persistent \
                else [self.seed, int(t), int(query) + 1]
            E[r] = np.random.default_rng(np.random.SeedSequence(key)).normal(size=H.shape[1])
        return H + self.sigma * E

    def averaged(self, H, tokens, N: int) -> np.ndarray:
        rels = [self.release(H, tokens, q) for q in range(N)]
        if all(np.array_equal(rels[0], r) for r in rels[1:]):
            return rels[0]  # identical releases (persistent token): the average IS one release, exactly
        return np.mean(rels, axis=0)


class NoiseLRTAttacker:
    """Exact Bayes/LRT for r = h + N(0, Sigma) with the population of clean vectors as the prior on h:
    p(r | s) = mean_{i: s_i = s} N(r; h_i, Sigma). Sigma known (isotropic sigma^2 / n_eff)."""
    name = "noise_lrt"
    tag = "A4"

    def __init__(self, contract: ReleaseContract):
        if contract.noise == "none" or contract.sigma is None:
            raise ValueError("noise_lrt needs a declared noise contract with sigma")
        self.contract = contract

    def fit(self, H_pop_clean, y_pop, *, auth: FitAuthorization, synthetic: bool):
        auth.check("noise_lrt population table", synthetic)
        self.H = np.asarray(H_pop_clean, dtype=np.float64)
        self.y = np.asarray(y_pop).astype(int)
        self.K = int(self.y.max()) + 1
        return self

    def predict_proba(self, R, n_eff: int = 1):
        R = np.asarray(R, dtype=np.float64)
        var = self.contract.sigma ** 2 / max(int(n_eff), 1)
        d2 = (R ** 2).sum(1)[:, None] - 2 * R @ self.H.T + (self.H ** 2).sum(1)[None, :]
        L = -0.5 * d2 / var
        logp = np.full((len(R), self.K), -np.inf)
        for k in range(self.K):
            m = self.y == k
            if m.any():
                logp[:, k] = logsumexp(L[:, m], axis=1)  # sum over members = prior * mean likelihood
        logp -= logsumexp(logp, axis=1, keepdims=True)
        return np.exp(logp)

    def access_record(self, surface: str = "rep", contract=None) -> AccessRecord:
        return AccessRecord(tag="A4", attacker=self.name, surface=surface,
                            fit_inputs=["clean representations of attacker population", "S(population)"],
                            eval_inputs=["release(target)"], requires=["known Sigma", "mechanism code"],
                            contract=self.contract.to_json())


class AdaptiveAttacker:
    """Trains a base attacker on releases the attacker simulates by running the defense (mechanism code
    + public parameters) on its OWN population inputs; scores the real target releases."""
    name = "adaptive"
    tag = "A2"

    def __init__(self, simulate, base: str = "gbt", cfg: dict | None = None, random_state: int = 0):
        self.simulate = simulate  # callable(inputs, rng) -> simulated releases
        self.base = FAMILIES[base](cfg, random_state)
        self.random_state = random_state

    def fit(self, X_pop_fit, y_fit, X_pop_val, y_val, *, auth: FitAuthorization, synthetic: bool):
        auth.check("adaptive attacker", synthetic)
        rng = np.random.default_rng(self.random_state)
        self.base.fit(self.simulate(X_pop_fit, rng), y_fit, self.simulate(X_pop_val, rng), y_val,
                      auth=auth, synthetic=synthetic)
        return self

    def predict_proba(self, R):
        return self.base.predict_proba(R)

    def access_record(self, surface: str = "rep", contract=None) -> AccessRecord:
        return AccessRecord(tag="A2", attacker=f"adaptive[{self.base.name}]", surface=surface,
                            fit_inputs=["attacker-population inputs", "S(population)",
                                        "defense-simulated releases"],
                            eval_inputs=["release(target), one draw"],
                            requires=["mechanism code", "public parameters"],
                            contract=None if contract is None else contract.to_json())


class RepeatedReleaseAttacker:
    """Averages N releases of each target and scores with a noise-aware base attacker (n_eff passed on).
    Valid only when the contract issues fresh noise per query; under a persistent token the N releases are
    identical, the average equals one release, and the record is marked invalid (N collapses to 1)."""
    name = "repeated_release"
    tag = "A3"

    def __init__(self, base: NoiseLRTAttacker, channel: ReleaseChannel, N: int):
        self.base, self.channel, self.N = base, channel, int(N)

    def predict_proba(self, H_target_clean, tokens):
        R = self.channel.averaged(H_target_clean, tokens, self.N)
        n_eff = self.channel.contract.effective_queries(self.N)
        return self.base.predict_proba(R, n_eff=n_eff)

    def access_record(self, surface: str = "rep") -> AccessRecord:
        c = self.channel.contract
        valid = c.issues_fresh_noise()
        return AccessRecord(tag=f"A3({self.N})", attacker=self.name, surface=surface,
                            fit_inputs=self.base.access_record().fit_inputs,
                            eval_inputs=[f"{self.N} releases of target"], requires=["query interface"],
                            contract=c.to_json(), valid=valid,
                            invalid_reason=None if valid else f"contract {c.noise}: no fresh noise per query",
                            queries=self.N, effective_queries=c.effective_queries(self.N))
