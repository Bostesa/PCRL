"""Frozen deterministic 17-token policy bank (spec section 2 + amendment M1).

Every policy is a pure map from LEGAL per-person inputs (X_A, H_A, stored T0
code, stored teacher posterior p, stored residual r, stored risk(11)) to the
SAME 17-token codebook as D17.  Non-D17 policies are PAIRED cost-regression
oracles (amendment M2): with g_i(z) = u_i(z) - sum_j lambda_j a_ij(z) in
per-person nats, each token's regressor targets the paired improvement

    Delta_i(z) = g_i(z) - g_i(D17(x_i))      (identically 0 at the D17 token)

and Delta_hat(D17(x)|x) := 0.  The policy deviates to argmin_z Delta_hat(z|x)
only if min_z Delta_hat(z|x) < -TAU (=0.002 nats); ties go to D17, then to
the lowest token id.
Labels are used only on nuisance_train rows, through the cost targets.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

N_TOKENS = 17
TAU = 0.002
LEGAL_INPUTS = ("x", "ha", "token_codes", "teacher_p", "residual", "risk")
INPUT_ALIASES = {"T0": "token_codes", "t0": "token_codes", "p": "teacher_p",
                 "r": "residual"}
FORBIDDEN_INPUTS = frozenset({"hb", "labels", "ids", "households", "household",
                              "weights", "role", "roles", "loss", "losses",
                              "y", "s", "SEX", "RAC1P", "same_residence"})
POLICY_NAMES = ("D17", "task_only", "local_priced", "coalition_priced", "all_priced_x2")
PRICE_GROUPS = {"task_only": ((), 0.),
                "local_priced": (("A/SEX", "A/RAC1P"), 1.),
                "coalition_priced": (("AB/SEX", "AB/RAC1P"), 1.),
                "all_priced_x2": (("A/SEX", "A/RAC1P", "AB/SEX", "AB/RAC1P"), 2.)}
HGB_PARAMS = {"max_depth": 3, "max_iter": 150, "learning_rate": 0.05,
              "min_samples_leaf": 40, "l2_regularization": 1.0}
VALIDATION_FRACTION = 0.2
SPLIT_SALT = "pcrl_shared_context_release_v1|policy_early_stopping|"


# ---------------------------------------------------------------------------
# Legal-input guard
# ---------------------------------------------------------------------------

def check_legal(inputs: Mapping) -> dict:
    """Canonicalize a per-person input mapping; reject any non-legal key."""
    if not isinstance(inputs, Mapping):
        raise TypeError("policy inputs must be a mapping of legal per-person arrays")
    out = {}
    for key, value in inputs.items():
        canonical = INPUT_ALIASES.get(key, key)
        if key in FORBIDDEN_INPUTS or canonical not in LEGAL_INPUTS:
            raise PermissionError(f"forbidden or undeclared deployment input: {key!r}")
        if canonical in out:
            raise ValueError(f"duplicate legal input {canonical!r}")
        out[canonical] = value
    return out


def legal_inputs(rows: Mapping) -> dict:
    """Pick exactly the legal columns out of a private pooled-role dictionary."""
    return {key: rows[key] for key in LEGAL_INPUTS if key in rows}


def _require(legal, keys):
    missing = [k for k in keys if k not in legal]
    if missing:
        raise ValueError(f"missing legal inputs {missing}")


def policy_features(inputs) -> np.ndarray:
    """[X_A(32), H_A(4), logit(p), r, risk(11)] -> (n, 49) float64."""
    legal = check_legal(inputs)
    _require(legal, ("x", "ha", "teacher_p", "residual", "risk"))
    x = np.asarray(legal["x"], dtype=np.float64)
    n = len(x)
    ha = np.asarray(legal["ha"], dtype=np.float64)
    p = np.clip(np.asarray(legal["teacher_p"], dtype=np.float64), 1e-6, 1 - 1e-6)
    r = np.asarray(legal["residual"], dtype=np.float64)
    risk = np.asarray(legal["risk"], dtype=np.float64)
    if (x.shape != (n, 32) or ha.shape != (n, 4) or p.shape != (n,) or r.shape != (n,)
            or risk.shape != (n, 11)):
        raise ValueError("legal policy inputs have unexpected shapes")
    z = np.column_stack((x, ha, np.log(p) - np.log1p(-p), r, risk))
    if not np.isfinite(z).all():
        raise ValueError("nonfinite legal policy feature")
    return z


def _codes(legal, n_states):
    _require(legal, ("token_codes",))
    t = np.asarray(legal["token_codes"])
    if t.ndim != 1 or t.dtype.kind not in "iu" or np.any(t < 0) or np.any(t >= n_states):
        raise ValueError("stored T0 codes must be integer states")
    return t.astype(np.int64)


# ---------------------------------------------------------------------------
# Frozen policy objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features):
        z = np.asarray(features, dtype=np.float64)
        mean = z.mean(0)
        scale = z.std(0)
        scale[scale < 1e-12] = 1.
        return cls(mean, scale)

    def transform(self, features):
        z = np.asarray(features, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != len(self.mean):
            raise ValueError("feature width differs from frozen standardizer")
        return (z - self.mean) / self.scale


@dataclass(frozen=True)
class D17Policy:
    name: str
    d17_tokens: np.ndarray

    def predict(self, inputs) -> np.ndarray:
        legal = check_legal(inputs)
        return self.d17_tokens[_codes(legal, len(self.d17_tokens))].copy()


@dataclass(frozen=True)
class PairedOracle:
    """17 frozen regressors of Delta(z) = g(z) - g(D17); D17 column forced to 0."""
    models: tuple

    def predict_delta(self, standardized_features, base_tokens) -> np.ndarray:
        z = np.asarray(standardized_features, dtype=np.float64)
        base = np.asarray(base_tokens, dtype=np.int64)
        if len(self.models) != N_TOKENS or base.shape != (len(z),):
            raise ValueError("paired oracle needs 17 regressors and one D17 token per row")
        delta = np.column_stack([m.predict(z) for m in self.models])
        if delta.shape != (len(z), N_TOKENS) or not np.isfinite(delta).all():
            raise ValueError("frozen paired regressor returned invalid predictions")
        delta[np.arange(len(z)), base] = 0.
        return delta


@dataclass(frozen=True)
class SwitchedCostPolicy:
    """D17-anchored paired cost-regression policy (amendments M1 + M2)."""
    name: str
    d17_tokens: np.ndarray
    standardizer: FeatureStandardizer
    oracle: PairedOracle
    tau: float = TAU
    pricing: dict = field(default_factory=dict)

    def predict_delta(self, inputs) -> np.ndarray:
        legal = check_legal(inputs)
        base = self.d17_tokens[_codes(legal, len(self.d17_tokens))]
        return self.oracle.predict_delta(self.standardizer.transform(policy_features(legal)), base)

    predict_costs = predict_delta   # paired costs relative to D17 (D17 column = 0)

    def predict(self, inputs) -> np.ndarray:
        legal = check_legal(inputs)
        base = self.d17_tokens[_codes(legal, len(self.d17_tokens))]
        return switched_argmin(self.predict_delta(legal), base, self.tau)


def paired_targets(costs, base_tokens) -> np.ndarray:
    """Delta_i(z) = g_i(z) - g_i(D17(x_i)); exactly 0 in each row's D17 column."""
    g = np.asarray(costs, dtype=np.float64)
    base = np.asarray(base_tokens, dtype=np.int64)
    if g.ndim != 2 or g.shape[1] != N_TOKENS or base.shape != (len(g),) or np.any(base < 0) \
            or np.any(base >= N_TOKENS) or not np.isfinite(g).all():
        raise ValueError("aligned finite (n,17) costs and D17 tokens required")
    return g - g[np.arange(len(g)), base][:, None]


def fit_paired_oracle(standardized_features, costs, base_tokens, pwgtp, households, *, seed: int):
    """Stable M2 oracle: paired targets + the registered HGB/early-stopping recipe.

    Returns (PairedOracle, record).  `costs` are raw per-person priced costs
    g_i(z) in nats on the fitting rows; `base_tokens` are D17(x_i).
    """
    targets = paired_targets(costs, base_tokens)
    models, record = fit_cost_regressors(standardized_features, targets, pwgtp, households, seed=seed)
    return PairedOracle(models), {**record, "target": "paired Delta_i(z) = g_i(z) - g_i(D17(x_i)) (M2)"}


def switched_policy_from_oracle(name, d17_tokens, standardizer, oracle, *, tau=TAU, pricing=None):
    """Frozen legal-input policy: argmin Delta_hat if min Delta_hat < -tau, else D17."""
    if not isinstance(oracle, PairedOracle):
        raise TypeError("PairedOracle required (amendment M2)")
    return SwitchedCostPolicy(name, np.asarray(d17_tokens, dtype=np.int64), standardizer,
                              oracle, float(tau), dict(pricing or {}))


def switched_argmin(g, base_tokens, tau=TAU) -> np.ndarray:
    """argmin_z g (lowest id on ties) only where it beats D17 by more than tau.

    With paired g (D17 column 0) this is exactly: deviate iff min_z Delta < -tau.
    """
    g = np.asarray(g, dtype=np.float64)
    base = np.asarray(base_tokens, dtype=np.int64)
    best = np.argmin(g, axis=1)
    gap = g[np.arange(len(g)), base] - g[np.arange(len(g)), best]
    return np.where(gap > tau, best, base).astype(np.int64)


@dataclass(frozen=True)
class PolicyBank:
    names: tuple
    policies: tuple

    def predict(self, inputs) -> np.ndarray:
        legal = check_legal(inputs)
        tokens = np.column_stack([p.predict(legal) for p in self.policies]).astype(np.int64)
        if tokens.shape[1] != len(self.names) or np.any(tokens < 0) or np.any(tokens >= N_TOKENS):
            raise ValueError("policy bank emitted an illegal token")
        return tokens

    def subset(self, names: Sequence[str]) -> "PolicyBank":
        index = [self.names.index(n) for n in names]
        if not index or index[0] != 0 or self.names[0] != "D17":
            raise ValueError("column 0 must remain exact D17")
        return PolicyBank(tuple(names), tuple(self.policies[i] for i in index))


# ---------------------------------------------------------------------------
# Pricing and cost targets
# ---------------------------------------------------------------------------

def price_groups(cuts: Sequence[Mapping], multipliers: Mapping[str, float]) -> dict:
    """Restrict round-0 T32 LP duals to each registered group; record fallbacks."""
    by_id = {cut["id"]: cut for cut in cuts}
    if set(by_id) != set(multipliers):
        raise ValueError("dual multipliers must cover exactly the calibrated bank")
    out = {}
    for name, (roles, scale) in PRICE_GROUPS.items():
        members = sorted(cid for cid, cut in by_id.items() if cut["role"] in roles)
        if not roles:
            out[name] = {"multipliers": {}, "fallback_used": False, "group_cuts": 0, "scale": 0.}
            continue
        values = {cid: max(0., float(multipliers[cid])) for cid in members}
        fallback = not members or all(v == 0 for v in values.values())
        if fallback:
            values = {cid: 1. / len(members) for cid in members} if members else {}
        out[name] = {"multipliers": {cid: scale * v for cid, v in values.items() if v > 0},
                     "fallback_used": bool(fallback), "group_cuts": len(members),
                     "scale": scale, "dual_sum": float(sum(values.values()))}
    return out


def priced_person_costs(task_losses, attack_rows: Sequence[Mapping], multipliers: Mapping[str, float],
                        task_mask) -> np.ndarray:
    """g_i(z) = u_i(z) - sum_j lambda_j a_ij(z) in per-person nats, task-valid rows.

    `attack_rows` items carry `id`, `valid_mask` over all rows and `losses` on
    their valid rows (e.g. `AR/fit_b._attack_loss_rows`).  A person without
    that attack's protected label contributes no attack term for it.
    """
    task_mask = np.asarray(task_mask, dtype=bool)
    u = np.asarray(task_losses, dtype=np.float64)
    if u.shape != (int(task_mask.sum()), N_TOKENS) or not np.isfinite(u).all():
        raise ValueError("task-valid per-person token losses required")
    g = np.zeros((len(task_mask), N_TOKENS))
    g[task_mask] = u
    seen = set()
    for row in attack_rows:
        cid = row["id"]
        if cid not in multipliers:
            continue
        seen.add(cid)
        mask = np.asarray(row["valid_mask"], dtype=bool)
        losses = np.asarray(row["losses"], dtype=np.float64)
        if mask.shape != task_mask.shape or losses.shape != (int(mask.sum()), N_TOKENS):
            raise ValueError("attack loss rows misaligned")
        g[mask] -= float(multipliers[cid]) * losses
    if seen != set(multipliers):
        raise ValueError("priced cut without replayed attack losses")
    return g[task_mask]


def _validation_mask(households):
    u = np.asarray([int.from_bytes(hashlib.sha256((SPLIT_SALT + str(h)).encode()).digest()[:8], "big") / 2**64
                    for h in households])
    return u < VALIDATION_FRACTION


def fit_cost_regressors(features, costs, pwgtp, households, *, seed: int):
    """17 HGB regressors with household-grouped 20% early-stopping validation.

    The boosting count is the staged-validation argmin (earliest on ties),
    refitted deterministically on the 80% training households.
    """
    z = np.asarray(features, dtype=np.float64)
    g = np.asarray(costs, dtype=np.float64)
    w = np.asarray(pwgtp, dtype=np.float64)
    hh = np.asarray(households).astype(str)
    if (z.ndim != 2 or g.shape != (len(z), N_TOKENS) or w.shape != (len(z),)
            or hh.shape != (len(z),) or not np.isfinite(g).all()):
        raise ValueError("aligned features/costs/weights/households required")
    w = w / w.mean()
    val = _validation_mask(hh)
    if val.all() or not val.any():
        raise ValueError("household early-stopping split is degenerate")
    fit = ~val
    models, record = [], []
    for token in range(N_TOKENS):
        probe = HistGradientBoostingRegressor(**HGB_PARAMS, early_stopping=False,
                                              random_state=int(seed))
        probe.fit(z[fit], g[fit, token], sample_weight=w[fit])
        curve = [float(np.average((stage - g[val, token]) ** 2, weights=w[val]))
                 for stage in probe.staged_predict(z[val])]
        best = int(np.argmin(curve)) + 1
        model = HistGradientBoostingRegressor(**{**HGB_PARAMS, "max_iter": best},
                                              early_stopping=False, random_state=int(seed))
        model.fit(z[fit], g[fit, token], sample_weight=w[fit])
        models.append(model)
        record.append({"token": token, "iterations": best,
                       "validation_mse": curve[best - 1], "validation_mse_first": curve[0]})
    return tuple(models), {"per_token": record, "fit_people": int(fit.sum()),
                           "validation_people": int(val.sum()),
                           "fit_households": int(len(np.unique(hh[fit]))),
                           "validation_households": int(len(np.unique(hh[val]))),
                           "params": HGB_PARAMS, "seed": int(seed),
                           "sample_weight": "PWGTP / mean(PWGTP)",
                           "early_stopping": "household-grouped 20% validation, staged argmin, refit on 80%"}


# ---------------------------------------------------------------------------
# Diagnostics and aliases (no outer rows)
# ---------------------------------------------------------------------------

def disagreement_diagnostics(tokens, names, codes, pwgtp, households, *, n_states=32) -> dict:
    """Per-policy disagreement with D17 (column 0), within-T32 and households."""
    d = np.asarray(tokens, dtype=np.int64)
    t = np.asarray(codes, dtype=np.int64)
    w = np.asarray(pwgtp, dtype=np.float64)
    hh = np.asarray(households).astype(str)
    base = d[:, 0]
    out = {}
    for m, name in enumerate(names):
        differ = d[:, m] != base
        per_state = []
        for s in range(n_states):
            member = t == s
            if member.any():
                per_state.append(float(differ[member].mean()))
        per_state = np.asarray(per_state)
        varying_states = sum(len(np.unique(d[t == s, m])) > 1 for s in range(n_states) if (t == s).any())
        out[name] = {"fraction_differs_from_D17_unweighted": float(differ.mean()),
                     "fraction_differs_from_D17_weighted": float(np.dot(w / w.sum(), differ)),
                     "affected_unique_households": int(len(np.unique(hh[differ]))),
                     "within_T32_disagreement_quantiles": {
                         str(q): float(np.quantile(per_state, q)) for q in (0, .25, .5, .75, 1.)},
                     "states_with_within_T32_token_variation": int(varying_states),
                     "token_histogram": np.bincount(d[:, m], minlength=N_TOKENS).tolist()}
    return out


def alias_ledger(tokens, names) -> dict:
    """Keep the first of every exactly identical token column; D17 always kept."""
    d = np.asarray(tokens, dtype=np.int64)
    retained, ledger = [], []
    for m, name in enumerate(names):
        match = next((r for r in retained if np.array_equal(d[:, names.index(r)], d[:, m])), None)
        if match is None:
            retained.append(name)
        else:
            ledger.append({"removed": name, "alias_of": match})
    if retained[0] != "D17":
        raise AssertionError("D17 must be the first retained column")
    return {"retained": retained, "removed": ledger,
            "rule": "exact identical token vectors on coefficient_split; earlier column kept"}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_bank(directory, bank: PolicyBank) -> dict:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "policy_bank.joblib"
    if path.exists():
        raise FileExistsError("frozen policy bank is write-once")
    temporary = root / "policy_bank.joblib.tmp"
    joblib.dump(bank, temporary, compress=3)
    temporary.replace(path)
    return {"file": "policy_bank.joblib", "sha256": _sha(path), "names": list(bank.names)}


def load_bank(directory, expected_sha256: str) -> PolicyBank:
    path = Path(directory) / "policy_bank.joblib"
    if _sha(path) != expected_sha256:
        raise ValueError("frozen policy bank hash mismatch")
    bank = joblib.load(path)
    if not isinstance(bank, PolicyBank) or bank.names[0] != "D17":
        raise ValueError("unexpected frozen policy bank schema")
    return bank


def dumps(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
