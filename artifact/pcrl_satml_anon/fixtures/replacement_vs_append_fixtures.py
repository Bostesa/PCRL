#!/usr/bin/env python3
"""Two fixtures bounding what Stage B's G2 result can be used to reject.

Both are exact finite constructions. Neither says anything about the real ACS codebooks; they
constrain the INFERENCE that may be drawn from an append-to-J redundancy measurement.

FIXTURE 1 — append-redundancy does not reject a replacement channel.
    H constant, J = (Y, S), T = Y, with Y and S independent fair bits.
    * Appending T to J adds exactly zero task information: Y is already a coordinate of J.
      So a G2-style "does T add to J" test correctly reports zero gain.
    * Replacing J by T keeps task utility identical (T = Y determines Y) and removes S entirely.
    A release that is strictly better on disclosure at equal utility therefore exists precisely
    where the append test reports "no gain". The append test cannot see it.

FIXTURE 2 — an additive probe can miss information its own inputs jointly determine.
    J, T independent uniform signs, Y = J * T.
    A logistic score additive in J and in one-hot(T) has zero accuracy above chance, although
    (J, T) determines Y. Capacity of the probe family, not absence of information, decides the
    outcome. This is a capacity counterexample; it is NOT a claim that ACS residence behaves
    like XOR.
"""
import numpy as np


def _entropy(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1)
    return float(-(p * np.log(p)).sum())


def fixture_1_replacement(n=200000, seed=0):
    """H constant, J = (Y, S), T = Y. Exact log losses from the empirical joint."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = rng.integers(0, 2, n)
    j = np.column_stack([y, s])          # the existing release: carries the task AND the sensitive bit
    t = y.copy()                         # the candidate code: carries the task only

    def best_log_loss(view_cols):
        """Bayes log loss of predicting y from a discrete view (the exact conditional table)."""
        if view_cols is None:
            q = np.bincount(y, minlength=2) / n
            return float(-(np.bincount(y, minlength=2) / n * np.log(q)).sum())
        key = np.unique(view_cols, axis=0, return_inverse=True)[1]
        loss = 0.0
        for k in range(key.max() + 1):
            m = key == k
            q = np.bincount(y[m], minlength=2) / m.sum()
            loss += -np.log(np.clip(q[y[m]], 1e-12, None)).sum()
        return float(loss / n)

    def sensitive_recovery(view_cols):
        """Bayes log loss of predicting s from a discrete view (lower = more disclosure)."""
        key = np.unique(view_cols, axis=0, return_inverse=True)[1]
        loss = 0.0
        for k in range(key.max() + 1):
            m = key == k
            q = np.bincount(s[m], minlength=2) / m.sum()
            loss += -np.log(np.clip(q[s[m]], 1e-12, None)).sum()
        return float(loss / n)

    ll_prior = best_log_loss(None)
    ll_j = best_log_loss(j)
    ll_j_plus_t = best_log_loss(np.column_stack([j, t]))
    ll_t = best_log_loss(t[:, None])
    s_j = sensitive_recovery(j)
    s_t = sensitive_recovery(t[:, None])
    return {
        "task_log_loss_prior": ll_prior,
        "task_log_loss_J": ll_j,
        "task_log_loss_J_plus_T": ll_j_plus_t,
        "task_log_loss_T_alone": ll_t,
        "append_gain_J_to_J_plus_T": ll_j - ll_j_plus_t,       # exactly 0: the append test sees nothing
        "replacement_task_change_J_to_T": ll_t - ll_j,          # 0: utility preserved by replacing
        "sensitive_log_loss_given_J": s_j,                      # 0: S fully disclosed by J
        "sensitive_log_loss_given_T": s_t,                      # log 2: S not disclosed by T
        "sensitive_disclosure_removed_nats": s_t - s_j,
    }


def fixture_2_additive_probe(n=200000, seed=1):
    """Y = J * T with independent signs; an additive logistic score cannot represent it."""
    rng = np.random.default_rng(seed)
    j = rng.choice([-1.0, 1.0], n)
    t = rng.choice([-1.0, 1.0], n)
    y = ((j * t) > 0).astype(int)

    # Best additive score a(J) + b(T): both marginals are independent of Y, so any additive
    # score has the same expected loss as the prior. Verified by exhaustive fit on the 4 cells.
    best_additive = None
    for a in np.linspace(-3, 3, 25):
        for b in np.linspace(-3, 3, 25):
            for c in np.linspace(-3, 3, 25):
                z = c + a * j + b * t
                p = 1 / (1 + np.exp(-z))
                ll = float(-np.mean(np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-12, None))))
                best_additive = ll if best_additive is None else min(best_additive, ll)
    joint_cells = (j > 0).astype(int) * 2 + (t > 0).astype(int)
    loss = 0.0
    for k in range(4):
        m = joint_cells == k
        q = np.bincount(y[m], minlength=2) / m.sum()
        loss += -np.log(np.clip(q[y[m]], 1e-12, None)).sum()
    return {
        "prior_log_loss": float(np.log(2)),
        "best_additive_in_J_and_onehot_T": best_additive,
        "joint_view_log_loss": float(loss / n),
        "additive_probe_gain_over_prior": float(np.log(2)) - best_additive,
        "joint_view_gain_over_prior": float(np.log(2)) - float(loss / n),
    }


if __name__ == "__main__":
    import json
    f1 = fixture_1_replacement()
    f2 = fixture_2_additive_probe()
    print(json.dumps({"fixture_1_replacement": f1, "fixture_2_additive_probe": f2}, indent=1))

    assert abs(f1["append_gain_J_to_J_plus_T"]) < 1e-9, "appending T to J must add exactly nothing"
    assert abs(f1["replacement_task_change_J_to_T"]) < 1e-9, "replacing J by T must preserve utility"
    assert f1["sensitive_disclosure_removed_nats"] > 0.69 - 1e-3, "replacement must remove S disclosure"
    assert f2["additive_probe_gain_over_prior"] < 1e-6, "additive probe must gain nothing"
    assert f2["joint_view_gain_over_prior"] > 0.69 - 1e-3, "the joint view determines Y"
    print("\nPASS")
    print(" 1. Append test sees zero gain while a replacement release keeps all task utility and")
    print("    removes the sensitive bit entirely -> append-redundancy cannot reject replacement.")
    print(" 2. An additive probe gains nothing where the joint view determines the label ->")
    print("    a null from one probe family is a capacity statement, not an information statement.")
