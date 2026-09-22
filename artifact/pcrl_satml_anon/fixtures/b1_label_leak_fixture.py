#!/usr/bin/env python3
"""Fixture: conditioning on per-row LOSS deciles leaks the label.

Stage B's B1 'probability decile' partition is built from the baseline probe's per-row log loss
(`stage_b._baseline_score` returns `probe['per_row_loss']`, and `_per_row` computes
-log p[i, y_i] using the person's own residence label y_i).

The leak is driven by CLASS SKEW, and residence is skewed: Stage B's prior log loss of 0.545 nats
corresponds to a base rate near 0.72. When the probe's predicted probabilities sit in a band above
0.5, the map y -> -log p[y] sends the two classes to disjoint loss ranges, so loss deciles encode
the label almost exactly.

Two cases are reported, and the contrast is the point:
  * skewed (the residence regime): y ~ Bern(0.72), p in [0.60, 0.85] carrying NO information about
    y. A table conditioned on loss deciles "predicts" y far below the true conditional entropy.
  * symmetric control: p centred on 0.5, where the two classes map to overlapping loss ranges and
    the same partition leaks nothing. This is why the defect is invisible in a balanced toy.
In both cases y is independent of everything the probe sees, so nothing is learnable; any reduction
below the entropy is the partition encoding the label.
"""
import numpy as np

def cross_fitted_table_loss(y, cells, fold, n_folds=5, pseudocount=0.5):
    """Same estimator shape as stage_b.ceiling_from_code (single side)."""
    y = np.asarray(y, np.int64); n_classes = int(y.max()) + 1
    loss = np.zeros(len(y))
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        _, key = np.unique(cells, return_inverse=True)
        table = np.full((key.max() + 1, n_classes), pseudocount)
        np.add.at(table, (key[tr], y[tr]), 1.0)
        table /= table.sum(1, keepdims=True)
        prior = (np.bincount(y[tr], minlength=n_classes) + pseudocount); prior = prior / prior.sum()
        seen = np.zeros(key.max() + 1, dtype=bool); seen[np.unique(key[tr])] = True
        p = np.where(seen[key[te]][:, None], table[key[te]], prior[None, :])
        loss[te] = -np.log(np.maximum(p[np.arange(te.sum()), y[te]], 1e-12))
    return float(loss.mean())

def case(n, seed, base_rate, p_lo, p_hi):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < base_rate).astype(np.int64)   # independent of everything the probe sees
    p1 = rng.uniform(p_lo, p_hi, n)                    # probe output, carries NO information about y
    per_row_loss = -np.log(np.where(y == 1, p1, 1 - p1))   # exactly stage_b._per_row
    fold = rng.integers(0, 5, n)
    decile = np.digitize(per_row_loss, np.quantile(per_row_loss, np.linspace(0, 1, 11)[1:-1]))
    prob_decile = np.digitize(p1, np.quantile(p1, np.linspace(0, 1, 11)[1:-1]))   # label-free control
    q = np.clip(np.mean(y), 1e-9, 1 - 1e-9)
    entropy = float(-(q * np.log(q) + (1 - q) * np.log(1 - q)))
    return {"base_rate": float(q), "true_conditional_entropy_nats": entropy,
            "loss_decile_partition_log_loss": cross_fitted_table_loss(y, decile, fold),
            "label_free_probability_decile_log_loss": cross_fitted_table_loss(y, prob_decile, fold),
            "n_rows": n, "n_cells": 10}

def main(n=3000, seed=0):
    out = {"skewed_residence_regime": case(n, seed, 0.72, 0.60, 0.85),
           "symmetric_control": case(n, seed + 1, 0.50, 0.20, 0.80)}
    for k in out:
        out[k]["leak_nats"] = (out[k]["true_conditional_entropy_nats"]
                               - out[k]["loss_decile_partition_log_loss"])
    return out

if __name__ == "__main__":
    import json
    r = main()
    print(json.dumps(r, indent=1))
    sk, sy = r["skewed_residence_regime"], r["symmetric_control"]
    assert sk["leak_nats"] > 0.4, "expected heavy leakage in the skewed regime"
    assert abs(sk["label_free_probability_decile_log_loss"]
               - sk["true_conditional_entropy_nats"]) < 0.03, "label-free control must not leak"
    assert abs(sy["leak_nats"]) < 0.03, "symmetric case should not leak (why a balanced toy hides it)"
    print("\nPASS: in the skewed residence regime, loss-decile conditioning recovers a label that is\n"
          "pure noise (leak %.3f nats); the label-free probability-decile control does not; and a\n"
          "symmetric toy shows no leak at all, which is why the defect is easy to miss."
          % sk["leak_nats"])
