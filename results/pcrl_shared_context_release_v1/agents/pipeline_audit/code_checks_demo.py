"""Synthetic demonstrations for CODE_CHECKS.md (no ACS data; read-only on code).

Run from the worktree root:
  PYTHONPATH=. /Users/nathansamson/PCRL/.venv/bin/python \
    results/pcrl_shared_context_release_v1/agents/pipeline_audit/code_checks_demo.py
"""
import numpy as np

from experiments.pcrl_adaptive_release_v1 import fit_b, refinement

rng = np.random.default_rng(20260924)


# ---- A1: nonnegativity by construction -------------------------------------
# min_z sum_{P} g(z) >= min_z sum_L g(z) + min_z sum_R g(z) because the parent
# argmin z* is feasible for both children. Brute-force check on random rows.
worst = np.inf
for _ in range(20000):
    n = rng.integers(2, 12)
    g = rng.normal(size=(n, 17)) * rng.exponential(size=(n, 1))
    left = rng.random(n) < .5
    if left.all() or not left.any():
        continue
    raw = g.sum(0).min() - g[left].sum(0).min() - g[~left].sum(0).min()
    worst = min(worst, raw)
print(f"A1 min raw gain over 20000 random splits = {worst:.3e} (>= 0 up to fp)")


# ---- A2: checking score is a re-optimized opportunity -----------------------
# Pure-noise population: every person's expected priced row is the same vector
# mu, so the population split gain is exactly 0 for every split. Pools are
# normalized per pool (rows sum to a per-pool average), mirroring
# priced_original_person_rows (fit_b.py:389, 406-408).
mu = np.linspace(0, .2, 17)


def pool(n):
    return (mu + rng.normal(scale=1., size=(n, 17))) / n


n_fit, n_chk, trials = 5257 // 32 * 2, 2215 // 32 * 2, 2000
fit_gain, chk_reopt, chk_frozen = [], [], []
for _ in range(trials):
    gf, gc = pool(n_fit), pool(n_chk)
    lf = rng.random(n_fit) < .5
    lc = rng.random(n_chk) < .5
    fit_gain.append(refinement.fixed_price_gain(gf, lf))
    chk_reopt.append(refinement.fixed_price_gain(gc, lc))
    # Frozen-action diagnostic: actions chosen on fitting rows, scored on checking rows.
    zp = gf.sum(0).argmin(); zl = gf[lf].sum(0).argmin(); zr = gf[~lf].sum(0).argmin()
    chk_frozen.append(gc.sum(0)[zp] - gc[lc].sum(0)[zl] - gc[~lc].sum(0)[zr])
fit_gain, chk_reopt, chk_frozen = map(np.asarray, (fit_gain, chk_reopt, chk_frozen))
passes = (chk_reopt + 1e-12 >= fit_b.MIN_CHECK_GAIN_RATIO * fit_gain) & (fit_gain > 1e-12)
print(f"A2 true gain 0; n_fit={n_fit}, n_chk={n_chk}")
print(f"   fitting gain: min {fit_gain.min():.4f} mean {fit_gain.mean():.4f}")
print(f"   checking (re-optimized, as coded): min {chk_reopt.min():.4f} "
      f"mean {chk_reopt.mean():.4f}; frac<0 = {(chk_reopt < 0).mean():.3f}")
print(f"   checking/fitting ratio >= 0.25 on pure noise: {passes.mean():.3f}")
print(f"   frozen-action checking gain: mean {chk_frozen.mean():.4f}; "
      f"frac<0 = {(chk_frozen < 0).mean():.3f}")


# ---- A3: empty positive-gain list labeled support_limited -------------------
class FrozenRisk:
    def probabilities(self, inputs):
        n = len(inputs.h_a)
        return {"SEX": np.tile([.4, .6], (n, 1)),
                "RAC1P": np.tile(np.ones(9) / 9, (n, 1))}


def rows(n, hh_offset):
    ha = np.zeros((n, 4)); ha[:, 0] = np.arange(n) % 2
    return {"x": np.zeros((n, 32)), "ha": ha,
            "token_codes": np.zeros(n, dtype=int), "teacher_p": np.full(n, .5),
            "residual": np.zeros(n), "households": np.arange(n) + hh_offset,
            "weights": np.ones(n)}


n = 800                                  # 400 unique households per child
fit_rows, chk_rows = rows(n, 0), rows(n, 10_000)
g = np.tile(np.r_[0., np.ones(16)], (n, 1)) / n   # every person prefers token 0 -> gain 0
sel = fit_b.select_nested_partition(fit_rows, chk_rows, FrozenRisk(), g, g,
                                    max_states=33, min_households=100,
                                    min_effective_households=100,
                                    feature_names=("h_a_0",))
direct = refinement.rank_splits(leaf_ids=np.zeros(n, int), leaf=0,
                                features={"h_a_0": fit_rows["ha"][:, 0]},
                                priced_rows=g, households=fit_rows["households"],
                                weights=fit_rows["weights"], feature_names=("h_a_0",))
print(f"A3 support-passing candidates from rank_splits: "
      f"{[(c.children_households, c.gain) for c in direct]}")
print(f"   select_nested_partition status = {sel['status']!r}; "
      f"candidate_history = {sel['candidate_history']}")


# ---- A4: "198 < 200" is necessary only if no household straddles a split ----
# Children are disjoint in PERSONS, not households (refinement.py:332-338). With
# a person-level feature (residual, task_posterior, risk), one household can
# count toward BOTH children's unique-household totals.
hh = np.repeat(np.arange(198), 2)                 # 198 households x 2 people
x = np.tile([0., 1.], 198)                        # the two members differ on the feature
g = np.zeros((396, 17)); g[x == 1, 0] = -1.; g[x == 0, 1] = -1.   # opposite preferred tokens
cands = refinement.rank_splits(leaf_ids=np.zeros(396, int), leaf=0,
                               features={"residual": x}, priced_rows=g / 396,
                               households=hh, weights=np.ones(396),
                               feature_names=("residual",))
print(f"A4 parent unique households = {len(np.unique(hh))}; admitted candidates: "
      f"{[(c.children_households, tuple(round(v, 1) for v in c.children_effective_weight), round(c.gain, 4)) for c in cands]}")
