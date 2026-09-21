# VALIDATION

## Independent replay — `REPLAY.json`, 14 / 14 pass

Headline contrasts were reconstructed from **per-person losses** by a different code path than the
one that produced them (plain anchor-mean of per-row differences, versus the household-weighted
shared-resample aggregate):

| quantity | recomputed | published | abs diff |
|---|---|---|---|
| `H+raw(pca32)` − `J`, unweighted | −0.015328 | −0.015328 | 0.00e+00 |
| `H+T(pca32,64)` − `H`, unweighted | −0.016791 | −0.016791 | 0.00e+00 |
| `H+T(pca32,64)` − `J`, unweighted | +0.005760 | +0.005760 | 8.67e-19 |

All eight published mean inclusive log losses for `pca32` and `zj` at `k=64` reconstruct to within
`5.1e-10`.

## Routing and layout assertions

* Wire layout asserted on **every** pool: `H_A` 4 columns, `H_B` 2, `Z_J` 16.
* **S1 routing**: the `H+T` candidate view was scanned for any 16-column block equal to `Z_J` —
  **none present**. A `J`-only predictor is therefore never available inside an S1 candidate's slate,
  as `RELEASE_CONTRACT.md` requires.
* Ancestor inclusion is applied to **every** view including the references, so no comparison is
  asymmetric in its favour.

## Identity check, reported as an identity

`zj`'s unquantized view reproduces `J` with **max per-row difference exactly 0.0** and zero variance.
This confirms the family is wired to the real `Z_J`. Per `PRECURSOR_CORRECTIONS.md` C3 it is **not**
evidence of anything and its zero variance is **not** precision.

## Fixture suite — 45 study fixtures, plus the inherited repository suite

| file | targets |
|---|---|
| `test_precursor_corrections.py` | the seven corrected errors, incl. the `min()` identity, label dependence of the withdrawn partition, the not-a-ceiling negativity, and the exact-arithmetic replacement counterexample |
| `test_ledger.py` | reconnect safety: content-hash reuse, atomic promotion, lock refuses a live duplicate, breaks only a genuinely dead holder, and a status file is explicitly **not** evidence of liveness |
| `test_tokens.py` | expected loss averages **losses not predictions** (and differs from the Jensen-biased alternative); `K` expanded rows are **one person** — weights sum to the person weight, PWGTP multiplies it, household and label carried, constant-token identity, weighted-gradient duplication identity; released token block is one-hot, never a row of `Q` |
| `test_synthetic.py` | enumeration completeness, coalition role strictly stricter than local, constant channel always feasible at `δ=0`, recorded comparison internally consistent, and **2016 containment** over machine-read artifacts and the live loader tree |

Two fixtures **caught real defects in my own work** and are the reason they are not in the
published record: an overclaim that no subset of the synthetic model's sensitive rates averages to
the overall rate (three do — corrected, and it is exactly why the local case shows no randomization
advantage), and a containment test that flagged the prose documenting the containment rule.

## Numerical records

`synthetic/DETERMINISTIC_COMPARISON.json` carries, per cell, the solver status, the **independently
recomputed** `cmi` per role, the constraint violation, and the exhaustive deterministic enumeration
count (`k**5`, verified). Every stochastic solution satisfies its budgets on recomputation to
`<= 1e-6`. No numerical incident, quarantine or corrupted array occurred; no unit was retried.

## What validation does *not* cover

No disclosure measurement exists, so there is no audit replay, no attacker-slate parity check and no
frozen-finalist verification — those stages never ran. The utility panel is labelled a **utility
panel, not a frontier**, because there is no disclosure axis to plot.
