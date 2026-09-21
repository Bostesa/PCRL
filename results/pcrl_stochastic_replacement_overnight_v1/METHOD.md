# METHOD — mathematical specification

Locked 2026-09-21 before any new validation outcome. Established mathematics is cited, not claimed.

## 1. Mechanism

`T = g(X_A)` is a finite code over permitted A-side inference inputs. Learn `Q` with
`Q[t,z] >= 0`, `sum_z Q[t,z] = 1`. Recipient `A` receives `H_A` plus **one sampled token** `Z ~ Q[T,:]`
as a public one-hot; recipient `B` receives `H_B` unchanged.

**Never released:** the row `Q[T,:]`, its logits, an expected prototype, or any seed that exposes a
person's sampling uniform. `Q` and the codebook are **public** and assumed known to attackers.

**Deployment randomness vs simulation randomness.** One token per person, drawn once and reused;
independent redraws are *not* the deployment contract. Scientific reproducibility uses a separate
simulation seed stream. A public seed combined with a public person identifier must not reveal that
person's sampling uniform — so deployment draws come from a **private** key held outside published
artifacts, and the published seed reproduces only the *expected* quantities, never a person's draw.
Repeated Monte Carlo draws are **not** new people.

## 2. Convex programme

For protected role `r` with attribute `S_r` and finite conditioning partition `C_r`, and a fixed cost
matrix `D`:

```
minimize_Q   sum_{t,z} p(t) Q[t,z] D[t,z]
subject to   Q[t,z] >= 0,  sum_z Q[t,z] = 1,
             I_Q(S_r ; Z | C_r) <= delta_r   for required roles r
```

`I_Q(S_r;Z|C_r) = sum_{s,c,z} a log(a/b)` with `a = p(s,c,z)` and `b = p(s|c) p(c,z)`, both **affine**
in `Q` under the mechanism's Markov property `Z — T — (S,C)`. Relative entropy is jointly convex, so
each constraint is a convex sublevel set and the programme is convex with a linear objective. This is
standard; see Rassouli & Gündüz, *On Perfect Privacy* (the `δ=0` linear-cost case already has an LP
there, including squared-error and error-probability utilities), and Makhdoumi et al. for why the
privacy funnel is nonconvex — its utility **constraint** `I(X;Y) >= R`, which we do not impose.

**No new nullspace theorem, no new privacy funnel, and no population privacy claim from a fitted
table are asserted.** A fitted-model constraint is a statement about the fitted model only.

**Zero budget.** At `δ = 0` prefer the equivalent **linear** system `A_r Q[:,z] = 0` per token per
required distribution/role, where `A_r[(s,c), t] = p_r(t,s,c) - p_r(s|c) p_r(t,c)`, then
**independently recompute** the modelled MI. This avoids an unnecessary relative-entropy boundary
problem. A constant `Q` is always feasible at `δ = 0`; failure to find a useful null direction is
**not** infeasibility of the channel simplex.

## 3. Constraint sets, budgets, partitions

* **L**: `A/SEX`, `A/RAC1P`.
* **C**: `L` plus `AB/SEX`, `AB/RAC1P`.
* Budgets `delta in {0, .002, .01}` nats per constrained role. **These are finite-model optimization
  settings and are never equated with the empirical `.001`-nat allowance relative to `J`.**
* Other forbidden roles are still audited, just not constrained.

**Partitions** (label-independent; frozen before outcomes). `C_A`: the non-redundant `H_A`
probabilities clustered into **4** cells by k-means on `representation_fit` with salt `20269022`,
crossed with public-coverage probability split at its `representation_fit` median → **2** bins, giving
`4 × 2 = 8` cells. `C_AB`: that, crossed with `H_B` coarsened to **2** bins by its first coordinate's
median → **16** cells. One registered coarser fallback, triggered by **feature-cell counts alone**
(any cell with fewer than 40 rows, or any (cell, class) with fewer than 10): drop the coverage split,
giving `4` and `8` cells.

Conditioning coordinates depend only on permitted predictions/features — never the sensitive label,
the residence label, a per-row loss, or correctness. (`PRECURSOR_CORRECTIONS.md` C4 is the defect this
rule exists to prevent.)

Distributions `p_r(s,c,t)` are estimated on `representation_fit` under **both** unweighted and PWGTP
weighting; if both scoring regimes are claimed, the **intersection** of both constraint sets is
imposed on a single `Q`, which remains convex for fixed distributions. Reported per cell: effective
counts, missing classes, pseudocounts, and sensitivity to smoothing. **A smoothed absent category
remains absent data.**

For an input-code state absent from cost/distribution fitting, a preregistered **common constant
output** row is used rather than an undefined cost or an arbitrary unconstrained row; its usage
frequency is reported.

## 4. Costs

### SUP diagnostic — fixed expected residence log loss

1. Fit baseline residence decoder `b(H_A)` on the **decoder-fit** household subset of
   `representation_fit`. **Freeze it.**
2. Public action set: `sigmoid(logit(b(H_A)) + a_z)` with offsets `a_z` **evenly spaced over
   `[-4, 4]`, including exactly `0`**. Output alphabets `K = 9` and `K = 17`.
3. `D[t,z]` = mean loss of action `z` over **cost-fit** observations with code `t` — a disjoint
   subset, so the decoder never saw the rows that set its costs.

The zero-offset constant mechanism reproduces `b(H_A)` exactly, which is the reference point of the
library. **Offsets, decoder and `Q` are never jointly optimized** — that would destroy convexity, and
no convexity claim is made for such a variant. Downstream **fresh probes** remain the actual utility
evaluation; `D` is only the optimizer's objective.

**Registered action-library fallback** (one only): replace the offset grid with `K` prototype-specific
regularized residence decoders depending on `H_A` alone, trained on the decoder-fit subset and frozen
before cost estimation, with a declared **global-decoder fallback** for prototype groups with fewer
than 100 decoder-fit rows. This changes the action family, so the report always identifies which
library produced a number.

### LF control — fixed normalized representation distortion

`K` prototypes fitted by k-means on the **cost-fit** inputs of that family (frozen before `Q`
optimization); `D[t,z] = ||centroid(t) - prototype(z)||^2 / mean_t,z ||centroid(t) - prototype(z)||^2`.
Scale, fitting rows and prototype rule fixed. This is an **objective control**, not evidence that
reconstruction is safe: reconstructing sensitive-correlated directions can be counterproductive.

## 5. Scoring a stochastic release

For a frozen attacker `f` and person `i` with code `t_i`, the **expected** per-person loss is
`sum_z Q[t_i, z] * loss(f(H_i, z), y_i)` — an average of **losses**, not of probabilities.
`log` of an averaged prediction is a different estimand and is not used.

Attackers for the token mechanism are fitted by **exact weighted expansion** where supported:
person `i` contributes `K` rows with weights `Q[t_i, z]` summing to that person's original weight
(PWGTP multiplies it), and the household identity is carried on every expanded row. Regularization and
epoch normalization must not treat `K` expanded rows as `K` independent people; a constant-token
identity fixture checks this. Where exact expansion is unsupported, a registered sampled training
schedule is used and its Monte Carlo variability reported.

**A sampled person-token pair is the released view.** Enumerating tokens for integration does not
grant an attacker all tokens. A sampled-wire parity check runs alongside expected scoring.
