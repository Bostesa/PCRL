# RESEARCH_DECISION — `pcrl_stochastic_channel_v1`

**Verdict: the study closes NEGATIVE at gate G2 (capacity), before the stochastic mechanism was ever
constrained.** All numbers are 2018 **development** on repeatedly used pools, validation split only;
the test split was never read. 2016 is sealed and untouched. Cloud spend **$0.00** against a $50
ceiling — the gate failed on a diagnostic that ran in seconds on CPU, so launching cloud compute for
stages the gate forbids would have been waste.

Stages C and D were **not run**. No grid was expanded. No threshold was changed.

## 1. The finding

A prespecified, label-free A-side code `T` carries **genuine** residence signal, and that signal is
**already subsumed by the released service predictions `J`**.

Validation residence log loss (nats, lower is better):

| seed | prior | `T` only | `J` only | `J + T` |
|---|---|---|---|---|
| 0 | 0.54501 | 0.53093 | **0.52507** | 0.52598 |
| 1 | 0.52859 | 0.51292 | **0.50114** | 0.50586 |
| 2 | 0.55554 | 0.53321 | **0.52856** | 0.53097 |

`T` alone beats the prior by 0.014–0.022 nats in every seed, so the quantizer is not noise. `J` alone
beats `T` alone in every seed. **Adding `T` to `J` does not improve on `J`, and is slightly worse.**

This is *not* "compression destroyed the signal" — the first outcome row of the registered outcome
table. It is the second structural issue the review memo raised in its §1, now measured directly:
a residual computed against `J` need not contain information missing from `J`. Here the A-side
inputs' residence-relevant content is, at this probe family's resolution, already exhausted by the
service predictions that were released. That is a statement about this **release contract**, and it
is the most useful thing this study produced.

## 2. G2, as measured

Probe family: the house utility family (`logistic` + `mlp`, selection by validation log loss), on the
historical subset indices, so the comparison is same-host.

| resolution | seed-mean advantage, unweighted | person-weighted | seeds positive | inclusion-respecting |
|---|---|---|---|---|
| `\|T\| = 64` (declared) | **−0.00268** | −0.00288 | **0 / 3** | 0.00000 in 3/3 |
| `\|T\| = 256` (the one predeclared fallback) | **−0.01136** | −0.00939 | **0 / 3** | 0.00000 in 3/3 |

Two accountings are reported because they answer different questions. **Code-only** gives the probe
`[H_A, Z_J, onehot(T)]` and nothing else; at `|T| = 64` that is 84 columns against the baseline's 20,
on 2048 fitting rows, so a finite learner can be *handicapped* by width — the same effect that
motivated in-slate J predictors on the protection side (`VERIFICATION.md` §A).
**Inclusion-respecting** also admits the J-only predictor, which is the accounting consistent with
METHOD's inclusion fact and the one G2 is judged on. It is exactly `0.00000` in all six cells: the
J-only predictor always wins.

Candidate-specific paired-household intervals are reported in `gates/G2.json` and were **not**
withheld on the grounds that historical precision was poor (`AMENDMENT_1.md` §4). At `|T| = 64` the
bootstrap SEs are 0.0048–0.0059 nats over ~2000 households; the advantage estimates are negative and
their one-sided upper bounds straddle zero, so the honest reading is **no demonstrated advantage**,
not a demonstrated deficit. The resampling unit is the household; replicating rows does not add
units, which is pinned by test.

Higher resolution is monotonically worse across all three seeds and both weightings — consistent with
both the probe-width handicap and with there being no extractable residence signal to find.

## 3. B1 — the fitted-model ceiling is not estimable here, and that is reported as such

| partition | ceiling by seed (nats) | diagnosis |
|---|---|---|
| per-column product, 3 bins × 20 columns | +0.0072 / +0.0043 / +0.0074 | **1.9–2.1 rows per cell** — unusable |
| baseline-probability deciles | −0.0693 / −0.0808 / −0.0881 | 10 × 64 = 640 cells on 2984 rows; the negative sign is a variance artifact, not a real deficit |

Conditioning on any reasonably fine baseline view crossed with 64 codes exhausts the 2,984 validation
rows. This is precisely the failure the review warned about in its §5 — conditioning on near-unique
raw `H` and reading off the result — so B1 is reported as **not reliably estimable at this sample
size** rather than as a number. G2 rests on B2, which is unambiguous. The sparsity was diagnosed from
rows-per-cell, a property of the partition and the data that is visible without reading any outcome.

## 4. Registered predictions: I was wrong on the more likely one

| # | registered | confidence | outcome |
|---|---|---|---|
| **P2** | Stage B shows a residence advantage over same-host J, positive but under `.003` nats | **55%** | **REFUTED** — negative in 3/3 seeds at both resolutions |
| **P3** | Stage B shows no usable residence advantage, stopping the study at the first gate | **30%** | **CONFIRMED** |
| P8 | precision check shows `z × SE > .001` on at least one endpoint | 50% | CONFIRMED literally; the over-general gloss withdrawn in `AMENDMENT_1.md` |
| P1, P4, P5, P6, P7 | — | — | **not reached**: the gate stopped the study before the mechanism was constrained |

P3 happened and I had it at 30% against P2's 55%. I was wrong about the more likely of the two, and
wrong in the optimistic direction. The overall expectation I put on the record — that the study would
close negative — held, but for a reason I ranked second.

## 5. What this does and does not establish

**Does:**
* At this resolution, this code and this probe family, the unconstrained release adds nothing to
  same-host J on residence, on the 2018 development pools.
* The code's residence content is subsumed by `J` rather than absent — a sharper and more useful
  negative than "compression destroyed it".

**Does not:**
* **Not a capacity theorem.** A different code, probe family, or resolution could differ. The review
  was explicit that failure of a simple release is an operational warning, not a theorem.
* **Not a statement about the ACS population.** Repeatedly used development pools.
* **Not a refutation of randomization.** The constrained stochastic mechanism was never reached,
  because the *unconstrained* release already failed. Nothing here bears on whether a
  role-constrained stochastic release would beat a deterministic one — that question is untouched.
* **Not a reserved-task claim in any direction.** Residence entered only its designated probe.

## 6. What survives for the manuscript

Independent of this negative verdict:

1. Two confirmed, sourced corrections to the predecessor's implementation, with fixes implemented and
   tested, and the correction's direction registered in advance (`VERIFICATION.md`, `STAGE_A.md`).
2. Validated finite-channel machinery treating the stochastic-optimization and nullspace results as
   **established mathematics**, with the coarse-conditioning counterexample that bounds what any
   fitted-model conditional-information claim can assert (`STAGE_4.md`).
3. An eight-point narrowing of the prior-art position (`VERIFICATION.md` §E).
4. `G0` as amended: limited precision near equality, and the prospective consequence that a
   confirmable pass requires a candidate that *reduces* additional sensitive recovery.
5. **The subsumption result in §1**, which offers a mechanism-level explanation for why this line of
   work has repeatedly gone negative: the release contract already hands out the A-side residence
   signal that an extension is trying to add.

## 7. Next step, if any — and what would not justify one

The honest next question is **not** a bigger grid on this code. It is whether any A-side statistic
carries residence signal *not* already in `J`. §1 says this quantizer does not, and §3 says the
sample size cannot support a fine-grained conditional answer. Either of those is a reason to change
the *question*, not to expand this design. A different code family, or a task whose signal is not
already released through `J`, would be a **separate prospective study** with its own registration.
