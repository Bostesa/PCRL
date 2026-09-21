# RESEARCH_DECISION — `pcrl_stochastic_replacement_overnight_v1`

**No competitive operating point was established.** The programme closed at its first scientific
gate, before any privacy constraint was applied, so **nothing here measures a utility/disclosure
tradeoff at all.**

All numbers are 2018 **development** on repeatedly used pools, **validation split only** — the test
split was never read. 2016 sealed, 2017 not opened. **Cloud spend $0.00** against a $50 ceiling.

---

## 1. The actual method change

The precursor asked whether a finite code `T` **adds** utility to `[H_A, Z_J]`. That is the wrong
question for a replacement, and `PRECURSOR_CORRECTIONS.md` C7 shows the extension framing cannot
deliver a strict disclosure improvement even in principle: an attacker may ignore anything appended,
so equality with `J` is the best attainable.

This study changed the **contract**, not the mechanism. Under **S1** (`RELEASE_CONTRACT.md`) the
release is `A = [H_A, Z]` with `Z_J` **removed**; `J` becomes a comparator. A replacement *can*
strictly remove information, which the exact-arithmetic counterexample in §3 of that document shows.

Prior external disclosure of `J` is **not established** — a full search of results, docs, paper
bodies, docstrings and the git log found no documented external recipient, and the project
repeatedly describes a hypothetical owner and locally-retained arrays. S1 is therefore scoped as
explicitly **hypothetical and prospective**, not as revocation.

## 2. Primary verdict

**Gate G_R (representation screen) FAILED for both registered code families at both registered
resolutions.** Per the registered rule the unquantized controls were completed, quantization loss was
separated from probe failure, and the **constrained ACS fitting branch is closed**. No `Q` was fitted
on ACS; stages A, Q, the audits and the finalist stress never ran.

| family | k | mean deficit vs `J`, unweighted | person-weighted | anchors with gain over `H` | pass |
|---|---|---|---|---|---|
| `pca32` | 64 | **+0.00576** | +0.00422 | 3 / 3 | no |
| `pca32` | 256 (fallback) | +0.01953 | +0.01744 | 1 / 3 | no |
| `zj` | 64 | +0.01903 | +0.01748 | 3 / 3 | no |
| `zj` | 256 (fallback) | +0.02255 | +0.02103 | 0 / 3 | no |

Allowance was `.001` nats. The best code tested misses it by roughly sixfold.

## 3. Strongest positive result

**The unquantized A-side view attains strictly lower residence loss than the same-host `J`
channel**, and it survives the frozen simultaneous correction (Bonferroni over the family of 68,
adjusted one-sided `z = 3.18`, 5,443 shared households, three anchors pooled):

| contrast | estimate | SE | adjusted one-sided upper | below zero? |
|---|---|---|---|---|
| `H+raw(pca32)` − `J`, unweighted | **−0.01533** | 0.00279 | **−0.00646** | **yes** |
| `H+raw(pca32)` − `J`, person-weighted | **−0.01850** | 0.00332 | **−0.00794** | **yes** |
| `H+T(pca32,64)` − `H`, unweighted | −0.01679 | 0.00374 | −0.00488 | yes |
| `H+T(pca32,64)` − `H`, person-weighted | −0.01680 | 0.00463 | −0.00207 | yes |

So the A-side inputs carry residence signal **beyond** `J`'s 16-coordinate auxiliary channel, and
even the 64-state code adds real capability over `H` alone. This **quantitatively contradicts the
precursor's headline**, which asserted the code's signal was "already subsumed by the released
service predictions".

**This is a utility result only.** No disclosure was measured, so it is not a tradeoff claim and
certainly not a competitive operating point.

## 4. Strongest counterexample to my own framing

My registered prediction **P1 — "the `zj` family passes and `pca32` fails" — was refuted on both
halves.** Both failed, and `pca32` was clearly the *better* of the two (+0.0058 against +0.0190).
My stated reason was wrong: `Z_J` is already a 16-dimensional compressed summary, so quantizing it
into 64 states destroys proportionally more than quantizing the richer 32-dimensional input space.

**P2 — "quantization costs measurable utility" — was confirmed decisively**, and it is the
diagnosis:

| family | k | `H+raw` deficit vs `J` | `H+T` deficit vs `J` | **quantization cost** |
|---|---|---|---|---|
| `pca32` | 64 | −0.01533 | +0.00576 | **+0.02109** |
| `pca32` | 256 | −0.01533 | +0.01953 | +0.03485 |
| `zj` | 64 | 0.00000 | +0.01903 | +0.01903 |

**The failure is the coder, not the probe.** The unquantized view has no deficit at all; the
64-state code gives away ~0.02 nats. At `k = 256` the one-hot block is 256 columns on 4,539 fitting
rows, so the registered fallback is additionally estimation-limited — which is why more resolution
made it worse, not better.

Internal check: `zj`'s unquantized view reproduces `J` **exactly**, max per-row difference `0.0`.
Per `PRECURSOR_CORRECTIONS.md` C3 that identity confirms the wiring and is **not** reported as
evidence of anything, and its zero variance is **not** precision.

## 5. Local versus coalition behaviour

Only measurable on the registered synthetic model, because no ACS audit ran. There the deterministic
family is enumerable, so the comparison is **certified** (exhaustive over all `2^5 = 32` and
`3^5 = 243` maps), not heuristic:

| constraint set | budget | stochastic | best deterministic | randomization advantage |
|---|---|---|---|---|
| **L** (local only) | 0 / .002 / .01 | 0.000000 | 0.000000 | **none, at any budget** |
| **C** (local + coalition) | 0 | 0.339397 | 0.400000 | **+0.060603** |
| **C** | .002 | 0.224613 | 0.400000 | **+0.175387** |
| **C** | .01 | 0.098447 | 0.400000 | **+0.301553** |

**In this model randomization buys nothing locally and a great deal under coalition constraints**,
where the deterministic optimum collapses to the constant map. The model deliberately admits an
exactly-private nonconstant deterministic map (three subsets of its sensitive rates average to the
overall rate), so the local null is a real finding rather than a rigged construction — a fixture
caught an earlier docstring overclaiming the opposite and the description was corrected.

This is a statement about a specified finite model. It is **not** an ACS finding and **not** a
general superiority claim over all deterministic mechanisms.

## 6. Remaining uncertainty, stated plainly

* **Whether a replacement can be competitive is unresolved, not answered.** The mechanism was never
  constrained on ACS. Nothing here is evidence for or against the stochastic release.
* **A failed finite predictor is not proof that useful information is absent** — and in this study
  the opposite was measured: the unquantized view beats `J`. The binding limitation is the
  **registered coder**, and only two code families at two resolutions were tested.
* **A better coder is an unregistered hypothesis.** It is the obvious next question and it is
  explicitly *not* claimed here. Testing it requires a new registration.
* **No disclosure measurement exists**, so the `.001` margin, the protection and utility routes, and
  the external-baseline comparisons were all untriggered.
* Repeatedly used 2018 development pools, task-informed throughout: residence guided the screen, so
  nothing here is a reserved-task result.

## 7. Registered predictions, resolved

| # | prediction | outcome |
|---|---|---|
| P1 | `zj` passes the screen, `pca32` fails | **REFUTED on both halves** (§4) |
| P2 | `H+raw` beats `H+T`; quantization costs utility | **CONFIRMED decisively** |
| P3 | SUP action library passes at `K=17`, maybe fails at `K=9` | **not reached** |
| P4 | LF does not reach a competitive operating point | **not reached** |
| P5 | a passing candidate would be SUP, improving a coalition endpoint first | **not reached**; the synthetic result is *consistent* with the coalition half, but that is a different model and not a test of P5 |
| P6 | more likely than not, closes without a competitive operating point | **held**, though via the representation screen rather than via the mechanism |

## 8. What was not tested

Stages A, Q, audits and finalist stress. `LEACE-on-A0`, `SPLINCE-on-A0` and the OptNet adaptation
were never compared, because the audit stage they belong to never ran; their artifacts live in the
unrestored `invariant` and `competitive` archive groups and restoring ~10 GiB for a closed branch was
not justified. `H`, `J` and `A0` were available and used. Commute was never touched.

## 9. Next question, and what would not justify one

The honest next question is **whether a better coder closes the ~0.02-nat quantization gap** — since
the information demonstrably exists in the A-side inputs and the deficit is attributable to the
code. That is a **separate prospective study** with its own registration. Expanding this one's grid,
adding a third code family, opening another year, or relaxing the `.001` allowance would each be an
outcome-driven change and none is authorized.
