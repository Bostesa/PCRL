# PROTOCOL — `pcrl_stochastic_replacement_overnight_v1`

**Locked 2026-09-21, before any new validation outcome of this study was read.** Branch
`research/pcrl-stochastic-replacement-overnight-v1`, based on the precursor pin
`cd895e4291b3dc2c0ed5e9c5482415d30f4e9f2a`.

Companion documents, all part of this registration: `RELEASE_CONTRACT.md` (which contract, and the
evidence), `PRECURSOR_CORRECTIONS.md` (what the precursor got wrong), `METHOD.md` (the mathematical
specification), `STATISTICAL_PLAN.md`, `DATA_USE.md`, `RUN_MATRIX.json`.

## 0. Boundaries

* **2016 is sealed.** No transform, label, fit, score, plot, selection, manifest entry, archive or
  cloud bundle file. Loaders are blocked from reaching it and a test asserts the block.
* **2017 is not opened.** A second spent year is lower priority than resolving the primary mechanism
  and audit quality.
* 2018 is the development pool. The precursor's 2018 results are already development evidence; this
  study is **also** development. Neither a fresh split of already-used people nor a new bootstrap
  restores independence from prior research.
* `H_A` and `H_B` byte-identical to the stored release on every pool, asserted per unit.
* **Contract: S1** (`RELEASE_CONTRACT.md`). `Z_J` is a comparator, never released in an S1 candidate.
  Prior external disclosure of `J` is **not established**; S1 is explicitly hypothetical/prospective.
* No rate constraint on `Z` is ever added.
* The whole programme is **task-informed development**: residence results guide the screen and the
  nomination, so calling an objective label-free does not make the selected method a reserved-task
  result.

## 1. Question

Does replacing the auxiliary channel with a small randomized finite token reach a better
utility/disclosure operating point than the `J` alternative, for recipients who never received `J`?

## 2. Pools and label use

Household-separated historical pools, reused unchanged:

| pool | rows (seed 0) | use in this study |
|---|---|---|
| `representation_fit` | 10,513 | codebook fitting (label-free); decoder-fitting and cost-fitting subsets; constraint distribution estimation |
| `downstream_fit` | 4,539 | utility probe fitting (**all** eligible households, no 2048 cap) |
| `downstream_validation` | 2,984 | screen and selection |
| `attacker_fit` / `attacker_validation` | 2,993 / 2,985 | audit attacker fitting and selection |
| `test` | 2,982 | frozen finalist evaluation only |

`representation_fit` is split by household into disjoint **decoder-fit** (first 50%) and **cost-fit**
(second 50%) subsets by the frozen hash rule in `METHOD.md`; the SUP baseline decoder is fitted on the
former and frozen before costs are estimated on the latter.

Label use is explicit per artifact in `DATA_USE.md`. Codebooks see **no labels of any kind**.

## 3. Stages and gates

* **R — representation screen** (§6 of the assignment, implemented in `replacement.py`).
  Two families (`pca32`, `zj`) × `k = 64`, with exactly one registered `k = 256` fallback each.
  Views compared: `H`, `H+J`, `H+T`, `H+raw`, each ancestor-inclusive with `H`.
  **Pass rule** (point-estimate scheduling, *not* a confidence claim): inclusion-respecting `H+T` is
  no worse than same-host `J` by more than `.001` nats in **both** weighted means, **and** shows a
  positive gain over `H` in **at least two** anchors under both weightings. First passing resolution
  per family is selected; every failed result is retained; all three anchors always evaluated.
  A `pca32` failure does **not** close the programme — `zj` is separately registered and must be
  completed. If both fail, complete the unquantized controls, diagnose quantization loss versus probe
  failure, and end the constrained ACS fitting branch. **No third code family is authorized.**
* **A — action-library check** (§7). Before spending on constrained fits, measure the unpenalized
  fixed-action SUP mechanism and the unconstrained LF mechanism, evaluated through **fresh probes**,
  under the same `.001` mean residence allowance versus `J` in both weightings, decided across all
  three anchors per family/output-size block. One registered fallback: replace the logit-offset grid
  with `K` prototype-specific regularized `H_A`-only residence decoders. If that also fails, close
  that SUP block as an **action-library limitation** — not as a privacy-constraint failure, since no
  privacy constraint has been applied yet. The LF block completes regardless.
* **Q — constrained fits** (§8), then **audits** (§10), then **frozen finalist stress** on `test`.

## 4. Gate ledger

`results/.../gates/` holds one JSON per gate with verdict, the measured quantities, and the decision
taken. States in the unit ledger: `planned`, `fitting`, `fitted`, `auditing`, `verified`, `failed`,
`duplicate`, `untriggered`. Reuse is decided by a content hash of the configuration.

## 5. Directional predictions, with reasons

Registered now; guesses are labelled as guesses.

1. **`zj` family passes the R screen; `pca32` fails it.** Reason: evidence, not a guess — the
   precursor measured that `[H_A, Z_J]` beats `[H_A, T_pca32]`, and `Z_J` is by construction the
   channel a J-derived code quantizes, so a `zj` code should preserve more of `J`'s residence content
   than a raw-input code. *Moderately confident.*
2. **`H+raw` beats `H+T` for both families**, i.e. quantization costs measurable utility. Reason:
   a 64-state code discards within-cell variation. *Confident; this is close to arithmetic.*
3. **The SUP fixed logit-offset library passes the A gate for at least one family at `K = 17`, and
   may fail at `K = 9`.** Reason: a 9-action library around a frozen `b(H_A)` is coarse. *Guess.*
4. **The LF control does not reach a competitive operating point.** Reason: a distortion objective
   is not aligned with residence, and reconstructing sensitive-correlated directions can be
   counterproductive. *Moderately confident.*
5. **If any candidate passes the full protection-route conjunction it will be SUP, not LF, and the
   strict improvement will appear on a coalition endpoint before a local one.** Reason: the coalition
   view is wider, so `J`'s auxiliary channel has more to leak there. *Guess.*
6. **Overall: more likely than not this closes without a competitive operating point.** The
   precursor's and predecessors' repeated negatives are the base rate. But unlike the precursor, the
   question being asked is now the right one for a replacement, so a real answer either way is
   expected rather than a mis-specified one.

## 6. What a negative result must state

Which contract was tested, which code families and resolutions were actually run, which action
library was used, which gate stopped it, and what was *not* tested. A failed finite predictor is
**not** proof that useful information does not exist (`PRECURSOR_CORRECTIONS.md` C2).

## 7. Adaptive branches, all registered before outcomes

`k = 256` fallback per family; the prototype-decoder action-library fallback; `K = 9` first with
`K = 17` as the expansion tier; one alternate numerical solver/tolerance schedule; one coarser
constraint partition derived from feature-cell counts alone. Exhausting these closes the branch
scientifically. **No outcome-driven threshold relaxation, third encoder family, extra year, enlarged
budget or new task is authorized.**
