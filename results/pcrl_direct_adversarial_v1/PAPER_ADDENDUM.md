# PAPER_ADDENDUM — direct adversarial refinement of the neural auxiliary channel

For Terminal 2. **Permitted and rejected claims are listed explicitly in §6.** Terminal
2 may revise prose freely but must not silently change a scientific endpoint, a
selection, a decision rule or a reported sign.

_Numeric sections are generated from `DEVELOPMENT_2018.json`, `PAIRED_INTERVALS.csv`,
`EXPLORATORY_2017.json`, `MECHANISM.json` and `STRESS_SUMMARY.json`. Where this file
states a number, that file is the authority._

## 1. What was done

The strongest developed protection channel in this line of work is `J`: the frozen
16-coordinate neural auxiliary channel trained against a nine-role observer ensemble
with coefficients `individual = -0.1`, `coalition = -0.1`. Four studies of spectral
moment penalties have failed to match it, and the previous study found that a
closed-form 2023 linear eraser (LEACE) applied to the `A0` channel matches `J` on every
sensitive endpoint.

This study asks whether a **directly adversarial refinement of that same channel**
improves the measured frontier. The starting point is the frozen `A0` channel,
recovered from its checkpoint and asserted **bitwise identical** to the released 2018
wire on all seven pools and all three seeds. Two widths: 16 (`A0` itself) and 8 (`A0`
followed by one deterministic training-only PCA projection).

The mapper minimises

```
U(theta) + beta * policy_penalty,
U(theta) = mean source CE from Z-only heads + 1.0 * teacher_distortion
```

against a refreshed ensemble of residual-logit attackers. For each protected role a
service-only predictor `p0_j` is fitted on that role's `H` view alone and frozen; every
attacker is parameterised as a correction to its logits, so the empirical incremental
gain `g = CE(p0_j) - CE(q_jk)` is zero at initialisation by construction.

## 2. Ingredient attribution — nothing here is new

* Madras, Creager, Pitassi & Zemel, *Learning Adversarially Fair and Transferable
  Representations*, ICML 2018 — <https://proceedings.mlr.press/v80/madras18a.html>.
  Adversarial encoder-versus-discriminator training and the transfer framing.
* Belrose et al., *LEACE*, 2023 — <https://arxiv.org/abs/2306.03819>. Closed-form
  affine concept erasure; used here as a control on a new input channel.
* SPLINCE and OptNet-ARL: the primary sources pinned in `BASELINE_ADAPTATIONS.md` at
  `73903b7f`, reused verbatim.

Attacker ensembles, attacker refresh, residual/offset parameterisation and
representation compression are all established. **The contribution claimed is an
adaptation to this project's immutable-service and recipient/coalition access contract,
and its empirical evaluation — not any of the ingredients.** Novelty is established
separately from performance, and any novelty statement is phrased as absence of
evidence after a bounded search.

## 3. Method facts a paper must state

* **Widths.** 16 starts bitwise at `A0`; 8 starts at a training-only PCA projection of
  the same `A0` output, with heads carried through algebraically. The PCA sees no
  reserved label.
* **Release contract.** `A = [H_A, Z]`, `B = H_B`, `AB = [A, B]`. `H_A` (4 coordinates)
  and `H_B` (2) are bitwise preserved in every released wire, asserted per pool per arm.
  `Z` is computed from permitted inference inputs only.
* **Reserved labels.** `same_residence` and `commute_over20` never enter encoder
  training, checkpoint selection, calibration or release selection; the label helper
  deletes those keys before returning. Because the project has inspected these tasks
  repeatedly across its history, **the overall development process is not blind to
  them** and must not be described as blind.
* **Selection.** Each checkpoint is scored against its **own freshly initialised**
  attacker slate trained for the same fixed budget, on an internal representation-training
  monitor fold. This is a selection yardstick, not a protection measurement.
* **Primary audit scope** is `kernel_expanded_independent` — **matched exposure**. The
  inherited `kernel_expanded_catchup` scope gives historical arms saved-observer
  catch-up attacks that no arm in this study received (`RUN_STATUS.md` amendment 4).

## 4. Statistics

Paired household-cluster bootstrap, 2000 replicates, cohort `SERIALNO` clusters drawn
once per replicate and applied to every seed. Two adjustments are reported side by side:
the reused within-contrast studentized max-|t| across the five family endpoints, and a
**candidate-wide** studentized Bonferroni over the entire searched family (86 contrasts
x 5 endpoints x 2 weightings = 860 comparisons, `z = 4.02`).

Rules are kept separate and never swapped: the historical `.001` coordination rule is a
**point-estimate** rule reused for continuity; utility noninferiority requires the
one-sided adjusted bound to be at most `.001`; equivalence requires its own two-sided
margin procedure; **absence of significance is neither**.

All uncertainty is **development uncertainty conditional on fitted systems**. It does
not undo repeated use of the 2018 pools. Three anchor seeds do not establish broad
training-population robustness.

## 5. Results

See `DEVELOPMENT_2018.md`, `EXPLORATORY_2017.md`, `FRONTIER_ANALYSIS.md`,
`ATTACK_STRENGTH.md`, `OPTIMIZATION_STABILITY.md` and `RESEARCH_DECISION.md`. The
headline verdict and the nominee conjunction are in `RESEARCH_DECISION.md` and are the
authority for any claim in a paper.

## 6. Permitted and rejected claims

### Permitted

* That the frozen `A0` channel was recovered and **proved bitwise identical** to its
  released 2018 wire before being used as a starting point.
* That the `leace_A0`, `splince_A0` and `optnet16_*` adaptations were **completed on
  2017** for the first time, with every reconstruction proved bitwise identical to the
  recorded 2018 artifacts.
* That measured recovery against **this declared finite attack family**, on these
  development pools, is what it is reported to be.
* That the reported contrasts are prespecified and that the simultaneous family
  includes every contrast searched.
* That a bitwise-identical channel reproduces its comparator's endpoints exactly under
  the matched-exposure scope (an end-to-end correctness result).

### Rejected — do not write these

* **Not** independence, differential privacy, a mutual-information bound, or protection
  against arbitrary attackers. Low measured recovery means a declared finite family did
  not find a gain inside its budget.
* **Not** a solved minimax problem. Finite alternating optimisation confers no
  certificate.
* **Not** a new conditional-information estimator. Subtracting the fixed `CE(p0_j)`
  creates no encoder gradient; `p0_j` does not depend on `theta`.
* **Not** a guarantee that teacher fidelity implies residence transfer; that is measured
  in `FRONTIER_ANALYSIS.md`, not assumed.
* **Not** a confirmation of anything. Every number is development, on repeatedly used
  pools, and the 2017 partition is spent.
* **Not** a width match when effective rank drops; ambient width is reported separately
  from realised rank.
* **Not** a reason to open 2016. **2016 remains sealed and was not touched.**
* **Not** a comprehensive benchmark of any cited method.

## 7. Figure and table sources

Every table in this study is generated from a machine-readable file and names it.
`PER_SEED.csv`, `PAIRED_INTERVALS.csv`, `PER_SEED_SIGNS.csv`, `SCORE_REPLAY.csv`,
`EXPLORATORY_2017.csv`, `MECH_*.csv`, `ARTIFACT_MANIFEST.json`. Per-person predictions
are on disk under each unit's `predictions.npz` and are hashed in the unit records.
