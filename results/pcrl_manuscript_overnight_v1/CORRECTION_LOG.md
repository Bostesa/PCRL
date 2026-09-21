# CORRECTION_LOG — overnight review session, 2026-09-21

Baseline manuscript `eb4aa96685568426c098effb00b857f1ae934b0d`. Diagnostic audited at
`cd895e4291b3dc2c0ed5e9c5482415d30f4e9f2a`. Replacement registration reviewed at `0af175c35164ef224b4dc8149b5a3aa309ac6280`.

Each row gives the original claim, the evidence that bears on it, the replacement wording, what it
affects, and whether any **execution** decision changed. No historical study record was edited; every
correction is additive. Where a defect is confined to interpretation, it is **not** propagated to results
it did not touch.

---

## N1 — "The code's residence content is already subsumed by `J`"

* **Original:** `pcrl_stochastic_channel_v1/RESEARCH_DECISION.md` §1 and the closing commit message:
  the A-side code's residence content "is **already subsumed** by the released service predictions".
* **Evidence:** `J+T` minus `J` is `−0.00091 / −0.00472 / −0.00241` nats against paired household
  bootstrap SEs of `0.00553 / 0.00479 / 0.00587`, i.e. **0.16, 0.98 and 0.41 SE**. The probe family is
  logistic + a 40-epoch MLP selected on validation log loss, fitted on 2048 rows, with 84 input columns
  against the baseline's 20. The logistic member degrades sharply when `T` is appended (0.5312 → 0.5494
  at anchor 0) while the MLP does not.
* **Replacement:** "At this probe capacity and these resolutions, adding the code to `J` produced no
  measurable improvement." Both "no incremental signal" and "insufficient probe capacity at this width
  and sample size" remain live; this design does not separate them.
* **Affects:** §VII of the paper and its conclusion. The direction of the finding is unchanged.
* **Execution decision changed:** none. G2 fails either way.

## N2 — The fitted-model ceiling from "baseline probability deciles"

* **Original:** a B1 ceiling of `−0.069 / −0.081 / −0.088` nats, described as "a variance artifact" of
  640 cells on 2,984 rows.
* **Evidence:** the partition coordinate is the baseline probe's **per-row log loss**
  (`stage_b._baseline_score` → `probe['per_row_loss']` → `_per_row`, which computes `−log p[i, y_i]`
  from `y_val`). It is a function of each person's own residence label. Signature: the decile baseline
  reaches **0.065–0.094 nats** where real residence log loss is ~0.52. Fixture
  `checks/b1_label_leak_fixture.py`: on a pure-noise label at the residence base rate, loss-decile
  conditioning "predicts" it at 0.057 nats against a true entropy of 0.590 — a 0.533-nat phantom
  reduction — while a label-free probability-decile control leaks nothing, and a balanced toy shows no
  leak at all.
* **Replacement:** the estimate is **withdrawn**, not reinterpreted. Sparsity is a second, smaller issue;
  the primary defect is conditioning on the outcome. No ceiling is used in any claim.
* **Affects:** §VII-A of the paper. The companion product partition is label-free but has 1.9–2.1 rows
  per cell and is unusable for a different reason.
* **Execution decision changed:** none, and this is load-bearing: G2's verdict is computed from the
  deployable probe comparison (B2). B1 gated, selected and stopped nothing, and the defect is
  deliberately **not** propagated to G2 or to any other study.

## N3 — "No anchor positive at either resolution"

* **Original:** `gates/G2.json` records `seeds_with_positive_advantage: 0` with no weighting named;
  `RESEARCH_DECISION.md` §4 says the advantage is "negative in 3/3 seeds at both resolutions".
* **Evidence:** person-weighted, anchor 0 is **positive** at both resolutions (`+0.00216` at `|T|=64`,
  `+0.00007` at `|T|=256`). Unweighted it is 0/3; person-weighted it is 1/3.
* **Replacement:** name the weighting wherever the count appears; report both.
* **Affects:** a sentence in §VII and Table II of the paper, which now prints both weightings per anchor.
* **Execution decision changed:** none. Anchor means are negative under both weightings.

## N4 — Provenance of the `prior` and `T`-only columns

* **Original:** the headline subsumption table and the closing commit message report `prior` and `T`-only
  validation log losses; "T alone beats the prior by 0.014–0.022 nats in every seed" is what establishes
  that the quantizer is not noise.
* **Evidence:** `J_only` and `J_plus_T` reproduce **exactly** from `STAGE_B_RESULTS.json`. `prior` and
  `T_only` appear in neither that file nor any committed code — `stage_b.py` fits only the `ref_J` probe
  and the `[J, onehot(T)]` probe.
* **Replacement:** reported as **reported-not-verified**, marked on the table itself.
* **Affects:** Table III of the paper and the sentence that the code carries genuine signal.
* **Execution decision changed:** none.

## N5 — "A `.001`-nat confirmation is not demonstrable for any mechanism"

* **Original:** the precursor's G0 report, and my own `G0_REVIEW.md` adopted the arithmetic while adding
  a different overreach.
* **Evidence:** the half-width is `z × SE`, and the paired SE belongs to the per-household loss
  differences of a **specific pair**. Terminal 1 withdrew the extrapolation in `AMENDMENT_1.md` on two
  grounds, both correct: the gate is one-sided so a sufficiently negative estimate passes at any observed
  half-width, and historical half-widths cannot lower-bound a new pair's SE.
* **Replacement:** "For the historical comparisons on these pools, precision is limited **near
  equality**." Nothing about mechanisms that do not exist.
* **Affects:** §IX of the paper.
* **Execution decision changed:** none; G0 was never an operational stopping gate.

## N6 — My own claim that a confirmable release would be "an uninteresting near-copy of `J`"

* **Original:** `results/pcrl_manuscript_review_v6/G0_REVIEW.md` §3 (mine): a `.001` confirmation
  "requires a release close enough to `J` that the demonstration is uninteresting".
* **Evidence:** the SE in question is that of paired per-household **sensitive-endpoint** losses for one
  comparison. A release can hold its sensitive predictions nearly equal to the reference — hence a small
  paired SE on those endpoints — while its **task utility** differs substantially. That is not a
  near-copy; it is the utility-first success route this programme registered, and the registration's
  §7 utility route requires exactly that shape.
* **Replacement:** low variance on the sensitive endpoints implies nothing about how interesting the
  release is. **Withdrawn.**
* **Affects:** §IX of the paper, and `G0_REVIEW.md` is annotated rather than rewritten.
* **Execution decision changed:** none.

## N7 — "A confirmable pass requires a candidate that *reduces* measured sensitive recovery"

* **Original:** `AMENDMENT_1.md` §4.4, offered as a prospective requirement for the next stage.
* **Evidence:** the quantity is the measured paired difference in fitted-attacker log loss on each
  sensitive endpoint versus the same-host reference. If the release **contains** `J`, the inclusion fact
  makes the *population* increment non-negative — exactly zero for a constant appended channel — so a
  confirmable strictly-negative bound would certify finite-probe noise rather than protection. If the
  release **replaces** `J`, a genuine reduction is achievable and the requirement is meaningful.
* **Replacement:** scope the requirement to the replacement contract, and state the quantity as the
  measured fitted-attack difference rather than as a privacy quantity. Note also that G5 was never
  implemented: it exists as registration text only.
* **Affects:** §III and §VIII of the paper; Appendix D item 6.
* **Execution decision changed:** none yet — but it does change what the **next** study may claim, which
  is why it was sent before stage Q rather than after.

## N8 — "`J` was already released to these recipients and remains accessible"

* **Original:** `pcrl_stochastic_channel_v1/REGISTRATION.md` §0, which used this to keep `Z_J` inside
  every release and every conditioning view.
* **Evidence:** no deployment record exists. In the source studies "released wire" denotes the
  experimental interface — arrays written to disk and audited — and the manuscript has described `J` as
  **internal** since v3. Terminal 1's own `RELEASE_CONTRACT.md` searched `results/**`, `docs/`,
  `paper-body/`, `*.tex`, docstrings and `git log --all` and found no external disclosure, against
  repeated project statements that the setting is stipulated and the artifacts local; it records the
  status as "unknown-but-undocumented".
* **Replacement:** `J` is a **comparator, not a compulsory component**. A release beside unchanged `H`
  for a recipient who never received `J` is the prospective design (scenario S1); the old-recipient
  threat model (S2) returns only if deployment evidence appears.
* **Affects:** §VIII of the paper, and the framing of what the capacity diagnostic rejects. With `J`
  mandatory, the capacity question was the **append** question, which cannot see a replacement win —
  demonstrated exactly in `checks/replacement_vs_append_fixtures.py` (appending `T` to `J` adds zero
  while replacing `J` with `T` keeps all task utility and removes 0.693 nats of disclosure).
* **Execution decision changed:** **yes, on Terminal 1's side.** The successor study registers scenario
  S1 and does not release `Z_J`. The precursor's registration stands unchanged.

## N9 — Residence-label-free proxy read as evidence about the utility target (carried from v6)

* **Original:** my v6 contribution assessment and, in weaker form, the v6 conclusion.
* **Evidence:** the proxy is the residual of `A_0`, the unprotected channel carrying the largest sensitive
  recovery in the paper (`+0.032` A/sex, `+0.051` A/race over `H`).
* **Replacement:** label-free is not neutral; "the objective itself paid for sensitive information"
  remains live. Already corrected in v6 and carried into §VI-E here.
* **Execution decision changed:** none.
