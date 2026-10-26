# Start here: reader-fit-aware release (campaign `pcrl_reader_fit_aware_release_v1`, RFA1)

A bounded fitted investigation, run on 2026-10-25 from 20:58:41Z, on the Florida ACS 2019 development cohort (admitted public-use
PUMS records). This is statistical privacy research on an existing census benchmark; SEX is the dataset's recorded demographic
variable whose recoverability is measured.

**What the service does.** It predicts an income class (2) and an occupation group (5) from six fields: AGEP, WKHP, COW, SCHL, MAR
and POBP. Each recipient receives only its own probability vector and argmax decision; the linked-pair recipient has both outputs of
the same person and the labelled auxiliary sample. These grants are assumptions of the benchmark, and the adversary is not claimed to
be new. Usefulness is judged by the unchanged 28 conditions.

**The question.** Does training the release against how an auxiliary-data reader **learns and generalizes** transfer better to
newly fitted recipients than training against a critic's current responses? The mechanism tested is the derivative of a held-out
reader objective through the reader's fit:
- **Start.** NEURAL_RAW; only its occupation network is trained.
- **Episodes.** Household-separated rows within MODEL_FIT: about 4,096 for auxiliary training and 2,400 for auxiliary evaluation.
- **Readers.** Exact ridge-logistic readers (linear and random Fourier features; R1 adds soft-binned table-like features).
- **Surrogate.** A symmetric held-out binormal-AUC surrogate.
- **The ingredient.** The implicit-function derivative through the fit.

The controls are:
- the unchanged start;
- a zero-privacy continuation;
- the same procedure with the fitted reader detached from its fitting path (STOPFIT, the isolating control);
- an ordinary lagging alternating critic of the same family;
- NEURAL_RAW's own recipe continued at comparable task cost;
- the frozen strong references.

Every number below is development evidence on adaptively reused MODEL_SELECT people, for the declared reader procedure. A lower
reader AUC is not an arbitrary-attacker privacy guarantee.

**Result: a bounded negative experiment** (`RESEARCH_DECISION.md`; governing records `claims/RFA_v1_final/` and
`claims/RFA_R1_final/`).
- **The mechanism.** F0 found the fit-path term active on MODEL_FIT (orthogonal share 0.10-0.11, a turn of about 6 degrees).
- **Usefulness.** The candidate kept all 28 usefulness conditions at the point.
- **Against the start.** On the registered cells it lowered linked-pair recovery by the declared readers below NEURAL_RAW (+0.0127,
  se 0.0059). The registered benefit over NEURAL_RAW was not confirmed on the second split (+0.0120, se 0.0075).
- **Against the controls.** It was not separated from the stopped-fit control (+0.0028, se 0.0045), the alternating critic (+0.0005,
  0.0010), continued task training or NEURAL_RAW's own recipe. On the second split the point edges over STOPFIT and ALTCRIT changed
  sign, and the benefit over NEURAL_RAW was not confirmed.
- **The revision.** The one revision added a table-like inner reader family aimed at the winning tree, table and categorical readers.
  It was not separated from its controls either, and its pair recovery was not lower than v1's (FC_R1 -0.0029, se 0.0046).
- **Not competitive.** The strong references keep an occupation output about 0.03 AUC less recoverable, and the 0.020 material target
  was not met.

## Plain answers

1. **What changed from the previous adversarial training?** The release was trained against readers fitted afresh on each episode's
   auxiliary-training households and scored on separate auxiliary-evaluation households. The derivative passes through the readers'
   weight fit. By contrast:
   - NEURAL_RAW's online critics lag, and they pass only a direct derivative of an in-batch risk.
   - SAR1's closed-form Fisher penalty is an in-sample best response. By the envelope theorem its gradient equals a stopped-fit
     gradient (checked to 1.4e-14).
2. **Did the derivative-through-fit operation actually affect the release?** Yes, measurably but slightly.
   - **F0.** The fit-path term was active on MODEL_FIT in both versions and both readings: orthogonal share 0.10-0.11, cos(full,
     stopped) 0.994-0.996. The share never fell below 0.05 before k\*.
   - **Outputs.** The released occupation outputs differ from STOPFIT's, with no exact alias in any grid. The occupation-view version
     means are 0.5971 / 0.5969 against RF_STOP_HI's 0.5986 / 0.5989.
   - **Spend (descriptive, post hoc).** The STOPFIT and ALTCRIT comparison points are saved at the candidate's own checkpoint. There
     they spent 0.00018-0.00057 nats more MODEL_FIT out-of-fold total NLL than the candidate, each 3.1-6.1 paired SE above zero. On
     MODEL_SELECT the difference was 0.00034-0.00041. This is a very small, resolved cost difference at equal step count, not a
     recovery difference. An earlier draft wrongly called it unresolved.
   - **Version 0.** In v1, version 0 barely moved: k\* = S20, with a spend of 0.0008 nats.
   - **Recovery.** The change did not produce a separated recovery difference (answer 4).
3. **Did both tasks and subgroup conditions survive?** Yes, at the point: 28 of 28 in both versions, in both readings.
   - **Not confidence-resolved.**
     - v1: version-1 occupation NLL vs Ucal (0.00737 +- 0.00133 of 0.010) and the version-1 men's guard (0.00442 +- 0.00074 of 0.005).
     - R1: the version-1 men's guard (0.00457 of 0.005).
   - **The tightest condition** at the selected points is the version-1 men's occupation guard: 0.885 of its cap in v1, 0.915 in R1.
   - **Not reserve-qualified.**
4. **Did it beat the stop-gradient and ordinary-critic controls at comparable usefulness?** No. It was not separated from either.
   Both controls are G1-feasible and are compared at matched MODEL_SELECT total-NLL spend at their bracket points.

   | Reading | vs STOPFIT | vs ALTCRIT |
   |---|---|---|
   | v1 | +0.0028 (0.0045) | +0.0005 (0.0010) |
   | R1 | -0.0014 (0.0041) | -0.0030 (0.0045) |

   On this grid, a v1 gain beyond STOPFIT larger than about 0.012 is not supported, and beyond ALTCRIT about 0.0025.
5. **Did that result survive fresh nonlinear readers and the second split?**
   - **Fresh readers.** The evaluation readers were fitted afresh for every member and cell, and no inner reader was reused. On the
     registered cells:
     - v1's pair winners were categorical-interaction, extra-trees, MLP and ignore-one readers (CATI1, ET25, MLP64,
       IGNORE_income__ET25);
     - R1's were extra-trees, MLP and categorical-interaction readers (ET25, MLP64, OUT_MLP, CATI1).

     Ignore-one readers also won some second-split cells in both readings.
   - **Second split, v1 (ROBUSTNESS_NOT_CONFIRMED).**
     - The point edges changed sign over STOPFIT (+0.0028 to -0.0009) and ALTCRIT (+0.0005 to -0.0023), and also over SUCC (+0.0037
       to -0.0036).
     - The benefit over NEURAL_RAW was not confirmed (+0.0120, se 0.0075).
     - A single-view recipient condition failed against RF_RAWJ_HI (income), H_LJ (occupation), RETIRED_B3CTL (both views) and SUCC
       (occupation).
   - **Second split, R1 (ROBUSTNESS_NOT_CONFIRMED).**
     - Sign changes over RF1_ALTC_HI (-0.0030 to +0.0016) and SUCC.
     - The benefit over NEURAL_RAW was not confirmed (+0.0124, 0.0070).
     - Recipient failures against RF1_RAWJ_HI, H_LJ, RETIRED_B3CTL and SUCC.
   - **Coverage.** B2TWIN, TWJ, W, TR100 and TT100 were not in the robustness bank.
   - **Learning curve.** The point rule flags a possible learnability effect in both readings, none resolved: in v1 against RF_TASK,
     RF_ALTC_HI and H_LJ; in R1 against H_LJ only.
6. **Did it beat the strong references under the unchanged conditions?** No (D3 NOT_SUPERIOR; D4 NOT_MET).
   - **Resolved below, with both recipient gates:** NEURAL_RAW (v1 only), W, TR100 and TT100.
   - **TWJ:** resolved below, but the occupation-view recipient gate fails.
   - **Not resolved:** B2TWIN, H_LJ and SUCC (point only, with failing occupation gates) and RETIRED_B3CTL (not positive).
7. **Is the result a mechanism component, a competitive candidate or a bounded negative experiment?** A bounded negative experiment,
   in the registered reading and in the adaptive revision.
8. **What specific evidence supports the next step?**
   - **No evidence that the fit path helps protection.**
     - The candidate was not separated from the stopped-fit control or from a lagging critic of the same family. The v1 development
       bounds on this grid are about 0.012 and 0.0025; R1's upper bounds are +0.0068 and +0.0060.
     - Its systematic contribution was small, even with the inner reader budget at the evaluation's own FIT size.
     - The one resolved difference is descriptive and post hoc: a task-cost saving of 0.0002-0.0006 nats at equal step count (answer
       2), with no separated recovery difference.

     Further learning-aware penalties or critic schedules on this start are not supported by this evidence.
   - **The gap is in the occupation output.** In the tree-based references that output is about 0.03 AUC less recoverable, at their
     own men's-guard cost (0.883 / 0.788 of cap).
   - **Usefulness binds early.**
     - On MODEL_FIT the budget stopped k\* at S20 / S70 (v1) and S30 / S60 (R1).
     - The failing components were the women's occupation Brier (v1 version 0, R1 version 0), the occupation NLL with the women's
       Brier (v1 version 1) and the occupation NLL (R1 version 1).
     - On MODEL_SELECT the tightest condition at the selected points is the version-1 men's guard (0.885 / 0.915 of cap).

   **Interpretation, not a tested claim:** a successor would need an occupation output whose recoverability approaches the tree-based
   references' within the guard budget. On this evidence that looks like a question about the output representation rather than
   about the critic.

## Read next

| File | What it holds |
|---|---|
| `RESEARCH_DECISION.md` | the decision, the five outcomes, all registered expectations against outcomes, what is not claimed |
| `ENDPOINTS.json`, `TABLES.md` | the endpoint table of both readings: utilities, reader-grid means, comparisons, second splits, FC_R1 |
| `MEASUREMENT_CONTRACT.md` | the measurement contract and the reviewer's adopted corrections |
| `PROTOCOL.md` | the registered design |
| `cards/` | the cards RF_RFIT_v1 and RF_RFIT_R1 |
| `roles/statistical_reviewer/` | the rules (RULES.md), the addendum and the reviews |
| `PROTOCOL_DEVIATIONS.md` | the deviations (append-only) |
| `claims/` | plans and decision records (`RFA_v1_final` and `RFA_R1_final` govern; earlier records kept); D2R; falsifier detail; FC_R1; `CAMPAIGN_SUMMARY.json`; `PER_RECIPE_PAIR_MEANS_corrected.json` (the first record is kept) |
| `roles/` | the investigator's design and derivative checks, the challenger's equivalence audit, the verifier's checks |
| `QUICKSTART.md` | commands: read results, run selftests, use a deployment package, re-check custody |
| `HANDOFF.md` | state, resources, data access, custody and its gaps |
