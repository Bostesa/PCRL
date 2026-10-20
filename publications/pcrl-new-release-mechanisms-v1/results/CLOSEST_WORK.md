# Closest-work comparison

Campaign `pcrl_new_release_mechanisms_v1`, Florida ACS 2019 development cohort, development evidence only. This file compares each
operation tried in this campaign with its closest published work and with the closest earlier PCRL attempts. It is written by the
coordinator from the mechanism cards (`cards/`), the failure map (`FAILURE_MAP.md`) and the agents' saved reports (`roles/`).

**No novelty is claimed for any operation.** No separate prior-art review was run for family B, so its novelty label is `NOT_CLAIMED`
(the rules never infer novelty from performance). Family A's own card names its prior art, so family A was declared `PRIOR_ART_FOUND`
(DECLARATIONS_V1); its twin's claims record shows G7 `NOT_CLAIMED` only because the invalid-grid branch does not read the declared label.
Citations below are limited to works the cards and earlier PCRL campaigns already named, plus standard references for output
perturbation and randomized response.

## Family A: propensity-routed finite cells (FA-PRC; closed at the capacity stage)

**Operation.** Each person is routed to one of a finite set of (income cell, occupation cell) pairs. The routing rule trades the task
cross-entropy against each cell's contribution to the plug-in mutual information between SEX and the cell, using a SEX propensity model
fitted on MODEL_FIT. The released vectors are label-refit cell decoders, so they identify the cell and nothing else.

**Closest published work.**
- The privacy funnel (Makhdoumi, Salamatian, Fawaz and Médard, 2014) and the information bottleneck with side information, IBSI
  (Chechik and Tishby, 2002), with deterministic information-bottleneck Lloyd updates (Strouse and Schwab, 2017). FA-PRC is a
  deterministic privacy-funnel quantizer of a task-only teacher.
- FARE (Jovanović et al., 2023), the published method that routes inputs to finite cells (tree leaves) with a certificate on what a
  downstream classifier can recover.

**Closest earlier PCRL attempts.** On this cohort, the shared-cell merging line (structural-mechanism pilot, SCM engineering benchmark,
SCM utility match, 2026-10-17; `FAILURE_MAP.md` row 5). On Adult, the finite codes of the decision-preserving compression line
(2026-10-06; row 7), which paid about 0.02 nats of occupation NLL.

**What differed.** Routing used the person's SEX propensity at release time (the cell choice is SEX-aware through π(x), never through the
recorded SEX), and the pair, income and occupation dependence terms were optimized together, with a guard term for men's and women's
occupation Brier.

**Outcome.** The registered capacity gate failed before any protected fit. It required the twin's nested held-out occupation NLL spend
over the continuous task teachers to satisfy estimate + 2 SE ≤ 0.004 nats (and accuracy loss ≤ 0.003); the measured spend was +0.0107 /
+0.0136 nats at 16 × 64 cells and +0.0137 / +0.0132 at 24 × 128 (`roles/coordinator/FA_CAPACITY_DECISION.md`,
`roles/designer_a/CLOSING_REPORT.md`). This campaign's own label-free capacity curve puts the occupation quantization cost on this
cohort at about 0.010 nats at 64 cells, falling to about 0.003 at 512 cells; the earlier finite codes on Adult paid about 0.02 nats.
Family A is closed as `FINITE_CELL_CAPACITY_NOT_ESTABLISHED`; the propensity-routing question itself was never tested.

## Family B, revision FB2: SEX-targeted rank-1 Gaussian noise on a frozen task-only base

**Operation.** The frozen base (HC_STRONG income, A1_PO_16 occupation; no refit) emits centred log-probabilities z(x). The release
divides them by a temperature fitted on CALIBRATION, adds ONE standard-normal scalar per person and release instance along a
person-specific direction v(x) in the joint space (z/T + c·ε·v), and emits the softmax of each task's block. For FB2_SEX, v(x) is the Fisher-whitened
SEX class-mean-difference direction (estimated on MODEL_FIT), regularized by the person's own task Fisher information; for the matched
control FB2_MINF, v(x) is the person's least task-informative direction, which uses no SEX information. Both cost about s²/2 nats per
person at second order. Each recipient receives only its own vector and argmax decision.

**Closest published work.**
- Output perturbation: noise calibrated to sensitivity (Dwork, McSherry, Nissim and Smith, 2006), the Gaussian mechanism (Dwork,
  Kenthapadi, McSherry, Mironov and Naor, 2006; Dwork and Roth, 2014), and randomized response (Warner, 1965). FB2 adds noise to released outputs, but it is not calibrated to any privacy budget and makes no
  differential-privacy claim.
- Privacy against statistical inference (du Pin Calmon and Fawaz, 2012) and the privacy funnel, where the optimal mapping of the
  released data is in general randomized and targeted at the private variable.
- Linear concept erasure (LEACE; Belrose et al., 2023) removes the class-mean difference deterministically. FB2 instead randomizes along
  a Fisher-whitened class-mean-difference direction and keeps the task information in the other directions.

**Closest earlier PCRL attempts.** The ACS 2018 stochastic-channel and replacement-channel studies (closed before any stochastic
constraint ran; the replacement coder cost +0.021 nats), the adaptive-release study (no stochastic advantage), the conditional budgeted
release (LP mixtures toward donors, pair reduction 0.0020, "no randomness advantage"), the frozen code-bank channels (identity action
alone over the occupation NLL cap), and this campaign's own card v1 (the same channel trained through; its task-only learner failed the
original limits).

**What differed.** Continuous noise on the frozen base's own log-probabilities: no code bank, no replacement coder and no quantization, so
the identity action is the base itself, which has measured slack under the original limits. The comparison with FB2_MINF holds the noise
law and second-order cost fixed and changes only whether the direction uses SEX, so it isolates targeting.

**Novelty.** `NOT_CLAIMED`. Every FB2 result carries the label "FB2 was designed after family-B screening on MODEL_SELECT (targeting
signal, F3); it is evaluated on the same people; development evidence only." and the scope "one released instance per person;
repeated-query protection not claimed".

**Outcome.** As in the earlier PCRL randomized and mixture studies ("no randomness advantage"), the channel lowered pair recovery
against its own base by about the amount a SEX-blind channel of the same form and cost did: FB2_SEX_S5 0.6486 against FB2_MINF_S5
0.6505 (Monte Carlo pooled pair AUC; benefit +0.0019, se 0.0089), from a base at 0.6605. Its falsifier F1 holds and the family closed (`TABLES.md`, `claims/FINAL_V2/`).
The targeted direction did cut in-sample linear SEX discriminability much more than the SEX-blind one (Designer B's fit-time records),
which the declared nonlinear readers did not reflect, in line with the earlier PCRL finding that linear erasure leaves nonlinear
recovery unchanged.

## The challenger's control: CTL_REM (SEX-blind remainder flattening)

**Operation.** On FB2_BASE's occupation vector, keep the argmax decision and the top probability exactly and move the remaining
probability mass toward fixed SEX-blind class shares fitted on CALIBRATION occupation labels. The strength is chosen by total-NLL spend
only. Income is FB2_BASE's, unchanged.

**Closest published work.** Output-coarsening defenses against inference from prediction vectors, such as releasing only the top class or
the top few classes, or rounding confidence values (discussed as mitigations by Shokri, Stronati, Song and Shmatikov, 2017).

**Role.** A SEX-blind, deterministic explanation of any FB2 benefit at matched spend (`cards/CTL_CARD_v2.md`). It makes no protection
claim and is never a candidate.

**Outcome (measured for the declared readers).** Partial flattening (CTL_REM_0p50) changed pair recovery by about 0.001 (0.6596 against
the base's 0.6605). Replacing the remainder by fixed shares (CTL_REM_1p00, decision and top probability only) lowered the occupation
view from 0.6428 to 0.5603 and the pair from 0.6605 to 0.6457 (resolved), but failed the occupation NLL clauses (1.8-2.2 times the cap)
and the men's guard (2.7-3.0 times the cap); its occupation Brier clauses passed. FB2 was not separated from this path at either
strength.

## References

- Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E. and Biderman, S. (2023). LEACE: Perfect linear concept
  erasure in closed form. NeurIPS.
- Chechik, G. and Tishby, N. (2002). Extracting relevant structures with side information. NIPS.
- du Pin Calmon, F. and Fawaz, N. (2012). Privacy against statistical inference. Allerton Conference.
- Dwork, C., Kenthapadi, K., McSherry, F., Mironov, I. and Naor, M. (2006). Our data, ourselves: privacy via distributed noise
  generation. EUROCRYPT.
- Dwork, C., McSherry, F., Nissim, K. and Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. TCC.
- Dwork, C. and Roth, A. (2014). The algorithmic foundations of differential privacy. Foundations and Trends in Theoretical Computer
  Science.
- Jovanović, N., Balunović, M., Dimitrov, D. I. and Vechev, M. (2023). FARE: Provably fair representation learning with practical
  certificates. ICML.
- Makhdoumi, A., Salamatian, S., Fawaz, N. and Médard, M. (2014). From the information bottleneck to the privacy funnel. IEEE
  Information Theory Workshop.
- Shokri, R., Stronati, M., Song, C. and Shmatikov, V. (2017). Membership inference attacks against machine learning models. IEEE
  Symposium on Security and Privacy.
- Strouse, D. J. and Schwab, D. J. (2017). The deterministic information bottleneck. Neural Computation.
- Warner, S. L. (1965). Randomized response: a survey technique for eliminating evasive answer bias. Journal of the American
  Statistical Association.
