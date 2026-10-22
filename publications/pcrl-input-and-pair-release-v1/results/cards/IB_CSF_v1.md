# Card IB_CSF_v1: coordinated keyed selective flattening (family IB, Designer B, redirected)

Campaign `pcrl_input_and_pair_release_v1`. Written by the coordinator from Designer B's proposal (`roles/designer_b/PROPOSAL.md`,
`mech_sketch.py`) with the challenger's redirect adopted (`roles/challenger/REVIEW_DESIGNS.md` findings 10-16;
`PROTOCOL_AMENDMENT_1.md` items 8-11), 2026-10-22 after 02:36Z. Governs IB_CSF, IB_CSFT, IB_IND and the IB brackets of the shared SPEND
path (card CTL_PATHS_v1). Code: `ipr/ib/csf.py` (`python3 -m ipr.ib.csf fit --version V --card <this card>`), with `ipr/paths.py`,
`ipr/ctlkit.py`, `ipr/ctl/twin.py`; committed with this card before the first fit.

**Declarations for the rules:** `operation_sex_blind: true` (the candidate has no SEX-dependent fitted input; SEX enters only the
MODEL_FIT strength-selection guards): SBC and RO are degenerate by construction and not built, D2 gates on TWIN and the SPEND path, met
label SEX_BLIND_IMPROVEMENT_BEYOND_TWIN_AND_SPEND, always quoted with its scope (no SEX-targeting claim and no claim about SEX
information beyond the twin's release is possible). No generic ZTILT path is built for IB. **Joint design:** declared jointly designed
(both outputs change, coordinated by one shared key), with `collapsed = {income_first: JOINT, occupation_first: JOINT}` (item 7).

## 1. Operation (CSF)

- Twin p_I(x), p_O(x): the shared cached task teachers (CTL_TWIN), unchanged.
- Tables m_I[d], m_O[k]: means of the twin's household out-of-fold vectors over the MODEL_FIT people with income decision d /
  occupation decision k (teacher-mean tables; each mean keeps its decision as argmax because every averaged row has its maximum there).
- Key u(x) = SHA-256('IPR_B_CSF|V<v>|' + the six compact fields' float64 bytes) / 16^13: one draw per six-field record, independent of
  SEX.
- Release: if u(x) < pi, r_I = m_I[argmax p_I(x)] and r_O = m_O[argmax p_O(x)] (the same people in both recipients); otherwise the
  twin's vectors bitwise. Each recipient receives only its own vector and argmax; decisions are kept exactly (both accuracy losses 0);
  deterministic in the six fields; no runtime SEX.

## 2. Closest previous implementations and what differs

CTL_REM (`nrm/ctl/release.py`): the occupation remainder flattened for everyone, top probability kept. DQ (`ipr/ctlkit.py`): SEX-blind
quantization of everyone, independent dithers per recipient. Row 4's bank channels (Adult): one draw per person between the identity
and a class medoid, per recipient. CSF is selective (a keyed fraction, everyone else untouched), collapses completely (the top
probability too), and coordinates the two recipients (the same people). Designer B's targeted form CSF-T is kept as an ablation: the
challenger judged its targeting layer a diluted form of the closed HQ grouping (row 12), worth only +0.0007 to +0.0040 in the
designer's own plug-in while halving the affordable strength.

## 3. Measured failure addressed

For a collapsed person neither recipient carries anything beyond the joint decision cell, so neither can re-supply what the other
lost (failures 2 and 3, for the collapsed fraction only; challenger advisory 15); income accuracy is unchanged (failure 5); the
operation is release-only and SEX-blind by construction (failure 6 cannot explain a gain it does not target). The collapsed people's
10 decision-cell rows are exactly identifiable to a table reader (failure 4 is neither addressed nor worsened).

## 4. Usefulness (label-free plug-in, disclosed, not outcomes)

Designer B's plug-in world: uniform CSF at pi 0.10 / 0.15 spends about 0.005 / 0.008 nats on income and 0.009 / 0.014 on occupation (V0)
for a pair change of about -0.006 / -0.009 from the twin's 0.639 (plug-in scale); the binding S2 component is the men's occupation
Brier guard at every strength (S2 ratio 1.01 at pi 0.075), so under S3 pi* is expected near 0.10-0.15. Accuracy loss is exactly 0.
Known gap of the inherited budgets: no income or overall Brier component (challenger advisory 17).

## 5. Twin

CTL_TWIN (card CTL_TWIN_v1); CSF at pi = 0 is the twin bitwise.

## 6. Strength grid and selection (MODEL_FIT only)

pi in {0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30}; pi* = the largest grid value such that it and every smaller
value meet **budget S3** over the twin (occupation NLL 0.012, income NLL 0.006, men's and women's occupation Brier 0.0045, occupation
accuracy loss 0.008, income accuracy loss 0.006; estimate + 2 household-clustered SE), estimated nested: 5 outer household folds
(IPR_IB_ncv_S<v>), the tables refitted on four folds' out-of-fold twin vectors and applied to the fifth. Per version; if no grid value
qualifies in either version, D1 is NO_FEASIBLE_STRENGTH.

## 7. Matched controls, ablations and the joint-design declaration

- **SPEND (gating):** DQ of the shared twin, two bracket points saved under CTL (card CTL_PATHS_v1), shared with family IA when they
  coincide.
- **SBC and RO:** degenerate by construction (a SEX-blind operation's SBC path and substituted RO path would be the operation itself);
  not built, never "not run".
- **Targeting ablation IB_CSFT** (descriptive): the same coordinated collapse at pi*, with the rates allocated over the 10 joint
  decision cells c = (income decision, occupation decision) in decreasing order of Var_c(z) / mean_c(kappa) until the cost equals
  uniform flattening of a fraction pi*; z = P(SEX | the twin's centred log-probability pair), HGB (15 leaves, 200 iterations, learning
  rate 0.05, minimum leaf 50) cross-fitted on 5 household folds (MODEL_FIT SEX, training only); kappa = KL(p_I || m_I[d]) / 0.004 +
  KL(p_O || m_O[k]) / 0.008.
- **Coordination ablation IB_IND** (descriptive; the separately designed recipients): uniform flattening at pi* with independent
  per-recipient keys (tags '...' and '...|O'), so each recipient is flattened for a fraction pi* of people, but not the same people.
- **D5 declaration (before the first fit, with its construction argument):** a sequential design whose second stage may use the first
  stage's release and the shared six-field key flattens exactly the people the first stage flattened (their first-stage vectors are
  decision-cell rows, identifiable from the release), which reproduces CSF bitwise; so both sequential orders equal the joint design by
  construction and D5 reads NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL (RULES.md 6: no joint-design advantage over sequential design can
  exist). The joint-versus-separate contrast is reported descriptively through IB_IND.
- **Descriptive MODEL_FIT checks** (never a stop): household cross-fitted readers on the nested releases; the data-rich readers.

## 8. Revisions (at most two; each needs an observed failure)

- **R1, split rates on nested keyed subsets.** Trigger: in both versions the binding S3 component at the first failing pi is an income
  component while every occupation component keeps at least 40% slack. Change: occupation flattened at pi and income at rho x pi on
  the same keyed people (u < rho pi; still coordinated), rho in {0, 0.25, 0.5}, chosen by the same rule. Control: SPEND reused if it
  still brackets.
- **R2, shallower collapse.** Trigger: pi* <= 0.025 in both versions. Change: collapsed people keep a 3-level income confidence code and
  a 2-level occupation top-probability code (cell means), which raises the affordable pi. Control: SPEND, and IND at the same depth.

## 9. Registered predictions, configurations and CPU

Benefit = comparator AUC - IB_CSF AUC on the registered grid (pair view).

| Prediction | Value / probability |
|---|---|
| pi* in {0.10, 0.125, 0.15} in both versions; binding component the men's occupation Brier guard | 0.6 |
| D1 TASK_FEASIBLE; accuracy loss exactly 0 | 0.85 / 1 |
| vs TWIN | +0.006 (interval -0.004 to +0.015); resolved 0.10 |
| vs the SPEND bracket Q* | +0.002 (-0.008 to +0.010) |
| D2 SEX_BLIND_IMPROVEMENT_BEYOND_TWIN_AND_SPEND | 0.03 |
| D2 NOT_SEPARATED | 0.75 |
| D2 PARTIAL_TWIN_ONLY | 0.12 |
| PROTECTION_NOT_ASSESSED | 0.02 |
| IB_CSFT vs IB_CSF within +-0.004; IB_IND vs IB_CSF within +-0.004 | 0.8 / 0.8 |
| D3 NOT_SUPERIOR / D4 NOT_MET / D5 NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL | 0.97 / 0.99 / 0.95 |

Configurations: IB_CSF 2, IB_CSFT 2, IB_IND 2 = 6, plus the IB brackets of the shared SPEND path (at most 4, fewer where they coincide
with IA's). CPU: under 10 CPU-minutes per version, plus the reader grid.

## 10. Novelty

NOT_CLAIMED. Local suppression and microaggregation (Willenborg and de Waal 2001; Hundepool et al. 2012); post-randomization (PRAM;
Gouweleeuw et al. 1998) and randomized response (Warner 1965); coordinated (complementary) suppression across linked releases (Cox
1980). No literature search was run in this campaign.
