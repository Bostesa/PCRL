# Card IA_SFG_v1: shared SEX-aware field generalization (family IA, Designer A)

Campaign `pcrl_input_and_pair_release_v1`. Written by the coordinator from Designer A's proposal (`roles/designer_a/PROPOSAL.md`,
`mech_sketch.py`) with the challenger's required changes adopted (`roles/challenger/REVIEW_DESIGNS.md` findings 3-9, 16-18;
`PROTOCOL_AMENDMENT_1.md` items 1-7, 10-11), 2026-10-22 after 02:36Z. Governs IA_SFG, IA_SBC_LO / IA_SBC_HI, IA_BLIND and the IA
brackets of the shared paths (card CTL_PATHS_v1). Code: `ipr/ia/sfg.py` (`python3 -m ipr.ia.sfg fit --version V --card <this card>`),
with `ipr/paths.py`, `ipr/ctlkit.py`, `ipr/ctl/twin.py`; committed with this card before the first fit.

**Declarations for the rules:** `operation_sex_blind: false`; `ro = {kind: GENERIC_ZTILT, operation_release_only: false}` (the
declared fallback, item 5 below); no joint-design claim (`joint: null`; both outputs change but the family is not declared jointly
designed: D5 = NOT_APPLICABLE_NOT_JOINT).

## 1. Operation

- **Menu** (fixed, label-free, maps of the compact coding Z = `ohn.fam_a.compact_po(X)`; COW and MAR are admitted block indices = code
  - 1, SCHL the SCHL code, POBP the numeric place-of-birth code, AGEP / WKHP standardized with their admitted levels; hours recovered
  from WKHP with the modal value = 40 h and the minimum = 1 h), a chain of coarser levels per field:

| Field | Levels (moderate -> intermediate -> strong) |
|---|---|
| AGEP | AGE5 (5-year bands, 8) -> AGE10 (10-year bands, 4) |
| WKHP | WK6 (1-19, 20-34, 35-39, 40, 41-49, 50+) -> WK5 (<35, 35-39, 40, 41-49, 50+) -> WK3 (<35, 35-44, 45+) |
| COW | COW5 (for-profit, nonprofit, government, self-employed, unpaid) -> COW4 (private, government, self-employed, unpaid) -> COW3 (private incl. unpaid, government, self-employed) |
| SCHL | SCH5 (<=15, 16-17, 18-20, 21, 22-24) -> SCH4 (<=17, 18-20, 21, >=22) -> SCH3 (<=17, 18-20, >=21) |
| MAR | MAR3 (married, previously married, never married) -> MAR2 (married, not married) -> MAR1 (suppressed) |
| POBP | POB4 (Florida, other US state, US territories, foreign or other) -> POB2 (Florida, elsewhere) -> POB1 (suppressed) |

- **Ladder** (MODEL_FIT; task labels and SEX as a training label): single-step statistics on 5 household folds (tag
  IPR_IA_ladder_S<v>, HGB A1_PO_16 settings at the cached twin's iteration counts T_inc / T_occ, the SEX model at the cached SEX-side
  T): dT_s = out-of-fold income + occupation NLL increase of step s alone over the identity; dI_s = out-of-fold log-loss increase of an
  HGB model of SEX on h_s(Z) over one on Z. Greedy order (`build_ladder`): at each step the move with the largest incremental dI / dT
  among each field's next coarser levels (a field keeps its last level). h_k = the first k ladder steps.
- **Task models** on h_k(Z) by the twin's construction (`fit_rung` = `nrm.fa.prc.fit_teachers`: inner household-CV T*, tag
  OHN_A_cv_S<v>, max 1000 iterations; full refit on all 20,000 MODEL_FIT people; out-of-fold on NRM_FA_xf_S<v>, the folds of the cached
  twin's out-of-fold predictions).
- **Release:** P_inc = f_inc(h_k(x)), P_occ = f_occ(h_k(x)); argmax decisions; deterministic; no SEX and no propensity at runtime;
  k = 0 is the twin. Both recipients see the same generalized record.

## 2. Closest previous implementations and what differs

Grouped feature selection (FAILURE_MAP row 8, Adult): per-recipient deletion of whole field groups, undone by the linked pair. SCM
(row 5): post-hoc merges of posterior cells. FRC ORD (row 12): SEX-blind coarsening of the twin's outputs. SFG merges levels inside the
raw fields on ACS FL, with the same generalized record for both recipients, refits the task models and releases continuous vectors;
SEX only orders a fixed menu. It is in FAILURE_MAP's "untested operations" (input-side generalization) and is no tilt,
standardization, reweighting, posterior-cell routing, channel, decoder, controller or leaf correction. Both families of this campaign
remove information by merging (challenger finding 1).

## 3. Measured failure addressed

Failures (2) super-additive pair and (3) leakage moving between recipients: both outputs are functions of h(x), so no reader of either
view or of the pair can exceed what h(X) carries about SEX. This ceiling has no bite at feasible rungs (the designer's plug-in ceiling
at k = 3, 0.656, is above the twin's realized pair 0.6506; challenger finding 9c). It does not credibly address (1) or (6), and it meets
(5): income pays most of the spend.

## 4. Usefulness (label-free feasibility, disclosed, not outcomes)

Designer A's plug-in (cached posteriors and propensity as the truth): MAR suppressed spends about 0.0085 nats (income 0.0056) for a
pair change of -0.005; MAR suppressed + three hours bands about 0.018 (income 0.011) for -0.013; with COW3 too, 0.035 for -0.024,
task-infeasible. The dual-design twin's published MODEL_SELECT headroom vs U0 (limit minus excess): income NLL 0.0186 / 0.0181, income
accuracy 0.0143 / 0.0110, men's occupation Brier guard 0.0089 / 0.0073, women's 0.0147 / 0.0137 (the twin is 0.0086 / 0.0081 nats better
than U0 on income NLL and 0.0161 / 0.0128 on occupation NLL). Under S3 the income clauses bind early: a near-trivial first rung (one
moderate step, plug-in pair change about -0.0003) is a likely CAND (challenger finding 4). Known gap of the inherited budgets: no
income or overall Brier component (challenger advisory 17).

## 5. Twin

CTL_TWIN (card CTL_TWIN_v1): the cached HGB task teachers, the same learner, settings, inner-CV T* rule, full refit and out-of-fold folds
as `fit_rung` on the raw inputs (k = 0). Equally trained (PROTOCOL 4, advisory 13 satisfied: the same re-selection on raw inputs is the
cached teachers' own construction).

## 6. Strength grid and selection (MODEL_FIT only)

Rungs k = 1, 2, ... of the SEX-ordered ladder, fitted in order; k* = the largest rung such that it and every smaller rung meet
**budget S3** over the twin (occupation NLL 0.012, income NLL 0.006, men's and women's occupation Brier 0.0045, occupation accuracy
loss 0.008, income accuracy loss 0.006; estimate + 2 household-clustered SE; out-of-fold on NRM_FA_xf_S<v>). The scan stops at the
first failing rung. Per version; CAND = the pair of per-version selections (flagged CAND_STRENGTH_DIFFERS_BY_VERSION by the rules if
they differ). If rung 1 fails in either version, D1 is NO_FEASIBLE_STRENGTH (no fallback).

## 7. Matched controls, ablation and descriptive checks

- **SBC (gating):** the same menu, ladder rule and `fit_rung` models with SEX replaced by a fixed-seed permutation of the MODEL_FIT SEX
  column (seed SeedSequence([v, 20261022, 1])); its rungs are fitted in order until one reaches CAND's spend + 2 SE; two bracket points
  (IA_SBC_LO / IA_SBC_HI). PSEUDO2 (`ipr.ctlkit`) only by the registered fallback rule (the permuted path ends without reaching CAND's
  spend + 2 SE). Expected (challenger finding 6): the permuted order is noise-driven, close to BLIND, and differs by version; it tests
  only "a real SEX signal beats noise" as the ordering.
- **SPEND (gating):** DQ of the shared twin, two bracket points saved under CTL (card CTL_PATHS_v1).
- **RO (gating): GENERIC_ZTILT, the declared fallback.** The registered default SUBSTITUTED path (SFG's ladder with SEX replaced by a
  cross-fitted release-only surrogate) is not built as a configuration, because the twin's released pair identifies the six-field
  record for about 93% of MODEL_FIT people (challenger `CHECK_SUPPORT.json`), so the surrogate is the record-level propensity up to its
  learner and the substituted ladder is SFG's own up to estimation noise. It is computed inside the fit unit on MODEL_FIT (soft target
  = the cross-fitted surrogate `ctlkit.ro_teacher`) and its equality with SFG's ladder is recorded (`substituted_ladder`), never
  released. The generic tilt's two bracket points are saved under CTL. D2's scope sentence for this declaration: the gain is not
  reproduced by the generic release-only tilt at matched spend; the operation driven only by the twin's release was not built.
- **Ablation (descriptive): BLIND**, the same menu and models with the ladder ordered by task spend only (the targeting component
  removed), at the first rung whose MODEL_FIT spend reaches CAND's (IA_BLIND).
- **Descriptive MODEL_FIT checks** (never a stop and never a decision input): household cross-fitted LOG_LR and HGB15 readers on the
  out-of-fold releases; the data-rich readers (`ipr.richreaders`) trained on them.

## 8. Revisions (at most two; each needs an observed failure)

- **R1, slack-weighted ladder.** Trigger: on MODEL_FIT, an income clause is the only failing clause at rung k* + 1 while every
  occupation component keeps at least 50% of its S3 slack. Change: dT = dNLL_inc / 0.006 + dNLL_occ / 0.012 in the ladder ratio; same
  menu and rule. Controls: a new SBC path (permuted target, same ratio); SPEND and RO reused if they still bracket.
- **R2, finer splits of the binding step.** Trigger: on MODEL_FIT, the rung after k* overshoots S3 by more than 50% on its binding
  component. Change: the binding step replaced by its two predeclared halves (WK3: merge only <35, or only 45+; COW3: merge only
  government, or only self-employed; MAR1: merge never with previously married first). Same rule and controls.
- Stop after the two revisions or when the budget forbids their controls (PROTOCOL 7).

## 9. Registered predictions, configurations and CPU

Benefit = comparator AUC - IA_SFG AUC on the registered grid (pair view unless stated).

| Prediction | Value / probability |
|---|---|
| D1 TASK_FEASIBLE (k* >= 1 in both versions) | 0.80; NO_FEASIBLE_STRENGTH 0.15 |
| k* = 1 in both versions | 0.5 |
| vs TWIN | +0.003 (interval -0.005 to +0.012); resolved 0.05 |
| vs the SBC bracket Q* | +0.002 (-0.010 to +0.012) |
| vs the SPEND bracket Q* | +0.002 (-0.008 to +0.012) |
| vs the RO bracket Q* | -0.005 (-0.020 to +0.008); RECIPIENT_GATE_FAILED_VS_RO 0.3 |
| D2 IMPROVEMENT_BEYOND_MATCHED_CONTROLS (GENERIC_ZTILT scope) | 0.02 |
| D2 NOT_SEPARATED | 0.60 |
| D2 PARTIAL_<...>_ONLY with RO failing | 0.20 |
| D2 NO_BRACKET_<path> flag | 0.05 |
| PROTECTION_NOT_ASSESSED | 0.02 |
| D3 NOT_SUPERIOR / D4 NOT_MET | 0.97 / 0.99 |
| substituted ladder equal to SFG's ladder up to rung k* (descriptive) | 0.8 |
| BLIND ablation vs IA_SFG within +-0.005 | 0.8 |

Configurations: IA_SFG 2, IA_SBC_LO / HI 4, IA_BLIND 2 = 8, plus the IA brackets of the shared SPEND and RO paths (at most 8, fewer
where they coincide with IB's). CPU: about 1-2 CPU-hours for both versions (ladder statistics 17 steps x 3 models x 5 folds; each rung's
T* inner CV and folds; the SBC and BLIND scans), plus the reader grid.

## 10. Novelty

NOT_CLAIMED. Field generalization and suppression (Samarati and Sweeney 1998; Sweeney 2002; Mondrian, LeFevre et al. 2006); the
privacy funnel (Makhdoumi et al. 2014) and the agglomerative information bottleneck (Slonim and Tishby 1999); optimized pre-processing
(Calmon et al. 2017). No literature search was run in this campaign.
