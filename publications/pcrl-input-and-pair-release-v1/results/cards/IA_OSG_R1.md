# Card IA_OSG_R1: occupation-side SEX-aware field generalization (family IA, revision R1)

Campaign `pcrl_input_and_pair_release_v1`. Written by the coordinator on 2026-10-22 after 03:35Z, after the registered IA reading
(`claims/IA_REG/DECISIONS.json`, 03:12Z) had been read. Committed with its code before its first fit. Governs IA_OSG,
IA_OSG_SBC_LO / IA_OSG_SBC_HI, IA_OSG_SPD_LO / IA_OSG_SPD_HI, IA_OSG_RO_LO / IA_OSG_RO_HI (the occupation-restricted path points,
saved under IA because they are this revision's own controls) and IA_OSG_BLIND.

**Status:** IA's first revision. Its diagnosis uses MODEL_SELECT outcomes, so the reading is adaptive as a whole (PROTOCOL 7); by
RULES 7 it carries the tag `ADAPTIVE_R1`. It is never IA's registered result, which stays IA_REG (NOT_SEPARATED).

**Not the revisions declared in IA_SFG_v1 section 8.** Neither declared revision was run (coordinator log, 02:47Z; MODEL_FIT records
only):
- the declared R1 (slack-weighted ladder): its trigger failed in version 0, where the women's occupation Brier guard keeps 45% of its
  slack against the 50% required;
- the declared R2 (finer splits of the binding step): its trigger holds in both versions, but the binding step (MAR suppression) has
  no predeclared half beyond MAR2, which the menu already contains, so it cannot change the release.

This card is a different revision, diagnosed from the registered reading. It takes IA's first revision slot and is numbered R1 for that
reason.

**Code:** `ipr/ia/osg.py` (`python3 -m ipr.ia.osg fit --version V --card <this card>`). Deployment runs through `ipr.ia.predict`
(payload operation `osg`). The restricted path points deploy through the same function (payload operations `dq` / `ro` with
`outputs = occupation`, through `ipr.paths.predict_path`). The plan is written by `python3 -m ipr.plan revision --bank IPR1_R1`.

**Declarations for the rules:**
- `operation_sex_blind: false`.
- `ro = {kind: GENERIC_ZTILT, operation_release_only: false}`. The RO path is the generic release-only tilt, here restricted to the
  occupation output.
- No joint-design claim (`joint: null`; only the occupation output changes): D5 = NOT_APPLICABLE_NOT_JOINT.

## 1. Observed failures addressed (with their records)

- **F1. The shared ladder is held back by income** (MODEL_FIT; `ia/FIT_V0.json`, `ia/FIT_V1.json`, `selection`).
  - The rung after k* (marital status suppressed, MAR1) fails budget S3 on income components only:
    - income accuracy 1.95 / 1.69 of the budget (estimate + 2 household-clustered SE);
    - income NLL 1.07 / 1.29 of the budget.
  - Every occupation component at that rung stays at or below 0.63 / 0.49 of its budget.
  - The single-step statistics put most of MAR1's task cost on income: dT_inc 0.0048 / 0.0043 against dT_occ 0.0022 / 0.0028 nats.
  - So the registered candidate generalizes almost nothing: MAR2 in version 0; AGE5 + MAR2 in version 1.
- **F2. Leakage moved to the income view** (MODEL_SELECT, registered grid IPR1_MAIN, 8 cells).
  - IA_SFG's income view is more recoverable than the twin's: winner SCORE AUC 0.5600 against 0.5498.
  - The D2 income recipient gate failed at the point against the twin, SBC and RO.
  - The refitted income model on the coarsened record leaks more, not less.
- **F3. The leakage sits in the occupation view** (MODEL_SELECT, IPR1_MAIN).
  - The twin's pair AUC 0.6480 is carried almost wholly by its occupation view (0.6443); the income view is 0.5498.
  - IA_SFG moved the occupation view by 0.0035 only.

## 2. Operation

- **Income recipient:** the twin's income vector and decision, unchanged (CTL_TWIN's income teacher on the raw record).
- **Occupation recipient:** P_occ = f_occ,k(h_k(x)), where:
  - the menu is IA_SFG_v1's (17 steps, unchanged);
  - single-step statistics come from the same MODEL_FIT folds, settings and fixed iteration counts as IA_SFG_v1 (tag
    IPR_IA_ladder_S<v>):
    - the task cost of a step is its occupation term only, dT_s = dT_occ,s (the out-of-fold occupation NLL increase of the step
      alone); the income term is dropped because the income output does not change;
    - dI_s is IA_SFG_v1's: the out-of-fold log-loss increase of an HGB model of SEX on h_s(Z) over one on Z;
  - the greedy ladder rule is `build_ladder` unchanged: the largest incremental dI / dT first, dT floored at 1e-5, each field moving
    only to a coarser level;
  - h_k = the first k ladder steps;
  - the occupation model on h_k(Z) uses the twin's construction for the occupation task only: `nrm.fa.prc.fit_teachers` with tasks
    ('occ',), inner household-CV T*, full refit, and out-of-fold predictions on the twin's folds NRM_FA_xf_S<v>.
- **Release:** P_inc = f_inc(x) (the twin's), P_occ = f_occ,k(h_k(x)); argmax decisions; deterministic. There is no SEX and no
  propensity at runtime. k = 0 is the twin.
- **Consistency check (recorded, never a stop):** the occupation dT and the dI of every step must equal those recorded in
  `ia/FIT_V<v>.json` (the same code, folds and seeds).

## 3. Closest previous implementations and what differs

- **IA_SFG_v1 (this family, registered).** Both recipients see h_k(x), so the ladder must satisfy both tasks at once, and income binds
  at the first informative step (F1). Here only the occupation input is generalized and the ladder is priced in occupation nats.
- **Grouped feature selection** (FAILURE_MAP row 8, Adult). Per-recipient deletion of whole field groups was undone by the linked
  pair. OSG differs in two ways: it merges levels inside fields rather than deleting groups, and it changes one recipient only.
- **The same risk applies.** The income vector still depends on the raw record, so a pair reader can recover through income part of
  what the occupation input loses. IA_SFG_v1's ceiling (no reader can exceed what h(X) carries) does not hold here.

## 4. Usefulness

- **Income:** income utility equals the twin's, bitwise.
- **Occupation:** bounded by S3, with its income components exactly zero.
- **Plug-in estimates** (label-free; Designer A's `roles/designer_a/FEASIBILITY.json`, version 0, the shared ladder's occupation view;
  under OSG the occupation recipient sees the same h, so the same view applies):
  - MAR suppressed: occupation view 0.6429 against 0.6487 at the identity (-0.0059), for 0.0028 occupation nats;
  - MAR suppressed + WK3: 0.6372 (-0.0115), for 0.0073 occupation nats. The plug-in women's occupation Brier change is +0.0032 there,
    which leaves little room inside S3's 0.0045 once 2 SE are added.

## 5. Twin

CTL_TWIN (card CTL_TWIN_v1), unchanged. The income output is the twin's.

## 6. Strength grid and selection (MODEL_FIT only)

- **Grid:** rungs k = 1, 2, ... of the OSG ladder, fitted in order.
- **Selection:** k* = the largest rung such that it and every smaller rung meet budget S3 over the twin. The components are occupation
  NLL 0.012, income NLL 0.006, men's and women's occupation Brier 0.0045, occupation accuracy loss 0.008 and income accuracy loss
  0.006, each as estimate + 2 household-clustered SE, out-of-fold on NRM_FA_xf_S<v>. The income components are exactly zero.
- **Stop:** the scan stops at the first failing rung.
- **Per version:** CAND = the pair of per-version selections (flagged CAND_STRENGTH_DIFFERS_BY_VERSION by the rules if they differ).
- **No fallback:** if rung 1 fails in either version, D1 is NO_FEASIBLE_STRENGTH.

## 7. Matched controls, ablation and descriptive checks

PROTOCOL 7 says a revision's card states its controls. The registered IA bracket points (spend about 0.001 nats) cannot bracket this
revision, and the operation changed. So every gating path is refitted here, each restricted to the output the revision changes:
income passes through bitwise from the twin, as in IA_OSG.

- **SBC (gating):**
  - the same occupation-only ladder rule and occupation models, with SEX replaced by IA_SFG_v1's fixed-seed permutation (seed
    SeedSequence([v, 20261022, 1]));
  - rungs are fitted in order until one reaches CAND's spend + 2 SE;
  - two bracket points, IA_OSG_SBC_LO / IA_OSG_SBC_HI;
  - PSEUDO2 is used only by the registered fallback rule.
- **SPEND (gating):**
  - SEX-blind dithered quantization of the twin's occupation output only (`ctlkit.dq_release`'s occupation coordinates: the same
    basis, step grid DQ_GRID_FINE and dither tag IPR_DQ_S<v>);
  - two bracket points, IA_OSG_SPD_LO / IA_OSG_SPD_HI;
  - it spends where the revision spends, so it is a stricter control than the registered both-output DQ path.
- **RO (gating): GENERIC_ZTILT, IA_SFG_v1's declared fallback kind.**
  - The generic release-only tilt of the registered RO path (`ctlkit.ro_path`, cross-fitted release-only surrogate, RO_GRID) is
    applied to the occupation output only.
  - Two bracket points, IA_OSG_RO_LO / IA_OSG_RO_HI.
  - Scope sentence: the gain is not reproduced by the generic release-only tilt of the occupation output at matched spend; the
    operation driven only by the twin's release was not built.
- **Ablation (descriptive): IA_OSG_BLIND.** The same occupation-only menu and models, with the ladder ordered by occupation task spend
  only (targeting removed), at the first rung whose MODEL_FIT spend reaches CAND's.
- **Descriptive MODEL_FIT checks** (never a stop and never a decision input):
  - household cross-fitted LOG_LR and HGB15 readers on the out-of-fold releases;
  - data-rich readers trained on them (`RICH_READERS_R1_V<v>`).

**Reading.**
- **Bank:** IPR1_R1, written by `ipr.plan revision` after the fits and before any of its grid output. It holds:
  - the shared twin, IA_OSG and its SBC, SPEND and RO bracket points (each present in both versions);
  - IA_OSG_BLIND;
  - the nine references, U0 and Ucal;
  - IA_SFG and the dual-design references, beside.
- **Cells:** the registered cells (R1, R2 x k0, k1, both versions; the IPR1_MAIN seeds).
- **Claims:** IPR1 rules, tag ADAPTIVE_R1.
- **Strongest-release follow-up:** if IA_OSG becomes IA's strongest feasible nonzero release, its refit bank and D2R follow under the
  rules.

## 8. Stop rule

- This is IA's first of at most two revisions.
- A second IA revision needs its own observed failure and card; none is planned.
- If IA_OSG is NO_FEASIBLE_STRENGTH or task-infeasible, the protection comparison is reported as not run, never as negative.
- If the budget cannot hold the control points, D2 is NOT_ASSESSED_CONTROLS_NOT_RUN.

## 9. Registered predictions (before the fit), configurations and CPU

Benefit = comparator AUC - IA_OSG AUC on IPR1_R1's registered cells (pair view unless stated).

| Prediction | Value / probability |
|---|---|
| D1 TASK_FEASIBLE (k* >= 1 in both versions) | 0.95 |
| k* includes MAR suppressed in both versions | 0.75 |
| k* includes WK3 or a COW step in at least one version | 0.35 |
| Income view bitwise the twin's | 1.0 (by construction) |
| Occupation view below the twin's at the point | 0.85 (by 0.003-0.012) |
| vs TWIN | +0.005 (interval -0.003 to +0.013); resolved 0.10 |
| vs the SBC bracket Q* | +0.004 (-0.008 to +0.014); resolved 0.08 |
| vs the SPEND bracket Q* (occupation-only DQ) | +0.002 (-0.010 to +0.012); resolved 0.05 |
| vs the RO bracket Q* (occupation-only generic tilt) | -0.003 (-0.016 to +0.008); resolved positive 0.03 |
| D2 IMPROVEMENT_BEYOND_MATCHED_CONTROLS (GENERIC_ZTILT scope) | 0.03 |
| D2 NOT_SEPARATED | 0.75 |
| D2 PARTIAL_<...>_ONLY | 0.15 |
| D2 NO_BRACKET_<path> flag | 0.05 |
| D3 NOT_SUPERIOR / D4 NOT_MET | 0.97 / 0.99 |
| IA_OSG_BLIND vs IA_OSG within +-0.005 | 0.7 |
| IA_OSG's pair point benefit over IA_SFG's positive (beside) | 0.7 |

- **Configurations:** IA_OSG 2, IA_OSG_SBC_LO / HI up to 4, IA_OSG_SPD_LO / HI up to 4, IA_OSG_RO_LO / HI up to 4, IA_OSG_BLIND 2:
  at most 16. 23 were saved before this card, so the campaign total is at most 39 of 48.
- **CPU:** about 0.3-0.6 CPU-hours for both versions (occupation-only statistics, rungs and scans), plus the reader grid.

## 10. Novelty

NOT_CLAIMED. Recipient-specific attribute generalization is local recoding and attribute generalization applied to one model's inputs
(Samarati and Sweeney 1998; Sweeney 2002; LeFevre et al. 2006); per-model feature restriction is standard. No literature search was
run in this campaign.
