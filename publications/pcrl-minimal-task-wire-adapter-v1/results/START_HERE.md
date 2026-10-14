# START HERE: minimal task-wire adapter (pcrl_minimal_task_wire_adapter_v1)

**Labels.** Completion: **COMPLETE**. Independent verification: 43/43 PASS. The off-device backup is pending. Scientific: **POINT_FEASIBLE_NO_COMPETITIVE_CRITERION**, independently recomputed. The independent re-solve confirmed the nominee's privacy-driven updates in both seeds.

**Caveats.** This is exploratory development on repeatedly used, exposed Adult people. Intervals are conditional on the fitted models and readers, and both model seeds reuse the same people.

**Plain answer.** We added a small learned correction (325 parameters) to the occupation head's predictions over a frozen trunk. It kept the original utility at the point level on FIT2 and INNER in both seeds, and its privacy term changed what the occupation recipient receives. Freshly trained readers recovered SEX from the occupation wire alone about 0.010 AUC less than from RAW_PROJ, and that reduction is established. The local-objective control achieved the same reduction. But recovery from the aligned income + occupation pair fell only 0.005, an established shortfall against the required 0.020. Trust constraints and coalition coupling made no measurable recovery difference. The previous guards did lower fresh recovery (pair 0.012–0.015, established), but they fail the original utility caps.

Read next: RESEARCH_DECISION.md (answers and decision), PRIMARY_ENDPOINTS.csv, DIAGNOSTIC_ENDPOINTS.csv, RECIPE_AND_CONTROLS.md, SOURCE_AND_EXPOSURE.md, QUICKSTART.md, figures/.

## All services (INNER, per seed; complete-view AUC = max over 18 fresh readers, pair includes ignore aliases)
| service | seed | status | epoch | eff./nominal occ. disclosure | income acc / NLL / Brier | occ. acc / NLL / Brier | r1 AUC | r2 AUC | pair AUC | INNER point clauses (per seed) |
|---|---|---|---|---|---|---|---|---|---|---|
| J_TRUST | 0 | selected point eligible e10 | 10 | rank 5 / 29 | 0.8492 / 0.3365 / 0.2069 | 0.4886 / 1.2739 / 0.6515 | 0.6861 | 0.8192 | 0.8468 | 12/12 |
| J_TRUST | 1 | selected point eligible e10 | 10 | rank 5 / 29 | 0.8483 / 0.3427 / 0.2087 | 0.4949 / 1.2672 / 0.6479 | 0.6928 | 0.8175 | 0.8357 | 12/12 |
| J_FREE | 0 | selected point eligible e10 | 10 | rank 5 / 29 | 0.8492 / 0.3365 / 0.2069 | 0.4846 / 1.2754 / 0.6522 | 0.6861 | 0.8168 | 0.8477 | 12/12 |
| J_FREE | 1 | selected point eligible e10 | 10 | rank 5 / 29 | 0.8483 / 0.3427 / 0.2087 | 0.4940 / 1.2683 / 0.6480 | 0.6928 | 0.8156 | 0.8393 | 12/12 |
| L_TRUST | 0 | selected point eligible e10 | 10 | rank 5 / 29 | 0.8492 / 0.3365 / 0.2069 | 0.4872 / 1.2745 / 0.6518 | 0.6861 | 0.8204 | 0.8449 | 12/12 |
| L_TRUST | 1 | selected point eligible e10 | 10 | rank 5 / 29 | 0.8483 / 0.3427 / 0.2087 | 0.4931 / 1.2679 / 0.6481 | 0.6928 | 0.8162 | 0.8372 | 12/12 |
| L_FREE | 0 | selected point eligible e5 | 5 | rank 5 / 29 | 0.8492 / 0.3365 / 0.2069 | 0.4872 / 1.2770 / 0.6533 | 0.6861 | 0.8258 | 0.8497 | 12/12 |
| L_FREE | 1 | selected point eligible e5 | 5 | rank 5 / 29 | 0.8483 / 0.3427 / 0.2087 | 0.4949 / 1.2668 / 0.6477 | 0.6928 | 0.8216 | 0.8392 | 12/12 |
| TASK_TRUST | 0 | selected point eligible e10 | 10 | rank 5 / 29 | 0.8492 / 0.3365 / 0.2069 | 0.4917 / 1.2721 / 0.6501 | 0.6861 | 0.8271 | 0.8522 | 12/12 |
| TASK_TRUST | 1 | selected point eligible e10 | 10 | rank 5 / 29 | 0.8483 / 0.3427 / 0.2087 | 0.4940 / 1.2662 / 0.6472 | 0.6928 | 0.8259 | 0.8405 | 12/12 |
| RAW_PROJ | 0 | reference primary comparator point eligible | 0 | rank 5 / 29 | 0.8492 / 0.3365 / 0.2069 | 0.4953 / 1.2721 / 0.6502 | 0.6861 | 0.8302 | 0.8514 | 12/12 |
| RAW_PROJ | 1 | reference primary comparator point eligible | 0 | rank 5 / 29 | 0.8483 / 0.3427 / 0.2087 | 0.4953 / 1.2661 / 0.6472 | 0.6928 | 0.8264 | 0.8408 | 12/12 |
| RAW_HYBRID | 0 | reference diagnostic greater disclosure | 0 | rank 16 / 29 | 0.8492 / 0.3365 / 0.2069 | 0.4953 / 1.2721 / 0.6502 | 0.6861 | 0.8389 | 0.8503 | 12/12 |
| RAW_HYBRID | 1 | reference diagnostic greater disclosure | 0 | rank 16 / 29 | 0.8483 / 0.3427 / 0.2087 | 0.4953 / 1.2661 / 0.6472 | 0.6928 | 0.8373 | 0.8457 | 12/12 |
| PRIOR_SEPARATE_GUARD_E5 | 0 | prior diagnostic fit2 point ineligible | 5 | rank 16 / 29 | 0.8492 / 0.3365 / 0.2069 | 0.4720 / 1.2858 / 0.6576 | 0.6861 | 0.8232 | 0.8386 | 6/12 |
| PRIOR_SEPARATE_GUARD_E5 | 1 | prior diagnostic fit2 point ineligible | 5 | rank 16 / 29 | 0.8483 / 0.3427 / 0.2087 | 0.4859 / 1.2797 / 0.6545 | 0.6928 | 0.8018 | 0.8273 | 8/12 |
| PRIOR_GLOBAL_UPDATE_GUARD_E5 | 0 | prior diagnostic fit2 point ineligible | 5 | rank 16 / 29 | 0.8492 / 0.3365 / 0.2069 | 0.4801 / 1.2804 / 0.6546 | 0.6861 | 0.8342 | 0.8361 | 6/12 |
| PRIOR_GLOBAL_UPDATE_GUARD_E5 | 1 | prior diagnostic fit2 point ineligible | 5 | rank 16 / 29 | 0.8483 / 0.3427 / 0.2087 | 0.4913 / 1.2755 / 0.6520 | 0.6928 | 0.8093 | 0.8367 | 8/12 |
| PRIOR_GLOBAL_UPDATE_GUARD_E5_OWNPROJ | 0 | prior diagnostic fit2 point ineligible | 5 | rank 5 / 29 | 0.8492 / 0.3365 / 0.2069 | 0.4801 / 1.2804 / 0.6546 | 0.6861 | 0.7909 | 0.8241 | 6/12 |
| PRIOR_GLOBAL_UPDATE_GUARD_E5_OWNPROJ | 1 | prior diagnostic fit2 point ineligible | 5 | rank 5 / 29 | 0.8483 / 0.3427 / 0.2087 | 0.4913 / 1.2755 / 0.6520 | 0.6928 | 0.7956 | 0.8264 | 8/12 |

- **Disclosure** is the effective occupation-representation rank out of the nominal 29-wide wire. Income is rank 1 of 21 everywhere.
- **"INNER point clauses (per seed)"** counts 12 per seed against U0/Ucal caps: accuracy 0.010, NLL 0.010, Brier 0.005.
- **Reader accounting:** 972 direct fits; the science lock planned 1,188 by mis-stating control banks as 216 per seed (actual 108 per seed).
- **Prior-guard rows** are fixed diagnostics that fail utility. They are not services.
- **Simultaneous status** for J_TRUST is in PRIMARY_ENDPOINTS.csv: utility 5 PASS, 19 UNRESOLVED, 0 violations; privacy pair benefit an established shortfall vs all five comparators.

## Component contrasts (diagnostic family K=45; recovery reduction = AUC_control − AUC_candidate, paired seeds)
| contrast | question | pair | occupation view |
|---|---|---|---|
| J_TRUST vs J_FREE | trust package | +0.0022 [−0.0041, +0.0086] unresolved | −0.0021 [−0.0082, +0.0039] unresolved |
| J_TRUST vs L_TRUST | coalition coupling at fixed trust | −0.0002 [−0.0060, +0.0056] unresolved | −0.0001 [−0.0052, +0.0051] unresolved |
| L_TRUST vs L_FREE | trust, local | +0.0034 [−0.0028, +0.0096] unresolved | +0.0054 [−0.0006, +0.0115] unresolved |
| J_TRUST vs TASK_TRUST | privacy term at fixed trust | +0.0051 [−0.0015, +0.0116] unresolved | +0.0082 [+0.0008, +0.0155] **established** |
| RAW_PROJ vs RAW_HYBRID | known compression | +0.0019 [−0.0060, +0.0097] unresolved | +0.0098 [−0.0016, +0.0212] unresolved |
| SEPARATE e5 vs RAW_HYBRID | did the old separate guard protect? | +0.0151 [+0.0044, +0.0257] **established** | +0.0256 [+0.0122, +0.0390] **established** |
| GLOBAL e5 vs RAW_HYBRID | did the old global guard protect? | +0.0116 [+0.0025, +0.0207] **established** | +0.0164 [+0.0046, +0.0282] **established** |
| GLOBAL own-proj vs GLOBAL | residual removal at fixed predictions | +0.0111 [−0.0003, +0.0226] unresolved | +0.0285 [+0.0146, +0.0424] **established** |
| GLOBAL own-proj vs RAW_PROJ | changed predictions under the head-row-space projection contract | +0.0209 [+0.0108, +0.0310] **established** | +0.0350 [+0.0219, +0.0482] **established** |

## Figures
- `figures/UTILITY_RECOVERY_PAIRED.png`: INNER utility vs pair recovery for every attacked service, with eligibility markers.
- `figures/FIT2_UTILITY_TRAJECTORIES.png`: utility only, every saved epoch.
- `figures/FITTING_VS_FIT2_LOSS.png`: overfit check.

## Chronology (UTC, 2026-10-14)
| time | event |
|---|---|
| 12:19 | campaign start |
| ~12:28 | protocol seal |
| ~12:33 | admission PASS |
| before lock | engineering tests (13 passed, after math review fixes) |
| 12:36:34 | SCIENCE_LOCK committed (ccf3d20) |
| 12:39:51 | training locked (8a2554f) |
| 12:40:59 | FIT2 once; SELECTION_LOCK (1ca9e30) |
| 12:46:29 | readers frozen; REPORTING_LOCK (c1b65da) |
| 12:47:55 | INNER opened once (acc842a) |

After that came deployment, backup and restore, independent verification, and publication. The prefit locks were committed locally and published with the clean package afterwards. They were not pushed before the fits; no retrospective-lock claim is made.
