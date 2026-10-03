# Research decision: joint complete-view method prototype

**Status: DEVELOPMENT evidence** on reused Adult rows; no fresh confirmation, no population/DP/MI guarantee.

**Pins.**
- Protocol lock: `b67664f` (before any real fit).
- Amendments: A1 `0629284` (finite centred logits; before any inner result) and A2 `a984a1e` (multiclass cell-conditional attacker; before any F/F0 outer output).
- `SELECTION_LOCK.json`: `197f323`, pushed before the single outer scoring.
- Independent replay: `INDEPENDENT_VERIFICATION.json`.

## Verdict

**Neither registered claim is established.**
- **No feasible configurations.** No LEACE-based arm (E, L, J, JP, S12, S21) met the registered utility gates on any seed: both tasks within one accuracy point of U, ≥ 80% retained gain, and ≥ 3 points above constant.
- **No J-vs-L claim.** L has no nominee, so the J-vs-L conjunction cannot be established.
- **No comparator.** No feasible comparator C\* exists, so claim 2 is NOT_ESTABLISHED (no substitution).
- **The cause is not the projection or training.** The mandatory final official LEACE alone (arm E) costs 4.9 points of income accuracy and 3.7 of occupation-group accuracy. It drives the predicted >50K-rate gap between men and women from 0.18 to 0.014 (true gap 0.20), i.e. near demographic parity. Both permitted tasks differ strongly by sex in Adult.

## Outer results (assessment, mean over 3 seeds; ✗ = INFEASIBLE closest-utility configuration)

Recovery is SEX AUC: local views v1, v2 and the coalition. Accuracy is from the deployed heads. Minority recall is for income (>50K).

| Arm | Status | v1 | v2 | Coalition | Synergy (coalition − max local) | Income accuracy | Occupation accuracy | Income minority recall |
|---|---|---|---|---|---|---|---|---|
| U | task-only reference | 0.861 | 0.877 | 0.873 | −0.004 | 0.843 | 0.476 | 0.61 |
| E | gates failed | 0.846 | 0.859 | 0.861 | 0.002 | 0.794 | 0.439 | 0.38 |
| L | ✗ | 0.744 | 0.810 | 0.830 | 0.021 | 0.782 | 0.436 | 0.22 |
| **J** | ✗ | 0.743 | 0.805 | **0.814** | 0.009 | 0.774 | 0.441 | 0.19 |
| JP | ✗ | 0.710 | 0.757 | 0.773 | 0.016 | 0.795 | 0.434 | 0.31 |
| S12 | ✗ | 0.741 | 0.789 | 0.816 | 0.025 | 0.784 | 0.448 | 0.23 |
| S21 | ✗ | 0.735 | 0.801 | 0.814 | 0.013 | 0.786 | 0.438 | 0.26 |
| F (official FARE) | ✗ (occupation) | 0.693 | 0.644 | **0.717** | 0.024 | **0.842** | 0.438 | 0.53 |
| F0 (zero-fairness twin) | gates pass | 0.798 | 0.857 | 0.874 | 0.017 | 0.849 | 0.465 | 0.59 |

The constant classifier scores 0.749 on income and 0.277 on occupation group. Full tables: `ACTUAL_TASK_UTILITY.csv`, `NATIVE_VS_AUDIT.csv`, `PRIMARY_ENDPOINTS.csv`, `SECONDARY_ENDPOINTS.csv`.

## Primary family (18 slots; z = 2.9913)

Claim 1 is computed on the INFEASIBLE configurations, so it is descriptive only.

| Clause | Point | Interval | Decision |
|---|---|---|---|
| P01: R_pair(L) − R_pair(J) > 0.02 | **0.017** | [0.009, 0.024] | NOT_ESTABLISHED (excludes zero; below target) |
| P02, P03: local guards (J − L < 0.01) | −0.001 / −0.005 | upper 0.009 / 0.003 | PASS / PASS |
| P04, P05: Acc(J) − Acc(U) > −0.01 | −0.068 / −0.035 | — | NOT_ESTABLISHED |
| P06, P07: retention | −0.050 / +0.005 | — | NOT_ESTABLISHED / NOT_ESTABLISHED |
| P08, P09: Acc(J) − const > 0.03 | 0.025 / 0.165 | — | NOT_ESTABLISHED / PASS |
| P10–P12 (claim 2) | — | — | NOT_ESTABLISHED (no C\*) |

**Claim 1: NOT_ESTABLISHED** (3 of 9 clauses pass, and the nominees are missing). **Claim 2: NOT_ESTABLISHED.** A clause count is not partial success.

## Secondary family (30 slots; z = 3.1440)

| Comparison | Point | Interval | Decision |
|---|---|---|---|
| J vs U, coalition | 0.059 | [0.049, 0.069] | PASS |
| J vs E, coalition | 0.048 | [0.038, 0.057] | PASS |
| J vs F0, coalition | 0.060 | [0.047, 0.074] | PASS |
| J vs S12, coalition | 0.003 | — | NOT_ESTABLISHED |
| J vs S21, coalition | 0.000 | — | NOT_ESTABLISHED |
| JP vs J, coalition | −0.040 | [−0.051, −0.030] | NOT_ESTABLISHED: JP is **lower** |
| F vs J, coalition | −0.097 | — | NOT_ESTABLISHED: FARE is **lower** |
| J vs L, probabilities only, coalition | 0.035 | [0.024, 0.047] | PASS |
| J vs L, hard decisions only, coalition | −0.002 | — | NOT_ESTABLISHED |
| J vs L, race (supported classes 1, 2, 4), coalition | 0.003 | — | NOT_ESTABLISHED |
| Projection vs penalty, income accuracy (J − JP) | −0.021 | — | NOT_ESTABLISHED |
| Projection vs penalty, occupation accuracy (J − JP) | 0.007 | lower −0.002, target −0.01 | PASS (non-inferior) |
| Coalition synergy > 0.02 | — | — | NOT_ESTABLISHED for any arm (L 0.021 [0.012, 0.029]) |

## Answers

1. **What the implemented algorithm does.**
   - Two separate MLP encoders release official-LEACE-erased 16-dimensional features plus centred logits of affine heads.
   - During training, whitened local critics (and, for J, a coalition critic) estimate SEX recoverability.
   - The privacy gradient is projected off the active task-loss guards (A-GEM per encoder), and steps are accepted only if no guard budget is exceeded.
   - It runs and deploys: `jcv.deploy` reproduces the saved releases bitwise.
2. **Did joint design beat the same method without coalition coupling?** **No; not established.**
   - Neither J nor L met the utility gates.
   - Descriptively, at their closest-utility configurations, J's coalition recovery is 0.017 lower than L's [0.009, 0.024]: below the 0.02 target, though the interval excludes zero. Local recovery is unchanged.
   - J also reduced the coalition-specific synergy (0.009 vs 0.021).
3. **Did it beat the strongest feasible matched control?** **No such control exists.** No arm met the gates with an L reference.
   - Descriptively, the plain penalty JP had **lower** coalition recovery than J (−0.040) and better income accuracy.
   - Official FARE had much lower recovery (−0.097) at U-level income accuracy, but lost 3.8 points on occupation group.
4. **Did both tasks remain useful?** **Not for any LEACE arm.**
   - J's income accuracy is 0.774 against U's 0.843; its gain over constant is 0.025 (below the 3-point floor); its minority (>50K) recall is 0.19 against 0.61.
   - Occupation group keeps a gain of 0.165 but loses 3.5 points.
   - FARE kept income (0.842, minority recall 0.53) but not occupation.
5. **Does any mathematical statement certify the measured privacy?** **No.**
   - The only exact statement is LEACE's linear guardedness on the fitting rows. It passes on all 48 erasure units (cross-covariance about 1e-15), and by affine closure it extends to the centred logits.
   - Nonlinear recovery of 0.71–0.86 is outside that theorem; this is not a refutation of it.
   - FARE's native certificate was refused by the wrapper's feature-hash guard. Under the record-identity guard it is vacuous (1.73) or unavailable.
   - Training critics' surrogates are not guarantees: JP's were about 0.02–0.05 while fresh attackers recovered 0.71–0.77.
6. **What is genuinely different from prior methods?** **Very little.**
   - The components are established: GEM/A-GEM projection, Song et al.'s critic surrogate, the LAFTR-style penalty, LEACE inside training, and PCRL's cross-purpose constraint.
   - The integration is a joint coalition critic over complete views with fixed information. It was implemented and tested, and it did not deliver a registered advantage.

## Registered predictions vs outcome (PROTOCOL §9)

| Prediction | Outcome |
|---|---|
| U high local recovery, synergy < 0.02 | **hit** (0.861 / 0.877; −0.004) |
| E nonlinear ≥ 0.75 | **hit** (0.85) |
| E utility passes | **miss** (−4.9 / −3.7 points) |
| L lowers local recovery vs E by > 0.05 | **hit** for v1 (0.10), borderline for v2 (0.049) |
| L "within the gates" | **miss** |
| Claim 1 NOT_ESTABLISHED | hit |
| Claim 2 NOT_ESTABLISHED | hit (C\* absent rather than F/L) |
| FARE lowest recovery; occupation gates exclude protective configurations | **hit** |
| Hard decisions well below primary views | **hit** (≈ 0.57 vs ≈ 0.81 for the protected arms; U 0.745 vs 0.873) |
| JP loses utility; no ≥ 0.02 J−JP difference | **miss**: JP had lower recovery and better income accuracy than J |

## Strongest favourable and adverse comparisons

- **Favourable** (descriptive, infeasible configurations).
  - J lowers coalition recovery relative to the task-only U by 0.059 [0.049, 0.069], and relative to post-hoc LEACE (E) by 0.048, while its local guards hold versus L.
  - On the XOR positive control it removed a coalition-only clue (pair 1.00 → 0.52) that local training could not see.
- **Adverse.**
  - The mandatory final LEACE makes every protected neural arm fail the one-point utility gates.
  - A plain penalty (JP) beat the projected update on coalition recovery (by 0.040) with better income accuracy.
  - Published FARE beat J by 0.097 at full income accuracy.

## Incomplete or infeasible arms

| Arm | Status |
|---|---|
| L, J, JP, S12, S21 | NO_FEASIBLE_NOMINEE (all seeds) |
| E | GATES_FAILED |
| F | NO_FEASIBLE_NOMINEE (occupation group) |
| F0 | Its own gates pass; it is labelled GATES_FAILED because it is anchored to F's infeasible occupation configuration (annotated in `SELECTION_LOCK.json`; no decision changes) |
| C\* | None |
| Original-PCRL adaptation; isolate-then-noise | Not run (optional): neither the architecture nor the noise semantics were admitted for these corrected inputs. The baseline set is not exhaustive. |

## Next step (prospective; not run)

A future design must either drop the mandatory final linear eraser for tasks whose labels depend on the protected attribute, or register a gate that permits the known parity cost. **Either would be a new protocol on new data**, not a rescoring of these outer results.
