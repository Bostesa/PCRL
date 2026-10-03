# Protocol: joint complete-view method prototype with matched controls

**Status: DEVELOPMENT.**
- All real data are Adult records already used by both repositories. Nothing here is fresh confirmation, a population privacy guarantee or a DP/MI guarantee.
- This protocol, the code pin, the input/role hashes, the unit manifest, families, thresholds, resampling, grid, fallback rules and `LOCK.json` are **pushed before any real model training**.

## 1. Data and roles (`DATA_ADMISSION.json`)

**Population and contract.**
- Adult, regenerated with the pinned loader; all 39,205 rows match the admitted record keys.
- Corrected input contract: 83 permitted columns. Removed: sex, race, income, occupation, fnlwgt, ids.
- `relationship ∈ {Husband, Wife}` determines SEX for about 46% of rows. It is kept as an ordinary permitted predictor and disclosed.

**Roles: oar-roles-v1, reused exactly.**

| Role | Rows | Use |
|---|---|---|
| defense_train | 19,230 | Encoders, heads, critics, LEACE maps, FARE trees |
| defense_val | 4,897 | Head C selection |
| attacker_fit | 6,065 | Inner and final attackers |
| attacker_val | 2,235 | Inner selection and final attacker selection |
| assessment | 5,243 | Single outer scoring |
| cert | 1,500 | FARE certificate |
| exposure-excluded | 17 | Never scored |

No defense gradient uses an outer label.

**Resampling unit.** The exact-record group, i.e. the de-duplicated record (5,243 groups in assessment). Adult has no household IDs; this grouping is not a household-independence proof.

## 2. Method and arms

See `METHOD_CARD.md`. The arms are U, E, L, J (candidate), JP (penalty ablation), S12 and S21 (neural sequential), F (official FARE per purpose) and F0 (its zero-fairness twin). β ∈ {0.1, 1, 10}; seeds 0, 1, 2. The schedule is in `LOCK.json["schedule"]`:
- 20 warm-start epochs, 20 protection epochs, batch 256;
- 5 critic steps per encoder step;
- LEACE refit every epoch plus a final refit;
- no early stopping: the final epoch is deployed.

**Matched budgets.** For every arm of a seed, the warm start, data order, critic initialisation and update counts are identical.
- **L:** 6 critics (3 per local view).
- **J / JP:** 6 critics (2 per view: v1, v2 and pair).
- **S12 / S21:** two 20-epoch stages that update one encoder each, with 2 + 4 critics.
- Each encoder receives 1,520 updates in every arm.

**Synthetic timing pilot** (done before the lock): about 0.9 CPU-s per protection epoch for J. The registered schedule fits the budget, so it was not reduced.

**Engineering rescue (registered, one shot).** If an arm has nonfinite gradients, or rejects every protection step, it is retried once at half the learning rate with the same maximum updates. The failed attempt is preserved in the record. Projection that removes every privacy direction while utility binds is a scientific limitation, not a defect.

## 3. Audits

| Audit | Design |
|---|---|
| Inner (selection) slate | LR(C = 1), MLP(64, 64), HGB(default). Fitted on attacker_fit, chosen by attacker_val log loss, scored as attacker_val AUC. The coalition bank adds the selected local attackers. |
| Final slate (primary views) | LR (5 C values); MLP with 4 width/depth settings; HGB (4 settings); a defense-aware MLP on the PCA-whitened release, dropping collapsed directions; for finite FARE releases, the cell-conditional Bayes attacker. Selection is on attacker_val log loss. The selected attacker is refitted at attacker seeds 0, 1, 2, and the scored quantity is SEX AUC on assessment averaged over those seeds. The coalition bank includes the ignore-other-view candidates. There is no assessment-based choice and no AUC orientation flip. |
| Secondary slate | The inner slate + defense-aware (+ cell-conditional for finite releases). Used for the output-only formats and the race audit on supported classes (≥ 30 rows in each scored role). |
| Utility | The deployed heads: accuracy, balanced accuracy, per-class and minority recall, log loss, Brier, ECE and a reliability table. A common refitted LR probe is reported separately. |
| Diagnostics | LEACE native checks (fitting rows) beside held-out linear diagnostics. FARE certificates per purpose. Label-only and constant references are diagnostics only. |
| Controls | Planted-leak and label-permutation controls are run on local and coalition views (`jcv/controls.py`; attacker_fit and attacker_val only). |

## 4. Running ladder

| Step | Action |
|---|---|
| A | Warm starts and U on all 3 seeds. |
| B | L, J, JP at all three β, plus E, F and S12/S21 on seed 0 (the runner may do all seeds together because the procedure is fixed). |
| C | Inner gates and selection (§5). |
| D | The same locked procedure on seeds 1 and 2. |
| E | Freeze the selected checkpoints, maps, heads, priors, role hashes and audit settings. **Push `SELECTION_LOCK.json` before the single outer scoring.** |
| F | Outer attacks, inference, independent replay, backup, closeout. |

## 5. Selection rules (`jcv/select.py`; inner roles only)

**Gates.** Per seed, on attacker_val, using the deployed heads, for **both** tasks:
- G1: Acc(arm) ≥ Acc(U) − 0.01;
- G2: (Acc − const) ≥ 0.8 (Acc(U) − const);
- G3: Acc − const ≥ 0.03.

Here const = the defense_train majority class.

**Selection order.**
1. **L first.** Among feasible β, minimise max(R_v1, R_v2). Ties go to the mean local recovery, then to the smaller β. The L nominee is frozen before it is used as a guard for other arms.
2. **J, JP, S12, S21.** Among β that are feasible and satisfy R_vi ≤ R_vi(L) + 0.01 for both recipients, minimise R_pair. Ties go to the smaller β.
3. **U and E** get no privacy tuning.
4. **F** uses its declared published rule within the gates. Per purpose, the admissible configuration with the lowest inner local AUC wins; ties go to the lower id.
5. **F0** is the γ = 0 twin at F's selected configuration.
6. **C\*** is the strongest feasible control. Among the {E, L, JP, S12, S21, F, F0} nominees that pass the gates and the local allowance versus L, choose the smallest inner R_pair. Ties are broken in the order E, L, JP, S12, S21, F, F0. If L wins, claim-2 slots alias claim 1.

**When nothing qualifies.**
- No feasible configuration gives `NO_FEASIBLE_NOMINEE`. The closest-utility configuration (largest worst-gate margin) is scored after the lock, labelled INFEASIBLE, and cannot pass a claim.
- No C\* gives claim 2 `NOT_ESTABLISHED`. No other comparator is substituted.
- A failed arm does not cancel the other arms.

## 6. Primary family (18 slots; `jcv/family.py`)

**Claim 1, J vs L** (one conjunction of 9 clauses):
- P01: R_pair(L) − R_pair(J) > 0.02;
- P02, P03: R_vi(J) − R_vi(L) < 0.01, judged on the upper bound;
- P04, P05: Acc_j(J) − Acc_j(U) > −0.01;
- P06, P07: Acc_j(J) − 0.8 Acc_j(U) − 0.2 const_j > 0;
- P08, P09: Acc_j(J) − const_j > 0.03.

**Claim 2, J vs C\*:**
- P10–P12: the same recovery clauses, with C\* in place of L;
- P13–P18: declared aliases of P04–P09.

**Inference.**
- Two-sided simultaneous Bonferroni intervals: point ± z·SE, with z = Φ⁻¹(1 − 0.05/36) = 2.991316.
- SE comes from 1,999 paired multinomial bootstrap replicates over assessment record groups (seed 20261013), with the same draws for every quantity.
- Endpoints average the per-seed statistics. Seed-level values and their spread are reported separately.
- A claim **passes only if all nine clauses pass and every seed has the required nominees**. A clause count is never reported as partial success.
- Intervals are conditional on fitted models, so they omit training variance. No extreme-percentile simultaneous coverage is claimed.

## 7. Secondary family (30 slots; separate Bonferroni; z = 3.143980)

- J vs L on output-only probabilities and hard decisions: coalition and local (6).
- Race stress audit, J vs L: coalition and local (3).
- Projection vs penalty (JP vs J): coalition, local and both accuracies (5).
- J vs each individual control's coalition recovery: U, E, JP, S12, S21, F, F0 (7).
- Coalition synergy, R_pair − max(R_v1, R_v2), per arm (9).

Trade-off curves and compute costs are descriptive.

## 8. Diagnostics registered from the review

- **Headroom.** Claim 1 needs coalition headroom. L's inner and outer synergy, R_pair − max local, is reported. If it is below 0.02, J can win only through local reductions; those are allowed, but local guards cap the cost.
- **Penalty mass.** J carries more penalty mass than L at the same β. The β grid spans a factor of 100 and lets L reach the higher strength; we report β-matched as well as selected comparisons descriptively.
- **Foreknowledge.** J vs S12/S21 includes foreknowledge of both purposes. Only J vs L holds information fixed.

## 9. Registered predictions (before any real fit)

| Item | Prediction |
|---|---|
| U | Local SEX recovery high on both views (~0.80–0.90, mainly via relationship and marital status). Coalition ≈ max local (synergy < 0.02). |
| E | LEACE leaves nonlinear recovery high (≥ 0.75). Utility passes. |
| L | Lowers local recovery relative to E by > 0.05 at the selected β, with tasks within the gates. |
| Claim 1 (J vs L) | **NOT_ESTABLISHED.** Little coalition synergy is expected, and the local guard leaves little room. P(PASS) ≈ 0.2. |
| Claim 2 (J vs C\*) | **NOT_ESTABLISHED.** P(PASS) ≈ 0.1. The likely C\* is F or L. |
| FARE | Lowest recovery among the feasible arms on income if a configuration passes the gates. The occupation-group gates may exclude the protective configurations. |
| Output-only hard decisions | Recovery well below the primary views for every arm. |
| Projection vs penalty | JP at large β loses utility; J keeps the gates. No ≥ 0.02 coalition difference is established. |

Findings outside this list will be labelled post hoc.

## 10. Repairs and amendments

Fixes are narrow, logged patches with regression tests, re-locked as dated amendments; the original locks are preserved. Only invalid affected units are rerun. Task labels, input exclusions, cohort, margins, seeds, objective and selection are not changed to obtain a favourable result.
