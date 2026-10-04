# Protocol: focused method improvement — ordinary penalty without mandatory linear erasure

## Exposure disclosure (registered before fits)

> This method change was motivated by the completed, already-opened Adult development study. All new results use previously exposed data and are EXPLORATORY DEVELOPMENT evidence. A pre-fit lock improves reproducibility and limits outcome-driven changes; it does not restore fresh confirmation.

## 1. Data, roles and reuse

**Inputs and labels, reused exactly from the predecessor.**
- The admitted input `adult_jcv.npz` (SHA256 `e0d9e54a…f12`): 83 permitted columns, the same preprocessing, row IDs, exclusions and roles (oar-roles-v1).
- Labels: income and occupation group.
- Protected attribute: SEX, the primary target for both recipients. Race is the secondary supported-class stress audit (classes 1, 2, 4).
- The disclosed relationship proxies are kept.

**Reused predecessor units** (alias records with `COMPLETE.json` hashes): the warm starts, U, E, JP at β ∈ {0.1, 1, 10}, the FARE grid trees, F0 (frozen `Z1` units) and their inner audits. All share identical inputs, code and deployment identities.

## 2. Arms (`METHOD_DELTA.md`)

| Arm | Role |
|---|---|
| PN | Candidate |
| LN | Local ordinary-penalty control |
| U, E, JP (erased), F, F0 | Mandatory references |
| J, L, S12, S21 | Historical diagnostics, kept with their actual (projected) update rules |

**New fits.** PN and LN at β ∈ {0.1, 1, 10}, seeds 0/1/2: **18 new nonzero-penalty encoder fits.**

## 3. Engineering checks before any nonzero fit (registered)

| Check | Expectation |
|---|---|
| Fixture | On synthetic data, PN and LN at β = 0 reproduce the task-only arm bitwise |
| Real data, β = 0 | On every seed, PN and LN at β = 0 reproduce U's model state bitwise and its release within 1e-12, with identical hard decisions and no map |
| Real data, fidelity | This study's copy of the training loop reproduces the predecessor's JP β = 1 seed 0 model bitwise |

- Failing receipts are kept, and nonzero fits refuse to start until parity passes.
- After parity, the β = 0 points are exact aliases of U (`nn__s{k}__U`), not new models. They are explicit utility-feasibility witnesses; if selected, they are labelled **TASK_ONLY_ALIAS** and cannot support a joint-training claim.

**Rescue.** One retry at half the learning rate applies, registered now, but only for nonfinite training. Poor scientific performance is not a trigger.

## 4. Selection (`pnx/select.py`; inner roles: attacker_fit → attacker_val)

**Gates** (both tasks, deployed heads; as in the predecessor):
- G1: Acc ≥ Acc(U) − 0.01;
- G2: retained gain ≥ 80%;
- G3: Acc − const ≥ 0.03.

**Selection order.**
1. **LN first**, among feasible β ∈ {0, 0.1, 1, 10}. Minimise the worse local SEX recovery; ties go to the mean local recovery, then the smaller β. LN is frozen before PN is selected.
2. **PN**, among β ∈ {0, 0.1, 1, 10} that are feasible and satisfy R_vi ≤ R_vi(LN) + 0.01 for both recipients. Minimise coalition recovery; ties go to the smaller β.
3. **JP (erased)**, by the same rule as PN over β ∈ {0.1, 1, 10}.
4. **E**: gates only. **U** is the reference.
5. **F**: per purpose, the declared published rule (lowest inner local AUC among gate-passing configurations; ties go to the lower id).
6. **F0**: the frozen `Z1` twin. Its **own gate status** and its **pairing to F** are separate fields; it is never labelled infeasible merely because F lacks a nominee.
7. **C\*** is chosen from LN, U, E, JP, F and F0: smallest inner coalition recovery among candidates that pass their own gates and the local allowance versus LN. Ties are broken in that order.

**Rules for the frozen selection.**
- No feasible configuration gives `NO_FEASIBLE_NOMINEE`, and the closest-utility configuration is scored descriptively as INFEASIBLE.
- If even U fails G3 on a seed, that seed has **NO_VALID_REFERENCE**: no arm is a nominee and both claims fail (implemented).
- The descriptive closest-utility fallback uses nonzero β only (β = 0 would be U itself).
- C\* records its status, its unit and whether it is a task-only release.
- Margins are never loosened.
- Model IDs, identity maps, heads, priors, controls and attacker settings are frozen. `SELECTION_LOCK.json` is pushed before the single outer scoring.
- No selection or rescue may use outer scores.

## 5. Outer audit (`pnx/outer.py`)

**Scored grid (descriptive).** Every complete grid point is scored: U, E, JP×3, PN×3, LN×3, F and F0 on each seed (39 units). The primary family uses only the frozen selections; the curves choose no winner.

**Attackers.**
- Final LR/MLP/HGB/defense-aware slate, refitted per release, with selection on attacker_val log loss and no outer AUC orientation flip.
- **Every coalition bank records the coalition's own slate table, both ignore-other-view tables and the bank decision**, even when a local attacker wins.

**Releases and views.**
- Primary release: features plus centred logits; outputs are canonicalised for every arm.
- Secondary views: probabilities and hard decisions.
- Race audit on supported classes.

**Utility.** Deployed-head accuracy, balanced accuracy, minority recall, log loss, Brier score and ECE (reported separately), plus a common probe.

**Native diagnostics.** These are measurements, not guaranteed passes:
- fitted-moment cross-covariance on the fitting rows, which is 0 up to rounding on the fitting rows for erased controls by construction; plus the rotation-invariant OLS R² of SEX on r1, r2 and [r1, r2], with a shuffled null;
- held-out correlation.

**FARE certificates** remain the predecessor's: unavailable or vacuous.

**Controls.** Label-only and constant references; real-data shuffled-label and planted-leak controls.

## 6. Critic-gap diagnosis (`pnx/critic_gap.py`; corrected after review R1/R2, before the lock)

**Snapshot.** The diagnosis runs on every PN and LN unit, using inner roles only, at **θ_{T−1}**: the model state at the last critic update. The saved online critics and their saved ZCA whitener belong to that state. Applying the saved whitener to the final state θ_T would amplify tiny changes (the eigenvalue floor gives about 10⁴×). That would be a whitening artefact, not critic weakness.

**Setup.** On the same training-time views (encoder + training head, identity map) and the same saved whitener:
- **online:** the saved critics;
- **fresh_def:** same architectures, freshly initialised and trained on the same defense_train rows with a fixed schedule;
- **fresh_att:** the same protocol trained on attacker_fit rows;
- the fitted-prior loss.

All are scored with SEX cross-entropy on the **same attacker_val rows**.

**Primary statistic (registered).** The mean over the bank's critics of CE(online critic j) − CE(fresh_def critic j), paired by critic index (same kind).

**Secondary.** Best-of-bank values and the inner sklearn slate, which are optimistic. Also recorded:
- **sensitivity:** θ_T with a ZCA whitener refitted on the same reference rows;
- covariance spectra and feature scales;
- the slate's loss on the deployed views.

**Rule.** This diagnostic informs any future optimiser change; it authorises no new training grid.

## 7. Families (`pnx/family.py`)

**Primary: 18 slots, z = Φ⁻¹(1 − 0.05/36) = 2.991316.**

| Claim | Clauses |
|---|---|
| A: PN vs LN | P01: coalition R(LN) − R(PN) > 0.02. P02, P03: local R(PN) − R(LN) < 0.01 (upper bound). P04–P09: the six utility clauses versus U and the constant. **Interpretation:** A tests joint vs local penalty training at their selected β. That contrast also changes penalty mass, bank sizes and clip share (METHOD_DELTA), so a pass is not a pure coupling effect. |
| B: PN vs C\* | The same nine clauses with C\*. The utility clauses are aliases. |

**When a claim passes.**
- **A** requires all nine clauses to pass and, on every seed, both PN and LN to be **nonzero** nominees. A TASK_ONLY_ALIAS on either side means the coupling component is not isolated, and A is NOT_ESTABLISHED.
- **B** requires nine passes, a nonzero PN on every seed and C\* on every seed.
- Missing nominees never shrink the family and never permit substitution.

**Secondary: 30 slots, z = Φ⁻¹(1 − 0.05/60) = 3.143980.**

| Group | Endpoints | Count |
|---|---|---|
| Erasure on/off | PN − JP, β-matched, averaged over β and seeds: both accuracies and three recovery views. Two-sided ABOVE / BELOW / NOT_RESOLVED versus 0. **Utility gain and recovery increase are both reported.** | 5 |
| Output-only formats | PN vs LN, probabilities and hard decisions | 6 |
| Supported-race audit | PN vs LN | 3 |
| Individual controls | PN vs each of LN, U, E, JP, F, F0 on the coalition | 6 |
| Synergy | Coalition minus best local, for PN, LN, U, E, JP, F, F0 | 7 |
| β-matched | PN vs LN on the coalition, per β | 3 |

**Inference.**
- Paired record-group bootstrap: B = 1,999, seed 20261015.
- Means of per-seed statistics; all seeds are shown with their spread.
- Intervals are conditional on the fitted models. Record groups are not households, and encoder or attacker seeds are not additional people.

## 8. Registered predictions (before fits)

1. **Removing erasure restores accuracy.** PN − JP accuracy is ABOVE 0 on income, by about +3 to +5 points, and probably on occupation group.
2. **Removing erasure raises recovery.** PN − JP recovery is ABOVE 0, by about +0.02 to +0.06.
3. **LN** has a feasible nonzero β on at least one seed (most likely 0.1). At β ≥ 1, utility gates likely fail.
4. **Claim A: NOT_ESTABLISHED.** Coalition synergy is small, so a ≥ 0.02 PN-over-LN coalition margin with local guards is unlikely. P(PASS) ≈ 0.15.
5. **Claim B: NOT_ESTABLISHED.** P(PASS) ≈ 0.1. The likely C\* is LN (nonzero) or F0 (from inner values already known: F0 passes its own gates and has lower coalition recovery than U).
6. **Critic gap.** The primary statistic, the mean paired CE(online) − CE(fresh_def) at θ_{T−1}, is **positive** (fresh critics read more) for most PN/LN units, by a few hundredths of a nat.

## 9. Budget and resources

- Local only, $0 cloud spend.
- Ceilings: 8 h elapsed and 12 CPU-h, with 2 h reserved for verification, backup and handoff.
- At most 2 heavy workers, under 8 GiB of memory, and at least 5 GiB free on the laptop.
- Private outputs go in `~/PCRL_eval_cache_private/pnx_v1`, mirrored to `<drive>/private_pnx_v1_20261003`.
