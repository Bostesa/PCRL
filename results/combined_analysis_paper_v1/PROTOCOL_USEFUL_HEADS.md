# Protocol: matched useful-head comparison (Adult income_prediction / sex)

**Status: development evidence. Written and pushed before any new fit.**

> These are reused development assessment rows (5,243 records, one record per group) that earlier studies in this repository have already scored many times.
> Naming primary endpoints and committing a lock does **not** turn them into fresh confirmation. Every result here is development evidence about these stored models and fitted attackers. No population privacy guarantee is inferred from attacker performance.

## 1. Question

The earlier studies compared feature releases using a common *refitted probe* for utility. Their output contracts released the outputs of separately fitted heads. This study asks a narrower question.

> When each arm releases the outputs of an **actually deployable task head** of one common family, fitted on that arm's own features, which arms keep task utility? And how much sex information can a recipient recover from those outputs?

## 2. Cell, arms and reuse (no new defense, no new encoder)

| Item | Value |
|---|---|
| Cell | Adult, purpose `income_prediction`, sensitive attribute `sex` (2 classes, both supported), PCRL Round-4 encoders, seeds 0, 1 and 2. All three seeds are reported; none is discarded. |
| Roles | `oar-roles-v1`, unchanged from the output-aware study. Exposure-cleaned (17 Adult test records equal to a training record are excluded from every scored role). Cert carve-out `oar-cert-v1`; head holdout `oar-head-v1` inside `defense_fit`. |
| **A** | Untreated `rep_p0` features. |
| **B** | The existing official target-LEACE map `adult__s<k>__income_prediction__B_sex` (concept-erasure 0.2.4), loaded and hash-verified, not refitted. |
| **F** | The existing validation-selected official FARE nominee: grid config 4 / 2 / 4 for seeds 0 / 1 / 2 (`selection/adult__s<k>.json`). Its saved cells (`FAREFIT_c<j>/cells.npy`) are reused as a one-hot release. |
| **F0** | The existing matched zero-fairness FARE tree (`FAREFIT_Z`): same configuration as the nominee, fairness weight 0. It is reused the same way. |

No defense grid is expanded and no candidate is re-selected. No encoder, LEACE map or FARE tree is refitted. No new population, ACS year, state or label is opened.

## 3. Common deployable head family

**Reuse.** Each arm's release head is the output-aware study's saved `HEAD__<A|B|F|FZ>` unit. They are reused without refitting, because they already match exactly across arms:

| Property | Value |
|---|---|
| Family | Multinomial logistic regression |
| Search | The benchmark U2 C grid |
| Fit role | `defense_fit` minus the `oar-head-v1` holdout |
| Selection | Holdout log loss |
| Runtime input | That arm's released features only |
| Release | The full log-probability vector for every row |

**Construction check (done before this protocol; no fit involved).** For all 12 heads:
- `logsumexp(L) = 0` to within 1e-12.
- The common offset `c = (l0 + l1)/2` equals `-(softplus(d) + softplus(-d))/2` of the margin `d = l1 - l0`, to within 1e-12.

So **the offset is a deterministic function of the margin.** The full logits, the (centred, offset) recoding, the centred margin and the probabilities all carry the same information about each row. Any recovery difference among those four formats is attacker-fitting variation, not information. The hard decision `1[d > 0]` is a coarsening.

**Utility reported.** For each arm and seed, on assessment:
- the head's accuracy, balanced accuracy, recall of the minority class (>50K), log loss, and gain over the constant;
- the constant is the attacker_fit majority class, accuracy 0.7492.

The historical refitted-probe accuracy (`U2__<arm>`) appears in a separate column and is **not** used in any endpoint.

## 4. Recipient contracts and formats

| Contract | Released to the recipient | Role |
|---|---|---|
| **Output-only** | The head's output in one format: `full` log-probabilities, `dc` (margin, offset), `centred` (margin), `prob`, `hard` (one-hot decision) | **Primary** |
| **Features plus own output** | The arm's features plus its own head's output (`full`, `centred`, `prob` or `hard`) | Secondary |
| Features only | The arm's features | Secondary, for interpretation |
| Features plus historical clean output | The arm's features plus the *untreated* PCRL `income_prediction` logits | **Adverse control.** This is the bypass that the earlier contracts allowed: a release that should not be permitted. |

**Attackers.** The benchmark slate is unchanged: L (5 C values), GBT (20 configurations), MLP (18).
- NL = GBT vs MLP, selected on attacker_val log loss and refitted at attacker seeds 0, 1 and 2.
- Recovery = sex macro AUC on assessment, averaged over the three attacker seeds, then over encoder seeds.
- **Features plus output:** the registered plus-surface rule applies. If the ignore-features or ignore-output candidate wins on validation, the component unit is scored.
- **Finite releases** (every F and F0 output, and all hard decisions): the cell-conditional attacker is also fitted and reported descriptively (NLDA). It is not in any family.
- **Banks:** the ignore-offset bank is {centred, prob}; the full bank is {full, dc, centred, prob} for output-only and {full, centred, prob} for features plus output. Selection is by validation log loss only.

## 5. Endpoint families (frozen in `cap/family.py`; asserted)

**Primary: 19 endpoints, output-only, means over the three seeds.**
- **Utility (9).** For each of B, F and F0:
  - `Acc(arm) − Acc(A)` > −0.01 (non-inferiority);
  - `Acc(arm) − Acc(const)` > 0.03;
  - retention `Acc(arm) − 0.8·Acc(A) − 0.2·Acc(const)` > 0.
- **Recovery (10).** For the `centred` and `hard` formats:
  - `R(A)−R(B)`, `R(A)−R(F)`, `R(A)−R(F0)`, `R(B)−R(F)`, `R(F0)−R(F)`, each > 0.02.

**Secondary: 33 endpoints, its own Bonferroni family.**
- Output-only `full` contrasts (5).
- Features plus own output `full`, `centred` and `hard` contrasts (15).
- Features-only contrasts (5).
- Offset: `R(full bank) − R(ignore-offset bank)` per arm, output-only (4).
- Bypass: `R(features + clean) − R(features + own full output)` per arm (4).

**Inference.**
- Paired multinomial bootstrap over assessment record groups: B = 1999, seed 20261041, the same draws for every unit.
- SE has ddof 1. Interval = point ± z·SE, with z the two-sided Bonferroni normal critical value:
  - primary: z = Φ⁻¹(1 − 0.05/38) = **3.007787**;
  - secondary: z = Φ⁻¹(1 − 0.05/66) = **3.171766**.
- **PASS** iff lower > target; otherwise **NOT_ESTABLISHED**.
- **Wording rule.** An interval that excludes zero but misses the target is reported as "a smaller difference, below the registered target", never as "not significant" or "no difference".
- Intervals are conditional on the fitted attackers and heads; between-seed spread is reported separately.

## 6. What is already known, and registered predictions

**Already known (not predictions).** The head accuracies were already printed in the output-aware study's unit records, and were re-read in the construction check above. Utility endpoints are therefore a planned *re-analysis* of known quantities:

| Arm | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| A | 0.800 | 0.852 | 0.844 |
| B | 0.798 | 0.852 | 0.844 |
| F | 0.798 | 0.845 | 0.823 |
| F0 | 0.801 | 0.839 | 0.843 |

So `Acc(F) − Acc(A)` has a mean of about −0.010 and is likely NOT_ESTABLISHED for non-inferiority. The retention of F is close to 0.

**Registered predictions for the new recovery fits** (written before any fit):
1. **Output-only centred.**
   - `R(A)−R(B)`: PASS (a linear head on LEACE features has a score uncorrelated with sex).
   - `R(A)−R(F)`, `R(A)−R(F0)`: PASS.
   - `R(B)−R(F)`: NOT_ESTABLISHED.
   - `R(F0)−R(F)`: NOT_ESTABLISHED.
2. **Output-only hard.**
   - `R(A)−R(B)`: PASS.
   - `R(A)−R(F)`, `R(A)−R(F0)`: PASS, but smaller than for centred.
   - `R(B)−R(F)`, `R(F0)−R(F)`: NOT_ESTABLISHED.
3. **Offset (secondary):** NOT_ESTABLISHED for every arm. The offset is a function of the margin, so the two banks see the same information.
4. **Bypass (secondary):** PASS for B, F and F0 (the clean logits restore removed information); NOT_ESTABLISHED for A.
5. **Features plus own output:** the features dominate; contrasts track the features-only contrasts.

Findings not on this list will be labelled post hoc.

## 7. Controls, verification, budget

**Controls.** Real-data shuffled-label null and planted-leak controls are run on attacker_fit/val only (seed 0) for each new interface type:
- output-only centred (A);
- output-only hard (F, finite);
- features plus hard (F, finite);
- output-only prob (B).

The null is flagged if its validation AUC is > 0.55; the planted leak must reach > 0.75.

**Independent verifier.** It does not import `cap`, `odx`, `oar`, `stored_model_eval` or `report`. It rebuilds the surfaces from the saved log-probabilities, the roles, the bank and plus selections, the AUCs, the bootstrap, the bounds and the decisions, and replays sampled attackers.

**Budget.**
- Elapsed and CPU: 10 elapsed hours and 12 CPU-hours in total, including verification.
- Workers and memory: at most two heavy workers, with `OMP_NUM_THREADS=1`; peak RSS below 6 GiB.
- Disk: at least 5 GiB free.
- Planned work: 84 new attack units plus 4 controls, expected well under 1 CPU-hour.

**Units.** 184 planned: 48 aliases, 84 new attack units, 48 banks and 4 controls. They are listed in `COVERAGE_AND_UNITS.csv` and `UNITS_PLANNED.json`.

**Failure rule.** A scientific failure is reported as such. Margins, seeds, labels, arms and the head family are not changed after this lock.
