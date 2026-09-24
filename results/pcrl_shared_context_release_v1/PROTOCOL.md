# Shared-context release v1: protocol (2018 development)

Registered 2026-09-24 before any release candidate of this study was fitted and before any new outcome was observed. Outcome-access status at registration: none. Design details: `DESIGN_SPEC.md` including Amendment M1. Reasons for the design: `RESPONSE_TO_HANDOFF.md`. Later changes go into `PROTOCOL_AMENDMENTS.md` with UTC time and outcome-access status.

## 1. Question

Can a release that uses permitted local information discarded by the 32-state code, sharing fitted parameters across people, reach a better residence-utility/sensitive-recovery tradeoff than D17 and than equally capable deterministic and adversarial controls? Evidence level sought: 2018 development (level 2 of the handoff). The 2018 households have been used repeatedly before, including this outer pool, which was opened once by the predecessor study. This study is not confirmation.

## 2. Data, roles, and access

- Data and roles are the predecessor's sanitized 2018 anchors 0, 1 and 2, with household roles from SHA256(`pcrl_adaptive_release_v1|household_id`): nuisance_train .20, audit_fit .20, coefficient_split .25, inner_selection .10, inner_check .10, outer .15.
- Roles are global across anchors, and the census reproduces the committed counts exactly (`agents/support/SUPPORT_CENSUS.json`).
- attacker_validation rows stay excluded, as in the predecessor.
- The outer role is opened exactly once, by the coordinator, after a committed and remotely verified selection lock.
- Legal deployment inputs, forbidden inputs and the wire contract are those of `DESIGN_SPEC.md` section 0.
- One persistent keyed token is released per record.

## 3. Panel (every anchor)

| Unit | Role in the study |
|---|---|
| NM1_U, NM1_P, NM4_U, NM4_P | Nested contextual policy mixture, K=1 and K=4 (K=2 if the registered support fallback triggers), utility-form and privacy-first-form LP. Nomination-eligible. |
| T32_U, T32_P | Same pipeline with eta fixed to 0 (parent-restricted control) |
| DET_SEL1, DET_SEL4 | Exact same-context deterministic selector (Amendment M1) |
| RD_TASK | Richer deterministic task-only policy |
| RD_PRIV | Richer deterministic privacy-trained policy |
| ADV_B1, ADV_B2 | PPAN-style adversarial categorical policy, beta 0.5 and 2.0 |
| D17 | Exact reference |
| Q_HIST | Historical task-directed Q. Pre-registered as the strongest available 17-token external reference: it passed all 8 measured sensitive clauses against J in the 2016 prospective study. |
| J, H_ONLY | Continuity references. J has a different, continuous interface and is not a matched control. |

The continuous erasers (LEACE, SPLINCE) are not refitted as 17-token controls. That is a stated limitation.

## 4. Fixed parameters

| Parameter | Value |
|---|---|
| delta | .001 |
| Alternation rounds | 6 |
| Policies (M) | 5 |
| Switching margin tau | .002 nats |
| Context counts | K in {1, 4} |
| K=2 support fallback | any K=4 context with < 300 unique coefficient households or household-weight ESS < 150 on any anchor |
| Seeds | predecessor convention: 20260924 + 10000·anchor + 100·round |
| ADV betas | {0.5, 2.0} |
| RD_PRIV | bisection ≤ 12 steps, 3 refit rounds |
| RD_TASK | 3 Lloyd rounds |

No coefficient sweep is run.

## 5. Independent audit

- The predecessor audit slate is used unchanged: six families; utility probe and the four sensitive roles A/SEX, A/RAC1P, AB/SEX, AB/RAC1P; own-release plus ignore-channel ancestor routes.
- Attackers and probes are fit on audit_fit, routes are chosen on inner_selection, and candidates are screened on inner_check.
- The audit attackers share audit_fit households with the optimization bank. They are separate fits, not independent data.
- Identical per-person laws on the admitted rows are audited once, and the alias ledger is kept.
- Positive controls (revealing channels) for AB/SEX and AB/RAC1P run on all three anchors before the lock. Detection means the H-minus-leak improvement is at least .01 nats in both weightings. If a control is not detected, audit strength for that role and anchor is marked uninformative.

## 6. Nomination (inner data only)

- **Contrasts:** three-anchor equal means on **inner_check** scores of the frozen validation-selected routes.
  - task = CE_Y(cand) − CE_Y(D17);
  - recovery = CE_S(D17) − CE_S(cand), where negative means less recovery for the candidate.
- **U screen:**
  - task ≤ −.003 in both weightings;
  - all 8 recovery contrasts ≤ +.001;
  - H capability (CE_Y(H) − CE_Y(cand)) ≥ .01 in both weightings.
  - Ranking key: (max task, mean task, max recovery, id).
- **P screen:**
  - task ≤ +.001 in both weightings;
  - AB/SEX recovery ≤ −.002 in both weightings;
  - the other 6 recovery contrasts ≤ +.001;
  - H capability ≥ .01.
  - Ranking key: (max AB/SEX, mean AB/SEX, max task, id).
- **Choosing the nominee:**
  - Each route's candidates are the four NM units (both LP forms are eligible for both routes).
  - The nominee is the rank-minimum among screen-passing candidates.
  - If none passes, it is the rank-minimum among all four, labelled `DIAGNOSTIC_ONLY` before the lock.
  - Never more than one nominee per route.
- **Comparator representatives:**
  - ADV is the rank-minimum of ADV_B1/ADV_B2 under the route's ranking key. Eligibility: for U, max recovery ≤ .001; for P, task ≤ .001. If neither is eligible, the rank-minimum overall.
  - RD_TASK and RD_PRIV have one representative each.

## 7. Locked development assessment

**Primary family.** Two slots (U, P), each with 4 comparators (D17, RD_TASK, RD_PRIV, ADV representative), 5 roles and 2 weightings, giving 80 endpoints. Bonferroni is computed from the generated list: two-sided 95%, z = norm.isf(.05/(2·80)).

Clauses and thresholds on the simultaneous interval's upper bound:

| Slot | Clause | Threshold |
|---|---|---|
| U | task | ≤ −.003 |
| U | 8 recovery guards | ≤ +.001 |
| P | task guard | ≤ +.001 |
| P | AB/SEX target | ≤ −.002 |
| P | 6 other recovery guards | ≤ +.001 |

**A route passes only if every one of its 40 clauses passes.** A pass against D17 alone is reported as such, not as a competitive advantage.

**Secondary family** (separately corrected, descriptive; not part of any pass):

- each nominee versus T32 (same LP form), DET_SEL (same K), the other-K NM unit, Q_HIST, J and H_ONLY;
- RD_TASK versus D17 (the task-sufficiency preflight).

**H-capability family:** 2 slots × 2 weightings, threshold .01.

**Uncertainty:**

- paired household bootstrap, 10,000 draws, seed 20260923, one multinomial resample over the union of household IDs across the three overlapping anchors;
- per-anchor weighted ratios, averaged equally over anchors;
- normal-approximation bounds, as in the predecessor.

Intervals condition on the fitted, selected objects. They do not cover the adaptive research history or the ACS survey design. An endpoint that is exactly zero because both releases chose the same H-only route is counted but labelled.

The endpoint list is generated by code, counted and hashed into `SELECTION_LOCK.json`. The lock is committed and pushed, the remote SHA is verified, and only then is the outer role opened, once, by the coordinator.

## 8. What will be reported regardless of outcome

For every unit:

- instantiation: policy disagreement with D17 within T32, and affected households;
- use: eta, A, within-T32 TV of the fitted law at every checkpoint;
- selection;
- all primary and secondary endpoints, with per-anchor estimates;
- the alias ledger;
- a separate count of logical slots and distinct released functions.

Also reported: positive-control results, and the three-level distinction (trained / used by the LP / selected).

## 9. Registered predictions (subjective, before fitting)

These are recorded to show calibration. The verdict uses results, not these.

| # | Prediction | P |
|---|---|---|
| 1 | At least one richer policy differs from D17 for > 10% of coefficient people on every anchor | .85 |
| 2 | NM4_U final round has within-T32 weighted mean TV > .02 on at least 2 anchors | .55 |
| 3 | The inner RD_TASK task contrast vs D17 is ≤ −.003 (X_A carries material task information beyond T0,H_A under this decoder family) | .35 |
| 4 | The U nominee passes the full U conjunction (all 40 clauses) | .03 |
| 5 | The P nominee passes the full P conjunction | .04 |
| 6 | Either nominee is inner-screen eligible, i.e. not DIAGNOSTIC_ONLY | .25 |
| 7 | On the outer assessment, the U nominee's task point estimate vs D17 is ≤ −.003 in both weightings | .15 |
| 8 | RD_TASK matches or beats the U nominee's task point estimate (within .001) | .55 |
| 9 | RD_PRIV matches or beats the P nominee on AB/SEX recovery at task cost ≤ +.001 (point estimates) | .45 |
| 10 | The ADV representative beats the P nominee on the same criterion | .25 |
| 11 | NM4 improves on NM1 of the same form by ≥ .001 on the route's primary contrast (inner) | .30 |
| 12 | DET_SEL4 is within .001 of NM4_U on the inner task contrast while inner-screen-comparable on recovery | .60 |

## 10. Stopping and resources

- Hard ceiling: 2026-09-25T13:26Z, or US$50, whichever comes first. Closeout starts no later than 11:30Z.
- Compute runs on one tagged c7i.8xlarge with a watchdog; the bounded retry policy covers technical failures only.
- Operational fixes never change margins, endpoints, nominees or comparison rules.
- If the deadline hits before the lock, no outer data are opened. The package is delivered as incomplete with resumable state.
