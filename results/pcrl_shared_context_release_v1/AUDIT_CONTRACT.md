# Independent audit contract (shared-context release v1)

Owner: baseline/audit agent. Written 2026-09-24 before any science fit and before any outcome was seen. Implementation: `experiments/pcrl_shared_context_release_v1/audit_panel.py` (infrastructure owner), which wraps the AR slate unchanged; positive controls in `poscontrol.py`. Sources cited: `agents/pipeline_audit/AUDIT_SELECTION_MAP.md`, `DESIGN_SPEC.md` §5, amendments M1–M5 (`PROTOCOL_AMENDMENTS.md`), and background §15–16.

**Principle.** Only this audit is privacy or utility evidence. Fixed-bank LP gains, frozen-attacker feasibility and best-response feasibility during fitting are optimization devices, not claims (M1.4; MATH_REVIEW §3.3: frozen attackers overstated a mixture's privacy by about 10 × delta).

## 1. Released object and legal inputs

- Every candidate, control and reference is audited as a per-person token law q_i(z), shape (n, 17), computed privately by `laws.person_law` from the six legal fields `x, ha, token_codes, teacher_p, residual, risk`. H_B, labels, ids, households, weights and roles are rejected before any artifact loads.
- The audited estimand is the exact expected-token cross-entropy E_{Z~q_i}[−log p(y_i | H_i, Z)], per original person, floored at 1e-9. No token draws are made.
- Kinds are `nested` (NM, T32, DET_SEL), `deterministic_policy` (RD_TASK, RD_PRIV; one-hot laws from `rd.load_law`), `adv_mlp` (ADV_B*; pruned softmax laws from `adv.load_law`), `historical_map` (D17, Q_HIST) and `h_only`. J is audited by the predecessor continuous-wire path (`AR/external_audit.py`) and is labeled contextual, not matched.
- Every unit is pinned by `release.json`: the SHA-256 of every file in it, re-verified after loading.

## 2. Families (the AR standard slate, unchanged)

The same slate is used for every audited role and every release:

- `logistic` (C = 1, lbfgs, with H × token one-hot interactions);
- `hist_gb_20` and `hist_gb_5` (150 iterations, 15 leaves, learning rate .1, l2 1; exact person-token expansion; min_samples_leaf = leaf × maximum per-person token support);
- `sampled_hist_gb_20` and `sampled_hist_gb_5` (one fixed token draw per person, seed 20260921 + seed; for deterministic laws they alias the exact trees);
- `mlp_120` ([64, 32] ReLU, Adam 1e-3, weight decay 1e-4, 120 epochs, checkpoint chosen on validation every 5 epochs).

Fits use balanced person weights .5 + .5·w/mean(w), so one predictor serves both weightings. The `catchup` slate (adds `mlp_360`) is not in the registered contract. It may be run only as a pre-registered stress test for every release alike.

The deliberately different family required by background §15 is the tree-based `hist_gb` pair next to the logistic and MLP families. No new family is added in v1. If a stronger network fails even the positive control, it is reported as an uninformative stress test.

**Support caveat (recorded, not changed).** Stochastic laws (NM, ADV) raise the maximum per-person support, which inflates the exact hist_gb min_samples_leaf by up to 17× and weakens those two auditors for stochastic releases only. The sampled_hist_gb, logistic and MLP auditors are unaffected. `audit_panel.hist_gb_record` logs support and the leaf actually used per release and role. ADV zeroes entries below 1e-3, which removes numerical dust but does not bound the maximum support.

## 3. Roles and seeds

| Step | Role |
|---|---|
| Fit every audit predictor | audit_fit |
| Select a route within each role (validation) | inner_selection |
| Score (the reported inner statistic) | inner_check |
| Outer development assessment | outer_assessment, only after the selection lock and remote-verified unlock (`audit_panel.verify_outer_gate`), replaying frozen inner routes with no refit or reselection |

- `assert_household_disjoint` is enforced at every fit.
- Seeds: `26000 + 1000·anchor + role_index` (`audit_panel.SEED_BASE`), identical across releases (common random numbers).
- H-only slates are fitted once per anchor and role and shared byte-for-byte by every release. The panel refuses an H score that differs across releases.

## 4. Validation selection and ignore-channel ancestors

Within each audited role, the route is the argmin of the balanced (0.5 U + 0.5 PWGTP) inner_selection loss over the candidate set below, with a lexical tie-break. The route is then frozen and scored on inner_check without refitting.

| Audited role | Candidate routes (standard slate each) |
|---|---|
| `utility:A/same_residence` (task utility probe) | the release slate on (H_A, z), plus the same-role H-only slate on H_A (12 routes) |
| `attack:A/SEX`, `attack:A/RAC1P` | the release slate on (H_A, z), plus the H-only slate on H_A (12 routes) |
| `attack:AB/SEX`, `attack:AB/RAC1P` | the AB release slate on (H_A, H_B, z), the AB H-only slate, the A same-release token slate on (H_A, z), and the B H-only slate on H_B (24 routes) |

- B never receives the token. A never sees H_B.
- H is unrounded and continuous.
- The fixed training decoder is not a route.
- An ignore-channel ancestor keeps a release from being penalised because its own fitted token model was worse than ignoring the token.

## 5. Endpoints

For candidate C and comparator B, averaged equally over anchors 0, 1 and 2:

- `Delta_task = CE_Y(C) − CE_Y(B)`. Negative is better.
- `Delta_recovery = CE_S(B) − CE_S(C)`. Positive means C makes sensitive inference easier.

**The 8 recovery endpoints.** {A/SEX, AB/SEX, A/RAC1P, AB/RAC1P} × {U = unweighted mean, W = PWGTP-weighted mean}.

**Task utility probe.** `utility:A/same_residence` × {U, W}.

**Routes (historical margins, unchanged).**

- U: Delta_task ≤ −0.003 in both weightings, and all 8 recovery endpoints ≤ +0.001.
- P: AB/SEX Delta_recovery ≤ −0.002 in both weightings, Delta_task ≤ +0.001 in both, and the other 6 recovery endpoints ≤ +0.001.

**H-capability.** The H-only minus candidate residence loss must be ≥ 0.01 nats in both weightings. It is a separate family (AR: 4 endpoints, z = 2.498). This check stops a constant or unusable release from winning through low recovery.

**Comparators for primary routes.**

- D17, RD_TASK, RD_PRIV and ADV (the inner-selected ADV_B*).
- Secondary family: T32_*, NM1 vs NM4, DET_SEL*, Q_HIST, J and H_ONLY.
- DET_SEL is the randomization-attribution control (M1.3).
- A baseline that wins is reported as a baseline win.

**Inference (reused unchanged from AR/TDR).**

- The endpoint list is generated by code, then counted and hashed before the lock. The Bonferroni correction is derived from that list.
- Paired household bootstrap with 10,000 common resamples over the union of households across anchors, seed 20260923. Bounds are the estimate ± z·SE.
- An endpoint that is exactly identical on every anchor has est = SE = 0. It is reported as such and does not count as an informative pass.
- An absolute fitted recovery and a difference of fitted recoveries are different estimands. A negative increment reflects finite training and selection, not negative conditional mutual information.
- Intervals condition on the fitted, selected objects. They do not model repeated use of 2018 and are not a full ACS complex-survey analysis.

## 6. How the fresh final attackers differ from the optimization banks

State this plainly in every report:

1. **Shared households, not independent data.** The audit fits on audit_fit and selects on inner_selection. These are the same households used by:
   - the NM round-0 attack bank and every alternation best response (fit_nm);
   - the RD_PRIV and ADV best-response attackers (rd/adv);
   - route and slate selection for all of them.

   The audit attackers are separate fits, not independent data. The different seeds (audit 26000+ vs fit_nm 20260924+ and RD/ADV best responses 56000+) are not data independence.
2. **Deterministic families can coincide exactly.**
   - For a deterministic release (RD_TASK, RD_PRIV, D17, DET_SEL, and NM when it collapses to a T32 kernel), `logistic` and `hist_gb` fits do not depend on the seed. The audit's release-route models of those families are then numerically the same as the corresponding best-response attackers fitted on the same law and rows.
   - Only the MLP and the sampled trees differ. So for those laws, "fresh" means a different selection and score role, not a different fit.
3. **The disjoint resource is inner_check.**
   - No fitting, pricing, bisection, checkpoint or round selection in any competitor or in NM reads inner_check. AR/NM only record an inner_check task diagnostic and never select on it.
   - The inner audit statistic is therefore scored on households that no optimizer or selector saw.
   - The outer development assessment is the only fully held-out score.
4. **The coefficient_split role is used by every optimizer** (LP coefficients, RD_PRIV bisection, ADV feasibility and ADV encoder training) and by no auditor.
5. **Winner's curse is symmetric but real.** Inner selection statistics are the minimum validation losses on inner_selection (AR caveat). Nominee selection uses them and gains optimism for every release. The inner_check score does not.

## 7. Positive controls (run before the lock)

- **Law.** A nondeployable label oracle: token = S (`AR/positive_control.revealing_token_law`, used unchanged through `poscontrol.run_control`).
- **Cells.** AB/SEX and AB/RAC1P on anchors 0, 1 and 2 (six cells; the predecessor ran only a0 AB/RAC1P).
- **Procedure.** The same standard slate, roles, H-only and legal ancestor routes as §4. Rows with a missing target label are excluded from all three inner roles, and the count is recorded.
- **"Detected".** On inner_check, the selected H-only route loss minus the selected revealing-route loss is ≥ 0.01 nats in both U and PWGTP. The wrapper asserts that this equals the predecessor's `detected_both_weightings`. 0.01 nats is a descriptive audit-power threshold, not a claim margin.
- **Failure.** If any cell is not detected, every "no additional recovery" statement for that role and anchor is reported as uninformative, not as survived.
- **Smoke.** On a 30% subsample of a0 (not science): AB/SEX improved U 0.692 and PWGTP 0.691 nats; AB/RAC1P improved U 1.215 and PWGTP 1.194 nats. Both were detected.
- **Limitation.** The oracle is maximal, so detection shows the slate can see a leak through the wire. It does not show sensitivity at the ±0.001–0.002-nat margins. A graded leak control was not implemented in v1, so no statement about detection power at the margin is licensed.

## 8. Exact-witness aliases (M4)

A privacy-constrained unit (NM*, T32*, RD_PRIV, ADV_B*, ADV_B*_P) whose selection lands on the exact D17 witness releases the one-hot D17 law. `rd.load_law` and `adv.load_law` return it bit-exactly, so `audit_panel.canonicalize` collapses the unit into D17's audit and records it in `ALIAS_LEDGER.json`. Each such unit is reported as `WITNESS_FALLBACK` (no checkpoint feasible) or `WITNESS_SELECTED_BY_RULE` (the witness won the unit's rule). Its excluded checkpoints and their violations are reported, and witness selections are counted per family and anchor. None of these is an uninstantiated test or a D17 win.

## 9. Same-host references and receipts

- Every release, D17, Q_HIST and H is audited in the same panel on the same host, with shared H slates. Nothing is subtracted from a historical audit.
- Each unit saves original-person and household U/W contributions and route selection receipts, both privately.
- A panel's `COMPLETE.json` inventories every artifact.
