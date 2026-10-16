# Pre-lock statistical review: ACS protected-tree recipe replication

Role: independent statistical reviewer. Date: 2026-10-15. Prospective review, done before SCIENCE_LOCK. I read code and documents only. I opened no exports, raw files, runs or `.private` data.

Files reviewed (sha256 prefix at review time):

- `acs_replication/report.py` e6cdd871828e4c38
- `acs_replication/select.py` 3f432687c6f9312c (re-read after the lead's mid-review edit that writes `two_task_valid`)
- `acs_replication/lock_science.py` 46de46906387dc57
- `acs_replication/readers.py` a176d76d913eec4a
- `acs_replication/data.py` 7e7cfae034789b27
- `acs_replication/schema.py` 113bbaeaff3a7c54

Predecessor conventions consulted: `task_headroom/inference.py`, `task_conditioned_representation/inference.py` (`bootstrap_auc`, `interval`) and `results/pcrl_frozen_tree_control_completion_v1/STATISTICAL_DESIGN_IUT.md`.

## Verdict

The architecture is sound:

- one paired household bootstrap shared across services, endpoints and seeds;
- three claims, each tested as an intersection-union test (IUT), with Bonferroni across the claims;
- reader winners and orientation chosen on ATTACK_VALIDATION;
- comparator configurations chosen on MODEL_SELECT;
- DEV_REPORT opened once.

**I found no selection on DEV_REPORT.**

**Six fixes are REQUIRED before lock.** Four concern decision logic or labels:

- the utility gate in the recipe claim is described as an α/3 test, but it is a point check;
- the "established" harm, violation and shortfall labels come from marginal bounds;
- the competitive claim does not hold LOCAL_TREE to the utility requirements the baselines had to meet;
- the planning output sizes single benefit clauses only.

Two concern wording and scope.

Critical values used below assume G ≈ 5,000 DEV_REPORT households (t with G − 1 df):

| Use | Level | t |
|---|---|---|
| IUT clause | one-sided 0.05/3 | 2.129 |
| IUT clause if α = 0.025 | one-sided 0.025/3 | 2.395 |
| Simultaneous display, K = 40 | two-sided 0.05/80 per side | 3.229 |

---

## REQUIRED fixes (must change before lock)

### R1. RECIPE_EFFECT mixes six α/3-tested clauses with a 24-clause point gate, and the text says every clause is tested at α/3

**Finding.** The decision requires all of the following:

- coalition benefit `iut_lower > 0.020` on both seeds (2 clauses);
- recipient worsening `iut_upper <= 0.010` (4 clauses);
- `utility_point_feasible_24`;
- `not utility_established_violation`.

As an IUT this is **valid for the six benefit and worsening clauses**. The rejection region is (R_benefit ∩ R_worsening ∩ point gate), which is contained in (R_benefit ∩ R_worsening). Adding a data-dependent gate can only shrink the rejection region. So P(false ESTABLISHED) ≤ α/3 whenever a benefit or worsening null is true.

The gate carries **no error control for utility**. If a utility clause is truly just over its cap, the point check passes with probability about 0.5. The docstring ("every clause at one-sided level alpha/3") and the lock text ("...point-pass with no established violation") therefore overclaim.

`not utility_established_violation` is logically implied by point-pass: if point ≤ cap, then point − c·SE ≤ cap. The term adds nothing, but it suggests an extra test.

The user asked for point feasibility and confidence-bound status to be reported separately. That can be done inside the same alpha. "Recipe + all 24 utility clauses pass at their α/3 bounds" is a nested, stronger conjunction. It shares the six clause tests and its rejection region is a subset of the recipe claim's, so it needs no extra alpha (nested-conjunction argument, STATISTICAL_DESIGN_IUT.md §2).

**Patch, `report.py`.** Replace the whole `# ---------------- claims` block (current lines 164–190) with the block below. The block also contains the R2 and R3 changes, so apply it once.

```python
    # ---------------- claims (decisions use iut_pass only; failing-direction labels use sim_status only)
    def members(pred, n):
        eps = [v for k, v in E.items() if pred(k)]
        if len(eps) != n:  # all([]) is True: never let a missing endpoint pass a conjunction
            raise AssertionError(f'claim roster mismatch: {len(eps)} endpoints, expected {n}')
        return eps
    ben = members(lambda k: k.startswith('RECIPE|coalition_benefit'), 2)
    wor = members(lambda k: k.startswith('RECIPE|') and '_recipient_worsening_' in k, 4)
    utl = members(lambda k: k.startswith('RECIPE|utility|'), 24)
    sg = members(lambda k: k.startswith('SUBGROUP|'), 4)
    rc = res['claims']['RECIPE_EFFECT'] = {
        'coalition_benefit_iut_pass_both_seeds': all(v['iut_pass'] for v in ben),
        'recipient_non_worsening_iut_pass_all': all(v['iut_pass'] for v in wor),
        'utility_point_feasible_24': all(v['point_pass'] for v in utl),
        'utility_iut_bound_pass_count_of_24': int(sum(v['iut_pass'] for v in utl)),
        'utility_simultaneous_status_counts': {s: sum(v['sim_status'] == s for v in utl) for s in ('PASS', 'UNRESOLVED', 'ESTABLISHED_VIOLATION')}}
    six = rc['coalition_benefit_iut_pass_both_seeds'] and rc['recipient_non_worsening_iut_pass_all']
    rc['decision'] = 'RECIPE_EFFECT_ESTABLISHED' if six and rc['utility_point_feasible_24'] else 'RECIPE_EFFECT_NOT_ESTABLISHED'
    rc['nested_decision_utility_bounds'] = ('RECIPE_EFFECT_AND_UTILITY_BOUNDS_ESTABLISHED' if six and rc['utility_iut_bound_pass_count_of_24'] == 24
                                            else 'RECIPE_EFFECT_AND_UTILITY_BOUNDS_NOT_ESTABLISHED')
    rc['decision_scope'] = ('error-controlled (IUT, one-sided alpha/3 per clause): coalition benefit > 0.020 and each recipient worsening <= 0.010 '
                            'for both fitted seed pairs, conditional on the fitted models and frozen readers; utility: point-feasibility gate only '
                            '(no error control); utility bounds: nested decision (same clause tests, no extra alpha)')
    # R3: LOCAL_TREE must meet the same utility requirements as the baselines it is compared with
    lt_select_feasible = bool(sel['fixed_services'].get('LOCAL_TREE', {}).get('point_feasible_24', False))
    same_requirements = lt_select_feasible and rc['utility_point_feasible_24']
    eligible = [f for f in REQUIRED_BASELINES if comp[f]['selection_status'] == 'ELIGIBLE']
    complete = all(comp[f]['service'] for f in REQUIRED_BASELINES) and bool(sel['frozen']['FAGTB'].get('two_task_valid', False))
    beats = {f: all(E[f'COMPETITIVE|coalition_benefit_vs_{f}|s{s}']['iut_pass'] for s in (0, 1)) for f in REQUIRED_BASELINES if comp[f]['service']}
    res['claims']['COMPETITIVE_COMPARISON'] = {
        'baselines': comp, 'eligible_required_baselines': eligible, 'complete': bool(complete),
        'local_tree_same_utility_requirements': {'model_select_point_feasible_24': lt_select_feasible,
                                                 'dev_report_point_feasible_24': rc['utility_point_feasible_24']},
        'superiority_iut_pass_vs': beats,
        'decision': ('INCOMPLETE' if not complete else
                     'LOCAL_TREE_FAILS_SAME_UTILITY_REQUIREMENTS' if not same_requirements else
                     'NO_ELIGIBLE_BASELINE' if not eligible else
                     'SUPERIOR_TO_ALL_ELIGIBLE' if all(beats[f] for f in eligible) else 'NOT_SUPERIOR_TO_ALL_ELIGIBLE'),
        'decision_scope': 'coalition (pair) recovery only; margin 0; one-sided alpha/3 per clause; both seeds; versus exactly the ELIGIBLE baselines listed'}
    # R2: QUALIFIED from the IUT bounds; HARM only from the K = 40 simultaneous intervals
    res['claims']['SUBGROUP_SUITABILITY'] = {
        'estimable': estimable_subgroups,
        'guards_iut_pass': {k: v['iut_pass'] for k, v in E.items() if k.startswith('SUBGROUP|')},
        'guards_simultaneous_status': {k: v['sim_status'] for k, v in E.items() if k.startswith('SUBGROUP|')},
        'decision': ('NOT_ESTIMABLE' if not estimable_subgroups else
                     'SUBGROUP_QUALIFIED' if all(v['iut_pass'] for v in sg) else
                     'ESTABLISHED_SUBGROUP_HARM' if any(v['sim_status'] == 'ESTABLISHED_VIOLATION' for v in sg) else
                     'SUBGROUP_UNRESOLVED'),
        'decision_scope': 'QUALIFIED: IUT over the 4 guards at one-sided alpha/3; HARM: K = 40 simultaneous lower bound > 0.005 only; crossing intervals are unresolved'}
```

**Patch, `lock_science.py`, line 83.** Replace the `'RECIPE_EFFECT'` rule string with:

```python
'RECIPE_EFFECT': 'ESTABLISHED iff (IUT, one-sided alpha/3 per clause) coalition benefit vs TASK_REFIT has lower bound > 0.020 on both seeds AND each recipient worsening has upper bound <= 0.010 on both seeds, AND (point-feasibility gate; a deterministic condition on the estimates, no error control) all 24 LOCAL_TREE utility clauses pass at the point. Reported separately: the count of utility clauses whose one-sided alpha/3 upper bound <= cap, and the nested decision RECIPE_EFFECT_AND_UTILITY_BOUNDS (all 30 clauses at their bounds; same clause tests, no extra alpha). Conditional on the two fitted seed pairs and the frozen readers.',
```

### R2. Failing-direction labels are read from marginal α/3 bounds; `ESTABLISHED_SUBGROUP_HARM` is uncontrolled

**Finding.** `endpoint()` sets ESTABLISHED_VIOLATION and ESTABLISHED_SHORTFALL from the marginal one-sided α/3 bounds. `ESTABLISHED_SUBGROUP_HARM` fires if **any** of four guards has `iut_lower > 0.005`. That is a union of four tests.

With every guard on its boundary, P(false harm) ≈ 1 − (1 − α/3)^4 ≈ 0.065 under independence, and about 0.033 if the two seeds were perfectly correlated. Both exceed α/3, and the first exceeds the whole α budget. The IUT controls only the positive decisions.

The validated predecessor (`task_headroom/inference.py`: `status_upper` and `status_benefit` applied to `interval(..., critical(K, ...))`) reads PASS, VIOLATION and SHORTFALL from the Bonferroni-K simultaneous intervals. STATISTICAL_DESIGN_IUT.md §2 says marginal violations and shortfalls are descriptive. The user requires crossing intervals to be "unresolved, not harm". The intervals that matter here are the simultaneous ones.

**Patch, `report.py`.** Replace `endpoint()` (lines 46–56) with:

```python
def endpoint(point, draws, kind, margin, c1, cs):
    """Wald-type bounds: point +/- t(G-1) x paired household-bootstrap SE (not a studentized bootstrap-t).
    iut_*: marginal one-sided (1 - alpha/3) bounds, used ONLY for positive-direction claim clauses (iut_pass).
    sim95_*: K = 40 Bonferroni two-sided bounds; the ONLY source of failing-direction labels (sim_status)."""
    se = float(np.std(draws, ddof=1))
    rec = {'point': float(point), 'se': se, 'kind': kind, 'margin': margin,
           'iut_lower': float(point - c1 * se), 'iut_upper': float(point + c1 * se),
           'sim95_lower': float(point - cs * se), 'sim95_upper': float(point + cs * se),
           'construction': 'point +/- t(G-1) x paired household-bootstrap SE; conditional on fitted models and readers'}
    if kind == 'benefit':
        rec['iut_pass'] = bool(rec['iut_lower'] > margin)
        rec['sim_status'] = 'ESTABLISHED' if rec['sim95_lower'] > margin else ('ESTABLISHED_SHORTFALL' if rec['sim95_upper'] < margin else 'UNRESOLVED')
    else:
        rec['iut_pass'] = bool(rec['iut_upper'] <= margin)
        rec['sim_status'] = 'PASS' if rec['sim95_upper'] <= margin else ('ESTABLISHED_VIOLATION' if rec['sim95_lower'] > margin else 'UNRESOLVED')
    rec['point_pass'] = bool(point > margin) if kind == 'benefit' else bool(point <= margin)
    return rec
```

Replace line 159 (`ep['material_margin_0.020_status'] = ...`) with:

```python
            ep['material_margin_0.020_iut_pass'] = bool(ep['iut_lower'] > 0.020)
            ep['material_margin_0.020_sim_status'] = 'ESTABLISHED' if ep['sim95_lower'] > 0.020 else ('ESTABLISHED_SHORTFALL' if ep['sim95_upper'] < 0.020 else 'UNRESOLVED')
```

The subgroup decision is already in the R1 block. No other code reads `['status']`. I checked `report.py` and the tests.

**Patch, `lock_science.py`, line 85.** Replace the `'SUBGROUP_SUITABILITY'` rule string with:

```python
'SUBGROUP_SUITABILITY': 'SUBGROUP_QUALIFIED iff all four guards (occupation summed-Brier excess vs U0 within each SEX group, both seeds) have one-sided alpha/3 upper bound <= 0.005 (IUT); ESTABLISHED_SUBGROUP_HARM iff any guard has K = 40 simultaneous lower bound > 0.005; otherwise SUBGROUP_UNRESOLVED (crossing intervals are unresolved, never harm); NOT_ESTIMABLE if fewer than 200 DEV_REPORT households contain a member of a SEX group (counted from SEX alone, before any outcome is computed). All PASS/VIOLATION/SHORTFALL/HARM labels are read from the K = 40 simultaneous intervals; marginal alpha/3 bounds decide only the positive claim clauses.'
```

### R3. The competitive claim does not apply "the same utility requirements" to LOCAL_TREE; the FAGTB validity default fails open

**Finding.** Baselines are ELIGIBLE only if they are point-feasible on all 24 MODEL_SELECT clauses. LOCAL_TREE faces no utility condition at all in `COMPETITIVE_COMPARISON`. Its MODEL_SELECT feasibility is computed (`SELECTION_LOCK.fixed_services.LOCAL_TREE.point_feasible_24`) but never used.

So a LOCAL_TREE that loses utility could be declared `SUPERIOR_TO_ALL_ELIGIBLE`, because a degraded release trivially leaks less. That contradicts "under the same output contract and utility requirements".

Separately, `sel['frozen']['FAGTB'].get('two_task_valid', True)` fails open when the key is absent. `select.py` now writes the key, but the default should fail closed.

**Minimal prespecified rule** (implemented in the R1 block): the decision is `SUPERIOR_TO_ALL_ELIGIBLE` iff all of the following hold.

1. The comparison is complete: every required family ran, and FAGTB is two-task valid.
2. LOCAL_TREE satisfies the baselines' own eligibility rule: MODEL_SELECT 24-clause point feasibility, both seeds. This is the symmetric requirement, and it is fixed before DEV_REPORT opens.
3. LOCAL_TREE passes the DEV_REPORT 24-clause point gate already used in RECIPE. This applies to LOCAL_TREE only, so it can only make the claim harder.
4. At least one required baseline is ELIGIBLE.
5. Coalition benefit vs every ELIGIBLE baseline has a one-sided α/3 lower bound > 0 on both seeds.

The eligible set is printed with the decision.

**Not required by the user's text:**

- Recipient non-worsening vs baselines would be coherent as extra IUT clauses, but it is not in the user's text and costs power. Report it descriptively (RECOMMENDED Rc9).
- **Do not add "utility parity vs baselines".** The utility requirements are absolute caps against U0 and Ucal. Parity would need its own equivalence margin and would be a different claim.

**Patch, `lock_science.py`, line 84.** Replace the `'COMPETITIVE_COMPARISON'` rule string with:

```python
'COMPETITIVE_COMPARISON': 'SUPERIOR_TO_ALL_ELIGIBLE (coalition recovery only) iff the comparison is complete, LOCAL_TREE meets the same utility requirements (the baselines\' eligibility rule: 24-clause MODEL_SELECT point feasibility on both seeds; plus the DEV_REPORT 24-clause point gate of RECIPE_EFFECT), at least one required baseline is ELIGIBLE, and coalition benefit vs every ELIGIBLE required baseline (NEURAL_RAW, FAGTB two-task, FARE) has one-sided alpha/3 lower bound > 0 on both seeds; INCOMPLETE if a required family is missing or invalid (never a win); LOCAL_TREE_FAILS_SAME_UTILITY_REQUIREMENTS if the LOCAL_TREE gate fails; the decision names the eligible set; ineligible baselines and recipient-level differences are descriptive.',
```

### R4. The planning output sizes single benefit clauses at the observed point only

**Finding.** The per-clause formula n = ((z₁₋α/₃ + z₀.₈)·σ_h/δ)², with σ_h = SE·√G and δ = point − margin, is the correct normal-approximation size for **one** clause. Four things are wrong with how it is used:

1. **Only benefit clauses are planned.** The worsening, utility and subgroup clauses are omitted. In the predecessor study the binding clause was a no-excess one (the s1 occupation Brier clause, 0.24 SE of slack; STATISTICAL_DESIGN_IUT.md §5).
2. **80% is per clause.** An IUT needs every clause to pass. Conjunction power is at most the minimum clause power, and is at least 1 − Σ(type-II errors) under any dependence.
3. **Futile clauses are skipped silently.** These are clauses whose point lies on the failing side, and more households cannot rescue them.
4. **Only the observed point is used.** It is optimistic when a favourable development result triggers the sealed assessment.

The note correctly says the output is not uncertainty about the true effect. The key name `households_for_80pct_at_observed_point` still invites misreading.

**Patch, `report.py`.** Replace the planning block (lines 192–199) with:

```python
    # ---------------- planning for a later sealed assessment (PLANNING ONLY; not inference, not evidence)
    from scipy import stats
    zc, zp = stats.norm.ppf(1 - ALPHA / N_CLAIMS), stats.norm.ppf(0.8)
    slack = lambda v: (v['point'] - v['margin']) if v['kind'] == 'benefit' else (v['margin'] - v['point'])
    def power(v, n, shrink):
        s = slack(v)
        if s <= 0:
            return 0.0
        return 1.0 if v['se'] <= 0 else float(stats.norm.cdf(shrink * s * np.sqrt(n) / (v['se'] * np.sqrt(G)) - zc))
    per_clause = {}
    for k, v in E.items():
        sd_h = v['se'] * np.sqrt(G)
        per_clause[k] = {'household_sd': sd_h, 'slack_at_development_point': slack(v)}
        if slack(v) <= 0:
            per_clause[k]['status'] = 'FUTILE_AT_DEVELOPMENT_POINT'
        else:
            for lab, sh in (('at_point', 1.0), ('at_half_slack', 0.5)):
                per_clause[k][f'households_single_clause_80pct_{lab}'] = float(np.ceil(((zc + zp) * sd_h / (sh * slack(v))) ** 2))
    claim_keys = {'RECIPE_EFFECT': [k for k in E if k.startswith('RECIPE|') and '|utility|' not in k],
                  'RECIPE_EFFECT_AND_UTILITY_BOUNDS': [k for k in E if k.startswith('RECIPE|')],
                  'COMPETITIVE_COMPARISON': [f'COMPETITIVE|coalition_benefit_vs_{f}|s{s}' for f in eligible for s in (0, 1)],
                  'SUBGROUP_SUITABILITY': [k for k in E if k.startswith('SUBGROUP|')]}
    per_claim = {}
    for name, keys in claim_keys.items():
        for mult in (1, 2, 4, 8, 16, 32):
            for lab, sh in (('at_point', 1.0), ('at_half_slack', 0.5)):
                pw = [power(E[k], mult * G, sh) for k in keys]
                if pw:
                    per_claim.setdefault(name, {})[f'{mult}xG|{lab}'] = {
                        'households': int(mult * G), 'min_clause_power': min(pw),
                        'conjunction_power_lower_bound_any_dependence': max(0.0, 1 - sum(1 - p for p in pw))}
    res['planning'] = {'note': ('PLANNING ONLY. Development (DEV_REPORT) points and household SDs are used as assumed true values for the SAME frozen '
                                'models and readers scored on new households. Not uncertainty about the true effect, not evidence that any clause holds, '
                                'and optimistic if a favourable development result triggers the sealed assessment. An IUT claim needs every clause: '
                                'conjunction power lies between conjunction_power_lower_bound_any_dependence and min_clause_power. '
                                'FUTILE clauses cannot be rescued by more households. Refitting models would add variance not represented here.'),
                       'alpha_per_clause_one_sided': ALPHA / N_CLAIMS, 'per_clause': per_clause, 'per_claim': per_claim}
```

`eligible` comes from the R1 block, which runs before this one.

### R5. The error-rate convention and coverage wording in the inference docstring

**Finding.** The docstring is copied verbatim into `SCIENCE_LOCK.inference.design`. It has five problems:

- It says "every clause at one-sided level alpha/3". That is false for the utility gate (R1).
- It says "descriptive coverage". The intervals do have approximate simultaneous coverage. They are simply not the decision rule.
- It never states the familywise level. With α = 0.05 split over three one-sided claims, the family is controlled at **one-sided 0.05**. Each claim is held to ≤ 0.0167 one-sided, which is stricter than the predecessor's 0.025 per decision. But the family level is double the conventional one-sided 0.025, which STATISTICAL_DESIGN_IUT.md §3 explicitly rejected for a single decision.
- The bound construction is unnamed. It is a Wald bound with a bootstrap SE, not a bootstrap-t.
- It does not say that the per-clause bounds are marginal.

The lead may keep α = 0.05 (a development-study choice, see Rc1), but the lock must state the convention explicitly.

**Patch, `report.py`.** Replace the module docstring (lines 1–12) with:

```python
"""Open DEV_REPORT once (after the committed REPORTING_LOCK) and evaluate the frozen roster.

Inference (SCIENCE_LOCK 'inference'): one nonparametric household (cluster) bootstrap (B = 2048, seed 2026101603) over DEV_REPORT
households, shared across every service, endpoint and both fitted seeds (paired). Every bound is Wald-type: point +/- t(G - 1) x
bootstrap SE (G = DEV_REPORT households); this is not a studentized bootstrap-t.
Decision family (positive direction only): three claims, each an intersection-union test (IUT) whose bound-based clauses are tested at
one-sided level alpha/3 (alpha = 0.05; Bonferroni across the three claims). Hence P(at least one of the three positive claim decisions is
false) <= 0.05 ONE-SIDED (each claim <= 0.0167 one-sided), approximately (Wald clause tests), conditional on the fitted models and readers.
RECIPE_EFFECT = 6 bound-based clauses plus a 24-clause utility POINT-feasibility gate (a deterministic condition on the estimates, with no
error control); its nested decision that also requires the 24 utility clauses to pass at their alpha/3 bounds shares the clause tests
and needs no further alpha. COMPETITIVE_COMPARISON = coalition benefit > 0 vs every eligible required baseline, both seeds, gated by
LOCAL_TREE meeting the same utility requirements. SUBGROUP_SUITABILITY = four guards. The IUT is NOT a simultaneous coverage statement:
per-clause alpha/3 bounds are marginal and can neither establish violations, shortfalls or harm nor be counted after a failed conjunction.
Reporting family: two-sided 95% intervals over the frozen K = 40 primary roster (Bonferroni t), with approximate simultaneous coverage
conditional on the fitted models and readers; every failing-direction label (ESTABLISHED_VIOLATION, ESTABLISHED_SHORTFALL,
ESTABLISHED_SUBGROUP_HARM) is read from these simultaneous intervals only.
Scope: the two seeds share people and are not independent samples; each seed is its own clause, so 'both seeds' is a two-fit
replication, not an estimate of between-fit variance. The household bootstrap treats DEV_REPORT households as a simple random sample
from a household superpopulation (unweighted ACS-sample composition); it is not ACS survey-design variance (no strata, PUMA clustering
or PWGTP1-80 replicate weights). PWGTP-weighted values are point summaries only.
"""
```

If the lead adopts Rc1 (α = 0.025), set `ALPHA = 0.025` and change "0.05 ONE-SIDED (each claim <= 0.0167 one-sided)" to "0.025 ONE-SIDED (each claim <= 0.0083 one-sided)".

### R6. Scope wording in SCIENCE_LOCK: weighted summaries, recommendation logic, estimability

**Findings.**

- `'weighted_secondary': '...; household bootstrap only; not survey-design variance'` reads as though the weighted summaries carry household-bootstrap intervals. They carry none (`pwgtp_weighted_point` and `pwgtp_weighted_auc` are points), and their target population is not stated.
- `'separation'` forbids an unqualified recommendation only "under established subgroup harm". That implicitly allows an unqualified recommendation when the subgroup result is UNRESOLVED, which conflicts with keeping SUBGROUP SUITABILITY as its own conclusion.

**Patch, `lock_science.py`, lines 86 and 88.** Replace both strings and add the two new keys:

```python
                      'separation': 'the three conclusions are reported separately and never merged; an UNQUALIFIED recommendation requires SUBGROUP_QUALIFIED; under SUBGROUP_UNRESOLVED any recommendation must state that subgroup suitability is unresolved; under ESTABLISHED_SUBGROUP_HARM (simultaneous intervals) no recommendation for that group; crossing intervals are unresolved, never established harm',
                      'familywise': 'decision family: one-sided FWER 0.05 over the three positive claim decisions (Bonferroni, alpha/3 per claim; IUT within claims); reporting family: K = 40 two-sided simultaneous 95% intervals; the two families carry separate guarantees and are labelled separately',
                      'scope': 'all intervals and decisions are conditional on the two fitted seed pairs and the frozen readers, in the unweighted DEV_REPORT household superpopulation; they do not quantify between-fit (training) variance or uncertainty about the recipe effect over refits',
                      'weighted_secondary': 'PWGTP-weighted point summaries (ratio estimators) of AUCs and utility, targeting the person-weighted Florida 2019 household-resident cohort analogue for the same fitted models; no interval is attached to weighted values; the household-bootstrap intervals apply only to the unweighted primary estimands; ACS design-based variance (PWGTP1-80 replicate weights, successive-difference replication) is not computed',
```

---

## RECOMMENDED

- **Rc1. Consider α = 0.025 one-sided for the decision family.** This matches the predecessor convention (STATISTICAL_DESIGN_IUT.md §3) and FDA/ICH practice for primary claims. The per-clause t rises from 2.129 to 2.395. At fixed DEV_REPORT size this costs power. For equal power it would need about 1.19 times the households. If α stays at 0.05, R5's explicit statement is mandatory.
- **Rc2. Use Holm instead of Bonferroni across the three claims.** It is uniformly more powerful and still valid under any dependence. Each claim's IUT p-value is the maximum of its clause p-values. Clause p-values are 1 − T_{G−1}((point − margin)/SE) for benefit clauses and 1 − T_{G−1}((margin − point)/SE) for no-excess clauses. Reject the claim with the smallest p-value if p ≤ α/3, the next if p ≤ α/2, and the last if p ≤ α. This is optional and must be fixed before lock if adopted.
- **Rc3. Add a pre-registered sensitivity check on the bounds.** Compute a one-sided percentile bound from the same 2,048 draws (about 34 draws lie beyond the 1.67% quantile) and the skewness of the bootstrap distribution for each roster endpoint. This matters most for the NLL clauses:
  - Per-person NLL differences can reach about ±27.6 nats because of the 1e-12 clip.
  - With a right-skewed excess, the Wald *upper* bound is anti-conservative, which risks a false PASS.
  - Pre-register "primary governs; every disagreement disclosed". The more conservative alternative is that PASS requires both bounds to agree.
- **Rc4. Report bootstrap Monte Carlo error.** The relative MC SD of the SE is 1/√(2(B−1)) ≈ 1.6%. Flag any clause whose decision would flip if its SE changed by ±3%.
- **Rc5. Show that the benefit is not an artefact of winner selection.** ATTACK_VALIDATION is small (cap 3,000 persons), and winner-selection noise can differ between releases. All slate scores are already stored in `DEV_REPORT_SCORES.npz`. Report descriptively:
  - the per-reader DEV_REPORT AUC matrix for LOCAL_TREE and TASK_REFIT;
  - same-reader differences;
  - the slate envelope (the maximum over readers), noting that it is selected on report data.
- **Rc6. Label endpoints outside the roster.** `srec['utility']` for TASK_REFIT, baselines, T_TREE and U0 carries `sim95_*` and status fields, although these endpoints are outside the K = 40 family. Add `'in_primary_family': False` and call them descriptive, or drop `sim95_*` for them.
- **Rc7. Store the roster in structured form.** Record it in SCIENCE_LOCK as `(key, claim, kind, margin)` tuples. The current strings carry annotations, so they cannot be matched to keys. In `report.py`, assert that `set(E)` equals the roster keys minus absent baselines, and read `K_PRIMARY` and the 200-household floor from the lock and `schema.SUPPORT` instead of hard-coding them.
- **Rc8. Use domain degrees of freedom for subgroup endpoints.** Use t with G_g − 1 df, where G_g is the number of households with a member of the group. The numerical effect is negligible at the expected size, but it is the correct reference.
- **Rc9. Add competitive add-ons at no alpha cost.**
  - A nested "material superiority" sub-decision (all eligible baselines, margin 0.020, `iut_pass`). Its rejection region is a subset of the margin-0 claim's, so it needs no extra alpha.
  - Per-recipient AUC differences vs each eligible baseline, reported as descriptive and outside K.
- **Rc10. Report LOCAL_TREE − TASK_REFIT utility differences directly.** These are descriptive, and show whether the coalition benefit was bought with utility loss relative to the matched control.
- **Rc11. Check that service output rows align with DEV_REPORT rows.** `report.py` checks the alignment of `DEV_REPORT_SCORES.npz` with `R['row_id']`, but not of the service outputs. Assert that each output file's `row_id__DEV_REPORT` (written by the workers, for example `stage_a.py` line 96) equals `R['row_id']`.
- **Rc12. Put the scope into the labels themselves.** Rename `RECIPE_EFFECT_ESTABLISHED` to `RECIPE_EFFECT_ESTABLISHED_WITH_POINT_FEASIBLE_UTILITY` and `SUPERIOR_TO_ALL_ELIGIBLE` to `COALITION_SUPERIOR_TO_ALL_ELIGIBLE`. If renamed, update the event text of FORECASTS.json F4 and F5 before the lock. R2 also makes F12 (harm) a simultaneous-interval event, so revise F12's probability before lock if needed.
- **Rc13. Housekeeping.** The `groups` dict (`report.py` line 120) stores full per-person vectors and masks that are never used. Remove it.

## NOTES (checked; no change needed)

- **N1. IUT validity.** For each claim, P(all clause tests reject | some clause null true) ≤ the level of the false clause's test (Berger 1982). Bonferroni over the three claims then gives FWER ≤ α for the three positive decisions under any dependence. Both seeds and both SEX groups enter as separate clauses inside one conjunction, which is valid regardless of their dependence. Validity is approximate, inherited from the Wald clause tests.
- **N2. K = 40 is the correct count.** Per seed there are 1 benefit + 2 worsening + 12 utility + 3 competitive + 2 subgroup = 20 endpoints. Bonferroni-t at α/(2K) is used correctly. It is conservative in two ways:
  - Ucal is one positive global temperature on centred U0 logits, which is argmax-invariant. So "accuracy loss vs U0" and "vs Ucal" are the same statistic: 4 duplicates, 36 distinct endpoints.
  - `crit_simultaneous` keeps K = 40 even if a baseline is absent.
- **N3. Bounds are adequate at this sample size.** For AUC U-statistics with about 4,000 persons per SEX class, paired AUC differences and bounded mean Brier differences are close to normal, so Wald bounds with a cluster-bootstrap SE are adequate (LeDell et al. 2015 report 94–95% coverage at n ≥ 5,000). t versus z is immaterial at G ≈ 5,000. NLL is the only clause type that needs the Rc3 check.
- **N4. No DEV_REPORT-dependent selection.**
  - Reader winners and orientation, including the 14 ignore-one aliases, are chosen on ATTACK_VALIDATION.
  - Comparator configurations and FAGTB validity are fixed by SELECTION_LOCK (MODEL_SELECT).
  - Ucal is fitted on CALIBRATION.
  - LOCAL_TREE (ρ = 1, E20) and TASK_REFIT are fixed.
  - The bootstrap seed and B are fixed.
  - The planted-bit control uses DEV_REPORT SEX only to build the control score, which is correct.
  - The estimability count uses SEX alone (ancillary).
  - One-time opening is enforced by the access log.
- **N5. The subgroup bootstrap is correct.** Resampling all households and restricting the weights to the domain (`W[:, m]`) is standard domain estimation. Mixed-sex households share a draw across both groups, the random domain size enters the ratio denominator, and the covariance between groups is preserved. Subsetting households first would be the wrong approach, and the code does not do that.
- **N6. The estimability floor is reasonable.** 200 households per SEX group is a sensible floor for a bounded mean difference. At 8,000 persons it will almost surely be met, so it acts as a sanity gate. It matches `schema.SUPPORT`.
- **N7. The planning formula itself is correct for one clause** (normal approximation, one-sided α/3, 80% power). R4 extends its coverage and fixes its labelling.
- **N8. Wording already correct.** "Not confirmation", "seeds share people and are not independent samples" and "not ACS survey-design variance" are all stated correctly. No text claims independent seeds or confirmation.
- **N9. Change of primary SE from the predecessor.** The predecessor recommended an influence-function SE as primary with the bootstrap as sensitivity. Here the bootstrap SE is primary. Both are consistent at this sample size, and the choice is acceptable if stated (R5 does this).
