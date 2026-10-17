# Engineering and claims review: PCRL autonomous method-development campaign

The engineering and claims reviewer (a separate agent) wrote this review, about 17:55–18:15Z on 2026-10-17. The subagent wrote no files, so the coordinator saved it from the hand-back:
- the verdict, the checked-and-correct section and must-fix items 1–3 are verbatim;
- items 4–20 and the optional items are condensed by the coordinator; every requested change and number is kept, but the full replacement paragraphs are summarized.

The coordinator's disposition is in `CLAIMS_REVIEW_DISPOSITION.md`.

**Reviewer's scope.** Read-only: no numerical job, no private array or label read, no commit, no edit.

**Census context.**
- Data: public-use 2019 ACS 1-Year PUMS Florida person records, restricted to the replication cohort: housing-unit residents aged 25–64, civilian employed (ESR 1 or 2), WKHP 1–99.
- The service releases an income-class probability vector (PINCP × ADJINC / 10^6 > 50,000) and an occupation-group probability vector (OCCP mapped to 5 civilian groups), each with its argmax decision. Both are computed from AGEP, WKHP, COW, SCHL, MAR and POBP.
- Readers infer the recorded SEX column (1 → 0 men, 2 → 1 women) from each output and from the linked pair.

**Checked against:**
- `report/DEV_REPORT_RESULTS.json` and the five CSV tables;
- `EXPERIMENT_LEDGER.csv`, `PROMOTION.json`, and the FINALIST, ANALYSIS and REPORTING locks;
- `READERS_SUMMARY.json`, the forecasts and their scores;
- `DEPLOYMENT_RESULTS.json` and `BACKUP_VERIFICATION.json`;
- `INDEPENDENT_VERIFICATION.json` (8/8 PASS, D1–D3) and `PROTOCOL_POSTRUN_CLARIFICATIONS.md`;
- the publication refs in git.

`RESEARCH_DECISION.md` was reviewed in its 14:04 working-tree version, which includes the verification block and the §4 within-SEX bullets.

## Verdict

Not ready to publish without the must-fix items below.

The claim labels and nearly all numbers match the record and the verifier. The main problems:
- one wrong table row (H_JJ);
- the 0.0199 point shown as "+0.020" next to the 0.020 target;
- missing lock-mandated qualifiers: J's income-Brier cost, M's NLL differences, and the IDENTICAL/REPLAY flags on the recipient count;
- the description of what the 6-level income output adds to the pair (on every observed role it adds no distinct split of people beyond the occupation output);
- incomplete selection disclosure: the finalist was control-designated and fitted after the interim exclusion, and the EDGE reasons are incomplete;
- an undisclosed PROTOCOL §9 timing deviation behind verifier D1;
- two factual errors in family C;
- a handoff whose reusable wording lacks the Census context and the mandatory qualifiers.

## Checked and correct (no action)

**Numbers.** Seed 0 / seed 1, at the stated rounding, these match DEV_REPORT_RESULTS / CSVs:
- pair AUCs: finalist 0.613/0.618, NEURAL_RAW 0.640/0.638, H_LJ 0.641/0.643, task-only 0.665/0.664, LOCAL-only twin 0.629/0.628, PUB_IBSI 0.630/0.623, T_TREE 0.664/0.664;
- benefit points and clause bounds for B, M and J;
- E counts: 24/4/8; 3 PASS / 25 UNRESOLVED of 28;
- the income-margin table;
- guards, all secondary NLL differences, and PUB_IBSI's guard 0.0051/0.0091;
- control bands: planted 1.0; shuffled 0.499–0.535;
- the §4 within-SEX bullets: +0.0038/−0.0005, +0.0022/+0.0087, +0.016/+0.012, +0.0065/+0.0060;
- MODEL_SELECT bars and family proxies;
- 22 + 22 evaluations, 17 trials and 3 promotable configurations.

**Labels.** E, B, M, J, N and IBSI equal the record and the verifier's recomputation.
- B's nested 0.020 target: seed 1 against NEURAL_RAW is a POINT_SHORTFALL; the others are UNRESOLVED.
- J's first failing condition: JOINT|pair_benefit_vs_B2TWIN|s1, UNRESOLVED.

**Deployment.**
- 7 packages reproduce the evaluated DEV_REPORT wire bitwise through the API and CSV paths, with 14/14 refusals each and no labels read.
- The missing NEURAL_RAW s1 package is disclosed in QUICKSTART.
- Verifier check 8: FINALIST and TASKONLY, 22/22 refusals.

**Publication receipts.**
- `7e306dcfd` was pushed at 17:01:39Z, before the first reference re-measurement (17:02:39Z) and the first fit (17:08:14Z).
- `614c9f36b` was pushed at 17:45:22Z, before the DEV_REPORT read at 17:45:30Z. START_HERE's "pushed before the DEV_REPORT re-read" is true.

**Custody.** Same-device and off-device restore-and-deploy checks are all_ok. The off-device read-back is WARM, and a cold re-check is pending. Coverage runs through `8ef240e88`; the file is untracked. No document claims otherwise.

**Absent from the main text, as required:** guarantee language, independence of the seeds, interval coverage of the search, LOOKUP optimality and novelty.

## Must-fix

### 1. RESEARCH_DECISION §5, H_JJ row: wrong numbers and missing statuses

Current:
> `| H_JJ (the JOINT_TREE parent) | 0.604 / 0.607 | fails / fails | −0.010 / −0.011: H_JJ reads lower, but is utility-ineligible |`

Replacement:
> `| H_JJ (the JOINT_TREE parent) | 0.603 / 0.606 | fails / fails | −0.009 (UNRESOLVED) / −0.011 (BELOW_ZERO), secondary family, COMPARATOR_INELIGIBLE: H_JJ reads lower at the point, but is utility-ineligible |`

Why: the DEV values are 0.60349 / 0.60649 (RECOVERY_TABLE 0.6035 / 0.6065), and SEC|H_JJ_minus_F|pair is −0.00950 (UNRESOLVED) / −0.01136 (BELOW_ZERO). The figures 0.604 / 0.607 are the earlier campaign's published values, or a double rounding.

### 2. RESEARCH_DECISION §5, NEURAL_RAW row: 0.0199 shown as 0.020

Current:
> `| NEURAL_RAW (eligible reference) | 0.640 / 0.638 | pass / pass | +0.027 / +0.020, clause > 0 both seeds |`

Replacement:
> `| NEURAL_RAW (eligible reference) | 0.640 / 0.638 | pass / pass | +0.027 / +0.0199, clause > 0 both seeds (seed 1 below 0.020 at the point) |`

Why: the point is 0.019888, and registered forecast F7 was scored false on exactly this value.

### 3. RESEARCH_DECISION §1, B row: 0.0199, flags, comparator status, replay

Replacement text for the B row:
- the linked-pair benefit (reference AUC − finalist AUC) against NEURAL_RAW is +0.027 / +0.0199, with clause bounds 0.0135 / 0.0059;
- against H_LJ it is +0.028 / +0.025, with clause bounds 0.0191 / 0.0129;
- all 8 recipient non-worsening clauses pass at the clause bound;
- the 2 occupation clauses against H_LJ are IDENTICAL_RELEASE (point = SE = 0), and the 2 against NEURAL_RAW are REPLAY_OF_PUBLISHED, so the 4 income clauses are the only new recipient evidence;
- both references meet all 12 DEV utility clauses at the point on both seeds;
- positive, below the original 0.020 target;
- the occupation component was chosen with knowledge of its published DEV_REPORT results;
- the refitted NEURAL_RAW and H_LJ pair readers reproduce the published reader scores bitwise (16/16 per seed).

Why: required by ANALYSIS_LOCK `mandatory_qualifiers`, DEVELOPMENT_REVIEW F4 and the READERS_SUMMARY `replay_vs_source`.

### 4. RESEARCH_DECISION §1, M row: mandatory qualifiers, alias, step-1 confound

Replacement text for the M row:
- Use the "declared readers" wording.
- Step 1: against the matched task-only control, +0.052 / +0.047, with clause bounds 0.0391 / 0.0336.
- Step 2: against H_LJ, +0.028 / +0.025, read from claim B's keys as an alias. It adds no comparison beyond B.
- The task-only control differs in both outputs: its occupation output alone reads 0.650 / 0.645, against the finalist's inherited 0.560 / 0.583. Step 1 therefore credits the inherited privacy-trained components as a whole, and the new income code's contribution is carried by step 2.
- Qualifiers against the task-only control: +0.019 / +0.015 more occupation NLL (marginal 95% intervals exclude 0) and +0.0050 / +0.0039 more income NLL (UNRESOLVED).
- Qualifiers against H_LJ: +0.0083 / +0.0076 more income NLL (ABOVE_ZERO) and identical occupation NLL.
- The privacy-trained components are the replication's stored Stage B corrections.

### 5. RESEARCH_DECISION §1, J row: the lock's mandatory qualifier is missing

Add, as the cost of the joint component: the finalist spends +0.0027 / +0.0027 more income summed Brier than the twin (marginal 95% [0.0002, 0.0051] / [0.0003, 0.0051]) and +0.0032 / +0.0036 more income NLL (UNRESOLVED). Name the twin as "same coarsening, calibration and occupation output, LOCAL-only statistic". Seed 1 is the first failing condition.

### 6. RESEARCH_DECISION §1: state the conjunctions explicitly

Insert the frozen conjunctions for E, B (with the 0.020 target), M (step 1, then step 2) and J. Also state the α split: 0.025/3 one-sided, clause critical value 2.395, family values 3.304 (52 keys) and 3.025 (20 keys).

### 7. Verifier D1 is clarified, not resolved; disclose the PROTOCOL §9 timing deviation

- **(a) RESEARCH_DECISION, verification block.** State that D1 is clarified, not resolved.
  - The claims draft was written at about 17:37Z, after the reviewer's adjudication (≈17:34Z) and the promotion record (17:35:23Z), so PROTOCOL §9 ("before a finalist is selected") was not met.
  - The record shows only that the conjunctions were hashed and committed before any final reader fit (17:43:02Z) and published before DEV_REPORT was read (pushed 17:45:22Z, read 17:45:30Z).
- **(b) PROTOCOL_POSTRUN_CLARIFICATIONS.md.** Append a "Deviation from PROTOCOL §9" paragraph to the same effect.

### 8. RESEARCH_DECISION scope: Census context and the full exclusion list

- **Census context.** Name the U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records, the income definition (PINCP × ADJINC / 10^6 above $50,000, 2019 dollars), the occupation definition (OCCP mapped to five civilian broad groups) and the cohort. State that SEX is never an input.
- **Intervals.** Say they are household-bootstrap Wald intervals (B = 2048), not ACS design-based (no PWGTP or replicate weights).
- **Exclusions.** Add the fixed household split, the α = 0.5 and m = 6 choices (CALIBRATION screens with a MODEL_SELECT-derived offset), and the earlier campaigns' use of MODEL_SELECT and DEV_REPORT. Note that the finalist's occupation output and the references' DEV results were published beforehand.

### 9. RESEARCH_DECISION "Selection": EDGE reasons and the origin of the finalist

**EDGE reasons.**
- 17 trials were considered.
- All three promotable configurations are EDGE through the shared seed-0 men's guard of 0.0044 (0.88 of the cap), inherited from the JOINT_TREE occupation output.
- B3CTL is also EDGE through its own income Brier on both seeds (0.87–0.88 of the cap).
- Had the shared guard been below 0.8 × cap, the rule would have selected B2.

**Origin of the finalist.**
- The finalist is a control-designated configuration.
- It was fitted at 17:24:25Z and 17:24:52Z as the coarse-only twin of the coupled B3, after B3 failed the seed-0 income Brier clause.
- α = 0.5 was the largest value inside the caps on a CALIBRATION screen (edge-seeking).

**The superseded interim record.**
- The interim exclusion, written at 17:23:49Z, would have excluded the eventual finalist.
- The interim record was never committed, was deleted at 17:33Z, and is reproduced in PROMOTION.json as superseded.
- The adjudication came with B3CTL's MODEL_SELECT results known, and it changed the finalist from B2 to B3CTL.

### 10. Informational redundancy of the 6-level income output

**Facts** (from `per_seed.*.coverage_attack_roles`). The finalist's linked pair has exactly as many distinct vectors as its occupation output on every role and seed:

| Role | Distinct vectors, seed 0 / seed 1 |
|---|---|
| DEV_REPORT | 5,619 / 5,619 |
| ATTACK_VALIDATION | 2,902 / 2,902 |
| ATTACK_FIT | 7,530 / 7,526 |

So, on these people, the income output is a function of the occupation vector. H_LJ is the same, with one extra distinct pair on ATTACK_FIT.

**Required changes.** Rewrite these places so they describe reader-relative differences, never added information:
- (a) RESEARCH_DECISION §3: the pair-minus-occupation gap;
- (b) §8 item 2: "Residual reader-level synergy";
- (c) §1: "at most 6 values; 6 distinct observed";
- (d) QUICKSTART §1;
- (e) RESEARCH_DECISION §2: the B2 twin sentence;
- (f) MECHANISM_CARDS: the B2 twin sentence.

### 11. B1 wording

"Does not carry the income information" contradicts the one-to-one observation. It should say a simple calibrated map (L2 logistic regression on the 4 log-ratios) does not recover income within the caps, which is a limit of the simple map, not an absence of information. Fix this in RESEARCH_DECISION and MECHANISM_CARDS.

### 12. Family C: one factual error and two overstated wordings

- **Every reader drops.** Every pair- and occupation-view proxy reader drops against T_TREE (by 0.011–0.033), but only 5 of 6 income-view readers drop: seed-1 income OUT_HGB rises by 0.011.
- **The temperature control.** Write "moves every reader proxy by at most ±0.0025", not "removes nothing".
- **The size of the reduction.** Occupation recovery falls 0.022 / 0.011 to 0.621 / 0.629, which is 0.039 / 0.033 above the bar. This is lower reader recovery, not information removal.
- **§3 wording.** "Removed, by construction, the SEX-propensity component; the proxy readers' recovery fell by a small amount".
- Fix MECHANISM_CARDS accordingly.

### 13. RESEARCH_DECISION §2, family B lesson

- **Not run:** a continuous-mixture control and the coupled B3 on seed 1.
- **Lesson (MODEL_SELECT selection feedback):**
  - the LOCAL-only coarse code read 0.007 / 0.008 below H_LJ;
  - the partly pair-aware code read a further 0.015 / 0.015 below;
  - coarsening and mixing are not separated at the mixed statistic;
  - J is not established on DEV.

### 14. RESEARCH_DECISION §7 and §8.1

- **The benefit over H_LJ.** Write "positive at the clause level on both seeds and below the original 0.020 target, for the declared readers". Coarsening and the JOINT half of the statistic are not separated.
- **Coupling.** "Not the active ingredient", rather than "contributes nothing".
- **Income utility.** Cite 0.0049 / 0.0049 and seed-1 accuracy loss 0.0096.
  - The JOINT-only code (α = 1) exceeded the Brier cap on CALIBRATION and was not run.
  - The coupled B3 exceeded it on MODEL_SELECT seed 0 with a pair proxy 0.0018 above the finalist's, so it was not more protective.

### 15. RESEARCH_DECISION §1, "Strongest supported conclusion": name the view

The comparison is between linked pairs, and the two seeds are the same people. From the occupation output alone, recovery is identical to H_LJ's.

### 16. START_HERE claims block and caveats

- **Claims B, E, M and J.** Restate them with the same numbers and qualifiers as items 3–5:
  - 0.0199;
  - the clause bounds;
  - IDENTICAL/REPLAY;
  - the E counts and edge values;
  - the step-1 confound and step-2 alias;
  - J's numbers and its income Brier cost.
- **Caveats to add:**
  - the selection was adaptive and edge-seeking;
  - the finalist was control-designated and promoted under the adjudication;
  - the conjunctions were written after the selection but before any reader fit;
  - the two seeds are the same people;
  - no distinct split is added by the income output.

### 17. MANUSCRIPT_CLAIMS_HANDOFF: reusable wording and qualifiers

- **Reusable quote.** Replace it with a version carrying:
  - the full Census context;
  - the 8-reader slate and 8,001-person auxiliary sample;
  - points, clause bounds and the 0.020 language;
  - a qualifier list with the E counts, both references' 12-clause status, CONFIDENCE_UNRESOLVED and the adaptive/edge/control-designated selection;
  - REPLAY/IDENTICAL, and the no-distinct-split qualifier.
- **Coarse income code (item 2).** Say that coarsening and mixing are not separated; add the income NLL cost against H_LJ.
- **Coupling and family C bullets.** Fix them as in items 11–12.
- **"Do not write" list.** Add: information-removal or "added information" language for the pair.

### 18. Investigator B role files

Keep the verbatim text and append a dated reviewer note to `REPORT.md` and `CARD_REVISIONS.md`:
- the "at most log 6" bound is vacuous for m ≥ 2 and should not be cited;
- the income output adds no distinct split beyond the occupation output on observed roles;
- the uncoupled matched task-only control was later run by the coordinator (ledger 43–44);
- the continuous-mixture control and the coupled B3 on seed 1 were never run.

### 19. START_HERE: the figure's interval bars are unlabelled

The right panel's bars are marginal 95% intervals of each AUC; the claims rest on paired differences, not on overlap. Open markers flag only failed DEV utility point clauses; PUB_IBSI also fails the men's guard (0.0051 / 0.0091).

### 20. Forward references to files that do not yet exist (publication gate)

These do not exist yet:
- `roles/verifier/`
- `roles/claims_reviewer/`
- `RESOURCE_ACCOUNTING.json`
- `HANDOFF.json`
- `PUBLICATION_RECORD.json`

Create each one, or mark it "(pending)", before publishing. START_HERE's resource statement must cite `RESOURCE_ACCOUNTING.json`.

## Optional

- **O1.** Narrow "requires JOINT_TREE-level occupation protection" to family A's observation.
- **O2.** The A2 bullet: add its seed-1 pair proxy, 0.6296, which is above both references.
- **O3.** The IBSI row: its ineligibility was designated on MODEL_SELECT at the freeze (seed-1 men's guard 0.00501; occupation screen). Add the reader asymmetry: about 470 distinct pairs.
- **O4.** Retitle the §4 diagnostics heading, or move the overall income Brier item.
- **O5.** Disclose the registered forecasts (E 0.50, B 0.25–0.30, M_TO 0.45, J 0.30; mean Brier 0.167).
- **O6.** Add a custody line to START_HERE.
- **O7.** QUICKSTART:
  - add the D3 threshold-tie note to "(deterministic)";
  - note that re-scoring is a logged MODEL_SELECT use;
  - note that the frozen references' refusals come from the strict input wrapper.
- **O8.** Give the receipt times in START_HERE.
- **O9.** The T_TREE row: give +0.051 / +0.046 (point difference, descriptive).
- **O10.** §8.3: write "it bound families A and C".
