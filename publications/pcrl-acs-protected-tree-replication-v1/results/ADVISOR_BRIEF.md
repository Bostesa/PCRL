# Advisor brief: does the protected-tree recipe transfer to ACS?

**Answer:** relative to matched continued task fitting, the local recipe lowered each recipient's own SEX recovery, but lowered the linked coalition's recovery by less than the pre-registered 0.020 margin.

**Setting:** Florida, ACS 2019 1-Year. This is a custom two-task benchmark of civilian workers aged 25–64:
- income above $50,000 in 2019 dollars;
- five official occupation groups.

SEX is the protected attribute. All results are development evidence on DEV_REPORT, which was opened once for analysis. The independent verifier later re-read it only to recompute the reported numbers (`LOCK_NOTES.md`). The sealed 20% of households stayed closed (see question 6).

**Pre-registered decisions:**

| Claim | Decision |
|---|---|
| Recipe effect | `RECIPE_EFFECT_NOT_ESTABLISHED` |
| Competitive comparison | `NOT_SUPERIOR_TO_ALL_ELIGIBLE` (complete) |
| Subgroup usefulness | `SUBGROUP_UNRESOLVED` |

**1. Did the stronger teacher create headroom in ACS?**
Yes, a modest amount, at the point; there is no interval. On MODEL_SELECT, the CatBoost teacher had lower NLL and Brier and higher accuracy than the equally exposed neural reference U0, for both tasks and both seeds:
- income NLL lower by about 0.007;
- occupation NLL lower by 0.008–0.012;
- accuracy up to +0.6 pp.

Balanced accuracy was mixed. The pre-registered gate is a different test: whether the teacher beats constant prediction, with class support passing. It passed (SUPPORTED_CONTINUE).

**2. Did the fixed local recipe keep utility and reduce recovery?**
- **Utility:** kept at the point. All 24 utility clauses pass at the point; this gate has no error control. In the simultaneous intervals 14 pass, 10 are unresolved and none is an established violation.
- **Each recipient's own SEX recovery:** fell. Against matched continued task fitting, AUC fell by 0.014–0.018 for income and 0.061–0.067 for occupation.
- **Coalition recovery:** this is what a party holding both recipients' outputs could infer. It fell only 0.007 and 0.0175, against the 0.020 target. On seed 0 the shortfall is established; on seed 1 it is unresolved.

**3. Did it beat continued task fitting and the closest published tree approach?**
- **Continued task fitting:** no, not by the required margin.
- **Eligible neural adversarial control:** no. At feasible utility, that control had lower coalition recovery, by 0.017 and 0.011 AUC at the point. This is established on seed 0 and unresolved on seed 1.
- **FAGTB** (the published adversarial tree booster; faithful binary port plus a documented multiclass adaptation): it had no utility-feasible λ on MODEL_SELECT, so it was ineligible under the frozen rule and is reported descriptively. On DEV_REPORT it failed every utility clause. Its coalition recovery was about the same as the recipe's (benefit −0.009 / +0.007).
- **Official FARE:** ineligible (no MODEL_SELECT-feasible setting), so it is reported descriptively. It removed more coalition signal than its no-fairness twin (0.640 / 0.633 vs 0.667 / 0.666). But both FARE and its twin fail every utility clause, so most of the cost is shared with compression.

**4. Was usefulness kept within both SEX groups?**
- **Women:** the simultaneous interval passes on seed 1 and is unresolved on seed 0 (upper bound 0.0058). The point excesses are −0.0013 and −0.0038, better than U0.
- **Men:** unresolved. The occupation Brier excess is +0.0002 and +0.0025.
  - The one-sided α/3 upper bounds (0.0051 and 0.0071) exceed the +0.005 guard, so the group is not qualified.
  - The K = 40 simultaneous intervals ([−0.0065, +0.0068] and [−0.0038, +0.0087]) cross the guard, so harm is not established either.

So neither harm nor suitability is established. Any recommendation must say this.

**5. What is established, unresolved, unsupported or incomplete?**
- **Established** (K = 40 simultaneous intervals):
  - recipient-level recovery reductions against continued task fitting (all four intervals lie wholly below 0);
  - on seed 0, a shortfall of the coalition benefit against 0.020;
  - on seed 0, a competitive shortfall against the neural control;
  - on seed 1, a positive coalition reduction; whether it reaches the 0.020 margin is unresolved.
- **Observed at the point, with no error control:**
  - teacher headroom over U0 on MODEL_SELECT;
  - all 24 LOCAL_TREE utility clauses (14 PASS and 10 unresolved in the simultaneous intervals).
- **Unresolved:**
  - usefulness for men (both seeds) and for women (seed 0);
  - the seed-1 coalition margin;
  - the seed-1 competitive comparison;
  - ten utility intervals.
- **Unsupported:** nothing; the benchmark is supported.
- **Incomplete:** nothing required. FAGTB's occupation model is an adaptation, labelled as such.

**The mechanism, descriptive.** The local objective makes each output less revealing on its own, but the pair of outputs still carries most of the teacher's SEX signal (91–96% of the teacher's above-chance coalition AUC). A diagnostic that adds the pair term (JOINT_TREE) cut coalition recovery to about 0.605. It was not nominated, and it failed a few income utility clauses.

**6. What runnable model exists, and is a later sealed assessment justified?**
- **Runnable models:** deployment packages map raw ACS fields to probabilities and decisions for:
  - the teacher;
  - LOCAL_TREE (both seeds);
  - the neural control.

  They were verified without labels. All 743 private files were restored byte-identically, once from the same-device backup and once from a separate copy on an external drive. A label-free LOCAL_TREE redeploy from the drive restore matched exactly (`BACKUP_VERIFICATION.json`). The immediate on-drive read-back may have been served partly from the file cache, so a cold re-verification command is recorded there.
- **Sealed assessment: not justified.**
  - At the development values, no sample size can establish the recipe and competitive claims (FUTILE).
  - Planning treats the development points as the true values for the same frozen models and readers. The seed-1 coalition interval [+0.006, +0.029] still includes values above 0.020.
  - Opening the sealed households would spend Florida 2019 to confirm a negative.
- **Sealed access disclosure (D2).** No model, file or summary used the sealed households beyond counts and a key-set hash. But the extraction's single eligibility pass parsed their records in memory, including both task labels, before roles were assigned, and then discarded them.

**Verification.** An independent verifier with separate code reproduced every decision and every result value it recomputed from the stored outputs; it only spot-checked the numbers in the documents. Ten of 11 checks pass. The failing check is a code-labelling defect with no effect on behaviour, disclosed as D1. A claims review corrected the document wording (`CLAIMS_REVIEW.md`).

**Recommended next step:** close this recipe-replication question as above. If the programme continues, start a newly registered development study of coalition-aware objectives, beginning from the pair term. Don't run a sweep here.
