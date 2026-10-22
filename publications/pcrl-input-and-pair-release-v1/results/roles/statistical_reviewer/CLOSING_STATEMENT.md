<!-- Saved verbatim by the coordinator from the statistical reviewer's hand-back (agent a2f38dcd3b9c3ee00, resumed at 04:53:48Z with
roles/coordinator/CLOSING_REVIEW_BRIEF.md), received 2026-10-22 shortly after 05:09Z. The reviewer was asked to return its statement
rather than write a file; the statement below is unedited apart from the harness's two-space line indentation, which is removed. It
was preceded by this cover note (verbatim): "The closing review is done. Every decision label follows from the frozen rules and the
records; I changed nothing in the repository, committed nothing, and read no label. The document drafts need corrections C1–C19, the
most important being C2: the claim that IA's "leakage moved to the income view" is not supported. Please save the text below
verbatim, with attribution, as `roles/statistical_reviewer/CLOSING_STATEMENT.md`, as was done for the pre-registration report." -->

# Closing statement of the statistical reviewer (campaign `pcrl_input_and_pair_release_v1`, rules IPR1-rules-v1)

**Status.**
- Writing started 2026-10-22T04:53:55Z and was completed 05:09:12Z (both read from `date -u`).
- Author: the statistical reviewer who wrote `RULES.md`, `RULES.json`, `ipr/rules.py`, `ipr/test_rules.py` and `REPORT_PREREGISTRATION.md`.
- I read no label file, nothing under `.private/` and no reader cache. I ran no real-data computation, changed no file and committed nothing.
- My only computations were invented-data checks:
  - `python3 -m ipr.test_rules`: 28 of 28 PASS, 0.64 CPU-s.
  - A closing-check script in my scratch directory, 0.10 CPU-s.
  - Opening the JSON records for reading, under 2 CPU-s in all (not timed individually).

**Inputs read** (records and code only):
- `PROTOCOL.md` (registered), `PROTOCOL_AMENDMENT_1.md`, `PROTOCOL_DEVIATIONS.md` items 1-10.
- The cards `IA_SFG_v1`, `IB_CSF_v1`, `CTL_TWIN_v1`, `CTL_PATHS_v1` and `IA_OSG_R1`.
- `roles/challenger/REVIEW_DESIGNS.md`, `HANDBACK.md` and `CHECK_SUPPORT.json`; `roles/coordinator/FS_SWAP_DECISION.md` and `COORDINATOR_LOG.md`.
- `claims/{IA_REG,IB_REG,IA_R1}/DECISIONS.json`, `claims/*_D2R/D2R.json`, `claims/superseded/`.
- `claims/PLAN_*.json`, `STRONGEST.json`, `BESIDE_IPR1_MAIN.json`, `BESIDE_IPR1_R1.json`, `CAMPAIGN_SUMMARY.json`, `paths/SHARED_ROLES.json`.
- `ia/FIT_V0`, `ia/FIT_V1`, `ia/FIT_R1_V0`, `ia/FIT_R1_V1`, `ib/FIT_V0`, `ib/FIT_V1`; `dev/UTILITY_LEDGER.csv`.
- `TABLES.md` (its data-rich reader section is the one I used; I did not open `rich/` separately).
- `roles/verifier/VERIFIER_REPORT.md` and, read only for a first look, the verifier's addendum records `A5_DECISIONS` (05:06:41Z). The addendum report itself is not saved yet.
- The coordinator's drafts as saved at 04:52-04:53Z: `RESEARCH_DECISION.md`, `START_HERE.md`, `ADVISOR_BRIEF.md`, `HANDOFF.md`, `VERIFICATION.md`.
- `ipr/claims.py` (how the `enhanced` entries are built) and `ipr/readerextra.py` (enhanced-winner selection).

**Sign conventions.** As registered:
- Pair benefit = comparator AUC − candidate AUC on the 8 registered cells, reported as mean (se_agg); positive favours the candidate.
- "Resolved" = mean − 2 se_agg > 0 with both version means > 0. "Resolved negative" is the mirror image.

## 1. Do the decision labels follow from the frozen rules and the records?

**Yes, for every reading.** All three DECISIONS.json records, the three D2R records, the strongest-release record and the campaign summary follow from `ipr/rules.py` applied to the inputs the claims builder passed.
- **Rules unchanged.** `RULES.json` (4c235046...), `ipr/rules.py` (4ef5fa35...) and `ipr/test_rules.py` (a5e11220...) were last committed in the registration commit adfe3e75a and are unchanged; every DECISIONS.json carries `rules_sha256` 4c235046....
- **Verifier replay, registered readings.** The verifier replayed IA_REG and IB_REG exactly with separate code (checks 1.1-5.4).
- **Verifier replay, later readings.** Its addendum records show its own replay of IA_R1's D1 and D2 (|Δ mean| = |Δ se| = 0) and of the three D2R readings matching the recorded labels. Until the addendum report is saved, documents must cite the report for these, not my reading of its records.

### 1.1 Inputs checked per reading

Times are UTC. "Cost" = pooled MODEL_SELECT paired total-NLL cost of CAND − point.

**IA_REG — registered.**
- *Registration:* card first commit 02:38:59Z; first MODEL_SELECT read 02:50:32Z.
- *Declarations passed:* equal to the card: `operation_sex_blind` false; `ro` {GENERIC_ZTILT, release-only false}; `joint` null.
- *Paths:*
  - SBC [IA_SBC_HI]. IA_SBC_LO is version 0 only and is excluded (deviation 2).
  - SPEND [IA_SPD_HI = CTL_DQ_0p177156].
  - RO [IA_RO_HI = CTL_RO_0p172614 / 0p208862].
  - The SPEND and RO lower points are TWIN. `alias_of` is empty.
- *Q\* and cost:* SBC_HI −0.0050; SPD_HI −0.0013; RO_HI −0.0017. TWIN is the only cheaper point (+0.0002) and is not resolved negative.
- *Fineness:* NOT_APPLICABLE, because CAND's MODEL_FIT spend s is −0.0001 / −0.0002.
- *Grid:* effective; 4800 checks; 0 equalities; strict NRM1 effective; registered reader and bootstrap seeds.
- *Labels:* D1 TASK_FEASIBLE; D2 NOT_SEPARATED (no comparison passes); D3 NOT_SUPERIOR; D4 NOT_MET; D5 NOT_APPLICABLE_NOT_JOINT; D6 NOT_CLAIMED; class TASK_ONLY_FEASIBLE_NO_PROTECTION.
- *Flags:* CAND_STRENGTH_DIFFERS_BY_VERSION (rungs 1 / 2), RO_GENERIC_ZTILT, READER_SENSITIVE_ENHANCED_IA_RO_HI / _IA_SPD_HI.

**IB_REG — registered.**
- *Registration:* same commit and read times as IA_REG.
- *Declarations passed:* equal to the card: `operation_sex_blind` true; `ro` null; SBC and RO paths null and not recorded as "not run"; `joint` {jointly designed, both outputs, collapsed income_first / occupation_first = JOINT / JOINT, separate IB_IND}.
- *Paths:* SPEND [IB_SPD_LO, IB_SPD_HI].
- *Q\* and cost:* IB_SPD_HI −0.0037. The cheaper points TWIN (+0.0104) and IB_SPD_LO (+0.0015) are not resolved negative.
- *Fineness:* FINE in both versions, checked by hand. Version 0: lower 0.0102 ≥ 0.75 s = 0.0081; upper 0.0149 ≤ s + 2 SE + 0.25 s = 0.0166. Version 1 likewise.
- *Grid:* effective; 4800 checks; 0 equalities.
- *Labels:* D2 NOT_SEPARATED, gating TWIN and SPEND only; D5 NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL; class TASK_ONLY_FEASIBLE_NO_PROTECTION.
- *Flags:* SEX_BLIND_OPERATION, SBC_DEGENERATE_BY_CONSTRUCTION, RO_DEGENERATE_BY_CONSTRUCTION, READER_SENSITIVE_ENHANCED_IB_SPD_HI.

**IA_R1 — ADAPTIVE_R1.**
- *Tag:* card first commit 04:11:44Z, after the first MODEL_SELECT read; revision 1. Every label carries the `ADAPTIVE_R1__` prefix; `registered` is false.
- *Declarations passed:* equal to card IA_OSG_R1: `operation_sex_blind` false; `ro` {GENERIC_ZTILT, release-only false}; `joint` null.
- *Paths:* SBC, SPEND and RO each [LO, HI], every point present in both versions.
- *Q\* and cost:* SBC_HI −0.0061; RO_HI −0.0008. There is no SPEND bracket: CAND costs +0.0043 / +0.0019 more than SPD_LO / SPD_HI, which gives NO_BRACKET_SPEND. No cheaper point is resolved negative (RO_LO: −0.0051, se 0.0060).
- *Fineness:* SBC COARSE_BOTH / COARSE_BELOW; SPEND COARSE_BELOW / FINE; RO FINE / FINE. This gives CONTROL_GRID_COARSE_SBC and _SPEND (checked by hand from the plan's recorded inputs).
- *Grid:* effective; 4608 checks; 0 equalities.
- *Labels:* D1 TASK_FEASIBLE with reserve RESERVE_ONE_SEED_ONLY, which gives the flag POINT_FEASIBLE_NOT_RESERVE_QUALIFIED; D2 NOT_SEPARATED; D3 NOT_SUPERIOR; D4 NOT_MET; D5 NOT_APPLICABLE_NOT_JOINT; D6 NOT_CLAIMED; class TASK_ONLY_FEASIBLE_NO_PROTECTION.
- *Flags:* CAND_STRENGTH_DIFFERS_BY_VERSION (rungs 2 / 4), NO_BRACKET_SPEND, CONTROL_GRID_COARSE_SBC / _SPEND, RO_GENERIC_ZTILT, READER_SENSITIVE_ENHANCED_IA_OSG_RO_HI.

**Other records.**
- **D2R:**
  - IA_REG_D2R (release IA_SFG; decisive TWIN plus the three HI brackets), IB_REG_D2R (IB_CSF; TWIN, IB_SPD_HI) and IA_R1_D2R (IA_OSG; TWIN, SBC_HI, RO_HI) are all ROBUSTNESS_NOT_CONFIRMED.
  - All refits are REFIT_GENUINELY_EFFECTIVE, with both seed pairs judged separately and 0 equalities.
  - In IA_R1_D2R the SPEND bracket is passed as an explicit null (no bracket), which the rules read as "no comparator". It is not "unrecorded", so the reading is not INCOMPLETE.
- **Strongest release:**
  - IA: IA_OSG (+0.0017078) over IA_SFG (+0.0016147), a difference of 0.0001.
  - The direct beside comparison IA_OSG vs IA_SFG is +0.0000931, equal to that difference, so the twin's cells are identical across banks IPR1_MAIN and IPR1_R1.
  - IB: IB_CSF.
- **Campaign summary:** K = 3 D2 looks (2 registered, 1 ADAPTIVE_R1); D2 met in 0.

### 1.2 Points settled by invented-data checks of the frozen rules

1. **A control point saved in one version only (IA_SBC_LO).**
   - The frozen rules pool every path point over both versions. A path point with costs in one version only makes D2 INCOMPLETE ("paired costs ... missing or malformed").
   - So leaving IA_SBC_LO out of the path (the plan writer's both-versions rule, deviation 2) was the only way to obtain any reading. It is a gap the frozen rules do not address explicitly.
   - Leaving it out removes only the cheaper-point check (c3) in version 0. When the bracket fails, adding a cheaper point, even one resolved less recoverable, leaves the label unchanged (checked).
   - SBC_FRONTIER fails at its bracket, and the verifier's own numbers show IA_SBC_LO would not be resolved less recoverable (O1). No label depends on it.
2. **The superseded early D2R records (deviation 9).**
   - `refit_effectiveness` returns REFIT_GENUINELY_EFFECTIVE when the robustness seed checks cover version 1 only. The frozen check needs every seed-dependent recipe checked, but not every version × repetition block.
   - That is why the early records carry REFIT_GENUINELY_EFFECTIVE although version 0's cells were missing.
   - Their D2R read NOT_RUN, so no wrong label arose. The reruns contain every block (8448 seed checks per seed pair for IA_REG).
3. **The campaign-summary sentence** is the frozen constant: "a D2 look is an intersection-union conjunction of four such comparisons (TWIN, SBC, SPEND, RO)". It prints the same words for a SEX-blind look. IB's look has two gating comparisons (TWIN, SPEND). Documents add that (C7).

### 1.3 Record observations that change no label

1. **IA_BLIND version 1 is bitwise IA_SFG version 1** (verifier O2). The targeting ablation is degenerate in version 1, so IA_SFG vs IA_BLIND (+0.0008, se 0.0060) rests on version 0 (+0.0017).
2. **The enhanced winner adds nothing here.** In every targeting entry of the three readings, the enhanced (TABLE_CV) winner's version means equal the non-table winner's exactly. So READER_SENSITIVE_ENHANCED_* repeats the targeting failures; it is not a separate reader-sensitivity result in this campaign (C16).
3. **IA_OSG_RO_HI fails G1 in version 1** (men's occupation guard at 1.005 of its cap). The rules gate G1 only for CAND and TWIN, so the RO comparison stands. But the point that "the tilt is less recoverable" is made against a release that is infeasible in one version (C5).
4. **IA_REG_D2R is not IA's family D2R.** It was computed while IA_SFG was IA's only operation configuration. Under RULES.md 8, IA's family D2R is IA_R1_D2R, because the strongest release moved to IA_OSG. IA_REG_D2R is an extra, descriptive reading of the registered candidate (C6).

### 1.4 Interpretation caveats inherited by the labels

- **IA_REG's controls.** IA_SFG spends almost nothing (MODEL_FIT −0.0001 / −0.0002; MODEL_SELECT +0.0004 / +0.0001 nats), so its SPEND and RO brackets are near-identity points (DQ step 0.177; tilt 0.17 / 0.21). Its SBC bracket, a whole permuted-ladder field step, spends 0.0050 nats more than CAND on MODEL_SELECT. That is a design limit of matching a near-zero-spend candidate with discrete steps. It favours the control and changes no label.
- **IB's comparison with the twin is version-1-dominated:** version means 0.0037 / 0.0102.
- **IA_OSG's spend grew from MODEL_FIT to MODEL_SELECT** (version 1 occupation NLL 0.0070 to 0.0120 nats). This explains both NO_BRACKET_SPEND and the missed reserve in version 1. MODEL_FIT selection is a selection device, not inference (RULES.md 11).
- **Realized se_agg was 0.0053-0.0081** for the decisive comparisons, smaller than the planned 0.009-0.018: the candidates are close to the twin, so paired differences are small. The per-comparison 2-SE threshold was therefore about 0.011-0.016 (section 4.8).

## 2. Required headline (to be quoted verbatim)

> This is development evidence on previously and adaptively used MODEL_SELECT records of the admitted ACS 2019 Florida cohort (8,001
> people), for the declared readers only: it is not confirmation, and lower recovery for these readers is not protection against every
> reader. Two independently designed release families, an input-side SEX-ordered field generalization (IA_SFG) and a SEX-blind
> coordinated keyed collapse of both outputs to decision-cell means (IB_CSF), met the original utility limits at the point, but neither
> reduced the readers' recovery of the recorded SEX attribute beyond its matched controls: D2 was met in 0 of 2 registered family
> readings and in 0 of 1 non-registered (ADAPTIVE_R1) reading, 3 D2 looks in total; one AGG_POSITIVE comparison passes a true null in
> about 0.07-2% of replicates under six moderate NRM1 known laws and about 8% when one version or split dominates, and with 3 looks the
> chance that at least one passes by chance is at most 3 times the per-look rate. Registered readings (linked-pair benefit = comparator
> AUC - candidate AUC, mean (se_agg) over the 8 registered cells): IA_SFG generalized almost nothing inside its utility budget (one
> marital-status merge, plus five-year age bands in one version) and is NOT_SEPARATED: +0.0016 (0.0058) over the equally trained twin,
> and +0.0033 (0.0056), +0.0006 (0.0053) and -0.0009 (0.0058) over its SEX-independent-target, SEX-blind-spending and generic
> release-only-tilt brackets. IB_CSF is NOT_SEPARATED: +0.0069 (0.0057) over the twin and +0.0007 (0.0081) over its SEX-blind
> spending bracket; as a SEX-blind operation it admits no SEX-targeting claim. Both are TASK_ONLY_FEASIBLE_NO_PROTECTION, and neither
> registered release's benefits are confirmed on refit cells (D2R ROBUSTNESS_NOT_CONFIRMED with genuinely effective refits). Beside
> them, never as a family result: IA's adaptive revision IA_OSG (occupation-only generalization, the income recipient keeping the
> twin's output) is ADAPTIVE_R1__NOT_SEPARATED; it lowered the occupation view (0.6351 against 0.6443) but not the linked pair (+0.0017
> (0.0062) over the twin), and at the point it is more recoverable than its occupation-restricted release-only tilt (-0.0072 (0.0064),
> unresolved, against a bracket point that fails the utility limits in one version). No reading is competitive with the nine inherited
> references (D3 NOT_SUPERIOR) and none meets the 0.020 material criterion (D4 NOT_MET): the candidates are 0.016-0.028 more
> recoverable than NEURAL_RAW and H_LJ. The +0.0016 to +0.0069 differences against the twin are smaller, unresolved results against a
> weaker baseline; at the 2-SE level the data exclude pair benefits over the twin above about 0.013 (IA_SFG), 0.018 (IB_CSF) and 0.014
> (IA_OSG). No release is a method, candidate or component; joint design is not claimed (IB's sequential orders equal its joint design
> by construction) and novelty is not claimed.

## 3. Required corrections to the coordinator's drafts

**C1. Headline.**
- Replace the `RESEARCH_DECISION.md` "## Headline" placeholder with section 2 above, verbatim.
- START_HERE, ADVISOR_BRIEF and HANDOFF keep pointing to it, and their short summaries must not contradict it (C2-C6 apply to them).

**C2. The income-view claim is not supported (the most important correction).**

Where it appears:
- `RESEARCH_DECISION.md` question 2, IA_REG bullet ("Its income view became more recoverable ... leakage moved to the income recipient").
- `RESEARCH_DECISION.md` Adverse ("Its refitted income model leaked more").
- `START_HERE.md` IA_SFG bullet; `ADVISOR_BRIEF.md` question 2; `HANDOFF.md` "What must not be said", IA bullet.
- `VERIFICATION.md` O4.
- PROTOCOL_DEVIATIONS item 8 (add the qualification there).
- The closing brief's own framing ("leakage moved to the income view").

Replace with:
> "IA_SFG's income-view winner AUC is 0.5600 against the twin's 0.5498, and its income recipient gate failed at the point against the
> twin, its SBC bracket and its release-only bracket. This rests on one repetition's cells (S1R1: +0.038 against the twin) in which the
> validation-selected income reader switched. In that cell the verifier's independent LR and HGB readers give +0.001 and +0.0002
> (verifier O4); the data-rich income readers give -0.0010 / +0.0003 in the two versions; the descriptive MODEL_FIT income readers give
> 0.5579 / 0.5560 against the twin's 0.5581 / 0.5571; and the SEX-blind dithered quantization of the twin (IA_SPD_HI), which removes
> information, reads 0.5580 on the same grid. This is a reader-selection sensitivity of the weak income view, not evidence that the
> refitted income model leaks more; no income-view statement is resolved."

Also:
- Card IA_OSG_R1 cannot be edited. Wherever its diagnosis is summarized, say that its failure F2 is not supported by these records and that the revision's rationale rests on F1 and F3 (C9).
- `VERIFICATION.md` "O4 ... does not matter" becomes "changes no label; it does qualify the income-view interpretation (C2)".

**C3. "Is spending" and "equals spending."**
- Where: RESEARCH_DECISION Favourable ("it equals what SEX-blind quantization buys at matched spend") and the Adverse heading "IB's gain is spending"; ADVISOR_BRIEF question 2 ("its gain is what spending buys").
- Replace with: "IB_CSF's gain over the twin is not separated from SEX-blind dithered quantization at matched spend (+0.0007, se 0.0081, against its SPEND bracket)."
- "Not separated" is never "equal" and never "is spending".

**C4. Unresolved contrasts are not "nothing".**
- "Coordination buys nothing measurable" (RESEARCH_DECISION question 6, ADVISOR_BRIEF question 6) becomes: "the coordination contrast (CSF against independent per-recipient keys, IND) is +0.0016 (se 0.0066), not resolved; coordination effects up to about 0.015 are not excluded."
- "Restricting the generalization to the occupation recipient bought nothing on the pair" becomes "made no detectable difference on the pair (+0.0001, se 0.0061, against IA_SFG)".
- "Its targeting and coordination ablations showed no gain" (RESEARCH_DECISION and COORDINATOR_LOG) becomes "showed no resolved gain".

**C5. IA_OSG against its own controls.**
- Where: RESEARCH_DECISION Adverse "IA_OSG is weaker than its own controls" and its RO bullets (questions 2 and 3); START_HERE; ADVISOR_BRIEF question 2; HANDOFF.
- Replace with:
  > "IA_OSG is not separated from any of its controls. At the point it is more recoverable than its occupation-restricted release-only
  > tilt: -0.0072 (0.0064) against the upper bracket IA_OSG_RO_HI, which fails G1 in version 1 (men's occupation Brier guard at 1.005
  > of its cap; the rules gate G1 only for the candidate and the twin), and -0.0051 (0.0060) against the cheaper, task-feasible
  > IA_OSG_RO_LO; on the refit cells -0.0057 (0.0066). None is resolved."
- For SPEND, state:
  > "No bracket on MODEL_SELECT: IA_OSG spent more than both occupation-only quantization points because its spend grew from MODEL_FIT to
  > MODEL_SELECT (version 1: 0.0070 to 0.0120 occupation nats); both points are at the point at least as hard to read (-0.0008, se
  > 0.0058, each)."

**C6. D2R per family.**
- Where: RESEARCH_DECISION D2R row and question 3; HANDOFF "Records to cite".
- State: "Under RULES.md 8 the family D2R readings are IA_R1_D2R (IA's strongest feasible nonzero release is IA_OSG) and IB_REG_D2R. IA_REG_D2R was computed while IA_SFG was IA's only operation configuration; it is an extra, descriptive reading of the registered candidate. All three are ROBUSTNESS_NOT_CONFIRMED with genuinely effective refits."
- Add: "IA_OSG was selected by 0.0001 against se_agg about 0.006, a tie in substance; the rule's tie tolerance is 1e-12, so the choice is mechanical and carries the winner's curse."

**C7. The campaign-summary sentence.**
- Wherever the frozen statement is quoted, add after it: "(IB's SEX-blind look has two gating comparisons, TWIN and SPEND; 'four such comparisons' is the frozen IPR1 wording)".

**C8. The one-version control point IA_SBC_LO** (RESEARCH_DECISION questions 1 and 2; deviation 2). Add:
> "The frozen rules pool each path point over both versions; a point recorded in one version only makes D2 INCOMPLETE (reviewer's
> invented-data check), so leaving IA_SBC_LO out of the path was the only way to obtain a reading. It removes only the cheaper-point
> check (c3) in version 0; SBC_FRONTIER fails at its bracket with or without it, and the verifier's numbers show it would not be resolved
> less recoverable (O1). As a saved configuration without utility evaluation or reader cells it breaches PROTOCOL 4's equal-evaluation
> clause (disclosed); no label depends on it."

**C9. IA's revision: numbering and trigger** (RESEARCH_DECISION "What was not run", ADVISOR_BRIEF "IA's revision", PROTOCOL_DEVIATIONS item 8). State:
> "IA_OSG fills IA's first revision slot (PROTOCOL 7 allows two), hence the tag ADAPTIVE_R1. It is not the slack-weighted ladder that
> IA_SFG_v1 declared as its R1, whose MODEL_FIT trigger failed in version 0 (45% against 50% slack); the declared R2 could not change
> the release. IA_OSG was designed after, and from, the registered MODEL_SELECT reading, so it is adaptive as a whole; its own card was
> committed before its fit (04:11:44Z). Its diagnosis rests on F1 (MODEL_FIT: income binds the shared ladder at MAR1) and F3 (the twin's
> pair recovery sits in its occupation view); F2 is not supported (C2)."

**C10. Precision and the meaning of NOT_SEPARATED.**
- Where: RESEARCH_DECISION "Untested questions: Precision" and the first Adverse bullet ("far from the resolution of the grid").
- Replace with:
  > "Realized se_agg for the decisive comparisons was 0.0053-0.0081, below the planned 0.009-0.018 because the candidates are close to
  > the twin, so the per-comparison 2-SE threshold was about 0.011-0.016. At the 2-SE level the data exclude pair benefits over the twin
  > above about 0.013 (IA_SFG), 0.018 (IB_CSF) and 0.014 (IA_OSG). In a normal approximation, a true benefit of 0.020 against every gating
  > control would have passed all of D2's comparisons with probability about 0.83 for IA's four and 0.66 for IB's two (reviewer's
  > invented-data calculation); effects below about 0.01 were not resolvable."

**C11. Reference wording.**
- Where: RESEARCH_DECISION question 4 ("Below every candidate: six references have a lower pair AUC") and Adverse ("All candidates are below the strong references on the pair").
- Replace with: "Six references are less recoverable (lower pair AUC) than every candidate. All three candidates are more recoverable than NEURAL_RAW and H_LJ, by 0.016-0.028, resolved in five of the six comparisons (not IB_CSF against NEURAL_RAW: -0.0158, se 0.0095)."

**C12. IB's binding constraint** (RESEARCH_DECISION "What was not run"; COORDINATOR_LOG).
- At π = 0.10, income NLL (1.22× / 1.001× of S3) and the men's occupation guard (1.26× / 1.17×) both fail, and in version 0 the women's guard too (1.18×) (verifier check 1.5).
- Do not write "the men's guard" alone.

**C13. The substituted release-only path** (RESEARCH_DECISION "Untested questions"). Replace "so it would nearly equal the operation itself" with:
> "It was not built (the declared GENERIC_ZTILT fallback, chosen before the fit because the twin's released pair identifies the six-field
> record for about 93% of MODEL_FIT people). The descriptive substituted ladder equals SFG's up to k* in version 1 but starts with a
> different step in version 0 (AGE5 against MAR2), so the premise held in one version only (deviation 3). IA's RO comparisons answer
> only the generic-tilt question and are not an information bound."

**C14. Registered bets.**
- RESEARCH_DECISION "The registered bets": add that IA_SFG's predicted RECIPIENT_GATE_FAILED_VS_RO occurred through the one-repetition income-view sensitivity (C2), not through the predicted mechanism.
- Add that IB_CSF_v1's predicted binding component (the men's guard) bound only jointly with income NLL.
- "The structural prediction held" and ADVISOR_BRIEF's "The results agree" become "the readings are consistent with the challenger's prediction". It was a qualitative prediction, not a test.

**C15. VERIFICATION.md.**
- O4: as in C2.
- "Not checked by the verifier": complete it from the addendum.
- No document may say the verifier replayed IA_R1, the D2R readings or STRONGEST until the addendum report is saved. "The verifier replayed every registered decision exactly" is correct as written.

**C16. The enhanced-winner flags** (wherever READER_SENSITIVE_ENHANCED_* is listed). Add:
> "In every targeting entry of the three readings the enhanced (TABLE_CV) winner equals the non-table winner, so these reported-only
> flags repeat the targeting failures and are not a separate reader-sensitivity result."

**C17. Deviation 9.** Add:
> "The superseded records' refit-effectiveness field reads REFIT_GENUINELY_EFFECTIVE although version 0's robustness cells were missing:
> the frozen check requires every seed-dependent recipe to be checked, not every version x repetition block (reviewer's invented-data
> check). Their D2R read NOT_RUN and they are never read; the reruns contain every block."

**C18. The previous campaign's release** (ADVISOR_BRIEF question 4; RESEARCH_DECISION question 4). After "DDR_PTC_R1 ... resolved less recoverable than IA_SFG and IA_OSG" add:
> "(beside: DB_PTC_R1 is the dual-design campaign's adaptive revision, chosen with a winner's curse on the same people and never
> separated from its own release-only control)."

**C19. IA_REG's control geometry.**
- State beside the IA_REG numbers: the SPEND and RO brackets are near-identity points, and the SBC bracket spends 0.0050 nats more than the candidate on MODEL_SELECT, because the permuted ladder's first step that reaches CAND's spend + 2 SE is a whole field step.
- This favours the control in the SBC comparison and changes no label (section 1.4).

## 4. What the evidence does and does not support

### 4.1 The input-side reading (IA_SFG)

**Supported.**
- Under S3 the SEX-ordered ladder stops at its first informative step: k* = MAR2 in version 0 and AGE5 + MAR2 in version 1.
- The next rung (marital status suppressed) fails S3 only on income: accuracy 1.95× / 1.69× and NLL 1.07× / 1.29× of the budget. Every occupation component stays at or below 0.63 / 0.49 of its budget.
- The candidate is a near-identity release:
  - MODEL_FIT out-of-fold spend −0.0001 / −0.0002 nats;
  - MODEL_SELECT +0.0004 / +0.0001;
  - pair AUC 0.6464 against the twin's 0.6480.
- It is NOT_SEPARATED from every control, and its benefit over the twin is bounded above by about 0.013 at 2 SE. The refit cells agree in direction (+0.0018, se 0.0055) without confirmation.
- The registered reading is therefore a closure test of SFG at S3, not a test of input-side generalization in general.

**Not supported.**
- That leakage moved to the income view (C2).
- Any statement about information removal at larger strengths: the MAR1, WK3 and COW3 rungs were never selected, and the data-processing ceiling has no bite at the feasible rung.
- Any claim that SEX targeting matters: the targeting ablation is +0.0008 (0.0060) and degenerate in version 1.

### 4.2 IA's adaptive revision IA_OSG

It is never IA's registered result, which stays IA_REG (NOT_SEPARATED).

**Trigger and diagnosis.**
- The coordinator diagnosed the registered reading at 03:25-03:35Z, after the verifier's report, and committed the card with its code at 04:11:44Z, before the fit. The reading is adaptive as a whole.
- **F1 is supported** by MODEL_FIT records: income binds the shared ladder.
- **F3 is supported:** the twin's pair 0.6480 is carried by its occupation view, 0.6443 against income 0.5498.
- **F2 is not supported** (C2).

**Numbering:** C9.

**Are its controls matched?** In the sense that matters, yes:
- every control shares the twin, the MODEL_FIT people, folds and seeds;
- SBC is the same occupation-only ladder and models with IA_SFG_v1's fixed permutation;
- SPEND and RO change only the occupation output, as the candidate does, so the income vector, which carries the pair link, is identical across candidate and controls (ALIAS_ZERO on income).

Three limits:
- The restriction to the occupation output is a choice made after the registered outcome (deviation 8), not a registered construction.
- The SPEND path did not bracket on MODEL_SELECT, and the SBC and SPEND grids are coarse.
- The RO upper bracket is task-infeasible in version 1.

None of these could have turned the reading into "met". They limit how adverse the comparisons are.

**What it shows.**
- The occupation view fell by 0.0092 at the point, as the card predicted (0.003-0.012). The data-rich occupation readers agree (+0.011 / +0.017).
- The linked pair moved by only +0.0017 (0.0062) against the twin and +0.0001 (0.0061) against IA_SFG. That is consistent with the pair reader recovering through the unchanged income vector (FAILURE_MAP note 2), stated as a reading of point estimates, not as a resolved effect.
- It is not separated from its controls and is point-worse than its release-only tilt (C5).
- D2R is not confirmed (refit cells: +0.0028 vs the twin, −0.0010 vs SBC, −0.0057 vs RO).
- It is not reserve-qualified in version 1.

### 4.3 The SEX-blind reading of IB

**The declaration held.**
- The verifier rebuilt IB_CSF and IB_IND bitwise from SEX-free inputs (check 4.6).
- SEX entered only through the MODEL_FIT strength guards. Those guards did fix π* = 0.075, together with income NLL, but as a utility constraint, not an input of the operation.

**What IB can answer:** whether keyed coordinated collapse of about 7% of people (561 / 605 at MODEL_SELECT) lowers recovery more than SEX-blind dithered quantization at matched spend.
- Not separated: +0.0007 (0.0081), upper 2-SE bound about 0.017.
- On the refit cells: −0.0006 (0.0069).
- Its +0.0069 against the twin is point only and version-1-dominated.

**What it cannot answer:** SEX targeting (there is no SEX-dependent fitted input), or SEX information beyond the twin's release (the release is a function of the twin's release and a SEX-independent key).

The targeting ablation CSF-T reads −0.0002 (0.0079). Targeting added nothing resolvable, consistent with the challenger's redirect.

### 4.4 IA's declared RO fallback

**Admissible.**
- RULES.md 3.2 allows a declared GENERIC_ZTILT fallback, and it was declared in the card before the first fit (verifier 4.8).
- The D2 scope sentence is the fallback's.
- Its premise held in version 1 only (C13).

**Supported:** at about matched total-NLL spend, the generic release-only tilt of the twin is at the point at least as hard to read as IA_SFG (−0.0009, se 0.0058). Its occupation-restricted form is at the point at least as hard to read as IA_OSG (C5). Neither is resolved.

**Not supported:** any statement that IA's gain needs, or does not need, SEX information beyond the twin's release.
- The substituted path was not built.
- Because the twin's released pair identifies the record for about 93% of MODEL_FIT people, a release-only control's power to explain a deterministic operation is a matter of learner capacity, not information (challenger finding 18).

### 4.5 IB's D5 collapse declaration

- The card declared, before its first fit and with its construction argument, that both sequential orders equal the joint design.
- The verifier built both orders and reproduced IB_CSF bitwise (561/561, 605/605 flattened people).
- So NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL follows mechanically. It means the coordinated collapse is itself a sequential design in either order, so no advantage of joint over sequential design can exist. It is not a negative finding about coordination.
- The only joint-versus-separate contrast is the descriptive IND ablation: +0.0016 (0.0066), unresolved (C4).
- The challenger had recommended declaring the joint design not built. The declaration is more informative and is correct given the verifier's rebuild.
- No joint-design claim is made.

### 4.6 What was not run

**FS_SWAP, the replacement family.**
- Decided at 02:45Z, before any MODEL_SELECT read, on Designer B's label-free plug-in: no better than occupation-only flattening or CSF at equal spend; predicted NOT_SEPARATED at 0.74.
- It is a planning decision, not a result, and no D2 look was spent on it.
- It was the only design aimed at the pair link itself, so the super-additivity failure remains untested here. IA_OSG's point estimates are consistent with that failure but do not test a link-breaking operation.

**IB's declared revisions.**
- Their triggers were checked on MODEL_FIT records at 02:47Z, before any grid output:
  - R1 needs income alone to bind, but the men's guard binds too;
  - R2 needs π* ≤ 0.025, but π* is 0.075.
- That decision is mechanical. The coordinator's further reasons (the ablations; the guard) are allocation judgments that use MODEL_SELECT outcomes, not evidence that a revision would fail.

**IA's declared revisions:** C9.

**Budget:** 39 of 48 configurations were used.

### 4.7 Novelty

- D6 is NOT_CLAIMED in every reading.
- The cards cite the antecedents: field generalization and suppression, the privacy funnel, microaggregation and local suppression, PRAM and randomized response, coordinated suppression.
- No literature search was run, and nothing in the evidence bears on novelty.

### 4.8 Resolution reached

- Realized se_agg 0.0053-0.0081 put the per-comparison threshold at about 0.011-0.016.
- The NOT_SEPARATED readings make a uniform benefit of 0.020 or more against every gating control unlikely. In the invented normal approximation, D2's power at 0.020 was about 0.83 for IA and 0.66 for IB; at 0.015, 0.47 and 0.39.
- They say little about effects below about 0.01.
- The upper 2-SE bounds against the twin are 0.013 / 0.018 / 0.014.
- These are approximate, model-based bounds on adaptively used records, with the NRM1 known-law caveat.

### 4.9 Registered expectations against outcomes (descriptive)

**Coordinator's protocol predictions:**
- P(CAND G1-feasible) 0.8: all three candidates were feasible.
- P(D2 met for at least one family) 0.08: met in none.
- P(D3) 0.03 and P(D4) 0.01: neither met.
- The predicted most-likely outcome (0.005-0.02 over the twin, not separated from a control) fits IB (+0.0069). IA (+0.0016) fell below that range.

**Card predictions:**
- Every observed pair benefit fell inside its registered interval.
- Misses: π* (0.075 against 0.10-0.15 at probability 0.6); IA_SFG's k* = 1 in both versions; the substituted-ladder equality (held in version 1 only); IA_OSG's SPEND bracket.

**Overall:** no release is a method, candidate or component. Both registered releases are TASK_ONLY_FEASIBLE_NO_PROTECTION.

## 5. Completion record

- Completed 2026-10-22T05:09:12Z (`date -u`, read immediately before this statement was returned). Started 04:53:55Z.
- Frozen rules unchanged since the registration commit adfe3e75a; `python3 -m ipr.test_rules` 28 of 28 PASS.
- **Invented-data checks:** 0.64 + 0.10 CPU-s measured; record inspection under about 2 CPU-s in all. No label was read, no real-data computation was run, no repository file was changed and nothing was committed.
