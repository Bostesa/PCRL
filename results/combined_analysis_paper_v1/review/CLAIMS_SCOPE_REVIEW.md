# Claims and scope review

**Role:** claims and literature reviewer. Read-only on the runner's files; written 2026-10-03.

**Packages reviewed:**
- the output diagnosis, `results/combined_output_diagnosis_v1/` at dc6a2e96;
- the output-aware removal study, `combined_output_aware_removal_v1/`;
- the matched benchmark, `combined_matched_removal_benchmark_v1/`;
- the reconciliation, `combined_evidence_reconciliation_v1/`;
- the runner's new `PROTOCOL_USEFUL_HEADS.md`, spot-checked.

**Rule.** Numbers were checked against the committed endpoint arrays:
- PRIMARY_ENDPOINTS.csv, S3_ENDPOINTS.csv, S4_ENDPOINTS.csv and S5_ENDPOINTS.csv;
- FROZEN_HEAD_UTILITY.csv, OUTPUT_SURFACE_RESULTS.csv, BANK_SELECTIONS.csv and PRIVATE_BACKUP_INDEX.json.

No verdict, threshold or lock is changed by this review. Every correction below is wording, scope or interpretation.
Where a number is touched, the stored value is unchanged and the prose is aligned to it.

**Line numbers** refer to the files at dc6a2e96.

**Columns in the tables:**
- **No.?** Does the correction change a reported number?
- **Dec.?** Does it change a registered decision?
- **Interp.?** Does it change the interpretation?

## Summary of findings

1. **FARE vs compression.**
   - No file uses the literal phrases "not significant" or "no difference".
   - Seven passages imply that the fairness term contributed nothing measurable beyond compression: "did almost as
     well", "just as well", "almost exactly as well", "matched by plain compression", "no contribution … was
     established", a bare "Not established" and "Not shown".
   - The stored contrast is R(FZ) − R(FARE) = **+0.0073721**, adjusted interval **[0.0018880, 0.0128561]**, target
     0.02 (S5_ENDPOINTS.csv row 2). The interval excludes zero.
   - Even the pre-repair value (AMENDMENT_S1: lower bound 0.0001) excluded zero.
   - Correct reading: *most of the reduction was reproduced by compression; FARE had a smaller additional improvement,
     below the registered 0.02 target.*
2. **"Constant head" means a constant decision, not constant scores.**
   - The HMDA fair_lending frozen head predicts class 1 for all 4,764 assessment rows on all three seeds.
   - Its scores vary: centred logits recover race at AUC 0.646 and sex at 0.655, while its task AUC is only
     0.515–0.516.
   - Its offset effect FC is **not** trivially zero: +0.0017 [0.0006, 0.0028] for race and +0.0023 [0.0008, 0.0038]
     for sex. Both intervals exclude zero, below target.
   - Five files describe the head as "constant" in a way that implies constant scores. One calls its FC result
     "trivial".
3. **The offset benefit was not seen on every seed.**
   - "Offset-attributable on every seed" means that an offset-using attacker was *selected* on every seed. It does
     not mean the offset helped on every seed.
   - On HMDA seed 1 (collapsed encoder) the full-bank candidates have equal validation loss (0.8362 vs 0.8373), and
     RESEARCH_DECISION §2 reports no change.
   - The benefit appeared on 5 of 6 encoder seeds.
   - Two summary files drop this qualification.
4. **"Pretrained" backbone.**
   - The word does not occur in any of the four latest packages, except where the reconciliation corrects it.
   - The source of the false description is the old NeurIPS manuscript (PDF(23) l.133–135 and Table 4; see
     `results/combined_empirical_preparation_v1/notes/methodology_pcrl_encoder/ARCHITECTURE.md` §1.2).
   - The stored-model studies use **Round-4** checkpoints. I checked Round-4 code at b96c412: it builds the backbone
     the same way.
     - `experiments/run_v2_dataset.py:161` calls `torch.manual_seed(seed)`.
     - `:178–180` builds a random `StandardEncoder([128,128]→64, dropout 0.3)`.
     - `pcrl/models/lora.py:176` freezes it.
     - The only `load_state_dict` (`:245`) reloads the run's own best.pt.
   - Correct description: *a frozen, seeded, randomly initialised MLP backbone, with per-purpose LoRA adapters and
     task heads trained on top.*
5. **Retired R²-to-accuracy guarantee.**
   - No statement in the four latest packages relies on it.
   - It is still live on public `origin/main` = 55e4cb1d1:
     - `pcrl/purposes/verification.py:208` `certified_accuracy_bound`;
     - used in `experiments/deployment_case_study.py:81,255` and `experiments/run_distribution_shift.py:68,371`.
   - The fix `origin/fix/retire-accuracy-guarantee` @ 5d4eda046 is **not** an ancestor of origin/main (checked with
     `git merge-base --is-ancestor`, 2026-10-03).
   - The manuscript must neither use the bound nor say the public API was fixed. At most it can say a repair branch
     exists and is unmerged.
   - No package claims it was fixed. The reconciliation states it correctly (CORRECTIONS item E; UPDATED_MEETING_BRIEF
     l.130; REBUTTAL_WORK_RECONCILIATION l.200–204).
6. **"22-to-8" and "26-to-19"** are **not** final-checkpoint vs best-validation counts. See section E.
7. **Minor issues:**
   - the backup file count;
   - "recoding of the occupation column";
   - the coalition "every contract" claim, which holds on the mean, not on every seed;
   - "at no accuracy cost";
   - a negative FC interval on Adult employment/race that should be read as an attacker-selection effect.

## A. The FARE correction

**Source values** (S5_ENDPOINTS.csv; FARE_USEFUL_TASK_RESULTS.md l.64–68):

| Quantity | Value |
|---|---|
| Complete-contract recovery: untreated | 0.731 |
| Complete-contract recovery: target LEACE | 0.713 |
| Complete-contract recovery: zero-fairness twin FZ | 0.569 |
| Complete-contract recovery: FARE | 0.562 |
| R(LEACE) − R(FARE) | 0.1512 [0.1373, 0.1651], PASS |
| R(FZ) − R(FARE) | 0.0074 [0.0019, 0.0129], target 0.02, NOT_ESTABLISHED |
| Acc(FARE) − Acc(untreated) | −0.0004 [−0.0025, 0.0016], PASS |

So FZ reproduces 0.1438 of FARE's 0.1512 advantage over LEACE (about 95 %), and FARE's remaining improvement is
0.0074.

| File | Original wording | Corrected wording | No.? | Dec.? | Interp.? |
|---|---|---|---|---|---|
| output_diagnosis/RESEARCH_DECISION.md l.86, l.88 | "## 5. Did FARE contribute beyond compression while keeping a useful task?" / "**Not established** (stage 5; Adult employment/age_group, chosen by a validation-only screen; …)." | "**A material contribution (≥ 0.02 AUC) beyond compression was not established.** Most of the reduction was reproduced by compression; FARE had a smaller additional improvement, +0.0074 AUC [0.0019, 0.0129], below the registered 0.02 target (stage 5; Adult employment/age_group, …)." | No | No (stays NOT_ESTABLISHED) | **Yes** |
| output_diagnosis/RESEARCH_DECISION.md l.91–92 | "But the zero-fairness tree with the same budget did almost as well: 0.569, a difference of +0.007 (lower bound 0.002, NOT_ESTABLISHED)." | "The zero-fairness tree with the same budget reproduced most of the reduction (0.569 vs FARE 0.562). FARE's additional improvement was +0.0074 AUC, adjusted interval [0.0019, 0.0129]: it excludes zero but is below the registered 0.02 target (NOT_ESTABLISHED for materiality)." | No (states the upper end) | No | **Yes** |
| output_diagnosis/RESEARCH_DECISION.md l.110 | "On the useful-task cell, FARE's advantage over LEACE is matched by plain compression." | "On the useful-task cell, most of FARE's advantage over LEACE (about 0.144 of 0.151 AUC) is reproduced by a same-budget tree with no fairness term; FARE's remaining improvement, 0.0074 [0.0019, 0.0129], is below the 0.02 target." | No | No | **Yes** |
| output_diagnosis/FARE_USEFUL_TASK_RESULTS.md l.71–73 | "But **the zero-fairness compression control does almost exactly as well**: no contribution of the fairness term beyond compression was established. The task's structure lets a 5-cell tree keep it while discarding most age information whatever γ is." | "Most of the reduction was reproduced by the same-budget zero-fairness tree (0.569 vs 0.562). FARE had a smaller additional improvement, +0.0074 AUC [0.0019, 0.0129], below the registered 0.02 target, so a material contribution of the fairness term was not established. In the feasible settings (1–2; 5–11 cells) the task's structure lets a small tree keep it while discarding most age information; settings 3–6 were infeasible, and γ ≥ 0.864 gives a single cell." | No | No | **Yes** (also "whatever γ is" is false: high γ collapses to one cell, S5_ADMISSION_NOTE) |
| output_diagnosis/ADVISOR_BRIEF.md l.33–36 | "5. **Does FARE help beyond compression on a useful task?** — **Not shown.** … — But the same tree without its fairness term does just as well (0.57)." | "5. **Does FARE help beyond compression on a useful task?** — **Only slightly, below our target.** … The same tree without its fairness term reached 0.57 against FARE's 0.56: most of the reduction is compression. FARE's additional improvement, +0.007 AUC [0.002, 0.013], excludes zero but is below the 0.02 target." | No | No | **Yes** |
| output_diagnosis/ADVISOR_BRIEF.md l.47 | "FARE's advantage here is matched by plain compression." | "Most of FARE's advantage here is reproduced by plain compression; the remaining +0.007 AUC is below the 0.02 target." | No | No | **Yes** |
| output_diagnosis/ADVISOR_BRIEF.md l.34–35 | "…it lowers recovery far below LEACE (0.56 vs 0.71) at no accuracy cost." | "…(0.56 vs 0.71) with accuracy within the 1-point margin (−0.0004, [−0.0025, 0.0016])." | No | No | Minor |
| output_diagnosis/PAPER_ADDENDUM.md l.50–51 | "…but a matched zero-fairness tree of the same budget achieved nearly the same reduction." | "…but a matched zero-fairness tree of the same budget reproduced most of the reduction (0.569 vs 0.562); FARE's additional improvement, 0.007 AUC [0.002, 0.013], was below our registered 0.02 target." | No | No | Yes (adds the remainder) |
| output_diagnosis/HANDOFF.json l.107 | `"fare_beyond_compression": "not established (FZ - FARE +0.007, lower 0.002); FARE beats LEACE by 0.151 at no accuracy cost; certificate unavailable/vacuous"` | `"fare_beyond_compression": "material (>=0.02 AUC) contribution NOT_ESTABLISHED; FZ - FARE = +0.0074 [0.0019, 0.0129] excludes zero but is below target; most of the reduction reproduced by compression; FARE below LEACE by 0.151 with accuracy -0.0004 [-0.0025, 0.0016] (within 1-pt margin); certificate unavailable/vacuous"` | No | No | **Yes** |
| output_diagnosis/AMENDMENT_S1_2026-10-03.md l.20 | "R(FZ) − R(FARE) \| 0.0076 / 0.0001 \| 0.0074 / 0.0019 \| NOT_ESTABLISHED (unchanged)" | Keep. Add: "Both the original and repaired lower bounds exceed 0; the NOT_ESTABLISHED verdict concerns the 0.02 materiality target only." | No | No | Yes |

**Do not write**, anywhere:
- "no (statistically) significant difference";
- "FARE adds nothing beyond compression";
- "just as well";
- "matched by".

The S5 task is easy (accuracy 0.974 vs untreated 0.975, constant 0.277). No single input column determines it. It
should not be presented as representative of difficult prediction.

## B. "Constant head" vs constant decision

**Source:** FROZEN_HEAD_UTILITY.csv (fair_lending, all seeds):
- `is_constant_prediction=True`, `prediction_frequencies=[0.0, 1.0]`;
- task_auc 0.5153 / 0.5146 / 0.5165.

**Source:** OUTPUT_SURFACE_RESULTS.csv `R|fair_lending_audit|race|FH|centred` = 0.646 and `…|sex|FH|centred` = 0.655.

| File | Original wording | Corrected wording | No.? | Dec.? | Interp.? |
|---|---|---|---|---|---|
| output_diagnosis/RESEARCH_DECISION.md l.30–31 | "The **HMDA fair_lending frozen head is exactly constant on every seed**: its gain is 0." | "The **HMDA fair_lending frozen head predicts the same class for every assessment row on every seed** (a constant decision; gain 0). Its scores are not constant: attackers recover race and sex from its centred logits at AUC 0.646 / 0.655, while its task AUC is only 0.515." | No | No | **Yes** |
| output_diagnosis/RESEARCH_DECISION.md l.63–64 | "HMDA fair_lending is trivial, because its head is constant." | "On HMDA fair_lending the offset adds little: +0.0017 [0.0006, 0.0028] (race) and +0.0023 [0.0008, 0.0038] (sex), excluding zero but below the 0.02 target. This is not trivial by construction: the head's scores vary and reveal race/sex at 0.646 / 0.655." | No | No | **Yes** |
| output_diagnosis/RESEARCH_DECISION.md l.66 | "CH: PASS on 14/14, but on HMDA fair_lending only because its decision is a constant." | Correct as written (decision-level). Keep. | — | — | — |
| output_diagnosis/RESEARCH_DECISION.md l.26–27 | "On seed 1 it is essentially a constant predictor." (HMDA underwriting) | "On seed 1 its decision is nearly constant: it predicts denial for 0.9 % of rows (denial recall 6.7 %), a gain of 0.57 points." | No | No | Minor |
| output_diagnosis/SCORE_DECOMPOSITION.md l.121 | "HMDA pricing × sex, and HMDA fair_lending × {race, sex}: that head is constant (see below)." | "HMDA pricing × sex (+0.0057 [−0.0004, 0.0119]); HMDA fair_lending × {race, sex} (+0.0017 / +0.0023, intervals excluding zero, below target). The fair_lending head's decision is constant (see below), but its scores are not." | No | No | **Yes** |
| output_diagnosis/SCORE_DECOMPOSITION.md l.132 | "(0.000: a constant head)" | "(0.000: a constant decision)" | No | No | Yes |
| output_diagnosis/ADVISOR_BRIEF.md l.14 | "**HMDA fair_lending:** the frozen head is a constant: it always predicts the majority class." | "**HMDA fair_lending:** the frozen head's *decision* is constant: it always predicts the majority class. Its scores still vary and reveal race at AUC 0.65." | No | No | **Yes** |
| output_diagnosis/FARE_USEFUL_TASK_RESULTS.md l.38 | "HMDA fair_lending (gain 0: a constant head)." | "HMDA fair_lending (gain 0: a constant decision)." | No | No | Yes |
| output_diagnosis/VALIDATION.md l.42–43 | "The HMDA fair_lending head is constant on all seeds, and its hard-surface attackers score exactly 0.5." | "The HMDA fair_lending head's decision is constant on all seeds (scores vary), and its hard-surface attackers score exactly 0.5." | No | No | Yes |
| output_diagnosis/HANDOFF.json l.103 | "…HMDA fair_lending frozen head constant" | "…HMDA fair_lending frozen head predicts one class for all rows (constant decision; scores non-constant, race AUC 0.646 from centred logits)" | No | No | Yes |
| output_diagnosis/RESEARCH_DECISION.md l.145; ADVISOR_BRIEF.md l.62; HANDOFF.json l.133 | "near-constant HMDA heads" | "HMDA heads whose decisions are constant or nearly constant" | No | No | Minor |

## C. Seed coverage of the offset benefit

**Source:** RESEARCH_DECISION.md l.50–51: all 3 Adult seeds; HMDA s0 and s2 show drops of 0.06 and 0.14; s1 shows
no change.

**Source:** BANK_SELECTIONS.csv, HMDA s1 FH: full-bank validation log losses 0.8362 (full, dc) vs 0.8373 / 0.8362
(centred, prob).

| File | Original wording | Corrected wording | No.? | Dec.? | Interp.? |
|---|---|---|---|---|---|
| output_diagnosis/PAPER_ADDENDUM.md l.37–38 | "…lowered measured recovery by 0.07 AUC in both primary cells (pre-registered; offset-attributable on every seed)." | "…lowered mean measured recovery by 0.072 (Adult) and 0.067 (HMDA) AUC (pre-registered; an offset-using attacker was selected on every seed). The reduction appeared on all three Adult seeds and on HMDA seeds 0 and 2; on the collapsed HMDA seed 1 there was none." | No | No | **Yes** |
| output_diagnosis/ADVISOR_BRIEF.md l.21–22 | "Removing it lowers measured recovery by **0.07 AUC** in both primary cells … The result passed its pre-registered test and is attributable to the offset on every seed." | "Removing it lowers mean measured recovery by **0.07 AUC** in both primary cells … It passed its pre-registered test; the reduction appears on 5 of 6 encoder seeds (none on the collapsed HMDA seed 1)." | No | No | **Yes** |
| output_diagnosis/SCORE_DECOMPOSITION.md l.69 | "Both passes are **offset-attributable**: an offset-using candidate was selected on every seed." | Keep, and add: "Selection is not benefit: on HMDA seed 1 the selected offset-using candidate matches the ignore-offset candidates and recovery is unchanged." | No | No | Yes |
| output_diagnosis/SCORE_DECOMPOSITION.md l.90–92 | "…Withholding it lowers measured recovery by 0.07 AUC in both primary cells…" | "…lowers mean measured recovery by about 0.07 AUC in both primary cells (no change on HMDA seed 1)…" | No | No | Yes |
| output_diagnosis/RESEARCH_DECISION.md l.48 | "**FC is offset-attributable:** an offset-using candidate was selected on every seed." | Acceptable because l.50–51 qualify it immediately. Keep the two lines adjacent in any excerpt. | — | — | — |
| output_diagnosis/RESEARCH_DECISION.md l.103–104 | "…lowers … by about 0.07 AUC on both primary cells, every seed where the head is not collapsed…" | Acceptable. More precise: "…on all three Adult seeds and HMDA seeds 0 and 2…" | No | No | Minor |
| output_diagnosis/HANDOFF.json l.104 | "offset-attributable every seed" | "offset-using attacker selected every seed; reduction on 5/6 encoder seeds (none on HMDA s1)" | No | No | Yes |
| output_aware/OUTPUT_SURFACES.md l.30–31 | "That offset adds 0.07 AUC of measured recovery without changing any decision." | "That offset adds about 0.07 AUC of mean measured recovery (not on every seed) without changing any decision." | No | No | Minor |

**Related wording:**
- RESEARCH_DECISION.md l.49 says the offset alone "is as revealing as the margin: 0.704 / 0.737". On HMDA the
  offset alone (0.737) exceeds the margin (0.701). Write: "about as revealing as the margin on Adult (0.704 vs
  0.708) and more revealing on HMDA (0.737 vs 0.701)."
- The S3 counterexample needs care. Adult employment × race has FC = −0.0172 [−0.0244, −0.0099]. The interval
  excludes zero on the *negative* side: the validation-selected full-bank attacker scored lower on assessment than
  the ignore-offset bank. This is an attacker-selection effect, not evidence that the offset protects. Write it that
  way if it is quoted; the word "counterexample" (RESEARCH_DECISION l.63) stays correct.

## D. Encoder description, retired guarantee, public API

| Item | Where | Wording to use | No.? | Dec.? | Interp.? |
|---|---|---|---|---|---|
| Backbone | Old NeurIPS manuscript PDF(23) l.133–135, Table 4 ("pretrained on the union of allowed tasks"). Not in the latest packages. | "Each encoder is a frozen, seeded, randomly initialised MLP ([128,128]→64, BatchNorm held in eval mode, dropout 0.3) with per-purpose LoRA adapters and task heads trained on top. The stored-model studies use the Round-4 checkpoints (seeds 0–2), not the Round-5/7 NeurIPS headline checkpoints." | No | No | **Yes** |
| "Frozen PCRL encoders" | All four packages | Acceptable, meaning "not retrained in this study". The first use in the paper should give the full description above. | — | — | Minor |
| R²→accuracy bound | origin/main 55e4cb1d1 (code); no package relies on it | Never cite it. If mentioned: "an earlier R²-to-accuracy bound is invalid (20-row counterexample: R² = 0, accuracy 0.9 vs bound 0.5) and is not used." | No | No | Yes |
| Public API | — | "A repair that retires the bound exists on an unmerged branch; the public main branch still contains it." Never write "fixed on main". | No | No | Yes |

## E. "22-to-8" and "26-to-19" are different comparisons, not one correction

**Where they come from:**
- `combined_evidence_reconciliation_v1/REBUTTAL_WORK_RECONCILIATION.md` l.76–77, l.93, l.296;
- `UPDATED_MEETING_BRIEF.md` l.77, l.92;
- `CORRECTIONS_TO_PREVIOUS_ASSESSMENT.md` item 6 (l.46–48);
- `EVIDENCE_INVENTORY.csv` rows R-10, R-11;
- `VERIFICATION_REPORT.json` (the "unified protocol" item).

These are cross-purpose concatenation-attack flag counts out of 33 cells:

| Model | Checkpoint | ABS (absolute recovery) | INCR (additional recovery over the strongest single release) |
|---|---|---|---|
| Submitted PCRL (R5 Adult/HMDA, R7 Diabetes) | final.pt | 26/33 | 22/33 |
| Rebuttal model: erase-layer union-LEACE + h_concat constraint, τ = 0.10, 200 epochs | best.pt | 19/33 | 8/33 |

**Each arrow changes three things at once:**
- the model;
- the checkpoint rule (final.pt vs best.pt);
- the criterion that was later chosen as the headline. Neither criterion was registered, and the switch was post hoc.

Under ABS, mean leakage on the nonlinear cells **rose** (19.4 → 24.2 pp on 20 of 22 cells). So these are **not**
final-checkpoint vs best-validation counts of one model.

The genuine same-model checkpoint pairs are different counts:
- strict linear compliance: 56/60 (final iterate) vs 54/60 (best validation);
- "clean": 6/60 vs 5/60;
- the old paper's 7/60 mixes the two rules (UPDATED_MEETING_BRIEF l.51–52; CORRECTIONS items 4–5).

| File | Original wording | Corrected wording | No.? | Dec.? | Interp.? |
|---|---|---|---|---|---|
| reconciliation/UPDATED_MEETING_BRIEF.md l.77 | "**Cross-purpose constraint.** It cuts *incremental* concatenation flags from 22/33 to 8/33." | "**Cross-purpose constraint.** Incremental flags are 22/33 for the submitted model (final.pt) and 8/33 for a different rebuttal model with a union eraser (best.pt). The comparison changes the model, the checkpoint rule and the post-hoc headline criterion together." | No | No | **Yes** |
| reconciliation/UPDATED_MEETING_BRIEF.md l.92 | "(26/33 → 19/33 flags, but the mean leak went up)" | "(absolute flags: 26/33 submitted model at final.pt, 19/33 rebuttal model at best.pt; mean absolute leakage on nonlinear cells rose, 19.4 → 24.2 pp on 20 of 22)" | No | No | Yes |
| reconciliation/REBUTTAL_WORK_RECONCILIATION.md l.93 | "The rebuttal genuinely reduced *concatenation increments*, from 22 to 8 incremental flags." | "Incremental flags were 22/33 on the submitted model (final.pt) and 8/33 on the rebuttal model (best.pt); this is not a like-for-like reduction." | No | No | Yes |

**Rule for the paper:**
- Report the four cells of the table above separately, with model, checkpoint rule and criterion.
- Never merge them into one "corrected" count.
- Keep the 56/54, 6/5 and 7 counts in their own sentence.
- Keep cross-purpose absolute recovery (ABS) separate from additional recovery over the strongest individual release
  (INCR).

## F. Other scope items

| File | Original wording | Corrected wording | No.? | Dec.? | Interp.? |
|---|---|---|---|---|---|
| output_diagnosis/RESEARCH_DECISION.md l.82 | "In every contract the pair exceeds **both** single recipients by more than 0.02." | "In every contract the pair's mean recovery exceeds both single recipients' by more than 0.02 (simultaneous lower bounds 0.023–0.137). On seed 2 the full-logit pair bank selected the income-only attacker, a difference of exactly 0 on that seed." | No | No | Yes |
| output_diagnosis/MULTI_RECIPIENT_RESULTS.md l.38 | "near a recoding of the occupation input column" | "an easy task (accuracy 0.89–1.00 vs constant 0.277); the verifier found no single input column that determines it" | No | No | Yes |
| output_diagnosis/RESEARCH_DECISION.md l.29–30; l.135; ADVISOR_BRIEF.md l.55 | "close to recodings of input columns" / "near an input recoding" / "close to an input recoding" | "easy tasks closely related to the inputs (no single input column determines them)" | No | No | Minor |
| output_diagnosis/RESEARCH_DECISION.md l.124–125; COST_AND_STORAGE.md l.33–34; VALIDATION.md l.65 | "3,487 files" / "verified 3,487/3,487" | "3,489 files (3,487 at the first backup plus 2 files re-synced afterwards), verified 3,489/3,489 uncached", per the authoritative PRIVATE_BACKUP_INDEX.json (`files: 3489`) and HANDOFF.json l.130 | **Yes** (count) | No | No |
| output_diagnosis/PAPER_ADDENDUM.md l.14 | "0.0156 [0.0098, 0.0214]: just below the 0.01 target" | "0.0156 [0.0098, 0.0214]: positive, but the lower bound is just below the 0.01 target (NOT_ESTABLISHED)" | No | No | Minor |
| output_diagnosis/ADVISOR_BRIEF.md l.26–27 | "It does **not** hold for the multiclass employment and education heads: removing the offset changes nothing there." | "…removing the offset changes measured recovery by −0.017 to +0.001 there (NOT_ESTABLISHED). It is also not established for HMDA pricing × sex and fair_lending × {race, sex}." | No | No | Minor |
| output_diagnosis/PAPER_ADDENDUM.md l.39–40 | "…a refitted head with higher utility leaked as much through its margin alone." | "…a refitted head with twice the utility gain (0.081 vs 0.039) leaked about as much through its margin alone (0.774, vs 0.779 for the frozen head's full logits and 0.708 for its offset-free output)." | No | No | Minor |
| output_aware/RESEARCH_DECISION.md l.191–197 | "The results support one specific, already-published direction: **Release contracts that do not ship the clean task output.** Hard decisions, or heads computed only from the protected features…" | Add: "Later analysis (output diagnosis) qualifies this. Low decision-level recovery partly reflects weak heads: HMDA usefulness was not established, and Adult recall of the minority class is 14–22 %. Two recipients' decisions together recover more race than either alone (0.611 vs 0.576)." | No | No | **Yes** |
| output_aware/RESEARCH_DECISION.md l.137–139 | "**Strongest favourable.** On HMDA seeds 0 and 2, official FARE with 4 cells keeps untreated task accuracy … while the measured sensitive recovery falls to chance (0.50)…" | Add: "HMDA's whole task gain is 0.6–2.3 points, and in the complete-contract replay the same-budget zero-fairness tree already reaches 0.532 on seed 2 (FARE_REPLAY_EXISTING.csv)." | No | No | Yes |
| output_aware/OUTPUT_SURFACES.md l.33–34 | "**Hard predictions sit just above chance.** On Adult they are even below the label-only reference…" | Add: "…these decisions are of limited use (Adult gain 0.039; HMDA gain not established), so low decision recovery partly reflects decisions that say little." | No | No | Yes |
| matched_benchmark/RESEARCH_DECISION.md l.87, l.159, l.168 | "The linear guarantee transfers to unseen rows." / "…transfers it to held-out rows" / "its linear guarantee generalises" | Already corrected in output_aware CORRECTION_ADDENDUM W5: "Measured held-out ρ₁² under target LEACE: 0.0006 (Adult) and 0.0001 (HMDA), on one already-used held-out split; not a distribution-free guarantee." Carry the corrected text into the paper. | No | No | Yes |
| matched_benchmark/RESEARCH_DECISION.md l.191–192 | "More pairs, seeds or noise levels on these encoders would not change the decision" | Already removed (W7). Do not reuse. | No | No | Yes |
| matched_benchmark/RESEARCH_DECISION.md l.114–115 | "An affine eraser cannot change a lookup on those values" | Already replaced (W8) by observed distinct-value counts. Do not reuse. | No | No | Yes |

## G. Spot check of the runner's new protocol

`results/combined_analysis_paper_v1/PROTOCOL_USEFUL_HEADS.md`:
- l.91 already carries the correct wording rule.
- l.42–44 correctly states that a refitted logistic-regression head's offset is a deterministic function of its
  margin.
- No "pretrained" wording was found.

No change is proposed.

## H. Verdict

**No registered decision changes.** The corrections affect interpretation in three places that matter for the paper:
- the FARE compression share;
- the fair_lending head (a constant decision with leaking scores);
- the seed coverage of the offset effect.

They also:
- correct one reported count (the backup files, 3,489);
- separate two count families (cross-purpose ABS/INCR across different models, and same-model final vs best
  checkpoints) that the reconciliation's brief compressed into arrows.
