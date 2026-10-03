# Evidence map notes (BOTH_REPOS_EVIDENCE_MAP.csv)

Built 2026-10-03 from read-only sources, with no fits. 248 rows × 23 columns, checked with Python `csv`: the column count is consistent and no row has an empty `family`. There is one row per result; no row merges two counts. The four families are separate bodies of work, and no code path connects them.

| Family | Rows | Verification |
|---|---|---|
| 1-purpose-specific-encoder | 42 | 29 independently recomputed, 13 author-reported |
| 2-attribute-removal-isolate-noise | 79 | 39 recomputed, 37 author-reported, 3 missing |
| 3-acs-finite-release | 42 | 28 replay PASS, 2 recomputed, 12 author-reported |
| 4-stored-model-output | 85 | 82 replay PASS, 2 author-reported, 1 pending |

**Status mapping** from the inventory's `verification_status`:
- "independently recomputed" stays as is. "unsupported by located artifacts" becomes `missing`.
- "checked against code and recorded aggregates" (aggregate-level, not per seed) becomes `author-reported only`; the row's notes say "Inventory status: checked…". "reported but not independently reproduced" also becomes `author-reported only`.
- Family 3 and 4 replay counts come from INDEPENDENT_VERIFICATION.json or the study docs. They are verifier scripts shipped inside each study, not external audits.

**Family placement:**
- **D-\*** (durable-guarantees, AAAI audit) go in family 2 (isolate/noise). D-03's Folktables CA 2018 cells belong to that 67-configuration audit, not to a finite-release study. **P-FOLKTABLES-R2** is the PCRL encoder trained on Folktables, so it is family 1.
- **Family 3** is the Sept 2026 ACS release lineage: the trunk of the `research/pcrl-*` branches (origin/ablations-facct-2026-07-24) and those branches. **None of it is in the 82-row inventory.** Rows come from each study's RESEARCH_DECISION or HANDOFF via `git show`; 7 headlines were re-checked against the docs.
- **Rows taken from reconciliation verification items, not inventory records:** D5 (LEACE on raw features), D6 (separate encoders), D7 (R-LACE/LEACE), E1 (utility conventions), F5 (VFAE worst-pair), F6 (isolate-then-noise advantage).

## Inventory record → map rows (EM-nnn)

```
P-R4-adult 1,4-6; P-R4-hmda 2,4-6; P-R4-diabetes 3-6; P-R4-adult-ckpt 122,128; P-R4-hmda-ckpt 128
P-ROUND5-adult 7,10-15; P-ROUND5-hmda 8,10-15; P-ROUND7-diabetes 9-14; P-ROUND5-diabetes 16-17; P-ROUND6-diabetes 18
P-HELDOUT-S3 19-20; P-CELEBA-PCRLV 21-22; P-CELEBA-v2-ckpt 93; P-INLP 43-45; P-LAFTR-PERPURPOSE-adult 46,49
P-LAFTR-PERPURPOSE-hmda-diabetes 47-50; P-SPLINCE 51-53; P-BIOS-L12 23; P-HEAD-AWARE-LEACE 24; P-BIOS-v2-rounds 25
P-FOLKTABLES-R2 26; P-CROSSPURP-CONSTRAINT 37-38; P-CROSSPURPOSE-ORIG 29,31,39-42; P-BASELINES-SINGLEPURPOSE 56
P-VARCONSTRAINT-PERDIM 27; R-01 57-60,62,64; R-02 11,13,57-58,61,63; R-05 65-68; R-07 65-68; R-08 69; R-09 70-72
R-10 29,31,39-40; R-11 30,32-35; R-12 30,32,36; R-13 37-38; R-14 73,76-78; R-15 74,76-78; R-16 75-78; R-17 76-78
R-23 70; D-01 79; D-02 80,82-84; D-03 81-84; D-04 82-89; D-05 90; D-06 91; D-07 92; D-08 93; D-09 94-98; D-10 99
D-11 94-98,100-101; D-12 82-84,87-89,102; D-13 103; D-14 104; D-15 105-107; D-16 108-109; D-17 94-96,110; D-18 111
D-19 112; D-20 113; D-21 114-116; D-22 117; D-23 118; D-24 119-120; D-25 121; O-PRESERVE-01 30,32,36,57-60,62,64
```

**Not a result (16 of 82 records; 66 map to rows):**
- Checkpoints only: P-R4-diabetes-ckpt, P-ROUND5-adult-ckpt, P-ROUND5-hmda-ckpt, P-ROUND7-diabetes-ckpt. No stored-model study has run on the R5/R7 checkpoints yet.
- Not executed or not completed: P-FACCT-ABL (registered and smoke-tested only); R-06 (VICReg Diabetes arm, cut by the 10-h cap, no file).
- Smoke runs or launch configuration: R-03, R-18 (smoke); R-04 (launch configuration).
- Lost or not read: R-19 (lost to the S3 lifecycle rule); R-20, R-21, R-22 (unread S3 checkpoint prefixes); D-26 (LAFTR-official raw directories, not located).
- Documents: D-27 (internal notes); D-28 (manuscript source).
- O-PRESERVE-01 preserves R-01 and R-12 and is mapped through their rows.

**Commits.** Every short hash resolved unambiguously, so no row needed `UNRESOLVED_SHORT` (expanded with `git rev-parse` in PCRL or in the durable-guarantees clone). EM-036 has no commit because its files are laptop-only; the preserved copies are at 07a9ca3ffaf27133b6cf955b3b8d526682961e8d.

## Cross-purpose counts: the exact source lines

Neither 22→8 nor 26→19 is a final-vs-best checkpoint pair. Each arrow changes three things at once: the model (submitted R5/R7 → rebuttal union-eraser retrain), the checkpoint rule (final.pt → best.pt) and the post-hoc criterion. They map to four rows: EM-029 = 22, EM-030 = 8, EM-031 = 26, EM-032 = 19.

- REBUTTAL_WORK_RECONCILIATION l.69–70: "**ABS:** the concatenation beats majority by more than 1 pp." / "**INCR:** the concatenation beats the strongest single-purpose release by more than 1 pp."
- l.76: "| Submitted PCRL (R5 Adult/HMDA, R7 Diabetes) | final.pt | 26/33 | 22/33 | recomputed |"
- l.77: "| Rebuttal: erase-layer union-LEACE + h_concat R² constraint (τ_cross = 0.10), 200 ep | best.pt | 19/33 | 8/33 | recomputed (laptop-only files, now preserved) |"
- l.93: "The rebuttal genuinely reduced *concatenation increments*, from 22 to 8 incremental flags."
- UPDATED_MEETING_BRIEF l.77: "It cuts *incremental* concatenation flags from 22/33 to 8/33."
- l.91–92: "Under absolute recovery, leakage on nonlinear cells *rose* (26/33 → 19/33 flags, but the mean leak went up)."
- Inventory R-11: "incremental flags 8/33 … absolute criterion on same data 19/33".

The genuine same-model checkpoint pairs each have their own rows: strict 56/60 (final.pt, EM-010) vs 54/60 (best.pt, EM-011); cleanly compliant 6/60 (EM-012) vs 5/60 (EM-013); 7/60 (EM-014) mixes the two rules.

Coalition rows keep absolute pair recovery (EM-184, 189, 194) apart from the individual releases and from the increments over the strongest individual release (EM-186, 193, 198).

**Corrections applied** (from CORRECTIONS_AND_SCOPE):
- FARE beyond compression: +0.0074 [0.0019, 0.0129], a smaller improvement below the 0.02 target (EM-201).
- HMDA fair_lending is a constant **decision**, not constant scores; its centred logits still give race 0.646 and sex 0.655 (EM-183).
- Offset removal passes on 5 of 14 pairs (EM-180). HMDA seed 1 shows no offset benefit; the benefit appears on 5 of 6 primary seeds (EM-173, 174).
- The refitted Adult head's gain is 0.0826 and the probe's is 0.0814 (EM-171). The stage-5 task is easy but is not a column recoding (EM-202).
- Backup count: the index lists 3,489 files (not a result row).

## Surprises worth flagging

1. **CORRECTIONS_AND_SCOPE scope item 5 mislabels the cross-purpose counts.** It calls 22→8 and 26→19 the "best-validation vs final-checkpoint" families. The sources above show they are the INCR and ABS criteria, each comparing two different models. That sentence needs fixing.
2. **ACS commits are not on this branch.** The commits named as this branch's recent history (ad2c088, 482f14d, 8a3c181, 4a36178, 02a0692) are not ancestors of research/combined-analysis-paper-v1. They are on the ablations-facct trunk.
3. **The matched benchmark has 185 FAIL items in its replay.** All are exploratory; the 183/183 primary-scope checks pass, and output-aware's custody re-run reproduces 24/24 primary endpoints.
4. **33/60 (Round-4 best.pt) has no itemised verification entry,** so it is marked author-reported. Separately, the R5 Diabetes counts disagree across sources: 18/18 (best.pt), 16/18 (final.pt) and 9/18 (commit message).
5. **Many "independent" replays are in-study scripts.** One ACS study (competitive_method) has 3 failed reference-identity checks, and utility_extension's cross-platform re-audit exceeded its tolerance.
6. **Several durable-guarantees artifact folders carry a person's name in their path.** Those paths were left out of the map.
