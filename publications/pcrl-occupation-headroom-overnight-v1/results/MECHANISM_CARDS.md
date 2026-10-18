# Mechanism cards: index

Each mechanism has a card. The card was written before the mechanism's first fit and committed by the coordinator before each
revision's fits. There is one exception: the P2 card was committed 71 s after its first fit (PROTOCOL_DEVIATIONS item 10). Code
provenance has its own exceptions: fits on an uncommitted protect.py (item 14), and family B / C code hashes (item 17). Later results
are appended with times; earlier text is never rewritten. Reports are the agents' closing texts, saved by the coordinator with a
disclosure.

| Mechanism | Card / report | Operation (deployed output) | Revisions | Outcome |
|---|---|---|---|---|
| A. Stronger categorical boosting (occupation, task-only) | `roles/investigator_a/CARD.md`, `REPORT.md` | HistGradientBoosting on a compact coding of the six fields; revision 1 codes POBP as its ordinal census code (`A1_PO_8`, `A1_PO_16`) | 1 of 2 (line exhausted at 6 of 12 configurations) | A1_PO_16 is witness 2. It beats T_TREE on NLL, Brier and accuracy on both seeds, and the men's guard is −0.0039 / −0.0023. The native-category A0 is worse than the teacher. |
| B. Complementary learners plus cross-fitted stacking (task-only) | `roles/investigator_b/CARD.md`, `REPORT.md` | Repaired tree (TREE_V2, POBP ordinal) and additive logistic basis; SINGLE = the temperature-scaled better learner, STACK = non-negative logit stack; B3 adds a CALIBRATION layer over the frozen TASK_REFIT | 2 of 2 | B2_SINGLE (the family's registered single-learner control) is witness 1, with occupation-side ratio 4.418. The stack B2_STACK adds −0.0025 / −0.0024 NLL out of fold but ranked third, because it weakens the men's guard. B3 did not meet its rule. The deployed bases are identical across seeds. |
| C. Residual task learning on the teacher (task-only) | `roles/investigator_c/CARD.md`, `REPORT.md` | Boosted residual on cross-fitted, exactly re-estimated teacher margins; revision 2 adds ordered block codes | 2 of 2 (abandon rule applied) | No headroom beyond TASK_REFIT (best +0.0001 / +0.0003 NLL). The teacher's in-sample optimism (+0.034 NLL) makes in-sample residual selection blind. |
| Conditional income branch (INC8) | `roles/investigator_inc/CARD.md`, `REPORT.md` | The source's STRONG income learner with POBP ordinal coding; H3's transplant step applied unchanged (`INC_H3_PO`); task-only twin `INC_TW_PO` | none allowed (defect only) | Repaired learner better in cross-fitting (Brier −0.0029 / −0.0028, NLL −0.0042 / −0.0041). Seed-1 income accuracy unchanged (ratio 2.840). The task-only twin reaches 3.144, below the unrepaired 3.296. The income view fails the recipient gate; no pair benefit. |
| Protection: coalition-aware occupation correction (PRIV24, CTRL12) | `roles/investigator_prot/CARD.md`, `REPORT.md` | `softmax(log p_W + s·Δθ)`: a leaf-score correction on the frozen T_TREE routing, trained against SEX critics of the linked pair (H3 income frozen) and of the occupation vector, scaled to an out-of-fold MODEL_FIT budget; arms CONT, LOC, PAIR, COMP | P1 initial; P2 (no in-training guard); P3 (static per-SEX task-loss weights, seed 0); plus a seed-1 LOC arm | At B80 it reads below the continuation twin and the B80 compression code in all 8 cells on both witnesses (declared readers; 17-18 of 18 fixed recipes per seed). At B50 the margins change sign. The B80 PAIR levels break the seed-0 reserve. The reserve-keeping local critic P2_LOC_B80 is 0.010 above SUCC. Nothing beats SUCC / RETIRED_B3CTL on both seeds within the reserve. Every configuration fails the per-cell recipient gate and the stability gate. |

Supporting records:
- Witness selection: `WITNESS_SELECTION.json` (frozen rule, 06:27:13Z).
- Nomination: `NOMINATION.json` and `NOMINATION_TABLE.csv` (07:45:00Z).
- Rules: `roles/statistical_reviewer/RULES.json` and `SELECTION_RULE.md`.
- Closest established methods: `roles/closest_work/EVIDENCE_TABLE.md`.
- Every fitted configuration: `EXPERIMENT_LEDGER.csv`.
