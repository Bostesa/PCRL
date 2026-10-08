# M3 matched-control rules — prefit proposal

Status: **PROPOSED FOR M0/M5 PREFIT REVIEW. No Adult fit, attack, or INNER_SELECTION scoring has used these rules.** All masks and pair rows come from M2's one immutable shared bank. M1's provisional primary profile contains the five masks obtained by deleting one of five categorical source groups while retaining four numeric and four categorical groups per recipient under the 77-encoded-column cap. The full Cartesian table has 25 ordered assignments. Actual recipient encoded widths and the exact raw union are reported for every pair.

## Common eligibility and order

Every rule uses the same bank, fitted heads, fitting-role task metrics and search critics. A mask is task-eligible only when its fitting-validation accuracy-loss, NLL-excess and Brier-excess margins are nonnegative against **U0, Ucal and F** (inclusive boundaries). A pair requires both masks task-eligible and M1-valid recipient capacities. No rule reads ATTACK_FIT_NEW, INNER_SELECTION, old assessment, or barred labels during selection. The task-only T objective uses only task fields; changing any critic AUC must leave T unchanged.

Select main T first: for each purpose independently, choose the task-eligible mask with the lowest fitting-validation natural-log NLL, then smallest canonical mask SHA. The T pair is the reference for fitting-stage local guards. For every other privacy rule, each recipient's oriented local search AUC must be at most its corresponding T AUC plus **0.010** (inclusive). This is a side constraint, not a change to T's task-only objective. The union-matched task-only objective is conditional on J's union quota and this fixed local guard; call it a task-only **objective** under privacy/geometry constraints, not a fully SEX-blind process.

On tied objective values, use summed two-task fitting-validation NLL and then `(income_mask_sha256, occupation_mask_sha256)` lexicographically. No best-seed or INNER outcome tie break is permitted. All feasible rows, including rows yielding identical masks for different rules, stay in the registry.

## Main rules (recipient profile matched; union may differ)

| Rule | Selection on the full 25-pair bank |
| --- | --- |
| `T` | Choose the two masks independently by each task's validation NLL as above; combine their registered pair. No AUC field is consulted. |
| `I` | Independently choose each task-eligible, locally guarded mask with lowest oriented local search AUC; tie by its task NLL, then mask SHA. Combine the two registered masks. |
| `S12` | Choose recipient 1 by local AUC among masks with at least one legal recipient-2 completion; tie by its task NLL and SHA. Freeze that first mask. Choose recipient 2 among legal completions by lowest coalition search AUC; tie by summed task NLL and mask SHA pair. |
| `S21` | The same local-first then coalition-aware completion in reverse recipient order. |
| `SHARED` | If a diagonal `(m,m)` exists and meets both tasks' caps and local guards, choose the diagonal with lowest coalition search AUC; tie by summed task NLL and mask SHA. Otherwise record incompatible/invalid, not an exact control. |

The sequential feasibility lookahead checks only whether a legal completion exists. It never uses the second-stage coalition score to change the first-stage local ranking. The first choice is immutable after selection. These are study controls, not a reproduction of Taylor et al.'s published setting.

## J-geometry union-matched rules

After M2 chooses J on fitting-role data, record J's **union group count and encoded-column count**. The attribution controls search all eligible table rows with that same union-size signature, without requiring J's union feature identity. The constraint is fixed before INNER_SELECTION scoring; no recovery outcome changes it.

| Rule | Objective within J's union-size signature |
| --- | --- |
| `T_U` | Lowest summed task validation NLL, then mask SHA pair, with the common utility/local guards. |
| `I_U` | Lowest sum of the two oriented local search AUCs, then summed task NLL, then mask SHA pair. This is a **local-objective rule under a union quota**, not fully independent selection. |
| `S12_U` | Recipient 1 local-first among masks with a quota-feasible completion; freeze it, then choose recipient 2 by coalition AUC, task NLL and mask SHA pair. |
| `S21_U` | Reverse-order version of `S12_U`. |

If no eligible pair exists for a union-matched rule, register its exact failure reason. A legitimate identical J/control mask pair is a valid tie. It must not be excluded or perturbed to manufacture a distinct comparator. M3 records mask-pair equivalence; M2/M4 may record a complete-release alias only after verifying both heads, calibration, wire fields and artifact hashes. Matching raw unions alone establishes raw-information equivalence, not a complete-view alias.

**Structural limit of this five-mask bank:** If J removes different categorical atoms for the two recipients, all 20 distinct-deletion pairs expose the identical full 83-column raw union. They can vary allocation, task heads and finite-reader behavior, but cannot differ in raw-union information. If J removes the same categorical atom for both recipients, its union has eight groups; matching both union group count and encoded width uniquely fixes that deleted atom because the five categorical widths differ. Every eligible equal-union control then selects the same mask pair, yielding a valid tie when the restored release identities also match. Accordingly, this bank's union-matched comparison cannot establish a raw-information advantage from feature choice at equal union capacity. Report this constraint even if a fresh attacker produces differing AUCs.

## Strongest-control registry

The matched **recipient capacity signature** is the sealed profile `(4 numeric groups, 4 categorical groups, at most 77 encoded columns)` for each recipient. The five masks have different realized encoded widths; report both widths for every row, but do not silently make exact realized width a new eligibility filter. For attribution, additionally match J's **actual** union source-group count and encoded width. Thus off-diagonal pairs share a full-83 union quota, while a diagonal union quota uniquely identifies its deleted categorical group in this bank. M0 fixed this convention before fitting.

Main pool: `T,I,S12,S21`, compatible `SHARED`, and `T_U,I_U,S12_U,S21_U` where their recipient profile matches. Attribution pool: the four union-matched controls plus any main rule whose actual union-size signature matches J's. An exact release alias stays in its pool and contributes a tie. Control eligibility requires point utility caps against U0, Ucal and F on **every evaluated seed**. Among eligible rules, choose the lowest paired-seed mean *final* coalition AUC; **exact numeric equality** uses the predeclared rule-ID order, without a tolerance band. Selection of the strongest rule uses the sealed final endpoint table; it does not change any fitted mask or reader. If a pool has no eligible valid rule, mark that dependent comparison incomplete. Main and attribution controls are selected separately; report each individual rule and alias.

The final privacy and utility verdict belongs to M4/M0 under frozen simultaneous inference. M3 search-reader scores only select masks. Neither a finite-bank optimum nor a fresh-reader weakness proves information removal. M5 independently checks the registry and selection before a claim.
