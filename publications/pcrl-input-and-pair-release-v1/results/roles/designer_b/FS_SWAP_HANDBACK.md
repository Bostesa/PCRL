# Designer B's hand-back on the replacement family FS_SWAP (verbatim)

Saved by the coordinator after 02:44Z on 2026-10-22, verbatim from Designer B's final message on the follow-up task of 02:36Z (a
subagent of this campaign; its words, not the coordinator's). The specification is `FS_SWAP_PROPOSAL.md` in this folder. One change:
the absolute worktree path in the file list is shortened to the repository-relative path (a personal path would be removed by the
publication sanitizer). Designer B's remark that `HANDBACK.md` in this folder was not written by it is correct: that file is the
coordinator's verbatim copy of Designer B's first hand-back message, with attribution.

---

The FS_SWAP spec is written. In my plug-in world the swap doesn't do what finding 20 hoped: it is no better than occupation-only flattening (OCC_FLAT) or CSF at equal spend, and only about 0.0006 better than DQ. I predict D2 NOT_SEPARATED (p 0.74).

1. **Operation.**
   - Swap selection: a person is swapped if u(x) < lambda, where u(x) = SHA-256(`IPR_FS_SWAP|V<v>|sel` | six compact fields) / 16^13.
   - Donor bank at deployment: the 20,000 MODEL_FIT records, with the deployed twin's occupation vectors. Rows are grouped by occupation decision cell, kept people-weighted, and ordered by a SHA-256 key on the record, so storage order doesn't matter.
   - Donor choice: a second keyed uniform v(x) picks row floor(v·|B_k|), stepping past rows with x's own record. The release is deterministic in the six fields.
   - Wires: income gets f_I(x) bitwise; occupation gets f_O(donor). Decisions and accuracy are kept exactly, SEX is never read, `operation_sex_blind: true`.
   - In the nested MODEL_FIT selection the bank is the other folds' people with their out-of-fold vectors.
   - Disclosed: swapped wires are MODEL_FIT bank members' vectors released under other identities.
2. **Strength and selection.** Lambda grid {0.01, 0.02, …, 0.10, 0.125, 0.15, 0.20}, budget S3, PROTOCOL 4 default rule on nested 5-fold household folds.
   - Plug-in S3 ratios: 0.82 / 0.84 at 0.05 and 1.31 / 1.24 at 0.075, so I expect lambda_S3 = 0.05.
   - The binding component is the men's occupation Brier guard (the women's from 0.075 in V1).
   - Summed occupation Brier is +0.0037 / +0.0038 at 0.05; it is a G1 clause but not an S3 component, so it is reported beside.
3. **Controls.**
   - Gating: TWIN and the shared SPEND path (DQ, tag `IPR_DQ_S<v>`). SBC and substituted RO are degenerate and not built; GENERIC_ZTILT is reported beside if it exists.
   - Ablation: ABL_OCC_FLAT, the same key with the decision-cell mean instead of a donor, matched on MODEL_FIT spend. This is the decisive test of finding 20's mechanism.
   - CSF is read descriptively from IB's configurations.
   - D5: NOT_APPLICABLE_ONE_OUTPUT_CHANGED, since only the occupation wire changes.
4. **Plug-in at lambda 0.05** (4 draws, V0 / V1):
   - pair .6379 / .6381 against the twin's .6410 / .6419;
   - DQ at equal KL .6384 / .6388;
   - OCC_FLAT at equal KL .6367 / .6374 (better than the swap);
   - CSF at equal KL .6372 / .6382;
   - at matched KL, CSF and DQ also sit lower on S3 (.58–.72 against .82–.84).
5. **Spend per swapped person.**
   - Occupation KL is 0.198 / 0.193, 2.2x the occupation cell-mean collapse and 1.4x CSF's two-recipient collapse.
   - Brier and the men's guard are exactly 2x the collapse.
   - Pair AUC drop per nat: swap 0.37 / 0.40, OCC_FLAT 0.47 / 0.51, CSF 0.45 / 0.43, DQ 0.19 / 0.26.
6. **Why it falls short.**
   - Keeping the occupation decision keeps the pair link through it: swapping everyone gives a pair of .5925 / .5951, against .5960 / .5970 for flattening everyone and a .577 floor.
   - Hiding who was swapped is worth only 0.002–0.0035 even at lambda 1.
   - A donor's wire carries the donor's SEX association, so the occupation view falls as much as the pair (−0.0044 / −0.0051 against −0.0031 / −0.0038). Finding 20's "own-view leakage almost unchanged" does not hold.
7. **D2 label predictions:** NOT_SEPARATED 0.74, NOT_ASSESSED_TASK_INFEASIBLE 0.10, NOT_ASSESSED_NO_FEASIBLE_STRENGTH 0.05, PARTIAL_TWIN_ONLY 0.03, READER_TARGETED 0.03, PARTIAL_SPEND_FRONTIER_ONLY 0.02, INCOMPLETE 0.02, SEX_BLIND_IMPROVEMENT_BEYOND_TWIN_AND_SPEND 0.01.
   - Point predictions: pair benefit vs the twin +0.001 to +0.007, the swap no better than OCC_FLAT (p 0.7), D4 not met (p 0.99).
   - Falsifier: the swap beating OCC_FLAT by at least 0.010 in both versions.
8. **Configurations:** CAND 2 + ABL_OCC_FLAT 2 = 4 new, reusing the shared SPEND points, plus at most 2 if those fail the 25% bracket rule — at most 6. Optional revision: joint-decision-cell donors (`swap_joint`), 13% cheaper per person with the same pair.
9. **CPU estimate:** under 2 CPU-minutes on the mechanism side; the reader grid is about 800–1,200 CPU-s.
10. **CPU used:** about 80 CPU-s for this task (in-script 70.6 s plus imports), one thread. Nothing labelled, nothing of MODEL_SELECT, no reader cache, no `.private` runs/access; no commits. `HANDBACK.md` in my directory was not written by me.

Files are in `results/pcrl_input_and_pair_release_v1/roles/designer_b/`:
- FS_SWAP_PROPOSAL.md
- fs_swap_sketch.py (modes `swap`, `swap_joint`, `occ_flat`, `csf`; selftest passes on invented data)
- fs_swap_feasibility.py
- FS_SWAP_FEASIBILITY.json
- fs_swap_point.py
- FS_SWAP_POINT.json
