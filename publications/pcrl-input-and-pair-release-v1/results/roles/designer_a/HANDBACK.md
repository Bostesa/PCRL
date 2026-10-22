# Designer A's hand-back summary (verbatim)

Saved by the coordinator after 01:53Z on 2026-10-22, verbatim from Designer A's final message (a subagent of this campaign; its
words, not the coordinator's). The proposal itself is `PROPOSAL.md` in this folder. One change: the absolute worktree path in
the last paragraph is shortened to the repository-relative path (a personal path would be removed by the publication sanitizer).

---

1. **Operation (SFG, shared SEX-aware field generalization):** both task models see only h_k(x), at training and at release. These are A1_PO_16 HGB models, with T* set by the twin's inner-CV rule. h_k(x) is the six fields coarsened by the first k steps of a ladder over a fixed, label-free menu of 12 generalizations (two coarser levels per field: age bands, hours bands, SCHL groups, COW groups, MAR merged or suppressed, POBP regions or suppressed). MODEL_FIT SEX is used only to order the menu: cross-fitted SEX log-loss gained per nat of task NLL. Strength grid k ∈ {2, 3, 4}. k* is the largest k whose MODEL_FIT out-of-fold clauses pass at estimate + 2 household SE (NLL 0.012 per task, accuracy 0.008 per task, men's and women's occupation Brier 0.0045). Before any MODEL_SELECT read, a MODEL_FIT reader gate (G1) applies.
2. **Closest previous implementations:** grouped feature selection (row 8, `results/pcrl_capacity_matched_feature_selection_v1/`, Adult) deleted whole field groups per recipient, and the linked pair undid it. SCM (row 5) merged posterior cells after the fact. SFG differs: it merges levels inside fields on ACS FL, gives both recipients the same h, refits the models and releases continuous vectors. It belongs to none of the closed families.
3. **Failure addressed:** the super-additive pair and leakage moving between recipients (failures 2 and 3). The pair is a function of h(x), so no reader can beat AUC(P(SEX | h(X))). This removes information instead of hiding it. It does not credibly address failures 1 or 6, and it runs into failure 5 (income pays most of the spend).
4. **Twin and controls:** the twin is DA_TWIN (the cached TASK teachers). The simplest matched control, BLIND, uses the same menu and ladder ordered by task spend only, with two points bracketing SFG's MODEL_FIT spend. Further controls: ATTR (the same ladder ordered by PSEUDO2), the release-only controls PROJ (E[SFG release | twin release]) and ZTILT, and DQ, all at matched spend.
5. **Label-free feasibility** (plug-in, with the cached pi and eta treated as truth). The plug-in twin pair is 0.654 against the realized 0.6506, so the method is calibrated. SFG k=2 (MAR suppressed) spends 0.0085 nats for a pair change of −0.005. k=3 (plus hours in 3 bands) spends 0.018 nats (0.011 of it income) for −0.013 in both versions, and the ceiling falls from 0.678 to 0.656. k=4 reaches −0.024 but is infeasible: income accuracy loss 0.015, women's occupation Brier +0.0095.
6. **Against the controls (plug-in):** SEX-blind generalization at equal spend gives only −0.0006 at 0.020 nats, so SFG is about 0.012 ahead. But field-structured (interaction-constrained) models move the pair by at most 0.0015 at 0.011-0.022 nats. PROJ keeps about 60% of SFG's gain at no extra spend. A release-only tilt at SFG's spend does as well or better (−0.012 to −0.019). Failure 6 is therefore predicted to repeat.
7. **Verdict:** nothing in my space plausibly beats its matched controls by 0.02. Registered predictions:
   - against the twin: +0.005 (k*=2, p 0.6) to +0.010 (k*=3);
   - against BLIND: +0.005 to +0.010;
   - against ZTILT: −0.005;
   - D2 not separated (p 0.85) and D4 not met (p 0.98).
   I recommend a lean run as the input-side closure test, or giving the budget to Designer B.
8. **Revisions:**
   - R1 re-weights the ladder by each task's slack. Trigger: an income clause is the only failing one and occupation keeps at least 50% slack.
   - R2 splits the next step into predeclared halves. Trigger: the next step overshoots the remaining budget by more than 50%. It is the likely one, because income accuracy binds at k=3.
9. **Configurations and CPU:** 10 core configurations (SFG_k* 2, BLIND bracket 4, PROJ 2, ZTILT 2) plus 6 optional (the second SFG rung, ATTR, DQ), so 16 at most, with the twin inherited. The CPU estimate is about 2.6 CPU-h: ordering 0.7, fits 0.8, selection 0.1, readers 1.0.
10. **CPU I used:** about 146 CPU-s, one thread. No label file, nothing of MODEL_SELECT, no reader cache and nothing of designer_b was opened; nothing was committed. One known flaw is disclosed in PROPOSAL.md: the k=6 point of `feasibility.py` lets a moderate step replace a strong one, and that point is not cited.

Files are in `results/pcrl_input_and_pair_release_v1/roles/designer_a/`:
- PROPOSAL.md
- mech_sketch.py (`selftest()` passes on invented data, 11.4 CPU-s)
- feasibility.py, feasibility2.py, feasibility3.py, with their outputs FEASIBILITY.json, FEASIBILITY2.json, FEASIBILITY3.json
- inspect_inputs.py
