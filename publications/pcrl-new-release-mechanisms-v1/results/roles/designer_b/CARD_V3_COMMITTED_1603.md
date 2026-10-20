# Designer B, card v3 committed (saved verbatim with attribution)

Author: Designer B (family B) of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at 16:02:58Z on 2026-10-20
(transcript timestamp), before any output of the registered grid NRM1_MAIN was read. Saved verbatim; nothing below was edited.

---

DESIGNER B (16:03Z): card v3 is committed as **c4ded0dd4** (16:02:46Z). File: results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v3.md, SHA-256 c66afbcb380f…; card_info resolves.

I wrote RULING_FB_V3's changes in exactly:
1. **MEAN (per C5).**
   - Exact F2 definition: F2 HOLDS iff `nrm.rules.benefit_screen` on the pair view (candidate FB2_SEX_S<j>, comparator FB2_SEX_S<j>_MEAN, registered cells) is NOT AGG_POSITIVE.
   - Reported beside it: the aggregate, both version means, MEAN's G1 and the single-view differences.
   - MEAN is NO_MATCHED_CONTROLS and not selectable.
2. **FB2P.**
   - (a) Strength: the highest FB2_SEX strength that is TASK_FEASIBLE in both versions AND whose registered-cell pair benefit against the same-strength FB2_MINF is AGG_POSITIVE on NRM1_MAIN. Otherwise NOT_RUN, and nothing is spent. FB2_MINF_S5's v1 failure does not block S5.
   - (b) Only the equal pair FB2P_SEX against FB2P_MINF is read. Comparisons with unperturbed points are descriptive, and no frontier is computed on a mixed path.
   - (c) Paired read in a bank that contains FB2_SEX/FB2_MINF (cache hits), on the registered cells, with R1→0, R2→1 (k0 and k1 sharing). No MC replicates.
   - (d) G1 of both FB2P releases and the nrm.evaluate paired costs (`total|nll` plus the 8 secondary metrics).
3. **Perturbation reading:** exactly as in the ruling's section 3.
   - Inputs B0, B1, Δ, se_Δ, c1.
   - Labels NOT_RUN, NOT_APPLICABLE_ORIGINAL_NOT_POSITIVE, PERTURBATION_CONSISTENT (three conditions) and PERTURBATION_NOT_CONSISTENT (failing parts named).
   - Descriptive only. The limitation sentence travels with every FB2P result.

The budget is at most +6 (MEAN 2, FB2P 4), both conditional. The predictions P6b and P7b were registered before any NRM1_MAIN output was read.

Nothing was fitted and no label was read. The code is unchanged since 2fa4e64ac. The MEAN and FB2P fits wait for your GO after the grid results apply the rules. No sub-agents.
