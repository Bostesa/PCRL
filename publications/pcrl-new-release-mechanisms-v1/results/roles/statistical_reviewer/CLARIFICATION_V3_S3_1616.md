# Statistical reviewer: clarification of RULING_FB_V3 section 3 (saved verbatim with attribution)

Author: the statistical reviewer of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator on 2026-10-20 before
16:15:46Z (the coordinator's `date -u` read just after receipt), in answer to the coordinator's question sent at about 16:15Z (is B0 read
once per cell in the shared FB2P bank, like B1?). The reply's own header says 16:16Z, a time ahead of that clock read; the file name and the
first version of this header (commit 9a77e0852) repeated "16:16Z". Corrected here visibly. Given before any output of the registered grid
NRM1_MAIN was read. Saved verbatim; nothing below the line was edited.

---

Statistical reviewer, reply on RULING_FB_V3 section 3 (16:16Z). Read-only; no NRM1_MAIN output read.

**CONFIRM for Δ and se_Δ; REPLACE for the applicability check and the disagreement label.**

**Confirmed: Δ and se_Δ.**
- B0 is read once per cell in the shared FB2P bank, with the same design as B1: the declared instance map (R1 → 0, R2 → 1, k0/k1 sharing) and cache hits of the NRM1_MAIN offset-0 readers.
- Δ = x1 − x0 is then like-for-like.
- se_Δ is the `se_agg` (`nrm.rules.aggregate`) of the per-cell entry, where:
  - cells = b1_c − b0_c;
  - cell_se = the SD of the paired bootstrap-draw differences (b1 − b0), with the weights shared within each cell.
- The NRM1_MAIN Monte Carlo reading of B0 (offsets +2/+4 pooled by `mc_cell`) is reported beside and never enters Δ.

**Replacement: the applicability check uses the registered reading.**
- NOT_APPLICABLE_ORIGINAL_NOT_POSITIVE is decided by the registered NRM1_MAIN reading of B0 (the `mc_cell`-pooled one).
- That is the same reading rule 2(a) uses to trigger the FB2P fit, so the label cannot occur for a fitted FB2P, as section 3 states.
- Why: the perturbation label qualifies the registered "SEX beyond MINF" statement, so its applicability must follow that statement's own reading.

**If B0-single and B0-MC disagree on AGG_POSITIVE:**
- Report the label defined above (applicability from B0-MC; Δ and se_Δ from B0-single).
- Attach the flag B0_SINGLE_INSTANCE_NOT_RESOLVED and show both B0 labels and aggregates.
- The flag never changes the label. It tells the reader that a NOT_CONSISTENT result may reflect single-draw Monte Carlo noise rather than the perturbation.

Everything else in section 3 is unchanged.
