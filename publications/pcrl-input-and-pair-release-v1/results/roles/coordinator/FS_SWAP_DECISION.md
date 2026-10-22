# Coordinator decision on the replacement family FS_SWAP

Written 2026-10-22 after 02:45Z (`date -u`), before any MODEL_SELECT read of this campaign and before any FS card existed.

**Decision: FS_SWAP is not run; the replacement-family slot (PROTOCOL 7) stays unused for now.**

- The challenger named the cross-recipient keyed swap as the one replacement family worth the reserve (review finding 20), on the
  premise that it breaks the pair link while leaving each recipient's own view almost unchanged.
- Designer B specified it (`roles/designer_b/FS_SWAP_PROPOSAL.md`, hand-back `FS_SWAP_HANDBACK.md`, saved verbatim) and evaluated it in
  the same label-free plug-in world as the campaign's other proposals. There the premise does not hold:
  - a donor's occupation wire carries the donor's SEX association, so the occupation view falls as much as the pair;
  - per swapped person the swap spends 2.2x the occupation decision-cell collapse, and at equal spend it is no better than
    occupation-only flattening (OCC_FLAT) or CSF and about 0.0006 better than DQ;
  - its predicted D2 is NOT_SEPARATED (designer's probability 0.74), and its own decisive ablation (OCC_FLAT) is predicted to beat it.
- Running it would add a third closure test whose expected outcome is already dominated by a cheaper construction inside the
  campaign (IB's coordinated collapse and its ablations bound the same mechanism class), which the brief asks not to fill time with.
  The slot is kept for an observed failure of a registered reading, should one call for a replacement.
- This is a planning decision on label-free evidence, not a result; no FS configuration was saved and no FS label was read.
