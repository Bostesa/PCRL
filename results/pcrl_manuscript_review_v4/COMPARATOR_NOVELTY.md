# COMPARATOR_NOVELTY — v4

Scope: bounded. Sources and verification status carried from
`results/pcrl_manuscript_review_v3/RELATED_WORK_COMPARISON.md` (LEACE, SPLINCE, SARL, OptNet-ARL, K-TOpt,
U-FaTE, LAFTR, Sankar et al., Stadler et al., Elazar & Goldberg). No new source was read in v4; LEACE's
formula is taken from the PDF-body transcription at 73903b7f, not from a web summary (a prior automated
pass hallucinated it). Absence below means *not found in what was read*, not *does not exist*.

## Study 4 (direct adversarial refinement)
Ingredients all prior: adversarial transferable representations (Madras et al. 2018), attacker
ensembles/refresh, residual-logit baselines, compression, LEACE/SPLINCE/OptNet-ARL as comparators.
Study 4 attributes these correctly. Result: not competitive with J or the executed external adaptations.

## Pending projection (Track P)
- Mathematically: LEACE's closed form (full-rank Σ) with a coalition-conditioned residual cross-moment
  and explicit rank control. Adaptation, not new mathematics.
- Conditional-moment / residualised targets: conceptually related to conditional-independence and
  debiased-nuisance literature (KCI, RCoT, DML — cited in v3); cross-fitting is DML practice.
- SARL/OptNet-ARL: trace-criterion subspace selection is SARL-like (v3 §6a audit showed spectral arms are
  SARL adaptations).
- Multiple-consumer formulations: Sankar et al. 2013 mention "multiple legitimate information consumers";
  whether they receive different releases was UNVERIFIED-ABSTRACT in v3 and remains so. Do not claim the
  multi-recipient framing is unprecedented.

## Defensible positioning
"We apply coalition-conditioned direction removal to strong existing channels beside an immutable
published output, with explicit rank controls, and evaluate it with recipient- and coalition-specific
audits." That is an adaptation plus an evaluation setting. A stronger novelty statement would require a
full prior-work comparison that has not been done.

## Competitive-claim requirements (four levels, kept separate)
1. Mechanism correctness — MATHEMATICAL_REVIEW M1–M8 satisfied; measured rank/support/tolerance per arm.
2. Competitive development trade-off — candidate selected without residence/commute labels or evaluation
   scores, one config across 3 anchor seeds; vs untouched J, executed LEACE/SPLINCE/OptNet adaptations and
   local controls; superiority, noninferiority (.001 utility margin, .001-nat sensitive tolerance) and
   zero-margin reported by name; one declared correction level.
3. Coalition-specific benefit — same source channel under local conditioning, matched rank/capacity.
4. Independent confirmation — fresh prospective protocol on an unused partition; 2016 sealed.
A local-control success is not a competitive result; losses to J/baselines must be shown beside it.

## Private overlap
The private AAAI overlap note stays at `.git/pcrl_private_v3/PRIVATE_AAAI_OVERLAP.md` (untracked, not
published). v4 adds no text from the private drafts and no identifying cross-citation. The private drafts
were not available to re-read this session, so the note was not updated; Study 4 and the pending programme
introduce nothing that its section 5 rule ("do not cross-quote ACS figures") does not already cover.
