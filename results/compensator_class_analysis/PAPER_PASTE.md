# PAPER_PASTE — block-0 compensator class analysis (§5.6)

## Verdict (data-driven, brutally honest)

**WALK BACK §5.6's directional-compensator framing for (0,6).** The 
144-cell mech-interp sweep contains **zero** heads whose mean OV write 
projects negatively along either the LEACE rank-1 gender direction or 
the class-mean gender direction. (0,6) is **not** a counter-balance 
head; it has the *largest* positive proj_LEACE in block 0 
(+0.388, vs. (0,7)=+0.054 and 
(0,10)=+0.105), writing in the *same* direction as 
the dominant heads.

## What the ablation evidence does say

Adding (0,6) to the {(0,7),(0,10)} ablation set drops probe 
accuracy from 31.9\% (top-2) to 
22.4\% (top-3) — a real 
9.5 pp degradation. But this 
degradation is not consistent with an *opposing-direction* writer; it 
is consistent with (0,6) being a **pronoun-swap-robust gender carrier**: 
after pronoun corruption, (0,6) still writes the original-gender 
direction (because its OV pulls from non-pronoun tokens), so ablating 
it on the corrupt run removes a salvage pathway for original-gender 
information. The OV-projection sign (clean cache) and the ablation-
degradation sign (corrupt cache) measure different things, and only 
the former bears on the directional-compensator claim.

## Recommended one-sentence framing for §5.6

> Of the 12 block-0 heads, none meets a directional-compensator 
> criterion (signed OV projection opposite to the dominant heads with 
> $|{\rm proj}_{\rm LEACE}|>0.15$); (0,6)'s ablation-degradation 
> signature (31.9\%$\to$22.4\%) 
> arises not from opposing directionality but from its role as a 
> pronoun-swap-robust gender carrier whose ablation removes a salvage 
> pathway for original-gender information.

## Don't claim

- *"Block-0 contains a class of compensator heads."* No directional 
  compensators exist in the 144-cell sweep.
- *"(0,6) writes opposite to (0,7), (0,10)."* (0,6) writes 
  **most strongly in the same direction**; it is the most-aligned 
  block-0 writer along the gender axis.

## Why the data is unambiguous on this

- Min `proj_LEACE` over **all 144** (block, head) cells: 
  +0.0002 (still positive).
- Min `proj_classmean` over **all 144** cells: 
  +0.0001 (still positive).
- Only **1 of 144** cells has negative $\Delta_{\rm logit}$: (1,8) 
  at −0.072, but its OV projection is still positive (+0.066 
  classmean). (1,8) is a behavioral negative-$\Delta$ head, not a 
  directional compensator.

## What's at stake

If §5.6 currently relies on "compensator class" or "counter-balance" 
language for (0,6), the language needs to be revised. The mechanistic 
story that *is* supported is structurally weaker but more accurate: 
block-0 attention has **diffuse positive contribution** to the gender 
axis (12/12 heads positive, mean proj_LEACE = 
+0.0958), with (0,7) and (0,10) 
accounting for the largest *causal* effects (Δ_logit) but not the 
largest *write magnitudes* (those are (0,6), (0,5), (0,4) — heads with 
low flip rate but high OV-norm). The asymmetry between write-magnitude 
ranking and causal-ranking is interesting in its own right and is 
consistent with the Stage-3 'DIFFUSE MEDIATION' headline.

## Numbers (block 0, sorted by proj_LEACE)

| head | proj_LEACE | proj_class | Δ_logit | flip_rate | classification |
|---|---|---|---|---|---|
| (0,6) | +0.3883 | +0.6369 | +0.0899 | 0.000 | co_writer_block0 |
| (0,5) | +0.2422 | +0.3997 | +0.0812 | 0.005 | co_writer_block0 |
| (0,4) | +0.1504 | +0.1910 | +0.0889 | 0.002 | co_writer_block0 |
| (0,10) | +0.1053 | +0.1204 | +0.0928 | 0.004 | dominant_writer |
| (0,9) | +0.0558 | +0.1166 | +0.0400 | 0.065 | neutral |
| (0,7) | +0.0537 | +0.0661 | +0.1412 | 0.029 | dominant_writer |
| (0,8) | +0.0533 | +0.0265 | +0.0634 | 0.008 | neutral |
| (0,0) | +0.0430 | +0.0251 | +0.0399 | 0.015 | neutral |
| (0,3) | +0.0206 | +0.0225 | +0.0821 | 0.008 | neutral |
| (0,11) | +0.0156 | +0.0633 | +0.0646 | 0.013 | neutral |
| (0,2) | +0.0155 | +0.0324 | +0.0672 | 0.024 | neutral |
| (0,1) | +0.0059 | +0.0510 | +0.0688 | 0.005 | neutral |
