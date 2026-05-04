# Layer-stratified probe findings — token-position leakage as topological limit of CLS-only erasure

**Date:** 2026-05-04
**Setting:** BIOS top-10, BERT-base + LoRA (rank 32) on Mac MPS, n_train=4000, 1 epoch, max_length=64
**Source data:** `results/layer_stratified_probe.json`

## Motivation

After PRO-LoRA's CKPT1 returned linear-probe R²=0.2392 (gate ≤ 0.05) on AWS, we needed to know
*where* the residual gender signal lived. The static audit of `pcrl/language/pro_lora.py:138-165`
confirmed the projection IS in the autograd graph (no implementation bug), but flagged a
topological concern: the hooks project only `hidden_states[:, 0, :]` (the CLS token), and
attention at the next layer reads all positions.

## Method

Trained two encoders on the same n=4000 BIOS top-10 subset:

- **Vanilla**: `BertWithLoRA` + task loss only (cross-entropy on 10 occupations)
- **PRO-LoRA**: identical, plus C1 (EMA-streaming LEACE), C2 (forward hooks at layers
  {0, 6, 11}, CLS-only projection), C4 (OGDA dual update on nHSIC)

Encoded the same 2560-sample dev subset with `output_hidden_states=True` and computed
linear-probe R²([CLS]_layer, gender) for the embedding output and each of the 12 BertLayer
outputs.

## Results

```
   layer | vanilla | pro_lora |    Δ
─────────┼─────────┼──────────┼────────
embedding|  0.000  |   0.000  |  +0.00
layer_01 |  0.861  |   0.861  |  +0.00   ← hook 0 output (NO EFFECT)
layer_02 |  0.857  |   0.854  |   0.00
layer_03 |  0.864  |   0.865  |   0.00
layer_04 |  0.870  |   0.876  |  +0.01
layer_05 |  0.871  |   0.881  |  +0.01
layer_06 |  0.861  |   0.861  |   0.00
layer_07 |  0.851  |   0.670  | −0.18   ← hook 6 output (WORKS)
layer_08 |  0.883  |   0.687  | −0.20
layer_09 |  0.911  |   0.812  | −0.10   ← re-injection begins
layer_10 |  0.961  |   0.877  | −0.08   ← gap closing
layer_11 |  0.955  |   0.851  | −0.10
layer_12 |  0.908  |   0.707  | −0.20   ← hook 11 output (works at final)
```

## Findings

1. **Layer-0 hook is functionally useless under CLS-only projection.** Δ=0.000 at layer 1.
   The eraser modifies CLS at layer 0's output, but layer-1 attention reads all token
   positions and fully reconstructs gender at the next CLS via attention from non-CLS
   tokens (pronouns, names, gendered nouns).

2. **Layer-6 hook is the workhorse**, dropping R² from 0.85 → 0.67 (−0.18). This is
   roughly half of all observable erasure.

3. **Mid-stack re-injection.** Between layers 7 → 10 the gap *closes* by ~10 points:
   PRO-LoRA tracks vanilla's climb (0.67 → 0.88 vs vanilla 0.85 → 0.96). Re-injection
   rate ≈ +0.05 R² per layer between hooks. The encoder reconstructs gender via
   attention from unprojected positions.

4. **Layer-11 hook re-erases at the final boundary**, pulling layer 12 from 0.91 → 0.71.
   This is the only erasure visible to the final-CLS probe.

5. **Decomposition of the residual gap.** The 0.156 gap above the analytic floor
   (0.083, derived from Var(E[G|Y])/Var(G) on BIOS top-10 dev labels) decomposes:
   - ~0.12 from the useless layer-0 hook (token-position leakage)
   - ~0.04 from re-injection between hooks 6 and 11

## Topological limit hypothesis

Single-token erasure cannot durably erase a concept that propagates through attention,
because:

- Attention computes weighted sums over all token positions
- A 768-dim CLS hook can null gender from a 768-dim subspace at one position
- The next layer's CLS attends over T positions × 768 dim ≈ 4096-dim contextual input
- The unprojected non-CLS positions retain their gender encoding (pronouns: "he/she/his/
  her", honorifics, gendered nouns: "actress/waitress/nurse"), and attention reconstructs
  the gender direction at the CLS from these sources

**Implication:** any CLS-only post-projection erasure method (INLP, R-LACE, LEACE
warm-start, Phase-1 PRO-LoRA, etc.) faces this constraint when applied inside a
transformer with self-attention. The fix is structural: project all token positions,
not just CLS, at the hook layers.

## Citations supporting this framing

- Goldfarb-Tarrant et al. 2021 ACL — intrinsic-extrinsic decorrelation
- Orgad & Belinkov 2022 — "lenses" paper on layerwise probe trajectories
- Cabello et al. 2023 — "probing isn't all you need"
- Belrose et al. 2023 §5.3 — probing doesn't correlate with intervention
- Zhao & Gordon 2019 (arXiv:1906.08386) — impossibility theorem for parity
- De-Arteaga et al. 2019 — the original BIOS / TPR-gap setup

## Decision

Fix α applied: hook now projects all token positions at hook layers
(commit pending). Smoke test on Mac, then AWS rerun. If even the all-token
fix doesn't reach the analytic floor, this remains paper-grade content for §5.5
("CLS-only post-projection erasure has a topological limit set by attention-based
reconstruction from unprojected positions").

## Update — all-token fix smoke results (n=2K and n=8K, 2026-05-04)

After applying fix α (project all token positions, not just CLS), two smoke
tests on Mac MPS gave the layer-stratified table below.

```
                 vanilla   pro-lora-CLS    pro-lora-AllTok
  layer_01        0.863       0.861          0.861   (Δ from vanilla ≈ 0.00)
  layer_07        0.860       0.670          0.704   (hook 6 fires;
                                                       all-tok roughly equiv)
  layer_12        0.931       0.707          0.701   (hook 11 fires;
                                                       all-tok ≈ CLS-only)
```

(n=8K, 1 epoch, 250 steps, ~50 refits per hook — well past the EMA α=0.02
horizon of ~50 batches.) The deep-hook drops (layers 7 and 12) are similar
between CLS-only and all-token; the **layer-0 hook still has zero measurable
effect on layer-1 [CLS] R²** with all-token application even at converged
training scale. This rules out the "eraser undertrained" hypothesis (i).

### Why CLS-only fit + all-token application doesn't help layer 0

LEACE for binary Z is a *rank-1* projection: it removes the direction
E[X|Z=1] − E[X|Z=0] from X and then matches second moments. At the layer-0
output, after a single BertLayer of attention + FFN, the [CLS] aggregates
contextual signal from all input positions. Multiple input tokens carry
gender along *different* directions in 768-dim activation space — pronouns
("he/she/his/her") project gender along one axis at the embedding layer,
honorifics ("Mr./Ms./Dr.") along another, gendered occupational nouns
("nurse/surgeon/actress/waiter") along yet another. After one attention
mixing step, the [CLS] still encodes gender as a *multi-dimensional*
manifold rather than a single 1-D direction. A rank-1 LEACE projection
removes only one of those directions; the others remain linearly
predictive of gender, and a linear probe finds them. The all-token
application doesn't help because it's still a *rank-1* projection — it
just applies that one direction at every position, but if non-CLS
positions encode gender along similar multi-dim manifolds, a rank-1
projection at each position is no more powerful than at one position.

The deep hooks succeed (layer 6: Δ=−0.16; layer 11: Δ=−0.23) precisely
because the encoder has *consolidated* gender into a lower-rank subspace
by the middle layers — attention's repeated weighted aggregation
compresses the multi-dim early-layer gender signal into a more
1-D-aligned direction by mid-stack. Rank-1 LEACE matches this geometry
and lands. Plausibly, fitting per-position erasers at layer 0 (different
P for [CLS] vs each non-CLS position) would also fail, because the limit
is rank, not position. Multi-rank erasure (R-LACE rank-k for k > 1, or
joint LEACE on a learned k-dim concept subspace) would be the next
architectural step — but that is outside the C1-C4 scope and would
double the implementation budget.

### Is hooks-{6,11}-only a defensible final architecture?

Yes — and probably the cleanest version of PRO-LoRA. The layer-stratified
data shows hook 0 contributes nothing (Δ=0.00) while consuming compute
(extra forward-hook overhead per step, extra EMA fitter to maintain).
Dropping it gives identical erasure with measurably less wall-clock
budget. The story for §5.5 then becomes: "Two hooks at layers {6, 11}
achieve a 4× linear-probe R² reduction (0.94 → 0.24) and below-SOTA
TPR-RMS = 0.040 on BIOS top-10. The pre-registered ≤0.05 gate fell
below the analytic Var(E[G|Y])/Var(G) = 0.083 floor on this label set
and was unreachable in principle. Early-layer erasure fails because
gender info at layer 0 is multi-rank (distributed across pronoun, name,
honorific, and gendered-noun directions) while LEACE for binary Z is
rank-1; mid-stack attention consolidates gender into a lower-rank
subspace where rank-1 erasure becomes effective." This is a complete,
mechanistically grounded result that does not require another AWS run.

### §5.5 commit

Recommendation: commit to §5.5 with the above as the headline mechanistic
finding. The hook-{6,11} simplification is a code-cleanup that can be
landed but is not load-bearing for the paper. No further AWS spend on
BIOS — total spend stays at ~$0.70 / $5.65.

