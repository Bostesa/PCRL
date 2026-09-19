# PAPER_ADDENDUM — for Terminal 2 (manuscript)

Study 5, `pcrl_competitive_method_v1`, branch `research/pcrl-competitive-method-v1`.
Every statement below is backed by `RESEARCH_DECISION.md` and its JSON/CSV ledgers.
Terminal 2 may revise prose; it must not change an endpoint, selection, decision rule
or reported sign.

## What may be written

1. **No competitive method was established.** Two prospectively defined tracks — a
   teacher-strength x initialisation factorial for adversarial refinement (168 slots)
   and a coalition-conditioned partial moment projection of frozen neural channels (204
   slots) — were fully fitted and audited on 2018 development pools. No prospectively
   nominated release met the registered conjunction (a sensitive endpoint significantly
   better than J and a strong external comparator, with residence noninferior at .001
   and no sensitive regression, under both weightings, candidate-wide within the
   declared family).
2. **The projection mechanism is an adaptation, not a new eigensystem.** It is the
   utility-free, infinite-penalty limit of the project's own residual spectral moment
   eigenproblem, applied to a frozen neural channel and unwhitened back into its metric
   (METHOD §2.5). Its only exact guarantee is the fixed-offset logit stationarity result
   of METHOD §2.7, under stated conditions; LEACE's complete-erasure guarantee is not
   claimed for partial projections.
3. **Protection on J trades against residence.** Coalition projection of J at rank 8
   lowers AB/SEX recovery significantly versus J (−0.0091) at a significant residence
   cost (+0.0101 in loss), matching ordinary LEACE applied to J within ±0.0011 on every
   endpoint. Local-only projections of the same rank behave the same: **no
   coalition-specific benefit on J**.
4. **A coalition-specific effect exists on the weaker A0 channel (exploratory).**
   Removing two coalition-conditioned directions from A0 lowers both local and
   coalition race recovery relative to local-only and feature-count-matched local
   projections of the same rank, significant under both weightings at the exploratory
   whole-grid level, with no significant residence difference. It is a mechanism
   finding on a channel far from J, not a competitive release.
5. **Adversarial refinement:** lowering the teacher coefficient helps A0-initialised
   training materially (race recovery roughly quartered at unchanged residence), but it
   is a coefficient change, not a method, and remains worse than J. J-initialised
   fine-tuning never improved on J under the registered selection rule. Coalition
   conditioning never beat the strength-matched local control L2 (0 of 128 cells).
6. **Diagnosis of the previous study** (SELECTION_DIAGNOSIS): every trajectory moved;
   39/72 unchanged channels was a selection outcome; the zero-gain clamp was inactive
   where an attacker existed.

## What must not be written

* That any release is Pareto-dominant, equivalent to, or noninferior to J, LEACE or
  SPLINCE on the basis of nonsignificance.
* That the stronger attack confirmed anything: the registered `MLP[256,256,128]`
  instrument was **weaker** than the standard audit and is reported as uninformative.
* That the development process was blind to residence or commute.
* That 2017 numbers are confirmation: they are exploratory on a repeatedly used year.
* That `ref_A0`/`ref_J` reproduced the historical audits exactly on all anchors: they
  did not on 3 cells, for an established candidate-set reason (amendment 4).

## Numbers Terminal 2 should cite (unweighted seed means, increments over H)

| release | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P | residence gain |
|---|---|---|---|---|---|
| J (ref) | +0.0015 | +0.0080 | +0.0000 | +0.0000 | +0.0215 |
| leace_A0 | +0.0039 | +0.0038 | +0.0097 | +0.0096 | +0.0220 |
| E_J_C_k8 (panel) | −0.0007 | −0.0011 | −0.0011 | −0.0000 | +0.0114 |
| E_J_C_k6 (panel) | −0.0004 | +0.0028 | −0.0024 | −0.0017 | +0.0166 |
| leace_J | +0.0000 | +0.0000 | −0.0001 | +0.0000 | +0.0115 |
| E_A0_C_k2 (exploratory) | +0.0163 | +0.0201 | +0.0223 | +0.0183 | +0.0309 |
| E_A0_L_k2 | +0.0112 | +0.0150 | +0.0425 | +0.0436 | +0.0291 |

2017 exploratory: see `EXPLORATORY_2017.md`.
