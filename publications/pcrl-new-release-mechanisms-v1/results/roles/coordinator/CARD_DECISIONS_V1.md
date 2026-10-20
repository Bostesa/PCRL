# Coordinator decisions on the v1 family cards (committed 15:19:15Z, before any family card was committed and before any family fit)

(Correction: the header first said "15:21Z", a time ahead of the clock; the file was written about 15:18Z and committed at 15:19:15Z.
See PROTOCOL_DEVIATIONS item 5.)

Inputs:
- the designers' proposals (roles/designer_a/PROPOSAL_WAVE1.md; roles/designer_b/PROPOSAL_WAVE1.md and PROPOSAL_WAVE1_HANDBACK2.md);
- the challenger's equivalence and controls review (roles/challenger/REVIEW_PROPOSALS.md);
- the statistical reviewer's frozen rules (NRM1-rules-v1, RULES.md §12 requirements: twin and simplest alternative declared before
  the fit; joint designs declare the separate design and both sequential orders; G3 needs a SEX-blind control path).

## Family A: FA-PRC (propensity-routed finite cells)

- **Approved, with one control replaced.**
  - LOC_S2 is dropped, because no joint-design claim will be made; G6 is not assessed.
  - SHUF_S2 is added. It keeps the same optimizer, t grid and cells, but replaces π, qO0 and qO1 by a household-permuted copy. It
    is the SEX-blind control at the same joint-cell count, so it matches on reader budget (challenger A(2), A(3)).
- **Configurations:** PRC_TWIN, PRC_S1, PRC_S2, PRC_S3, ORD_S1, ORD_S2, ORD_S3, HQ_S2, SHUF_S2, each in both versions: 18.
  Contingent, each recorded with its evidence:
  - R0 twin ladder (24, 128): +2;
  - one ORD bracketing entry, chosen by measured MODEL_SELECT spend only, if `nrm.ctl.frontier` returns EXTEND_CONTROL or
    AHEAD_VS_SPARSE_PATH: +2.
- **Declared controls.**
  - Twin = PRC_TWIN (λ = 0, never reads SEX).
  - Simplest alternatives: ORD (SEX-blind coarsening path, the G3 control path), SHUF_S2 (untargeted re-routing at the same cells)
    and HQ_S2 (post-hoc SEX-aware grouping on q only, the closed SCM/IBSI line).
- **Required reporting.**
  - People per released vector and per joint value, for every release.
  - PRC vs TWIN at the same cells is the clean contrast.
  - A PRC-vs-ORD advantage is credited only if it also holds for smooth readers and for the MODEL_FIT cross-fitted LOOKUP.
  - Fitted-vs-held-out divergence is reported per strength.
  - Verdicts use spend measured on MODEL_SELECT by `nrm.evaluate`, never the nested MODEL_FIT estimates.
- **Fit order.**
  1. PRC_TWIN in both versions, with the internal capacity gate.
  2. The coordinator evaluates the twin on MODEL_SELECT.
  3. Only then the protected and control paths.

## Family B: FSC-CA (channel-aware joint training, SEX-targeted coupled Gaussian logit channel)

- **Approved, with the SEX-blind channel controls replaced.**
  - PCA is dropped. Its top principal axis is the task-dominant direction, so "SEX below PCA" could pass trivially (challenger B(2)).
  - In its place: MINF_S2 and MINF_S3, the same rank-1 coupled construction along each person's minimum-Fisher direction ("noise
    where the task is cheapest"), with the same s, cap, R and keys and the same channel-aware training.
  - Added: ISO_S2, the implemented full-rank Fisher-isotropic, uncoupled channel at matched second-order cost, the strong SEX-blind
    control.
- **Configurations:** FB_TWIN, FB_SEX_S2, FB_SEX_S3, FB_MINF_S2, FB_MINF_S3, FB_ISO_S2, FB_SEX_S2_ID, FB_SEX_S2_MEAN, FB_POSTHOC_S2,
  each in both versions: 18.
- **Falsification F1, restated.** The targeting rationale fails if, at matched realized MODEL_SELECT cost, the SEX path's linked
  recovery is not below the MINF path at S2 and S3. ISO is reported beside.
- **Declared controls.**
  - Twin = FB_TWIN.
  - Simplest alternatives: MINF (untargeted noise of the same form, the G3 control path), ISO, ID (identity channel of the candidate
    learner), MEAN (deterministic construction) and POSTHOC (no joint adaptation).
- **Challenger controls on FB_TWIN.**
  - CTL_REM (remainder flattening; deterministic, SEX-blind; keeps the decision and the top probability): +2. Its strength is chosen
    by spend only, bracketing FB_SEX_S2's realized MODEL_SELECT spend.
  - Contingent: CTL_BLIND_OCC at one bracketing level, +2.
- **Not claimed.** No coupling or joint-design claim: both the SEX and MINF paths are coupled, and ISO is not.
- **Card statements required.**
  - Deployed randomness for people outside the admitted rows: per-request keyed draws from a service CSPRNG key, one released
    instance per person. Repeated-query protection is not claimed.
  - Custody of the private root key: never published; replay only from the private tree or the backup.
  - SEX-labelled exposure: δ and Σ_w are a closed-form LDA-type critic using MODEL_FIT SEX on in-sample logits.
  - Per-task spend is read beside the linked cost.

## Budget after these decisions

| Item | Configurations |
|---|---|
| Firm: A 18 + B 18 + CTL_REM 2 | 38 |
| Contingent: A 4 + B 2 | 6 |
| Remaining, of 60 | 16 (ablations, training perturbation, revisions, one reviewed replacement) |

The statistical reviewer is asked to confirm that the committed cards conform to RULES.md §12. This runs in parallel with the fits
and is recorded with its answer.
