# Protocol amendment 1: adjudication of the challenger's design review (campaign `pcrl_input_and_pair_release_v1`)

Coordinator. Written 2026-10-22 from 02:36Z (`date -u`), after the registration (adfe3e75a, receipt e909fb8c0 verified 02:14:15Z) and
before any mechanism card of IA or IB was committed, before any fit of IA or IB and before any MODEL_SELECT read. The registration
files are not edited; this dated file records how the challenger's review (`roles/challenger/REVIEW_DESIGNS.md`, hand-back
`roles/challenger/HANDBACK.md`, both saved verbatim) changes the two families' cards. Every change below is within the registered
protocol and rules (PROTOCOL sections 4, 6, 7 and 8; RULES.md 3.2-3.4 and 6).

## Family IA (Designer A, SFG): PROCEED_WITH_REQUIRED_CHANGES, adopted in full

1. **Budget S3** (finding 3), one of the predeclared budgets: occupation NLL 0.012, income NLL 0.006, men's and women's occupation
   Brier 0.0045, occupation accuracy loss 0.008, income accuracy loss 0.006. The designer's looser income components are not used.
   The card corrects the headroom sentence: the dual-design twin is 0.0086 / 0.0081 nats better than U0 on income NLL and 0.0161 /
   0.0128 on occupation NLL (published MODEL_SELECT records).
2. **Rung k = 1 in the grid, and the rungs scanned in order** (finding 4): the selection rule needs every smaller rung to pass, so
   rungs are fitted in ladder order and the scan stops at the first rung that fails S3. The card discloses that a near-trivial first
   rung (one moderate step) is a likely CAND.
3. **BLIND is the targeting-off ablation, not an SBC path** (finding 5); **the gating SBC is the permuted-SEX ladder** (finding 6),
   with PSEUDO2 only by the registered fallback rule (the permuted path cannot reach CAND's spend + 2 SE); ATTR is not built. The card
   records that the permuted ladder is expected to be noise-driven, close to BLIND and different by version
   (SBC_CONSTRUCTION_DIFFERS_BY_VERSION is computed by the rules from the records; both versions use the permutation unless the
   fallback fires).
4. **Intermediate levels in the menu** (finding 7): each of WKHP, COW, SCHL, MAR and POBP gets a predeclared intermediate level
   between its moderate and strong levels (WK5, COW4, SCH4, MAR2, POB2; definitions in the card), so the ladder has 17 rungs of
   smaller spend steps. The coordinator's definitions differ in detail from the designer's R2 halves (one intermediate level per field,
   a chain) and are stated in the card; the fineness of the saved control points is computed and reported by the rules
   (`grid_fineness`, reported, never gating).
5. **No PROJ; RO = GENERIC_ZTILT as the card's declared fallback** (finding 8). The registered default for an operation with a
   SEX-dependent fitted input is SUBSTITUTED (RULES.md 3.2). The challenger's check (`CHECK_SUPPORT.json`) shows that the twin's released
   pair identifies the six-field record for 92.9% / 92.7% of MODEL_FIT people, so a release-only surrogate of SEX is the record-level
   propensity up to its learner, and the substituted ladder is SFG's own ladder up to estimation noise: its comparison would be close
   to CAND against itself. The card therefore declares `ro = {kind: GENERIC_ZTILT, operation_release_only: false}`, the fallback the
   rules allow, whose D2 scope sentence is "the gain is not reproduced by the generic release-only tilt at matched spend; the
   operation driven only by the twin's release was not built". The substituted ladder is still computed on MODEL_FIT inside the fit
   unit (no configuration, never released) and its equality with SFG's ladder is recorded, so the release-only question is answered
   descriptively. `operation_sex_blind: false`.
6. **Card corrections** (finding 9): the designer's MODEL_FIT reader gate becomes a descriptive check that never stops anything (it
   is not "G1"); strength selection uses each rung's own construction (`ipr.ia.sfg.fit_rung`: T* re-selected by the inner household
   CV, out-of-fold on the cached twin's folds); the data-processing ceiling is stated as having no bite at feasible rungs.
7. **Predicted gate failure against RO** (finding 18): the card predicts RECIPIENT_GATE_FAILED_VS_RO as a likely D2 outcome (the
   generic tilt flattens the income view more than SFG at matched spend).

## Family IB (Designer B): REDIRECT, adopted with one change

8. **CAND = the uniform, SEX-blind coordinated keyed collapse CSF at budget S3** (findings 10-13), declared
   `operation_sex_blind: true`: SBC and RO are degenerate by construction and not built, and D2 gates on TWIN and the SPEND path
   (met label SEX_BLIND_IMPROVEMENT_BEYOND_TWIN_AND_SPEND, quoted with its scope: no SEX-targeting claim). Designer B's CSF-T becomes the
   **targeting ablation** and IND (independent per-recipient keys) the **coordination ablation**, both at CAND's strength. S3 replaces
   the designer's S2 (the men's occupation Brier guard binds at every strength in the designer's own grid). No generic ZTILT path is
   built for IB (the rules make it optional for a SEX-blind operation).
9. **D5** (finding 14): the challenger recommends declaring the joint design not built. The coordinator instead declares, in the card
   and before the first fit, that **both sequential orders equal the joint design by construction**: a sequential design whose second
   stage may use the first stage's release and the shared six-field key flattens exactly the people the first stage flattened (their
   first-stage vectors are decision-cell rows), which reproduces CSF bitwise. Under RULES.md 6 this gives
   `NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL` (no joint-design advantage over sequential design can exist), a more informative statement
   than NOT_ASSESSED_MISSING_CONTROLS; the separately designed recipients (IND) are evaluated and reported as the coordination
   ablation (descriptive). Neither label is a joint-design claim.

## Shared control code

10. **The DQ dither tag is family-independent** (finding 16), and the SPEND and RO bracket points of both families are saved under the
    shared family CTL (card CTL_PATHS_v1, configuration ids CTL_DQ_<step> / CTL_RO_<strength>) by the family fit units, once: a point
    that already exists is reused and counted once. The two versions run in separate lanes and the two families of one version run
    in sequence in the same lane, so no point is written twice.
11. Advisory 17 (the inherited budgets carry no income or overall Brier component) is recorded in both cards as a known gap.

## Replacement family (finding 20)

12. The challenger's cross-recipient keyed swap is the one replacement family worth the reserve. At 02:36Z the coordinator asked the
    redirected Designer B to specify it as `FS_SWAP` (one-page proposal, sketch, plug-in estimate, predictions). Its card is read by the
    challenger and the statistical reviewer before it is committed (PROTOCOL 7) and, if it is ready in time, it is committed before
    the first MODEL_SELECT read of this campaign (a REVISED_FS reading under RULES.md 7); otherwise it is an ADAPTIVE_FS reading.

## Allocation (48 configurations)

| Block | Configurations |
|---|---|
| CTL_TWIN (saved) | 2 |
| Shared SPEND and RO points (CTL; up to two per path and version per family, fewer where they coincide) | up to 12 |
| IA: CAND 2, SBC 4, BLIND ablation 2 | 8 |
| IB: CAND 2, CSF-T ablation 2, IND ablation 2 | 6 |
| FS_SWAP (if adopted): CAND 2, one ablation 2 | 4 |
| Reserve (revisions) | 16 or more |
