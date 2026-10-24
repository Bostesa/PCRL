# Start here: releases built from the strongest frozen anchors (campaign `pcrl_strong_anchor_release_v1`, SAR1)

A bounded development investigation, run on 2026-10-24 from 04:02:04Z on the Florida ACS 2019 development cohort (admitted public-use
PUMS records). This is statistical privacy research on an existing census benchmark. SEX is the dataset's recorded demographic
variable.

**What the service does.** It predicts an income class (2 classes) and an occupation group (5) from six fields: AGEP, WKHP, COW, SCHL,
MAR and POBP. Each of two recipients receives only its own probability vector and argmax decision. The six fields are model inputs,
not disclosed fields. One instance is released per person.

**What is measured.** How well fitted readers infer the recorded SEX attribute from what the recipients receive: from each output
alone and from the linked pair. The usefulness conditions are the source's 28: 12 task clauses and 2 within-SEX occupation guards
per model version, with the original caps.

**What changed from earlier campaigns.** They improved weak task-only starting services. This one started from the two strongest
usable frozen releases, frozen before any fit (`anchors/ANCHOR_SELECTION.md`):
- **A = SUCC:** H3 income with JOINT_TREE E20 occupation. Strong protection, but its binding condition is the men's occupation guard,
  +0.0044 / +0.0039 against a 0.005 cap.
- **B = NEURAL_RAW:** wide usefulness margins (at most 0.34 / 0.57 of cap). Its pair recovery sits in its occupation view.

**Result: no competitive release method is established** (`RESEARCH_DECISION.md`).
- **No reading met D2, D3 or D4.**
  - D2 is a reliable gain over the release's own start and matched controls.
  - D3 is superiority to the strong references.
  - D4 is the material target.
- **Both D2 looks were adaptive revisions** (FB_R1, FA_R2). The two registered readings found no feasible strength. No multiplicity
  correction is applied.
- **FA** (investigator A; on SUCC; card, then two revisions). It asked whether a strong protected release can recover usefulness
  margin while keeping its recovery profile.
  - **What it is.** The second revision, FA_TFRB_R2, is a cross-fit-bagged occupation residual distilled toward the admitted
    teachers. SUCC's income is kept bitwise.
  - **Usefulness.** It met all 28 conditions at the point. That is point eligibility only: 27 of 28 pass the confidence conjunction,
    with version 1's men's guard unresolved, and it is not reserve-qualified.
  - **Margin.** It improved SUCC's binding men's guard, just resolved on reused people (pooled −0.0029, se 0.0013).
  - **Recovery profile.** Not kept: version 0's occupation view rose by +0.025; version 1 improved.
  - **Against its start and control.** Over SUCC +0.0066 (se 0.0141) and over the untargeted control +0.0066 (0.0140), both
    unresolved. In version 1 that control point is SUCC itself.
  - **Against the references.** +0.0241 / +0.0208 over NEURAL_RAW and H_LJ at the point only; on a second reader split they fall to
    +0.0183 / +0.0158.
- **FB** (investigator B; on NEURAL_RAW; card, then one revision). It tried a different construction of the outputs, aimed at
  recovery.
  - **What it is.** The revision FB_PFO is occupation-only pair Fisher-discriminant training.
  - **Usefulness.** It met all 28 conditions at the point, with the same caveat: 27 of 28 pass the confidence conjunction, and it is
    not reserve-qualified.
  - **Against its start.** +0.0125 over NEURAL_RAW (0.0056), resolved.
  - **Against ALT.** It is not separated from NEURAL_RAW's own recipe continued to matched spend (ALT): +0.0012 (0.0074). That edge
    **changes sign on the second split** (−0.0012).
  - **Against the strong releases.** Its pair is more recoverable than SUCC's (−0.0050) and RETIRED_B3CTL's (−0.0130).
  - **Stop.** FB was stopped by decision after revision 1, with one revision unused.
- **FS** (the replacement family; on SUCC). A conditional pair-increment residual on SUCC's income, with occupation frozen.
  - It had no feasible strength in version 0 (version 1 had lambda 0.05), so there is no candidate.
  - A revision was declined on its predicted effect, about 0.005 over SUCC.
- **The three outcomes:**
  1. No reliable component gain. The only resolved wins over the zero-protection continuations are weak-twin attributions.
  2. Not superior to the strong references.
  3. The material target (a resolved 0.020 against NEURAL_RAW and H_LJ) is not established.
- **Deployment.** The 13 raw-input packages match the evaluated outputs bitwise for the full evaluated batch. For other request batch
  sizes they match within 6e-16 per probability, with identical decisions (`deploy/DEPLOYMENT_TOLERANCE.json`).

**Read next.**

| File | What it holds |
|---|---|
| `RESEARCH_DECISION.md` | the decision, the three outcomes, per-reading table, limits, plain answers, what prevents the next claim |
| `ENDPOINTS.json` | machine-readable endpoints per reading (labels, recovery per view and version, comparisons, usefulness, robustness) |
| `TABLES.md` / `TABLES.json` | every member's usefulness and recovery; every pair comparison |
| `PROTOCOL.md`, `roles/statistical_reviewer/RULES.md` | the registered design and decision rules (immutable); `PROTOCOL_DEVIATIONS.md` (14 items, append-only) |
| `cards/` | the prospective mechanism cards (FA_TFR_v1, FB_PFD_v1, FB_PFO_R1, FA_TFRO_R1, FA_TFRB_R2, FS_CPI_v1) |
| `roles/` | coordinator log; investigators' proposals, hand-backs and stop memo; reviewer's rules, reviews and claims review; verifier's checks (all verbatim) |
| `QUICKSTART.md` | commands: read results, use a deployment package, re-check custody |
| `HANDOFF.md` | state, custody and its gaps, owner actions, what a continuation would need |

All recovery results are exploratory development evidence on reused MODEL_SELECT households, for the declared reader bank. Intervals
do not correct the adaptive search, and nothing here is a confirmation.
