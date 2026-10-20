# Coordinator adjudication of the pre-registration reviews (2026-10-20, written from 23:30Z, `date -u`)

Inputs: the challenger's review (`roles/challenger/REVIEW_PREREGISTRATION.md`, 4 BLOCKER, 9 SHOULD FIX, 8 NOTE) and the statistical
reviewer's report and frozen rules (`roles/statistical_reviewer/REPORT_PREREGISTRATION.md`, RULES.md section 10, C1-C11). No label had
been read by anyone. Each item: decision and reason. "Adopted" items are applied to PROTOCOL.md, the card FA_FRC_v1.md and the code
before the registration commit; rule changes are made by the statistical reviewer in the files it owns.

## Challenger findings

| # | Decision | Reason / what changes |
|---|---|---|
| 1 QUAL slack S1 vs assessment at S2 (BLOCKER) | **Adopted (primary text).** Q2 uses the S2 increments: occupation NLL 0.008, income NLL 0.004, occupation accuracy loss 0.006, income accuracy loss 0.004, guards 0.0030, occupation summed Brier 0.0030, income summed Brier 0.0020. | D2 and its HQ / SHUF controls sit at S2, so "useful task margin" for the registered comparison means S2 slack. Known before registration and disclosed: A at (16, 64) meets Q1, fails this Q2 (version-1 men's guard 0.00255 + 0.0030 = 0.00555 > 0.005) and meets the S1 form; `QUALIFIED_WITHOUT_PROBABILITY_REPAIR` can occur only at (24, 128) or (32, 256). The fallback b* was not adopted (one assessment budget, no data-dependent budget). |
| 2 P3 reads in-sample deployed routing (BLOCKER) | **Adopted.** P3 reads the out-of-fold-routed fit cells, is labelled "routing optimized on these people (optimistic)" and is never a credit input. The decision-relevant MODEL_FIT reader check is the nested held-out LOOKUP of P1. | Differential in-sample bias in PRC's favour. Code: `frc.fa.route.stage_check`. |
| 3 F4 and reader targeting missing from D2 (BLOCKER) | **Adopted with one change.** F4 (the PRC configuration selected at S2: nested held-out pair LOOKUP AUC below TWIN's by more than 2 household-bootstrap SE in both versions) is computed in P1 and committed before any MODEL_SELECT read of a protected release, and is a required conjunct of D2. It is **not** a stop of the evaluation: the brief asks for the numerical comparison to be completed and allows diagnostic comparisons of ambiguous results, so the protected releases are still read, and a failed F4 makes D2 `NOT_MET_F4` and may justify revision R2 (the registered remedy for "fitted objective falls, held-out LOOKUP does not"). Reader targeting adopted as written: D2 also needs a point benefit > 0 in both versions for the non-table pair winner (the grid's `nolookup` extra) and for the nested MODEL_FIT LOOKUP, against TWIN and every bracketing control; otherwise `READER_TARGETED`. The governing card is FA_FRC_v1 (it now carries falsifiers 1-5). | Deviation from FA_PRC_v1's "stop before any MODEL_SELECT read" disclosed. |
| 4 frc/devreaders.py NRM1 constants (BLOCKER) | **Already fixed** at about 23:17Z, before the report arrived (FRC1-devgrid-v1, tags FRC1_devgrid_R<r>, offset 700, seed 2026102100; the selftest's forbidden list now includes 600; READER_SPLITS.json records the NRM1 overlaps). **Additional defect found by the coordinator** while checking this: `nrm.readerextra.cell_extras` imports `nrm.devreaders` relatively, so the extras would have used NRM1's splits (it would have failed on the bootstrap shape). Fixed by a campaign copy `frc/readerextra.py` imported by `frc/devreaders.py` and locked. | |
| 5 HQ / SHUF not spend-matched; ORD contingent (SHOULD FIX) | **Adopted with changes.** Bracketing points HQ_BR, SHUF_BR and ORD_BR: the first strength (grid lambda, or ladder rung) in path order whose nested MODEL_FIT total-NLL spend over the twin (point) is >= that of the PRC configuration selected at S2; saved only if not an alias of an already saved point (at most 6 configurations). "Beyond" HQ, SHUF and ORD is `nrm.rules.frontier_comparison` against the control paths [HQ_S2, HQ_BR], [SHUF_S2, SHUF_BR] and [TWIN, ORD_S1-S3, ORD_BR] with MODEL_SELECT paired total-NLL costs; no path point spending at least as much gives `NOT_SEPARATED_FROM_TASK_SPENDING` (`CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT`), never credit. | The bracket uses the frontier rule's primary cost (total NLL), and every bracket is chosen on MODEL_FIT, never on MODEL_SELECT spend (release parameters are never chosen on MODEL_SELECT labels). |
| 6 SHUF guard on the donor's occupation posteriors (SHOULD FIX) | **Adopted** (a card change before any SHUF fit has run): SHUF uses the keyed-hash donor copy of pi only; its guard uses the person's own qO for both qO0 and qO1 (SEX-blind task fidelity). | Isolates targeting from a wasted, task-irrelevant guard. Code: an embedding mode of `frc.fa.route`. |
| 7 Reader-budget asymmetry (SHOULD FIX) | **Adopted.** The pair-view `frc.readerbudget.contrast` is recorded for every D2 / D3 comparison; an asymmetric D3 comparison is credited only with the same non-table and nested-LOOKUP point conditions as D2 (D2 already requires them). The FIT-fraction curve (0.25 / 0.5 / 1.0) is reported for the pair view of TWIN, the PRC configuration at S2 and the bracketing controls (descriptive). | |
| 8 Additive two-table reader (SHOULD FIX) | **Already present**: the grid extra TABLE_CV (pair view) is an additive logit on one-hot(income value) + one-hot(occupation value) with cross-validated L2, then the joint table shrunk towards it; it and the ENHANCED winner are reported beside the registered winner (descriptive). No new recipe. | |
| 9 D2 undefined cases (SHOULD FIX) | **Adopted**: `NOT_ASSESSED_NO_FEASIBLE_STRENGTH`; an alias of PRC_S1 is assessed as that configuration with its brackets; `NOT_ASSESSED_EXACT_ALIAS` (and stop) if every nonzero selection transmits exactly the twin's vectors. | |
| 10 No-fallback rule (SHOULD FIX) | **Adopted** (also the statistical reviewer's C3). | |
| 11 T6 forking path (SHOULD FIX) | **Adopted (alternative text)**: a T6 repair is labelled POST_HOC; its card is committed before any MODEL_SELECT read of its outputs; it can give only `POST_HOC_REPAIR_QUALIFIED`, never `REPAIRED_TASK_ONLY_RELEASE`. T6 is kept because the brief permits one documented structural repair. | |
| 12 Reproduction sufficiency (SHOULD FIX) | **Adopted**: (a) verifier recomputes B1 bitwise with separate code; (b) and (c) added to the selftest; (d) people routed into out-of-fold-empty cells counted per role; (e) per-fold B1 counts equal A's counts (recorded). | |
| 13 ORD ladders (SHOULD FIX) | **Adopted**: per-capacity ladders registered, finest to coarsest. | |
| 14 Partition fixed | Noted (consistent with the selftest). | |
| 15 Non-nested T* shared | Adopted: stated in PROTOCOL 4.1. | |
| 16 Decision flips | Adopted: cells and people whose A and B1 decisions differ, with accuracy / Brier differences restricted to them, are reported; prediction 6's distances were measured on the A twin (stated). | |
| 17 Fold-level dependence | **Adopted**: the Q-PE SE is max(household-clustered SE, SD of the 5 outer-fold means / sqrt(5)). | |
| 18 Power | Adopted: minimum detectable pair benefit about 0.015 (2 se_agg); "not separated" is the default reading. | |
| 19 E4 conditional | **Not adopted.** The brief asks for the ablation and the refit of the strongest feasible nonzero release even if the first interval crosses zero; both run whenever such a release exists. Budget: 48 still holds (PROTOCOL 12). | |
| 20 HQ sparse at 128 / 256 cells | Adopted: HQ's fitted-minus-held-out divergence is reported beside every HQ reading. | |
| 21 MODEL_SELECT SEX in QUAL | Adopted: disclosed (utility selection common to PRC and its controls). | |

## Statistical reviewer's requested changes

| Item | Decision |
|---|---|
| C1 grid constants | Already fixed (see finding 4). |
| C2 men's / women's rows from T3 only | Adopted. |
| C3 per-capacity construction choice | Adopted. |
| C4 `QUALIFIED_B1_REPAIR_NOT_RESOLVED` | Adopted. |
| C5 no feasible strength | Adopted for PRC (`NOT_ASSESSED_NO_FEASIBLE_STRENGTH`, no fallback budget). For HQ / SHUF superseded by the brackets of finding 5; an empty control path (no S2 strength and no bracket) is compared at the identity, flag `<name>_AT_IDENTITY`. |
| C6 D3 / D4 at the S2 configuration | Adopted. |
| C7 ORD path and spend reporting | Adopted, with ORD_BR added to the path; realized spending of every control reported beside D2. |
| C8 effective refit definition | Adopted. |
| C9 QUAL disclosure | Adopted in its S2 form (finding 1). |
| C10 repair judged by the same rules | Adopted (with the POST_HOC label of finding 11). |
| C11 outcome classes | Adopted. |
