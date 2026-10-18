# Handoff to the manuscript and claims owners (file-based; overnight occupation-headroom campaign)

**Status.** No configuration qualified under the frozen nomination rule (NOMINATION.json, 07:45:00Z; replayed 34 / 34 by the
independent verifier).
- No finalist was locked and no reporting receipt was made. No ATTACK or DEV_REPORT label was read; only DEV_REPORT's label-free raw
  inputs were used, for the deployment checks.
- None of the final claims registered for a finalist (W, E, R, P, N, C, M, T, O in `roles/statistical_reviewer/CLAIMS_DRAFT.md`) was
  evaluated: each is **NOT_EVALUATED (no finalist)**. J stays NOT_CLAIMABLE and V NOT_CLAIMED by registration.
- Development-stage labels (SELECTION_RULE §7) are in RESEARCH_DECISION.md section 6.
- Everything below is development evidence on MODEL_SELECT and the development reader grid. Those records were used adaptively for
  selection and diagnosis, and they had been used before, so they are not fresh data.

## Statements the evidence supports (use the wording and the qualifiers)

1. **Occupation headroom (development).**
   - **Wording.** "On the development role, a gradient-boosted occupation learner that codes place of birth by its ordinal census
     code improved occupation NLL over the frozen teacher on both seeds (B2_SINGLE −0.0026 / −0.0031; A1_PO_16 −0.0040 / −0.0046;
     SE about 0.0017). It improved the men's guard from the teacher's −0.0010 / +0.0006 to about −0.004 / −0.002 (limit +0.005),
     with the women's guard near the teacher's level. With H3 income the complete release kept the 2-SE development reserve on all
     28 conditions."
   - **Qualifiers.**
     - The development role was used adaptively.
     - The two seeds are not independent fits. Family A's seed-1 model is bitwise the seed-0 boosting path truncated at the seed-1
       tree count (A1_PO_16: 154 of 168 trees; `checks/A_SEED_IDENTITY.json`). Family B's base learners are identical across seeds.
       So the witnesses' second seed mostly re-evaluates nearly the same model against seed-specific references.
     - The gain over the equally exposed TASK_REFIT is about 0.5-0.6 SE for B2_SINGLE and 1.4-1.5 SE for A1_PO_16.
     - The whole-release minimum is H3's seed-1 income accuracy (2.862 SE units). Its two clauses (against U0 and Ucal) are
       unresolved at the Bonferroni level.
2. **Coalition-aware occupation correction against matched controls, at the B80 level (development).**
   - **Wording.** "A critic-trained occupation correction that reads the linked income-occupation pair, with H3's income frozen,
     lowered pair recovery by the declared development readers beyond a matched continuation twin (critic weight zero) and a
     compression code at matched out-of-fold cost. At the B80 level it did so in all 8 grid cells on both witnesses and both seeds,
     under 18 of 18 fixed reader recipes per seed against the twin and 17-18 of 18 against the compression code (for example
     P2A_PAIR_B80 against its twin, +0.040 AUC)."
   - Call it a coalition-aware occupation correction, never a joint two-output design.
   - **Qualifiers.**
     - These are declared readers only, on a development grid of two household splits.
     - At B50 the margins change sign.
     - The B80 PAIR levels break the seed-0 men's reserve. The reserve-keeping local-critic arm P2_LOC_B80 also beats both controls
       in all 8 cells.
3. **Not superior to the inherited releases (development).**
   - **Wording.** "The correction did not lower recovery below the inherited JOINT_TREE releases (SUCC, RETIRED_B3CTL) on both seeds
     within the reserve. Its best seed-0 point (pair AUC 0.581) was not reproduced by the seed-1 refit (0.612)."
   - The best reserve-keeping configuration (P2_LOC_B80) stays 0.010 above SUCC.
4. **Budget transfer is the limiting engineering fact.**
   - **Wording.** "The out-of-fold budget projection mis-transferred the men's guard cost: +0.0016 to +0.0029 summed Brier above its
     projection on seed 0 and −0.0008 to −0.0012 below it on seed 1. On seed 0 that is 30-54% of the men's reserve slack (0.0054),
     so one nominal level over-spent on one seed and under-spent on the other. The witnesses' seed-1 men's slack was also smaller
     (0.0041)."
   - This is a diagnosis, not a tested remedy.
5. **Recipient gate structure.**
   - **Wording.** "On this grid, every H3-anchored release fails the per-cell recipient gate through H3's own income view: +0.031
     above H_LJ in two seed-0 cells (one household split, two reader seeds); seed 1 passes. This is consistent with split noise.
     Under a normal-noise model with the observed split SD (about 0.021), a true zero difference fails against one comparator with
     probability about 0.81."
   - This is a methodological observation; no rule was changed. All admissible configurations also fail the stability gate, so the
     outcome does not depend on the recipient gate.

## Statements that must not be made

- Any competitive, superior or "successor" claim, or any claim of meeting the 0.020 target. P2A_PAIR_B80's 8-cell means against
  H_LJ / NEURAL_RAW (+0.022 / +0.020) are a point-only development result; its seed-1 means are +0.003 / +0.0025.
- Any protection, privacy or fairness guarantee, protection against all readers, or information removal. A deterministic output is
  not assumed one-to-one, and a recalibration may change fitted readers without removing information.
- "Confirmed", "replicated" or "fresh". The two seeds are not replications, and the third household split R3 rotates previously used
  people.
- A joint two-output design (J) or algorithmic novelty (V). The ingredients are known: ordinal coding of a categorical, stacking,
  critic-trained corrections (an adaptation of conditional adversarial debiasing) and budgeted projection.
- That income accuracy was repaired. It was not: 2.840 vs H3's 2.862 on seed 1, and the task-only repair lowered HC_STRONG's 3.296 to
  3.144.
- Any DEV_REPORT or ATTACK number for this campaign. None exists.

## Numbers and where they live

| What | Where |
|---|---|
| Utility per configuration and seed | `UTILITY_FRONTIER.csv`, `EXPERIMENT_LEDGER.csv` |
| Grid benefits, labels and recipient maxima | `READER_STABILITY.csv`, `GRID_RECOVERY.csv`, `NOMINATION.json` / `NOMINATION_TABLE.csv` |
| Frontier | `NOMINATION_FRONTIER.csv` |
| Fixed-recipe and probe readers | `roles/reader_robustness/RR_FINAL__CANDIDATE_TABLE.csv` (not independently recomputed) |
| Cost decompositions | `decomposition/` |
| Independent recomputation | `roles/verifier/CHECKS.json` |
| Claims review | `roles/claims_review/REPORT.md` |

## Disclosures a manuscript must carry

Item numbers refer to PROTOCOL_DEVIATIONS.md.
- **Timing and code provenance.**
  - The P2 card was committed 71 s after its first fit (10).
  - Protection fits ran on an uncommitted file whose P2 path was unchanged (14).
  - Family B records no code hash, and its code was committed after its first fits; four family-C code hashes are absent from git
    (17).
- **Verification.** The verifier widened its own access-check rule after seeing FAIL results; the platform's safety monitor flagged
  this, and it was restored in the verifier's records (15-16).
- **Label reads.** Logged diagnostic MODEL_SELECT reads went outside the two registered entry points (9, 13, 19). The scorer reads
  MODEL_FIT SEX for a diagnostic (4).
- **Adaptive choices.**
  - Families A and B's ordinal-POBP result was relayed to family C before C's revision 2 (5).
  - A family-A diagnostic setting changed after a MODEL_SELECT read (12).
  - Family B chose its revisions after MODEL_SELECT feedback.
  - The P2 / P3 budgets and weights were set from logged MODEL_SELECT reserve slacks.
  - The income branch was DEV-guided through the source's published DEV value (registered leg B).
- **Records and restrictions.**
  - One agent's closing text was cut short by a platform safeguard (20).
  - Two family-A record texts carry errata (18).
  - A known receipt-tool defect was not exercised (21).
  - 61 of the 201 admitted copies have no expected hash, and 59 of those are not pinned by any committed source record. The H3+O1 /
    TW+O1 references rest on the source's private copy (verifier note D3).

Questions about this handoff go to the campaign record (RESEARCH_DECISION.md, PROTOCOL_DEVIATIONS.md), not to individuals.
