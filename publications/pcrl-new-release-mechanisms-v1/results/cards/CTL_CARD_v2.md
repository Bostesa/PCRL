# CTL_CARD_v2: CTL_REM on FB2_BASE (family-specific control card)

Author: challenger, 2026-10-20 (about 15:55Z). Code: `nrm/ctl/run.py`, `nrm/ctl/post.py`, `nrm/ctl/release.py`.

- **Lineage.** It specializes CTL_CARD_v1 (generic) for FB2, as the statistical reviewer's ruling RULING_FB_R1_FINAL C7 requires.
- **Role.** It is a second SEX-blind G3 path, next to Designer B's own declared path.
- **Commit timing.** It is committed before any CTL label read and before any CTL fit.

## 1. Operation

CTL_REM post-processes the saved deterministic task-only base **FB/FB2_BASE** (both versions; HC_STRONG income + A1_PO_16
occupation). FB2_BASE passes 24 clauses + 4 guards, per the coordinator at 15:52Z.

- **Income:** the FB2_BASE income vector, unchanged.
- **Occupation:** `flatten_remainder(FB2_BASE occupation, R_v, s)`.
  - Every argmax and its probability are kept.
  - The non-argmax remainder moves toward SEX-blind class shares R_v[argmax], with weight min(s, cap). The cap keeps every
    decision.
- **Decisions:** the argmax of the emitted vectors, so both recipients' decisions equal FB2_BASE's.
- **Determinism:** the release is deterministic, with one release per person (R = 1); keys are refused.
- **Transmitted:** each recipient's float64 probability vector and argmax decision, and nothing else.

## 2. Closest prior PCRL attempts, and what differs

- **Decoders and calibration** (`pcrl_heldout_calibration_v1`, `pcrl_adult_learned_decoder_release_v1`): an invertible decoder
  or temperature change removes no information. REM is many-to-one in the low-probability log-odds, so it is not invertible.
- **Decision-preserving finite codes** (`pcrl_decision_preserving_compression_v1`): those quantized the whole vector and paid
  about 0.02 nats of occupation NLL. REM keeps the top probability exactly and acts only on the remainder.
- **SEX-blind coarsening** (SCM's ORD): REM is SEX-blind in the same way, but it keeps decision and confidence.

Bottleneck addressed: FB2's rationale places SEX information in low-Fisher directions, i.e. the log-odds among unlikely
occupations. REM removes exactly that information deterministically and SEX-blind. It is the simplest explanation of a FB2
benefit at matched spend.

## 3. Why (control role)

If REM reaches FB2_SEX_S3's or FB2_SEX_S5's recovery at no more total-NLL spend, the SEX-targeted channel's benefit is explained
by SEX-blind, deterministic removal of remainder information. This card makes no protection claim for CTL_REM.

## 4. Task-only twin

FB2_BASE, saved by Designer B as the family-B twin. CTL_REM never refits it. The CTL parameters are only the remainder shares R_v.

## 5. Simplest controls

This card declares one SEX-blind G3 path. The other declared SEX-blind path is Designer B's.

## 6. Falsifying or invalidating outcomes (control role)

- **Decision-reader check.** Decisions equal FB2_BASE's by construction. If a decision-only reader (DEC_LR) reads CTL_REM
  differently from FB2_BASE beyond reader noise, the reader pipeline is investigated.
- **Bracketing.** If the realized MODEL_SELECT spend of the fitted levels does not bracket a target:
  - S5 unreachable at the strongest level reads NOT_SEPARATED on this path, as declared.
  - A realized under-shoot is recorded as INCOMPLETE per reviewer C6.
  - No further level is fitted.

## 7. Fitting plan

**Labels:** CALIBRATION occupation only, operation `calibrate`, through `nrm.common.load` with this card, inside queue units.
- No SEX label.
- No MODEL_FIT label.
- No MODEL_SELECT label.

**Remainder shares R_v:** `fit_remainder_shares(FB2_BASE CALIBRATION occupation, y_occupation)`, Jeffreys α = 0.5, on all
CALIBRATION people.

**Spend prediction:**
- Out-of-fold total NLL of CTL_REM minus FB2_BASE, in nats per person. Income is unchanged, so total = occupation.
- Computed on CALIBRATION by 5-fold household cross-fitting (fold tag `NRM_CTL_REM_cv_S<v>`), with household-clustered SE.
- The occupation summed-Brier spend is also recorded.
- It is label-only and uses no SEX.

**Ladder (predeclared):** s ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00}.

**Targets (spend only).** These are the realized MODEL_SELECT total-NLL spends of the SEX points over FB2_BASE, as recorded by
the coordinator in `roles/coordinator/FB2_SPEND_DECISIONS.md`:

| Point | V0 | V1 | Pooled |
|---|---|---|---|
| FB2_SEX_S3 | 0.0072 | 0.0076 | 0.0074 |
| FB2_SEX_S5 | 0.0216 | 0.0237 | 0.0226 |

**Choice rule** (`nrm.ctl.run.choose`; spend only, never recovery):
- For each target, take the smallest ladder level whose predicted CALIBRATION spend is at or above the target in both versions:
  "at or just above", with no SE margin (Z_BRACKET = 0).
- A target that no level reaches gets the strongest level (1.00). It is reported as unbracketed; S5 then reads NOT_SEPARATED on
  this path.
- If both targets give the same level, add the next lower ladder level.
- At most 2 levels, and the same levels in both versions.

**Sizes and seeds.**
- Actual training size of the CTL parameters: all CALIBRATION people (2,982) and their households.
- There are no random seeds; the folds are deterministic hashes.

**Execution:**
1. Queue units CTL_REM_PLAN_V0 and CTL_REM_PLAN_V1 (stage fit; the label read; no release saved).
2. `python3 -m nrm.ctl.run choose --targets <targets JSON>` (label-free).
3. Queue units CTL_REM_FIT_V0 and CTL_REM_FIT_V1 (stage fit; label-free). They save through `nrm.release.save`:
   - family CTL;
   - config_id `CTL_REM_<level>`, e.g. CTL_REM_0p40;
   - kind `control`;
   - all six roles.

## 8. Readers

The coordinator's formal grid and screening split, with the same recipes, splits, cells and seeds as FB2.
- The release is continuous and deterministic.
- The decision views are identical to FB2_BASE's.

## 9. Budget

- At most 2 levels × 2 versions = 4 configurations (reviewer C7; coordinator ruling Q1).
- CPU:
  - plan: under 60 CPU-s per version;
  - fit: under 30 CPU-s per version;
  - plus grid readers.

## 10. Revision plan and verdict rules

**Revisions:**
- No other strengths, and no level change after any MODEL_SELECT recovery is seen.
- A fit that fails a check is diagnosed and refit under a new configuration id only with the coordinator's approval.

**Verdict rules:**
- The frozen G3 rule of `nrm/rules.py`: DOMINATES against each declared SEX-blind path.
- Frontier reading with `nrm.ctl.frontier` conventions.
- Unbracketed reads NOT_SEPARATED, as declared. INCOMPLETE follows C6.

**Deployment:** `nrm.ctl.release` provides `predict`, `model_paths` (CTL_PARAMS.npz plus FB2_BASE's model paths) and
`CODE_MODULES` (`nrm.ctl.*` plus family B's modules).
