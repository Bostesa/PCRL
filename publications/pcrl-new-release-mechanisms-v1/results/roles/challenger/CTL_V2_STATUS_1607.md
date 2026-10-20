# Challenger, CTL_REM runner and card v2 status (saved verbatim with attribution)

Author: the challenger of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 16:07Z on 2026-10-20,
before any output of the registered grid NRM1_MAIN was read. Saved verbatim; nothing below was edited.

Coordinator note (not part of the report): the flagged deployment point is already handled. `nrm.deploy.build` keys any
model path under the campaign runs folder as `NRM::` and any other repository file as `REPO::` (nrm/deploy.py, the
model_paths loop in `build`), for every family, so CTL needs no change.

---

Items 1 and 2 are ready and committed. The CTL_REM fits have not run yet, because both plan units are waiting in the queue.

**Item 1: `nrm/ctl/run.py` and `nrm/ctl/release.py`** (commit `bcd9df4fd`, plus fix `ac488d353`). Invented-data selftest: `python3 -m nrm.ctl.run selftest` → ok, 14/14; `nrm.ctl.post` is still 39/39. The runner has three steps:
- **plan** (queue unit; the label read): reads CALIBRATION occupation labels only, with operation 'calibrate' and card `CTL_CARD_v2.md`.
  - It fits SEX-blind remainder shares on FB2_BASE's CALIBRATION occupation vectors.
  - It predicts each ladder level's total-NLL spend over FB2_BASE by 5-fold household cross-fitting on CALIBRATION, with household-clustered SE.
  - It saves no release. It reads no SEX label and no MODEL_FIT or MODEL_SELECT label.
- **choose** (label-free): the spend-only rule. Each target gets the smallest strength whose predicted spend is at or above it in both versions. An unreachable target gets the strongest level, reported as NOT_SEPARATED on this path.
- **fit** (queue unit, label-free): saves family CTL, config `CTL_REM_<level>`, kind `control`, for all six roles.
  - Income is FB2_BASE's, unchanged.
  - Occupation is the remainder flattening, with decision and top probability kept.
  - The parameter file is write-once.

Deployment works through `nrm.ctl.release`:
- `nrm.deploy.family_module('CTL')` resolves to it, providing `predict` (deterministic; keys and any draw other than 0 are refused), `model_paths` and `CODE_MODULES`.
- `model_paths` is the CTL params file plus FB2_BASE's own model paths, and `CODE_MODULES` includes family B's modules.
- The selftest showed `predict` reproduces the saved release bitwise from features, row-independently.
- **One thing for you to check:** FB2_BASE's model paths include files outside the campaign runs folder. `nrm.deploy.build` currently requires paths under RUNS, so it needs the same handling for CTL as for FB2.

**Item 2: `cards/CTL_CARD_v2.md`** (commit `485a329ca`, before any CTL read or fit). It follows the ten sections of `cards/README.md`.
- **Strength ladder** (predeclared): 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00.
- **Targets:** your per-version total-NLL spends over FB2_BASE, recorded in `roles/challenger/CTL_REM_TARGETS.json`.

  | Point | V0 | V1 | Pooled |
  |---|---|---|---|
  | FB2_SEX_S3 | 0.0072 | 0.0076 | 0.0074 |
  | FB2_SEX_S5 | 0.0216 | 0.0237 | 0.0226 |

- **Rule:** "at or just above" each target, with no SE margin as you asked. The predicted SE will be reported beside each choice, because REM's per-person cost varies a lot and realized spend could fall below a target. If both targets pick the same level, the next lower level is added. At most 2 levels × 2 versions = 4 configurations, with the same levels in both versions.

**Item 3, status.** Units CTL_REM_PLAN_V0 (queued 15:54Z) and CTL_REM_PLAN_V1 (queued 16:05Z) are waiting for slots. Two formal-grid jobs (`nrm.devreaders`) have held both slots since 15:52Z. A monitor will bring me back when the plans are written. Then I will run choose and the two fit units, and message you with the saved configuration ids and their predicted spends. If I don't report back, send me a message to resume.

CPU charged so far: 30 CPU-s for the challenger. The plan and fit units are charged by the queue.
