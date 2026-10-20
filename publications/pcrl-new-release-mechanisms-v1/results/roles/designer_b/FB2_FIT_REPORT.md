# Designer B, FB2 fit report (saved verbatim with attribution)

Author: Designer B (family B) of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 15:49Z on 2026-10-20
(written 15:48Z). Saved verbatim; nothing below was edited. Its flag (b) about the instance map refers to an older version of
nrm/devreaders.py: the locked grid code (1128afd67) already maps R3 -> 6 and R4 -> 7 and allows the offsets on R1/R2 only.

---

DESIGNER B (15:48Z): the 10 FB2 releases are saved and ready for you to evaluate FB2_BASE first, under C3. No sub-agents, one thread throughout, and no MODEL_SELECT label read by family B.

**Order followed: code, then card, then fit**
- **Code (committed before the card):**
  - d14ce4526: `nrm/fb/posthoc.py`, `release.py`, `learner.py`.
  - 0a8b2de9d: FB2_BASE emits the exact base.
  - ENG1-08 fixed:
    - one model folder and one write-once root key per (revision, version): `runs/FB/MODELS/<rev>/V<v>`;
    - FB1 keeps the layout it was fitted with (`MODELS/V<v>`);
    - configurations are selected by revision (`nrm.fb.release.revision_of`);
    - the revision tag is written in every FB2 `CHANNEL_<cid>.json`;
    - `predict` / `model_paths` dispatch FB2 to `nrm.fb.posthoc`.
  - ENG1-13 fixed: the learner docstring now says joint from-scratch learners and cites V3X_FB_INVARIANCE for the SEX-blind claim.
  - Selftests: posthoc 21/21 under python3 and under the torch venv; learner 20/20; channel 27/27.
- **Card:** `cards/FB_FSC_v2.md`, commit b163f3233 at 15:45:14Z (SHA-256 3f727693…). It declares everything you and the final ruling (C1-C7) required:
  - twin FB2_BASE, with the C3 gate first;
  - G2: MINF_S3 for SEX_S3 and MINF_S5 for SEX_S5;
  - MINF_* recorded as NO_MATCHED_CONTROLS and not selectable;
  - G3 paths: MINF, and CTL_REM on FB2_BASE (up to 4, spend-only, under the challenger's CTL v2 card);
  - strengths {0.12, 0.21}; contingent MINF_S6 at 0.26 and MEAN (C5);
  - R = 8, the instance map R1→0, R2→1, R3→6, R4→7, and MC members at +2/+4 for all four randomized configurations (S5 pair required);
  - F1 per C6; G6 NOT_ASSESSED_MISSING_CONTROLS; G7 NOT_CLAIMED;
  - the scope sentence and the FB2 label sentence, plus the C1 deviation;
  - root-key custody and deployed randomness;
  - SEX-labelled exposure;
  - the updated closest prior work;
  - predictions P1b-P5b.
- **Fit:** FB2_FIT_V0 (exit 0, 20.0 CPU-s) and FB2_FIT_V1 (exit 0, 19.9 CPU-s), under python3. Configurations saved per version: FB2_BASE, FB2_SEX_S3, FB2_SEX_S5, FB2_MINF_S3, FB2_MINF_S5. The ledger shows 10 FB2 entries, so family B now holds 28.
- **Label reads (4, all with the v2 card, 15:45:31Z):** per version, MODEL_FIT `fit` ['sex'] (20,000 people) and CALIBRATION `calibrate` ['income', 'occupation'] (2,982 people). No MODEL_FIT task label was read.

**Gates and checks (label-free)**
- **Base gate (inside the fit):** the base computed from features (`ihs.fam_h.predict` HC_STRONG; `ohn.fam_a.predict` A1_PO_16) is bitwise equal to ohn's hash-checked stored releases for all 6 roles in both versions. The fit refuses before writing anything otherwise.
- **Deployment parity (out of queue, 234 CPU-s, charged):** `nrm.fb.release.predict` reproduces draw 0 bitwise for all 10 configurations × 6 roles.
  - FB2_BASE is an exact alias of the stored base on every role: TWJ's income and W's occupation (C3).
  - Every `model_paths` file exists: 22 files for deterministic configurations, 23 for randomized.

**Design decisions to note**
1. **FB2_BASE is the frozen base emitted unchanged.** It has no temperature, reads no CALIBRATION labels, and skips any log/softmax round trip, so that C3's exact aliases hold. Only the noisy SEX and MINF channels get CALIBRATION temperatures, matched between the two paths.
   - This is a second use of CALIBRATION income: HC_STRONG's manifest shows it used CALIBRATION income for its stack and Platt map. Disclosed in the card and in the records.
2. **Out-of-fold geometry (A7).** The occupation part of δ and Σ_w uses A1_PO_16's admitted CROSSFIT `MODEL_FIT__oof`; its decision agreement with the deployed A1 is recorded. HC_STRONG has no out-of-fold bank (RELEASE.npz holds per-role arrays only), so income is in-sample. Disclosed.
   - Wording slip in the committed card: it says HC_STRONG's STEP1.json holds "CV metrics only". The file actually holds CV fold/grid/selection metadata and no per-person arrays. The claim itself is unchanged.

**FIT-TIME DIAGNOSTICS** (MODEL_FIT, label-free self-expected cost KL(base‖emitted) per task, and one-draw linked Δ² with SEX; not evidence for any verdict). Values are V0 / V1:

| Configuration | Income KL | Occupation KL | Linked Δ² |
|---|---|---|---|
| BASE | 0 / 0 | 0 / 0 | 0.288 / 0.288 |
| SEX_S3 | 0.0026 / 0.0026 | 0.0053 / 0.0054 | 0.236 / 0.230 (−18% / −20%) |
| SEX_S5 | 0.0087 / 0.0088 | 0.0144 / 0.0146 | 0.183 / 0.174 (−36% / −40%) |
| MINF_S3 | 0.0001 / 0.0002 | 0.0081 / 0.0082 | 0.268 / 0.271 |
| MINF_S5 | 0.0005 / 0.0006 | 0.0238 / 0.0240 | 0.257 / 0.246 |

- MINF puts almost all of its spend on occupation. MINF_S5's occupation spend (about 0.024 nats self-expected) is close to the base's occupation slack (0.026 / 0.023 vs U0), so the MINF S5 point may hit the occupation NLL cap or the men's guard.

**Flags for you**
- **(a) nrm.deploy.build.** It keys `model_paths` relative to this campaign's `runs/`, but the FB2 base sources live outside it: the ihs mirror for HC_STRONG and the stage files, the ohn runs for A1_PO_16, and two governance JSONs. Build needs an external-path key, for example OHN:: / MIRROR:: / repo-relative. `nrm.deploy.store` (label-free) should work as is.
- **(b) Instance map.** `nrm.devreaders` REP_INDEX is R1→0, R2→1, R3→2, R4→3, and it refuses any draw ≥ R. The declared FB2 map (R1→0, R2→1, R3→6, R4→7) plus offsets +2/+4 only fits R = 8 if the members exist for the R1/R2 blocks alone (instances 2-5). The card records this; please align devreaders before the FB2 grid.

**Files**
- Code: `nrm/fb/posthoc.py`, `nrm/fb/release.py`, `nrm/fb/learner.py`, `nrm/fb/channel.py`.
- Card: `results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v2.md`.
- Model files: `.private/pcrl_new_release_mechanisms_v1/runs/FB/MODELS/FB2/V{0,1}/`.
- Releases: `runs/FB/FB2_*/V{0,1}/`.

CPU charged out of queue under designer_b: 40 + 25 + 10 + 260 s (upper estimates). Queue units for FB: about 56 s (FB_FIT) plus 40 s (FB2_FIT).
