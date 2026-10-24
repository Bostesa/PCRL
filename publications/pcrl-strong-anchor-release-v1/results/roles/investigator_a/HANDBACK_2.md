# Investigator A second hand-back, after the coordinator's adjudication (saved verbatim by the coordinator, about 05:08Z on 2026-10-24)

## Investigator A hand-back: FA_TFR revised under your adjudication

All seven required changes are made in `sar/fa/tfr.py` and `CARD_DRAFT.md`. The selftest passes all 24 checks (19.1 CPU-s). Revised 05:05Z. Still label-free: no label file opened, nothing read from MODEL_SELECT, no git, no shared code touched.

**Before the first fit: the pins for SUCC's 7 frozen files per version are not in this campaign's `SOURCE_ADMISSION.json` yet.** A new label-free `preflight` checks them; `fit` runs it first and refuses before any label read. It currently exits 1 for both versions.
- The current hashes, for you to pin from your own admission process, are in `results/pcrl_strong_anchor_release_v1/roles/investigator_a/preflight_V0.json` and `preflight_V1.json`.
- Any key ending in the file's last three path components (for example `H3_STRONG_LJDIR_PB/S0/STATE.json`) holding `{'sha256': ...}` is found.
- The TASK.pkl pins are already found under `source_private_artefacts`.

### The changes, by item
1. **CALIBRATION exposure is corrected.** H3 (STRONG stack, β budget, Platt map) and Ucal's temperatures are in-sample on CALIBRATION; only the occupation side (JOINT_TREE), the HGB teachers and the residual are out of sample. Income-side CALIBRATION spends over SUCC are disclosed as biased in SUCC's favour (conservative for CAND), with no honesty claim. This is stated in the card, the module docstring and every record. I also added a dated correction note at the top of PROPOSAL.md.
2. **CTRL is the λ = 0 path FA_TFR_TASK with origin START.**
   - The path is S_PATH = {0, 0.05, …, 1}. s = 0 returns SUCC's own vectors exactly, checked bitwise in both the path function and `predict`.
   - **LO** = the point with the largest CALIBRATION total-NLL spend over SUCC that is ≤ CAND's.
   - **HI** = the point with the smallest spend ≥ CAND's + 2 SE, using the clustered SE of that point's paired difference against CAND.
   - Flags: `CONTROL_PATH_MAY_NOT_BRACKET` when no HI exists, plus `CONTROL_PATH_NO_LO_POINT` when no LO exists.
   - Comparisons are on spend values, not on s, and ties go to the smaller s. The selftest covers an improving CAND, a spending CAND (LO = START, HI missing), no LO, and a path shuffled so spend is not monotone in s.
   - The closest-point match is gone everywhere.
3. **CONT** is FA_TFR_CONT (λ = 0, s = 1), declared gating in the record and the card.
4. **Specifications: at most 8** (CAND 2 + CTRL 4 + CONT 2), stated in the card.
5. **`results/pcrl_strong_anchor_release_v1/fa/FIT_V<v>.json`** holds per version:
   - CAND's config, ρ\*, status and its CALIBRATION total-NLL spend over SUCC (value, SE, all components);
   - CTRL's origin, selection data, every S_PATH point's spend and SE against CAND, the LO/HI ids with spends, and the flags;
   - CONT's config;
   - the per-ρ selection table;
   - the TASK.pkl and frozen-file hashes.
6. **Card statements added:**
   - CALIBRATION income and occupation labels (never SEX) are read only for selection part (a) and the bracket, with identical exposure for the controls;
   - the usefulness result is MODEL_SELECT with unchanged caps;
   - TASK.pkl is pinned from this campaign's SOURCE_ADMISSION. `task_cache` now reads only that file.
7. No shared-code change was needed.

### Three choices of mine you may want to overrule
- **Guard reference point.** Part (b) of the strength rule now compares each ρ against its LO point instead of the closest point (s = 1 if no LO). It is the conservative reference: at least as useful as CAND. This may lower ρ\* slightly.
- **Endpoint aliasing.** If LO or HI lands at s = 0 it is recorded as an alias of START; at s = 1, as an alias of FA_TFR_CONT. Neither is saved twice, so the spec count can fall below 8. Set `ALIAS_ENDPOINTS = False` in `tfr.py` to save duplicates instead.
- **Frozen-file check at deployment.** `predict` now refuses if SUCC's frozen files differ from the hashes recorded at fit time.

### Predictions
The registered table is unchanged. A dated 05:04Z note in the card adds:

| Quantity | Prediction |
|---|---|
| Pair benefit over LO | +0.015 (80% interval −0.003 to +0.033) |
| Pair benefit over HI | +0.008 (80% interval −0.008 to +0.025) |
| Resolved over both | 0.20 |
| `CONTROL_PATH_MAY_NOT_BRACKET` in at least one version | 0.35 |
| No LO point in at least one version | 0.10 |
| At least one CTRL point is an alias | 0.45 |
| ρ\* ≤ 1 | about 0.65 (was 0.60), from the guard reference change |

### CPU
About 140 of my 600 CPU-s used in total. The estimate per fit is unchanged at about 5-7 CPU-min per version.

Run deploys with the torch venv for bitwise parity, since the residual's matrix products may differ in the last bits under plain `python3`'s numpy.

Files are in `<repository>/.claude/worktrees/pcrl-strong-anchor-release-v1/`:
- `sar/fa/tfr.py`
- `sar/fa/__init__.py`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/CARD_DRAFT.md`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/PROPOSAL.md`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/selftest_output.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/preflight_V0.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/preflight_V1.json`
