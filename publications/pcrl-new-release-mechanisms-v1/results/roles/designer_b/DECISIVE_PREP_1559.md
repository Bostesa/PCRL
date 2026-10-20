# Designer B, decisive-phase preparation (saved verbatim with attribution)

Author: Designer B (family B) of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 15:59Z on 2026-10-20.
Saved verbatim; nothing below was edited.

---

DESIGNER B (15:59Z): both decisive-phase pieces are coded and selftested, and the code is committed. I ran no fit and did not commit the card. No sub-agents; one thread.

**Code commit 2fa4e64ac** (`nrm/fb/posthoc.py`, `nrm/fb/release.py`)
1. **MEAN (C5).** `fit_mean2` / CLI `fit-mean --version V --of FB2_SEX_S<j> --card ...` builds FB2_SEX_S<j>_MEAN, kind 'ablation'.
   - It is the exact deterministic mean E[emitted] of the fitted FB2_SEX_S<j> channel: same frozen geometry, temperatures, cap and s, read from its write-once CHANNEL file (SHA recorded). Computed by 48-node Gauss-Hermite on the frozen base. R = 1; keys are refused.
   - It reads no label and fits nothing new (0 calibrator and 0 critic fits). The base bitwise gate still applies.
   - It refuses if the source channel is not fitted.
   - Selftest: equals the exact quadrature mean of the fitted SEX channel bitwise; within 0.0033 of a 4,000-draw Monte Carlo average.
2. **Training perturbation, revision FB2P.** Configurations FB2P_SEX_S3/S5 and FB2P_MINF_S3/S5, kind 'perturbation'.
   - SEX geometry: re-estimated from a household bootstrap of MODEL_FIT, seed [7300, v, 0].
   - Temperatures: re-estimated from a household bootstrap of CALIBRATION, seed [7300, v, 1]. SEX and MINF share the same resample (equally perturbed); MINF has no geometry to resample.
   - Release key: a fresh write-once root at `runs/FB/MODELS/FB2P/V<v>`, so the draws are also fresh.
   - Seeds and sizes (households, drawn, distinct; people, distinct) are recorded in each record and CHANNEL file.
   - Limitation, stated in every record: the frozen base models are not refitted.
   - One revision per fit unit is enforced. Run with `fit --configs FB2P_SEX_S5,FB2P_MINF_S5`.
   - `release.py` now has revisions FB1 / FB2 / FB2P. predict/model_paths dispatch FB2P to its own folder and root.

**Selftests**
- posthoc: 30/30 under python3 and under the torch venv.
- learner: 20/20.
- channel: 27/27.
- Label-free spot check after the layout change: real FB1 releases (FB_TWIN, SEX_S2, ISO_S2, MEAN) and real FB2 releases (BASE, SEX_S5, MINF_S3) still reproduce draw 0 bitwise on DEV_REPORT, both versions.

**Card v3 draft (uncommitted):** `results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v3.md`
- MEAN strength rule (C5): the highest FB2_SEX strength that is TASK_FEASIBLE in both versions AND DOMINATES_CONTROL_FRONTIER against both the MINF and CTL_REM paths; otherwise MEAN is not fitted.
- FB2P strength rule: the highest FB2_SEX strength that is TASK_FEASIBLE in both versions. Your 15:53Z utility read gives S5 (0.21). FB2_MINF_S5's v1 failure is reported but does not change the rule; if the grid's utility record differs, the rule is applied to the grid record.
- Configurations: MEAN +2 (conditional), FB2P +4; order after NRM1_MAIN.
- Proposed descriptive reading (PERTURBATION_CONSISTENT / NOT_CONSISTENT / NOT_RUN): never decisive, and the statistical reviewer must confirm or replace it, because RULES.md has no perturbation label.
- Predictions P6b and P7b, the FB2 label and scope sentences, and the limitation.

**CPU** charged out of queue under designer_b: +45 s (upper estimate).

Ready to commit v3 and fit on your GO.
