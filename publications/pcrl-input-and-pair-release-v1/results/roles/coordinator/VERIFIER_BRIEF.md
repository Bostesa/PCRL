# Verifier brief (coordinator, campaign `pcrl_input_and_pair_release_v1`)

You are the independent verifier. Purpose of the campaign: a statistical machine-learning benchmark on admitted public-use ACS 2019
Florida records; two task recipients (income, occupation) receive probability vectors; the privacy evaluation measures whether
classifiers can infer the recorded SEX column from them. Families: IA (shared SEX-aware field generalization, SFG: both task models
refitted on a coarsened record), IB (coordinated keyed selective flattening, CSF: SEX-blind), and possibly the replacement family
FS_SWAP (cross-recipient keyed swap); the shared twin CTL_TWIN; shared SPEND (DQ) and RO (generic release-only tilt) points saved under
CTL and named per version by role members (`results/pcrl_input_and_pair_release_v1/paths/SHARED_ROLES.json`). Decisions follow the
frozen IPR1 rules (`roles/statistical_reviewer/RULES.md`, `ipr/rules.py`).

Independently reproduce the decisive numbers with your OWN code (do not import `ipr.evaluate`, `ipr.devreaders`, `ipr.rules` or
`ipr.claims` for the recomputation; you may import `ipr.common` for the logged label access, `ipr.release` to load saved releases and
the family modules' `predict` functions to recompute releases):

1. **Release integrity.** For each saved configuration (`.private/pcrl_input_and_pair_release_v1/runs/CONFIG_LEDGER.jsonl`), the
   RELEASE.npz hash equals its RELEASE.json; the deployment function (`ipr.ia.predict`, `ipr.ib.predict`, `ipr.ctl.predict`, and the FS
   family's if any) recomputes the MODEL_SELECT arrays bitwise from admitted features.
2. **Utility.** Recompute the 24 clauses and 4 guards for the twin, each family's CAND and its control points (role members resolved per
   version) from MODEL_SELECT labels (accuracy loss, NLL excess, summed Brier excess against U0_S<v> and Ucal_S<v> for both tasks, men's
   and women's occupation summed Brier excess vs U0) and compare with the saved `EVAL_MS.json` records (tolerance 1e-9 on point values).
3. **Reader grid.** For the registered bank's decisive comparisons (each CAND vs TWIN and its bracketing control points, and vs
   NEURAL_RAW / H_LJ): recompute per-cell SCORE AUCs of the recorded winners where possible, or refit at least two cells per comparison
   with your own simple readers (logistic regression on log-probabilities; a gradient-boosting reader) to check the direction and size of
   the differences; state which numbers are exact replays and which are independent approximations.
4. **Decisions.** Re-derive the D2 conjuncts (pair benefit aggregate and se_agg, recipient gates, the bracket choice from paired total-NLL
   costs, the cheaper-control check, targeting) from the grid records with your own arithmetic and compare with `claims/*/DECISIONS.json`;
   check the card declarations the rules cannot verify (IA: `ro = GENERIC_ZTILT` as declared fallback and the recorded substituted
   ladder; IB: `operation_sex_blind`, which you can check by confirming the CSF release depends on no SEX-fitted quantity, and the D5
   collapse declaration).
5. **Access.** Check that every label read in `.private/pcrl_input_and_pair_release_v1/access/LABEL_ACCESS.jsonl` happened after the
   registration receipt (02:14:15Z), that fitting reads carried a committed card, that no MODEL_SELECT read came from a fitting entry
   point, and that the first MODEL_SELECT read followed the commit of the IA and IB cards (9b70c5d77, 02:38:59Z).

Run every numerical job through the queue (`python3 -m ipr.jobqueue run --unit IPR_VER_<name>_V<n> --stage verify --rss-gib 2 --
python3 <your script>`), with your scripts under `results/pcrl_input_and_pair_release_v1/roles/verifier/code/` (label reads use
`ipr.common.load(role, fields, 'verify', purpose)` from there). Keep CPU under 0.6 CPU-hours. You may not be able to write report files:
if the harness refuses, return the full report text (CHECKS table: check, expected, observed, PASS/FAIL) in your final message.
