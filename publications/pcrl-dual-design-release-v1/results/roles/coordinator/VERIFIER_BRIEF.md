# Verifier brief (coordinator, campaign `pcrl_dual_design_release_v1`)

You are the independent verifier. Purpose of the campaign: a statistical machine-learning benchmark on admitted public-use ACS 2019
Florida records; two task recipients (income, occupation) receive probability vectors; the privacy evaluation measures whether
classifiers can infer the recorded SEX column from them. Two release families were fitted (DA: SEX-standardized task posteriors; DB:
propensity-tilt cancellation) with matched controls; decisions follow the frozen DDR1 rules.

Independently reproduce the decisive numbers with your OWN code (do not import `ddr.evaluate`, `ddr.devreaders`, `ddr.rules` or
`ddr.claims` for the recomputation; you may import `ddr.common` for the logged label access and `ddr.release` / `ddr.da.std` /
`ddr.db.ptc` to load saved releases or models):

1. **Release integrity.** For each saved configuration (`.private/pcrl_dual_design_release_v1/runs/CONFIG_LEDGER.jsonl`), the
   RELEASE.npz hash equals its RELEASE.json; the deployment function (`ddr.da.predict` / `ddr.db.predict`) recomputes the
   MODEL_SELECT arrays bitwise from admitted features.
2. **Utility.** Recompute the 24 clauses and 4 guards for DA_TWIN, DA_STD, DB_PTC and their control points from MODEL_SELECT labels
   (accuracy loss, NLL excess, summed Brier excess against U0_S<v> and Ucal_S<v> for both tasks, men's and women's occupation summed
   Brier excess vs U0) and compare with `runs/<F>/<C>/EVAL_MS.json` (tolerance 1e-9 on point values).
3. **Reader grid.** For the registered bank's decisive comparisons (each candidate vs TWIN and its bracketing controls, and vs
   NEURAL_RAW / H_LJ): recompute per-cell SCORE AUCs of the recorded winners from the saved bootstrap-free records where possible, or
   re-fit at least two cells per comparison with your own simple readers (logistic regression on log-probabilities; a gradient
   boosting reader) to check the direction and size of the differences (descriptive cross-check; state clearly which numbers are
   exact replays and which are independent approximations).
4. **Decisions.** Re-derive the D2 conjuncts (pair benefit aggregate and se_agg, recipient gates, the bracket choice from paired
   total-NLL costs, the cheaper-control check, targeting) from the grid records with your own arithmetic and compare with
   `claims/*/DECISIONS.json`.
5. **Access.** Check that every label read in `.private/pcrl_dual_design_release_v1/access/LABEL_ACCESS.jsonl` happened after the
   registration receipt (17:42:42Z), that fitting reads carried a committed card, that no MODEL_SELECT read came from a fitting entry
   point, and that the first MODEL_SELECT read followed the commit of both cards (5f2fd02a9, 18:06:14Z).

Run every numerical job through the queue (`python3 -m ddr.jobqueue run --unit DDR_VER_<name>_V<n> --stage verify --rss-gib 2 --
python3 <your script>`), with your scripts under `results/pcrl_dual_design_release_v1/roles/verifier/code/` (label reads use
`ddr.common.load(role, fields, 'verify', purpose)` from there). Keep CPU under 0.5 CPU-hours. You may not be able to write report
files: if the harness refuses, return the full report text (CHECKS table: check, expected, observed, PASS/FAIL) in your final message.
