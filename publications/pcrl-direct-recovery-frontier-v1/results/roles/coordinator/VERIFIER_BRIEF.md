# Verifier brief (campaign `pcrl_direct_recovery_frontier_v1`, DRF1; coordinator, written 2026-10-25 from 03:01Z)

You are the independent verifier. Use SEPARATE code that you write under `results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/`
(the only verifier entry point the access layer authorizes: operation 'verify', `drf.common.load(role, fields, 'verify', purpose)`,
roles MODEL_FIT and MODEL_SELECT; every read is logged; reads run only inside queue units: `python3 -m drf.jobqueue run --unit
DRF_VER_<name>_V<k> --stage verify --rss-gib 2 -- python3 results/.../roles/verifier/code/<script>.py ...`). Do not import the
campaign's metric, objective, solver, rules or reader code to compute what you check (you may import `drf.common` for data access,
`drf.release.load` / the frozen loaders to read saved arrays, and the admitted reader recipes `ffv/readers.py`,
`scm_bench/readers.py` only for the reader refits). Record every check with PASS / FAIL and keep failed checks; never relax a check
after seeing its failure. Charge out-of-queue CPU with `python3 -m drf.jobqueue charge`.

Check (bounded; the campaign's records are under `results/pcrl_direct_recovery_frontier_v1/` and `.private/pcrl_direct_recovery_frontier_v1/`):
1. **Finite-law objective.** The fit records' objective is computed on MODEL_FIT people routed by their OUT-OF-FOLD statistics (the
   deployed MODEL_FIT arrays use the full-pool teachers, which are in-sample on training rows, so they are not the fitted law). With
   your own code: rebuild the fine cells from the frozen out-of-fold statistics (the files pinned in `drf.dr.SOURCE_PINS`), apply the
   saved A_inc / A_occ (`runs/DR/<config>/V<v>/MODEL.pkl`), build the emitted law (identical emitted vectors merged), and recompute
   the exact optimal-reader AUC*_pair, AUC*_inc, AUC*_occ (estimated law pi; empirical law true SEX), the plug-in MI terms and the 8
   spends; compare with `dr/FIT_V<v>.json` (tolerance 1e-9). Also check on MODEL_SELECT that the deployed release equals the saved
   wires and that its atoms emit the saved tables.
2. **Decisive task metrics.** The 28 conditions on MODEL_SELECT for DR_AUC and every control (your own metric code against U0 / Ucal
   frozen arrays), compared with the evaluation records; the CTRL bracket selection recomputed from the nested records.
3. **Reader comparisons.** The aggregate paired screen (mean, se_agg, version means, recipient gates) of the decisive comparisons
   recomputed from the grid's cell records with your own code; the D-labels recomputed by your own reading of RULES.md.
4. **Reader refits.** Refit a bounded sample (about 6) of winning reader cells with the admitted recipes and the recorded seeds and
   splits; compare SCORE AUCs.
5. **Integrity.** Ordering (registration receipt before the first label read; card commit before its fit; plan commit before the
   reading's first MODEL_SELECT read); specification count; no MODEL_SELECT read by `drf.dr`; no closed-role label; deployment packages
   (when built) reproduce the evaluated wires.
Return a report with each check, its inputs and its result; your final message is saved verbatim.
