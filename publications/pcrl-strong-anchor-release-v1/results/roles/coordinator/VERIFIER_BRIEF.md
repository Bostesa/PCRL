# Independent verifier brief (coordinator; drafted 2026-10-24 04:40Z, completed before launch)

Campaign `pcrl_strong_anchor_release_v1` (package `sar/`). You verify, with your own code, the decisive results of this campaign from
its frozen outputs. You do not reuse `sar/rules.py`, `sar/claims.py`, `nrm/rules.py` or `sar/devreaders.py` for the numbers you
recompute (you may read them to understand definitions); write your code under `results/pcrl_strong_anchor_release_v1/roles/verifier/code/`
(the only entry point authorized for the `verify` operation of `sar.common.load`), and your records under
`results/pcrl_strong_anchor_release_v1/roles/verifier/`. Label reads run only inside queue units (`python3 -m sar.jobqueue run --unit
SAR_VERIFY_<n> --stage verify --rss-gib 2 -- <interpreter> <your script> ...`), and only for the roles MODEL_FIT, CALIBRATION and
MODEL_SELECT through `sar.common.load(..., operation='verify', purpose=...)`.

Checks (each PASS / FAIL with numbers; a failed check is preserved, never relaxed; an adjudication, if any, is a separate entry):
1. **Utility.** For every release that enters a decision (candidates, their START, CONT and CTRL bracket points, the references
   NEURAL_RAW and H_LJ): the 12 task clauses and 2 within-SEX guards per version recomputed from the saved MODEL_SELECT arrays and the
   MODEL_SELECT labels with your own metric code (accuracy loss, NLL excess, summed Brier excess against U0_S<v> / Ucal_S<v>; men's and
   women's occupation summed-Brier excess against U0), against `runs/<F>/<C>/EVAL_MS.json` (tolerance 1e-9).
2. **Selections.** Each candidate's strength and each CTRL bracket point recomputed from the saved MODEL_FIT selection records with
   your own implementation of the card's rule (no label read needed where the records hold the per-strength spends; otherwise a
   logged MODEL_FIT read).
3. **Grid.** For a stated sample of cells and members: the winner SCORE AUC and the household-bootstrap SE recomputed from the stored
   reader scores, and the per-comparison aggregate (block means over version x repetition; se_agg) recomputed from the GRID.json /
   BOOT.npz records with your own code; every decisive comparison's label recomputed from the frozen rules' definitions.
4. **Reader refits.** A bounded random sample (stated seed) of winning readers refitted from scratch on the frozen splits and seeds;
   their SCORE AUCs must reproduce the grid's (bitwise, or a stated tolerance with the reason).
5. **Outcome.** Each reading's decisions (D1, UM, D2, D2R, D3, D4, D5, D6, outcome class) recomputed from the recomputed inputs.
6. **Access.** The label-access log: no label read before the verified registration receipt; every fitting read after its card's
   commit; no MODEL_SELECT label read inside a fitting module; no closed role read.

Deliver `VERIFICATION_REPORT.md` and `CHECKS.json` (each check with its inputs' SHA-256), then a hand-back with the totals.
