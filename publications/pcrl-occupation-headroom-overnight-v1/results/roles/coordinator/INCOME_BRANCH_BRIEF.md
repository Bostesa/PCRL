# Brief for the conditional income-accuracy branch (coordinator; used only if the frozen trigger fires at stage-1 close)

Read `AGENT_BRIEF_COMMON.md`, `PROTOCOL.md` section 5 (item 5), and `roles/statistical_reviewer/RULES.json` (`income_branch`).

Trigger (frozen, evaluated once in `WITNESS_SELECTION.json`): a PROMISING_WITNESS exists AND (leg A: the MODEL_SELECT whole-release
reserve of H3 income + the top witness fails only on income accuracy clauses, OR leg B: the anchor's already published DEV_REPORT
income accuracy clauses fail the registered replication rule; leg B is TRUE from the published record: seed-1 accuracy 0.83 SE).
A branch opened through leg B is DEV-guided: its motivation comes from a DEV_REPORT result of the source campaign. That is disclosed
in every record; no new DEV_REPORT read is involved, and every selection quantity is measured on MODEL_SELECT as usual.

Permitted (at most 8 seed-specific configurations, budget class INC8):
- one bounded income-task repair branch addressing TASK QUALITY OR DECISION ERRORS while preserving the Brier / NLL headroom of H3:
  a genuinely justified income task-model correction or ensemble (for example a household-cross-fitted stack of the source's STRONG
  income learner with a complementary income learner, or a capacity-limited residual correction of the H3 income log-odds aimed at
  decision errors near the 0.5 threshold), with a MATCHED TASK-ONLY TWIN (the same model without H3's transplanted, critic-trained
  income corrections);
- the probability-plus-argmax contract; every parameter fitted on MODEL_FIT (task labels; SEX only if a transplanted stored
  correction requires it, as in H3) or CALIBRATION (calibrators), never on MODEL_SELECT;
- fixed-H3 comparisons kept: each branch configuration is a complete release (new income + a selected witness occupation, or a
  privacy-stage occupation derived from one); its matched component is the same occupation output beside H3's income.
Not permitted: another sweep over H3's old privacy strength (beta), six-level bins or mixture weights; any relaxation of the reserve.
If the branch fails, the occupation witness is packaged as a component and the reserve is not relaxed.

Deliverables: a card section before the first fit (operation, the measured failure it addresses, why it might fix seed-1 accuracy
without spending Brier / NLL headroom, matched twin, controls, triggers); code `ohn/fam_inc.py` (`selftest`, `fit`, `predict`
returning {'income': (n, 2)} for all six roles, `verify`); complete releases saved as RELEASE.npz / RELEASE.json under
runs/INC/<config>/S<s>/ (both tasks, all six roles) so `ohn.devreaders` can load them as {"type": "complete"} members; utility scored
on MODEL_SELECT with the registered evaluator (the coordinator adds an `--income` route to the scorer when the branch opens).
