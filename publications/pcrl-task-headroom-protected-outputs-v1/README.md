# PCRL: task headroom and protected prediction outputs (exploratory)

**Result:** CatBoost task teachers gave real task headroom. Local leaf-score privacy training (L_TREE) cut linked-coalition SEX recovery from probability+decision outputs by **0.0505 AUC [0.0257, 0.0754]** against the incumbent RAW_PROJ, and every original point utility clause passes. Against three historical score codes the gain (0.014–0.018) is unresolved, and joint design was not established.

**Labels:** `POINT_FEASIBLE_NO_COMPETITIVE_CRITERION`, `JOINT_COMPONENT_NOT_ESTABLISHED`. Independent verification: 9 of 9 stages PASS.

**What is here:**
- **Start:** `results/START_HERE.md`, then `results/ADVISOR_BRIEF.md` and `results/RESEARCH_DECISION.md`.
- **Code:** `code/task_headroom` (campaign code) and `code/task_headroom_independent` (verifier).
- **Receipts:** `prefit/` and `stage_b_prefit/`, pushed before any fit and before any protected fit.

**What is not here:** people, labels, predictions, model weights, packages and absolute paths all stay private. The data are repeatedly used Adult development roles; this is not confirmation.

**Run:** `python -m task_headroom.deploy --package <private package> ...` (see `results/QUICKSTART.md`).
