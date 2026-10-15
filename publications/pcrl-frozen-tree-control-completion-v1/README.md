# PCRL: frozen-candidate control completion and confirmation feasibility (exploratory)

**Recommendation:** `ORIGINAL_CONFIRMATION_CRITERION_NOT_PRACTICALLY_FEASIBLE` + `NO_COMPATIBLE_UNEXPOSED_POPULATION_IDENTIFIED`.

**Frozen candidate:** L_TREE (CatBoost depth-4 teachers plus local leaf-score privacy corrections, ρ=1, epoch 20) from publication `c8ae518190100decf39e605f191747d0f10723e2`. It was not retuned.

**Completed controls** (exploratory, already-used Adult development people; coalition SEX AUC benefit):

| L_TREE vs | benefit | status |
|---|---|---|
| T_REFIT E20 (task-only continuation at the same epoch) | 0.085 [0.065, 0.105] | established |
| fresh, equally exposed neural RAW-J | 0.066 [0.045, 0.088] | beyond 0.020 |
| access-honoring official FARE | lower recovery than L_TREE | utility-infeasible; descriptive, not beaten |

**Confirmation feasibility:**
- The original criterion cannot be confirmed at the observed margins over the historical codes (0.014–0.018).
- A narrower, new claim would need about 317,000 independent evaluation groups (IUT, one-sided α = 0.025). The binding constraint is the occupation Brier slack.
- No compatible unexposed population was identified.

**What is here:**
- **Start with** `results/START_HERE.md`.
- **Code:** `code/tree_completion` (campaign code, including the runnable power model `power.py` on public aggregate inputs and the confirmation runner, whose fresh execution is disabled) and `code/tree_completion_independent` (the verifier).
- **Receipts:** `prefit/` holds the receipt pushed to this GitHub branch before any new fit (`ba54ecc`; scrub `4db5534`).

**What is not here:** people, labels, predictions, model weights, packages and absolute paths all stay private. Reproduction needs the private role exports and the frozen packages.
