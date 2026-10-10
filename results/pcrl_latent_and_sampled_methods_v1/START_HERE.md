# Latent repair and sampled confidence: development result

This is an exploratory Adult development study. The same `INNER_SELECTION` people have been examined repeatedly in predecessor studies; the old assessment stayed embargoed. Two paired model seeds reuse the same 2,235 people and 2,234 duplicate groups. The results are not independent confirmation.

## What was tested

The feature service released a **ten-bit hard code per recipient**, plus a probability vector and decision computed only from that code. We fitted a task-only compressor (T), fixed-kernel privacy repair (D-K), local, both freeze-first sequential, and ordinary adversarial controls. We also tested an unchanged wider service (F), which releases 16 scaled latent values per recipient plus its task output. Fresh attackers saw each recipient's complete release and the aligned pair.

D-K lowered pair SEX AUC from **0.832582 to 0.805905** in seed 0 and from **0.835960 to 0.793778** in seed 1 relative to T. The equal-seed reduction was **0.034429**, with a fixed-family simultaneous interval **[0.013719, 0.055138]**; the 0.020 target is unresolved. D-K failed the required occupation utility limits. Its mean occupation NLL excess against U0 was **0.032253** with lower interval bound **0.010368**, above the 0.010 cap. Its mean occupation Brier excess against Ucal was **0.014797** with lower bound **0.006994**, above the 0.005 cap. All six ten-bit arms missed the full two-task, two-reference point utility gate. There is no eligible ten-bit matched method win or joint-design result.

F passed all 24 seed/task/reference **point** utility clauses, but nine of its 36 seed and mean interval clauses remained unresolved. Its pair SEX AUC was **0.869335/0.862196** across seeds. F has a wider disclosure than the ten-bit services and is a descriptive utility/privacy tradeoff, not a capacity-matched repair result.

The score service was designed to transmit one cached sample from a finite confidence channel. Its 16 bounded fits used the locked same-bank stochastic, deterministic, product, local and sequential arms. Fourteen searches completed without a feasible selected policy; two reverse sequential fits ended after infeasible first stages and are technically incomplete. **No protected sampled release was selected or attacked.** The fixed historical references and identity/class-only diagnostics have separate fresh complete-view attacks, independently replayed across both seeds. The task-only T* score reference had pair SEX AUC **0.846745/0.829728**; inherited P* had **0.802702/0.803579** and passed the point utility caps in both seeds. The class-only diagnostic had **0.736103/0.734125** but failed utility. Independent endpoint replay matched all **1,161 estimable** slots of the registered 6,885-clause family; the other **5,724** are missing with null bounds. These fixed cells do not constitute a protected sampled-channel comparison.

## Where to look

- `PROTOCOL.md`, `CAMPAIGN_LOCK.json`, and dated amendments: what was fixed before each stage.
- `PRIMARY_ENDPOINTS.csv`: D per-seed tasks, local/pair AUCs and all 567 signed simultaneous clauses; A fixed-cell task/AUC rows and all 6,885 registered clause slots, including missing ones.
- `RESEARCH_DECISION.md` and `COMPONENT_LEDGER.md`: scientific interpretation and component attribution.
- `INDEPENDENT_VERIFICATION.json`: exact independent replay scope and limitations.
- `UNIT_STATUS.csv`: the physical queue and charged CPU; `A_PLANNED_UNIT_STATUS.csv` records all 220 logical A units, including aliases and units not triggered; `A_FIXED_ATTACK_PHYSICAL_ATTEMPTS.csv` preserves failed and corrected attack attempts.
- `QUICKSTART.md`: commands actually exercised; private checkpoints and person-level arrays stay outside Git.

The D-K one-row real-input deployment, six negative contract tests, same-device byte restore and restored numerical replay passed. A full Git bundle was restored at an earlier versioned branch tip; a latest-tip bundle and off-device custody are tracked in `BACKUP_AND_RESTORE.json`.
