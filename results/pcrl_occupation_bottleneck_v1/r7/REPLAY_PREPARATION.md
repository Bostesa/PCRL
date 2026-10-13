# Independent replay preparation

R7 performs metadata and code work only. ROOT alone registers and launches every numerical verifier through the shared queue. No numerical files were opened, hashed, imported, fitted, or tested by R7.

The pending `REPLAY_PATCH.diff` covers an invocation-local cache of exact bootstrap envelope inputs, a flat INNER truth filename guard before hashing, and direct selection of each registered epoch from the pinned saved checkpoint-index JSON list. The cache retains the original ordered reader loop, AUC routine, bootstrap draws, seed averaging, thresholds, family size, and interval arithmetic. Its key includes the exact truth-table SHA, truth column, score-table SHA, and ordered reader keys. No cached object is mutated by endpoint subtraction or sign multiplication. Numerical equivalence still requires ROOT's queued verification.

Do not apply the patch while reader units still pin the current replay code. After all those units finish, ROOT may apply the patch, pin the new exact code bytes, and register the final replay units. No shared reporting or diagnostic lock is edited by R7.

Run the metadata builder with a new private destination:

```sh
python3 results/pcrl_occupation_bottleneck_v1/r7/build_audit_config.py \
  --output-dir .private/occupation_bottleneck/r7/preparation_002
```

An incomplete bank or trajectory receipt set writes only `PREPARATION_STATUS.json` and exits 2. It does not generate an incomplete runnable replay config. Once complete, it produces `AUDIT_CONFIG.json`, `C3_CONFIG.json`, and their exact source lists for ROOT registration. These source lists include the config bytes, code bytes, original source-role pins, admitted checkpoint pins, numerical pins copied from completed receipts, and pinned saved path indices. ROOT must register these sources as a `verification` stage unit, with the config's exact `unit_id` and public output.

The audit config covers 72 ATT banks, 66 canonical full FIT trajectories, source admission for the nominal 84-package roster, 18 four-checkpoint paths, and six U0/RAW/JOINT anchors across three authorized feature roles (18 pair checks, 36 recipient geometry checks). The C3 config covers the FIT01-trained task tables, FIT2 label association, complete frozen reader contracts, and supported strata for all 72 banks. These remain exploratory results on the inherited development people.

To include the fixed gradient witnesses, supply a metadata JSON containing only `gradients` and/or `gradient_regeneration` lists in the verifier interface:

```sh
python3 results/pcrl_occupation_bottleneck_v1/r7/build_audit_config.py \
  --output-dir .private/occupation_bottleneck/r7/preparation_003 \
  --gradient-items .private/occupation_bottleneck/root/INDEPENDENT_GRADIENT_ITEMS.json
```

The final reporting replay also needs original INNER frozen wires, separately pinned task/SEX truth under the canonical committed reporting lock, source U0/Ucal references, real control scores, and the frozen FIT01 label table. The numerical final report and every headline gate must be independently replayed before campaign closeout. Config preparation and code review do not establish numerical PASS.
