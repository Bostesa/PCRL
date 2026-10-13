# Independent replay preparation

R7 performs metadata and code work only. ROOT alone registers and launches every numerical verifier through the shared queue. No numerical files were opened, hashed, imported, fitted, or tested by R7.

The initial `REPLAY_PATCH.diff` covers an invocation-local cache of exact bootstrap envelope inputs, a flat INNER truth filename guard before hashing, and direct selection of each registered epoch from the pinned saved checkpoint-index JSON list. `FINAL_REPLAY_PATCH.diff` includes the complete final audit and reporting replay. The cache retains the original ordered reader loop, AUC routine, bootstrap draws, seed averaging, thresholds, family size, and interval arithmetic. Its key includes the exact truth-table SHA, truth column, score-table SHA, and ordered reader keys. No cached object is mutated by endpoint subtraction or sign multiplication.

The patch was applied after ROOT reported all 72 reader banks and 66 trajectories complete. `FINAL_CODE_PINS.json` gives the current exact code bytes. No shared reporting or diagnostic lock is edited by R7. The prepatch config directory is retained as preparation evidence and must not be launched.

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

The complete audit with ROOT's four fixed gradient witnesses is prepared at `.private/occupation_bottleneck/r7/audit_final_v1/`. It contains 4 tensor summaries covering 8 batches, plus independent batch-zero regeneration for each of the 4 states/seeds. Its two configs and source lists are ready for ROOT queue registration.

After ROOT's reporting unit completes, prepare final headline and inference checks with another new destination:

```sh
python3 results/pcrl_occupation_bottleneck_v1/r7/build_headline_config.py \
  --report-config .private/occupation_bottleneck/root/configs/ROOT_REPORT_CONFIG.json \
  --output-dir .private/occupation_bottleneck/r7/headlines_final_v1 \
  --inference-dir results/pcrl_occupation_bottleneck_v1/r8
```

Use the actual completed ROOT report config path. Omit `--inference-dir` until both runner inference receipts exist; a later invocation with a fresh destination prepares the utility and privacy configs. The headline worker replays all 72 packages on both FIT2 and INNER from the original frozen wires, class metrics and U0/Ucal references; verifies original-to-derived person-level metric, truth and score arrays; checks the 12 seed clauses and 24 paired clauses; preserves historical RAW ineligibility; checks all 144 supported-stratum banks and 12 exact decision-reader alias packages; and independently regenerates the FIT01 label table. Real ATT and INNER controls are recomputed before comparing the constant bool dependency table. Failed controls yield MISSING privacy endpoints while retaining K108. Utility K864 is unchanged.

ROOT can verify the cache separately with the queue-only synthetic fixture:

```sh
python -m results.pcrl_occupation_bottleneck_v1.r7.cache_equivalence_fixture_v2 \
  --output results/pcrl_occupation_bottleneck_v1/r7/CACHE_EQUIVALENCE_REPLAY.json
```

This fixture requires an authenticated verification queue slot before importing numerical libraries. It compares cached and uncached serialized results exactly, including repeated two-seed envelopes and reversed signs, and requires fewer AUC calls. R7 has not executed it.

ROOT executed the original fixture and preserved its class-scope setup failure. ROOT's corrected `cache_equivalence_fixture_v2` passed as `OB_INDEPENDENT_CACHE_EQUIVALENCE_V2`: serialized cached and uncached results were exactly equal, with 12 versus 96 AUC calls. Use that corrected module for future execution; the first source and failed queue receipt remain retained.

ROOT's `OB_INDEPENDENT_AUDIT_R7` and `OB_INDEPENDENT_C3_R7` completed. The audit replays 66 trajectories, 72 ATT banks, 18 geometry role pairs, 18 paths, 4 stored gradient summaries, and 4 independently regenerated gradient batches. All four regeneration checks report PASS. Source admission reports 84 nominal packages, 66 canonical packages, 20 exact source aliases, no missing package, and no INNER label bytes read or hashed. R7 inspected these JSON receipts only. Final INNER headline and inference replay remain required before campaign closeout.
