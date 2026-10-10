# Bank-anchored release: start here

This focused campaign tested whether changing which code an already useful ten-bit Adult bank sends can protect SEX while keeping income and occupation predictions useful. It is exploratory development on reused people.

**Result.** The exact W_TSTAR bank runs as B0. Seven alternative channel policies were fitted on each of two source seeds. B0, T, D, P, L, J and S12 each passed all 24 original common-INNER point utility checks; S21 passed 22 and was not attacked. None has a complete simultaneous utility-bound pass. The best new-bank pair recovery was P at AUC 0.832301, versus B0 at 0.838251, a 0.005951 improvement below the 0.020 target. J reached 0.833786 and did not beat P. The eligible older P* reference reached 0.803141. No competitive privacy or joint-design gain was established.

[RESEARCH_DECISION.md](RESEARCH_DECISION.md) explains the comparisons, uncertainty and scope. [PRIMARY_ENDPOINTS.csv](PRIMARY_ENDPOINTS.csv) gives each seed, arm and recipient/pair view. [HEADLINE_COMPARISONS.csv](HEADLINE_COMPARISONS.csv) gives selected paired differences and simultaneous intervals. [ADVISOR_BRIEF.md](ADVISOR_BRIEF.md) is the short handoff.

## What ran and what can be run

[PROTOCOL.md](PROTOCOL.md), [CAMPAIGN_LOCK.json](CAMPAIGN_LOCK.json), [OPTIMIZATION_PROXY.md](OPTIMIZATION_PROXY.md) and [CANDIDATE_REGISTRY.json](CANDIDATE_REGISTRY.json) record the prospective comparison. The public code is in `bank_anchored/`; per-person inputs, labels, releases, models, draws and attacker predictions remain in owner-only `.private` directories. [R1 quickstart](r1/QUICKSTART.md) documents executed raw-input B0/J commands, expected 83-column input schema, one-fixed-release sampling contract and the recipient wire. [R1 executable manifest](r1/R1_EXECUTABLE_RELEASE_MANIFEST.json) binds source and policy hashes. R5 independently replayed utility, all 38 effective fresh-attack units, the 1,575-slot recovery interval family, selection, and four fixed HGB refits; [NUMERICAL_CLOSEOUT_V2.json](r5/NUMERICAL_CLOSEOUT_V2.json) gives the checks.

A versioned same-device snapshot and restore checked all 55 files with uncached hashes. The first restored numerical CLI attempt exposed a missing transitive runtime import. The V2 fixed-source successor exposed another lazy decoder import and is preserved as a failure; final corrected restored numerical replay is in progress. The snapshot is not an off-device backup; no writable external drive was mounted. [RESTORE_INDEX_V2.json](r1/RESTORE_INDEX_V2.json) records current custody.

Six native agent roles worked under one machine-wide heavy-process queue. No cloud resource or new data was used. The original assessment and new HEAD_VALIDATION labels stayed closed. The two source seeds do not create independent people. [SOURCE_AND_EXPOSURE.json](SOURCE_AND_EXPOSURE.json) lists roles and source pins.
