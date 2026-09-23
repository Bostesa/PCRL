# Evidence table — what each claim rests on

Every row names a pinned commit. Verification scripts read evidence through the Git object store, so
nothing here depends on a working-tree copy. **No new scientific fits and no cloud compute were used.**

| finding | evidence commit | files | verified by |
|---|---|---|---|
| Prospective ACS 2016 decision: Q and D17 each 8/10 — all eight sensitive clauses pass, both task clauses fail | `5e154e5c4fdaeb23d327a0ebefe838525f1a19cb` | `INFERENCE_2016.json`, `PRIMARY_CLAIMS_RESULTS.md`, `RESEARCH_DECISION.md`, `PAPER_ADDENDUM.md` | `verify_prospective_2016.py`, 29 checks; every bound re-derived as `estimate + z·SE` |
| Q vs D17/D33: worse unweighted task loss, unresolved weighted, no sensitive advantage | same | `INFERENCE_2016.json` secondary rows | same script |
| Q vs RR75/W75: better task, higher recovery on 6 of 8 endpoints each | same | same | same script |
| Supervised 2018 development result (task gain 0.01625/0.01639, 7 of 8 bounds unresolved) | `f4bdf4cd5bf74c634feeec50aef78bff249667e4` | `HEADLINE_EVIDENCE.json`, `NARRATIVE_MANIFEST.json` | `verify_task_directed.py`, 24 checks |
| Claims audit corrections (residence supervision, local policy, 2/4 cells, prior art, composition lemma) | `33124f965861ccfcaa710aff4a56880c5ab3957e` | `patches/0002-*`, `THEORY_REPAIRS.md`, `PRIOR_ART_MATRIX.md` | `verify_t3_integration.py`, 15 checks incl. source re-derivation |
| Rare-class audit, retired guarantee, composition boundary, shared-union comparison | `0176f149e91c02b8e2d202eb25ea9cba8ae019dc` | `V2_DOMINANT_AXIS_LATEST_SUMMARY.md`, `docs/ACCURACY_CERTIFICATE_RETIREMENT.md`, `redesign_20260907_gaussian_v1/TABLE.md` | source-read at pin; recorded in the claim ledger |
| PCRL rebuttal improvements (per-dim std, rank-8 ablation, VICReg sweep) | `0176f149e` | `results/rebuttal/**` | read directly; **these belong to PCRL, not the other paper** |
| Manuscript scope statements (12 required facts) | working tree + `origin/main` | `papers/pcrl_satml_final_v1/main.tex` | `verify_manuscript_statements.py`, 12 checks |
| Public-main repair not merged | `5d4eda04639aae10733e4d72c2ceaae0e849ede5` | `fix/retire-accuracy-guarantee` | `git branch -r --contains`: `origin/main` does **not** contain it |

## Counts used in the paper

| count | value | source |
|---|---|---|
| 2016 final pool | 23,684 people in 15,928 shared households across three anchors | `INFERENCE_2016.json` |
| 2016 panel execution | 156/156 fit units, 156/156 score units, 0 failures, 0 quarantined, 10,000/10,000 draws | same |
| 2016 endpoints | 20 primary, 70 secondary, 114 descriptive | same |
| 2016 independent replay | per-person loss within 3.55e-15; 204 point estimates within 1.39e-17 | `FINAL_HANDOFF.json` |
| 2018 programme | 81 nominal primary maps (54+18+9); 126 of 162 maps completed; 125 reduced kernel groups; 297 audits | `f4bdf4cd5` |
