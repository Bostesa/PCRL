# Methodology role notes (2026-10-02)

## Files in this directory

| File | What it is |
|---|---|
| `heldout_linear_inventory.csv` | HL-01..HL-13. Every linear evaluation of a defended representation that was found, with its fit, selection and scoring rows traced in code. Produced by two read-only code traces (PCRL refs and dg@956f5c88), then spot-checked by hand. |
| `failure_classification.csv` | FC-01..FC-18. Historical "failures" mapped to C1–C5. |
| `heldout_linear_recount.py` / `_output.json` | Reads committed git objects only (`origin/main`). |
| `scale_composition_check.py` / `_output.json` | Synthetic data; system python3, numpy only. |
| `leace_tolerance_check.py` / `_output.json` | Synthetic data; PCRL `.venv` python (torch, concept-erasure 0.2.4, sklearn). |
| `batch_r2_bias_check.py` / `_output.json` | Synthetic data; system python3. |
| `protocol_config.json` | Machine-readable protocol for the evaluator. |

## Pending operations

None of these was run here.

| # | Operation | Authorisation class | Resolves |
|---|---|---|---|
| P-1 | Frozen forward pass of the Adult CROSSPURP `canonical_iterate.pt` (seeds 0–2), then the native in-sample test R² on the same test rows. Needs the checkpoint location from `CHECKPOINT_INVENTORY.csv`. | Forward pass plus the method's own closed-form statistic | FC-11: C1 vs C2 |
| P-2 | Recount the stored `run_eval_multi` outputs on `cross-purpose-rebuttal-2026-05-18` (train→test ridge R² for Adult, HMDA and Diabetes). | Git read only | HL-06 |
| P-3 | Eigen-spectrum of the test-row Gram, compared with 1e-6, for R5/R7 final.pt and the erase-pilot checkpoints (where they survive). | Forward pass only | D1: whether "no change in leakage under rescaling" holds |
| P-4 | Per-batch native statistic with permuted attribute labels on stored checkpoints, plus the val eigen-spectrum. | Forward pass plus closed-form statistic. **Ask first**, since it computes regressions on real rows. | D4 hypothesis |
| P-5 | Whether the dg fitted defense parameters (Q, LEACE map, σ, channel weights) were saved, per cell. | Artifact agent (`ARTIFACT_TO_EVALUATION_MAP.csv` already says the dg-trained channel weights were never written) | B (U1 on unread PCRL rows); R11 estimability |
| P-6 | Per-classifier breakdown of the PCRL `PostHocAuditorSuite` (LR vs RF/SVM/XGB). It is not stored. | New attacker fit on real rows. **Needs approval.** | FC-12 attribution |

## Decisions left to the coordinator or advisor

- **Panel slots** CELL-A/B/C in `PILOT_PROTOCOL.md` §2.
- **Role shares** (50/15/35 of the non-defense-fit pool) and whether the secondary overlap arm runs.
- **Output object** per cell: hard label, score or logit.
- **Primary family** P1–P3, or a substitute (at most 3 endpoints).
- **Header names** in `ATTACKER_ACCESS_TABLE.csv`. The family and release-count columns are named `family`
  and `release_count`, with the allowed values given in the spec. Rename them if the evaluator expects
  the literal parenthesised header.
