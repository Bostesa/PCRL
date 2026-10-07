# Exposure ledger (lcr)

## Registered exposure statement (verbatim, before any real fit)

"This study is motivated by opened Adult development results from qpc and cbp and the earlier PCRL lineage. All real-data roles have been used historically. This is exploratory development evidence. Nominal intervals condition on fitted artifacts and do not correct for the adaptive research history. A masked assessment prevents additional selection leakage but does not make the rows fresh. No confirmation population is opened."

## What is already known before this study

Published assessment aggregates from qpc (evidence `9dd06da`) and cbp (tip `7f3ec67`) for maps admitted here:

| Release (assessment means) | Value |
|---|---|
| U continuous, pair SEX AUC | 0.858 (cbp 0.8585) |
| DIRECT-TASK i8o64 (D0) | Pair 0.849; occupation log-loss excess about 0.0030; all four confidence bounds pass (qpc and cbp) |
| FINE-TASK i8o64 (D0) | Pair 0.846 |
| JOINT λ 0.1 (D0) | Pair 0.813; occupation log-loss excess 0.008072 (upper bound about 0.012). Preservation not established; violation not established. |
| JOINT λ 0.01 (D0; cbp P\*) | Pair benefit vs FINE-TASK 0.00217 [0.00019, 0.00415] |
| SEQ-12 / SEQ-21 λ 0.1 (D0) | Pair 0.816 / 0.816 |
| Decisions alone | Pair 0.739 |
| cbp inner rows | Several λ 0.06 and 0.08 maps met ordinary eligibility and the local guards but failed cbp's 0.006/0.0035 every-seed headroom; they were never assessed as nominees |

The learned-decoder and constrained-release design was chosen knowing those numbers. They are motivation, not untouched evidence.

## Data roles (unchanged)

| Role | Rows | Use here |
|---|---:|---|
| OSF_DEFENSE_FIT | 15,434 | Fine partitions stay fixed and label-blind. **New in this study:** the coarse assignments and the learned decoder (D1) are fitted with this role's TRUE TASK LABELS and SEX. This is a material change from the label-blind codebook fitting of qpc and cbp. |
| HEAD_VALIDATION | 1,500 | Historical head selection only; not repurposed; no label read |
| AUDIT_FIT | 6,065 | Attacker fitting |
| INNER_SELECTION | 2,235 | Attacker selection, ordinary inner eligibility, comparators and nominees |
| OSF_DEVELOPMENT_ASSESSMENT | 13,936 (13,929 groups) | Masked until the pushed lcr EVALUATION_LOCK; opened once. It was opened historically in qpc and cbp: the mask prevents additional selection leakage but does not make the rows fresh |

**Mask scope.**
- No assessment label is read before the lcr EVALUATION_LOCK is pushed.
- No prior per-person assessment prediction (qpc or cbp `outer__*` units) is copied into the lcr store or read by
  fitting, selection or debugging. lcr.admit copies no outer unit.
- Only the cbp INNER audits of the admitted D0 maps (AUDIT_FIT / INNER_SELECTION) are admitted.

**Input.**
- `adult_jcv.npz`, sha256 `e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12`.
- The exact 83 permitted columns, order, duplicate rules and group assignments.
- Explicit SEX, race, true task labels, person IDs and role indicators are never inference inputs.

## Populations

- No new state, year or population is opened.
- ACS 2016 is spent. Texas 2018 carries a prior reservation.
- No other cohort is assumed unexposed without reconciling both repositories and the private ledgers.
- No confirmation population is spent by this study.

## Historical custody issue (recorded; not rewritten)

Earlier committed history on predecessor branches contains an identifying drive-folder name. No history rewrite or
force-push is authorised. New public files use placeholders only and are scanned before every push.

## Realized use (appended at closeout, 2026-10-07)

The registered mechanism gate failed (MECHANISM_GATE_NOT_MET), so no Adult fit ran. In practice:
- the authorized material change (OSF_DEFENSE_FIT task labels and SEX for the new supervised fitting) was never
  exercised;
- no lcr attacker was trained on AUDIT_FIT;
- no selection used INNER_SELECTION;
- the assessment labels were never unsealed, because no EVALUATION_LOCK exists;
- HEAD_VALIDATION was not repurposed.

The study's real-data operations were limited to three things:
- source admission: teacher forward passes and release re-encoding;
- one deployment of the admitted Q map on the 83 permitted columns;
- custody copying and restoring.

None of them reads a task label or SEX.
