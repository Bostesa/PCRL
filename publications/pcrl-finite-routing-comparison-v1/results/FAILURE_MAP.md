# Failure map: family FA (propensity-routed finite cells), campaign `pcrl_finite_routing_comparison_v1`

Drafted by the challenger (2026-10-20, 23:08-23:22Z, before registration; saved with attribution). The coordinator edited it at
23:40Z only to match the adjudicated protocol: F4 is a D2 conjunct, not a stop (stop condition 5); the T6 repair is labelled
POST_HOC; SHUF uses the person's own qO in its guard; brackets are chosen on MODEL_FIT nested total-NLL spend. Every edit is marked
"(coordinator)".

Sources: the NRM1 published records (cards/FA_PRC_v1.md; roles/designer_a/{CLOSING_REPORT.md, GATE_V*_*.json,
CAPACITY_INFO_LABEL_FREE.json}; roles/coordinator/FA_CAPACITY_DECISION.md; RESEARCH_DECISION.md; TABLES.md; FAILURE_MAP.md;
roles/challenger/CLOSING_REPORT.md) and this campaign's PROTOCOL.md section 0. No data and no labels were read.

All of this is development evidence on adaptively used records, for the declared readers only.

## 1. What this family has run

| Operation | Status | Evidence (v0 / v1) |
|---|---|---|
| Task teachers (income HGB, A1_PO_16 occupation) and out-of-fold embeddings over 5 household folds | fitted | T*: income 332 / 393, occupation 168 / 154 |
| Task-only twin 16 x 64, KL k-means, label-refit decoders (A, alpha = 20) | fitted, saved, evaluated on MODEL_SELECT | Point-feasible: worst clause / cap 0.100 / 0.510 (binding: v1 men's guard +0.00255). Occupation NLL vs U0 -0.0033 / -0.0012 (A1: -0.0161 / -0.0128). Reserve ratio 2.311 / 1.289 (not reserve-qualified). Pair / income / occupation AUC 0.6509 / 0.5657 / 0.6233 (NEURAL_RAW 0.6361, H_LJ 0.6327, W 0.6565) |
| Internal capacity gate, 16 x 64 | FAIL | Occupation NLL spend over CONT +0.0107 +/- 0.0013 / +0.0136 +/- 0.0013. Occupation accuracy loss +0.0026 / +0.0048. Gate: estimate + 2 SE <= 0.004 / 0.003 |
| Twin 24 x 128 (R0) | fitted and saved (contingent; disclosed deviation); never evaluated on MODEL_SELECT | Gate FAIL: +0.0137 +/- 0.0014 / +0.0132 +/- 0.0014; occupation accuracy loss +0.0057 / +0.0031 |
| Label-free capacity curve | recorded | Fitted occupation KL 0.0098 / 0.0067 / 0.0046 / 0.0031 at 64 / 128 / 256 / 512 cells (v0). Income 0.0009 / 0.0004 / 0.0002 at 16 / 24 / 32. Approximate decoder term 0.0055 (64) / 0.0092 (128): a hypothesis, not a finding |
| Label-free pre-registration check (this campaign) | done | Bitwise reproduction. Prototype vs out-of-fold cell mean: max 0.0019-0.0062 (occupation). Decoders vs prototypes: up to 0.09-0.14. Deployed vs out-of-fold routing of MODEL_FIT people: 56-76% agreement |

**Never run in this family:**
- pi-routed PRC at any strength (S1-S3);
- HQ, SHUF, the ORD ladder and LOC;
- the per-SEX guard's effect, and the lambda continuation on real data;
- any MODEL_FIT LOOKUP check of a protected release;
- robustness readers on a finite release;
- B0 / B1 tables;
- capacity (32, 256);
- kappa = 4 (R3) and exact single-person moves (R2).

**Closed adjacent lines (not reopened):**
- SCM/IBSI post-hoc cell merging: benefit not separated from task spending;
- FB / FB2 Gaussian channels: F1 holds;
- leaf-score corrections;
- Adult learned decoders: re-decoding a fixed partition removed no information;
- Adult finite codes: joint about equal to sequential;
- CTL_REM remainder flattening: full collapse lowered pair AUC by 0.0148 at 1.8-3.0 times the caps.

## 2. Failure modes this campaign can detect

| Failure mode | Where | Consequence |
|---|---|---|
| Decoder estimation error of A at fixed routing | T1 nested + T3 MODEL_SELECT, B1 - A | PE_ERROR_* label |
| Prototype lag (last Lloyd codeword differs from the cell mean) | B0 - B1 | descriptive |
| Quantization error remaining after the repair | B1 spend over CONT; original gate recomputed | reported, never the eligibility rule |
| No utility slack at the assessed budget | QUAL on MODEL_SELECT | stop |
| Implementation drift | T0 bitwise; T1 <= 1e-12 against the GATE records; verifier recomputation of B1 | stage stops |
| A table change alters the partition | identity checks (nested per fold; all six roles) | assertion stops |
| The fitted objective does not carry over to held-out people | nested in-sample minus held-out LOOKUP; fitted pi-MI vs measured recovery | falsifier F4 (a D2 conjunct; coordinator) |
| A nonzero strength changes nothing | changed vectors, re-routed people | exact alias; never saved twice |
| Untargeted re-routing explains the gain | SHUF at bracketing spend | not "beyond SHUF" |
| Task-score-only grouping explains the gain | HQ at bracketing spend | not "beyond HQ" |
| Utility spending explains the gain | ORD frontier with a bracketing point | NOT_SEPARATED_FROM_TASK_SPENDING |
| Leakage moves to the other recipient | per-view and ignore-one readers; recipient gates | gate failure |
| Advantage visible only to LOOKUP-type readers | non-table winner; nested MODEL_FIT LOOKUP | READER_TARGETED |
| Reader-budget asymmetry | frc.readerbudget.contrast; FIT-fraction curve | credited only with non-table readers |
| Versions disagree | aggregate screen | not AGG_POSITIVE |
| A near-limit point treated as robust | reserve beside G1 | never "confidently feasible" |

## 3. Failure modes this campaign cannot detect

- Recovery by readers outside the declared bank; by readers with more labelled data than reader FIT (about 4,000 people; the nested
  MODEL_FIT LOOKUP is the only denser check); or by recipients holding the published tables or auxiliary data.
- Anything at confirmation level:
  - MODEL_SELECT was used adaptively;
  - versions share all training rows and the MODEL_SELECT people;
  - reader repetitions share households;
  - a 0.020 benefit is not confirmable on about 17,500 households;
  - effects below about 0.015 pair AUC (2 se_agg) cannot be resolved.
- Decoder error for other teachers, for capacities beyond (32, 256), for other shrinkage (no sweep), or in other cohorts, years or
  states.
- A joint-design contribution (no separate or sequential controls).
- Comparisons with untested routes: randomized response on decisions; input-side recoding or erasure; SEX-balanced reweighting; MMP;
  coupled-versus-decoupled randomness; continuous-mixture and temperature-matched controls.
- Repeated-query or linkage risk; anyone outside the admitted cohort; the closed roles.
- The effects of the non-nested T* and of the deployed-versus-out-of-fold routing shift (both shared by every arm).
- Whether a stronger optimizer (R2) would find a better PRC, unless R2 is run.

## 4. Stop conditions (in order)

1. **Reproduction failure** at T0, T1 or T2 (bitwise, <= 1e-12, partition identity): stop that stage until it is diagnosed and
   recorded.
2. **Improper label read** (any read outside `frc.common.load`, before the receipt, from a closed role, or of MODEL_SELECT labels
   inside a fitting module): stop and report.
3. **No qualifying construction** at (16, 64), (24, 128) or, under its trigger, (32, 256), and no qualifying POST_HOC repair
   (coordinator): FA stops with NO_QUALIFYING_TASK_ONLY_RELEASE; protected routing remains untested. (Known in advance under the S2
   rule: A at (16, 64) fails Q2.)
4. **No feasible strength:** every nonzero PRC selection is NOT_FEASIBLE or an exact alias of TWIN. Record NOT_ASSESSED and stop.
5. **F4 fails at MODEL_FIT** (coordinator: changed from a stop to a D2 conjunct by the adjudication): D2 is NOT_MET_F4; the protected
   releases are still evaluated on MODEL_SELECT as diagnostic evidence; F4 is the registered trigger of revision R2.
6. **F1-F3 on MODEL_SELECT after the allowed revisions:** the family closes and no further PRC configurations are fitted. A
   successor needs a new FS_ card, reviewed by the challenger and the statistical reviewer.
7. **Revisions used up:** two protected-stage revisions or one T6 repair used means no further revision.
8. **Resource limits:** 48 configurations (the 49th is refused), 12 CPU-hours in total including failed attempts and agent work, and
   the research window end at 06:49:23Z (closeout 07:19:23Z). Close out with what exists; unfinished stages are recorded as NOT_RUN.
