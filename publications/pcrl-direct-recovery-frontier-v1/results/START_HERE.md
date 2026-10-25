# Start here: direct optimization of recovery in a bounded finite release family (campaign `pcrl_direct_recovery_frontier_v1`, DRF1)

A bounded development investigation, run on 2026-10-25 from 02:17:51Z, on the Florida ACS 2019 development cohort (admitted
public-use PUMS records). This is statistical privacy research on an existing census benchmark. SEX is the dataset's recorded
demographic variable whose recoverability is measured.

**What the service does.** It predicts an income class (2 classes) and an occupation group (5) from six fields: AGEP, WKHP, COW,
SCHL, MAR and POBP. Each of two recipients receives only its own probability vector and argmax decision; one instance is released
per person. The usefulness conditions are the source's original 28: 12 task clauses and 2 within-SEX occupation guards per model
version, with the original caps.

**The question.** Can a realizable many-to-one probability release keep those usefulness limits while reaching a recovery level that
the strongest existing releases have not reached reliably? The concrete change tested is alignment with the recovery objective under
explicit task constraints:
- **Objective.** Exact optimal-reader AUC of the finite law of the emitted wire and SEX, for the linked pair and each view, instead of a
  dependence proxy.
- **Constraints.** Eight usefulness spends inside the search, instead of a strength sweep followed by a utility screen.
- **Family.** Deterministic lookups from 64 / 256 legal cells (task-twin codeword x propensity quartile, functions of the six fields
  only) to at most 16 / 64 atoms, emitting teacher-mean tables.
- **Isolating comparison.** The same family and search with the previous plug-in MI objective (DR_MI), plus a SEX-blind coarsening
  path bracketed at matched spend.

This overlaps privacy funnels and the earlier SCM / FRC1 finite routing; no novelty is claimed.

**Result: no method claim** (`RESEARCH_DECISION.md`). Failure stage 3: the legal mapping keeps the original usefulness but ties its
controls.
- **Optimistic diagnostic** (MODEL_FIT only). A legal-cell construction fitted on the true-SEX law of the fitted people was useful
  in-sample: 0.023-0.024 below the SEX-blind path at matched spend.
  - It was not useful on the nested held-out estimator. There its held-out exact pair AUC* was 0.044 / 0.041 above the SEX-blind
    path at matched nested spend.
  - Its nested spends were 2.6-2.9 times the occupation NLL budget and 3.4-4.0 times the men's guard budget.
- **v1 legal mapping** (registered; realized-label constraints). Its constraints did not transfer. On MODEL_SELECT it failed the men's
  occupation guard and occupation NLL, and occupation Brier in version 1 (stage 2).
- **R1 legal mapping** (registered headline revision; expected-cost constraints). It kept all 28 conditions in both versions. That is
  at the point only: 20 of 28 pass the confidence conjunction, none is violated, and it is not reserve-qualified.
  - **Recovery and the coarsening control.** Pair AUC 0.6049 / 0.6104: below the SEX-blind coarsening at matched total NLL by +0.034
    (se 0.010), but only on that primary cost; it spends more income NLL and Brier.
  - **The MI objective.** Not separated: +0.0076 (0.0117). The adaptive R2, with a tripled search, gave +0.0039 (0.0082).
  - **The strong start.** Not below the unchanged strong start SUCC: +0.0008 (se 0.0120). Both single-view recipient conditions fail
    against SUCC (income version 1, occupation version 0), and it spends resolved more total NLL (+0.0097, se 0.0033).
  - **The references.** Resolved below NEURAL_RAW only (+0.023, se 0.011; recipient gates pass). The 0.020 material target was not
    met.
- **Robustness reader splits** (R3 / R4, with reader seeds k2 / k3 beside R1 / R2; no retuning).
  - The pair benefits over the SEX-blind coarsening bracket and over the twin were `ROBUSTNESS_CONFIRMED`. The coarsening comparison
    remains separated on total NLL only, and the twin comparison is never sufficient.
  - The point benefits over the MI objective, NEURAL_RAW and H_LJ were not confirmed.
  - The SUCC comparison changed sign (+0.0008 to -0.0002), and a single-view recipient condition failed against SUCC and
    RETIRED_B3CTL.

**Read next.**

| File | What it holds |
|---|---|
| `RESEARCH_DECISION.md` | the decision, the seven answers, uncertainty, expectations against outcomes, what is not claimed |
| `ENDPOINTS.json`, `FRONTIER.json` / `FRONTIER.csv` | machine-readable endpoints per reading; the usefulness-recovery frontier (MODEL_SELECT rows, MODEL_FIT in-sample and nested rows, optimistic rows) |
| `TABLES.md` / `TABLES.json` | usefulness and recovery of every member; every paired comparison; MODEL_FIT diagnostics |
| `PROTOCOL.md`, `roles/statistical_reviewer/RULES.md` | the registered design and decision rules (immutable); `PROTOCOL_DEVIATIONS.md` (append-only) |
| `cards/` | the mechanism card DR_DRF_v1 and the revision cards DR_DRF_R1 (MODEL_FIT trigger) and DR_DRF_R2 (adaptive) |
| `roles/` | coordinator log; investigator A's exact-AUC derivation and tiny exact instances; reviewer's rules, pre-registration and claims reviews; verifier's checks and hand-backs (verbatim) |
| `QUICKSTART.md` | commands: read results, run selftests, use a deployment package, re-check custody |
| `HANDOFF.md` | state, resources, data access, custody and its gaps, owner actions |

All recovery results are exploratory development evidence on reused MODEL_SELECT households, for the declared reader bank. Intervals
do not correct the adaptive search, and nothing here is a confirmation.
