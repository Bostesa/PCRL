# Shared brief for every agent of campaign `pcrl_finite_routing_comparison_v1` (coordinator, 2026-10-20, written 23:05Z)

You are one role in a bounded research campaign run by a coordinator. Read this whole file before doing anything.

## 1. What the campaign is

A software benchmark of machine-learning release mechanisms on already admitted public-use U.S. Census ACS 2019 1-Year PUMS person
records for Florida (the development cohort of earlier PCRL campaigns). Two prediction tasks: an income class (2 classes) and an
occupation group (5 classes). Each recipient (income recipient, occupation recipient) receives only its own float64 probability vector
and its argmax decision. The privacy evaluation measures whether separate classifiers ("readers") can infer the dataset's recorded SEX
column from those transmitted predictions (income view, occupation view, and the linked pair of both outputs of one person). Use the
dataset's terminology: "the recorded SEX attribute", "SEX recovery by the declared development readers". Lower recovery for these
readers is never protection against every reader. All findings are exploratory development evidence on previously and adaptively
used records (never confirmation).

**Objective.** Complete the finite-routing comparison that the previous campaign (`pcrl_new_release_mechanisms_v1`, code `nrm/`,
publication 682068d52) did not reach. Its family A (propensity-routed finite cells, card `results/pcrl_new_release_mechanisms_v1/
cards/FA_PRC_v1.md`, code `nrm/fa/prc.py`) closed at an internal capacity gate before any protected fit: its task-only twins (16 x 64
and 24 x 128 cells) passed the original point utility limits, but their nested held-out occupation NLL spend over the continuous task
teachers (+0.0107 to +0.0137 nats) failed the extra gate "estimate + 2 SE <= 0.004". Designer A's label-free note suggested that the
twin's **label-refit cell decoders** (probabilities estimated from MODEL_FIT labels inside each cell, shrunk to the codeword with 20
pseudo-counts) add an estimation term (about 0.0055 nats at 64 cells, 0.0092 at 128). That is a hypothesis, not a finding.

The study:
1. **Fixed-routing probability comparison** (the question): on identical teachers, routing functions, partitions, capacities and
   training exposure, compare A = label-refit decoders (the saved releases) with B = teacher-mean probabilities. B1 = the mean of the
   OUT-OF-FOLD teacher predictions inside each final routed cell (5 household folds `NRM_FA_xf_S<v>`, fold models at the full-pool T*;
   the same cells the label-refit decoders count); B0 = the stored clustering prototype (the routing codeword). They differ: a label-free
   pre-registration check found max |prototype - OOF cell mean| of 0.0019-0.0062 on occupation, because the last Lloyd step's codeword
   is the mean of the previous partition. Routing stays fixed: a probability change never changes the partition (checked).
2. **Task feasibility** under the ORIGINAL requirements (24 clauses + 4 within-SEX occupation Brier guards vs U0_S<v> / Ucal_S<v> on
   MODEL_SELECT, `nrm.evaluate` definitions). The 0.004-nat capacity gate stays in its original record and is NOT this study's
   eligibility rule. The two-SE reserve is reported separately. A registered qualification rule (section 5 of PROTOCOL.md) decides
   whether a task-only construction has enough slack for a small protection comparison.
3. If a construction qualifies: **fit the actual propensity-aware routing operation** of card FA_PRC_v1 with the selected probability
   construction (probabilities recomputed by the same rule after routing changes a partition), along a short predeclared budget path,
   with its controls (equally trained task-only twin, task-score-only grouping HQ, shuffled-signal SHUF, ordinary compression path ORD,
   the nine admitted references), a fresh reader grid, a bounded component ablation and a classifier refit of the strongest feasible
   nonzero release.
4. Separate decisions: repaired task-only release; protection component beyond matched controls; improvement over the strong original
   references; the 0.020 material benefit; joint-design contribution (not assessed unless its controls exist); novelty (not claimed).

Do not reopen the completed Gaussian-channel (family B / FB2) or leaf-score correction studies. Preserve the previous negative result.

## 2. Data boundaries (non-negotiable)

- Only the admitted Florida ACS 2019 cohort, roles MODEL_FIT (20,000 people, 13,337 households), CALIBRATION and MODEL_SELECT (8,001
  people, 5,306 households). ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels are closed (DEV_REPORT features are used label-free for
  deployment parity only). The sealed assessment and capped-out records are closed. No download of any people or population.
- Six runtime fields only: AGEP, WKHP, COW, SCHL, MAR, POBP (admitted 145-column encoding; the A1_PO_16 compact coding).
- Every label read goes through `frc.common.load` (logged; refuses reads before the registration receipt, closed roles, MODEL_SELECT
  labels inside fitting modules, reads outside a queue unit). Never read labels any other way. Never read a label before the
  coordinator says the registration receipt is verified.
- MODEL_SELECT labels are never a fitting input. Release parameters are fitted on MODEL_FIT (or CALIBRATION) only.
- Runtime prediction receives only the six fields. SEX is used only as a training label where the committed card says so, and as an
  evaluation label.

## 3. Code and infrastructure

- New campaign package `frc/` (derived from `nrm/` with this campaign's paths, names, entry points and limits). The source `nrm/` code
  is imported unchanged where reused. Private state: `.private/pcrl_finite_routing_comparison_v1/` (never committed). Public results:
  `results/pcrl_finite_routing_comparison_v1/`.
- Read-only source inputs: `.private/pcrl_new_release_mechanisms_v1/runs/FA/{PRC_TWIN,PRC_TWIN_C128}/V{0,1}/` (MODEL.pkl/json,
  RELEASE.npz/json), `.private/pcrl_new_release_mechanisms_v1/fa_cache/V{0,1}/TASK.pkl` (task teachers, out-of-fold predictions), the
  GATE records `results/pcrl_new_release_mechanisms_v1/roles/designer_a/GATE_V*.json`. Never write anything under the source's paths.
- **Every numerical job** (fit, nested estimate, evaluation, reader grid, verifier replay) runs through the shared queue:
  `python3 -m frc.jobqueue run --unit <UNIQUE_ID> --stage <stage> --rss-gib <GiB> -- <interpreter> -m <module> ...`. At most two heavy
  jobs run at once (slot locks), one numerical thread each. Unit ids are append-only (a retry gets a new suffix). Failed attempts are
  kept and counted.
- Work outside the queue (selftests on invented data, small label-free scripts) must be charged:
  `python3 -m frc.jobqueue charge --label <role> --cpu-s <upper estimate> --note "<what>"`.
- Budgets: 12 CPU-hours in total (everything, including failed attempts and agents' work); research window 22:49:23Z to 06:49:23Z,
  closeout until 07:19:23Z; at most 48 new release configurations (controls included; `frc.release.save` refuses the 49th); at most
  six live agents including the coordinator. **You must not spawn sub-agents or helpers of any kind.**
- Times in every record and message come from `date -u` at the moment of writing. Never round a time ahead of the clock.

## 4. Discipline

- Register before measuring: protocol, rules and the governing card are committed (and the registration receipt verified on GitHub)
  before any new computation that uses real labels. Cards and code are committed before the fits they govern.
- A prediction or decision rule written after seeing an outcome is labelled post hoc, up front.
- Never write "promising", "confirmed", "replicated", "targeting helps", "useful component" (unless the frozen rule assigns that label),
  or any general impossibility claim. "Not separated" is not "equivalent". "Point-feasible" is not "confidently feasible".
- Versions 0 and 1 share every training row and the MODEL_SELECT people: they are procedure repeats, not independent replications.
  Reader repetitions share households; ineffective reader seeds are not replications.
- Commit only your own files, with plain messages (no trailers, no co-author lines). Do not push. Do not switch branches. Do not touch
  `main`, manuscripts, earlier campaigns' locks or results.
- Agents cannot write files the coordinator did not ask for; return your report as text in your final message (the coordinator saves
  it verbatim with attribution) unless your assignment names files you own.
- Send no messages to people. No network access other than what your assignment names.

## 5. Roles

- **Coordinator** (the main session): protocol, card, queue, evaluation, reader grid, decisions, documents, publication.
- **Statistical reviewer**: freezes the claims and selection rules (RULES.md / RULES.json / frc/rules.py) before outcomes.
- **Challenger**: attacks the design, names the simplest alternative explanations, checks that the controls can falsify the claims.
- **Engineering reviewer**: reviews code for correctness, access-layer integrity, fixed-partition guarantees, reproducibility and
  deployment refusals.
- **Independent verifier**: replays the decisive numbers with separate code under `results/<campaign>/roles/verifier/code/`.
- **Implementer** (if used): writes assigned modules to a spec.
