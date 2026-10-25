# Design brief: direct-recovery finite release (campaign `pcrl_direct_recovery_frontier_v1`, code `drf/`, tag DRF1)

Coordinator, written 2026-10-25 from 02:36Z (`date -u`), before any label read of this campaign. Exploratory development on the
previously and adaptively used Florida ACS 2019 development cohort (admitted public-use PUMS records). Statistical privacy research:
SEX is the recorded demographic column whose recoverability from released task outputs is measured. Nothing here is confirmation.

## 1. Question and the changed ingredient

Can a realizable many-to-one probability release keep the original 28 usefulness conditions (12 task clauses + 2 within-SEX
occupation guards per version, two versions) while reaching a linked-pair recovery that the strongest existing releases have not
reached reliably?

The closed finite-routing line (NRM1 FA_PRC_v1, FRC1, FSA1; `nrm/fa/prc.py`, `frc/fa/route.py`, `fsa/fa/solver.py`) minimized a
Lagrangian: task cross-entropy to codewords + lambda x (pi-weighted plug-in mutual information of the pair and of each recipient's cell)
+ lambda kappa x a per-SEX guard, with a damped Lloyd / tilt-line optimizer, then chose lambda afterwards by a nested held-out utility
screen. Its protected path saturated (spent about 0.002 nats of a 0.008-0.012 budget) and was not separated from task-only controls.

**Changed here (one concrete change, two parts):**
1. the objective is the **exact optimal-reader AUC** of the emitted finite law (pair + income view + occupation view), not mutual
   information or another dependence proxy;
2. the usefulness conditions enter the optimizer as **explicit constraints** (every move must keep every spend within its budget),
   instead of a penalty path followed by a utility screen.

The comparison that isolates it: the same family, fine cells, moves, constraints, data exposure and optimizer budget, with the old
dependence objective (the PRC pi-weighted plug-in MI, same pair / local weights) in place of the exact AUC (control OBJ_MI). This
overlaps privacy funnels (deterministic, information-bottleneck style) and the earlier SCM / FRC finite routing; no novelty is
presumed (D6 NOT_CLAIMED).

## 2. Data and runtime contract (unchanged definitions)

- Six fields AGEP, WKHP, COW, SCHL, MAR, POBP (admitted 145-column encoding X). Exact income (2) and five-class occupation tasks. Each
  recipient receives only its own float64 probability vector and argmax decision; the linked-pair recipient has person-level linkage.
  One released instance per person. No cell identifier, routing key, original field, donor record or extra feature is disclosed.
- MODEL_FIT (20,000 people, 13,337 households): all fitting (task teachers are the admitted NRM1 / FRC1 teachers; release parameters,
  output support, mapping rules and tables are fitted here; SEX a training label for the objective and the within-SEX guards).
- MODEL_SELECT (8,001 people, 5,306 households): logged development utility evaluation and reader fitting / selection / scoring on
  prospectively separated household subroles. Never a fitting input of a task learner, calibration map or release parameter.
- **CALIBRATION is not read by this campaign** (no labels). ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT labels, the sealed assessment and
  capped-out households stay closed; their features are used label-free only to compute the release on every role (deployment parity),
  as in earlier campaigns.
- At inference the mapping uses only the six fields and its frozen model. SEX and true task labels may inform training, never release.
  No person identifier is used to memorize an assignment.

## 3. The release family F (deterministic, finite)

- **Statistics (frozen, admitted, version-specific):** the NRM1 task teachers qI (income) and qO (occupation) (A1_PO_16 HGB settings;
  `nrm` fa_cache TASK.pkl, MODEL_FIT out-of-fold predictions; full-pool models deployed) and the FRC1 P1 SEX-propensity teacher
  pi = P(SEX = 1 | x) (`frc` fa_cache SEXSIDE.pkl, out-of-fold on MODEL_FIT; full-pool model deployed). All are functions of the six
  fields at runtime.
- **Fine cells (the legal input representation):** income fine cell (k, m) and occupation fine cell (l, m), where k (16) and l (64)
  are the admitted task-only twin's nearest codewords (NRM1 PRC_TWIN book; the FRC1 TWB_C64 partition) and m (4) is the quartile bin
  of pi (thresholds frozen from MODEL_FIT out-of-fold pi). 64 income and 256 occupation fine cells.
- **Atoms and emitted vectors:** income atoms 1..16, occupation atoms 1..64 (capacity (16, 64), the twin's). A mapping is a table
  A_I: (k, m) -> income atom, A_O: (l, m) -> occupation atom. The emitted vector of an atom is the B1 table: the mean MODEL_FIT
  out-of-fold teacher posterior of the people the mapping sends there (clipped at 1e-6, renormalized), which is exactly the object
  used for every task cost inside the optimizer. Decisions = argmax.
- **Identity action:** A_I(k, m) = k, A_O(l, m) = l reproduces the admitted twin TWB_C64 (B1) exactly; it is always the start, so
  infeasibility cannot come from omitting the family's admitted release. SUCC (the strong anchor) is not in the family: its components
  are in-sample on MODEL_FIT, so a quantized SUCC cannot be estimated there without in-sample bias; SUCC is the unchanged strong START
  comparator, and a quantized approximation is never equated with it.
- Recovery depends only on the partition of people into observable atoms (identical emitted vectors are merged before any recovery
  computation); task costs depend on the partition and the emitted tables.

## 4. Objectives, constraints, optimizer

- **DR (candidate):** minimize J_AUC = (AUC*_pair - 1/2) + (AUC*_inc - 1/2) + (AUC*_occ - 1/2), where AUC* is the exact optimal-reader
  AUC (likelihood-ratio ranking, ties 1/2) of the finite law of (emitted wire, SEX). Law: the **estimated** conditional law (each MODEL_FIT
  person contributes pi to the SEX = 1 mass and 1 - pi to the SEX = 0 mass of its atom).
- **OBJ_MI (control, previous objective):** minimize J_MI = MI_pair + MI_inc + MI_occ (PRC's pi-weighted plug-in information, nats),
  everything else identical.
- **Constraints (both):** the MODEL_FIT spend over the identity (twin) on 8 metrics (income / occupation NLL, summed Brier and accuracy
  loss; men's and women's occupation summed Brier), with the emitted B1 tables and true MODEL_FIT labels and SEX, must stay <= budget.
  Budget per metric and version = 0.6 x the twin's published MODEL_SELECT slack to the original cap (min over U0 / Ucal; FRC1
  `dev/UTILITY_LEDGER.csv`): an informed, adaptive setting from published aggregates (no new MODEL_SELECT read); disclosed.
- **Optimizer (bounded, deterministic):** feasible local search from the identity. Sweeps over fine cells in a seeded order; for each
  fine cell all alternative atoms of its recipient are evaluated exactly; the best feasible improving move is taken; stop at a sweep
  without improvement or at the sweep cap. Bounded restarts. Internal iterations are recorded, never counted as specifications.
- **Tiny exact instances** (synthetic): exhaustive constrained optimum versus the local search, to distinguish an optimizer defect from
  a failed objective.

## 5. Optimistic constructions (MODEL_FIT only; diagnostics, not candidates)

- **OPT_S (oracle):** fine cells split by the person's true SEX ((k, m, s), (l, m, s)); true labels; in-sample. It assigns outputs using
  true per-person SEX: illegal at release by construction.
- **OPT_L:** the legal fine cells with the true-SEX empirical law, in-sample (memorization / estimation optimistic).
- Both deterministic; neither relaxes another deployment restriction. Their in-sample values and their nested held-out values are
  reported. Neither success nor failure proves population feasibility or impossibility. No optimistic result gates the legal fit.

## 6. The legal mapping and the comparisons

- **DR** is fitted on all MODEL_FIT (estimated law) and deployed from the six fields. Internal validation: nested MODEL_FIT estimator
  (5 outer household folds; the release optimization and tables refitted on outer-training households, task spends and held-out
  LOOKUP / exact recovery on the held-out fold; teachers fixed at their admitted out-of-fold statistics: disclosed).
- **Comparators on MODEL_SELECT:**
  - START = SUCC (unchanged strong starting release, frozen, both versions);
  - CTRL = the simplest untargeted operation in the same family at comparable cost: the SEX-blind task-only coarsening path from the
    twin (greedy atom merges by least total-NLL increase); two bracket points saved before any MODEL_SELECT read on the MODEL_FIT nested
    total-NLL spend (largest <= DR's spend; smallest >= DR's spend + 2 SE), MODEL_SELECT frontier rule as inherited;
  - OBJ_MI = the previous dependence objective in the same family (section 4);
  - TWIN = the family origin TWB_C64 (frozen; reported, a weak twin);
  - the nine references NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100 (+ U0 / Ucal diagnostics), re-read on the new
    grid with the same reader fitting and selection opportunity.
- **Readers:** fresh grid DRF1 inside MODEL_SELECT (repetitions R1, R2 registered; R3, R4 robustness; FIT 0-4 / VALIDATION 5-6 / SCORE
  7-9 household tenths; reader seeds MODEL_SEED_BASE + 1100 + 10 v + k, offsets 300-1000 forbidden; bootstrap base 2026102500); the
  admitted 16-recipe bank on the transmitted vectors and decisions only. Each recipient view and the linked pair.

## 7. Budget

At most 32 new version-specific release specifications (each compared operating point counts; optimistic constructions included);
at least 16 reserved for legal mappings, controls and follow-through. Initial plan: OPT_S 2, OPT_L 2, DR 2, OBJ_MI 2, CTRL 4 = 12.
Two evidence-driven revisions, each with a prospective note. Eight elapsed hours from 02:17:51Z, 12 CPU-hours, zero paid compute, one
queue, at most two heavy jobs with one thread each, at most six live agents including the coordinator.
