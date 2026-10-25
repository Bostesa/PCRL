# DR_DRF_v1: direct-recovery finite release (family DR, coordinator)

Campaign `pcrl_direct_recovery_frontier_v1` (DRF1). Exploratory development on previously and adaptively used public-use ACS 2019
1-Year PUMS Florida records, not confirmation. Written 2026-10-25 from 02:50Z (`date -u`), before any label read of this campaign.
Governs every fit of family DR (units `DRF_FIT_*`, `DRF_OPT_*`; `python3 -m drf.dr ...`). Code: `drf/dr.py` (family, legal mapping,
nested estimator, stages), `drf/solver.py` (objective, constraints, optimizer), `drf/finite_law.py` (exact finite-law AUC,
independent; investigator A). Design: `roles/coordinator/DESIGN_BRIEF.md`; protocol `PROTOCOL.md`; rules `roles/statistical_reviewer/RULES.md`.

## 1. Finite input representation and permitted assignment information

- Runtime statistics, all functions of the six fields (AGEP, WKHP, COW, SCHL, MAR, POBP) through frozen admitted models: the NRM1
  task teachers qI, qO (A1_PO_16 HGB settings, full-pool refits as deployed in FRC1's TWB_C64) and the FRC1 propensity teacher
  pi = P(SEX = 1 | x) (full-pool refit), clipped to [0.01, 0.99].
- Fine cells: income (k, m), occupation (l, m); k / l = nearest codeword (cross-entropy) of qI / qO in the admitted twin book (16 /
  64 codewords); m = quartile bin of pi with thresholds frozen from MODEL_FIT out-of-fold pi. 64 income and 256 occupation fine cells.
- The mapping may use only these fine cells (hence only the six fields and the frozen models). It never uses SEX, a task label, a
  person or household identifier, a donor record or any undisclosed recipient information.

## 2. Emitted probability vectors and decisions

- Income recipient: T_inc[A_inc(k, m)] (2 probabilities); occupation recipient: T_occ[A_occ(l, m)] (5); decisions = argmax.
  Atoms: at most 16 income and 64 occupation (capacity (16, 64), the twin's). Only these vectors and decisions are released; one
  instance per person; deterministic.
- T = B1 tables: an atom's vector is the mean MODEL_FIT out-of-fold teacher posterior of the MODEL_FIT people the mapping sends to it,
  clipped at 1e-6 and renormalized (an atom with no MODEL_FIT person emits its twin codeword, counted). The optimizer computes every
  task cost from exactly these tables, and the deployed release emits exactly them (no prototype / emission mismatch).

## 3. Recovery objective and usefulness constraints

- Objective (DR_AUC): minimize (AUC*_pair - 1/2) + (AUC*_inc - 1/2) + (AUC*_occ - 1/2), the exact optimal-reader AUCs of the finite law
  of (emitted wire, SEX) for the linked pair and each view (likelihood-ratio ranking; ties 1/2; identical emitted vectors merged).
  Law: estimated (each MODEL_FIT person contributes pi / 1 - pi). Lower MI or any other proxy is not a substitute endpoint.
- Constraints (every accepted state): MODEL_FIT spend over the identity (twin) with the emitted tables and true MODEL_FIT labels /
  SEX, for income and occupation NLL, summed Brier and accuracy loss and the men's and women's occupation summed Brier, each <=
  0.6 x the twin's published MODEL_SELECT slack to the original cap (min over U0 / Ucal; FRC1 `dev/UTILITY_LEDGER.csv`, pinned). This
  is an informed, adaptive setting from published aggregates; no new MODEL_SELECT read. The 16 values are frozen in `dr/BUDGETS.json`
  (committed with the registration; `drf.dr.frozen_budgets` refuses any other) and copied into each
  fit record. The original 28 conditions on MODEL_SELECT remain the only usefulness criterion.

## 4. Legal mapping class

Deterministic lookup tables A_inc: 64 -> 16 and A_occ: 256 -> 64 on the fine cells, with B1 emission. The identity A_inc(k, m) = k,
A_occ(l, m) = l is the admitted task-only twin TWB_C64 (FRC1, qualified) and is always the start: infeasibility cannot come from
omitting the family's admitted release. SUCC is not a member (its components are in-sample on MODEL_FIT); a quantized approximation
is never equated with a continuous anchor.

## 5. The changed ingredient and the comparison that isolates it

- Changed: exact optimal-reader AUC (instead of the PRC pi-weighted plug-in MI) as the objective, with the usefulness conditions as
  explicit constraints inside the search (instead of a lambda path followed by a utility screen).
- Isolating comparison: DR_MI, the same family, fine cells, moves, constraints, data exposure, seed and optimizer budget with the PRC
  pi-weighted plug-in MI objective (pair + income + occupation). Their task-spend mismatch is reported; a spend difference is not
  credited to the objective.
- Untargeted control at comparable cost: DR_CTRL_LO / DR_CTRL_HI, two points of the SEX-blind task-only coarsening path of the twin
  (greedy least-total-NLL atom merges), chosen before any MODEL_SELECT read on the nested MODEL_FIT total-NLL spend (largest <= DR_AUC's;
  smallest >= DR_AUC's + 2 household-clustered SE of their paired difference). A point equal to the origin is the twin itself.
- The family change (finite release vs continuous SUCC) is reported separately (START comparison), never credited to the objective.

## 6. Optimistic constructions (MODEL_FIT only; diagnostics, never candidates)

| Name | True per-person labels | Estimated conditional law | Legal input representation | Relaxes determinism / other |
|---|---|---|---|---|
| OPT_L | yes (true-SEX empirical law, in-sample) | no | yes | no / in-sample evaluation on the fitted people |
| OPT_S | yes (and routing by the true SEX) | no | no (SEX-split fine cells) | no / not deployable |

Both use the same constraints and optimizer as DR_AUC. In-sample and nested held-out values are reported; neither proves population
feasibility or impossibility, and neither gates the legal fit.

## 7. Bounded optimizer, support, revisions, accounting

- Optimizer: feasible best-improvement local search over single fine-cell moves (every alternative atom of the cell's recipient is
  evaluated exactly) in a seeded order; a sweep without an improving single move is followed by the best feasible improving atom MERGE
  (every unordered pair of used atoms of each recipient, evaluated exactly); the search stops when neither improves, or after 30
  sweeps; then 2 perturbation restarts (16 random feasible single moves each) from that seed's incumbent; the whole procedure from 2
  seeds (1000 v and 1000 v + 1 for the full fit; 1000 v + 100 (fold + 1) and + 1 in the nested folds), keeping the best. Identical for
  DR_AUC, DR_MI, OPT_L and OPT_S. Sweeps, moves, merges and candidate evaluations are recorded as internal iterations, never as
  specifications. The merge move and the seeds were added before registration after investigator A's tiny exact instances
  (`roles/investigator_a/TINY_EXACT.json`: no implementation defect, but single-move local optima in about 10% of tiny cases, mostly
  closed by a joint multi-cell move); the re-run with these settings is `TINY_EXACT_V2.json`. A "no move" result never shows that the
  identity is optimal. Every final release is also re-scored on its merged emitted wires (`drf.finite_law`).
- Support: one capacity (16, 64) with 4 pi bins, frozen now. No capacity sweep.
- Nested MODEL_FIT estimator: 5 outer household folds (`DRF1_nest_S<v>`); the release optimization and the tables are refitted on
  the outer-training households; held-out task spends, held-out LOOKUP AUC (training true-SEX atom rates, Beta(1) shrinkage) and the
  held-out fold's exact AUC are recorded. The admitted teachers stay at their out-of-fold statistics (not refitted; disclosed).
- Specifications (version-specific): OPT_L 2, OPT_S 2, DR_AUC 2, DR_MI 2, DR_CTRL_LO 2, DR_CTRL_HI 2 = 12 of 32 (a bracket point at
  the origin is the twin and costs none). At most two evidence-driven revisions, each with its own prospective card (new config ids).
- Registered predictions (coordinator, before any label): P(DR_AUC G1-feasible on MODEL_SELECT in both versions) 0.6; DR_AUC's
  MODEL_FIT in-sample exact pair AUC below DR_MI's in both versions 0.8; P(D2 met) 0.10; P(D3) 0.03; P(D4) 0.02. Expected: the
  men's occupation guard binds first; DR_AUC lowers the twin's grid pair AUC (FRC1 grid 0.6555) by 0.01-0.03, mostly not separated
  from the coarsening control or not reaching SUCC.

## 8. Failure interpretation

Failure stages and their meaning follow RULES.md (the brief's section 8): no useful optimistic point (a failure within this finite
model, estimated law and optimizer, not a general impossibility); optimistic success but legal failure (estimation, representation,
optimization or generalization); utility kept but tied with controls (no mechanism advantage); controls beaten but strong references
missed (at most a scoped component result if robustness supports it); original competitive conditions passed (exploratory evidence
requiring fresh confirmation). What would contradict the rationale: DR_AUC not below DR_MI in its own exact objective (optimizer
defect, checked on tiny exact instances), or below in-sample but not on held-out households (estimation / generalization), or no
lower recovery than the coarsening control at matched spend (no advantage of targeting).
