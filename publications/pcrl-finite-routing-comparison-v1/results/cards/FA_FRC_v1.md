# FA_FRC_v1: propensity-routed finite cells with partition-preserving probability constructions (family FA, coordinator)

Campaign `pcrl_finite_routing_comparison_v1`. Exploratory development on previously used public-use ACS 2019 1-Year PUMS Florida
records, not confirmation. Written 2026-10-20 from 23:20Z (`date -u`), before any label read of this campaign.

- **Governs:** every fit of family FA in this campaign (units `FRC_NEST_*`, `FRC_TWB_*`, `FRC_PATHS_*`, `FRC_CFG_*`, `FRC_CHECK_*`,
  `FRC_ABL_*`).
- **Code:** `frc/fa/route.py` (deployment interface `frc/fa/release.py`), importing the source `nrm/fa/prc.py` unchanged.
  Invented-data selftest: `python3 -m frc.fa.route selftest` (45 checks, all PASS at 23:37Z; it includes bitwise equality of
  construction A with the source pipeline (every entry except the changed SHUF), the nested estimator and the source spend, the SHUF
  checks, label-perturbation checks and the Lloyd fixed point B1 = B0).
- **Source mechanism:** `results/pcrl_new_release_mechanisms_v1/cards/FA_PRC_v1.md` (sections 1, 4, 5, 7, 10). Everything not changed
  below is exactly that card.
- **Protocol:** `results/pcrl_finite_routing_comparison_v1/PROTOCOL.md`; rules: `roles/statistical_reviewer/RULES.md`.
- **Readers:** "recovery" means SEX recovery by the declared development readers from the transmitted vectors. It is never
  protection against every reader.

## 1. Operation (raw fields -> income vector, occupation vector; decisions = argmax)

As FA_PRC_v1 section 1: Z = `ohn.fam_a.compact_po(X)`; statistics qI, qO (task teachers, A1_PO_16 learner settings, full-pool refits
at the inner-CV T*), and, in protected configurations only, pi = P(SEX = 1 | x) and qO0 / qO1 (SEX-conditional occupation posteriors);
the routing rule (k, l) = argmin CE(qI, cI_k) + CE(qO, cO_l) + t [D(pi; rP_kl) + D(pi; rI_k) + D(pi; rO_l)] + t kappa [guard]
(t = 0 for the twin and ORD).

**Changed: the emitted vector of each cell (the routing rule and its partition are untouched).**

| Construction | Emitted vector of a cell | Used by |
|---|---|---|
| A | label-refit decoder (MODEL_FIT label counts of the out-of-fold-routed people + 20 x codeword) / (n + 20) | the source twins; the nested comparison |
| B1 | mean of the out-of-fold teacher predictions (5 household folds `NRM_FA_xf_S<v>`) of the people whose out-of-fold route is the cell, clipped at 1e-6 and renormalized; an unoccupied cell emits its prototype (counted) | T2 twins; every protected-stage configuration if selected |
| B0 | the stored routing prototype (codeword) | the nested comparison only (diagnostic) |

- Means use out-of-fold predictions, never full-fit predictions of training rows (those are in-sample; only 56-76% of MODEL_FIT
  people keep their out-of-fold cell under the full-fit teachers).
- One construction is selected by PROTOCOL section 6.2 and used consistently in every protected-stage configuration and control,
  recomputed on each configuration's own final out-of-fold partition.
- **Disclosure.** Unchanged: each recipient sees only its own vector (and decision); the linked pair sees both. Internal cells
  emitting identical vectors are one observable value (checked: distinct vectors = occupied cells).

## 2. Closest prior attempt; what differs

- The source family (FA_PRC_v1) never fitted its protected stage. Its task-only twins (A) passed the original point limits but
  failed the internal capacity gate. This card separates the probability construction from the partition and asks whether the
  decoders caused avoidable held-out error (PROTOCOL Q-PE).
- B1 is the Bregman (KL) centroid of the final partition, that is one Lloyd codeword update without re-routing. It is not a new
  mechanism; it is the plug-in probability the routing objective itself assumes.
- Not novel: deterministic privacy funnel / IBSI family; FARE is the published finite-cell input-routing method.

## 3. Task-only twin and its stages

- **T0 (`repro`, label-free):** the saved NRM1 twins PRC_TWIN (16 x 64) and PRC_TWIN_C128 (24 x 128) reproduce their saved
  releases on all six roles and their books from the cached out-of-fold teachers, bitwise.
- **T1 (`nest`, MODEL_FIT income and occupation, no SEX):** the NRM1 gate estimator (5 outer household folds `NRM_FA_ncv_S<v>`,
  whole pipeline refitted on each outer-training part at the full-pool T*), emitting CONT, A, B0, B1 on identical held-out cells;
  reproduces the GATE records (A) to <= 1e-12 and the saved decoders from the full-pool labels bitwise.
- **T2 (`twb`, label-free):** TWB_C64 and TWB_C128 = the saved twin's teachers and book with B1 tables from the cached full-pool
  out-of-fold predictions; partition identity with the saved A release on all six roles is asserted.
- **T5 (conditional):** TWB_C256 (or TWA_C256 if A is chosen at (32, 256)); the book is refitted label-free from the cached
  out-of-fold teachers (KL k-means++ seed SeedSequence([v, 32, 256]), 50 Lloyd steps).
- SEX is never read by a twin unit.

## 4. Protected stage (only after the committed qualification decision `decisions/QUAL_DECISION.json`)

Exactly FA_PRC_v1 sections 1 and 7, with the selected construction recomputed on every partition:
- **P1 (`paths`, MODEL_FIT income, occupation, SEX as a training label):** full-pool SEX-side teachers (pi; occ_s on [Z, s]) with
  inner household CV T*, full refits and out-of-fold predictions; the nested estimator over the PRC, HQ and SHUF chains (lambda in
  {0.25, 0.5, 1, 2, 4, 8}, tilts lambda / 2^j, j = 0..6) and the ORD ladder of the protection capacity ((12, 48), (8, 32), (6, 24),
  (4, 16), (3, 12), (2, 8), with (16, 64) prepended at (24, 128) and (24, 128), (16, 64) prepended at (32, 256)); selection by
  `nrm.fa.prc.select_path` at S1-S3 (spend over the twin, nested estimate + 2 household-clustered SE <= budget, every metric).
  P1 also records the bracketing points (the first strength in path order whose nested total-NLL spend over the twin is >= that of
  the PRC configuration selected at S2, for HQ, SHUF and ORD), the nested held-out pair LOOKUP AUC of every entry with a MODEL_FIT
  household-bootstrap SE of its difference from the twin, and F4 for the S2 configuration (a D2 conjunct, not a stop).
- **P2 (`cfg`):** PRC_S1, PRC_S2, PRC_S3, ORD_S1, ORD_S2, ORD_S3, HQ_S2, SHUF_S2, HQ_BR, SHUF_BR, ORD_BR from the full-pool
  statistics. NOT_FEASIBLE budgets are not fitted; aliases are not saved twice. Each record carries the fitted objective against the
  twin's, the fitted pi-weighted pair MI, the people re-routed relative to the twin (MODEL_FIT out-of-fold and MODEL_SELECT),
  occupancy and the reader budget. PRC and the ablations store all four teachers; HQ, SHUF and ORD store the task teachers only (SHUF
  also its donor bank).
- **P3 (`check`, MODEL_FIT SEX as an evaluation label):** household 5-fold cross-fitted LOOKUP of the out-of-fold-routed fit cells
  of every saved release; labelled "routing optimized on these people (optimistic)"; never a credit input.

**Change from FA_PRC_v1 (before any SHUF fit has ever run; adjudication of challenger 6):** SHUF replaces pi by the keyed-hash donor
copy (`NRM_FA_shuf_S<v>`, bank = the training rows' out-of-fold pi) and sets the guard's qO0 = qO1 = the person's own qO, so the
guard is a SEX-blind task-fidelity term instead of a pull towards another person's occupation posterior. The chain is the source
chain with no source restriction on that copy (`frc.fa.route.chain_inputs`).
- **E4 (`cfg` with ABL_LOC / ABL_PAIR and `--ablation-lambda`):** the bounded ablation of PROTOCOL 10.5 at the strongest feasible
  nonzero release's lambda.

Budgets (unchanged from FA_PRC_v1):

| Budget | occ NLL | inc NLL | men's / women's occ Brier | occ acc loss | inc acc loss |
|---|---|---|---|---|---|
| S1 | 0.004 | 0.002 | 0.0015 | 0.003 | 0.002 |
| S2 | 0.008 | 0.004 | 0.0030 | 0.006 | 0.004 |
| S3 | 0.012 | 0.006 | 0.0045 | 0.008 | 0.006 |

## 5. Controls and falsifying outcomes

Controls (PROTOCOL section 8): TWIN; HQ (task-score-only grouping, same search) and SHUF (shuffled signal, same search) at S2 and
at their bracketing strengths; the ORD compression path with its bracketing rung; the nine references. Falsifying outcomes, read
on MODEL_SELECT (`frc.evaluate` spend, FRC1 grid readers) except F4 (MODEL_FIT):
1. **Q-PE:** B1 is not below A in nested held-out occupation NLL in both versions -> H-PE is not supported; A stays the construction.
2. **TWIN:** the PRC configuration selected at S2 is not `AGG_POSITIVE` against TWIN -> no protection from routing at this budget.
3. **SHUF:** PRC_S2 is not `DOMINATES_CONTROL_FRONTIER` against the SHUF path at equal or higher control spend -> untargeted
   re-routing explains any gain.
4. **HQ:** the same against the HQ path -> propensity routing adds nothing beyond task-score grouping.
5. **ORD:** the same against the ORD path -> not separated from ordinary compression.
6. **F4 (MODEL_FIT, before any MODEL_SELECT read of a protected release):** the nested held-out pair LOOKUP AUC of the S2
   configuration is not below the twin's by more than 2 SE in both versions -> D2 not met (`NOT_MET_F4`); a registered trigger of R2.
7. **Reader targeting:** a benefit not shown (point, both versions) by the non-table winner and by the nested MODEL_FIT LOOKUP is
   `READER_TARGETED` and credits nothing (the objective is itself the pi-weighted plug-in LOOKUP statistic).

## 6. Fitting plan (one thread; all through `frc.jobqueue`; every command ends `--card results/pcrl_finite_routing_comparison_v1/cards/FA_FRC_v1.md`)

| Step | Units | Command (`python3 -m frc.fa.route ...`) |
|---|---|---|
| T0 | FRC_REPRO_V0_V1, FRC_REPRO_V1_V1 | `repro --version v` |
| T1 | FRC_NEST_C64_V<v>_V1, FRC_NEST_C128_V<v>_V1 | `nest --capacity C64 / C128 --version v` |
| T2 | FRC_TWB_C64_V<v>_V1, FRC_TWB_C128_V<v>_V1 | `twb --config TWB_C64 / TWB_C128 --version v` |
| T5 | FRC_NEST_C256_V<v>_V1, FRC_TWB_C256_V<v>_V1 | conditional (PROTOCOL 6.4) |
| P1 | FRC_PATHS_V<v>_V1 | `paths --version v` |
| P2 | FRC_CFG_V<v>_V1 | `cfg --config PRC_S1,PRC_S2,PRC_S3,ORD_S1,ORD_S2,ORD_S3,HQ_S2,SHUF_S2,HQ_BR,SHUF_BR,ORD_BR --version v` |
| P3 | FRC_CHECK_V<v>_V1 | `check --config <saved configurations> --version v` |
| E4 | FRC_ABL_V<v>_V1 | `cfg --config ABL_LOC,ABL_PAIR --ablation-lambda <lambda> --version v` |

Unit ids are append-only; a retry gets a new suffix (_V2, ...). CPU estimate: T1 about 1 min per unit; P1 about 40 min per version at
16 x 64 (about 2 h at 24 x 128); P2 about 5 min; P3 about 1 min.

## 7. Configurations

TWB_C64 and TWB_C128 (4); TWB_C256 or TWA_C256 (2, conditional); PRC_S1-S3, ORD_S1-S3, HQ_S2, SHUF_S2 (at most 16); HQ_BR, SHUF_BR,
ORD_BR (at most 6); ABL_LOC, ABL_PAIR (4). Any revision uses new configuration ids under a new card version.

## 8. Revision plan (at most two revisions, each with a committed card before its fit)

- R2 optimizer (FA_PRC_v1): the fitted pi objective falls but the held-out LOOKUP does not -> exact single-person moves under the same
  objective (rule refit), not a new lambda grid.
- R3 guard (FA_PRC_v1): the men's guard binds first along the nested path -> kappa = 4, once.
- Each revision record: the failure, the change, the control it is compared with, its stopping rule.
- **Stop:** falsifiers 2-5 on MODEL_SELECT after the allowed revisions, or no qualifying task-only release (PROTOCOL 6.6) -> the
  family closes; protected routing is reported as untested in the second case.

## Addendum A1 (2026-10-20T23:55:53Z, before P1; engineering review findings 1-5, 9, 10)

- **F4 bootstrap (registered here):** B = 400 MODEL_FIT household draws, numpy default_rng(SeedSequence([2026102100, v, 4])), the
  same weights for the twin and every entry (paired). The SE is conditional on the fitted LOOKUP tables (table refit variance is
  excluded); this is stated in every F4 record.
- **Source pins:** every read of a source artefact (TASK.pkl, the saved twins' MODEL.pkl and RELEASE.npz, the GATE records) must
  equal its SOURCE_ADMISSION.json pin.
- **Write-once guards:** a stage refuses to start when its write-once outputs exist; P2 and P3 require the SEX-side teachers to be
  the ones the P1 selection used (SHA-256).
- **Ablation strength:** the ablation lambda must be the selected grid lambda of a saved PRC configuration of the same version.
- No operation, rule, budget or control of this card changes; T0-T2 ran under the card's earlier text (its SHA-256 is in their
  records and in the access log).
