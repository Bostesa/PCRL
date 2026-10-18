# Exposure map (registration file; written before any fit or label read of this campaign)

Campaign `pcrl_correction_transfer_stability_v1` (exploratory development on previously used records, not confirmation).
Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, development roles of the protected-tree
replication. Labels are the income class (2), the occupation group (5) and the recorded SEX column; SEX is a training-time label
and an evaluation label, never a runtime input or output.

How this map was made. Every row was established by reading the code that fitted the component (the module and function are
named in the row) and, where a number is quoted, the component's own stored record (tree counts, coefficients, strengths). It
corrects the predecessor table `results/pcrl_occupation_headroom_overnight_v1/EXPOSURE_TABLE.md` where the code says something
different (section 3). No label was read to write it. Statements about this campaign's new components describe the registered
design (`DESIGN_BRIEF.md`, `cts/common.py`); if the code that implements them differs, the code record wins and the difference is
a deviation.

Roles (people / households): MODEL_FIT (MF) 20,000 / 13,337; CALIBRATION (CAL) 2,982 / 1,974; MODEL_SELECT (MS) 8,001 / 5,306.
ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels are closed for the whole campaign (their features may be used label-free);
the sealed assessment and capped-out records are closed. Partitions of MF (`PARTITIONS.json`, label-free household hashing):

| Partition | FIT | GUARD | CHECK |
|---|---|---|---|
| D0 (seed 0) | 8,419 / 5,603 | 2,816 / 1,866 | 2,787 / 1,866 (pool) |
| D1 (seed 1) | 8,390 / 5,603 | 2,825 / 1,866 | 2,807 / 1,866 (pool) |
| R0 (seed 0) | 10,523 / 7,002 | 3,499 / 2,333 | C0 = 2,981 / 2,001 |
| R1 (seed 1) | 10,506 / 7,002 | 3,516 / 2,333 | C1 = 2,997 / 2,001 |

If the support check moves a partition to the next tag suffix, these counts change and `PARTITIONS.json` is the record.

Words used below. "In-sample on X": the component's parameters were fitted with the labels of the people in X. "Excluded-household":
no household of the evaluated people contributed a label to the layer named. "Seed s" is the replication seed of the frozen
component (D0, R0 use seed 0; D1, R1 use seed 1).

## 1. Exposure table

| Component | Operation (as in the code) | Labels seen (people / role) | Label-free exposure | Used here as | Consequence |
|---|---|---|---|---|---|
| **Shared input** | | | | | |
| 145-column input encoding (`acs_replication/extract.py` `fit_encoding`, `encode`) | AGEP and WKHP z-scored with MF mean and SD (stored float32); COW, SCHL, MAR, POBP one-hot; a level is kept iff at least 10 MF people have it, otherwise the block's OTHER column (every block has one) | none | inputs of all 20,000 MF people (moments, level counts), including every GUARD, CHECK and C0/C1 person | input of every component | Track F re-collapses on FIT (a kept level with fewer than 10 FIT people goes to OTHER); the result equals a dictionary built from FIT codes alone. The z-map still carries all-MF moments, but it is strictly increasing and affine and the tree learners bin by value order, so fitted trees do not depend on it (up to floating-point rounding of bin edges). No neural layer is refitted here. |
| **Replication Stage A (per seed s)** | | | | | |
| Internal early-stopping split (`acs_replication/data.py` `internal_split`) | 15% of MF households by a label-free hash (seed + 17001); whole-household class-support repair moves a held-out household into the fitting part if a class of either task is missing there | income and occupation of MF, only for the repair check (0 households moved on both seeds) | MF household ids | the same split selects U0's epoch count and both T_TREE tree counts | a label use in principle, none in effect on these seeds. Track R calls the same function on FIT. |
| U0 per task (`acs_replication/stage_a.py`, `neural.py` `fit`) | network d-64-64-16-K, Adam lr 1e-3, batch 512, at most 30 epochs, patience 5; epoch count on the internal split; refitted from scratch on all MF people | its task label, all 20,000 MF people | MF inputs | utility reference: 24 clauses (with Ucal) and the 4 subgroup guards | in-sample on every MF person, so its values on FIT, GUARD, CHECK and C0/C1 are optimistic. The witness's baseline slack against U0 is therefore measured on MS only. |
| Ucal per task (`neural.py` `fit_temperature`) | U0 logits divided by one temperature per task, bounded scalar search on CAL NLL | U0's MF labels; CAL task label, all 2,982 | as U0 | utility reference | in-sample on MF (through U0) and on CAL (temperature). CAL is not a clean holdout against Ucal. |
| T_TREE per task (`stage_a.py`, `acs_replication/trees.py`) | CatBoost 1.2.8, depth 4, lr 0.05, l2 3, at most 800 trees, one thread; tree count = first minimum of the internal-split loss + 1 (seed 0: income 587, occupation 800 = the cap; seed 1: 529, 799); refitted on all MF people with that count and the same CatBoost seed; converted to a numpy routing checked bitwise against CatBoost | its task label, all 20,000 MF people | MF inputs | FROZEN routing of the Track F correction; base of TASK_REFIT, LOCAL_TREE, JOINT_TREE; one column of H3's stack; task-only reference | the routing (which people share a leaf) was learned from the labels of every MF person, including every GUARD, CHECK and C0/C1 person; out of sample on CAL, MS and later roles. The occupation tree count sits at the 800 cap, so a refit on fewer people may also hit the cap. |
| **Replication Stage B (per seed s; `acs_replication/stage_b.py`, `leafcorr.py`)** | | | | | |
| TASK_REFIT | leaf-score corrections theta on the T_TREE routing of both tasks; AdamW lr 0.01, weight decay 0.01, 20 epochs of 40 steps of 256 rows, E20 kept; lambda = 0 | MF income and occupation, task batches over all 20,000 | MF inputs; label-free probe (512) and reference (256) rows of MF | task-only reference | in-sample on MF. SEX is read by the run and the critics are trained, but with lambda = 0 they carry no gradient: theta does not depend on SEX. |
| LOCAL_TREE | same loop; objective adds risk(income) + risk(occupation) from online SEX critics (linear, MLP32, MLP64 per wire; 80 warm rounds, 5 critic rounds per update); coefficient lambda_J, calibrated at theta = 0 from 16 draws as the ratio of task to joint-objective gradient norms (the same lambda as JOINT_TREE on each seed) | MF income, occupation and SEX, all 20,000 (task pool and privacy pool are both all of MF) | as above | its income correction D_L = s_L - s_T is transplanted inside H3; LOCAL income = H_LJ and H_LL income | in-sample on MF; SEX entered at training time for every MF person, including every GUARD, CHECK and C0/C1 person. The income correction's gradient uses only the income-wire critics, but its strength coefficient comes from the joint objective, which contains the occupation wire. |
| JOINT_TREE | as LOCAL_TREE plus the linked-pair critic; both recipients trained together | MF income, occupation and SEX, all 20,000; the pair term ties the income correction to the occupation recipient | as above | D_J = s_J - s_T is H3's budgeted direction; JOINT occupation (E20) is the occupation output of H_LJ, SUCC, TWJ, B2TWIN and RETIRED_B3CTL | in-sample on MF; D_J depends on MF income, occupation and SEX labels of every MF person. |
| **Inherited references** | | | | | |
| NEURAL_RAW (`acs_replication/neural_raw.py`) | starts from U0; all network weights trained with the joint objective, the same critics, row draws and calibration as the tree arms (Adam 1e-4, clip 5); grid rho {1, 3, 10} x checkpoints {E2, E5, E10, E20}; rho 1 E2 chosen by the replication's registered rule on MS | MF task labels and SEX (all); MS labels for the choice | MF inputs | eligible neural reference | in-sample on MF; its MS values are post-selection. |
| H_LJ, H_JJ, H_LL (frozen bank) | no fit: H_LJ = LOCAL income + JOINT occupation; H_JJ = JOINT_TREE; H_LL = LOCAL_TREE (manifest checks bitwise identity with the sources) | as LOCAL / JOINT | as above | H_LJ eligible tree reference; H_JJ, H_LL descriptive; H_LJ's MS standard errors are the reserve rule's SE floor | as LOCAL / JOINT; the reserve floor is an MS quantity. |
| B2TWIN, RETIRED_B3CTL (`amd/fam_b.py` `fit`) | income = L2 logistic map h of a one-hot 6-level code of LOCAL (B2) or 0.5 LOCAL + 0.5 JOINT (B3) income log-odds; aligned bins (3 quantile bins on each side of 0; thresholds = quantiles of the MF log-odds); uncoupled (h does not use the occupation output); occupation = JOINT E20; h fitted on all CAL; 0.5 chosen on a CAL screen; the uncoupled twin and the finalist promotion were adaptive steps on MS development proxies | through LOCAL / JOINT: MF income, occupation, SEX; CAL income (all 2,982); MS labels (adaptive choice) | MF income log-odds of in-sample models (thresholds; no label read by that step) | code references | in-sample on MF and CAL. |
| SUCC, TWJ (`ohn/devrun.py` `COMBINED`) | no fit: SUCC = H3 income + JOINT E20 occupation (the source successor); TWJ = HC_STRONG income + JOINT E20 occupation | as H3 / HC_STRONG plus JOINT | as above | strong references | as H3 / HC_STRONG and JOINT. |
| O1_SEXBAL (`ihs/fam_o.py`; not a comparator here) | JOINT occupation with a CAL temperature and a SEX-balanced per-class bias fitted on MF against the stored U0 MF output | CAL occupation; MF occupation and SEX | MF inputs | not used | predecessor row confirmed. |
| **H3_STRONG_LJDIR_PB and HC_STRONG (per seed; `ihs/fam_h.py`, `ihs/bases.py`)** | | | | | |
| Step 1 HGB income learner (`step1`) | scikit-learn HistGradientBoosting, 7 settings (compact coding: 8/16/32 leaves x min leaf 20/80; one-hot: 16 leaves, min leaf 20), lr 0.05, at most 600 trees, l2 1; setting and tree count = smallest household 5-fold out-of-fold Brier on MF; refitted on all MF (seed 0: 16 leaves, min leaf 20, 94 trees; seed 1: 8 leaves, min leaf 80, 185 trees). An ExtraTrees comparison model is fitted too and not used | MF income, all 20,000; no SEX | MF inputs | first column of the STRONG stack | in-sample on every MF person, including GUARD, CHECK and C0/C1. |
| STRONG stack | 3-coefficient logistic regression of income on (s_HGB, s_T), all CAL | CAL income, all 2,982 | | HC_STRONG base | in-sample on CAL; both inputs out of sample there. |
| LJDIR transplant and beta budget (`budget_select`, H3 = precision-aware rule) | d = a (s_B + D_L + beta D_J) + b; beta = largest value of 0, 0.05, ..., 2 that, with every smaller value, keeps household 5-fold cross-fitted CAL costs against the cross-fitted task-only base inside B1 (Brier and NLL plus 1 SE within 0.0045 and 0.0080; expected accuracy cost within 0.0035); chosen beta 0.25 (seed 0), 0.20 (seed 1). B1 was set from T_TREE's MS income values and the reserve thresholds | CAL income; through D_L, D_J: MF income, occupation and SEX of every MF person; MS aggregates (B1 values) | | H3 = the frozen income anchor of Track F and the income of SUCC | the transplanted corrections are in-sample on every MF person. |
| Platt map | (a, b) logistic fit of income on s_B + D_L + beta D_J, all CAL | CAL income | | final H3 income map | in-sample on CAL. H3 is family H's second revision, nominated on MS by its source campaign's registered rule, and was evaluated once on DEV_REPORT there. |
| HC_STRONG | Step 1 + stack, no transplant (beta = 0); the Platt refit is the identity on the stacked score | MF income (HGB), CAL income; no SEX | MF inputs | TW anchor; TWJ income; Track R task-only twin recipe | in-sample on MF and CAL. |
| **Source occupation witnesses (occupation-headroom campaign; per seed)** | | | | | |
| A1_PO_16 deployed (`ohn/fam_a.py` `run_pipeline`, `fit`) | HGB multinomial on compact_po (AGEP, WKHP numeric; COW, MAR native categories; SCHL ordinal; POBP ordinal code), 16 leaves, min leaf 100, lr 0.025, l2 3; T* (at most 1000) = first minimum of the pooled inner 5-fold household out-of-fold NLL over all MF (seed-specific folds); refitted on all MF with T*; no calibrator (a CAL temperature is fitted as a diagnostic only, not applied) | MF occupation, all 20,000; CAL occupation (diagnostic only); no SEX, no income | MF inputs | Track F witness recipe (refitted on FIT); source witness of P2A | in-sample on MF. Its coding (revision 1) was decided after A0's seed-0 results (including an MS score) and MF-only diagnostics, and it was ranked second on MS by the witness selection: an adaptive choice. |
| A1_PO_16 cross-fitting (CROSSFIT.npz) | for each of the 5 seed-independent OHN1_crossfit household folds k: inner CV restricted to MF minus fold k gives T*_k; refit on MF minus fold k (model FOLD_k); predicts fold k (MODEL_FIT__oof); the five fold models' mean on MS (MODEL_SELECT__foldavg) | fold model k: occupation of MF minus fold k | MF inputs | base logits that trained and projected the source corrections | the witness's own cross-fitting is fully nested (inner CV inside each outer fold). Each household's out-of-fold value is out of fold for the witness, but the vector as a whole is not free of any one fold: the values of fold j (j != k) come from model FOLD_j, which trained on fold k. |
| B2_SINGLE (`ohn/fam_b.py` `run_base`, `fit_layer`) | two learners tuned by inner household CV: TREE_V2 (HGB, POBP ordinal, lr 0.05, at most 500, leaves {4, 8, 15} x min leaf {50, 200}, checkpoints of 10) and ADD_V2 (L2 multinomial logistic on a spline / indicator / interaction basis, capacity S or L x C {0.003, 0.01, 0.03}); the learner with the lower temperature-scaled inner out-of-fold NLL is released with that temperature; full nested cross-fitting over OHN1_crossfit (tuning, refit, choice and temperature redone per outer fold); fold models are not stored | MF occupation, all (fold pipeline k: without fold k) | basis parameters (spline ranges, WKHP values held by at least 200 people) from all MF inputs, shared by the fold pipelines | Track F sensitivity witness (one partition); source witness of P1, P2, P3 | as A1_PO_16; ranked first on MS by the witness selection (adaptive). |
| **Source protection stage (`ohn/protect.py`; P2 / P2A = guard 'none')** | | | | | |
| Fold pipeline k (`crossfit_train` -> `train_correction`) | theta on the FROZEN T_TREE occupation routing of seed s; base = the witness's cross-fitted MF logits; task cross-entropy + lambda x critic risk (LOC: occupation wire; PAIR: occupation wire and the linked pair with H3's MF income wire; CONT: lambda = 0); lambda calibrated at theta = 0; trained on MF minus fold k, predicts Delta on fold k | occupation and SEX of the other four folds, directly | MF inputs | (source) out-of-fold correction delta_oof | NOT nested. Fold k's labels still reach the correction that is scored on fold k: (i) the base logits of the training rows came from witness models FOLD_j (j != k) fitted on fold k's occupation labels; (ii) the routing was fitted on fold k's occupation labels; (iii) PAIR's income wire is H3, fitted with fold k's income labels (HGB, T_TREE, Stage B) and, through D_L and D_J, its occupation and SEX labels. |
| Full pipeline | the same training on all MF people with the same cross-fitted base, routing and income wire | MF occupation and SEX, all 20,000 | MF inputs | (source) deployed correction Delta_full | train/deploy mismatch: trained on the cross-fitted witness logits but applied to the deployed witness (in-sample on MF, a different and more confident base). |
| End projection (`project`, `witness_slack`, `budget_at`) | scale s = the largest point of the 101-point grid on [0, 1] with every smaller point feasible, refined by 30 bisection steps; costs = incremental men's / women's / all summed Brier, NLL and accuracy loss of probs(z_oof + s delta_oof) against the out-of-fold witness at s = 0, on all MF people; budget = f x min over references of the witness's MS reserve slack (margin - 2 x max(own SE, H_LJ floor SE)), f = 0.5 (B50) or 0.8 (B80) | MF occupation and SEX, all (costs); MS occupation and SEX through the witness's logged MS margins and SEs (budget) | | the OLD rule reproduced here | the costs that set s come from fold thetas on out-of-fold witness logits, while the release is Delta_full on the deployed witness: neither the base nor the correction is the deployed one. The budget is an MS quantity, so the MS cost of the chosen scale is not independent of the choice. Guard targets from MS slack enter training only for guards 'equal' and 'static' (P1, P3), not for P2. |
| **This campaign, Track F (per partition P with associated seed s)** | | | | | |
| Dictionary re-collapse | label-free, on FIT | none | FIT level counts; all-MF z-map (tree-invariant) | input coding of the witness and of REFIT routing | no label. A GUARD or CHECK person with a level that is rare in FIT falls into OTHER, as a new person would. |
| Witness W_P (A1_PO_16 recipe) | T* by 5-fold household inner CV on FIT; refit on FIT; cross-fitting over 5 outer household folds of FIT, each with full nested inner CV; fold models kept for the ensemble variant | FIT occupation; no SEX, no income | FIT inputs | witness and base of every correction | out of sample on GUARD, CHECK and MS; in-sample on FIT. The out-of-fold base vector has the same non-nesting as the source (values of fold j come from a model trained on fold k). |
| Routing REFIT | T_TREE occupation recipe on FIT: tree count on the 15% household internal split of FIT, refit on FIT, numpy routing checked against CatBoost | FIT occupation (the internal split's repair check also looks at FIT income if both tasks are passed) | FIT inputs | routing of the leaf-score correction | GUARD and CHECK excluded. Not refitted inside the outer folds except in the nested variant. |
| Routing FROZEN | the admitted T_TREE occupation teacher of seed s | MF occupation, all 20,000, including GUARD, CHECK, C0 and C1 | MF inputs | routing of the leaf-score correction (factor level) | costs on GUARD and CHECK under FROZEN routing are not excluded-household for the routing. |
| Directions PAIR, LOC, CONT (`ohn.protect.train_correction`, guard 'none') | 5 fold pipelines on the witness's outer folds of FIT (each trained on FIT minus fold k with the cross-fitted witness logits as base) and 1 full-FIT pipeline | FIT occupation and SEX (critics for LOC and PAIR); CONT reads SEX with zero weight; PAIR also uses H3's income wire on the FIT people | H3 income values on FIT, a function of H3 parameters fitted with every MF household | Delta_k, Delta_full | Verified in the code: the income wire reaches theta only through the pair-view critic term, so LOC and CONT do not depend on H3 at all (LOC's income-view critics are trained but carry no gradient; CONT has lambda = 0). PAIR depends on H3, whose parameters saw the income, occupation and SEX labels of every GUARD, CHECK, C0 and C1 household. |
| Releases r_s and e_s | r_s = softmax(log p_W + s Delta_full); e_s = mean over k of softmax(log p_Wk + s Delta_k); ladder s = 0, 0.1, ..., 1 | none (composition) | | ladder releases | the ensemble variant still mixes training bases (cross-fitted logits) with deployment bases (each fold model's own output). |
| OOF estimator | fold pipeline k on fold k of FIT, cross-fitted witness base (exactly the source `project` costs) | FIT occupation and SEX | | estimator OOF | not excluded-household: fold k's labels entered the witness base (always), the routing (FROZEN, and REFIT unless refitted per fold) and, for PAIR, H3. The nested variant refits every upstream layer inside each outer fold; H3 and FROZEN routing stay exposed in it. |
| GUARD estimator | deployed release (and ensemble) on GUARD | GUARD occupation and SEX, used only to choose a strength | | strength choice | excluded-household for the witness, REFIT routing and the directions; not for FROZEN routing; not for H3 (PAIR only). |
| CHECK estimator | realized costs on CHECK (D: diagnostic; R: C0 / C1, only after the committed REPAIR_FREEZE) | CHECK occupation and SEX | | transfer check | same layer exposure as GUARD, and never used for a choice. C0 / C1 are fresh only for layers fitted in this campaign: every frozen layer (U0, Ucal, T_TREE, Stage B, H3, the source witnesses and corrections) was fitted with their labels. |
| MS estimator and baseline slack | realized costs on MS; the witness's MS margin to each occupation-side cap (accuracy, NLL, Brier vs U0 and Ucal; men's and women's guards vs U0) with household-clustered SE; reserve slack = margin - 2 SE | MS income, occupation and SEX (benchmark operation) | | 24 clauses + 4 guards; the budget input of every rule | out of sample for every fitted layer (all are fitted on MF or CAL), but MS is adaptively used: by the whole lineage, by the source witness selection and here by every rule's budget. MS costs of a rule-chosen strength are not independent of the choice. |
| **This campaign, Track R (per partition R0, R1)** | | | | | |
| T_TREE refit (income, occupation) | the Stage A recipe on FIT: internal 15% household split of FIT for the tree count, refit on FIT, numpy routing checked against CatBoost | FIT income and occupation | FIT inputs | refitted teachers and routing | GUARD and C0 / C1 excluded; MF people outside FIT excluded. |
| Stage B refit (LOCAL_TREE, JOINT_TREE; TASK_REFIT if affordable) | `leafcorr.Trainer` / `run_arm` on FIT: task cross-entropy, SEX critics, 16-draw calibration, E20; probe and reference rows drawn from FIT by the same label-free hash order | FIT income, occupation and SEX | FIT inputs | refitted D_L, D_J | as above. |
| H3-recipe income refit and HC-recipe twin | Step 1 grid, household 5-fold CV and refit on FIT; STRONG stack, beta budget (cross-fitted CAL, the same B1 values) and Platt map on all CAL, with the refitted T_TREE income, D_L and D_J | FIT income (and through D_L, D_J: FIT occupation and SEX); CAL income, all 2,982 | FIT inputs | refitted income release for all six roles (ATTACK and DEV_REPORT from features only) | the refitted income lineage never saw GUARD or C0 / C1. CAL is shared with the frozen H3 and between R0 and R1, so it is not an independent holdout. B1 stays the source value, which was set from MS quantities. |
| **Rules, readers and roles as data** | | | | | |
| Strength rules | OLD: OOF FIT costs and MS reserve slack (B80, B50). SIMPLE and UNCERTAINTY: GUARD costs (with household-aware SEs for UNCERTAINTY) and MS reserve slack, fixed ladder | as the estimators they read | | chosen strengths | a rule never reads CHECK or C0 / C1. |
| Support check and source replay | class x SEX counts per partition role before any fit (support_count); recomputation of source records on the rows the source used (source_replay) | MF occupation and SEX counts, all roles including C0 / C1; MF labels for the replay | | partition validity (tag suffix); source numbers | no fit and no new release is evaluated with them on a panel. |
| Development readers (DEV_GRID) | readers of SEX from released vectors; FIT 50% / VALIDATION 20% / SCORE 30% of MS households; repetitions R1, R2 x reader seeds k0, k1 | MS SEX | MS released vectors | recovery evidence | development proxies on an adaptively used role, not a fresh assessment. |
| ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT | labels closed | none | features (deployment parity) | release computation only | no fresh-role result exists in this campaign. |

## 2. Which comparisons are excluded-household, layer by layer

"Yes" means no household of the evaluated people contributed a label to that layer.

| Evaluated on | Witness | Directions | FROZEN routing | REFIT routing | H3 income (PAIR critic) | U0 / Ucal |
|---|---|---|---|---|---|---|
| OOF, fold k of FIT (source design) | no: the base values of the training rows came from models fitted on fold k | yes for the fold-k pipeline itself | no | no (fitted on all of FIT) | no | no (in-sample) |
| OOF, nested variant | yes | yes | no | yes (refitted per fold) | no | no |
| GUARD | yes | yes | no | yes | no | no |
| CHECK (D) | yes | yes | no | yes | no | no |
| CHECK (R: C0 / C1), Track F | yes | yes | no | yes | no | no |
| CHECK (R: C0 / C1), Track R refitted income | yes | yes | no | yes | yes (refitted lineage) | no |
| MS | yes (out of sample) | yes | yes | yes | yes | yes, but MS is adaptively used and sets every budget |

Consequences for this campaign:
- Track F: H3's income saw every MF household, so a complete income-plus-occupation release is never wholly excluded-household on
  any MF subset, and every PAIR direction carries GUARD and CHECK information through H3. LOC and CONT carry none.
- FROZEN routing saw every GUARD and CHECK household; REFIT routing is the excluded-household alternative.
- U0 and Ucal are in-sample on MF (and Ucal on CAL), so the witness's baseline slack is measured on MS, not on GUARD or CHECK; the
  incremental cost of a correction (against the uncorrected release on the same people) and the baseline slack are reported
  separately.
- D0 / D1 FIT sets share about 36% of the pool and R0 / R1 FIT sets about 56%: repetitions are different training partitions of
  the same people, not independent population replications.

## 3. Corrections and additions to the predecessor exposure table

1. T_TREE: "No CatBoost is installed, so it cannot be refitted per fold" no longer holds; CatBoost 1.2.8 is installed in a private
   virtual environment for fitting only (inference stays numpy). The occupation tree count is at or next to the 800 cap on both
   seeds (800, 799).
2. The internal early-stopping split is not purely label-free: its class-support repair reads the task labels (it moved no household).
3. TASK_REFIT: the run reads SEX and trains critics, with zero weight on theta.
4. LOCAL / JOINT: the task pool and the privacy pool are both all 20,000 MF people. The JOINT income correction (D_J, used by H3)
   also depends on MF occupation labels through the pair term. LOCAL uses the joint-objective coefficient lambda_J.
5. NEURAL_RAW: the eligible configuration (rho 1, E2) was chosen on MS by the replication's selection rule.
6. B2TWIN / RETIRED_B3CTL: the code thresholds are quantiles of in-sample MF income log-odds; the income map h is fitted on all CAL
   and is uncoupled from the occupation output; the mixing weight 0.5 came from a CAL screen; the twin and promotion were adaptive on MS.
7. H3: its budget values B1 were set from MS reference values; its transplanted D_J carries MF occupation and SEX labels.
8. The reserve rule's SE floor is H_LJ's MS standard error, an MS quantity.
9. Missing from the cross-fitting paragraph: the source protection stage used the witness's cross-fitted outputs as the base of its
   own cross-fitting without nesting, and its projection combined fold corrections and cross-fitted bases while the release deploys
   the full-data correction on the deployed witness.

## 4. Possible explanations of the source transfer error, and how this campaign tests each

The source campaign's out-of-fold projection missed the realized men's-guard cost on MS by +0.0016 to +0.0029 summed Brier on
seed 0 and by -0.0008 to -0.0012 on seed 1 (slack 0.0041-0.0054). The explanations below are POSSIBLE, not established.

1. Non-nested upstream cross-fitting. The correction scored on fold k was trained on witness logits from models that had seen
   fold k, so its out-of-fold cost may be biased. Test: on at least one partition, the nested variant (every upstream layer refitted
   inside each outer fold) against the non-nested OOF at the same strengths, both against GUARD and CHECK realized costs.
2. Frozen routing. The leaves were fitted with every MF person's occupation label, so a correction attached to them may behave
   differently on new people. Test: the FROZEN versus REFIT routing factor on D0 and D1, comparing OOF, GUARD, CHECK and MS costs.
3. Exposed H3 anchor. PAIR's critic saw H3's in-sample MF income, which is more confident and more SEX-entangled than H3 on new
   people. Test: LOC versus PAIR (H3 enters only PAIR); Track R refits the H3 recipe on FIT, if it fits the budget.
4. Train/deploy mismatch. The scale was set with fold corrections on cross-fitted bases, but the release is the full-data
   correction on the deployed witness. Test: GUARD costs of the deployed release and of the deployed ensemble against the OOF
   costs at the same strengths, then CHECK.
5. Correction magnitude. The full-data correction may be larger or differently shaped than the fold corrections, and the cost
   curve in s is not linear. Test: costs over the whole fixed ladder for every estimator, and the label-free size of each
   correction (fold versus full) at equal strength.
6. Subgroup sample variation. The binding men's-guard cost is measured on a subset whose household-clustered SE is of the same
   order as the transfer error, and the error changed sign across seeds. Test: household-clustered SEs on GUARD, CHECK and MS, two
   partitions per seed association (D0 / D1, R0 / R1), and the UNCERTAINTY rule.
