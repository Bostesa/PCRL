# Data-use registration

Only the saved2018 three-anchor development cohorts are used. Raw path is allowlisted to `data/folktables/2018/1-Year/psam_p06.csv` within this study's owned `private/inputs` root. No global resolver is modified or permitted. Archive manifests and restored files are hash-verified. 2016 remains sealed; no path or archive containing2016 is bundled. 2017 is not used.

Inherited cohort:30,000 people in20,147 households. Pool proportions are representation_fit35%, source_validation10%, downstream_fit15%, downstream_validation10%, attacker_fit10%, attacker_validation10%, test10%, split by household with original anchor-specific seeds. Saved split-row hashes and actual counts are recorded in SPLITS.json. All anchors reuse the same overall cohort, so they are dependent.

| Object | Fitting rows | Labels / features accessed |
|---|---|---|
| p(PCA32,H_A), b(H_A) | first48% of RF households by new hash | residence; internal12% selects model/checkpoint |
| Risk predictors | same48% of RF | SEX and RAC1P training labels |
| Code, action dictionary, C_A/C_AB | teacher60% of RF | frozen probabilities, permitted features, H; soft p targets for action offsets, no actual Y for costs |
| Q cost and all joint tables | remaining40% of RF | actual residence, protected labels, H partitions, code, PWGTP |
| Supervised LEACE/SPLINCE | teacher-fit48% of RF | joint protected labels; SPLINCE residence/income/employment covariance |
| Fresh attackers and task probes | original downstream_fit union attacker_fit | appropriate target only; full permitted recipient view |
| Attack validation | original attacker_validation | target loss; no new model fitting |
| Mechanism/utility validation and selection | original downstream_validation | task labels and declared validation comparisons |
| Current-run evaluation | original test, after selection freeze | all registered endpoints only |

Each hash role is by household, not person. Dictionary fitting may reuse teacher-training and internal-selection households, but mechanism estimation is disjoint from both. Attack/probe fitting pools are also disjoint from RF and validation/evaluation. No cross-fitting teacher is later silently refitted. Validation prediction selection is shared between unweighted and PWGTP reporting. Inference conditions on the fitted models.

Historical preprocessing: per-anchor feature preprocessing and PCA32 were fitted on the entire original representation_fit pool. Thus the current mechanism-estimation40% contributed features to that historical transform. New role splitting does not undo this historical data use. Historical H models and their selections have additional original downstream/source fitting and validation history documented in ARTIFACT_DEFINITIONS; this study treats H as a frozen existing service, not a newly independent transform. Every2018 statement remains development evidence.

The current-run test outcomes and feature NPZ members are not accessed by the fitting loader. CSV reads skip all unrequested raw rows before parsing label columns. Reading a test pool requires an explicit selection-frozen permit. Hashing whole stored artifacts for integrity does not evaluate their held-out members. Private artifacts retain ordered person and household IDs for alignment, bootstrap and replay; only aggregates and hashes are committed publicly.
