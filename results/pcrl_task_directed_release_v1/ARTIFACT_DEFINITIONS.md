# Artifact definition audit — PCRL task-directed release

Audit date: 2026-09-21. Authoritative source commit: `e3415b94deb8d71c4870d4c392bb8ad4b464a847`. This read-only audit examined the replacement worktree, verified committed source bytes, inspected public metadata, hashed narrowly named artifacts, and loaded only saved split row-index arrays. It did not load raw source data, any year-2016 files, serialized models, or release values; it did not access cloud services or fit anything.

Machine-readable detail, full metadata and every observed/expected artifact hash: `artifact_definition_audit.json`. All 33 examined source/manifest files match the stated commit. Of 54 targeted artifact paths, 33 have a local copy in the declared search roots; all available copies with a committed expected hash match.

## Implementation-critical findings

- H_A contains only income/employment class probability pairs; coverage belongs exclusively to H_B. Local A conditioning must never use coverage probabilities. Existing DAX and spectral code route correctly; no offending local coverage-conditioning implementation was located in the examined predecessor modules.
- Stored A0/J releases are authoritative. Portable_state checks exact checkpoint identity separately from float32 execution agreement <=1e-5. Historical strict loaders require bit-exact recomputation and may fail on a different platform.
- LEACE/SPLINCE-on-A0 erase only the 16-coordinate A0 channel; numerical ranks 6,6,7 do NOT shrink stored width: their stored A/B/AB widths remain 20/2/22.
- Original executed OptNet encoders were not persisted. Encoder persistence was added later, but existing optnet_complete.json causes run() to skip before persistence. Historical release arrays are the actual executed comparison objects.
- Original erasure_baselines.run persists release arrays and metadata, not projection matrices. Do not invent an existing fitted affine-map artifact or re-fit and label it byte-identical.
- The original 2018 household cohort is already development-exposed. Seed variants reuse the same 30000 people / 20147 households and are not independent samples.
- Historical attacker cap4096 exceeds actual attacker pools (~3000), so do not describe them as4096 distinct fitting people.

## Exact ownership and allowed inference inputs

`results/redesign_20260909_acs_fixed_predictions_v1/INPUT_SCHEMA.json` fixes H_A to `[income0,income1,employment0,employment1]`, and H_B to `[coverage0,coverage1]`. Class order is `[0,1]`; both original float64 probability columns survive. Coverage belongs to B. A/public_coverage denotes a forbidden prediction target available to an A attacker; it does not give that attacker the B coverage output.

The correct local conditioning implementation is `experiments/pcrl_direct_adversarial_v1/attackers.py:42 service_view`: all A/* roles receive `ha`; all AB/* roles receive `cat(ha,hb)`. The independent spectral implementation, `experiments/pcrl_nonlinear_rank_v1/maps.py:65 build_roles`, uses `qA(ha[:,[1,3]])` locally and `qAB([ha[:,[1,3]],hb[:,1]])` for coalitions. `experiments/acs_residual_spectral.py:196 SpectralModel.features` also reads only PCA32 and H_A.

**Coverage ownership bug trace:** any local A conditioner containing coverage/H_B would violate those committed definitions. I did not locate that offending operation in the examined predecessor implementations; the concrete existing functions above are correct. The report does not relabel them buggy without evidence.

PCA32 comes from the ten-column covariate allowlist: `AGEP,WKHP,SCHL,MAR,RELP,CIT,DIS,DEAR,DEYE,DREM`. `acs_transfer_data.py:148 CovariatePreprocessor` fits numeric statistics and categorical levels on representation_fit, with separate missing and unseen states. `run_acs_transfer.py:152` fits full-SVD PCA on the same representation-fitting data and saves it in `results/redesign_20260907_acs_transfer_v1/seed_s/release_maps.joblib`. A0/J then use the saved fixed_predictions `pca.npz` and the checkpoint standardizer: float64 subtract/divide, then float32 tensor. No H_B, labels, household ID, row ID or survey weight is a mapper input.

## J and A0: exact executed objects

`experiments/acs_fixed_predictions_training.py:15 Auxiliary` wraps a 32→64→16 ReLU mapper with two 16→1 source heads. Forward parameters total3186. The saved J wire is A=`[H_A,Z_J]`20 dimensions; B=H_B2; AB=`[H_A,Z_J,H_B]`22. Public-derived source-head probabilities are separately stored in `derived/*` and available only where the interface publishes them.

`train_seed` at line125 executes60 source epochs,20 observer warmup epochs, then a common full state/optimizer/RNG fork into80 continuation epochs. Each continuation minibatch makes3 observer updates then1 mapper update; batch256, Adam lr.001, no weight decay. Selection is the fixed final epoch. A0 uses only source loss. J uses `source-.1*individual-.1*coalition`; source is `.25*BCE_income+.25*BCE_employment`. The `.5*coverage` term is fixed and logged but omitted from backward. `acs_coalition_training.py:165 observer_losses` entropy-normalizes each role CE, averages roles within a view, and defines individual as the half-sum of A/B means; coalition is the AB mean. B has zero mapper gradient and shares the exact A0 B observer trajectory/Adam slots.

Checkpoint/release pattern: `results/redesign_20260909_acs_fixed_predictions_v1/seed_s/training/{A0,J}/{final.pt,releases.npz}`. All three seeds currently exist in the main checkout and match RELEASE_MANIFEST.json hashes. Exact hashes are listed below and in JSON.

`experiments/pcrl_competitive_method_v1/run_fit_e.py:53 load_mapper` extracts `branch.mapper.*`, source heads, and input_mean/input_scale. The replacement uses `utility_extension/extension.py:234 portable_state`: checkpoint identity is exact, stored release arrays stay authoritative, and recomputation is checked at max absolute1e-5 because float32 host arithmetic can differ. Older `dax.load_frozen_state` and `frozen_channel` require bit-exact execution and can fail across hosts.

## LEACE-on-A0 and SPLINCE-on-A0

`experiments/pcrl_invariant_baselines_v1/erasure_baselines.py:171 fit_leace`, `:216 fit_splince`, `:297 build_releases` define both baselines. Input is the stored A0 auxiliary channel (wire/A columns4:), not the full A wire. Sensitive fitting targets are concatenated one-hot SEX2+RAC1P9+public_coverage2 on complete representation_fit cases. Application is `mean+(x-mean)@P.T`; H_A/H_B are appended unchanged.

LEACE uses covariance-eigenvalue rank stabilization1e-10, float64 affine concept-erasure, shrinkage=False, constrain_cov_trace=False, svd_tol1e-10, then projection SVD cleanup1e-10. SPLINCE invokes `pcrl/baselines/splince.py::fit_splince` with condition gate1e6, preserves covariance with `[income_binary,civilian_at_work]`, and refuses a silent LEACE fallback: infeasible returns no named SPLINCE release.

All three committed fits are feasible. Numerical projection ranks are6,6,7; storage remains16 coordinates, so A/B/AB dimensions remain20/2/22. Seed2 has no complete-case support for RAC1P zero-based class3, reducing removed centered concept rank from10 to9. The guarantee concerns fitting moments and affine prediction on the transformed channel only. The augmented release retains H_A and can leak nonlinearly. SPLINCE sees two authorized source-task labels that the spectral/OptNet residual-teacher comparators do not; this adaptation and label-access difference are recorded.

The original `run()` writes release arrays and JSON, **not serialized projection/mean maps**. No fitted affine-map artifact should be invented. The current local copies of the three LEACE and three SPLINCE release files are absent in the searched roots. Committed release hashes exist in each `seed_s/erasure_baselines.json`; archive group `invariant` contains the releases (and exec_inv contains LEACE execution inputs).

## Executed OptNet definition and persistence limit

`experiments/pcrl_invariant_baselines_v1/optnet_arl.py` is the controlled Sadeghi/Wang/Boddeti OptNet-ARL adaptation. It is not an Amos–Kolter QP layer. `build_roles` obtains frozen whitened V128 from `SpectralModel.features(PCA32,H_A)`: standardized PCA32 plus96RFF coordinates, nuisance-residualized on local H_A, then saved whitening. Utility target R32 is PCA teacher residualized on qA(H_A).

The executed encoder is float64 V128→64 ReLU→16. It minimizes weighted protected explained-fractions on `[H_role,Z]` minus teacher explained-fraction on Z. Per step, a Gaussian kernel of instance-normalized rows is constructed; column-centered kernel m and centered target u give `p3=solve(m.T@m+n*gamma*I,m.T@u)` and explained fraction `(||m@p3||²+n*gamma||p3||²)/||u||²`. This is the implemented expression; it is not a new mathematical audit of the cited paper.

Every seed ran policies L1=(local1,coalition0), L2=(2,0), C1=(1,1), with local weight divided across3 roles and coalition across2. Actual `optnet_complete.json` records **1200 Adam steps per start,2 starts,batch512,lr3e-4,weight_decay2e-4,gamma1e-4**. The CLI default1500 is not the executed budget. Start selection uses final training objective on the same held-in fixed batch; no downstream or attacker outcome is consulted. Multi-role aggregation, residual continuous teacher, code-derived target-energy normalization, and approximate minibatch kernel players are explicit adaptation assumptions.

**Original encoder persistence is absent.** `run_exploratory_2017.py:54 NOT_TRANSPORTED` states that the executed run did not save its encoders. `optnet_arl.run` later gained a joblib dump, but existing completion markers return early before it. An archived `optnet_encoders.joblib` is not established by the newer source. Historical2018 release arrays are the executed objects; a new fit cannot silently inherit their identity. All9 OptNet release files and all3 named encoder files are absent locally in the examined roots.

## Actual household splits

Source: `results/redesign_20260907_acs_transfer_v1/schema_support.json`. Eligibility is19≤AGEP≤34 and PWGTP>0, then a random whole-household prefix with sample_seed1200000 and cap30000. The resulting cohort has30000 people and20147 households; all3 split seeds partition this same cohort. `acs_transfer_data.py:87 split_households` sorts unique household IDs, permutes with1210000+seed, and rounds cumulative household fractions35/10/15/10/10/10/10percent. Saved split arrays are original raw row offsets.

| Pool | Seed0 persons | Seed1 persons | Seed2 persons | Households per seed |
|---|---:|---:|---:|---:|
| representation_fit | 10513 | 10428 | 10551 | 7051 |
| source_validation | 3004 | 3010 | 2979 | 2015 |
| downstream_fit | 4539 | 4521 | 4464 | 3022 |
| downstream_validation | 2984 | 3041 | 3024 | 2015 |
| attacker_fit | 2993 | 3021 | 2941 | 2015 |
| attacker_validation | 2985 | 2964 | 3062 | 2014 |
| test | 2982 | 3015 | 2979 | 2015 |

I independently loaded only saved row-index arrays: each count and raw-row array hash matches the committed manifest, each seed partitions30000 unique rows, and the sorted cohort hash agrees across seeds. Household counts/ID hashes are copied from the committed manifest; household identities were not reconstructed from raw data. The precise per-pool raw-row and household hashes are in JSON.

DAX internal training folds are different from these7 external pools: SHA256 of `pcrl_direct_adversarial_v1/internal_household_fold/v1|SERIALNO`, first8hex divided by2^32, partitions representation_fit into p0_fit[0,.25),mapper_fit[.25,.8),monitor[.8,1]. Existing seed variants are not independent confirmation samples. Original2018 test households already informed this line of research.

## Inherited roles and attacker slates

Authorized utility roles are A/{income_binary,civilian_at_work,same_residence} and B/{public_coverage,commute_over20}. Forbidden audit roles are A/{public_coverage,commute_over20,SEX,RAC1P}, B/{income_binary,civilian_at_work,same_residence,SEX,RAC1P}, AB/{SEX,RAC1P}. These11 roles are fixed in `acs_fixed_predictions_audits.py:32`; the9 saved-observer roles exclude residence and commute. B is structurally unchanged, so its scores are constant across releases.

Five fresh candidates: logisticC1/lbfgs/maxiter500/tol1e-4; two MLP64,32 restarts with seed offset10000; HistGB150 iterations,15 leaves,learning_rate.1,l2=1,leaf20 or5,no early stopping. MLPs run one360-epoch trajectory with nested120/360 selected checkpoints, batch256,lr.001,no weight_decay,validation every5, min validationCE then earliest epoch. The fitting cap4096 exceeds actual attacker_fit pool sizes2993/3021/2941. Subsets use seed1240000+100*seed+target_index. Selection remains unweighted; PWGTP scores use the same selected predictions.

`acs_spectral_audits.py` adds RFF-kernel ridge candidates with256features, median distance bandwidth from up to512 fitting rows, ridge alphas1e-4/.01/1. It retains original scopes and adds `kernel_` scope names. Standard independent uses wire predictors; expanded independent adds public-derived functions; expanded catchup adds actual saved-observer continuation. Unchanged saved observers are diagnostic-only. Catchup uses exact saved float32 state, identity coordinates/no new normalizer, an Adam reset, and nested120/360 additional attacker-fitting epochs. Existing adapter only admits widths2/4/6/20/22 and [width,64,32,K] architecture.

Erasure/OptNet/spectral baselines have no own saved training adversary; their generic audit reports0 own catchup trajectories and retains accessible historical H-ancestor catchup candidates. J has own saved observers and publicly derived source heads. Those exposures are distinct and should remain explicitly recorded.

`select_pools`, `inherit_singletons`, and `route_ancestor` ensure all accessible H-only predictors and all A/B singleton sensitive predictors remain candidates under AB, not only earlier selected winners. For S1 `[H_A,Z_new]`, J is a comparator, so no J-only predictor belongs in the candidate slate. The stochastic extension `slate.py` assumes S0 `[H_A,Z_J,R]`; its J column routes cannot be reused unchanged.

Historical utility probing uses logisticC1 plus MLP40 on cap2048. Replacement Section6 explicitly changes to all eligible downstream fitting households, logisticC{.03,.3,3}, and MLP200 with wd{0,1e-4}, same opportunity for H/J/candidate/raw views. This does not automatically change the historical privacy slate.

## Compatibility caveats requiring explicit treatment

- Artifact resolver caveat: inputs.Registry.resolve invokes nonlinear_rank.inputs.resolve, not the DAX module resolve wrapper. Thus the DAX extra invariant root is NOT automatically searched through Registry.resolve; direct dax.resolve calls do search it. Declare all roots explicitly in a new loader.
- Replay only from verified checkpoint plus original input mean/scale, keeping stored release arrays authoritative on already-scored2018 pools; do not treat numerical tolerance as permission to silently substitute a different trained model.
- Raw-data helpers remain broad: run_acs_residual_spectral.load_labels calls load_cohort (reads features, source, heldout and audit columns); all_labels materializes reserved labels before some callers discard them. Merely retaining only authorized keys afterwards is not a strict never-read boundary.
- Erasure authorized_task_matrix docstring says reserved labels never read, but calls all_labels over the whole frame and only selects authorized outputs. If the new protocol forbids reading residence/commute at fit time, use a narrow loader rather than this helper.
- Portable ancestor audit comparison is a separate tolerance from mapper replay: utility_extension.portability uses CE1e-7 and ranking1e-3 with exact other fields. acs_spectral_audits historically asserts full metric-dict exact equality.
- The new S1 release does not contain J. J-only predictors cannot be routed into its candidate slate; accessible H-only and same-release singleton predictors can. S0 extension slate.py assumes [H_A,Z_J,R] and must not be reused unchanged for S1.
- Existing historical saved-observer catchup adapter only supports2/4/6/20/22 widths and exact source [width,64,32,K] models. A newly sized task channel cannot claim historical own-observer catchup without a real compatible saved observer.
- Do not infer active service deployment from historical release files; replacement RELEASE_CONTRACT.md distinguishes stipulated S1 evaluation from irrevocable S0 recipient access.

## Artifact hashes and local availability

Search roots were the main PCRL checkout, replacement/invariant/utility/stochastic worktrees, residual-spectral worktree, and new task-directed worktree. Only these named relative paths were checked. A missing path in this inventory is not a proof that no archive copy exists. No cloud restoration was attempted.

| Artifact relative path | Expected SHA256 | Local |
|---|---|---|
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_0/split_rows.npz` | `669fcff80bf61b3d8f17b7cb07d676a2b38030fe638942bfa4bae90f12311f8a` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_0/pca.npz` | `a2015174bb67e1aa03370dda7eef3233ce08f551fbbc2bcb276396cfad36130a` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_0/anchors.npz` | `2a40cdf734631be40d07f0b24de1d7211b10a984c9c23b5c7da8db1c19ebf9fa` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_0/teacher.npz` | `8d4aabd1e68302bba1be472f6bee26673ea414a896993b17d260bfc9b9f22e07` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_0/training/H/final.pt` | `d80949bc78ba41a29e896e876d8507ffff0bafb91460f96ab3349416aefe7287` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_0/training/H/releases.npz` | `d68aebd5ea72b1c0991cb3282816a965c8c4934357cc7c8dc007b9982c8c54a0` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_0/training/A0/final.pt` | `379ca674fac9ea2a7726e660422d1a40337af45b84a887f661b02645c1c41c63` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_0/training/A0/releases.npz` | `db48086dff1f83bc93e670c7aa9b84c09379946d54dbe7888723307ec595dfe5` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_0/training/J/final.pt` | `8a824ca61d2cb5142c364232fe840699ecfff09d8913964d7525d3e25ccb952d` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_0/training/J/releases.npz` | `a905982eb310cc3bdf2dc7392445ef19201bd648eb56fc650874102fd8173626` | present; matched |
| `results/pcrl_invariant_baselines_v1/seed_0/releases/leace_A0/releases.npz` | `25e58a7c950d740961190d9eb79c5384bdadd6d799a0b13b754b26e2953c68b7` | absent |
| `results/pcrl_invariant_baselines_v1/seed_0/releases/splince_A0/releases.npz` | `dca7533cc63dab12984590af0ab6fbf6ea3f997a7a9ded9cac1db13033f87e67` | absent |
| `results/pcrl_invariant_baselines_v1/seed_0/releases/optnet16_L1/releases.npz` | `fd6be514cbc9b50f87ffdcdd2a88b38f071e9d7a54120765dc4719757227785f` | absent |
| `results/pcrl_invariant_baselines_v1/seed_0/releases/optnet16_L2/releases.npz` | `ff81a84952360f4b768416b77018ac394c616e9e12fa61307e87263100f51e13` | absent |
| `results/pcrl_invariant_baselines_v1/seed_0/releases/optnet16_C1/releases.npz` | `e57b1cfce30b239ed5c8aa3913bc016cdf83dfe8af9abc051dd589f3e29e9269` | absent |
| `results/pcrl_invariant_baselines_v1/seed_0/optnet_encoders.joblib` | `no committed file hash located` | absent |
| `results/redesign_20260910_acs_residual_spectral_v1/seed_0/maps.joblib` | `0cc29ac818e611c35b742b9082fec8695742b7e9c80b5a82d4de95ada6df4aec` | absent |
| `results/redesign_20260907_acs_transfer_v1/seed_0/release_maps.joblib` | `no committed file hash located` | present; hash recorded in JSON |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_1/split_rows.npz` | `d8543a6857e68adcbb2345781eba135a850463285a77a76b42b5c3e701233d1e` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_1/pca.npz` | `99f5e4d5a7dc8ad51157519583a62ac5f7fe898c7557fcd743a6a5a020d6382b` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_1/anchors.npz` | `957f78dfd36862fe2b39e1685433d671d44a2b8d3d511b14a52a9636ad037379` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_1/teacher.npz` | `c37047422bc374eb8541cf87636373ded469b204b463d59681d795d707e130c6` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_1/training/H/final.pt` | `6244bd0927eda2236e07ce083884c90613ead21d316790a2c1dd3e298b036779` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_1/training/H/releases.npz` | `700aa64eec0a99935f9e597499257aced5618fe988e40eeebba325322dc54e9d` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_1/training/A0/final.pt` | `76935ed9303575d1c7fda498bc1d143be8d9c6c4e6bab1e6d845c7ed2a969e59` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_1/training/A0/releases.npz` | `b61c4be69b1f99622e0e67384951e5729b605196083ebb05c2698688b365961a` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_1/training/J/final.pt` | `b6f7763b71dfd2a33bc40e10d76d0ab8b3952dd1b10cac74739ebb375e20eeac` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_1/training/J/releases.npz` | `57aae7057e972700abbafe1b16d362f65bf2a227d847ac219fea0f295a3ac15a` | present; matched |
| `results/pcrl_invariant_baselines_v1/seed_1/releases/leace_A0/releases.npz` | `8f533aeb7cb12db0f258eeea1da14334b17b717f1525a3caf0bfe70afa184293` | absent |
| `results/pcrl_invariant_baselines_v1/seed_1/releases/splince_A0/releases.npz` | `050fd01414adc2d36964aaffa9089a960d2642252b3b1c033c1455d081c855b9` | absent |
| `results/pcrl_invariant_baselines_v1/seed_1/releases/optnet16_L1/releases.npz` | `4f9f9fbb251a8a6c7526afb57ca38b19d8126eca003af0a7cce5c097cccf670a` | absent |
| `results/pcrl_invariant_baselines_v1/seed_1/releases/optnet16_L2/releases.npz` | `f29198d7e9ab6b351537320b50d4c0a373d674823efef2fb4429955f84dfb99e` | absent |
| `results/pcrl_invariant_baselines_v1/seed_1/releases/optnet16_C1/releases.npz` | `e95b0a9fc4a5b89587d79dcf8f289eb25407cf8f07689b01cd498bd0d13b800f` | absent |
| `results/pcrl_invariant_baselines_v1/seed_1/optnet_encoders.joblib` | `no committed file hash located` | absent |
| `results/redesign_20260910_acs_residual_spectral_v1/seed_1/maps.joblib` | `2f0207853dd6595b11c5d51875fc3fe856371c7d6f97d99198fd0ee19dafb85b` | absent |
| `results/redesign_20260907_acs_transfer_v1/seed_1/release_maps.joblib` | `no committed file hash located` | present; hash recorded in JSON |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_2/split_rows.npz` | `07d9b48e99a05f30396c8dd46d2a3e7b8bd182ca9e8ef55baa55604530e65038` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_2/pca.npz` | `36f37df30eef4ae68224d1456836e53de40c54fe2be97ac1dfd0c4ae734ba89f` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_2/anchors.npz` | `8319e40110eda68071107bbc648bcb0be1e3255975c5d7222e038a0c23a20768` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_2/teacher.npz` | `3c39c9ad65c5419d79af5c9fa24606f1d0aed7be5fdb61551fa44cee4ff2bb9e` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_2/training/H/final.pt` | `fa119b1abb96f6f2647596e9133bf415f13faacc4f23318e9328c584dd930770` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_2/training/H/releases.npz` | `558d709bfebde75b400e1506073594fbdbe6839d82be5995e9b4ea72aaa7b56b` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_2/training/A0/final.pt` | `131743adb7a7c4df31ab262e592151a0c53b7e5d3713770a53ab488fe95bfac0` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_2/training/A0/releases.npz` | `8ad0d0fb7d576517b9d3ebbeacf09708fa8fa925009e7702f6326219be875d2c` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_2/training/J/final.pt` | `086a07294620687306737ba34afc20e83087caf3aee648aaae146d0a3a5c3390` | present; matched |
| `results/redesign_20260909_acs_fixed_predictions_v1/seed_2/training/J/releases.npz` | `a9afc6712f33ce5e2e78dcaeee99f60c750de98cfc4008436e0706329c1fc417` | present; matched |
| `results/pcrl_invariant_baselines_v1/seed_2/releases/leace_A0/releases.npz` | `49fe62d3781f324fa7f294605cfa90368ca68996457f60cbb8f9a09f5d83d11e` | absent |
| `results/pcrl_invariant_baselines_v1/seed_2/releases/splince_A0/releases.npz` | `0b1678e4829522e33edf55ec4ad150a00e5ea34e7c9ad20437410a0c0a5d0a5e` | absent |
| `results/pcrl_invariant_baselines_v1/seed_2/releases/optnet16_L1/releases.npz` | `5d64a66144f9a8607221863713bafbdcda1c476da385ea3ef9d91c92ab665e2b` | absent |
| `results/pcrl_invariant_baselines_v1/seed_2/releases/optnet16_L2/releases.npz` | `86677f27826b2211b333afa1f563ba1065cf4e7d4a7e46191192bf0c81d46b28` | absent |
| `results/pcrl_invariant_baselines_v1/seed_2/releases/optnet16_C1/releases.npz` | `bbd92e7094804c626292da3e2f56969114a434bd7d6404899d0deb4855da9bd0` | absent |
| `results/pcrl_invariant_baselines_v1/seed_2/optnet_encoders.joblib` | `no committed file hash located` | absent |
| `results/redesign_20260910_acs_residual_spectral_v1/seed_2/maps.joblib` | `7c2da63b81eefa5cc159b254f2284ca000064d8544bb9c788272c94c923c9fa9` | absent |
| `results/redesign_20260907_acs_transfer_v1/seed_2/release_maps.joblib` | `no committed file hash located` | present; hash recorded in JSON |

`ARCHIVE_MANIFEST.md` and `RESTORE.md` report exact-version archives; group invariant contains baseline releases/audits, exec_inv has LEACE inputs, exec_main has the fixed historical execution inputs. The public stochastic restore record reports exec_main2545files,0bad; this audit independently confirmed currently available required-file hashes. Original OptNet non-persistence is a distinct limitation from missing local archived releases.
