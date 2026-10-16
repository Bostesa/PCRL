"""Frozen configuration of the prepared MMP pilot (PREPARED, NOT RUN in this campaign).

Everything a future run needs is fixed here before any real-data science: arms, access, gates, readers, endpoints,
inference, controls, units, runtime, nominee and stopping rules. A future evaluation on the repeatedly used
Florida 2019 DEVELOPMENT people is exploratory; a new lock does not erase their selection history.
"""
PILOT = {
    'schema': 'MD1-mmp-pilot-config-v1',
    'status': 'PREPARED_NOT_RUN',
    'exploratory_label': ('EXPLORATORY DEVELOPMENT PILOT on repeatedly used Florida ACS 2019 DEVELOPMENT people; not confirmation; '
                          'SEALED_ASSESSMENT stays closed; nominal error rates only'),
    'population_and_contract': {
        'source_publication': 'f4ea199 (ACS protected-tree replication); frozen comparators admitted in 3a1f509 (frozen hybrids)',
        'cohort_fields_tasks_roles': 'unchanged from the source SCIENCE_LOCK (six permitted fields; income > $50k; five occupation groups; '
                                     'household roles MODEL_FIT/CALIBRATION/MODEL_SELECT/ATTACK_FIT/ATTACK_VALIDATION/DEV_REPORT)',
        'release_contract': 'primary: each recipient receives its own probability vector and argmax decision, deterministic, from the six fields',
        'utility_caps': 'accuracy loss <= 0.010, NLL excess <= 0.010, summed Brier excess <= 0.005 vs U0 and Ucal, both tasks, both seeds (24 clauses)',
    },
    'arms': {
        'MMP': 'candidate: SEX-aware CatBoost teachers (T_TREE registry: depth 4, lr 0.05, <= 800 trees, l2 3, 15% household internal '
               'early stopping, refit) fitted on MODEL_FIT with SEX as an extra input; release m = (1 - pibar) q^0 + pibar q^1, '
               'pibar = MODEL_FIT female share; evaluated at SEX = 0 and 1, no person SEX read',
        'MMP_TWIN': 'task-only twin from the same components: (1 - pihat(x)) q^0 + pihat(x) q^1, pihat a 5-fold household cross-fitted '
                    'CatBoost SEX classifier on X (MODEL_FIT); isolates the marginalisation step',
        'NEURAL_RAW_rho1.0_E2': 'existing eligible neural benchmark (frozen outputs)',
        'TASK_REFIT_E20': 'existing tree-family task-only control (frozen outputs)',
        'T_TREE': 'existing SEX-blind teacher (frozen outputs)',
        'LOCAL_TREE_E20': 'strongest existing local (non-joint) protected twin (frozen outputs)',
        'U0, Ucal': 'utility references (frozen outputs)',
        'published_baselines': 'FAGTB and official FARE had no MODEL_SELECT-feasible configuration in this population (source study); they '
                               'are reported as ineligible, not re-run. No published attribute-inference defense implements MMP '
                               '(marginalisation over a sensitive attribute is known from anti-discrimination profiling); explicit gap.',
        'joint_or_sequential_arms': 'not applicable: MMP makes no joint-component claim',
    },
    'equal_exposure': ('every new arm uses MODEL_FIT features, task labels and SEX only for fitting, the same household internal split '
                       'and the same CatBoost registry; CALIBRATION is unused (Ucal only); selection and gates use MODEL_SELECT only; '
                       'readers use ATTACK_FIT/ATTACK_VALIDATION only; the transmitted contract is identical for every arm'),
    'stage0_gates_model_fit_and_model_select_only': {
        'G0_sex_gain': 'held-out internal-validation NLL gain from adding SEX to the teacher must be > 0 for at least one task on both seeds '
                       '(otherwise Delta is ~0 and MMP is ~ the SEX-blind teacher): STOP',
        'G1_utility_F3': 'MMP passes all 24 MODEL_SELECT point clauses (frozen selection rule step 1): else STOP',
        'G2_premise_F1': 'frozen OUT_LR coalition proxy (trained on MODEL_FIT pair wire with MODEL_FIT SEX, AUC on MODEL_SELECT): '
                         'MMP_TWIN minus MMP >= 0.02 on both seeds: else STOP',
        'G3_nonlinear_premise': 'the same with an OUT_HGB proxy: MMP_TWIN minus MMP >= 0.01 on both seeds: else STOP (symmetric nonlinear gate)',
        'diagnostics_reported_not_gating': ['label-free merge diagnostic: distinct encoded inputs vs distinct MMP release pairs on MODEL_SELECT',
                                            'nonlinear-minus-linear proxy gap for every arm',
                                            'out-of-fold linear decomposition of pair vs occupation-only signal for MMP and MMP_TWIN'],
    },
    'stage1_readers_attack_roles_only': {
        'slate': '7 readers (OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB) x 3 views (income, occupation, pair)',
        'pair_bank': '21 candidates per arm: 7 pair readers + 7 income-only + 7 occupation-only (ignore-one) aliases',
        'selection': 'fit on ATTACK_FIT with SEX; orientation and winner by ATTACK_VALIDATION AUC; DEV_REPORT scored from releases only',
        'rng': 'reader seed MODEL_SEED_BASE + s (task_conditioned_representation.attacks)',
        'learning_curve': 'pair slate refit on nested 25/50/100% household subsets of ATTACK_FIT for MMP, MMP_TWIN and NEURAL_RAW',
        'controls': 'MMP wire: shuffled-SEX null (OUT_HGB; acceptance AUC in [0.45, 0.55]) and planted SEX bit (OUT_HGB; acceptance >= 0.99)',
    },
    'stage2_report_after_committed_pilot_reporting_lock': {
        'primary_endpoints': [
            'COMPETITIVE coalition benefit vs NEURAL_RAW, both seeds (lower bound > 0; nested material margin 0.020)',
            'RECIPE coalition benefit vs MMP_TWIN, both seeds (lower bound > 0.020)',
            'recipient worsening vs MMP_TWIN, income and occupation, both seeds (upper bound <= 0.010)',
            '24 utility clauses on DEV_REPORT (point gate + confidence field)',
            'subgroup guards: occupation summed-Brier excess vs U0 within each SEX group, both seeds (<= 0.005)',
        ],
        'K_primary': 36,
        'secondary_descriptive': ['single-recipient AUCs for every arm', 'pair minus best single', 'learning-curve slopes', 'merge diagnostic'],
        'inference': 'paired household bootstrap B = 2048 (new pilot seed), Wald t(G-1); clause bounds one-sided alpha/3; family K = 36',
        'decisions_kept_separate': ['point feasibility', 'utility confidence', 'recovery reduction vs twin', 'competitive vs NEURAL_RAW',
                                    'subgroup', 'novelty (prior-work only)'],
    },
    'nominee_rule': 'MMP is the single registered nominee (no grid, no strength, no checkpoint); no rescue, retune or new nominee',
    'stopping_rule': 'any Stage-0 gate failure stops the pilot before any reader is fitted and is reported as NOT_SUPPORTED_AT_GATE',
    'units': {
        'stage0': '2 seeds x (2 SEX-aware teachers + 5 propensity folds) = 14 CatBoost fits; 4 tasks x arms clause evaluations; 4 proxy fits per arm',
        'stage1': '2 arms x 2 seeds x 21 reader fits = 84 new fits (+84 refits of frozen comparators for equal treatment) + 3 x 3 learning-curve subsets x 7 pair readers x 2 seeds = 126 + 4 control fits',
        'stage2': 'one bootstrap evaluation (B = 2048)',
    },
    'runtime_estimate_cpu_h': {'stage0': 0.5, 'stage1': 1.0, 'stage2': 0.1, 'total_with_margin': 2.5,
                               'basis': 'source T_TREE fits (minutes each, single thread) and the frozen-hybrids reader run (250 fits, 0.3 CPU-h total campaign)'},
    'dependencies_not_in_this_campaign': [
        'admitted Florida 2019 role exports and extraction receipt (source private tree; restored and hash-checked by acs_hybrids.restore_admit)',
        'frozen comparator outputs: frozen-hybrids FROZEN_BANK_S{0,1} and U0/Ucal outputs (private)',
        'CatBoost 1.2.8 (source private pylib; this campaign installed the same version into .private/mechanism_design/pylib for rehearsal)',
        'a committed PILOT_AUTHORIZATION.json naming this protocol hash and an explicit user authorisation (never created here)',
    ],
}
