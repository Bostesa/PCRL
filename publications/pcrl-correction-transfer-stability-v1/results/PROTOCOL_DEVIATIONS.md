# Protocol deviations, amendments and disclosures (append-only; times UTC, 2026-10-18 unless stated)

1. **Out-of-queue label-free preflight before the queue opened (18:56).** The representative source-output check
   (`cts.preflight outputs`, label-free) ran directly with `python3` and default BLAS threads (about 24 CPU-s user + system) before
   `cts.jobqueue` was initialized at 19:01. It read no label. Charged at 19:01 together with the other pre-queue label-free work
   (60 CPU-s upper estimate, ledger label COORD_PREREG_LABEL_FREE_V1).
2. **Routing recipe corrected before any fit (19:07).** The refit investigator found that the first version of the REFIT routing
   reduced the replication's CatBoost seed modulo 2^31 and used a one-task class-support repair; both were corrected to the exact
   Stage A recipe before the registration commit. The routing reproduction (`CTS_ROUTING_REPRO_S0_V1`) then refitted the admitted
   seed-0 occupation teacher exactly.
3. **Code changed while a unit waited for a slot (19:19).** `CTS_D0_DIR_REFIT_LOC_V1` registered at 19:17 and launched at 19:19:15
   while new, not-yet-imported modules (`cts/deploy.py`, `cts/freeze.py`) were being written; the queue recorded the digest change
   and `cts/freeze.py` as uncommitted at launch. The code the unit imports (cts.trackf, cts.common, ohn.protect, acs_replication)
   was committed and unchanged. No other unit launched with uncommitted imported code (checked from the ledger).
4. **Access amendment (19:20, before first use).** `ACCESS_AMENDMENTS.json` authorizes the entry point `cts.questiona` for the
   operations fit, select, check_diag and check_repair (question A per-person increments); committed before its first use.
5. **Sensitivity witness preprocessing (19:29, before its fit).** PROTOCOL section 6 states that AGEP / WKHP enter through the
   admitted all-MODEL_FIT z-map, to which the TREE learners are invariant. The sensitivity witness (B2_SINGLE recipe, partition alias
   D0B) also contains the additive logistic learner ADD_V2, which is not invariant to that map; its AGEP / WKHP are therefore
   re-standardized with FIT moments, and its spline / frequent-WKHP featurizer is fitted on FIT inputs only (the recipe's absolute
   200-person WKHP threshold kept, giving 6 indicators on D0's FIT instead of the source's 13 on all MODEL_FIT). The dictionary,
   z-map and featurizer are fitted once on all FIT and shared by its fold pipelines (as the source did on MODEL_FIT; label-free).
   Recorded in its WITNESS.json 'adaptations'. Decision by the coordinator before the fit.
6. **Engineering patches applied before any correction outcome (19:46-19:47).** The engineering review (no blocker; 5 major, 15 minor)
   supplied 11 patches, applied and committed (4a17c64a3) before any ladder, selection or realized outcome of a correction was
   computed. One changes a registered computation: for the ensemble deployment, the recipe gates G2 / G3 and the tie-break reserve
   ratios now use the ensemble's increment against the deployed witness W (whose MODEL_SELECT margins define the reserve) instead of
   its increment against the uncorrected ensemble (ENG-01; the transfer diagnostics keep the latter). Not patched, disclosed: the
   frozen ens rules budget e_s - e_0 against W's reserve, and ens 'identity' is the uncorrected ensemble e_0, not W.
7. **Code committed while units waited for a slot (19:20-19:47).** Several units registered before and launched after commits of new
   modules or patches (the queue records the digest at registration and at launch; `digest_changed_while_waiting`). At every launch
   the imported campaign code was committed and clean except `CTS_D0_DIR_REFIT_LOC_V1` (item 3); the changes concerned modules those
   units do not import, the default-preserving Track R income option of cts.trackf (19:20) and the reviewed patches (19:46).
8. **Amendment timestamps (verifier ATTENTION, recorded 20:09).** The four entries of ACCESS_AMENDMENTS.json carry
   "utc": "2026-10-18T19:20:00+00:00", but the file was first committed at 19:14:44Z (commit 019241108): the coordinator wrote an
   estimated time instead of the actual one. The commit time is the record; the entries were committed before the first use of
   cts.questiona, so the before-first-use rule holds. The file is not edited.
9. **Adaptive training-size diagnostic (amendment, 20:16, after written statistical review, before its fits).** Motivated by the
   observed D / R witness margins (NO_RECIPE): the A1_PO_16 recipe refitted on the whole non-panel pool (partition P70, 14,022 people,
   70% of MODEL_FIT; panels still excluded; no GUARD / CHECK role) and on the same rows and folds with the admitted dictionary (P70A),
   each evaluated on MODEL_SELECT against both seeds' references, with paired differences between fits and the dictionary collapse
   at each size. Predictions and the reading rule are registered in `diagnosis/TRAINING_SIZE_BET.json` before the fits; the review is
   `roles/statistical_reviewer/REVIEW_TRAINING_SIZE_DIAGNOSTIC.md`. Labelled ADAPTIVE; kept out of every selection; it cannot change
   NO_RECIPE. Two fitted release specifications.
