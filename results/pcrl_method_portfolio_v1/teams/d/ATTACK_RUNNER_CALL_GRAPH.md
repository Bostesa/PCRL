# Team D fresh-attack execution path

This is the private, S3-queued attack unit for one locked D arm, source seed and
attacker seed. It is development evaluation on reused Adult roles, not a fresh
confirmation. `attack_runner.py` must run only after S1's effective science
seal and S4 role-file admission; no Adult attack has yet run.

1. S1's `CAMPAIGN_LOCK.json` registers `attack_units[attack_run_id]` with the
   exact D arm, source seed, attacker seed, `source_run_id`, U checkpoint SHA
   and `SCREEN` or `FINALIST` stage. It also pins the four D attack source
   files and S2 `attack_eval.py`. `D_RAW_J` remains refused until separate
   source parity and S4 admission are amended into the contract.
2. S3 admits one heavy compound process with one numerical thread and a
   reserved wall, CPU and RSS ceiling. The process checks the local Git seal,
   immutable unit tuple, source checkpoint pin and code hashes before any
   role-label read.
3. The runner verifies the source FIT, CAL and RELEASE receipt chain and loads
   immutable private ATTACK_FIT_NEW and INNER_SELECTION wire packages. It
   checks each package's S2 role/row hash, source checkpoint SHA, calibrator
   SHA, exact wire members and owner-only custody. It does not open raw source
   inputs or the old assessment.
4. It reads ATTACK_FIT_NEW SEX through S2 `SafeRoleStore(...).read(...,
   procedure="attack")`. For each recipient and aligned pair, the D adapter
   builds the complete S2 `AttackView`: categorical one-hot ID and ten bits
   where present, emitted decoded feature, exact calibrated float64
   probabilities, decision, all public metadata literals, plus deterministic
   public-only defense-aware features. Row/unit equality is checked.
5. S2 `fit_slate` fits LR, MLP, HGB and distinct public-only defense-aware MLP
   on ATTACK_TRAIN and orients/chooses the envelope on ATTACK_TUNE. It writes
   one private model, prediction and timed completion receipt per reader. For
   each finite-code view, D additionally fits the fixed α = 0.5, 1, 5 direct
   categorical CellReaders on ATTACK_TRAIN and attaches their frozen TUNE and
   INNER predictions through `add_external_fixed_readers`. That is 12 learned
   plus 9 specialist fits for a code arm, or 12 learned fits for a continuous
   arm. Ignore-one-recipient pair readers are exact aliases, zero new fits.
6. All three fitted slates, TUNE-selected orientations, specialists and pair
   aliases are saved in private frozen custody. `FROZEN_COMPLETE.json` is
   written before INNER_SELECTION SEX is opened for scoring. S2 `score_slate`
   reports each reader and the fixed-slate envelope on INNER. No INNER label
   selects an alpha, architecture, orientation or release.
7. The private `COMPLETE.json` holds source/lock hashes, 21 or 12 actual fit
   units, zero-fit aliases and aggregate AUC. Per-person predictions remain
   private for S4 replay. Public integration uses only reviewed aggregates.

Each compound attack process reserves the full cost of its internal readers.
If it stops midway, completed model/prediction/receipt files remain private,
but the runner refuses reuse of that attack ID. S3 records the partial unit as
incomplete; any retry needs a dated S1-registered new ID and charges every
executed fit again. Saving a model file is checkpoint custody, not a second
training fit. S4 independently replays release views, fit/tune separation,
reader predictions and scores before a D endpoint can be called valid.
