# S4 numerical verifier admission draft

Status: design only. No S4 Adult role value has been read and no Adult verifier
unit has been queued. The existing S4 identity and V2 control synthetic RED
tests are registered separately from this prospective admission.

## Exact historical inputs

The verifier lock must list both committed B fit-lock SHA-256 values, the
original B V2 control-lock SHA-256 and the amended R2 control-lock SHA-256.
For every one of the 16 B arm/seed screens and two feature-anchor seeds, an
immutable unit entry binds:

- exact parent S3 unit ID and queue receipt SHA;
- exact parent fit-lock SHA, lock commit, archived S4 review and local seal;
- exact `ATTEMPT_INDEX.json` SHA or `ANCHOR_RECEIPT.json` SHA;
- exact V2 probe S3 unit ID, queue receipt SHA, V2 control-lock SHA, private
  V2 index/receipt SHA, and each of three view report/custody receipt SHAs;
- one fresh private S4 output path, its expected S3 unit ID, source script SHA,
  seed, arm, CPU/wall/RSS reservation and `data_access` scope.

The five historical B V1 probe results and the failed original anchor V2 s0
unit remain diagnostic entries only. No verifier may treat their PASS/FAIL
statuses as the uniform V2 gate. The reported V2 PASS status is a claim to
recompute from its saved models, labels, views and vectors.

## Role allowlist

B replay may read admitted ATTACK_FIT_NEW features and SEX, INNER_SELECTION
features, SEX, income, occupation, and the seed-specific U0/Ucal references.
The feature anchor also needs ATTACK_FIT_NEW references. The role files must
match `DATA_ACCESS_MANIFEST.json` by full file SHA, exact NPZ members and
ordered row/unit pair hash before their values are used. The old assessment,
HEAD_VALIDATION, all-role archive and FIT/CAL label values are out of scope.

The anchor alone may read the pinned source U checkpoint and public heads from
the admitted hcal cache. B replay may read only its frozen fit checkpoint,
sampler secret, public head/calibration records, saved ATT/INNER releases,
reader models, vectors and endpoint package. V2 replay may read its three
probe models and NPZ custody package per view. Hash every private artifact
against its parent receipt before loading it. Public reports contain only
aggregate metrics, residual maxima, hashes, status and limitations.

## Queue sequence

1. Run the S4 V3 identity missing-module RED and subsequent GREEN, then the
   independent V2 control custody RED/GREEN. These are `data_access=NONE`.
2. S1 commits a separate `S4_VERIFIER_ADMISSION.json` with the 18 exact unit
   entries and the reviewed S4 source/test SHA map. S4 writes a scoped prefit
   review; S1 commits a local verifier seal. Neither changes a historical
   fit or V2 control lock.
3. S3 adds a distinct `s4_adult_replay` mode. It checks the verifier lock,
   review, seal, exact script at `argv[1]`, unit/arm/seed, parent receipts,
   source SHA, role scope, private output and resource limits at enqueue and
   launch. The child independently checks its live S3 process group before
   any person-level file read.
4. Smoke one B and one anchor replay under the same global semaphore. Reserve
   remaining units from measured CPU/RSS, within the campaign ceiling. Every
   unit writes one immutable private report or a preserved failure receipt.
5. After 18 successful replays, a separately budgeted aggregate unit may
   recompute the strongest-control selection and simultaneous paired-group
   intervals from independently checked saved arrays. Until then, numerical
   endpoints and anchor local AUC caps remain pending.

S4 code must not import Team B, Team A, S2 attack/evaluator, endpoint or
selection implementations. It may use NumPy, Torch and trusted serialized
sklearn models after their exact file hashes pass. Any fit control failure or
replay mismatch blocks the affected result without changing the frozen
thresholds or hiding the saved attempt.
