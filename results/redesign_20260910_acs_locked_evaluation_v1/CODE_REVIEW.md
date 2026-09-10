# Independent code review — locked ACS evaluation

Scope: `experiments/acs_locked_evaluation.py`, `scripts/run_acs_locked_evaluation.py`, `scripts/inventory_acs_locked_objects.py`, focused tests, object replay, and the provenance-blocker wording. Reviewed against the supplied zero-unused-household branch of the requested evaluation. This review follows the requesting-code-review skill as the independently dispatched reviewer; no recursive reviewer was needed.

## Scientific and provenance assessment

The blocker conclusion is supported. Every authorized target person is already in earlier v2 fitting/validation/test exposure, established by exact raw-to-cache reconstruction plus preserved executed-run evidence. `DATA_INDEPENDENCE.md` correctly separates this broad research-use exclusion from the retained transfer preprocessor/PCA, which fitted only their historical representation pools. It does not misdescribe public-file loading as model fitting, claim fresh class support, or call the blocker a scientific null/confirmation. All fresh phases guard on independence before obtaining new labels. No new sample or performance is consumed.

Canonical source namespace, string identifier handling, whole-household exclusion including older members, and deterministic complete-household prefix selection are correct. The expanded H projection is performed before the saved candidate standardizer. The unchanged original metric clipping/normalization is reused. Selected output IDs are fixed across both weightings. Signed additional gain is not clamped. The six bootstrap contrasts have the correct signs; household numerators and task-specific PWGTP/count denominators share multiplicities across conditions and three seeds. Aggregation is the mean of seed-specific metrics, not a probability ensemble. Undefined replicates are retained; zero-variance/undefined contrasts cannot silently become a six-family pass.

## Important findings communicated to root

1. **Raw data identity was not rechecked by the inference runner.** `verify()` originally checked lock records/model/local artifact bytes but not the inventory's original-file hashes; modified covariate values with unchanged IDs could have reached a nonempty run. Root has added a loop over `DATA_USE_INVENTORY.json.source_sha256`, covering the original raw CSV and all recorded source inputs. This is resolved in the currently observed source.

2. **Missing fitted files could escape the object inventory.** `candidate()` originally called bind only for files already present, so missing model/preprocessing could leave `required_missing=0`. Root has changed it to infer required filenames from metadata family and call bind even when absent; compatible inventory replay also checks those expected files exist. Resolved in currently observed source.

3. **Historical hash coverage omitted extended attacker manifests and accepted unverified metadata bases.** The original glob missed `results/redesign_20260908_acs_preservation_v1/extended/seed_{0,1,2}/local_artifacts.json`, although those manifests contain exact model/preprocessing hashes for the optional exposed controls. Also files without historical hashes were labeled “reviewed committed metadata” without checking reviewed Git bytes. Root was asked to include extended manifests and verify committed metadata against the reviewed commit, or explicitly identify non-historical/unverified context dependencies. This does not indicate that current core selected models are wrong, but the inventory must not overstate provenance.

4. **Scientific output completion hashes were not checked before score/uncertainty.** The original `score()` merely asserted `INFERENCE_COMPLETION.complete`, then read arrays without checking per-unit files/LOCK identities. Root was asked to validate all 18 unique system completion records and the hashes of predictions/release/metadata files before downstream phases. Otherwise accidental altered arrays could be scored despite the verified-unit recovery requirement. The zero-cohort guard prevents this in the current study.

5. **Bootstrap resume accepted arbitrary saved draws.** The original `uncertainty()` read existing draws without validating `(2000,n_households)`, values, or equality to the prescribed deterministic RNG stream. Root was asked to regenerate expected deterministic draws and compare exactly, or bind and verify a complete draw record. This is latent in the nonempty path only.

6. **Explicit in-memory state guarding was partial.** The original runner hashes candidate objects and preprocessor/PCA, but not the temporary native anchor models or mapper checkpoint/teacher around inference. The actual functional graph is nonmutating and independent historical replay confirms unchanged states. Root was asked to extend explicit before/after checks to cover every promised state object in the generic runner.

## Checks performed

- `pytest tests/test_acs_locked_evaluation.py tests/test_acs_locked_independence.py -q`: **17 passed** during review.
- AST inspection of evaluator and runner: zero calls named fit, fit_transform, partial_fit, backward, step, train, fit_candidates, fit_prior, or argmin.
- The **new** `inference_graph` was applied to the full historical representation-fit pool for every one of the 18 systems. All **90 wire/derived arrays were bitwise equal** to the parent's saved arrays. Mapper tensors and teacher fingerprints were unchanged. No fresh row was scored.
- A preliminary five-row-slice comparison differed from full-batch saved mapper outputs by at most **1.9073486328125e-06**. Replaying the identical full historical batch removed every difference. This is ordinary float32 GEMM batch-shape sensitivity, not a state mismatch or a changed fitted system. Exact model-output claims must specify identical input/batch shape; unrelated batch shapes should not be required to reproduce bitwise values. No scientific threshold or metric tolerance was changed.

## Review boundary

The runner is still being extended for controls/comparisons by root. This note records reviewed behavior and communicated findings, not approval of an unseen final diff. The current study's only valid scientific outcome remains blocked; no fresh superiority, support, interval, matching or tradeoff result is available. The publication should state this explicitly and list all such pending units as unavailable. After the above provenance/integrity fixes and final regression checks, the blocked-outcome package can be published without claiming independent scientific success.

## Follow-up resolution review

Root's subsequent source changes address all six findings: full data/source SHA replay; required candidate filenames by family; extended manifests and reviewed Git-blob hash checks; 18-unit inference completion/hash verification before score and uncertainty; exact prescribed bootstrap-draw comparison; and explicit anchor, mapper-checkpoint and teacher state checks. I read these changes after the initial note. A separate temporary-file exercise confirmed bootstrap resume accepts exact existing draws and rejects a single altered draw.

The existing `FROZEN_OBJECTS.json` must also be enriched/regenerated with the additional historical provenance; the inventory's compatible-resume fast path returns the existing object rather than adding new hashes. Root has been explicitly told to preserve earlier inventory evidence while producing the final corrected inventory before lock. No fresh scientific outcome has been viewed, so these are pre-outcome implementation/provenance corrections.

Subject to the final frozen artifact reflecting those fixes and root's final checks on the newly added comparison code, no unresolved important defect was found in the reviewed blocker path, identifier logic, release graph, signed-gain arithmetic or bootstrap arithmetic.
