# Additive historical raw-joint reference — scope proposal

The source inventory confirms that the historical best development reference exists for paired seeds0/1: `RAW-J|b0.3`, epoch40, historically named C*. Both serialized encoder checkpoints and actual deployed scaler/logistic head files match the committed OSF manifest and each unit's COMPLETE metadata. The complete byte pins are in `HISTORICAL_RAW_JOINT_REFERENCE_AVAILABILITY.json`; concrete locations are private.

The source architecture is separate 83→64→64→16 encoders with ReLU after the first two affine layers and an identity representation map. The actual deployed outputs come from each 16-dimensional feature through its own StandardScaler/LogisticRegression pipeline. Binary centered logits are (−d/2,+d/2); multiclass logits are the centered classifier decision rows. Probabilities use that classifier's actual `predict_proba`, and decisions use first-class argmax. The numerical restoration worker verifies actual estimator types, classes, scaler metadata, encoder features, probabilities, logits and decisions before accepting the checkpoint.

The historical permitted-column order has SHA256 `7101f2fb37b623a08221c32b33eedf132cf0c4c376bb4042ab9cf16da08ad975`, identical to the current useful-service schema. Both sources use the same five numeric predictors standardized on full OSF_DEFENSE_FIT and the same78 fixed one-hot columns. The historical manifest therefore reuses current role-only exports and the source bank's exact feature-name/schema metadata. It does not use that bank's score codes, probabilities or routing. Source class order is income[0,1], occupation[0,1,2,3,4,5]. New restoration updates are zero; historical training was40 epochs at beta0.3.

Register `H_RAW_CSTAR` as an additional descriptive reference outside the prelocked primary pool. Keep primary K51 and secondary K33 intact. Disclose that current FIT outcomes were already known when this available historical source was admitted. The additional reference cannot alter candidate strengths, checkpoints, fallbacks or reader recipes. The ordinary freshly trained RAW_JOINT remains the required matched adversarial control.

Before any INNER task or SEX result opens, freeze this fixed two-seed source, its zero-update restored checkpoints and its fresh reader banks together with the existing releases. Report current original point utility and full-wire AUC descriptively; report output-only AUC separately if produced by the shared reader bank. Include it in independent restoration and reader-output replay. Do not substitute historical assessment numbers for current outcomes or make a matched optimization attribution from its unequal historical training/head exposure.

The historical `osf.deploy.release` calls a whole-unit verifier that would checksum its mixed-role `release.npz`. The new queued restoration worker authenticates only the explicit safe model/head/record/COMPLETE files and permitted feature-only role exports. It does not call that verifier, load historical person arrays or run new fitting. Its common checkpoint format is directly readable by the shared utility and attack workers.

Proposed root-owned scope-lock fields required by the restoration worker are `status: FROZEN`, `arm_id: H_RAW_CSTAR`, `K_primary: 51`, `primary_pool_changed: false`. Its queued spec supplies `unit_id`, `arm_id`, `seed`, `scope_lock` and SHA, original `science_lock` and SHA, private historical `manifest` and SHA, `private_output`, and `public_output`. Run:

```sh
python -m task_conditioned_representation.historical_reference_worker --spec "$TCR_HISTORICAL_RESTORE_SPEC"
```

At this proposal, source availability and safe file byte identity are verified. Current numerical restoration and outcomes remain pending queued execution. The coordinator owns the dated amendment and final locks; the original protocol and useful-source manifest retain their contents.
