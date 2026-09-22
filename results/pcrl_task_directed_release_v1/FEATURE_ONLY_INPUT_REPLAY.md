# Feature-only frozen input replay

Checked 128 fixed representation-fit rows per anchor. All three anchors reproduce both saved PCA32 and checkpoint-standardized float32 inputs exactly (maximum absolute error 0.0). H_A bytes and dtype are unchanged. Private export/reload also reproduces identical outputs.

The check read only AGEP, WKHP, SCHL, MAR, RELP, CIT, DIS, DEAR, DEYE and DREM from the owned 2018 CSV, the representation_fit raw-row-index member, representation_fit PCA member, and representation_fit/A service member. It read no task/protected columns or test members and fitted nothing. IDs and row values are absent from this report.

The frozen chain is historical CovariatePreprocessor.transform, saved PCA.transform, then (PCA.astype(float64) - J.input_mean) / J.input_scale, cast to float32. Original preprocessing/PCA used the entire inherited representation-fit pool; new teacher/mechanism splitting does not change that historical feature use.

| Anchor | Rows | PCA max error | Standardized max error | H_A exact | Export exact |
|---|---:|---:|---:|---|---|
| 0 | 128 | 0.0 | 0.0 | True | True |
| 1 | 128 | 0.0 | 0.0 | True | True |
| 2 | 128 | 0.0 | 0.0 | True | True |

All source artifact hashes, checked-row feature hashes and deployment-manifest hashes are in [FEATURE_ONLY_INPUT_REPLAY.json](FEATURE_ONLY_INPUT_REPLAY.json). The declared absolute numerical acceptance tolerance was 1e-5; all measured errors were zero.

Standalone private bundles are `private/deployment_inputs/anchor_{0,1,2}/`, each containing `preprocessing.json`, `pca.joblib`, `standardizer.npz` and `MANIFEST.json`. Preserve them in the private archive with the public inference/preprocessing source and environment record. They contain frozen parameters, not raw person rows. Existing bundles are never overwritten.

Runtime API: `FrozenInputMap.load_export(directory).transform(raw_covariate_dataframe, h_a)` accepts exactly the ten named raw columns plus four finite H_A coordinates. Extra label, protected, ID, survey-weight or B columns are rejected. `emit_token(h_a, probabilities, rng=...)` emits only an integer token and a byte-identical read-only H_A copy. Persistent wire reuse remains caller-owned.

Environment used for this replay: `{"joblib": "1.5.3", "numpy": "2.4.2", "pandas": "3.0.1", "python": "3.13.7", "scikit_learn": "1.8.0"}`.

Historical preprocessing source SHA-256: `1cb0290cabd83e4efba1fd990c31721f76ff9a847669951942b6aef14d6fa0a5`.
Inference source SHA-256: `551d994d29fd62cdabb89b1a7b762fd0cecb78101a5263b0f45b7f4c8f57a992`.
Aggregate JSON SHA-256: `f080a868336a53e8eff5d5c3c9ca02890f458e6e72b0189115925a79d66bc50d`.

Synthetic verification: 15 inference tests passed, including no-fit guards, forbidden-column rejection, historical missing/unseen handling, exact H preservation, portable roundtrip, artifact hash rejection and token-only emission. No model was refitted.
