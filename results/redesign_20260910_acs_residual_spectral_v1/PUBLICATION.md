# Residual spectral publication package

This is an evidence-packaging snapshot, not an assertion of completed scientific evaluation. Publish only paths listed in `PUBLICATION_FILES.txt`, together with separately reviewed source/tests/docs. No raw `data/` path, fitted object, prediction array, row identifier, fold assignment, or bandwidth subset is included. Originals remain untouched and local.

`EVIDENCE_MANIFEST.json` lists every public content hash/size and each exact or sanitized export. `LOCAL_ARTIFACTS.json` inventories every omitted original by path, hash, and bytes. No checkpoint download or exact saved-model replay is provided by this public package; refitting is a new reproduction.

`NUMERICAL_SUMMARY.json` projects numerical and held-out-moment diagnostics into compact aggregate evidence. Fitted vectors/matrices and row-level arrays are replaced by hashes/shapes. Eigenvalues, class support and class moment norms remain. Sanitized audit selections retain validation curves and scoring evidence; their original fitted preprocessing vectors remain local. An export marked `lossless_original_bytes: true` decompresses to exact original bytes; existing compressed CSV inputs preserve exact decompressed CSV bytes and receive a canonical gzip header; other exports are explicitly documented projections.

All gzip archives use an empty filename header and mtime zero. Repeated runs on unchanged input produce identical package bytes. The package fails if a source file changes while read or exceeds the public size cap. Rerun after final evaluation/report logs close to inventory the completed evidence. Neither historical files nor git staging are modified.
