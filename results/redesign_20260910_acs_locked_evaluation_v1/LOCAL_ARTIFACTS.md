# Omitted local evidence

[LOCAL_ARTIFACTS.json](LOCAL_ARTIFACTS.json) binds the new identifier/exclusion/sample manifests, full first inventories/lock and logs by path, byte count and SHA256. Required historical objects remain at the paths in [FROZEN_OBJECTS.json](FROZEN_OBJECTS.json), with original reuse provenance. No raw record, household identifier, fitted model/preprocessor, cache or person-level prediction is published.

The empty sample manifests can be reproduced exactly by the independent exposure verifier. Full model replay requires the original omitted binaries; a new fit or approximate replacement is not recovery. Publication-receipt bytes are created after the final commit and retained locally.
