# Application-screen regression notes

Seven targeted tests passed in 1.05 seconds. The command and actual final output
are recorded in `regression_tests.log`; writing that log did not rerun them.

The first invocation returned six passes and one failure in 1.07 seconds. The
failure was in the tiny integration fixture: its mock of
`subprocess.check_output` returned a string for every call, including Python's
`platform.platform()` byte-output query. That caused an `AttributeError` while
decoding the fixture's hardware metadata. The test fixture was corrected to
mock the platform string explicitly. No application-screen source or experiment
configuration changed, and no experimental run was invalidated or superseded.

The tests use fabricated arrays and temporary files. They cover participant and
row isolation, task-fit-only scaling, reconstruction of saved model probabilities,
the binary active/sedentary mapping with genuine six-class activity as the attack
target, fitting attacker probabilities only on attacker rows, unsupported-class
failure, and the boundary preventing official-test content access. The loader
test mocks the historical oracle-value checks because their reference numbers
do not describe the fabricated fixture; the actual loader and learned-reference
paths execute on tiny local files.

A separate read-only artifact check confirmed the recorded source and model
checkpoint SHA256 values, the 4,697/1,361/1,294-row partition of all 7,352 HAR
training windows, disjoint participant groups of 13/4/4, binary model classes,
and finite archived probabilities. It did not retrain a model or read official
test content. The screen remains a development-only policy diagnostic, not a
representation-method or final-test evaluation.
