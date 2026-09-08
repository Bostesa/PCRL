# Executed validation

No historical research program was rerun. No ACS model run was invalidated or
superseded. Before the freeze, independent source review aligned tree L2 and
cross-entropy probability normalization with the written protocol. These were
pre-run implementation corrections, not changes made against final outcomes.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_acs_transfer_data.py tests/test_acs_transfer_models.py tests/test_acs_transfer_heads.py tests/test_acs_transfer_runner.py
```

Final focused run: **30 passed in 2.26 seconds** (3.36s command wall included
regenerating the final report immediately before pytest). Tests cover strict
input exclusions, raw cohort/person deduplication, household isolation, missing
labels, source versus withheld task access, fitting-only preprocessing,
category support, matched update schedules, epoch-zero/source selection,
frozen states, saved/reloaded probabilities, weighted scores, exposed controls,
and saved selection before final-test transformation.

The artificial miniature exercised every fitting/saving/evaluation path with
small budgets and temporary fixture records: all seven releases and34 saved
selections,1.249s internal runtime, all integrity flags true. It measured no ACS
scientific outcome. See [PIPELINE_CHECK.json](PIPELINE_CHECK.json).

All three real seeds passed in-process frozen neural/tree/PCA/preprocessing,
cached release and saved-selection checks. [Independent verification](VERIFICATION.md)
reconstructed household pools, allowed preprocessing, source label masks,
candidate selection and schedules from raw inputs. It verified522 binary-file
SHA256 hashes and21 bitwise-exact source-validation release replays, with no
training. The JSON includes replay source and command. Serialization changes
joblib object hashes, so saved-file hashes and output replay are distinguished
from in-process object hashes.

[Independent score replay](SCORE_REPLAY.json) reconstructed targets from the raw
CSV and saved test rows without importing the experiment scorer. All240 candidate
records,480 unweighted/PWGTP score sets and32,186 numerical values matched;
maximum error4.44e−16, tolerance2e−12. It also verified102 family selections,
90 MLP epoch minima,240 fitting-target hashes and12 matched schedule groups.
There were no fitting warnings or numerical prior fallbacks. Exposed trees
failed on one rare supported race category; the absent fitting category and
corresponding limitations are explicitly reported rather than repaired by
changing the race schema or extending training.

The final summarizer produced720 target/split rows,3,015 per-class rows and504
paired metric rows. Tables and the human research decision were checked against
raw per-seed values. Plot labels were inspected and made readable; CSV output
line endings were normalized in the reporting script only. Original metrics,
selections, protocol and execution-source hashes are unchanged.

Publication validation checks the compact-artifact manifest, staged byte hashes,
local evidence links and source syntax. Historical publication verification
checks the original250 result files and35 frozen source entries remain exact.
No dataset, model weights, cached releases, credential, or unrelated untracked
work is included in the ACS publication commit. The publication check record
is separate from scientific verification.
