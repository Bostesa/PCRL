# Publication validation

This records packaging checks separately from the completed research tests.
No research experiment, dataset download, representation retraining, attack
fitting or proposed ACS experiment was run for publication.

* Original preservation: all **3,570** original files in the six completed
  result directories match the publication manifest, including all omitted
  local files. The **250** staged original evidence files also match their
  original byte hashes. Original experiment commits and protocol/source hashes
  are unchanged.
* A clean export of the Git index contains all **250** published artifacts,
  matches **35** frozen source-manifest entries, and resolves **97** local links
  across **34** canonical reports/protocols. This export has no untracked local
  checkpoints, caches, datasets or duplicate source archives.
* Imported all **16** runner/helper/report/verification entry modules from that
  export, exercising **35** local modules in
  **1.676 s**. All imported project paths were inside the export.
  Existing installed dependencies were used; no fresh environment was installed.
* Compiled all **50** staged Python source/test files without running training.
  Independently checked local import closure and actual recorded dependencies;
  added observed tqdm and torchvision pins needed by eager package imports.
* Reviewed staged source changes against historical manifests, report claims,
  source dependency order, evidence selection, artifact sizes and omission
  boundaries. A targeted credential-pattern scan found no private-key headers
  or common GitHub/AWS/OpenAI token patterns in the staged payload. The only
  new binary blobs are four small PNG/PDF plots. No new file exceeds 5 MB.
* Whitespace checks pass for implementation, documentation and packaging.
  The full staged diff reports 151 pre-existing CRLF CSV lines and one trailing
  space in the original prediction-audit table. Those evidence bytes are
  deliberately unchanged; `.gitattributes` preserves original line endings.
* Earlier research tests were reused: the foundation's 156 passing tests and
  later stage-specific checks/logs remain published. No suite was rerun merely
  to package unchanged research code. Counts overlap across stages.

One packaging harness invocation initially resolved the virtual-environment
Python symlink to its underlying interpreter, losing that environment's package
lookup. Using the absolute `.venv/bin/python` path corrected the harness; the
isolated import check then passed. This did not execute or invalidate a research
run. No implementation change was needed.

Machine-readable records:

- [Isolated artifact/link check](export_integrity_check.json)
- [Isolated import paths and runtime](export_import_check.json)
- [Staged payload review and hashes](staged_review.json)
- [Original included/omitted artifact hashes](artifact_manifest.json)
- [Historical source equivalence](source_equivalence.json)

Recheck a clone with `python scripts/verify_redesign_publication.py`. On the
original machine only, add `--include-local` to check omitted files as well.
The staged payload review excludes its own JSON file to avoid a self-hash cycle.
Remote branch identity and authenticated access to the review index/four reports
are checked after committing and pushing, then reported with commit-pinned URLs.
