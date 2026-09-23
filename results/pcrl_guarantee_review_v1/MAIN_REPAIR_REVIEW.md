# Review of `fix/retire-accuracy-guarantee` @ 5d4eda04639aae10733e4d72c2ceaae0e849ede5

**Base:** `origin/main` @ 55e4cb1d1b603a52e5ae16fd5c71d41ebb9b3827. **Not merged.** This review changes
nothing on main or on the branch.

**Diff** (`git diff --stat origin/main fix/retire-accuracy-guarantee`):

| File | Change |
|---|---|
| docs/ACCURACY_CERTIFICATE_RETIREMENT.md | +47 |
| pcrl/evaluation/certificates.py | ±24 |
| pcrl/purposes/verification.py | ±21 |
| tests/test_accuracy_bound.py | −324 (deleted) |
| tests/test_accuracy_bound_retired.py | +50 |

## Verdict

**Approve, with two small strengthening requests** (R1, R2). They are optional but recommended before
merging.

The code change is the claims-audit patch 0001, unchanged. It makes `certified_accuracy_bound` and
`NonlinearComplianceCertificate.check` raise `NotImplementedError`, and removes the only call sites in
`generate_report` and `print_compliance_table`.

The deleted file is correctly deleted:
- Every one of its 14 tests calls `certified_accuracy_bound`, either directly or via the shared helper
  (`TestCertifiedAccuracyBound`, `TestBoundHoldsOnSyntheticData`, `TestBoundOnPCRLSynthetic`).
- Every one asserts properties of the refuted bound: monotonicity, a majority floor, and "bound holds" on
  synthetic data.
- None can survive the retirement, and keeping any of them would assert the false theorem.

**Coverage lost versus coverage kept.** The deleted file incidentally exercised
`LinearComplianceCertificate` and an end-to-end `generate_report` run. Both remain covered on the branch:
- `LinearComplianceCertificate` by `tests/test_composition.py` and `tests/test_dominant_axis.py`;
- `generate_report` end-to-end by `tests/test_generate_report_alignment.py`.

`git grep` on origin/main confirms these files exist and are not modified. No valid behaviour loses its
only test.

**The new tests reject the false guarantee, not merely its API.**
`test_zero_affine_r2_with_ninety_percent_threshold_accuracy` builds the exact 20-row counterexample in
rational arithmetic: Cov(h,A) = 0, hence affine R² = 0, while the threshold accuracy is 9/10. At R² = 0 the
old theorem asserted accuracy ≤ π_maj = 1/2, so the test contradicts it. The three API tests fail on
unpatched main and pass on the branch. This was verified in a scratch checkout for patch 0001, and the
failing pattern is 3 of 4, with the counterexample test passing on both by design.

## Suite status (claimed versus verified)

- **Claimed** by the manuscript owner's integration note: the full suite on the branch gives 208 passed
  and 7 failed, and all 7 reproduce on clean origin/main (6 torch-autograd tests in
  `test_folktables_fixes.py`, plus `test_lora.py::test_n_trainable_matches_lora_count`).
- **Verified here:**
  - (a) The branch worktree's pytest cache (`/private/tmp/pcrl-main-fix/.pytest_cache/v/cache/lastfailed`)
    lists exactly those 7 node ids. It also lists one stale entry for the deleted
    `tests/test_accuracy_bound.py`, from a run made before its removal.
  - (b) The 4 retirement tests pass with this machine's Python.
- **Not verified here:** an independent full-suite run on either clean main or the branch. No
  torch-enabled Python exists on this machine, and none was installed. The "same 7 failures on clean
  main" statement therefore rests on the owner's run. The clean-main worktree left no pytest cache.

## Requests

- **R1 (recommended): import-based API tests when torch is present.** The AST-based tests avoid torch but
  are brittle. `test_nonlinear_certificate_check_raises` asserts that the first statement after the
  docstring is `raise`, so a harmless refactor would break it. Add a torch-guarded test that actually
  calls the API:
  ```python
  torch = pytest.importorskip("torch")
  from pcrl.purposes.verification import NonlinearComplianceCertificate, certified_accuracy_bound
  with pytest.raises(NotImplementedError):
      certified_accuracy_bound(0.0, 0.5, 2)
  with pytest.raises(NotImplementedError):
      NonlinearComplianceCertificate().check(np.zeros((4, 1)), np.array([0, 1, 0, 1]))
  ```
  Also assert that a `generate_report` smoke run returns `nonlinear_bound is None` for every report.
  `test_generate_report_alignment.py` already builds such a run and can be extended.
- **R2 (recommended): make the counterexample test state what it refutes.** Add one line asserting that
  the old formula's value at R² = 0, max(π_maj, π_maj + sqrt(0·k·π_maj(1−π_maj))) = 1/2, is less than the
  observed 9/10. This keeps the refutation even if the counterexample is later moved.
- **R3 (information).** Three experiment scripts on main call the retired function
  (`experiments/deployment_case_study.py`, `run_distribution_shift.py`, `run_extended_baselines.py`).
  After the merge they raise, which is fail-closed by design. The README on the branch does not advertise
  the bound.

- **R4 (recommended; a gap in the original patch 0001, now owned).** On the branch, the class docstring
  of `NonlinearComplianceCertificate` still presents the smoothing derivation, whose step 5 chains into
  the refuted bound, without a retirement notice. Only `check` raises.

**Follow-up patch for R1, R2 and R4:** `0001-followup-on-fix-retire-accuracy-guarantee.patch`, in this
directory, against 5d4eda04. It passes locally: 4 passed, and 1 skipped because the torch-guarded test
skips without torch. Apply with `git am` on the fix branch. It is not pushed to that branch, because the
branch is the manuscript owner's.

## Authorisation

Merging into public `main` is the owner's decision. The review found no defect that should block it.
