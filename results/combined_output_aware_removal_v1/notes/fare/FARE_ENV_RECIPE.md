# Official FARE: environment recipe, pins and adaptations

Owner: method admission, 2026-10-03.

- **Compatibility work:** about 25 minutes, from 05:05Z to about 05:30Z. That is well inside the 2-hour limit, and there is **no blocker**.
- **Isolation:** nothing was installed into the shared project venv (`~/PCRL/.venv`).
- **Private paths:** everything below lives under `~/PCRL_eval_cache_private/oar_v1/` (written `<oar>` from here on).

## Pins

| item | value |
|---|---|
| FARE repository | https://github.com/eth-sri/fare, cloned to `<oar>/vendor/fare` |
| FARE commit used | `89cb1b66ed268c16659cbf7428c43e60da2df641`, "Fix build issues". This is HEAD of the default branch on 2026-10-03. |
| earlier pin `89cb1b66` | exists and **equals HEAD** (same full SHA as the durable-guarantees pin) |
| git tree of HEAD | `2e06b0fdd00a6c59ebfe89527436c1b1af21c281` |
| sha256 tree hash, `*.py` | `56a447007fb963090d5f7ea65de3d8b69b9bcddf1669e86234ebab4a52d89cbd` (109 files) |
| sha256 tree hash, `sktree/*` | `2b0c24d89e416fc0b3ff5bef5e7773978485259f3b29de68e0a658aa437c139e` (14 files: .py, .pyx, .pxd, build.sh) |
| `code/src/tree/alphabeta_adversary.py` | `85138f6db01b3dd01edc0a7f347f230b7f0fc62f18874facfe98572ee7cacbcd` |
| `code/src/tree/main.py` | `e1338a57f768f07d42245147729fe157bc8bee72f4c628b6d4d06fba6daefae0` |
| `sktree/_criterion.pyx` (official) | `ec12afa8f7f2f54ad9b1bf61768d1aca337fb4c9cbb7c2f05d4ea6f9ddd6acd4` |
| scikit-learn base (`install.sh`) | commit `fd60379f95f5c0d3791b2f54c4d070c0aa2ac576`, downloaded as a GitHub archive zip with sha256 `0020f3075c4a30833574544689c0edf88540126c2f13bed74673137ad94f829d`, extracted to `<oar>/vendor/scikit-learn` |
| patched build source | local git commit `f1635bfb…` = sklearn fd60379f + the official `sktree/*` overlay, unmodified; then `97e1d0d8…` = the single buffer fix below |
| `sklearn/tree/_criterion.pyx` (built) | `4fa73adee14f3f2e515725d9abceca0a158371bdf4a55eeafd891efce0127f47` |
| installed `.so` (sha256) | `_criterion` `94985b66…8c7b`, `_splitter` `17eaaa8f…445a`, `_tree` `9f383d72…5d68`, `_utils` `342dac33…1f55` |
| paper PDF | sha256 `bb3108f5e981cc3a08e026409a46a067441a3b036d7da602325759ee142853b2` |

**Tree hash construction.** For each tracked file (`git ls-files '*.py'`), with paths sorted under `LC_ALL=C`, write the line `"<sha256(file)>  <path>\n"`. The tree hash is the sha256 of the concatenated lines. `oar.fare_official.official_tree_sha256()` recomputes it at call time.

The repository has **no license file**. We run and cite it only, and do not redistribute its code.

## Environment `<oar>/env_fare`

- **Python:** CPython 3.9.12 (uv-managed, arm64). This is the version in the official `fareenv.yml`.
- **Build:** no OpenMP (`SKLEARN_NO_OPENMP=1`), Apple clang 17.

```bash
uv venv --python 3.9.12 <oar>/env_fare
VIRTUAL_ENV=<oar>/env_fare uv pip install --python <oar>/env_fare/bin/python \
  "setuptools<60" wheel cython==0.29.30 numpy==1.23.1 scipy==1.9.1 joblib==1.2.0 threadpoolctl==3.1.0 \
  statsmodels==0.13.2 pandas==1.4.3 python-box==6.0.2 tqdm==4.64.0 matplotlib==3.5.2 fairlearn==0.7.0 \
  folktables==0.0.12 pip
uv pip uninstall scikit-learn            # fairlearn pulled scikit-learn 1.6.1; the patched build replaces it
# official sktree/build.sh step (cp *.py *.pyx *.pxd ../scikit-learn/sklearn/tree), then:
cd <oar>/vendor/scikit-learn && SKLEARN_NO_OPENMP=1 <oar>/env_fare/bin/python -m pip install --no-build-isolation --no-deps -v .
# only for the official entry point (reproduction gate); the wrapper does not import these:
uv pip install torch==1.12.1 torchvision==0.13.1 pillow==9.2.0 tensorboard==2.9.1 "protobuf<3.20" dill==0.3.5.1 texttable==1.6.4
```

**Resulting versions.** Python 3.9.12, numpy 1.23.1, scipy 1.9.1, scikit-learn 1.2.dev0 (patched), statsmodels 0.13.2, Cython 0.29.30, setuptools 59.8.0. The full `pip freeze` is in `<oar>/env_fare_freeze.txt`.

**Pins versus `fareenv.yml`.** All pins follow `fareenv.yml` except two:

- `pillow` would have been 11.3 and was pinned back to 9.2.0;
- `wandb`, `tensorflow`, `parallel` and `cmake` were not installed, because neither the tree code nor the certificate code imports them.

**Size.** The environment is about 650 MB, the scikit-learn source about 215 MB, and the public ACS data cache (`vendor/fare/code/data`) 338 MB.

**Calling the official code.** `oar/fare_official.py` runs it in this environment: in-process when it is imported there, otherwise through `<oar>/env_fare/bin/python -m oar.fare_official --worker <job>` launched from the project venv. Three environment variables override the locations: `OAR_FARE_PYTHON`, `OAR_FARE_ROOT` and `OAR_FARE_SKLEARN_SRC`.

## Adaptations (complete list)

None of these adaptations changes the published objective, the criterion arithmetic or the certificate formulas.

### A1. Compiled buffer-size fix (the only change to compiled code)

**Problem.** `sklearn/tree/_criterion.pyx`, `ClassificationCriterion.__cinit__`, allocates the nine sensitive-count buffers (`sum_{total,left,right}_sens[_y0|_y1]`) with shape `(n_outputs, max_n_classes)`, where `max_n_classes` is the number of *task* classes. Upstream marks this `# TODO add max_n_sens`.

The buffers are indexed by group code up to `n_sens − 1`. With binary y and 5 race groups, writes therefore go out of bounds; Cython bounds-checking is off.

**Evidence from the unfixed official build** (logged in `<oar>/overflow_probe_unfixed.log`):

- with 3 or 5 groups, `ValueError: array is too big` in 5 of 6 runs;
- otherwise, different trees across repeated fits in the same process.

**Fix.** The buffers get `sens_width = max(max_n_classes, max(n_sens))` columns. The diff is 3 added lines and 9 changed allocations (`<oar>/oar_sens_buffer_fix.diff`).

**Verification.**

- When #groups ≤ #classes (every binary-s case and the paper's 3-group/4-class case), the shapes are unchanged.
- The reproduction gate is bit-identical before and after the fix: dp_ub 0.15712399439471292, and the same z_test hash.
- After the fix, fits with 5 groups are deterministic, and their root split matches a brute-force search of the published FairGini objective (test_f).

### A2. Tree serialization (wrapper side; no official file is touched)

**Problem.** The patched `Tree.__reduce__` (`sktree/_tree.pyx:708-712`) passes 4 constructor arguments, but `Tree.__cinit__` (L668) requires 6 (`cat_pos`, `cat_maxval`). As a result, `pickle.loads` of a fitted official tree fails.

**Fix.** The wrapper stores three things: the estimator shell, the 6 constructor arguments (with an all-False `cat_pos` and zero `cat_maxval`, since we have no categorical features), and the official `Tree.__getstate__()`. To rebuild, it calls `Tree(*args).__setstate__(state)`, which is what a working `__reduce__` would do.

**Verification.** Round trip: identical cells and fingerprint (test_d), and identical `predict` (test_h).

### A3. Certificate budget for more than 3 groups

**Problem.** The official multi-group script keeps ε_b = ε_s = 0.005 per pair (`main.py:425`). For 4 or more groups the Lemma 5.2 budget becomes negative, giving NaN, and the max then silently reports 0.

**Fix.** The wrapper passes ε_b = ε_s = (δ/#pairs)/10 and ε_c = 0.8·δ/#pairs to the unmodified `AlphaBetaAdversary` constructor. Its `eps_glob` and `eps_ab` are constructor arguments, so no code is changed.

**Verification.** For 1 pair at δ = 0.05 this gives exactly 0.005 / 0.04 / 0.005, matching the official values. The wrapper refuses any non-finite bound.

### A4. Cell ids as z

**What changes.** The certificate is given integer cell ids, not the median vectors.

**Why this is equivalent.** The official adversary identifies cells only through `np.unique(z, axis=0)`. Using ids also avoids a spurious `len(unique) == k` failure if two leaves had identical medians.

**Verification.** The gate bound is bit-identical with medians and with ids (`FARE_ADMISSION_CHECKS.json`). The official median representation is still available through `embed()`.

### A5. Lemma 5.1 rows

**What the official code does.** It runs Lemma 5.1 on `z_train`, the tree's own fit rows.

**What the wrapper does.** It rebuilds that same input from the stored aggregate per-cell × group fit counts. The official function uses only those counts, so the result is identical.

### A6. Group recoding

**Why it is needed.** The official criterion indexes buffers by the raw s value and counts groups as `len(np.unique(s))` (`_classes.py:247`).

**What the wrapper does.** `fit` recodes s to 0..G−1 and stores the original codes. It also recodes y, which scikit-learn re-encodes anyway.

### A7. Seed mapping and output routing

**Seed.** The official code fixes `random_state = 43` (`main.py:33`). The wrapper uses 43 + seed, so seed 0 reproduces the official setting.

**Output.** The compiled splitter `printf`s every split. The worker sends fd 1 to /dev/null and flushes the C stdio buffer.

### A8. Threads

Calibration, and the recommended runs, use `OMP/MKL/OPENBLAS/VECLIB_MAXIMUM_THREADS = 1`. The build has no OpenMP, so the tree is single-threaded in any case.

## Knowledge reused from durable-guarantees @ 956f5c88

**Reused (read-only clone in `<oar>/vendor/durable-guarantees`):**

- the same FARE commit, sklearn base, Python 3.9.12 and no-OpenMP build;
- the reproduction-gate configuration and target;
- one process per fit;
- routing `printf` to /dev/null.

**Corrected.**

- The earlier study attributed its "memory corruption across sequential fits" to the patched build in general. The cause is A1, and it is specific to more than 2 groups with a binary task. Binary-s fits are deterministic in-process.
- Its HMDA "dp_ub = 0.000" values are the A3 artefact.

**Not reused.** None of its baseline labels or results are used as evidence.
