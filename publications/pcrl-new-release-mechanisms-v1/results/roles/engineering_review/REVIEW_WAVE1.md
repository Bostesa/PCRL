# Engineering review, wave 1 (before outcomes; saved verbatim with attribution)

Author: the engineering / claims reviewer of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 15:34Z on
2026-10-20. Read-only review. Saved verbatim; nothing below was edited. The coordinator's dispositions are in
roles/coordinator/ADJUDICATIONS.md.

---

ENGINEERING / CLAIMS REVIEWER: wave-1 engineering review, written 15:33Z on 2026-10-20

This was a read-only review. I edited no file outside my scratch folder and read no labels or real data. I ran one label-free probe, <session scratch>, which stubs `nrm.common.load` so nothing is loaded. It used 5 CPU-s, charged with `nrm.jobqueue charge --label engineering_reviewer`. I spawned no sub-agents.

**Code reviewed** (the tree changed several times during the review):
- coordinator modules at HEAD fc61c797d, including the evaluate.py tag change 9b69a6a96;
- family A at commit 300d017a6, card f19614b05. The working tree was being edited during the review; line numbers below are given for both;
- family B at commit e4da5a8fc, card 48fbac50f;
- `nrm/ctl` at 5cfe2fb5d;
- the `nrm/rules.py` integration points.

**Context checked in the ledgers:**
- FB_FIT_V0 / FB_FIT_V1 started at 15:24:24Z on clean code, after the card commit at 15:24:05Z. They saved 18 configurations: 12 randomized, R = 8.
- FA_TWIN and FA_GATE ran clean at 15:28-15:30Z under python3 3.14.3 / numpy 2.4.4. FB ran under the torch venv (3.13.7 / numpy 2.4.2).
- 20 of the 60 configurations are used.

No finding is a BLOCKER: I found nothing that invalidates the results saved so far.

---

### ENG1-01 SHOULD-FIX: recorded uncommitted-code paths are truncated (nrm/jobqueue.py:125-127 and :142-144)
- **Cause:** `_git()` returns `stdout.strip()`. The first porcelain line, " M path", loses its leading space, and `l[3:]` then cuts the first character of the path.
- **Evidence in the append-only ledger:**
  - FB_FIT_V0/V1, the NRM_REF_SCREEN_* units and VER_ADMISSION_V1 record `uncommitted_code_paths: ['rm/fa/prc.py']`;
  - VER_SELECT_SELFTEST_V1 and VER_RULES_AGREE_V1 record `['rm/devreaders.py', 'nrm/fa/prc.py']`, where the real file was nrm/devreaders.py.
- **Failure:** any audit that matches dirty paths against imported modules misses the first dirty file. A fit launched with its own module modified would show a path that does not exist. The bug is inherited unchanged from `tsz.jobqueue`. The verifier's R9 compares hashes, so it is not affected.
- **Fix:**
  - use `git status --porcelain -z` and split on '\0', without stripping;
  - disclose the existing rows as known-truncated (do not rewrite them);
  - optionally refuse fit/evaluate/readers launches whose entry package is dirty. At present the queue records dirty code but never refuses it.

### ENG1-02 SHOULD-FIX: the MODEL_SELECT gate can be bypassed through the coordinator's helpers (nrm/common.py:158-175 and :187-193; closeout.py:86)
- **Cause:** `_caller()` attributes a read to the first frame outside `nrm.common`.
- **Probe result** (called from a function whose module is `nrm.fa.*`):

  | Call | Attributed to | Result |
  |---|---|---|
  | `nrm.evaluate.ms(...)` | `nrm.evaluate:ms` | AUTHORIZED: MODEL_SELECT income, occupation and SEX |
  | `nrm.screen._load_sex(...)` | `nrm.screen:_load_sex` | AUTHORIZED |
  | the same `C.load` called directly | `nrm.fa.*` | REFUSED |

- **Failure:** a family helper, or an agent's script outside the queue, could read MODEL_SELECT labels inside a fit, and the log would show an ordinary benchmark read.
  - closeout's `MODEL_SELECT_reads_outside_benchmark_screen_readers_verify` can never be non-empty, because `load` already refuses MODEL_SELECT for fit and calibrate.
  - Today no file in nrm/fa, nrm/fb or nrm/ctl imports evaluate, screen or devreaders (checked with grep).
  - The verifier's R10 (v1_access.py:114-122) detects this after the fact, but only for queue units. Reads with `unit_id = None` escape it.
- **Fix:**
  - for MODEL_SELECT operations, refuse when any frame in the stack, or `__main__.__spec__.name`, belongs to a fitting entry point (`FIT_ENTRY`);
  - log the main module and the full module chain with every read;
  - refuse label reads that have no NRM_UNIT_ID;
  - replace the closeout check at line 86 with R10's main-module and stage cross-check.

### ENG1-03 SHOULD-FIX: the 60-configuration budget is not safe under concurrent saves (nrm/release.py:79-81 and :101-108)
- **Cause:** `save` reads the ledger, checks the count and appends, without a lock.
- **Evidence:** two units do save concurrently. FB_FIT_V0 and FB_FIT_V1 interleaved saves #11-#18 within the same seconds (15:24:47-53Z).
- **Failure:** with 59 used, two units can both read 59 and both save, giving 61 configurations. `configuration_number` can also repeat.
- **Fix:** take an `fcntl.flock` on a lock file around the read, the check, the npz/json write and the ledger append.

### ENG1-04 SHOULD-FIX: the custody restore check will fail on the randomized FB releases (nrm/restore_check.py:44, loop at :34-46)
- **Cause:** it calls `from_features(..., draw=0)` without draw keys. `nrm.fb.release.predict` raises PermissionError for every randomized configuration without keys (release.py:88-90).
- **Failure:** 12 such releases are already saved. When nothing is packaged, the check re-runs every release, so it aborts with an uncaught exception at closeout.
- **Second risk:** it re-runs FA (fitted under 3.14.3 / numpy 2.4.4) and FB (torch venv 3.13.7 / numpy 2.4.2) in one process. FB forces the torch interpreter, so FA's bitwise parity would be tested under a different numpy and BLAS. FA routing is an argmin over BLAS products, so a one-unit-in-the-last-place difference can move a person to another cell.
  - The same applies to a multi-package `nrm.deploy check`.
- **Fix:**
  - pass `keys=deploy._keys_for(rec, 'MODEL_SELECT', S['row_id'])`;
  - add a family filter and run one restore unit and one `deploy check` unit per family, each under the family's fit interpreter;
  - record per-release exceptions as failures instead of aborting.

### ENG1-05 SHOULD-FIX: the wire hashes that selection needs do not exist in one consistent format
- **What `rules.select_candidate` needs** (rules.py:813-835 and 793-795): `wire_sha256_eval == wire_sha256_grid` per version, as exact dict equality, plus `reference_wires` for all 9 BENCHMARK references.
- **What exists today:**

  | Source | What it stores |
  |---|---|
  | `nrm.evaluate` (evaluate.py:177-179) | no wire hash at all (complete_eval's `wire_sha256` and the per-draw hashes are discarded) |
  | `nrm.release.save` (release.py:99) | `arr_sha` per role: dtype, shape and probabilities, no argmax |
  | `nrm.devreaders` (devreaders.py:455-470) | `SB.wire_sha` (probabilities + argmax) per task; a list over draws for randomized releases |

- **Failure:** if the selection driver fills the eval hashes from RELEASE.json, every row becomes NOT_SELECTABLE_OUTPUT_MISMATCH and the outcome is a wrong NO_DEVELOPMENT_CANDIDATE. If reference hashes come in another format, alias-of-reference detection silently never fires.
- **Fix:**
  - use one function everywhere: `SB.wire_sha` on the MODEL_SELECT arrays per task, as a list over draws for randomized releases;
  - store it per version in EVAL_MS.json;
  - compute the reference wires with the same function through `comparator_release`;
  - add a selftest that the eval, grid and reference hashes agree for one deterministic and one randomized saved release.

### ENG1-06 SHOULD-FIX: randomized-release evaluation records are internally inconsistent (nrm/evaluate.py:110-127, :138-146, :96-98)
- **Stale draw-0 fields:** `utility_of` deep-copies draw 0's utility and replaces only value, se, se_mc and draw_values. These keep their draw-0 values:
  - per clause and guard: point_pass, margin, margin_se_units;
  - top level: point_pass_12, guard_pass, by_sex, task_summary.
- **Failure:** if the draw-averaged occupation NLL excess is 0.0102 but draw 0 is 0.0098, the stored clause says point_pass True while value > cap.
- **Reserve column:** `reserve_of` (the UTILITY_LEDGER.csv reserve columns) leaves out se_mc. The governing `rules.reserve_status` adds se_mc in quadrature, so the public ledger can show MET where the rule gives NOT_MET.
- **Monte Carlo SE:** `paired_costs` with unequal draw counts treats all Ra × Rb pairs as independent, so the MC SE is too small. `total|nll` has no se_mc.
- **Fix:**
  - recompute or drop the derived fields from the averaged values;
  - label the ledger reserve columns "without MC", or call `rules.reserve_status`;
  - compute each release's MC SE from its per-draw means and combine the two in quadrature.

### ENG1-07 SHOULD-FIX: FA can save a configuration in one version only (nrm/fa/prc.py stage_configs: committed 300d017a6 about l.1283-1288; working tree l.1305-1309)
- **Cause:** NOT_FEASIBLE and ALIAS_OF are decided separately for each version from SELECTION_V<v>.json.
- **Failure:** suppose V0 selects λ 2 for both S1 and S2 (so PRC_S2 is recorded as an alias and not saved), while V1 selects different λ. Then PRC_S2 exists only in V1, and the same can happen to ORD_* or to a NOT_FEASIBLE budget.
  - The configuration then can never be selected or compared:
    - `nrm.evaluate` marks version 0 missing;
    - `rules.point_eligibility` returns INCOMPLETE;
    - `devreaders run --seed 0` crashes in `RL.load`.
- **Fix:** decide feasibility and aliasing per budget across both versions before saving, so a configuration is saved in both versions or neither. If only one version aliases, save the alias there too and record `alias_of`.
- **Also:** detect a protected selection whose release is bitwise equal to PRC_TWIN (a stalled chain at large λ, seen on invented data) and record ALIAS_OF_PRC_TWIN instead of spending two configurations.

### ENG1-08 SHOULD-FIX: no FB revision can be fitted with the current file layout (nrm/fb/learner.py:314-318 and :328; nrm/fb/release.py:27-33 and :44-45)
- **Cause:** RELEASE_ROOT.bin and LEARNER_*.pt are per version under MODELS/V<v> and written with O_EXCL / 'xb'. `predict` and `model_paths` always read MODELS/V<v>.
- **Failure:**
  - an R1/R3 revision (FB2_*, FB3_*) refuses before training ("version v was already fitted");
  - passing `root=` writes the models elsewhere, but `predict` and `deploy` cannot find them;
  - `fit_version` also re-saves all CONFIGS, which `RL.save` refuses after the training.
- **Fix before any FB revision:**
  - a model directory per revision tag (MODELS/<rev>/V<v>);
  - CONFIGS selected by revision;
  - the tag recorded in CHANNEL_<cid>.json;
  - a root key per (revision, version).

### ENG1-09 SHOULD-FIX: the planned random-replacement and merge controls are biased or cannot be deployed (nrm/ctl/post.py:441-463; CTL_CARD_v1.md C5 and lines 114-115)
- **(a) Biased bank:** the replacement bank is the twin's MODEL_FIT vectors. For FB (no cross-fitting) and FA (teachers and decoders fitted on MODEL_FIT) these are in-sample and more confident than held-out vectors.
  - Replacing MODEL_SELECT people's vectors with them inflates the control's utility cost. The SEX-blind control then looks worse than it should, which favours the candidate (a strawman control).
  - Fix: `bank_role='CALIBRATION'`, i.e. the twin's held-out vectors from the same release function, or out-of-fold vectors where they exist.
- **(b) Not deployable:** the draws come from a row-order random generator per (config, version, role, recipient), not keyed per person.
  - If C5 becomes the decisive control, its package cannot meet `nrm.deploy`'s per-request keyed contract: a subset or reordered request changes the draws.
  - Fix: per-person keyed draws, for example BLAKE2b with a private key as in `nrm.fb.channel`, with the coupling recorded.
- **(c) C2 merges:** these run on in-sample vectors when a family does not cross-fit. Record this, and keep posteriors on CALIBRATION as the card already states.

### ENG1-10 NOTE: fix two files before the grid lock (nrm/devreaders.py:99-100 and :542)
DEVGRID_CODE_LOCK.json is not written yet. The lock covers nrm/release.py and nrm/devreaders.py, and `make_lock` is write-once. Apply any change to these two files, such as ENG1-03, before running `nrm.devreaders lock`. After that, any edit fails `lock_check` for good, with no amendment path.

### ENG1-11 NOTE: the card gate does not tie a card to its family (nrm/common.py:113-126; closeout.py:87)
- `card_info` accepts any committed file under cards/, including README.md or another family's card. For example, an `nrm.fa` fit with `--card cards/FB_FSC_v1.md` passes.
- `save` checks the record's card separately from the card logged at the read.
- Fix: require a prefix match (FA_* for nrm.fa, FB_* for nrm.fb, CTL_* for nrm.ctl), refuse README.md, and have save check that the unit's logged read used the same card.
- closeout's `late_card` check can never fire, because the card is checked at read time.

### ENG1-12 NOTE: nrm.devreaders draw default and missing MC assembly
- The docstring (devreaders.py:33-34) says an nrm member's "draw" defaults to 0. The code (line 453) defaults to 'rep' (R1→0, R2→1, R3→2, R4→3), which matches the FB card. Fix the docstring.
- `rules.mc_cell` and `cell_se_mc` are never assembled by any code: `paired()` gives `cells` and `cell_se` only. With one draw per cell, Monte Carlo variation is absorbed into `se_reader_split`.
- Reporting MC uncertainty separately, as the protocol requires, needs the coordinator to build `mc_cell` entries from the draw-offset members (+2, +4). R = 8 covers the largest draw index needed (7).

### ENG1-13 NOTE: stale FB learner docstring (nrm/fb/learner.py:1-20)
- The module docstring still describes shared stage-1 networks plus six continuation epochs at lr 3e-4. The code (`fit_learner`) and the committed card train every learner jointly from scratch with its own early stopping. Fix the docstring.
- The twin and the MINF/ISO controls are fitted in the unit that reads MODEL_FIT SEX. In the code, SEX reaches only the 'sex' geometry, so these three never use it. The verifier's invariance check (V3X_FB_INVARIANCE) should be the cited evidence for calling them SEX-blind.

### ENG1-14 NOTE: ORD's operating point used SEX (FA stage_paths)
ORD's operating point is chosen under budgets that include men's and women's Brier, computed with MODEL_FIT SEX. The record discloses this. Claims should say "SEX-blind mapping, SEX-informed operating-point selection" rather than plain SEX-blind (protocol 5.8). T* is fixed from the full pool in the nested estimator (disclosed as non-nested).

### ENG1-15 NOTE: TORCH_NUM_THREADS is not a torch setting (nrm/jobqueue.py:42-43)
torch ignores it. FB calls `torch.set_num_threads(1)` itself and FA uses `threadpoolctl`, so current units are single-threaded. Any new torch entry point must set this explicitly. The CATI readers are logistic regression and ET uses `n_jobs=1`, so no reader is multi-threaded.

---

### Verified with no defect found
- **Access layer:** closed roles are refused; CALIBRATION SEX is not admitted; MODEL_SELECT is refused for fit and calibrate; the card gate is enforced at the read; entry-point prefix matching is exact.
- **nrm.devreaders:**
  - constants: seed base +600, R1-R4 tags, bootstrap seed;
  - each FIT, VALIDATION and SCORE person gets exactly one draw (the same instance index for that cell);
  - stacks are validated per draw;
  - the cache key includes the drawn wire.
- **nrm.screen:** readers fit on one half and score the other, with draws 0 and 1 screened separately.
- **nrm.deploy:** keyed draws for randomized packages and refused keys for deterministic ones. Field and schema refusals run before the key check, so they are not masked.
- **FA:** no runtime SEX (π from X; the SEX-conditional occupation teacher is evaluated at s = 0 and s = 1); household-grouped cross-fitting and outer folds; selection uses only MODEL_FIT nested estimates; `predict` produces the saved wire.
- **FB:**
  - private `os.urandom` root, so draws cannot be recomputed from the published code;
  - draws keyed by (role, instance, row_id) and independent of row order;
  - one coupled scalar per person;
  - temperatures fitted on CALIBRATION only;
  - MODEL_SELECT never read.
- **nrm.evaluate:** metrics on each emitted vector and its own argmax, averaged over draws; `total|nll` and the secondary paired-cost keys match what `rules.frontier_comparison` expects.

**Order of fixes:**
1. ENG1-07 before FA stage 3.
2. ENG1-03 (and any other release.py change), then ENG1-10's lock.
3. ENG1-05 before selection.
4. ENG1-04 before custody.
5. ENG1-08 before any FB revision.
