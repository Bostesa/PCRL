# Confirmation plan: later population test (proposal only)

Status: draft, 2026-09-24. **This document authorises nothing** (handoff sec. 12, 16). It targets result
level 3 (advantage survives a later genuinely prospective population test); this programme aims at level 2.
No individual-level ACS file (any state, any year) was downloaded or opened; no model output influenced it.

## 1. Prior-use inventory (what was searched, what was found, limits)

Searched on 2026-09-24:

- **PCRL git history, all 77 local refs / 402 commits** (every `PCRL-terminal-*`, `PCRL-t2-*`,
  `PCRL-claims-*` checkout is a worktree of the same repository, so `--all` covers their commits).
  `git log --all -G` over `*.py *.md *.json *.sh *.yaml *.yml *.txt` for: non-06 `psam_[ph]NN` state files;
  non-CA `ss1Xp..` files; `folktables/20{10-17,19-23}`; `acs_20{10-15,19-23}`; `survey_year`/`year=` 2019-2023;
  quoted postal codes of the 12 largest states; `Texas|New York`. **No data-use hit.** The only
  `New York` hits are font-name entries in a figure cache and a literature note (K-TOpt evaluates
  Folktables on WA/NY: external work, not ours).
- **Branch-tip tally of every PUMS file name referenced** (all refs): `psam_p06` (3622), `psam_h06` (19),
  `csv_pca.zip` (329), `csv_pca_2017.zip` (250), `csv_pca_2016.zip` (200), `ss16pca` (213). California only.
- **Loaders:** `pcrl/data/folktables.py` (default 2018) and `experiments/run_folktables.py --states`
  (default `CA`) accept other states. Uncommitted ad hoc runs with other states cannot be excluded from git evidence.
- **`/Users/nathansamson/durable-guarantees`** (116 commits, 2 branches): all 1533 state/year records in
  history are `CA`/`2018`; `utils/folktables_io.py` pins `STATE="CA"`, `YEAR="2018"`; `data_cache/folktables`
  holds only `*_CA_2018_train` files.
- **Local file names only** (home directory to depth 9, excluding Library, caches, venvs): PUMS inputs are
  `psam_p06.csv` (2018 CA, three copies), `csv_pca_2016.zip`/`ss16pca.csv`, `csv_pca_2017.zip` and the
  2016/2017 dictionaries. No TX, NY or 2019+ PUMS file exists locally.
- **S3 object names** (`AWS_PROFILE=vein`): `pcrl-ux-archive-ed9d21fd` has one matching name,
  `psam_p06.csv`; `pcrl-bios-overnight-20260504` has none.

Limits: the archive holds tar+zstd chunks whose members were not listed (only the manifests committed to git
were searched); deleted EC2 disks, other machines, unfetched remote-only branches and renamed files are not
covered; the pickaxe did not cover other file types. HMDA code referencing `states=CA, years=2023` is
another dataset. **No prior PCRL use of any non-CA state or of 2019-2023 ACS was found. This is a
bounded search, not a certificate.**

## 2. Schema check (official dictionaries only)

Inputs the predecessor needs (`experiments/acs_transfer_data.py`, `run_acs_fixed_predictions.py`,
`pcrl_task_directed_release_v1/data.py`):
- X_A = PCA32 of the ten allowlisted covariates `AGEP, WKHP` (numeric) and `SCHL, MAR, RELP, CIT, DIS,
  DEAR, DEYE, DREM` (categorical), using the CA-fitted `CovariatePreprocessor`.
- H_A (4) = frozen anchor-head probabilities for `income_binary` (PINCP) and `civilian_at_work` (ESR) on X_A.
- H_B (2) = the frozen anchor-head probability for `public_coverage` (PUBCOV).
- Task label `MIG`. Audits `SEX`, `RAC1P`. Keys `SERIALNO, SPORDER`, weight `PWGTP`, checks `RT, ST`.

Checked against the official dictionaries for 2018 (14 Nov 2019), 2019 (15 Oct 2020), 2021, 2022 and 2023
(www2.census.gov `.../pums/data_dict/PUMS_Data_Dictionary_YYYY.txt`). Each variable's value block was
diffed against 2018:
- **2018, any state:** one national dictionary, so TX (ST=48) and NY (ST=36) files share CA's schema exactly.
  `docs/ACS_2018_SCHEMA_NOTES.md` applies unchanged. All listed variables are present.
  The official files exist: `.../pums/2018/1-Year/csv_ptx.zip` (47,257,733 B) and `csv_pny.zip` (34,958,982 B),
  both last modified 2019-10-31. Only HTTP HEAD requests were made; neither file was downloaded.
- **2019+:** `RELP` (codes 00-17) is **absent** and is replaced by `RELSHIPP` (20-38). The new codes
  split spouse and partner by same/opposite sex, and they merge "roomer/boarder" (11) with
  "housemate/roommate" (12) into 34. No exact map back to RELP exists, so the frozen preprocessor and PCA
  cannot be applied verbatim.
  Also: `CIT` labels reworded (codes unchanged); `PINCP` text edits only; 2023 renames `ST` to `STATE` and adds
  `JWMNP=888` "suppressed for select PUMAs". `MIG, SEX, RAC1P, PWGTP, SERIALNO, SPORDER, AGEP, WKHP, SCHL,
  MAR, DIS, DEAR, DEYE, DREM, ESR, PUBCOV` blocks are unchanged 2018->2023.
- **2020:** no standard 1-year release. Census issued only an experimental PUMS with experimental weights and
  advises against comparing it with standard ACS estimates. It is ineligible.

## 3. Recommended destination (predeclared design criteria, not performance)

**Primary: Texas, ACS 2018 1-Year person file (`csv_ptx.zip` -> `psam_p48.csv`).** Criteria, in order:
1. No recorded prior PCRL use (sec. 1).
2. Schema identical to development, so every frozen object applies with no crosswalk (sec. 2).
3. A standard, non-experimental release.
4. The largest non-CA state. Scaling 2018 state population totals suggests roughly 0.7x CA's cohort; this is
   **an unverified expectation, not a count.**
**Registered fallback: TX+NY 2018 pooled** (state-stratified household partition). It is triggered *only*
by the label-free admission count in sec. 4 falling short of the sec. 6 requirement. NY alone is smaller, so it
is never a substitute. If the pooled count is also short, do not run. No other switch is allowed, and none after
any label is read.
Not recommended now: CA 2019/2021-2023. That would be a temporal claim, and it needs a non-invertible
RELSHIPP->RELP input change that must be registered as part of the method.

## 4. What the test does and does not establish

- **Geographic transport is a different claim from temporal confirmation.** TX 2018 has the same survey
  year as development. It tests whether a CA-2018-developed, frozen release keeps its advantage in a
  different state's population. It says nothing about later years, and a win or loss does not transfer to
  CA 2019+. The write-up must name the claim "same-year geographic transport".
- **A different state does not establish that every person is new.**
  - Known: a housing-unit address lies in exactly one state, so CA-2018 and TX-2018 *household records* are
    distinct sample units. `SERIALNO` disjointness from the CA 2018 key set is checked at admission (keys only).
  - Not known: PUMS has no person-level linkage. A person who moved between states during the survey year
    could be sampled at two addresses in different months. We found no Census documentation of cross-address
    person de-duplication, so this cannot be ruled out or checked. It is expected to be rare. Such movers would
    concentrate in the outcome class `MIG != 1`.
  - Likewise, TX-2018 residents who lived in CA during 2016/2017 (spent years) cannot be identified.
- For a *later year of the same state*, ACS design ensures no address is sampled more than once in five years.
  So repeat addresses are excluded by design, but people who move can be sampled again. This is recorded
  for a future temporal study; it is not part of this proposal.
- CA 2018 developmental exposure, the 2016 prospective result and repeated 2017 use are all unchanged by this
  test. The 2016 result does not confirm a newly fitted model.

## 5. Admission and household handling (label-free until the lock)

1. Download the official zip once. Record the URL, byte size, SHA256 of the zip and of `psam_p48.csv`, and the
   dictionary hash. Commit this admission manifest before step 3.
2. Read only feature, key and weight columns (`FEATURES + SERIALNO, SPORDER, PWGTP, RT, ST`). Check `RT=='P'` and
   `ST==48`, check that feature codes are dictionary-valid, and check that `SERIALNO` is disjoint from the CA 2018 set.
3. Cohort rule identical to development: `19 <= AGEP <= 34`, `PWGTP > 0`. Duplicate `(SERIALNO, SPORDER)` keys
   abort admission, because a label-free replay cannot certify a dedup rule. Count rows and households, and
   compute household-aggregated PWGTP ESS = (sum_h W_h)^2 / sum_h W_h^2.
4. Household partition by `sha256("PCRL-shared-context-confirmation-v1|2018|48|" + SERIALNO)` (TX+NY
   fallback: the same scheme per state). It mirrors the 2016 design: fitting 50% (attacker_fit 25 /
   task_fit 25), validation 20% (attacker_validation 10 / task_validation 10), **final 30%**. Every person in a
   household (group-quarters persons have their own serials) lands in one pool. No cross-household or
   cross-pool leakage: preprocessors are frozen CA objects; destination heads and auditors are fitted only on
   fitting pools and selected on validation pools. Commit array hashes of keys, weights, raw rows and every
   partition.
5. Label support on fitting and validation pools only: all 9 RAC1P classes and both SEX classes, retaining
   absent classes. It is never computed on the final pool. Sparse RAC1P classes (3, 4, 5, 7) are reported, not
   merged.
6. Admission passes only if the number of final-pool households is at least `H_final_req` (sec. 6).
   Otherwise, apply the sec. 3 fallback rule once, or stop.

## 6. Outcomes, estimands and sample-size planning

Definitions are identical to 2018 (`labels_from_frame`):
- `same_residence = 1[MIG==1]`, valid MIG in {1,2,3}, blank masked.
- `SEX` two-class and `RAC1P` nine-class, with no collapse.
- Weightings: unweighted and PWGTP; fit weighting `0.5 + 0.5*PWGTP/mean(PWGTP)`. Raw PINCP (no ADJINC),
  `ESR==1`, `PUBCOV==1` are read only if destination anchor/B-side audits are registered.
- `Delta_task = CE_Y(C) - CE_Y(B)`; `Delta_recovery = CE_S(B) - CE_S(C)`. Route U and route P margins
  are exactly as in handoff sec. 16.

Paired household estimator. For endpoint e and weighting a (a_i = 1 or PWGTP_i), per household h:
`D_h = sum_{i in h} a_i (l_i^C - l_i^B)` and `A_h = sum_{i in h} a_i`. Then `Delta_hat = sum_h D_h / sum_h A_h`,
with linearised residual `u_h = (D_h - Delta_hat*A_h) / mean_h(A_h)`, per-household SD `s_e = SD_h(u_h)`, and
`SE(Delta_hat) ~= s_e / sqrt(H)`. The paired household bootstrap resamples households with replacement,
using identical draws for both arms, and estimates the same SE. Plug-in from the 2018 outer development
assessment: `s_e = SE_boot_2018(e) * sqrt(H_2018)`. Cross-fitting does not enlarge H.

Required final-pool households for endpoint e. M = registered endpoint family size (Bonferroni, one-sided
alpha/M each); K = conjunctive components in the route (power 1-beta/K each, so all hold jointly with prob
>= 1-beta); kappa >= 1 = transport SD inflation registered before admission, never estimated from TX
outcomes. Let `z = z_{1-alpha/M} + z_{1-beta/K}`.
- (a) Detect a true task difference of -0.003 against 0: `H = ceil( kappa^2 z^2 s_task^2 / 0.003^2 )`.
- (b) Detect a true AB/SEX recovery reduction of 0.002 against 0: `H = ceil( kappa^2 z^2 s_ABSEX^2 / 0.002^2 )`.
- (c) Establish the margin, where the upper bound must clear -0.003 (task) or -0.002 (AB/SEX), at planning
  effect `Delta*` beyond the margin: `H = ceil( kappa^2 z^2 s_e^2 / (|Delta*| - margin)^2 )`.
  If `|Delta*| = margin`, power equals alpha, so there is no finite H.
- (d) Non-inferiority, each protected comparison and each task cost must be at most +0.001 at planning
  value `Delta* <= 0`: `H = ceil( kappa^2 z^2 s_e^2 / (0.001 - Delta*)^2 )`. This margin is small and may bind.
- `H_final_req = max` over every registered endpoint and both weightings; admitted total `= H_final_req/0.30`.
  Arithmetic example only: alpha=.05, beta=.20, M=K=10 gives z = 2.576 + 2.054 = 4.630. Bonferroni is used as
  instructed; it is conservative for intersection-union conjunctions.

Coordinator fills these from the 2018 development assessment (**placeholders, do not invent**):

| Quantity | Unweighted | PWGTP |
|---|---|---|
| `s_task` (Delta_task, nominee vs B) | `<<FILL>>` | `<<FILL>>` |
| `s_ABSEX` (Delta_recovery AB/SEX) | `<<FILL>>` | `<<FILL>>` |
| max `s_e` over the remaining protected comparisons | `<<FILL>>` | `<<FILL>>` |
| `s_taskcost` (route P task cost) | `<<FILL>>` | `<<FILL>>` |
| `H_2018` (outer assessment households) | `<<FILL>>` | same |
| planning `Delta*` per endpoint (registered, pre-admission) | `<<FILL>>` | `<<FILL>>` |
| `M`, `K`, `alpha`, `beta`, `kappa` | `<<FILL>>` | same |

## 7. Candidate lock

- **Precondition:** one locked development nominee per route (labelled diagnostic if it failed inner
  selection) and an independently verified level-2 report.
- `CONFIRMATION_LOCK.json` pins the SHA256 of:
  - every frozen object: release maps and mixture parameters, the CA `CovariatePreprocessor`, PCA32, the
    H_A/H_B anchor heads, J, D17, comparator B and the strong matched controls;
  - the head/auditor fitting and selection code and config, the endpoint list, margins, weightings, alpha,
    M, K and bootstrap B (10,000) and seed;
  - the admission manifest and partition hashes, and the git commit.
  It also carries the registered predictions, with subjective probabilities, per the "register bets first"
  rule.
- An independent verifier replays the lock hashes and the label-free admission, then signs before any
  label column is read.
- **Single opening owner:** one named agent holds the evaluation permit, following the
  `validate_evaluation_permit` / `lock_verified()` pattern. The final-pool label read is pool-gated and
  logged, and it happens exactly once. Nobody else reads final labels. Opening marks TX 2018 as spent.

## 8. Evaluation procedure

1. Fit destination task heads and auditors on the TX fitting pools, and select on the TX validation pools
   with the locked rules. Freeze them and hash them into an addendum. Release maps are never refit.
2. Open the final pool once. Score C, B, D17 and every registered baseline and control on identical rows.
3. Compute both weightings and one-sided Bonferroni-adjusted bounds from the paired household bootstrap.
   Apply route U / P and the secondary checks: H-alone reference at .01 nats, no constant-release win,
   support and audit validity.
4. Report every endpoint and every baseline, including baseline wins, failures and absent RAC1P classes. The
   verdict is immutable. No re-run on TX, no runner-up substitution and no post hoc endpoints; anything
   extra is labelled exploratory.
5. An independent replay reproduces the scores from the hashes. Disclose legacy dependence: the frozen
   inputs encode CA-2018 development exposure.

## 9. Decision boundary

**Whether to acquire TX 2018 and run this evaluation is a separate, later decision**, made after a locked,
verified development candidate exists. No data have been acquired; none should be acquired automatically.
