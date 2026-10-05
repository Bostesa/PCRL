# Confirmation plan (prospective, short)

**Status: DRAFT, written before any fit of this study.** Nothing has been acquired, downloaded or opened. No
confirmation file was read, and no HTTP request was made for one. The plan does not depend on this study's result,
which does not exist yet.

Owner: data and provenance role. The usage re-audit was run read-only on 2026-10-05 (UTC). Its scope is in §3.

## 1. When a confirmation is justified

| Development label (prompt §16) | Action |
|---|---|
| DEVELOPMENT_ADVANTAGE_ESTABLISHED | Freeze the entire procedure. Draft a population-specific confirmation protocol from §2–§4 and ask the user to choose the population. Nothing is acquired automatically (prompt §5, §16). |
| EXPERIMENTAL_NO_ADVANTAGE | No confirmation. Every candidate below stays unspent. |
| INCOMPLETE_OR_INVALID | No confirmation. Repair the development comparison first. |

A change made after NEW_DEVELOPMENT_ASSESSMENT opens creates a new development candidate, never a confirmation
candidate. That includes the grid, thresholds, arms, attackers and nominees.

## 2. What could be confirmed

- **No fresh Adult cohort exists in the 83-column schema.** Every Adult row is exposed (`EXPOSURE_LEDGER.md`):
  - old assessment and cert are spent;
  - the PCRL v2 validation split was spent as a selection set;
  - the loader's dropped rows have missing values.

  The deployed Adult checkpoint therefore cannot be confirmed on new people.
- **An ACS adaptation is method transport, not confirmation.** It changes the input and label contract:
  - capital-gain and capital-loss have no direct PUMS equivalent;
  - the 2018 `RELP` spouse code is expected to be sex-neutral, so the Husband/Wife SEX proxy weakens or disappears;
  - occupation_group needs an `OCCP` crosswalk;
  - the >50K income threshold is in different-year dollars;
  - 2019 and later replace `RELP` with `RELSHIPP`.

  Any ACS run would be registered as transport of the frozen procedure, refit end to end. It would not be confirmation of this Adult checkpoint.

## 3. Candidate populations: audited status

| Candidate | Status | Evidence (re-checked 2026-10-05 UTC) |
|---|---|---|
| Adult, all roles of this study | **Exposed** | `ROLE_MANIFEST.json`, `EXPOSURE_LEDGER.md` |
| Adult old assessment (5,243) / cert (1,500) / rgj DEVELOPMENT_ASSESSMENT (3,397) | **Spent** | rgj and jcv role manifests; rgj `EVALUATION_LOCK.json` |
| ACS CA 2018 | **Spent.** It is the dg Folktables state/year and fed the PCRL v2 ACS encoders. The AGEP 19–34 cohort is exhausted. | dg full history: `STATE = "CA"`, `YEAR = "2018"` are the only values ever set in `utils/folktables_io.py`. `ad2c08872:results/redesign_20260910_acs_locked_evaluation_v1/DATA_INDEPENDENCE.md` |
| ACS CA 2017 | **Spent** (final partition opened 2026-09-17) | `349efa454:results/redesign_20260917_acs_spectral_transport_v1/` |
| ACS CA 2016 | **Spent** (scored once) | `5e154e5c4:results/pcrl_final_prospective_v1/EVALUATION_LOCK.json` and `INFERENCE_2016.json` present |
| ACS TX 2018 | **Reserved.** It is the named primary confirmation cohort of the shared-context programme. It was never acquired. Using it here would spend it there too. | PCRL pickaxe (all refs): `csv_ptx` / `psam_p48` appear only in plan or audit commits `66e8186f8`, `b67f04b5e` and `2be8ceabe`. `537e74a44:.../pcrl_shared_context_release_v1/CONFIRMATION_PLAN.md`. No file on local disk (Spotlight and the data folders) or in any drive inventory. No dg reference. |
| ACS NY 2018 | **No use found.** It is named only as the shared-context pooled fallback. | `csv_pny` only in `66e8186f8` and `2be8ceabe`; `psam_p36` only in `2be8ceabe`. Absent from local disk, drive inventories and dg history. |
| Other ACS states or years | **No use found.** 2019 and later have a different schema. | Drive inventories (3.26 M entries across both relocation folders) list only CA person files (`psam_p06`, `ss16pca`). No other-state hits in the PCRL refs or dg history. |

Re-audit scope:
- PCRL: all 50 remote heads, each equal to its local tracking ref, and the full `git log --all -S` history.
- durable-guarantees: full history of a clone at `956f5c8`, with every object present.
- Local Spotlight and the data folders.
- Both drive relocation inventories.
- The shared handoff directory `pcrl_week_finish_v2/evaluation/`: no handoff claims a new population.

"No use found" is limited to these scopes; it is not proof of non-use elsewhere.

## 4. If a development advantage is established

1. **Do not default to TX 2018.** The user decides whether one population may serve both programmes.
2. **Register and push before any download:**
   - population, filter and household-grouped allocation with a fixed salt;
   - the Adult→ACS contract and label definitions (capital columns, relationship, `OCCP` crosswalk, income threshold);
   - the frozen procedure, grid, selection rule, endpoints, allowances, multiplicity and a precision target;
   - a one-shot evaluation rule;
   - the triggering development result.
3. **Admission is label-blind:** dictionary and schema checks and label-free counts only. Labels are read once, after the confirmation's evaluation lock is pushed.
4. **Re-run this audit immediately before the decision:**
   - `git fetch --all`;
   - `git log --all -S` for the chosen state's file names in both repositories;
   - the drive inventories and Spotlight;
   - the open confirmation plans and handoffs.

   Record the date in the protocol.
5. **Claims a confirmation could support:** the frozen procedure's advantage under transport to that population and contract. It could not support population privacy, DP or MI, other attributes, or properties of the Adult checkpoint.
