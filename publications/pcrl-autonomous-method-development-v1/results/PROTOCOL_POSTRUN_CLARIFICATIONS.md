# Post-run clarifications

These clarifications resolve the independent verifier's discrepancies D1–D3 (`INDEPENDENT_VERIFICATION.json`). Locked files keep their bytes. No release, endpoint, number or label changes.

## D1: the statistical reviewer's self-reported times (low; custody narrative)

**Problem.** Two files give self-reported writing times that cannot be right:
- `CLAIMS_AND_ENDPOINTS_DRAFT.md` says it was written "at about 17:50Z".
- `DEVELOPMENT_REVIEW.md` says "written 17:30–17:50Z".

Neither time is possible. The record shows the following sequence:

| Time (UTC) | Event | Evidence |
|---|---|---|
| ≈17:34Z | DEVELOPMENT_REVIEW.md written | file modification time; seen by the coordinator at 17:34:42Z |
| 17:37:13Z | FINALIST_LOCK.json committed | commit `ad37c120c` |
| ≈17:37Z | CLAIMS_AND_ENDPOINTS_DRAFT.md written | file modification time; seen by the coordinator at 17:37:21Z |
| 17:42:54Z | ANALYSIS_LOCK.json written; it carries the draft's SHA-256 | the lock file |
| 17:43:02Z | ANALYSIS_LOCK.json and the draft committed | commit `345e0adb0` |
| 17:43:14Z onward | first final reader unit | queue ledger |
| 17:45:22Z | reporting receipt pushed | `614c9f36b` |
| 17:45:30Z | coordinator's one logged DEV_REPORT read | access log |

The draft's worked instance names the finalist. The reviewer had adjudicated the finalist in DEVELOPMENT_REVIEW.md §0, before writing the draft.

**What the record shows:**
- the endpoint roster and the claim conjunctions were fixed, hashed and committed before any final reader fit;
- they were published before DEV_REPORT was read.

**What the record cannot show:** whether the draft existed before the finalist lock was committed. The two events fall in the same minute.

**Deviation from PROTOCOL §9.** §9 asks for the claim conjunctions "before a finalist is selected". Both the reviewer's adjudication (≈17:34Z) and the promotion record (17:35:23Z) precede the draft (≈17:37Z). The conjunctions were therefore written with the finalist and its MODEL_SELECT results known. This is a timing deviation from §9. It precedes every final reader fit and the DEV_REPORT read. D1 is clarified, not resolved.

## D2: MODEL_SELECT size (info; documentation)

`PROTOCOL.md` §3 gives MODEL_SELECT as "8,000". The admitted export, the bank and every evaluation use **8,001** people, the cap of 8,000 plus a one-person household overshoot.

## D3: a threshold tie on a non-evaluated role (info; numerical fragility)

**What happens.** The finalist's income code bins the statistic 0.5 × LOCAL_TREE + 0.5 × JOINT_TREE income log-odds at empirical quantiles. Those quantiles were computed label-free on MODEL_FIT, so a few MODEL_FIT people sit exactly on a threshold:
- 2 people on seed 0 and 4 on seed 1;
- a last-bit change in the summation order of the tree scores moves such a person to the adjacent bin;
- the probability then changes by up to 0.19, with the decision unchanged.

**Effect on the evaluated roles: none.** On MODEL_SELECT and DEV_REPORT, the nearest statistic is at least 3.5e-5 from any threshold, and the verifier's independent recomputation matches bitwise. The deployment packages use the same summation code path as the release and reproduce DEV_REPORT bitwise.

**Note for deployment elsewhere.** A different floating-point summation order could flip a person who lies within about 1e-15 of a threshold. A production version should store thresholds that are not data values, for example midpoints between adjacent order statistics.
