# Cost, archive and closeout

## Time

| Event | Time (UTC, 2026-09-24) |
|---|---|
| Execution start (first discovery command) | 17:26:27 |
| Pre-fit registration commit (`ba39149`) | 17:48:02 |
| Launch commit (`8ac7577`) | ~18:52 |
| Instance launch | 18:53:41 |
| Runner start | 18:56:37 |
| All 58 units complete | 19:48:40 |
| Lock commit (`e5d555a`) | 19:54:21 |
| Outer unlock (remote-verified) | 19:56:26 |
| Assessment written | 19:58:06 |
| Independent verification complete | ~20:12 |
| Post-lock archives verified | ~20:14 |
| Termination requested | 20:15:55 |

- **Elapsed:** about 3 h from execution start to the termination request, and about 3.5 h to this closeout. That is well inside the 20 h ceiling (2026-09-25 13:26Z) and the calendar ceiling.

## Money

| Item | Upper estimate |
|---|---|
| Compute: `i-0732026d054f6d3ee` (c7i.8xlarge, on-demand $1.428/h), 18:53:41 → ~20:16 (≤ 1.40 h) | ≤ $2.00 |
| 120 GiB gp3 root volume, same interval, at $0.08/GiB-month | ≈ $0.02 |
| S3 requests and same-day storage | < $0.10 |
| **Total incremental estimate** | **≈ $2.1, against the $50 ceiling** |

- **Ledger:** `cloud cost --record` reported $2.0378 at 20:15:53. That figure is event-based; AWS billing posts later and is not a final invoice.
- **Other workloads:** no other workload's resources were created, modified or stopped. Two unrelated instances seen at start (`s1-aws-s1` and `p0-pilot`) were left untouched.

## Retained private storage

- **Where:** `s3://pcrl-ux-archive-ed9d21fd/pcrl_shared_context_release_v1/`, versioned with SSE-AES256. There is no lifecycle expiry on this bucket, unlike the older 7-day bucket.
- **Size:** 2.76 GB across all versions at 20:15, which is about **$0.06/month**.
- **Contents:**
  - 58 unit archives plus 58 receipts, each with a read-back-verified manifest (116 of 116);
  - 4 post-lock archives, read-back verified: outer scores, unlock receipt, assessment, post-hoc, reports and verifier scratch.
- **Deliberately not re-archived:** the restored outer-labelled originals. They already exist under the predecessor's pinned input versions.

## Local disk

- **Mac:** nothing was deleted by this study. Free disk fell from 7.1 GiB to about 4.6 GiB during the session, partly from other activity on the machine.
- **This study's temporary private files:** in the session scratchpad (restore test about 300 MB, manifests). They were deleted at 20:17Z after S3 read-back verification.

## Resource shutdown checklist

- [x] Study instance terminate requested (tag-checked, `Study=pcrl_shared_context_release_v1`).
- [x] Instance reached `terminated` at 20:16:34Z. Its root volume was deleted on termination (0 volumes attached or tagged).
- [x] Study security group `sg-01e1c3010494c470c` (zero ingress, study-tagged) deleted at 20:16:50Z. A fresh query found 0 live study instances, 0 study volumes and 0 study security groups.
- [x] No EventBridge rules or schedules were created. The host timers died with the instance.
