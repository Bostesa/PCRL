# COST_AND_SHUTDOWN

## Spend

| item | amount |
|---|---|
| Authorized incremental ceiling | **$50.00** |
| EC2 / compute created by this assignment | **$0.00** |
| S3 requests | **$0.00** — no object was fetched; the existing restore was reused after hash verification |
| New S3 storage created | **$0.00** |
| Data transfer | **$0.00** |
| **Total incremental** | **$0.00** |

**Why nothing was spent.** The whole executed programme is small: the representation screen ran in
**121 s** on CPU across both families, both resolutions and three anchors; the certified synthetic
comparison and all fixtures take seconds. Gate `G_R` then **failed**, and the registration forbids
running the constrained fits, audits and finalist stress after that failure — so there was no heavy
work left to place on a cloud host. Launching an instance to execute forbidden stages, or to re-run a
two-minute diagnostic, would have been waste rather than diligence. Ceilings are limits, not budgets
to consume.

No small benchmark suggested a GPU was needed, because no job large enough to need one was ever
authorized to run.

## Resources created by this assignment

| resource | count |
|---|---|
| EC2 instances | **0** |
| EBS volumes | **0** |
| Security groups / key pairs / IAM roles | **0** |
| S3 objects written | **0** |
| Scheduled jobs / watchdog timers on cloud hosts | **0** |

**Nothing to shut down.** The shutdown step is a confirmed no-op, verified against live cloud state
rather than inferred: `describe-instances` filtered on
`tag:study=pcrl_stochastic_replacement_overnight_v1` returns **empty**.

## Read-only note on a foreign instance

`describe-instances` shows one running instance that is **not this assignment's** and was **not
touched**:

| field | value |
|---|---|
| id | `i-0fed8843075a831af` |
| type | `g7e.2xlarge` |
| state | `running`, launched 2026-09-20T07:56:50Z |
| tags | `Name=s1-aws-s1`, `study=s1-aws`, `purpose=study` |

It belongs to a different study and is explicitly outside this assignment's scope, so it was not
stopped, terminated or repurposed. **No accrued-cost figure is restated here**: an earlier session
estimated one, and repeating that estimate now would present a stale number as current fact. Its
current *state* is recorded above; its cost is for its owner to assess.

Five `in-use` EBS volumes exist in the account, none tagged to this study and none created by it.

## Local resources

An isolated Python environment under the session scratchpad (torch 2.14.0, cvxpy 1.9.3, scikit-learn
1.9.1). No project dependency was upgraded. The restored input tree in
`/Users/nathansamson/PCRL-terminal-1-stochastic` was **reused, not re-downloaded**, after verifying
the 9 key input files and a 40-file random sample against the archive manifest: all matched.

**Nothing was deleted.** No attacker weights, no broad directories, no unrelated project's files.

## Archival

This study's outputs are text, JSON and two figure files totalling well under 1 MiB, committed to git
and pushed. Git is the system of record for them; a chunked S3 archive would add storage cost for no
durability gain. The one large input set is already archived upstream in
`pcrl-ux-archive-ed9d21fd` (no lifecycle rule) and was re-verified rather than re-uploaded, so the
restoration path remains intact.
