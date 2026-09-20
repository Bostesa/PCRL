# COST_AND_SHUTDOWN — `pcrl_utility_extension_v1`

Prices were read live from the AWS Pricing API before launch, not assumed: m7i.2xlarge (8 vCPU,
32 GiB, x86-64) On-Demand **$0.4032/h** in us-east-1; gp3 storage $0.08/GB-month, i.e. $0.0219/h for
the 200 GB root volume. These are **estimates from verified unit prices and measured elapsed time**;
they are not a billing statement, and delayed billing data cannot enforce an exact cap.

## Measured compute

| host | purpose | running time |
|---|---|---|
| host 1 | bootstrap; found the verification stream bug | ~24 min |
| host 2 | relaunch with the fix; T0 PASS, T1 FAIL (incident 1) | ~12 min |
| host 3 | diagnosis, Amendment 1, T0/T1 PASS, T2 pilot, closeout | ~52 min (incl. two rehearsal stop/start cycles) |
| **total** | | **~1.47 instance-hours** |

| item | estimate |
|---|---|
| EC2 compute, 1.47 h x $0.4032 | **$0.59** |
| EBS gp3 while the volumes existed (~1.5 h x 200 GB) | **$0.03** |
| S3 PUT requests (~22.4k, mostly the run sync) | **$0.11** |
| Data transfer (in free; same-region reads free; small egress to the laptop) | **< $0.02** |
| **incremental compute/network total** | **~$0.75** against a $35 target |

## Ongoing storage (separate from compute)

| prefix | size | note |
|---|---|---|
| `chunks/` | 49.40 GB | 35 verified archive chunks: 29 historical-results chunks, 3 execution-bundle chunks, 3 run-results chunks |
| `run/` | 5.43 GB | browsable copy of the run outputs (redundant with `results_run__*` chunks; deletable) |
| `manifests/` + `verification/` | 57.8 MB | per-file SHA-256 manifests and the AWS read-back records |
| **total** | **54.89 GB** | **~$1.26/month** at S3 Standard $0.023/GB-month |

First 30 days of storage plus compute: **~$2.0**, against the agent-selected $50 operating ceiling.
Nothing expires automatically: the bucket has **no lifecycle rules**, so archived evidence is never
silently deleted (unlike the 7-day bucket used by earlier PCRL work).

## Shutdown mechanisms, and the rehearsals

Three independent stops, all installed before the scientific run started:

1. **In-host deadline** — a persistent systemd timer at an absolute UTC deadline (`Persistent=true`,
   so it still fires if the host was stopped across it). Verified armed for 16:23Z.
2. **AWS-side idle stop** — a CloudWatch alarm (CPU < 3% for 30 min) with an EC2 stop action, for the
   case where the worker process dies but the instance lives.
3. **AWS-side deadline stop** — an EventBridge one-time schedule calling `ec2:StopInstances`, under a
   role that can only stop instances tagged `pcrl-task=utility-extension-v1`.

Both AWS-side paths were **rehearsed against the actual instance ID** and observed to stop it, then
the instance was restarted; only after both rehearsals passed was the `START` flag written. The
scheduler also carried its own cost guard ($30) and time guard (45-minute closing reserve); neither
was reached — the run ended on a scientific gate, not a budget.

Instance-initiated shutdown behaviour was **stop**, not terminate, so a closeout racing the deadline
could not destroy unarchived evidence. Confirmed official behaviour used here: stopping an instance
does not delete it and does not end EBS charges
(<https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html>), which is why the volumes were
released only by terminating after the outputs were archived and verified
(<https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity.html> for the
checksum semantics relied on in the archive).

## Final state (verified by API after closeout)

| resource | state |
|---|---|
| all three instances | **terminated** |
| task-tagged EBS volumes | **none** (deleted on termination) |
| idle-stop alarm, deadline schedule | **deleted** |
| security group (no ingress), two IAM roles, instance profile | retained, **$0/month**, task-scoped |
| S3 bucket | retained, private, versioned, SSE-S3, no lifecycle — **the only ongoing charge** |
| elastic IPs | 4 exist in the account, all associated and belonging to other projects; not touched |
