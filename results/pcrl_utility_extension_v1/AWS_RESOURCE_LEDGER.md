# AWS_RESOURCE_LEDGER — `pcrl_utility_extension_v1` (public, redacted)

Account identifiers, the bucket name, instance IDs and object version IDs are held only in the
private ledger under the git common directory (`pcrl_parallel_handoff_v5/terminal_1/private/`).
Everything task-owned is tagged `pcrl-task=utility-extension-v1`.

| resource | what | scope | final state |
|---|---|---|---|
| S3 bucket | task-specific archive | all four public-access blocks on; versioning on; SSE-S3 default encryption; **no lifecycle rules** | retained (54.89 GB, ~$1.26/month) |
| IAM role + instance profile | cloud host identity | `s3:GetObject/PutObject/ListBucket` on **this bucket only**, plus the AWS-managed SSM core policy for an operator channel | retained, $0 |
| IAM role | deadline stop | `ec2:StopInstances` restricted by condition to instances tagged `pcrl-task=utility-extension-v1` | retained, $0 |
| Security group | host networking | **no inbound rules** | retained, $0 |
| EC2 instances (3) | one runner; two replaced after infrastructure faults | m7i.2xlarge, 200 GB encrypted gp3, shutdown behaviour = stop, IMDSv2 required | **all terminated** |
| EBS volumes | root volumes | encrypted, delete-on-termination | **none remain** |
| CloudWatch alarm | idle-CPU stop | CPU < 3% for 30 min -> stop this instance | deleted after the run |
| EventBridge schedule | deadline stop | one-time `StopInstances` at the hard deadline | deleted after the run |

## Resources deliberately NOT touched

The account also holds other projects' resources: a `fedpub-*` bucket, a `vein-terraform-state`
bucket, an `s1-aws-*` bucket, that project's `s1-aws-s1-stop` schedule, four associated elastic IPs,
and about twenty **stopped** `pcrl-*` GPU instances from earlier studies (April-May 2026) with their
volumes. None was started, stopped, modified or deleted. The earlier PCRL bucket
`pcrl-bios-overnight-20260504` has a 7-day expiry lifecycle and was **not** used for this archive;
this study created its own bucket with no expiry so evidence cannot age out.

## Things this ledger does not claim

Costs elsewhere in this study are estimates from live unit prices and measured elapsed time, not
billed amounts. A stopped instance is not deleted storage: the volumes here were released by
**terminating** the instances after the outputs were archived and verified, and the API confirms no
task-tagged volume remains.
