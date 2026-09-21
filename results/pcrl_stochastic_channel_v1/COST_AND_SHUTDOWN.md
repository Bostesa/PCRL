# COST_AND_SHUTDOWN — `pcrl_stochastic_channel_v1`

## Spend

| item | amount |
|---|---|
| Authorized incremental ceiling | $50.00 |
| **EC2 / compute actually spent** | **$0.00** |
| S3 requests (restore of one 0.30 GiB chunk, GET + manifest) | **< $0.01** |
| New storage created in S3 | **$0.00** (nothing uploaded) |
| **Total** | **< $0.01** |

**Why $0 on compute.** Gate G2 is a capacity diagnostic that runs in ~4 s per anchor on CPU. All
three anchors, both declared resolutions, and the subsumption check completed locally in well under a
minute. G2 **failed**, and the registration forbids running stages C and D after that failure, so
there was nothing left that needed cloud compute. Launching an instance to execute forbidden stages,
or to re-run a four-second diagnostic, would have been waste rather than diligence.

The instruction to keep heavy work off the laptop is respected: the heavy work in this study would
have been the Stage C/D audit matrix (attacker slates × conditions × seeds), and that work does not
exist, because the gate stopped it.

## Cloud resources

| resource | status |
|---|---|
| EC2 instances created by this study | **none** |
| EBS volumes created | **none** |
| Security groups / key pairs / IAM roles created | **none** |
| S3 objects written | **none** |
| S3 objects read | `pcrl_utility_extension_v1/manifests/exec_main__0000.json`, `chunks/exec_main__0000.tar.zst` (version `vqtdaPnnad06uT5j41O78Y8z35mI2XkN`) |

**Nothing to shut down.** There is no task-created compute, so the shutdown step is a confirmed
no-op rather than an omission. Verified: no instance was ever launched by this study; `launch.py` was
never invoked.

## Restore and its verification

One chunk restored, into **this worktree** (`/Users/nathansamson/PCRL-terminal-1-stochastic`) and
*not* into the main checkout, so `main` and every unrelated worktree stay untouched. The resolver
finds it because `FALLBACK_ROOTS[0]` is the module's own worktree root.

| field | value |
|---|---|
| chunk | `exec_main__0000` |
| object version | `vqtdaPnnad06uT5j41O78Y8z35mI2XkN` |
| stream sha256 expected | `a59beb0aa0d141a2bfe433454036d3f81fb51fdaac0aa209756c1ce2e2158943` |
| stream sha256 observed | `a59beb0aa0d141a2bfe433454036d3f81fb51fdaac0aa209756c1ce2e2158943` |
| files checked | **2,545** |
| files bad | **0** |
| verified | **true** |
| seconds | 13.7 |

Record kept at `RESTORE_VERIFICATION.json`. Contents: the 2018 raw ACS extract plus the fixed-
predictions and transfer artifacts — `anchors.npz` and `training/J/releases.npz` for all three
anchors, which are the stored `H_A`, `H_B` and `Z_J`.

**Nothing was deleted.** The restore is additive; the archive copy is untouched and remains the
system of record. No new deletion ledger entry is needed. The archive bucket
`pcrl-ux-archive-ed9d21fd` has **no lifecycle rule**, so the restored inputs remain recoverable
indefinitely and were not re-uploaded.

## Archival verification of this study's own outputs

This study's outputs are text and JSON totalling **144 KiB**, committed to git on branch
`research/pcrl-stochastic-channel-v1` and pushed to `origin`. Git is the system of record for them;
there is nothing large enough to warrant a chunked S3 archive, and creating one would add storage
cost for no durability gain. The one binary-ish input set (the restored chunk) is already archived
and verified upstream.

## Reproduction cost

Data-free fixtures: `python -m pytest tests/pcrl_stochastic_channel_v1/ -q` — 61 tests, ~12 s.
Stage B against restored inputs: ~4 s per anchor per resolution, CPU only, no cloud.
