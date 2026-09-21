# RUN_STATUS

Programme **closed** at gate `G_R`. Branch `research/pcrl-stochastic-replacement-overnight-v1`,
based on precursor pin `cd895e4291b3dc2c0ed5e9c5482415d30f4e9f2a`.

| stage | state | note |
|---|---|---|
| Precursor corrections | **complete** | 7 items, 10 fixtures |
| Release contract | **complete** | S1; prior J disclosure not established |
| Registration | **complete** | PROTOCOL, METHOD, STATISTICAL_PLAN, DATA_USE, RUN_MATRIX |
| R — representation screen | **complete, FAILED** | 2 families × 2 resolutions × 3 anchors |
| A — action library | **untriggered** | gate closed the branch first |
| Q — constrained fits | **untriggered** | 0 of 144 nominal ACS fits run |
| Audits | **untriggered** | no disclosure measured |
| Finalist stress (`test`) | **untriggered** | test split never read |
| Synthetic certified comparison | **complete** | registered under §9, independent of the ACS gate |
| Independent replay | **complete** | 14/14 checks pass |

## Unit counts

Nominal Stage-R units 12 (2 families × 2 resolutions × 3 anchors), executed as 4 family/resolution
units over 3 anchors each. Unique configurations 4, exact duplicates 0, numerical failures 0,
quarantines 0. Nominal Q fits 144, **untriggered 144**. Nominal action-library blocks 8,
**untriggered 8**.

Per-unit ledger with content-hash configuration IDs: `ledger/`.

## Owned cloud resources

**None.** No instance, volume, security group, key pair, IAM role or S3 object was created by this
assignment. The foreign instance `i-0fed8843075a831af` (tag `study=s1-aws`) is outside this
assignment and was not touched; see `COST_AND_SHUTDOWN.md` for its read-only current state.

## Blockers

None. The branch is **scientifically closed**, not paused. There is no authorized work left
outstanding and therefore no resume command: continuing would require a new registration.
