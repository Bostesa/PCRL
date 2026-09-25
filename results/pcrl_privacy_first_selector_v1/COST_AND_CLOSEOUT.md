# Cost, archive and closeout

## Time (UTC)

| Event | Time |
|---|---|
| Registration commit `5f91117` | 2026-09-24 23:34:02 |
| Launch commit `9bb77ac` | 23:43:55 |
| Instance `i-07d9bc2619c9f5b11` launched | 23:44:19 |
| Basis manifest (before any solve) | 23:49:33 |
| Amendment A1 `bf040be`, then re-solve | ~23:51 |
| Amendment A2 `5576bea` | ~23:58 |
| Inner audits complete | 23:58:57 |
| Lock `7fcfaf1` pushed and remote-verified | 2026-09-25 00:02:15 |
| Outer unlock | 00:02:45 |
| Outer originals restored | 00:03:09 |
| Outer scored and assessment written | ~00:04–00:07 |
| Independent verification (Parts B and C, on the host) | ~00:12 |
| Termination requested / reached | 00:14:06 / 00:14:35 |
| Study security group `sg-07748c7024ec577d4` deleted | 00:14:36 |

## Money

| Item | Upper estimate |
|---|---|
| c7i.8xlarge on-demand at $1.428/h, 0.50 instance-hours | ≤ $0.71 |
| 120 GiB gp3 root volume for the same interval | ≈ $0.01 |
| S3: 213.8 MB across all versions under `pcrl_privacy_first_selector_v1/` | ≈ $0.005/month |
| **Total** | **≈ $0.72**, against ceilings of $50 and 20 instance-hours |

- These are the event-based ledger figures (`cloud cost --record`), not an AWS invoice.
- Unrelated running instance `p0-pilot` (i-0e507056c6231b717) was seen and not touched.

## Retained private archives

Location: `s3://pcrl-ux-archive-ed9d21fd/pcrl_privacy_first_selector_v1/`. The bucket is versioned, uses SSE-AES256 and has no lifecycle expiry. Every archive was read back and SHA-256-verified.

| Key (archive/…) | Version | SHA-256 | Bytes |
|---|---|---|---|
| units.tar.gz | Vro.AnmMB373x.e9AN1ktNNH_lIntx4w | ae0def2cf48cf5a8841254778f88cdc39b225fbad06eb0e1a025cb80b72b86c1 | 19,745 |
| inner_panels.tar.gz | S.pwXU4zpn.WcLIat66sdLhtwWWxFK.y | 6f000aa79dfc90a657e7b8bce97e5c4a43993dba9324964219748556c03004db | 195,489,173 |
| superseded_solve_A1.tar.gz | DZM1WloRgnX5tpCtO_8bcQ30aKGPEjP2 | 9cca3bb720aa8c6b72dfbde21308595a12b427055e0ce87cc218d87db6f565fd | 20,212 |
| outer_scores.tar.gz | Ul6G42s1fiUttP9V8DFkcLFFJQK_NJym | 9c11eb7e5c788d1f7378afc2cdbeaa913c2fec18dad27ffb69dbb46e9921ef17 | 10,005,879 |
| gate_receipts.tar.gz (unlock, original restore, prior restore) | U0ofok4uQrBzOPsFSOLrgCsQK6wLhkEr | 247c5e345ffac031f4b3c5546a8f92e2a03d77809d435789499aecb898b3bf6a | 2,058 |

- Host logs are under `control/logs/`.
- Deliberately not re-archived:
  - the predecessor unit archives, which are pinned in the SC `MODEL_MANIFEST.json`;
  - the outer-labelled originals, which are pinned in AR's input store.

## Local

- The Mac scratchpad holds label-free coefficient blocks for verifier Part A and synthetic fixtures. No person-level outer data was copied to the Mac.

## Shutdown checklist

- [x] Study instance terminated (tag-checked, `Study=pcrl_privacy_first_selector_v1`). Its root volume was deleted on termination.
- [x] Study security group deleted. A fresh query shows 0 study instances, 0 study volumes and 0 study security groups.
- [x] No EventBridge rules or schedules were created. The host timers ended with the instance.
