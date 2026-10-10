# R4 fresh-attack claims handoff

Status: **development point results; independent numerical replay and simultaneous recovery intervals pending**. These are reused INNER_SELECTION people. The three cached draws per stochastic arm measure mechanism variation, not three populations. No old assessment or new confirmation population was opened.

## What was actually run

All eight locked arms passed the original INNER utility point gate (24/24 clauses per arm over two seeds). R4 then fitted independent final attackers to all 40 fixed arm/seed/draw release instances: P0 and V_D had one deterministic instance per seed; O_J, V_J, V_P, V_L, V_S12, and V_S21 had three cached instances per seed. Every instance had recipient 1, recipient 2, and aligned coalition complete views. Each view had eight fitted readers, including nonlinear, categorical, probability-only, code-bit, and public-policy-aware readers; the coalition also had ignore-one-recipient aliases. Each view had two fitted shuffled-label/planted-leak controls and an exact null check. This yields 120 complete views and 1,200 fitted reader/control models. All 40 units and all view controls completed and passed. R4 fit SEX readers only on ATTACK_FIT_NEW and opened INNER_SELECTION SEX only after reader freeze. R4 did not read source task labels or old assessment rows.

The actual wire retained original P0 global code IDs, decoded probability, decision, and fixed public metadata. The public selected policy and a prospectively declared uniform-original-code-pair derived feature were available to defense-aware readers. This feature is a data-free prior over all P0 input-code pairs, not a claim about the private provider input distribution. No score used an averaged stochastic output or a repeated-query view.

## Point results

Equal-seed, draw-mean coalition SEX AUC from the fixed reader envelope:

| Arm | Pair AUC | P0 minus arm |
| --- | ---: | ---: |
| P0 | 0.803141 | 0 |
| O_J | 0.802718 | +0.000423 |
| V_J | 0.801862 | +0.001279 |
| V_P | 0.801749 | +0.001392 |
| V_D | 0.803141 | 0 |
| V_L | 0.802506 | +0.000635 |
| V_S12 | 0.802170 | +0.000970 |
| V_S21 | 0.800789 | +0.002351 |

The most favorable observed point contrast is P0 minus V_S21 = **0.0023514542 AUC**, far below the prospective +0.020 coalition target. V_S21 has the lowest equal-seed draw-mean pair AUC in the fixed arms. The FIT-only nominee V_L has pair AUC **0.0017162975 higher** than V_S21; joint nominee V_J is **0.0010728958 higher**. Thus these point results supply no competitive or joint-design advantage. Per-seed and per-view values, including adverse recipient shifts, are in the point closeout and 120-row table. R3's paired group × draw simultaneous intervals and R5's independent attack replay determine the final clause statuses.

V_D's actual emitted recipient JSONL wire hashes equal P0 for both ATT/INNER roles, both seeds, and both recipients. V_D still has separately pinned policy receipts and separately fitted R4 attackers. Its equal AUC is an exact-wire alias check, not a privacy gain.

The historical A_CLASS_ONLY predicted-label/decision-only diagnostic reported pair AUC 0.736103/0.734125 and failed confidence utility. Its old T* bank source bindings differ from the current P0 bank. It is historical context only, not a matched current control or a proven current decision floor. A true-task-label-only output is not an admitted inference-time release in this campaign. The corrected source record is `HISTORICAL_DECISION_REFERENCE_V2.json`; its V1 wording error and dated correction remain preserved.

## Claim boundary for the paper

The fixed FIT-only rule nominated V_L before final attack outcomes. Forty attacks and several arm comparisons were then made on reused development people. These point values can report an exploratory negative result for this bank-anchored design; they cannot be presented as independent confirmation or as a general impossibility theorem. A structural component earns attribution only through its fixed matched arm: V_D is exact identity; V_P, V_L, both sequential orders, and joint arms are separately scored. Current points do not isolate a benefit from joint design, randomness, or the public-policy-aware feature.

## Exact source pins

- Campaign science lock `r0/CAMPAIGN_LOCK.json`: `237feaf85dd19c497c5f377ab0f7fffab72bc41cb8d2c277b21976d6c0cd4ae6`.
- Final attack lock `r0/ATTACK_STAGE_LOCK.json`: `2fd5596e3d399f02efcb867fc8853f945aca3a595223a608a82fb303399e50a1`.
- FIT-only nomination `r3/R3_INTERNAL_READER_NOMINATION_V1.json`: `e2f558401c80837e55fdd8c5b2a14e9bb942bfc9b324c169e41c0b41f62399f0`.
- R4 prepared index `r4/ATTACK_PREPARED_INDEX.json`: `20727ee40db5125773a76dc36622af7aaaf6f7c6f3ddca12b77affa3153442cb`.
- R4 effective map `r4/ATTACK_EFFECTIVE_MAP.json`: `98b4dd6ad202a9f1c236547276f28fa0b53339a61c7d802aed3413bccc2d410f`.
- R4 attack aggregate `r4/ATTACK_AGGREGATE.json`: `a0090e8b0d61a7194e6f71da5a61709de1efb6d4c09be71745b7cd8b59820340`.
- R4 complete AUC table `r4/ATTACK_AUC_TABLE.csv`: `995d4428286a4ba1b35e36fef6e6b60502a9d4957704d181ea613e4923cacc36`.
- R4 point closeout `r4/ATTACK_POINT_CLOSEOUT.json`: `8b04235c81ee9167ca1a85ff6e2b21578ac6eda8fbdbbfbc61f2e88c23ad2097`.
- Historical diagnostic correction `r4/HISTORICAL_DECISION_REFERENCE_V2.json`: `e767fbd9b00a0d112f120af4e94dd50d7fce251306f5bbed5c8100808a0e737f`.
- Independent prefit R5 queue audit `r5/ATTACK_QUEUE_STATIC_AUDIT.json`: `1c6ac35eb970c9218a18ae6d0e5257d10d40e38b3291db38e5b3e44a2697b593`.

## Commands used and runnable entrypoints

Run from the incumbent-view worktree with `PCRL_VENV_PYTHON` pointing to the repository venv **entrypoint** (do not dereference its Python symlink). These are the argument forms used locally; owner-only plan and queue specs contain private paths and remain under `r4/.private/`.

```sh
"$PCRL_VENV_PYTHON" -m incumbent_view.r4_build_specs prepare --config results/pcrl_incumbent_view_risk_v1/r4/.private/ATTACK_OWNER_PLAN.json
"$PCRL_VENV_PYTHON" -m incumbent_view.r4_build_specs finalize --index results/pcrl_incumbent_view_risk_v1/r4/ATTACK_PREPARED_INDEX.json --stage-lock results/pcrl_incumbent_view_risk_v1/r0/ATTACK_STAGE_LOCK.json --stage-lock-sha256 2fd5596e3d399f02efcb867fc8853f945aca3a595223a608a82fb303399e50a1 --repo "$PCRL_WORKTREE" --python "$PCRL_VENV_PYTHON"
"$PCRL_VENV_PYTHON" -m incumbent_view.r4_aggregate --repo "$PCRL_WORKTREE" --prepared-index results/pcrl_incumbent_view_risk_v1/r4/ATTACK_PREPARED_INDEX.json --prepared-sha256 20727ee40db5125773a76dc36622af7aaaf6f7c6f3ddca12b77affa3153442cb --effective-map results/pcrl_incumbent_view_risk_v1/r4/ATTACK_EFFECTIVE_MAP.json --effective-sha256 98b4dd6ad202a9f1c236547276f28fa0b53339a61c7d802aed3413bccc2d410f --output-json results/pcrl_incumbent_view_risk_v1/r4/ATTACK_AGGREGATE.json --output-csv results/pcrl_incumbent_view_risk_v1/r4/ATTACK_AUC_TABLE.csv
```

The 40 `incumbent_view.r4_worker` invocations were run by the shared `r0.compute_queue` from the owner-only specs indexed by `r4/ATTACK_QUEUE_INDEX.json`; this handoff does not claim they were launched as standalone shells. All listed outputs are write-once and already exist. A replay should use a restored copy or fresh campaign namespace, rather than overwrite them.
