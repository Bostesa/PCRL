# Mathematical and implementation verifier: files and units (Phase 1)

**Role.** The independent mathematical and implementation verifier, following `roles/coordinator/VERIFIER_BRIEF.md`.

**Data handling.**
- Everything here is synthetic, or computed from published aggregate tables (EVIDENCE_PACK table A, `STAGE_A_HEADROOM.json`, and the published reader-slate matrices in the source and frozen-hybrid aggregates).
- No person-level arrays were opened. Nothing under `.private/` was read except this role's own unit logs (`.private/mechanism_design/compute/units/V_*`).
- The coordinator's `mechanism_design` code was not imported or copied. Only `mechanism_design.jobqueue` (the queue runner) and `mechanism_design.no_real_data` (the guard) were used.
- The designers' laws were re-entered from their proposals and scripts, and every quantity was recomputed with separate code.

**Proposals assessed** (hashes checked before reading):

| Designer | Proposal | SHA-256 |
|---|---|---|
| Designer 1 | MMP | `2b7273ae01d6449036f25c7a6f5e74efb8d2843f1251cbceb8e4c7024c9ff368` |
| Designer 2 | RCCS | `83653f2db759bb8339edd96984ffa89683dc2f84478d37304b489894f4009f21` |

## Files

| File | What it is |
|---|---|
| `MATH_ASSESSMENT.md` | The Phase 1 assessment: one section per proposal, plus a comparison |
| `V_MMP_EXACT_V1.json` | Output of unit `V_MMP_EXACT_V1` |
| `V_RCCS_GAUSS_V1.json` | Output of unit `V_RCCS_GAUSS_V1` |
| `V_RCCS_FINITE_V1.json` | Output of unit `V_RCCS_FINITE_V1` |
| `V_RCCS_MAP_V1.json` | Output of unit `V_RCCS_MAP_V1` |
| `V_RCCS_MAP2_V1.json` | Output of unit `V_RCCS_MAP2_V1` |
| `PROTOTYPE_VERIFICATION.md` | Phase 2: independent recomputation of the MMP prototype, with registered predictions P1–P7 and verdicts on the coordinator's three failed checks |
| `PROTOTYPE_VERIFICATION.json` | Phase 2 key recomputed numbers and check verdicts |
| `V_PROTO_FINITE_V1.json`, `V_PROTO_CONT_V1.json`, `V_PROTO_HD_V1.json`, `V_PROTO_HD_V2.json`, `V_PROTO_HD_ALIGN_V1.json` | Phase 2 unit outputs (`V_PROTO_CONT_V1.json` is written by unit `V_PROTO_CONT_V2`) |

Code lives in `mechanism_design_independent/` at the worktree root:

| Module | Role |
|---|---|
| `__init__.py` | Package marker |
| `exact.py` | Own exact finite-law toolkit. Bayes AUC by LR-sorted ROC with ties counted ½; task metrics matching `acs_replication/data.py`; within-SEX Brier; set partitions by restricted-growth strings |
| `v_mmp.py` | Designer-1 checks |
| `v_rccs_gauss.py` | Designer-2 linear algebra, Gaussian model and published-AUC readings |
| `v_rccs_finite.py` | Designer-2 exact finite laws |
| `v_rccs_map.py` | Synthetic Monte Carlo of the literal RCCS map against linear, quadratic, HGB and MLP readers |
| `v_rccs_map2.py` | RCCS vs MIN with the same map family and budget: the analogue of falsifier 1 |
| `v_proto_finite.py`, `v_proto_cont.py`, `v_proto_hd.py`, `v_proto_hd2.py`, `v_proto_hd_align.py`, `v_proto_summary.py` | Phase 2: the six finite laws (exact); the two continuous laws (own sampler, readers and quadrature); the high-dimensional probe P7, its strong-reader check and its alignment diagnostic; the summary assembly |

## Units

All units ran through the campaign queue with stage `verifier`, using:

`PYTHONPATH=<worktree> ~/PCRL/.venv/bin/python -m mechanism_design.jobqueue run --unit <ID> --stage verifier --rss-gib <r> -- ~/PCRL/.venv/bin/python -m mechanism_design_independent.<module>`

| Unit | Module | What it checks |
|---|---|---|
| `V_MMP_EXACT_V1` | `v_mmp` | Recomputes laws N, F and R: AUC\*, identities, NLL bound, exact frontier with local, S12 and S21 controls. Also: AUC\*(S;H) in law R; the within-SEX Brier formula on 300 random constant-gap laws; a non-constant-gap counterexample; a law where MMP *raises* pair Bayes AUC; the Brier-optimal mixing weight |
| `V_RCCS_GAUSS_V1` | `v_rccs_gauss` | The amplification identity and three counter-shift cost formulas on random instances. The designer's Gaussian model under four cost/leak conventions, plus the LOCAL replay. Published-AUC readings for both seeds, with sampling spread, implied R², share decompositions and reader-class sensitivity |
| `V_RCCS_FINITE_V1` | `v_rccs_finite` | Designer 2's four 8-cell laws, exhaustive over 4,140 partitions, under the designer's Brier-only budgets and under the binding three-clause budgets |
| `V_RCCS_MAP_V1` | `v_rccs_map` | The literal RCCS map with an oracle propensity, in a smooth synthetic law: criterion-class readers vs HGB/MLP, at 8,000 and 100,000 training rows |
| `V_RCCS_MAP2_V1` | `v_rccs_map2` | RCCS vs MIN at matched budget B = 0.009; pair and occupation-only views; linear, HGB and MLP readers |
| `V_PROTO_FINITE_V1` | `v_proto_finite` | Phase 2: recomputes the six finite prototype laws exactly and compares them with `PROTOTYPE_RESULTS.json` |
| `V_PROTO_CONT_V1` | `v_proto_cont` | **failed**: a key-name bug in my own script |
| `V_PROTO_CONT_V2` | `v_proto_cont` | The two continuous laws: population quadrature, fitted LR/HGB/MLP, the matched and full erasers |
| `V_PROTO_HD_V1` | `v_proto_hd` | Post-hoc probe P7 (registered before the run): 10 fields, no pure-proxy field |
| `V_PROTO_HD_V2` | `v_proto_hd2` | P7 robustness check: strong HGB readers on 1M rows |
| `V_PROTO_HD_ALIGN_V1` | `v_proto_hd_align` | P7 diagnostic: field-mediated vs direct SEX effects |
| `V_PROTO_SUMMARY_V1` | `v_proto_summary` | Assembles `PROTOTYPE_VERIFICATION.json` |

Unit logs are in `.private/mechanism_design/compute/units/V_*/`, and the CPU each unit used is recorded in the campaign ledger.
