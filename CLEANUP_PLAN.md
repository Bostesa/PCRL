# PCRL repo cleanup plan — anonymous NeurIPS 2026 submission

Phase-1 audit only. **No file was modified outside this document.** Wait for explicit "approved" before Phase 2.

---

## 1.1 Current structure

```
.                                       (605 tracked files; 4 at root)
├── .gitignore
├── README.md                           Reproduction guide for the headline numbers
├── LAUNCH_README.md                    AWS launch guide for BIOS_LAYER12 — DEANONYMIZING (S3 bucket, AMI, IAM profile)
├── requirements.txt                    Unpinned deps (>=) — fine for Phase 2 review
├── launch_bios_layer12.sh              UNTRACKED, gitignored. Personal AWS launcher. Leave as-is.
├── pcrl/                               Core package: data, evaluation, language (BIOS), models, purposes, training, utils, vision (CelebA), baselines
├── experiments/                        ~80 entry-point scripts (training, evaluation, probes)
├── scripts/                            ~35 analysis + table-builder + paper-paste utilities
├── tests/                              16 unit tests
├── configs/                            adult.yaml, adult_3purpose.yaml, bios_pcrl_layer12/{P1,P2,P3}.yaml
├── data/                               Gitignored; only data/README.md tracked. Contains UCI HAR, Adult, HMDA, Diabetes, CelebA, folktables locally
├── checkpoints/                        Gitignored; ~80 local run dirs, none tracked
├── logs/                               Gitignored; nothing tracked
├── infra/                              SHOULD be gitignored but isn't — `infra/inlp/` and `infra/splince/` were committed before the gitignore rule. Both contain DEANONYMIZING AWS state (bucket names, AMI, IAM profile, GitHub clone URL pinning Bostesa/PCRL)
├── figures/                            PDF figures referenced by the paper
├── paper-body/tables/                  LaTeX tables shipped with the paper (11 .tex + 1 .md)
└── results/                            ~120 sub-directories of experiment artifacts (JSON, CSV, MD, .tex, a few .pt + .png)
```

Top-level tracked files (4): `.gitignore`, `README.md`, `LAUNCH_README.md`, `requirements.txt`. **No LICENSE.** **No setup.py / pyproject.toml.** No notebooks anywhere in the repo (`*.ipynb` count = 0 — clean).

UNKNOWN — needs human review:
- `infra/laftr/` is referenced by recent commits (`ade5cd9 Add Stage 2/3 worker queue runner (infra/laftr/run_worker.sh)`) but the file is no longer in tree (must have been deleted). Confirm intent.
- `paper-body/` directory name is non-standard. Likely just a holdover; suggest rename to `tables/` in Phase 2 or leave.

---

## 1.2 PII findings

**Summary:** the repo's source code is largely clean (the prior anonymization commit `135e440` did most of the work). The remaining problems are concentrated in (a) AWS launch infrastructure that was tracked by accident, (b) absolute warm-start paths embedded in result JSONs/CSVs, and (c) the git history itself, which still attributes every commit to `Bostesa <natestesa@gmail.com>` and contains the deanonymizing GitHub clone URL.

### Hits by category

| # | Category | Tracked-file hits | Notes |
|---|---|---:|---|
| A | Author name (Nathan/Samson/nsamson/nathans) | **0** in code; 4 in result JSON paths only | author home dir appeared in 4 metrics.json (smoke runs). Listed under F. |
| B | Advisor (Yus/Roberto/ryus) | **1** | `scripts/build_splince_vs_pcrl.py:226` — comment `# Paper-paste paragraph (Dr. Yus's flowing-prose voice)` |
| C | Lab member (Christian/Badolato) | 0 | clean |
| D | Institution (UMBC/DAMS) | 0 | clean |
| E | Email addresses (regex) | **0** in tracked code | git author email is `natestesa@gmail.com` — see §git history below |
| F | Personal absolute paths (`/Users/`, `/home/<user>`, `C:\Users\`) | **47+ in tracked files** | See breakdown below |
| G | AWS instance IDs (`i-[0-9a-f]{8,17}`) | **9 hits across 4 tracked files** | `LAUNCH_README.md`, `infra/inlp/launch.sh`, `infra/splince/launch.sh`, `results/bios_layer12/SHUTDOWN.txt`, `results/paper_critique_responses/PAPER_PASTE.md` |
| H | AWS keys (`AKIA…`, `aws_access_key_id`, `aws_secret`) | 0 | clean |
| I | GCP / W&B / `entity=` / `WANDB_ENTITY` / `GOOGLE_APPLICATION_CREDENTIALS` | 0 | clean |
| J | Slack/Zulip/Discord/Zoom/Notion URLs | 0 | clean |
| K | Google Docs/Drive personal links | 0 | clean |
| L | Acknowledgments naming people | 0 | one false positive: `experiments/run_bios.py:1129` has `--acknowledge-…` argparse help, no name |
| M | TODO/FIXME/XXX/HACK | 0 in tracked code | clean |
| N | SSH/PGP/private-key blobs | 0 | clean |

### F. Personal absolute paths — full list (deduplicated by file)

`${REPO_ROOT}/...` (4 files, all smoke-test outputs, **highest priority — these are deanonymizing**):
- `results/v2_pcrl_variance_constrained/_smoke/adult_s0/metrics.json:6`
- `results/v2_perdim_lagrangian/_smoke/hmda_s0/metrics.json:6`
- `results/splince_benchmark_smoke/adult_s0/metrics.json:4`
- `results/splince_benchmark_smoke/splince_results.json:9`

`${REPO_ROOT}/...` (43 files, **AWS default username — generic but still revealing that runs were on EC2 + reveals checkpoint layout**):
- 33 in `results/splince_benchmark/**/metrics.json` and `splince_results.json`
- 10 in `results/v2_pcrl_variance_constrained/**/metrics.json` and `results.json`
- 4 in `results/v2_perdim_lagrangian/**/metrics.json` and `results.json`
- 2 in `results/adult/{laftr,pcrl}_sweep.csv` (path columns)
- 1 in `results/har_real/pcrl_sweep.csv`
- 2 in `infra/inlp/user_data.sh`, `infra/splince/user_data.sh` (intentional — set $HOME on EC2)
- 2 in `scripts/crosspurp/{run_eval,build_report}.py` (sys.path fallback for a script intended to run on EC2)
- 1 in `experiments/run_leopard_overnight.py` (`LEOPARD_PATH` env-var default)

### G. AWS instance IDs + S3 buckets — full list

| File | Content |
|---|---|
| `LAUNCH_README.md:52` | `AMI_ID = ami-012ba162b9cd2729c`; `IAM_INSTANCE_PROFILE = pcrl-overnight`; bucket `pcrl-bios-layer12-20260505` |
| `infra/inlp/launch.sh` | bucket `pcrl-bios-overnight-20260504`, AMI `ami-05603a42e5254c4bb`, KEY `pcrl-gpu-key`, IAM `pcrl-bios-s3-writer`, REGION `us-east-1`, **`https://github.com/Bostesa/PCRL.git`** |
| `infra/inlp/user_data.sh` | bucket `pcrl-bios-overnight-20260504`, **`git clone https://github.com/Bostesa/PCRL.git`** |
| `infra/splince/launch.sh` | bucket `pcrl-bios-overnight-20260504`, AMI `ami-05603a42e5254c4bb`, KEY `pcrl-gpu-key`, IAM `pcrl-bios-s3-writer`, **`https://github.com/Bostesa/PCRL.git`** |
| `infra/splince/user_data.sh` | bucket `pcrl-bios-overnight-20260504`, **`git clone https://github.com/Bostesa/PCRL.git`** |
| `results/bios_layer12/SHUTDOWN.txt` | 5 instance IDs: `i-03248b45c2fa0a23c`, `i-0d6e6023900eecfa1`, `i-0056a26214f415fd4`, `i-0f1c52accd564b488`, `i-0d73cf44812a22f0a`, `i-08710bf92faed9b81` |
| `results/paper_critique_responses/PAPER_PASTE.md:418` | `i-08710bf92faed9b81` |

**Critical: 5 places in tracked files contain `https://github.com/Bostesa/PCRL.git` — single biggest deanonymization risk.** The `infra/{inlp,splince}/` files were committed before `infra/` was added to .gitignore.

### git history concerns (do **not** rewrite without explicit user request)

- All recent commits attribute author `Bostesa <natestesa@gmail.com>`. `git log` exposes this whether or not the file tree is anonymized.
- Commit messages reference real instance IDs and S3 buckets (e.g. `4f64fdf BIOS Round 2 launch #5 — VERDICT…`).
- The 4open.science anonymization mirror normally hides commit metadata; for a vanilla GitHub link, the author/email and messages would deanonymize.
- **Recommendation:** flag in CLEANUP_REPORT.md, tell the user `git filter-repo` is the only fix and is destructive; do not run it ourselves. Confirm 4open.science's redaction is sufficient before push.

---

## 1.3 Paper-claim → artifact mapping

19 of 20 claims have a backing artifact in the tree. **#11 (Cross-purpose Prop 6 bound utilization)** is the only gap, and the generating script + JSON exist in `/tmp/tier2_prop6/` — a 60-second copy-into-results away from being recoverable.

| # | Claim | Status | Path |
|---|---|---|---|
| 1 | 60-cell compliance grid (Table 1) | **present** | `results/v2_adult_ROUND5/{per_seed_results,summary,dominant_axis_audit}.json` + same for `v2_hmda_ROUND5/`, `v2_diabetes_ROUND7/` |
| 2 | 56/60 strict pass headline | derivable from #1 | — |
| 3 | 7/60 cleanly compliant | derivable from #1 | `results/v2_adult_ROUND5/collapse_diagnostic.json` and analogues |
| 4 | 49/56 collapse-compliant decomp | derivable from #1 + #3 | — |
| 5 | τ ∈ {0.01, 0.025, 0.05} sweep | **present** | `results/tier1_analyses/tau_sensitivity.json` + `tau_sensitivity_table.tex` + `tau_sensitivity_plot.pdf` |
| 6 | Round-4 baseline (46/60, mean R²=0.038) | **present** | `results/v2_adult_ROUND4/`, `v2_hmda_ROUND4/`, `v2_diabetes_ROUND4/` (per_seed + summary + final_vs_best) |
| 7 | Diabetes c.d.s./gender Zhao-Gordon slack 0.0011 | **present** | `results/reviewer_dropins/zhao_gordon_table.{csv,json}` (20 cells) — generated by `scripts/zhao_gordon_certificate.py` (untracked, needs commit) |
| 8 | Convex-Combo Identity 33/33 | **present** | same `dominant_axis_audit.json` files as #1 (column `convex_combo_residual`) |
| 9 | HMDA underwriting/race amplification 12.7×, 10.7×, 3.2× | **present** | same `dominant_axis_audit.json` files as #1 |
| 10 | Cross-purpose 26/33 (A) and 22/33 (B) | **present** | `results/cross_purpose_laftr/DUAL_CRITERIA.json` + `results/v2_cross_purpose/aggregate.json` + `per_seed_results.json` |
| 11 | Cross-purpose Prop 6 bound utilization 45–70% | **partial** | Backing JSON `numerical_verification_v2.json` and verifier `verify_bound_v2.py` live in `/tmp/tier2_prop6/`. Move into `results/reviewer_dropins/` and commit. |
| 12 | Adult held-out validation seed 3 (6/8, +0.0111 mean Δ) | **present** | `results/v2_adult_HELDOUT_S3/{per_seed_results,summary,dominant_axis_audit}.json` |
| 13 | CelebA train R²≤0.005 + Smiling acc 0.752 ± 0.010 | **present** | `results/v2_celeba_R5_FULL/{train_set_r2,cross_purpose,training_log}.json` + `results/celeba/celeba_main_table.csv` + `paper-body/tables/celeba_final_multiseed.tex` |
| 14 | CelebA 6-architecture audit (worst-case −6.3pp Male, −8.2pp Young) | **present** | `results/celeba/{baseline_v2_fixed,celeba_perpair,concat_perpair,laftr_per_purpose,pcrl_ensemble_perpair,pcrl_vs_laftr_ensemble}.csv` + `paper-body/tables/celeba_cross_purpose_multiseed.tex` |
| 15 | SPLINCE comparison (3/60 health, 60/60 R²-only, 2/60 +Δ_aud) | **present** | `results/splince_benchmark/{splince_results,splince_vs_pcrl_summary}.json` + per-cell `metrics.json` |
| 16 | Compute cost (train time, params, latency) | **present** | `results/tier1_analyses/{compute_cost.json, compute_cost_table.tex, PAPER_PASTE_compute.md}` |
| 17 | Figure 2 drift trajectory (Adult income/race s2 vs income/sex s2) | **present** | `results/v2_drift_trajectories/{data.json, drift_chart.png, trajectories.csv}` + `experiments/analyze_drift.py` |
| 18 | LAFTR baseline 25% pass under PCRL criterion | **present** | `results/laftr_benchmark/STAGE2_ADULT.md` + `STAGE2_ADULT_TABLE.tex` + `adult/` per-cell |
| 19 | Per-purpose INLP baseline 72% pass | **present** | `results/inlp_benchmark/{inlp_results.json, PAPER_PASTE.md, pcrl_vs_laftr_vs_inlp_table.tex}` + per-dataset subdirs |
| 20 | Per-class OvR Diabetes age_bucket (rank 24, K=10) | **present** | `results/v2_diabetes_OVR_DIABETES_PROBE/{per_seed_results, summary}.json` |

**Coverage:** 19/20 fully present, 1/20 partial-but-recoverable. Phase 2 step 8 will move #11 into the repo.

---

## 1.4 Repo hygiene findings

- **Files >50MB:** all are gitignored already. None tracked. The on-disk hits are local-only (`checkpoints/celeba_*.pt`, `data/hmda_raw/hmda_2023_ca.csv`, `data/folktables/2018/1-Year/psam_p06.csv`, `data/UCI HAR Dataset/train/X_train.txt`).
- **Hardcoded absolute paths:** see §1.2 F. 47+ hits, mostly `${REPO_ROOT}/...` in result JSONs.
- **Hardcoded API keys / tokens / credentials:** **none beyond the AWS context already in §1.2.G**.
- **Dead Python files:** none confirmed dead via static inspection in this audit. Each top-level script in `experiments/` and `scripts/` is referenced by README, by infra user-data, or by another script. Defer "dead-code" pruning unless explicitly requested.
- **`.DS_Store` / `Thumbs.db` / `__MACOSX`:** none in tracked set. `data/__MACOSX/` exists on disk but is gitignored. `.gitignore` already covers `.DS_Store` and `__MACOSX/`.
- **IDE files:** `.gitignore` already covers `.idea/`, `.vscode/`. `.claude/` is gitignored. No personal IDE config tracked.
- **TODO/FIXME/XXX/HACK in tracked code:** 0 hits.
- **`requirements.txt` versions:** all unpinned (`>=`). Acceptable for a review repo, but consider pinning major versions for reproducibility (e.g. `torch>=2.0,<3`).
- **Notebooks (`*.ipynb`):** 0 in repo. No `nbstripout` step needed in Phase 2.
- **`infra/` is gitignored but partially tracked.** `infra/inlp/`, `infra/splince/` were committed before the gitignore rule. Phase 2 should `git rm` them.
- **`logs/heldout_s3/` exists on disk and is gitignored — fine.**
- **README cites `experiments/run_v2_dataset.py`** (line 44 of README). Verify that file actually exists; it isn't in the listing I did. Possibly the entry-point is named differently — flag for Phase 2.
- **Untracked work in progress (38 paths):** several reviewer-dropin docs, the new Tier-1/2/3 scripts (`compute_pareto_certificate.py`, `sadeghi_boddeti_sanity.py`, `zhao_gordon_certificate.py`, `theorem5_lambda_star.py`, `tier1_*.py`, `varconstraint_diagnostics.py`, `heldout_s3_compare.py`), and several results/ dirs that back paper claims. **All are paper-supporting and should be committed in Phase 2 step 1, after the PII pass.**

---

## 1.5 Proposed structure

The current layout is close to the proposed default; the main edits are (a) untrack `infra/`, (b) add `LICENSE` and `docs/`, (c) optionally drop the `paper-body/` rename. I am **not** proposing to merge `experiments/` into `scripts/` — they have a meaningful split (entry-points vs. analysis utilities) and the README documents it. Heavy reorg risks breaking the dozens of import paths across `experiments/run_*.py` files for marginal cleanliness gain.

```
.
├── README.md                           rewritten per §1.6
├── LICENSE                             ADD (MIT default — confirm in Phase 2 step 9)
├── requirements.txt
├── .gitignore                          extend to cover infra/, LAUNCH_README.md, launch_bios_layer12.sh
├── pcrl/                               unchanged
├── experiments/                        unchanged (training/eval entry-points)
├── scripts/                            unchanged (analysis + table builders); commit the ~10 untracked Tier-1/2/3 scripts after PII scrub
├── tests/                              unchanged
├── configs/                            unchanged
├── data/                               unchanged (only data/README.md tracked)
├── results/                            unchanged but PII-scrubbed; add the 9 untracked subdirs that back paper claims (zhao_gordon, splince_benchmark, inlp_benchmark, cross_purpose_laftr, paper_critique_responses, reviewer_dropins, tier1_analyses, theorem5_check, v2_adult_HELDOUT_S3, v2_pcrl_variance_constrained, v2_perdim_lagrangian)
├── paper-body/tables/                  unchanged (LaTeX tables shipped with paper)
├── figures/                            unchanged
└── docs/                               ADD
    ├── REPRODUCIBILITY.md              one-command-per-claim reproduction guide (consolidates README "Reproducing" section)
    └── DATA.md                         absorb data/README.md and add license notes per dataset
```

**Proposed `git rm` (untrack but keep on disk via `git rm --cached`):**
- `LAUNCH_README.md` → personal AWS launch guide. Replace with a non-deanonymizing README about how to run on a fresh GPU (no S3 / no IAM specifics).
- `infra/inlp/launch.sh`, `infra/inlp/user_data.sh`, `infra/inlp/aggregate.py`
- `infra/splince/launch.sh`, `infra/splince/user_data.sh`
  (All four launch/user_data scripts contain the deanonymizing GitHub URL + bucket name. `aggregate.py` is generic — re-add to `scripts/` after grep-scrub.)
- `results/bios_layer12/SHUTDOWN.txt` (pure operational log, contains 5 instance IDs)
- `results/bios_layer12/NEGATIVE_RESULT_NOTE.md` — keep but scrub instance IDs and bucket names if any.

**Files to delete outright:** none proposed. Every other tracked file has a paper-traceable purpose.

---

## 1.6 README assessment

The current `README.md` (172 lines) covers **most** of the NeurIPS-README checklist. The shortfalls are: missing LICENSE callout, no consolidated headline-results table, no claim-to-artifact map, citation block (anonymous placeholder) absent, and one stale claim (`22/33` should be `26/33` or rephrased — see prior conversation thread).

| Item | Current README | Action |
|---|---|---|
| Title matching the paper | ✓ "PCRL: Purpose-Conditioned Representation Learning" | keep |
| One-paragraph what-this-does | ✓ lines 3–8 | keep |
| Headline results table | ✗ — numbers scattered through prose | **add** a 1-table summary at the top (56/60, 7/60, 26/33, 22/33, etc.), pulled from the actual JSON/CSV |
| Repo structure tree | ✓ lines 17–28 | keep, update for `docs/` and removal of `infra/`, `LAUNCH_README.md` |
| Setup (3-5 commands max) | partial — only `requirements.txt` install implied | **add** explicit `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |
| One-command reproduction of the headline | ✓ "Strict pass at τ=0.05" section gives 3 commands | keep, but verify `experiments/run_v2_dataset.py` exists (not visible in listing) |
| Paper-claim → artifact mapping | ✗ | **add** the 20-row table from §1.3 in compact form |
| Citation block (anonymous placeholder) | ✗ | **add** a stub `@inproceedings{anonymous2026pcrl, ...}` that cites the anonymous title and the OpenReview/4open.science URL placeholder |
| License | ✗ | **add** "Released under the MIT License — see `LICENSE`." once Phase 2 writes the LICENSE file |
| Cross-purpose number | currently says `22/33 flagged amplifications` (line 76) | reconcile with §5.5 of paper — change to `26/33 above majority +1pp` (Criterion A) per the abstract fix already applied to `results/reviewer_dropins/PAPER_PASTE.md` |

---

## Stop. Awaiting approval to proceed to Phase 2.

When you reply **"approved"** I will execute the following commits in order, **without pushing**:

1. `.gitignore` — add `infra/`, `LAUNCH_README.md`, `launch_bios_layer12.sh`, `wandb/`, `outputs/`, `lightning_logs/`, `data/raw/`, `*.ckpt`, `*.pt > 50MB` (keep existing rules).
2. `git rm --cached` — `LAUNCH_README.md`, `infra/inlp/{launch.sh,user_data.sh,aggregate.py}`, `infra/splince/{launch.sh,user_data.sh}`, `results/bios_layer12/SHUTDOWN.txt`. Move `infra/inlp/aggregate.py` to `scripts/inlp_aggregate.py` after grep-scrub.
3. PII scrub — sed-replace `${REPO_ROOT}/` and `${REPO_ROOT}/` → `${REPO_ROOT}/` in result JSONs/CSVs (47+ files); strip `Dr. Yus`-named comment in `scripts/build_splince_vs_pcrl.py`; strip instance ID from `results/paper_critique_responses/PAPER_PASTE.md:418`.
4. Notebook scrub — N/A (no notebooks).
5. Rewrite `README.md` per §1.6 with real numbers pulled from `results/`.
6. Write `docs/REPRODUCIBILITY.md` and `docs/DATA.md` (absorb `data/README.md`).
7. Commit the 9 untracked `results/` subdirs and ~10 untracked Tier-1/2/3 scripts that back paper claims (after PII scrub).
8. Move `/tmp/tier2_prop6/{numerical_verification_v2.json, verify_bound_v2.py}` into `results/reviewer_dropins/prop6_bound_utilization/` and commit (closes claim #11).
9. Add `LICENSE` (MIT — confirm before writing).
10. Re-run the §1.2 grep. If anything remains, stop and report.

Constraints I will respect: no history rewriting, no dataset commits, no model-checkpoint commits, no `git push`, no `git filter-repo`. The git-history PII (Bostesa/natestesa@gmail.com author + GitHub clone URL in old commits) will be flagged in `CLEANUP_REPORT.md` for human decision.
