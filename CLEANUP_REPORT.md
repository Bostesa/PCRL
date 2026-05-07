# PCRL repo cleanup — Phase 2 report

Phase 2 executed against `CLEANUP_PLAN.md` with three additions from the
approval message: (a) verified `experiments/run_v2_dataset.py` exists
before referencing it in the README, (b) MIT license dated 2026, holder
"Anonymous", (c) README cites both cross-purpose criteria with the exact
26/33 + 22/33 split and full A/B definitions.

## Commits

Created on branch `crosspurp-constraint-2026-05-06`. **Nothing was pushed.**

| # | Commit | Title |
|--:|---|---|
| 1 | `8554865` | Extend .gitignore: untrack LAUNCH_README.md, outputs/, lightning_logs/ |
| 2 | `d09c5f7` | Untrack AWS launch scripts and operational logs (instance IDs, S3 buckets, GitHub clone URL) |
| 3 | `4fe3c25` | PII scrub: personal absolute paths → `${REPO_ROOT}`; advisor-name comment; instance ID in env-var default |
| 4 | `a12153e` | Rewrite README (anonymous header, headline-results table, paper-claim→artifact map, both cross-purpose criteria, MIT callout) |
| 5 | `7a7d33a` | Add `docs/REPRODUCIBILITY.md` + `docs/DATA.md` |
| 6 | `a3875c6` | Commit paper-supporting result artifacts and Tier-1/2/3 analysis scripts; .gitignore skips eval_reps.npz |
| 7 | `1ea908e` | Add Prop 6 cross-purpose bound utilization verifier + numerical results (claim #11) |
| 8 | `d359762` | Add MIT LICENSE (2026, Anonymous holder) |
| 9 | `5847a40` | Untrack laftr_benchmark .npz representation blobs (75 MB; reproducible from encoders + raw data) |

## What was removed

| Path | Why |
|---|---|
| `LAUNCH_README.md` | Personal AWS launch guide (S3 bucket, AMI, IAM profile). Contained `pcrl-bios-layer12-20260505` bucket name and `pcrl-overnight` IAM profile. Now in `.gitignore`. |
| `infra/inlp/launch.sh` | Contained `https://github.com/Bostesa/PCRL.git` clone URL, S3 bucket `pcrl-bios-overnight-20260504`, KEY name `pcrl-gpu-key`, IAM profile `pcrl-bios-s3-writer`. |
| `infra/inlp/user_data.sh` | Same. |
| `infra/inlp/aggregate.py` | Path-dependent on EC2 layout; not paper-cited. |
| `infra/splince/launch.sh` | Same as inlp launcher. |
| `infra/splince/user_data.sh` | Same. |
| `results/bios_layer12/SHUTDOWN.txt` | 5 AWS instance IDs; pure operational log. Deleted from disk and from index. |
| 24 × `results/laftr_benchmark/**/*.npz` | 75 MB of intermediate float-array representations, reproducible from encoders + raw data. Repo-size hygiene; covered by new `*.npz` gitignore rule. |

## PII findings: before vs after

| Category | Before (Phase 1) | After (Phase 2) |
|---|--:|--:|
| A. Author name in tracked code | 0 | 0 |
| B. Advisor name in tracked code | 1 (comment in `scripts/build_splince_vs_pcrl.py`) | 0 |
| C. Lab member name | 0 | 0 |
| D. Institution name | 0 | 0 |
| E. Email addresses in tracked text | 0 | 0 |
| F. Personal absolute paths in tracked text | 47+ | 0 |
| G. AWS instance IDs in tracked text | 9 (across 4 files) | 0 |
| G. S3 buckets in tracked text | 5 | 0 |
| G. GitHub clone URL `Bostesa/PCRL.git` in tracked text | 5 | 0 |
| H. AWS keys / secrets | 0 | 0 |
| I. GCP / W&B credentials | 0 | 0 |
| J. Slack/Zulip/Discord URLs | 0 | 0 |
| K. Google Docs/Drive personal links | 0 | 0 |
| L. Acknowledgments naming people | 0 | 0 |
| M. TODO/FIXME/XXX | 0 | 0 |
| N. SSH/PGP/private keys | 0 | 0 |

**Final Phase 2 sweep across all tracked text files: 0 residual hits.**

## Paper-claim coverage

| State | Phase 1 | Phase 2 |
|---|--:|--:|
| Fully present | 19 / 20 | **20 / 20** |
| Partial (script exists, output missing) | 1 / 20 (#11) | 0 / 20 |
| Absent | 0 / 20 | 0 / 20 |

Claim #11 (Cross-purpose Prop 6 bound utilization 45–70%) was closed by
commit `1ea908e`, which moved `numerical_verification_v2.json` and the
`verify_bound_v2.py` verifier from `/tmp/tier2_prop6/` into
`results/reviewer_dropins/prop6_bound_utilization/` (with personal-path
scrub on the verifier scripts). All 20 numbered claims in the paper now
have a committed artifact backing them.

## What is still incomplete

- **Round-4 baseline numbers (#6) are committed but not re-derivable from a
  single command.** The Round-4 grid was run before the optimizer-schedule
  fix (commit `dbe0fdc`); reproducing it requires checking out the pre-fix
  commit. Documented in `docs/REPRODUCIBILITY.md` Stage 4. Not blocking
  for the paper; the JSONs are committed.
- **`infra/laftr/` was referenced by an old commit message but the files
  are no longer in the tree.** Likely deleted before this audit; no action
  needed.
- **`paper-body/` directory name is non-standard.** Could rename to
  `tables/` for cleanliness; not required.

## git history concerns (NOT addressed — requires explicit user decision)

`git log` and `git show` still expose deanonymizing metadata. **None of the
Phase 2 commits rewrote history.** The risks:

1. **Author identity:** every recent commit attributes
   `Bostesa <natestesa@gmail.com>`. `git log --pretty=fuller` reveals this
   on every commit. Visible to anyone who clones the repo.
2. **Bucket/instance IDs in commit messages:** several commits reference
   real instance IDs in their subject lines (e.g. the BIOS Round 2 launch
   commits, the per-day overnight commits). Visible via `git log --grep`.
3. **The deanonymizing `https://github.com/Bostesa/PCRL.git` URL** lived in
   `infra/inlp/{launch,user_data}.sh`, `infra/splince/{launch,user_data}.sh`,
   and `LAUNCH_README.md` until commit `d09c5f7`. Phase 2 untracked them
   (file tree is clean now), but **`git show <pre-d09c5f7>:LAUNCH_README.md`
   will still print the URL**. The blob is still in the object database.

Two paths to fully resolve, in order of safety:

- **Recommended: trust 4open.science's history redaction.** The
  anonymous.4open.science mirror (the channel the user is publishing
  through) hides commit metadata and authors by default. If reviewers only
  see the 4open.science snapshot, the file tree clean is sufficient.
  Verify on the rendered PCRL-29E1 link: pull up `git log` in the web UI
  — if 4open.science shows commit messages with PII, the repo will need
  more work before push.
- **Destructive option (only if 4open.science doesn't redact):** run
  `git filter-repo` to rewrite all commit authors to "Anonymous
  <anonymous@example.com>", strip references to instance IDs / bucket
  names from commit messages, and remove the deanonymizing files from
  every historical revision. **This rewrites every SHA in the repo** and
  invalidates open PRs / branches / pulled clones. **I did not run it.**
  If you want this, run it locally and inspect with `git log --all
  --pretty=fuller` before force-pushing. Do not delegate that command.

## Final pre-push checklist

```sh
# 1. Review the diff against main
git log --stat origin/main..HEAD

# 2. Re-run the PII scrub one more time, just to be sure
git ls-files | xargs grep -EHIn '/Users/[a-zA-Z]+|/home/[a-zA-Z]+/PCRL|i-[0-9a-f]{8,17}|github\.com/Bostesa|natestesa|Bostesa' 2>/dev/null
# expect zero output

# 3. Decide the git-history question (recommend 4open.science redaction first)

# 4. Push (only after #3 is decided)
git push origin crosspurp-constraint-2026-05-06

# 5. Re-anonymize on anonymous.4open.science
#    The existing PCRL-29E1 link auto-updates from the new push. Open it,
#    spot-check that LAUNCH_README.md / infra/inlp/ / infra/splince/ /
#    results/bios_layer12/SHUTDOWN.txt are no longer present in the
#    rendered tree, and that no commit message exposes a Bostesa SHA.
```
