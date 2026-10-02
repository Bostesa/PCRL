# Shared context for the evidence-reconciliation roles (2026-10-01, evening)

Assignment: **evidence reconciliation, not a method search**. No new training, hyperparameter search,
attacker fitting, fresh-population evaluation, cloud compute, deletions of archived evidence, or merges
into main. Allowed: reading GitHub branches, reading the external drive in place (single-member
extraction to scratch is fine), deterministic recounts from stored outputs, small synthetic fixtures.
Label any reconstructed evidence clearly.

## Status vocabulary (use exactly)
- **independently recomputed** — recomputed by our own standalone code from stored per-seed/per-person outputs.
- **checked against code and recorded aggregates** — code path read; aggregate values match stored files.
- **reported but not independently reproduced** — a number appears in prose/HEADLINE/summary only.
- **unsupported by located artifacts** — no artifact located after searching GitHub branches, the drive, and manifests.
Availability classes: on GitHub (branch@commit:path) | external drive only (archive + member path +
sha256 from inventory) | referenced archive only (e.g. S3 manifest entry; not read) | prose only |
registered but not executed.

## Pins and locations
- PCRL GitHub: https://github.com/Bostesa/PCRL ; local repo /Users/nathansamson/PCRL (primary checkout
  belongs to another session: read refs via `git -C /Users/nathansamson/PCRL show/grep/ls-tree/log`, never
  checkout). Run `git -C /Users/nathansamson/PCRL fetch origin` already done; list all remote branches.
- durable-guarantees: read-only clone
  /private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad/dg
  @ 956f5c883f515646aa457db55ecbd74b913768b2 (also check `git ls-remote https://github.com/Bostesa/durable-guarantees` for new refs).
- External drive (MOUNTED, read-only use): /Volumes/YOTUO/NathanSamson-Mac-relocated-2026-09-30/
  (archives/*.tar, inventories/*.json.gz = per-member path→{size, sha256, mtime}, git/*.bundle,
  RESTORE.md, manifest.json) and /Volumes/YOTUO/NathanSamson-Mac-relocated-2026-09-25/ (batches/,
  inventories/, git-bundles/, manifest.json). Key archives: fl-PCRL-main-results-ignored.tar (12 GB, PCRL
  git-ignored results incl. per-seed outputs), fl-PCRL-main-checkpoints.tar (6.8 GB),
  fl-PCRL-main-data-celeba.tar, tree-durable-guarantees.tar (2.1 GB; incl. 721 untracked analysis files),
  tree-laftr_cells.tar, tree-fnf.tar, wt-PCRL-superpowers-laftr-hard-r2-2026-05-17.tar, other wt-PCRL-*.tar.
  To find a file: query the inventory json.gz (python gzip+json) — do NOT untar whole archives. To read a
  member: `tar -xOf <archive> '<member path>'` (or `-x -C <scratch> -T list.txt` for several) into your
  scratch dir /private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad/recon/<role>/.
  Never write to the drive.
- S3 archive pcrl-ux-archive-ed9d21fd (no lifecycle) is referenced by manifests in PCRL branches; AWS
  credentials may be expired — record manifest entries, do not download buckets.
- Previous assessment (to be corrected where wrong): PCRL branch research/combined-empirical-preparation-v1
  @ f381bc2, dir results/combined_empirical_preparation_v1/ (MEETING_BRIEF.md, BOTH_REPOS_INDEX.json,
  EVIDENCE_CROSSWALK.csv, METHODOLOGY_IMPROVEMENTS.md, EVALUATION_PROTOCOL_DRAFT.md, notes/*/FINDINGS.md,
  INDEPENDENT_CHECKS.json, fixtures/). Its main limits: official reviews unavailable; drive unmounted;
  LAFTR-hard full output, durable raw arrays, PCRL ignored results not inspected.

## Manuscripts (hashes verified)
- NeurIPS PCRL supplied PDF sha256 397d85d5…894f (26 pp) = local build "Formatting_Instructions_For_NeurIPS_2026 (23).pdf"
  (sha256 4a705d5e…) content + venue reviewer-copy stamp text on every page (no other word differences).
  Text: /private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad/recon/b.txt (the (23) build).
- AAAI "Outputs Leak" supplied PDF sha256 2d86afad…b077 (9 pp) = local (58) build, Jul 29 04:57 EDT. Supplement
  PDF (65) sha256 9921d6cc… (5 pp proofs). Source zip ~/Downloads/AAAI27___Outputs_Leak_What_They_Use.zip.

## Chronology anchors
- NeurIPS PCRL: submitted May 5–7 2026; reviews written Jun 22–26, released to authors Jul 24; no
  substantive author response in the export; withdrawn Aug 21.
- AAAI: submitted Jul 21–28 (revisions to Aug 2); reviews Aug 23–30 (AI review Jul 27); visible with
  decision Sep 24 (Phase-1 reject).
- Many "rebuttal" branches (May 2026) PREDATE the official reviews: record that an experiment can address
  a concern without having been done in response to it.

## Reviews (private)
Official exports and the meeting transcript are in a PRIVATE local file:
~/Documents/PCRL_private_review_sources_20261001/PCRL_Meeting_and_Reviews_Followup_Prompt.txt
Never copy reviewer handles, submission numbers, author names from the export, or verbatim review text
into the public package. Public pseudonyms: NeurIPS N-R1, N-R2, N-R3 (order as in the export: first,
second, third human review); AAAI A-R1 (first human review), A-R2 (second human review), A-AI (AI review).

## Write rules
- Write only under your role dir `results/combined_evidence_reconciliation_v1/notes/<role>/` in
  /Users/nathansamson/PCRL/.worktrees/combined-evidence-reconciliation-v1 (verification role may also write
  `results/combined_evidence_reconciliation_v1/fixtures/`). Private material only under
  ~/Documents/PCRL_private_review_sources_20261001/<role>/.
- The harness may refuse files named like reports (SUMMARY.md, FINDINGS.md). Prefer data files (CSV/JSON)
  and `<topic>_notes.md`; if a write is refused, put the content in your final reply instead — do not
  work around the refusal.
- Every claim: `repo@commit:path[:line]`, or `archive::member (sha256)`, or manuscript page/section.
- Final reply: a concise summary (<= 700 words) with EARLY FLAGS first.
