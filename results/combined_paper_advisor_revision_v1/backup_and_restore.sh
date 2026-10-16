#!/usr/bin/env bash
# Back up the manuscript revision and verify that the PDFs rebuild, byte for byte, from the backup alone.
#
# Usage: backup_and_restore.sh DEST_DIR [BASE_COMMIT]
#   DEST_DIR     directory that receives combined_paper_advisor_revision_v1_<UTC stamp>/
#   BASE_COMMIT  prerequisite of the incremental bundle (default: the admitted manuscript commit 3b5065805)
#
# Writes: branch_since_base.bundle (git history since BASE_COMMIT), source.tar (git archive of the manuscript folders and
# this results folder at HEAD), TIP_COMMIT, MANIFEST.sha256 and RESTORE_CHECK.json. The restore check extracts source.tar
# into a fresh temporary directory, deletes the committed PDFs there, rebuilds them with the fixed build date used for the
# committed PDFs (SOURCE_DATE_EPOCH=1792108800, FORCE_SOURCE_DATE=1) and compares SHA-256 values with HEAD.
# Nothing here reads real-person data.
set -euo pipefail
DEST="$1"
BASE="${2:-3b5065805bac588bf2f3214c656f9069eb533ccb}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
TIP="$(git rev-parse HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$DEST/combined_paper_advisor_revision_v1_${STAMP}"
mkdir -p "$OUT"
export SOURCE_DATE_EPOCH=1792108800 FORCE_SOURCE_DATE=1

git bundle create "$OUT/branch_since_base.bundle" "$BASE..$BRANCH" 2> "$OUT/bundle_create.log"
git archive --format=tar -o "$OUT/source.tar" HEAD \
  papers/combined_empirical_v2 papers/combined_empirical_v1 results/combined_paper_advisor_revision_v1
echo "$TIP" > "$OUT/TIP_COMMIT"
(cd "$OUT" && shasum -a 256 branch_since_base.bundle source.tar TIP_COMMIT > MANIFEST.sha256)
bundle_ok=false
if git bundle verify "$OUT/branch_since_base.bundle" > "$OUT/bundle_verify.log" 2>&1; then bundle_ok=true; fi

R="$(mktemp -d)"
tar -xf "$OUT/source.tar" -C "$R"
rm -f "$R/papers/combined_empirical_v2/main.pdf" "$R/results/combined_paper_advisor_revision_v1/ADVISOR_BRIEF.pdf"
(cd "$R/papers/combined_empirical_v2" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1)
(cd "$R/results/combined_paper_advisor_revision_v1" \
  && pdflatex -interaction=nonstopmode -halt-on-error ADVISOR_BRIEF.tex > build.log 2>&1 \
  && pdflatex -interaction=nonstopmode -halt-on-error ADVISOR_BRIEF.tex >> build.log 2>&1)

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
committed_main="$(git show "HEAD:papers/combined_empirical_v2/main.pdf" | shasum -a 256 | cut -d' ' -f1)"
committed_brief="$(git show "HEAD:results/combined_paper_advisor_revision_v1/ADVISOR_BRIEF.pdf" | shasum -a 256 | cut -d' ' -f1)"
rebuilt_main="$(sha "$R/papers/combined_empirical_v2/main.pdf")"
rebuilt_brief="$(sha "$R/results/combined_paper_advisor_revision_v1/ADVISOR_BRIEF.pdf")"
pages_main="$(pdfinfo "$R/papers/combined_empirical_v2/main.pdf" | awk '/^Pages/ {print $2}')"
warnings="$(grep -c -E 'LaTeX Warning|undefined' "$R/papers/combined_empirical_v2/main.log" || true)"

cat > "$OUT/RESTORE_CHECK.json" <<EOF
{
 "tip_commit": "$TIP",
 "bundle_prerequisite": "$BASE",
 "bundle_verify_ok": $bundle_ok,
 "manuscript_pdf": {"committed_sha256": "$committed_main", "rebuilt_sha256": "$rebuilt_main",
                    "identical": $([ "$committed_main" = "$rebuilt_main" ] && echo true || echo false),
                    "pages": $pages_main, "latex_warnings": $warnings},
 "advisor_brief_pdf": {"committed_sha256": "$committed_brief", "rebuilt_sha256": "$rebuilt_brief",
                       "identical": $([ "$committed_brief" = "$rebuilt_brief" ] && echo true || echo false)},
 "build_env": "SOURCE_DATE_EPOCH=1792108800 FORCE_SOURCE_DATE=1; latexmk -pdf main.tex; pdflatex x2 ADVISOR_BRIEF.tex",
 "checked_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
rm -rf "$R"
echo "$OUT"
cat "$OUT/RESTORE_CHECK.json"
