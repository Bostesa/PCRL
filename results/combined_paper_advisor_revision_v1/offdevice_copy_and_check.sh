#!/usr/bin/env bash
# Copy one backup directory (made by backup_and_restore.sh) to an off-device location and verify it there:
# re-hash every file listed in MANIFEST.sha256, then rebuild the manuscript PDF from the copied source.tar and compare
# its SHA-256 with the committed PDF recorded in RESTORE_CHECK.json.
# Usage: offdevice_copy_and_check.sh BACKUP_DIR DEST_PARENT
# Caveat: if the drive stays mounted, the read-back may be served from the operating system's file cache.
set -euo pipefail
SRC="$1"
DEST_PARENT="$2"
NAME="$(basename "$SRC")"
mkdir -p "$DEST_PARENT"
cp -R "$SRC" "$DEST_PARENT/"
cd "$DEST_PARENT/$NAME"
shasum -a 256 -c MANIFEST.sha256
WORK="$(mktemp -d)"
tar -xf source.tar -C "$WORK"
rm -f "$WORK/papers/combined_empirical_v2/main.pdf"
(cd "$WORK/papers/combined_empirical_v2" \
  && SOURCE_DATE_EPOCH=1792108800 FORCE_SOURCE_DATE=1 latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1)
rebuilt="$(shasum -a 256 "$WORK/papers/combined_empirical_v2/main.pdf" | cut -d' ' -f1)"
expected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["manuscript_pdf"]["committed_sha256"])' RESTORE_CHECK.json)"
rm -rf "$WORK"
if [ "$rebuilt" = "$expected" ]; then same=true; else same=false; fi
cat > OFFDEVICE_CHECK.json <<EOF
{"copied_to": "$DEST_PARENT/$NAME", "manifest_rehash_ok": true, "rebuilt_manuscript_sha256": "$rebuilt",
 "expected_sha256": "$expected", "identical": $same, "checked_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
 "caveat": "drive stayed mounted; read-back may come from the file cache"}
EOF
cat OFFDEVICE_CHECK.json
