"""Verify published redesign evidence without fitting models or loading datasets.

Run from a clone with Python's standard library. --include-local additionally
checks omitted artifacts when the original evidence is available locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--include-local', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    publication = root / 'results/redesign_publication_20260907'
    manifest = json.loads((publication / 'artifact_manifest.json').read_text())
    errors = []
    checked = 0
    for row in manifest['files']:
        if row['availability'] != 'published_verbatim' and not args.include_local:
            continue
        path = root / row['path']
        if not path.is_file():
            errors.append(f"Missing artifact: {row['path']}")
            continue
        content = path.read_bytes()
        if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
            errors.append(f"Changed artifact: {row['path']}")
        checked += 1
    source_record = json.loads((publication / 'source_equivalence.json').read_text())
    for row in source_record['frozen_executable_sources']:
        path = root / row['source']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['original_sha256']:
            errors.append(f"Changed executed source: {row['source']}")

    docs = [root / name for name in (
        'PCRL_NEXT_STAGE_RESEARCH_REPORT.md', 'PCRL_REDESIGN_STATUS.md',
        'docs/PCRL_REVIEW_INDEX.md', 'docs/PCRL_PUBLICATION.md',
        'docs/PCRL_APPLICATION_SELECTION.md', 'docs/PCRL_PRIOR_WORK.md',
        'docs/ACCURACY_CERTIFICATE_RETIREMENT.md',
    )]
    docs += [root / row['path'] for row in manifest['files']
             if row['availability'] == 'published_verbatim' and row['path'].endswith('.md')]
    links_checked = 0
    for doc in docs:
        if not doc.is_file():
            errors.append(f'Missing document: {doc.relative_to(root)}')
            continue
        # Canonical publication documents use ordinary inline Markdown links.
        for target in re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)', doc.read_text()):
            parsed = urlsplit(target.strip('<>'))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            relative = unquote(parsed.path)
            resolved = root / relative.lstrip('/') if relative.startswith('/') else doc.parent / relative
            if not resolved.exists():
                errors.append(f'Broken local link: {doc.relative_to(root)} -> {target}')
            links_checked += 1
    result = {
        'artifact_files_checked': checked,
        'omitted_artifacts_checked': args.include_local,
        'frozen_source_entries_checked': len(source_record['frozen_executable_sources']),
        'documents_checked': len(docs), 'local_links_checked': links_checked,
        'errors': errors,
        'scope': 'Byte integrity and local link existence only; no research rerun or checkpoint replay.',
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
