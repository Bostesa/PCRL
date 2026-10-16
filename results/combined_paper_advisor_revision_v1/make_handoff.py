"""Write HANDOFF.json for the advisor-ready manuscript revision.

Usage: python make_handoff.py EXTRA_JSON
EXTRA_JSON holds run-time facts (commits, backup results, remote SHA) merged into the record.
Hashes are computed here from the files in the worktree. Nothing here reads real-person data.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

FILES = [
    'papers/combined_empirical_v2/main.tex',
    'papers/combined_empirical_v2/references.bib',
    'papers/combined_empirical_v2/fig_recipient.tex',
    'papers/combined_empirical_v2/main.pdf',
    'papers/combined_empirical_v2/figures/fig_formats_pairs.pdf',
    'papers/combined_empirical_v2/figures/fig_useful_head.pdf',
    'papers/combined_empirical_v1/main.tex',
    'papers/combined_empirical_v1/references.bib',
    'papers/combined_empirical_v1/main.pdf',
    'results/combined_paper_advisor_revision_v1/ADVISOR_BRIEF.tex',
    'results/combined_paper_advisor_revision_v1/ADVISOR_BRIEF.pdf',
    'results/combined_paper_advisor_revision_v1/METHOD_SCOPE_PROPOSAL.md',
    'results/combined_paper_advisor_revision_v1/CHANGE_NOTE.md',
    'results/combined_paper_advisor_revision_v1/CLAIM_SOURCE_CHECK.md',
    'results/combined_paper_advisor_revision_v1/SOURCE_RESOLUTION.md',
    'results/combined_paper_advisor_revision_v1/EVIDENCE_EXTENSIONS.md',
    'results/combined_paper_advisor_revision_v1/MANUSCRIPT_DIFF.diff',
    'results/combined_paper_advisor_revision_v1/MANUSCRIPT_WORD_DIFF.txt',
    'results/combined_paper_advisor_revision_v1/README.md',
    'results/combined_paper_advisor_revision_v1/backup_and_restore.sh',
    'results/combined_paper_advisor_revision_v1/offdevice_copy_and_check.sh',
]


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main():
    extra = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else {}
    rec = {
        'schema': 'CPR1-handoff-v1',
        'branch': 'research/combined-paper-advisor-revision-v1',
        'admitted_source': {'commit': '3b5065805bac588bf2f3214c656f9069eb533ccb',
                            'path': 'papers/combined_empirical_v1/'},
        'revised_source': {'path': 'papers/combined_empirical_v2/',
                           'build': 'cd papers/combined_empirical_v2 && latexmk -pdf main.tex'},
        'unavailable_version': 'PCRL_Combined_Empirical_Revised.pdf / _Source.zip: not on this host; not revised',
        'adversary_patch_v2': {'source_commit': 'c9709a29863a70ef2e88b5e53a8a6e20413e03a0',
                               'sha256': '3d296d2be026c381d8c5b3fa7bd6b3c6dc1d7ecd51d0f3efc32526f2b9d4e068',
                               'applied_commit': '133d2c740'},
        'evidence_cutoff': 'numeric record of the admitted manuscript (experiments through the useful-head '
                           'comparison, 2026-10-03); no new statistics; later corrections used only to repair claims',
        'scientific_runs': 'none: no defense training, attacker fitting, new statistics, real-person array reads, '
                           'population admission or sealed-assessment access',
        'sha256': {p: sha(p) for p in FILES if (ROOT / p).is_file()},
    }
    rec.update(extra)
    (OUT / 'HANDOFF.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: rec[k] for k in ('branch', 'sha256')}, indent=1))


if __name__ == '__main__':
    main()
