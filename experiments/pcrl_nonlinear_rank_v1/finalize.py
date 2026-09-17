"""Write HANDOFF.json: the committed SHA and the aggregate-artifact hashes."""
from __future__ import annotations

import argparse
import datetime
import subprocess
from pathlib import Path

from .inputs import OUT, read_json, sha_file, write_json

AGGREGATE = ('PROTOCOL.md', 'METHOD.md', 'RANK_DIAGNOSTIC.md', 'RANK_SPECTRUM.json',
             'RANK_SPECTRUM.csv', 'MATRIX.json', 'PROTOCOL_FREEZE.json', 'RUN_STATUS.md',
             'VALIDATION.md', 'DEVELOPMENT_2018.md', 'DEVELOPMENT_2018.json',
             'EXPLORATORY_2017.md', 'PER_SEED.csv', 'CRITERIA_PER_SEED.csv',
             'CRITERIA_SUMMARY.csv', 'PAIRED_INTERVALS.csv', 'SCORE_REPLAY.csv',
             'WITHHOLDING.csv', 'WITHHOLDING_DOMINANCE.csv', 'RESEARCH_DECISION.md',
             'PAPER_ADDENDUM.md', 'NEXT_CONFIRMATION_SPEC.md', 'REPRODUCE.md',
             'FIT_SUMMARY.json', 'DIAGNOSTICS_SUMMARY.json', 'DEV_2018_SUMMARY.json',
             'PREDICTION_REPLAY.json', 'EVIDENCE_MANIFEST.json', 'FIGURES.json')


def run(out: Path = OUT) -> dict:
    out = Path(out)
    root = out.parents[1]
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                         cwd=root).stdout.strip()
    branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True,
                            text=True, cwd=root).stdout.strip()
    remote = subprocess.run(['git', 'ls-remote', 'origin', f'refs/heads/{branch}'],
                            capture_output=True, text=True, cwd=root).stdout.split()
    matrix = read_json(out / 'MATRIX.json') if (out / 'MATRIX.json').exists() else {}
    replay = read_json(out / 'PREDICTION_REPLAY.json') if (out / 'PREDICTION_REPLAY.json').exists() else {}
    report = read_json(out / 'DEVELOPMENT_2018.json') if (out / 'DEVELOPMENT_2018.json').exists() else {}
    record = {
        'terminal': 'terminal_a',
        'study': 'pcrl_nonlinear_rank_v1',
        'branch': branch,
        'committed_sha': sha,
        'remote_sha': remote[0] if remote else None,
        'remote_matches_local': bool(remote and remote[0] == sha),
        'baseline': '349efa454afd907389760fd1f59fd8806a215efd',
        'worktree': str(root),
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'verdict': ('NO-GO. No candidate improved the tradeoff against both the local controls '
                    'and the frozen neural J, so none is nominated for confirmation. All eight '
                    'registered coordination cells fail.'),
        'ledger': {k: matrix.get(k) for k in
                   ('nominal_interface_records', 'unique_new_fits', 'reused_historical',
                    'void_by_alias')},
        'prediction_replay': {'checked': replay.get('predictions_checked'),
                              'all_clean': replay.get('all_clean')},
        'score_replay': report.get('score_replay'),
        'aggregate_artifact_sha256': {name: sha_file(out / name)
                                      for name in AGGREGATE if (out / name).exists()},
        'figures_sha256': {p.name: sha_file(p) for p in sorted((out / 'figures').glob('*'))
                           if p.is_file()} if (out / 'figures').exists() else {},
        'local_only_not_published': ('fitted maps, 2018/2017 releases, fitted candidates and raw '
                                    'per-person predictions stay on this machine; see '
                                    'EVIDENCE_MANIFEST.json and .gitignore'),
        'constraints_held': [
            '2016 untouched: no 2016 model output, score, attack, plot or final-label read',
            'every 2017 number is labelled EXPLORATORY CROSS-YEAR DEVELOPMENT, not a confirmation',
            'no paid or remote compute; no merge into main; no force-push',
            'writes confined to experiments/, tests/ and results/pcrl_nonlinear_rank_v1',
            'nothing historical refitted; historical inputs resolved read-only and hashed',
            'single worker and one BLAS thread throughout',
        ],
    }
    write_json(out / 'HANDOFF.json', record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    r = run(a.out)
    print('sha', r['committed_sha'][:12], '| remote matches:', r['remote_matches_local'],
          '| aggregate artifacts', len(r['aggregate_artifact_sha256']))


if __name__ == '__main__':
    main()
