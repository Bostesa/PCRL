"""Tier-0 storage inventory: untracked/ignored files in PCRL worktrees.

Read-only. Records logical size, allocated bytes (st_blocks*512), nlink, inode,
mtime and a coarse category per file. Never opens file contents. Paths with a
'2016' year token are flagged SEALED and excluded from every candidate list.
"""
import csv, json, os, re, subprocess, sys
from collections import defaultdict
from pathlib import Path

WORKTREES = [l.split()[1] for l in subprocess.run(
    ['git', '-C', str(Path(__file__).parent), 'worktree', 'list', '--porcelain'],
    capture_output=True, text=True, check=True).stdout.splitlines() if l.startswith('worktree ')]
SEALED = re.compile(r'(^|[^0-9])2016([^0-9]|$)')
EXT = {'.pt': 'weights_pt', '.pth': 'weights_pt', '.joblib': 'fitted_joblib', '.pkl': 'fitted_joblib',
       '.npz': 'arrays', '.npy': 'arrays', '.parquet': 'tabular', '.csv': 'tabular', '.json': 'records',
       '.jsonl': 'records', '.md': 'reports', '.pdf': 'pdf', '.png': 'image', '.log': 'logs',
       '.aux': 'latex_aux', '.pyc': 'pycache', '.tar': 'archive', '.gz': 'archive', '.zip': 'archive'}


def untracked(wt):
    out = subprocess.run(['git', '-C', wt, 'ls-files', '-z', '--others'], capture_output=True, check=True).stdout
    return [p for p in out.decode().split('\0') if p]


def main(out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seen = {}
    agg = defaultdict(lambda: [0, 0, 0])
    with open(out_dir / 'untracked_files.tsv', 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['worktree', 'relpath', 'logical', 'allocated', 'nlink', 'dev_ino', 'mtime_ns', 'category', 'sealed', 'first_seen_at'])
        for wt in WORKTREES:
            for rel in untracked(wt):
                p = os.path.join(wt, rel)
                try:
                    st = os.lstat(p)
                except FileNotFoundError:
                    continue
                if not os.path.isfile(p) or os.path.islink(p):
                    cat = 'symlink_or_special'
                else:
                    cat = EXT.get(os.path.splitext(rel)[1].lower(), 'other')
                    if '__pycache__' in rel or '.pytest_cache' in rel: cat = 'pycache'
                key = f'{st.st_dev}:{st.st_ino}'
                first = seen.setdefault(key, p)
                sealed = bool(SEALED.search(rel))
                w.writerow([wt, rel, st.st_size, st.st_blocks * 512, st.st_nlink, key, st.st_mtime_ns, cat, int(sealed), '' if first == p else first])
                a = agg[(wt, rel.split('/')[0] + ('/' + rel.split('/')[1] if rel.startswith('results/') and rel.count('/') > 1 else ''), cat)]
                a[0] += st.st_size; a[2] += 1
                if first == p: a[1] += st.st_blocks * 512
    rows = sorted(([*k, *v] for k, v in agg.items()), key=lambda r: -r[4])
    with open(out_dir / 'untracked_by_dir_category.tsv', 'w') as fh:
        fh.write('worktree\tdir\tcategory\tlogical\tallocated_unique_inode\tfiles\n')
        for r in rows: fh.write('\t'.join(map(str, r)) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
