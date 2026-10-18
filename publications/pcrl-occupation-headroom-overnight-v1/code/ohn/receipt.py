"""Public receipt verification for the gates of the overnight occupation-headroom campaign.

    python -m ohn.receipt verify --kind registration --commit <publication commit> --research-commit <research commit>
    python -m ohn.receipt verify --kind reporting    --commit <publication commit> --research-commit <research commit>

Checks, with `git ls-remote` against the GitHub remote (https://github.com/Bostesa/PCRL.git; origin is not a local mirror), that
the publication branch tip equals the receipt commit; that the receipt commit's tree holds the sanitized copy of every gated file;
and that every gated file is committed and unchanged in this worktree. Writes the private verified record that
`ohn.common.registered()` / `dev_report_truth` require (REGISTRATION_RECEIPT_VERIFIED.json / REPORTING_RECEIPT_VERIFIED.json under
.private/<campaign>/access/) and appends the public summary to results/<campaign>/RECEIPTS.json. Nothing is backdated: the record
carries the verification time.
"""
from __future__ import annotations
import argparse
import json
import subprocess

from . import common as C

NAME = 'pcrl-occupation-headroom-overnight-v1'


def _git(*a):
    r = subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def verify(kind, commit, research_commit):
    files = C.REGISTRATION_FILES if kind == 'registration' else ('REPORTING_LOCK.json',)
    rc, url, _ = _git('remote', 'get-url', 'origin')
    if rc != 0 or url != C.REMOTE_URL:
        raise PermissionError(f'origin is {url!r}, not the GitHub remote {C.REMOTE_URL}')
    rc, out, err = _git('ls-remote', 'origin', f'refs/heads/{C.PUBLICATION_BRANCH}')
    if rc != 0 or not out:
        raise PermissionError(f'ls-remote failed or branch absent: {err}')
    remote_sha = out.split()[0]
    blobs, missing = {}, []
    for f in files:
        rc, b, _ = _git('rev-parse', f'{commit}:publications/{NAME}/results/{f}')
        if rc != 0:
            missing.append(f)
        blobs[f] = b if rc == 0 else None
    not_committed = [f for f in files if not C.committed(C.RESULTS / f)]
    rc, _, _ = _git('merge-base', '--is-ancestor', research_commit, 'HEAD')
    rec = {'kind': kind, 'receipt_commit': commit, 'remote_sha': remote_sha, 'remote_url': url,
           'remote_sha_verified': remote_sha == commit and not missing and not not_committed and rc == 0,
           'research_commit': research_commit, 'research_commit_is_ancestor_of_HEAD': rc == 0,
           'files_sha256': {f: C.sha(C.RESULTS / f) for f in files}, 'published_blobs': blobs, 'missing_in_receipt': missing,
           'not_committed': not_committed, 'verified_utc': C.now()}
    path = C.REG_RECEIPT if kind == 'registration' else C.REP_RECEIPT
    if path.exists():
        raise PermissionError(f'{path.name} already exists (write-once)')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    pub = C.RESULTS / 'RECEIPTS.json'
    allrec = json.loads(pub.read_text()) if pub.is_file() else {'schema': 'OHN1-receipts-v1', 'receipts': []}
    allrec['receipts'].append({k: v for k, v in rec.items() if k not in ('files_sha256',)} | {'files_sha256': rec['files_sha256']})
    C.write_json(pub, allrec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--kind', required=True, choices=('registration', 'reporting'))
    v.add_argument('--commit', required=True); v.add_argument('--research-commit', required=True)
    a = ap.parse_args(argv)
    rec = verify(a.kind, a.commit, a.research_commit)
    print(json.dumps(rec, indent=1))
    if not rec['remote_sha_verified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
