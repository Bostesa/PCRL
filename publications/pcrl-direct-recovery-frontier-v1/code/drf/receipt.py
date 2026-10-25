"""Public receipt verification for the gates of the PCRL direct-recovery frontier campaign.

    python3 -m drf.receipt verify --kind registration  --commit <publication commit> --research-commit <research commit>

(Derived unchanged from nrm.receipt, itself derived from tsz.receipt, the repaired receipt check.)

Checks, with `git ls-remote` against the GitHub remote (https://github.com/Bostesa/PCRL.git; `origin` here is that remote, not a
local mirror), that the publication branch tip equals the receipt commit; that the receipt commit's tree holds the sanitized copy
of every gated file with the same bytes as the committed research copy; and that every gated file is committed and unchanged in
this worktree.

Repair of the inherited defect (occupation-headroom engineering review E13, PROTOCOL_DEVIATIONS item 21): `ohn.receipt verify`
wrote its write-once private record even when verification FAILED, which would have blocked every later retry. Here every attempt,
failed or verified, is appended to the append-only RECEIPT_ATTEMPTS.jsonl; ONLY a verified attempt creates the write-once record
that the access layer requires (exclusive create), and a second verified attempt for the same gate is refused. Nothing is
backdated: records carry the verification time. Tests: `python3 -m drf.test_receipt` (temporary directories, fake git; no data).
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

from . import common as C

GATES = {'registration': {'files': C.REGISTRATION_FILES, 'record': 'REGISTRATION_RECEIPT_VERIFIED.json'}}


def _git_real(*a):
    r = subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def verify(kind, commit, research_commit, *, git=_git_real, results=None, access=None, committed=None, now=C.now,
           remote_url=C.REMOTE_URL, branch=C.PUBLICATION_BRANCH, name=C.PUBLICATION_NAME):
    results = Path(results) if results is not None else C.RESULTS
    access = Path(access) if access is not None else C.ACCESS
    committed = committed or (lambda p: C.committed(p))
    gate = GATES[kind]
    files = gate['files']
    problems = []
    rc, url, _ = git('remote', 'get-url', 'origin')
    if rc != 0 or url != remote_url:
        problems.append(f'origin is {url!r}, not the GitHub remote {remote_url}')
    rc, out, err = git('ls-remote', 'origin', f'refs/heads/{branch}')
    remote_sha = out.split()[0] if rc == 0 and out else None
    if remote_sha is None:
        problems.append(f'ls-remote failed or branch absent: {err}')
    elif remote_sha != commit:
        problems.append(f'remote tip {remote_sha} differs from the receipt commit {commit}')
    blobs, missing, differ = {}, [], []
    for f in files:
        rc, b, _ = git('rev-parse', f'{commit}:publications/{name}/results/{f}')
        blobs[f] = b if rc == 0 else None
        if rc != 0:
            missing.append(f)
            continue
        rc2, local_blob, _ = git('hash-object', str(results / f))
        if rc2 != 0 or local_blob != b:
            differ.append(f)
    not_committed = [f for f in files if not committed(results / f)]
    rc, _, _ = git('merge-base', '--is-ancestor', research_commit, 'HEAD')
    if missing:
        problems.append(f'missing in the receipt tree: {missing}')
    if differ:
        problems.append(f'published bytes differ from the research copy: {differ}')
    if not_committed:
        problems.append(f'not committed: {not_committed}')
    if rc != 0:
        problems.append('research commit is not an ancestor of HEAD')
    ok = not problems
    rec = {'kind': kind, 'receipt_commit': commit, 'remote_sha': remote_sha, 'remote_url': url, 'remote_sha_verified': ok,
           'research_commit': research_commit, 'files_sha256': {f: C.sha(results / f) if (results / f).is_file() else None for f in files},
           'published_blobs': blobs, 'problems': problems, 'verified_utc': now()}
    access.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(access / 'RECEIPT_ATTEMPTS.jsonl', 'a') as f:            # every attempt, append-only
        f.write(json.dumps({**rec, 'outcome': 'verified' if ok else 'failed'}, sort_keys=True) + '\n')
    if not ok:
        return rec                                                     # no write-once record on failure (the repaired defect)
    path = access / gate['record']
    try:
        with open(path, 'x') as f:                                     # write-once, exclusive create
            f.write(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    except FileExistsError:
        raise PermissionError(f'{path.name} already exists (write-once; the gate was verified before)')
    pub = results / 'RECEIPTS.json'
    allrec = json.loads(pub.read_text()) if pub.is_file() else {'schema': 'DRF1-receipts-v1', 'receipts': []}
    allrec['receipts'].append(rec)
    pub.write_text(json.dumps(allrec, indent=1, sort_keys=True) + '\n')
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--kind', required=True, choices=tuple(GATES))
    v.add_argument('--commit', required=True); v.add_argument('--research-commit', required=True)
    a = ap.parse_args(argv)
    rec = verify(a.kind, a.commit, a.research_commit)
    print(json.dumps(rec, indent=1))
    if not rec['remote_sha_verified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
