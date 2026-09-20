"""Durable private archive of PCRL result bytes: plan -> upload -> remote verify -> guarded delete.

Design rules (Tier 0, B2):
* chunks hold <= CHUNK_BYTES of *input* bytes (one oversized file gets its own chunk);
* tar is streamed through zstd straight into `aws s3 cp -`: no local staging copy;
* every file's SHA-256 is computed from the exact bytes placed in the tar, and the
  compressed stream's SHA-256 is computed from the exact bytes sent to S3;
* an upload is NOT a verification. `verify` (run on AWS) re-reads each object,
  checks the stream hash, extracts to a disposable directory and re-hashes every file;
* `delete` unlinks only manifest-listed regular files whose verification record
  passed and whose size / mtime / inode / SHA-256 are unchanged since the manifest,
  after lstat/realpath containment checks. Symlinks are archived as links and never
  deleted. Anything that changed is skipped and reported for re-inventory.
* Year-2016 paths are rejected at plan time (sealed).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

CHUNK_BYTES = 2 * 2 ** 30
SEALED = re.compile(r'(^|[^0-9])2016([^0-9]|$)|ss16|acs_2016_admission')
READ = 1 << 20


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    with os.fdopen(fd, 'w') as fh:
        json.dump(payload, fh, indent=1)
        fh.write('\n')
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(READ), b''):
            h.update(block)
    return h.hexdigest()


def git_head(root):
    r = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return r.stdout.strip() or None


def tracked_set(root, sub):
    out = subprocess.run(['git', '-C', str(root), 'ls-files', '-z', '--', sub], capture_output=True, check=True).stdout
    return {p for p in out.decode().split('\0') if p}


# ------------------------------------------------------------------ plan
def plan(groups, out):
    """groups: list of {label, root, subdirs, note}. Only UNTRACKED files are archived-for-delete;
    tracked files already live in git and are left alone."""
    chunks, cur, cur_bytes = [], [], 0
    rejected = []

    def flush(group):
        nonlocal cur, cur_bytes
        if cur:
            chunks.append({'group': group['label'], 'entries': cur, 'input_bytes': cur_bytes})
        cur, cur_bytes = [], 0

    for group in groups:
        root = Path(group['root'])
        for sub in group['subdirs']:
            tracked = tracked_set(root, sub)
            if (root / sub).is_file():
                walker = [(str((root / sub).parent), [], [(root / sub).name])]
            elif not (root / sub).exists():
                rejected.append({'path': sub, 'reason': 'missing'})
                continue
            else:
                walker = os.walk(root / sub, followlinks=False)
            for dirpath, dirnames, filenames in walker:
                dirnames.sort()
                rel_dir = os.path.relpath(dirpath, root)
                for name in sorted(filenames) + sorted(d for d in dirnames if os.path.islink(os.path.join(dirpath, d))):
                    rel = os.path.normpath(os.path.join(rel_dir, name))
                    full = root / rel
                    if SEALED.search(rel):
                        rejected.append({'path': rel, 'reason': 'sealed_2016_pattern'})
                        continue
                    if rel in tracked or name == '.DS_Store':
                        continue
                    st = os.lstat(full)
                    if os.path.islink(full):
                        entry = {'rel': rel, 'type': 'symlink', 'target': os.readlink(full), 'size': 0}
                    elif os.path.isfile(full):
                        entry = {'rel': rel, 'type': 'file', 'size': st.st_size, 'mtime_ns': st.st_mtime_ns,
                                 'ino': st.st_ino, 'dev': st.st_dev}
                    else:
                        rejected.append({'path': rel, 'reason': 'special_file'})
                        continue
                    if cur and cur_bytes + entry['size'] > CHUNK_BYTES:
                        flush(group)
                    cur.append(entry)
                    cur_bytes += entry['size']
        flush(group)
    groups_out = [{**g, 'head': git_head(g['root'])} for g in groups]
    for i, c in enumerate(chunks):
        c['chunk_id'] = f'{c["group"]}__{i:04d}'
    write_json_atomic(out, {'created_utc': utcnow(), 'chunk_bytes_limit': CHUNK_BYTES, 'groups': groups_out,
                            'chunks': chunks, 'rejected': rejected})
    total = sum(c['input_bytes'] for c in chunks)
    print(f'{len(chunks)} chunks, {total / 2 ** 30:.2f} GiB input, {len(rejected)} rejected')


# ------------------------------------------------------------------ upload
class HashingReader(io.RawIOBase):
    """File wrapper hashing exactly the bytes tarfile reads."""

    def __init__(self, fh):
        self.fh, self.h, self.n = fh, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, b):
        data = self.fh.read(len(b))
        self.h.update(data)
        self.n += len(data)
        b[:len(data)] = data
        return len(data)


def upload(plan_path, chunk_ids, bucket, prefix, manifest_dir, zstd_level=3, threads=2):
    p = json.load(open(plan_path))
    groups = {g['label']: g for g in p['groups']}
    manifest_dir = Path(manifest_dir)
    for chunk in p['chunks']:
        if chunk_ids and chunk['chunk_id'] not in chunk_ids:
            continue
        mpath = manifest_dir / f'{chunk["chunk_id"]}.json'
        if mpath.exists() and json.load(open(mpath)).get('upload_ok'):
            print('skip (uploaded)', chunk['chunk_id'])
            continue
        g = groups[chunk['group']]
        root = Path(g['root'])
        key = f'{prefix}/chunks/{chunk["chunk_id"]}.tar.zst'
        t0 = time.time()
        zstd = subprocess.Popen(['zstd', f'-{zstd_level}', f'-T{threads}', '-q', '-c'],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        aws = subprocess.Popen(['aws', 's3', 'cp', '-', f's3://{bucket}/{key}', '--only-show-errors',
                                '--expected-size', str(max(chunk['input_bytes'], 1)),
                                '--sse', 'AES256', '--checksum-algorithm', 'SHA256'],
                               stdin=subprocess.PIPE)
        files, err = [], []
        stream = {'h': hashlib.sha256(), 'n': 0}

        def pump():
            for block in iter(lambda: zstd.stdout.read(READ), b''):
                stream['h'].update(block)
                stream['n'] += len(block)
                aws.stdin.write(block)
            aws.stdin.close()

        th = threading.Thread(target=pump)
        th.start()
        try:
            with tarfile.open(fileobj=zstd.stdin, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for e in chunk['entries']:
                    full = root / e['rel']
                    arc = f'{chunk["group"]}/{e["rel"]}'
                    if e['type'] == 'symlink':
                        ti = tarfile.TarInfo(arc)
                        ti.type, ti.linkname = tarfile.SYMTYPE, e['target']
                        tar.addfile(ti)
                        files.append({**e})
                        continue
                    st = os.lstat(full)
                    if (st.st_size, st.st_mtime_ns, st.st_ino) != (e['size'], e['mtime_ns'], e['ino']):
                        err.append({'rel': e['rel'], 'reason': 'changed_since_plan'})
                        continue
                    ti = tar.gettarinfo(str(full), arcname=arc)
                    with open(full, 'rb') as fh:
                        hr = HashingReader(fh)
                        tar.addfile(ti, io.BufferedReader(hr, READ))
                    if hr.n != e['size']:
                        err.append({'rel': e['rel'], 'reason': 'short_read'})
                    files.append({**e, 'sha256': hr.h.hexdigest()})
        finally:
            zstd.stdin.close()
        zstd.wait()
        th.join()
        rc = aws.wait()
        head = {}
        if rc == 0:
            r = subprocess.run(['aws', 's3api', 'head-object', '--bucket', bucket, '--key', key,
                                '--checksum-mode', 'ENABLED', '--output', 'json'], capture_output=True, text=True)
            head = json.loads(r.stdout) if r.returncode == 0 else {'error': r.stderr}
        man = {'chunk_id': chunk['chunk_id'], 'group': chunk['group'], 'source_head': g['head'],
               'source_note': g.get('note'), 'readers': g.get('readers'),
               'key': key, 'version_id': head.get('VersionId'), 'stored_bytes': head.get('ContentLength'),
               's3_checksum': {k: v for k, v in head.items() if k.startswith('Checksum')},
               'archive_sha256': stream['h'].hexdigest(), 'archive_bytes': stream['n'],
               'input_bytes': sum(f['size'] for f in files), 'n_files': len(files), 'files': files,
               'errors': err, 'aws_rc': rc, 'zstd_rc': zstd.returncode, 'seconds': round(time.time() - t0, 1),
               'upload_ok': rc == 0 and zstd.returncode == 0 and head.get('ContentLength') == stream['n'],
               'uploaded_utc': utcnow()}
        write_json_atomic(mpath, man)
        print(chunk['chunk_id'], 'ok' if man['upload_ok'] else 'FAIL', f'{stream["n"] / 2 ** 20:.0f} MiB',
              f'{man["seconds"]}s', f'errors={len(err)}', flush=True)
        if man['upload_ok']:
            subprocess.run(['aws', 's3', 'cp', str(mpath), f's3://{bucket}/{prefix}/manifests/{mpath.name}',
                            '--only-show-errors', '--sse', 'AES256'], check=False)


# ------------------------------------------------------------------ verify (runs on AWS)
_STREAM = """set -o pipefail
mkfifo "{work}/.pipe"
SHACMD="$(command -v sha256sum || echo shasum -a 256)"
aws s3api get-object --bucket {bucket} --key "{key}" --version-id "{vid}" "{work}/.pipe" > "{work}/.meta.json" &
GET=$!
cat "{work}/.pipe" | tee >($SHACMD > "{work}/.sha") | zstd -d -q | tar -x {strip} -C "{dest}"
RC=$?
wait $GET; GRC=$?
sleep 1
exit $(( RC | GRC ))"""


def verify(bucket, prefix, manifest_keys, scratch, out_dir):
    out_dir = Path(out_dir)
    for mkey in manifest_keys:
        man = json.loads(subprocess.run(['aws', 's3', 'cp', f's3://{bucket}/{mkey}', '-'],
                                        capture_output=True, check=True).stdout)
        work = Path(tempfile.mkdtemp(dir=scratch))
        t0 = time.time()
        # stream: s3 body -> FIFO -> tee(sha256) -> zstd -d -> tar -x.
        # The body MUST go to a FIFO, not /dev/stdout: `s3api get-object` also prints its response
        # metadata JSON on stdout, which would corrupt the stream and its hash.
        pipe = subprocess.run(['bash', '-c', _STREAM.format(
            work=work, bucket=bucket, key=man['key'], vid=man['version_id'], strip='', dest=work)],
            capture_output=True, text=True)
        got_stream = (work / '.sha').read_text().split()[0] if (work / '.sha').exists() else None
        bad, checked = [], 0
        for f in man['files']:
            path = work / man['group'] / f['rel']
            if f['type'] == 'symlink':
                ok = path.is_symlink() and os.readlink(path) == f['target']
            else:
                ok = path.is_file() and not path.is_symlink() and path.stat().st_size == f['size'] \
                    and sha_file(path) == f['sha256']
            checked += 1
            if not ok:
                bad.append(f['rel'])
        rec = {'chunk_id': man['chunk_id'], 'key': man['key'], 'version_id': man['version_id'],
               'stream_sha256_expected': man['archive_sha256'], 'stream_sha256_observed': got_stream,
               'files_checked': checked, 'files_bad': bad[:50], 'n_bad': len(bad), 'pipe_rc': pipe.returncode,
               'stderr': pipe.stderr[-2000:], 'seconds': round(time.time() - t0, 1), 'verified_utc': utcnow(),
               'host': os.uname().nodename}
        rec['verified'] = (pipe.returncode == 0 and got_stream == man['archive_sha256'] and not bad
                           and checked == man['n_files'] and not man['errors'])
        write_json_atomic(out_dir / f'{man["chunk_id"]}.verify.json', rec)
        subprocess.run(['aws', 's3', 'cp', str(out_dir / f'{man["chunk_id"]}.verify.json'),
                        f's3://{bucket}/{prefix}/verification/{man["chunk_id"]}.verify.json',
                        '--only-show-errors', '--sse', 'AES256'], check=False)
        subprocess.run(['rm', '-rf', str(work)], check=False)
        print(man['chunk_id'], 'VERIFIED' if rec['verified'] else 'FAILED', rec['seconds'], flush=True)


# ------------------------------------------------------------------ guarded delete (local)
def open_paths_under(root):
    r = subprocess.run(['lsof', '-Fn', '+c0'], capture_output=True, text=True)
    return {l[1:] for l in r.stdout.splitlines() if l.startswith('n' + str(root))}


def delete(plan_path, manifest_dir, verify_dir, ledger, owner, archive_uri_prefix, dry_run=False):
    p = json.load(open(plan_path))
    groups = {g['label']: g for g in p['groups']}
    ledger = Path(ledger)
    total = 0
    for chunk in p['chunks']:
        mpath = Path(manifest_dir) / f'{chunk["chunk_id"]}.json'
        vpath = Path(verify_dir) / f'{chunk["chunk_id"]}.verify.json'
        if not (mpath.exists() and vpath.exists()):
            continue
        man, ver = json.load(open(mpath)), json.load(open(vpath))
        if not (man.get('upload_ok') and ver.get('verified') and ver['stream_sha256_observed'] == man['archive_sha256']
                and ver['version_id'] == man['version_id']):
            print('not verified, keep', chunk['chunk_id'])
            continue
        root = Path(groups[man['group']]['root']).resolve()
        busy = open_paths_under(root)
        freed, skipped = 0, []
        for f in man['files']:
            if f['type'] != 'file':
                continue
            path = root / f['rel']
            try:
                st = os.lstat(path)
            except FileNotFoundError:
                continue
            real = Path(os.path.realpath(path))
            reason = None
            if os.path.islink(path) or not real.is_relative_to(root) or real != path:
                reason = 'symlink_or_escape'
            elif (st.st_size, st.st_mtime_ns, st.st_ino) != (f['size'], f['mtime_ns'], f['ino']):
                reason = 'changed_since_manifest'
            elif str(path) in busy:
                reason = 'open_by_process'
            elif sha_file(path) != f['sha256']:
                reason = 'content_hash_changed'
            if reason:
                skipped.append({'rel': f['rel'], 'reason': reason})
                continue
            if not dry_run:
                os.unlink(path)
            freed += st.st_blocks * 512
        rec = {'utc': utcnow(), 'owner': owner, 'chunk_id': man['chunk_id'], 'group': man['group'],
               'reason': 'archived + remotely verified (per-file sha256); exact manifest-listed copies unlinked',
               'archive': f'{archive_uri_prefix}/{man["key"]}', 'version_id': man['version_id'],
               'n_files_manifest': man['n_files'], 'n_skipped': len(skipped), 'skipped': skipped[:100],
               'allocated_bytes_freed': freed, 'dry_run': dry_run,
               'restore': f'infra/pcrl_utility_extension_v1/restore.sh {man["chunk_id"]}'}
        with open(ledger, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        total += freed
        print(chunk['chunk_id'], f'freed {freed / 2 ** 30:.2f} GiB', f'skipped {len(skipped)}', flush=True)
    print(f'TOTAL freed {total / 2 ** 30:.2f} GiB (dry_run={dry_run})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    a = sp.add_parser('plan'); a.add_argument('groups'); a.add_argument('out')
    a = sp.add_parser('upload'); a.add_argument('plan'); a.add_argument('--bucket', required=True)
    a.add_argument('--prefix', required=True); a.add_argument('--manifests', required=True)
    a.add_argument('--chunks', nargs='*'); a.add_argument('--threads', type=int, default=2)
    a = sp.add_parser('verify'); a.add_argument('--bucket', required=True); a.add_argument('--prefix', required=True)
    a.add_argument('--manifest-keys', nargs='+', required=True); a.add_argument('--scratch', required=True)
    a.add_argument('--out', required=True)
    a = sp.add_parser('delete'); a.add_argument('plan'); a.add_argument('--manifests', required=True)
    a.add_argument('--verify-dir', required=True); a.add_argument('--ledger', required=True)
    a.add_argument('--owner', default='terminal_1'); a.add_argument('--archive-uri', required=True)
    a.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.cmd == 'plan':
        plan(json.load(open(args.groups)), args.out)
    elif args.cmd == 'upload':
        upload(args.plan, set(args.chunks or []), args.bucket, args.prefix, args.manifests, threads=args.threads)
    elif args.cmd == 'verify':
        verify(args.bucket, args.prefix, args.manifest_keys, args.scratch, args.out)
    elif args.cmd == 'delete':
        delete(args.plan, args.manifests, args.verify_dir, args.ledger, args.owner, args.archive_uri, args.dry_run)
