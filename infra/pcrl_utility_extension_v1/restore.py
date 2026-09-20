"""Restore archive chunks IN PLACE (group root layout) and verify every file against its manifest.

  python restore.py --bucket B --prefix P --chunks exec_main__0000 ... --roots roots.json --verify-dir D
Streams s3 -> sha256 of the compressed object -> zstd -d -> tar -x (first path component, the
group label, is stripped into the group's root). Then re-hashes every manifest file in place.
A mismatch is recorded and the chunk is marked not verified; nothing is silently accepted.
"""
import argparse, hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import archive
from archive import sha_file, write_json_atomic, utcnow


def restore(bucket, prefix, chunk, roots, verify_dir):
    mkey = f'{prefix}/manifests/{chunk}.json'
    man = json.loads(subprocess.run(['aws', 's3', 'cp', f's3://{bucket}/{mkey}', '-'], capture_output=True, check=True).stdout)
    root = Path(roots[man['group']])
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    work = tempfile.mkdtemp()
    p = subprocess.run(['bash', '-c', archive._STREAM.format(
        work=work, bucket=bucket, key=man['key'], vid=man['version_id'],
        strip='--strip-components=1', dest=root)], capture_output=True, text=True)
    got = open(f'{work}/.sha').read().split()[0] if os.path.exists(f'{work}/.sha') else None
    bad = []
    for f in man['files']:
        path = root / f['rel']
        if f['type'] == 'symlink':
            ok = path.is_symlink() and os.readlink(path) == f['target']
        else:
            ok = path.is_file() and path.stat().st_size == f['size'] and sha_file(path) == f['sha256']
        if not ok:
            bad.append(f['rel'])
    rec = {'chunk_id': chunk, 'key': man['key'], 'version_id': man['version_id'], 'mode': 'restore_in_place',
           'stream_sha256_expected': man['archive_sha256'], 'stream_sha256_observed': got,
           'files_checked': man['n_files'], 'n_bad': len(bad), 'files_bad': bad[:50], 'pipe_rc': p.returncode,
           'stderr': p.stderr[-1500:], 'seconds': round(time.time() - t0, 1), 'verified_utc': utcnow(),
           'host': os.uname().nodename}
    rec['verified'] = p.returncode == 0 and got == man['archive_sha256'] and not bad and not man['errors']
    write_json_atomic(Path(verify_dir) / f'{chunk}.verify.json', rec)
    subprocess.run(['aws', 's3', 'cp', str(Path(verify_dir) / f'{chunk}.verify.json'),
                    f's3://{bucket}/{prefix}/verification/{chunk}.verify.json', '--only-show-errors', '--sse', 'AES256'])
    print(chunk, 'RESTORED+VERIFIED' if rec['verified'] else 'FAILED', rec['seconds'], flush=True)
    return rec['verified']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bucket', required=True); ap.add_argument('--prefix', required=True)
    ap.add_argument('--chunks', nargs='+', required=True); ap.add_argument('--roots', required=True)
    ap.add_argument('--verify-dir', required=True)
    a = ap.parse_args()
    roots = json.load(open(a.roots))
    ok = all([restore(a.bucket, a.prefix, c, roots, a.verify_dir) for c in a.chunks])
    sys.exit(0 if ok else 1)
