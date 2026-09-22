"""Independent archive integrity regressions; all content is synthetic, no AWS."""
import hashlib
import json

import pytest

from experiments.pcrl_task_directed_release_v1 import archive


def record(name, content):
    return {'path': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


def test_pack_rejects_manifest_path_escape_before_reading(tmp_path):
    root = tmp_path / 'source'; root.mkdir()
    content = b'synthetic outside artifact'
    (tmp_path / 'outside.bin').write_bytes(content)
    with pytest.raises(ValueError, match='path|escape|Unsafe'):
        archive.pack(root, [record('../outside.bin', content)], tmp_path / 'part.tar.zst')


def test_restore_symlink_parent_cannot_create_any_outside_directory(tmp_path):
    root = tmp_path / 'source'; (root / 'link/new').mkdir(parents=True)
    content = b'synthetic model'; (root / 'link/new/model.bin').write_bytes(content)
    records = [record('link/new/model.bin', content)]
    packed = tmp_path / 'part.tar.zst'; archive.pack(root, records, packed)
    restore = tmp_path / 'restore'; restore.mkdir()
    outside = tmp_path / 'outside'; outside.mkdir()
    (restore / 'link').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escape|symbolic'):
        archive.verify_and_restore(packed, records, restore_root=restore, restore_prefixes=['link'])
    assert not (outside / 'new').exists()


def test_inventory_requires_every_accepted_receipt_artifact(tmp_path):
    base = tmp_path / 'results' / archive.STUDY / 'private/run/anchor_0/audits/Q'
    base.mkdir(parents=True)
    receipt = {'artifact_hashes': {'missing_model.pt': '0' * 64}}
    (base / 'COMPLETE.json').write_text(json.dumps(receipt))
    with pytest.raises((ValueError, FileNotFoundError), match='artifact|missing|No such'):
        archive.inventory(tmp_path)


def test_private_destination_requires_public_access_block_and_fresh_prefix(monkeypatch):
    calls = []
    def public(*args):
        calls.append(args)
        return {'PublicAccessBlockConfiguration': {'BlockPublicAcls': False}}
    monkeypatch.setattr(archive, 'aws_json', public)
    with pytest.raises(ValueError, match='private|public'):
        archive.validate_destination('synthetic-private-bucket', 'synthetic/archive')
    def occupied(*args):
        if args[1] == 'get-public-access-block':
            return {'PublicAccessBlockConfiguration': {k: True for k in
                ('BlockPublicAcls', 'IgnorePublicAcls', 'BlockPublicPolicy', 'RestrictPublicBuckets')}}
        return {'KeyCount': 1, 'Contents': [{'Key': 'synthetic/archive/part-0000.tar.zst'}]}
    monkeypatch.setattr(archive, 'aws_json', occupied)
    with pytest.raises(ValueError, match='empty|exists|occupied'):
        archive.validate_destination('synthetic-private-bucket', 'synthetic/archive')


def test_encrypted_put_is_conditional_and_version_pinned(monkeypatch,tmp_path):
    path = tmp_path / 'artifact'; path.write_bytes(b'synthetic')
    calls = []
    def fake(*args):
        calls.append(args)
        if args[1] == 'put-object': return {'VersionId': 'synthetic-version'}
        return {'VersionId': 'synthetic-version', 'ServerSideEncryption': 'AES256', 'ContentLength': path.stat().st_size}
    monkeypatch.setattr(archive, 'aws_json', fake)
    header = archive.encrypted_put('synthetic-private-bucket', 'synthetic/key', path)
    put, head = calls
    assert put[put.index('--if-none-match') + 1] == '*'
    assert put[put.index('--server-side-encryption') + 1] == 'AES256'
    assert head[head.index('--version-id') + 1] == 'synthetic-version'
    assert header['VersionId'] == 'synthetic-version'


def test_complete_receipt_and_deployment_inventory_is_unique(tmp_path):
    base = tmp_path / 'results' / archive.STUDY / 'private/run/anchor_0/maps/Q'
    base.mkdir(parents=True)
    content = b'complete synthetic weights'; (base / 'weights.bin').write_bytes(content)
    (base / 'ACCEPTED.json').write_text(json.dumps({'artifact_hashes': {
        'weights.bin': hashlib.sha256(content).hexdigest()}}))
    deploy = tmp_path / 'results' / archive.STUDY / 'private/deployment_inputs/anchor_0'
    deploy.mkdir(parents=True)
    (deploy / 'pca.joblib').write_bytes(content)
    (deploy / 'MANIFEST.json').write_text(json.dumps({'files': {
        'pca.joblib': hashlib.sha256(content).hexdigest()}}))
    records = archive.inventory(tmp_path)
    assert len(records) == 4 and len({r['path'] for r in records}) == 4
    (deploy / 'pca.joblib').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact'):
        archive.inventory(tmp_path)


def test_hard_linked_sources_are_archived_as_independent_regular_members(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    first = source / 'first'; first.write_bytes(b'synthetic duplicate inode')
    (source / 'second').hardlink_to(first)
    records = [record(n, first.read_bytes()) for n in ('first', 'second')]
    packed = tmp_path / 'part.tar.zst'; archive.pack(source, records, packed)
    assert archive.verify_and_restore(packed, records)['verified_files'] == 2
    with pytest.raises(ValueError, match='Duplicate'):
        archive.pack(source, [records[0], records[0]], tmp_path / 'duplicate.tar.zst')
    with pytest.raises(ValueError, match='omitted'):
        archive.verify_and_restore(packed, [*records, record('missing', b'not present')])


def test_restore_atomic_commit_cannot_overwrite_concurrent_target(tmp_path,monkeypatch):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'model').write_bytes(b'new model')
    packed = tmp_path / 'part.tar.zst'; records = [record('model', b'new model')]
    archive.pack(source, records, packed)
    link = archive.os.link
    def competing_writer(src,dst):
        dst.write_bytes(b'concurrent existing artifact')
        link(src,dst)
    monkeypatch.setattr(archive.os, 'link', competing_writer)
    with pytest.raises(FileExistsError):
        archive.verify_and_restore(packed, records, restore_root=tmp_path / 'restore', restore_prefixes=['model'])
    assert (tmp_path / 'restore/model').read_bytes() == b'concurrent existing artifact'
    assert not (tmp_path / 'restore/model.restore-tmp').exists()


def test_publish_streams_and_manifest_are_encrypted_read_back_and_scope_restored(tmp_path,monkeypatch):
    root = tmp_path / 'source'; out = root / 'results' / archive.STUDY; out.mkdir(parents=True)
    (out / 'SELECTION.json').write_text(json.dumps({'selection_frozen': True}))
    code = root / 'experiments' / archive.STUDY / 'synthetic.py'; code.parent.mkdir(parents=True)
    code.write_bytes(b'# synthetic source\n')
    model = out / 'private/run/anchor_0/model.bin'; model.parent.mkdir(parents=True)
    model.write_bytes(b'synthetic archive weights')
    records = [record(str(p.relative_to(root)), p.read_bytes()) for p in (code,model)]
    monkeypatch.setattr(archive, 'ROOT', root); monkeypatch.setattr(archive, 'OUT', out)
    monkeypatch.setattr(archive, 'inventory', lambda: records)
    objects = {}; calls = []
    def fake(*args):
        calls.append(args); command = args[1]
        if command == 'get-public-access-block':
            return {'PublicAccessBlockConfiguration': {k: True for k in
                ('BlockPublicAcls','IgnorePublicAcls','BlockPublicPolicy','RestrictPublicBuckets')}}
        if command == 'list-objects-v2': return {'KeyCount': 0}
        key = args[args.index('--key') + 1]
        if command == 'put-object':
            assert key not in objects and args[args.index('--if-none-match') + 1] == '*'
            assert args[args.index('--server-side-encryption') + 1] == 'AES256'
            objects[key] = archive.Path(args[args.index('--body') + 1]).read_bytes()
            return {'VersionId': 'v1'}
        assert args[args.index('--version-id') + 1] == 'v1'
        if command == 'head-object':
            return {'VersionId': 'v1', 'ContentLength': len(objects[key]), 'ServerSideEncryption': 'AES256'}
        assert command == 'get-object'
        archive.Path(args[-1]).write_bytes(objects[key]); return {'VersionId': 'v1'}
    monkeypatch.setattr(archive, 'aws_json', fake)
    index = archive.publish('synthetic-private-bucket', 'synthetic/task', tmp_path / 'archive',
                            tmp_path / 'restored', 'synthetic-commit')
    assert index['all_streams_and_files_verified'] and index['restore_completed']
    assert index['files'] == index['expected_restored_files'] == 2
    assert index['chunks'][0]['per_file_verified'] == index['chunks'][0]['restored_files'] == 2
    assert index['manifest_version_id'] == 'v1'
    assert (tmp_path / 'restored' / model.relative_to(root)).read_bytes() == model.read_bytes()
    assert sum(c[1] == 'put-object' for c in calls) == 2
