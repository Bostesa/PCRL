"""Synthetic archive-only regression: no live artifacts or AWS operations."""
import hashlib
import io
import json
import subprocess
import tarfile

import pytest

from experiments.pcrl_task_directed_release_v1 import config

from experiments.pcrl_task_directed_release_v1 import archive

KEY = 'audit/1/Trisk_rr_0.75_a33'
DIGEST = 'd0a1e406e19a5a47b01bd7db5aabc8b3c1bd75be5145fb4582a7e28201653c66'
NAME = f'results/{config.STUDY}/private/locks/{DIGEST}.lock'


def content(**updates):
    value = {'pid': 12345, 'key': KEY, 'utc': '2026-09-22T05:14:14.758034+00:00'}
    value.update(updates)
    return json.dumps(value).encode()


def record(name, data):
    return {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def write_source(tmp_path, data, name=NAME):
    root = tmp_path/'source'; path = root/name
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    return root, path


def untrusted_archive(path, name, data, *, kind=tarfile.REGTYPE, linkname=''):
    # Build a synthetic adversarial stream without going through the candidate packer.
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w') as handle:
        member = tarfile.TarInfo(name); member.type = kind; member.linkname = linkname
        member.size = len(data) if kind == tarfile.REGTYPE else 0
        handle.addfile(member, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    with path.open('xb') as output:
        subprocess.run(['zstd', '-q', '-c'], input=raw.getvalue(), stdout=output, check=True)


def test_actual_scheduled_key_hash_roundtrips_inventory_pack_stream_and_restore(tmp_path):
    assert hashlib.sha256(KEY.encode()).hexdigest() == DIGEST and '2016' in DIGEST
    assert archive.safe_relative(NAME).as_posix() == NAME
    data = content(); root, path = write_source(tmp_path, data)
    records = archive.inventory(root)
    assert records == [record(NAME, data)]
    packed = tmp_path/'part.tar.zst'; archive.pack(root, records, packed)
    assert archive.verify_and_restore(packed, records)['verified_files'] == 1
    restore = tmp_path/'restored'
    result = archive.verify_and_restore(packed, records, restore_root=restore,
        restore_prefixes=[f'results/{config.STUDY}/private/locks'])
    assert result['restored_files'] == 1 and (restore/NAME).read_bytes() == path.read_bytes()
    with pytest.raises(ValueError, match='overwrites'):
        archive.verify_and_restore(packed, records, restore_root=restore, restore_prefixes=[NAME])


@pytest.mark.parametrize('name', [
    'data/2016/file.csv', 'x/ss16pca.npz', '../'+NAME, '/'+NAME,
    NAME.replace('/locks/', '/other/'), NAME.replace(config.STUDY, 'other_study'),
    NAME+'.json', NAME+'/payload', NAME.replace(DIGEST, DIGEST.upper()),
    NAME.replace(DIGEST, DIGEST[:-1]), NAME.replace('/locks/', '/locks/2016/'),
    NAME.replace('/locks/', '/locks//'), NAME.replace('/locks/', '/locks/../locks/'),
    NAME.replace('/locks/', '/locks\\'), NAME+'\x00',
])
def test_exception_does_not_expand_to_other_paths(name):
    with pytest.raises(ValueError):
        archive.safe_relative(name)


BAD = [b'not JSON', b'', b'[]', content(key='audit/0/other'), content(pid=True),
       content(pid=0), content(pid=-1), content(pid=1.5), content(utc='not a date'),
       content(utc='2026-09-22T05:14:14'), content(utc='2026-09-22T05:14:14+02:00'),
       content(extra='disguised payload'), content(key='data/2016/secret.csv'),
       b'{"pid":1,"pid":12345,"key":"audit/1/Trisk_rr_0.75_a33","utc":"2026-09-22T05:14:14+00:00"}',
       b'x'*4097]


@pytest.mark.parametrize('data', BAD)
def test_inventory_and_pack_reject_misleading_lock_contents(tmp_path, data):
    root, _ = write_source(tmp_path, data)
    with pytest.raises(ValueError, match='lock|Lock'):
        archive.inventory(root)
    with pytest.raises(ValueError, match='lock|Lock'):
        archive.pack(root, [record(NAME, data)], tmp_path/'bad.tar.zst')


@pytest.mark.parametrize('restore', [False, True])
@pytest.mark.parametrize('data', [content(key='audit/0/other'), content(extra='hidden'), b'[]', b'x'*4097])
def test_stream_validation_rejects_bad_lock_even_when_manifest_hash_matches(tmp_path, data, restore):
    packed = tmp_path/'untrusted.tar.zst'; untrusted_archive(packed, NAME, data)
    target = tmp_path/'restore'
    with pytest.raises(ValueError, match='lock|Lock'):
        archive.verify_and_restore(packed, [record(NAME, data)],
            restore_root=target if restore else None, restore_prefixes=[NAME] if restore else [])
    assert not (target/NAME).exists()
    assert not list(target.rglob('*.restore-tmp'))


def test_hash_matching_sealed_or_non_job_key_does_not_disguise_contents(tmp_path):
    for index, key in enumerate(('audit/1/forbidden_2016', 'data/2016/secret.csv', '../outside', 'arbitrary_payload')):
        digest = hashlib.sha256(key.encode()).hexdigest()
        name = f'results/{config.STUDY}/private/locks/{digest}.lock'
        root, _ = write_source(tmp_path/str(index), content(key=key), name)
        with pytest.raises(ValueError, match='lock|Lock'):
            archive.inventory(root)


def test_valid_noncollision_lock_is_checked_too(tmp_path):
    key = 'programme/selection_freeze'
    name = f'results/{config.STUDY}/private/locks/{hashlib.sha256(key.encode()).hexdigest()}.lock'
    root, _ = write_source(tmp_path, content(key=key), name)
    assert len(archive.inventory(root)) == 1
    (root/name).write_bytes(content(key='different'))
    with pytest.raises(ValueError, match='lock|Lock'):
        archive.inventory(root)


@pytest.mark.parametrize('key', [
    'prepare/0', 'map/2/Trisk_C_0.01_a33', 'evaluation/1/constant_best',
    'benchmark', 'programme/selection_freeze', 'programme/final_evidence',
    'branch/A/registration', 'branch/C/resource_schedule', 'branch/A/tables/2',
    'baseline_supplement/registration', 'baseline_supplement/schedule',
    'baseline_supplement/0/mechanism40', 'baseline_supplement/2/union88',
])
def test_all_registered_lock_namespaces_are_preserved(tmp_path, key):
    name = f'results/{config.STUDY}/private/locks/{hashlib.sha256(key.encode()).hexdigest()}.lock'
    data = content(key=key)
    root, _ = write_source(tmp_path, data, name)
    assert archive.inventory(root) == [record(name, data)]


def test_pack_serializes_the_validated_lock_bytes(tmp_path, monkeypatch):
    data = content(); root, path = write_source(tmp_path, data)
    original = archive._read_lock_metadata
    def mutate_after_validated_read(name, source):
        result = original(name, source)
        source.write_bytes(content(extra='unvalidated later bytes'))
        return result
    monkeypatch.setattr(archive, '_read_lock_metadata', mutate_after_validated_read)
    packed = tmp_path/'snapshot.tar.zst'
    archive.pack(root, [record(NAME, data)], packed)
    target = tmp_path/'restored'
    archive.verify_and_restore(packed, [record(NAME, data)], restore_root=target, restore_prefixes=[NAME])
    assert (target/NAME).read_bytes() == data and path.read_bytes() != data


def test_task_lock_symlinks_remain_forbidden_at_pack_and_restore(tmp_path):
    data = content(); root, path = write_source(tmp_path, data)
    real = root/'real'; path.rename(real); path.symlink_to(real)
    with pytest.raises(ValueError, match='symbolic'):
        archive.pack(root, [record(NAME, data)], tmp_path/'link.tar.zst')
    bad = tmp_path/'tar-link.tar.zst'
    untrusted_archive(bad, NAME, b'', kind=tarfile.SYMTYPE, linkname='/outside')
    with pytest.raises(ValueError, match='member'):
        archive.verify_and_restore(bad, [record(NAME, b'')])
