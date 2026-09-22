import hashlib
from pathlib import Path
import pytest
from experiments.pcrl_task_directed_release_v1.archive import pack,verify_and_restore,safe_relative,chunk_records


def test_archive_complete_member_hash_readback_and_restore(tmp_path):
 root=tmp_path/'source';root.mkdir();(root/'weights.bin').write_bytes(b'unique model weights')
 record={'path':'weights.bin','bytes':20,'sha256':hashlib.sha256(b'unique model weights').hexdigest()}
 archive=tmp_path/'part.tar.zst';info=pack(root,[record],archive)
 result=verify_and_restore(archive,[record],restore_root=tmp_path/'restored',restore_prefixes=['weights.bin'])
 assert result['verified_files']==result['restored_files']==1
 assert (tmp_path/'restored/weights.bin').read_bytes()==(root/'weights.bin').read_bytes()
 assert info['files']==1
 bad={**record,'sha256':'0'*64}
 with pytest.raises(ValueError,match='hash mismatch'):verify_and_restore(archive,[bad])
 with pytest.raises(ValueError,match='overwrites'):verify_and_restore(archive,[record],restore_root=tmp_path/'restored',restore_prefixes=['weights.bin'])


def test_archive_paths_and_chunking():
 for path in ('../credentials','/absolute','data/2016/file.csv','x/ss16pca.npz'):
  with pytest.raises(ValueError):safe_relative(path)
 assert len(chunk_records([{'bytes':7},{'bytes':7},{'bytes':3}],10))==2
