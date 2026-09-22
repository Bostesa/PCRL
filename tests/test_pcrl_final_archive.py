"""Private archive streams must restore fitted weights and predictions exactly."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.pcrl_final_prospective_v1.archive_run import pack, verify_restore
from experiments.pcrl_final_prospective_v1.common import sha


class ArchiveRoundtripTest(unittest.TestCase):
    def test_restores_exact_files_and_detects_change(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            source.mkdir()
            rel = 'results/pcrl_final_prospective_v1/private/runs/acs2016/model.bin'
            model = source/rel
            model.parent.mkdir(parents=True)
            model.write_bytes(b'fitted model bytes')
            records = [{'path': rel, 'bytes': model.stat().st_size, 'sha256': sha(model)}]
            part = root/'part.tar.zst'
            pack(source, records, part)
            restored = root/'restored'
            result = verify_restore(part, records, restored, [rel])
            self.assertEqual(result['verified_files'], 1)
            self.assertEqual((restored/rel).read_bytes(), model.read_bytes())
            model.write_bytes(b'changed after inventory')
            with self.assertRaisesRegex(ValueError, 'source hash'):
                pack(source, records, root/'changed.tar.zst')


if __name__ == '__main__':
    unittest.main()
