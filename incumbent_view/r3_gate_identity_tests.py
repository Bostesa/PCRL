"""Data-free regression for protected score gates in the witness queue stage."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .r3_gate_worker import run,sha


class GateIdentityRegression(unittest.TestCase):
    def test_protected_arm_reaches_four_source_inventory_in_witness_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            config=root/'config.json'
            config.write_text(json.dumps({'schema':'pcrl-incumbent-r3-gate-config-v1',
                                          'unit_id':'fixture','arm_id':'O_J','source_units':{},
                                          'output':str(root/'never_written.json')}))
            state={'units':{'fixture':{'spec':{'sources':[{'path':str(config),'sha256':sha(config)}]}}}}
            (root/'state.json').write_text(json.dumps(state))
            active={'unit_id':'fixture','stage':'witness','seal_sha256':'a'*64}
            with patch.dict(os.environ,{'PCRL_QUEUE_ROOT':str(root)}),patch(
                    'incumbent_view.r3_gate_worker.assert_active_from_env',return_value=active):
                with self.assertRaisesRegex(PermissionError,'four source units required'):
                    run(config)


if __name__=='__main__':unittest.main()
