"""Data-free dual-lock bridge fixtures; R0 runs this under the shared queue."""
import json
import tempfile
import unittest
from pathlib import Path
from .r3_worker import lock_bridge,sha


class LockBridgeFixtures(unittest.TestCase):
    def test_witness_stays_on_admission(self):
        admission='a'*64
        cfg={'arm_id':'P0','pins':{'seal':{'sha256':admission}}}
        active={'stage':'witness','seal_sha256':admission}
        self.assertEqual(lock_bridge(cfg,active,{'seal':{'sha256':admission}}, {},{}),admission)

    def test_protected_science_amendment_then_dated_successor(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);admission='a'*64
            science_path=root/'science.json';science_path.write_text('{}')
            science=sha(science_path)
            successor=root/'amendment.json';successor.write_text(json.dumps({'campaign_lock_sha256':science}))
            newer=sha(successor)
            cfg={'arm_id':'V_J','pins':{'seal':{'sha256':science},'admission_lock':{'sha256':admission}}}
            active={'stage':'witness','seal_sha256':admission}
            state={'seal':{'sha256':admission},'amendments':[
                {'path':str(science_path),'sha256':science,'stages':['protection']},
                {'path':str(successor),'sha256':newer,'stages':['witness']}]}
            registered={successor.resolve():newer}
            self.assertEqual(lock_bridge(cfg,active,state,{'amendment_sha256':newer},registered),science)
            with self.assertRaises(PermissionError):lock_bridge(cfg,active,state,{'amendment_sha256':science},registered)
            with self.assertRaises(PermissionError):lock_bridge(cfg,{'stage':'witness','seal_sha256':science},state,{'amendment_sha256':newer},registered)
            successor.write_text(json.dumps({'campaign_lock_sha256':'b'*64}))
            state['amendments'][-1]['sha256']=sha(successor)
            registered={successor.resolve():sha(successor)}
            with self.assertRaises(PermissionError):lock_bridge(cfg,active,state,{'amendment_sha256':sha(successor)},registered)

    def test_unregistered_science_refused(self):
        admission='a'*64;science='b'*64
        cfg={'arm_id':'V_J','pins':{'seal':{'sha256':science},'admission_lock':{'sha256':admission}}}
        active={'stage':'witness','seal_sha256':admission}
        state={'seal':{'sha256':admission},'amendments':[]}
        with self.assertRaises(PermissionError):lock_bridge(cfg,active,state,{}, {})


if __name__=='__main__':unittest.main()
