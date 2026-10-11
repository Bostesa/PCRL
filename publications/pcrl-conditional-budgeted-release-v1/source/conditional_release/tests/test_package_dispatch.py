"""Hash-before-dispatch plumbing; no fitting or person arrays."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from conditional_release.source import sha_file
from conditional_release.deployment import load_control_package

class Tests(unittest.TestCase):
    def test_narrow_wide_dispatch(self):
        for schema,target in (('CBR1-control-policy-v1','conditional_release.controls.load_control_policy'),
                              ('CBR1-wide-control-policy-v1','conditional_release.control_wide_policy.load_wide_control_policy')):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'POLICY_RECEIPT.json';p.write_text(json.dumps({'schema':schema}));sha=sha_file(p)
                with patch(target,return_value='admitted') as mock:
                    self.assertEqual(load_control_package(d,{},sha),'admitted')
                    mock.assert_called_once_with(d,{},expected_receipt_sha256=sha)
    def test_missing_or_wrong_hash_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'POLICY_RECEIPT.json';p.write_text(json.dumps({'schema':'unregistered'}))
            with self.assertRaises(PermissionError):load_control_package(d,{},'0'*64)
            with self.assertRaises(PermissionError):load_control_package(d,{},sha_file(p))

if __name__=='__main__':unittest.main()
