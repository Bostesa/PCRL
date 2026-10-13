import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .attack_worker import _policy


class TestReaderControlDispatch(unittest.TestCase):
    def test_bound_narrow_and_wide_loaders(self):
        f=Path(__file__).resolve().parents[1]/"results/pcrl_conditional_budgeted_release_v1/r5/CONTROL_DISPATCH_FIXTURES.json"
        fixture=json.loads(f.read_text())
        with tempfile.TemporaryDirectory() as d:
            receipt=Path(d)/"POLICY_RECEIPT.json"
            manifest={"engineering":"no roles"}
            for case in fixture["accepted"]:
                with self.subTest(schema=case["package_schema"]):
                    receipt.write_text(json.dumps({"schema":case["package_schema"]}))
                    digest=hashlib.sha256(receipt.read_bytes()).hexdigest()
                    cfg={"type":"control","directory":d,"receipt_sha256":digest}
                    module=importlib.import_module(case["module"])
                    sentinel=object()
                    with patch.object(module,case["loader"],return_value=sentinel) as loader:
                        self.assertIs(_policy(cfg,manifest),sentinel)
                        loader.assert_called_once_with(d,manifest,expected_receipt_sha256=digest)
            receipt.write_text(json.dumps({"schema":fixture["rejected_schema"]}))
            cfg={"type":"control","directory":d,"receipt_sha256":hashlib.sha256(receipt.read_bytes()).hexdigest()}
            with self.assertRaises(PermissionError):_policy(cfg,manifest)
            cfg["receipt_sha256"]=fixture["rejected_hash"]
            with self.assertRaises(PermissionError):_policy(cfg,manifest)


if __name__=="__main__":unittest.main()
