"""Data-free operator wrapper failure checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .dispatch import run_unit
from .queue import sha256


class DispatchTests(unittest.TestCase):
    def test_preexisting_queued_spec_must_equal_sealed_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            study = base / "repo/results/pcrl_capacity_matched_feature_selection_v1"
            private = study / ".private"
            private.mkdir(parents=True)
            prior_slot = (base / "pcrl-method-portfolio-v1/results/"
                          "pcrl_method_portfolio_v1/shared/.private/compute/slot0.lock")
            prior_slot.parent.mkdir(parents=True)
            prior_slot.touch()
            (study / "RESOURCE_PLAN.json").write_text(json.dumps({
                "epoch_start_utc": "2026-10-08T23:18:01+00:00"}))
            seal = study / "LOCAL_SEAL_RECEIPT.json"
            seal.write_text('{"status":"PASS"}')
            registry = private / "UNIT_SPECS.json"
            registry.write_text(json.dumps({"unit_specs": {
                "TEST-UNIT": {"unit_id": "TEST-UNIT",
                              "seal_sha256": "$LOCAL_SEAL_SHA256"}}}))
            (study / "CAMPAIGN_LOCK.json").write_text(json.dumps({
                "private_unit_registry_sha256": sha256(registry)}))
            altered = private / "altered.json"
            altered.write_text('{"unit_id":"TEST-UNIT","altered":true}')

            class FakeQueue:
                def __init__(self, *args, **kwargs):
                    self.ran = False

                def admit_science(self, **kwargs):
                    pass

                def status(self):
                    return {"units": {"TEST-UNIT": {
                        "status": "QUEUED", "spec_path": str(altered),
                        "spec_sha256": sha256(altered)}}}

                def run_once(self, **kwargs):
                    self.ran = True
                    raise AssertionError("altered queued unit was launched")

            with patch("results.pcrl_capacity_matched_feature_selection_v1."
                       "m0.dispatch.FocusedQueue", FakeQueue):
                with self.assertRaisesRegex(PermissionError, "preexisting queued unit"):
                    run_unit(study, "TEST-UNIT", "one external heavy fixture")


if __name__ == "__main__":
    unittest.main()
