"""Synthetic timing cannot bypass its global R03 slot."""

from pathlib import Path
import unittest
from unittest.mock import patch

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls import (
    smoke_diagnostic_release as smoke,
)


class DiagnosticSmokeTest(unittest.TestCase):
    def test_direct_launch_refuses_before_bank_construction(self):
        with patch.object(smoke, "assert_active_from_env",
                          side_effect=RuntimeError("no queue slot")), \
             patch.object(smoke, "_bank") as bank:
            with self.assertRaisesRegex(RuntimeError, "no queue slot"):
                smoke.run(unit_id="A-DIAG-SMOKE", arm_id="A_CLASS_ONLY",
                          role="ATTACK_FIT_NEW", rows=4064,
                          output_path=Path("absent"))
        bank.assert_not_called()

    def test_role_shapes_match_predeclared_synthetic_units(self):
        self.assertEqual(smoke.ROLE_SHAPES["ATTACK_FIT_NEW"], (4064, 4061))
        self.assertEqual(smoke.ROLE_SHAPES["INNER_SELECTION"], (2235, 2234))


if __name__ == "__main__":
    unittest.main()
