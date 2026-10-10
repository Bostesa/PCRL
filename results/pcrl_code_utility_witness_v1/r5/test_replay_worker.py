"""Data-free fail-closed test for R5's numerical queue boundary."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from results.pcrl_code_utility_witness_v1.r5.replay_worker import run


class ReplayWorkerTest(unittest.TestCase):
    def test_direct_launch_refuses_before_private_config_read(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                run(Path("/this/private/config/does/not/exist.json"))


if __name__ == "__main__":
    unittest.main()
