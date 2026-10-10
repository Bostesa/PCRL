"""Invented SQLite input for the queued A cache custody boundary."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import unittest

from results.pcrl_latent_and_sampled_methods_v1.r18.a_cache_custody import (
    copy_closed_cache,
)


class ClosedCacheCopyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent = self.root / "completed_parent"
        self.parent.mkdir(mode=0o700)
        self.source = self.parent / "cache.sqlite3"
        with sqlite3.connect(self.source) as connection:
            connection.execute("CREATE TABLE releases (key TEXT PRIMARY KEY, z1 INTEGER, z2 INTEGER)")
            connection.execute("INSERT INTO releases VALUES ('invented', 1, 2)")
        os.chmod(self.source, 0o600)
        self.restore = self.root / "restore"
        self.restore.mkdir(mode=0o700)
        self.destination = self.restore / "cache.sqlite3"
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def test_closed_cache_copy_is_fresh_private_and_byte_exact(self) -> None:
        actual = copy_closed_cache(self.source, self.destination, self.digest)
        self.assertEqual(actual, self.digest)
        self.assertEqual(hashlib.sha256(self.destination.read_bytes()).hexdigest(), self.digest)
        self.assertEqual(stat.S_IMODE(self.destination.stat().st_mode), 0o600)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.digest)
        with sqlite3.connect(self.destination) as connection:
            self.assertEqual(connection.execute("SELECT z1,z2 FROM releases").fetchone(), (1, 2))

    def test_live_sidecar_refuses_before_copy(self) -> None:
        for suffix in ("-wal", "-shm", "-journal"):
            with self.subTest(suffix=suffix):
                sidecar = Path(str(self.source) + suffix)
                sidecar.write_bytes(b"synthetic uncheckpointed transaction")
                with self.assertRaisesRegex(ValueError, "sidecar"):
                    copy_closed_cache(self.source, self.destination, self.digest)
                self.assertFalse(self.destination.exists())
                sidecar.unlink()

    def test_source_symlink_or_public_mode_refuses(self) -> None:
        alias = self.parent / "alias.sqlite3"
        alias.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "symlink"):
            copy_closed_cache(alias, self.destination, self.digest)
        self.assertFalse(self.destination.exists())
        os.chmod(self.source, 0o644)
        with self.assertRaisesRegex(PermissionError, "private"):
            copy_closed_cache(self.source, self.destination, self.digest)
        self.assertFalse(self.destination.exists())

    def test_wrong_hash_or_existing_destination_refuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "hash"):
            copy_closed_cache(self.source, self.destination, "0" * 64)
        self.assertFalse(self.destination.exists())
        self.destination.write_bytes(b"prior snapshot")
        with self.assertRaisesRegex(FileExistsError, "exists"):
            copy_closed_cache(self.source, self.destination, self.digest)
        self.assertEqual(self.destination.read_bytes(), b"prior snapshot")


if __name__ == "__main__":
    unittest.main()
