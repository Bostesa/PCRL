"""Data-free checks for private diagnostics in restored CLI replay."""

import json
import stat
from types import SimpleNamespace

import pytest

from bank_anchored import r1_restore_replay_v2 as replay


def test_nested_cli_failure_stays_in_owner_only_diagnostic(tmp_path, monkeypatch):
    output = tmp_path / "private"
    output.mkdir(mode=0o700)
    monkeypatch.setattr(replay.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="private stdout", stderr="private source traceback"))
    with pytest.raises(RuntimeError, match="private diagnostic SHA-256") as error:
        replay._run_restored_cli(["python", "-m", "bank_anchored.r1_deploy"],
                                 code_root=tmp_path, env={}, output=output,
                                 seed=0, arm="B0", purpose=1)
    assert "private source traceback" not in str(error.value)
    diagnostic = output / "NESTED_CLI_FAILURE.json"
    saved = json.loads(diagnostic.read_text())
    assert saved["stderr"] == "private source traceback"
    assert saved["seed"] == 0 and saved["purpose"] == 1
    assert stat.S_IMODE(diagnostic.stat().st_mode) == 0o600


def test_nested_cli_success_parses_only_json_stdout(tmp_path, monkeypatch):
    monkeypatch.setattr(replay.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='{"schema":"ok"}', stderr=""))
    assert replay._run_restored_cli(["python"], code_root=tmp_path, env={},
                                    output=tmp_path, seed=1, arm="J", purpose=0) == {
                                        "schema": "ok"}
    assert not (tmp_path / "NESTED_CLI_FAILURE.json").exists()
