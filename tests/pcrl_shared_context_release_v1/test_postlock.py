"""Post-lock original-object restore: gate first, exact pins, SHA-verified, private."""
from __future__ import annotations

import hashlib
import json
import stat

import pytest

from experiments.pcrl_shared_context_release_v1 import audit_panel, postlock


def fake_index(tmp_path, payloads):
    anchors = {}
    for anchor, blob in payloads.items():
        sha = hashlib.sha256(blob).hexdigest()
        anchors[str(anchor)] = {"prepared": {"sha256": sha, "archive": {
            "member_sha256": sha, "member_path": f"results/x/private/run/anchor_{anchor}/prepared.joblib"}}}
    return {"anchors": anchors}


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    payloads = {a: f"original-{a}".encode() for a in (0, 1, 2)}
    pins = {a: {"key": f"k{a}", "version_id": f"v{a}", "bytes": len(b),
                "sha256": hashlib.sha256(b).hexdigest()} for a, b in payloads.items()}
    monkeypatch.setattr(postlock, "ORIGINAL_OBJECTS", pins)
    index = fake_index(tmp_path, payloads)
    monkeypatch.setattr(postlock.data, "index", lambda path: index)
    monkeypatch.setattr(postlock.data, "member_record",
                        lambda value, anchor, kind: value["anchors"][str(anchor)][kind])
    monkeypatch.setattr(postlock, "RESTORE_RECEIPT", tmp_path / "private" / "ORIGINAL_RESTORE.json")
    by_key = {pins[a]["key"]: payloads[a] for a in payloads}
    fetched = []

    def fetch(key, version, target):
        fetched.append((key, version))
        target.write_bytes(by_key[key])
    return fetch, fetched, by_key


def test_restore_refuses_without_gate_and_touches_nothing(tmp_path, pinned):
    fetch, fetched, _ = pinned
    root = tmp_path / "private" / "original_2018_restore"
    with pytest.raises(PermissionError):
        postlock.restore_outer("0" * 64, fetch=fetch, restore_root=root)  # real gate: no lock
    assert fetched == [] and not root.exists()


def test_restore_after_gate_is_exact_verified_and_private(tmp_path, pinned):
    fetch, fetched, _ = pinned
    root = tmp_path / "private" / "original_2018_restore"
    gate_calls = []
    receipt = postlock.restore_outer("f" * 64, fetch=fetch, restore_root=root,
                                     gate=lambda path, sha: gate_calls.append((path, sha)))
    assert gate_calls == [(audit_panel.LOCK_PATH, "f" * 64)]
    assert fetched == [("k0", "v0"), ("k1", "v1"), ("k2", "v2")]
    target = root / "results/x/private/run/anchor_1/prepared.joblib"
    assert target.read_bytes() == b"original-1"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert receipt["all_verified"] and json.loads(postlock.RESTORE_RECEIPT.read_text())["lock_sha256"] == "f" * 64
    again = postlock.restore_outer("f" * 64, fetch=fetch, restore_root=root, gate=lambda p, s: None)
    assert {r["action"] for r in again["records"]} == {"already_present_verified"}


def test_restore_rejects_bytes_that_differ_from_the_pin(tmp_path, pinned):
    _, _, _ = pinned
    root = tmp_path / "private" / "original_2018_restore"
    with pytest.raises(ValueError, match="differs from its SHA-256 pin"):
        postlock.restore_outer("f" * 64, restore_root=root, gate=lambda p, s: None,
                               fetch=lambda key, version, target: target.write_bytes(b"tampered!!"))
    assert not list(root.rglob("*.joblib")) and not list(root.rglob("*.partial"))


def test_registered_pins_match_the_pinned_index():
    value = json.loads((audit_panel.ROOT / postlock.data.INDEX_RELATIVE).read_text())
    for anchor, pin in postlock.ORIGINAL_OBJECTS.items():
        assert value["anchors"][str(anchor)]["prepared"]["sha256"] == pin["sha256"]
        assert pin["key"].endswith(f"original_prepared_anchor_{anchor}.joblib")
