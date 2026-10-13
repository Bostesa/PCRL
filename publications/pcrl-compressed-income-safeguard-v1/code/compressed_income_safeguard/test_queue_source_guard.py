"""Prove byte hashing refuses barred role paths before any file open."""
from pathlib import Path
import pytest
from .scheduler import digest

@pytest.mark.parametrize('role',['HEAD_VALIDATION','OSF_DEVELOPMENT_ASSESSMENT','all_role'])
def test_hash_rejects_forbidden_role_before_open(role):
    path=Path.cwd()/'finite_guard_missing_fixture'/role/'labels.npz'
    with pytest.raises(PermissionError):digest(path)
