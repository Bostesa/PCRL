"""Safety guards: no network, no scientific fits without explicit authorisation, caches outside git."""
from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path


class ScientificFitRefused(PermissionError):
    pass


class NetworkRefused(PermissionError):
    pass


@dataclass(frozen=True)
class FitAuthorization:
    """Who may fit an attacker / utility probe. Synthetic fixtures are always allowed; real inputs need
    the explicit --execute-scientific-fits flag."""
    synthetic: bool = False
    execute_scientific_fits: bool = False

    @classmethod
    def synthetic_only(cls):
        return cls(synthetic=True)

    def check(self, what: str, data_is_synthetic: bool) -> None:
        if data_is_synthetic and self.synthetic:
            return
        if not data_is_synthetic and self.execute_scientific_fits:
            return
        raise ScientificFitRefused(
            f"refusing to fit {what}: "
            + ("non-synthetic inputs require --execute-scientific-fits" if not data_is_synthetic
               else "synthetic fits require --synthetic"))


_ORIG_SOCKET = socket.socket


def install_network_guard() -> None:
    """Make any socket creation in this process raise (the evaluator never acquires data)."""

    class _NoSocket(_ORIG_SOCKET):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **k):
            raise NetworkRefused("stored_model_eval never opens network connections")

    socket.socket = _NoSocket  # type: ignore[misc]
    socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(  # type: ignore[assignment]
        NetworkRefused("stored_model_eval never opens network connections"))


def inside_git_worktree(path: Path) -> Path | None:
    """Return the enclosing git work tree root if `path` (or its nearest existing parent) is inside one."""
    p = Path(path).expanduser().resolve()
    while not p.exists():
        p = p.parent
    for q in [p, *p.parents]:
        if (q / ".git").exists():
            return q
    return None


def require_outside_git(path: Path, what: str = "cache") -> Path:
    root = inside_git_worktree(path)
    if root is not None:
        raise PermissionError(f"{what} directory {path} is inside git work tree {root}; "
                              "model outputs and row-level arrays must live outside git")
    return Path(path).expanduser().resolve()
