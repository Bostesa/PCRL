"""Per-person token-law dispatcher for every audited shared-context release.

`person_law(source, inputs)` returns the private (n, 17) law q(z | x_i) of one
pinned release on one set of original people. `source` is either a fitted unit
directory holding `release.json` (kinds "nested", "deterministic_policy",
"adv_mlp") or an in-memory spec for a predecessor kind ("historical_map" for
D17 / historical Q as 32x17 matrices indexed by the stored T0 code, "h_only"
for the shared H ancestor). J is audited by the predecessor's continuous-wire
path (`AR/external_audit.py`) and is deliberately not a kind here.

Only the six legal deployable fields may reach a law: `x` (X_A, 32), `ha`
(H_A, 4), `token_codes` (stored T0), `teacher_p`, `residual` and `risk` (11).
H_B, labels, ids, households, weights and roles are rejected before any
artifact is loaded. Every returned law is validated: shape (n, 17), float64,
finite, nonnegative, rows summing to one within 1e-9. The sole exception is
"h_only", whose law is the one-column bookkeeping wire `ones((n, 1))` used by
the inherited auditor for the H-only ancestor.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

N_TOKENS = 17
N_STATES = 32
ROW_SUM_TOL = 1e-9
DESCRIPTOR = "release.json"
DESCRIPTOR_SCHEMA = "pcrl-sc-release-descriptor-v1"
LEGAL_INPUTS = ("x", "ha", "token_codes", "teacher_p", "residual", "risk")
FORBIDDEN_INPUTS = frozenset({
    "hb", "labels", "ids", "households", "household", "weights", "role",
    "roles", "loss", "losses", "y", "s", "SEX", "RAC1P", "same_residence",
    "aux", "J"})
UNIT_KINDS = {  # kind -> module exposing load_law(unit_dir) -> callable(legal) -> (n, 17)
    "nested": "release",
    "deterministic_policy": "rd",
    "adv_mlp": "adv",
}
SPEC_KINDS = ("historical_map", "h_only")
KINDS = (*UNIT_KINDS, *SPEC_KINDS)
HISTORICAL_MAPS = ("D17", "Q")
PACKAGE = "experiments.pcrl_shared_context_release_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def law_identity(law: np.ndarray) -> str:
    """Exact structural identity: dtype, shape and float64 bytes."""
    value = np.ascontiguousarray(np.asarray(law, dtype=np.float64))
    digest = hashlib.sha256(b"float64")
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def rows_identity(ids: np.ndarray) -> str:
    """Order-sensitive identity of the original people a law was evaluated on."""
    digest = hashlib.sha256()
    for value in np.asarray(ids).ravel():
        raw = str(value).encode()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# legal inputs
# ---------------------------------------------------------------------------

def check_inputs(inputs: Mapping) -> dict[str, np.ndarray]:
    """Validate a legal input mapping; refuse any hidden or undeclared field."""
    if not isinstance(inputs, Mapping):
        raise TypeError("law inputs must be a mapping of legal per-person arrays")
    keys = set(inputs)
    forbidden = sorted(keys & FORBIDDEN_INPUTS)
    if forbidden:
        raise PermissionError(f"forbidden release inputs: {forbidden}")
    undeclared = sorted(keys - set(LEGAL_INPUTS))
    if undeclared:
        raise PermissionError(f"undeclared release inputs: {undeclared}")
    missing = sorted(set(LEGAL_INPUTS) - keys)
    if missing:
        raise ValueError(f"missing legal release inputs: {missing}")
    out = {key: np.asarray(inputs[key]) for key in LEGAL_INPUTS}
    n = len(out["token_codes"])
    shapes = {"x": (n, 32), "ha": (n, 4), "token_codes": (n,), "teacher_p": (n,),
              "residual": (n,), "risk": (n, 11)}
    for key, shape in shapes.items():
        if out[key].shape != shape:
            raise ValueError(f"legal input {key} has shape {out[key].shape}, expected {shape}")
        if key != "token_codes" and not np.isfinite(out[key]).all():
            raise ValueError(f"nonfinite legal input {key}")
    codes = out["token_codes"]
    if n == 0 or not np.issubdtype(codes.dtype, np.integer) or \
            codes.min() < 0 or codes.max() >= N_STATES:
        raise ValueError("stored T0 codes must be nonempty integers in 0..31")
    return out


def legal_inputs(rows: Mapping) -> dict[str, np.ndarray]:
    """Project an admitted role dictionary onto the six legal fields only."""
    missing = [key for key in LEGAL_INPUTS if key not in rows]
    if missing:
        raise ValueError(f"role rows lack legal fields {missing}")
    return check_inputs({key: rows[key] for key in LEGAL_INPUTS})


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate_law(law: Any, n: int, *, columns: int = N_TOKENS,
                 name: str = "release") -> np.ndarray:
    value = np.asarray(law)
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{name}: law must be numeric")
    value = value.astype(np.float64, copy=False)
    if value.shape != (n, columns):
        raise ValueError(f"{name}: law shape {value.shape}, expected {(n, columns)}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: nonfinite law")
    if np.any(value < 0):
        raise ValueError(f"{name}: negative law mass")
    worst = float(np.max(np.abs(value.sum(axis=1) - 1.0))) if n else 0.0
    if worst > ROW_SUM_TOL:
        raise ValueError(f"{name}: law rows differ from 1 by {worst:.3g} > {ROW_SUM_TOL}")
    return np.ascontiguousarray(value)


def support_record(law: np.ndarray) -> dict:
    """Per-row token support: drives the inherited hist_gb min-leaf rule."""
    p = np.asarray(law, dtype=np.float64)
    support = (p > 0).sum(axis=1)
    positive = p[p > 0]
    return {"rows": int(len(p)), "max_support": int(support.max()) if len(p) else 0,
            "support_histogram": np.bincount(support, minlength=p.shape[1] + 1).tolist(),
            "stochastic_rows": int((support > 1).sum()),
            "min_positive_mass": float(positive.min()) if positive.size else None,
            "expanded_rows": int(support.sum())}


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------

def _inventory(root: Path) -> dict[str, str]:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink inside a release unit")
        if path.is_file() and path.name != DESCRIPTOR and "__pycache__" not in path.parts:
            out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def write_descriptor(unit_dir: str | Path, kind: str, *, release_id: str,
                     anchor: int, pins: Mapping[str, str] | None = None,
                     extra: Mapping[str, Any] | None = None) -> dict:
    """Write-once `release.json`; pins default to every file in the unit."""
    if kind not in UNIT_KINDS:
        raise ValueError(f"unit descriptor kind must be one of {sorted(UNIT_KINDS)}")
    if anchor not in (0, 1, 2):
        raise ValueError("registered anchor required")
    root = Path(unit_dir)
    pins = dict(_inventory(root) if pins is None else pins)
    if not pins:
        raise ValueError("release descriptor needs at least one pinned artifact")
    record = {"schema": DESCRIPTOR_SCHEMA, "kind": kind, "release_id": release_id,
              "anchor": anchor, "pins": pins, **dict(extra or {})}
    encoded = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target = root / DESCRIPTOR
    if target.exists():
        if target.read_text() != encoded:
            raise FileExistsError("release descriptor is write-once and differs")
        return record
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    temporary.write_text(encoded)
    os.replace(temporary, target)
    return record


def read_descriptor(unit_dir: str | Path) -> dict:
    root = Path(unit_dir).resolve()
    record = json.loads((root / DESCRIPTOR).read_text())
    if record.get("schema") != DESCRIPTOR_SCHEMA or record.get("kind") not in UNIT_KINDS:
        raise ValueError("unknown release descriptor schema or kind")
    pins = record.get("pins")
    if not isinstance(pins, dict) or not pins:
        raise ValueError("release descriptor has no pinned artifacts")
    for relative, expected in pins.items():
        part = Path(relative)
        target = (root / part).resolve()
        if part.is_absolute() or not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"unsafe or missing pinned artifact {relative}")
        if sha256_file(target) != expected:
            raise ValueError(f"pinned release artifact differs: {relative}")
    return {**record, "unit_dir": str(root),
            "descriptor_sha256": sha256_file(root / DESCRIPTOR)}


def historical_spec(name: str, anchor: int, *, index_path: str | Path | None = None,
                    map_path: str | Path | None = None,
                    map_sha256: str | None = None) -> dict:
    """In-memory spec for a predecessor 32x17 map (D17 or historical Q)."""
    if name not in HISTORICAL_MAPS or anchor not in (0, 1, 2):
        raise ValueError("historical map must be D17 or Q on a registered anchor")
    if (index_path is None) == (map_path is None):
        raise ValueError("give exactly one of the pinned index or a pinned map file")
    spec = {"kind": "historical_map", "map_name": name, "anchor": anchor}
    if index_path is not None:
        spec["index_path"] = str(Path(index_path).resolve())
    else:
        if not isinstance(map_sha256, str) or len(map_sha256) != 64:
            raise ValueError("direct map file requires its SHA-256 pin")
        spec.update(map_path=str(Path(map_path).resolve()), map_sha256=map_sha256)
    return spec


def h_only_spec() -> dict:
    return {"kind": "h_only"}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _load_historical(spec: Mapping) -> tuple[np.ndarray, dict]:
    name, anchor = spec["map_name"], spec["anchor"]
    if "index_path" in spec:
        from experiments.pcrl_task_aligned_cuts_v1 import data
        value = data.index(spec["index_path"])
        record = data.member_record(value, anchor, "map", name, "Q.npz")
        q = data.load_map(value, anchor, name)  # verifies member SHA
        file_sha = record["sha256"]
    else:
        path = Path(spec["map_path"])
        file_sha = sha256_file(path)
        if file_sha != spec["map_sha256"]:
            raise ValueError("historical map file differs from its SHA-256 pin")
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)
            if len(keys) != 1:
                raise ValueError("historical map archive must hold one array")
            q = np.asarray(archive[keys[0]], dtype=np.float64)
    q = validate_law(q, N_STATES, name=f"historical {name}")
    return q, {"kind": "historical_map", "map_name": name, "anchor": anchor,
               "map_file_sha256": file_sha, "map_array_sha256": law_identity(q)}


class PinnedLaw:
    """A loaded, hash-verified release law; call with legal inputs only."""

    def __init__(self, kind: str, fn: Callable[[dict], np.ndarray], descriptor: dict,
                 columns: int = N_TOKENS):
        self.kind, self._fn, self.descriptor, self.columns = kind, fn, descriptor, columns

    def __call__(self, inputs: Mapping) -> np.ndarray:
        legal = check_inputs(inputs)
        n = len(legal["token_codes"])
        # Each call receives fresh copies: a law cannot mutate shared rows.
        law = self._fn({key: value.copy() for key, value in legal.items()})
        return validate_law(law, n, columns=self.columns,
                            name=self.descriptor.get("release_id", self.kind))


def load(source: str | Path | Mapping) -> PinnedLaw:
    """Resolve a unit directory or an in-memory predecessor spec to a law."""
    if isinstance(source, Mapping):
        kind = source.get("kind")
        if kind == "h_only":
            return PinnedLaw("h_only", lambda legal: np.ones((len(legal["token_codes"]), 1)),
                             {"kind": "h_only", "wire": "H"}, columns=1)
        if kind == "historical_map":
            q, descriptor = _load_historical(source)
            return PinnedLaw("historical_map", lambda legal: q[legal["token_codes"]].copy(),
                             descriptor)
        if kind in UNIT_KINDS and "unit_dir" in source:
            return load(source["unit_dir"])
        raise ValueError(f"unknown in-memory release spec kind {kind!r}")
    descriptor = read_descriptor(source)
    module = importlib.import_module(f"{PACKAGE}.{UNIT_KINDS[descriptor['kind']]}")
    fn = module.load_law(descriptor["unit_dir"])
    if not callable(fn):
        raise TypeError(f"{module.__name__}.load_law must return a callable law")
    # Re-verify pins after the module loaded its artifacts (no swap mid-load).
    if read_descriptor(source)["pins"] != descriptor["pins"]:
        raise ValueError("release artifacts changed while loading")
    return PinnedLaw(descriptor["kind"], fn, descriptor)


def person_law(source: str | Path | Mapping | PinnedLaw, inputs: Mapping) -> np.ndarray:
    """(n, 17) private per-person token law of one pinned release (see module doc)."""
    check_inputs(inputs)  # refuse hidden fields before any artifact is loaded
    law = source if isinstance(source, PinnedLaw) else load(source)
    return law(inputs)
