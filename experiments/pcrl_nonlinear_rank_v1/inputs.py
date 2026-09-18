"""Read-only resolution of historical artifacts, with recorded hashes.

This worktree is a fresh checkout of the baseline commit, so the ignored local
artifacts of the completed studies (fitted spectral maps, 2018 releases, the
2017 partitions) did not travel with git. They are resolved from the original
checkout / the study's own worktree **read-only** and hashed; nothing historical
is refitted, moved or modified.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HIST_ROOT = Path('/Users/nathansamson/PCRL')

# Read-only fallback roots searched in order for an ignored local artifact.
FALLBACK_ROOTS = (
    ROOT,
    Path(os.environ.get('PCRL_NLR_FALLBACK',
                        '/Users/nathansamson/.config/superpowers/worktrees/PCRL/residual-spectral-20260910')),
    HIST_ROOT,
)

DEV_NAME = 'redesign_20260910_acs_residual_spectral_v1'
TRANSPORT_NAME = 'redesign_20260917_acs_spectral_transport_v1'
FIXED_NAME = 'redesign_20260909_acs_fixed_predictions_v1'
TRANSFER_NAME = 'redesign_20260907_acs_transfer_v1'

OUT = ROOT / 'results/pcrl_nonlinear_rank_v1'
SEEDS = (0, 1, 2)

HIST_INTERFACES = ('H', 'E', 'A0', 'L025', 'L20', 'J')
HIST_SPECTRAL = tuple('spectral_' + a for a in ('S0', 'M025', 'M1', 'L025', 'L1', 'C025', 'C1', 'L2'))

# The three protected local roles and two coalition roles of the original penalty.
SPECTRAL_SCHEMA = {'SEX': 2, 'RAC1P': 9, 'public_coverage': 2}


class MissingArtifact(FileNotFoundError):
    pass


def resolve(relative: str | Path) -> Path:
    """First existing path for ``relative`` across the read-only fallback roots."""
    relative = Path(relative)
    for root in FALLBACK_ROOTS:
        candidate = root / relative
        if candidate.exists():
            return candidate
    raise MissingArtifact(f'{relative} not found under any of {[str(r) for r in FALLBACK_ROOTS]}')


def sha_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(array) -> str:
    import hashlib
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _finite(obj):
    """Replace non-finite floats with None so ``allow_nan=False`` cannot abort a write.

    A NaN or infinity is a real signal (an undefined ratio, a degenerate interval),
    so it is preserved as an explicit null rather than silently coerced to a number.
    """
    import math
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def write_json(path, obj) -> Path:
    """Atomic JSON write; readers never observe a partial file."""
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_finite(obj), indent=2, allow_nan=False, sort_keys=False) + '\n'
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    with os.fdopen(fd, 'w') as handle:
        handle.write(text)
    os.replace(tmp, path)
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def arrays(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ registry
@dataclass
class Registry:
    """Every read-only input this study touched, with its hash."""

    entries: dict

    @classmethod
    def new(cls) -> "Registry":
        return cls(entries={})

    def add(self, path) -> Path:
        path = Path(path).resolve()
        key = str(path)
        if key not in self.entries:
            self.entries[key] = {'sha256': sha_file(path), 'bytes': path.stat().st_size}
        return path

    def resolve(self, relative) -> Path:
        return self.add(resolve(relative))

    def dump(self) -> dict:
        return {'count': len(self.entries), 'files': dict(sorted(self.entries.items()))}


# ------------------------------------------------------------------ 2018 development inputs
def dev_path(*parts) -> str:
    return str(Path('results') / DEV_NAME / Path(*parts))


def fixed_path(*parts) -> str:
    return str(Path('results') / FIXED_NAME / Path(*parts))


def load_spectral_model(seed: int, registry: Registry):
    """The frozen 2018 SpectralModel for one seed (never refitted)."""
    import joblib
    return joblib.load(registry.resolve(dev_path(f'seed_{seed}', 'maps.joblib')))


def load_matrix_diagnostics(seed: int, registry: Registry) -> dict:
    return read_json(registry.resolve(dev_path(f'seed_{seed}', 'matrix_diagnostics.json')))


def load_pools(seed: int, registry: Registry):
    """Historical PCA teacher coordinates, released anchors and pool row indices."""
    p = fixed_path(f'seed_{seed}')
    rows = arrays(registry.resolve(str(Path(p) / 'split_rows.npz')))
    teacher = arrays(registry.resolve(str(Path(p) / 'pca.npz')))
    anchors = arrays(registry.resolve(str(Path(p) / 'anchors.npz')))
    return rows, teacher, anchors


def load_representation_labels(seed: int, registry: Registry):
    """Representation-fit sensitive labels and household IDs, exactly as the 2018 fit used them.

    Mirrors ``run_acs_residual_spectral.fit_maps``: the raw CSV is read with only
    the four columns that stage is permitted to see, and indexed by the saved
    per-pool raw-row arrays. No residence or commute column is loaded.
    """
    import pandas as pd
    raw_path = registry.resolve('data/folktables/2018/1-Year/psam_p06.csv')
    raw = pd.read_csv(raw_path, usecols=['SEX', 'RAC1P', 'PUBCOV', 'SERIALNO'], dtype={'SERIALNO': str})
    rows, _, _ = load_pools(seed, registry)
    frame = raw.iloc[rows['representation_fit']]
    sex = frame.SEX.to_numpy(float)
    race = frame.RAC1P.to_numpy(float)
    pub = frame.PUBCOV.to_numpy(float)
    labels = {
        'SEX': np.where(np.isin(sex, [1, 2]), (sex - 1), -1).astype(np.int64),
        'RAC1P': np.where(np.isin(race, list(range(1, 10))), (race - 1), -1).astype(np.int64),
        'public_coverage': np.where(np.isin(pub, [1, 2]), (pub == 1).astype(int), -1).astype(np.int64),
    }
    return labels, frame.SERIALNO.to_numpy()
