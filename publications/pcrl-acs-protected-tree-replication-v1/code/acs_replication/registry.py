"""Service output registry: every fitted service's released probabilities per seed / role / task.

Keys inside the private output files follow '<SERVICE>__<ROLE>__<task>'. Only probability vectors are stored; the
released wire of a recipient is (probability vector, argmax decision). Nothing else leaves a worker.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import numpy as np

PRIV = Path(__file__).resolve().parent.parent / '.private' / 'acs_replication'
RUNS = PRIV / 'runs'
FILES = {'stage_a': 'STAGE_A_{s}/STAGE_A_OUTPUTS.npz', 'stage_b': 'STAGE_B_{s}/STAGE_B_OUTPUTS_E20.npz',
         'neural_raw': 'NEURAL_RAW_{s}/NEURAL_RAW_OUTPUTS.npz', 'fare': 'FARE_{s}/FARE_OUTPUTS.npz',
         'fagtb': 'FAGTB_{s}/FAGTB_OUTPUTS.npz'}


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(seed, roles=None, families=None):
    """Return ({service: {role: {task: probs}}}, {file: sha256}); missing families are skipped and reported."""
    out, hashes, missing = {}, {}, []
    for fam, pattern in FILES.items():
        if families is not None and fam not in families:
            continue
        p = RUNS / pattern.format(s=seed)
        if not p.is_file():
            missing.append(fam); continue
        hashes[str(p.relative_to(PRIV))] = sha(p)
        with np.load(p) as z:
            for key in z.files:
                if '__' not in key or key.startswith(('row_id__', 'unit__')):
                    continue
                svc, role, task = key.split('__')
                if roles is not None and role not in roles:
                    continue
                out.setdefault(svc, {}).setdefault(role, {})[task] = z[key].copy()
    return out, hashes, missing


def wire(p):
    p = np.asarray(p, dtype=np.float64)
    return {'probability': p, 'decision': p.argmax(1).astype(np.int64)}
