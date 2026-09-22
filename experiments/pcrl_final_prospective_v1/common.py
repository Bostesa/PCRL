"""Shared constants, hashing and atomic persistence for the final prospective study."""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np

STUDY = 'pcrl_final_prospective_v1'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results'/STUDY
PRIVATE = OUT/'private'
INPUTS = PRIVATE/'inputs'
RESTORED = PRIVATE/'restore'/'results'/'pcrl_task_directed_release_v1'/'private'/'run'
PRED = 'pcrl_task_directed_release_v1'
PINNED = {
    'task_directed': 'f4bdf4cd5bf74c634feeec50aef78bff249667e4',
    'replacement': 'e3415b94deb8d71c4870d4c392bb8ad4b464a847',
    'manuscript_review': '0176f149e91c02b8e2d202eb25ea9cba8ae019dc',
    'evidence_paper_2016_admission': '0d8f4b67b6d4961dfa133289d0167c874d2f4794',
}
ANCHORS = (0, 1, 2)

# Panel: short name -> exact frozen release family.
PANEL = {
    'H': 'H', 'J': 'J', 'Q': 'T0_L_0.01_a17', 'D17': 'T0_U_unconstrained_a17',
    'D33': 'T0_U_unconstrained_a33', 'C': 'continuous_task',
    'E': 'leace_supervised_mechanism40', 'S': 'splince_supervised_mechanism40',
    'RR75': 'T0_rr_0.75', 'W75': 'T0_withhold_0.75',
}
CANDIDATES = ('Q', 'D17')
SECONDARY_COMPARATORS = ('D17', 'D33', 'C', 'E', 'S', 'RR75', 'W75')
PRIMARY_ROLES = ('attack:A/SEX', 'attack:A/RAC1P', 'attack:AB/SEX', 'attack:AB/RAC1P')
TASK_ROLE = 'utility:A/same_residence'
B_ROLES = ('attack:B/SEX', 'attack:B/RAC1P')
WEIGHTINGS = ('unweighted', 'PWGTP')

# 2016 pools (admission partition; label-blind, committed before this study).
FIT_POOL = 'fitting'
ATTACK_VAL = 'validation/attacker_validation'
TASK_VAL = 'validation/task_validation'
FINAL = 'final_evaluation'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean(value), indent=2, allow_nan=False)+'\n'
    if immutable and path.exists():
        if path.read_text() != text:
            raise FileExistsError(f'Refusing to replace immutable record {path}')
        return
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())
