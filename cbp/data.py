"""Role loader for the confidence-budgeted privacy study (cbp): the pinned osf/dpc/qpc roles, reused EXACTLY.

``load()`` returns qpc.data.load()'s D unchanged (pinned dpc.data -> osf.data; all 39,170 rows; 83 permitted columns),
after qpc.data's pinned-blob checks and role verification (counts, exact-record groups, row-id and group-set hashes,
zero group overlap between OSF_DEFENSE_FIT and the attacker / selection / assessment roles):

    OSF_DEFENSE_FIT 15,434 | HEAD_VALIDATION 1,500 | AUDIT_FIT 6,065 | INNER_SELECTION 2,235 |
    OSF_DEVELOPMENT_ASSESSMENT 13,936 rows / 13,929 exact-record groups

Assessment labels (sex, race, y_income, y_occupation_group, y dict) are -1 at load. ``load(unseal=True)`` refuses
unless (a) the calling module is cbp.assess and (b) results/pcrl_confidence_budgeted_privacy_v1/EVALUATION_LOCK.json is
committed and byte-identical on origin/research/pcrl-confidence-budgeted-privacy-v1 (fetched now). The qpc and dpc
gates open only for their own studies, so after this gate passes the pinned osf.data.load(unseal=True) is called
directly (as qpc.data does), followed by the same verification. ``labels_for`` is qpc.data.labels_for (dpc procedure
vocabulary + qpc's narrower procedures); qpc.utility uses it unchanged.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from osf import data as OD
from qpc import data as QD

WT = Path(__file__).resolve().parents[1]
REL = "results/pcrl_confidence_budgeted_privacy_v1"
PKG = WT / REL
BRANCH = "research/pcrl-confidence-budgeted-privacy-v1"
EVALUATION_LOCK = PKG / "EVALUATION_LOCK.json"
UNSEAL_CALLER = "cbp.assess"
ASSESS = QD.ASSESS
FIT = QD.FIT
EXPECTED = QD.EXPECTED
labels_for = QD.labels_for
row_index = QD.row_index


def _caller_module(depth):
    f = sys._getframe(depth)
    name = f.f_globals.get("__name__")
    if name == "__main__":
        spec = f.f_globals.get("__spec__")
        name = getattr(spec, "name", None) or "__main__"
    return name


def evaluation_lock_pushed(lock=None, branch=BRANCH):
    lock = Path(lock or EVALUATION_LOCK)
    rel = f"{REL}/EVALUATION_LOCK.json"
    if not lock.exists():
        return False, "EVALUATION_LOCK.json does not exist"
    subprocess.run(["git", "-C", str(WT), "fetch", "-q", "origin", branch], capture_output=True)
    r = subprocess.run(["git", "-C", str(WT), "show", f"origin/{branch}:{rel}"], capture_output=True)
    if r.returncode != 0:
        return False, "EVALUATION_LOCK.json is not on origin"
    if r.stdout != lock.read_bytes():
        return False, "local EVALUATION_LOCK.json differs from origin"
    return True, "EVALUATION_LOCK.json byte-identical on origin"


def unseal_gate(caller):
    if caller != UNSEAL_CALLER:
        raise PermissionError(f"REFUSED: only {UNSEAL_CALLER} may unseal assessment labels (caller: {caller})")
    ok, why = evaluation_lock_pushed()
    if not ok:
        raise PermissionError(f"REFUSED: assessment labels stay sealed: {why}")
    return why


def load(verify=True, unseal=False):
    gate = unseal_gate(_caller_module(2)) if unseal else None
    if not unseal:
        D = QD.load(verify=verify, unseal=False)
    else:
        if verify:
            bad = [f for f, ok in QD.loader_blob_checks().items() if not ok]
            if bad:
                raise SystemExit(f"REFUSED: pinned loader differs from its blob at the pin: {bad}")
        D = OD.load(verify=verify, unseal=True)
        if verify:
            _, bad = QD.verify_D(D)
            if bad:
                raise SystemExit(f"REFUSED: roles differ from the pinned manifests: {bad}")
    D["study"] = "pcrl_confidence_budgeted_privacy_v1"
    D["unseal_gate"] = gate
    if not unseal:
        a = D["idx"][ASSESS]
        assert D["sealed"] and all(np.all(D[k][a] == -1) for k in QD.LABEL_KEYS), "assessment labels not sealed"
    return D
