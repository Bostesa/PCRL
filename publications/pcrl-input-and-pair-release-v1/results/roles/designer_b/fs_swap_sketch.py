"""Designer B, campaign pcrl_input_and_pair_release_v1: reference sketch of FS_SWAP, the cross-recipient keyed swap
(FS_SWAP_PROPOSAL.md; the challenger's finding 20). Label-free; plain numpy; the coordinator integrates it into `ipr/`.

Operation (release side; the task-only twin f = (f_I, f_O) is fixed, SEX is never read):
  bank      the MODEL_FIT people's six-field records b_j with the twin's occupation vectors f_O(b_j) (at deployment the deployed twin
            applied to the bank records; inside the nested MODEL_FIT selection the out-of-fold vectors of the other folds' people).
            Within each occupation decision cell k (argmax f_O(b_j) = k) the bank rows are put in a canonical order, the SHA-256 of
            (tag | record bytes), so with deployed bank vectors the release does not depend on how the rows were stored (ties are
            identical records, which carry identical deployed vectors).
  select    u(x) = SHA-256(tag|sel | six compact fields) / 16^13; person x is swapped iff u(x) < lambda.
  donor     v(x) = SHA-256(tag|donor | six compact fields) / 16^13; k = argmax f_O(x); j = floor(v(x) |B_k|) in cell k's ordered
            bank; if that bank record equals x's record, step to the next row (cyclically); a cell with no other record -> no swap.
  release   income recipient: f_I(x) unchanged (bitwise). Occupation recipient: f_O(donor) if swapped, else f_O(x). The donor shares
            x's occupation decision, so both decisions and both accuracies are kept exactly; the occupation wire of a swapped person
            is a genuine twin vector of a random same-decision bank member, so the wire's marginal distribution within each decision
            cell is (up to the keyed draw) the bank's, while the link between x's two wires beyond the decision is cut.
  modes     'swap' (the candidate: donors from the same occupation decision cell); 'swap_joint' (donors from the same joint
            (income decision, occupation decision) cell: R1 variant); 'occ_flat' (ablation: the same selected people get the
            occupation decision-cell mean instead of a donor; CSF's occupation part); 'csf' (both recipients collapsed, CSF).

    fit(Z, y_inc, y_occ, sex, unit, version, strength, twin_inc=None, twin_occ=None, twin_fn=None, mode='swap',
        bank_Z=None, bank_inc=None, bank_occ=None) -> state
    release(state, Z, twin_inc=None, twin_occ=None) -> (P_inc (n, 2), P_occ (n, 5))
    selftest()  (invented data only)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech_sketch import _standin_twin, clipn, decision_tables, keyed_uniform  # noqa: E402

STRENGTHS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.125, 0.15, 0.20)   # registered lambda grid
EXPLORATORY_GRID = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30)            # FS_SWAP_FEASIBILITY.json's spend grid
MODES = ('swap', 'swap_joint', 'occ_flat', 'csf')
K_INC, K_OCC = 2, 5


def _order_key(rec, tag):
    return hashlib.sha256(f'{tag}|bank|{np.ascontiguousarray(rec, np.float64).tobytes().hex()}'.encode()).hexdigest()


def build_bank(bank_Z, bank_inc, bank_occ, tag, joint):
    """Ordered donor lists per cell: occupation decision k, or the joint cell (income decision, occupation decision)."""
    bank_Z = np.asarray(bank_Z, np.float64)
    bank_occ = clipn(bank_occ)
    cell = bank_occ.argmax(1) + (K_OCC * clipn(bank_inc).argmax(1) if joint else 0)
    lists = {}
    for c in np.unique(cell):
        idx = np.flatnonzero(cell == c)
        keys = np.array([_order_key(bank_Z[j], tag) for j in idx])
        lists[int(c)] = idx[np.argsort(keys, kind='mergesort')]   # ties = identical records (identical deployed vectors)
    return {'Z': bank_Z, 'occ': bank_occ, 'lists': lists, 'joint': joint}


def fit(Z, y_inc, y_occ, sex, unit, version, strength, twin_inc=None, twin_occ=None, twin_fn=None, mode='swap',
        bank_Z=None, bank_inc=None, bank_occ=None):
    """FS_SWAP state. `sex` is never read. Default bank: the fit rows themselves with their (out-of-fold) twin vectors."""
    del sex
    if mode not in MODES:
        raise ValueError(mode)
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError('strength lambda must be in [0, 1]')
    Z = np.asarray(Z, np.float64)
    if twin_inc is None or twin_occ is None:
        twin_inc, twin_occ, fn = _standin_twin(Z, np.asarray(y_inc), np.asarray(y_occ), np.asarray(unit), version)
        twin_fn = twin_fn or fn
    tag = f'IPR_FS_SWAP|V{int(version)}'
    if bank_Z is None:
        bank_Z, bank_inc, bank_occ = Z, twin_inc, twin_occ
    return {'lam': float(strength), 'mode': mode, 'version': int(version), 'tag': tag, 'twin_fn': twin_fn,
            'bank': build_bank(bank_Z, bank_inc, bank_occ, tag, joint=(mode == 'swap_joint')),
            'T_inc': decision_tables(twin_inc, K_INC), 'T_occ': decision_tables(twin_occ, K_OCC)}


def donors(state, Z, pI, pO):
    """Bank row index of each person's donor (-1 when the cell has no other record)."""
    B = state['bank']
    Z = np.asarray(Z, np.float64)
    cell = clipn(pO).argmax(1) + (K_OCC * clipn(pI).argmax(1) if B['joint'] else 0)
    v = keyed_uniform(Z, state['tag'] + '|donor')
    out = np.full(len(Z), -1)
    for i in range(len(Z)):
        lst = B['lists'].get(int(cell[i]))
        if lst is None or len(lst) == 0:
            continue
        j0 = int(np.floor(v[i] * len(lst)))
        for step in range(len(lst)):                       # skip bank rows with x's own record (cyclically)
            j = lst[(j0 + step) % len(lst)]
            if not np.array_equal(B['Z'][j], Z[i]):
                out[i] = j
                break
    return out


def release_from_twin(state, Z, pI, pO):
    pI, pO = clipn(pI), clipn(pO)
    sel = keyed_uniform(Z, state['tag'] + '|sel') < state['lam']
    rI, rO = pI.copy(), pO.copy()
    if state['mode'] in ('swap', 'swap_joint'):
        d = donors(state, Z, pI, pO)
        ok = sel & (d >= 0)
        rO[ok] = state['bank']['occ'][d[ok]]
    else:
        rO[sel] = state['T_occ'][pO[sel].argmax(1)]
        if state['mode'] == 'csf':
            rI[sel] = state['T_inc'][pI[sel].argmax(1)]
    return rI, rO                     # inputs, bank vectors and tables are already floored and normalized (no second pass)


def release(state, Z, twin_inc=None, twin_occ=None):
    Z = np.asarray(Z, np.float64)
    if twin_inc is None or twin_occ is None:
        twin_inc, twin_occ = state['twin_fn'](Z)
    return release_from_twin(state, Z, twin_inc, twin_occ)


def selftest():
    rng = np.random.default_rng(11)
    n = 2400
    sex = rng.integers(0, 2, n)
    Z = np.column_stack([rng.integers(18, 80, n), np.clip(rng.normal(38 + 4 * sex, 9, n).round(), 1, 99), rng.integers(1, 9, n),
                         rng.integers(1, 25, n), rng.integers(1, 6, n), rng.integers(1, 60, n)]).astype(float)
    Z[:40] = Z[40:80]                                                    # some identical records
    unit = rng.integers(0, 1600, n)
    y_inc = (0.05 * Z[:, 1] + 0.15 * Z[:, 3] + rng.normal(0, 1.5, n) > 5.5).astype(int)
    y_occ = np.clip(((Z[:, 3] + 2 * Z[:, 2] + rng.normal(0, 4, n)) / 7).astype(int), 0, 4)
    oI, oO, fn = _standin_twin(Z, y_inc, y_occ, unit, 0)
    st = fit(Z, y_inc, y_occ, sex, unit, 0, 0.3, twin_inc=oI, twin_occ=oO, twin_fn=fn)
    tI, tO = fn(Z)                                                       # deployment: the twin's deployed models
    rI, rO = release(st, Z)
    tI, tO = clipn(tI), clipn(tO)
    assert np.array_equal(rI, tI), 'the income wire is the twin bitwise'
    assert np.array_equal(rO.argmax(1), tO.argmax(1)), 'occupation decisions kept exactly'
    changed = np.any(rO != tO, axis=1)
    assert 0.2 < changed.mean() < 0.4, changed.mean()
    d = donors(st, Z, tI, tO)
    sel = keyed_uniform(Z, st['tag'] + '|sel') < 0.3
    assert np.all(d[sel] >= 0) and not np.any([np.array_equal(st['bank']['Z'][d[i]], Z[i]) for i in np.flatnonzero(sel)])
    assert np.allclose(rO[sel], st['bank']['occ'][d[sel]]), 'a swapped wire is a bank member vector'
    assert np.array_equal(rO[~sel], tO[~sel])
    rI2, rO2 = release(st, Z)
    assert np.array_equal(rO, rO2), 'deterministic in the six fields'
    st_p = fit(Z, y_inc, y_occ, rng.permutation(sex), unit, 0, 0.3, twin_inc=oI, twin_occ=oO, twin_fn=fn)
    assert np.array_equal(release(st_p, Z)[1], rO), 'SEX is never read'
    perm = rng.permutation(n)                                            # deployed bank: storage order does not matter
    sd = fit(Z, y_inc, y_occ, sex, unit, 0, 0.3, twin_inc=oI, twin_occ=oO, twin_fn=fn, bank_Z=Z, bank_inc=tI, bank_occ=tO)
    sdp = fit(Z, y_inc, y_occ, sex, unit, 0, 0.3, twin_inc=oI, twin_occ=oO, twin_fn=fn, bank_Z=Z[perm], bank_inc=tI[perm],
              bank_occ=tO[perm])
    assert np.array_equal(release(sd, Z)[1], release(sdp, Z)[1]), 'canonical bank order'
    sj = fit(Z, y_inc, y_occ, sex, unit, 0, 0.3, twin_inc=oI, twin_occ=oO, twin_fn=fn, mode='swap_joint')
    dj = donors(sj, Z, tI, tO)
    bI = clipn(oI).argmax(1)
    assert np.all(bI[dj[sel]] == tI[sel].argmax(1)), 'joint-cell donors share the income decision'
    sf = fit(Z, y_inc, y_occ, sex, unit, 0, 0.3, twin_inc=oI, twin_occ=oO, twin_fn=fn, mode='occ_flat')
    fI, fO = release(sf, Z)
    assert np.array_equal(fI, tI) and len(np.unique(fO[sel].round(12), axis=0)) <= K_OCC
    s0 = fit(Z, y_inc, y_occ, sex, unit, 0, 0.0, twin_inc=oI, twin_occ=oO, twin_fn=fn)
    assert np.array_equal(release(s0, Z)[1], tO), 'lambda = 0 is the twin'
    return 'fs_swap_sketch selftest passed'


if __name__ == '__main__':
    print(selftest())
