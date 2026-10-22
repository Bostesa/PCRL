"""The shared SPEND (DQ) and generic RO control paths of the PCRL input-and-pair release campaign, built on the shared twin inside a
family's fit unit (label-free code: the MODEL_FIT labels arrive as arguments from the family entry point; PROTOCOL section 4).

    spend_and_ro(tI, tO, Z, yI, yO, sex, unit, version, cand_L, twin_L, family)  -> saved-point payloads and records
    predict_path(P, tI, tO, Z)                                                     -> the released vectors of a saved path point

Bracket rule (PROTOCOL 4; `bracket_points`, the dual-design rule): LO = the largest grid point whose MODEL_FIT out-of-fold total-NLL
spend over the twin is at most CAND's; HI = the smallest grid point whose spend is at least CAND's + 2 household-clustered SE of the
paired difference (point minus CAND); if none reaches it, the largest grid point, flagged CONTROL_PATH_MAY_NOT_BRACKET. Fineness
(PROTOCOL 4): the record states whether the grid holds a point in [0.75, 1] x and in [1, 1.25] x CAND's spend.
"""
from __future__ import annotations

import numpy as np

from . import ctlkit as CK
from . import famkit as FK


def bracket_points(path_L, cand_L, twin_L, unit, order):
    cand_sp = float((cand_L['tot_nll'] - twin_L['tot_nll']).mean())
    rows, lo, hi = {}, None, None
    for g in order:
        d = path_L[g]['tot_nll'] - twin_L['tot_nll']
        diff = path_L[g]['tot_nll'] - cand_L['tot_nll']
        rows[str(g)] = {'spend': float(d.mean()), 'diff_vs_cand': float(diff.mean()), 'se_diff': FK.cluster_se(diff, unit)}
        if rows[str(g)]['spend'] <= cand_sp:
            lo = g
    for g in order:
        r = rows[str(g)]
        if r['spend'] >= cand_sp + 2.0 * r['se_diff']:
            hi = g
            break
    flag = None
    if hi is None:
        hi, flag = order[-1], 'CONTROL_PATH_MAY_NOT_BRACKET'
    if lo == hi:
        lo = None
    sp = [r['spend'] for r in rows.values()]
    fine = {'below': any(0.75 * cand_sp <= s <= cand_sp for s in sp), 'above': any(cand_sp <= s <= 1.25 * cand_sp for s in sp)}
    return {'cand_spend': cand_sp, 'grid': rows, 'lo': lo, 'hi': hi, 'flag': flag, 'fineness': fine,
            'fineness_ok': bool(fine['below'] and fine['above'])}


def spend_and_ro(tI, tO, Z, yI, yO, sex, unit, version, cand_L, twin_L, family, dq_grid=CK.DQ_GRID_FINE, ro_grid=CK.RO_GRID,
                 log=None, ro=True):
    """DQ (SEX-blind) and generic RO paths of the shared twin at a family's CAND spend. Returns {'SPEND': {...}, 'RO': {...}}, each
    with 'points': {'LO' | 'HI': (strength, payload, (rI_oof, rO_oof))} and 'bracket'."""
    out = {}
    basis = CK.dq_fit(tI, tO)
    tag = f'IPR_DQ_S{int(version)}'                 # family-independent: equal steps on the shared twin alias across families
    dq_rel = {d: CK.dq_release(basis, tI, tO, Z, d, tag) for d in dq_grid}
    dq_L = {d: FK.person_losses(*dq_rel[d], yI, yO) for d in dq_grid}
    br = bracket_points(dq_L, cand_L, twin_L, unit, tuple(dq_grid))
    out['SPEND'] = {'bracket': br, 'points': {k: (g, {'op': 'dq', 'dq_basis': basis, 'delta': float(g), 'dq_tag': tag}, dq_rel[g])
                                              for k, g in (('LO', br['lo']), ('HI', br['hi'])) if g is not None}}
    if log:
        log(f'SPEND path bracket lo={br["lo"]} hi={br["hi"]} flag={br["flag"]} fineness_ok={br["fineness_ok"]}')
    if not ro:                                       # a SEX-blind family: no RO path is built (PROTOCOL 4)
        return out
    rel, state, rec = CK.ro_path(tI, tO, sex, unit, version, tuple(ro_grid))
    ro_L = {a: FK.person_losses(*rel[a], yI, yO) for a in ro_grid}
    brr = bracket_points(ro_L, cand_L, twin_L, unit, tuple(ro_grid))
    out['RO'] = {'bracket': brr, 'record': rec,
                 'points': {k: (g, {'op': 'ro', 'ro_state': state, 'alpha': float(g)}, rel[g])
                            for k, g in (('LO', brr['lo']), ('HI', brr['hi'])) if g is not None}}
    if log:
        log(f'RO path bracket lo={brr["lo"]} hi={brr["hi"]} flag={brr["flag"]} fineness_ok={brr["fineness_ok"]}')
    return out


SHARED_CARD = 'results/pcrl_input_and_pair_release_v1/cards/CTL_PATHS_v1.md'


def shared_id(key, g):
    """Configuration id of a shared path point: CTL_DQ_<step> / CTL_RO_<strength> (6 decimals, '.' written as 'p')."""
    return f"CTL_{'DQ' if key == 'SPEND' else 'RO'}_{float(g):.6f}".replace('.', 'p')


def save_shared(sr, version, roles, save, base, people, households, cpu, yI, yO, twin_L, unit, sex, family, log=None):
    """Save the SPEND / RO bracket points of a family under the shared family CTL (card CTL_PATHS_v1), once: a point that already
    exists (the other family's bracket) is reused, counted once and recorded as reused. Returns {key: {'LO' | 'HI': config id}} and
    {config id: 'saved' | 'reused'}."""
    from . import famkit as FK
    from . import release as RL
    from .ctl import twin as TW
    ids, status = {}, {}
    for key in ('SPEND', 'RO'):
        if key not in sr:
            continue
        ids[key] = {}
        for side, (g, payload, oof_pair) in sr[key]['points'].items():
            cfg = shared_id(key, g)
            ids[key][side] = cfg
            if RL.exists('CTL', cfg, version):
                status[cfg] = 'reused'
                continue
            meta = FK.write_model('CTL', cfg, version, {**base, **payload})
            arrays = FK.release_all(TW.predict, cfg, version, roles)
            sp = FK.spend(FK.person_losses(*oof_pair, yI, yO), twin_L, unit, sex)
            what = (f"shared {'SPEND path (DQ, SEX-blind dithered quantization)' if key == 'SPEND' else 'RO path (generic release-only tilt)'} "
                    f'point of the shared twin at strength {float(g):.6g} (first bracketed family {family})')
            rec = FK.record(SHARED_CARD, 'control', what, (0, 0), 'MODEL_FIT: income, occupation, sex',
                            'MODEL_FIT task labels reach the cached twin (inherited); for RO, MODEL_FIT SEX reaches the release-only '
                            'surrogate and the tilt weights (cross-fitted); MODEL_FIT task and SEX labels reach the bracket selection',
                            people, households, cpu, {'op': payload['op'], 'strength': float(g), 'path': key,
                                                     'first_family': family, 'model_sha256': meta['model_sha256'],
                                                     'oof_spend_over_twin': FK.jsonable_spend(sp)})
            save('CTL', cfg, version, arrays, rec)
            status[cfg] = 'saved'
            if log:
                log(f'shared {key} point {cfg} saved')
    return ids, status


def predict_path(P, tI, tO, Z):
    """Released vectors of a saved SPEND / RO point from the twin's vectors on the same people. A point restricted to the occupation
    output (payload `outputs = occupation`, card IA_OSG_R1) passes the twin's income vector through bitwise."""
    if P.get('outputs') not in (None, 'occupation'):
        raise ValueError(f'unknown path restriction {P.get("outputs")!r}')
    if P['op'] == 'dq':
        rI, rO = CK.dq_release(P['dq_basis'], tI, tO, Z, float(P['delta']), P['dq_tag'])
    elif P['op'] == 'ro':
        rI, rO = CK.ro_release(P['ro_state'], tI, tO, float(P['alpha']))
    else:
        raise ValueError(f'not a shared path operation: {P["op"]}')
    if P.get('outputs') == 'occupation':
        return tI, rO
    return rI, rO


def selftest():
    rng = np.random.default_rng(5)
    n = 2400
    unit = np.arange(n) // 2
    sex = rng.integers(0, 2, n)
    tO = CK.softmax(rng.normal(size=(n, 5)) + 0.7 * (sex - 0.5)[:, None] * np.array([1.0, -0.5, 0.0, 0.3, -0.8]))
    tI = CK.softmax(rng.normal(size=(n, 2)) * 0.8 + 0.5 * (sex - 0.5)[:, None] * np.array([-1.0, 1.0]))
    yI = (rng.random(n) < tI[:, 1]).astype(np.int64)
    yO = (tO.cumsum(1) < rng.random(n)[:, None]).sum(1)
    Z = np.column_stack([rng.integers(20, 60, n), rng.integers(1, 80, n), rng.integers(0, 8, n), rng.integers(1, 24, n),
                         rng.integers(1, 6, n), rng.integers(1, 500, n)]).astype(float)
    twin_L = FK.person_losses(tI, tO, yI, yO)
    cand = CK.dq_release(CK.dq_fit(tI, tO), tI, tO, Z, 0.8, 'selftest_cand')
    cand_L = FK.person_losses(*cand, yI, yO)
    res = spend_and_ro(tI, tO, Z, yI, yO, sex, unit, 0, cand_L, twin_L, 'TEST', dq_grid=CK.DQ_GRID_FINE[:30], ro_grid=CK.RO_GRID[:30])
    ok = {'spend_bracket': res['SPEND']['bracket']['hi'] is not None,
          'ro_has_points': bool(res['RO']['points']),
          'fineness_recorded': 'fineness_ok' in res['SPEND']['bracket']}
    k, (g, payload, oof) = next(iter(res['SPEND']['points'].items()))
    ok['spend_predict_equals_oof'] = bool(np.allclose(predict_path(payload, tI, tO, Z)[1], oof[1]))
    ok['all_ok'] = all(ok.values())
    return ok


if __name__ == '__main__':
    import json
    r = selftest()
    print(json.dumps(r, default=str))
    raise SystemExit(0 if r['all_ok'] else 1)
