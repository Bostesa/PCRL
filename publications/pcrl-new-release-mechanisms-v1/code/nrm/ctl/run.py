"""CTL_REM runner (challenger; governed by results/pcrl_new_release_mechanisms_v1/cards/CTL_CARD_v2.md).

CTL_REM post-processes a named saved DETERMINISTIC base release (default FB/FB2_BASE):
  income      = the base's income vector, unchanged;
  occupation  = nrm.ctl.post.flatten_remainder(base occupation, R_v, s): every argmax and its probability are kept and the non-argmax
                remainder moves toward SEX-blind class shares R_v[argmax] with weight min(s, cap) (cap keeps every decision).
R_v is fitted on the base's CALIBRATION occupation vectors with CALIBRATION occupation labels (operation 'calibrate'). No SEX label, no
MODEL_FIT label and no MODEL_SELECT label is read by this module.

Steps (each run through the queue; the card is committed first):
  plan   --version V   label read (CALIBRATION occupation): fits R_v on all of CALIBRATION, and predicts the total-NLL spend over the
                       base of every REM_LADDER level by 5-fold household cross-fitting on CALIBRATION (out-of-fold; household-clustered
                       SE). Writes runs/CTL/_PLAN/PLAN_V<v>.json and SHARES_V<v>.npz (write-once). Saves no release.
  choose --targets F   label-free: the card's spend-only rule on both versions' plans and the coordinator's realized MODEL_SELECT
                       total-NLL spends of the candidate strengths over the base (JSON {"V0": {"S3": c, "S5": c}, "V1": {...}}).
                       Writes runs/CTL/_PLAN/CHOICE.json (write-once).
  fit    --version V   label-free: for each chosen level writes the write-once parameter file and saves the release of all six roles
                       through nrm.release.save (family CTL, config_id CTL_REM_<level>, kind 'control').
  selftest             invented arrays, fake access and a temporary folder: no data, no label, no ledger.
Deployment (nrm.deploy family interface; exported by nrm.ctl.release): predict, model_paths, code_modules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import post as P

FAMILY = 'CTL'
BASE = ('FB', 'FB2_BASE')
CARD = 'results/pcrl_new_release_mechanisms_v1/cards/CTL_CARD_v2.md'
REM_LADDER = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00)
N_FOLDS = 5
FOLD_TAG = 'NRM_CTL_REM_cv_S{v}'
Z_BRACKET = 0.0           # a level brackets a target c when its predicted spend >= c + Z_BRACKET * SE in both versions ("at or
                          # just above each target", coordinator 15:52Z; CTL_CARD_v2)
TASKS = ('income', 'occupation')
REQUIRED = ('mechanism_card', 'kind', 'description', 'n_model_fits', 'n_calibrator_fits', 'n_crossfit_fits', 'n_critic_fits',
            'roles_read', 'exposure', 'training_people', 'training_households', 'fit_cpu_s', 'randomized')
_DEPS_OVERRIDE = None     # selftest only


def level_id(s):
    return 'CTL_REM_' + f'{float(s):.2f}'.replace('.', 'p')


def _key(s):
    return f'{float(s):.2f}'


# ------------------------------------------------------------------------------------------------ dependencies (real or fake)
class Deps:
    def __init__(self, runs, load_labels, load_base, save, base_module, roles, base=BASE):
        self.runs, self.load_labels, self.load_base, self.save = Path(runs), load_labels, load_base, save
        self.base_module, self.roles, self.base = base_module, tuple(roles), tuple(base)


def real_deps(base=BASE):
    """Real access. The base family module is imported lazily (only predict / model_paths / code_modules need it)."""
    from .. import common as C
    from .. import release as RL
    return Deps(runs=C.RUNS, load_labels=C.load, load_base=lambda v: RL.load(base[0], base[1], v), save=RL.save,
                base_module=None, roles=C.ROLES6, base=base)


def _deps():
    return _DEPS_OVERRIDE if _DEPS_OVERRIDE is not None else real_deps()


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _write_json_x(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(path, 'x') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=float)
    return _sha(path)


def _save_npz_x(path, **arrs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(path, 'xb') as f:
        np.savez(f, **arrs)
    return _sha(path)


def plan_dir(d):
    return d.runs / FAMILY / '_PLAN'


def param_file(d, config_id, version):
    return d.runs / FAMILY / config_id / f'V{int(version)}' / 'CTL_PARAMS.npz'


# ------------------------------------------------------------------------------------------------ statistics
def _cluster_mean_se(v, unit):
    """Mean of per-person values and its household-clustered SE."""
    _, inv = np.unique(np.asarray(unit), return_inverse=True)
    S = np.bincount(inv, weights=v)
    n = np.bincount(inv).astype(float)
    m = S.sum() / n.sum()
    e = S - m * n
    H = len(n)
    return float(m), float(np.sqrt(H / (H - 1) * (e ** 2).sum()) / n.sum())


def _folds(unit, version):
    u, inv = np.unique(np.asarray(unit).astype(str), return_inverse=True)
    tag = FOLD_TAG.format(v=int(version))
    f = np.array([int.from_bytes(hashlib.sha256(f'{tag}|{x}'.encode()).digest()[:8], 'big') % N_FOLDS for x in u])
    return f[inv]


def _deltas(P0, Q, y):
    rows = np.arange(len(y))
    dn = np.log(np.maximum(P0[rows, y], 1e-300)) - np.log(np.maximum(Q[rows, y], 1e-300))
    E = np.eye(P0.shape[1])[y]
    db = ((Q - E) ** 2).sum(1) - ((P0 - E) ** 2).sum(1)
    return dn, db


# ------------------------------------------------------------------------------------------------ plan / choose / fit
def plan(version, card, d):
    """CALIBRATION label read; remainder shares on all of CALIBRATION; cross-fitted spend of every ladder level."""
    t0 = time.process_time()
    base, brec = d.load_base(version)
    if brec.get('randomized'):
        raise ValueError('CTL_REM needs a deterministic base release')
    lab = d.load_labels('CALIBRATION', ('occupation',), 'calibrate',
                        f'CTL_REM remainder shares and cross-fitted spend prediction on {d.base[0]}/{d.base[1]} V{version} '
                        f'(card {card}; label-only, no SEX)', card)
    y = np.asarray(lab['occupation']).astype(np.int64)
    unit = np.asarray(lab['unit'])
    Pc = P.check_prob(base['CALIBRATION']['occupation'], 'base CALIBRATION occupation')
    if len(Pc) != len(y):
        raise ValueError('base CALIBRATION rows differ from the CALIBRATION label rows')
    R = P.fit_remainder_shares(Pc, 'CALIBRATION', y)
    fold = _folds(unit, version)
    levels = {}
    oof = {s: (np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y), bool)) for s in REM_LADDER}
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        Rk = P.fit_remainder_shares(Pc[tr], 'CALIBRATION', y[tr])
        for s in REM_LADDER:
            Q = P.flatten_remainder(Pc[te], Rk, s)
            dn, db = _deltas(Pc[te], Q, y[te])
            oof[s][0][te], oof[s][1][te] = dn, db
            oof[s][2][te] = np.abs(Q - Pc[te]).max(1) > 0
    for s in REM_LADDER:
        m, se = _cluster_mean_se(oof[s][0], unit)
        mb, seb = _cluster_mean_se(oof[s][1], unit)
        levels[_key(s)] = {'s': s, 'config_id': level_id(s), 'spend': m, 'se': se, 'occ_brier_spend': mb, 'occ_brier_se': seb,
                           'changed_fraction': float(oof[s][2].mean())}
    out = {'version': int(version), 'base': {'family': d.base[0], 'config_id': d.base[1],
                                             'release_sha256': brec.get('release_sha256')},
           'card': card, 'spend_definition': 'out-of-fold total NLL (income + occupation; income unchanged) of CTL_REM minus the '
                                             'base, nats per person, CALIBRATION, 5 household folds, household-clustered SE',
           'fold_tag': FOLD_TAG.format(v=int(version)), 'n_people': int(len(y)), 'n_households': int(len(np.unique(unit))),
           'ladder': list(REM_LADDER), 'levels': levels, 'shares': R.tolist(), 'cpu_s': round(time.process_time() - t0, 2),
           'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    out['shares_sha256'] = _save_npz_x(plan_dir(d) / f'SHARES_V{int(version)}.npz', R=R)
    out['plan_sha256'] = _write_json_x(plan_dir(d) / f'PLAN_V{int(version)}.json', out)
    return out


def _load_plan(d, version):
    return json.loads((plan_dir(d) / f'PLAN_V{int(version)}.json').read_text())


def choose(plans, targets, write_dir=None):
    """The card's spend-only rule. plans: {v: plan}; targets: {'V<v>': {'S3': c, 'S5': c}} (realized MODEL_SELECT total-NLL spend
    of the candidate strengths over the base). Never reads recovery."""
    vs = sorted(plans)

    def reach(name):
        ok = [s for s in REM_LADDER
              if all(plans[v]['levels'][_key(s)]['spend'] >= float(targets[f'V{v}'][name]) + Z_BRACKET *
                     plans[v]['levels'][_key(s)]['se'] for v in vs)]
        return (min(ok), True) if ok else (REM_LADDER[-1], False)

    def below(s):
        i = REM_LADDER.index(s)
        return REM_LADDER[i - 1] if i > 0 else None

    names = sorted(targets[f'V{vs[0]}'])           # target names from the version entries (other keys, e.g. a definition, ignored)
    first, bracketed = {}, {}
    for n in names:
        first[n], bracketed[n] = reach(n)
    picks = sorted(set(first.values()))
    if len(picks) == 1 and below(picks[0]) is not None:
        picks = [below(picks[0]), picks[0]]
    out = {'rule': f'per target: the smallest ladder level whose predicted CALIBRATION total-NLL spend >= the target'
                   f'{"" if not Z_BRACKET else f" + {Z_BRACKET} SE"} in both versions; a target no level reaches gets the strongest '
                   'level (unbracketed, reported: NOT_SEPARATED on this path); one distinct level -> add the next lower level',
           'targets': targets, 'first_bracketing_level': first, 'bracketed_by_prediction': bracketed,
           'unbracketed_reads': {n: 'NOT_SEPARATED on the CTL_REM path' for n in names if not bracketed[n]}, 'levels': picks[:2],
           'config_ids': [level_id(s) for s in picks[:2]],
           'plans_sha256': {f'V{v}': plans[v].get('plan_sha256') for v in vs},
           'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if write_dir is not None:
        out['choice_sha256'] = _write_json_x(Path(write_dir) / 'CHOICE.json', out)
    return out


def fit(version, card, d):
    """Label-free: save the chosen CTL_REM levels of one version (all six roles)."""
    t0 = time.process_time()
    choice_path = plan_dir(d) / 'CHOICE.json'
    plan_path = plan_dir(d) / f'PLAN_V{int(version)}.json'
    shares_path = plan_dir(d) / f'SHARES_V{int(version)}.npz'
    choice = json.loads(choice_path.read_text())
    pl = json.loads(plan_path.read_text())
    hashes = {'plan_sha256': _sha(plan_path), 'choice_sha256': _sha(choice_path), 'shares_sha256': _sha(shares_path)}
    with np.load(shares_path) as z:
        R = z['R'].copy()
    base, brec = d.load_base(version)
    if brec.get('randomized') or brec.get('release_sha256') != pl['base']['release_sha256']:
        raise ValueError('the base release differs from the planned base')
    saved = []
    for s in choice['levels']:
        cfg = level_id(s)
        arrays = {r: {'income': np.array(base[r]['income'], dtype=np.float64, copy=True),
                      'occupation': P.flatten_remainder(base[r]['occupation'], R, s)} for r in d.roles}
        pf = param_file(d, cfg, version)
        psha = _save_npz_x(pf, R=R, s=np.float64(s), base_family=np.array(d.base[0]), base_config=np.array(d.base[1]),
                           base_release_sha256=np.array(str(brec.get('release_sha256'))))
        lv = pl['levels'][_key(s)]
        record = {
            'mechanism_card': card, 'kind': 'control',
            'description': f'CTL_REM s={s:.2f}: decision- and top-probability-preserving remainder flattening of '
                           f'{d.base[0]}/{d.base[1]} occupation toward SEX-blind CALIBRATION shares; income = the base income',
            'n_model_fits': 0, 'n_calibrator_fits': 1, 'n_crossfit_fits': N_FOLDS, 'n_critic_fits': 0,
            'roles_read': 'CALIBRATION: occupation (remainder shares and the cross-fitted spend prediction of the plan step)',
            'exposure': 'CALIBRATION occupation labels reached the remainder shares (one fit per version, shared by the version\'s '
                        'CTL_REM levels); no SEX, MODEL_FIT or MODEL_SELECT label reached any CTL parameter; the base parameters '
                        'are those of the frozen base release',
            'training_people': pl['n_people'], 'training_households': pl['n_households'],
            'fit_cpu_s': round(time.process_time() - t0, 3), 'randomized': False,
            'base': {'family': d.base[0], 'config_id': d.base[1], 'version': int(version),
                     'release_sha256': brec.get('release_sha256')},
            'ctl': {'operation': 'REM', 's': float(s), 'ladder': list(REM_LADDER), 'shares': R.tolist(),
                    'shares_role': 'CALIBRATION', 'predicted_spend': lv, 'choice_levels': choice['levels'],
                    'choice_targets': choice['targets'], **hashes, 'params_file': str(pf), 'params_sha256': psha}}
        d.save(FAMILY, cfg, int(version), arrays, record)
        saved.append(cfg)
    return saved


# ------------------------------------------------------------------------------------------------ deployment interface
def _params(config_id, version, d):
    with np.load(param_file(d, config_id, version)) as z:
        return {'R': z['R'].copy(), 's': float(z['s']), 'base_family': str(z['base_family']),
                'base_config': str(z['base_config'])}


def _base_module(family, d):
    if family == d.base[0] and d.base_module is not None:
        return d.base_module
    from ..deploy import family_module
    return family_module(family)


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deterministic: the base's prediction from admitted features, then the saved remainder flattening (keys refused)."""
    if keys is not None:
        raise PermissionError('draw keys are refused by a deterministic CTL configuration')
    if not isinstance(draw, (int, np.integer)) or int(draw) != 0:
        raise ValueError('a deterministic CTL configuration has only draw 0')
    d = _deps()
    par = _params(config_id, version, d)
    out = _base_module(par['base_family'], d).predict(par['base_config'], int(version), X, feature_names)
    return {'income': np.ascontiguousarray(np.asarray(out['income'], dtype=np.float64)),
            'occupation': P.flatten_remainder(np.asarray(out['occupation'], dtype=np.float64), par['R'], par['s'])}


def model_paths(config_id, version):
    d = _deps()
    par = _params(config_id, version, d)
    base = _base_module(par['base_family'], d)
    return [str(param_file(d, config_id, version))] + [str(p) for p in base.model_paths(par['base_config'], int(version))]


def code_modules():
    d = _deps()
    base = _base_module(d.base[0], d)
    return ('nrm.ctl.post', 'nrm.ctl.run', 'nrm.ctl.release', base.__name__) + tuple(getattr(base, 'CODE_MODULES', ()))


# ------------------------------------------------------------------------------------------------ selftest (invented arrays)
def selftest():
    global _DEPS_OVERRIDE
    checks = {}
    rng = np.random.default_rng(11)
    tmp = Path(tempfile.mkdtemp(prefix='nrm_ctl_run_selftest_'))
    roles = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
    names = tuple(f'f{j}' for j in range(6))
    feats = {r: {'X': rng.normal(size=(n, 6)), 'row_id': np.arange(n) * 3 + 1, 'unit': np.arange(n) // 2}
             for r, n in zip(roles, (900, 1200, 500, 300, 200, 250))}
    Wts = {v: (np.random.default_rng(100 + v).normal(size=(6, 2)), np.random.default_rng(200 + v).normal(size=(6, 5)) * 1.5)
           for v in (0, 1)}

    def softmax(L):
        L = L - L.max(1, keepdims=True)
        E = np.exp(L)
        return E / E.sum(1, keepdims=True)

    def fake_predict(config_id, version, X, feature_names, draw=0, keys=None):
        if keys is not None:
            raise PermissionError('deterministic')
        rowwise = lambda W: (X[:, :, None] * W[None]).sum(1)          # row-independent (no BLAS blocking)
        return {'income': softmax(rowwise(Wts[version][0])), 'occupation': softmax(rowwise(Wts[version][1]))}

    import types
    FakeBase = types.SimpleNamespace(__name__='fake.base', CODE_MODULES=('fake.base.core',), predict=fake_predict,
                                     model_paths=lambda config_id, version: [str(tmp / f'base_V{version}.bin')])

    base_arrays = {v: {r: FakeBase.predict('BASE', v, feats[r]['X'], names) for r in roles} for v in (0, 1)}
    y_cal = {v: (rng.random(1200)[:, None] > np.cumsum(base_arrays[v]['CALIBRATION']['occupation'], 1)).sum(1).clip(0, 4)
             for v in (0, 1)}
    reads, saved = [], {}
    cur = {'v': 0}

    def load_labels(role, fields, operation, purpose, card):
        reads.append((role, tuple(fields), operation, card))
        if role != 'CALIBRATION' or operation != 'calibrate' or set(fields) - {'occupation'} or not card:
            raise PermissionError('selftest fake access: only CALIBRATION occupation under calibrate with a card')
        return {'row_id': feats['CALIBRATION']['row_id'], 'unit': feats['CALIBRATION']['unit'], 'occupation': y_cal[cur['v']]}

    def save(family, config_id, version, arrays, record):
        if [k for k in REQUIRED if k not in record] or (family, config_id, version) in saved:
            raise ValueError('incomplete record or overwrite')
        saved[(family, config_id, version)] = (arrays, record)

    d = Deps(runs=tmp, load_labels=load_labels, load_base=lambda v: (base_arrays[v], {'randomized': False,
                                                                                       'release_sha256': f'base{v}'}),
             save=save, base_module=FakeBase, roles=roles, base=('FAKE', 'BASE'))
    _DEPS_OVERRIDE = d
    try:
        plans = {}
        for v in (0, 1):
            cur['v'] = v
            plans[v] = plan(v, 'CARD_PATH', d)
        checks['plan_reads_only_calibration_occupation'] = all(r[:3] == ('CALIBRATION', ('occupation',), 'calibrate') for r in reads)
        sp = {v: np.array([plans[v]['levels'][_key(s)]['spend'] for s in REM_LADDER]) for v in (0, 1)}
        se = {v: np.array([plans[v]['levels'][_key(s)]['se'] for s in REM_LADDER]) for v in (0, 1)}
        checks['plan_spends_finite_positive_se'] = all(np.isfinite(sp[v]).all() and (se[v] > 0).all() for v in (0, 1))
        top = min(sp[0][-1], sp[1][-1])
        tg = {f'V{v}': {'S3': 0.25 * top, 'S5': 0.6 * top} for v in (0, 1)}
        ch = choose(plans, tg, write_dir=plan_dir(d))
        ok_br = all(ch['bracketed_by_prediction'].values())
        for n in ('S3', 'S5'):
            s = ch['first_bracketing_level'][n]
            i = REM_LADDER.index(s)
            ok_br &= all(plans[v]['levels'][_key(s)]['spend'] >= tg[f'V{v}'][n] for v in (0, 1)) and s in ch['levels'] and \
                (i == 0 or any(plans[v]['levels'][_key(REM_LADDER[i - 1])]['spend'] < tg[f'V{v}'][n] for v in (0, 1)))
        checks['choice_levels_at_or_just_above_targets'] = bool(ok_br and 1 <= len(ch['levels']) <= 2)
        un = choose(plans, {f'V{v}': {'S3': 10.0, 'S5': 20.0} for v in (0, 1)})
        checks['unreachable_targets_get_strongest_level_reported'] = un['levels'] == [REM_LADDER[-2], REM_LADDER[-1]] and \
            not any(un['bracketed_by_prediction'].values()) and set(un['unbracketed_reads']) == {'S3', 'S5'}
        one = choose(plans, {f'V{v}': {'S3': 0.5 * top, 'S5': 0.5 * top} for v in (0, 1)})
        checks['single_level_gets_next_lower_level'] = len(one['levels']) == 2 and \
            REM_LADDER.index(one['levels'][1]) - REM_LADDER.index(one['levels'][0]) == 1
        for v in (0, 1):
            fit(v, 'CARD_PATH', d)
        checks['fit_saves_chosen_levels_in_both_versions'] = set(saved) == {(FAMILY, level_id(s), v) for s in ch['levels']
                                                                            for v in (0, 1)}
        good = True
        for (fam, cfg, v), (arr, rec) in saved.items():
            for r in roles:
                b = base_arrays[v][r]
                q = arr[r]['occupation']
                good &= np.array_equal(arr[r]['income'], b['income']) and (q.argmax(1) == b['occupation'].argmax(1)).all() \
                    and np.allclose(q.max(1), b['occupation'].max(1), atol=1e-12) and np.allclose(q.sum(1), 1, atol=1e-12)
            good &= rec['kind'] == 'control' and rec['randomized'] is False and rec['n_critic_fits'] == 0
        checks['income_unchanged_decisions_and_top_probability_kept'] = bool(good)
        bit = True
        for (fam, cfg, v), (arr, rec) in saved.items():
            for r in roles:
                out = predict(cfg, v, feats[r]['X'], names)
                bit &= np.array_equal(out['occupation'], arr[r]['occupation']) and np.array_equal(out['income'], arr[r]['income'])
        checks['predict_from_features_reproduces_release_bitwise'] = bool(bit)
        cfg0 = level_id(ch['levels'][0])
        sub = predict(cfg0, 0, feats['DEV_REPORT']['X'][:17], names)['occupation']
        checks['predict_row_independent'] = bool(np.array_equal(sub, saved[(FAMILY, cfg0, 0)][0]['DEV_REPORT']['occupation'][:17]))
        try:
            predict(cfg0, 0, feats['DEV_REPORT']['X'], names, keys={'role': 'DEV_REPORT', 'row_id': feats['DEV_REPORT']['row_id']})
            checks['keys_refused'] = False
        except PermissionError:
            checks['keys_refused'] = True
        try:
            predict(cfg0, 0, feats['DEV_REPORT']['X'], names, draw=1)
            checks['draw_other_than_0_refused'] = False
        except ValueError:
            checks['draw_other_than_0_refused'] = True
        mp = model_paths(cfg0, 1)
        checks['model_paths_pin_params_and_base'] = mp[0].endswith('CTL_PARAMS.npz') and mp[1].endswith('base_V1.bin')
        checks['code_modules_include_base'] = {'nrm.ctl.post', 'nrm.ctl.run', 'fake.base', 'fake.base.core'} <= set(code_modules())
        try:
            fit(0, 'CARD_PATH', d)
            checks['write_once_refit_refused'] = False
        except (FileExistsError, ValueError):
            checks['write_once_refit_refused'] = True
        info = {'ladder_spend_V0': [round(x, 5) for x in sp[0]], 'ladder_se_V0': [round(x, 5) for x in se[0]], 'targets': tg,
                'levels': ch['levels'], 'bracketed_by_prediction': ch['bracketed_by_prediction']}
    finally:
        _DEPS_OVERRIDE = None
    ok = all(checks.values())
    print('CTL_RUN', json.dumps({'ok': ok, 'checks': checks}))
    print('CTL_RUN_INFO', json.dumps(info))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python3 -m nrm.ctl.run')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('plan', 'fit'):
        p = sub.add_parser(name)
        p.add_argument('--version', type=int, required=True)
        p.add_argument('--card', default=CARD)
    c = sub.add_parser('choose')
    c.add_argument('--targets', required=True, help='JSON {"V0": {"S3": c, "S5": c}, "V1": {...}}: realized MODEL_SELECT '
                                                    'total-NLL spends of the candidate strengths over the base')
    sub.add_parser('selftest')
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return 0 if selftest() else 1
    d = real_deps()
    if a.cmd == 'plan':
        res = plan(a.version, a.card, d)
        print(json.dumps({k: res[k] for k in ('version', 'n_people', 'n_households', 'plan_sha256', 'cpu_s')}))
        print(json.dumps({k: (round(v['spend'], 5), round(v['se'], 5)) for k, v in res['levels'].items()}))
    elif a.cmd == 'choose':
        targets = json.loads(Path(a.targets).read_text())
        res = choose({v: _load_plan(d, v) for v in (0, 1)}, targets, write_dir=plan_dir(d))
        print(json.dumps({k: res[k] for k in ('levels', 'config_ids', 'first_bracketing_level', 'bracketed_by_prediction',
                                              'unbracketed_reads')}))
    elif a.cmd == 'fit':
        print(json.dumps({'saved': fit(a.version, a.card, d)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
