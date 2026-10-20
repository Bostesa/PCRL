"""Family B learner (torch, one thread): the task-only twin and the channel-aware continuation of the two task networks under the
FSC channel (nrm.fb.channel), CALIBRATION temperatures, and the release of all family-B configurations of one version.

Census context: public-use ACS 2019 1-Year PUMS Florida person records; six raw fields -> admitted 145-column encoding X -> income (2)
and occupation (5) probability vectors, one per recipient, decision = argmax of the emitted vector. SEX is read only on MODEL_FIT, as
the label of the channel geometry (delta, Sigma_w); never a runtime input or output. CALIBRATION (no SEX) fits temperatures only.
MODEL_SELECT labels are never read here (feedback only through nrm.evaluate / nrm.screen outputs).

Per version v (both versions; seeds and the internal household split depend on v only):
  stage 1   per task, network d-128-ReLU-128-ReLU-32-linear K (the U0 Recipe A shape, widened 2x); Adam lr 1e-3, batch 512, at most
            40 epochs, early stopping (patience 5, min delta 1e-4) on a household-grouped 15% internal split of MODEL_FIT keyed by v;
            refit from the same initialisation on ALL MODEL_FIT people for the selected epoch count (non-nested; actual training size
            20,000 people / 13,337 households).
  stage 2   every learner continues from the stage-1 networks for 6 epochs (Adam lr 3e-4, batch 512, both task losses summed):
              'none'   task-only continuation (the equally trained twin);
              'sex:Sj' the expected task loss of the EMITTED vectors under the FSC channel (mode 'sex', strength s_j): one training
                       draw per example per step, reparameterised (gradient through the deterministic logits; the per-person noise
                       operator is recomputed from the detached logits); the channel geometry is re-estimated at the start of every
                       epoch from the current deterministic logits of all MODEL_FIT people (SEX labels);
              'pca:Sj' the same with the SEX-blind direction (mode 'pca', same s_j: same second-order Fisher cost by construction).
            The permutation stream and the training-noise stream are identical across the seven learners of a version (paired).
  release   geometry re-estimated on the final learner's MODEL_FIT logits and frozen; per-task temperatures fitted on CALIBRATION
            (golden section on log T in [log 0.5, log 3], two coordinate sweeps; randomized channels: mean emitted NLL over 8 keyed
            calibration draws; mean-vector construction: its exact NLL); emitted vectors for all six roles with R = 8 independent
            one-draw-per-person repetitions keyed by a private per-(config, version) key (nrm.fb.channel.release_key).

Configurations of one version (8; 16 over both versions; the coordinator's allocation dropped the S1 strength):
  FB_TWIN (twin), FB_SEX_S2/S3 (candidate), FB_PCA_S2/S3 (SEX-blind matched control), FB_SEX_S2_ID (identity channel of the S2
  learner), FB_SEX_S2_MEAN (deterministic mean vector of the S2 channel), FB_POSTHOC_S2 (S2 channel on the twin learner, no
  adaptation). Both the SEX and the PCA paths use the same coupled rank-1 draw, so their comparison isolates targeting; no
  joint-design or coupling claim is made (an uncoupled control is not funded).
Model files (write-once, under runs/FB/MODELS/V<v>/, read by nrm.fb.release.predict): LEARNER_<learner>.pt, CHANNEL_<config>.json
(frozen geometry, temperatures, strength, cap, feature names), RELEASE_ROOT.bin (private 32-byte root of the release keys).

Commands:
  selftest                                   invented data only (fake access hooks; nothing written outside a temporary folder)
  fit --version V --card <committed card>    real fit (through the queue; refuses before the registration receipt)
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import channel as CH
from . import release as RL

FAMILY = 'FB'
TASKS = ('income', 'occupation')
SCHEMA = 'NRM-FB-learner-v1'
STRENGTHS = {'S2': 0.09, 'S3': 0.12}        # S1 (0.06) dropped by the coordinator's allocation (16 configurations)
SPEC = {
    'architecture': 'per task: Linear(d,128)-ReLU-Linear(128,128)-ReLU-Linear(128,32)-Linear(32,K) (U0 Recipe A shape, widened 2x)',
    'train': {'optimizer': 'Adam (both task networks, summed emitted task loss)', 'lr': 1e-3, 'batch': 512, 'max_epochs': 40,
              'patience': 5, 'min_delta': 1e-4, 'val_fraction': 0.15,
              'refit': 'from the same initialisation on all MODEL_FIT people for the selected epoch count (non-nested)'},
    'calibration': {'bounds_log_t': [float(np.log(0.5)), float(np.log(3.0))], 'sweeps': 2, 'golden_iters': 24, 'draws': 8},
    'release_draws': 8, 'cap': CH.CAP, 'strengths': STRENGTHS,
}
LEARNERS = ('none',) + tuple(f'{m}:{j}' for m in ('sex', 'pca') for j in STRENGTHS)
CONFIGS = {
    'FB_TWIN': {'learner': 'none', 'mode': 'none', 's': 0.0, 'kind': 'twin',
                'description': 'task-only twin: stage-1 networks + 6 task-only continuation epochs; identity channel'},
    **{f'FB_SEX_{j}': {'learner': f'sex:{j}', 'mode': 'sex', 's': s, 'kind': 'candidate',
                       'description': f'channel-aware learner + FSC channel, SEX-targeted direction, s = {s}'} for j, s in STRENGTHS.items()},
    **{f'FB_PCA_{j}': {'learner': f'pca:{j}', 'mode': 'pca', 's': s, 'kind': 'control',
                       'description': f'SEX-blind matched control: channel-aware learner + FSC channel, top-PC direction, s = {s}'}
       for j, s in STRENGTHS.items()},
    'FB_SEX_S2_ID': {'learner': 'sex:S2', 'mode': 'none', 's': 0.0, 'kind': 'ablation',
                     'description': 'identity channel of the FB_SEX_S2 learner (no noise)'},
    'FB_SEX_S2_MEAN': {'learner': 'sex:S2', 'mode': 'sex', 's': STRENGTHS['S2'], 'mean_vector': True, 'kind': 'ablation',
                       'description': 'deterministic construction: mean vector E[emitted] of the FB_SEX_S2 channel'},
    'FB_POSTHOC_S2': {'learner': 'none', 'mode': 'sex', 's': STRENGTHS['S2'], 'kind': 'ablation',
                      'description': 'S2 SEX-targeted channel on the twin learner (no channel-aware adaptation; frozen-learner analogue)'},
}
SEED_INIT, SEED_TRAIN = 7100, 7200
VAL_KEY = hashlib.blake2b(b'NRM-FB internal-validation draws (fit-time only; not a release key)', digest_size=32).digest()


# ------------------------------------------------------------------ access hooks (the selftest replaces them)
_NO_REAL_ACCESS = False


def _load(role, fields, operation, purpose, card):
    if _NO_REAL_ACCESS:
        raise PermissionError('selftest: no real label read')
    from .. import common as C
    return C.load(role, fields, operation, purpose, card=card)


def _features(role):
    if _NO_REAL_ACCESS:
        raise PermissionError('selftest: no real feature read')
    from .. import common as C
    return C.features(role)


def _save(config_id, version, arrays, record):
    from .. import release as REL
    return REL.save(FAMILY, config_id, version, arrays, record)


def _roles():
    from .. import common as C
    return C.ROLES6


# ------------------------------------------------------------------ torch pieces
def _torch():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def network(d, k):
    torch = _torch()
    nn = torch.nn
    return nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 32), nn.Linear(32, k))


def household_split(unit, v, frac):
    """Household-grouped internal split keyed by the version: (fit_rows, val_rows)."""
    u = np.asarray(unit)
    hh = np.unique(u)
    score = np.array([int.from_bytes(hashlib.blake2b(f'NRM-FB-split|V{int(v)}|{h}'.encode(), digest_size=8).digest(), 'little') for h in hh])
    val_hh = set(hh[np.argsort(score, kind='stable')[:int(round(frac * len(hh)))]].tolist())
    is_val = np.array([x in val_hh for x in u.tolist()])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def logits_z(nets, X):
    """Deterministic centred logits z (n, 5), float64, from admitted features."""
    torch = _torch()
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    with torch.no_grad():
        li = nets['income'].eval()(Xt).double().numpy(); lo = nets['occupation'].eval()(Xt).double().numpy()
    return CH.to_z(li, lo)


def _mode_s(learner):
    return ('none', 0.0) if learner == 'none' else (learner.split(':')[0], STRENGTHS[learner.split(':')[1]])


def _init_nets(d, v):
    torch = _torch()
    nets = {}
    for i, t in enumerate(TASKS):
        torch.manual_seed(SEED_INIT + 10 * int(v) + i)          # same initialisation for every learner of a version (paired)
        nets[t] = network(d, CH.K[t])
    return nets


def _train_epoch(nets, opt, Xt, yi, yo, rows, X_np, sex, mode, s, v, phase, e, batch, H2t, H5t):
    """One epoch of the summed emitted task loss over `rows`; the geometry is re-estimated from the current deterministic logits of
    `rows` at the start of the epoch. Permutation and training-noise streams depend only on (v, phase, epoch): paired learners."""
    torch = _torch()
    geom = CH.geometry(logits_z(nets, X_np[rows]), sex[rows], mode) if mode in ('sex', 'pca') else {'mode': mode}
    chan = {'mode': mode, 's': s, 'geom': geom}
    order = rows[np.random.default_rng([SEED_TRAIN + int(v), phase, e, 0]).permutation(len(rows))]
    noise_rng = np.random.default_rng([SEED_TRAIN + int(v), phase, e, 1])
    for t in TASKS:
        nets[t].train()
    tot, nb = 0.0, 0
    for b0 in range(0, len(order), batch):
        b = order[b0:b0 + batch]
        eps = noise_rng.standard_normal((len(b), CH.DZ))
        z = torch.cat([nets['income'](Xt[b]) @ H2t.T, nets['occupation'](Xt[b]) @ H5t.T], 1)
        if mode != 'none' and s > 0:
            op = CH.operators(z.detach().double().numpy(), chan)
            dz = op * eps[:, :1] if op.ndim == 2 else np.einsum('nab,nb->na', op, eps)
            z = z + torch.from_numpy(dz.astype(np.float32))
        loss = torch.nn.functional.cross_entropy(z[:, :1] @ H2t, yi[b]) + torch.nn.functional.cross_entropy(z[:, 1:] @ H5t, yo[b])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tot += float(loss.detach()) * len(b); nb += len(b)
    return geom, tot / max(nb, 1), order


def _val_emitted_nll(nets, X_np, ys, geom, mode, s, rows, v):
    """Summed emitted NLL of the validation people under the current channel; fixed per-person draws (common random numbers)."""
    z = logits_z(nets, X_np[rows])
    if mode == 'none' or s == 0:
        pi, po = CH.probs(z)
        P = {'income': pi, 'occupation': po}
    else:
        chan = {'mode': mode, 's': s, 'geom': geom}
        P = CH.emit(z, chan, VAL_KEY, f'VAL{int(v)}', rows, 1, tag_prefix='NRM-FB-VAL')
    return sum(CH.task_metrics(P[t], ys[t][rows])['nll'] for t in TASKS)


def fit_learner(X, ys, sex, unit, v, learner, spec=None):
    """Joint from-scratch fit of both task networks under the learner's channel (identity for 'none'): early stopping on the summed
    emitted NLL of a household-grouped 15% internal split, then refit from the same initialisation on all rows for the selected
    epoch count. Every learner of a version shares the initialisation, the split, the permutations and the training-noise draws."""
    torch = _torch()
    sp = spec or SPEC['train']
    mode, s = _mode_s(learner)
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    yi = torch.from_numpy(np.ascontiguousarray(ys['income'], dtype=np.int64))
    yo = torch.from_numpy(np.ascontiguousarray(ys['occupation'], dtype=np.int64))
    H2t, H5t = torch.from_numpy(CH.H2.astype(np.float32)), torch.from_numpy(CH.H5.astype(np.float32))
    fit_rows, val_rows = household_split(unit, v, sp['val_fraction'])
    log = {'learner': learner, 'mode': mode, 's': s, 'fit_rows': int(len(fit_rows)), 'val_rows': int(len(val_rows)), 'search': [], 'refit': []}
    nets = _init_nets(X.shape[1], v)
    opt = torch.optim.Adam([p for t in TASKS for p in nets[t].parameters()], lr=sp['lr'])
    best, best_e, bad = float('inf'), 0, 0
    order_sha, first_sha = hashlib.sha256(), None
    for e in range(1, sp['max_epochs'] + 1):
        geom, tr, order = _train_epoch(nets, opt, Xt, yi, yo, fit_rows, X, sex, mode, s, v, 0, e, sp['batch'], H2t, H5t)
        order_sha.update(order.tobytes())
        first_sha = first_sha or hashlib.sha256(order.tobytes()).hexdigest()
        g_now = CH.geometry(logits_z(nets, X[fit_rows]), sex[fit_rows], mode) if mode in ('sex', 'pca') else {'mode': mode}
        val = _val_emitted_nll(nets, X, ys, g_now, mode, s, val_rows, v)
        log['search'].append({'epoch': e, 'train_emitted_loss': tr, 'val_emitted_nll': val, 'lda_delta2_fit_rows': g_now.get('lda_delta2')})
        if val < best - sp['min_delta']:
            best, best_e, bad = val, e, 0
        else:
            bad += 1
            if bad >= sp['patience']:
                break
    nets = _init_nets(X.shape[1], v)
    opt = torch.optim.Adam([p for t in TASKS for p in nets[t].parameters()], lr=sp['lr'])
    all_rows = np.arange(len(X))
    for e in range(1, best_e + 1):
        geom, tr, order = _train_epoch(nets, opt, Xt, yi, yo, all_rows, X, sex, mode, s, v, 1, e, sp['batch'], H2t, H5t)
        order_sha.update(order.tobytes())
        log['refit'].append({'epoch': e, 'train_emitted_loss': tr, 'lda_delta2_at_epoch_start': geom.get('lda_delta2')})
    log.update({'selected_epochs': best_e, 'best_val_emitted_nll': best, 'order_stream_sha256': order_sha.hexdigest(),
                'first_epoch_order_sha256': first_sha})
    return nets, log


# ------------------------------------------------------------------ channel record, temperatures and emission
def channel_for(nets, X_fit, sex_fit, cfg):
    if cfg['mode'] in ('none', 'iso'):
        return {'mode': cfg['mode'], 's': cfg['s'], 'geom': {'mode': cfg['mode']}, 'cap': CH.CAP}
    return {'mode': cfg['mode'], 's': cfg['s'], 'geom': CH.geometry(logits_z(nets, X_fit), sex_fit, cfg['mode']), 'cap': CH.CAP}


def scaled(chan, temps):
    return {**chan, 'geom': CH.scale_geometry(chan['geom'], *temps)}


def emitted(z, chan, temps, key, role, row_id, reps, mean_vector=False, tag_prefix='NRM-FB'):
    zs = z / np.array([temps[0]] + [temps[1]] * 4)
    ch = scaled(chan, temps)
    if mean_vector:
        return CH.mean_vector(zs, ch)
    return CH.emit(zs, ch, key, role, row_id, reps, tag_prefix=tag_prefix)


def _golden(f, lo, hi, iters):
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc <= fd:
            b, d, fd = d, c, fc; c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd; d = a + g * (b - a); fd = f(d)
    return (a + b) / 2


def fit_temperatures(z_cal, y_cal, chan, cal_key, row_id, mean_vector=False, spec=None):
    """Per-task temperatures on CALIBRATION minimising the mean emitted NLL (common keyed draws across evaluations)."""
    sp = spec or SPEC['calibration']
    reps = 1 if (chan['mode'] == 'none' or chan['s'] == 0 or mean_vector) else sp['draws']
    lt = [0.0, 0.0]

    def nll(task_index, log_t):
        t = list(lt); t[task_index] = log_t
        P = emitted(z_cal, chan, (float(np.exp(t[0])), float(np.exp(t[1]))), cal_key, 'CALIBRATION', row_id, reps, mean_vector,
                    tag_prefix='NRM-FB-CAL')
        task = TASKS[task_index]
        return CH.task_metrics(P[task], y_cal[task])['nll']
    for _ in range(sp['sweeps']):
        for i in range(2):
            lt[i] = _golden(lambda x: nll(i, x), *sp['bounds_log_t'], sp['golden_iters'])
    return (float(np.exp(lt[0])), float(np.exp(lt[1])))


# ------------------------------------------------------------------ one version: all configurations
def fit_version(v, card, root=None, spec=None, log=print):
    """Real fit of every family-B configuration of version v (through the queue, after the receipt and the committed card)."""
    t_all = time.process_time()
    v = int(v)
    sp = spec or SPEC
    F = _load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit',
              f'NRM family B V{v}: task-only twin and channel-aware learners (income, occupation task losses); SEX only for the '
              'channel geometry (SEX-mean difference and within-SEX covariance of the linked centred logits)', card)
    Cal = _load('CALIBRATION', ('features', 'income', 'occupation'), 'calibrate',
                f'NRM family B V{v}: per-task temperatures of every configuration (calibration parameters only)', card)
    X, unit, sex = F['X'], F['unit'], np.asarray(F['sex'], np.int64)
    ys = {t: np.asarray(F[t], np.int64) for t in TASKS}
    ycal = {t: np.asarray(Cal[t], np.int64) for t in TASKS}
    pdir = RL.model_dir(v, root)
    pdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    key_path = RL.root_key_file(v, root)
    if key_path.exists():
        raise PermissionError(f'{key_path} exists: version {v} was already fitted (append-only)')
    root_key = os.urandom(32)
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, root_key); os.close(fd)
    cal_key = CH.release_key(root_key, 'CALIBRATION-DRAWS', v)
    learners, logs, cpu = {}, {}, {}
    for name in LEARNERS:
        t0 = time.process_time()
        learners[name], logs[name] = fit_learner(X, ys, sex, unit, v, name, sp['train'])
        cpu[name] = time.process_time() - t0
        log(json.dumps({'version': v, 'learner': name, 'cpu_s': round(cpu[name], 1), 'selected_epochs': logs[name]['selected_epochs'], 'best_val_emitted_nll': round(logs[name]['best_val_emitted_nll'], 5)}))
    for name, nets in learners.items():                    # model files first: every saved release names them with hashes
        with open(RL.learner_file(name, v, root), 'xb') as f:
            _torch().save({t: nets[t].state_dict() for t in TASKS}, f)
    roles = {r: _features(r) for r in _roles()}
    z_role = {name: {r: logits_z(learners[name], roles[r]['X']) for r in roles} for name in LEARNERS}
    twin_fit = {t: CH.task_metrics(CH.probs(z_role['none']['MODEL_FIT'])[i], ys[t]) for i, t in enumerate(TASKS)}
    saved = {}
    for cid, cfg in CONFIGS.items():
        t0 = time.process_time()
        nets = learners[cfg['learner']]
        chan = channel_for(nets, X, sex, cfg)
        mv = bool(cfg.get('mean_vector'))
        temps = fit_temperatures(z_role[cfg['learner']]['CALIBRATION'], ycal, chan, cal_key, roles['CALIBRATION']['row_id'], mv, sp['calibration'])
        randomized = not (chan['mode'] == 'none' or chan['s'] == 0 or mv)
        key = CH.release_key(root_key, cid, v)
        reps = sp['release_draws'] if randomized else 1
        arrays = {r: emitted(z_role[cfg['learner']][r], chan, temps, key, r, roles[r]['row_id'], reps, mv) for r in roles}
        ch_rec = {'schema': SCHEMA + '-channel', 'config_id': cid, 'version': v, 'learner': cfg['learner'], 'mode': chan['mode'],
                  's': chan['s'], 'cap': CH.CAP, 'mean_vector': mv, 'randomized': randomized, 'draws': reps,
                  'temperatures': [float(temps[0]), float(temps[1])], 'geometry': chan['geom'],
                  'feature_names': list(map(str, F['feature_names'])), 'tag_prefix': 'NRM-FB',
                  'key_derivation': 'BLAKE2b(key = root, message = "NRM-FB|<config>|V<v>")'}
        with open(RL.channel_file(cid, v, root), 'x') as f:
            json.dump(ch_rec, f, indent=1, sort_keys=True)
        mfiles = [RL.learner_file(cfg['learner'], v, root), RL.channel_file(cid, v, root)] + ([key_path] if randomized else [])
        model_sha = {Path(m).name: hashlib.sha256(Path(m).read_bytes()).hexdigest() for m in mfiles}
        fit_diag = {t: CH.task_metrics(arrays['MODEL_FIT'][t], ys[t]) for t in TASKS}
        zf = arrays['MODEL_FIT']
        d2 = CH.lda_delta2(CH.z_from_probs(zf['income'][0] if randomized else zf['income'],
                                           zf['occupation'][0] if randomized else zf['occupation']), sex)
        own = 0 if cid in ('FB_SEX_S2_ID', 'FB_SEX_S2_MEAN', 'FB_POSTHOC_S2') else 1      # derived configurations reuse a learner
        record = {
            'mechanism_card': card, 'kind': cfg['kind'], 'description': cfg['description'],
            'n_model_fits': 2 * own, 'n_calibrator_fits': 1, 'n_crossfit_fits': 0, 'n_critic_fits': 0,
            'roles_read': 'MODEL_FIT: income, occupation, sex (fit); CALIBRATION: income, occupation (calibrate)',
            'exposure': 'MODEL_FIT task labels reach the networks; MODEL_FIT SEX reaches only the channel geometry (and, through the '
                        'channel-aware loss, the networks of the sex:* learners); CALIBRATION task labels reach only the two temperatures',
            'training_people': int(len(X)), 'training_households': int(len(np.unique(unit))),
            'fit_cpu_s': float(cpu.get(cfg['learner'], 0.0) * own + (time.process_time() - t0)),
            'randomized': randomized, 'draws': reps,
            'seed_handling': ('emitted draw r of person i = Box-Muller of keyed BLAKE2b(key = private per-(config, version) key derived '
                              'from a private 32-byte root, message = "NRM-FB|<role>|r<r>|<row_id>"); independent of row order; R = 8 '
                              'repetitions = 8 independent one-draw-per-person release instances') if randomized else 'deterministic',
            'noise_coupling': ('one scalar N(0,1) draw per person per repetition shared by the income and occupation recipients along '
                               'the person\'s own rank-1 direction (maximal coupling)') if randomized else 'none',
            'channel': {'mode': chan['mode'], 's': chan['s'], 'cap': CH.CAP, 'mean_vector': mv, 'temperatures': list(temps),
                        'geometry': chan['geom']},
            'learner': cfg['learner'], 'training_log': logs[cfg['learner']], 'spec': sp, 'model_files_sha256': model_sha,
            'coupling_claim': 'none (SEX and PCA paths share the coupled rank-1 draw; the comparison isolates targeting)',
            'fit_time_diagnostics_MODEL_FIT_in_sample': {'emitted': fit_diag, 'twin_identity': twin_fit,
                                                         'lda_delta2_one_draw': d2},
            'schema': SCHEMA,
        }
        saved[cid] = _save(cid, v, arrays, record)
        log(json.dumps({'version': v, 'config': cid, 'temps': [round(x, 4) for x in temps], 'randomized': randomized,
                        'fit_nll': {t: round(fit_diag[t]['nll'], 5) for t in TASKS}, 'lda_delta2': round(d2, 5)}))
    return {'version': v, 'saved': {k: str(p) for k, p in saved.items()}, 'cpu_s': time.process_time() - t_all}


# ------------------------------------------------------------------ selftest (invented data only)
def _invented_world(rng, d=40):
    roles = {'MODEL_FIT': 1600, 'CALIBRATION': 400, 'MODEL_SELECT': 300, 'ATTACK_FIT': 60, 'ATTACK_VALIDATION': 50, 'DEV_REPORT': 70}
    W, start = {}, 0
    Wi, Wo, ws = rng.normal(size=(d, 2)), rng.normal(size=(d, 5)), rng.normal(size=d)
    for r, n in roles.items():
        sex = rng.integers(0, 2, n)
        X = (rng.random((n, d)) < 0.3).astype(np.float64)
        X[:, :5] = (rng.random((n, 5)) < (0.2 + 0.4 * sex[:, None])).astype(np.float64)    # invented SEX-correlated columns
        li, lo = X @ Wi * 0.4, X @ Wo * 0.4
        y_inc = np.array([rng.choice(2, p=p) for p in CH.softmax(li)])
        y_occ = np.array([rng.choice(5, p=p) for p in CH.softmax(lo)])
        unit = start + np.arange(n) // 2
        W[r] = {'X': X, 'feature_names': tuple(f'f{j}' for j in range(d)), 'row_id': np.arange(start, start + n) * 7 + 3,
                'unit': unit, 'income': y_inc, 'occupation': y_occ, 'sex': sex}
        start += n
    return W


def selftest():
    global _NO_REAL_ACCESS, _load, _features, _save, _roles
    t0 = time.process_time()
    rng = np.random.default_rng(20261020)
    W = _invented_world(rng)
    calls, store = [], {}
    saved_hooks = (_load, _features, _save, _roles)
    checks, info = {}, {}
    # refusal before any fake is installed: the real access layer is never reached in the selftest
    _NO_REAL_ACCESS = True
    with tempfile.TemporaryDirectory(prefix='nrm_fb_refuse_') as tmp_refuse:
        try:
            fit_version(0, 'results/none.md', root=tmp_refuse)
            checks['real_access_refused'] = False
        except PermissionError:
            checks['real_access_refused'] = not any(Path(tmp_refuse).iterdir())     # refused before anything was written

    def fake_load(role, fields, operation, purpose, card):
        allowed = {('MODEL_FIT', 'fit'), ('CALIBRATION', 'calibrate')}
        labels = sorted(set(fields) & {'income', 'occupation', 'sex'})
        calls.append({'role': role, 'operation': operation, 'labels': labels, 'card': card})
        if (role, operation) not in allowed or (role == 'CALIBRATION' and 'sex' in labels):
            raise PermissionError('selftest policy')
        return {k: W[role][k] for k in ('X', 'feature_names', 'row_id', 'unit', *labels)}

    def fake_features(role):
        return {k: W[role][k] for k in ('X', 'feature_names', 'row_id', 'unit')}

    def fake_save(cid, v, arrays, record):
        from .. import release as REL
        randomized = bool(record['randomized'])
        for r, n in ((r, len(W[r]['row_id'])) for r in W):
            for t in TASKS:
                a = REL._check_prob(arrays[r][t], CH.K[t], f'{r} {t}', randomized)
                if a.shape[-2] != n or (randomized and a.shape[0] != record['draws']):
                    raise ValueError(f'{cid} {r} {t}: shape {a.shape}')
        missing = [k for k in REL.REQUIRED if k not in record]
        if missing or (randomized and not all(k in record for k in ('draws', 'seed_handling', 'noise_coupling'))):
            raise ValueError(f'{cid}: record misses {missing}')
        store[cid] = {'arrays': arrays, 'record': json.loads(json.dumps(record, default=float))}
        return f'(selftest) {cid}/V{v}'
    _load, _features, _save, _roles = fake_load, fake_features, fake_save, (lambda: tuple(W))
    tiny = {**SPEC, 'train': {**SPEC['train'], 'max_epochs': 4, 'patience': 2},
            'calibration': {**SPEC['calibration'], 'golden_iters': 8, 'sweeps': 1}}
    try:
        with tempfile.TemporaryDirectory(prefix='nrm_fb_selftest_') as tmp:
            res = fit_version(0, 'results/pcrl_new_release_mechanisms_v1/cards/FB_SELFTEST.md', root=tmp, spec=tiny, log=lambda s: None)
            checks['all_configurations_saved'] = sorted(store) == sorted(CONFIGS)
            checks['allocation_16'] = len(CONFIGS) == 8 and 'FB_SEX_S1' not in CONFIGS and 'FB_PCA_S1' not in CONFIGS
            checks['randomized_flags'] = all(store[c]['record']['randomized'] == (c in ('FB_SEX_S2', 'FB_SEX_S3', 'FB_PCA_S2', 'FB_PCA_S3',
                                                                                         'FB_POSTHOC_S2'))
                                             for c in CONFIGS)
            checks['randomized_stack_shape'] = store['FB_SEX_S2']['arrays']['MODEL_SELECT']['occupation'].shape == (8, 300, 5)
            checks['label_reads_policy'] = len(calls) == 2 and \
                calls[0] == {'role': 'MODEL_FIT', 'operation': 'fit', 'labels': ['income', 'occupation', 'sex'], 'card': calls[0]['card']} and \
                calls[1] == {'role': 'CALIBRATION', 'operation': 'calibrate', 'labels': ['income', 'occupation'], 'card': calls[1]['card']}
            orders = {store[c]['record']['training_log']['first_epoch_order_sha256'] for c in CONFIGS}
            checks['paired_streams_across_learners'] = len(orders) == 1
            # identity config = softmax of the S2 learner's temperature-scaled logits; mean config is deterministic
            rec_id = store['FB_SEX_S2_ID']['record']
            checks['identity_config_deterministic'] = rec_id['randomized'] is False and rec_id['draws'] == 1
            checks['mean_vector_config_deterministic'] = store['FB_SEX_S2_MEAN']['record']['randomized'] is False
            # deployment (nrm.fb.release.predict, the nrm/deploy.py contract) reproduces RELEASE draw 0 bitwise for every role
            RL._ROOT_OVERRIDE = tmp
            dep = {}
            for cid in CONFIGS:
                rec = store[cid]['record']
                for r in W:
                    keys = {'role': r, 'row_id': W[r]['row_id']} if rec['randomized'] else None
                    out = RL.predict(cid, 0, W[r]['X'], W[r]['feature_names'], draw=0, keys=keys)
                    ref = store[cid]['arrays'][r]
                    dep[f'{cid}|{r}'] = all(np.array_equal(out[t], ref[t][0] if rec['randomized'] else ref[t]) for t in TASKS)
            checks['predict_reproduces_draw0_every_config_every_role'] = all(dep.values())
            Wd = W['DEV_REPORT']
            o3 = RL.predict('FB_SEX_S2', 0, Wd['X'], Wd['feature_names'], draw=3, keys={'role': 'DEV_REPORT', 'row_id': Wd['row_id']})
            checks['predict_reproduces_instance_3'] = all(np.array_equal(o3[t], store['FB_SEX_S2']['arrays']['DEV_REPORT'][t][3]) for t in TASKS)
            sw = list(Wd['feature_names']); sw[0], sw[1] = sw[1], sw[0]
            nan = Wd['X'].copy(); nan[0, 0] = np.nan
            probes = {'randomized_without_keys': (lambda: RL.predict('FB_SEX_S2', 0, Wd['X'], Wd['feature_names']), PermissionError),
                      'deterministic_with_keys': (lambda: RL.predict('FB_TWIN', 0, Wd['X'], Wd['feature_names'],
                                                                     keys={'role': 'DEV_REPORT', 'row_id': Wd['row_id']}), PermissionError),
                      'deterministic_draw_1': (lambda: RL.predict('FB_SEX_S2_MEAN', 0, Wd['X'], Wd['feature_names'], draw=1), ValueError),
                      'short_keys': (lambda: RL.predict('FB_PCA_S3', 0, Wd['X'], Wd['feature_names'],
                                                        keys={'role': 'DEV_REPORT', 'row_id': Wd['row_id'][:-1]}), PermissionError),
                      'reordered_names': (lambda: RL.predict('FB_TWIN', 0, Wd['X'], tuple(sw)), ValueError),
                      'non_finite': (lambda: RL.predict('FB_TWIN', 0, nan, Wd['feature_names']), ValueError),
                      'wrong_width': (lambda: RL.predict('FB_TWIN', 0, Wd['X'][:, :-1], Wd['feature_names'][:-1]), ValueError),
                      'dropped_config_S1': (lambda: RL.predict('FB_SEX_S1', 0, Wd['X'], Wd['feature_names']), ValueError)}
            pr = {}
            for k_, (f, exc) in probes.items():
                try:
                    f(); pr[k_] = False
                except exc:
                    pr[k_] = True
            info['predict_refusals'] = pr
            checks['predict_refusals'] = all(pr.values())
            mp = {c: RL.model_paths(c, 0) for c in CONFIGS}
            checks['model_paths_exist_and_key_only_if_randomized'] = all(
                all(Path(x).is_file() for x in mp[c]) and (any(x.endswith('RELEASE_ROOT.bin') for x in mp[c]) == store[c]['record']['randomized'])
                for c in CONFIGS)
            checks['model_file_hashes_in_record'] = all(
                {Path(x).name: hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in mp[c]} == store[c]['record']['model_files_sha256']
                for c in CONFIGS)
            checks['twin_and_s2_learners_differ'] = not np.array_equal(store['FB_TWIN']['arrays']['MODEL_SELECT']['occupation'],
                                                                       store['FB_SEX_S2_ID']['arrays']['MODEL_SELECT']['occupation'])
            checks['geometry_frozen_in_record'] = store['FB_SEX_S2']['record']['channel']['geometry']['mode'] == 'sex' and \
                store['FB_PCA_S2']['record']['channel']['geometry']['mode'] == 'pca'
            checks['temperatures_in_bounds'] = all(0.5 - 1e-9 <= t <= 3.0 + 1e-9 for c in CONFIGS for t in store[c]['record']['channel']['temperatures'])
            info['fit_time_diagnostics'] = {c: {'nll': {t: round(store[c]['record']['fit_time_diagnostics_MODEL_FIT_in_sample']['emitted'][t]['nll'], 4)
                                                        for t in TASKS},
                                                'lda_delta2': round(store[c]['record']['fit_time_diagnostics_MODEL_FIT_in_sample']['lda_delta2_one_draw'], 4),
                                                'temps': [round(x, 3) for x in store[c]['record']['channel']['temperatures']]} for c in CONFIGS}
            info['fit_cpu_s'] = round(res['cpu_s'], 1)
            # a second fit of the same version refuses (append-only key)
            try:
                fit_version(0, 'x', root=tmp, spec=tiny, log=lambda s: None); checks['second_fit_refused'] = False
            except PermissionError:
                checks['second_fit_refused'] = True
    finally:
        _load, _features, _save, _roles = saved_hooks
        RL._ROOT_OVERRIDE = None
    ok = all(bool(x) for x in checks.values())
    print(json.dumps({'schema': SCHEMA + '-selftest', 'ok': ok, 'checks': {k: bool(x) for k, x in checks.items()}, 'info': info,
                      'invented_data_only': True, 'cpu_s': round(time.process_time() - t0, 1)}, indent=1, default=float))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='nrm.fb.learner')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    f = sub.add_parser('fit'); f.add_argument('--version', type=int, required=True); f.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return 0 if selftest() else 1
    print(json.dumps(fit_version(a.version, a.card), default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
