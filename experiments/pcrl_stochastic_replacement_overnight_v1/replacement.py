"""Inputs, the two registered code families, and the Section-6 representation screen.

Contract tested here is **S1** (`RELEASE_CONTRACT.md`): `H_A`/`H_B` preserved exactly, the release is
`A = [H_A, Z]`, and `[H_A, Z_J]` is a *comparator*. `Z_J` is computed internally to build the second
code family but is **never released** in an S1 candidate, and a `J`-only predictor is never placed in
an S1 candidate's attack or utility slate.

Registered before any outcome:

* Code families. `pca32` = standardized A-side PCA32 inference inputs, exactly the precursor's
  `x_all`. `zj` = the frozen J channel `Z_J` computed internally (16 coordinates).
* Codebooks. Label-free k-means, `k = 64`, fitted on the declared representation-fitting rows.
  Exactly one registered fallback per family at `k = 256`. Seeds, `n_init`, and preprocessing frozen
  below.
* Probe slate. Regularized logistic over a fixed `C` grid plus a sufficiently trained MLP over a
  fixed weight-decay grid; best-validation checkpoint kept by the fitter. **Identical opportunities
  for every view**, references included.
* Rows. **All** eligible declared downstream-fitting households, not the precursor's 2048-row cap,
  which handicapped wide one-hot inputs.

This is a practical representation screen. It is not an estimate of Bayes capacity, and a J-derived
code need not *add* utility to J -- it needs to *preserve* enough utility to be a plausible
replacement.
"""
from __future__ import annotations

import time

import numpy as np

from experiments.pcrl_direct_adversarial_v1 import inputs as dax

# ------------------------------------------------------------------ frozen registration constants
FAMILIES = ('pca32', 'zj')
K_PRIMARY = 64
K_FALLBACK = 256
CODE_SALT = 20269021                      # frozen before outcomes
KMEANS_N_INIT = 10
RESIDENCE = 'same_residence'

# Identical opportunity for every view, references included.
LOGISTIC_C = (0.03, 0.3, 3.0)
MLP_WEIGHT_DECAY = (0.0, 1e-4)
MLP_EPOCHS = 200
MLP_HIDDEN = (64, 32)
MLP_BATCH = 256
MLP_LR = 1e-3
MLP_VALIDATION_INTERVAL = 5

SCREEN_ALLOWANCE = 0.001                  # nats, point-estimate scheduling rule vs same-host J
SCREEN_MIN_ANCHORS_GAIN_OVER_H = 2


def probe_schedule() -> list:
    """The frozen slate, as an explicit list so it can be hashed into the protocol."""
    out = [{'family': 'logistic', 'C': c} for c in LOGISTIC_C]
    out += [{'family': 'mlp', 'weight_decay': wd} for wd in MLP_WEIGHT_DECAY]
    return out


# ------------------------------------------------------------------ inputs
def seed_inputs(seed: int) -> dict:
    """Frozen state, wires, both code-family input spaces, labels, folds, weights, serials."""
    from experiments.pcrl_utility_extension_v1 import extension as ext
    from experiments.run_acs_residual_spectral import load_labels
    from experiments.pcrl_nonlinear_rank_v1.inputs import HIST_ROOT

    registry = dax.Registry.new()
    state, j, portability = ext.portable_state(seed, registry)
    jm = j['model']
    if not (np.array_equal(jm['input_mean'], state['a0']['input_mean'])
            and np.array_equal(jm['input_scale'], state['a0']['input_scale'])):
        raise AssertionError('J standardiser differs from A0 standardiser')

    ha = {p: np.asarray(state['anchors'][f'{p}/A'], np.float64) for p in dax.POOLS}
    hb = {p: np.asarray(state['anchors'][f'{p}/B'], np.float64) for p in dax.POOLS}
    zj = {p: np.asarray(j['channel'][p], np.float64) for p in dax.POOLS}
    pca = {p: dax.standardize(state['pca'][p], jm['input_mean'], jm['input_scale'])
           for p in dax.POOLS}

    for p in dax.POOLS:
        if ha[p].shape[1] != dax.H_A_WIDTH or hb[p].shape[1] != dax.H_B_WIDTH:
            raise AssertionError(f'anchor width changed at {p}')
        if zj[p].shape[1] != dax.A0_WIDTH:
            raise AssertionError(f'J channel width changed at {p}')

    frame, pools, labels, weights = load_labels(HIST_ROOT, seed)
    serials = {p: np.asarray(frame.iloc[pools[p]]['SERIALNO']) for p in pools}
    return {'seed': seed, 'ha': ha, 'hb': hb, 'zj': zj,
            'inputs': {'pca32': pca, 'zj': zj},
            'labels': labels, 'weights': weights, 'serials': serials, 'pools': pools,
            'portability': portability}


# ------------------------------------------------------------------ codebooks
def fit_code(x_fit: np.ndarray, k: int, seed: int, family: str):
    """Label-free k-means codebook. Exposes `assign` only."""
    from sklearn.cluster import KMeans
    rs = CODE_SALT + 1000 * FAMILIES.index(family) + seed
    km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=rs).fit(x_fit)

    class Code:
        n = k
        centers = km.cluster_centers_
        inertia = float(km.inertia_)
        random_state = rs

        @staticmethod
        def assign(x):
            return km.predict(np.asarray(x, np.float64)).astype(np.int64)

    return Code


def onehot(t, n) -> np.ndarray:
    out = np.zeros((len(t), n), dtype=np.float64)
    out[np.arange(len(t)), t] = 1.0
    return out


# ------------------------------------------------------------------ probes
def fit_view(x_fit, y_fit, x_val, y_val, seed, weights_val=None) -> dict:
    """Fit the whole frozen slate on one view and select by validation log loss.

    Every view gets the identical schedule, so references are never handicapped relative to
    candidates or the reverse.
    """
    from experiments.acs_transfer_heads import fit_candidates, metrics
    best, per_candidate = None, {}
    for spec in probe_schedule():
        if spec['family'] == 'logistic':
            budget = {'logistic': {'C': spec['C'], 'max_iter': 2000}}
            tag = f"logistic_C{spec['C']}"
        else:
            budget = {'mlp': {'hidden': list(MLP_HIDDEN), 'epochs': MLP_EPOCHS,
                              'batch_size': MLP_BATCH, 'lr': MLP_LR,
                              'weight_decay': spec['weight_decay'],
                              'validation_interval': MLP_VALIDATION_INTERVAL}}
            tag = f"mlp_wd{spec['weight_decay']}"
        fitted = fit_candidates(x_fit, y_fit, x_val, y_val, 2, seed,
                                families=(spec['family'],), budget=budget)
        cand = fitted['candidates'][spec['family']]
        ll = float(cand.metadata['validation_scores']['log_loss'])
        per_candidate[tag] = ll
        if best is None or (ll, tag) < (best['log_loss'], best['tag']):
            best = {'tag': tag, 'log_loss': ll, 'candidate': cand}

    probs = np.asarray(best['candidate'].predict_proba(x_val), np.float64)
    y = np.asarray(y_val, np.int64)
    out = {'selected': best['tag'], 'log_loss': best['log_loss'], 'per_candidate': per_candidate,
           'per_row_loss': -np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12)),
           'input_dim': int(np.asarray(x_fit).shape[1]), 'fit_rows': int(len(y_fit)),
           'validation_rows': int(len(y_val))}
    if weights_val is not None:
        out['log_loss_person_weighted'] = float(
            metrics(y_val, probs, 2, weights=weights_val)['log_loss'])
    return out


def ancestor_inclusive(candidate: dict, ancestor: dict) -> dict:
    """Best of a view and its accessible ancestor, per weighting, on the same rows.

    Legitimate here because `H` is genuinely contained in `[H_A, Z]` and in `[H_A, Z_J]`, so the
    ancestor predictor really is available to a recipient. Unlike the precursor's J case this is not
    degenerate: the candidate can and does win.
    """
    pick_unw = candidate if candidate['log_loss'] <= ancestor['log_loss'] else ancestor
    out = {'log_loss': float(pick_unw['log_loss']), 'source_unweighted': pick_unw['selected'],
           'per_row_loss': pick_unw['per_row_loss'],
           'candidate_alone': float(candidate['log_loss']),
           'ancestor_alone': float(ancestor['log_loss'])}
    if 'log_loss_person_weighted' in candidate and 'log_loss_person_weighted' in ancestor:
        cw, aw = candidate['log_loss_person_weighted'], ancestor['log_loss_person_weighted']
        pick_w = candidate if cw <= aw else ancestor
        out['log_loss_person_weighted'] = float(min(cw, aw))
        out['source_person_weighted'] = pick_w['selected']
        out['per_row_loss_person_weighted'] = pick_w['per_row_loss']
    return out


# ------------------------------------------------------------------ the screen
def screen_seed(inp: dict, family: str, k: int) -> dict:
    """One anchor of the Section-6 screen: H, H+J, H+T, H+raw, all ancestor-inclusive."""
    tick = time.perf_counter()
    seed = inp['seed']
    labels, weights = inp['labels'], inp['weights']
    x = inp['inputs'][family]

    valid_fit = labels['downstream_fit'][RESIDENCE] >= 0
    valid_val = labels['downstream_validation'][RESIDENCE] >= 0
    y_fit = labels['downstream_fit'][RESIDENCE][valid_fit]
    y_val = labels['downstream_validation'][RESIDENCE][valid_val]
    w_val = (np.asarray(weights['downstream_validation'])[valid_val] if weights else None)

    code = fit_code(x['representation_fit'], k, seed, family)
    t_fit = code.assign(x['downstream_fit'])[valid_fit]
    t_val = code.assign(x['downstream_validation'])[valid_val]

    ha_f, ha_v = inp['ha']['downstream_fit'][valid_fit], inp['ha']['downstream_validation'][valid_val]
    zj_f, zj_v = inp['zj']['downstream_fit'][valid_fit], inp['zj']['downstream_validation'][valid_val]
    xr_f, xr_v = x['downstream_fit'][valid_fit], x['downstream_validation'][valid_val]

    pseed = 1250000 + 100 * seed
    views = {
        'H': (ha_f, ha_v),
        'H_plus_J': (np.column_stack([ha_f, zj_f]), np.column_stack([ha_v, zj_v])),
        'H_plus_T': (np.column_stack([ha_f, onehot(t_fit, k)]),
                     np.column_stack([ha_v, onehot(t_val, k)])),
        'H_plus_raw': (np.column_stack([ha_f, xr_f]), np.column_stack([ha_v, xr_v])),
    }
    fitted = {name: fit_view(f, y_fit, v, y_val, pseed, w_val) for name, (f, v) in views.items()}

    h = fitted['H']
    incl = {name: ancestor_inclusive(fitted[name], h)
            for name in ('H_plus_J', 'H_plus_T', 'H_plus_raw')}
    incl['H'] = {'log_loss': h['log_loss'], 'per_row_loss': h['per_row_loss'],
                 'candidate_alone': h['log_loss'], 'ancestor_alone': h['log_loss'],
                 'source_unweighted': h['selected']}
    if 'log_loss_person_weighted' in h:
        incl['H']['log_loss_person_weighted'] = h['log_loss_person_weighted']
        incl['H']['per_row_loss_person_weighted'] = h['per_row_loss']
        incl['H']['source_person_weighted'] = h['selected']

    j_ll, t_ll = incl['H_plus_J']['log_loss'], incl['H_plus_T']['log_loss']
    rec = {'seed': seed, 'family': family, 'k': k,
           'codebook': {'inertia': code.inertia, 'random_state': code.random_state,
                        'labels_used': 'none', 'n_states': k},
           'occupancy': np.bincount(t_val, minlength=k).tolist(),
           'raw': {n: {kk: vv for kk, vv in f.items() if kk != 'per_row_loss'}
                   for n, f in fitted.items()},
           'inclusive_log_loss': {n: incl[n]['log_loss'] for n in incl},
           'deficit_T_vs_J_unweighted': float(t_ll - j_ll),
           'gain_T_over_H_unweighted': float(incl['H']['log_loss'] - t_ll),
           'rows': {'fit': int(len(y_fit)), 'validation': int(len(y_val))},
           'seconds': time.perf_counter() - tick}
    if w_val is not None:
        jw, tw = incl['H_plus_J']['log_loss_person_weighted'], incl['H_plus_T']['log_loss_person_weighted']
        rec['inclusive_log_loss_person_weighted'] = {
            n: incl[n]['log_loss_person_weighted'] for n in incl}
        rec['deficit_T_vs_J_person_weighted'] = float(tw - jw)
        rec['gain_T_over_H_person_weighted'] = float(
            incl['H']['log_loss_person_weighted'] - tw)
    rec['_paired'] = {'serial': inp['serials']['downstream_validation'][valid_val],
                      'weights': w_val,
                      'per_row': {n: incl[n]['per_row_loss'] for n in incl}}
    return rec


def screen_family(seeds, family: str, k: int, cache: dict) -> dict:
    """Apply the registered screen rule across all three anchors. Never cherry-picks an anchor."""
    per = [screen_seed(cache[s], family, k) for s in seeds]
    def mean(key):
        return float(np.mean([p[key] for p in per]))
    deficit_unw, deficit_pw = mean('deficit_T_vs_J_unweighted'), mean('deficit_T_vs_J_person_weighted')
    gains_unw = [p['gain_T_over_H_unweighted'] for p in per]
    gains_pw = [p['gain_T_over_H_person_weighted'] for p in per]
    anchors_with_gain = sum(1 for a, b in zip(gains_unw, gains_pw) if a > 0 and b > 0)
    passed = (deficit_unw <= SCREEN_ALLOWANCE and deficit_pw <= SCREEN_ALLOWANCE
              and anchors_with_gain >= SCREEN_MIN_ANCHORS_GAIN_OVER_H)
    return {'family': family, 'k': k, 'pass': bool(passed),
            'mean_deficit_T_vs_J': {'unweighted': deficit_unw, 'person_weighted': deficit_pw},
            'allowance': SCREEN_ALLOWANCE,
            'gain_T_over_H_by_anchor': {'unweighted': gains_unw, 'person_weighted': gains_pw},
            'anchors_with_gain_both_weightings': anchors_with_gain,
            'required_anchors': SCREEN_MIN_ANCHORS_GAIN_OVER_H,
            'mean_inclusive_log_loss': {
                n: float(np.mean([p['inclusive_log_loss'][n] for p in per]))
                for n in per[0]['inclusive_log_loss']},
            'mean_inclusive_log_loss_person_weighted': {
                n: float(np.mean([p['inclusive_log_loss_person_weighted'][n] for p in per]))
                for n in per[0]['inclusive_log_loss_person_weighted']},
            'per_anchor': [{kk: vv for kk, vv in p.items() if kk != '_paired'} for p in per],
            'rule': ('point-estimate scheduling rule, NOT a confidence non-inferiority claim: '
                     'inclusion-respecting H+T no worse than same-host J by more than .001 nats in '
                     'both weighted means, and a positive gain over H in at least two anchors under '
                     'both weightings'),
            '_per_anchor_paired': [p['_paired'] for p in per]}
