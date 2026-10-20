"""Family B revision FB2 (card cards/FB_FSC_v2.md): the frozen FSC channel applied POST HOC (no training through the channel) to the
strongest admitted task-only base TWA_v = HC_STRONG S<v> income + A1_PO_16 S<v> occupation. numpy / scikit-learn only (python3).

Base (frozen, never refitted here):
  income       ihs.fam_h.predict(v, 'HC_STRONG', X, names)['income']: the task-only strong-income twin of the income-headroom
               campaign. Per its manifest it read MODEL_FIT income (Step-1 HGB) and CALIBRATION income (stack, budget cross-fitting,
               Platt map). Its predict also reads stored T_TREE / LOCAL_TREE / JOINT_TREE parameters (income offset 0).
  occupation   ohn.fam_a.predict(v, 'A1_PO_16', X, names): A1_PO_16 HGB on the POBP-ordinal coding, fitted on all MODEL_FIT.
  The fit computes the base from features for all six roles and requires bitwise equality with the hash-checked stored releases
  (ohn.devrun.anchor_income('TW', v, role); ohn.devrun.load_occupation('A', 'A1_PO_16', v)) before anything is written.
Channel (nrm.fb.channel; the frozen law of card v1, not trained through):
  z            centred log-probabilities of the base.
  'sex'        d_i = (F_i + s^2 Sigma_w^-1)^-1 Sigma_w^-1 delta.
               delta and Sigma_w come from MODEL_FIT SEX (SEX-labelled exposure: a closed-form LDA-type critic), computed on the
               geometry logits: occupation = A1_PO_16's admitted 5-fold household cross-fitted MODEL_FIT__oof (CROSSFIT.npz,
               hash-checked); income = HC_STRONG in-sample (no admitted out-of-fold bank exists). Disclosed.
  'minf'       each person's minimum-Fisher eigenvector (SEX-blind).
  common       unit Fisher norm; cap 1.5; one shared N(0,1) scalar per person per instance; R = 8 instances.
  keys         BLAKE2b(root, "NRM-FB|<config>|V<v>") with root = runs/FB/MODELS/FB2/V<v>/RELEASE_ROOT.bin (one per (revision,
               version)); messages "NRM-FB|<role>|r<r>|<row_id>".
  temperatures per-task, fitted on CALIBRATION on the emitted draws (noisy channels only; matched between SEX and MINF). This is a
               second use of CALIBRATION income, after HC_STRONG's own stack and Platt map: disclosed.
Configurations (revision FB2, both versions):
  firm         FB2_BASE (twin: the frozen base emitted unchanged - no temperature, exact alias of TWJ income and W occupation),
               FB2_SEX_S3 (0.12), FB2_SEX_S5 (0.21), FB2_MINF_S3, FB2_MINF_S5.
  contingent   (fitted only when the card's conditions hold) FB2_MINF_S6 (0.26, spend-only bracketing level),
               FB2_SEX_S3_MEAN / FB2_SEX_S5_MEAN: the exact deterministic mean vector E[emitted] of the FITTED FB2_SEX_S<j> channel
               (its frozen geometry and temperatures, 48-node Gauss-Hermite; decisive phase; one strength; fit_mean2, which reads no
               label).
Revision FB2P (training perturbation; card v3): FB2P_SEX_S<j> / FB2P_MINF_S<j> re-estimate the channel's fitted parameters
  - the SEX geometry delta / Sigma_w from a household bootstrap resample of MODEL_FIT;
  - the CALIBRATION temperatures from a household bootstrap resample of CALIBRATION, the same resample for SEX and MINF;
  - a fresh write-once root key (runs/FB/MODELS/FB2P/V<v>);
  - fixed bootstrap seeds [7300, v, 0] (MODEL_FIT) and [7300, v, 1] (CALIBRATION), sizes recorded.
  The frozen base models (HC_STRONG, A1_PO_16) are NOT refitted (limitation).

Commands:
  python3 -m nrm.fb.posthoc selftest
      invented data only (fake hooks; temporary folder).
  python3 -m nrm.fb.posthoc fit --version V --card <card> [--configs C1,C2]
      real fit (through the queue). Default: the 5 firm FB2 configurations; FB2P configurations by --configs.
  python3 -m nrm.fb.posthoc fit-mean --version V --of FB2_SEX_S<j> --card <card>
      the exact mean vector of a fitted SEX channel (no label read).
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import channel as CH
from . import learner as L          # numpy helpers only (fit_temperatures, emitted, scaled, _invented_world); torch is not imported
from . import release as RL

FAMILY = 'FB'
REVISION = 'FB2'
TASKS = ('income', 'occupation')
SCHEMA = 'NRM-FB2-posthoc-v1'
STRENGTHS2 = {'S3': 0.12, 'S5': 0.21}
CONFIGS2 = {
    'FB2_BASE': {'mode': 'none', 's': 0.0, 'kind': 'twin', 'exact_base': True,
                 'description': 'FB2 twin: the task-only base TWA itself (HC_STRONG income + A1_PO_16 occupation, frozen), emitted '
                                'unchanged (identity channel, no temperature; exact alias of TWJ income and W occupation)'},
    **{f'FB2_SEX_{j}': {'mode': 'sex', 's': s, 'kind': 'candidate',
                        'description': f'post-hoc SEX-targeted coupled FSC channel on the frozen TWA base, s = {s}'}
       for j, s in STRENGTHS2.items()},
    **{f'FB2_MINF_{j}': {'mode': 'minf', 's': s, 'kind': 'control',
                         'description': f'G3 SEX-blind control path: post-hoc minimum-Fisher coupled channel on the frozen TWA base, '
                                        f's = {s} (NO_MATCHED_CONTROLS; not selectable)'} for j, s in STRENGTHS2.items()},
}
CONTINGENT2 = {
    'FB2_MINF_S6': {'mode': 'minf', 's': 0.26, 'kind': 'control',
                    'description': 'contingent spend-only bracketing level of the G3 MINF path, s = 0.26 (not selectable)'},
    **{f'FB2_SEX_{j}_MEAN': {'mode': 'sex', 's': s, 'mean_vector': True, 'mean_of': f'FB2_SEX_{j}', 'kind': 'ablation',
                             'description': f'decisive-phase deterministic construction: the exact mean vector E[emitted] of the fitted '
                                            f'FB2_SEX_{j} channel (its frozen geometry and temperatures)'}
       for j, s in STRENGTHS2.items()},
}
ALL_CONFIGS2 = {**CONFIGS2, **CONTINGENT2}
PERTURB2 = {
    **{f'FB2P_SEX_{j}': {'mode': 'sex', 's': s, 'kind': 'perturbation', 'perturbs': f'FB2_SEX_{j}',
                         'description': f'training perturbation of FB2_SEX_{j}: geometry from a household bootstrap of MODEL_FIT, '
                                        f'temperatures from a household bootstrap of CALIBRATION, fresh root key; frozen base'}
       for j, s in STRENGTHS2.items()},
    **{f'FB2P_MINF_{j}': {'mode': 'minf', 's': s, 'kind': 'perturbation', 'perturbs': f'FB2_MINF_{j}',
                          'description': f'equally perturbed control of FB2P_SEX_{j}: temperatures from the same household bootstrap of '
                                         f'CALIBRATION, fresh root key; frozen base (MINF has no SEX geometry)'}
       for j, s in STRENGTHS2.items()},
}
REVISION_OF = {**{c: 'FB2' for c in ALL_CONFIGS2}, **{c: 'FB2P' for c in PERTURB2}}
CONF = {**ALL_CONFIGS2, **PERTURB2}
BOOT_SEED = 7300
PERTURB_LIMITATION = ('the frozen base models (HC_STRONG income, A1_PO_16 occupation) are not refitted; only the channel\'s fitted '
                      'parameters (SEX geometry, CALIBRATION temperatures) and the release key are re-drawn')
SPEC2 = {'calibration': L.SPEC['calibration'], 'release_draws': 8, 'cap': CH.CAP, 'strengths': STRENGTHS2,
         'base': {'income': 'ihs.fam_h.predict(v, HC_STRONG)', 'occupation': 'ohn.fam_a.predict(v, A1_PO_16)'}}
CODE_MODULES_BASE = ('ihs.fam_h', 'ihs.bases', 'ihs.common', 'ohn.fam_a', 'ohn.common', 'ohn.devrun', 'acs_replication.trees')
LABEL = ('FB2 was designed after family-B screening on MODEL_SELECT (targeting signal, F3); it is evaluated on the same people; '
         'development evidence only')


# ------------------------------------------------------------------ hooks (the selftest replaces them)
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


def base_probs(v, X, names):
    """The frozen TWA base from admitted features: (income (n, 2), occupation (n, 5)) float64."""
    from ihs import fam_h
    from ohn import fam_a
    inc = np.asarray(fam_h.predict(int(v), 'HC_STRONG', X, names)['income'], dtype=np.float64)
    occ = np.asarray(fam_a.predict(int(v), 'A1_PO_16', X, names), dtype=np.float64)
    return inc, occ


def stored_base(v, role):
    """The same base read through ohn's hash-checked loaders (admitted stored releases)."""
    from ohn import devrun as D
    inc = np.asarray(D.anchor_income('TW', int(v), role), dtype=np.float64)
    occ = np.asarray(D.load_occupation('A', 'A1_PO_16', int(v), roles=(role,))[0][role], dtype=np.float64)
    return inc, occ


def oof_occupation(v):
    """A1_PO_16's admitted out-of-fold MODEL_FIT occupation bank (5-fold household cross-fitting; hash-checked by the loader)."""
    from ohn import devrun as D
    _, _, cf = D.load_occupation('A', 'A1_PO_16', int(v), roles=('MODEL_FIT',))
    if cf is None or 'MODEL_FIT__oof' not in cf:
        raise PermissionError('A1_PO_16 has no admitted out-of-fold MODEL_FIT bank')
    return np.asarray(cf['MODEL_FIT__oof'], dtype=np.float64)


def base_files(v):
    """Every file the base computation and its hash-checked loaders read (pinned through model_paths):
      - HC_STRONG state, release files (RELEASE.npz / RELEASE.json) and the Step-1 learner;
      - the admitted stage files read by ihs.fam_h.predict;
      - the A1_PO_16 run directory (OCC.npz / OCC.json, CROSSFIT.npz, models; mutable EVAL_ / VERIFY records excluded);
      - the two runtime governance admissions."""
    from ohn import common as OC, deploy as OD, devrun as D
    from .. import common as C
    files = []
    for d in (OC.MIRROR / 'runs' / 'H' / 'HC_STRONG' / f'S{int(v)}', OC.MIRROR / 'runs' / 'H' / 'STEP1' / f'S{int(v)}'):
        files += [p for p in sorted(d.iterdir()) if p.is_file() and not p.name.startswith(('EVAL_', 'VERIFY'))]
    for rel in (f'admitted/stage_a/S{int(v)}/TREE_TEACHER_income.json', f'admitted/stage_a/S{int(v)}/TREE_TEACHER_occupation.json',
                f'admitted/stage_b/S{int(v)}/JOINT_TREE__E20.npz', f'admitted/stage_b/S{int(v)}/LOCAL_TREE__E20.npz'):
        files.append(OC.MIRROR / rel)
    rd = D.occ_dir('A', 'A1_PO_16', int(v))
    files += [p for p in sorted(rd.rglob('*')) if p.is_file() and not p.name.startswith(('EVAL_', 'VERIFY'))]
    files += [C.ROOT / g for g in OD.GOVERNANCE]
    return [str(p) for p in files]


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def household_bootstrap(unit, seed):
    """Row indices of a household bootstrap resample (households drawn with replacement, all members of each drawn household)."""
    u = np.asarray(unit)
    hh, inv = np.unique(u, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    counts = np.bincount(inv, minlength=len(hh))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    draw = np.random.default_rng(list(seed)).integers(0, len(hh), len(hh))
    idx = np.concatenate([order[starts[h]:starts[h] + counts[h]] for h in draw])
    return idx, {'seed': [int(x) for x in seed], 'households': int(len(hh)), 'households_drawn': int(len(draw)),
                 'distinct_households_drawn': int(len(np.unique(draw))), 'people_in_resample': int(len(idx)),
                 'distinct_people_in_resample': int(len(np.unique(idx))), 'people_in_role': int(len(u))}


def _base_gate(v, roles):
    """The frozen base for every role from features, required bitwise equal to the hash-checked stored releases."""
    t0 = time.process_time()
    base = {r: base_probs(v, roles[r]['X'], roles[r]['feature_names']) for r in roles}
    check = {}
    for r in roles:
        si, so = stored_base(v, r)
        check[r] = {'income_bitwise': bool(np.array_equal(base[r][0], si)), 'occupation_bitwise': bool(np.array_equal(base[r][1], so)),
                    'income_max_abs': float(np.max(np.abs(base[r][0] - si))), 'occupation_max_abs': float(np.max(np.abs(base[r][1] - so)))}
    if not all(c['income_bitwise'] and c['occupation_bitwise'] for c in check.values()):
        raise AssertionError(f'the base from features differs from the hash-checked stored releases: {check}')
    return base, check, time.process_time() - t0


# ------------------------------------------------------------------ fit (FB2 configurations of one version)
def fit_version2(v, card, configs=None, root=None, spec=None, log=print):
    t_all = time.process_time()
    v = int(v)
    sp = spec or SPEC2
    configs = list(CONFIGS2) if configs is None else list(configs)
    unknown = [c for c in configs if c not in CONF]
    if unknown or not configs:
        raise ValueError(f'unknown or empty FB2 configuration list: {unknown or configs}')
    if any(CONF[c].get('mean_of') for c in configs):
        raise ValueError('MEAN configurations are fitted by fit_mean2 (frozen parameters of the fitted SEX channel; no label read)')
    revs = {REVISION_OF[c] for c in configs}
    if len(revs) != 1:
        raise ValueError(f'one revision per fit unit (got {sorted(revs)})')
    rev = revs.pop()
    existing = [c for c in configs if RL.channel_file(c, v, root, rev).exists()]
    if existing:
        raise PermissionError(f'{rev} V{v}: {existing} already fitted (append-only)')
    F = _load('MODEL_FIT', ('features', 'sex'), 'fit',
              f'NRM family B revision {rev} V{v}: SEX only, for the closed-form channel geometry (SEX-mean difference and within-SEX '
              'covariance of the frozen TWA base\'s linked centred logits); no task label (the base is frozen)', card)
    Cal = _load('CALIBRATION', ('features', 'income', 'occupation'), 'calibrate',
                f'NRM family B revision {rev} V{v}: per-task temperatures (calibration parameters only)', card)
    roles = {r: _features(r) for r in _roles()}
    if not (np.array_equal(F['row_id'], roles['MODEL_FIT']['row_id']) and np.array_equal(Cal['row_id'], roles['CALIBRATION']['row_id'])):
        raise AssertionError('labelled rows differ from the feature rows')
    sex = np.asarray(F['sex'], np.int64)
    ycal = {t: np.asarray(Cal[t], np.int64) for t in TASKS}
    base, check, base_cpu = _base_gate(v, roles)
    t0 = time.process_time()
    oof = oof_occupation(v)
    if oof.shape != base['MODEL_FIT'][1].shape:
        raise AssertionError('out-of-fold bank shape differs from MODEL_FIT')
    oof_agree = float(np.mean(oof.argmax(1) == base['MODEL_FIT'][1].argmax(1)))
    if oof_agree < 0.5:
        raise AssertionError(f'out-of-fold bank does not align with the MODEL_FIT rows (decision agreement {oof_agree:.3f})')
    base_cpu += time.process_time() - t0
    z = {r: CH.z_from_probs(*base[r]) for r in roles}
    z_geo = CH.z_from_probs(base['MODEL_FIT'][0], oof)          # income in-sample (no admitted OOF bank), occupation out-of-fold
    geo_source = {'income': 'HC_STRONG in-sample MODEL_FIT probabilities (no admitted out-of-fold bank exists)',
                  'occupation': 'A1_PO_16 CROSSFIT.npz MODEL_FIT__oof (admitted 5-fold household cross-fitting, hash-checked)',
                  'oof_vs_deployed_decision_agreement': oof_agree,
                  'fisher': 'per person from the deployed base probabilities (as at runtime)'}
    pert = None
    rid_cal = np.asarray(roles['CALIBRATION']['row_id'])
    z_geo_fit, sex_fit, z_cal, y_cal, r_cal = z_geo, sex, z['CALIBRATION'], ycal, rid_cal
    if rev == 'FB2P':                                            # training perturbation of the channel's fitted parameters
        i_fit, b_fit = household_bootstrap(F['unit'], (BOOT_SEED, v, 0))
        i_cal, b_cal = household_bootstrap(Cal['unit'], (BOOT_SEED, v, 1))
        z_geo_fit, sex_fit = z_geo[i_fit], sex[i_fit]
        z_cal, y_cal, r_cal = z['CALIBRATION'][i_cal], {t: ycal[t][i_cal] for t in TASKS}, rid_cal[i_cal]
        pert = {'MODEL_FIT_geometry_resample': b_fit, 'CALIBRATION_temperature_resample': b_cal, 'fresh_root_key': True,
                'limitation': PERTURB_LIMITATION}
    pdir = RL.model_dir(v, root, rev)
    pdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    key_path = RL.root_key_file(v, root, rev)
    if key_path.exists():
        root_key = key_path.read_bytes()                        # one root per (revision, version); contingent configs reuse it
        key_reused = True
    else:
        root_key = os.urandom(32)
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, root_key); os.close(fd)
        key_reused = False
    cal_key = CH.release_key(root_key, f'{rev}-CALIBRATION-DRAWS', v)
    names = list(map(str, roles['MODEL_FIT']['feature_names']))
    bfiles = base_files(v)
    pbase = {t: p for t, p in zip(TASKS, base['MODEL_FIT'])}
    saved = {}
    for cid in configs:
        cfg = CONF[cid]
        t0 = time.process_time()
        geom = CH.geometry(z_geo_fit, sex_fit, 'sex') if cfg['mode'] == 'sex' else {'mode': cfg['mode']}
        chan = {'mode': cfg['mode'], 's': cfg['s'], 'cap': CH.CAP, 'geom': geom}
        mv = bool(cfg.get('mean_vector'))
        exact = bool(cfg.get('exact_base'))
        if exact:                                                # the base itself: no temperature, no log / softmax round trip
            temps = (1.0, 1.0)
        else:
            temps = L.fit_temperatures(z_cal, y_cal, chan, cal_key, r_cal, mv, sp['calibration'])
        randomized = cfg['mode'] != 'none' and cfg['s'] > 0 and not mv
        key = CH.release_key(root_key, cid, v)
        reps = sp['release_draws'] if randomized else 1
        if exact:
            arrays = {r: {'income': base[r][0].copy(), 'occupation': base[r][1].copy()} for r in roles}
        else:
            arrays = {r: L.emitted(z[r], chan, temps, key, r, roles[r]['row_id'], reps, mv) for r in roles}
        zf = arrays['MODEL_FIT']
        kl = {}
        for t in TASKS:                                          # label-free: mean KL(base || emitted), base as the label law
            P = zf[t] if zf[t].ndim == 3 else zf[t][None]
            kl[t] = float(np.mean([np.mean(np.sum(pbase[t] * (np.log(np.maximum(pbase[t], 1e-300)) - np.log(np.maximum(P[r_], 1e-300))), 1))
                                   for r_ in range(len(P))]))
        d2 = CH.lda_delta2(CH.z_from_probs(zf['income'][0] if randomized else zf['income'],
                                           zf['occupation'][0] if randomized else zf['occupation']), sex)
        ch_rec = {'schema': SCHEMA + '-channel', 'revision': rev, 'perturbation': pert, 'config_id': cid, 'version': v, 'mode': cfg['mode'], 's': cfg['s'],
                  'cap': CH.CAP, 'mean_vector': mv, 'exact_base': exact, 'randomized': randomized, 'draws': reps,
                  'temperatures': [float(temps[0]), float(temps[1])], 'geometry': geom, 'geometry_source': geo_source if cfg['mode'] == 'sex' else None,
                  'feature_names': names, 'tag_prefix': 'NRM-FB', 'base': SPEC2['base'],
                  'key_derivation': 'BLAKE2b(key = root, message = "NRM-FB|<config>|V<v>")'}
        with open(RL.channel_file(cid, v, root, rev), 'x') as f:
            json.dump(ch_rec, f, indent=1, sort_keys=True)
        mfiles = [RL.channel_file(cid, v, root, rev)] + ([key_path] if randomized else [])
        record = {
            'mechanism_card': card, 'kind': cfg['kind'], 'description': cfg['description'], 'revision': rev, 'label': LABEL,
            'perturbation': pert, 'perturbs': cfg.get('perturbs'),
            'n_model_fits': 0, 'n_calibrator_fits': 0 if exact else 1, 'n_crossfit_fits': 0,
            'n_critic_fits': 1 if cfg['mode'] == 'sex' else 0,
            'critic_note': ('critic = the closed-form LDA-type SEX geometry (delta, Sigma_w) on MODEL_FIT SEX; occupation logits '
                            'out-of-fold (A1_PO_16 CROSSFIT), income logits in-sample (HC_STRONG has no admitted out-of-fold bank)')
                           if cfg['mode'] == 'sex' else 'no SEX read by this configuration\'s channel',
            'roles_read': ('MODEL_FIT: sex (fit; channel geometry only); CALIBRATION: income, occupation (calibrate; temperatures)'
                           + ('; FB2_BASE itself uses neither (exact base)' if exact else '')),
            'exposure': ('frozen base: HC_STRONG income (MODEL_FIT income for its Step-1 HGB; CALIBRATION income for its stack, budget '
                         'cross-fitting and Platt map) and A1_PO_16 occupation (all MODEL_FIT occupation). This configuration: '
                         + ('MODEL_FIT SEX reaches the channel geometry (SEX-labelled exposure, LDA-type critic); '
                            if cfg['mode'] == 'sex' else 'no SEX; ')
                         + ('no label of this campaign reaches the emitted vectors (exact base)' if exact else
                            'CALIBRATION task labels reach the two temperatures (second use of CALIBRATION income after HC_STRONG)')),
            'training_people': int(len(F['row_id'])), 'training_households': int(len(np.unique(F['unit']))),
            'training_note': 'the frozen base was trained by its source campaigns (A1_PO_16: all MODEL_FIT; HC_STRONG: MODEL_FIT + '
                             'CALIBRATION); this configuration fits only the channel geometry (MODEL_FIT) and two temperatures (CALIBRATION)',
            'fit_cpu_s': float(base_cpu / len(configs) + (time.process_time() - t0)),
            'randomized': randomized, 'draws': reps,
            'seed_handling': ('emitted draw r of person i = Box-Muller of keyed BLAKE2b(key = private per-(config, version) key derived '
                              f'from the private 32-byte {rev} root of this version, message = "NRM-FB|<role>|r<r>|<row_id>"); '
                              'independent of row order; R = 8 independent one-draw-per-person release instances') if randomized else 'deterministic',
            'noise_coupling': ({'sex': 'one scalar N(0,1) draw per person per repetition shared by the income and occupation recipients '
                                       'along the person\'s own rank-1 direction (maximal coupling)',
                                'minf': 'one scalar N(0,1) draw per person per repetition along the person\'s minimum-Fisher direction '
                                        '(same coupled rank-1 construction; the direction generically lies in one recipient\'s block)'}
                               [cfg['mode']]) if randomized else 'none',
            'coupling_claim': 'none (SEX and MINF share the coupled rank-1 construction; the comparison isolates targeting)',
            'channel': {'mode': cfg['mode'], 's': cfg['s'], 'cap': CH.CAP, 'mean_vector': mv, 'temperatures': list(temps), 'geometry': geom,
                        'trained_through': False, 'root_key_reused': key_reused},
            'base_check_vs_stored': check, 'base_files_sha256': {p: _sha(p) for p in bfiles},
            'model_files_sha256': {Path(m).name: _sha(m) for m in mfiles},
            'fit_time_diagnostics_MODEL_FIT_in_sample': {'kl_base_to_emitted': kl, 'lda_delta2_one_draw_deployed_logits': d2},
            'spec': dict(sp), 'schema': SCHEMA,
        }
        saved[cid] = _save(cid, v, arrays, record)
        log(json.dumps({'version': v, 'config': cid, 'temps': [round(x, 4) for x in temps], 'randomized': randomized,
                        'kl_base_to_emitted': {t: round(kl[t], 5) for t in TASKS}, 'lda_delta2': round(d2, 5)}))
    return {'version': v, 'saved': {k: str(p) for k, p in saved.items()}, 'cpu_s': time.process_time() - t_all}


def fit_mean2(v, of_config, card, root=None, log=print):
    """The exact mean vector of a FITTED FB2_SEX channel (its frozen geometry and temperatures; Gauss-Hermite): deterministic,
    kind 'ablation'. Reads no label (every parameter is the source channel's)."""
    t_all = time.process_time()
    v = int(v)
    cid = f'{of_config}_MEAN'
    if cid not in CONTINGENT2 or CONTINGENT2[cid].get('mean_of') != of_config:
        raise ValueError(f'no MEAN construction is declared for {of_config}')
    rev = REVISION_OF[cid]
    if RL.channel_file(cid, v, root, rev).exists():
        raise PermissionError(f'{cid} V{v} already fitted (append-only)')
    src_path = RL.channel_file(of_config, v, root, rev)
    src = json.loads(src_path.read_text())                       # FileNotFoundError when the source channel is not fitted
    if src['config_id'] != of_config or int(src['version']) != v or src.get('revision') != rev or src['mode'] != 'sex' \
            or not src['randomized']:
        raise PermissionError(f'{src_path.name} is not the fitted randomized SEX channel {of_config} V{v}')
    roles = {r: _features(r) for r in _roles()}
    base, check, base_cpu = _base_gate(v, roles)
    chan = {'mode': src['mode'], 's': src['s'], 'cap': src['cap'], 'geom': src['geometry']}
    temps = (float(src['temperatures'][0]), float(src['temperatures'][1]))
    arrays = {r: L.emitted(CH.z_from_probs(*base[r]), chan, temps, None, None, None, 1, mean_vector=True) for r in roles}
    ch_rec = {**{k: src[k] for k in ('mode', 's', 'cap', 'temperatures', 'geometry', 'geometry_source', 'feature_names', 'tag_prefix',
                                     'base')},
              'schema': SCHEMA + '-channel', 'revision': rev, 'perturbation': None, 'config_id': cid, 'version': v, 'mean_vector': True,
              'exact_base': False, 'randomized': False, 'draws': 1, 'mean_of': of_config, 'source_channel_sha256': _sha(src_path),
              'gauss_hermite_nodes': CH.GH_NODES}
    with open(RL.channel_file(cid, v, root, rev), 'x') as f:
        json.dump(ch_rec, f, indent=1, sort_keys=True)
    cfg = CONTINGENT2[cid]
    bfiles = base_files(v)
    record = {
        'mechanism_card': card, 'kind': cfg['kind'], 'description': cfg['description'], 'revision': rev, 'label': LABEL,
        'mean_of': of_config, 'source_channel_sha256': ch_rec['source_channel_sha256'],
        'n_model_fits': 0, 'n_calibrator_fits': 0, 'n_crossfit_fits': 0, 'n_critic_fits': 0,
        'critic_note': f'no new fit: the SEX geometry and temperatures are {of_config}\'s (frozen)',
        'roles_read': f'none (no label read by this unit; every parameter is frozen from {of_config})',
        'exposure': f'as {of_config}: MODEL_FIT SEX reached its geometry (LDA-type critic), CALIBRATION task labels its temperatures; '
                    'reused unchanged',
        'training_people': int(len(roles['MODEL_FIT']['row_id'])), 'training_households': int(len(np.unique(roles['MODEL_FIT']['unit']))),
        'training_note': f'parameters fitted by {of_config} (geometry: MODEL_FIT; temperatures: CALIBRATION); the frozen base by its sources',
        'fit_cpu_s': float(time.process_time() - t_all), 'randomized': False, 'draws': 1, 'seed_handling': 'deterministic',
        'noise_coupling': 'none (deterministic mean of the coupled channel)', 'coupling_claim': 'none',
        'channel': {'mode': 'sex', 's': src['s'], 'cap': src['cap'], 'mean_vector': True, 'temperatures': list(temps),
                    'geometry': src['geometry'], 'trained_through': False},
        'base_check_vs_stored': check, 'base_files_sha256': {p: _sha(p) for p in bfiles},
        'model_files_sha256': {RL.channel_file(cid, v, root, rev).name: _sha(RL.channel_file(cid, v, root, rev))},
        'schema': SCHEMA,
    }
    saved = _save(cid, v, arrays, record)
    log(json.dumps({'version': v, 'config': cid, 'mean_of': of_config, 'temps': list(temps), 'base_cpu_s': round(base_cpu, 1)}))
    return {'version': v, 'saved': {cid: str(saved)}, 'cpu_s': time.process_time() - t_all}


# ------------------------------------------------------------------ deployment (nrm/deploy.py contract, via nrm.fb.release)
def channel_record2(config_id, version):
    if config_id not in CONF or int(version) not in (0, 1):
        raise ValueError(f'unknown FB2 / FB2P configuration {config_id} V{version}')
    rev = REVISION_OF[config_id]
    rec = json.loads(RL.channel_file(config_id, version, None, rev).read_text())
    if rec['config_id'] != config_id or int(rec['version']) != int(version) or rec.get('revision') != rev:
        raise PermissionError('channel file does not belong to this configuration')
    return rec


def model_paths2(config_id, version):
    rec = channel_record2(config_id, version)
    rev = REVISION_OF[config_id]
    paths = [str(RL.channel_file(config_id, version, None, rev))]
    if rec['randomized']:
        paths.append(str(RL.root_key_file(version, None, rev)))
    return paths + base_files(version)


def predict2(config_id, version, X, feature_names, draw=0, keys=None):
    rec = channel_record2(config_id, version)
    X = RL._check_inputs(X, feature_names, rec['feature_names'])
    if not isinstance(draw, (int, np.integer)) or int(draw) < 0:
        raise ValueError('draw must be a non-negative integer')
    if rec['randomized']:
        if keys is None or 'role' not in keys or 'row_id' not in keys or len(keys['row_id']) != len(X):
            raise PermissionError('a randomized FB2 configuration needs out-of-band per-request draw keys (one row id per row)')
    else:
        if keys is not None:
            raise PermissionError('draw keys are refused by a deterministic FB2 configuration')
        if int(draw) != 0:
            raise ValueError('a deterministic FB2 configuration has only draw 0')
    inc, occ = base_probs(version, X, feature_names)
    if rec.get('exact_base'):                                    # FB2_BASE: the frozen base itself
        return {'income': inc, 'occupation': occ}
    z = CH.z_from_probs(inc, occ)
    chan = {'mode': rec['mode'], 's': rec['s'], 'cap': rec['cap'], 'geom': rec['geometry']}
    temps = tuple(rec['temperatures'])
    if rec['mean_vector']:
        return L.emitted(z, chan, temps, None, None, None, 1, mean_vector=True)
    if not rec['randomized']:
        return L.emitted(z, chan, temps, None, None, None, 1)
    key = CH.release_key(RL.root_key_file(version, None, REVISION_OF[config_id]).read_bytes(), config_id, version)
    zs = z / np.array([temps[0]] + [temps[1]] * 4)
    return CH.emit_one(zs, L.scaled(chan, temps), key, keys['role'], np.asarray(keys['row_id']), int(draw))


# ------------------------------------------------------------------ selftest (invented data only)
def selftest():
    global _NO_REAL_ACCESS, _load, _features, _save, _roles, base_probs, stored_base, base_files, oof_occupation
    t0 = time.process_time()
    rng = np.random.default_rng(20261020)
    W = L._invented_world(rng)
    d = W['MODEL_FIT']['X'].shape[1]
    Wi, Wo = rng.normal(size=(d, 2)) * 0.5, rng.normal(size=(d, 5)) * 0.5
    calls, store, checks, info = [], {}, {}, {}
    saved_hooks = (_load, _features, _save, _roles, base_probs, stored_base, base_files, oof_occupation)
    _NO_REAL_ACCESS = True
    with tempfile.TemporaryDirectory(prefix='nrm_fb2_refuse_') as tmp_refuse:
        try:
            fit_version2(0, 'results/none.md', root=tmp_refuse)
            checks['real_access_refused'] = False
        except PermissionError:
            checks['real_access_refused'] = not any(Path(tmp_refuse).iterdir())

    def fake_load(role, fields, operation, purpose, card):
        labels = sorted(set(fields) & {'income', 'occupation', 'sex'})
        calls.append({'role': role, 'operation': operation, 'labels': labels})
        if (role, operation) not in {('MODEL_FIT', 'fit'), ('CALIBRATION', 'calibrate')} or (role == 'CALIBRATION' and 'sex' in labels):
            raise PermissionError('selftest policy')
        return {k: W[role][k] for k in ('X', 'feature_names', 'row_id', 'unit', *labels)}

    def fake_features(role):
        return {k: W[role][k] for k in ('X', 'feature_names', 'row_id', 'unit')}

    def fake_save(cid, v, arrays, record):
        from .. import release as REL
        randomized = bool(record['randomized'])
        for r in W:
            for t in TASKS:
                a = REL._check_prob(arrays[r][t], CH.K[t], f'{r} {t}', randomized)
                if a.shape[-2] != len(W[r]['row_id']) or (randomized and a.shape[0] != record['draws']):
                    raise ValueError(f'{cid} {r} {t}: shape {a.shape}')
        missing = [k for k in REL.REQUIRED if k not in record]
        if missing or (randomized and not all(k in record for k in ('draws', 'seed_handling', 'noise_coupling'))):
            raise ValueError(f'{cid}: record misses {missing}')
        store[(cid, v)] = {'arrays': arrays, 'record': json.loads(json.dumps(record, default=float))}
        return f'(selftest) {cid}/V{v}'

    def fake_base(v, X, names):
        X = np.asarray(X, np.float64)
        return CH.softmax(X @ Wi + 0.1 * int(v)), CH.softmax(X @ Wo)

    def fake_stored(v, role):
        return fake_base(v, W[role]['X'], W[role]['feature_names'])

    def fake_oof(v):                                             # a smoother (held-out-like) occupation bank, same rows
        return CH.softmax(0.8 * (W['MODEL_FIT']['X'] @ Wo))

    tiny = {**SPEC2, 'calibration': {**SPEC2['calibration'], 'golden_iters': 8, 'sweeps': 1}}
    try:
        with tempfile.TemporaryDirectory(prefix='nrm_fb2_selftest_') as tmp:
            bdir = Path(tmp) / 'base_files'; bdir.mkdir()
            for nm in ('HC_STRONG_RELEASE.json', 'A1_OCC.npz', 'A1_OCC.json'):
                (bdir / nm).write_text(nm)
            mroot = Path(tmp) / 'models'
            _load, _features, _save, _roles = fake_load, fake_features, fake_save, (lambda: tuple(W))
            base_probs, stored_base, oof_occupation = fake_base, fake_stored, fake_oof
            base_files = (lambda v: [str(p) for p in sorted(bdir.iterdir())])
            res = fit_version2(0, 'results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v2.md', root=mroot, spec=tiny, log=lambda s: None)
            firm = [(c, 0) for c in CONFIGS2]
            checks['firm_configurations_saved'] = sorted(store) == sorted(firm) and len(CONFIGS2) == 5
            checks['revision_layout_resolves'] = RL.model_dir(0, None, 'FB2').name == 'V0' and RL.model_dir(0, None, 'FB2').parent.name == 'FB2' \
                if RL._ROOT_OVERRIDE is None else False
            checks['randomized_flags'] = all(store[(c, 0)]['record']['randomized'] == (c != 'FB2_BASE') for c in CONFIGS2)
            checks['randomized_stack_shape'] = store[('FB2_SEX_S5', 0)]['arrays']['MODEL_SELECT']['occupation'].shape == (8, 300, 5)
            checks['label_reads_policy'] = calls == [{'role': 'MODEL_FIT', 'operation': 'fit', 'labels': ['sex']},
                                                     {'role': 'CALIBRATION', 'operation': 'calibrate', 'labels': ['income', 'occupation']}]
            checks['base_check_recorded_bitwise'] = all(all(x['income_bitwise'] and x['occupation_bitwise']
                                                            for x in store[k]['record']['base_check_vs_stored'].values()) for k in firm)
            checks['sex_geometry_only_for_sex_and_oof'] = all(
                (store[(c, 0)]['record']['channel']['geometry']['mode'] == 'sex') == c.startswith('FB2_SEX') for c in CONFIGS2) and \
                all(store[(c, 0)]['record']['n_critic_fits'] == (1 if c.startswith('FB2_SEX') else 0) for c in CONFIGS2) and \
                all(store[(c, 0)]['record']['n_model_fits'] == 0 for c in CONFIGS2)
            g = store[('FB2_SEX_S3', 0)]['record']['channel']['geometry']
            g_oof = CH.geometry(CH.z_from_probs(fake_base(0, W['MODEL_FIT']['X'], None)[0], fake_oof(0)), W['MODEL_FIT']['sex'], 'sex')
            g_ins = CH.geometry(CH.z_from_probs(*fake_base(0, W['MODEL_FIT']['X'], None)), W['MODEL_FIT']['sex'], 'sex')
            checks['geometry_uses_oof_occupation'] = g['delta'] == g_oof['delta'] and g['delta'] != g_ins['delta']
            checks['revision_tag_and_label'] = all(store[k]['record']['revision'] == 'FB2' and store[k]['record']['label'] == LABEL for k in firm)
            checks['not_trained_through'] = all(store[k]['record']['channel']['trained_through'] is False for k in firm)
            pi, po = fake_base(0, W['MODEL_SELECT']['X'], None)
            checks['base_config_is_exact_base_alias'] = bool(np.array_equal(store[('FB2_BASE', 0)]['arrays']['MODEL_SELECT']['occupation'], po) and
                                                             np.array_equal(store[('FB2_BASE', 0)]['arrays']['MODEL_SELECT']['income'], pi)) and \
                store[('FB2_BASE', 0)]['record']['channel']['temperatures'] == [1.0, 1.0] and store[('FB2_BASE', 0)]['record']['n_calibrator_fits'] == 0
            # contingent configurations later: the version's root key is reused; a fitted configuration is refused
            fit_version2(0, 'results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v2.md', configs=['FB2_MINF_S6'],
                         root=mroot, spec=tiny, log=lambda s: None)
            n_calls = len(calls)
            fit_mean2(0, 'FB2_SEX_S5', 'results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v3.md', root=mroot, log=lambda s: None)
            checks['contingent_reuses_root_key'] = store[('FB2_MINF_S6', 0)]['record']['channel']['root_key_reused'] is True and \
                store[('FB2_SEX_S5_MEAN', 0)]['record']['randomized'] is False
            checks['mean_reads_no_label'] = len(calls) == n_calls and store[('FB2_SEX_S5_MEAN', 0)]['record']['n_calibrator_fits'] == 0
            src = store[('FB2_SEX_S5', 0)]['record']['channel']
            chs = {'mode': 'sex', 's': src['s'], 'cap': src['cap'], 'geom': src['geometry']}
            Wd0 = W['DEV_REPORT']
            zd = CH.z_from_probs(*fake_base(0, Wd0['X'], None))
            gh = L.emitted(zd, chs, tuple(src['temperatures']), None, None, None, 1, mean_vector=True)
            checks['mean_is_exact_mean_of_fitted_sex_channel'] = all(
                np.array_equal(gh[t], store[('FB2_SEX_S5_MEAN', 0)]['arrays']['DEV_REPORT'][t]) for t in TASKS)
            mc = CH.emit(zd / np.array([src['temperatures'][0]] + [src['temperatures'][1]] * 4), L.scaled(chs, tuple(src['temperatures'])),
                         CH.release_key((mroot / 'FB2' / 'V0' / 'RELEASE_ROOT.bin').read_bytes(), 'FB2_SEX_S5', 0), 'MC',
                         Wd0['row_id'], 4000)
            info['mean_vs_mc_max_abs'] = float(max(np.abs(mc[t].mean(0) - gh[t]).max() for t in TASKS))
            checks['mean_matches_mc_average_of_draws'] = info['mean_vs_mc_max_abs'] < 0.01
            try:
                fit_mean2(1, 'FB2_SEX_S3', 'x', root=mroot, log=lambda s: None); checks['mean_needs_fitted_source'] = False
            except FileNotFoundError:
                checks['mean_needs_fitted_source'] = True
            # training perturbation FB2P (SEX and its MINF control at one strength): bootstrap geometry and temperatures, fresh key
            fit_version2(0, 'results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v3.md', configs=['FB2P_SEX_S5', 'FB2P_MINF_S5'],
                         root=mroot, spec=tiny, log=lambda s: None)
            rp, rm = store[('FB2P_SEX_S5', 0)]['record'], store[('FB2P_MINF_S5', 0)]['record']
            checks['perturbation_fresh_root_key'] = (mroot / 'FB2P' / 'V0' / 'RELEASE_ROOT.bin').read_bytes() != \
                (mroot / 'FB2' / 'V0' / 'RELEASE_ROOT.bin').read_bytes() and rp['channel']['root_key_reused'] is False
            checks['perturbation_recorded_and_matched'] = rp['perturbation']['MODEL_FIT_geometry_resample']['seed'] == [BOOT_SEED, 0, 0] and \
                rp['perturbation']['CALIBRATION_temperature_resample'] == rm['perturbation']['CALIBRATION_temperature_resample'] and \
                rp['perturbation']['CALIBRATION_temperature_resample']['seed'] == [BOOT_SEED, 0, 1] and rp['kind'] == 'perturbation' and \
                rp['perturbs'] == 'FB2_SEX_S5' and rm['perturbs'] == 'FB2_MINF_S5' and rm['channel']['geometry'] == {'mode': 'minf'}
            checks['perturbation_changes_geometry'] = rp['channel']['geometry']['delta'] != store[('FB2_SEX_S5', 0)]['record']['channel']['geometry']['delta']
            bi, bmeta = household_bootstrap(W['MODEL_FIT']['unit'], (BOOT_SEED, 0, 0))
            checks['household_bootstrap_whole_households'] = bmeta['households_drawn'] == bmeta['households'] and \
                all(np.isin(W['MODEL_FIT']['unit'][bi], W['MODEL_FIT']['unit']).tolist()) and \
                np.array_equal(bi, household_bootstrap(W['MODEL_FIT']['unit'], (BOOT_SEED, 0, 0))[0])
            info['perturbation'] = rp['perturbation']
            try:
                fit_version2(0, 'x', configs=['FB2P_SEX_S5', 'FB2_SEX_S3'], root=mroot, spec=tiny, log=lambda s: None)
                checks['one_revision_per_unit'] = False
            except ValueError:
                checks['one_revision_per_unit'] = True
            try:
                fit_version2(0, 'x', configs=['FB2_SEX_S3'], root=mroot, spec=tiny, log=lambda s: None); checks['refit_refused'] = False
            except PermissionError:
                checks['refit_refused'] = True
            # deployment through nrm.fb.release (revision dispatch) reproduces draw 0 bitwise for every configuration and role
            RL._ROOT_OVERRIDE = mroot
            dep = {}
            for (cid, v) in store:
                rec = store[(cid, v)]['record']
                for r in W:
                    keys = {'role': r, 'row_id': W[r]['row_id']} if rec['randomized'] else None
                    out = RL.predict(cid, 0, W[r]['X'], W[r]['feature_names'], draw=0, keys=keys)
                    ref = store[(cid, v)]['arrays'][r]
                    dep[f'{cid}|{r}'] = all(np.array_equal(out[t], ref[t][0] if rec['randomized'] else ref[t]) for t in TASKS)
            checks['predict_reproduces_draw0_every_config_every_role'] = all(dep.values()) and len(dep) == 9 * 6
            Wd = W['DEV_REPORT']
            o5 = RL.predict('FB2_MINF_S5', 0, Wd['X'], Wd['feature_names'], draw=5, keys={'role': 'DEV_REPORT', 'row_id': Wd['row_id']})
            checks['predict_reproduces_instance_5'] = all(np.array_equal(o5[t], store[('FB2_MINF_S5', 0)]['arrays']['DEV_REPORT'][t][5])
                                                          for t in TASKS)
            sw = list(Wd['feature_names']); sw[0], sw[1] = sw[1], sw[0]
            nan = Wd['X'].copy(); nan[0, 0] = np.nan
            probes = {'randomized_without_keys': (lambda: RL.predict('FB2_SEX_S3', 0, Wd['X'], Wd['feature_names']), PermissionError),
                      'deterministic_with_keys': (lambda: RL.predict('FB2_BASE', 0, Wd['X'], Wd['feature_names'],
                                                                     keys={'role': 'DEV_REPORT', 'row_id': Wd['row_id']}), PermissionError),
                      'deterministic_draw_1': (lambda: RL.predict('FB2_SEX_S5_MEAN', 0, Wd['X'], Wd['feature_names'], draw=1), ValueError),
                      'short_keys': (lambda: RL.predict('FB2_SEX_S5', 0, Wd['X'], Wd['feature_names'],
                                                        keys={'role': 'DEV_REPORT', 'row_id': Wd['row_id'][:-1]}), PermissionError),
                      'reordered_names': (lambda: RL.predict('FB2_BASE', 0, Wd['X'], tuple(sw)), ValueError),
                      'non_finite': (lambda: RL.predict('FB2_BASE', 0, nan, Wd['feature_names']), ValueError),
                      'unknown_config': (lambda: RL.predict('FB2_SEX_S2', 0, Wd['X'], Wd['feature_names']), ValueError),
                      'unfitted_contingent': (lambda: RL.predict('FB2_SEX_S3_MEAN', 0, Wd['X'], Wd['feature_names']), FileNotFoundError)}
            pr = {}
            for k_, (f, exc) in probes.items():
                try:
                    f(); pr[k_] = False
                except exc:
                    pr[k_] = True
            info['predict_refusals'] = pr
            checks['predict_refusals'] = all(pr.values())
            mp = {c: RL.model_paths(c, 0) for (c, _) in store}
            checks['model_paths_pin_base_sources_and_key_only_if_randomized'] = all(
                all(Path(x).is_file() for x in mp[c]) and any(x.endswith('A1_OCC.npz') for x in mp[c]) and
                any(x.endswith('A1_OCC.json') for x in mp[c]) and any(x.endswith('HC_STRONG_RELEASE.json') for x in mp[c]) and
                (any(x.endswith('RELEASE_ROOT.bin') for x in mp[c]) == store[(c, 0)]['record']['randomized']) for c in mp)
            checks['code_modules_pin_base_sources'] = set(CODE_MODULES_BASE) <= set(RL.CODE_MODULES)
            checks['fb1_layout_unchanged'] = RL.revision_of('FB_SEX_S2') == 'FB1' and RL.revision_of('FB2_MINF_S6') == 'FB2' and \
                RL.revision_of('FB2P_MINF_S5') == 'FB2P'
            info['fit_time'] = {c: store[(c, 0)]['record']['fit_time_diagnostics_MODEL_FIT_in_sample'] for c in CONFIGS2}
            info['fit_cpu_s'] = round(res['cpu_s'], 2)
            RL._ROOT_OVERRIDE = None
            # a base that differs from the stored release is refused before anything is written
            stored_base = (lambda v, role: (fake_stored(v, role)[0], fake_stored(v, role)[1] * (1 + 1e-12)))
            n_before = len(store)
            try:
                fit_version2(1, 'x', root=Path(tmp) / 'm2', spec=tiny, log=lambda s: None); checks['base_mismatch_refused'] = False
            except AssertionError:
                checks['base_mismatch_refused'] = len(store) == n_before and not (Path(tmp) / 'm2').exists()
    finally:
        _load, _features, _save, _roles, base_probs, stored_base, base_files, oof_occupation = saved_hooks
        RL._ROOT_OVERRIDE = None
    ok = all(bool(x) for x in checks.values())
    print(json.dumps({'schema': SCHEMA + '-selftest', 'ok': ok, 'checks': {k: bool(x) for k, x in checks.items()}, 'info': info,
                      'invented_data_only': True, 'cpu_s': round(time.process_time() - t0, 2)}, indent=1, default=float))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='nrm.fb.posthoc')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    f = sub.add_parser('fit'); f.add_argument('--version', type=int, required=True); f.add_argument('--card', required=True)
    f.add_argument('--configs', default=None, help='comma-separated FB2 or FB2P configuration ids (default: the 5 firm FB2 ones)')
    m = sub.add_parser('fit-mean'); m.add_argument('--version', type=int, required=True); m.add_argument('--of', required=True)
    m.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return 0 if selftest() else 1
    if a.cmd == 'fit-mean':
        print(json.dumps(fit_mean2(a.version, a.of, a.card), default=str))
        return 0
    cfgs = None if a.configs is None else [c for c in a.configs.split(',') if c]
    print(json.dumps(fit_version2(a.version, a.card, configs=cfgs), default=str))
    return 0


if __name__ == '__main__':          # run in the canonical module object (nrm.fb.release dispatches to nrm.fb.posthoc, not __main__)
    sys.exit(importlib.import_module('nrm.fb.posthoc').main())
