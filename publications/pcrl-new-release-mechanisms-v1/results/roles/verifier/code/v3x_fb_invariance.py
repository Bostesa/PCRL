"""V3X training-exposure experiment for Family B (torch interpreter): are the twin and the SEX-blind control learners invariant to
the MODEL_FIT SEX labels, and do the saved learners reproduce from the admitted training labels?

For version v and learners L in ('none', 'minf:S2', 'iso:S2') (twin and SEX-blind controls) plus 'sex:S2' (positive control: must
DEPEND on SEX), the family's own pure fitting function nrm.fb.learner.fit_learner (it returns networks; it writes nothing) is run
twice: with the true MODEL_FIT SEX and with SEX permuted across households (household-level permutation, fixed seed). Checks:
  - SEX-blind learners: state dicts bitwise identical under the permutation;
  - positive control 'sex:S2': state dicts differ under the permutation;
  - every learner fitted with the true SEX equals the saved runs/FB/MODELS/V<v>/LEARNER_<name>.pt bitwise (reproduction).
The channel files of the SEX-blind configurations must carry no SEX geometry. Labels: MODEL_FIT income, occupation, SEX through
nrm.common.load(operation 'verify') called from THIS script (logged). No release is saved; nothing is written outside records/.

    TORCH_PY .../v3x_fb_invariance.py --tag V1 --version 0
"""
from __future__ import annotations
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

sys.path.insert(0, str(V.ROOT))


def load_fit(purpose):
    import nrm.common as C
    F = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'verify', purpose)
    return F


def sd_bytes(nets, torch):
    buf = {}
    for t in ('income', 'occupation'):
        for k, x in nets[t].state_dict().items():
            buf[f'{t}.{k}'] = x.detach().cpu().numpy().tobytes()
    return buf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--version', type=int, default=0)
    a = ap.parse_args()
    v = a.version
    F = load_fit(f'verify: Family B V{v} SEX-invariance of the twin and SEX-blind control learners and reproduction of saved learners '
                 '(family fit function rerun in memory; no release saved)')
    import nrm.fb.learner as L
    torch = L._torch()
    X, unit = F['X'], np.asarray(F['unit'])
    sex = np.asarray(F['sex'], np.int64)
    ys = {t: np.asarray(F[t], np.int64) for t in ('income', 'occupation')}
    hh, inv = np.unique(unit, return_inverse=True)
    rng = np.random.default_rng(20261020)
    # household-level permutation: each household's SEX vector is replaced by the SEX vector of another household of the same size
    perm_sex = sex.copy()
    by_size = {}
    for g in range(len(hh)):
        by_size.setdefault(int((inv == g).sum()), []).append(g)
    idx_of = [np.flatnonzero(inv == g) for g in range(len(hh))]
    for size, gs in by_size.items():
        tgt = rng.permutation(gs)
        for g, h in zip(gs, tgt):
            perm_sex[idx_of[g]] = sex[idx_of[h]]
    res = {'version': v, 'sex_changed_people': int((perm_sex != sex).sum()), 'learners': {}}
    mdir = V.RUNS / 'FB' / 'MODELS' / f'V{v}'
    for name in ('none', 'minf:S2', 'iso:S2', 'sex:S2'):
        nets_a, log_a = L.fit_learner(X, ys, sex, unit, v, name)
        nets_b, log_b = L.fit_learner(X, ys, perm_sex, unit, v, name)
        A, B = sd_bytes(nets_a, torch), sd_bytes(nets_b, torch)
        same_perm = A.keys() == B.keys() and all(A[k] == B[k] for k in A)
        saved_path = mdir / f"LEARNER_{name.replace(':', '_')}.pt"
        sd = torch.load(saved_path, weights_only=True)
        S = {f'{t}.{k}': x.detach().cpu().numpy().tobytes() for t in ('income', 'occupation') for k, x in sd[t].items()}
        same_saved = S.keys() == A.keys() and all(S[k] == A[k] for k in A)
        res['learners'][name] = {'invariant_to_sex_permutation': bool(same_perm), 'reproduces_saved_learner': bool(same_saved),
                                 'selected_epochs': [log_a['selected_epochs'], log_b['selected_epochs']],
                                 'saved_file_sha256': V.file_sha(saved_path)}
    chans = {}
    for cfg in ('FB_TWIN', 'FB_MINF_S2', 'FB_MINF_S3', 'FB_ISO_S2'):
        ch = json.loads((mdir / f'CHANNEL_{cfg}.json').read_text())
        chans[cfg] = {'geometry': ch.get('geometry'), 'sex_free_geometry': set((ch.get('geometry') or {}).keys()) <= {'mode'}}
    res['channels_sex_free'] = chans
    blind_ok = all(res['learners'][n]['invariant_to_sex_permutation'] for n in ('none', 'minf:S2', 'iso:S2'))
    pos_ok = not res['learners']['sex:S2']['invariant_to_sex_permutation']
    repro_ok = all(x['reproduces_saved_learner'] for x in res['learners'].values())
    chan_ok = all(x['sex_free_geometry'] for x in chans.values())
    V.write_record(f'V3X_FB_INVARIANCE__{a.tag}_V{v}.json', res)
    chk = {'id': f'V3X.FB_SEX_INVARIANCE.V{v}__{a.tag}',
           'what': f'Family B V{v}: twin and SEX-blind control learners (none, minf:S2, iso:S2) bitwise invariant to a household-level '
                   'permutation of MODEL_FIT SEX; positive control sex:S2 changes; all four refits reproduce the saved learner files; '
                   'SEX-blind channel files carry no SEX geometry',
           'inputs': {**V.inputs_sha([mdir / f"LEARNER_{n}.pt" for n in ('none', 'minf_S2', 'iso_S2', 'sex_S2')]),
                      'code': V.inputs_sha([V.ROOT / 'nrm' / 'fb' / 'learner.py', V.ROOT / 'nrm' / 'fb' / 'channel.py'])},
           'result': {**res, 'blind_invariant': blind_ok, 'positive_control_detects_sex_use': pos_ok, 'reproduction': repro_ok,
                      'channel_files_sex_free': chan_ok},
           'numerical_status': 'PASS' if (repro_ok and pos_ok) else 'FAIL',
           'procedural_status': 'PASS' if (blind_ok and chan_ok) else 'FAIL'}
    print(json.dumps(V.append_checks([chk]), indent=1))
    print(json.dumps(chk['result'], indent=1, default=V._jd)[:3000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
