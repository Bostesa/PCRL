"""Coordinator combinations: a release whose income output comes from one stored release and whose occupation output comes
from another (no new model fit; every role is copied bitwise from the two sources).

Used for (a) a successor that pairs the best income-side output with the best occupation-side output, (b) its fixed-occupation
control (the same income output beside the unmodified JOINT_TREE occupation output) and (c) its fixed-income control (the same
occupation output beside H_LJ's own income output). A combination is a new seed-specific release configuration and counts
toward the 36 (n_model_fits = 0). Sources may be campaign releases (family, config) or the frozen references 'H_LJ' (both
outputs recomputed and checked by ihs.bases: LOCAL_TREE income, JOINT_TREE occupation).

    python -m ihs.combine --seed s --config CMB_ID --income F:CONFIG|H_LJ --occupation F:CONFIG|H_LJ --note "..."
"""
from __future__ import annotations
import argparse
import json
import time

from . import bases as B
from . import common as C
from . import devrun as D

ROLES = C.ROLES6


def source(spec, seed):
    if spec == 'H_LJ':
        inc = B.base_outputs(seed, 'LOCAL_TREE', ROLES); occ = B.base_outputs(seed, 'JOINT_TREE', ROLES)
        return {r: {'income': inc[r]['prob']['income'], 'occupation': occ[r]['prob']['occupation']} for r in ROLES}, {'source': 'H_LJ (recomputed, checked)'}
    fam, cfg = spec.split(':')
    outs, rec = D.load_release(fam, cfg, seed)
    return outs, {'source': f'{fam}:{cfg}', 'release_sha256': rec['release_sha256'], 'n_model_fits': int(rec.get('n_model_fits') or 0)}


def run(seed, config_id, inc_spec, occ_spec, note):
    t0 = time.process_time()
    a, ra = source(inc_spec, seed); b, rb = source(occ_spec, seed)
    roles = [r for r in ROLES if r in a and r in b]
    outs = {r: {'income': a[r]['income'], 'occupation': b[r]['occupation']} for r in roles}
    rec = {'mechanism': f'combination: income output of {inc_spec}, occupation output of {occ_spec}', 'inputs_used': 'the six permitted raw fields (through the two sources)',
           'roles_read': 'features only (no label): the two source releases', 'labels_used_at_training': 'none in this step (the sources record theirs)',
           'n_model_fits': int(ra.get('n_model_fits', 0)) + int(rb.get('n_model_fits', 0)),
           'n_model_fits_note': 'no new fit in the combination step; the count carries the fits behind both source outputs',
           'fit_cpu_s': time.process_time() - t0, 'income_source': ra, 'occupation_source': rb, 'note': note}
    D.save_release('CMB', config_id, seed, outs, rec)
    print(json.dumps({'config': config_id, 'seed': seed, 'roles': roles}))


FAMILY_MODULE = {'P': 'ihs.fam_p', 'O': 'ihs.fam_o', 'H': 'ihs.fam_h'}


def _predict_spec(spec, seed, X, names, task):
    import importlib
    from acs_replication import trees
    if spec == 'H_LJ':
        arm = 'LOCAL_TREE' if task == 'income' else 'JOINT_TREE'
        return trees.probabilities(task, B.base_raw(seed, arm, X, names, 'E20', tasks=(task,))[task])
    fam, cfg = spec.split(':')
    return importlib.import_module(FAMILY_MODULE[fam]).predict(seed, cfg, X, names)[task]


def predict(seed, config_id, X, names):
    """Recompute a combination from features alone: each output through its source family's own predict."""
    rec = json.loads((D.release_dir('CMB', config_id, seed) / 'RELEASE.json').read_text())
    inc_spec = rec['income_source']['source'].split(' ')[0]
    occ_spec = rec['occupation_source']['source'].split(' ')[0]
    return {'income': _predict_spec(inc_spec, seed, X, names, 'income'), 'occupation': _predict_spec(occ_spec, seed, X, names, 'occupation')}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--config', required=True)
    ap.add_argument('--income', required=True); ap.add_argument('--occupation', required=True); ap.add_argument('--note', default='')
    a = ap.parse_args()
    run(a.seed, a.config, a.income, a.occupation, a.note)
