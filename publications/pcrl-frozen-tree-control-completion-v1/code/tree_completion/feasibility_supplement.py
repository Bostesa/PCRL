"""Class-conditional means of the per-group influence values (aggregate; for prevalence-shift scenarios).

Under a SEX-prevalence (label) shift with unchanged class-conditional distributions, a utility statistic's
expected value becomes pi' m_pos + (1 - pi') m_neg; with phi_g the per-group deviation from the INNER mean,
m_c - mean = E[phi | class c] (groups are SEX-pure; rows per group ~ 1). AUC influence rows have zero mean
within each class by construction.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def run(spec_path):
    from .evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'verification')
    with np.load(verify(cfg['phi']['path'], cfg['phi']['sha256'])) as z:
        phi = z['phi']; gs = z['group_sex']; names = [str(x) for x in z['names']]
    out = {'schema': 'TC1-feasibility-supplement-v1', 'queue_unit_id': active['unit_id'], 'names': names,
           'class_mean_phi_pos': phi[:, gs == 1].mean(1).tolist(), 'class_mean_phi_neg': phi[:, gs == 0].mean(1).tolist(),
           'groups_pos': int((gs == 1).sum()), 'groups_neg': int((gs == 0).sum()), 'definition': __doc__}
    with Path(cfg['output']).open('x') as f:
        json.dump(out, f)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
