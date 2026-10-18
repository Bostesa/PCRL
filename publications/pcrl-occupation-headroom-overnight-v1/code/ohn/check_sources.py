"""Unit OHN_SOURCE_CHECK: label-free load check of the income anchors and frozen occupation sources on all six roles."""
import json

import numpy as np

from . import common as C
from . import devrun as D


def main():
    out = {}
    for s in C.SEEDS:
        for role in C.ROLES6:
            inc = D.anchor_income('H3', s, role); tw = D.anchor_income('TW', s, role)
            occ_j = D.source_occupation('JOINT', s, role); occ_t = D.source_occupation('T_TREE', s, role)
            o1 = D.source_occupation('O1', s, role)
            out[f'S{s}/{role}'] = {'n': int(len(inc)), 'inc_shape': list(inc.shape), 'occ_shape': list(occ_j.shape),
                                   'H3_vs_TW_income_maxdiff': float(np.abs(inc - tw).max()),
                                   'O1_vs_JOINT_maxdiff': float(np.abs(o1 - occ_j).max()),
                                   'T_TREE_vs_JOINT_maxdiff': float(np.abs(occ_t - occ_j).max())}
    print(json.dumps(out, indent=0))


if __name__ == '__main__':
    main()
