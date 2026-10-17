"""Synthetic check of the cross-fitted selection proxy (invented data): a planted dependence is read above chance, an
independent release reads near chance, and a finite-valued release is read by LOOKUP."""
import json

import numpy as np

from scm_bench.select_scm import proxy


def main():
    r = np.random.default_rng(3); n = 4000
    unit = np.repeat(np.arange(n // 2), 2); sex = (r.random(n) < 0.47).astype(int)
    def rel(dep):
        z = r.normal(size=n) + dep * sex
        p1 = 1 / (1 + np.exp(-z)); inc = np.column_stack((1 - p1, p1))
        occ = r.dirichlet(np.ones(5), n)
        return {'income': inc, 'occupation': occ}
    planted = proxy(rel(1.0), sex, unit, 0); null = proxy(rel(0.0), sex, unit, 0)
    cells = r.integers(0, 40, n); tab = r.dirichlet(np.ones(5), 40); tab1 = r.dirichlet(np.ones(2), 40)
    sex_c = (r.random(n) < np.linspace(0.2, 0.8, 40)[cells]).astype(int)
    finite = proxy({'income': tab1[cells], 'occupation': tab[cells]}, sex_c, unit, 0)
    out = {'planted_arm_proxy': planted['arm_proxy'], 'null_arm_proxy': null['arm_proxy'], 'finite_lookup': finite['LOOKUP']['mean']}
    out['all_ok'] = bool(planted['arm_proxy'] > 0.65 and 0.45 < null['arm_proxy'] < 0.56 and finite['LOOKUP']['mean'] > 0.6)
    print(json.dumps(out, indent=1)); assert out['all_ok']


if __name__ == '__main__':
    main()
