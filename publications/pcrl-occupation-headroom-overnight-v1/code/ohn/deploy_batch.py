"""Build and check the deployment packages of the nomination candidates in one pass (label-free).

    python3 -m ohn.deploy_batch --candidates NAME [NAME ...] --report results/<campaign>/deploy/DEPLOYMENT_CHECKS_<tag>.json

NAME = '<FAMILY>/<CONFIG>@<ANCHOR>' (anchor H3, TW or COMPLETE; 'SOURCE/JOINT@H3' is the unmodified source successor). Each name is
built for both seeds as runs/DEPLOY/<FAMILY>__<CONFIG>__<ANCHOR>_S<s> (write-once; an existing package is reused, and ohn.deploy
re-hashes every dependency before running it), then every package is checked by ohn.deploy.check: bitwise reproduction of the
evaluated DEV_REPORT release through the wrapper API, the CSV wrapper and the underlying function, and every refusal. The report is
what ohn.nominate reads as the runnable-package evidence.
"""
from __future__ import annotations
import argparse
import json

from . import common as C
from . import deploy as DP


def package_dir(name, seed):
    fam_cfg, anchor = name.split('@')
    fam, cfg = fam_cfg.split('/')
    return C.RUNS / 'DEPLOY' / f'{fam}__{cfg}__{anchor}_S{seed}', anchor, f'{fam}:{cfg}'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', nargs='+', required=True); ap.add_argument('--report', required=True)
    a = ap.parse_args(argv)
    pkgs, built = [], []
    for name in a.candidates:
        for s in C.SEEDS:
            d, anchor, occ = package_dir(name, s)
            if not (d / 'PACKAGE.json').is_file():
                DP.build(anchor, occ, s, d); built.append(d.name)
            pkgs.append(str(d))
    rec = DP.check(pkgs, a.report)
    print(json.dumps({'built': built, 'n_packages': len(pkgs), 'all_bitwise': rec['all_bitwise'], 'all_refused': rec['all_refused']}))
    if not (rec['all_bitwise'] and rec['all_refused']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
