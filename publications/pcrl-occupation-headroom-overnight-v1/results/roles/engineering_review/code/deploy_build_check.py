"""Engineering review: build deployment packages and check them in ONE process (label-free), so that no module edit can fall
between the build hash and the check; the coordinator's own `ohn.deploy.build` / `ohn.deploy.check` code is used unchanged, with
check()'s temporary raw-input CSV kept under .private/ and removed afterwards. Optionally audits runtime file access per package
(deploy_review.audit: files opened outside the package manifest, label-bearing files, subprocesses).

    <python> results/pcrl_occupation_headroom_overnight_v1/roles/engineering_review/code/deploy_build_check.py \
        --tag TAG --specs ANCHOR:FAMILY:CONFIG:SEED [...] --report OUT.json [--audit]

ANCHOR is H3, TW or COMPLETE (COMPLETE: the module's predict returns both outputs, e.g. COMPLETE:INC:INC_H3_PO:0).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from ohn import common as C  # noqa: E402
from ohn import deploy as DP  # noqa: E402

BASE = C.RUNS / 'DEPLOY_REVIEW'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--specs', nargs='+', required=True)
    ap.add_argument('--report', required=True); ap.add_argument('--audit', action='store_true')
    ap.add_argument('--deploy-module', help='a pinned or proposed ohn/deploy.py version to use instead of ohn.deploy')
    a = ap.parse_args(argv)
    global DP
    import deploy_review as DR
    if a.deploy_module:
        DP = DR.use_deploy_module(a.deploy_module)
    t0 = time.process_time()
    tmp = BASE / f'tmp_{a.tag}'
    tmp.mkdir(parents=True, exist_ok=True, mode=0o700)
    tempfile.tempdir = str(tmp)
    dirs, built = [], {}
    for spec in a.specs:
        anchor, fam, cfg, seed = spec.split(':')
        out = BASE / f'{a.tag}_{anchor}_{fam}_{cfg}_S{seed}'
        meta = DP.build(anchor, f'{fam}:{cfg}', int(seed), out)
        dirs.append(str(out))
        built[out.name] = {'spec': spec, 'n_dependencies': len(meta['depends_sha256']),
                           'dependency_keys': sorted(meta['depends_sha256'])}
    rec = DP.check(dirs, a.report)
    res = {'tag': a.tag, 'interpreter': sys.executable.split('/')[-1], 'python': sys.version.split()[0],
           'numpy': __import__('numpy').__version__, 'sklearn': __import__('sklearn').__version__,
           'deploy_module_sha256': C.sha(a.deploy_module) if a.deploy_module else C.sha(C.ROOT / 'ohn' / 'deploy.py'),
           'deploy_module_source': 'proposed patch' if a.deploy_module else 'ohn/deploy.py (working tree = HEAD)',
           'all_bitwise': rec['all_bitwise'], 'all_refused': rec['all_refused'], 'built': built,
           'check_refusals': {k: v['refusals'] for k, v in rec['packages'].items()},
           'check_extra': {k: {x: v.get(x) for x in ('code_closure_unchanged_since_build', 'environment_at_build')} for k, v in rec['packages'].items()},
           'check_environment': rec.get('environment')}
    if a.audit:
        res['audit'] = {}
        for d in dirs:
            au = DR.audit(d)
            res['audit'][Path(d).name] = {k: au.get(k) for k in ('unlisted_runtime_data_files', 'label_bearing_files_opened', 'subprocesses',
                                                                   'sockets', 'repo_code_modules_not_in_manifest', 'third_party_libraries_loaded',
                                                                   'cpu_s_release_all_dev_rows', 'error')}
    shutil.rmtree(tmp, ignore_errors=True)
    res['cpu_s'] = round(time.process_time() - t0, 2)
    Path(a.report).with_name(Path(a.report).stem + '_SUMMARY.json').write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'tag': a.tag, 'all_bitwise': res['all_bitwise'], 'all_refused': res['all_refused'], 'cpu_s': res['cpu_s']}))
    return 0 if (rec['all_bitwise'] and rec['all_refused']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
