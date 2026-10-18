"""Unit OHN_ADMIT_V1: admit the income-headroom successor campaign's private artifacts read-only, by hash, into a mirror of its own
private layout (.private/pcrl_income_headroom_successor_v1/ in this worktree) so that its released code (ihs.*) runs unchanged
for the income anchors. No label array is opened here.

Accepted only when the SHA-256 equals a committed record of the source (research/pcrl-income-headroom-successor-v1 @ 2af3c9db2;
public package 89fd2f6e6):
- admitted/<path> (exports, frozen bank, stage A / B teachers and corrections, lineage releases, packages): the source
  SOURCE_ADMISSION.json `copies` (124 copies; exports additionally against the public extraction summary, the frozen bank
  against FROZEN_BANK_MANIFEST);
- runs/H/H3_STRONG_LJDIR_PB/S<s>/*: the source FINALIST_LOCK `package_files_sha256`; runs/H/STEP1/S<s>/*: the source H3 deployment
  PACKAGE.json `depends_sha256` (whose RELEASE / STATE entries equal the FINALIST_LOCK hashes);
- runs/H/HC_STRONG, runs/H/HC_TTREE_LJDIR_PB (matched twin and component control), runs/O/{O1_SEXBAL, OC_TCAL, OC_TBCAL,
  OC_BFIT} (the occupation recalibration family): RELEASE.npz against its RELEASE.json `release_sha256`; for the twin, also the
  committed SERVICE_BANK_MANIFEST `release_records`; twin and component control ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT
  arrays against the committed SERVICE_BANK_MANIFEST wire hashes;
- runs/FINAL/SERVICE_BANK_S<s>.npz: the source SERVICE_BANK_MANIFEST;
- runs/DEPLOY/<package>/PACKAGE.json: its RELEASE / STATE dependencies against the files admitted above.
Access logs are snapshot-copied and hashed only. The pinned source documents are hashed (research tip) and compared with the
public package's blobs (sanitised copies may differ only by the publication sanitiser).

Output: results/<campaign>/SOURCE_ADMISSION.json (write-once).
"""
from __future__ import annotations
import json
import shutil
import subprocess

import numpy as np

from . import common as C

H_CONFIGS = ('H3_STRONG_LJDIR_PB', 'HC_STRONG', 'HC_TTREE_LJDIR_PB')
O_CONFIGS = ('O1_SEXBAL', 'OC_TCAL', 'OC_TBCAL', 'OC_BFIT')
PACKAGES = tuple(f'{c}_S{s}' for c in H_CONFIGS for s in C.SEEDS)
PINNED_DOCS = ('START_HERE.md', 'RESEARCH_DECISION.md', 'CONFIRMATION_FEASIBILITY.md', 'CONFIRMATION_FEASIBILITY_RESULTS.json',
               'INDEPENDENT_VERIFICATION.json', 'PROTOCOL_DEVIATIONS.md', 'MECHANISM_CARDS.md', 'PROTOCOL.md',
               'SOURCE_ADMISSION.json', 'SERVICE_BANK_MANIFEST.json', 'READER_PLAN.json', 'FINALIST_LOCK.json',
               'ANALYSIS_LOCK.json', 'REPORTING_LOCK.json', 'NOMINATION.json', 'EXPERIMENT_LEDGER.csv', 'UTILITY_RESERVE_RULE.md',
               'roles/statistical_reviewer/RESERVE_RULE.json', 'roles/statistical_reviewer/NOMINATION_RULE.md',
               'roles/statistical_reviewer/CLAIMS_DRAFT.md', 'report/DEV_REPORT_RESULTS.json', 'report/PRIMARY_ENDPOINTS.csv',
               'HANDOFF_DECISION.json', 'DEPLOYMENT_CHECKS.json')


def _git(*a):
    r = subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=True)
    return r.returncode, r.stdout


def plan():
    """(destination relative to the mirror, source path, expected sha256 or None, basis)."""
    src = C.IHS_SRC_PRIV
    adm = json.loads((C.IHS_RESULTS / 'SOURCE_ADMISSION.json').read_text())['copies']
    fl = json.loads((C.IHS_RESULTS / 'FINALIST_LOCK.json').read_text())['package_files_sha256']
    sbm = json.loads((C.IHS_RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    out = []
    for rel, rec in sorted(adm.items()):
        out.append((f'admitted/{rel}', src / 'admitted' / rel, rec['sha256'], 'source SOURCE_ADMISSION.json'))
    h3pkg = {s: json.loads((src / 'runs' / 'DEPLOY' / f'H3_STRONG_LJDIR_PB_S{s}' / 'PACKAGE.json').read_text())['depends_sha256'] for s in C.SEEDS}
    for s in C.SEEDS:
        for name, h in sorted(fl[f'S{s}'].items()):
            out.append((f'runs/H/H3_STRONG_LJDIR_PB/S{s}/{name}', src / 'runs/H/H3_STRONG_LJDIR_PB' / f'S{s}' / name, h,
                        'source FINALIST_LOCK package_files_sha256'))
        for name in ('HGB.pkl', 'STEP1.json'):
            out.append((f'runs/H/STEP1/S{s}/{name}', src / 'runs/H/STEP1' / f'S{s}' / name, h3pkg[s][f'runs/H/STEP1/S{s}/{name}'],
                        'source H3 deployment PACKAGE.json depends_sha256'))
        for fam, cfgs in (('H', H_CONFIGS[1:]), ('O', O_CONFIGS)):
            for cfg in cfgs:
                d = src / 'runs' / fam / cfg / f'S{s}'
                if not d.is_dir():                       # e.g. OC_TBCAL was fitted on seed 0 only by the source campaign
                    continue
                for p in sorted(d.iterdir()):
                    if p.is_file():
                        out.append((f'runs/{fam}/{cfg}/S{s}/{p.name}', p, None, 'source run file (checked for internal consistency below)'))
        out.append((f'runs/FINAL/SERVICE_BANK_S{s}.npz', src / 'runs/FINAL' / f'SERVICE_BANK_S{s}.npz', sbm['bank_files_sha256'][f'S{s}'],
                    'source SERVICE_BANK_MANIFEST'))
    for pkg in PACKAGES:
        out.append((f'runs/DEPLOY/{pkg}/PACKAGE.json', src / 'runs/DEPLOY' / pkg / 'PACKAGE.json', None,
                    'source package record (dependencies checked below)'))
    return out


def run():
    rows, problems = {}, []
    for dst, srcp, expected, basis in plan():
        p = C.MIRROR / dst
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not srcp.is_file():
            problems.append(f'missing source file: {dst}'); continue
        if not p.exists():
            shutil.copyfile(srcp, p)
        h = C.sha(p)
        if expected is not None and h != expected:
            problems.append(f'hash differs: {dst}')
        rows[dst] = {'sha256': h, 'expected_sha256': expected, 'basis': basis, 'bytes': p.stat().st_size}
    sbm = json.loads((C.IHS_RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    checks = {}
    for s in C.SEEDS:
        for fam, cfgs in (('H', H_CONFIGS), ('O', O_CONFIGS)):
            for cfg in cfgs:
                if not (C.MIRROR / f'runs/{fam}/{cfg}/S{s}/RELEASE.json').is_file():
                    checks[f'{fam}/{cfg}/S{s} not present in the source (seed not fitted there)'] = True
                    continue
                rec = json.loads((C.MIRROR / f'runs/{fam}/{cfg}/S{s}/RELEASE.json').read_text())
                ok = rec['release_sha256'] == rows[f'runs/{fam}/{cfg}/S{s}/RELEASE.npz']['sha256']
                checks[f'{fam}/{cfg}/S{s} release record matches array'] = ok
                if not ok:
                    problems.append(f'release record does not match its array: {fam}/{cfg} S{s}')
        per = sbm['per_seed'][f'S{s}']
        for svc, cfg in (('SUCCESSOR', 'H3_STRONG_LJDIR_PB'), ('TW', 'HC_STRONG')):
            ok = per['release_records'][svc] == rows[f'runs/H/{cfg}/S{s}/RELEASE.npz']['sha256']
            checks[f'{svc} S{s} release equals SERVICE_BANK_MANIFEST release_records'] = ok
            if not ok:
                problems.append(f'{svc} S{s}: release differs from the committed bank record')
        with np.load(C.MIRROR / f'runs/H/HC_TTREE_LJDIR_PB/S{s}/RELEASE.npz') as z, \
                np.load(C.MIRROR / f'runs/H/HC_STRONG/S{s}/RELEASE.npz') as zt:
            for svc, zz in (('CC', z), ('TW', zt)):
                for t in C.TASKS:
                    for role in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'):
                        ok = C.wire_sha(zz[f'{role}__{t}']) == per['wire_sha256'][svc][t][role]
                        checks[f'{svc} S{s} {role} {t} wire equals manifest'] = ok
                        if not ok:
                            problems.append(f'{svc} S{s} {role} {t}: wire differs from the committed manifest')
    for pkg in PACKAGES:
        meta = json.loads((C.MIRROR / f'runs/DEPLOY/{pkg}/PACKAGE.json').read_text())
        for rel, h in meta['depends_sha256'].items():
            key = rel if rel.startswith('runs/') else f'admitted/{rel}'
            if key in rows:
                ok = rows[key]['sha256'] == h
            else:
                ok = False
                problems.append(f'{pkg}: dependency {rel} not admitted')
            checks[f'{pkg} dependency {rel}'] = ok
            if key in rows and not ok:
                problems.append(f'{pkg}: dependency {rel} hash differs')
    logs = {'ihs_access/DEV_REPORT_REREAD.jsonl': C.IHS_SRC_PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl',
            'ihs_access/MODEL_SELECT_USE.jsonl': C.IHS_SRC_PRIV / 'access' / 'MODEL_SELECT_USE.jsonl',
            'ihs_access/ATTACK_LABEL_READS.jsonl': C.IHS_SRC_PRIV / 'access' / 'ATTACK_LABEL_READS.jsonl',
            'lineage_access/ACCESS_LOG_snapshot.jsonl': C.SRC_ACCESS_LOG}
    snapshots = {}
    for dst, srcp in logs.items():
        p = C.PRIV / 'admitted_logs' / dst
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if srcp.is_file():
            shutil.copyfile(srcp, p)
            lines = [l for l in p.read_text().splitlines() if l.strip()]
            snapshots[dst] = {'sha256': C.sha(p), 'lines': len(lines), 'bytes': p.stat().st_size}
        else:
            snapshots[dst] = {'status': 'NOT_PRESENT'}
    lineage = [json.loads(l) for l in (C.PRIV / 'admitted_logs/lineage_access/ACCESS_LOG_snapshot.jsonl').read_text().splitlines() if l.strip()]
    reads = [e.get('event') for e in lineage if str(e.get('event', '')).startswith('DEV_REPORT_TRUTH')]
    pinned = {}
    for f in PINNED_DOCS:
        p = C.IHS_RESULTS / f
        rc, blob = _git('rev-parse', f'{C.IHS_PUBLICATION}:publications/pcrl-income-headroom-successor-v1/results/{f}')
        rc2, local_blob = _git('rev-parse', f'{C.IHS_TIP}:results/{C.IHS_CAMPAIGN}/{f}')
        pinned[f] = {'sha256_at_research_tip': C.sha(p) if p.is_file() else None, 'git_blob_research_tip': local_blob.strip() if rc2 == 0 else None,
                     'git_blob_public_package': blob.strip() if rc == 0 else None,
                     'identical_to_public_blob': (rc == 0 and rc2 == 0 and blob.strip() == local_blob.strip())}
    rec = {'schema': 'OHN1-source-admission-v1', 'campaign': C.CAMPAIGN, 'utc': C.now(),
           'source': {'branch': 'research/pcrl-income-headroom-successor-v1', 'research_tip': C.IHS_TIP,
                      'public_package_commit': C.IHS_PUBLICATION,
                      'public_package_url': f'https://github.com/Bostesa/PCRL/tree/{C.IHS_PUBLICATION}/publications/pcrl-income-headroom-successor-v1'},
           'mirror': '.private/pcrl_income_headroom_successor_v1 (inside this worktree; read-only by convention; hash-checked on every load)',
           'copies': rows, 'n_copies': len(rows), 'consistency_checks': checks, 'access_log_snapshots': snapshots,
           'lineage_dev_report_truth_reads_before_this_campaign': {'count': len(reads), 'events': reads,
                                                                   'note': 'the lineage log also holds non-read lines (e.g. one CORRECTION record)'},
           'pinned_source_documents': pinned,
           'labels_read': 'none (exports are copied and hashed; no label array is opened by this unit)',
           'sealed_assessment': 'not admitted; every loader refuses it', 'problems': problems,
           'status': 'PASS' if not problems else 'FAIL'}
    C.write_json_x(C.RESULTS / 'SOURCE_ADMISSION.json', rec)
    print(json.dumps({'n_copies': len(rows), 'problems': problems, 'lineage_dev_reads': len(reads),
                      'checks_failed': [k for k, v in checks.items() if not v],
                      'pinned_not_identical': [f for f, v in pinned.items() if not v['identical_to_public_blob']]}))
    return rec


if __name__ == '__main__':
    run()
