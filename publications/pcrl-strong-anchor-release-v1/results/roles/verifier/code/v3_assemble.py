"""Verifier wave 3 (label-free): append the wave-3 entries (deployment packages, publication leak scan) to CHECKS.json after checking
the file against the hash it had at the end of wave 2; earlier entries are kept unchanged. The coordinator's deploy/DEPLOYMENT_CHECK.json
and DEPLOYMENT_VALIDITY.json are read only here, for the final comparison.

    python3 .../roles/verifier/code/v3_assemble.py --utc <date -u> --prev-sha256 <sha of CHECKS.json after wave 2>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

CHK = {'a_depends': 'every depends_sha256 entry re-hashes (CODE:: module file, SAR:: under runs/, REPO:: under the repository root)',
       'b_stored_equals_release': 'the stored deployment wire equals the evaluated RELEASE.npz bitwise (both tasks, six roles) and the '
                                  'release hash chain holds (RELEASE.npz = RELEASE.json = PACKAGE.json release_sha256; the stored wire is a '
                                  'pinned dependency)',
       'c_model_select_wire': 'the MODEL_SELECT wire hashes (float64 vectors + int64 argmax) equal wire_sha256_eval of the EVAL_MS record '
                              'that evaluated this release',
       'd_validity': 'probability validity of the 12 stored arrays: finite, within [0, 1], rows sum to 1 within 1e-9',
       'e_blackbox': 'sar.deploy.release on 50 DEV_REPORT raw rows (seed 20261026, the same rows for every package) equals the stored '
                     'DEV_REPORT wire bitwise; decisions equal the stored argmax',
       'f_refusals': 'three refusal probes: an extra SEX field, a missing POBP field, a raw_score request'}


def rel(p):
    return str(Path(p).relative_to(V.ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--utc', required=True)
    ap.add_argument('--prev-sha256', required=True)
    a = ap.parse_args()
    cp = V.VDIR / 'CHECKS.json'
    if V.fsha(cp) != a.prev_sha256:
        raise PermissionError('CHECKS.json differs from its end-of-wave-2 content')
    W = json.loads(cp.read_text())
    dp, gp, sp = V.WORK / 'deploy3.json', V.WORK / 'deploy3_diag.json', V.WORK / 'pubscan3.json'
    D, G, S = (json.loads(p.read_text()) for p in (dp, gp, sp))
    new = []
    for name, r in D['packages'].items():
        pk = V.PRIV / 'deploy' / name / 'PACKAGE.json'
        ins = {rel(pk): V.fsha(pk), rel(dp): V.fsha(dp)}
        for key, what in CHK.items():
            x = r[key]
            new.append({'id': f'W3_DEPLOY_{name}_{key}', 'wave': 3, 'what': f'{name}: {what}', 'inputs_sha256': ins,
                        'recomputed': {k: v for k, v in x.items() if k != 'pass'}, 'record': 'package files and saved releases',
                        'status': 'PASS' if x['pass'] else 'FAIL',
                        'tolerance': {'d_validity': 1e-9, 'e_blackbox': 'bitwise'}.get(key, 'bitwise / exact'),
                        'unit': 'none (label-free, torch venv, outside the queue)'})
    # separate entries (the bitwise FAILs above stand): tolerance view and the full-batch diagnostic
    for name, r in D['packages'].items():
        e = r['e_blackbox']
        if e['pass'] or 'per_task' not in e:
            continue
        mx = max(t['max_abs'] for t in e['per_task'].values())
        dec = all(t['decision_equal_stored_argmax'] for t in e['per_task'].values())
        new.append({'id': f'W3_DEPLOY_{name}_e_blackbox_tolerance', 'wave': 3,
                    'what': f'{name}: separate entry for the failed bitwise check (that FAIL stands): the 50-row black-box output equals the stored '
                            'DEV_REPORT wire within 1e-15 with identical decisions',
                    'inputs_sha256': {rel(dp): V.fsha(dp)}, 'recomputed': {'max_abs': mx, 'decisions_identical': dec},
                    'record': 'stored DEV_REPORT wire', 'status': 'PASS' if mx <= 1e-15 and dec else 'FAIL', 'tolerance': 1e-15,
                    'unit': 'none (label-free)'})
    for name, r in G['packages'].items():
        ok = all(t['full_batch_bitwise'] and t['full_batch_decisions_equal'] for t in r.values())
        new.append({'id': f'W3_DEPLOY_{name}_full_batch_bitwise', 'wave': 3,
                    'what': f'{name}: diagnostic: sar.deploy.release on ALL 5,943 DEV_REPORT rows (the batch the stored wire was computed on) '
                            'reproduces the stored wire bitwise; 50-row and 1-row batches differ by the recorded ULP-level amounts',
                    'inputs_sha256': {rel(gp): V.fsha(gp)}, 'recomputed': r, 'record': 'stored DEV_REPORT wire',
                    'status': 'PASS' if ok else 'FAIL', 'tolerance': 'bitwise', 'unit': 'none (label-free)',
                    'note': 'the release depends on the batch composition at the ULP level (float64 matrix products); decisions are unaffected'})
    # comparison with the coordinator's own records (read only now)
    cc = json.loads((V.RES / 'deploy' / 'DEPLOYMENT_CHECK.json').read_text())
    cv = json.loads((V.RES / 'deploy' / 'DEPLOYMENT_VALIDITY.json').read_text())
    mine_full = {n: all(t['full_batch_bitwise'] for t in G['packages'][n].values()) for n in G['packages']}
    theirs = {n: all(all(v.values()) for v in p['bitwise_equal_to_stored_DEV_REPORT_wire'].values()) for n, p in cc['packages'].items()}
    agree_full = all(theirs.get(n) == mine_full[n] for n in mine_full)
    sample = {n: D['packages'][n]['e_blackbox']['pass'] for n in D['packages']}
    new.append({'id': 'W3_DEPLOY_compare_coordinator_records', 'wave': 3,
                'what': 'comparison with the coordinator\'s deploy/DEPLOYMENT_CHECK.json and DEPLOYMENT_VALIDITY.json (read after my checks): '
                        'their all-rows bitwise parity agrees with my full-batch diagnostic; their validity verdicts agree with mine (d); their '
                        '"all_bitwise" does not extend to the 50-row sample (batch dependence)',
                'inputs_sha256': {rel(V.RES / 'deploy' / 'DEPLOYMENT_CHECK.json'): V.fsha(V.RES / 'deploy' / 'DEPLOYMENT_CHECK.json'),
                                  rel(V.RES / 'deploy' / 'DEPLOYMENT_VALIDITY.json'): V.fsha(V.RES / 'deploy' / 'DEPLOYMENT_VALIDITY.json')},
                'recomputed': {'full_batch_bitwise_mine': mine_full, 'sample50_bitwise_mine': sample,
                               'validity_mine': {n: D['packages'][n]['d_validity']['pass'] for n in D['packages']},
                               'refusals_mine': {n: D['packages'][n]['f_refusals']['pass'] for n in D['packages']}},
                'record': {'all_bitwise': cc['all_bitwise'], 'all_refused': cc['all_refused'], 'per_package_bitwise': theirs,
                           'validity_all_ok': cv['all_ok'], 'validity_tolerance': cv['tolerance_row_sum']},
                'status': 'PASS' if agree_full and cv['all_ok'] == all(D['packages'][n]['d_validity']['pass'] for n in D['packages'])
                and cc['all_refused'] == all(D['packages'][n]['f_refusals']['pass'] for n in D['packages']) else 'FAIL',
                'tolerance': 'exact', 'unit': 'none (label-free)',
                'note': 'the coordinator compared all DEV_REPORT rows in one batch; on a 50-row batch the six FB packages differ by <= 5.6e-16'})
    # publication leak scan
    hits = S['line_hits']
    by = {}
    for h in hits:
        by.setdefault(h['pattern'], []).append(f"{h['file']}:{h['line']}")
    judged = {
        'personal_path_users': ('FAIL_IF_ANY', 'absolute home paths'),
        'user_name': ('FAIL_IF_ANY', 'the user\'s name (<owner>) inside the drive-relocation folder name; a personal identifier in an '
                                     'anonymous publication: redact (e.g. <relocation run>)'),
        'email': ('FAIL_IF_ANY', 'e-mail addresses'),
        'github_token': ('FAIL_IF_ANY', 'tokens'), 'aws_key': ('FAIL_IF_ANY', 'keys'), 'openai_anthropic_key': ('FAIL_IF_ANY', 'keys'),
        'private_key_block': ('FAIL_IF_ANY', 'keys'), 'secret_word': ('FAIL_IF_ANY', 'secret / password / api-key words'),
        'token_word': ('FAIL_IF_ANY', 'the word token'),
        'volumes_path': ('HARMLESS', '/Volumes/YOTUO is the backup drive label and /System/Volumes/Data a system path; no user name'),
        'worktree_path': ('HARMLESS', 'sanitized <repository>/... prefixes or relative worktree names of this project; no user name or '
                                      'absolute personal path (cosmetic: the sanitized literal in code/v_decide.py no longer resolves)'),
        'home_tilde_path': ('HARMLESS', '~/storage-relocation-20261022/ is home-relative and carries no user name'),
        'private_tmp_path': ('FAIL_IF_ANY', 'session temp paths'),
        'repo_owner_account': ('HARMLESS', 'the public remote https://github.com/Bostesa/PCRL.git (the owner keeps this repository public '
                                           'by decision)'),
        'scratchpad_session_path': ('HARMLESS', 'the word "scratchpad" in the reviewer\'s hand-backs; no path'),
    }
    pi = {rel(sp): V.fsha(sp)}
    for pat, (rule, why) in judged.items():
        h = by.get(pat, [])
        status = 'PASS' if (rule == 'HARMLESS' or not h) else 'FAIL'
        new.append({'id': f'W3_PUB_{pat}', 'wave': 3, 'what': f'publication dry build ({S["n_files"]} files): lines matching {pat}',
                    'inputs_sha256': pi, 'recomputed': {'hits': h, 'n': len(h)}, 'record': 'none expected' if rule != 'HARMLESS' else 'judged',
                    'status': status, 'tolerance': 0, 'unit': 'none (label-free)', 'note': why if h else None})
    new.append({'id': 'W3_PUB_binary_files', 'wave': 3, 'what': 'publication dry build: no binary or array file (.npz, .npy, .pkl, .pt, archives; '
                'NUL bytes or non-UTF-8 content)', 'inputs_sha256': pi, 'recomputed': S['binary'], 'record': 'none',
                'status': 'PASS' if not S['binary'] else 'FAIL', 'tolerance': 0, 'unit': 'none (label-free)'})
    new.append({'id': 'W3_PUB_record_level_data', 'wave': 3, 'what': 'publication dry build: no record-level data (JSON arrays of >= 500 numbers, '
                'person-level keys with >= 100-element lists, lines with >= 200 comma-separated numbers, CSV tables with person-level columns)',
                'inputs_sha256': pi, 'recomputed': {'record_level': S['record_level'], 'csv_tables': S['csv_tables']},
                'record': 'none', 'status': 'PASS' if not S['record_level'] and not any(t['person_level_columns'] for t in S['csv_tables'])
                else 'FAIL', 'tolerance': 0, 'unit': 'none (label-free)',
                'note': 'the only CSV is results/dev/UTILITY_LEDGER.csv: 44 rows of per-release aggregate utility values'})
    nf = [c['id'] for c in new if c['status'] != 'PASS']
    W['checks'] = W['checks'] + new
    W['waves']['3'] = {'utc': a.utc, 'totals': {'checks': len(new), 'pass': len(new) - len(nf), 'fail': len(nf)}, 'failed': nf,
                       'units': [], 'outside_queue_cpu_s': D['cpu_s'] + G['cpu_s'],
                       'code_sha256': {rel(p): V.fsha(p) for p in sorted(HERE.glob('*.py'))}, 'sha256_before': a.prev_sha256}
    allf = [c['id'] for c in W['checks'] if c['status'] != 'PASS']
    W['totals_all_waves'] = {'checks': len(W['checks']), 'pass': len(W['checks']) - len(allf), 'fail': len(allf), 'failed': allf}
    V.jdump(cp, W)
    print(json.dumps({'wave3': W['waves']['3']['totals'], 'wave3_failed': nf, 'all': W['totals_all_waves']['checks'],
                      'all_fail': W['totals_all_waves']['fail']}))


if __name__ == '__main__':
    main()
