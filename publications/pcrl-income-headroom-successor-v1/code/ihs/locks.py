"""Final-stage locks for a promoted successor (PROTOCOL.md section 10; used only if NOMINATION.json names a finalist).

    python -m ihs.locks freeze --twin F:CONFIG [--cc F:CONFIG]      # bank + READER_PLAN + ANALYSIS_LOCK + FINALIST_LOCK
    python -m ihs.locks reporting                                   # REPORTING_LOCK after the final readers

- Final bank (RUNS/FINAL/SERVICE_BANK_S<s>.npz; manifest SERVICE_BANK_MANIFEST.json): SUCCESSOR (the nominee), NEURAL_RAW and
  H_LJ (frozen-hybrids bank), TW (the nominee's matched task-only twin), CC (a declared matched component control, if any),
  RETIRED_B3CTL, B2TWIN and PUB_IBSI (the source campaign's final bank), on ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT
  (label-free outputs; DEV_REPORT outputs are released vectors, not labels). At most 8 effective releases per seed; exact
  aliases (equal wires for both tasks on the three roles) are deduplicated.
- READER_PLAN.json: the validation's enhanced common slate with its full estimator parameters and library versions (the
  `.venv` interpreter, numpy 2.4.2 / scikit-learn 1.8.0 / scipy 1.17.1), selection rules, controls and the refit roster.
- ANALYSIS_LOCK.json: the endpoint roster built here from generic roles (CLAIMS_DRAFT.md), multiplicity, critical values,
  bootstrap seed, control bands, and the SHA-256 of ihs/claims.py (the verbatim encoding of the claims draft).
- FINALIST_LOCK.json: the nominee, both seed packages (release and parameter hashes), the selection record (NOMINATION.json,
  EXPERIMENT_LEDGER.csv hashes), comparators and controls, the output contract, and the hashes of the three locks above.
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from . import common as C
from . import devrun as D

ROLES3 = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
REFS = ('NEURAL_RAW', 'H_LJ')
BOOT_SEED, BOOT_B = 2026101705, 2048
ALPHA = 0.025


def _release(spec, seed, roles=ROLES3):
    fam, cfg = spec.split(':')
    outs, rec = D.load_release(fam, cfg, seed, roles=roles)
    return outs, rec


def build_bank(seed, nominee, twin, cc=None):
    frozen, fids = C.frozen_bank(seed, roles=ROLES3)
    amd, aids = C.amd_service_bank(seed, roles=ROLES3)
    services = {}
    for r in REFS:                                             # references first, so an alias always keeps the reference's name (reviewer L1)
        services[r] = {role: {t: frozen[r][role][t] for t in C.TASKS} for role in ROLES3}
    services['SUCCESSOR'], rec_s = _release(nominee, seed)
    services['TW'], rec_t = _release(twin, seed)
    if cc:
        services['CC'], _ = _release(cc, seed)
    for name, src in (('RETIRED_B3CTL', 'FINALIST'), ('B2TWIN', 'B2TWIN'), ('PUB_IBSI', 'PUB_IBSI')):
        services[name] = {role: {t: amd[src][role][t] for t in C.TASKS} for role in ROLES3}
    ids = {}
    for role in ROLES3:
        X = C.load_role(role, ('features',))
        ids[f'row_id__{role}'] = X['row_id']; ids[f'unit__{role}'] = X['unit']
        if not (np.array_equal(fids[f'row_id__{role}'], X['row_id']) and np.array_equal(aids[f'row_id__{role}'], X['row_id'])):
            raise AssertionError(f'{role}: bank row order differs from the export')
    wires = {svc: {t: tuple(C.wire_sha(services[svc][r][t]) for r in ROLES3) for t in C.TASKS} for svc in services}
    aliases, kept = {}, []
    for svc in services:
        twin_of = next((k for k in kept if wires[k] == wires[svc]), None)
        if twin_of:
            aliases.setdefault(twin_of, [twin_of]).append(svc)
        else:
            kept.append(svc); aliases.setdefault(svc, [svc])
    if len(kept) > 8:
        raise AssertionError(f'more than 8 effective releases: {kept}')
    arrays = dict(ids)
    for svc in kept:
        for role in ROLES3:
            for t in C.TASKS:
                arrays[f'{svc}__{role}__{t}'] = np.ascontiguousarray(services[svc][role][t], dtype=np.float64)
    p = C.RUNS / 'FINAL' / f'SERVICE_BANK_S{seed}.npz'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('wb') as f:
        np.savez(f, **arrays)
    return {'path': str(p.relative_to(C.ROOT)), 'sha256': C.sha(p), 'kept': kept, 'aliases': aliases,
            'wire_sha256': {svc: {t: dict(zip(ROLES3, wires[svc][t])) for t in C.TASKS} for svc in services},
            'release_records': {'SUCCESSOR': rec_s['release_sha256'], 'TW': rec_t['release_sha256']}}


def reader_plan():
    from ffv.readers import NEW8, CONTROL_READERS, params_of
    from . import final_readers as FR
    return {'schema': 'IHS1-reader-plan-v1', 'versions': {'numpy': '2.4.2', 'scipy': '1.17.1', 'sklearn': '1.8.0'},
            'interpreter': '/Users/<you>/PCRL/.venv/bin/python (Python 3.13.7): the versions of the validation READER_BANK_LOCK',
            'source_slate': list(FR.SRC8), 'new_recipes': list(NEW8), 'estimator_params': {n: params_of(n) for n in NEW8},
            'selection': {'original': 'source 8 (pair: 8 pair readers + 16 ignore-one alternatives); maximum oriented ATTACK_VALIDATION AUC; exact ties to the earlier candidate',
                          'enhanced': 'source-first order: source 8 [+ ignore-one] then new 8 [+ ignore-one]; same rule; AV near tie < 0.010 flagged'},
            'fit_role': 'ATTACK_FIT (8,001)', 'select_role': 'ATTACK_VALIDATION (3,000)', 'score_role': 'DEV_REPORT (released outputs only)',
            'model_seed': 'MODEL_SEED_BASE + s', 'controls': {'readers': list(CONTROL_READERS), 'shuffled_permutation_seed': 'NULL_SEED + s (2026101602 + s)',
                                                              'bands': {'planted_min': 0.95, 'shuffled': [0.45, 0.55]}},
            'refit_roster': {'services': list(FR.ROSTER_SERVICES), 'model_seed_refits': 'seed + 1000 j, j = 1..4 (seed-dependent winners only)',
                             'aux_bootstrap_refits': f'household bootstrap of ATTACK_FIT, resample seeds {FR.AUX_SEED} + 10 s + j, j = 1..4'},
            'verifier_refit_roster': {                                                              # reviewer L2 (PROTOCOL section 10)
                'pair_readers': [f'PAIR__{svc}__{n}' for svc in ('SUCCESSOR', 'NEURAL_RAW', 'H_LJ') for n in ('ET5', 'HGB31', 'MLP6432', 'CATI1')],
                'winners': 'the ENHANCED and ORIGINAL pair winners of SUCCESSOR, NEURAL_RAW and H_LJ on both seeds',
                'requirement': 'refitted independently from this plan with the same interpreter and seeds; AV AUCs and DEV_REPORT scores compared with the stored ones, bitwise where possible'},
            'code_sha256': {f: C.sha(C.ROOT / f) for f in ('ihs/final_readers.py', 'ffv/readers.py', 'ffv/features.py', 'scm_bench/readers.py',
                                                            'task_conditioned_representation/attacks.py')}}


def endpoint_roster(cc_declared):
    eps = []
    others = list(REFS) + ['TW'] + (['CC'] if cc_declared else [])
    for s in C.SEEDS:
        for t in C.TASKS:
            for r in ('U0', 'Ucal'):
                for m, cap in C.CAPS.items():
                    eps.append({'key': f'UTILITY|S|{t}|{r}|{m}|s{s}', 'family': 'P', 'claim': 'E', 'seed': s, 'type': 'utility', 'service': 'SUCCESSOR',
                                'task': t, 'ref': r, 'metric': m, 'kind': 'no_excess', 'margin': cap})
                    eps.append({'key': f'UTILITY|H_LJ|{t}|{r}|{m}|s{s}', 'family': 'D', 'claim': 'R2/O floor', 'seed': s, 'type': 'utility', 'service': 'H_LJ',
                                'task': t, 'ref': r, 'metric': m, 'kind': 'no_excess', 'margin': cap})
        for g in (0, 1):
            eps.append({'key': f'SUBGROUP|S|sex{g}|s{s}', 'family': 'P', 'claim': 'E', 'seed': s, 'type': 'guard', 'service': 'SUCCESSOR', 'group': g,
                        'kind': 'no_excess', 'margin': C.GUARD})
            eps.append({'key': f'SUBGROUP|H_LJ|sex{g}|s{s}', 'family': 'D', 'claim': 'O floor', 'seed': s, 'type': 'guard', 'service': 'H_LJ', 'group': g,
                        'kind': 'no_excess', 'margin': C.GUARD})
        for x in others:
            svc = x
            eps.append({'key': f'BEN|{x}|s{s}', 'family': 'P', 'claim': 'P' if x in REFS else ('M' if x == 'TW' else 'C'), 'seed': s, 'type': 'auc',
                        'view': 'pair', 'a': svc, 'b': 'SUCCESSOR', 'kind': 'benefit', 'margin': 0.0, 'refs': [0.020]})
            for v in ('income', 'occupation'):
                eps.append({'key': f'NW|{x}|{v}|s{s}', 'family': 'P', 'claim': 'P' if x in REFS else ('M' if x == 'TW' else 'C'), 'seed': s, 'type': 'auc',
                            'view': v, 'a': 'SUCCESSOR', 'b': svc, 'kind': 'no_excess', 'margin': 0.010})
        for x in ('RETIRED_B3CTL', 'B2TWIN', 'PUB_IBSI'):
            eps.append({'key': f'DESC|BEN|{x}|s{s}', 'family': 'D', 'claim': 'D', 'seed': s, 'type': 'auc', 'view': 'pair', 'a': x, 'b': 'SUCCESSOR',
                        'kind': 'difference', 'margin': 0.0})
            for t in C.TASKS:
                for m in ('nll', 'brier'):
                    eps.append({'key': f'DESC|LOSS|{x}|{t}|{m}|s{s}', 'family': 'D', 'claim': 'D', 'seed': s, 'type': 'loss_diff', 'a': 'SUCCESSOR', 'b': x,
                                'task': t, 'metric': m, 'kind': 'difference', 'margin': 0.0})
        for x in ['TW'] + (['CC'] if cc_declared else []):                                             # reviewer C6
            for t in C.TASKS:
                for m in ('nll', 'brier'):
                    eps.append({'key': f'DESC|LOSS|{x}|{t}|{m}|s{s}', 'family': 'D', 'claim': f"{'M' if x == 'TW' else 'C'} qualifier", 'seed': s,
                                'type': 'loss_diff', 'a': 'SUCCESSOR', 'b': x, 'task': t, 'metric': m, 'kind': 'difference', 'margin': 0.0})
        for r in REFS:
            eps.append({'key': f'DESC|TWBEN|{r}|s{s}', 'family': 'D', 'claim': 'M qualifier (twin pair benefit vs the eligible reference)', 'seed': s,
                        'type': 'auc', 'view': 'pair', 'a': r, 'b': 'TW', 'kind': 'difference', 'margin': 0.0})
    return eps


def analysis_lock(cc_declared, occupation_inherited, control_designated):
    from amd.report import crit
    eps = endpoint_roster(cc_declared)
    n_claims = 3 if cc_declared else 2
    K = {'P': sum(e['family'] == 'P' for e in eps), 'D': sum(e['family'] == 'D' for e in eps)}
    G = 3955
    return {'schema': 'IHS1-analysis-lock-v1', 'claims_source': 'roles/statistical_reviewer/CLAIMS_DRAFT.md (registered 74376912c)',
            'claims_code_sha256': C.sha(C.ROOT / 'ihs' / 'claims.py'), 'report_code_sha256': C.sha(C.ROOT / 'ihs' / 'report.py'),
            'declared': {'TW': True, 'CC': bool(cc_declared), 'occupation_inherited': bool(occupation_inherited), 'control_designated': bool(control_designated)},
            'inference': {'bootstrap_seed': BOOT_SEED, 'bootstrap_B': BOOT_B, 'alpha_family_one_sided': ALPHA, 'n_claims': n_claims, 'K': K,
                          'critical_values_expected': {'c1': crit(G, ALPHA / n_claims), 'c_P': crit(G, 0.05 / (2 * K['P'])),
                                                       'c_D': crit(G, 0.05 / (2 * K['D'])), 'marginal': crit(G, 0.025)},
                          'note': 'family D is descriptive (marginal intervals are the reported quantity); c_D is computed only for completeness'},
            'controls_bands': {'planted_min': 0.95, 'shuffled': [0.45, 0.55]},
            'endpoints': eps}


def freeze(twin, cc=None):
    nom = json.loads((C.RESULTS / 'NOMINATION.json').read_text())
    if nom['outcome'] != 'FINALIST_NOMINATED':
        raise PermissionError('no nominated finalist: nothing to freeze, DEV_REPORT stays closed')
    nominee = nom['nominated'].replace('/', ':')
    rec = next(r for r in nom['configurations'] if f"{r['family']}:{r['config_id']}" == nominee)
    banks = {f'S{s}': build_bank(s, nominee, twin, cc) for s in C.SEEDS}
    occ_inherited = all(banks[f'S{s}']['wire_sha256']['SUCCESSOR']['occupation'] == banks[f'S{s}']['wire_sha256']['H_LJ']['occupation'] for s in C.SEEDS)
    man = {'schema': 'IHS1-service-bank-manifest-v1', 'utc': C.now(), 'bank_files_sha256': {k: v['sha256'] for k, v in banks.items()},
           'per_seed': banks, 'release_contract': 'each recipient: its own float64 probability vector and argmax decision; coalition: the aligned pair; nothing else'}
    C.write_json_x(C.RESULTS / 'SERVICE_BANK_MANIFEST.json', man)
    C.write_json_x(C.RESULTS / 'READER_PLAN.json', reader_plan())
    C.write_json_x(C.RESULTS / 'ANALYSIS_LOCK.json', analysis_lock(cc is not None, occ_inherited, 'CONTROL_DESIGNATED_FINALIST' in rec['flags']))
    pk = {}
    for s in C.SEEDS:
        fam, cfg = nominee.split(':')
        d = D.release_dir(fam, cfg, s)
        pk[f'S{s}'] = {p.name: C.sha(p) for p in sorted(d.iterdir()) if p.is_file()}
    lock = {'schema': 'IHS1-finalist-lock-v1', 'utc': C.now(), 'nominee': nominee, 'nomination_record': rec,
            'nomination_sha256': C.sha(C.RESULTS / 'NOMINATION.json'), 'ledger_sha256': C.sha(C.RESULTS / 'EXPERIMENT_LEDGER.csv'),
            'package_files_sha256': pk, 'matched_task_twin': twin, 'matched_component_control': cc,
            'control_nomination_records': {k: next((r for r in nom['configurations'] if f"{r['family']}:{r['config_id']}" == v), None)
                                           for k, v in (('TW', twin), ('CC', cc)) if v},
            'eligible_references': list(REFS), 'diagnostic': ['RETIRED_B3CTL'], 'descriptive': ['B2TWIN', 'PUB_IBSI'],
            'occupation_output_inherited': occ_inherited,
            'output_contract': 'each recipient receives only its own float64 probability vector and argmax decision, computed from AGEP, WKHP, COW, SCHL, MAR, POBP',
            'locks_sha256': {n: C.sha(C.RESULTS / n) for n in ('SERVICE_BANK_MANIFEST.json', 'READER_PLAN.json', 'ANALYSIS_LOCK.json')},
            'no_return_to_development': 'after the DEV_REPORT evaluation the campaign does not return to development'}
    C.write_json_x(C.RESULTS / 'FINALIST_LOCK.json', lock)
    print(json.dumps({'nominee': nominee, 'kept': {k: v['kept'] for k, v in banks.items()}, 'occupation_inherited': occ_inherited}, indent=1))


def reporting():
    rl = {'schema': 'IHS1-reporting-lock-v1', 'utc': C.now(), 'readers': {}, 'attack_validation_controls': {}}
    bands = json.loads((C.RESULTS / 'ANALYSIS_LOCK.json').read_text())['controls_bands']
    for s in C.SEEDS:
        d = C.RUNS / f'READERS_S{s}'
        rl['readers'][f'S{s}'] = {'record_sha256': C.sha(d / 'READERS_SEED.json'), 'av_scores_sha256': C.sha(d / 'AV_SCORES.npz'),
                                  'dev_report_scores_sha256': C.sha(d / 'DEV_REPORT_SCORES.npz')}
        rr = json.loads((d / 'READERS_SEED.json').read_text())                                    # reviewer C3: ATTACK_VALIDATION controls
        for view, crec in rr['controls'].items():
            for name, c in crec.items():
                ok = bands['shuffled'][0] <= c['shuffled_attack_validation_auc'] <= bands['shuffled'][1] and c['planted_attack_validation_auc'] >= bands['planted_min']
                rl['attack_validation_controls'][f'S{s}|{view}|{name}'] = {'shuffled': c['shuffled_attack_validation_auc'],
                                                                          'planted': c['planted_attack_validation_auc'], 'inside_bands': ok}
    rl['attack_validation_controls_all_inside'] = all(v['inside_bands'] for v in rl['attack_validation_controls'].values())
    if not rl['attack_validation_controls_all_inside']:
        rl['deviation'] = 'an ATTACK_VALIDATION control is outside its band: recorded before the DEV_REPORT read (PROTOCOL_DEVIATIONS.md)'
    rl['locks_sha256'] = {n: C.sha(C.RESULTS / n) for n in ('FINALIST_LOCK.json', 'ANALYSIS_LOCK.json', 'SERVICE_BANK_MANIFEST.json', 'READER_PLAN.json')}
    rl['events'] = list(C.DEV_EVENTS)
    C.write_json_x(C.RESULTS / 'REPORTING_LOCK.json', rl)
    print(json.dumps(rl, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--twin', required=True); f.add_argument('--cc', default=None)
    sub.add_parser('reporting')
    a = ap.parse_args()
    freeze(a.twin, a.cc) if a.cmd == 'freeze' else reporting()
