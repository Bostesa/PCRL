"""Aggregate per-unit fit records into committed ledgers, and count every status.

Outputs (committed; bulk per-unit artifacts stay local):

* `TRACK_N_FITS.json` — every Track N unit: selected step, selection score components,
  runtime, release hash, and whether the released channel is the untouched start;
* `TRACK_E_FITS.json` — every Track E map: realised rank, distortion, residual declared
  moment energy per role, boundary degeneracy, whitening and spectra per channel;
* `COUNTS.json` — expected, fitted, infeasible, duplicate, audited and pending, per track.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .common import OUT, read_json, sha_file, utcnow, write_json_atomic

SEEDS = (0, 1, 2)


def track_n() -> dict:
    rows = []
    for seed in SEEDS:
        context = read_json(OUT / f'seed_{seed}' / 'track_n_context.json')
        start_hash = {}
        for path in sorted((OUT / f'seed_{seed}' / 'fits').glob('N_*/fit_record.json')):
            r = read_json(path)
            sel = r['monitor_scores'][r['selected_index']]
            first = r['monitor_scores'][0]
            rows.append({
                'seed': seed, 'unit': r['unit'], 'init': r['init'], 'gamma': r['gamma'],
                'policy': r['policy'], 'beta': r['beta'], 'block': r['block'],
                'selected_step': r['selected_step'],
                'selected_source': sel['source'], 'selected_distortion': sel['distortion'],
                'selected_penalty': sel['penalty'], 'selected_score': sel['monitor_score'],
                'step0_score': first['monitor_score'],
                'step0_distortion': first['distortion'],
                'runtime_seconds': r['runtime_seconds'],
                'release_sha256': r['release']['sha256'],
                'channel_hash_test': r['channel_hash']['test']})
        write_json_atomic(OUT / f'seed_{seed}' / 'track_n_ledger_ref.json',
                          {'j_initial_distortion': context['j_initial_distortion_vs_A0_teacher']})
    by = defaultdict(list)
    for row in rows:
        by[row['init'], row['gamma'], row['block']].append(row['selected_step'])
    summary = {f'{i}|gamma={g:g}|{b}': {'units': len(v), 'step0': sum(s == 0 for s in v),
                                        'selected_steps': dict(Counter(v))}
               for (i, g, b), v in sorted(by.items())}
    record = {'generated_utc': utcnow(), 'units': rows, 'selection_summary': summary,
              'total_runtime_seconds': float(sum(r['runtime_seconds'] for r in rows)),
              'j_initial_distortion_vs_A0_teacher': {
                  seed: read_json(OUT / f'seed_{seed}' / 'track_n_context.json')
                  ['j_initial_distortion_vs_A0_teacher'] for seed in SEEDS}}
    write_json_atomic(OUT / 'TRACK_N_FITS.json', record)
    return record


def track_e() -> dict:
    units, contexts = [], {}
    for seed in SEEDS:
        fit = read_json(OUT / f'seed_{seed}' / 'track_e_fit.json')
        for u in fit['units']:
            units.append({k: v for k, v in u.items() if k != 'release'} |
                         {'release_sha256': (u.get('release') or {}).get('sha256')})
        contexts[seed] = {'contexts': fit['contexts'], 'nuisance': fit['nuisance'],
                          'basis': fit['basis'], 'J_parity_all_bitwise':
                          all(v['bitwise_identical'] for v in fit['J_parity'].values()),
                          'maps_sha256': fit['maps_sha256'],
                          'runtime_seconds': fit['runtime_seconds']}
    # Mechanism: declared-moment energy removed, by policy and k (A0 and J separately).
    mech = defaultdict(list)
    for seed in SEEDS:
        fit = read_json(OUT / f'seed_{seed}' / 'track_e_fit.json')
        for u in fit['units']:
            if 'moment_energy_after' not in u:
                continue
            before = fit['contexts'][u['channel']]['energy']
            local = sum(u['moment_energy_after'][r] for r in ('A/public_coverage', 'A/SEX',
                                                              'A/RAC1P'))
            local0 = sum(before[r] for r in ('A/public_coverage', 'A/SEX', 'A/RAC1P'))
            coal = sum(u['moment_energy_after'][r] for r in ('AB/SEX', 'AB/RAC1P'))
            coal0 = sum(before[r] for r in ('AB/SEX', 'AB/RAC1P'))
            mech[u['channel'], u['policy'], str(u['k'])].append(
                {'local_fraction_remaining': local / local0,
                 'coalition_fraction_remaining': coal / coal0,
                 'distortion': u['distortion_normalised']})
    mechanism = {f'{c}|{p}|k={k}': {key: float(np.mean([r[key] for r in v]))
                                   for key in v[0]}
                 for (c, p, k), v in sorted(mech.items())}
    record = {'generated_utc': utcnow(), 'units': units, 'per_seed': contexts,
              'mechanism_declared_moment_energy_remaining': mechanism}
    write_json_atomic(OUT / 'TRACK_E_FITS.json', record)
    return record


def counts() -> dict:
    matrix = read_json(OUT / 'MATRIX.json')
    out = {}
    for track in ('N', 'E'):
        slots = matrix[f'track_{track}']['slots']
        c = Counter()
        for seed in SEEDS:
            identity_path = OUT / f'seed_{seed}' / 'audit_identity.json'
            identity = read_json(identity_path) if identity_path.exists() else {}
            fit = read_json(OUT / f'seed_{seed}' / 'track_e_fit.json') if track == 'E' else None
            infeasible = {u['unit'] for u in fit['units'] if u.get('status') == 'SCOPED INFEASIBLE'} \
                if fit else set()
            for s in slots:
                c['expected'] += 1
                release = OUT / f'seed_{seed}' / 'releases' / s['unit'] / 'releases.npz'
                if s['unit'] in infeasible:
                    c['infeasible'] += 1
                    continue
                if not release.exists():
                    c['not_fitted'] += 1
                    continue
                c['fitted'] += 1
                entry = identity.get(s['unit'])
                if entry is None:
                    c['pending_audit'] += 1
                elif entry['duplicate']:
                    c['duplicate_of_audited_release'] += 1
                    if entry['audited_as'].startswith('ref_'):
                        c['duplicate_of_untouched_start'] += 1
                else:
                    c['audited_distinct'] += 1
        out[track] = dict(c)
    refs = sum(1 for seed in SEEDS for r in ('ref_A0', 'ref_J')
               if (OUT / f'seed_{seed}' / r / 'complete.json').exists())
    record = {'generated_utc': utcnow(), 'tracks': out, 'references_audited': refs,
              'nominal_total': matrix['nominal_registry_total']}
    write_json_atomic(OUT / 'COUNTS.json', record)
    return record


if __name__ == '__main__':
    import json
    track_n()
    track_e()
    print(json.dumps(counts(), indent=1))
